"""Generate/manipulate pyorbit bunches."""
from __future__ import print_function
import math
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d

from bunch import Bunch
from bunch import BunchTwissAnalysis
import orbit_mpi
from orbit_mpi import mpi_comm
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op
from spacecharge import Grid2D
from orbit.py_linac.lattice import BaseLinacNode
from orbit.bunch_generators import GaussDist1D
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist1D
from orbit.bunch_generators import KVDist2D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import TwissAnalysis
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist1D
from orbit.bunch_generators import WaterBagDist2D
from orbit.bunch_generators import WaterBagDist3D
from orbit.utils import consts

from btfsim import analysis


def get_coord_array(bunch):
    """Return Nx6 coordinate array from bunch."""
    X = np.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        X[i, 0] = bunch.x(i)
        X[i, 1] = bunch.xp(i)
        X[i, 2] = bunch.y(i)
        X[i, 3] = bunch.yp(i)
        X[i, 4] = bunch.z(i)
        X[i, 5] = bunch.dE(i)
    return X


def decimate(bunch, factor=1):
    """Reduce the number of macro-particles in the bunch.

    This just skips every `fac` indices, so we assume that the bunch
    coordinates were generated randomly.
    """
    n_parts0 = bunch.getSizeGlobal()
    if not factor or not (1 <= factor < n_parts0):
        print("No decimation for fac={}.".format(factor))
    new_bunch = Bunch()
    bunch.copyEmptyBunchTo(new_bunch)
    for i in range(0, n_parts0, factor):
        new_bunch.addParticle(
            bunch.x(i), bunch.xp(i),
            bunch.y(i), bunch.yp(i),
            bunch.z(i), bunch.dE(i),
        )
    new_bunch.macroSize(factor * bunch.macroSize())
    new_bunch.copyBunchTo(bunch)
    return bunch


def resample(bunch, n_parts=1, rms_factor=0.05):
    """Up/down-sample to obtain requested number of particles.

    Upsampling should probably be done by sampling from estimated pdf...

    Parameters
    ----------
    bunch : Bunch
        The pyorbit bunch to resample.
    n_parts : int
        The number of desired particles in the bunch.
    """
    n_parts0 = bunch.getSizeGlobal()
    mult = float(n_parts) / float(n_parts0)
    n_parts = int(n_parts)
    print("Resampling bunch from {} to {} particles...".format(n_parts0, n_parts))
    print("mult = {:.3f}".format(mult))

    if mult == 1:
        return []

    coords0 = get_coord_array(bunch)
    # Down-sample if n_parts0 > n_parts_new
    if mult < 1:
        ind = np.random.permutation(np.arange(n_parts0))[0:n_parts]

    # Up-sample if n_parts0 < n_parts_new. (This way is a lot of work.)
    elif mult > 1:
        nnew = n_parts - n_parts0

        # -- normal distribution of new particles will be ~1% rms width
        rmswidths = np.sqrt((coords0**2).mean(axis=0))
        scale = rmswidths * rms_factor
        # -- longitudinal will be 10x smaller
        scale[4] *= 0.1
        scale[5] *= 0.1

        # -- get integer multiplier (round up)
        intmult = int(np.ceil(mult))
        # -- explode each coordinate into intmultx particles (Gaussian cloud)
        # -- this will create a bunch with integere x original size (ie, 2x, 3x, etc..)
        newcoords = np.random.normal(
            loc=coords0, 
            scale=scale, 
            size=[intmult, n_parts0, 6],
        )
        reshape_coords = np.zeros([intmult * n_parts0, 6], dtype="f8")
        for i in range(6):
            reshape_coords[:, i] = newcoords[:, :, i].ravel()

        coords0 = reshape_coords.copy()

        # -- and downsample to desired number
        ind = np.random.permutation(np.arange(len(coords0)))[0:n_parts]

    # -- make new bunch and place re-sampled coordinates
    newbunch = Bunch()
    bunch.copyEmptyBunchTo(newbunch)  # copy attributes
    for i in ind:
        newbunch.addParticle(
            coords0[i, 0],
            coords0[i, 1],
            coords0[i, 2],
            coords0[i, 3],
            coords0[i, 4],
            coords0[i, 5],
        )
    # -- keep same current by re-setting macrosize
    newmacrosize = bunch.macroSize() * (1.0 / mult)
    newbunch.macroSize(newmacrosize)
    # -- over-write bunch_in
    newbunch.copyBunchTo(bunch)
    print("Done resampling.")
    return bunch

def attenuate(bunch, factor=1.0):
    """Adjust current by changing macrosize

    factor : float
        The fractional beam current attenuation.
    """
    bunch.macroSize(factor * bunch.macroSize())

def decorrelate_x_y_z(bunch):
    """Remove cross-plane correlations in the bunch by permuting 
    (x, x'), (y, y'), (z, z') pairs."""
    X = get_coord_array(bunch)
    for i in (0, 2, 4):
        idx = np.random.permutation(np.arange(X.shape[0]))
        X[:, i : i + 2] = X[idx, i : i + 2]
    for i, (x, xp, y, yp, z, dE) in enumerate(X):
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.z(i, z)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.dE(i, dE)

def shift(bunch, x, xp, y, yp, z, dE):
    print('Shifting bunch centroid...')
    for i in range(bunch.getSize()):
        bunch.x(i, bunch.x(i) + x)
        bunch.y(i, bunch.y(i) + y)  
        bunch.z(i, bunch.z(i) + z)
        bunch.xp(i, bunch.xp(i) + xp) 
        bunch.yp(i, bunch.yp(i) + yp)
        bunch.dE(i, bunch.dE(i) + dE)
    print('Bunch shifted.')

def center(bunch):
    """Shift the bunch so that first-order moments are zero."""
    twiss = BunchTwissAnalysis()
    twiss.analyzeBunch(bunch)
    return shift(bunch, *[twiss.getAverage(i) for i in range(6)])

def reverse(bunch):
    """Reverse the bunch propagation direction.

    Since the tail becomes the head of the bunch, the sign of z
    changes but the sign of dE does not change.
    """
    for i in range(bunch.getSize()):
        bunch.xp(i, -bunch.xp(i))
        bunch.yp(i, -bunch.yp(i))
        bunch.z(i, -bunch.z(i))
        

class BunchCalculator:
    """Calculate parameters from Bunch object.
    
    Attributes
    ----------
    coords : ndarray, shape (N, 6)
        The phase space coordinate array.
    cov : ndarray, shape(6, 6)
        The covariance matrix.
    """
    def __init__(self, bunch):
        self.coords = get_coord_array(bunch)
        self.coords[:, :4] *= 1e3  # mm, mrad, MeV
        self.coords[:, 5] *= 1e6  # keV
        self.cov = np.cov(self.coords.T)
        self.twiss_analysis = BunchTwissAnalysis()
        self.twiss_analysis.analyzeBunch(bunch)
        self.n_parts = bunch.getSizeGlobal()
        self.gamma = bunch.getSyncParticle().gamma()
        self.beta = bunch.getSyncParticle().beta()
        self.mass = bunch.getSyncParticle().mass()

    def twiss(self, dim="x", emit_norm_flag=False):
        """Return rms 2D Twiss parameters.
        
        beta: [mm/mrad]
        alpha: []
        eps: [mm*mrad]
        disp: [m]
        dispp: []
        """
        i = dim
        if type(dim) is str:
            i = ["x", "y", "z"].index(i)
        alpha, beta = analysis.twiss(self.cov, dim=dim)
        eps = analysis.emittance(self.cov, dim=dim)
        if emit_norm_flag and dim == "z":
            eps *= self.gamma**3 * self.beta
        disp = self.twiss_analysis.getDispersion(i)
        dispp = self.twiss_analysis.getDispersionDerivative(i)
        return {
            'beta': beta,
            'alpha': alpha,
            'eps': eps,
            'disp': disp,
            'dispp': dispp,
        }

    def norm_coords(self, scale_emittance=False):
        """Return coordinates normalized by rms Twiss parameters.
        
        The normalization occurs in x-x', y-y', and z-z'. The parameter
        `scale_emittance` will additional divide the coordinates by
        the square root of the rms emittanc.
        """
        X = self.coords
        Xn = np.zeros(X.shape)
        for i, dim in enumerate(['x', 'y', 'z']):
            twiss = self.twiss(dim=dim)
            alpha = twiss["alpha"]["value"]
            beta = twiss["beta"]["value"]
            i *= 2
            Xn[:, i] = X[:, i] / np.sqrt(beta)
            Xn[:, i + 1] = (np.sqrt(beta) * X[:, i + 1]) + (alpha * X[:, i] / np.sqrt(beta))
            if scale_emittance:
                eps = twiss["eps"]["value"]
                Xn[:, i:i+2] = Xn[:, i:i+2] / np.sqrt(eps)
        return Xn
        

class Base_BunchGenerator(object):
    """Base class for bunch generators.
    
    Attributes
    ----------
    """
    def __init__(self, mass=0.939294, charge=-1, ekin=0.0025, curr=40.0, freq=402.5e6):
        self.bunch = Bunch()
        self.bunch.mass(mass)
        self.bunch.charge(charge)
        self.bunch.getSyncParticle().kinEnergy(ekin)
        self.beta = self.bunch.getSyncParticle().beta()
        self.beam_current = curr  # beam current [mA]
        self.bunch_frequency = freq  # RF frequency [Hz]

    def get_kin_energy(self):
        return self.bunch.getSyncParticle().kinEnergy()

    def set_kin_energy(self, e_kin=0.0025):
        self.bunch.getSyncParticle().kinEnergy(e_kin)
        self.beta = self.bunch.getSyncParticle().beta()

    def get_z_to_phase_coeff(self):
        """Returns the coefficient to calculate phase in degrees from the z-coordinate."""
        self.rf_wavelength = consts.speed_of_light / self.bunch_frequency
        bunch_lambda = self.beta * self.rf_wavelength
        phase_coeff = -360.0 / bunch_lambda
        return phase_coeff

    def get_beam_current(self):
        """Returns the beam currect [mA]."""
        return self.beam_current

    def set_beam_current(self, current):
        """Set the beam currect [mA]."""
        self.beam_current = current

    def calc_macro_size(self, bunch):
        macrosize = self.beam_current * 1.0e-3 / self.bunch_frequency
        macrosize /= math.fabs(bunch.charge()) * consts.charge_electron
        macrosize /= bunch.getSizeGlobal()
        return macrosize

    def get_bunch(self, filename):
        bunch = Bunch()
        self.bunch.copyEmptyBunchTo(bunch)
        bunch.readBunch(filename)
        return bunch

    def read_parmteq_bunch(self, filename):
        """Reads bunch with expected header format:

        Number of particles    =
        Beam current           =
        RF Frequency           =
        The input file particle coordinates were written in double precision.
        x(cm)             xpr(=dx/ds)       y(cm)             ypr(=dy/ds)       phi(radian)        W(MeV)
        """
        # -- read header
        header = np.genfromtxt(filename, max_rows=3, usecols=[0, 1, 2, 3, 4], dtype=str)
        n_parts = int(header[0, 4])
        current = np.float(header[1, 3])
        freq = np.float(header[2, 3])

        # -- read data
        data = np.loadtxt(filename, skiprows=5)
        # -- trim off-energy particles
        e_kin = np.mean(data[:, 5])  # center energy
        ind = np.where(np.abs(data[:, 5] - e_kin) < (0.05 * e_kin))[0]
        n_parts = len(ind)

        # -- get particle coordinates
        de = (data[ind, 5] - e_kin) * 1e-3  # MeV to GeV
        x = data[ind, 0] * 1e-2  # cm to m
        xp = data[ind, 1]  # radians
        y = data[ind, 2] * 1e-2  # cm to m
        yp = data[ind, 3]  # radians
        phi = data[ind, 4]  # radians

        # -- make bunch
        bunch = Bunch()
        self.bunch.copyEmptyBunchTo(bunch)

        # -- change bunch attributes based on input distribution
        self.bunch_frequency = freq * 1e6
        self.set_kin_energy(e_kin=e_kin * 1e-3)
        phase_coeff = self.get_z_to_phase_coeff()
        z = np.rad2deg(phi) / phase_coeff

        # -- add particles to bunch
        for i in range(n_parts):
            bunch.addParticle(x[i], xp[i], y[i], yp[i], z[i], de[i])

        # assign current
        self.set_beam_current(current)
        macrosize = self.calc_macro_size(bunch)
        bunch.macroSize(macrosize)

        return bunch

    def dump_parmila_file(self, **kwargs):
        """Dump the Parmila bunch into the file.

        Optional arguments:
        bunch = bunch instance, default self.bunch
        phase_init, default -45 deg
        filename, default "./GeneratedInputBunch_parmila.txt"
        """
        bunch = kwargs.get("bunch", self.bunch)
        phase_init = kwargs.get("phase_init", -45.0)
        fileName = kwargs.get("filename", "GeneratedInputBunch_parmila.txt")

        e_kin = bunch.getSyncParticle().kinEnergy()
        n_particles = bunch.getSize()
        n_parts_global = bunch.getSizeGlobal()
        beam_current = (
            bunch.macroSize() * n_parts_global * self.bunch_frequency * 1.0e3
        ) * (math.fabs(bunch.charge()) * consts.charge_electron)
        parmila_out = open(fileName, "w")
        parmila_out.write("Parmila data from *****  Generated by pyORBIT \n")
        parmila_out.write("Structure number       =          1 \n")
        parmila_out.write("Cell or element number =          0 \n")
        parmila_out.write("Design particle energy =%11.6g     MeV \n" % e_kin)
        parmila_out.write("Number of particles    =%11d           \n" % n_particles)
        parmila_out.write("Beam current           =%11.7f         \n" % beam_current)
        parmila_out.write("RF Frequency           =   402.5000     MHz \n")
        parmila_out.write("Bunch Freq             =   402.5000     MHz \n")
        parmila_out.write("Chopper fraction       =   0.680000  \n")
        parmila_out.write(
            "The input file particle coordinates were written in double precision. \n"
        )
        parmila_out.write(
            "   x(cm)             xpr(=dx/ds)       y(cm)             ypr(=dy/ds)       phi(radian)        W(MeV) \n"
        )
        part_wave_lenghth = (
            consts.speed_of_light
            / self.bunch_frequency
            * bunch.getSyncParticle().beta()
        )
        for i in range(n_particles):
            (x, xp, y, yp, z, dE) = (
                bunch.x(i),
                bunch.xp(i),
                bunch.y(i),
                bunch.yp(i),
                bunch.z(i),
                bunch.dE(i),
            )
            phi = 2 * np.pi * (z / part_wave_lenghth + phase_init / 360.0)
            kinE = (dE + e_kin) * 1.0e3  # we need in [MeV], but pyORBIT is in [GeV]
            x = x * 100.0  # pyORBIT in [m] and intermediate file for Parmila in [cm]
            y = y * 100.0  # pyORBIT in [m] and intermediate file for Parmila in [cm]
            xp = xp  # pyORBIT in [rad] and intermediate file for Parmila in [rad]
            yp = yp  # pyORBIT in [rad] and Pintermediate file for armila in [rad]
            parmila_out.write(
                "%18.11g%18.11g%18.11g%18.11g%18.11g%18.11g \n"
                % (x, xp, y, yp, phi, kinE)
            )
        parmila_out.close()

    def center_bunch(self, bunch):
        """Calculate bunch means and shift to center on 0."""
        n_parts_total = bunch.getSizeGlobal()
        if n_parts_total == 0.0:
            return
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        main_rank = 0
        data_type = mpi_datatype.MPI_DOUBLE
        n_parts = bunch.getSize()
        (x_avg, xp_avg, y_avg, yp_avg, z_avg, dE_avg) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(n_parts):
            x_avg += bunch.x(i)
            xp_avg += bunch.xp(i)
            y_avg += bunch.y(i)
            yp_avg += bunch.yp(i)
            z_avg += bunch.z(i)
            dE_avg += bunch.dE(i)
        (x_avg, xp_avg, y_avg, yp_avg, z_avg, dE_avg) = orbit_mpi.MPI_Allreduce(
            (x_avg, xp_avg, y_avg, yp_avg, z_avg, dE_avg),
            data_type,
            mpi_op.MPI_SUM,
            comm,
        )
        x_avg /= n_parts_total
        xp_avg /= n_parts_total
        y_avg /= n_parts_total
        yp_avg /= n_parts_total
        z_avg /= n_parts_total
        dE_avg /= n_parts_total
        for i in range(n_parts):
            bunch.x(i, bunch.x(i) - x_avg)
            bunch.xp(i, bunch.xp(i) - xp_avg)
            bunch.y(i, bunch.y(i) - y_avg)
            bunch.yp(i, bunch.yp(i) - yp_avg)
            bunch.z(i, bunch.z(i) - z_avg)
            bunch.dE(i, bunch.dE(i) - dE_avg)

        msg = (
            "Centered bunch, shifted coordinates by: \n"
            + "dx  = %.4f mm, \n"
            + "dy  = %.4f mm, \n"
            + "dxp = %.4f mrad, \n"
            + "dyp = %.4f mrad, \n"
            + "dz  = %.4f mm, \n"
            + "dE  = %.4f keV "
        ) % (
            x_avg * 1e3,
            y_avg * 1e3,
            xp_avg * 1e3,
            yp_avg * 1e3,
            z_avg * 1e3,
            dE_avg * 1e6,
        )
        print(msg)
        return bunch


class BunchGenerator(Base_BunchGenerator):
    """Generates the pyORBIT BTF Linac Bunches.

    Twiss parameters has the fol following units: x in [m], xp in [rad]
    and the X and Y emittances are un-normalized. The longitudinal emittance
    is in [GeV*m].
    """
    def __init__(
        self,
        twiss_x,
        twiss_y,
        twiss_z,
        mass=0.939294,
        charge=-1,
        ekin=0.0025,
        curr=40.0,
        freq=402.5e6,
    ):
        self.twiss = (twiss_x, twiss_y, twiss_z)
        super(BunchGenerator, self).__init__(
            mass=mass, charge=charge, ekin=ekin, curr=curr, freq=freq
        )

    def get_bunch(self, n_parts=0, dist_class=WaterBagDist3D, cut_off=-1.0):
        """Return bunch with particular number of particles."""
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        rank = orbit_mpi.MPI_Comm_rank(comm)
        size = orbit_mpi.MPI_Comm_size(comm)
        data_type = mpi_datatype.MPI_DOUBLE
        main_rank = 0
        bunch = Bunch()
        self.bunch.copyEmptyBunchTo(bunch)
        macrosize = self.beam_current * 1.0e-3 / self.bunch_frequency
        macrosize /= math.fabs(bunch.charge()) * consts.charge_electron
        distributor = None
        if dist_class in [WaterBagDist3D, KVDist3D]:
            distributor = dist_class(self.twiss[0], self.twiss[1], self.twiss[2])
        else:
            distributor = dist_class(
                self.twiss[0], self.twiss[1], self.twiss[2], cut_off
            )
        bunch.getSyncParticle().time(0.0)
        for i in range(n_parts):
            (x, xp, y, yp, z, dE) = distributor.getCoordinates()
            (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
                (x, xp, y, yp, z, dE), data_type, main_rank, comm
            )
            if i % size == rank:
                bunch.addParticle(x, xp, y, yp, z, dE)
        n_parts_global = bunch.getSizeGlobal()
        bunch.macroSize(macrosize / n_parts_global)
        return bunch


class BunchGeneratorTransverse(Base_BunchGenerator):
    """Generates the pyORBIT BTF Linac Bunches.

    The transverse phase spaces are generated by PhaseSpaceGenerator2D class instances
    The longitudinal phase space is generated according to designated distribution function,
    defined as Twiss object. Emittance is in [GeV*m].
    """
    def __init__(
        self,
        phase_sp_gen_x,
        phase_sp_gen_y,
        twiss_z,
        mass=0.939294,
        charge=-1,
        ekin=0.0025,
        curr=40.0,
        freq=402.5e6,
        method="cdf",
    ):
        """
        method = 'cdf' (default): Uses cumulative distribution function
        to pick coordinates for a randomly generated probability.
        (aka inverse transform sampling)
        Gives precise number of particles requested (n_parts)

        method = 'grid': deposits particles on a grid according to
        probability density function, applies random kernel to 'shake'
        particles away from grid points.
        Returns slightly fewer particles than requested ( < n_parts)
        """
        self.twiss_z = twiss_z
        self.phase_sp_gen_x = phase_sp_gen_x
        self.phase_sp_gen_y = phase_sp_gen_y
        self.method = method
        super(BunchGeneratorTransverse, self).__init__(
            mass=mass, charge=charge, ekin=ekin, curr=curr, freq=freq
        )

    def get_bunch(self, n_parts=0, dist_class=WaterBagDist1D, cut_off=-1.0):
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        rank = orbit_mpi.MPI_Comm_rank(comm)
        size = orbit_mpi.MPI_Comm_size(comm)
        data_type = mpi_datatype.MPI_DOUBLE
        main_rank = 0
        bunch = Bunch()
        self.bunch.copyEmptyBunchTo(bunch)
        macrosize = self.beam_current * 1.0e-3 / self.bunch_frequency
        macrosize /= math.fabs(bunch.charge()) * consts.charge_electron
        # -- distributor for z-distribution
        distributor = None
        if dist_class in [WaterBagDist1D, KVDist1D]:
            distributor = dist_class(self.twiss_z)
        else:
            distributor = dist_class(self.twiss_z, cut_off)
        bunch.getSyncParticle().time(0.0)

        if self.method == "cdf":
            for i in range(n_parts):
                (z, dE) = distributor.getCoordinates()
                (x, xp) = self.phase_sp_gen_x.get_x_xp()
                (y, yp) = self.phase_sp_gen_y.get_y_yp()
                (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        elif self.method == "grid":
            # Create x and y pdfs.
            self.phase_sp_gen_x.gen_pdf()
            self.phase_sp_gen_y.gen_pdf()
            # Sample from the pdfs.
            xcoord, xpcoord = self.phase_sp_gen_x.grid_sample(n_parts=n_parts)
            ycoord, ypcoord = self.phase_sp_gen_y.grid_sample(n_parts=n_parts)
            # Add particles to the bunch. (This is necessary because the
            # grid method does not return exact n_parts.)
            n_parts = min(len(xcoord), len(ycoord))
            for i in range(n_parts):
                z, dE = distributor.getCoordinates()
                x, xp = (xcoord[i], xpcoord[i])
                y, yp = (ycoord[i], ypcoord[i])
                x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), 
                    data_type, 
                    main_rank, 
                    comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        else:
            raise ValueError(
                "'{}' is not an available method for transverse 2D bunch generation"
                .format(self.method)
            )
        n_parts_global = bunch.getSizeGlobal()
        bunch.macroSize(macrosize / n_parts_global)
        return bunch


class BunchGenerator6D(Base_BunchGenerator):
    """Generates the pyORBIT BTF Linac Bunches.

    The x-x', y-y', and z-dE coordinates each individually generated
    by PhaseSpaceGenerator2D class instances.
    """
    def __init__(
        self,
        phase_sp_gen_x,
        phase_sp_gen_y,
        phase_sp_gen_z,
        mass=0.939294,
        charge=-1,
        ekin=0.0025,
        curr=40.0,
        freq=402.5e6,
        method="cdf",
    ):
        """
        method = 'cdf' (default): Uses the cumulative distribution 
        function to pick coordinates for a randomly generated probability.
        (aka inverse transform sampling). This returns the precise number 
        of requested particles.

        method = 'grid': Deposits particles on a grid according to probability
        density function, applies random kernel to 'shake' particles away from 
        grid points. This will return slightly fewer particles than requested.
        
        To do: move sampling routines to PhaseSpaceGenerator6D.
        """
        self.phase_sp_gen_z = phase_sp_gen_z
        self.phase_sp_gen_x = phase_sp_gen_x
        self.phase_sp_gen_y = phase_sp_gen_y
        self.method = method
        super(BunchGenerator6D, self).__init__(
            mass=mass, charge=charge, ekin=ekin, curr=curr, freq=freq
        )

    def get_bunch(self, n_parts=0, **kwargs):
        """Returns the pyORBIT bunch with particular number of particles."""
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        rank = orbit_mpi.MPI_Comm_rank(comm)
        size = orbit_mpi.MPI_Comm_size(comm)
        data_type = mpi_datatype.MPI_DOUBLE
        main_rank = 0
        bunch = Bunch()
        self.bunch.copyEmptyBunchTo(bunch)
        macrosize = self.beam_current * 1.0e-3 / self.bunch_frequency
        macrosize /= math.fabs(bunch.charge()) * consts.charge_electron

        bunch.getSyncParticle().time(0.0)

        # -- deposit particles according to CDF method
        if self.method == "cdf":
            for i in range(n_parts):
                (z, dE) = self.phase_sp_gen_z.get_z_zp()
                (x, xp) = self.phase_sp_gen_x.get_x_xp()
                (y, yp) = self.phase_sp_gen_y.get_y_yp()
                (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        # -- deposit particles according to Grid-based method
        elif self.method == "grid":
            # -- create pdf's for x and y distributions
            self.phase_sp_gen_z.gen_pdf()
            self.phase_sp_gen_x.gen_pdf()
            self.phase_sp_gen_y.gen_pdf()
            # -- sample distributions
            (zcoord, zpcoord) = self.phase_sp_gen_z.grid_sample(n_parts=n_parts)
            (xcoord, xpcoord) = self.phase_sp_gen_x.grid_sample(n_parts=n_parts)
            (ycoord, ypcoord) = self.phase_sp_gen_y.grid_sample(n_parts=n_parts)
            # -- add particles to bunch
            n_parts = min(
                [len(xcoord), len(ycoord), len(zcoord)]
            )  # grid method does not return exact n_parts
            for i in range(n_parts):
                (z, dE) = (zcoord[i], zpcoord[i])
                (x, xp) = (xcoord[i], xpcoord[i])
                (y, yp) = (ycoord[i], ypcoord[i])
                (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        else:
            raise ValueError(
                "'{}' is not an available method for transverse 2D bunch generation."
                .format(self.method)
            )
        n_parts_global = bunch.getSizeGlobal()
        bunch.macroSize(macrosize / n_parts_global)
        return bunch