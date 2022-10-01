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
    def __init__(self, mass=0.939294, charge=-1, ekin=0.0025, current=40.0, freq=402.5e6):
        self.bunch = Bunch()
        self.bunch.mass(mass)
        self.bunch.charge(charge)
        self.bunch.getSyncParticle().kinEnergy(ekin)
        self.current = current  # beam current [mA]
        self.frequency = freq  # RF frequency [Hz]

    def get_z_to_phase_coeff(self):
        """Returns the coefficient to calculate phase in degrees from the z-coordinate."""
        self.rf_wavelength = consts.speed_of_light / self.frequency
        bunch_lambda = self.beta * self.rf_wavelength
        phase_coeff = -360.0 / bunch_lambda
        return phase_coeff

    def get_macro_size(self, bunch):
        # Why not use bunch.macroSize()?
        macro_size = self.current * 1.0e-3 / self.frequency
        macro_size /= math.fabs(bunch.charge()) * consts.charge_electron
        macro_size /= bunch.getSizeGlobal()
        return macro_size

    def read_pyorbit(self, filename):
        bunch = Bunch()
        self.bunch.copyEmptyBunchTo(bunch)
        bunch.readBunch(filename)
        return bunch

    def read_parmteq(self, filename):
        """Reads bunch with expected header format:

        Number of particles    =
        Beam current           =
        RF Frequency           =
        The input file particle coordinates were written in double precision.
        x(cm)             xpr(=dx/ds)       y(cm)             ypr(=dy/ds)       phi(radian)        W(MeV)
        """
        # Read header.
        header = np.genfromtxt(filename, max_rows=3, usecols=[0, 1, 2, 3, 4], dtype=str)
        n_parts = int(header[0, 4])
        current = np.float(header[1, 3])
        freq = np.float(header[2, 3])

        # Read data, triming off-energy particles.
        data = np.loadtxt(filename, skiprows=5)
        kin_energy = np.mean(data[:, 5])  # center energy
        ind = np.where(np.abs(data[:, 5] - kin_energy) < (0.05 * kin_energy))[0]
        n_parts = len(ind)

        # -- get particle coordinates
        de = (data[ind, 5] - kin_energy) * 1e-3  # MeV to GeV
        x = data[ind, 0] * 1e-2  # cm to m
        xp = data[ind, 1]  # radians
        y = data[ind, 2] * 1e-2  # cm to m
        yp = data[ind, 3]  # radians
        phi = data[ind, 4]  # radians

        # Change bunch attributes based on input distribution.
        bunch = Bunch()
        self.bunch.copyEmptyBunchTo(bunch)
        self.frequency = freq * 1e6
        self.getSyncParticle().kinEnergy(kin_energy * 1e-3)
        phase_coeff = self.get_z_to_phase_coeff()
        z = np.rad2deg(phi) / phase_coeff

        # Add particles to bunch.
        for i in range(n_parts):
            bunch.addParticle(x[i], xp[i], y[i], yp[i], z[i], de[i])

        # Set beam current.
        self.set_beam_current(current)
        macrosize = self.calc_macro_size(bunch)
        bunch.macroSize(macrosize)
        return bunch


class BunchGeneratorTwiss(Base_BunchGenerator):
    """Generate bunch from {twiss_x, twiss_y, twiss_z}."""
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
        self.twiss_x = twiss_x
        self.twiss_y = twiss_y
        self.twiss_z = twiss_z
        super(BunchGeneratorTwiss, self).__init__(
            mass=mass, charge=charge, ekin=ekin, curr=curr, freq=freq,
        )

    def get_bunch(self, n_parts=0, dist_class=WaterBagDist3D, **kws):
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        rank = orbit_mpi.MPI_Comm_rank(comm)
        size = orbit_mpi.MPI_Comm_size(comm)
        data_type = mpi_datatype.MPI_DOUBLE
        main_rank = 0
        bunch = Bunch()
        self.bunch.copyEmptyBunchTo(bunch)
        macrosize = self.current * 1.0e-3 / self.frequency
        macrosize /= math.fabs(bunch.charge()) * consts.charge_electron
        distributor = dist_class(self.twiss[0], self.twiss[1], self.twiss[2], **kws)
        bunch.getSyncParticle().time(0.0)
        for i in range(n_parts):
            x, xp, y, yp, z, dE = distributor.getCoordinates()
            x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
                (x, xp, y, yp, z, dE), 
                data_type, main_rank, comm,
            )
            if i % size == rank:
                bunch.addParticle(x, xp, y, yp, z, dE)
        bunch.macroSize(macrosize / bunch.getSizeGlobal())
        return bunch


class BunchGenerator_XXP_YYP_TwissZ(Base_BunchGenerator):
    """Generate bunch from {f(x, x'), f(y, y'), twiss_z}."""
    def __init__(
        self,
        phase_space_gen_x,
        phase_space_gen_y,
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
        self.phase_space_gen_x = phase_space_gen_x
        self.phase_space_gen_y = phase_space_gen_y
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
        macrosize = self.current * 1.0e-3 / self.frequency
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
                (x, xp) = self.phase_space_gen_x.get_x_xp()
                (y, yp) = self.phase_space_gen_y.get_y_yp()
                (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        elif self.method == "grid":
            # Create x and y pdfs.
            self.phase_space_gen_x.gen_pdf()
            self.phase_space_gen_y.gen_pdf()
            # Sample from the pdfs.
            xcoord, xpcoord = self.phase_space_gen_x.grid_sample(n_parts=n_parts)
            ycoord, ypcoord = self.phase_space_gen_y.grid_sample(n_parts=n_parts)
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


class BunchGenerator_XXP_YYP_ZZP(Base_BunchGenerator):
    """Generate bunch from {f(x, x'), f(y, y'), f(z, dE)}."""
    def __init__(
        self,
        phase_space_gen_x,
        phase_space_gen_y,
        phase_space_gen_z,
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
        self.phase_space_gen_z = phase_space_gen_z
        self.phase_space_gen_x = phase_space_gen_x
        self.phase_space_gen_y = phase_space_gen_y
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
        macrosize = self.current * 1.0e-3 / self.frequency
        macrosize /= math.fabs(bunch.charge()) * consts.charge_electron

        bunch.getSyncParticle().time(0.0)

        if self.method == "cdf":
            for i in range(n_parts):
                z, dE = self.phase_space_gen_z.get_z_zp()
                x, xp = self.phase_space_gen_x.get_x_xp()
                y, yp = self.phase_space_gen_y.get_y_yp()
                x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        elif self.method == "grid":
            self.phase_space_gen_z.gen_pdf()
            self.phase_space_gen_x.gen_pdf()
            self.phase_space_gen_y.gen_pdf()
            zcoord, zpcoord = self.phase_space_gen_z.grid_sample(n_parts=n_parts)
            xcoord, xpcoord = self.phase_space_gen_x.grid_sample(n_parts=n_parts)
            ycoord, ypcoord = self.phase_space_gen_y.grid_sample(n_parts=n_parts)
            n_parts = min(len(xcoord), len(ycoord), len(zcoord))  
            for i in range(n_parts):
                z, dE = zcoord[i], zpcoord[i]
                x, xp = xcoord[i], xpcoord[i]
                y, yp = ycoord[i], ypcoord[i]
                x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        else:
            raise ValueError(
                "'{}' is not an available method for transverse 2D bunch generation."
                .format(self.method)
            )
        bunch.macroSize(macrosize / bunch.getSizeGlobal())
        return bunch