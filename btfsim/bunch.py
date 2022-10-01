"""Generate/manipulate pyorbit bunches.

Note that some routines are parallel but not efficient.
"""
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


def bunch_coord_array(bunch):
    """Return Nx6 coordinate array from Bunch."""
    X = np.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        X[i, 0] = bunch.x(i)
        X[i, 1] = bunch.xp(i)
        X[i, 2] = bunch.y(i)
        X[i, 3] = bunch.yp(i)
        X[i, 4] = bunch.z(i)
        X[i, 5] = bunch.dE(i)
    return X


class BunchManager:
    """Class to manipulate Bunch object."""
    def __init__(self, bunch=None):
        self.bunch = bunch
        
    def decimate(self, factor=1):
        """Reduce the number of macro-particles in the bunch.

        This just skips every `fac` indices, so we assume that the bunch
        coordinates were generated randomly.
        """
        n_parts0 = self.bunch.getSizeGlobal()
        if not factor or not (1 <= factor < n_parts0):
            print("No decimation for fac={}.".format(factor))
        print('Decimating bunch by factor {}...'.format(factor))
        new_bunch = Bunch()
        self.bunch.copyEmptyBunchTo(new_bunch)
        for i in range(0, n_parts0, factor):
            new_bunch.addParticle(
                self.bunch.x(i), self.bunch.xp(i),
                self.bunch.y(i), self.bunch.yp(i),
                self.bunch.z(i), self.bunch.dE(i),
            )
        new_bunch.macroSize(factor * self.bunch.macroSize())
        new_bunch.copyBunchTo(self.bunch)
        print('Done decimating bunch.')
    
    def resample(self, n_parts=1, rms_factor=0.05):
        """Up/down-sample to obtain requested number of particles.

        Upsampling should probably be done by sampling from estimated pdf...

        Parameters
        ----------
        bunch : Bunch
            The pyorbit bunch to resample.
        n_parts : int
            The number of desired particles in the bunch.
        """
        n_parts0 = self.bunch.getSizeGlobal()
        mult = float(n_parts) / float(n_parts0)
        n_parts = int(n_parts)
        print("Resampling bunch from {} to {} particles...".format(n_parts0, n_parts))
        print("mult = {:.3f}".format(mult))

        if mult == 1:
            return []

        coords0 = bunch_coord_array(self.bunch)
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
        newmacrosize = self.bunch.macroSize() * (1.0 / mult)
        newbunch.macroSize(newmacrosize)
        # -- over-write bunch_in
        newbunch.copyBunchTo(self.bunch)
        print("Done resampling.")

    def attenuate(self, fac=1.0):
        """Adjust current by changing macrosize

        att : float
            The fractional attenuation.
        """
        self.bunch.macroSize(fac * self.bunch.macroSize())

    def decorrelate_x_y_z(self):
        """Remove inter-plane correlations by permuting (x, x'), (y, y'), (z, z') pairs."""
        X = bunch_coord_array(self.bunch)
        for i in (0, 2, 4):
            idx = np.random.permutation(np.arange(X.shape[0]))
            X[:, i : i + 2] = X[idx, i : i + 2]
        for i, (x, xp, y, yp, z, dE) in enumerate(X):
            self.bunch.x(i, x)
            self.bunch.y(i, y)
            self.bunch.z(i, z)
            self.bunch.xp(i, xp)
            self.bunch.yp(i, yp)
            self.bunch.dE(i, dE)

    def shift(self, x, xp, y, yp, z, dE):
        print('Shifting bunch centroid...')
        for i in range(self.bunch.getSize()):
            self.bunch.x(i, self.bunch.x(i) + x)
            self.bunch.y(i, self.bunch.y(i) + y)  
            self.bunch.z(i, self.bunch.z(i) + z)
            self.bunch.xp(i, self.bunch.xp(i) + xp) 
            self.bunch.yp(i, self.bunch.yp(i) + yp)
            self.bunch.dE(i, self.bunch.dE(i) + dE)
        print('Bunch shifted.')
        
    def center(self):
        """Shift the bunch so that first-order moments are zero."""
        twiss = BunchTwissAnalysis()
        twiss.analyzeBunch(self.bunch)
        self.shift(*[twiss.getAverage(i) for i in range(6)])
        
    def reverse(self):
        """Reverse the bunch propagation direction.

        Since the tail becomes the head of the bunch, the sign of z
        changes but the sign of dE does not change.
        """
        for i in range(bunch.getSize()):
            bunch.xp(i, -bunch.xp(i))
            bunch.yp(i, -bunch.yp(i))
            bunch.z(i, -bunch.z(i))
        
    def dump(self, filename):
        print('Writing bunch coordinates to {}'.format(filename))
        self.bunch.dumpBunch(filename)
        print('Done writing bunch coordinates.')
        
        
class BunchCalculator:
    def __init__(self, bunch, file=None):
        if isinstance(bunch, Bunch):
            self.bunch = bunch
        elif type(bunch) == str:  # load bunch from file
            self.bunch = Bunch()
            self.bunch.readBunch(bunch)
        self.coords = bunch_coord_array(bunch)
        self.coords[:, :4] *= 1e3  # mm, mrad, MeV
        self.coords[:, 5] *= 1e6  # keV
        self.cov = np.cov(self.coords.T)
        self.filename = file
        self.twiss_analysis = BunchTwissAnalysis()
        self.twiss_analysis.analyzeBunch(bunch)
        self.n_parts = bunch.getSizeGlobal()
        self.gamma = bunch.getSyncParticle().gamma()
        self.beta = bunch.getSyncParticle().beta()
        self.mass = bunch.getSyncParticle().mass()

    def twiss(self, dim="x", emit_norm_flag=False):
        """Return rms 2D Twiss parameters."""
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
            "beta": {"value": beta, "unit": "mm/mrad"},
            "alpha": {"value": alpha, "unit": ""},
            "eps": {"value": eps, "unit": "mm-mrad"},
            "disp": {"value": disp, "unit": "m"},
            "dispp": {"value": dispp, "unit": ""},
        }

    def norm_coords(self, scale_emittance=False):
        """Return coordinates normalized by rms Twiss parameters in x-x', y-y', z-z'."""
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

    def radial_density(self, dr=0.1, dim="x"):
        raise NotImplementedError
        
        
class BunchCompare:
    """Class to compare two bunches.
    
    Attributes
    ----------
    """
    def __init__(self, bunch1, bunch2):
        if isinstance(bunch1, Bunch):
            self.bunch1 = bunch1
        elif type(bunch1) == str:
            self.bunch1 = Bunch()
            self.bunch1.readBunch(bunch1)
            print("Loading bunch {}".format(bunch1))
        if isinstance(bunch2, Bunch):
            self.bunch2 = bunch2
        elif type(bunch2) == str:
            self.bunch2 = Bunch()
            self.bunch2.readBunch(bunch2)
            print("Loading bunch {}".format(bunch2))

        self.coord1 = bunch_coord_array(bunch1)
        self.coord2 = bunch_coord_array(bunch2)
        self.coordlist = ["x", "xp", "y", "yp", "z", "dE"]

        self.mins = np.min(
            np.vstack([np.min(self.coord1, axis=0), np.min(self.coord1, axis=0)]),
            axis=0,
        )
        self.mins = np.max(
            np.vstack([np.max(self.coord1, axis=0), np.max(self.coord1, axis=0)]),
            axis=0,
        )

    def compare2D(self, d1="x", d2="xp", nbins=10):
        if type(nbins) == int:
            nbins = np.repeat(nbins, 2)

        i1 = self.coordlist.index(d1)
        i2 = self.coordlist.index(d2)

        bins = []
        bins.append(np.linspace(self.mins[i1], self.maxs[i1], nbins[0]))
        bins.append(np.linspace(self.mins[i2], self.maxs[i2], nbins[1]))

        d = self.getHistDiff(bins, cols=[i1, i2])
        return d

    def compare6D(self, nbins=10):
        if type(nbins) == int:
            nbins = np.repeat(nbins, 6)

        bins = []
        for i in range(6):
            bins.append(np.linspace(self.mins[i], self.maxs[i], nbins[i]))

        d = self.getHistDiff(bins)
        return d

    def get_hist_diff(self, bins, cols=range(6)):
        H1, edges1 = np.histogramdd(self.coord1[:, cols], bins=bins, density=True)
        H2, edges2 = np.histogramdd(self.coord2[:, cols], bins=bins, density=True)

        diff = np.abs(H1.flatten() - H2.flatten())
        sum = H1.flatten() + H2.flatten()
        ind = np.where(sum != 0)[0]

        d = np.sum(diff[ind] / sum[ind])
        return d
    
    
class Beamlet:
    """Class to create beamlet out of specified bunch distribution.

    optional arguments:

    center of slice:
    x,y [mm]
    xp,yp [mrad]
    z [mm]
    dE [keV]

    slice width:
    xwidth,ywidth = .200 mm
    xpwidth,ypwidth = .200 mm/L
    L = 0.947 (HZ04-->HZ06 slit separation)
    zwidth = 2 deg. (~BSM resolution)
    dEwidth = 0.4 keV (~energy slit resolution

    """

    def __init__(self, bunch_in, z2phase, **kwargs):
        self.z2phase = z2phase
        self.bunch_in = bunch_in

    def slice(self, **kwargs):

        # -- location of bunch slice
        xslice = kwargs.get("x", None)
        xpslice = kwargs.get("xp", None)
        yslice = kwargs.get("y", None)
        ypslice = kwargs.get("yp", None)
        zslice = kwargs.get("z", None)
        dEslice = kwargs.get("dE", None)

        # -- physical width of slits [mm]
        xw = kwargs.get("xwidth", 0.2)
        xpw = kwargs.get("xpwidth", 0.2)
        yw = kwargs.get("ywidth", 0.2)
        ypw = kwargs.get("ypwidth", 0.2)

        # -- width of z in deg.
        zw = kwargs.get("zwidth", 0.4)  # close to 1 pixel width

        # -- width of dE in keV
        dEw = kwargs.get("dEwidth", 2)
        # per Cathey thesis, energy uncertainty is
        # ~1.3 keV for 0.8 mm slit, 0.6 keV for screen

        # -- distance between transverse slits
        L = kwargs.get("L", 0.947)  # [m]
        # L = L*1e3 #[convert to mm]
        # Ldipoo2slit = 0.129 # distance dipole exit to VS06 (energy) slit
        # rho = 0.3556 # dipole bending radius
        # Lslit2dipo = 1.545

        # -- convert to bunch units (meters, rad, GeV)
        xw = 0.5 * xw
        xpw = 0.5 * xpw / L
        yw = 0.5 * yw
        ypw = 0.5 * ypw / L
        dEw = 0.5 * dEw
        zw = 0.5 * zw

        # -- be verbose, also convert to [m, rad, GeV]
        print("selecting in:")
        if not (xslice is None):
            print("%.6f < x < %.6f mm" % (xslice - xw, xslice + xw))
            xslice *= 1e-3
            xw *= 1e-3
        if not (xpslice is None):
            print("%.6f < x' < %.6f mrad" % (xpslice - xpw, xpslice + xpw))
            xpslice *= 1e-3
            xpw *= 1e-3
        if not (yslice is None):
            print("%.6f < y < %.6f mm" % (yslice - yw, yslice + yw))
            yslice *= 1e-3
            yw *= 1e-3
        if not (ypslice is None):
            print("%.6f < y' < %.6f mrad" % (ypslice - ypw, ypslice + ypw))
            ypslice *= 1e-3
            ypw *= 1e-3
        if not (zslice is None):
            print("%.6f < z < %.6f deg" % (zslice - zw, zslice + zw))
            zslice /= self.z2phase
            zw /= self.z2phase
        if not (dEslice is None):
            print("%.6f < dE < %.6f keV" % (dEslice - dEw, dEslice + dEw))
            dEslice *= 1e-6
            dEw *= 1e-6

        n = self.bunch_in.getSizeGlobal()
        beamlet = Bunch()  # make new empty
        for i in range(n):
            x = self.bunch_in.x(i)
            xp = self.bunch_in.xp(i)
            y = self.bunch_in.y(i)
            yp = self.bunch_in.yp(i)
            z = self.bunch_in.z(i)
            dE = self.bunch_in.dE(i)
            # -- check each dimension to see if particle is within slice,
            # if slice is specified.
            if not (xslice is None):
                if not (x < xslice + xw and x > xslice - xw):
                    continue
            if not (xpslice is None):
                if not (xp < xpslice + xpw and xp > xpslice - xpw):
                    continue
            if not (yslice is None):
                if not (y < yslice + yw and y > yslice - yw):
                    continue
            if not (ypslice is None):
                if not (yp < ypslice + ypw and yp > ypslice - ypw):
                    continue
            if not (zslice is None):
                if not (z < zslice + zw and z > zslice - zw):
                    continue
            if not (dEslice is None):
                if not (dE < dEslice + dEw and dE > dEslice - dEw):
                    continue
            beamlet.addParticle(x, xp, y, yp, z, dE)

        print("Beamlet has %i particles" % beamlet.getSizeGlobal())
        return beamlet
    
    
class AdaptiveWeighting:
    """Dynamic macro-particle weight adjustment.
    
    Attributes
    ----------
    """
    def __init__(self, z2phase, macrosize0):
        self.z2phase = z2phase
        self.macrosize0 = macrosize0

        # -- initialize history arrays in hist dict
        hist_keys = ["s", "macro"]
        hist_init_len = 10000
        self.hist = dict(
            (hist_keys[k], np.zeros(hist_init_len)) for k in range(len(hist_keys))
        )
        self.hist["node"] = []

    def action_entrance(self, params_dict):
        node = params_dict["node"]
        bunch = params_dict["bunch"]
        pos = params_dict["path_length"]
        if params_dict["old_pos"] == pos:
            return
        if params_dict["old_pos"] + params_dict["pos_step"] > pos:
            return
        params_dict["old_pos"] = pos
        params_dict["count"] += 1

        # -- count how many particles inside 1 RF period
        noutside = 0
        ntotal = bunch.getSize()
        for i in range(ntotal):
            phi = bunch.z(i) * self.z2phase
            if np.abs(phi) > 180.0:  # if outside 1 RF period
                noutside += 1

        macrosize = self.macrosize0 * ntotal / (ntotal - noutside)
        bunch.macroSize(macrosize)

        self.hist["s"][params_dict["count"]] = pos
        self.hist["node"].append(node.getName())
        self.hist["macro"][params_dict["count"]] = macrosize

    def action_exit(self, params_dict):
        self.action_entrance()


class Base_BunchGenerator(object):
    """Base class for bunch generators.
    
    Attributes
    ----------
    """
    def __init__(self, mass=0.939294, charge=-1, ekin=0.0025, curr=40.0, freq=402.5e6):
        self.bunch = Bunch()
        syncPart = self.bunch.getSyncParticle()
        self.bunch.mass(mass)
        self.bunch.charge(charge)
        syncPart.kinEnergy(ekin)
        self.beta = syncPart.beta()

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
        """Returns the beam currect in mA."""
        return self.beam_current

    def set_beam_current(self, current):
        """Sets  the beam currect in mA."""
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

    def dump_bunch(self, **kwargs):
        """Dump bunch to coordinates file.

        optional args:
        bunch = bunch instance
        filename = name to send output, default "./GeneratedInputBunch.txt"
        """
        bunch = kwargs.get("bunch", self.bunch)
        fileName = kwargs.get("filename", "GeneratedInputBunch.txt")
        bunch.dumpBunch(fileName)

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
        twissX,
        twissY,
        twissZ,
        mass=0.939294,
        charge=-1,
        ekin=0.0025,
        curr=40.0,
        freq=402.5e6,
    ):
        self.twiss = (twissX, twissY, twissZ)
        super(BunchGenerator, self).__init__(
            mass=mass, charge=charge, ekin=ekin, curr=curr, freq=freq
        )

    def get_bunch(self, n_parts=0, dist_class=WaterBagDist3D, cut_off=-1.0):
        """Return bunch with particular number of particles."""
        print(
            "Generating bunch based on Twiss parameters with distributor %s"
            % dist_class.__name__
        )

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

    The transverse phase spaces are generated by  PhaseSpaceGenerator2D class instances
    The longitudinal phase space is generated according to designated distribution function,
    defined as Twiss object. Emittance is in [GeV*m].
    """
    def __init__(
        self,
        phase_sp_gen_x,
        phase_sp_gen_y,
        twissZ,
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
        self.twissZ = twissZ
        self.phase_sp_gen_x = phase_sp_gen_x
        self.phase_sp_gen_y = phase_sp_gen_y
        self.method = method
        super(BunchGeneratorTransverse, self).__init__(
            mass=mass, charge=charge, ekin=ekin, curr=curr, freq=freq
        )

    def get_bunch(self, n_parts=0, dist_class=WaterBagDist1D, cut_off=-1.0):
        """Returns the pyORBIT bunch with particular number of particles."""
        print(
            "Generating bunch based on X,Y emittance scans with %s method and Z Twiss parameters with distributor %s"
            % (self.method, dist_class.__name__)
        )

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
            distributor = dist_class(self.twissZ)
        else:
            distributor = dist_class(self.twissZ, cut_off)
        bunch.getSyncParticle().time(0.0)

        # --------------------------------------------------------------------------------------
        # -- deposit particles according to CDF method
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
        # -- deposit particles according to Grid-based method
        elif self.method == "grid":
            # -- create pdf's for x and y distributions
            self.phase_sp_gen_x.gen_pdf()
            self.phase_sp_gen_y.gen_pdf()
            # -- sample distributions
            (xcoord, xpcoord) = self.phase_sp_gen_x.grid_sample(n_parts=n_parts)
            (ycoord, ypcoord) = self.phase_sp_gen_y.grid_sample(n_parts=n_parts)
            # -- add particles to bunch
            n_parts = min(
                [len(xcoord), len(ycoord)]
            )  # this is necessary because grid method does not return exact n_parts
            for i in range(n_parts):
                (z, dE) = distributor.getCoordinates()
                (x, xp) = (xcoord[i], xpcoord[i])
                (y, yp) = (ycoord[i], ypcoord[i])
                (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        else:
            raise ValueError(
                "'%s' is not an available method for transverse 2D bunch generation"
                % (self.method)
            )
        # ---------------------------------------------------------------------------------------------
        n_parts_global = bunch.getSizeGlobal()
        bunch.macroSize(macrosize / n_parts_global)
        return bunch


class BunchGenerator6D(Base_BunchGenerator):
    """Generates the pyORBIT BTF Linac Bunches.

    The transverse phase spaces are generated by  PhaseSpaceGenerator2D class instances
    The longitudinal phase space is reconstructed from
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
        method = 'cdf' (default): Uses cumulative distribution function
        to pick coordinates for a randomly generated probability.
        (aka inverse transform sampling)
        Gives precise number of particles requested (n_parts)

        method = 'grid': deposits particles on a grid according to
        probability density function, applies random kernel to 'shake'
        particles away from grid points.
        Returns slightly fewer particles than requested ( < n_parts)
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
        print("6D bunch generation assuming no inter-plane correlations.")

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
                "'%s' is not an available method for transverse 2D bunch generation"
                % (self.method)
            )

        n_parts_global = bunch.getSizeGlobal()
        bunch.macroSize(macrosize / n_parts_global)
        return bunch


class PhaseSpaceGenerator2D:
    """Generates (x,x') pairs from the 2D distribution.

    The 2D distribution is read from the file.
    x and x' coordinates in the file are in [mm] and [mrad]
    The grid2D table is normalized to maximum value equals 1.
    The input file has the following structure:
    % comment line
    0.0       x1       x2       ...   x_nx
    xp1    val_1_1   val_1_2    ... val_1_nx
    xp2    val_2_1   val_2_2    ... val_2_nx
    ...
    xp_nxp val_nxp_1 val_nxp_2  ... val_nxp_nx
    """
    def __init__(self, file_name, x_max=1.0e36, xp_max=1.0e36, threshold=3e-4):

        # -- initialize variables that are defined/used in sub-functions
        self.pdf = []

        # -- separator
        if file_name[-4:] == ".txt":
            delimiter = " "
        elif file_name[-4:] == ".csv":
            delimiter = ","

        arr_in = np.genfromtxt(
            file_name, comments="%", delimiter=delimiter, filling_values=0.0
        )

        # -- [0,0] component of arr_in is linear correlation that should be inserted into bunch
        self.slope = arr_in[0, 0]

        # -- parse info from data file; get x endpoints
        res_arr = arr_in[0, 1:]
        self.nx = len(res_arr) - 1
        endpoints = np.array([float(res_arr[1]), float(res_arr[-1])])
        self.x_min = np.min(endpoints)
        self.x_max = np.max(endpoints)
        self.x_step = (self.x_max - self.x_min) / (self.nx - 1)

        xp_arr = arr_in[1:, 0]
        self.val_matrix = arr_in[1:, 1:]

        # -- threshold
        val_peak = self.val_matrix.max().max()
        self.val_matrix[self.val_matrix < val_peak * threshold] = 0.0
        # -- re-normalize to sum
        self.val_matrix /= self.val_matrix.sum().sum()
        # -- get x' endpoints
        self.nxp = len(xp_arr)
        self.xp_min = xp_arr[0]
        self.xp_max = xp_arr[len(xp_arr) - 1]
        self.xp_step = (self.xp_max - self.xp_min) / (self.nxp - 1)
        # print("debug x_min=", self.x_min, " x_max=", self.x_max,
        #       " xp_min=", self.xp_min, " xp_max=", self.xp_max)
        # print("debug xp_step=", self.xp_step, " nx=", self.nx,
        #       " nxp=", self.nxp, " lenMtrY=", len(val_matrix), " lenMtrX=", len(val_matrix[0]))
        # --------------------------------------------------------------
        # -- make 2D grid object for PDF
        self.grid2D = Grid2D(
            self.nx, self.nxp, self.x_min, self.x_max, self.xp_min, self.xp_max
        )
        self.x_min_gen = self.x_min
        self.xp_min_gen = self.xp_min
        if self.x_min_gen < -math.fabs(x_max):
            self.x_min_gen = -math.fabs(x_max)
        if self.xp_min_gen < -math.fabs(xp_max):
            self.xp_min_gen = -math.fabs(xp_max)
        self.x_max_gen = self.x_max
        self.xp_max_gen = self.xp_max
        if self.x_max_gen > math.fabs(x_max):
            self.x_max_gen = math.fabs(x_max)
        if self.xp_max_gen > -math.fabs(xp_max):
            self.xp_max_gen = math.fabs(xp_max)
        # --------------------------------------------------------------
        # -- deposit values in PDF grid
        for ix in range(self.nx):
            for ixp in range(self.nxp):
                val = self.val_matrix[ixp][ix]
                # -- assign value to 2D grid
                self.grid2D.setValue(val, ix, ixp)
        # --------------------------------------------------------------
        # -- make 2D grid for CDF
        self.int_grid2D = Grid2D(
            self.nx, self.nxp, self.x_min, self.x_max, self.xp_min, self.xp_max
        )
        for ix in range(self.nx):
            self.int_grid2D.setValue(0.0, ix, 0)
            for ixp in range(1, self.nxp):
                val_0 = self.int_grid2D.getValueOnGrid(ix, ixp - 1)
                val = (
                    self.grid2D.getValueOnGrid(ix, ixp - 1)
                    + self.grid2D.getValueOnGrid(ix, ixp)
                ) / 2.0
                self.int_grid2D.setValue(val + val_0, ix, ixp)
        self.s_int_arr = [
            0.0,
        ]
        for ix in range(1, self.nx):
            val_0 = self.s_int_arr[ix - 1]
            val = (
                self.int_grid2D.getValueOnGrid(ix - 1, self.nxp - 1)
                + self.int_grid2D.getValueOnGrid(ix, self.nxp - 1)
            ) / 2.0
            self.s_int_arr.append(val + val_0)
        s_tot = self.s_int_arr[self.nx - 1]
        if s_tot > 0.0:
            for ix in range(self.nx):
                self.s_int_arr[ix] = self.s_int_arr[ix] / s_tot
        for ix in range(self.nx):
            s_tot = self.int_grid2D.getValueOnGrid(ix, self.nxp - 1)
            if s_tot > 0.0:
                for ixp in range(self.nxp):
                    val = self.int_grid2D.getValueOnGrid(ix, ixp) / s_tot
                    self.int_grid2D.setValue(val, ix, ixp)
        # print("debug =================================gen created =============")

    # ------------------------------------------------------------------------------------------------------------
    # -- functions for method = cdf (default)

    def get_x(self):
        g = random.random()
        ind_x_0 = 0
        ind_x_1 = self.nx - 1
        count_max = 200
        count = 0
        val0 = self.s_int_arr[ind_x_0]
        val1 = self.s_int_arr[ind_x_1]
        while not abs(ind_x_1 - ind_x_0) <= 1:
            count += 1
            if count > count_max:
                print("debug problem with X generation g=", g)
                print("debug val0=", val0, " val1=", val1)
                print("debug ind_xp_0=", ind_xp_0, " ind_xp_1=", ind_xp_1)
                print("debug s_int_arr=", self.s_int_arr)
                sys.exit(1)
            if g > val0 and g <= val1:
                ind_new = int((ind_x_0 + ind_x_1) / 2)
                if g > self.s_int_arr[ind_new]:
                    ind_x_0 = ind_new
                    val0 = self.s_int_arr[ind_x_0]
                else:
                    ind_x_1 = ind_new
                    val1 = self.s_int_arr[ind_x_1]
        coeff = (g - val0) / (val1 - val0)
        x = self.x_min + (ind_x_0 + coeff) * self.x_step
        return (x, coeff, ind_x_0, ind_x_1)

    def get_xp(self, coeff, ind_x_0, ind_x_1):
        g = random.random()
        ind_x = ind_x_0
        if g > coeff:
            ind_x = ind_x_1
        g = random.random()
        ind_xp_0 = 0
        ind_xp_1 = self.nxp - 1
        count_max = 200
        count = 0
        val0 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_0)
        val1 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_1)
        if val1 == 0.0:
            if ind_x == ind_x_0:
                ind_x = ind_x_1
            else:
                ind_x = ind_x_0
        val0 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_0)
        val1 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_1)
        while not abs(ind_xp_1 - ind_xp_0) <= 1:
            count += 1
            if count > count_max:
                print("debug problem with XP generation g=", g)
                print("debug ind_x=", ind_x)
                print("debug val0=", val0, " val1=", val1)
                print("debug coeff=", coeff)
                print("debug  ind_x_0=", ind_x_0, "   ind_x_1=", ind_x_1)
                print("debug ind_xp_0=", ind_xp_0, " ind_xp_1=", ind_xp_1)
                print("debug s_int_arr[ind_x_0]=", self.s_int_arr[ind_x_0])
                print("debug s_int_arr[ind_x_1]=", self.s_int_arr[ind_x_1])
                print("debug s_int_arr=", self.s_int_arr)
                for ixp in range(self.nxp):
                    print(
                        "debug ix=",
                        ind_x_0,
                        " ixp=",
                        ixp,
                        " val0=",
                        self.int_grid2D.getValueOnGrid(ind_x_0, ixp),
                        " val1=",
                        self.int_grid2D.getValueOnGrid(ind_x_1, ixp),
                    )
                sys.exit(1)
            if g > val0 and g <= val1:
                ind_new = int((ind_xp_0 + ind_xp_1) / 2)
                if g > self.int_grid2D.getValueOnGrid(ind_x, ind_new):
                    ind_xp_0 = ind_new
                    val0 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_0)
                else:
                    ind_xp_1 = ind_new
                    val1 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_1)
        coeff = (g - val0) / (val1 - val0)
        xp = self.xp_min + (ind_xp_0 + coeff) * self.xp_step
        return xp

    def get_x_xp(self):
        count_max = 1000
        count = 0
        while -1 < 0:
            (x, coeff, ind_x_0, ind_x_1) = self.get_x()
            xp = self.get_xp(coeff, ind_x_0, ind_x_1)

            if abs(x) < self.x_max_gen and abs(xp) < self.xp_max_gen:
                # -- add in linear correlation
                x += xp * self.slope
                return (x / 1000.0, xp / 1000.0)
            count += 1
            if count > count_max:
                print("debug problem with X and XP generation count=", count)
                print("debug x=", x, " xp=", xp)
                print(
                    "debug self.x_max_gen =",
                    self.x_max_gen,
                    " self.xp_max_gen =",
                    self.xp_max_gen,
                )
                sys.exit(1)

    def get_y_yp(self):
        return self.get_x_xp()

    def get_z_zp(self):
        return self.get_x_xp()

    # ------------------------------------------------------------------------------------------------------------
    # -- functions for method = grid

    def gen_pdf(self):
        # create 2D numpy array out of 2D grid object;
        # (just normalizes scan data)
        self.pdf = self.val_matrix / self.val_matrix.sum()

    def grid_sample(self, n_parts=0):
        # make number density array out of PDF grid
        # returns arrays of x and x' coordinates
        #

        # -- make number density grid based on PDF
        Ndistr = np.floor(self.pdf * n_parts).astype(int)
        n_parts = Ndistr.sum()

        # -- define coordinate grid
        xaxis = np.linspace(self.x_min, self.x_max, num=self.nx)
        xpaxis = np.linspace(self.xp_min, self.xp_max, num=self.nxp)
        [XG, XPG] = np.meshgrid(xaxis, xpaxis)

        # -- loop through grid + deposit Ndistr(i,j) particles at each point
        # could be combined into 1 loop (do x and y simultaneously),
        # but left general in case x and y scans have different resolutions
        X, XP = np.zeros([2, n_parts])
        counter = 0
        for i in range(np.shape(Ndistr)[0]):
            for j in range(np.shape(Ndistr)[1]):
                if Ndistr[i, j] > 0:
                    counterend = counter + Ndistr[i, j]
                    X[counter:counterend] = XG[i, j]
                    XP[counter:counterend] = XPG[i, j]
                    counter = counterend

        # -- spread particles out from gridpoint locations through a uniform kernel
        # scale from mm, mrad to meters, rad
        # right now width of each kernel is twice grid spacing, is this intentional?
        x = (X + (np.random.rand(1, n_parts) - 0.5) * 2 * self.x_step) / 1000.0
        xp = (XP + (np.random.rand(1, n_parts) - 0.5) * 2 * self.xp_step) / 1000.0

        # -- random sampling to remove correlations in x-y, xp-yp
        xind = np.random.permutation(range(np.shape(x)[1]))

        # -- return shuffled x,x' distribution
        return (x[0, xind], xp[0, xind])


class PhaseSpaceGeneratorZPartial:
    """Generates (z, dE) distribution using 1D e-profile.

    Assumes Gaussian phase to fit specified emittance and beta function.
    """
    def __init__(
        self, 
        file_name, 
        twissZ, 
        zdistributor=GaussDist1D, 
        cut_off=-1, 
        threshold=1e-3,
    ):
        emitZ = twissZ.emittance
        betaZ = twissZ.beta
        alphaZ = twissZ.alpha

        print("========= Input Twiss ===========")
        print(
            "alpha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f "
            % (alphaZ, betaZ, emitZ * 1.0e6)
        )

        ## -- load input dE distribution
        deinput = np.loadtxt(file_name)
        deval = deinput[:, 0] * 1e-3  # units GeV

        # -- input file is already thresholded and normalized (sum=1)
        pdf = deinput[:, 1]

        # -- calculate RMS alpha to preserve emittance and beta
        self.dErms = np.sqrt(np.sum(pdf * deval**2) / np.sum(pdf))
        alphaZ = -np.sqrt(betaZ * self.dErms**2 / emitZ - 1)

        # -- construct cumulative distribution function
        dEstep = (
            np.mean(deval[1:] - deval[0:-1]) * 1e3
        )  # this unit has to be in MeV to give correct cdf
        cdf = np.zeros(len(pdf))
        for i in range(1, len(cdf)):
            cdf[i] = np.sum(pdf[0:i]) * dEstep

        # -- function that returns dE [GeV] given index in range [0,1]
        self.invcdf = interp1d(cdf, deval, bounds_error=False, fill_value="extrapolate")

        print("========= Adjusted Twiss ===========")
        print(
            "alpha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f "
            % (alphaZ, betaZ, emitZ * 1.0e6)
        )

        twissZ = TwissContainer(alphaZ, betaZ, emitZ)

        # -- distributor for 2D Gaussian distribution
        self.distributor = None
        if zdistributor == WaterBagDist1D:
            self.distributor = zdistributor(twissZ)
        else:
            self.distributor = zdistributor(twissZ, cut_off)

    def get_z_zp(self):
        """Sample from z-dE distribution."""
        (z, dE) = self.distributor.getCoordinates()
        ind01 = norm.cdf(
            dE, scale=self.dErms
        )  # index of particle from [0,1] distribution
        dE = self.invcdf(ind01)
        return (z, dE)

    def gen_pdf(self):
        print("Method 'gen_pdf' not enabled")

    def grid_sample(self, n_parts=0):
        print("Method 'grid_sample' not enabled")
        return (0, 0)


def dump_bunch_coordinates(file_name, bunch):
    fl_out = None
    comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    rank = orbit_mpi.MPI_Comm_rank(comm)
    size = orbit_mpi.MPI_Comm_size(comm)
    data_type = mpi_datatype.MPI_DOUBLE
    if rank == 0:
        fl_out = open(file_name, "w")
        s = " x[mm]  xp[mrad]  y[mm]  yp[mrad]  z[mm]  dE[MeV] "
        fl_out.write(s + "\n")
    for cpu_rank in range(size):
        n_parts = bunch.getSize()
        n_parts = orbit_mpi.MPI_Bcast(n_parts, mpi_datatype.MPI_INT, cpu_rank, comm)
        for i in range(n_parts):
            x = xp = y = yp = z = dE = 0.0
            if cpu_rank == rank:
                x = bunch.x(i)
                y = bunch.y(i)
                z = bunch.z(i)
                xp = bunch.xp(i)
                yp = bunch.yp(i)
                dE = bunch.dE(i)
            x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
                (x, xp, y, yp, z, dE), data_type, cpu_rank, comm
            )
            if rank == 0:
                s = " %12.5g  %12.5g    %12.5g    %12.5g    %12.5g  %12.5g " % (
                    x * 1000.0,
                    xp * 1000.0,
                    y * 1000.0,
                    yp * 1000.0,
                    z * 1000.0,
                    dE * 1000.0,
                )
                fl_out.write(s + "\n")
    if rank == 0:
        fl_out.close()