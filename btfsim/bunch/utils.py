"""Utilities for PyORBIT bunch manipulation."""
from __future__ import print_function
import sys
import os

import numpy as np
import pandas as pd

from bunch import Bunch
from bunch import BunchTwissAnalysis

from btfsim.analysis import stats
from btfsim.analysis import dist


def fit_gauss(x, mu=0.0, *params):
    A, sigma = params
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def bunch_coord_array(bunch):
    """Return bunch coordinate array."""
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
        
    def center_bunch(self):
        """Shift the bunch so that first-order moments are zero."""
        twiss = BunchTwissAnalysis()
        twiss.analyzeBunch(self.bunch)
        self.shift(*[twiss.getAverage(i) for i in range(6)])
        
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

    def twiss(self, dim="x", dispersion_flag=0, emit_norm_flag=0):
        """Return rms 2D Twiss parameters."""
        i = ["x", "y", "z"].index(dim)
        alpha, beta = stats.twiss(self.cov, dim=dim)
        eps = stats.emittance(self.cov, dim=dim)
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

    def norm_coords(self, dim="x"):
        """Return coordinates normalized by rms Twiss parameters in x-x', y-y', z-z'."""
        x, xp = np.zeros([2, self.bunch.getSize()])
        X = bunch_coord_array(self.bunch())

        twiss = self.twiss(dim=dim, emit_norm_flag=1)
        alpha = twiss["alpha"]["value"]
        beta = twiss["beta"]["value"]

        xn = x / np.sqrt(beta)
        xnp = alpha * x / np.sqrt(beta) + xp * np.sqrt(beta)
        return xn, xnp

    def radial_density(self, dr=0.1, dim="x"):
        raise NotImplementedError
        
        
class BunchCompare:
    """Class to compare two bunches."""
    
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
    """Dynamic macro-particle weight adjustment."""

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