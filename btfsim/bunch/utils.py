"""Utilities for manipulation of bunches."""
import numpy as np
from bunch import Bunch, BunchTwissAnalysis
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit


# Gaussian function for fitting
def fit_gauss(x, mu=0.0, *params):
    A, sigma = params
    return A * np.exp(-((x - mu)**2) / (2.0 * sigma**2))

            
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
            np.vstack([np.min(self.coord1, axis=0), 
                       np.min(self.coord1, axis=0)]),
            axis=0,
        )
        self.mins = np.max(
            np.vstack([np.max(self.coord1, axis=0), 
                       np.max(self.coord1, axis=0)]),
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


class BunchCalculation:
    
    def __init__(self, bunch, file=None):
        if isinstance(bunch, Bunch):
            self.bunch = bunch
        elif type(bunch) == str:  # load bunch from file
            self.bunch = Bunch()
            self.bunch.readBunch(bunch)
        self.coords = bunch_coord_array(bunch)
        self.cov = np.cov(self.coords.T)
        self.filename = file
        self.twiss_analysis = BunchTwissAnalysis()
        self.twiss_analysis.analyzeBunch(bunch)
        self.nParts = bunch.getSizeGlobal()
        self.gamma = bunch.getSyncParticle().gamma()
        self.beta = bunch.getSyncParticle().beta()
        self.mass = bunch.getSyncParticle().mass()
    
    def mean(self):
        """Return centroid in 6D phase space."""
        return np.mean(self.coords, axis=0)

    def twiss(self, dim='x', dispersion_flag=0, emit_norm_flag=0):
        """Return rms 2D Twiss parameters."""
        i = ['x', 'y', 'z'].index(dim)
        d = self.twiss_analysis.getDispersion(ind)
        dp = self.twiss_analysis.getDispersionDerivative(ind)
        i = 2 * ['x', 'y', 'z'].index(dim)
        sigma = self.cov[i:i+2, i:i+2]
        eps = np.sqrt(np.linalg.det(sigma))
        beta = sigma[0, 0] / eps
        alpha = -sigma[0, 1] / eps
        if emit_norm_flag and dim == 'z':
            eps *= self.gamma**3 * self.beta
        return {
            "beta": {"value": b, "unit": "mm/mrad"},
            "alpha": {"value": a, "unit": ""},
            "eps": {"value": e, "unit": "mm-mrad"},
            "disp": {"value": d, "unit": "m"},
            "dispp": {"value": dp, "unit": ""},
        }

    def extent(self, fraction):
        """Calculate max radial extent [m] containing fraction of particles.

        bunch: Bunch() instance
        fraction: fraction of particles to count. Eg, 0.90 for 90% extent.
        """
        radii = np.linalg.norm(self.coords[:, [0, 2]], axis=0)
        radii = np.sort(radii)
        if (X.shape[0] * fraction < 100.0):
            nn = self.coords.shape[0] - 1
        else:
            nn = np.round(self.coords.shape[0] * fraction)
        try:
            fraction = r[int(nn)]
        except:
            fraction = 0.0
        return fraction
        
    def norm_coords(self, dim='x'):
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


class BunchTrack:
    """Class to store beam evolution data.
    
    This class also has as a method that is called on action entrance to add 
    to the beam evolution array.

    dispersion_flag = 0; set to 1 to subtract dispersive term from emittances
    emit_norm_flag = 0; set to 1 fo calculate normalized emittances
    """
    def __init__(self, dispersion_flag=0, emit_norm_flag=0):
        self.twiss_analysis = BunchTwissAnalysis()
        self.dispersion_flag = dispersion_flag
        self.emit_norm_flag = emit_norm_flag
        histkeys = [
            "s",
            "npart",
            "nlost",
            "alpha_x",
            "beta_x",
            "eps_x",
            "alpha_y",
            "beta_y",
            "eps_y",
            "alpha_z",
            "beta_z",
            "eps_z",
            "sigxx",
            "sigyy",
            "sigxy",
            "r90",
            "r99",
            "dx",
            "dpx",
        ]
        histinitlen = 10000
        self.hist = dict(
            (histkeys[k], np.zeros(histinitlen)) for k in range(len(histkeys))
        )
        self.hist["node"] = []

    def action_entrance(self, paramsDict):
        """Executed at entrance of node."""
        node = paramsDict["node"]
        bunch = paramsDict["bunch"]
        pos = paramsDict["path_length"]
        if paramsDict["old_pos"] == pos:
            return
        if paramsDict["old_pos"] + paramsDict["pos_step"] > pos:
            return
        paramsDict["old_pos"] = pos
        paramsDict["count"] += 1

        # -- update statement
        nstep = paramsDict["count"]
        npart = bunch.getSize()
        print(
            "Step %i, Nparts %i, s=%.3f m, node %s"
            % (nstep, npart, pos, node.getName())
        )

        calc = BunchCalculation(bunch)
        twissx = calc.Twiss(
            dim="x",
            dispersion_flag=self.dispersion_flag,
            emit_norm_flag=self.emit_norm_flag,
        )
        (alphaX, betaX, emittX) = (
            twissx["alpha"]["value"],
            twissx["beta"]["value"],
            twissx["eps"]["value"],
        )
        (dispX, disppX) = (twissx["disp"]["value"], twissx["dispp"]["value"])
        twissy = calc.Twiss(
            dim="y",
            dispersion_flag=self.dispersion_flag,
            emit_norm_flag=self.emit_norm_flag,
        )
        (alphaY, betaY, emittY) = (
            twissy["alpha"]["value"],
            twissy["beta"]["value"],
            twissy["eps"]["value"],
        )
        twissz = calc.Twiss(
            dim="z",
            dispersion_flag=self.dispersion_flag,
            emit_norm_flag=self.emit_norm_flag,
        )
        (alphaZ, betaZ, emittZ) = (
            twissz["alpha"]["value"],
            twissz["beta"]["value"],
            twissz["eps"]["value"],
        )
        nParts = bunch.getSizeGlobal()
        gamma = bunch.getSyncParticle().gamma()
        beta = bunch.getSyncParticle().beta()

        ## -- compute twiss (this is somehow more robust than above...but doesn't include dispersion flag..)
        # self.twiss_analysis.analyzeBunch(bunch)
        # (alphaX,betaX,emittX) = (self.twiss_analysis.getEffectiveAlpha(0),self.twiss_analysis.getEffectiveBeta(0),self.twiss_analysis.getEffectiveEmittance(0)*1.0e+6)
        # (alphaY,betaY,emittY) = (self.twiss_analysis.getEffectiveAlpha(1),self.twiss_analysis.getEffectiveBeta(1),self.twiss_analysis.getEffectiveEmittance(1)*1.0e+6)
        # (alphaZ,betaZ,emittZ) = (self.twiss_analysis.getTwiss(2)[0],self.twiss_analysis.getTwiss(2)[1],self.twiss_analysis.getTwiss(2)[3]*1.0e+6)

        x_rms = np.sqrt(betaX * emittX) / 10.0  # [cm]
        y_rms = np.sqrt(betaY * emittY) / 10.0  # [cm]
        z_rms = np.sqrt(betaZ * emittZ)  # [m]

        # Compute covariance matrix.
        Sigma = calc.cov()
        
        # -- compute 90%, 99% extent
        r90 = calc.Extent(0.90) * 1e2
        r99 = calc.Extent(0.99) * 1e2

        # Correctly assign the number of particles for the 0th step
        if paramsDict["count"] == 1:
            self.hist["npart"][paramsDict["count"] - 1] = nParts

        # -- assign history arrays in hist dict
        self.hist["s"][paramsDict["count"]] = pos
        self.hist["node"].append(node.getName())
        self.hist["npart"][paramsDict["count"]] = nParts
        self.hist["xrms"][paramsDict["count"]] = x_rms
        self.hist["yrms"][paramsDict["count"]] = y_rms
        self.hist["zrms"][paramsDict["count"]] = z_rms
        self.hist["alpha_x"][paramsDict["count"]] = alphaX
        self.hist["beta_x"][paramsDict["count"]] = betaX
        self.hist["eps_x"][paramsDict["count"]] = emittX
        self.hist["dx"][paramsDict["count"]] = dispX
        self.hist["dpx"][paramsDict["count"]] = disppX
        self.hist["alpha_y"][paramsDict["count"]] = alphaY
        self.hist["beta_y"][paramsDict["count"]] = betaY
        self.hist["eps_y"][paramsDict["count"]] = emittY
        self.hist["alpha_z"][paramsDict["count"]] = alphaZ
        self.hist["beta_z"][paramsDict["count"]] = betaZ
        self.hist["eps_z"][paramsDict["count"]] = emittZ
        self.hist["sigxx"][paramsDict["count"]] = sigxx
        self.hist["sigyy"][paramsDict["count"]] = sigyy
        self.hist["sigxy"][paramsDict["count"]] = sigxy
        self.hist["r90"][paramsDict["count"]] = r90
        self.hist["r99"][paramsDict["count"]] = r99
        self.hist["nlost"][paramsDict["count"]] = self.hist["npart"][0] - nParts

    def action_exit(self, paramsDict):
        """
        Executed at exit of node
        """
        self.action_entrance(paramsDict)

    def cleanup(self):
        # -- trim 0's from hist
        ind = np.where(self.hist["xrms"] == 0)[0][1]
        for key, arr in self.hist.iteritems():
            self.hist[key] = arr[1:ind]

    def writehist(self, **kwargs):
        """
        Save history data
        optional argument:
        filename = location to save data
        """

        # --- file name + location
        defaultfilename = "btf_output_data.txt"
        filename = kwargs.get("filename", defaultfilename)

        # -- open files to write data
        file_out = open(filename, "w")
        header = "s[m], nparts, xrms [cm], yrms [cm], zrms [cm], ax, bx, ex[mm-mrad], ay, by, ey[mm-mrad], az, bz, ez[m-GeV], sigx[cm], sigy[cm], sigxx[cm2], sigyy[cm2], sigxy[cm2], r90[cm], r99[cm], Dx [m], Dxp \n"
        file_out.write(header)

        for i in range(len(self.hist["s"])):
            line = "%.3f %i " % (self.hist["s"][i], self.hist["npart"][i])
            line += "%.3f %.3f %.3f " % (
                self.hist["xrms"][i],
                self.hist["yrms"][i],
                self.hist["zrms"][i],
            )
            line += "%.3f %.3f %.6f " % (
                self.hist["alpha_x"][i],
                self.hist["beta_x"][i],
                self.hist["eps_x"][i],
            )
            line += "%.3f %.3f %.6f " % (
                self.hist["alpha_y"][i],
                self.hist["beta_y"][i],
                self.hist["eps_y"][i],
            )
            line += "%.3f %.3f %.6f " % (
                self.hist["alpha_z"][i],
                self.hist["beta_z"][i],
                self.hist["eps_z"][i],
            )
            line += "%.5f %.5f %.6f " % (
                self.hist["sigx"][i],
                self.hist["sigy"][i],
                self.hist["sigxx"][i],
            )
            line += "%.6f %.6f " % (self.hist["sigyy"][i], self.hist["sigxy"][i])
            line += "%.4f %.4f " % (self.hist["r90"][i], self.hist["r99"][i])
            line += "%.4f %.4f \n" % (self.hist["dx"][i], self.hist["dpx"][i])
            file_out.write(line)

        file_out.close()


class spTrack:
    """
    This class holds array with beam evolution data
    Copy of bunchTrack class modified for single-particle tracking (no twiss/size data)
    """

    def __init__(self):

        # -- initialize history arrays in hist dict
        histkeys = ["s", "npart", "x", "xp", "y", "yp", "z", "dE"]
        histinitlen = 10000
        self.hist = dict(
            (histkeys[k], np.zeros(histinitlen)) for k in range(len(histkeys))
        )
        self.hist["node"] = []

    def action_entrance(self, paramsDict):
        """
        Executed at entrance of node
        """
        node = paramsDict["node"]
        bunch = paramsDict["bunch"]
        pos = paramsDict["path_length"]
        if paramsDict["old_pos"] == pos:
            return
        if paramsDict["old_pos"] + paramsDict["pos_step"] > pos:
            return
        paramsDict["old_pos"] = pos
        paramsDict["count"] += 1

        # -- update statement
        nstep = paramsDict["count"]
        npart = bunch.getSize()
        print(
            "Step %i, Nparts %i, s=%.3f m, node %s"
            % (nstep, npart, pos, node.getName())
        )

        nParts = bunch.getSizeGlobal()

        # -- get particle position, momenta
        x = bunch.x(0) * 1000.0
        xp = bunch.xp(0) * 1000.0
        y = bunch.y(0) * 1000.0
        yp = bunch.yp(0) * 1000.0
        z = bunch.z(0) * 1000.0
        dE = bunch.dE(0) * 1000.0

        # -- assign history arrays in hist dict
        self.hist["s"][paramsDict["count"]] = pos
        self.hist["node"].append(node.getName())
        self.hist["npart"][paramsDict["count"]] = nParts
        self.hist["x"][paramsDict["count"]] = x
        self.hist["y"][paramsDict["count"]] = y
        self.hist["z"][paramsDict["count"]] = z
        self.hist["xp"][paramsDict["count"]] = xp
        self.hist["yp"][paramsDict["count"]] = yp
        self.hist["dE"][paramsDict["count"]] = dE

    def action_exit(self, paramsDict):
        """
        Executed at exit of node
        """
        self.action_entrance(paramsDict)

    def cleanup(self):
        # -- trim 0's from hist
        ind = np.where(self.hist["npart"][1:] == 0)[0][0] + 1
        for key, arr in self.hist.iteritems():
            self.hist[key] = arr[0:ind]

    def writehist(self, **kwargs):
        """
        Save history data
        optional argument:
        filename = location to save data
        """

        # --- file name + location
        defaultfilename = "btf_output_data.txt"
        filename = kwargs.get("filename", defaultfilename)

        # -- open files to write data
        file_out = open(filename, "w")
        header = "s[m], nparts, x [mm], xp[mrad], y[mm], yp[mrad], z[mm?], dE[MeV?] \n"
        file_out.write(header)

        for i in range(len(self.hist["s"]) - 1):
            line = "%.3f %s %i %.6f %.6f %.6f %.6f %.6f %.6f \n" % (
                self.hist["s"][i],
                self.hist["node"][i].split(":")[-1],
                self.hist["npart"][i],
                self.hist["x"][i],
                self.hist["xp"][i],
                self.hist["y"][i],
                self.hist["yp"][i],
                self.hist["z"][i],
                self.hist["dE"][i],
            )
            file_out.write(line)

        file_out.close()


class Beamlet:
    """
    Class to create beamlet out of specified bunch distribution.

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


class adaptiveWeighting:
    """This class adjusts macroparticle weight dynamically."""

    def __init__(self, z2phase, macrosize0):
        self.z2phase = z2phase
        self.macrosize0 = macrosize0

        # -- initialize history arrays in hist dict
        histkeys = ["s", "macro"]
        histinitlen = 10000
        self.hist = dict(
            (histkeys[k], np.zeros(histinitlen)) for k in range(len(histkeys))
        )
        self.hist["node"] = []

    def action_entrance(self, paramsDict):
        """
        Executed at entrance of node
        """
        node = paramsDict["node"]
        bunch = paramsDict["bunch"]
        pos = paramsDict["path_length"]
        if paramsDict["old_pos"] == pos:
            return
        if paramsDict["old_pos"] + paramsDict["pos_step"] > pos:
            return
        paramsDict["old_pos"] = pos
        paramsDict["count"] += 1

        # -- count how many particles inside 1 RF period
        noutside = 0
        ntotal = bunch.getSize()
        for i in range(ntotal):
            phi = bunch.z(i) * self.z2phase
            if np.abs(phi) > 180.0:  # if outside 1 RF period
                noutside += 1

        macrosize = self.macrosize0 * ntotal / (ntotal - noutside)
        bunch.macroSize(macrosize)

        # -- assign history arrays in hist dict
        self.hist["s"][paramsDict["count"]] = pos
        self.hist["node"].append(node.getName())
        self.hist["macro"][paramsDict["count"]] = macrosize

    def action_exit(self, paramsDict):
        self.action_entrance()
