"""Generate/manipulate pyorbit bunches."""
from __future__ import print_function
import math
import os
import random
import sys

from psdist import ap
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


def initialize_bunch(mass=0.939294, charge=-1, kin_energy=0.0025, current=0.040, freq=402.5e6):
    """Initialize bunch with specified parameters.
    
    Parameters
    ----------
    mass : float
        Particle mass [GeV/c^2].
    charge : float
        Particle charge [elementary charge units].
    kin_energy : float
        Synchronous particle kinetic energy [GeV].
    current : float
        Beam current [A].
    freq : float
        RF frequency [Hz].
        
    Returns
    -------
    Bunch
    """
    bunch = Bunch()
    bunch.mass(mass)
    bunch.charge(charge)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    bunch = set_current(bunch, current=current, freq=freq)
    return bunch
    
    
def set_current(bunch, current=None, freq=None):
    """Set macro-size from current [A] and frequency [Hz]."""    
    charge_bunch = current / freq
    charge_particle = abs(float(bunch.charge()) * consts.charge_electron)
    intensity = charge_bunch / charge_particle
    macro_size = intensity / bunch.getSizeGlobal()
    bunch.macroSize(macro_size)
    return bunch

    
def get_z_to_phase_coeff(bunch, freq=None):
    """Return coefficient to calculate phase [degrees] from z [m]."""
    wavelength = consts.speed_of_light / freq
    return -360.0 / (bunch.getSyncParticle().beta() * wavelength)


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


def decimate(bunch, factor=1, verbose=0):
    """Reduce the number of macro-particles in the bunch.

    This just skips every `fac` indices, so we assume that the bunch
    coordinates were generated randomly.
    """
    n_parts0 = bunch.getSizeGlobal()
    if not factor or not (1 <= factor < n_parts0):
        print('No decimation for factor={}.'.format(factor))
    if verbose:
        print('Decimating bunch by factor {}...'.format(factor))
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
    if verbose:
        print('Decimation complete.',
              'New bunch has {} macro-particles.'.format(new_bunch.getSize()))
    return bunch


def attenuate(bunch, factor=1.0):
    """Adjust current by changing macrosize

    factor : float
        The fractional beam current attenuation.
    """
    bunch.macroSize(factor * bunch.macroSize())
    return bunch

    
def decorrelate_x_y_z(bunch, verbose=0):
    """Remove cross-plane correlations in the bunch by permuting 
    (x, x'), (y, y'), (z, z') pairs."""
    if verbose:
        print("Decorrelating x-x', y-y', z-dE...")
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
    print("Decorrelation complete.")
    return bunch
        
    
def shift(bunch, x, xp, y, yp, z, dE, verbose=0):
    if verbose:
        print(
            "Shifting bunch centroid...",
            "  delta_x = {:.3f}".format(x),
            "  delta_x = {:.3f}".format(y),
            "  delta_z = {:.3f}".format(z),
            "  delta_xp = {:.3f}".format(xp),
            "  delta_yp = {:.3f}".format(yp),
            "  delta_dE = {:.3f}".format(dE),
        )
    for i in range(bunch.getSize()):
        bunch.x(i, bunch.x(i) + x)
        bunch.y(i, bunch.y(i) + y)  
        bunch.z(i, bunch.z(i) + z)
        bunch.xp(i, bunch.xp(i) + xp) 
        bunch.yp(i, bunch.yp(i) + yp)
        bunch.dE(i, bunch.dE(i) + dE)
    if verbose:
        print("Bunch shift complete.")
    return bunch

    
def center(bunch):
    """Shift the bunch so that first-order moments are zero."""
    twiss = BunchTwissAnalysis()
    twiss.analyzeBunch(bunch)
    deltas = [twiss.getAverage(i) for i in range(6)]
    return shift(bunch, *deltas)


def reverse(bunch, verbose=0):
    """Reverse the bunch propagation direction.

    Since the tail becomes the head of the bunch, the sign of z
    changes but the sign of dE does not change.
    """
    for i in range(bunch.getSize()):
        bunch.xp(i, -bunch.xp(i))
        bunch.yp(i, -bunch.yp(i))
        bunch.z(i, -bunch.z(i))
            
    
def load(
    filename=None,
    file_format='pyorbit',
    verbose=False,
    bunch=None,
):
    """Load bunch from coordinates file.

    Parameters
    ----------
    filename : str
        Path the file.
    file_format : str        
        'pyorbit':
            The expected header format is:
        'parmteq':
            The expected header format is:
                Number of particles    =
                Beam current           =
                RF Frequency           =
                The input file particle coordinates were written in double precision.
                x(cm)             xpr(=dx/ds)       y(cm)             ypr(=dy/ds)       phi(radian)        W(MeV)
    verbose : bool
        Whether to print intro/exit messages.
    bunch : Bunch
        If None, create a new bunch; otherwise, load into this bunch.
    """
    if verbose:
        print("Reading bunch from file '{}'...".format(filename))
    if not os.path.isfile(filename):
        raise ValueError("File '{}' does not exist.".format(filename))
    if bunch is None:
        bunch = Bunch()
    if file_format == "pyorbit":
        bunch.readBunch(filename)
    elif file_format == "parmteq":
        # Read data.
        header = np.genfromtxt(filename, max_rows=3, usecols=[0, 1, 2, 3, 4], dtype=str)
        n_parts = int(header[0, 4])
        current = np.float(header[1, 3])
        freq = np.float(header[2, 3])  * 1e6  # MHz to Hz
        data = np.loadtxt(filename, skiprows=5)
        
        # Trim off-energy particles.
        kin_energy = np.mean(data[:, 5])  # center energy [MeV]
        ind = np.where(np.abs(data[:, 5] - kin_energy) < (0.05 * kin_energy))[0]
        n_parts = len(ind)
        bunch.getSyncParticle().kinEnergy(kin_energy * 1e-3)
                
        # Unit conversion.
        dE = (data[ind, 5] - kin_energy) * 1e-3  # MeV to GeV
        x = data[ind, 0] * 1e-2  # cm to m
        xp = data[ind, 1]  # radians
        y = data[ind, 2] * 1e-2  # cm to m
        yp = data[ind, 3]  # radians
        phi = data[ind, 4]  # radians
        z = np.rad2deg(phi) / get_z_to_phase_coeff(bunch, freq=freq)
        
        # Add particles.
        for i in range(n_parts):
            bunch.addParticle(x[i], xp[i], y[i], yp[i], z[i], dE[i])
    else:
        raise KeyError("Unrecognized format '{}'.".format(file_format))
    if verbose:
        print("Bunch loaded (nparts={} macrosize={}).".format(bunch.getSize(), bunch.macroSize()))
    return bunch
        

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
        self.coords[:, :5] *= 1e3  # x=y=z[mm], x'=y'=[mrad]
        self.coords[:, 5] *= 1e6  # dE=[keV]
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
        j = 2 * i
        sigma = self.cov[j:j+2, j:j+2]
        alpha, beta = ap.twiss(sigma)
        eps = ap.apparent_emittance(sigma)        
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
    
        
def gen_bunch_twiss(
    n_parts=0,
    twiss_x=None,
    twiss_y=None,
    twiss_z=None,
    dist_gen=None,
    **dist_gen_kws
):
    """Generate bunch from Twiss parameters."""
    comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    rank = orbit_mpi.MPI_Comm_rank(comm)
    size = orbit_mpi.MPI_Comm_size(comm)
    data_type = mpi_datatype.MPI_DOUBLE
    main_rank = 0
    bunch = Bunch()
    distributor = dist_gen(twiss_x, twiss_y, twiss_z, **dist_gen_kws)
    bunch.getSyncParticle().time(0.0)
    for i in range(n_parts):
        x, xp, y, yp, z, dE = distributor.getCoordinates()
        x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
            (x, xp, y, yp, z, dE), data_type, main_rank, comm,
        )
        if i % size == rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch


def gen_bunch_xxp_yyp_twissz(
    n_parts=0,
    phase_space_gen_x=None,
    phase_space_gen_y=None,
    twiss_z=None,
    method='cdf',
    **dist_gen_kws
):
    """Generate bunch from f(x, x'), f(y, y'), twiss_z."""
    comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    rank = orbit_mpi.MPI_Comm_rank(comm)
    size = orbit_mpi.MPI_Comm_size(comm)
    data_type = mpi_datatype.MPI_DOUBLE
    main_rank = 0
    bunch = Bunch()
    distributor = dist_gen(twiss_z, **dist_gen_kws)
    bunch.getSyncParticle().time(0.0)
    if method == "cdf":
        for i in range(n_parts):
            z, dE = distributor.getCoordinates()
            x, xp = phase_space_gen_x.get_x_xp()
            y, yp = phase_space_gen_y.get_y_yp()
            x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
                (x, xp, y, yp, z, dE), data_type, main_rank, comm
            )
            if i % size == rank:
                bunch.addParticle(x, xp, y, yp, z, dE)
    elif method == "grid":
        phase_space_gen_x.gen_pdf()
        phase_space_gen_y.gen_pdf()
        xcoord, xpcoord = phase_space_gen_x.grid_sample(n_parts=n_parts)
        ycoord, ypcoord = phase_space_gen_y.grid_sample(n_parts=n_parts)
        n_parts = min(len(xcoord), len(ycoord))
        for i in range(n_parts):
            z, dE = distributor.getCoordinates()
            x, xp = (xcoord[i], xpcoord[i])
            y, yp = (ycoord[i], ypcoord[i])
            x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
                (x, xp, y, yp, z, dE), data_type, main_rank, comm
            )
            if i % size == rank:
                bunch.addParticle(x, xp, y, yp, z, dE)
    else:
        raise ValueError("Invalid method '{}'.".format(method))
    return bunch


def gen_bunch_xxp_yyp_zzp(
    n_parts=0,
    phase_space_gen_x=None,
    phase_space_gen_y=None,
    phase_space_gen_z=None,
    method='cdf',
):
    """Generate bunch from f(x, x'), f(y, y'), f(z, dE)."""
    comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    rank = orbit_mpi.MPI_Comm_rank(comm)
    size = orbit_mpi.MPI_Comm_size(comm)
    data_type = mpi_datatype.MPI_DOUBLE
    main_rank = 0
    bunch = Bunch()
    bunch.getSyncParticle().time(0.0)
    if self.method == "cdf":
        for i in range(n_parts):
            z, dE = phase_space_gen_z.get_z_zp()
            x, xp = phase_space_gen_x.get_x_xp()
            y, yp = phase_space_gen_y.get_y_yp()
            x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
                (x, xp, y, yp, z, dE), data_type, main_rank, comm
            )
            if i % size == rank:
                bunch.addParticle(x, xp, y, yp, z, dE)
    elif self.method == "grid":
        phase_space_gen_z.gen_pdf()
        phase_space_gen_x.gen_pdf()
        phase_space_gen_y.gen_pdf()
        zcoord, zpcoord = phase_space_gen_z.grid_sample(n_parts=n_parts)
        xcoord, xpcoord = phase_space_gen_x.grid_sample(n_parts=n_parts)
        ycoord, ypcoord = phase_space_gen_y.grid_sample(n_parts=n_parts)
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
        raise ValueError("Invalid method '{}'.".format(method))
    return bunch