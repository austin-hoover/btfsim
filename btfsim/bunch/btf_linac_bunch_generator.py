"""Generate PyORBIT bunches in the BTF linac.

It is parallel, but it is not efficient.
"""
from __future__ import print_function
import math
import sys
import os
import random
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

from bunch import Bunch
import orbit_mpi
from orbit_mpi import mpi_comm
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op
from spacecharge import Grid2D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import KVDist1D, KVDist2D, KVDist3D
from orbit.bunch_generators import GaussDist1D, GaussDist2D
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import WaterBagDist1D, WaterBagDist2D, WaterBagDist3D
from orbit.bunch_generators import TwissAnalysis
from orbit.py_linac.lattice import BaseLinacNode
from orbit.utils import consts


class Base_BunchGenerator(object):
    """Base class for bunch generators."""

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
        nparticles = int(header[0, 4])
        current = np.float(header[1, 3])
        freq = np.float(header[2, 3])

        # -- read data
        data = np.loadtxt(filename, skiprows=5)
        # -- trim off-energy particles
        e_kin = np.mean(data[:, 5])  # center energy
        ind = np.where(np.abs(data[:, 5] - e_kin) < (0.05 * e_kin))[0]
        nparticles = len(ind)

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
        for i in range(nparticles):
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
        nPartsTotal = bunch.getSizeGlobal()
        if nPartsTotal == 0.0:
            return
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        main_rank = 0
        data_type = mpi_datatype.MPI_DOUBLE
        nParts = bunch.getSize()
        (x_avg, xp_avg, y_avg, yp_avg, z_avg, dE_avg) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(nParts):
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
        x_avg /= nPartsTotal
        xp_avg /= nPartsTotal
        y_avg /= nPartsTotal
        yp_avg /= nPartsTotal
        z_avg /= nPartsTotal
        dE_avg /= nPartsTotal
        for i in range(nParts):
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


class BTF_Linac_BunchGenerator(Base_BunchGenerator):
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
        super(BTF_Linac_BunchGenerator, self).__init__(
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


class BTF_Linac_TrPhaseSpace_BunchGenerator(Base_BunchGenerator):
    """Generates the pyORBIT BTF Linac Bunches.

    The transverse phase spaces are generated by  PhaseSpaceGenerator class instances
    The longitudinal phase space is generated according to designated distribution function,
    defined as Twiss object. Emittance is in [GeV*m].
    """

    def __init__(
        self,
        phaseSpGenX,
        phaseSpGenY,
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
        self.phaseSpGenX = phaseSpGenX
        self.phaseSpGenY = phaseSpGenY
        self.method = method
        super(BTF_Linac_TrPhaseSpace_BunchGenerator, self).__init__(
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
                (x, xp) = self.phaseSpGenX.get_x_xp()
                (y, yp) = self.phaseSpGenY.get_y_yp()
                (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        # -- deposit particles according to Grid-based method
        elif self.method == "grid":
            # -- create pdf's for x and y distributions
            self.phaseSpGenX.gen_pdf()
            self.phaseSpGenY.gen_pdf()
            # -- sample distributions
            (xcoord, xpcoord) = self.phaseSpGenX.grid_sample(n_parts=n_parts)
            (ycoord, ypcoord) = self.phaseSpGenY.grid_sample(n_parts=n_parts)
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


class BTF_Linac_6DPhaseSpace_BunchGenerator(Base_BunchGenerator):
    """Generates the pyORBIT BTF Linac Bunches.

    The transverse phase spaces are generated by  PhaseSpaceGenerator class instances
    The longitudinal phase space is reconstructed from
    """

    def __init__(
        self,
        phaseSpGenX,
        phaseSpGenY,
        phaseSpGenZ,
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
        self.phaseSpGenZ = phaseSpGenZ
        self.phaseSpGenX = phaseSpGenX
        self.phaseSpGenY = phaseSpGenY
        self.method = method
        super(BTF_Linac_6DPhaseSpace_BunchGenerator, self).__init__(
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
                (z, dE) = self.phaseSpGenZ.get_z_zp()
                (x, xp) = self.phaseSpGenX.get_x_xp()
                (y, yp) = self.phaseSpGenY.get_y_yp()
                (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
                    (x, xp, y, yp, z, dE), data_type, main_rank, comm
                )
                if i % size == rank:
                    bunch.addParticle(x, xp, y, yp, z, dE)
        # -- deposit particles according to Grid-based method
        elif self.method == "grid":
            # -- create pdf's for x and y distributions
            self.phaseSpGenZ.gen_pdf()
            self.phaseSpGenX.gen_pdf()
            self.phaseSpGenY.gen_pdf()
            # -- sample distributions
            (zcoord, zpcoord) = self.phaseSpGenZ.grid_sample(n_parts=n_parts)
            (xcoord, xpcoord) = self.phaseSpGenX.grid_sample(n_parts=n_parts)
            (ycoord, ypcoord) = self.phaseSpGenY.grid_sample(n_parts=n_parts)
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


class PhaseSpaceGenerator:
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
        Nparts = Ndistr.sum()

        # -- define coordinate grid
        xaxis = np.linspace(self.x_min, self.x_max, num=self.nx)
        xpaxis = np.linspace(self.xp_min, self.xp_max, num=self.nxp)
        [XG, XPG] = np.meshgrid(xaxis, xpaxis)

        # -- loop through grid + deposit Ndistr(i,j) particles at each point
        # could be combined into 1 loop (do x and y simultaneously),
        # but left general in case x and y scans have different resolutions
        X, XP = np.zeros([2, Nparts])
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
        x = (X + (np.random.rand(1, Nparts) - 0.5) * 2 * self.x_step) / 1000.0
        xp = (XP + (np.random.rand(1, Nparts) - 0.5) * 2 * self.xp_step) / 1000.0

        # -- random sampling to remove correlations in x-y, xp-yp
        xind = np.random.permutation(range(np.shape(x)[1]))

        # -- return shuffled x,x' distribution
        return (x[0, xind], xp[0, xind])


class PhaseSpaceGeneratorZPartial:
    """Generates (z, dE) distribution using 1D e-profile.

    Assumes Gaussian phase to fit specified emittance and beta function.
    """

    def __init__(
        self, file_name, twissZ, zdistributor=GaussDist1D, cut_off=-1, threshold=1e-3
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


class DumpBunchAccNode(BaseLinacNode):
    """Dumps the x, x', y, y', z, dE into the ASCII file."""

    def __init__(self, file_name="bunch_dump.dat", name="DumpBunch"):
        BaseLinacNode.__init__(self, name)
        self.file_name = file_name
        self.skip = False

    def set_skip(self, skip):
        self.skip = skip

    def set_file_name(self, file_name):
        self.file_name = file_name

    def track(self, paramsDict):
        """The track method will be called during the bunch track."""
        if self.skip:
            return
        bunch = paramsDict["bunch"]
        dump_bunch_coordinates(self.file_name, bunch)


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
        nParts = bunch.getSize()
        nParts = orbit_mpi.MPI_Bcast(nParts, mpi_datatype.MPI_INT, cpu_rank, comm)
        for i in range(nParts):
            (x, xp, y, yp, z, dE) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            if cpu_rank == rank:
                (x, xp, y, yp, z, dE) = (
                    bunch.x(i),
                    bunch.xp(i),
                    bunch.y(i),
                    bunch.yp(i),
                    bunch.z(i),
                    bunch.dE(i),
                )
            (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
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


def bunch_transformer_func(bunch):
    """Reverse all xp, yp, z coordinates of the bunch.

    We have to change the sign of the z because the tail will be the head
    of the bunch, but the sign of dE will not change because of the same reason.
    """
    nParts = bunch.getSize()
    for i in range(nParts):
        (xp, yp, z, dE) = (bunch.xp(i), bunch.yp(i), bunch.z(i), bunch.dE(i))
        bunch.xp(i, -xp)
        bunch.yp(i, -yp)
        bunch.z(i, -z)
        # --- dE should not change the sign
        # bunch.dE(i,-dE)


def reverse_bunch_coordinate(bunch, axis_ind):
    """Reverse a particular coordinate of the bunch.

    axis_ind = 0 - x-coord
    axis_ind = 1 - xp-coord
    axis_ind = 2 - y-coord
    axis_ind = 3 - yp-coord
    axis_ind = 4 - z-coord
    axis_ind = 5 - dE-coord
    """
    nParts = bunch.getSize()
    for i in range(nParts):
        (x, xp, y, yp, z, dE) = (
            bunch.x(i),
            bunch.xp(i),
            bunch.y(i),
            bunch.yp(i),
            bunch.z(i),
            bunch.dE(i),
        )
        if axis_ind == 0:
            bunch.x(i, -x)
        if axis_ind == 1:
            bunch.xp(i, -xp)
        if axis_ind == 2:
            bunch.y(i, -y)
        if axis_ind == 3:
            bunch.yp(i, -yp)
        if axis_ind == 4:
            bunch.z(i, -z)
        if axis_ind == 5:
            bunch.dE(i, -dE)
