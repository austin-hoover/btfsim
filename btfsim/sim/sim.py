from __future__ import print_function
import sys
import os
import time
import numpy as np

from bunch import Bunch, BunchTwissAnalysis
from spacecharge import SpaceChargeCalcUnifEllipse, SpaceChargeCalc3D
from orbit.bunch_generators import TwissContainer, TwissAnalysis
from orbit.bunch_generators import KVDist1D, KVDist2D, KVDist3D
from orbit.bunch_generators import GaussDist1D, GaussDist2D, GaussDist3D
from orbit.bunch_generators import WaterBagDist1D, WaterBagDist2D, WaterBagDist3D
from orbit.lattice import AccActionsContainer
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_bend_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice
from orbit.space_charge.sc3d import setSC3DAccNodes, setUniformEllipsesSCAccNodes
import orbit.utils.consts as consts

import btfsim.bunch.btf_linac_bunch_generator as gen_bunch
import btfsim.bunch.utils as butils
import btfsim.lattice.lattice_factory as gen_lattice
from btfsim.util.default import Default


class Sim:
    """Class to hold simulation model.

    Attributes
    ----------
    ekin : float
        Kinetic energy of synchronous particle [GeV]
    mass : float
        Mass per particle [GeV/c^2]        
    beta : float
        Synchronous particle velocity relative to speed of light.
    gamma : float
        Lorentz factor 1 / sqrt(1 - beta**2)
    frequency : float
        
    bunch_in : Bunch
        Input bunch to the simulation. It is copied to a `bunch_track`
        for tracking.
    bunch_track : Bunch
        Bunch used for tracking; overwritten each time simulation is run.
    lattice : orbit.lattice.AccLattice
        Lattice for tracking.
    latgen : btfsim.lattice.generate_btf_lattice.LatticeGenerator
        Instance of lattice generator class.
    """
    def __init__(self, outdir=None):
        self.ekin = 0.0025  # [GeV]
        self.mass = 0.939294  # [GeV/c^2]
        self.charge = -1.0  # elementary charge units
        self.gamma = (self.mass + self.ekin) / self.mass
        self.beta = np.sqrt(self.gamma * self.gamma - 1.0) / self.gamma
        self.freq = 402.5e6
        print("relativistic. gamma =", self.gamma)
        print("relativistic  beta =", self.beta)

        self.default = Default()
        self.outdir = outdir
        if self.outdir is None:
            self.outdir = os.path.join(os.getcwd(), '/data')
        if not os.path.exists(self.outdir):  # -- make directory if not there yet
            os.mkdir(self.outdir)

        self.movie_flag = 0  
        self.dispersion_flag = 1
        self.emit_norm_flag = 0

    def init_all(self, init_bunch=True):
        """Set up full simulation with all default values."""
        self.init_lattice()
        self.init_apertures()
        self.init_sc_nodes()
        if init_bunch:
            self.init_bunch()

    def enable_movie(self, **kwargs):
        self.movie_flag = 1
        savefolder = kwargs.pop("savedir", "data/")
        self.movie = MovieBase(savefolder, **kwargs)

    def run(self, start=0.0, stop=None, out='default', verbose=0):
        """Run the simulation.

        out : str
            Location of output bunch file. If 'default', use 'btf_output_bunch_end.txt'. 
            If None, does not save anything.
        """
        if stop is None:
            stop = self.lattice.getLength()

        # Parse default output filename.
        if stop == self.lattice.getLength():
            default_output_filename = "btf_output_bunch_end.txt"
        else:
            default_output_filename = "btf_output_bunch_{}.txt".format(stop)
        if out == 'default':
            out = default_output_filename
        if out is not None:
            output_filename = os.path.join(self.outdir, out)

        # Figure out which nodes are at stop/start position.
        if type(start) in [float, int]:
            startnode, startnum, zs_startnode, ze_startnode = self.lattice.getNodeForPosition(start)
        elif type(start) is str:
            startnode = self.lattice.getNodeForName(start)
            startnum = self.lattice.getNodeIndex(startnode)
            zs_startnode = startnode.getPosition() - 0.5 * startnode.getLength()
            ze_startnode = startnode.getPosition() + 0.5 * startnode.getLength()
        else:
            raise TypeError("Invalid type {} for `start`.".format(type(start)))
            
        if type(stop) in [float, int]:            
            print("max simulation length = {:.3f}".format(self.lattice.getLength()))
            if stop > self.lattice.getLength():
                stop = self.lattice.getLength()
            stopnode, stopnum, zs_stopnode, ze_stopnode = self.lattice.getNodeForPosition(stop)
        elif type(stop) is str:
            stopnode = self.lattice.getNodeForName(stop)
            stopnum = self.lattice.getNodeIndex(stopnode)
            zs_stopnode = stopnode.getPosition() - 0.5 * stopnode.getLength()
            ze_stopnode = stopnode.getPosition() + 0.5 * stopnode.getLength()
        else:
            raise TypeError("Invalid type {} for `stop`.".format(type(start)))
            
        print(
            "Running simulation from {:.4f} m to {:.4f} m (nodes {} to {})"
            .format(zs_startnode, ze_stopnode, startnum, stopnum)
        )

        # Setup
        paramsDict = {
            "old_pos": -1.0,
            "count": 0,
            "pos_step": 0.005,
        } 
        actionContainer = AccActionsContainer("BTF Bunch Tracking")

        ## Add tracking action; load different tracking routine if bunch has only 
        ## one particle (single-particle tracking).
        if self.bunch_in.getSize() > 1:
            self.tracker = butils.BunchTracker(
                dispersion_flag=self.dispersion_flag, 
                emit_norm_flag=self.emit_norm_flag
            )
        elif self.bunch_in.getSize() == 1:
            self.tracker = butils.SingleParticleTracker()
        actionContainer.addAction(self.tracker.action_exit, AccActionsContainer.EXIT)

        # Run the simulation.
        if self.movie_flag == 1:
            actionContainer.addAction(self.movie.makeFrame, AccActionsContainer.EXIT)
        time_start = time.clock()
        self.bunch_track = Bunch()
        self.bunch_in.copyBunchTo(self.bunch_track)
        self.lattice.trackBunch(
            self.bunch_track,
            paramsDict=paramsDict,
            actionContainer=actionContainer,
            index_start=startnum,
            index_stop=stopnum,
        )
        
        # Save the last time step.
        paramsDict["old_pos"] = -1
        self.tracker.action_entrance(paramsDict)

        # Wrap up.
        time_exec = time.clock() - time_start
        print("time[sec]=", time_exec)
        if out is not None:
            self.bunch_track.dumpBunch(output_filename)
            print("Dumped output bunch to file {}".format(output_filename))
        self.tracker.cleanup()
        self.tracker.hist["s"] += zs_startnode

    def run_reverse(self, start=0.0, stop=None, out='default'):
        """Execute the simulation in reverse.
        
        The simulation will start at `stop` (position or node) and end at `start`.
        """
        if stop is None:
            stop = self.lattice.getLength()
        print("Running simulation in reverse from s={} to s={}.".format(stop, start))

        # Reverse start and stop (if specified as floats).
        if type(start) is float:            
            start = self.lattice.getLength() - start
        if type(stop) is float:
            stop = self.lattice.getLength() - stop
            
        # Parse default output filename.
        if stop == 0.0:
            default_output_filename = "reverse_output_bunch_start.txt"
        else:
            default_output_filename = "reverse_output_bunch_{}.txt".format(stop)
        if out == 'default':
            out = default_output_filename
        if out is not None:
            output_filename = os.path.join(self.outdir, out)

        # Reverse the lattice.
        self.latgen.lattice.reverseOrder()
        self.lattice = self.latgen.lattice

        # Reverse the bunch coordinates.
        gen_bunch.bunch_transformer_func(self.bunch_in)

        # Run in reverse (start <==> stop). Do not auto-save output bunch; the bunch
        # coordinates need to be reversed first.
        self.run(start=stop, stop=start, out=None)

        # Un-reverse the lattice.
        self.latgen.lattice.reverseOrder()
        self.lattice = self.latgen.lattice

        # Un-reverse the bunch coordinates, then save them.
        gen_bunch.BunchTransformerFunc(self.bunch_track)
        if out:
            self.bunch_track.dumpBunch(output_filename)

        # Also un-reverse the initial bunch coordinates.
        gen_bunch.BunchTransformerFunc(self.bunch_in)

        # Wrap up.
        if type(stop) in [float, int]:
            stopnode, stopnum, zs_stopnode, ze_stopnode = self.lattice.getNodeForPosition(stop)
        elif type(stop) == str:
            stopnode = self.lattice.getNodeForName(stop)
            stopnum = self.lattice.getNodeIndex(stopnode)
            zs_stopnode = stopnode.getPosition() - 0.5 * stopnode.getLength()
            ze_stopnode = stopnode.getPosition() + 0.5 * stopnode.getLength()
        self.tracker.hist["s"] = 2.0 * ze_stopnode - self.tracker.hist["s"]

    def init_lattice(
        self,
        beamlines=["MEBT1", "MEBT2", "MEBT3"],
        xml=None,
        mstatename=None,
        mdict=None,
        units="Amps",
        maxdriftlen=0.012,
        coef_filename=None,
    ):
        """Initialize lattice from xml file.
        
        beamlines : list[str]
            The names of the beamlines to load.
        xml : str
            Name of xml file.
        mstatename : str
            Location of the .mstate file. if None, the default lattice is loaded.
        mdict : dict
            Dictionary of magnet name:current pairs. Specified quads are updated to 
            have this value. Overwrites values set by .mstate file if both are specified.
        units : str
            Passed to change quads to make appropriate conversion. Default: 'Amps'.
        maxdriftlen : float
            Maximum drift length [m].
        coef_filename : str
            File name for magnet coefficients.
        """
        if xml is None:  
            xml = os.path.join(self.default.home, self.default.defaultdict["XML_FILE"])
        print('xml:', xml)
        
        self.latgen = lattice_factory.LatticeGenerator(
            xml=xml, 
            beamlines=beamlines, 
            maxdriftlen=maxdriftlen, 
            coef_filename=coef_filename,
        )
        if mstatename:
            self.change_quads(filename=mstatename, units=units)
        if mdict:
            self.change_quads(dict=mdict, units=units)
        self.lattice = self.latgen.lattice

    def change_quads(self, **kwargs):
        """Change lattice quadrupole strengths.
        
        Can either import new xml lattice or update quads in existing lattice.
        """
        # -- if filename is passed,
        filename = kwargs.pop("filename", None)
        thisdict = kwargs.pop("dict", [])
        
        # Set units for quad values: current [Amps] or GL [Tesla].
        units = kwargs.pop("units", "Amps") 

        # Set quads according to mstate file.
        if filename:
            # if filename[-6:] == 'mstate':
            self.latgen.load_quads(filename=filename, units=units)
            # if loaded from mstate, units are in Tesla by default, no need to specify

        # Set quads according to dict.
        if thisdict:
            self.latgen.update_quads(dict=thisdict, units=units)

        # Assume any remaining named arguments are quad:current pairs.
        for quad, current in kwargs.items():
            self.latgen.update_quads(dict={quad: current}, units=units)

        # Update the lattice model.
        self.lattice = self.latgen.lattice

    def init_sc_nodes(self, minlen=0.015, solver="ellipse", nellipse=1, 
                      gridmult=6, n_bunches=None):
        """Set up space charge nodes.

        minlen = 0.015
        solver = 'ellipse' (default) or 'fft'
        nellipse = 1 (number of ellipses, only relevant if solver=ellipse)
        gridmult = 6 (size of grid = 2**gridmult. only relevant if solver=fft)
        n_bunches = None (number of neighboring bunches to include in calculation)
                    If None, a single bunch is modelled using "normal" Grid3D solver
                    If =0, a single bunch is modelled using new Grid3D with periodic boundary
                    If >0, n_bunches are modeled (odd numbers round up to even. Can only
                    model an even number of neighboring bunches)
                    Right now new Grid3D only available on andrei's version of py-orbit
        """
        sc_path_length_min = minlen
        sc_solver = solver
        nellipse = int(nellipse)
        gridmult = int(gridmult)

        print("Set up Space Charge nodes. ")
        if sc_solver == "ellipse":
            # Uniformly charged ellipsoid space charge solver. The more ellipsoids
            # are used, the more accurate of space charge calculation. This 
            # ellipse method can be used for the initial estimate because it
            # is faster than the FFT method.
            calcUnifEllips = SpaceChargeCalcUnifEllipse(nellipse)
            space_charge_nodes = setUniformEllipsesSCAccNodes(
                self.latgen.lattice, sc_path_length_min, calcUnifEllips
            )
        else:
            # 3D FFT space charge solver. The number of macro-particles 
            # should be increased by m**3 when the grid resolution increases
            # by factor m.
            size_x = size_y = size_z = 2**gridmult
            calc3d = SpaceChargeCalc3D(size_x, size_y, size_z)
            if n_bunches:
                calc3d.numExtBunches(n_bunches)
                calc3d.freqOfBunches(self.freq)
            space_charge_nodes = setSC3DAccNodes(
                self.latgen.lattice, sc_path_length_min, calc3d)

        max_sc_length = 0.0
        min_sc_length = self.latgen.lattice.getLength()
        for sc_node in space_charge_nodes:
            scL = sc_node.getLengthOfSC()
            if scL > max_sc_length:
                max_sc_length = scL
            if scL < min_sc_length:
                min_sc_length = scL

        self.lattice = self.latgen.lattice
        self.scnodes = space_charge_nodes

    def init_apertures(self, aprt_pipe_diameter=0.04):
        aprtNodes = Add_quad_apertures_to_lattice(self.latgen.lattice)
        aprtNodes = Add_bend_apertures_to_lattice(
            self.latgen.lattice, aprtNodes, step=0.1
        )
        aprt_drift_step = 0.1
        pos_aprt_start = 0.0
        pos_aprt_end = self.lattice.getLength()
        aprtNodes = Add_drift_apertures_to_lattice(
            self.latgen.lattice,
            pos_aprt_start,
            pos_aprt_end,
            aprt_drift_step,
            aprt_pipe_diameter,
            aprtNodes,
        )
        ## This will print out the all aperture nodes and their positions
        # for node in aprtNodes:
        #     print "aprt=", node.getName()," pos =", node.getPosition()
        self.lattice = self.latgen.lattice
        print("===== Aperture Nodes Added =======")

    def init_single_particle(self, x=0.0, xp=0.0, y=0.0, yp=0.0, z=0.0, dE=0.0):
        """Initialize bunch with one particle. ([m], [rad], [MeV])."""
        # Convert to [m], [rad], [GeV].
        x0 = x * 1e-3
        y0 = y * 1e-3
        xp0 = xp * 1e-3
        yp0 = yp * 1e-3
        z0 = z * 1e-3
        dE0 = dE * 1e-3
        self.bunch_in = Bunch()
        self.bunch_in.charge(self.charge)
        self.bunch_in.addParticle(x0, xp0, y0, yp0, z0, dE0)
        self.bunch_in.getSyncParticle().kinEnergy(self.ekin)
        self.bunch_in.macroSize(1)
        self.z2phase = 1.0  # dummy coefficient for z to phase (fix this)
        nparts = self.bunch_in.getSize()
        print("Bunch Generation completed with {} macroparticles.".format(nparts))

    def init_bunch(
        self, 
        bunch_generator='twiss', 
        center=True, 
        twiss=None,
        bunch_filename=None,
        bunch_file_format='pyorbit',
        **kwargs
    ):
        """Initialize simulation bunch.

        Only sampling of 2 independent phase spaces x-x' and y-y' is implemented. 
        Default is 200k particles, 40 mA peak, 3D waterbag distribution with 
        default Twiss parameters.

        Parameters
        ----------
        bunch_generator : str
            Options are {'twiss', 'load', 'twiss', '2d', '2dx3', '2d+E'}.
            if gen=="twiss", need to define all arguments ax,bx,ex, etc or accept defaults above
            if gen=="2d", need to supply arguments:
                xfile = path to distribution map for x phase space
                yfile = path to distribution map for y phase space
                dist is used to determine 2D longitudinal distribution only
                threshold is fractional threshold for provided emittance data
            if gen=="2d+E", need to supply arguments:
                xfile = path to distribution map for x phase space
                yfile = path to distribution map for y phase space
                efile = path to measurement of energy profile
                dist is used to determine 1D phase distribution only
                threshold is fractional threshold for provided emittance data
            if gen=="2dx3", need to supply arguments:
                xfile = path to distribution map for x phase space
                yfile = path to distribution map for y phase space
                zfile = path to measurement of energy profile
                threshold is fractional threshold for emittance data
            if gen=="load", need to supply arguments:
                file = path to file containing macro particle distribution
                fileformat = 'pyorbit' ('parmteq' is also an option)
        dist : str
            Distribution name (if generating from Twiss parameters). Options 
            are {'waterbag', 'kv', 'gaussian'}.
        threshold : float
            1e-6
        cutoff = -1 {only used if dist=gaussian)
        current = 0.040 {A}
        nparts = 200,000 {not valid if gen=load}
        center : bool
            Whether to center the bunch.
        twiss : dict
            Input Twiss parameters normalized RMS value; emit: mm-mrad; beta: mm/mrad.
        """
        if bunch_generator not in ['load', 'twiss', '2d', '2dx3', '2d+E']:
            raise KeyError(
                "bunch_generator={} not valid. allowed generators: ['load', 'twiss', '2d', '2dx3', '2d+E']"
                .format(bunch_generator)
            )
        # Default Twiss parameters (not used if generating function does not call for them...)
        if twiss is None:
            twiss = dict()
        twiss.setdefault('alpha_x', -1.9899)
        twiss.setdefault('beta_x', 0.19636)
        twiss.setdefault('eps_x', 0.160372)
        twiss.setdefault('alpha_y', 1.92893)
        twiss.setdefault('beta_y', 0.17778)
        twiss.setdefault('eps_y', 0.160362)
        twiss.setdefault('alpha_z', 0.0)
        twiss.setdefault('beta_z', 0.6)
        twiss.setdefault('eps_z', 0.2)

        # Make emittances un-normalized XAL units [m*rad].
        twiss['eps_x'] = 1.0e-6 * twiss['eps_x'] / (self.gamma * self.beta)
        twiss['eps_y'] = 1.0e-6 * twiss['eps_y'] / (self.gamma * self.beta)
        twiss['eps_z'] = 1.0e-6 * twiss['eps_z'] / (self.gamma**3 * self.beta)

        # Transform to pyorbit emittance [GeV*m].
        twiss['eps_z'] = twiss['eps_z'] * self.gamma**3 * self.beta**2 * self.mass
        twiss['beta_z'] = twiss['beta_z'] / (self.gamma**3 * self.beta**2 * self.mass)
        if bunch_generator == "twiss":
            print("========= PyORBIT Twiss ===========")
            print(
                "alpha beta emitt[mm*mrad] X= %6.4f %6.4f %6.4f "
                % (alpha_x, beta_x, eps_x * 1.0e6)
            )
            print(
                "alpha beta emitt[mm*mrad] Y= %6.4f %6.4f %6.4f "
                % (alpha_y, beta_y, eps_y * 1.0e6)
            )
        if bunch_generator in ["twiss", "2d", "2d+E"]:
            print(
                "alpha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f "
                % (alpha_z, beta_z, eps_z * 1.0e6)
            )

        if bunch_generator == "load":
            if bunch_filename is None:
                bunch_filename = os.path.join(
                    self.default.home,
                    self.default.defaultdict["BUNCH_IN"],
                )
            bunch_gen = gen_bunch.Base_BunchGenerator()

            # -- generate bunch by reading in file w/  macroparticle coordinates
            if not os.path.isfile(bunch_filename):
                raise ValueError("Bunch file '{}' does not exist.".format(bunch_filename))
            print("Reading in bunch from file %s..." % bunch_filename)
            if bunch_file_format == "pyorbit":
                self.bunch_in = Bunch()
                self.bunch_in.readBunch(bunch_filename)
            elif bunch_file_format == "parmteq":
                self.bunch_in = bunch_gen.read_parmteq_bunch(bunch_filename)
            else:
                raise KeyError(
                    "Do not recognize format %s for bunch file" % (bunch_file_format)
                )
            # -- read in nparts and macrosize
            nparts = self.bunch_in.getSize()

            macrosize = self.bunch_in.macroSize()
            # -- overwrite mass and charge:
            # self.bunch_in.getSyncParticle().kinEnergy(self.ekin)
            self.bunch_in.mass(self.mass)
            self.bunch_in.charge(self.charge)

            self.z2phase = bunch_gen.get_z_to_phase_coeff()
            self.current = bunch_gen.get_beam_current() * 1e-3

            ## -- to-do: insert function to extract peak current from coordinates
            ## peak current = N_peak * macroSize * consts.charge_electron * self.beta * consts.speed_of_light
            ## where N_peak is peak number density
            print("Bunch read completed. Imported %i macroparticles." % nparts)

        # Generate bunch by Twiss params or measured distributions.
        if bunch_generator in ["twiss", "2d", "2d+E", "2dx3"]:
            nparts = kwargs.get("nparts", 200000)
            self.current = kwargs.get("current", 0.040)
            bunchdistributor = kwargs.get("dist", "gaussian") 
            cut_off = kwargs.get("cutoff", -1)
            distributor_class = None
            if bunchdistributor == "gaussian":
                if bunch_generator == "twiss":
                    distributor_class = GaussDist3D
                elif bunch_generator in ["2d", "2d+E"]:
                    distributor_class = GaussDist1D
            elif bunchdistributor == "waterbag":
                if bunch_generator == "twiss":
                    distributor_class = WaterBagDist3D
                elif bunch_generator in ["2d", "2d+E"]:
                    distributor_class = WaterBagDist1D
            elif bunchdistributor == "kv":
                if bunch_generator == "twiss":
                    distributor_class = KVDist3D
                elif bunch_generator in ["2d", "2d+E"]:
                    distributor_class = KVDist1D
            else:
                raise ValueError(
                    "Unrecognized distributor {}. Accepted classes are 'gaussian', 'waterbag', 'kv'."
                    .format(bunchdistributor)
                )

            # Make generator instances.
            if bunch_generator == "twiss":
                twiss_x = TwissContainer(alpha_x, beta_x, eps_x)
                twiss_y = TwissContainer(alpha_y, beta_y, eps_y)
                twiss_z = TwissContainer(alpha_z, beta_z, eps_z)
                print("Generating bunch based off twiss parameters (N = {})".format(nparts))
                bunch_gen = gen_bunch.BTF_Linac_bunch_generator(
                    twiss_x,
                    twiss_y,
                    twiss_z,
                    mass=self.mass,
                    charge=self.charge,
                    ekin=self.ekin,
                    curr=self.current * 1e3,
                    freq=self.freq,
                )

            elif bunch_generator in ["2d", "2d+E", "2dx3"]:
                xfilename = kwargs.get("xfile", "")
                yfilename = kwargs.get("yfile", "")
                sample_method = kwargs.get("sample", "cdf")
                thres = kwargs.get("threshold", 1e-6)

                phaseSpGenX = gen_bunch.PhaseSpaceGen(xfilename, threshold=thres)
                phaseSpGenY = gen_bunch.PhaseSpaceGen(yfilename, threshold=thres)
                twiss_z = TwissContainer(alpha_z, beta_z, eps_z)

                if bunch_generator == "2d":

                    print(
                        "Generating bunch based off 2d emittance measurements ( N = %i )"
                        % nparts
                    )
                    bunch_gen = gen_bunch.BTF_Linac_TrPhaseSpace_bunch_generator(
                        phaseSpGenX,
                        phaseSpGenY,
                        twiss_z,
                        mass=self.mass,
                        charge=self.charge,
                        ekin=self.ekin,
                        curr=self.current * 1e3,
                        freq=self.freq,
                        method=sample_method,
                    )

                elif bunch_generator == "2d+E":

                    efilename = kwargs.get("efile", "")
                    phaseSpGenZ = gen_bunch.PhaseSpaceGenZPartial(
                        efilename,
                        twiss_z,
                        zdistributor=distributor_class,
                        cut_off=cut_off,
                    )

                    print(
                        "Generating bunch based off 2d emittance + 1d energy profile measurements ( N = %i )"
                        % nparts
                    )
                    bunch_gen = gen_bunch.BTF_Linac_6DPhaseSpace_BunchGenerator(
                        phaseSpGenX,
                        phaseSpGenY,
                        phaseSpGenZ,
                        mass=self.mass,
                        charge=self.charge,
                        ekin=self.ekin,
                        curr=self.current * 1e3,
                        freq=self.freq,
                        method=sample_method,
                    )
                elif bunch_generator == "2dx3":

                    zfilename = kwargs.get("zfile", "")
                    phaseSpGenZ = gen_bunch.PhaseSpaceGen(zfilename, threshold=thres)

                    print(
                        "Generating bunch based off 2d emittances in x,y,z planes ( N = %i )"
                        % nparts
                    )
                    # -- is this the right method?
                    bunch_gen = gen_bunch.BTF_Linac_6DPhaseSpace_BunchGenerator(
                        phaseSpGenX,
                        phaseSpGenY,
                        phaseSpGenZ,
                        mass=self.mass,
                        charge=self.charge,
                        ekin=self.ekin,
                        curr=self.current * 1e3,
                        freq=self.freq,
                        method=sample_method,
                    )

            # -- set the initial kinetic energy in GeV
            bunch_gen.setKinEnergy(self.ekin)

            # -- generate bunch
            self.bunch_in = bunch_gen.get_bunch(
                nParticles=int(nparts), 
                distributor_class=distributor_class,
            )

            # -- save coefficient for Z to Phase
            self.z2phase = bunch_gen.get_z_to_phase_coeff()

            # -- report
            nparts = self.bunch_in.getSize()
            print("Bunch Generation completed with %i macroparticles." % nparts)

        if center:
            self.center_bunch()

    def center_bunch(self):
        """Force the bunch to be centered.
        
        Typically used to correct small deviations from 0.
        """
        print("Centering bunch...")
        twiss_analysis = BunchTwissAnalysis()
        twiss_analysis.analyzeBunch(self.bunch_in)
        x_avg = twiss_analysis.getAverage(0)
        y_avg = twiss_analysis.getAverage(2)
        z_avg = twiss_analysis.getAverage(4)
        xp_avg = twiss_analysis.getAverage(1)
        yp_avg = twiss_analysis.getAverage(3)
        dE_avg = twiss_analysis.getAverage(5)
        print("before correction: x={:.6f}, x'={:.6f}, y={:.6f}, y'={:.6f}, z={:.6f}, dE={:.6f}"
              .format(x_avg, xp_avg, y_avg, yp_avg, z_avg, dE_avg))
        for part_id in range(self.bunch_in.getSize()):
            self.bunch_in.x(part_id, self.bunch_in.x(part_id) - x_avg)
            self.bunch_in.y(part_id, self.bunch_in.y(part_id) - y_avg)
            self.bunch_in.xp(part_id, self.bunch_in.xp(part_id) - xp_avg)
            self.bunch_in.yp(part_id, self.bunch_in.yp(part_id) - yp_avg)
            self.bunch_in.z(part_id, self.bunch_in.z(part_id) - z_avg)
            self.bunch_in.dE(part_id, self.bunch_in.dE(part_id) - dE_avg)
        print("Bunch centered.")

    def shift_bunch(self, x0=0.0, xp0=0.0, y0=0.0, yp0=0.0, z0=0.0, dE0=0.0):
        print('Shifting bunch centroid...')
        for i in range(self.bunch_in.getSize()):
            self.bunch_in.x(i, self.bunch_in.x(i) + x0)
            self.bunch_in.y(i, self.bunch_in.y(i) + y0)  
            self.bunch_in.z(i, self.bunch_in.z(i) + z0)
            self.bunch_in.xp(i, self.bunch_in.xp(i) + xp0) 
            self.bunch_in.yp(i, self.bunch_in.yp(i) + yp0)
            self.bunch_in.dE(i, self.bunch_in.dE(i) + dE0)
        print('Bunch shifted.')

    def attenuate_bunch(self, att):
        """Adjust current without changing the number of particles.
        
        att : float
            The fractional attenuation.
        """
        macrosize = self.bunch_in.macroSize()
        self.bunch_in.macroSize(macrosize * att)

    def decimate_bunch(self, dec):
        """Reduce number of macro-particles without changing beam current.
        
        dec : float
            Bunch is decimated by factor 10**dec.
        """
        nparts0 = self.bunch_in.getSizeGlobal()
        stride = nparts0 / 10.0**dec
        if (0 < dec < np.log10(nparts0)):
            ind = np.arange(0, nparts0, stride).astype(int)
            nparts = len(ind)
            newbunch = Bunch()
            self.bunch_in.copyEmptyBunchTo(newbunch)
            for i in ind:
                newbunch.addParticle(
                    self.bunch_in.x(i),
                    self.bunch_in.xp(i),
                    self.bunch_in.y(i),
                    self.bunch_in.yp(i),
                    self.bunch_in.z(i),
                    self.bunch_in.dE(i),
                )
            newmacrosize = self.bunch_in.macroSize() * stride
            newbunch.macroSize(newmacrosize)
            newbunch.copyBunchTo(self.bunch_in)
        else:
            print("No decimation for 10^{}.".format(dec))

    def resample_bunch(self, nparts, rms_factor=0.05):
        """Up/down-sample to obtain requested number of particles.

        nparts = number of desired particles in bunch.
        """
        nparts0 = self.bunch_in.getSizeGlobal()
        mult = float(nparts) / float(nparts0)
        nparts = int(nparts)
        print("Resampling bunch from %i to %i particles..." % (nparts0, nparts))

        print("mult = %.3f" % mult)
        print(mult == 1)

        # -- make an array of existing coordinates (length=nparts0)
        coords0 = np.zeros([nparts0, 6])
        for i in range(nparts0):
            coords0[i, :] = [
                self.bunch_in.x(i),
                self.bunch_in.xp(i),
                self.bunch_in.y(i),
                self.bunch_in.yp(i),
                self.bunch_in.z(i),
                self.bunch_in.dE(i),
            ]

        if mult == 1:
            return []
        else:

            # -- down-sample if nparts0 > nparts-new
            if mult < 1:
                ind = np.random.permutation(np.arange(nparts0))[0:nparts]

            # -- up-sample if nparts0 < nparts-new
            # -- this way is a lot of work
            elif mult > 1:
                nnew = nparts - nparts0

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
                    loc=coords0, scale=scale, size=[intmult, nparts0, 6])
                reshape_coords = np.zeros([intmult * nparts0, 6], dtype="f8")
                for i in range(6):
                    reshape_coords[:, i] = newcoords[:, :, i].flatten()

                coords0 = reshape_coords.copy()

                # -- and downsample to desired number
                ind = np.random.permutation(np.arange(len(coords0)))[0:nparts]

            # -- make new bunch and place re-sampled coordinates
            newbunch = Bunch()
            self.bunch_in.copyEmptyBunchTo(newbunch)  # copy attributes
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
            newmacrosize = self.bunch_in.macroSize() * (1.0 / mult)
            newbunch.macroSize(newmacrosize)

            # -- over-write bunch_in
            newbunch.copyBunchTo(self.bunch_in)

            print("done resampling")
            return newbunch
        
    def decorrelate_bunch(self):
        """Remove inter-plane correlations by permuting (x, x'), (y, y'), (z, z') pairs."""
        print('Removed correlations between x-y-z phase planes.')
        self.bunch_in = butils.decorrelate_x_y_z(self.bunch_in)
    
    def dump_bunch(self, filename):
        """Dump the bunch coordinate array."""
        self.bunch_in.dumpBunch(filename)

    def dump_parmila(self, **kwargs):
        filename = kwargs.get("filename", [])
        bunch_gen = gen_bunch.Base_BunchGenerator()
        if filename:
            bunch_gen.dump_parmila_file(
                self.bunch_in, phase_init=-0.0, fileName=filename
            )
            print("Bunch dumped to %s" % filename)
        else:
            bunch_gen.dump_parmila_file(self.bunch_in, phase_init=-0.0)
            print("Bunch dumped to default file")
