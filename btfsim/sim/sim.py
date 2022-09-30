from __future__ import print_function
import sys
import os
import time

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from spacecharge import SpaceChargeCalcUnifEllipse
from spacecharge import SpaceChargeCalc3D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import TwissAnalysis
from orbit.bunch_generators import KVDist1D
from orbit.bunch_generators import KVDist2D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import GaussDist1D
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import WaterBagDist1D
from orbit.bunch_generators import WaterBagDist2D
from orbit.bunch_generators import WaterBagDist3D
from orbit.lattice import AccActionsContainer
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_bend_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
import orbit.utils.consts as consts

import btfsim.bunch.btf_linac_bunch_generator as bg
import btfsim.bunch.utils as butils
from btfsim.lattice import lattice_factory
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
    def __init__(self, outdir=None, tracker_kws=None):
        """Constructor.
        
        tracker_kws : dict
            Key word arguments passed to BunchTracker.
        """
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
 
        self.tracker_kws = tracker_kws
        if self.tracker_kws is None:
            self.tracker_kws = dict()

    def init_all(self, init_bunch=True):
        """Set up full simulation with all default values."""
        self.init_lattice()
        self.init_apertures()
        self.init_sc_nodes()
        if init_bunch:
            self.init_bunch()        

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
            startnode, start_num, zs_startnode, ze_startnode = self.lattice.getNodeForPosition(start)
        elif type(start) is str:
            startnode = self.lattice.getNodeForName(start)
            start_num = self.lattice.getNodeIndex(startnode)
            zs_startnode = startnode.getPosition() - 0.5 * startnode.getLength()
            ze_startnode = startnode.getPosition() + 0.5 * startnode.getLength()
        else:
            raise TypeError("Invalid type {} for `start`.".format(type(start)))
            
        if type(stop) in [float, int]:            
            print("max simulation length = {:.3f}.".format(self.lattice.getLength()))
            if stop > self.lattice.getLength():
                stop = self.lattice.getLength()
            stopnode, stop_num, zs_stopnode, ze_stopnode = self.lattice.getNodeForPosition(stop)
        elif type(stop) is str:
            stopnode = self.lattice.getNodeForName(stop)
            stop_num = self.lattice.getNodeIndex(stopnode)
            zs_stopnode = stopnode.getPosition() - 0.5 * stopnode.getLength()
            ze_stopnode = stopnode.getPosition() + 0.5 * stopnode.getLength()
        else:
            raise TypeError("Invalid type {} for `stop`.".format(type(start)))
            
        print(
            "Running simulation from s = {:.4f} [m] to s = {:.4f} [m] (nodes {} to {})."
            .format(zs_startnode, ze_stopnode, start_num, stop_num)
        )

        # Setup
        params_dict = {
            "old_pos": -1.0,
            "count": 0,
            "pos_step": 0.005,
        } 
        action_container = AccActionsContainer("BTF Bunch Tracking")

        # Add tracking action; load different tracking routine if bunch has only 
        # one particle (single-particle tracking).
        if self.bunch_in.getSize() > 1:
            self.tracker = butils.BunchTracker(**self.tracker_kws)
        elif self.bunch_in.getSize() == 1:
            self.tracker = butils.SingleParticleTracker(**self.tracker_kws)
        action_container.addAction(self.tracker.action_exit, AccActionsContainer.EXIT)

        # Run the simulation.
        time_start = time.clock()
        self.bunch_track = Bunch()
        self.bunch_in.copyBunchTo(self.bunch_track)
        self.lattice.trackBunch(
            self.bunch_track,
            paramsDict=params_dict,
            actionContainer=action_container,
            index_start=start_num,
            index_stop=stop_num,
        )
        
        # Save the last time step.
        params_dict["old_pos"] = -1
        self.tracker.action_entrance(params_dict)

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
        bg.bunch_transformer_func(self.bunch_in)

        # Run in reverse (start <==> stop). Do not auto-save output bunch; the bunch
        # coordinates need to be reversed first.
        self.run(start=stop, stop=start, out=None)

        # Un-reverse the lattice.
        self.latgen.lattice.reverseOrder()
        self.lattice = self.latgen.lattice

        # Un-reverse the bunch coordinates, then save them.
        bg.BunchTransformerFunc(self.bunch_track)
        if out:
            self.bunch_track.dumpBunch(output_filename)

        # Also un-reverse the initial bunch coordinates.
        bg.BunchTransformerFunc(self.bunch_in)

        # Wrap up.
        if type(stop) in [float, int]:
            stopnode, stop_num, zs_stopnode, ze_stopnode = self.lattice.getNodeForPosition(stop)
        elif type(stop) == str:
            stopnode = self.lattice.getNodeForName(stop)
            stop_num = self.lattice.getNodeIndex(stopnode)
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
            self.update_quads(filename=mstatename, units=units)
        if mdict:
            self.update_quads(spdict=mdict, units=units)
        self.lattice = self.latgen.lattice

    def update_quads(self, filename=None, units='Amps', **spdict):
        """Change lattice quadrupole strengths.
        
        Can either import new xml lattice or update quads in existing lattice.
        """
        if filename is not None:
            self.latgen.load_quads(filename=filename, units=units)
        self.latgen.update_quads(units=units, **spdict)
        self.lattice = self.latgen.lattice

    def init_sc_nodes(self, min_dist=0.015, solver='fft', n_ellipsoid=1, 
                      gridmult=6, n_bunches=None):
        """Initialize space charge nodes.

        min_dist : float
            Minimum distance between nodes [m].
        solver : {'ellipsoid', 'fft'}
            Type of space charge solver.
        n_ellipsoid : int
            Number of ellipsoids (if using 'ellipsoid' solver).
        gridmult : int
            Size of grid is 2**gridmult (if using 'fft' solver).
        n_bunches : int (even)
            Number of neighboring bunches to include in calculation. If None, a 
            single bunch is modeled using the normal Grid3D solver. If 0, a 
            single bunch is modeled using a new Grid3D with periodic boundary
            conditions. If > 0, `n_bunches` are modeled. (Odd numbers round up to 
            even; can only model an even number of neighboring bunches.)
        """
        print("Initializing space charge nodes...")
        n_ellipsoid = int(n_ellipsoid)
        gridmult = int(gridmult)
        if solver == 'ellipsoid':
            # Uniformly charged ellipsoid space charge solver. The more ellipsoids
            # are used, the more accurate of space charge calculation. This 
            # ellipse method can be used for the initial estimate because it
            # is faster than the FFT method.
            calc = SpaceChargeCalcUnifEllipse(n_ellipsoid)
            sc_nodes = setUniformEllipsesSCAccNodes(self.latgen.lattice, min_dist, calc)
        else:
            # 3D FFT space charge solver. The number of macro-particles 
            # should be increased by m**3 when the grid resolution increases
            # by factor m.
            size_x = size_y = size_z = 2**gridmult
            calc = SpaceChargeCalc3D(size_x, size_y, size_z)
            if n_bunches:
                calc.numExtBunches(n_bunches)
                calc.freqOfBunches(self.freq)
            sc_nodes = setSC3DAccNodes(self.latgen.lattice, min_dist, calc)
        self.lattice = self.latgen.lattice
        self.scnodes = sc_nodes
        print('Space charge nodes initialized.')

    def init_apertures(self, aprt_pipe_diameter=0.04):
        """Initialize apertures."""
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
        """Initialize bunch with one particle."""
        self.bunch_in = Bunch()
        self.bunch_in.charge(self.charge)
        self.bunch_in.addParticle(x, xp, y, yp, z, dE)
        self.bunch_in.getSyncParticle().kinEnergy(self.ekin)
        self.bunch_in.macroSize(1)
        self.z2phase = 1.0  # dummy coefficient for z to phase (fix this)
        n_parts = self.bunch_in.getSize()
        print('Generated single-particle bunch.')

    def init_bunch(
        self, 
        gen_type='twiss', 
        dist='waterbag',
        center=True, 
        twiss=None,
        bunch_filename=None,
        bunch_file_format='pyorbit',
        n_parts=200000,
        current=0.040,
        cut_off=-1,
        xfilename='',
        yfilename='',
        zfilename='',
        sample_method='cdf',
        thresh=1e-6,
    ):
        """Initialize simulation bunch.

        Only sampling of 2 independent phase spaces x-x' and y-y' is implemented. 
        Default is 200k particles, 40 mA peak, 3D waterbag distribution with 
        default Twiss parameters.

        Parameters
        ----------
        gen_type : str
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
            
        cutoff : int
            Cutoff for truncated Gaussian distribution.
        current :
            Beam current [A].
        n_parts : int
            Number of macro-particles (unused if loading bunch from file).
        center : bool
            Whether to center the bunch.
        twiss : dict
            Input Twiss parameters normalized RMS value; emit: mm-mrad; beta: mm/mrad.
        """
        if gen_type not in ['load', 'twiss', '2d', '2dx3', '2d+E']:
            raise KeyError(
                "gen_type={} not valid. allowed generators: ['load', 'twiss', '2d', '2dx3', '2d+E']"
                .format(gen_type)
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
        alpha_x = twiss['alpha_x']
        alpha_y = twiss['alpha_y']
        alpha_z = twiss['alpha_z']
        beta_x = twiss['beta_x']
        beta_y = twiss['beta_y']
        beta_z = twiss['beta_z']
        eps_x = twiss['eps_x']
        eps_y = twiss['eps_y']
        eps_z = twiss['eps_z']
        twiss_x = TwissContainer(alpha_x, beta_x, eps_x)
        twiss_y = TwissContainer(alpha_y, beta_y, eps_y)
        twiss_z = TwissContainer(alpha_z, beta_z, eps_z)
        
        # Make emittances un-normalized XAL units [m*rad].
        eps_x = 1.0e-6 * eps_x / (self.gamma * self.beta)
        eps_y = 1.0e-6 * eps_y / (self.gamma * self.beta)
        eps_z = 1.0e-6 * eps_z / (self.gamma**3 * self.beta)

        # Transform to pyorbit emittance [GeV*m].
        eps_z = eps_z * self.gamma**3 * self.beta**2 * self.mass
        beta_z = beta_z / (self.gamma**3 * self.beta**2 * self.mass)
        if gen_type == "twiss":
            print("========= PyORBIT Twiss ===========")
            print('alpha_x = {:6.4f}'.format(alpha_x))
            print('beta_x = {:6.4f} [mm/mrad]'.format(beta_x))
            print('eps_x = {:6.4f} [mm*mrad]'.format(1.0e6 * eps_x))
            print('alpha_y = {:6.4f}'.format(alpha_y))
            print('beta_y = {:6.4f} [mm/mrad]'.format(beta_y))
            print('eps_y = {:6.4f} [mm*mrad]'.format(1.0e6 * eps_y))
        if gen_type in ["twiss", "2d", "2d+E"]:
            print('alpha_z = {:6.4f}'.format(alpha_z))
            print('beta_z = {:6.4f} [mm/mrad]'.format(beta_z))
            print('eps_z = {:6.4f} [mm*MeV]'.format(1.0e6 * eps_z))
            
        if gen_type == "load":
            if bunch_filename is None:
                bunch_filename = os.path.join(
                    self.default.home, 
                    self.default.defaultdict["BUNCH_IN"],
                )
            bunch_gen = bg.Base_BunchGenerator()

            if not os.path.isfile(bunch_filename):
                raise ValueError("Bunch file '{}' does not exist.".format(bunch_filename))
            print("Reading in bunch from file %s..." % bunch_filename)
            if bunch_file_format == "pyorbit":
                self.bunch_in = Bunch()
                self.bunch_in.readBunch(bunch_filename)
            elif bunch_file_format == "parmteq":
                self.bunch_in = bg.read_parmteq_bunch(bunch_filename)
            else:
                raise KeyError("Do not recognize format {} for bunch file".format(bunch_file_format))
                
            n_parts = self.bunch_in.getSize()
            macrosize = self.bunch_in.macroSize()
            self.bunch_in.getSyncParticle().kinEnergy(self.ekin)
            self.bunch_in.mass(self.mass)
            self.bunch_in.charge(self.charge)
            self.z2phase = bunch_gen.get_z_to_phase_coeff()
            self.current = bunch_gen.get_beam_current() * 1e-3

            ## TODO: insert function to extract peak current from coordinates
            ## peak current = N_peak * macroSize * consts.charge_electron * self.beta * consts.speed_of_light
            ## where N_peak is peak number density
            print("Bunch read completed. Imported {} macroparticles.".format(n_parts))

        elif gen_type in ["twiss", "2d", "2d+E", "2dx3"]:
            self.current = current
            dist_class = None
            if dist == "gaussian":
                if gen_type == "twiss":
                    dist_class = GaussDist3D
                elif gen_type in ["2d", "2d+E"]:
                    dist_class = GaussDist1D
            elif dist == "waterbag":
                if gen_type == "twiss":
                    dist_class = WaterBagDist3D
                elif gen_type in ["2d", "2d+E"]:
                    dist_class = WaterBagDist1D
            elif dist == "kv":
                if gen_type == "twiss":
                    dist_class = KVDist3D
                elif gen_type in ["2d", "2d+E"]:
                    dist_class = KVDist1D
            if dist_class is None:
                raise ValueError(
                    "Unrecognized distributor {}. Accepted classes are 'gaussian', 'waterbag', 'kv'."
                    .format(dist)
                )
            if gen_type == "twiss":
                print("Generating bunch based off twiss parameters (N = {})".format(n_parts))
                bunch_gen = bg.BTF_Linac_BunchGenerator(
                    twiss_x,
                    twiss_y,
                    twiss_z,
                    mass=self.mass,
                    charge=self.charge,
                    ekin=self.ekin,
                    curr=self.current * 1e3,
                    freq=self.freq,
                )
            elif gen_type in ["2d", "2d+E", "2dx3"]:
                phase_sp_gen_x = bg.PhaseSpaceGen(xfilename, threshold=thresh)
                phase_sp_gen_y = bg.PhaseSpaceGen(yfilename, threshold=thresh)
                twiss_z = TwissContainer(alpha_z, beta_z, eps_z)
                if gen_type == "2d":
                    print("Generating bunch based off 2d emittance measurements (n_parts = {})."
                          .format(n_parts))
                    bunch_gen = bg.BTF_Linac_TrPhaseSpace_BunchGenerator(
                        phase_sp_gen_x,
                        phase_sp_gen_y,
                        twiss_z,
                        mass=self.mass,
                        charge=self.charge,
                        ekin=self.ekin,
                        curr=self.current * 1e3,
                        freq=self.freq,
                        method=sample_method,
                    )

                elif gen_type == "2d+E":
                    efilename = kwargs.get("efile", "")
                    phase_sp_gen_z = bg.PhaseSpaceGenZPartial(
                        efilename,
                        twiss_z,
                        zdistributor=dist_class,
                        cut_off=cut_off,
                    )
                    print(
                        "Generating bunch based off 2d emittance + 1d energy profile measurements (n_parts = {})."
                        .format(n_parts)
                    )
                    bunch_gen = bg.BTF_Linac_6DPhaseSpace_BunchGenerator(
                        phase_sp_gen_x,
                        phase_sp_gen_y,
                        phase_sp_gen_z,
                        mass=self.mass,
                        charge=self.charge,
                        ekin=self.ekin,
                        curr=self.current * 1e3,
                        freq=self.freq,
                        method=sample_method,
                    )
                elif gen_type == "2dx3":
                    phase_sp_gen_z = bg.PhaseSpaceGen(zfilename, threshold=thresh)
                    print(
                        "Generating bunch based off 2d emittances in x,y,z planes (n_parts = {})."
                        .format(n_parts)
                    )
                    # -- is this the right method?
                    bunch_gen = bg.BTF_Linac_6DPhaseSpace_BunchGenerator(
                        phase_sp_gen_x,
                        phase_sp_gen_y,
                        phase_sp_gen_z,
                        mass=self.mass,
                        charge=self.charge,
                        ekin=self.ekin,
                        curr=self.current * 1e3,
                        freq=self.freq,
                        method=sample_method,
                    )
            bunch_gen.set_kin_energy(self.ekin)
            self.bunch_in = bunch_gen.get_bunch(
                n_parts=int(n_parts), 
                dist_class=dist_class,
            )
            self.z2phase = bunch_gen.get_z_to_phase_coeff()
            n_parts = self.bunch_in.getSize()
            print("Bunch Generation completed with {} macroparticles.".format(n_parts))
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

    def decimate_bunch(self, fac=1):
        """Reduce number of macro-particles without changing beam current."""
        return butils.decimate_bunch(self.bunch_in, fac=fac)

    def resample_bunch(self, n_parts, rms_factor=0.05):
        """Up/down-sample to obtain requested number of particles.

        n_parts = number of desired particles in bunch.
        """
        n_parts0 = self.bunch_in.getSizeGlobal()
        mult = float(n_parts) / float(n_parts0)
        n_parts = int(n_parts)
        print("Resampling bunch from %i to %i particles..." % (n_parts0, n_parts))

        print("mult = %.3f" % mult)
        print(mult == 1)

        # -- make an array of existing coordinates (length=n_parts0)
        coords0 = np.zeros([n_parts0, 6])
        for i in range(n_parts0):
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

            # -- down-sample if n_parts0 > n_parts-new
            if mult < 1:
                ind = np.random.permutation(np.arange(n_parts0))[0:n_parts]

            # -- up-sample if n_parts0 < n_parts-new
            # -- this way is a lot of work
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
                    loc=coords0, scale=scale, size=[intmult, n_parts0, 6])
                reshape_coords = np.zeros([intmult * n_parts0, 6], dtype="f8")
                for i in range(6):
                    reshape_coords[:, i] = newcoords[:, :, i].flatten()

                coords0 = reshape_coords.copy()

                # -- and downsample to desired number
                ind = np.random.permutation(np.arange(len(coords0)))[0:n_parts]

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
        bunch_gen = bg.Base_BunchGenerator()
        if filename:
            bunch_gen.dump_parmila_file(self.bunch_in, phase_init=-0.0, 
                                        fileName=filename)
            print("Bunch dumped to {}".format(filename))
        else:
            bunch_gen.dump_parmila_file(self.bunch_in, phase_init=-0.0)
            print("Bunch dumped to default file")
