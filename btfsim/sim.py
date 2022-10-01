from __future__ import print_function
import sys
import os
import time

import numpy as np
import pandas as pd

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

from btfsim import bunch_utils as bu
from btfsim.bunch_utils import BunchCalculator
from btfsim.lattice import LatticeGenerator
from btfsim.default import Default


# Define default Twiss parameters at RFQ exit.
INIT_TWISS = {
    'alpha_x': -1.9899,
    'beta_x': 0.19636,
    'eps_x': 0.160372,
    'alpha_y': 1.92893,
    'beta_y': 0.17778,
    'eps_y': 0.160362,
    'alpha_z': 0.0,
    'beta_z': 0.6,
    'eps_z': 0.2,
}


class BunchTracker:
    """Class to store bunch evolution data.
    
    Attributes
    ----------
    history : DataFrame
        Stores various scalar beam parameters at each node.
    plotter : btfsim.plot.Plotter
        Plot routine manager. BunchTracker can decide when the plotter
        should activate.
    plot_norm_coords : bool
        Whether to normalize the coordinates by the rms Twiss parameters
        before plotting.
    plot_scale_emittance : bool
        Whether to scale the coordinates by the rms emittances in 
        each phase plane before plotting.
    emit_norm_flag : bool
    dispersion_flag : bool
    """
    def __init__(
        self, 
        save_bunch=None,
        plotter=None,
        plot_norm_coords=False,
        plot_scale_emittance=False,
        dispersion_flag=False, 
        emit_norm_flag=False,
    ):
        self.save_bunch = save_bunch if save_bunch else dict()
        self.plotter = plotter
        self.plot_norm_coords = plot_norm_coords
        self.plot_scale_emittance = plot_scale_emittance
        self.dispersion_flag = dispersion_flag
        self.emit_norm_flag = emit_norm_flag
        self.twiss_analysis = BunchTwissAnalysis()
        keys = [
            "s",
            "node",
            "n_parts",
            "n_lost",
            "disp_x",
            "dispp_x",
            "r90",
            "r99",
        ]
        # Add 2D alpha, beta, and emittance keys.
        for dim in ["x", "y", "z"]:
            for name in ["alpha", "beta", "eps"]:
                keys.append("{}_{}".format(name, dim))
        # Add covariance matrix element keys (sig_11 = <xx>, sig_12 = <xx'>, etc.).
        for i in range(6):
            for j in range(i + 1):
                keys.append("sig_{}{}".format(j + 1, i + 1))
        self.history = {key: [] for key in keys}
        
    def delete_history(self):
        for key in self.history:
            self.history[key] = []

    def action(self, params_dict):
        node = params_dict["node"]
        bunch = params_dict["bunch"]
        position = params_dict["path_length"]
        if params_dict["old_pos"] == position:
            return
        if params_dict["old_pos"] + params_dict["pos_step"] > position:
            return
        params_dict["old_pos"] = position
        params_dict["count"] += 1

        # Print update statement.
        n_step = params_dict["count"]
        n_part = bunch.getSize()
        print("Step {}, n_parts {}, s={:.3f} [m], node {}"
              .format(n_step, n_part, position, node.getName()))

        # Compute bunch parameters.
        calc = BunchCalculator(bunch)
        twiss_x, twiss_y, twiss_z = [
            calc.twiss(dim=dim, emit_norm_flag=self.emit_norm_flag) 
            for dim in ('x', 'y', 'z')
        ]
        n_parts = bunch.getSizeGlobal()
        gamma = bunch.getSyncParticle().gamma()
        beta = bunch.getSyncParticle().beta()

        # Correctly assign the number of particles for the 0th step.
        if params_dict["count"] == 1:
            self.history["n_parts"][params_dict["count"] - 1] = n_parts

        # Update history.
        self.history["s"].append(position)
        self.history["node"].append(node.getName())
        self.history["n_parts"].append(n_parts)
        self.history["alpha_x"].append(twiss_x['alpha'])
        self.history["beta_x"].append(twiss_x['beta'])
        self.history["eps_x"].append(twiss_x['eps'])
        self.history["disp_x"].append(twiss_x['disp'])
        self.history["dispp_x"].append(twiss_x['disp'])
        self.history["alpha_y"].append(twiss_y['alpha'])
        self.history["beta_y"].append(twiss_y['beta'])
        self.history["eps_y"].append(twiss_y['eps'])
        self.history["alpha_z"].append(twiss_z['alpha'])
        self.history["beta_z"].append(twiss_z['beta'])
        self.history["eps_z"].append(twiss_z['eps'])
        for i in range(6):
            for j in range(i + 1):
                key = "sig_{}{}".format(j + 1, i + 1)
                self.history[key].append(calc.cov[j, i])
        self.history["n_lost"].append(self.history["n_parts"][0] - n_parts)
        
        # Make plots.
        if self.plotter is not None:
            info = dict()
            for key in self.history:
                info[key] = self.history[key][-1]
            info['step'] = params_dict['count']
            info['node'] = params_dict['node'].getName()
            info['gamma'] = params_dict['bunch'].getSyncParticle().gamma()
            info['beta'] = params_dict['bunch'].getSyncParticle().beta()  
            if self.plot_norm_coords:
                data = calc.norm_coords(scale_emittance=self.plot_scale_emittance)
            self.plotter.plot(data=data, info=info, verbose=True)
                                                                    
    def cleanup(self):
        # Convert history lists to numpy arrays.
        for key in self.history:
            self.history[key] = np.array(self.history[key])
        # Trim zeros from history.
        ind = np.where(np.self.history["sig_11"] == 0)[0][1]
        for key, arr in self.history.iteritems():
            istart = 0 if key == "node" else 1
            self.history[key] = arr[istart:ind]

    def write_hist(self, filename=None, sep=" "):
        """Save history data.

        filename = location to save data
        """
        if filename is None:
            filename = "history.dat"
        keys = list(self.history)
        data = np.array([self.history[key] for key in keys]).T
        df = pd.DataFrame(data=data, columns=keys)
        df.to_csv(filename, sep=sep, index=False)
        return df


class Simulation:
    """Class to hold simulation model.

    Attributes
    ----------
    kin_energy : float
        Kinetic energy of synchronous particle [GeV]
    mass : float
        Mass per particle [GeV/c^2]        
    beta : float
        Synchronous particle velocity relative to speed of light.
    gamma : float
        Lorentz factor 1 / sqrt(1 - beta**2)
    frequency : float
        Bunch frequency.
    bunch_in : Bunch
        Input bunch to the simulation. It is copied to a `bunch_out`
        for tracking.
    bunch_out : Bunch
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
        self.params = dict()
        self.params['kin_energy'] = 0.0025  # [GeV]
        self.params['mass'] = 0.939294  # [GeV/c^2]
        self.params['charge'] = -1.0  # elementary charge units
        self.params['gamma'] = (self.params['mass'] + self.params['kin_energy']) / self.params['mass']
        self.params['beta'] = np.sqrt(self.params['gamma'] * self.params['gamma'] - 1.0) / self.params['gamma']
        self.params['freq'] = 402.5e6

        self.default = Default()
        self.outdir = outdir
        if self.outdir is None:
            self.outdir = os.path.join(os.getcwd(), '/data')
        if not os.path.exists(self.outdir):  # -- make directory if not there yet
            os.mkdir(self.outdir)
            
        self.bunch_in = None
        self.bunch_out = None
        self.tracker_kws = tracker_kws
        if self.tracker_kws is None:
            self.tracker_kws = dict()
                        
    def init_all(self, init_bunch=True):
        """Set up full simulation with all default values."""
        self.init_lattice()
        self.init_apertures()
        self.init_sc_nodes()
        self.tracker = BunchTracker(**self.tracker_kws)
        if init_bunch:
            self.init_bunch()        

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
        
        self.latgen = LatticeGenerator(
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
                calc.freqOfBunches(self.params['freq'])
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
        
        To-do: break down into smaller functions. Should bunch generation be 
        handled in simulation scripts?

        Parameters
        ----------
        gen_type : str
            Options are {'twiss', 'load', 'twiss', '2d', '2dx3', '2d+E'}.
            if gen=="twiss", need to define all arguments alpha, beta, etc.
            or accept the defaults below.
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
        # Twiss parameters are not used if generating function does not call for them.
        if twiss is None:
            twiss = dict()
        for key in INIT_TWISS:
            twiss.setdefault(key, INIT_TWISS[key])
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
        eps_x = 1.0e-6 * eps_x / (self.params['gamma'] * self.params['beta'])
        eps_y = 1.0e-6 * eps_y / (self.params['gamma'] * self.params['beta'])
        eps_z = 1.0e-6 * eps_z / (self.params['gamma']**3 * self.params['beta'])

        # Transform to pyorbit emittance [GeV*m].
        eps_z = eps_z * self.params['gamma']**3 * self.params['beta']**2 * self.params['mass']
        beta_z = beta_z / (self.params['gamma']**3 * self.params['beta']**2 * self.params['mass'])
        if verbose:
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
            print("Reading bunch from file {}...".format(bunch_filename))
            if not os.path.isfile(bunch_filename):
                raise ValueError("File '{}' does not exist.".format(bunch_filename))
                
            bunch_gen = bu.Base_BunchGenerator()
            if bunch_file_format == "pyorbit":
                self.bunch_in = Bunch()
                self.bunch_in.readBunch(bunch_filename)
            elif bunch_file_format == "parmteq":
                self.bunch_in = bu.read_parmteq_bunch(bunch_filename)
            else:
                raise KeyError("Do not recognize format {} for bunch file".format(bunch_file_format))
                
            n_parts = self.bunch_in.getSize()
            macrosize = self.bunch_in.macroSize()
            self.bunch_in.getSyncParticle().kinEnergy(self.params['kin_energy'])
            self.bunch_in.mass(self.params['mass'])
            self.bunch_in.charge(self.params['charge'])
            self.z2phase = bunch_gen.get_z_to_phase_coeff()
            self.current = bunch_gen.get_beam_current() * 1e-3

            # TODO: insert function to extract peak current from coordinates
            # peak current = N_peak * macroSize * consts.charge_electron * self.params['beta'] * consts.speed_of_light
            # where N_peak is peak number density

            print("Bunch read complete {} macroparticles.".format(n_parts))

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
                raise ValueError("Unrecognized distributor {}.".format(dist))
            if gen_type == "twiss":
                bunch_gen = bu.BunchGeneratorTwiss(
                    twiss_x,
                    twiss_y,
                    twiss_z,
                    mass=self.params['mass'],
                    charge=self.params['charge'],
                    kin_energy=self.params['kin_energy'],
                    curr=self.current * 1e3,
                    freq=self.params['freq'],
                )
            elif gen_type in ["2d", "2d+E", "2dx3"]:
                phase_sp_gen_x = bu.PhaseSpaceGen(xfilename, threshold=thresh)
                phase_sp_gen_y = bu.PhaseSpaceGen(yfilename, threshold=thresh)
                twiss_z = TwissContainer(alpha_z, beta_z, eps_z)
                if gen_type == "2d":
                    bunch_gen = bu.BunchGenerator_XXP_YYP_TwissZ(
                        phase_sp_gen_x,
                        phase_sp_gen_y,
                        twiss_z,
                        mass=self.params['mass'],
                        charge=self.params['charge'],
                        kin_energy=self.params['kin_energy'],
                        curr=self.current * 1e3,
                        freq=self.params['freq'],
                        method=sample_method,
                    )
                elif gen_type == "2d+E":
                    efilename = kwargs.get("efile", "")
                    phase_sp_gen_z = bu.PhaseSpaceGenZPartial(
                        efilename,
                        twiss_z,
                        zdistributor=dist_class,
                        cut_off=cut_off,
                    )
                    bunch_gen = bu.BunchGenerator_XXP_YYP_ZZP(
                        phase_sp_gen_x,
                        phase_sp_gen_y,
                        phase_sp_gen_z,
                        mass=self.params['mass'],
                        charge=self.params['charge'],
                        kin_energy=self.params['kin_energy'],
                        curr=self.current * 1e3,
                        freq=self.params['freq'],
                        method=sample_method,
                    )
                elif gen_type == "2dx3":
                    phase_sp_gen_z = bu.PhaseSpaceGen(zfilename, threshold=thresh)
                    bunch_gen = bu.BunchGenerator_XXP_YYP_ZZP(
                        phase_sp_gen_x,
                        phase_sp_gen_y,
                        phase_sp_gen_z,
                        mass=self.params['mass'],
                        charge=self.params['charge'],
                        kin_energy=self.params['kin_energy'],
                        curr=self.current * 1e3,
                        freq=self.params['freq'],
                        method=sample_method,
                    )
            bunch_gen.set_kin_energy(self.params['kin_energy'])
            self.bunch_in = bunch_gen.get_bunch(
                n_parts=int(n_parts), 
                dist_class=dist_class,
            )
            self.z2phase = bunch_gen.get_z_to_phase_coeff()
            n_parts = self.bunch_in.getSize()
            print("Bunch generation complete with {} macroparticles.".format(n_parts))
                    
    def run(self, start=0.0, stop=None, out='default', verbose=0):
        """Run the simulation.
        
        Parameters
        ----------
        start/stop : float or str
            Start/stop position or node name.
        out : str
            Location of output bunch file. If 'default', use 'btf_output_bunch_end.txt'. 
            If None, does not save anything.
        """
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
        if stop is None:
            stop = self.lattice.getLength()
        if type(start) in [float, int]:
            start_node, start_num, zs_start_node, ze_start_node = self.lattice.getNodeForPosition(start)
        elif type(start) is str:
            start_node = self.lattice.getNodeForName(start)
            start_num = self.lattice.getNodeIndex(start_node)
            zs_start_node = start_node.getPosition() - 0.5 * start_node.getLength()
            ze_start_node = start_node.getPosition() + 0.5 * start_node.getLength()
        else:
            raise TypeError("Invalid type {} for `start`.".format(type(start)))
        if type(stop) in [float, int]:            
            print("max simulation length = {:.3f}.".format(self.lattice.getLength()))
            if stop > self.lattice.getLength():
                stop = self.lattice.getLength()
            stop_node, stop_num, zs_stop_node, ze_stop_node = self.lattice.getNodeForPosition(stop)
        elif type(stop) is str:
            stop_node = self.lattice.getNodeForName(stop)
            stop_num = self.lattice.getNodeIndex(stop_node)
            zs_stop_node = stop_node.getPosition() - 0.5 * stop_node.getLength()
            ze_stop_node = stop_node.getPosition() + 0.5 * stop_node.getLength()
        else:
            raise TypeError("Invalid type {} for `stop`.".format(type(start)))
        if verbose:
            print(
                "Running simulation from s = {:.4f} [m] to s = {:.4f} [m] (nodes {} to {})."
                .format(zs_start_node, ze_stop_node, start_num, stop_num)
            )
            
        # Add tracking action at each node entrance.
        self.tracker = BunchTracker(**self.tracker_kws)
        action_container = AccActionsContainer("bunch tracker")
        action_container.addAction(self.tracker.action, AccActionsContainer.ENTRANCE)

        # Run the simulation.
        params_dict = {
            "old_pos": -1.0,
            "count": 0,
            "pos_step": 0.005,
        } 
        time_start = time.clock()
        self.bunch_out = Bunch()
        self.bunch_in.copyBunchTo(self.bunch_out)
        self.lattice.trackBunch(
            self.bunch_out,
            paramsDict=params_dict,
            actionContainer=action_container,
            index_start=start_num,
            index_stop=stop_num,
        )
        
        # Save the last time step.
        params_dict["old_pos"] = -1
        self.tracker.action(params_dict)

        # Wrap up.
        time_exec = time.clock() - time_start
        print("time[sec]=", time_exec)
        if out is not None:
            self.bunch_out.dumpBunch(output_filename)
            print("Dumped output bunch to file {}".format(output_filename))
        self.tracker.cleanup()
        self.tracker.history["s"] += zs_start_node

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
        self.bunch_in = bu.reverse(self.bunch_in)

        # Run in reverse (start <==> stop). Do not auto-save output bunch; the bunch
        # coordinates need to be reversed first.
        self.run(start=stop, stop=start, out=None)

        # Un-reverse the lattice.
        self.latgen.lattice.reverseOrder()
        self.lattice = self.latgen.lattice

        # Un-reverse the bunch coordinates, then save them.
        self.bunch_out = bu.reverse(self.bunch_out)
        if out:
            self.bunch_out.dumpBunch(output_filename)

        # Also un-reverse the initial bunch coordinates.
        self.bunch_in = bu.reverse(self.bunch_in)

        # Wrap up.
        if type(stop) in [float, int]:
            stop_node, stop_num, zs_stop_node, ze_stop_node = self.lattice.getNodeForPosition(stop)
        elif type(stop) == str:
            stop_node = self.lattice.getNodeForName(stop)
            stop_num = self.lattice.getNodeIndex(stop_node)
            zs_stop_node = stop_node.getPosition() - 0.5 * stop_node.getLength()
            ze_stop_node = stop_node.getPosition() + 0.5 * stop_node.getLength()
        self.tracker.history["s"] = 2.0 * ze_stop_node - self.tracker.history["s"]
        
    def reset(self):
        self.bunch_out = None
        self.tracker.delete_history()