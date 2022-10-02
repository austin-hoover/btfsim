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
from btfsim import reconstruct as rec
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


class Monitor:
    """Class to record bunch evolution data.
    
    Attributes
    ----------
    history : dict[str: list]
        Each key is a the name of the parameter; each value is a 
        list of the parameter's value that is appended to each time
        the beam is tracked by a lattice node.
    plotter : btfsim.plot.Plotter
        Plot routine manager. Monitor can decide when this plotter 
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
        for dim in ["x", "y", "z"]:
            for name in ["alpha", "beta", "eps"]:
                keys.append("{}_{}".format(name, dim))
        for i in range(6):
            for j in range(i + 1):
                keys.append("sig_{}{}".format(j + 1, i + 1))
        self.history = {key: [] for key in keys}
        
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
        # Turn history lists into ndarrays.
        for key in self.history:
            self.history[key] = np.array(self.history[key])
        # Trim zeros.
        ind = np.where(np.self.history["sig_11"] == 0)[0][1]
        for key, arr in self.history.iteritems():
            istart = 0 if key == "node" else 1
            self.history[key] = arr[istart:ind]
            
    def forget(self):
        """Delete all data."""
        for key in self.history:
            self.history[key] = []

    def write(self, filename=None, sep=" "):
        """Save history data.

        Parameters
        ----------
        filename : str 
            Path to file.
        sep : str
            Delimeter in file (" ", ",", etc.)
            
        Returns
        -------
        DataFrame
            Pandas dataframe containing
        """
        keys = list(self.history)
        data = np.array([self.history[key] for key in keys]).T
        df = pd.DataFrame(data=data, columns=keys)
        df.to_csv(filename, sep=sep, index=False)
        return df


class Simulation:
    """Class to hold simulation model.

    Attributes
    ----------
    bunch_in : Bunch
        Input bunch to the simulation. It is copied to a `bunch_out`
        for tracking.
    bunch_out : Bunch
        Bunch used for tracking; overwritten each time simulation is run.
    lattice : orbit.lattice.AccLattice
        Lattice for tracking.
    latgen : btfsim.lattice.LatticeGenerator
        Instance of LatticeGenerator class.    
    monitor_kws : Monitor
        Class which monitors bunch during simulation. The class must implement
        the method `action(self, params_dict)`, which is called at the entrance
        of each node in the lattice.
    """
    def __init__(self, outdir=None, monitor_kws=None):
        """Constructor.
        
        monitor_kws : dict
            Key word arguments passed to Monitor.
        """
        self.default = Default()
        self.outdir = outdir
        if self.outdir is None:
            self.outdir = os.path.join(os.getcwd(), 'data')
        utils.ensure_path_exists(self.outdir)            
        self.bunch_in = None
        self.bunch_out = None
        self.monitor_kws = monitor_kws if monitor_kws else dict()
        self.monitor = Monitor(**self.monitor_kws)
        self.action_container = AccActionsContainer('monitor')
        self.action_container.addAction(self.monitor.action, AccActionsContainer.ENTRANCE)
      
    def init_all(self, init_bunch=True):
        """Set up full simulation with all default values."""
        self.init_lattice()
        self.init_apertures()
        self.init_sc_nodes()
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
        for node in aprtNodes:
            print("aprt={}", node.getName(), " pos =", node.getPosition())
        self.lattice = self.latgen.lattice
        print("Aperture nodes added.")

    def init_bunch(self, bunch):
        """Set the initial bunch."""
        self.bunch_in = bunch

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
        # Configure output file name.
        if stop == self.lattice.getLength():
            default_output_filename = "btf_output_bunch_end.txt"
        else:
            default_output_filename = "btf_output_bunch_{}.txt".format(stop)
        if out == 'default':
            out = default_output_filename
        if out is not None:
            output_filename = os.path.join(self.outdir, out)

        # Figure out where to start/stop.
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
            
        # Add tracking actions; these are called at every node.
        self.refresh()

        # Propagate a copy of the input bunch.
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
        self.monitor.action(params_dict)

        # Wrap up.
        time_exec = time.clock() - time_start
        print("time = {:.3f} [sec]".format(time_exec))
        if out is not None:
            self.bunch_out.dumpBunch(output_filename)
            print("Dumped output bunch to file {}".format(output_filename))
        self.monitor.cleanup()
        self.monitor.history["s"] += zs_start_node

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
        self.monitor.history["s"] = 2.0 * ze_stop_node - self.monitor.history["s"]
        
    def refresh(self):
        """Prepare for new run."""
        self.bunch_out = None
        self.monitor.forget()