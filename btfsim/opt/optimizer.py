from __future__ import print_function

import numpy as np
from scipy.optimize import minimize

from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes

import btfsim.sim.simulation_main as main
from btfsim.lattice.btf_quad_func_factory import btf_quad_func_factory as quadfunc


class Matcher:

    fodomark1 = "MEBT:MARK1"
    fodomark2 = "MEBT:MARK2"
    screenloc = "MEBT:FLANGE2"

    def __init__(self, **kwargs):
        """
        optional arguments:
        knobs: list of quadrupole names to use for optimization
                   default : ['QV10','QH11','QV12','QH13']
        targets: list of parameter names to use as targets
                         default : ['ax','bx','ay','by']
                         target values are passed on call to optimizer

        mstatefile: name of mstate file for initial condition on quads. Otherwise uses a default settings file
        filename: name of file to output results. Default 'optimization_progress.txt'
        analytic_quads: default False; if true implements analytic quad model instead of hard-edged
        analytic_quad_names: if analytic_quads=False, can specify list of quad names to use analytic model.
                                                By default, changes model for all quads.
        """

        self.knobs = kwargs.pop("knobs", ["QV10", "QH11", "QV12", "QH13"])
        self.targetnames = kwargs.pop("targets", ["ax", "bx", "ay", "by"])
        self.mstatefile = kwargs.pop("mstate", None)
        self.xmlfile = kwargs.pop("xml", None)
        self.coeffilename = kwargs.pop("coeffilename", "")
        self.filename = kwargs.pop("outfile", "optimization_progress.txt")
        self.analyticQuads = kwargs.pop("analytic_quads", False)
        self.quad_names = kwargs.pop("analytic_quad_names", [])
        nbunches = kwargs.pop("nbunches", None)

        # -- variables defining optimization region
        # (defaults to matching quads section, Q10-13)
        beamline = kwargs.pop("beamline", ["MEBT3"])

        # -- init sim of BTF in desired beamline area
        self.sim = main.simBTF()
        if self.mstatefile:
            if self.xmlfile:
                self.sim.initLattice(
                    beamline=beamline,
                    mstatename=self.mstatefile,
                    xml=self.xmlfile,
                    coeffilename=self.coeffilename,
                )
            else:
                self.sim.initLattice(
                    beamline=beamline,
                    mstatename=self.mstatefile,
                    coeffilename=self.coeffilename,
                )
        else:
            if self.xmlfile:
                self.sim.initLattice(
                    beamline=beamline, xml=self.xmlfile, coeffilename=self.coeffilename
                )
            else:
                self.sim.initLattice(beamline=beamline, coeffilename=self.coeffilename)
        self.sim.initApertures()

        if self.analyticQuads:
            z_step = 0.001
            Replace_Quads_to_OverlappingQuads_Nodes(
                self.sim.accLattice,
                z_step,
                accSeq_Names=beamline,
                quad_Names=self.quad_names,
                EngeFunctionFactory=quadfunc,
            )

        if nbunches:
            solver = "fft"
        else:
            solver = "ellipse"

        self.sim.initSCnodes(solver=solver, n_bunches=nbunches)
        self.sim.initBunch(**kwargs)

        self.counter = 0

        # -- open filename and print header
        header = self.knobs + self.targetnames
        with open(self.filename, "w") as filehandle:
            for item in header:
                filehandle.write("%s," % item)
            filehandle.write("f\n")

    def opt2target_function(self, x, *target):
        """
        Optimize list of paramters to specified target values

        x is list of quad values to change.
        target is tuple of values for target (values of beam parameters
        at end of simulation length, where parameter names are specified
        in initialization).
        """

        # -- change quads
        changedict = {}
        for i in range(len(x)):
            changedict[self.knobs[i]] = x[i]

        print(changedict)
        self.sim.changeQuads(dict=changedict)

        # -- run simulation over specified length
        self.sim.run(start=self.tstart, stop=self.tstop, out=None)
        self.sim.tracker.writehist(filename="data/opt_run_latest.txt")
        # self.counter += 1

        # -- calculate error function (rms difference between target
        # and calculated value)
        f = 0.0
        fvec = []
        for i in range(len(target)):
            targetval = target[i]
            thistarget = self.targetnames[i]
            key = thistarget.split("-")[-1]
            if "end-" in thistarget:
                val = self.sim.tracker.hist[key][-1]
            elif "max-" in thistarget:
                val = self.sim.tracker.hist[key].max()
            elif "min-" in thistarget:
                val = self.sim.tracker.hist[key].min()
            elif "avg-" in thistarget:
                val = self.sim.tracker.hist[key].mean()
            elif not ("-" in thistarget):
                val = self.sim.tracker.hist[thistarget][-1]
            else:
                raise ValueError("Cannot figure out what you mean by %s" % thistarget)
            diffval = targetval - val
            f += diffval**2
            fvec.append(val)

        f = np.sqrt(f)

        writetofile = list(x) + fvec
        with open(self.filename, "a") as filehandle:
            for item in writetofile:
                filehandle.write("%.6f," % item)
            filehandle.write("%.6f\n" % f)

        return f

    def opt2equal_function(self, x):
        """
        Optimize to make two parameters equal each other.
        This is done to test a matching algorithm for application to BTF

        x is list of quad values to change.
        Target is automatically assumed that parameters should be equal to each other.
        (parameter names are specified in initialization).
        """

        # -- change quads
        changedict = {}
        for i in range(len(x)):
            changedict[self.knobs[i]] = x[i]

        print(changedict)
        self.sim.changeQuads(dict=changedict)

        # -- run simulation over specified length
        self.sim.run(start=self.tstart, stop=self.tstop, out=None)

        # -- calculate error function (rms difference between target
        # and calculated value)
        fvec = []
        for i in range(len(self.targetnames)):
            thistarget = self.targetnames[i]
            key = thistarget.split("-")[-1]
            if "end-" in thistarget:
                val = self.sim.tracker.hist[key][-1]
            elif "max-" in thistarget:
                val = self.sim.tracker.hist[key].max()
            elif "min-" in thistarget:
                val = self.sim.tracker.hist[key].min()
            elif "avg-" in thistarget:
                val = self.sim.tracker.hist[key].mean()
            elif not ("-" in thistarget):
                val = self.sim.tracker.hist[thistarget][-1]
            else:
                raise ValueError("Cannot figure out what you mean by %s" % thistarget)
            fvec.append(val)

        f = 0.0
        for i in range(len(fvec)):
            for j in range(len(fvec)):
                diffval = fvec[i] - fvec[j]
                f += diffval**2

        f = np.sqrt(f)

        writetofile = list(x) + fvec
        with open(self.filename, "a") as filehandle:
            for item in writetofile:
                filehandle.write("%.6f," % item)
            filehandle.write("%.6f\n" % f)

        return f

    def minimize_function(self, x, weights):
        """
        x is list of quad values to change.
        weights is tuple of values for weights on target values (values of beam parameters
        at end of simulation length, where parameter names are specified
        in initialization).
        """

        # -- change quads
        changedict = {}
        for i in range(len(x)):
            changedict[self.knobs[i]] = x[i]

        print(changedict)
        self.sim.changeQuads(dict=changedict)

        # -- run simulation over specified length
        self.sim.run(start=self.tstart, stop=self.tstop, out=None)
        self.sim.tracker.writehist(filename="data/opt_min_latest.txt")

        # -- calculate error function (rms difference between target
        # and calculated value)
        f = 0.0
        fvec = []
        for i in range(len(weights)):
            thistarget = self.targetnames[i]
            key = thistarget.split("-")[-1]
            if "end-" in thistarget:
                val = self.sim.tracker.hist[key][-1]
            elif "max-" in thistarget:
                val = self.sim.tracker.hist[key].max()
            elif "min-" in thistarget:
                val = self.sim.tracker.hist[key].min()
            elif not ("-" in thistarget):
                val = self.sim.tracker.hist[thistarget][-1]
            else:
                raise ValueError("Cannot figure out what you mean by %s" % thistarget)
            f += (val * weights[i]) ** 2
            fvec.append(val)

        f = np.sqrt(f)

        writetofile = list(x) + fvec
        with open(self.filename, "a") as filehandle:
            for item in writetofile:
                filehandle.write("%.6f," % item)
            filehandle.write("%.6f\n" % f)

        return f

    def optimize_to_target(self, x0, target, **kwargs):
        """
        kwargs:
        start = float position or name of node, start of opt. region
                        default: 0.
        stop = float position or name of node, end of opt. region
                        default: "MEBT:MARK1" (for matching quad optimization)
        remaining kwargs are options for solver:
        {'ftol','maxiter','disp'}
        """
        method = kwargs.pop("method", "Nelder-Mead")

        # -- variables defining optimization region
        # (defaults to matching quads section, Q10-13)
        self.tstart = kwargs.pop("start", 0.0)
        self.tstop = kwargs.pop("stop", self.fodomark1)

        # automatic bounds 10 A
        bounds = [(-10, 10)] * len(x0)

        opt_result = minimize(
            self.opt2target_function,
            x0,
            args=target,
            method=method,
            bounds=bounds,
            options=kwargs,
        )

        return opt_result

    def optimize_to_equal(self, x0, **kwargs):
        """
        kwargs:
        start = float position or name of node, start of opt. region
                        default: 0.
        stop = float position or name of node, end of opt. region
                        default: "MEBT:MARK1" (for matching quad optimization)
        remaining kwargs are options for solver:
        {'ftol','maxiter','disp'}
        """
        method = kwargs.pop("method", "Nelder-Mead")

        # -- variables defining optimization region
        # (defaults to matching quads section, Q10-13)
        self.tstart = kwargs.pop("start", 0.0)
        self.tstop = kwargs.pop("stop", self.screenloc)

        # automatic bounds 10 A
        bounds = [(-10, 10)] * len(x0)

        opt_result = minimize(
            self.opt2equal_function, x0, method=method, bounds=bounds, options=kwargs
        )

        return opt_result

    def optimize_to_min(self, x0, weights, **kwargs):
        """
        x0 = list of initial conditions
        weight = list of weights
        kwargs:
        start = float position or name of node, start of opt. region
                        default: 0.
        stop = float position or name of node, end of opt. region
                        default: "MEBT:MARK1" (for matching quad optimization)
        remaining kwargs are options for solver:
        method = 'Nelder-Mead'
        {'ftol','maxiter','disp'}
        """
        method = kwargs.pop("method", "Nelder-Mead")

        # -- variables defining optimization region
        # (defaults to matching quads section, Q10-13)
        self.tstart = kwargs.pop("start", 0.0)
        self.tstop = kwargs.pop("stop", self.fodomark1)

        opt_result = minimize(
            self.minimize_function, x0, args=weights, method=method, options=kwargs
        )

        return opt_result


class PeriodicMatcher:

    fodomark1 = "MEBT:MARK1"
    fodomark2 = "MEBT:MARK2"

    def __init__(self, **kwargs):
        """
        optional arguments:
        knobs: list of quadrupole names to use for optimization
                   default : ['QV10','QH11','QV12','QH13']
        targets: list of parameter names to use as targets
                         default : ['ax','bx','ay','by']
                         target values are passed on call to optimizer

        analytic_quads: default False; if true implements analytic quad model instead of hard-edged
        """

        # self.knobs = kwargs.pop('knobs',['QV10','QH11','QV12','QH13'])
        # self.targetnames = kwargs.pop('targets',['ax','bx','ay','by'])
        # self.mstatefile = kwargs.pop('mstate',None)
        # self.filename = kwargs.pop('outfile','optimization_progress.txt')
        self.analyticQuads = kwargs.pop("analytic_quads", False)
        xmlfile = kwargs.pop("xml", None)
        nbunches = kwargs.pop("nbunches", None)

        # -- variables defining optimization region
        # (defaults to matching quads section, Q10-13)
        beamline = kwargs.pop("beamline", ["MEBT3"])

        # -- only use fft solver if modeling neighboring bunches
        # otherwise, ellipse solver is faster
        if nbunches:
            solver = "fft"
        else:
            solver = "ellipse"

        # -- init sim of BTF in FODO line for finding periodic match
        self.simprd = main.simBTF()
        self.simprd.initLattice(beamline=beamline, xml=xmlfile)
        self.simprd.initApertures()
        if self.analyticQuads:
            z_step = 0.001
            Replace_Quads_to_OverlappingQuads_Nodes(
                self.simprd.accLattice,
                z_step,
                accSeq_Names=beamline,
                EngeFunctionFactory=quadfunc,
            )

        self.simprd.initSCnodes(solver=solver, n_bunches=nbunches)
        # self.simprd.z2phase = self.sim.z2phase

    def optperiodic_function(self, x, *args):
        # right now, unused. simple find works better

        ax0 = x[0]
        bx0 = x[1]
        ay0 = x[2]
        by0 = x[3]

        current = args[0]
        twissz = args[1]
        az0 = twissz[0]
        bz0 = twissz[1]
        ez0 = twissz[2]

        self.simprd.initBunch(
            gen="twiss",
            nparts=20000,
            ax=ax0,
            bx=bx0,
            ay=ay0,
            by=by0,
            az=az0,
            bz=bz0,
            ez=ez0,
            dist="gaussian",
            current=current,
        )

        self.simprd.run(start=self.fodomark1, stop=self.fodomark2, out=None)

        ax1 = self.simprd.tracker.hist["ax"][0]
        bx1 = self.simprd.tracker.hist["bx"][0]
        ax2 = self.simprd.tracker.hist["ax"][-1]
        bx2 = self.simprd.tracker.hist["bx"][-1]

        ay1 = self.simprd.tracker.hist["ay"][0]
        by1 = self.simprd.tracker.hist["by"][0]
        ay2 = self.simprd.tracker.hist["ay"][-1]
        by2 = self.simprd.tracker.hist["by"][-1]

        f = np.sqrt(
            (ax2 - ax1) ** 2 + (bx2 - bx1) ** 2 + (ay2 - ay1) ** 2 + (by2 - by1) ** 2
        )

        return f

    def optimize_periodic_match(
        self,
        twiss0,
        twissz,
        current=0.04,
        nparts=20000,
        niter=30,
        plotFlag=False,
        ftol=0.01,
    ):
        az0 = twissz[0]
        bz0 = twissz[1]
        ez0 = twissz[2]

        # -- the simple way
        f = 100.0
        i = 0
        twiss_hist = np.zeros([niter, 4])
        while f > ftol and i < niter:
            ax0 = twiss0[0]
            bx0 = twiss0[1]
            ay0 = twiss0[2]
            by0 = twiss0[3]

            self.simprd.initBunch(
                gen="twiss",
                nparts=nparts,
                ax=ax0,
                bx=bx0,
                ay=ay0,
                by=by0,
                az=az0,
                bz=bz0,
                ez=ez0,
                dist="gaussian",
                current=current,
            )

            self.simprd.run(start=self.fodomark1, stop=self.fodomark2, out=None)

            ax2 = self.simprd.tracker.hist["ax"][-1]
            bx2 = self.simprd.tracker.hist["bx"][-1]
            ay2 = self.simprd.tracker.hist["ay"][-1]
            by2 = self.simprd.tracker.hist["by"][-1]

            diff = np.array([ax2 - ax0, bx2 - bx0, ay2 - ay0, by2 - by0])
            f = np.sqrt(np.mean(diff**2))

            twiss0 = (
                np.mean([ax0, ax0, ax2]),
                np.mean([bx0, bx0, bx2]),
                np.mean([ay0, ay0, ay2]),
                np.mean([by0, by0, by2]),
            )
            twiss_hist[i, :] = [
                np.mean([ax0, ax0, ax2]),
                np.mean([bx0, bx0, bx2]),
                np.mean([ay0, ay0, ay2]),
                np.mean([by0, by0, by2]),
            ]
            print(twiss0)
            i += 1

        print("N iteration = %i" % i)
        return twiss0
