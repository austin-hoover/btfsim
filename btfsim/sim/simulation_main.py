# -- btfsim modules
import btfsim.lattice.generate_btf_lattice as gen_lattice
import btfsim.bunch.btf_linac_bunch_generator as gen_bunch
import btfsim.util.Defaults as default
import btfsim.bunch.butilsities as butils
from btfsim.plot.Movie import MovieBase

# -- pyorbit modules
import orbit.utils.consts as consts
from orbit.space_charge.sc3d import setSC3DAccNodes, setUniformEllipsesSCAccNodes
from spacecharge import SpaceChargeCalcUnifEllipse, SpaceChargeCalc3D
from orbit.bunch_generators import TwissContainer, TwissAnalysis
from orbit.bunch_generators import KVDist1D, KVDist2D, KVDist3D
from orbit.bunch_generators import GaussDist1D, GaussDist2D, GaussDist3D
from orbit.bunch_generators import WaterBagDist1D, WaterBagDist2D, WaterBagDist3D
from bunch import Bunch, BunchTwissAnalysis
from orbit.lattice import AccActionsContainer
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_bend_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice

# -- other things
import os
import numpy as np
import time
#from PyQt5 import QtCore


class simBTF():
    """
    Class to hold simulation model.

    Initialize has optional argument: outdir = path to directory for output data
    If not specified, defaults to folder 'data/' in current working directory

    ##Attributes:
    ekin
    mass
    gamma
    beta
    frequency
    c
    peakcurrent
    bunch_in

    ##Instances:
    accLattice
    lat
    bunch_in
    bunch_track

    ##Methods:
    run
    init_lattice
    initSCnodes
    initBunch
    changeQuads

    """
    def __init__(self, outdir=[]):
        #super(simBTF,self).__init__()

        # -- some default parameters
        self.ekin = 0.0025 # in [GeV]
        self.mass = 0.939294 # in [GeV]
        self.charge = -1 # particle charge
        self.gamma = (self.mass + self.ekin)/self.mass
        self.beta = np.sqrt(self.gamma*self.gamma - 1.0)/self.gamma
        print("relat. gamma=",self.gamma)
        print("relat.  beta=",self.beta)
        self.freq = 402.5e+6

        # -- initialize default file locations
        self.defaults = default.getDefaults()

        # -- default save location will be <cwd>/data/ unless specified
        if outdir:
            self.outdir=outdir
        else:
            self.outdir = os.getcwd() + '/data/'
        if not(os.path.exists(self.outdir)): # -- make directory if not there yet
            os.mkdir(self.outdir)

        # -- call BunchTwissAnalysis method
        #self.twiss_analysis = BunchTwissAnalysis()

        # -- set flags
        self.movieFlag = 0 # if 1 will save movie frames (takes a lot of storage space)
        self.dispersion_flag = 1 # if 1 will correct for dispersion in recorded twiss parameters
        self.emit_norm_flag = 0 # if 1 will calculated normalized emittances

    def initAll(self, bunchFlag=1):
        """
        Set up full simulation with all default values
        """

        # -- init default lattice
        self.init_lattice()

        # -- set up apertures
        self.initApertures()

        # -- set up SC nodes
        self.initSCnodes()

        # -- initialize default bunch if flag =/= 0
        if bunchFlag:
            self.initBunch()

    def enableMovie(self,**kwargs):
        self.movieFlag = 1
        savefolder = kwargs.pop('savedir','data/')
        self.movie = MovieBase(savefolder,**kwargs)

    def run(self, **kwargs):
        """
        Execute simulation

        optional keyword args:
        start = 0
        stop = length of defined accLattice 
        out = 'btf_output_bunch_end.txt'; location of output bunch file. out=None, no file saved
        """

        # -- parse start/stop
        maxsimlen = self.accLattice.getLength()
        start = kwargs.get('start', 0.0)  # default start at s=0
        stop = kwargs.get('stop', maxsimlen)  # default run til end of defined lattice

        # -- parse/default output filename "out"
        if stop == self.accLattice.getLength():
            default_output_filename = "btf_output_bunch_end.txt"                       
        else:
            default_output_filename = "btf_output_bunch_%s.txt"%str(stop)
        out = kwargs.get('out',default_output_filename)
        if out:
            output_filename = self.outdir + out

        # -- figure out which nodes are at stop, start position
        if type(start) == float or type(start) == int:
            startnode,startnum,zs_startnode,ze_startnode = self.accLattice.getNodeForPosition(start)
        elif type(start) == str:
            startnode = self.accLattice.getNodeForName(start)
            startnum = self.accLattice.getNodeIndex(startnode)
            zs_startnode = startnode.getPosition() - 0.5*startnode.getLength()
            ze_startnode = startnode.getPosition() + 0.5*startnode.getLength()
        else:
            raise TypeError("don't understand type %s for 'start'"%str(type(start)))

        if type(stop) == float  or type(stop) == int:
            # -- check sim length
            print("max sim len = %.3f"%maxsimlen)
            if stop > maxsimlen: stop=maxsimlen # don't run longer than simulation
            stopnode,stopnum,zs_stopnode,ze_stopnode = self.accLattice.getNodeForPosition(stop)
        elif type(stop) == str:
            stopnode = self.accLattice.getNodeForName(stop)
            stopnum = self.accLattice.getNodeIndex(stopnode)
            zs_stopnode = stopnode.getPosition() - 0.5*stopnode.getLength()
            ze_stopnode = stopnode.getPosition() + 0.5*stopnode.getLength()
        else:
            raise TypeError("don't understand type %s for 'stop'"%str(type(stop)))	

        print("Running simulation from %.4f m to %.4f m (nodes %i to %i)"%(zs_startnode,ze_stopnode,startnum,stopnum))

        # -- set up run
        paramsDict = {"old_pos":-1.,"count":0,"pos_step":0.005}    #"pos_step":0.005
        actionContainer = AccActionsContainer("BTF Bunch Tracking")

        # -- add tracking action 
        # --- load different tracking routine if bunch has 1 particle (single-particle tracking)
        if self.bunch_in.getSize() > 1:
            self.tracker = butils.bunchTrack(dispersion_flag=self.dispersion_flag, emit_norm_flag=self.emit_norm_flag)
        elif self.bunch_in.getSize() == 1:
            self.tracker = butils.spTrack()
        actionContainer.addAction(self.tracker.action_exit, AccActionsContainer.EXIT)

        ## -- add movie action 
        if self.movieFlag == 1:
            actionContainer.addAction(self.movie.makeFrame, AccActionsContainer.EXIT)

        time_start = time.clock()
        self.bunch_track = Bunch()
        self.bunch_in.copyBunchTo(self.bunch_track)


        # -- run (track bunch)
        self.accLattice.trackBunch(self.bunch_track, paramsDict = paramsDict, actionContainer = actionContainer, index_start=startnum, index_stop=stopnum)

        # -- save last time step
        paramsDict["old_pos"]=-1
        self.tracker.action_entrance(paramsDict)

        # -- timer
        time_exec = time.clock() - time_start
        print("time[sec]=",time_exec)

        # -- save output bunch if out =/= None
        if out:
            print('dumped output bunch to file %s'%output_filename)
            self.bunch_track.dumpBunch(output_filename)

        # -- cleanup extra zeros
        self.tracker.cleanup()
        # -- shift hist['s'] to start from startze
        self.tracker.hist['s'] += zs_startnode

    def reverse(self,**kwargs):
        """
        Execute simulation

        optional keyword args:
        start = 0.
        stop = length of defined accLattice 
        out = 'btf_output_bunch_end.txt'; location of output bunch file. out=None, no file saved

        Note: simulation will /start/ at stop (position or node) and /end/ at start. 
        """

        # -- parse start/stop
        maxsimlen = self.accLattice.getLength()
        start = kwargs.get('start',0.0) # default start at s=0
        stop = kwargs.get('stop',maxsimlen) # default run til end of defined lattice

        print("Running simulation in reverse from %s m to %s."%(str(stop),str(start)))

        # -- reverse start and stop (if specified as floats)
        if isinstance(start,float):
            start = maxsimlen - start
        if isinstance(stop,float):
            stop = maxsimlen - stop

        # -- parse/default output filename "out"
        if stop == 0.0:
            default_output_filename = "reverse_output_bunch_start.txt"                       
        else:
            default_output_filename = "reverse_output_bunch_%s.txt"%str(stop)
        out = kwargs.get('out',default_output_filename)
        if out:
            output_filename = self.outdir + out

        # -- reverse lattice 
        self.lat.accLattice.reverseOrder()
        self.accLattice = self.lat.accLattice

        # -- reverse bunch coordinates
        gen_bunch.BunchTransformerFunc(self.bunch_in)

        # -- run in reverse (start <==> stop)
        # -- do not auto-save output bunch; need to reverse coordinates first
        self.run(start=stop,stop=start,out=None)

        # -- un-reverse lattice 
        self.lat.accLattice.reverseOrder()
        self.accLattice = self.lat.accLattice

        # -- un-reverse bunch and save
        gen_bunch.BunchTransformerFunc(self.bunch_track)
        if out:
            self.bunch_track.dumpBunch(output_filename)

        # -- also un-reverse initial bunch coordinates
        gen_bunch.BunchTransformerFunc(self.bunch_in)

        # -- shift hist['s'] to start from startze
        # -- figure out which nodes are at stop, start position
        if type(stop) == float  or type(stop) == int:
            # -- check sim length
            stopnode,stopnum,zs_stopnode,ze_stopnode = self.accLattice.getNodeForPosition(stop)
        elif type(stop) == str:
            stopnode = self.accLattice.getNodeForName(stop)
            stopnum = self.accLattice.getNodeIndex(stopnode)
            zs_stopnode = stopnode.getPosition() - 0.5 * stopnode.getLength()
            ze_stopnode = stopnode.getPosition() + 0.5 * stopnode.getLength()
        self.tracker.hist['s'] = 2 * ze_stopnode - self.tracker.hist['s']


    def init_lattice(self,beamline=["MEBT1", "MEBT2", "MEBT3"], xml=None, mstatename=None, 
                     mdict=None, units='Amps', ds=0.012, coeffilename=''):
        """
        initialize lattice based on xml file and fill w/ default quad values or mstate values

        optional arguments:
        beamline = name of beamlines to load. 
        mstatename = None; Location of .mstate file. If specified, quad values from this solution are loaded.
        magdict = None; Dictionary of magnet name/current pairs. Specified quads are updated to have this value. 
                Overwrites values set by .mstate file if both are specified.
        units = 'Amps' ; passed to change quads to make appropriate conversion
        """
        if xml: # -- load lattice from xml file
            print(xml)
            self.lat = gen_lattice.GenLattice(xml=xml, beamline=beamline, ds=ds, coeffilename=coeffilename) 
        else: # -- load lattice in default xml file
            xml = self.defaults.homedir + self.defaults.defaultdict["XML_FILE"]
            print(xml)
            self.lat = gen_lattice.GenLattice(xml=xml, beamline=beamline, ds=ds, coeffilename=coeffilename) 

        if mstatename: # -- set quads according to .mstate file
            self.changeQuads(filename=mstatename,units=units)
        # else:  # -- set default quad values in default_quads.txt
        #     self.lat.defaultQuads() 

        if mdict:
            self.changeQuads(dict=mdict, units=units)

        self.accLattice = self.lat.accLattice


    def changeQuads(self,**kwargs):
        """
        Change lattice; can either import new xml lattice or update quads in existing lattice
        """
        # -- if filename is passed, 
        filename = kwargs.pop('filename',None)
        thisdict = kwargs.pop('dict',[])
        units = kwargs.pop('units','Amps') # units for quad values. default is current [Amps], 
                                          # but GL [Tesla] also an option

        # -- set quads according to mstate file
        if filename:
            #if filename[-6:] == 'mstate':
            self.lat.loadQuads(filename=filename,units=units)
            # if loaded from mstate, units are in Tesla by default, no need to specify

        # -- set quads according to dict
        if thisdict:
            self.lat.updateQuads(dict=thisdict,units=units)

        # -- assume any remaining named arguments are quad/current pairs
        for quad,current in kwargs.items():
            self.lat.updateQuads(dict={ quad : current},units=units)

        # -- update accel. model
        self.accLattice = self.lat.accLattice

    def initSCnodes(self,minlen = 0.015, solver='ellipse', nellipse=1, gridmult=6, n_bunches=None):
        """
        Set up SC nodes

        Optional keyword args:
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

        if (sc_solver == "ellipse"):
            # set of uniformly charged ellipses Space Charge    The bigger number of ellipse, the more accurate of sapce charge calculation
            # Ellipse method can be used for the initial estimate, because it calculates faster than FFT method.
            calcUnifEllips = SpaceChargeCalcUnifEllipse(nellipse)
            space_charge_nodes = setUniformEllipsesSCAccNodes(self.lat.accLattice,sc_path_length_min,calcUnifEllips)
        else:
            # set FFT 3D Space Charge      FFT Poission solver, the more sizes (grids), the more accurate of space charge calculation
            #Particle number should be increased by multiplier**3 when grid increases by a multiplier
            sizeX = 2**gridmult                    
            sizeY = 2**gridmult 
            sizeZ = 2**gridmult 
            calc3d = SpaceChargeCalc3D(sizeX,sizeY,sizeZ)
            # right now can only model neighboring bunches if using Andrei's version of pyorbit
            if n_bunches: 
                calc3d.numExtBunches(n_bunches)
                calc3d.freqOfBunches(self.freq)
            space_charge_nodes =  setSC3DAccNodes(self.lat.accLattice,sc_path_length_min,calc3d)

        max_sc_length = 0.
        min_sc_length = self.lat.accLattice.getLength()
        for sc_node in space_charge_nodes:
            scL = sc_node.getLengthOfSC()
            if(scL > max_sc_length): max_sc_length = scL
            if(scL < min_sc_length): min_sc_length = scL

        # -- 'return' accLattice
        self.accLattice = self.lat.accLattice
        self.scnodes = space_charge_nodes

    def initApertures(self,d=0.04):
        """
        Arguments:
        d = 0.04 m (pipe wall diameter)
        """

        aprt_pipe_diameter = d

        aprtNodes = Add_quad_apertures_to_lattice(self.lat.accLattice)

        aprtNodes = Add_bend_apertures_to_lattice(self.lat.accLattice, aprtNodes, step = 0.1)

        aprt_drift_step = 0.1
        pos_aprt_start = 0.
        pos_aprt_end = self.accLattice.getLength()
        aprtNodes = Add_drift_apertures_to_lattice(self.lat.accLattice, pos_aprt_start, pos_aprt_end, aprt_drift_step, aprt_pipe_diameter, aprtNodes)

        #---- This will print out the all aperture nodes and their positions
        # #---- You can comment this part out if you wish
        # for node in aprtNodes:
        # 	print "aprt=",node.getName()," pos =",node.getPosition()
        self.accLattice = self.lat.accLattice
        print("===== Aperture Nodes Added =======")

    def initSingleParticle(self,**kwargs):
        """
        Initialize bunch with one particle

        Arguments:

        x0 = 0.0
        xp0 = 0.0
        y0 = 0.0
        yp0 = 0.0
        z0 = 0.0
        dE0 = 0.0		

        """

        # -- initial position/angle in mm, mrad, MeV 
        x0 = float(kwargs.get('x',0.0))
        xp0 = float(kwargs.get('xp',0.0))
        y0 = float(kwargs.get('y',0.0))
        yp0 = float(kwargs.get('yp',0.0))
        z0 = float(kwargs.get('z',0.0))
        dE0 = float(kwargs.get('dE',0.0))

        x0 /= 1000.
        xp0 /= 1000.
        y0 /= 1000.
        yp0 /= 1000.
        z0 /= 1000.
        dE0 /= 1000.


        self.bunch_in = Bunch()
        self.bunch_in.charge(self.charge)

        # -- add particle
        self.bunch_in.addParticle(x0,xp0,y0,yp0,z0,dE0)

        # -- set the initial kinetic energy in GeV
        self.bunch_in.getSyncParticle().kinEnergy(self.ekin)
        # -- set to a single particle size
        self.bunch_in.macroSize(1)

        # -- dummy coefficient for Z to Phase (fix this later)
        self.z2phase = 1.

        # -- report
        nparts = self.bunch_in.getSize()
        print("Bunch Generation completed with %i macroparticles."%nparts)


    def initBunch(self,**kwargs):
        """
        Initialize bunch, either based on twiss parameters or sampling of phase space distribution
        Right now, only sampling of 2 independent phase spaces X-X' and Y-Y' is implemented
        Default (with no arguments) is:
        200k particles, 40 mA peak, 3D waterbag distribution with default twiss parameters 

        Arguments:
        gen = 'twiss' {also: '2d' or 'load'}
        dist = 'waterbag' {also: 'kv' or 'gaussian'}
        threshold = 1e-6
        cutoff = -1 {only used if dist=gaussian)
        current = 0.040 {A}
        nparts = 200,000 {not valid if gen=load}
        centering = 1
        ax = -1.9899
        bx = 0.19636
        ex = 0.160372
        ay = 1.92893
        by = 0.17778
        ey = 0.16362
        az = 0.
        bz = 0.6
        ez = 0.2

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

        Input Twiss params are normalized RMS value; emit: mm-mrad; beta: mm/mrad

        """

        bunchgenerator = kwargs.get('gen','twiss') # options are twiss, 2dxy, 2d+E, 2dxyz (others to be added later)
        # -- check if valid option:
        if not(bunchgenerator in ['load','twiss','2d','2dx3','2d+E']):
            raise KeyError('gen=%s not valid. allowed generators: [ load, twiss , 2d , 2dx3 , 2d+E ]'%bunchgenerator)


        centering = kwargs.get('centering',1) # if 0, does not center bunch coordinates.

        # -- default values of twiss params (not used if generating function does not call for them...)
        alphaX = float(kwargs.get('ax',-1.9899))
        betaX = float(kwargs.get('bx',0.19636))
        emitX = float(kwargs.get('ex',0.160372))
        alphaY = float(kwargs.get('ay',1.92893))
        betaY = float(kwargs.get('by',0.17778))
        emitY = float(kwargs.get('ey',0.160362))
        alphaZ = float(kwargs.get('az',0.))
        betaZ = float(kwargs.get('bz',0.6))
        emitZ = float(kwargs.get('ez',0.2))

        #---make emittances un-normalized XAL units [m*rad]
        emitX = 1.0e-6*emitX/(self.gamma*self.beta)
        emitY = 1.0e-6*emitY/(self.gamma*self.beta)
        emitZ = 1.0e-6*emitZ/(self.gamma**3*self.beta)

        #---- transform to pyORBIT emittance[GeV*m]
        emitZ = emitZ*self.gamma**3*self.beta**2*self.mass
        betaZ = betaZ/(self.gamma**3*self.beta**2*self.mass)
        if bunchgenerator=='twiss':
            print("========= PyORBIT Twiss ===========")
            print("alpha beta emitt[mm*mrad] X= %6.4f %6.4f %6.4f "%(alphaX,betaX,emitX*1.0e+6))
            print("alpha beta emitt[mm*mrad] Y= %6.4f %6.4f %6.4f "%(alphaY,betaY,emitY*1.0e+6))
        if bunchgenerator in ['twiss','2d','2d+E']:
            print("alpha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f "%(alphaZ,betaZ,emitZ*1.0e+6))

        # -- can load bunch through file
        if bunchgenerator == 'load':
            defaultbunchfilename = self.defaults.homedir + self.defaults.defaultdict["BUNCH_IN"]
            bunchfilename = kwargs.get('file',defaultbunchfilename)
            bunchfileformat = kwargs.get('fileformat','pyorbit')

            # -- load bunch generator
            bunch_gen = gen_bunch.Base_BunchGenerator()

            # -- generate bunch by reading in file w/  macroparticle coordinates
            print("Reading in bunch from file %s..."  %bunchfilename)
            if bunchfileformat == 'pyorbit':
                self.bunch_in = Bunch()
                self.bunch_in.readBunch(bunchfilename)
            elif bunchfileformat == 'parmteq':
                self.bunch_in = bunch_gen.readParmteqBunch(bunchfilename)
            else:
                raise KeyError('Do not recognize format %s for bunch file'%(bunchfileformat))
            # -- read in nparts and macrosize
            nparts = self.bunch_in.getSize()

            macrosize = self.bunch_in.macroSize()
            # -- overwrite mass and charge:
            #self.bunch_in.getSyncParticle().kinEnergy(self.ekin)
            self.bunch_in.mass(self.mass)
            self.bunch_in.charge(self.charge)

            self.z2phase = bunch_gen.getZtoPhaseCoeff()
            self.current = bunch_gen.getBeamCurrent()*1e-3

            ## -- to-do: insert function to extract peak current from coordinates
            ## peak current = N_peak * macroSize * consts.charge_electron * self.beta * consts.speed_of_light
            ## where N_peak is peak number density
            print("Bunch read completed. Imported %i macroparticles."%nparts)

        # -- or, can generate bunch by Twiss params or measured distributions
        if bunchgenerator in ["twiss","2d","2d+E","2dx3"]:

            nparts = kwargs.get('nparts',200000)
            self.current = kwargs.get('current',0.040)
            # -- choose distributor class
            bunchdistributor = kwargs.get('dist','gaussian') # options are waterbag, kv or gaussian
            cut_off = kwargs.get("cutoff",-1) 

            # -- use generator to make bunch according to specified distributor class
            distributorClass = None
            if bunchdistributor == 'gaussian':
                if bunchgenerator == 'twiss':
                    distributorClass = GaussDist3D
                elif bunchgenerator in ['2d','2d+E']:
                    distributorClass = GaussDist1D
            elif bunchdistributor == 'waterbag':
                if bunchgenerator == 'twiss':
                    distributorClass = WaterBagDist3D
                elif bunchgenerator in ['2d','2d+E']:
                    distributorClass = WaterBagDist1D
            elif bunchdistributor == 'kv':
                if bunchgenerator == 'twiss':
                    distributorClass = KVDist3D
                elif bunchgenerator in ['2d','2d+E']:
                    distributorClass = KVDist1D
            else: raise ValueError("Do not recognize distributor %s. Accepted classes are 'gaussian', 'waterbag','kv'"%bunchdistributor)


            # -- make generator instances
            if bunchgenerator == "twiss":

                twissX = TwissContainer(alphaX,betaX,emitX)
                twissY = TwissContainer(alphaY,betaY,emitY)
                twissZ = TwissContainer(alphaZ,betaZ,emitZ)

                print("Generating bunch based off twiss parameters ( N = %i )"%nparts)
                bunch_gen = gen_bunch.BTF_Linac_BunchGenerator(twissX, twissY, twissZ, \
                                                               mass=self.mass, charge=self.charge, \
                                                               ekin=self.ekin, curr=self.current*1e3, \
                                                               freq=self.freq)

            elif bunchgenerator in ["2d","2d+E","2dx3"]:
                xfilename = kwargs.get('xfile','')
                yfilename = kwargs.get('yfile','')
                sample_method = kwargs.get('sample','cdf') 
                thres = kwargs.get('threshold',1e-6)

                phaseSpGenX = gen_bunch.PhaseSpaceGen(xfilename,threshold=thres)
                phaseSpGenY = gen_bunch.PhaseSpaceGen(yfilename,threshold=thres)
                twissZ = TwissContainer(alphaZ,betaZ,emitZ)

                if bunchgenerator == "2d":

                    print("Generating bunch based off 2d emittance measurements ( N = %i )"%nparts)
                    bunch_gen = gen_bunch.BTF_Linac_TrPhaseSpace_BunchGenerator( phaseSpGenX, phaseSpGenY, twissZ, \
                                                                            mass=self.mass, charge=self.charge, \
                                                                            ekin=self.ekin, curr=self.current*1e3, \
                                                                            freq=self.freq, method=sample_method)

                elif bunchgenerator == '2d+E':

                    efilename = kwargs.get('efile','')
                    phaseSpGenZ = gen_bunch.PhaseSpaceGenZPartial(efilename,twissZ, zdistributor=distributorClass,cut_off=cut_off)

                    print("Generating bunch based off 2d emittance + 1d energy profile measurements ( N = %i )"%nparts)
                    bunch_gen = gen_bunch.BTF_Linac_6DPhaseSpace_BunchGenerator( phaseSpGenX, phaseSpGenY, phaseSpGenZ, \
                                                                                mass=self.mass, charge=self.charge, \
                                                                                ekin=self.ekin, curr=self.current*1e3, \
                                                                                freq=self.freq, method=sample_method)		
                elif bunchgenerator == '2dx3':

                    zfilename = kwargs.get('zfile','')
                    phaseSpGenZ = gen_bunch.PhaseSpaceGen(zfilename, threshold=thres)

                    print("Generating bunch based off 2d emittances in x,y,z planes ( N = %i )"%nparts)
                    # -- is this the right method?
                    bunch_gen = gen_bunch.BTF_Linac_6DPhaseSpace_BunchGenerator( phaseSpGenX, phaseSpGenY, phaseSpGenZ, \
                                                                                mass=self.mass, charge=self.charge, \
                                                                                ekin=self.ekin, curr=self.current*1e3, \
                                                                                freq=self.freq, method=sample_method)	

            # -- set the initial kinetic energy in GeV
            bunch_gen.setKinEnergy(self.ekin)

            # -- generate bunch
            self.bunch_in = bunch_gen.getBunch(nParticles = int(nparts), distributorClass = distributorClass)

            # -- save coefficient for Z to Phase
            self.z2phase = bunch_gen.getZtoPhaseCoeff()

            # -- report
            nparts = self.bunch_in.getSize()
            print("Bunch Generation completed with %i macroparticles."%nparts)

        if centering:
            self.centerBunch()


    def centerBunch(self):
        """
        Bunch center after generating can have small deviation from the (0,0,0,0,0,0)
        This function will force centering the bunch.
        """
        twiss_analysis = BunchTwissAnalysis()		
        twiss_analysis.analyzeBunch(self.bunch_in)	
        #-----------------------------------------------
        # let's center the beam
        (x_avg,y_avg) = (twiss_analysis.getAverage(0),twiss_analysis.getAverage(2))
        (xp_avg,yp_avg) = (twiss_analysis.getAverage(1),twiss_analysis.getAverage(3))
        (z_avg,dE_avg) = (twiss_analysis.getAverage(4),twiss_analysis.getAverage(5))
        for part_id in range(self.bunch_in.getSize()):
            self.bunch_in.x(part_id,self.bunch_in.x(part_id) - x_avg)
            self.bunch_in.y(part_id,self.bunch_in.y(part_id) - y_avg)
            self.bunch_in.xp(part_id,self.bunch_in.xp(part_id) - xp_avg)
            self.bunch_in.yp(part_id,self.bunch_in.yp(part_id) - yp_avg)
            self.bunch_in.z(part_id,self.bunch_in.z(part_id) - z_avg)
            self.bunch_in.dE(part_id,self.bunch_in.dE(part_id) - dE_avg)
        #-----------------------------------------------
        print('bunch centered \n before correction x=%.6f, x\'=%.6f, y=%.6f, y\'=%.6f, z=%.6f, dE=%.9f'%(x_avg,xp_avg,y_avg,yp_avg,z_avg,dE_avg))

    def shiftBunch(self,**kwargs):

        # -- initial position/angle
        x0 = float(kwargs.get('x0',0.0))
        xp0 = float(kwargs.get('xp0',0.0))
        y0 = float(kwargs.get('y0',0.0))
        yp0 = float(kwargs.get('yp0',0.0))

        # -- shift centroid according to x0, y0 etc
        for i in range(self.bunch_in.getSize()):
            x = self.bunch_in.x(i) # retrieve value
            self.bunch_in.x(i,x+x0) # add offset
            xp = self.bunch_in.xp(i) # retrieve value
            self.bunch_in.xp(i,xp+xp0) # add offset
            y = self.bunch_in.y(i) # retrieve value
            self.bunch_in.y(i,y+y0) # add offset
            yp = self.bunch_in.yp(i) # retrieve value
            self.bunch_in.yp(i,yp+yp0) # add offset

    def attenuateBunch(self,att):
        """
        att = fractional attenuation (0--> 1)
        Function adjusts current without changing number of particles
        (can also be used to increase current)
        """
        macrosize = self.bunch_in.macroSize()
        self.bunch_in.macroSize(macrosize*att)

    def decimateBunch(self,dec):
        """
        dec = power of 10 (dec < log(nparts))
        Function reduces number of particles in bunch by factor of 10 
        without changing current in bunch
        """
        nparts0 = self.bunch_in.getSizeGlobal()
        # -- only decimate if >0 and < log10(nparts)
        if (dec < np.log10(nparts0)) & (dec > 0):
            ind = np.arange(0,nparts0,nparts0/10**dec).astype('int')
            nparts = len(ind)

            # -- make new bunch and place every 10^dec particle inside
            newbunch = Bunch()
            self.bunch_in.copyEmptyBunchTo(newbunch) # copy attributes
            for i in ind:
                newbunch.addParticle(self.bunch_in.x(i),
                                     self.bunch_in.xp(i),
                                     self.bunch_in.y(i),
                                     self.bunch_in.yp(i),
                                     self.bunch_in.z(i),
                                     self.bunch_in.dE(i))

            # -- keep same current by re-setting macrosize
            newmacrosize = self.bunch_in.macroSize()*(nparts0/10**dec)
            newbunch.macroSize(newmacrosize)

            # -- over-write bunch_in
            newbunch.copyBunchTo(self.bunch_in)			

        else:
            print("No decimation for 10^%f"%dec)

    def resampleBunch(self,nparts,rms_factor=0.05):
        """
        Up-sample or down-sample to obtain requested number of particles. 

        nparts = number of desired particles in bunch. 
        """
        nparts0 = self.bunch_in.getSizeGlobal()
        mult = float(nparts)/float(nparts0)
        nparts = int(nparts)
        print('Resampling bunch from %i to %i particles...'%(nparts0,nparts) )

        print('mult = %.3f' % mult)
        print(mult==1)

        # -- make an array of existing coordinates (length=nparts0)
        coords0 = np.zeros([nparts0,6])
        for i in range(nparts0):
            coords0[i,:] = [self.bunch_in.x(i),
                            self.bunch_in.xp(i),
                            self.bunch_in.y(i),
                            self.bunch_in.yp(i),
                            self.bunch_in.z(i),
                            self.bunch_in.dE(i)]

        if mult == 1:
            return []
        else:

            # -- down-sample if nparts0 > nparts-new
            if mult < 1:
                ind = np.random.permutation(np.arange(nparts0))[0:nparts]	

            # -- up-sample if nparts0 < nparts-new
            # -- this way is a lot of work
            elif mult > 1:
                nnew = nparts-nparts0

                # -- normal distribution of new particles will be ~1% rms width 
                rmswidths = np.sqrt((coords0**2).mean(axis=0))
                scale = rmswidths*rms_factor
                # -- longitudinal will be 10x smaller
                scale[4] *= .1
                scale[5] *= .1

                # -- get integer multiplier (round up)
                intmult = int(np.ceil(mult))
                # -- explode each coordinate into intmultx particles (Gaussian cloud)
                # -- this will create a bunch with integere x original size (ie, 2x, 3x, etc..)
                newcoords = np.random.normal(loc=coords0,scale=scale,size=[intmult,nparts0,6])
                reshape_coords = np.zeros([intmult*nparts0,6],dtype='f8')
                for i in range(6):
                    reshape_coords[:,i] = newcoords[:,:,i].flatten()

                coords0 = reshape_coords.copy()

                # -- and downsample to desired number
                ind = np.random.permutation(np.arange(len(coords0)))[0:nparts]

            # -- make new bunch and place re-sampled coordinates
            newbunch = Bunch()
            self.bunch_in.copyEmptyBunchTo(newbunch) # copy attributes
            for i in ind:
                newbunch.addParticle(coords0[i,0],
                                     coords0[i,1],
                                     coords0[i,2],
                                     coords0[i,3],
                                     coords0[i,4],
                                     coords0[i,5])

            # -- keep same current by re-setting macrosize
            newmacrosize = self.bunch_in.macroSize()*(1./mult)
            newbunch.macroSize(newmacrosize)

            # -- over-write bunch_in
            newbunch.copyBunchTo(self.bunch_in)	


            print('done resampling')

            return newbunch


    def dumpParmila(self,**kwargs):
        filename = kwargs.get('filename',[])
        bunch_gen = gen_bunch.Base_BunchGenerator()
        if filename:
            bunch_gen.dumpParmilaFile(self.bunch_in, phase_init = -0.0, fileName = 	filename)
            print("Bunch dumped to %s"%filename)
        else: 
            bunch_gen.dumpParmilaFile(self.bunch_in, phase_init = -0.0)
            print("Bunch dumped to default file")