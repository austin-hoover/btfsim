"""Script to run BTF PyORBIT simulation."""
import sys

sys.path.append('../../')
from btfsim.sim import simulation_main as main
import btfsim.lattice.magUtilities as magutil
import btfsim.bunch.bunchUtilities as butil
from btfsim.lattice.btf_quad_func_factory import BTF_QuadFunctionFactory as quadfunc
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes


start = 'HZ04'
stop = 'MEBT3'
mstatefile = '/home/kruisard/Dropbox/Data/mstates/Snapshot_20200831_155501.mstate'
bunchfilename = 'bunches/2Dx3_200902_HZ04_34mA_200k'

irun = 0

####################################################################################
## Set-up parameters
####################################################################################

# flags
replace_quads = 1 # if 1, replaced FODO quads with overlapping fields model
dispersionFlag = 0 # if 1, dispersion corrected for in twiss calculation.

# space charge calculation
sclen = 0.01 # [meters]
gridmult= 6
n_bunches = 3 # number of bunches to model (for field calculation with overlapping bunches)

# initial centroid position
x0 = 0. #mm
y0 = 0.#6.17 #mm
xp0 = 0. #mrad
yp0 = 0.#-2.2
dE0 = 0. #Mev


###############################################################################
# -- SETUP SIM
###############################################################################

runname = sys.argv[0][0:-3].split('_')[1]
textfilename = runname + '.txt'

# -- init sim of BTF (whack into separate sections)
sim = main.simBTF()

# -- set dispersion flag
sim.dispersionFlag = dispersionFlag

# -- init default lattice w/ mstate solution
#sim.initLattice(beamline=["MEBT1","MEBT2","MEBT3"],mstatename=mstatefile)
sim.initLattice(beamline=["MEBT1","MEBT2"],mstatename=mstatefile)


# -- manually adjust quad currents
qdict = {}
qdict['QV07'] = 5.2#-sim.lat.magdict['QV07']['current']
qdict['QH08'] = -4.5#-sim.lat.magdict['QH08']['current']
qdict['QV09'] = 4.5#-sim.lat.magdict['QV09']['current']
sim.changeQuads(dict=qdict)



###############################################################################
# -- OVERLAPPING NODES
###############################################################################
# -- replace quads with analytic model (must come after lattice init but before SC nodes)
quad_names = [] # leaving this empty will change all quads in sequence(s)
nPMQs=19 # there are 19 pmq's in FODO
for j in range(nPMQs):
	qname = 'MEBT:FQ%02.0f'%(j+1)
	quad_names.append(qname)

z_step = 0.001
if replace_quads == 1:
    Replace_Quads_to_OverlappingQuads_Nodes(sim.accLattice,\
                                            z_step, \
                                            accSeq_Names = ["MEBT3"], \
                                            quad_Names = quad_names, \
                                            EngeFunctionFactory = quadfunc)



# -- set up SC nodes
sim.initSCnodes(minlen = sclen, solver='fft', gridmult=gridmult,n_bunches=n_bunches)

# -- load initial bunch into sims[0]
sim.initBunch(gen="load",file=bunchfilename)
sim.decimateBunch(4) # reduce number of macroparticles to 10^n
#sim.attenuateBunch(34./41.)  # reduce bunch current
sim.shiftBunch(x0=x0*1e-3,y0=y0*1e-3,xp0=xp0*1e-3,yp0=yp0*1e-3) # move centroid

###############################################################################
# -- Run, track bunch
###############################################################################
if type(start) == str:
    startarg = "MEBT:" + start
else:
    startarg = start
if type(stop) == str:
    stoparg = "MEBT:" + stop
else:
    stoparg = stop



bunchout =  runname+'_%s_%s_bunch_%i.txt'%(str(start),str(stop),irun)

# -- run
#sim.run(start=startarg,stop=stoparg,out = bunchout)
sim.run(start=startarg,out = bunchout)


hist_out_name = 'data/'+runname+'_%s_%s_hist_%i.txt'%(str(start),str(stop),irun)
sim.tracker.writehist(filename=hist_out_name)


