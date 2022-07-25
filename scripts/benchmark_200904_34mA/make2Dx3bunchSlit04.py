import numpy as np
import sys
sys.path.append('/mnt/c/Users/k0r/soft/btf-pyorbit-simulation/')
sys.path.append('/media/sf_OracleShared/soft/btf-pyorbit-simulation/')
import btfsim.sim.simulation_main as main
import btfsim.bunch.bunchUtilities as butil

####################################################################################
## Files
####################################################################################

emitdir = 'emittance-files/'

# 1st emit, 35 mA, Snapshot_20200831_155501.mstate
xfile = emitdir + 'emittance-data-x-20200902-1599057774224.csv'
yfile = emitdir + 'emittance-data-y-20200902-1599060458388.csv'
zfile = emitdir + 'BSM-2d_02-Sep-2020_16_08_27-emittance-z-dE.csv' # core
threshold = 0.01

out_bunch_filename = '2Dx3_200902_HZ04_34mA_10M'
####################################################################################
## Set-up parameters
####################################################################################

beam_current = 0.034
bunch_frequency= 402.5e6
nParticles = 10e6

ekin = 0.0025 # in [GeV]
mass = 0.939294 # in [GeV]
gamma = (mass + ekin)/mass
beta = np.sqrt(gamma*gamma - 1.0)/gamma

method = 'cdf'

sim = main.simBTF()

# -- generate 2D bunch out of emittance measurements
sim.initBunch(gen="2dx3",xfile=xfile,yfile=yfile,zfile=zfile,threshold=threshold,
              nparts=nParticles,current=beam_current)

sim.bunch_in.dumpBunch(out_bunch_filename)

# -- analyze bunch
bcalc = butil.bunchCalculation(sim.bunch_in)
twissx = bcalc.Twiss(plane='x')
print(twissx)
twissy = bcalc.Twiss(plane='y')
print(twissy)
twissz = bcalc.Twiss(plane='z')
print(twissz)





# -- propagate via simulation
mstatefile = '/home/kruisard/Dropbox/Data/mstates/Snapshot_20200831_155501.mstate'

# -- init default lattice w/ mstate solution
sim.initLattice(beamline=["MEBT1","MEBT2","MEBT3"],mstatename=mstatefile)


# -- set up apertures
sim.initApertures(d=0.04)

# -- set up SC nodes
sclen = 0.001
sim.initSCnodes(minlen = sclen, solver='fft',gridmult=6)


## -- first run backwards to RFQ exit.
out_bunch_name= out_bunch_filename.replace('HZ04','RFQ0')
start = 0.
stop = "MEBT:HZ04"
sim.reverse(start=start, stop=stop, out = out_bunch_name)

# -- save output
hist_out_name = 'data/btf_hist_toRFQ_10M.txt'
sim.tracker.writehist(filename=hist_out_name)

# -- run forwards to VS06
#out_bunch_name= '2Dx3_190326_190724_FC12_26mA_200k'
#start = "MEBT:HZ04"
#stop = "MEBT:FC12"
#sim.run(start=start, stop=stop, out = out_bunch_name)
#
## -- save output
#hist_out_name = 'data/btf_hist_toFC12.txt'
#sim.tracker.writehist(filename=hist_out_name)


