"""
This script uses Andrei's method of bunch generation.

Reads in input from 2D X and Y emittance scans; (2D intensity arrays produced by emittance GUI)
uses method from mebt_bunch_generator.py to sample phase space distribution
Assumes Gaussian longitudinal distribution with specified Twiss Parameters.

K. Ruisard
2/7/18
"""
import math
import time
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist1D, GaussDist1D, KVDist1D
from btf_linac_bunch_generator import BTF_Linac_TrPhaseSpace_BunchGenerator, PhaseSpaceGen
from src.util.Defaults import getDefaults

default = getDefaults()

#####################
# -- input parameters
#####################
nParticles = 100000
e_kin_rfq = 0.0025 # [GeV]
peak_current = 12 # [mA]
alphaZ = 0 
betaZ = 0.6
emittZ = 0.2
xfilename = '/home/kruisard/Documents/BTF-measurement/GenerateBunch/from-sasha/1549030763838_X_mat.txt'
yfilename = '/home/kruisard/Documents/BTF-measurement/GenerateBunch/from-sasha/1549031343944_Y_mat.txt'

#####################
# -- unit conversions ( for longitudinal distribution ) 
#####################
mass = 0.939294    # in [GeV]
gamma = (mass + e_kin_rfq)/mass
beta = math.sqrt(gamma*gamma - 1.0)/gamma

#---make emittances un-normalized XAL units [m*rad]
emittZ = 1.0e-6*emittZ/(gamma**3*beta)

#---- transform to pyORBIT emittance[GeV*m]
emittZ = emittZ*gamma**3*beta**2*mass
betaZ = betaZ/(gamma**3*beta**2*mass)
alphaZ = -alphaZ # why is this?

print " ========= PyORBIT Twiss ==========="
print " alpha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f "%(alphaZ,betaZ,emittZ*1.0e+6)

# -- make twiss object
twissZ = TwissContainer(alphaZ,betaZ,emittZ)

print "Start Bunch Generation."

# -- make prob. distr. functions for X and Y
phaseSpGenX = PhaseSpaceGen(xfilename,threshold=3e-4)
phaseSpGenY = PhaseSpaceGen(yfilename,threshold=3e-4)

# -- create generator for 6D (uncorrelated) distribution
bunch_gen = BTF_Linac_TrPhaseSpace_BunchGenerator(phaseSpGenX,phaseSpGenY,twissZ)

# -- set the initial kinetic energy in GeV
bunch_gen.setKinEnergy(e_kin_rfq)

#-- set the beam peak current in mA
bunch_gen.setBeamCurrent(peak_current)

# -- Create bunch
t0 = time.clock()
bunch_cdf = bunch_gen.getBunch(nParticles = nParticles, method = 'cdf', distributorClassZ = GaussDist1D)
t1 = time.clock()
bunch_grid = bunch_gen.getBunch(nParticles = nParticles, method = 'grid', distributorClassZ = GaussDist1D)
t2 = time.clock()

# -- center bunches
bunch_cdf = bunch_gen.centerBunch(bunch_cdf)
bunch_grid = bunch_gen.centerBunch(bunch_grid)


# -- write to file
bunch_cdf.dumpBunch(default.bunchdir + "test_bunch_cdf.dat")
bunch_grid.dumpBunch(default.bunchdir + "test_bunch_grid.dat")

"""
# -- save pdf to file as well
import numpy as np
xfilenameout = '/home/kruisard/Documents/BTF-measurement/GenerateBunch/generate-4d-bunch/1549030763838_X_pdf.txt'
yfilenameout = '/home/kruisard/Documents/BTF-measurement/GenerateBunch/generate-4d-bunch/1549031343944_Y_pdf.txt'
np.savetxt(xfilenameout,phaseSpGenX.pdf)
np.savetxt(yfilenameout,phaseSpGenY.pdf)
"""

print("Bunch Generation completed.")
print("CDF method %.3f s" %(t1-t0))
print("Grid method %.3f s" %(t2-t1))
