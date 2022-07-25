#! /usr/bin/env python

"""
This script will track the bunch through the BTF lattice.
"""
class forward_track:
	def __init__(self,ax,bx,ex,ay,by,ey,az,bz,ez,Q01,Q02,Q03,Q04,Q05,Q06,pos_cal,current,Part_Num,sc_sover,elpsN):
		import sys
		import math
		import random
		import time
		import numpy as np

		from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory

		# from linac import the C++ RF gap classes
		from linac import BaseRfGap, MatrixRfGap, RfGapTTF

		from orbit.bunch_generators import TwissContainer
		from orbit.bunch_generators import WaterBagDist3D, GaussDist3D, KVDist3D

		from bunch import Bunch
		from bunch import BunchTwissAnalysis
		from orbit_utils import BunchExtremaCalculator

		from orbit.lattice import AccLattice, AccNode, AccActionsContainer

		from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
		from orbit.py_linac.lattice_modifications import Add_bend_apertures_to_lattice
		from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
		from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice
		from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice

		from btf_linac_bunch_generator import BTF_Linac_BunchGenerator
		from btf_linac_bunch_generator import DumpBunchAccNode
		from btfsim.lattice.Lattice_change_class_FODO import new_lattice

		import btfsim.util.Defaults as default
		defaults = default.getDefaults()

		random.seed(100)

		#---- define the peak current in mA
		#peak_current = 40.0

		# print "debug peak_current[mA]=",peak_current

		names = ["MEBT",]
		latticeFileName =  defaults.latticedir + "FODO_lattice.xml"    # Read lattice configuration
		newLattice = new_lattice(latticeFileName,pos_cal)   #Produce new lattice configuration, name is "btf_mebt.xml"
		magNumbers = newLattice.MagNum()

		#---- create the factory instance
		btf_linac_factory = SNS_LinacLatticeFactory()
		btf_linac_factory.setMaxDriftLength(0.012)    #0.00001   0.016  float(Drift_len)

		#---- the XML file name with the structure
		xml_file_name = defaults.latticedir + "temp_btf_mebt.xml"

		#---- make lattice from XML file 
		accLattice = btf_linac_factory.getLinacAccLattice(names,xml_file_name)

		# print "Linac lattice is ready. L=",accLattice.getLength()

		#---- Set up a new field in quads (different from the lattice XML file)
	
		quads = accLattice.getQuads()
		quad_name_dict = {}
		for quad in quads:
			quad_name_dict[quad.getName()] = quad

		if (magNumbers ==4):
			quad_name_dict["MEBT:QV10"].setParam("dB/dr",-float(Q01)/quad_name_dict["MEBT:QV10"].getLength())
			quad_name_dict["MEBT:QH11"].setParam("dB/dr",-float(Q02)/quad_name_dict["MEBT:QH11"].getLength())
			quad_name_dict["MEBT:QV12"].setParam("dB/dr",-float(Q03)/quad_name_dict["MEBT:QV12"].getLength())
			quad_name_dict["MEBT:QH13"].setParam("dB/dr",-float(Q04)/quad_name_dict["MEBT:QH13"].getLength())
		else:
			if (magNumbers ==5):
				quad_name_dict["MEBT:QV10"].setParam("dB/dr",-float(Q01)/quad_name_dict["MEBT:QV10"].getLength())
				quad_name_dict["MEBT:QH11"].setParam("dB/dr",-float(Q02)/quad_name_dict["MEBT:QH11"].getLength())
				quad_name_dict["MEBT:QV12"].setParam("dB/dr",-float(Q03)/quad_name_dict["MEBT:QV12"].getLength())
				quad_name_dict["MEBT:QH13"].setParam("dB/dr",-float(Q04)/quad_name_dict["MEBT:QH13"].getLength())
				quad_name_dict["MEBT:QH14"].setParam("dB/dr",-float(Q05)/quad_name_dict["MEBT:QH14"].getLength())
			else:
				quad_name_dict["MEBT:QV10"].setParam("dB/dr",-float(Q01)/quad_name_dict["MEBT:QV10"].getLength())
				quad_name_dict["MEBT:QH11"].setParam("dB/dr",-float(Q02)/quad_name_dict["MEBT:QH11"].getLength())
				quad_name_dict["MEBT:QV12"].setParam("dB/dr",-float(Q03)/quad_name_dict["MEBT:QV12"].getLength())
				quad_name_dict["MEBT:QH13"].setParam("dB/dr",-float(Q04)/quad_name_dict["MEBT:QH13"].getLength())
				quad_name_dict["MEBT:QH14"].setParam("dB/dr",-float(Q05)/quad_name_dict["MEBT:QH14"].getLength())
				quad_name_dict["MEBT:QV15"].setParam("dB/dr",-float(Q06)/quad_name_dict["MEBT:QV15"].getLength())
	   		
		#-----------------------------------------------------
		# Set up Space Charge Acc Nodes
		#-----------------------------------------------------
		from orbit.space_charge.sc3d import setSC3DAccNodes, setUniformEllipsesSCAccNodes
		from spacecharge import SpaceChargeCalcUnifEllipse, SpaceChargeCalc3D
		sc_path_length_min = 0.015   #0.000001    0.015

		print "Set up Space Charge nodes. "

		if (sc_sover == "Ellipse"):
		# set of uniformly charged ellipses Space Charge    The bigger numbre of ellipse, the more accurate of sapce charge calculation
			nEllipses = int(elpsN )                                # Ellipse method can be used for the initial estimate, because it calculates faster than FFT method.
			calcUnifEllips = SpaceChargeCalcUnifEllipse(nEllipses)
			space_charge_nodes = setUniformEllipsesSCAccNodes(accLattice,sc_path_length_min,calcUnifEllips)

		else:
			# set FFT 3D Space Charge      FFT Poission solver, the more sizes (grids), the more accurate of space charge calculation
			sizeX = 32*2                       #Particle number should be increased by multiplier**3 when grid increases by a multiplier
			sizeY = 32*2
			sizeZ = 32*2
			calc3d = SpaceChargeCalc3D(sizeX,sizeY,sizeZ)
			space_charge_nodes =  setSC3DAccNodes(accLattice,sc_path_length_min,calc3d)
		

		max_sc_length = 0.
		min_sc_length = accLattice.getLength()
		for sc_node in space_charge_nodes:
			scL = sc_node.getLengthOfSC()
			if(scL > max_sc_length): max_sc_length = scL
			if(scL < min_sc_length): min_sc_length = scL
		print "maximal SC length =",max_sc_length,"  min=",min_sc_length

		print "===== Aperture Nodes START  ======="
		aprtNodes = Add_quad_apertures_to_lattice(accLattice)

		aprtNodes = Add_bend_apertures_to_lattice(accLattice, aprtNodes, step = 0.1)

		aprt_pipe_diameter = 0.04
		aprt_drift_step = 0.1
		pos_aprt_start = 0.
		pos_aprt_end = 10.
		aprtNodes = Add_drift_apertures_to_lattice(accLattice, pos_aprt_start, pos_aprt_end, aprt_drift_step, aprt_pipe_diameter, aprtNodes)


		#---- This will print out the all aperture nodes and their positions
		#---- You can comment this part out if you wish
		# for node in aprtNodes:
		# 	print "aprt=",node.getName()," pos =",node.getPosition()

		# print "===== Aperture Nodes Added ======="

		#-----TWISS Parameters at the entrance of MEBT ---------------
		# transverse emittances are unnormalized and in pi*mm*mrad
		# longitudinal emittance is in pi*eV*sec
		e_kin_ini = 0.0025 # in [GeV]
		mass = 0.939294    # in [GeV]
		gamma = (mass + e_kin_ini)/mass
		beta = math.sqrt(gamma*gamma - 1.0)/gamma
		print "relat. gamma=",gamma
		print "relat.  beta=",beta
		frequency = 402.5e+6
		v_light = 2.99792458e+8  # in [m/sec]

		if (float(ax) == 0.0 and float(bx) == 0.0 and float(ex) == 0.0 and float(ay) == 0.0 and float(by) == 0.0 and float(ey) == 0.0 and \
			float(az) == 0.0 and float(bz) == 0.0 and float(ez) == 0.0):

			bunch_gen = BTF_Linac_BunchGenerator(TwissContainer(0.0,0.2,0.1),TwissContainer(0.0,0.2,0.1),TwissContainer(0.0,0.2,0.1))
			bunch_in = Bunch()
			bunch_in.readBunch(defaults.defaultdict["BUNCHOUTDIR"]+"Input_distributin_external.txt")
			print ("Beam current = %.2f mA")%(float(current)*frequency*float(Part_Num)*1.6021773e-19*1000)

		else:
			peak_current = float(current)
			print "debug peak_current[mA]=",peak_current

			(alphaX,betaX,emittX) = (float(ax), float(bx), float(ex))   # Transverse emittance: normalized RMS value, mm-mrad; beta: mm/mrad   
			(alphaY,betaY,emittY) = ( float(ay), float(by), float(ey))
			(alphaZ,betaZ,emittZ) = ( float(az), float(bz), float(ez))    # Longitudinal emittance: normalized RMS value, mm-mrad; beta: mm/mrad

			alphaZ = - alphaZ

		#---make emittances un-normalized XAL units [m*rad]
			emittX = 1.0e-6*emittX/(gamma*beta)
			emittY = 1.0e-6*emittY/(gamma*beta)
			emittZ = 1.0e-6*emittZ/(gamma**3*beta)
			print " ========= XAL Twiss ==========="
			print " aplha beta emitt[mm*mrad] X= %6.4f %6.4f %6.4f "%(alphaX,betaX,emittX*1.0e+6)
			print " aplha beta emitt[mm*mrad] Y= %6.4f %6.4f %6.4f "%(alphaY,betaY,emittY*1.0e+6)
			print " aplha beta emitt[mm*mrad] Z= %6.4f %6.4f %6.4f "%(alphaZ,betaZ,emittZ*1.0e+6)

		#---- long. size in mm
			sizeZ = math.sqrt(emittZ*betaZ)*1.0e+3

		#---- transform to pyORBIT emittance[GeV*m]
			emittZ = emittZ*gamma**3*beta**2*mass
			betaZ = betaZ/(gamma**3*beta**2*mass)

			print " ========= PyORBIT Twiss ==========="
			print " aplha beta emitt[mm*mrad] X= %6.4f %6.4f %6.4f "%(alphaX,betaX,emittX*1.0e+6)
			print " aplha beta emitt[mm*mrad] Y= %6.4f %6.4f %6.4f "%(alphaY,betaY,emittY*1.0e+6)
			print " aplha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f "%(alphaZ,betaZ,emittZ*1.0e+6)

			twissX = TwissContainer(alphaX,betaX,emittX)
			twissY = TwissContainer(alphaY,betaY,emittY)
			twissZ = TwissContainer(alphaZ,betaZ,emittZ)

			print "Start Bunch Generation."
			bunch_gen = BTF_Linac_BunchGenerator(twissX,twissY,twissZ)

		#set the initial kinetic energy in GeV
			bunch_gen.setKinEnergy(e_kin_ini)

		#set the beam peak current in mA
			bunch_gen.setBeamCurrent(peak_current)

		#bunch_in = bunch_gen.getBunch(nParticles = 1500000, distributorClass = WaterBagDist3D)
			bunch_in = bunch_gen.getBunch(nParticles = int(Part_Num), distributorClass = GaussDist3D)
		#bunch_in = bunch_gen.getBunch(nParticles = 10000, distributorClass = KVDist3D)

		bunch_gen.dumpParmilaFile(bunch_in, phase_init = -0.0, fileName =defaults.defaultdict["BUNCHOUTDIR"]+"Distribution_Input_new.txt")
		print "Bunch Generation completed."
		# print "Beam current is: ", bunch_gen.getBeamCurrent()

		#track through the lattice 
		paramsDict = {"old_pos":-1.,"count":0,"pos_step":0.05}
		actionContainer = AccActionsContainer("Test Design Bunch Tracking")

		# pos_start = 0.

		twiss_analysis = BunchTwissAnalysis()
		# bunch_extrema_cal = BunchExtremaCalculator()

		#file_extrema_out = open("Extrema_out.txt","w")
		# file_out = open("pyorbit_btf_twiss_sizes_ekin.txt","w")
		# file_percent_particle_out = open("Percent_particle_coords.txt","w")

		# s = " Node   position "
		# s += "   alphaX betaX emittX  normEmittX"
		# s += "   alphaY betaY emittY  normEmittY"
		# s += "   alphaZ betaZ emittZ  emittZphiMeV"
		# s += "   sizeX sizeY sizeZ_deg"
		# s += "   eKin Nparts "
		# file_out.write(s+"\n")

		self.accLattice = accLattice
		self.bunch_in = bunch_in
		self.paramsDict = paramsDict
		self.actionContainer = actionContainer
		self.lattice_L = accLattice.getLength()
		self.twiss_analysis = twiss_analysis
		self.bunch_gen = bunch_gen
		self.AccActionsContainer  = AccActionsContainer
		self.frequency = frequency
		self.v_light = v_light

	def sim(self):
		return self.accLattice, self.bunch_in, self.paramsDict, self.actionContainer,self.lattice_L,\
				self.twiss_analysis,  self.bunch_gen, self.AccActionsContainer, self.frequency,self.v_light

		
		
		
	




