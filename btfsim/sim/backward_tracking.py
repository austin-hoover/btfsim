#! /usr/bin/env python

"""
This script will 
1. generate the bunch from the emittance phase-space table for X,Y
2. transform the bunch for the backward tracking
3. create backward lattice for BTF
4. track the bunch to the entrance of the BTF MEBT
5. transform the bunch for forward tracking
6. dump the bunch at the MEBT entrance for future tracking
"""
class back_tracking:
	def __init__(self,QH01,QH02,QH03,QH04,filename,field_selection):
		import sys
		import math
		import random
		import time

		from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory

		# from linac import the C++ RF gap classes
		from linac import BaseRfGap, MatrixRfGap, RfGapTTF

		from orbit.bunch_generators import TwissContainer
		from orbit.bunch_generators import KVDist1D, KVDist2D, KVDist3D
		from orbit.bunch_generators import GaussDist1D, GaussDist2D, GaussDist3D
		from orbit.bunch_generators import WaterBagDist1D, WaterBagDist2D, WaterBagDist3D


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
		from btf_linac_bunch_generator import PhaseSpaceGen
		from btf_linac_bunch_generator import BTF_Linac_TrPhaseSpace_BunchGenerator
		from btf_linac_bunch_generator import BunchTransformerFunc
		from btf_linac_bunch_generator import BunchX_XP_Y_YP_Z_dE_Centering
		from btf_linac_bunch_generator import DumpBunchCoordinates
		from btf_linac_bunch_generator import ReveseBunchCoordinate
		from btf_linac_bunch_generator import plotBunch

		from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
		from  btf_enge_func_factory import BTF_EngeFunctionFactory

		random.seed(100)

		names = ["MEBT",]

		#---- create the factory instance
		btf_linac_factory = SNS_LinacLatticeFactory()
		btf_linac_factory.setMaxDriftLength(0.001)    # 0.00001

		#---- the XML file name with the structure
		xml_file_name = "./btf_mebt.xml"

		#---- make lattice from XML file 
		accLattice = btf_linac_factory.getLinacAccLattice(names,xml_file_name)

		#---reverse the order of accelerator elements
		#accLattice.getNodes().reverse()
		#accLattice.initialize()
		accLattice.reverseOrder()
		print "Linac lattice is ready. L=",accLattice.getLength()
		#---- Set up a new field in quads (different from the lattice XML file)
		if (field_selection == "HardEdge"):
			quad_field_dict = {}
			quad_field_dict["MEBT:QH01"]= -float(QH01)/accLattice.getNodeForName("MEBT:QH01").getLength()
			quad_field_dict["MEBT:QV02"]= +float(QH02)/accLattice.getNodeForName("MEBT:QV02").getLength()
			quad_field_dict["MEBT:QH03"]= -float(QH03)/accLattice.getNodeForName("MEBT:QH03").getLength()
			quad_field_dict["MEBT:QV04"]= +float(QH04)/accLattice.getNodeForName("MEBT:QV04").getLength()

			quads = accLattice.getQuads()
			for quad in quads:
				if(quad_field_dict.has_key(quad.getName())):
					field = quad_field_dict[quad.getName()]
					quad.setParam("dB/dr",field)
					#print "debug quad new field[T/m]=",field
		else:
			z_step = 0.005 
			int_field_Q1 =  -float(QH01)    #1.313
			int_field_Q2 =   float(QH02)
			int_field_Q3 =  -float(QH03)
			int_field_Q4 =   float(QH04)

			quads = accLattice.getQuads()
			quad_name_dict = {}	
			for quad in quads:
				quad_name_dict[quad.getName()] = quad
			#=======================================
			#Replace_Quads_to_OverlappingQuads_Nodes(accLattice,z_step,["MEBT",],["MEBT:QV02"],
			#BTF_EngeFunctionFactory) should always be below #quads = accLattice.getQuads()
			#==================================================
			Replace_Quads_to_OverlappingQuads_Nodes(accLattice,z_step,["MEBT",],[],BTF_EngeFunctionFactory)
			quad_name_dict["MEBT:QH01"].setParam("dB/dr",int_field_Q1/quad_name_dict["MEBT:QH01"].getLength())
			quad_name_dict["MEBT:QV02"].setParam("dB/dr",int_field_Q2/quad_name_dict["MEBT:QV02"].getLength())
			quad_name_dict["MEBT:QH03"].setParam("dB/dr",int_field_Q3/quad_name_dict["MEBT:QH03"].getLength())
			quad_name_dict["MEBT:QV04"].setParam("dB/dr",int_field_Q4/quad_name_dict["MEBT:QV04"].getLength())
		#-----------------------------------------------------
		# Set up Space Charge Acc Nodes
		#-----------------------------------------------------
		from orbit.space_charge.sc3d import setSC3DAccNodes, setUniformEllipsesSCAccNodes
		from spacecharge import SpaceChargeCalcUnifEllipse, SpaceChargeCalc3D
		sc_path_length_min = 0.001   #0.000001

		print "Set up Space Charge nodes. "

		# set of uniformly charged ellipses Space Charge
		nEllipses = 1
		calcUnifEllips = SpaceChargeCalcUnifEllipse(nEllipses)
		space_charge_nodes = setUniformEllipsesSCAccNodes(accLattice,sc_path_length_min,calcUnifEllips)

		"""
		# set FFT 3D Space Charge
		sizeX = 64
		sizeY = 64
		sizeZ = 64
		calc3d = SpaceChargeCalc3D(sizeX,sizeY,sizeZ)
		space_charge_nodes =  setSC3DAccNodes(accLattice,sc_path_length_min,calc3d)
		"""

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

		aprt_pipe_diameter = 10.03
		aprt_drift_step = 0.1
		pos_aprt_start = 0.
		pos_aprt_end = 10.
		aprtNodes = Add_drift_apertures_to_lattice(accLattice, pos_aprt_start, pos_aprt_end, aprt_drift_step, aprt_pipe_diameter, aprtNodes)

		#---- the size is a total distance between scraper plates. The center is at 0.
		#---- The function AddScrapersAperturesToLattice is for symmetric position of the plates.
		#---- If you need something else do it yourself.
		"""
		x_size = 0.08
		y_size = 0.08
		aprtNodes = AddScrapersAperturesToLattice(accLattice,"MEBT:Slit#1",x_size,y_size,aprtNodes)

		x_size = 0.042
		y_size = 0.042
		aprtNodes = AddScrapersAperturesToLattice(accLattice,"MEBT:Slit#2",x_size,y_size,aprtNodes)
		"""

		"""
		#---- This will print out the all aperture nodes and their positions
		#---- You can comment this part out if you wish
		for node in aprtNodes:
			print "aprt=",node.getName()," pos =",node.getPosition()

		print "===== Aperture Nodes Added ======="


		#----- Kinematics parameters
		e_kin_ini = 0.0025 # in [GeV]
		mass = 0.939294    # in [GeV]
		gamma = (mass + e_kin_ini)/mass
		beta = math.sqrt(gamma*gamma - 1.0)/gamma
		print "relat. gamma=",gamma
		print "relat.  beta=",beta
		frequency = 402.5e+6
		v_light = 2.99792458e+8  # in [m/sec]

		#---- Twiss parameters for Z direction from the design values
		#---- at the first BTF MEBT slit
		(alphaZ,betaZ,emittZ) = ( -12.1478 , 3250.6746, 0.0167*1.0e-6)

		twissZ = TwissContainer(alphaZ,betaZ,emittZ)
		
		#----- read the emittance information from files
		phaseSpGenX = PhaseSpaceGen("1476739191342_hor_plane.txt")
		phaseSpGenY = PhaseSpaceGen("1476740771284_ver_plane.txt")

		#print phaseSpGenX
		#print "Start Bunch Generation."
		bunch_gen = BTF_Linac_TrPhaseSpace_BunchGenerator(phaseSpGenX,phaseSpGenY,twissZ)

		#set the initial kinetic energy in GeV
		bunch_gen.setKinEnergy(e_kin_ini)

		#set the beam peak current in mA
		bunch_gen.setBeamCurrent(peak_current)

		bunch_in = bunch_gen.getBunch(nParticles = 50000, distributorClassZ = WaterBagDist1D)
		#bunch_in = bunch_gen.getBunch(nParticles = 50000, distributorClassZ = GaussDist1D)
		#bunch_in = bunch_gen.getBunch(nParticles = 50000, distributorClassZ = KVDist1D)

		print "Bunch Generation completed."
		
		#---- set the average value for each coordinate of the bunch to 0.
		BunchX_XP_Y_YP_Z_dE_Centering(bunch_in)
		"""
		print('Reading in Bunch from %s'%(filename))
		bunch_in = Bunch()
		bunch_in.readBunch(filename)
#		bunch_in.readBunch("Input_distributin_external_new.txt")

		#---- it is possible that we have wrong sign of Twiss alpha, so
		#---- we have to transform some of coordinates
		#ReveseBunchCoordinate(bunch_in,0) # reverse x
		#ReveseBunchCoordinate(bunch_in,1) # reverse xp
		#ReveseBunchCoordinate(bunch_in,2) # reverse y
		#ReveseBunchCoordinate(bunch_in,3) # reverse yp

		"""
		#----- print out the beam parameters at the 1st slit position
		twiss_analysis = BunchTwissAnalysis()
		twiss_analysis.analyzeBunch(bunch_in)
		x_rms = math.sqrt(twiss_analysis.getTwiss(0)[1]*twiss_analysis.getTwiss(0)[3])*1000.
		y_rms = math.sqrt(twiss_analysis.getTwiss(1)[1]*twiss_analysis.getTwiss(1)[3])*1000.
		z_rms = math.sqrt(twiss_analysis.getTwiss(2)[1]*twiss_analysis.getTwiss(2)[3])*1000.
		z_rms_deg = bunch_gen.getZtoPhaseCoeff(bunch_in)*z_rms/1000.0
		(alphaX,betaX,emittX) = (twiss_analysis.getTwiss(0)[0],twiss_analysis.getTwiss(0)[1],twiss_analysis.getTwiss(0)[3]*1.0e+6)
		(alphaY,betaY,emittY) = (twiss_analysis.getTwiss(1)[0],twiss_analysis.getTwiss(1)[1],twiss_analysis.getTwiss(1)[3]*1.0e+6)
		(alphaZ,betaZ,emittZ) = (twiss_analysis.getTwiss(2)[0],twiss_analysis.getTwiss(2)[1],twiss_analysis.getTwiss(2)[3]*1.0e+6)
		print "======= PyORBIT Twiss params at BTF Emittance Dev."
		print "(alphaX,betaX,emittX) = (%6.4f,  %6.4f,  %6.4f) "%(alphaX,betaX,emittX)
		print "(alphaY,betaY,emittY) = (%6.4f,  %6.4f,  %6.4f) "%(alphaY,betaY,emittY)
		print "(alphaZ,betaZ,emittZ) = (%6.4f,  %6.4f,  %6.4f) "%(alphaZ,betaZ,emittZ)
		print "sx[mm],sy[mm],sz[deg] =  %6.4f   %6.4f   %6.4f  "%(x_rms,y_rms,z_rms_deg)
		print "======================================================================="
		"""
		#---- You can see the particles on Gnuplot graphs
		#plotBunch(bunch_in,0)
		#plotBunch(bunch_in,1)
		#plotBunch(bunch_in,2)

		#---- Dump the coordinates of the particles in the bunch to the file without the usual bunch header
		#DumpBunchCoordinates("bunch_slit_1_coords_" + name + ".txt",bunch_in)


		#--- prepare the bunch to track it backwards
		BunchTransformerFunc(bunch_in)

		#-----Tracking bunch from slit #1 to the beginning of BTF MEBT ------

		#---- Find index of the slit in the backward lattice
		slit_1_node = accLattice.getNodeForName("MEBT:Slit#1")
		slit_1_Ind = accLattice.getNodeIndex(slit_1_node)

		#---- The last index in the backward lattice is the initial nodel in the forward laatice
		lastInd = len(accLattice.getNodes())

		#set up design 
		accLattice.trackDesignBunch(bunch_in,None,None,slit_1_Ind)

		print "Design tracking completed."

		#track through the lattice 
		paramsDict = {"old_pos":-1.,"count":0,"pos_step":0.02}
		actionContainer = AccActionsContainer("Backward Bunch Tracking")


		#---- lattice length is the position of the first node in the forward lattice
		lattice_length = accLattice.getLength()

		#---- The slit #1 position in the backward lattice
		node_pos_dict = accLattice.getNodePositionsDict()
		(slit_1_pos_Before,slit_1_pos_After) = node_pos_dict[slit_1_node]
		slit_1_pos = slit_1_pos_Before

		#---- the initial path of the particle
		paramsDict["path_length"] = slit_1_pos

		self.accLattice = accLattice
		self.bunch_in = bunch_in
		self.paramsDict = paramsDict
		self.actionContainer = actionContainer
		self.lattice_L = accLattice.getLength()
		self.AccActionsContainer  = AccActionsContainer
		# self.frequency = frequency
		# self.v_light = v_light
		self.slit_1_Ind = slit_1_Ind
		self.BunchTransformerFunc = BunchTransformerFunc

	def sim(self):
		return self.accLattice, self.bunch_in, self.paramsDict, self.actionContainer,self.lattice_L,\
			self.AccActionsContainer,self.slit_1_Ind,self.BunchTransformerFunc


	# 	def action_entrance(paramsDict):
	# 		node = paramsDict["node"]
	# 		bunch = paramsDict["bunch"]
	# 		pos = paramsDict["path_length"]
	# 		if(paramsDict["old_pos"] == pos): return
	# 		if(paramsDict["old_pos"] + paramsDict["pos_step"] > pos): return
	# 		paramsDict["old_pos"] = pos
	# 		paramsDict["count"] += 1
			
	# 	def action_exit(paramsDict):
	# 		action_entrance(paramsDict)
			
			
	# 	actionContainer.addAction(action_entrance, AccActionsContainer.ENTRANCE)
	# 	actionContainer.addAction(action_exit, AccActionsContainer.EXIT)

	# 	accLattice.trackBunch(bunch_in, paramsDict = paramsDict, actionContainer = actionContainer, index_start = slit_1_Ind)

	# 	#---- Transform coordinates for the forward tracking  
	# 	BunchTransformerFunc(bunch_in)

	# 	x_b,xp_b = [0]*bunch_in.getSize(),[0]*bunch_in.getSize()
	# 	y_b,yp_b = [0]*bunch_in.getSize(),[0]*bunch_in.getSize()
	# 	for m in range(bunch_in.getSize()):
	# 		x_b[m] = bunch_in.x(m)
	# 		xp_b[m] = bunch_in.xp(m)
	# 		y_b[m] = bunch_in.y(m)
	# 		yp_b[m]= bunch_in.yp(m)
	# 	#---- dump bunch for the future tracking from the beginning of the BTF lattice
	# 	bunch_in.dumpBunch("Bunch_at_RFQ_exit_" + name)
	# 	self.x_b = x_b
	# 	self.xp_b = xp_b
	# 	self.y_b = y_b
	# 	self.yp_b = yp_b
	# def cord_RFQ_exit(self):
	# 	return self.x_b,self.xp_b,self.y_b,self.yp_b


