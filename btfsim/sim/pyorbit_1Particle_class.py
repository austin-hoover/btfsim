#! /usr/bin/env python

"""
This script will track the bunch through the BTF lattice.
"""
class dispersion_curve_cal:
	def __init__(self,Q07,Q08,Q09,E_dis):
		import sys
		import math
		import random
		import time

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

		from btf_linac_bunch_generator_1P import BTF_Linac_BunchGenerator
		from btf_linac_bunch_generator_1P import DumpBunchAccNode
		# from Halo_treatment import Halo_management

		random.seed(100)

		#---- define the peak current in mA
		peak_current = 0.0

		print "debug peak_current[mA]=",peak_current

		names = ["MEBT",]

		#---- create the factory instance
		btf_linac_factory = SNS_LinacLatticeFactory()
		btf_linac_factory.setMaxDriftLength(0.016)

		#---- the XML file name with the structure
		xml_file_name = "./Dispersion_alys_lattice.xml"

		#---- make lattice from XML file 
		accLattice = btf_linac_factory.getLinacAccLattice(names,xml_file_name)

		print "Linac lattice is ready. L=",accLattice.getLength()

		#---- Set up a new field in quads (different from the lattice XML file)
		quads = accLattice.getQuads()
		quad_name_dict = {}
		for quad in quads:
			quad_name_dict[quad.getName()] = quad
		quad_name_dict["MEBT:QV07"].setParam("dB/dr",+float(Q07)/quad_name_dict["MEBT:QV07"].getLength())
		quad_name_dict["MEBT:QH08"].setParam("dB/dr",-float(Q08)/quad_name_dict["MEBT:QH08"].getLength())
		quad_name_dict["MEBT:QV09"].setParam("dB/dr",+float(Q09)/quad_name_dict["MEBT:QV09"].getLength())

		#-----------------------------------------------------
		# Set up Space Charge Acc Nodes
		#-----------------------------------------------------
		from orbit.space_charge.sc3d import setSC3DAccNodes, setUniformEllipsesSCAccNodes
		from spacecharge import SpaceChargeCalcUnifEllipse, SpaceChargeCalc3D
		sc_path_length_min = 0.015

		print "Set up Space Charge nodes. "

		"""
		# set of uniformly charged ellipses Space Charge    The bigger numbre of ellipse, the more accurate of sapce charge calculation
		nEllipses = 13                                # Ellipse method can be used for the initial estimate, because it calculates faster than FFT method.
		calcUnifEllips = SpaceChargeCalcUnifEllipse(nEllipses)
		space_charge_nodes = setUniformEllipsesSCAccNodes(accLattice,sc_path_length_min,calcUnifEllips)
		"""

		# set FFT 3D Space Charge      FFT Poission solver, the more sizes (grids), the more accurate of space charge calculation
		sizeX = 32*2                        #Particle number should be increased by multiplier**3 when grid increases by a multiplier
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

		aprt_pipe_diameter = 0.08
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

		#---- This will print out the all aperture nodes and their positions
		#---- You can comment this part out if you wish
		for node in aprtNodes:
			print "aprt=",node.getName()," pos =",node.getPosition()

		print "===== Aperture Nodes Added ======="

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

		#------ emittances are normalized - transverse by gamma*beta and long. by gamma**3*beta 

		(alphaX,betaX,emittX) = (-1.9899, 0.19636, 0.160372)   # Transverse emittance: normalized RMS value, mm-mrad; beta: mm/mrad   
		(alphaY,betaY,emittY) = ( 1.92893, 0.17778, 0.160362)
		(alphaZ,betaZ,emittZ) = ( -0.015682, 0.670939, 0.222026802)    # Longitudinal emittance: normalized RMS value, mm-mrad; beta: mm/mrad


		alphaZ = -alphaZ

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
		bunch_in = bunch_gen.getBunch(nParticles = 1, distributorClass = GaussDist3D, E_offset = float(E_dis) /1000.0) # E_dis unit: MeV
		#bunch_in = bunch_gen.getBunch(nParticles = 10000, distributorClass = KVDist3D)

		bunch_gen.dumpParmilaFile(bunch_in, phase_init = -0.0, fileName = 	"Distribution_Input_new.txt")
		print "Bunch Generation completed."


		#---- Add a child bunch dump node to the center of the bend
		#dumpBunchAccNode = DumpBunchAccNode("bunch_in_bend.dat","dump_bunch_in_bend")
		#bend1_node = accLattice.getNodeForName("MEBT:DH1")
		#nParts = bend1_node.getnParts()
		#n_dump_ind = int(nParts/2)
		#print "debug bend=",bend1_node.getName()," nParts=",nParts," n_dump_ind=",n_dump_ind
		#bend1_node.addChildNode(dumpBunchAccNode,dumpBunchAccNode.BODY,n_dump_ind,dumpBunchAccNode.BEFORE)

		#set up design
		accLattice.trackDesignBunch(bunch_in)

		print "Design tracking completed."

		#track through the lattice 
		paramsDict = {"old_pos":-1.,"count":0,"pos_step":0.005}
		actionContainer = AccActionsContainer("Test Design Bunch Tracking")

		pos_start = 0.

		twiss_analysis = BunchTwissAnalysis()
		bunch_extrema_cal = BunchExtremaCalculator()

		file_out = open("pyorbit_btf_twiss_sizes_ekin.txt","w")

		s = " Node   position "
		s += "   alphaX betaX emittX  normEmittX"
		s += "   alphaY betaY emittY  normEmittY"
		s += "   alphaZ betaZ emittZ  emittZphiMeV"
		s += "   sizeX sizeY sizeZ_deg"
		s += "   eKin Nparts "
		file_out.write(s+"\n")
		print " N node   position    sizeX  sizeY  sizeZdeg  eKin Nparts "

		file_extrema_out = open("Dispersion_cal_bunch_extrema.txt","w")

		def action_entrance(paramsDict):
			node = paramsDict["node"]
			bunch = paramsDict["bunch"]
			pos = paramsDict["path_length"]
			if(paramsDict["old_pos"] == pos): return
			if(paramsDict["old_pos"] + paramsDict["pos_step"] > pos): return
			paramsDict["old_pos"] = pos
			paramsDict["count"] += 1
			gamma = bunch.getSyncParticle().gamma()
			beta = bunch.getSyncParticle().beta()
			twiss_analysis.analyzeBunch(bunch)
			#---- getEffective(***) returns the raw parameters not corrected for x-energy or y-energy correlation
			(alphaX,betaX,emittX) = (twiss_analysis.getEffectiveAlpha(0),twiss_analysis.getEffectiveBeta(0),twiss_analysis.getEffectiveEmittance(0)*1.0e+6)
			(alphaY,betaY,emittY) = (twiss_analysis.getEffectiveAlpha(1),twiss_analysis.getEffectiveBeta(1),twiss_analysis.getEffectiveEmittance(1)*1.0e+6)
			#---- by default there are corrections for x-energy or y-energy correlation
			#(alphaX,betaX,emittX) = (twiss_analysis.getTwiss(0)[0],twiss_analysis.getTwiss(0)[1],twiss_analysis.getTwiss(0)[3]*1.0e+6)
			#(alphaY,betaY,emittY) = (twiss_analysis.getTwiss(1)[0],twiss_analysis.getTwiss(1)[1],twiss_analysis.getTwiss(1)[3]*1.0e+6)	
			x_rms = math.sqrt(betaX*emittX)
			y_rms = math.sqrt(betaY*emittY)
			z_rms = math.sqrt(twiss_analysis.getTwiss(2)[1]*twiss_analysis.getTwiss(2)[3])*1000.
			z_to_phase_coeff = bunch_gen.getZtoPhaseCoeff(bunch)
			z_rms_deg = z_to_phase_coeff*z_rms/1000.0
			nParts = bunch.getSizeGlobal()	
			(alphaZ,betaZ,emittZ) = (twiss_analysis.getTwiss(2)[0],twiss_analysis.getTwiss(2)[1],twiss_analysis.getTwiss(2)[3]*1.0e+6)		 
			norm_emittX = emittX*gamma*beta
			norm_emittY = emittY*gamma*beta
			#---- phi_de_emittZ will be in [pi*deg*MeV]
			phi_de_emittZ = z_to_phase_coeff*emittZ	
			eKin = bunch.getSyncParticle().kinEnergy()*1.0e+3
			#---- rms sizes in cm
			s = " %35s  %4.5f "%(node.getName(),(pos+pos_start)*100.)
			s += "   %6.4f	%6.4f	%6.4f	%6.4f   "%(alphaX,betaX,emittX*5,norm_emittX)
			s += "   %6.4f	%6.4f	%6.4f  %6.4f   "%(alphaY,betaY,emittY*5,norm_emittY)
			s += "   %6.4f	%6.4f	%6.4f  %6.4f   "%(alphaZ,betaZ / ((1000/360.)*(v_light*beta/frequency)*1.0e+3),emittZ,phi_de_emittZ*5)
			s += "   %5.3f  %5.3f  %5.3f "%(x_rms/10.,y_rms/10.,z_rms_deg)                    #Units of x_rms and y_rms: cm
			s += "  %10.6f   %8d "%(eKin,nParts)
			file_out.write(s +"\n")
			s_prt = " %5d  %35s  %4.5f "%(paramsDict["count"],node.getName(),pos+pos_start)
			s_prt += "  %5.3f  %5.3f   %5.3f "%(x_rms,y_rms,z_rms_deg)
			s_prt += "  %10.6f   %8d "%(eKin,nParts)
			print s_prt
			#------ bunch extrema calculation - values in cm
			(xMin,xMax,yMin,yMax,zMin,zMax) = bunch_extrema_cal.extremaXYZ(bunch)
			s = " %35s  %4.5f  "%(node.getName(),(pos+pos_start)*100.)
			s += "  %12.5e  %12.5e  %12.5e  %12.5e  %12.5e  %12.5e"%(xMin*100.,xMax*100.,yMin*100.,yMax*100.,zMin*100.,zMax*100.)
			file_extrema_out.write(s +"\n")
			#------ flush for safe Cntrl-C cancel
			file_out.flush()
			file_extrema_out.flush()
			
		def action_exit(paramsDict):
			action_entrance(paramsDict)
			
			
		actionContainer.addAction(action_entrance, AccActionsContainer.ENTRANCE)
		actionContainer.addAction(action_exit, AccActionsContainer.EXIT)


		time_start = time.clock()

		accLattice.trackBunch(bunch_in, paramsDict = paramsDict, actionContainer = actionContainer)

		time_exec = time.clock() - time_start
		print "time[sec]=",time_exec

		bunch_in.dumpBunch("Dispersion_cal_Output.txt")

		file_out.close()
		file_extrema_out.close()

		dr=0.01
		# plot = Halo_management(dr)

