import sys
import os
import numpy as np
import time
import shutil
from PyQt5.QtWidgets import	QApplication, QFileDialog, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import json # only used for pretty printing of ordered dictionary

# -- local
#sys.path.append(os.path.abspath('../../'))
import guidesign
import btfsim.lattice.magUtilities as magutil
import btfsim.bunch.bunchUtilities as bunchutil
import btfsim.util.Defaults as default
import btfsim.util.Utilities as util
from btfsim.sim.simulation_main import simBTF
import btfsim.plot.Plotter as plotter
import orbit.utils.consts as consts


###############################################################################
## Threaded Classes
###############################################################################

class SimThread(QThread):
	tsignal = pyqtSignal(float)
	inbunchsignal = pyqtSignal(int,float,dict,dict,dict)
	pltbunchsignal = pyqtSignal(object)
	outbunchsignal = pyqtSignal(int,float,dict,dict,dict)
	
	# -- get default directories
	defaults = default.getDefaults()
	
	def __init__(self,start=0.,stop=100.,bunchArgs={},solverArgs={},latticeArgs={},aperArgs={},reverseFlag=False):
		super(SimThread,self).__init__()
		
		self.startloc = start
		self.stoploc = stop
		self.reverseFlag=reverseFlag
		
		# -- init simulation inside pyQt thread
		self.sim = simBTF(outdir = self.defaults.outdir)
		
		# -- assign parameters to simulation
		self.sim.initLattice(**latticeArgs)
		self.sim.initApertures(**aperArgs)
		self.sim.initSCnodes(**solverArgs)
		self.sim.initBunch(**bunchArgs)
		
		self.beamline = latticeArgs["beamline"]
		self.max_sim_len = self.sim.lat.accLattice.getLength()
		
		# -- if empty run simulation to end
		if self.stoploc=="MEBT:":
			self.stoploc = self.max_sim_len
			
		# -- make name for stop location
		if type(self.stoploc)==float:
			if np.round(self.stoploc,3) >= np.round(self.max_sim_len,3):
				self.stopname = "_end"
			else:
				self.stopname = "_s%.0fcm"%(self.stoploc*100.)
		else: self.stopname = self.stoploc.split(':')[1]
			
		# -- make name for start location
		if type(self.startloc)==float:
			self.startname = "_s%.0fcm"%(self.startloc*100.)
		else: self.startname = self.startloc.split(':')[1]
		
			
		
	def run(self):
		tstart = time.time()
		
		# -- get into about initial bunch and report to main class 
		nparts = self.sim.bunch_in.getSizeGlobal()
		curr = self.sim.bunch_in.macroSize()*self.sim.freq*nparts*consts.charge_electron*1000 #Unit:mA
                bunchcalc = bunchutil.bunchCalculation(self.sim.bunch_in)
                twissx = bunchcalc.Twiss(plane='x',emitnormFlag =1)
                twissy = bunchcalc.Twiss(plane='y',emitnormFlag =1)
                twissz = bunchcalc.Twiss(plane='z',emitnormFlag =1)
                
                self.inbunchsignal.emit(nparts,curr,twissx,twissy,twissz)
		
		if self.reverseFlag: # -- backward run
			# -- name for output bunch
			bunch_out_name = 'bunch_' + self.beamline[-1] + self.startname + '.txt'
			# -- run
			self.sim.reverse(start=self.startloc,stop=self.stoploc, out = bunch_out_name)
			# -- save output
			hist_out_path = self.defaults.outdir + 'reversehist_' + self.beamline[-1] + self.stopname + '.txt'
			self.sim.tracker.writehist(filename=hist_out_path)
		else: # -- forward run
			# -- name for output bunch
			bunch_out_name = 'bunch_' + self.beamline[-1] + self.stopname + '.txt'
			# -- run
			self.sim.run(start=self.startloc,stop=self.stoploc, out = bunch_out_name)
			# -- save output
			hist_out_path = self.defaults.outdir + 'hist_' + self.beamline[-1] + self.stopname + '.txt'
			self.sim.tracker.writehist(filename=hist_out_path)
		
		# -- get info about output bunch to send to GUI window
		nparts_out = self.sim.bunch_track.getSizeGlobal()
		curr_out = self.sim.bunch_track.macroSize()*self.sim.freq*nparts*consts.charge_electron*1000 #Unit:mA
		bunchcalc_out = bunchutil.bunchCalculation(self.sim.bunch_track)
		twissx_out = bunchcalc_out.Twiss(plane='x',emitnormFlag =1)
		twissy_out = bunchcalc_out.Twiss(plane='y',emitnormFlag =1)
		twissz_out = bunchcalc_out.Twiss(plane='z',emitnormFlag =1)
		
		# -- emitters
		tend = time.time()
		ttotal = tend - tstart
		self.tsignal.emit(ttotal) # -- send message on total time of simulation
		self.outbunchsignal.emit(nparts_out,curr_out,twissx_out,twissy_out,twissz_out)
		self.pltbunchsignal.emit(self.sim.bunch_track)
                
# this makes 1D projection plots; I've written a plotter, need to connect to GUI still.
class TwissThread(QThread):
	twiss = pyqtSignal(float,float,float,float,float,float,float,float,float,list,list,float,float,float,float,\
		float,float,float,float,float,int,int,list,list,list,list)    
	proc= pyqtSignal(int,int,int) 
	Plot_1D_lin = pyqtSignal(list,list,list,list)    
	Plot_1D_log = pyqtSignal(list,list,list,list) 

	def __init__(self, parent, ft,gh,gv,out_filename):
		super(TwissThread, self).__init__(parent)
		self.threshold = ft
		self.grid_h = gh
		self.grid_v = gv
		self.out_filename = out_filename

	def run(self):
		grid_h = self.grid_h
		grid_v = self.grid_v
		TR = self.threshold
		filename = self.out_filename
		aaa = Twiss_cal_threshold(filename,TR,grid_h,grid_v)
		dr = 0.3

		xn  = []
		xpn = []
		yn  = []
		ypn = []
		x1,xp1,x1_n,xp1_n, index1,gridx_N,y1,yp1,y1_n,yp1_n,indey1,gridy_N, alphaz,betaz,ez_phi_rms,\
		xmin, xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax,Part_TN= aaa.data_particle()
		#=================================================================================
		#=================================================================================
		if (float(TR) == 0):
			unique, counts = np.unique(index1, return_counts=True)
			counter_dict = dict(zip(unique, counts))
			for i in counter_dict.keys():
				if i < (grid_h * grid_v):
					gridx_N[i] = counter_dict[i]

			xn = x1
			xpn = xp1
			new_index_counter = gridx_N
		else: 
			unique, counts = np.unique(index1, return_counts=True)
			counter_dict = dict(zip(unique, counts))
			for i in counter_dict.keys():
				if i < (grid_h * grid_v):
					gridx_N[i] = counter_dict[i]

			new_index_counter = [0]*grid_v*grid_h      
			for i in range(grid_v*grid_h):
				if (i %20 ==0 ):
					self.proc.emit(i,grid_v*grid_h,0)
				if (float(gridx_N[i])/np.max(gridx_N) >= float(TR) ) :
					new_index_counter[i] = gridx_N[i]
					for j in range(len(x1_n)):
						if (index1[j] == i):
							xn.append(x1[j])
							xpn.append(xp1[j])

			xn = np.array(xn)
			xpn = np.array(xpn)

		xn2_avg=np.mean(xn**2)     # get column mean value
		xpn2_avg=np.mean(xpn**2)
		xxpn_avg=np.mean(xn*xpn)
		ex_rms_n=math.sqrt(xn2_avg*xpn2_avg-xxpn_avg**2)
		alphax= -xxpn_avg/ex_rms_n
		betax = xn2_avg/ex_rms_n

		x_max=np.max(xn)
		x_min=np.min(xn)
		tmpx = np.arange(x_min-0.1, x_max+0.1, dr)
		countx = [0]*len(tmpx)
		countx = np.histogram(xn, tmpx)[0]
		x1 = np.array(zip(*zip(tmpx,countx))[0])
		x2 = np.array(zip(*zip(tmpx,countx))[1])
		#=================================================================================
		#=================================================================================
		if (float(TR) == 0):
			unique, counts = np.unique(indey1, return_counts=True)
			counter_dict = dict(zip(unique, counts))
			for i in counter_dict.keys():
				if i < (grid_h * grid_v):
					gridy_N[i] = counter_dict[i]

			yn = y1
			ypn = yp1
			new_indey_counter = gridy_N
		
		else:
			unique, counts = np.unique(indey1, return_counts=True)
			counter_dict = dict(zip(unique, counts))
			for i in counter_dict.keys():
				if i < (grid_h * grid_v):
					gridy_N[i] = counter_dict[i]
					
			new_indey_counter = [0]*grid_v*grid_h
			for i in range(grid_v*grid_h):
				if (i %20 ==0 ):
					self.proc.emit(i,grid_v*grid_h,1)
				if (float(gridy_N[i])/np.max(gridy_N) >= float(TR) ) :
					new_indey_counter[i] = gridy_N[i]
					for j in range(len(y1_n)):
						if (indey1[j] == i):
							yn.append(y1[j])
							ypn.append(yp1[j])
			
			yn = np.array(yn)
			ypn = np.array(ypn)

		yn2_avg=np.mean(yn**2)     # get column mean value
		ypn2_avg=np.mean(ypn**2)
		yypn_avg=np.mean(yn*ypn)
		ey_rms_n=math.sqrt(yn2_avg*ypn2_avg-yypn_avg**2)
		alphay= -yypn_avg/ey_rms_n
		betay = yn2_avg/ey_rms_n

		y_max=np.max(yn)
		y_min=np.min(yn)
		tmpy = np.arange(y_min-0.1, y_max+0.1, dr)
		county = [0]*len(tmpy)
		county = np.histogram(yn, tmpy)[0]
		y1 = np.array(zip(*zip(tmpy,county))[0])
		y2 = np.array(zip(*zip(tmpy,county))[1])

		self.twiss.emit(alphax,betax,ex_rms_n,alphay,betay,ey_rms_n, alphaz,betaz,ez_phi_rms,new_index_counter,new_indey_counter,\
			xmin, xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax,Part_TN,grid_h,grid_v,list(xn),list(xpn),list(yn),list(ypn))
		self.Plot_1D_lin.emit(list(x1),list(x2),list(y1),list(y2))
		self.Plot_1D_log.emit(list(x1),list(x2),list(y1),list(y2))		
		
###############################################################################
## Main Class for connecting threads to GUI widgets
###############################################################################	
		
class Simulation():
	# -- get default directories
	defaults = default.getDefaults()

	def __init__(self):		
		
		# -- init gui
		self.gui = guidesign.WindowPane(beamline=["MEBT1","MEBT2"])
		
		# -- init simulation outside pyQt thread 
		# (this just to get some default values)
		self.sim = simBTF()
		
		# -- connect function to buttons
		# input + defaults
		self.gui.button1.clicked.connect(self.onClick_read_bunch)
		self.gui.button_g.clicked.connect(self.onClick_read_mstate)
		self.gui.button_Ini.clicked.connect(self.onClick_set_default)
		self.gui.checkbox_amps.clicked.connect(self.onClick_amps)
		self.gui.checkbox_tesla.clicked.connect(self.onClick_tesla)
		
		# run
		self.gui.button_run.clicked.connect(self.onClick_run)
		
		# analysis, plotting, saving
		self.gui.quads_save.clicked.connect(self.onClick_savequads)
		self.gui.button_o1.clicked.connect(self.onClick_calTws)
		self.gui.button_o2.clicked.connect(self.onClick_saveTws)
		self.gui.TwoD_replot_button.clicked.connect(self.onClick_replot_bunch)
		self.gui.save_bunch_button.clicked.connect(self.onClick_SaveBunchAs)
		self.gui.save_hist_button.clicked.connect(self.onClick_SaveHist)
		
	###############################################################################
	## Click functions
	###############################################################################
		
	def onClick_read_bunch(self):#================================= Read input distribution file
		self.openFileNameDialog1()

	def onClick_read_mstate(self):  #=================================Read gradients file
		# 1st make sure correct units box is checked
		self.gui.checkbox_amps.setChecked(True)
		self.onClick_amps()
		# -- load file dialog
		self.openFileNameDialog2()
		
	def onClick_set_default(self): #======================================Defaut setting!
		
		## -- if starting at beginning of BTF, use default Twiss params
		if self.gui.beamline[0] == "MEBT1": 
			PATH = self.defaults.homedir + self.defaults.defaultdict['TWISS_IN'] # path to default bunch parameter file
			if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
				print "Twiss file exists and is readable"

				data_output = np.genfromtxt(PATH, dtype = float,delimiter='',\
											skip_header=2,comments=None,usecols = (1, 2, 3))
				self.gui.text_xa.setText(str(data_output[0,0]))       
				self.gui.text_xb.setText(str(data_output[0,1]))
				self.gui.text_xe.setText(str(data_output[0,2]))
				self.gui.text_ya.setText(str(data_output[1,0]))     
				self.gui.text_yb.setText(str(data_output[1,1]))
				self.gui.text_ye.setText(str(data_output[1,2]))
				self.gui.text_za.setText(str(data_output[2,0]))      
				self.gui.text_zb.setText(str(data_output[2,1]))
				self.gui.text_ze.setText(str(data_output[2,2]))
				self.gui.text_PN.setReadOnly(False)
				self.gui.textbox0.setText("")
			else:
				QMessageBox.about(self.gui, "", "'%s' is missing or is not readable!"%(PATH))
				
		## -- if starting at MEBT2 section, use output bunch from MEBT1
		elif self.gui.beamline[0] == "MEBT2":
			PATH = self.defaults.outdir + 'bunch_MEBT1_end.txt'
			if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
				print("MEBT1 output bunch file exists and is readable")
				print(PATH)
				self.gui.textbox0.setText(PATH)
			else:
				QMessageBox.about(self.gui, "", "'%s' is missing; try running simulation for MEBT1 section!"%(PATH))
				
		## -- if starting at MEBT3 section, use output bunch from MEBT2
		elif self.gui.beamline[0] == "MEBT3":
			PATH = self.defaults.outdir + 'bunch_MEBT2_end.txt'
			if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
				print("MEBT2 output bunch file exists and is readable")
				self.gui.textbox0.setText(PATH)
			else:
				QMessageBox.about(self.gui, "", "'%s' is missing; try running simulation for MEBT2 section!"%(PATH))
				
  		## -- load default quad values
		# 1st make sure correct units box is checked
		self.gui.checkbox_amps.setChecked(True)
		self.onClick_amps()
		PATH =  self.defaults.homedir + self.defaults.defaultdict['BTF_QUADS']  # path to default quad settings file
		if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
			print "Magnet settings file exists and is readable"
			self.gui.text_Gfile.setText(PATH)
			currents = np.genfromtxt(PATH, dtype=['|S4',float],delimiter=',',\
									 comments='#',usecols=(0,1),names='quadname,curr')
			for j in range(len(currents)):
				for i in range(len(self.gui.quadLabels)):
					label = self.gui.quadLabels[i]
					if label == currents["quadname"][j]:
						self.gui.quadBoxes[i].setValue(currents["curr"][j])
		else:
			QMessageBox.about(self.gui, "", "'%s' is missing or is not readable!"%(PATH))

		## -- set default bunch parameters
		self.gui.text_T.setText("0.0")                  #("0.0000001")
		self.gui.text_I0.setText("40.0")
		self.gui.text_att.setText("1.0")
		self.gui.text_PN.setText("20000")
		self.gui.elps_N.setText("1")
		self.gui.text_att.repaint()
		
		## -- defaults for emittance plot grid
		self.gui.TwoD_nx_entry.setText('100')
		self.gui.TwoD_nxp_entry.setText('100')
		self.gui.TwoD_ny_entry.setText('100')
		self.gui.TwoD_nyp_entry.setText('100')

	def onClick_amps(self):
		
		# -- check if tesla box is checked 
		# (aka, check if this is a switch or just a double click
		if self.gui.checkbox_tesla.isChecked():

			# -- uncheck tesla box
			self.gui.checkbox_tesla.setChecked(False)	

			# -- convert all entries to I [A]
			mc = magutil.magConvert()
			for i in range(len(self.gui.quadLabels)):
				quadname = self.gui.quadLabels[i].text()
				quadGL = self.gui.quadBoxes[i].value()
				quadcurr = mc.gl2c(quadname,quadGL)
				self.gui.quadBoxes[i].setValue(quadcurr)
		else:
			pass

		
		
	def onClick_tesla(self):
		
		# -- check if amps box is checked 
		# (aka, check if this is a switch or just a double click
		if self.gui.checkbox_amps.isChecked():

			# -- uncheck tesla box
			self.gui.checkbox_amps.setChecked(False)	

			# -- convert all entries to I [A]
			mc = magutil.magConvert()
			for i in range(len(self.gui.quadLabels)):
				quadname = self.gui.quadLabels[i].text()
				quadcurr = self.gui.quadBoxes[i].value()
				quadGL = mc.c2gl(quadname,quadcurr)
				self.gui.quadBoxes[i].setValue(quadGL)
		else:
			pass
		
		

	
	def enable_run_button(self):
		self.gui.button_run.setEnabled(True)
	
	def onClick_run(self):#=================================================Run
		
		self.gui.button_run.setEnabled(False)
		
		## -- some defaults
		if (self.gui.text_T.text() ==""): self.gui.text_T.setText("0.0") # threshold for twiss calc
		if (self.gui.elps_N.text() ==""): self.gui.elps_N.setText("1") # number of solver ellipses
		
		# -- get stop/start postition
		startStr = self.gui.text_S0.text()
		try:
			start = float(startStr) # -- change number to string
		except ValueError: # -- if non-numerical string is passed, keep as string
			start = "MEBT:" + str(startStr)
			
		stopStr = self.gui.text_S.text()
		try:
			stop = float(stopStr) # -- change number to string
		except ValueError: # -- if non-numerical string is passed, keep as string
			stop = "MEBT:" + str(stopStr)
			
		## -- check if reverse (backwards) run
		reverseFlag = self.gui.checkBox_reverse.isChecked()
		
		## -- determine what type of bunch input is used
		loadBunchFlag = False
		twissFlag = False
		if (self.gui.text_xa.text() and self.gui.text_xb.text() and self.gui.text_xe.text() \
			and self.gui.text_ya.text() and self.gui.text_yb.text() and self.gui.text_ye.text() \
			and  self.gui.text_za.text() and self.gui.text_zb.text() and self.gui.text_ze.text()):
			twissFlag = True
		if self.gui.textbox0.text():
			file = self.gui.textbox0.text()
			loadBunchFlag = True
			if file[-6:] == '.twiss':
				twissFlag = True
			else:
				twissFlag = False

		## -- check that all needed params are defined
		warning_message = ""
		
		# -- beam params set?
		if (self.gui.text_I0.text() == "" or self.gui.text_att.text() == "" or self.gui.text_PN.text() == ""):
			warning_message += "No beam parameters set! \n"
			
		# -- quad gradients set? (at least one should be!)
		ncurrentset = 0
		for widget in self.gui.quadBoxes:
			if float(widget.text()) != 0.0:
				ncurrentset += 1
		if ncurrentset == 0:
			warning_message += "No Gradients Entry! \n"
			
		# -- beam defined by twiss param or bunch file loaded
		if not(twissFlag) and not(loadBunchFlag):
			warning_message += "No beam definition!"
			
		## -- do not execute if missing info; otherwise run
		if warning_message: # if user forgot to fill out info, inform them
			QMessageBox.about(self.gui, "", warning_message)
			self.enable_run_button() # and re-enable run button
		else: # everything looks good? then carry on
		
			print('starting simulation')
			#===================================================Set progressBar
			self.gui.progress = QProgressDialog(self.gui)
			self.gui.progress.setLabelText(" \n \n                      "+\
										   "Simulation in progress..."+\
										   "							")
			self.gui.progress.setStyleSheet('font: bold;')
			self.gui.progress.setMinimumDuration(0)
			self.gui.progress.setWindowModality(Qt.WindowModal)
			self.gui.progress.setRange(0,100) 
			#=====================================================Set progressBar
		
			# -- get dictionary of quad set points from GUI boxes
			magdict = {}
			for i in range(len(self.gui.quadBoxes)):
				magdict[self.gui.quadLabels[i].text()] = self.gui.quadBoxes[i].value()
			if self.gui.checkbox_amps.isChecked(): units='Amps'
			elif self.gui.checkbox_tesla.isChecked(): units='Tesla'

			## -- make dictionaries of parameters to pass to simulation
			# -- lattice
			latticeArgs = {'beamline':self.gui.beamline, 'mdict':magdict, 'units':units}
			
			# -- apertures
			aperArgs = {'d':0.04}
			
			# -- solver 
			solver = self.gui.sc_cb.currentText()
			nelps = self.gui.elps_N.text()
			solverArgs = {'minlen':0.015, 'solver':solver, 'nellipse':nelps, 'gridmult':6}
			
			# -- bunch
			# -- import bunch coordinates 
			if loadBunchFlag and not(twissFlag):
				file = self.gui.textbox0.text()
				bunchArgs = {'gen':"load",'file':file}
			elif loadBunchFlag and twissFlag:
				file = self.gui.textbox0.text()
				fdat = np.genfromtxt(file,comments='#',usecols=(1,2,3),names=True)
				self.gui.text_xa.setText('%.6f'%(fdat['alpha'][0]))
				self.gui.text_ya.setText('%.6f'%(fdat['alpha'][1]))
				self.gui.text_za.setText('%.6f'%(fdat['alpha'][2]))
				self.gui.text_xb.setText('%.6f'%(fdat['beta'][0]))
				self.gui.text_yb.setText('%.6f'%(fdat['beta'][1]))
				self.gui.text_zb.setText('%.6f'%(fdat['beta'][2]))
				self.gui.text_xe.setText('%.6f'%(fdat['emittance'][0]))
				self.gui.text_ye.setText('%.6f'%(fdat['emittance'][1]))
				self.gui.text_ze.setText('%.6f'%(fdat['emittance'][2]))
			# -- or generate Gaussian bunch
			if twissFlag:
				curr = float(self.gui.text_I0.text())*1e-3 # Current should be in A
				nparts = int(self.gui.text_PN.text())
				ax = float(eval(self.gui.text_xa.text()))
				bx = float(eval(self.gui.text_xb.text()))
				ex = float(eval(self.gui.text_xe.text()))
				ay = float(eval(self.gui.text_ya.text()))
				by = float(eval(self.gui.text_yb.text()))
				ey = float(eval(self.gui.text_ye.text()))
				az = float(eval(self.gui.text_za.text()))
				bz = float(eval(self.gui.text_zb.text()))
				ez = float(eval(self.gui.text_ze.text()))
				bunchArgs = {'gen':'twiss','current':curr,'nparts':nparts,\
							 'ax':ax,'bx':bx,'ex':ex,\
							 'ay':ay,'by':by,'ey':ey,\
							 'az':az,'bz':bz,'ez':ez,\
							 'dist':'gaussian','cutoff':3}
			
			## -- create simulation thread
			self.simthread = SimThread(start=start,stop=stop,\
									   latticeArgs=latticeArgs,\
									   aperArgs=aperArgs,\
									   solverArgs=solverArgs,\
									   bunchArgs=bunchArgs,\
									   reverseFlag=reverseFlag)
			
			
			
			# -- make connections:
			self.simthread.tsignal.connect(self.msg_total_time)
			self.simthread.inbunchsignal.connect(self.msg_inbunch_params)
			self.simthread.finished.connect(self.gui.progress.close)
			self.simthread.finished.connect(self.plot_beam_hist)
			self.simthread.pltbunchsignal.connect(self.plot_bunch)
			self.simthread.outbunchsignal.connect(self.msg_outbunch_params)
			self.simthread.finished.connect(self.enable_run_button)
                        
			# -- and GO!
			self.simthread.start()
			print('sim thread running')
			
	
	def onClick_saveTws(self):              #============================================ Save data
		if (self.gui.text_xa_o.text()==""):
			QMessageBox.about(self.gui, "", "No Data!")
		else:
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			fileName, _ = QFileDialog.getSaveFileName(self.gui,"QFileDialog.getSaveFileName()","",\
													  "All Files (*);;Text Files (*.txt)", options=options)
			if fileName:
				file_extrema_out = open(fileName,"w")
				file_extrema_out.write("#    alpha    beta    emittance \n")
				file_extrema_out.write("#           (mm/mrad) (mm-mrad) \n")
				file_extrema_out.write("x   " + "%6.4f   %6.4f   %6.4f \n"% (float(self.gui.text_xa_o.text()),\
																		  float(self.gui.text_xb_o.text()),\
																		  float(self.gui.text_xe_o.text())))
				file_extrema_out.write("y   " + "%6.4f   %6.4f   %6.4f \n"% (float(self.gui.text_ya_o.text()),\
																		  float(self.gui.text_yb_o.text()),\
																		  float(self.gui.text_ye_o.text())))
				file_extrema_out.write("z   " + "%6.4f   %6.4f   %6.4f \n"% (float(self.gui.text_za_o.text()),\
																		  float(self.gui.text_zb_o.text()),\
																		  float(self.gui.text_ze_o.text())))
				# QMessageBox.about(self, "", "Save completed!") 
			
			# -- disable this for now.. not sure why this is here. write distribution coordinates with noise removed?
			# -- (but only x,y,x',y'?)
			#dis_after = open(self.defaults.outdir + "Distribution_after_noise_removing_BTF.txt", 'w')
			#dis_after.write("x(mm)  xp(mard)   y(mm)   yp(mrad)" + '\n')
			#np.savetxt(dis_after,zip(self.xn,self.xpn,self.yn,self.ypn),fmt ="%.8f")    
	
	def onClick_savequads(self): # ================================= save quad gradients to txt file
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getSaveFileName(self.gui,"QFileDialog.getSaveFileName()","",\
												  "All Files (*);;Text Files (*.txt)", options=options)
		if fileName:
			FileforQuads = open(fileName,"w+")
			FileforQuads.write("#BTF_Quads_Current (Unit:A) \n")
			for i in range(len(self.gui.quadLabels)):
				label = self.gui.quadLabels[i].text()
				if self.gui.checkbox_amps.isChecked():
					curr = self.gui.quadBoxes[i].value()
				elif self.gui.checkbox_tesla.isChecked():
					mc = magutil.magConvert()
					curr = mc.gl2c(label,self.gui.quadBoxes[i].value())
				FileforQuads.write("%s,  %6.5f \n"%(label,curr))
				
	def onClick_SaveBunchAs(self):              #============================================ Save bunch data under desired file name
		# This function saves current output bunch data (at designated endpoint)
		# It works by copying auto-generated bunch file from default name/location into specified name/location
		if (self.gui.text_FN.text()==""):
			QMessageBox.about(self.gui, "", "No Data!")
		else:
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			fileName, _ = QFileDialog.getSaveFileName(self.gui,"QFileDialog.getSaveFileName()","",\
													  "All Files (*);;Text Files (*.txt)", options=options)
			if fileName:
                                self.bunch_out.dumpBunch(fileName)
				#shutil.copy2(self.bunch_out_path,fileName)
	
	def onClick_SaveHist(self):
		if (self.gui.text_FN.text()==""): # check if run is complete if final particle number is printed
			QMessageBox.about(self.gui, "", "No Data!")
		else:
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			fileName, _ = QFileDialog.getSaveFileName(self.gui,"QFileDialog.getSaveFileName()","",\
													  "All Files (*);;Text Files (*.txt)", options=options)
			fileName = fileName.split('.')[0] # get filename without extension
			# -- save data to txt file
			self.simthread.sim.tracker.writehist(filename = fileName+'.txt')
			# -- save plot to png
			self.gui._static_fig.savefig(fileName + '.png')
	
	def onClick_calTws(self):
		PATH1=self.defaults.defaultdict["OUTDIR"]+'Distribution_Output_BTF.txt'
		PATH2=self.defaults.defaultdict["OUTDIR"]+'Distribution_BTF_End.txt'

		if (os.path.isfile(PATH1) and os.access(PATH1, os.R_OK))==True and (os.path.isfile(PATH2) and os.access(PATH2, os.R_OK))==False:
			print 'Distribution_Output_BTF.txt exists'
			self.calTws(PATH1)
		else:
			if (os.path.isfile(PATH1) and os.access(PATH1, os.R_OK))==False and (os.path.isfile(PATH2) and os.access(PATH2, os.R_OK))==True:
				print 'Distribution_BTF_End.txt exists'
				self.calTws(PATH2)
			else:
				if (os.path.isfile(PATH1) and os.access(PATH1, os.R_OK))==True and (os.path.isfile(PATH2) and os.access(PATH2, os.R_OK))==True:
					print "Two files exists"
					t1 = os.stat(PATH1).st_mtime
					t2 = os.stat(PATH2).st_mtime
					if (t1>t2):
						outpt_filenm = PATH1
					else:
						outpt_filenm = PATH2
					self.calTws(outpt_filenm)
				else:
					QMessageBox.about(self.gui, "", "No distribution files!") 
					
	###############################################################################
	## Sub functions
	###############################################################################

	def msg_total_time(self,total_time):
		tt_minutes = total_time/60.
		print("Total simulation time: %.3f min."%tt_minutes)
		self.gui.timer_out.setText("%.3f min"%tt_minutes)
		
	def msg_inbunch_params(self,nparts,curr,twissx,twissy,twissz):
		self.gui.text_I0.setText("%.6f"%curr)
		self.gui.text_PN.setText(str(nparts))
                # -- fill in twiss param box
                self.gui.text_xa.setText('%.6f'%twissx['alpha']['value'])
                self.gui.text_xb.setText('%.6f'%twissx['beta']['value'])
                self.gui.text_xe.setText('%.6f'%twissx['emit']['value'])
                self.gui.text_ya.setText('%.6f'%twissy['alpha']['value'])
                self.gui.text_yb.setText('%.6f'%twissy['beta']['value'])
                self.gui.text_ye.setText('%.6f'%twissy['emit']['value'])
                self.gui.text_za.setText('%.6f'%twissz['alpha']['value'])
                self.gui.text_zb.setText('%.6f'%twissz['beta']['value'])
                self.gui.text_ze.setText('%.6f'%twissz['emit']['value'])

	def msg_outbunch_params(self,nparts,curr,twissx,twissy,twissz):

                self.gui.text_FN.setText("%i"%nparts)
                #self.gui.text_FI.setText("%.6f"%curr)

                # -- fill in twiss param box
                self.gui.text_xa_o.setText('%.6f'%twissx['alpha']['value'])
                self.gui.text_xb_o.setText('%.6f'%twissx['beta']['value'])
                self.gui.text_xe_o.setText('%.6f'%twissx['emit']['value'])
                self.gui.text_ya_o.setText('%.6f'%twissy['alpha']['value'])
                self.gui.text_yb_o.setText('%.6f'%twissy['beta']['value'])
                self.gui.text_ye_o.setText('%.6f'%twissy['emit']['value'])
                self.gui.text_za_o.setText('%.6f'%twissz['alpha']['value'])
                self.gui.text_zb_o.setText('%.6f'%twissz['beta']['value'])
                self.gui.text_ze_o.setText('%.6f'%twissz['emit']['value'])

                
	def openFileNameDialog1(self):  
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self.gui,"QFileDialog.getOpenFileName()", "",\
												  "All Files (*);;Python Files (*.py)", options=options)
		if fileName:
			self.gui.textbox0.setText(fileName)
		
	def loadBunchFile(self): # -- right now not used; Bunch is loaded fresh each time simulation is run.
		fileName = self.gui.textbox0.text()
		if fileName:
			print('Read Input File')
			fileRead = open(fileName,'r+')
			line1 = fileRead.readlines()[0]  
			if ( line1.strip() != "% PARTICLE_ATTRIBUTES_CONTROLLERS_NAMES"):
				QMessageBox.about(self.gui, "", "Wrong file!!! Header must read % PARTICLE_ATTRIBUTES_CONTROLLERS_NAMES")
			else:
				data_output = np.loadtxt(fileName,skiprows=(14))
				x1 = np.array(data_output[:,0])
				Particle_N = len(x1)

				self.macroSize = np.genfromtxt(fileName, dtype = float,delimiter='', skip_header=3,max_rows=1,comments=None,usecols = 3)
				cur_I = self.macroSize*self.sim.freq*Particle_N*consts.charge_electron*1000 #Unit:mA

				self.gui.text_I0.setText(str('%.2f' %cur_I))
				self.gui.text_att.setText("1.0")
				self.gui.text_PN.setText(str('%6.i' %Particle_N))
				self.gui.text_PN.setReadOnly(True)
				self.gui.text_xa.setText("")
				self.gui.text_xb.setText("")
				self.gui.text_xe.setText("")
				self.gui.text_ya.setText("")
				self.gui.text_yb.setText("")
				self.gui.text_ye.setText("")
				self.gui.text_za.setText("")
				self.gui.text_zb.setText("")
				self.gui.text_ze.setText("")
				self.gui.text_FN.setText("")
				print('Read Input File complete')
		else:
			QMessageBox.about(self.gui, "", "Please specify path to bunch file!")
				
	def openFileNameDialog2(self):  
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self.gui,"QFileDialog.getOpenFileName()", "",\
												  "All Files (*);;Python Files (*.py)", options=options)
		if fileName:
			self.gui.text_Gfile.setText(fileName)
			print "====================",fileName
			buttonReply = QMessageBox.question(self.gui, '', "Is the right gradients file chosen?",\
											   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
			if buttonReply == QMessageBox.Yes:
				fileext = fileName.split('.')[-1]
				if fileext == 'mstate':
					cstate = magutil.loadQuadReadback(fileName)
					mc = magutil.magConvert()
					print(json.dumps(cstate)) # print out magnets with current settings
					for quadname in cstate:
						for i in range(len(self.gui.quadLabels)):
							label = self.gui.quadLabels[i].text()
							if label == quadname:
								quadcurr = mc.gl2c(quadname,cstate[quadname])
								self.gui.quadBoxes[i].setValue(quadcurr)
					
				elif fileext == 'txt':
					print('Read Gradients File.')
					fileRead = open(fileName,'r+')
					line1 = fileRead.readlines()[0]
					if ( line1.strip() != "#BTF_Quads_Current (Unit:A)"):
						QMessageBox.about(self.gui, "", "Wrong file!!! header must read: #BTF_Quads_Current (Unit:A)")
					else:
						currents = np.genfromtxt(fileName, dtype=['|S4',float],delimiter=',',comments='#',usecols=(0,1),names='quadname,curr')
						for j in range(len(currents)):
							for i in range(len(self.gui.quadLabels)):
								label = self.gui.quadLabels[i].text()
								if label == currents["quadname"][j]:
									print("Found %s"%label)
									self.gui.quadBoxes[i].setValue(currents["curr"][j])
				else: QMessageBox.about(self.gui, "", "Wrong file!!! Expecting extension .txt or .mstate")
			else:
				print('No clicked.')
				
	def procBar_sim(self,pos,ttL,step,Tr):
		if ((ttL-pos) > step):
			self.gui.progress.setValue(pos /ttL*100)
			self.gui.progress.setCancelButtonText(str('%.4f' %pos))
		else:
			if (Tr ==0):
				self.gui.progress.setValue(100)
				QMessageBox.information(self.gui,"Indicator","Finished!")
			else:
				self.gui.progress.setValue(100)
				
	def calTws(self,optfilname):
		self.gui.progress = QProgressDialog(self.gui)
		self.gui.progress.setLabelText(" Calculating of Twiss Parameters is in process...")
		self.gui.progress.resize(500,100)
		self.gui.progress.forceShow()     
		self.gui.progress.setWindowModality(Qt.WindowModal)
		self.gui.progress.setMinimum(0)
		self.gui.progress.setMaximum(100)
		self.gui.progress.setAutoClose(True)

		print "The file used for Twiss parameter calculation is:\n ", optfilname

		if (self.gui.text_T.text()==""):
			threshold= 0.0
			self.gui.text_T.setText("0.0")
			self.gui.text_T.repaint()
		else:
			threshold= float(self.gui.text_T.text())/100

		grid_h = 30
		grid_v = 30
		self.thresh = threshold

		self.thread2 = Twiss_Cal(self.gui,threshold,grid_h,grid_v,optfilname)
		if (threshold ==0):
			self.progress.cancel()
			self.thread2.twiss.connect(self.twisscal) 
		else:
			self.thread2.proc.connect(self.procBar_twiss)
			self.thread2.twiss.connect(self.twisscal) 
		self.thread2.Plot_1D_lin.connect(self.OneDplot_lin)
		self.thread2.Plot_1D_log.connect(self.OneDplot_log)
		self.thread2.start()
		
		
	def procBar_twiss(self, val,Num_t,n):
		if (n ==0):            
			self.gui.progress.setValue(val /(2*float(Num_t))*100) 
			self.gui.progress.setCancelButtonText("Running")
		else:
			if (n == 1):
				if ((Num_t-val) >20):
					self.gui.progress.setValue((val+float(Num_t)) /(2*float(Num_t))*100)
					self.gui.progress.setCancelButtonText("Running")
				else:
					self.gui.progress.setValue(100)
					QMessageBox.information(self,"Indicator","Finished!")

	def twisscal(self,ax,bx,ex,ay,by,ey,az,bz,ez,new_index_counter,new_indey_counter,\
				 xmin, xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax,Part_TN,grid_h,grid_v,xn,xpn,yn,ypn):
		
		self.gui.text_xa_o.setText(str('%.4f' %ax))
		self.gui.text_xb_o.setText(str('%.4f' %bx))
		self.gui.text_xe_o.setText(str('%.4f' %ex))
		self.gui.text_ya_o.setText(str('%.4f' %ay))
		self.gui.text_yb_o.setText(str('%.4f' %by))
		self.gui.text_ye_o.setText(str('%.4f' %ey))
		self.gui.text_za_o.setText(str('%.4f' %az))
		self.gui.text_zb_o.setText(str('%.4f' %bz))
		self.gui.text_ze_o.setText(str('%.4f' %ez))
		self.gui.text_FN.setText(str('%6.i' %Part_TN))

		self.xn = xn
		self.xpn = xpn
		self.yn = yn
		self.ypn = ypn

		self.gui.TwoDx_ax.clear()
		self.gui.TwoDy_ax.clear()

		outputx = np.reshape(new_index_counter, (grid_h,grid_v))
		self.gui.TwoDx_ax.imshow(outputx,interpolation='lanczos',cmap='bone',origin='lower',aspect = 'auto', extent=[xmin, xmax, xpmin, xpmax],vmax=abs(outputx).max(), vmin=-abs(outputx).max())
		self.gui.TwoDx_ax.set_title('x-xp (mm-mrad)')
		self.gui.TwoDx_ax.figure.canvas.draw()

		outputy = np.reshape(new_indey_counter, (grid_h,grid_v))
		self.gui.TwoDy_ax.imshow(outputy,interpolation='lanczos',cmap='bone',origin='lower',aspect = 'auto', extent=[ymin,ymax,ypmin,ypmax],vmax=abs(outputy).max(), vmin=-abs(outputy).max())
		self.gui.TwoDy_ax.set_title('y-yp (mm-mrad)')
		self.gui.TwoDy_ax.figure.canvas.draw()

		self.gui.text_FN.repaint() 
		
	###############################################################################
	## Plotting
	###############################################################################
		
	def plot_bunch(self, bunch_out):
		
		# -- clear axes
		self.gui.TwoDx_ax.clear()
		self.gui.TwoDy_ax.clear()

				# -- check if log scale box is checked
		if self.gui.TwoD_log_toggle.isChecked():
			logFlag = True
		else: logFlag = False

		# -- get grid resolution from GUI input
		nx = int(self.gui.TwoD_nx_entry.text())
		nxp = int(self.gui.TwoD_nxp_entry.text())
		ny = int(self.gui.TwoD_ny_entry.text())
		nyp = int(self.gui.TwoD_nyp_entry.text())
		
				
		## -- get latest file
		#bunchfile = util.get_latest_file(self.defaults.outdir,'bunch_*')
		
		# -- init plotting function
		myplt = plotter.plotBunch(bunch_out,fontsize=8)
		
		# -- x-x' plot
	   	self.gui.TwoDx_ax.set_title('x-xp (mm-mrad)')
		myplt.plot2d('x','xp',show=False,nbins=[nx,nxp],logscale=logFlag,\
					 axis=self.gui.TwoDx_ax, figure = 'gcf', colorbarFlag=False)
		self.gui.TwoDx_ax.figure.canvas.draw()
		
		# -- y-y' plot
		self.gui.TwoDy_ax.set_title('y-yp (mm-mrad)')
		myplt.plot2d('y','yp',show=False,nbins=[ny,nyp],logscale=logFlag,\
					 axis=self.gui.TwoDy_ax, figure = 'gcf', colorbarFlag=False)
		self.gui.TwoDy_ax.figure.canvas.draw()

		# -- not sure what this does?
		self.gui.text_FN.repaint()

		# -- save bunch name (emitted by simulation thread)
		self.bunch_out = bunch_out

	def onClick_replot_bunch(self):
		self.plot_bunch(self.bunch_out)
	
	def plot_beam_hist(self):
		self.gui._static_ax.clear()
		
		myplt = plotter.plotMoments(self.simthread.sim.tracker.hist,fontsize=12)
			
		myplt.plot('s',['xrms','yrms'],show=False,figure='gcf',axis = self.gui._static_ax,linestyle='-')
		myplt.plot('s',['r90','r99'],show=False,figure='gcf',axis = self.gui._static_ax,color='b',linestyle=':')
		self.gui._static_ax.set_xlabel('Position (m)')
		self.gui._static_ax.set_ylabel('Size (cm)')
		
		self.gui._static_ax.tick_params(direction='in')
		self.gui._static_ax.grid()
		self.gui._static_ax.set_xlim([0,self.simthread.max_sim_len])
		self.gui._static_ax.figure.canvas.draw()
		
	
	 

## Main	   
def	main():
	app = QApplication(sys.argv)
	form = Simulation()
	form.gui.show()
	app.exec_()

if __name__	== '__main__':
	main()
