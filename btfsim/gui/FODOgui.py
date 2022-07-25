import sys
import time
import math
import numpy as np
import os
import os.path
import shutil

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QMessageBox
from PyQt5.QtWidgets import QLabel, QMainWindow, QLineEdit, QAction,QTableWidget,QTableWidgetItem,QTextEdit,QFormLayout, QProgressBar,QComboBox
from PyQt5.QtWidgets import  QInputDialog, QFileDialog,QSizePolicy, QTabWidget,QSpacerItem, QBoxLayout,QSpinBox,QDoubleSpinBox,QProgressDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import random
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter,NullLocator

import btfsim.util.Defaults as default
from btfsim.sim.forward_tracking_FODO import forward_track

sys.path.append('/home/kruisard/Documents/BTF-simulations/BTF-GUI-zhouli/BTF_FODO_Dispersion_GUI/')
from btfsim.util.Twiss_cal_with_threshold import Twiss_cal_threshold   

class Lattice_Sim(QtCore.QThread):
	sim = QtCore.pyqtSignal(float,float,float,float) 
	twissCal = QtCore.pyqtSignal(str) 
	Beamsizeplot = QtCore.pyqtSignal()

	def __init__(self, parent, ax,bx,ex,ay,by,ey,az,bz,ez,Q01,Q02,Q03,Q04,Q05,Q06,pos_cal,current,Part_Num,thresh,scSVR,elpsN):
		super(Lattice_Sim, self).__init__(parent)
		self.ax = ax;  self.bx = bx;  self.ex = ex;  self.ay = ay;  self.by = by;  self.ey = ey;  self.az = az;  self.bz = bz;  self.ez = ez;
		self.Q01 = Q01;   self.Q02 = Q02;    self.Q03 = Q03;   self.Q04 = Q04;   self.Q05 = Q05;    self.Q06 = Q06;
		self.pos_cal = pos_cal;    self.current = current;    self.Part_Num = Part_Num; self.TR = thresh
		self.scSVR = scSVR; self.elpsN = elpsN

		# -- get default directory structure
		self.defaults = default.getDefaults()
		
	#====================================================Main function of a thread, executed by run() function
	def run(self):
		sss = forward_track(self.ax,self.bx,self.ex,self.ay,self.by,self.ey,self.az,self.bz,self.ez, self.Q01,self.Q02,\
			self.Q03,self.Q04,self.Q05,self.Q06, self.pos_cal, self.current, self.Part_Num,self.scSVR, self.elpsN)

		accLattice, bunch_in, paramsDict, actionContainer, ttlength, twiss_analysis, bunch_gen, AccActionsContainer,frequency,v_light= sss.sim()

		file_out = open(self.defaults.defaultdict["OUTDIR"] + "fodo_output_twiss.txt.txt","w")
		file_percent_particle_out = open(self.defaults.defaultdict["OUTDIR"] + "fodo_output_percent_particles.txt","w")
		pos_start = 0.

		def action_entrance(paramsDict):
			node = paramsDict["node"]
			bunch = paramsDict["bunch"]
			pos = paramsDict["path_length"]
			if(paramsDict["old_pos"] == pos):return
			if(paramsDict["old_pos"] + paramsDict["pos_step"] > pos): return
			paramsDict["old_pos"] = pos
			paramsDict["count"] += 1
			gamma = bunch.getSyncParticle().gamma()
			beta = bunch.getSyncParticle().beta()
			twiss_analysis.analyzeBunch(bunch)
				   
			(alphaX,betaX,emittX) = (twiss_analysis.getEffectiveAlpha(0),twiss_analysis.getEffectiveBeta(0),twiss_analysis.getEffectiveEmittance(0)*1.0e+6)
			(alphaY,betaY,emittY) = (twiss_analysis.getEffectiveAlpha(1),twiss_analysis.getEffectiveBeta(1),twiss_analysis.getEffectiveEmittance(1)*1.0e+6)
		
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
			s += "   %6.4f  %6.4f   %6.4f   %6.4f   "%(alphaX,betaX,emittX*5,norm_emittX)
			s += "   %6.4f  %6.4f   %6.4f  %6.4f   "%(alphaY,betaY,emittY*5,norm_emittY)
			s += "   %6.4f  %6.4f   %6.4f  %6.4f   "%(alphaZ,betaZ / ((1000/360.)*(v_light*beta/frequency)*1.0e+3),emittZ,phi_de_emittZ*5)
			s += "   %5.3f  %5.3f  %5.3f "%(x_rms/10.,y_rms/10.,z_rms_deg)                    #Units of x_rms and y_rms: cm
			s += "  %10.6f   %8d "%(eKin,nParts)
			file_out.write(s +"\n")                 
			file_out.flush()

			self.sim.emit(pos+pos_start,ttlength, paramsDict["pos_step"],float(self.TR))
			
			x_i, y_i = [0]*bunch.getSize(), [0]*bunch.getSize()
			for i in range(bunch.getSize()):               
				x_i[i] = bunch.x(i)
				y_i[i] = bunch.y(i)
			r = 100*np.linalg.norm([x_i,y_i], axis=0)

			r.sort()
			nn1 = np.round(bunch.getSize() * 0.90)
			nn2 = np.round(bunch.getSize() * 0.99)

			cor = "  %35s  %4.5f   %12.9e    %12.9e    %12.9e    %12.9e "%(node.getName(),(pos+pos_start)*100.,r[int(nn1)],r[int(nn2)],x_rms/10.0,y_rms/10.0)
			file_percent_particle_out.write(cor +"\n")   #Unit in 'file_percent_particle_out' is cm
			file_percent_particle_out.flush()
			
		def action_exit(paramsDict):
			action_entrance(paramsDict)

		actionContainer.addAction(action_exit, AccActionsContainer.EXIT)

		time_start = time.clock()

		accLattice.trackBunch(bunch_in, paramsDict = paramsDict, actionContainer = actionContainer)

		time_exec = time.clock() - time_start
		print "time[sec]=",time_exec

		output_filename = self.defaults.defaultdict["BUNCHOUTDIR"] + "Distribution_Output_FODO.txt"
		bunch_in.dumpBunch(output_filename)
		
		file_out.close()
		file_percent_particle_out.close()
		self.Beamsizeplot.emit()
		self.twissCal.emit(output_filename)


class Twiss_Cal(QtCore.QThread):
	twiss = QtCore.pyqtSignal(float,float,float,float,float,float,float,float,float,list,list,float,float,float,float,\
		float,float,float,float,float,int,int,list,list,list,list)    
	proc= QtCore.pyqtSignal(int,int,int)
	Plot_1D_lin = QtCore.pyqtSignal(list,list,list,list)    
	Plot_1D_log = QtCore.pyqtSignal(list,list,list,list)      
	
	
	def __init__(self, parent, ft,gh,gv,out_filename):
		super(Twiss_Cal, self).__init__(parent)
		self.threshold = ft
		self.grid_h = gh
		self.grid_v = gv
		self.out_filename = out_filename
		self.defaults = default.getDefaults()

	def run(self):
		grid_h = self.grid_h
		grid_v = self.grid_v
		TR = self.threshold
		filename = self.defaults.defaultdict["BUNCHOUTDIR"]+"Distribution_Output_FODO.txt"
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
					# cord_num.append(i)                        #cord_num: Grid coordinates number including particle number bigger than certain density
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
					# cord_num.append(i)                        #cord_num: Grid coordinates number including particle number bigger than certain density
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
#=====================================================================================================================
#=====================================================================================================================
class WindowFODO(QWidget):
 
	def __init__(self):
		super(WindowFODO,self).__init__()
		self.freq = 402.5e+6
		self.si_e_charge = 1.6021773e-19 
		
		# -- get default directory structure
		self.defaults = default.getDefaults()

		mainlayout = QGridLayout()

		mainlayout1 = QHBoxLayout()        
		mainlayout1.addWidget(self.GradientGroupBox())
		mainlayout1.addWidget(self.CurrentGroupBox())
		mainlayout1.addWidget(self.RunningS())
		mainlayout2 = QGridLayout()    #QHBoxLayout()
		mainlayout2.addWidget(self.OneDPlotGroupBox(),0,0)
		mainlayout2.addWidget(self.DisOutputGroupBox(),0,1)
		mainlayout2.addWidget(self.TwoDEmitplotGroupBox(),0,2)
		mainlayout2.setColumnStretch(0,2)
		mainlayout2.setColumnStretch(1,2)
		mainlayout2.setColumnStretch(2,4)
		mainlayout.addLayout(mainlayout1,0,0)
		mainlayout.addWidget(self.SizePlotGroupBox(),1,0)
		mainlayout.addLayout(mainlayout2,2,0)

		mainlayout.setRowMinimumHeight(1,220)
		mainlayout.setSpacing(4)
		mainlayout.setContentsMargins(1,1, 1, 1)
		self.setLayout(mainlayout)
		self.setWindowTitle("FODO_Experiment_Gui")
		self.show()

	def RunningS(self):
		HgroupBox = QGroupBox("Running  Settings")
		HBox = QVBoxLayout()       #QVBoxLayout() QGridLayout()
	   
		self.button_Ini = QPushButton(' Default  Settings ',self)
		self.button_Ini.clicked.connect(self.on_click3)
		self.button_run = QPushButton('Run',self)
		self.button_run.clicked.connect(self.on_click4)
		self.label_o1_1 = QLabel('Output at', self)
		self.text_S = QLineEdit(self)
		self.text_S.setFixedWidth(90)
		self.label_o1_2 = QLabel('m', self)
		self.sc_svr = QLabel('SC_Solvers:', self)
		self.sc_cb = QComboBox()
		self.sc_cb.addItem("Ellipse")
		self.sc_cb.addItem("FFT")
		self.sc_cb.currentIndexChanged.connect(self.selectionchange)
		self.elps = QLabel('Ellipse numbres:', self)
		self.elps_N = QLineEdit(self)

		SC_layout = QHBoxLayout()
		SC_layout.addWidget(self.sc_svr)
		spacerFt0 = QSpacerItem(60, 1, QSizePolicy.Maximum)
		SC_layout.addItem(spacerFt0)
		SC_layout.addWidget(self.sc_cb)

		elpsN_layout = QHBoxLayout()
		spacereps = QSpacerItem(130, 1, QSizePolicy.Maximum)
		elpsN_layout.addItem(spacereps)
		elpsN_layout.addWidget(self.elps)
		elpsN_layout.addWidget(self.elps_N)

		sc_sum_layout = QVBoxLayout()
		sc_sum_layout.addLayout(SC_layout)
		sc_sum_layout.addLayout(elpsN_layout)
		sc_sum_layout.setSpacing(2)

		EndPlayout = QHBoxLayout()
		EndPlayout.addWidget(self.label_o1_1)
		EndPlayout.addWidget(self.text_S)
		EndPlayout.addWidget(self.label_o1_2)
	
		spacerIn0 = QSpacerItem(1, 10, QSizePolicy.Maximum)
		HBox.addItem(spacerIn0)
		HBox.addWidget(self.button_Ini)

		HBox.addLayout(sc_sum_layout)
		# HBox.addLayout(elpsN_layout)
		spacerIn = QSpacerItem(1, 15, QSizePolicy.Maximum)
		HBox.addItem(spacerIn)
		HBox.addLayout(EndPlayout)
		HBox.addWidget(self.button_run)        
		HgroupBox.setLayout(HBox)
		return HgroupBox

	def selectionchange(self):
		if (self.sc_cb.currentText()=="FFT"):
			self.elps_N.setReadOnly(True)
		else:
			self.elps_N.setReadOnly(False)
   
	def GradientGroupBox(self):
		QGgroupBox = QGroupBox("Set Quads Gradients")
		G_layout = QVBoxLayout()        #QVBoxLayout()QGridLayout()
		self.label_g0 = QLabel('*Read from file ("FODO_field_M#.txt")', self)
		self.button_g = QPushButton('Select',self)
		self.button_g.clicked.connect(self.on_click2)
		self.label_g10 = QLabel('*Manual Input (Unit: T)', self)
		self.quads_save = QPushButton('Save_Quads',self)
		self.quads_save.clicked.connect(self.savequads)

		self.label_Q1 = QLabel('QV10')
		self.label_Q2 = QLabel('QH11')
		self.label_Q3 = QLabel('QV12')
		self.label_Q4 = QLabel('QH13')
		self.label_Q5 = QLabel('QH33')
		self.label_Q6 = QLabel('QV34')

		self.text_Gfile = QLineEdit(self)         # for gradients input file
		Magstep = 0.002

		select_row =  QHBoxLayout()
		spacer = QSpacerItem(30, 1, QSizePolicy.Maximum)
		select_row.addItem(spacer)
		select_row.addWidget(self.text_Gfile)
		select_row.addWidget(self.button_g) 

		self.sp0 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp1 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp2 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		row012 = QHBoxLayout() 
		row012.addWidget(self.label_Q1)
		row012.addWidget(self.sp0)
		row012.addWidget(self.label_Q2)
		row012.addWidget(self.sp1)
		row012.addWidget(self.label_Q3)
		row012.addWidget(self.sp2)
		self.sp3 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp4 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp5 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		row345 = QHBoxLayout()
		row345.addWidget(self.label_Q4)
		row345.addWidget(self.sp3)
		row345.addWidget(self.label_Q5)
		row345.addWidget(self.sp4)
		row345.addWidget(self.label_Q6)
		row345.addWidget(self.sp5)

		row_manual = QHBoxLayout()
		spacerSQ = QSpacerItem(190, 1, QSizePolicy.Maximum)        
		row_manual.addWidget(self.label_g10)
		row_manual.addItem(spacerSQ)
		row_manual.addWidget(self.quads_save)
		spacer = QSpacerItem(200, 1, QSizePolicy.Maximum)
		row_manual.addItem(spacer)

		G_layout.addWidget(self.label_g0)
		G_layout.addLayout(select_row)
		G_layout.addLayout(row_manual)
		G_layout.addLayout(row012)
		G_layout.addLayout(row345)

		G_layout.setSpacing(4)   
		QGgroupBox.setLayout(G_layout)
		return QGgroupBox
		
	def CurrentGroupBox(self):
		CgroupBox = QGroupBox("Input Distribution and Beam Current Setting")
		clayout = QVBoxLayout()

		self.label1 = QLabel('*Read from file ("Distribution_BTF_End.txt")', self)
		self.button1 = QPushButton('Select',self)
		self.button1.clicked.connect(self.on_click1)
		self.textbox0 = QLineEdit(self)               # for input distribution file

		self.label_seperate = QLabel('   ---------------------------------------------------', self)

		self.label_I0 = QLabel('I0 =', self)
		self.label_I1 = QLabel('Att = ', self)
		self.label_I2 = QLabel('mA', self)
		self.label_PN = QLabel('Initial particle number:', self)
		self.label_FN = QLabel('Final particle number:', self)
		self.text_I0 = QLineEdit(self)                 # for Current setting
		self.text_att = QLineEdit(self)                  # for current attenuation setting
		self.text_PN = QLineEdit(self)                 # Particle number presetting
		self.text_PN.setReadOnly(True)
		self.text_FN = QLineEdit(self)                  # Final particle number
		self.text_FN.setReadOnly(True) 

		Dis_input = QHBoxLayout()
		spacerI0 = QSpacerItem(30, 1, QSizePolicy.Maximum)
		Dis_input.addItem(spacerI0)
		Dis_input.addWidget(self.textbox0)
		Dis_input.addWidget(self.button1)
		spacerI1 = QSpacerItem(130, 1, QSizePolicy.Maximum)

		CurrentSet = QHBoxLayout()
		CurrentSet.addWidget(self.label_I0)
		CurrentSet.addWidget(self.text_I0)
		CurrentSet.addWidget(self.label_I2)
		spacerC0 = QSpacerItem(60, 1, QSizePolicy.Maximum)
		CurrentSet.addItem(spacerC0)
		CurrentSet.addWidget(self.label_I1)
		CurrentSet.addWidget(self.text_att)
		spacerC1 = QSpacerItem(150, 1, QSizePolicy.Maximum)

		PreParticle = QHBoxLayout()
		PreParticle.addWidget(self.label_PN)        
		spacerPr1 = QSpacerItem(60, 1, QSizePolicy.Maximum)
		PreParticle.addItem(spacerPr1)
		PreParticle.addWidget(self.text_PN )

		FinParticle = QHBoxLayout()
		FinParticle.addWidget(self.label_FN)       
		spacerFi1 = QSpacerItem(64, 1, QSizePolicy.Maximum)
		FinParticle.addItem(spacerFi1)
		FinParticle.addWidget(self.text_FN)
	
		clayout.addWidget(self.label1)
		clayout.addLayout(Dis_input)
		clayout.addWidget(self.label_seperate)
		spacerF = QSpacerItem(1, 10, QSizePolicy.Maximum)
		clayout.addItem(spacerF)
		clayout.addLayout(CurrentSet)
		clayout.addLayout(PreParticle)
		clayout.addLayout(FinParticle)

		clayout.setSpacing(4)
		CgroupBox.setLayout(clayout)
		return CgroupBox

	def SizePlotGroupBox(self):
		SPgroupBox = QGroupBox("Beam Sizes")
		playout = QGridLayout()   #  QHBoxLayout 
		self.label_p0 = QLabel('*x_RMS', self)
		self.label_p0.setStyleSheet('QLabel {color: magenta}')
		self.label_p1 = QLabel('*y_RMS ', self)
		self.label_p1.setStyleSheet('QLabel {color: green}')
		# self.label_p2 = QLabel('*90%', self)
		# self.label_p2.setStyleSheet('QLabel {color: red}')
		self.label_p3 = QLabel('*99%', self)
		self.label_p3.setStyleSheet('QLabel {color: Blue}')
		labellayout = QGridLayout()   #QBoxLayout
		labellayout.addWidget(self.label_p0,1,0)
		labellayout.addWidget(self.label_p1,2,0)
		# labellayout.addWidget(self.label_p2,3,0,Qt.AlignLeft)
		labellayout.addWidget(self.label_p3,3,0,Qt.AlignLeft)
		playout.addLayout(labellayout,0,0)       
		static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
		playout.addWidget(static_canvas,0,1)
		self._static_ax = static_canvas.figure.subplots()
		self._static_ax.set_position([0.07, 0.2, 0.92, 0.7])

		playout.setColumnStretch(0,1)
		playout.setColumnStretch(1,20)

		SPgroupBox.setLayout(playout)
		return SPgroupBox

	def plot_beam_size(self):
		self._static_ax.clear()
		data_extre = np.genfromtxt(self.defaults.defaultdict["OUTDIR"] +'fodo_output_percent_particles.txt', delimiter='', comments=None)
		x_extre1 = map(float, zip(*data_extre)[1])
		# x_extre2 = map(float, zip(*data_extre)[2])
		x_extre3 = map(float, zip(*data_extre)[3])
		x_extre4 = map(float, zip(*data_extre)[4])
		x_extre5 = map(float, zip(*data_extre)[5])
		# self._static_ax.plot(x_extre1,x_extre2,'r-')
		self._static_ax.plot(x_extre1,x_extre3,'b-')
		self._static_ax.plot(x_extre1,x_extre4,'m-')
		self._static_ax.plot(x_extre1,x_extre5,'g-')
		self._static_ax.set_xlabel('Position (cm)')
		self._static_ax.set_ylabel('Size (cm)')
		#self._static_ax.yaxis.set_major_locator(NullLocator())   # Hiding ticks and lable 
		self._static_ax.tick_params(direction='in')
		self._static_ax.grid()        
		self._static_ax.set_title('X,Y Beam Size')
		self._static_ax.figure.canvas.draw()

	def OneDPlotGroupBox(self):
		onegroupBox = QGroupBox("1D Profile")
		olayout = QGridLayout()     #QVBoxLayout()

		tabs = QTabWidget()
		tab1 = QWidget()   
		tab2 = QWidget()
		tabs.setTabPosition(0)

		tabs.addTab(tab1,"lin")
		tabs.addTab(tab2,"log")

		linlayout = QHBoxLayout()
		loglayout = QHBoxLayout()

		lin_canvas = FigureCanvas(Figure(figsize=(5, 3)))
		log_canvas = FigureCanvas(Figure(figsize=(5, 3)))
		linlayout.addWidget(lin_canvas)
		loglayout.addWidget(log_canvas)

		tab1.setLayout(linlayout)
		tab2.setLayout(loglayout)

		olayout.addWidget(tabs)
		linlayout.setContentsMargins(0,0,0,0)
		loglayout.setContentsMargins(0,0,0,0)

		self.OneD_lin = lin_canvas.figure.subplots()
		# self.OneD_lin.set_position([0.15, 0.22, 0.80, 0.70])
		self.OneD_log = log_canvas.figure.subplots()
		# self.OneD_log.set_position([0.15, 0.22, 0.80, 0.70])
		onegroupBox.setLayout(olayout)
		return onegroupBox

	def OneDplot_lin(self,x1,x2,y1,y2):
		self.OneD_lin.clear()
		self.OneD_lin.plot(x1,x2,'r-', lw=1,label='x')
		self.OneD_lin.legend(loc=1)
		self.OneD_lin.plot(y1,y2,'b-',lw=1,label='y')
		self.OneD_lin.legend(loc=1)
		self.OneD_lin.set_xlabel('position (mm)')
		self.OneD_lin.set_ylabel('Amp (a.u)')
		self.OneD_lin.yaxis.set_major_locator(NullLocator())   # Hiding ticks and lable 
		self.OneD_lin.tick_params(direction='in')      # ticks inside 
		self.OneD_lin.figure.canvas.draw()   

	def OneDplot_log(self,x1,x2,y1,y2):
		self.OneD_log.clear()       
		self.OneD_log.semilogy(x1,x2,'r-', lw=1,label='x')
		self.OneD_log.legend(loc=1)
		self.OneD_log.semilogy(y1,y2,'b-',lw=1,label='y')
		self.OneD_log.legend(loc=1)
		self.OneD_log.set_xlabel('position (mm)')
		self.OneD_log.set_ylabel('Amp (a.u)')
		self.OneD_log.yaxis.set_major_locator(NullLocator())   # Hiding ticks and lable 
		self.OneD_log.tick_params(direction='in')      # ticks inside 
		self.OneD_log.figure.canvas.draw() 

	def DisOutputGroupBox(self):
		OPgroupBox = QGroupBox("Output Settings and Results")
		oplayout = QGridLayout()       #QGridLayout()   QVBoxLayout()     
		
		self.button_o1 = QPushButton('TwissCal',self)
		self.button_o1.setStyleSheet('font: 14px; min-width: 4.6em;')
		self.button_o1.clicked.connect(self.calTws_click)
		self.button_o2 = QPushButton('SaveTws',self)
		self.button_o2.setFixedWidth(120)
		self.button_o2.setStyleSheet('font: 14px; min-width: 4.6em;')
		self.button_o2.clicked.connect(self.on_click8)
		
		self.button_o3 = QPushButton('SaveBunchAs',self)
		self.button_o3.setFixedWidth(120)
		self.button_o3.setStyleSheet('font: 12px; min-width: 4.6em;')
		self.button_o3.clicked.connect(self.onclick_SaveBunchAs)
		
		self.label_o2_0 = QLabel('Threshold of Background       ', self)
		self.label_o2_1 = QLabel('Particles = ', self)
		self.label_o2_2 = QLabel(' %', self)
		self.label_o3 = QLabel('(mm/mrad)', self)
		self.label_o4 = QLabel('(mm-mrad)', self)
		self.label_o5 = QLabel('(deg/keV)', self)
		self.label_o6 = QLabel('(deg-keV)', self)
		self.label_a = QLabel('alpha', self)
		self.label_b = QLabel('beta', self)
		self.label_e = QLabel('emittance(rms)', self)
		self.label_ox = QLabel('X', self)
		self.label_oy = QLabel('Y', self)
		self.label_oz = QLabel('Z', self)
		self.text_T = QLineEdit(self)           #for data about twiss calculation threshold 
		self.text_xa_o = QLineEdit(self)
		self.text_xb_o = QLineEdit(self)
		self.text_xe_o = QLineEdit(self)
		self.text_ya_o = QLineEdit(self)
		self.text_yb_o = QLineEdit(self)
		self.text_ye_o = QLineEdit(self)
		self.text_za_o = QLineEdit(self)
		self.text_zb_o = QLineEdit(self)
		self.text_ze_o = QLineEdit(self)
		
		Btnlayout = QVBoxLayout()
		Btnlayout.addWidget(self.button_o2)
		Btnlayout.addWidget(self.button_o1)

		TBPlayout = QVBoxLayout()
		PNlayout = QHBoxLayout()
		PNlayout.addWidget(self.label_o2_1)
		PNlayout.addWidget(self.text_T)
		PNlayout.addWidget(self.label_o2_2)
		spacer0 = QSpacerItem(100, 1, QSizePolicy.Maximum)
		PNlayout.addItem(spacer0)
		spacer1 = QSpacerItem(1, 5, QSizePolicy.Maximum)
		TBPlayout.addItem(spacer1)
		TBPlayout.addWidget(self.label_o2_0)
		TBPlayout.addLayout(PNlayout)
		spacer2 = QSpacerItem(1, 10, QSizePolicy.Maximum)
		TBPlayout.addItem(spacer2)
		TBPlayout.setSpacing(0)

		rowlayout0 = QHBoxLayout()
		rowlayout0.addLayout(TBPlayout)
		rowlayout0.addLayout(Btnlayout)

		rowlayout2 = QGridLayout()
		rowlayout2.addWidget(self.label_a,0,1,Qt.AlignCenter)
		rowlayout2.addWidget(self.label_b,0,4,Qt.AlignCenter)
		rowlayout2.addWidget(self.label_e,0,7,Qt.AlignCenter)
		rowlayout2.addWidget(self.label_o3,1,4,Qt.AlignCenter)
		rowlayout2.addWidget(self.label_o4,1,7,Qt.AlignCenter)
		rowlayout2.setVerticalSpacing(0)
		rowlayoutx = QHBoxLayout()
		rowlayoutx.addWidget(self.text_xa_o)
		rowlayoutx.addWidget(self.text_xb_o)
		rowlayoutx.addWidget(self.text_xe_o)
		rowlayouty = QHBoxLayout()
		rowlayouty.addWidget(self.text_ya_o)
		rowlayouty.addWidget(self.text_yb_o)
		rowlayouty.addWidget(self.text_ye_o)
		rowlayoutz = QHBoxLayout()
		rowlayoutz.addWidget(self.text_za_o)
		rowlayoutz.addWidget(self.text_zb_o)
		rowlayoutz.addWidget(self.text_ze_o)
		rowunitz = QHBoxLayout()
		spacer4 = QSpacerItem(70, 1, QSizePolicy.Maximum)
		rowunitz.addItem(spacer4)
		rowunitz.addWidget(self.label_o5)
		rowunitz.addWidget(self.label_o6)
			   
		oplayout.addLayout(rowlayout0,0,0,1,4)
		oplayout.addLayout(rowlayout2,2,1)
		oplayout.addLayout(rowlayoutx,3,1)    #,1,6) 
		oplayout.addLayout(rowlayouty,4,1)
		oplayout.addLayout(rowunitz,5,1)
		oplayout.addLayout(rowlayoutz,6,1)       
		oplayout.addWidget(self.label_ox,3,0)
		oplayout.addWidget(self.label_oy,4,0)
		oplayout.addWidget(self.label_oz,6,0) 

		oplayout.setSpacing(4)       
		OPgroupBox.setLayout(oplayout)
		return OPgroupBox

	def TwoDEmitplotGroupBox(self):
		onegroupBox = QGroupBox("2D Emittance")
		tlayout = QHBoxLayout()    #QGridLayout()
		TwoDx_canvas = FigureCanvas(Figure(figsize=(5, 3)))
		tlayout.addWidget(TwoDx_canvas)
		TwoDy_canvas = FigureCanvas(Figure(figsize=(5, 3)))
		tlayout.addWidget(TwoDy_canvas)
		self.TwoDx_ax = TwoDx_canvas.figure.subplots()
		self.TwoDx_ax.set_position([0.15, 0.15, 0.80, 0.70])
		self.TwoDy_ax = TwoDy_canvas.figure.subplots()
		self.TwoDy_ax.set_position([0.15, 0.15, 0.80, 0.70])
		onegroupBox.setLayout(tlayout)
		return onegroupBox

	def on_click1(self):  #================================= Read input distribution file
		self.openFileNameDialog1()
		
	def openFileNameDialog1(self): 
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
		if fileName:
			self.textbox0.setText(fileName)
			buttonReply = QMessageBox.question(self, '', "Is the right input file chosen?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
			if buttonReply == QMessageBox.Yes:
				print('Read Input File.')
				fileRead = open(fileName,'r+')
				line1 = fileRead.readlines()[0]  
				if ( line1.strip() != "% PARTICLE_ATTRIBUTES_CONTROLLERS_NAMES"):
					QMessageBox.about(self, "", "Wrong file!!!")
				else:
					data_output = np.loadtxt(fileName,skiprows=(14))
					x1 = np.array(data_output[:,0])
					Particle_N = len(x1)
					
					self.macroSize = np.genfromtxt(fileName, dtype = float,delimiter='', skip_header=3,max_rows=1,comments=None,usecols = 3)
					cur_I = self.macroSize*self.freq*Particle_N*self.si_e_charge*1000    #Unit: mA

					self.text_PN.setText(str('%6.i' %Particle_N))
					self.text_I0.setText(str('%.4f' %cur_I))
					self.text_att.setText("1.0")
					self.text_FN.setText("")
				
			else:
				print('No clicked.')

	def on_click2(self):   #=================================Read gradients file
		self.openFileNameDialog2()

	def openFileNameDialog2(self):  
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
		if fileName:
			self.text_Gfile.setText(fileName)
			print "====================",fileName
			buttonReply = QMessageBox.question(self, '', "Is the right gradients file chosen?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
			if buttonReply == QMessageBox.Yes:
				fileext = fileName.split('.')[-1]
				if fileext == 'mstate':
					cstate = magutil.loadState(fileName)
					mc = magutil.magConvert()
					self.gstate = mc.current2kappa(cstate)
					print(self.gstate)
					self.sp0.setValue(float(self.gstate["QV10"]))
					self.sp1.setValue(float(self.gstate["QH11"]))
					self.sp2.setValue(float(self.gstate["QV12"]))
					self.sp3.setValue(float(self.gstate["QH13"]))
					self.sp4.setValue(float(self.gstate["QH33"]))
					self.sp5.setValue(float(self.gstate["QV34"]))
					
				elif fileext == 'txt':
					print('Read Gradients File.')
					fileRead = open(fileName,'r+')
					line1 = fileRead.readlines()[0]   
					if ( line1.strip() != "FODO_Quads_Gradients (Unit:T)"):
						QMessageBox.about(self, "", "Wrong file!!!")
					else:
						line = np.genfromtxt(fileName, dtype = float,delimiter='', skip_header=1,comments=None,usecols = (1))
						self.sp0.setValue(float(line[0]))   #'%.4f' %alphax))
						self.sp1.setValue(float(line[1]))
						self.sp2.setValue(float(line[2]))
						self.sp3.setValue(float(line[3]))
						self.sp4.setValue(float(line[4]))
						self.sp5.setValue(float(line[5]))
				else: QMessageBox.about(self, "", "Wrong file!!! Expecting extension .txt or .mstate")
			else:
				print('No clicked.')

	def on_click3(self):   #======================================Initializing
		PATH=self.defaults.defaultdict['BUNCHOUTDIR'] + 'Distribution_BTF_End.txt'
		if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
			print "File exists and is readable"

			data_output = np.loadtxt(PATH,skiprows=(14))
			x1 = np.array(data_output[:,0])
			Particle_N = len(x1)
					
			self.macroSize = np.genfromtxt(PATH, dtype = float,delimiter='', skip_header=3,max_rows=1,comments=None,usecols = 3)
			cur_I = self.macroSize*self.freq*Particle_N*self.si_e_charge*1000    #Unit: mA

			self.text_PN.setText(str('%6.i' %Particle_N))
			self.text_I0.setText(str('%.4f' %cur_I))
			self.text_att.setText("1.0")
			self.textbox0.setText(PATH)
		else:
			QMessageBox.about(self, "", "'%s' is missing or is not readable!!! Please run BTF_GUI first!" %PATH)

		PATH=self.defaults.defaultdict['FODO_QUADS']
		if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
			print "File exists and is readable"

			Field = np.genfromtxt(PATH, dtype = float,delimiter='', skip_header=1,comments=None,usecols = (1))
			self.sp0.setValue(float(Field[0]))   #'%.4f' %alphax))
			self.sp1.setValue(float(Field[1]))
			self.sp2.setValue(float(Field[2]))
			self.sp3.setValue(float(Field[3]))
			self.sp4.setValue(float(Field[4]))            
			self.sp5.setValue(float(Field[5]))
			self.text_Gfile.setText(PATH)
			self.text_S.setText("7.500465174")
			self.text_T.setText("0.0")                  #("0.0000001")
			self.elps_N.setText("5")
			self.text_att.repaint()
		else:
			QMessageBox.about(self, "", "'%s' is missing or is not readable!"%(PATH))

	def on_click4(self):       #=================================================Run
		#===================================================Set progressBar
		self.progress = QProgressDialog(self)
		self.progress.setLabelText(" \n \n                                         Lattice Simulation is in process...                                               ")
		self.progress.setStyleSheet('font: bold;')
		self.progress.setMinimumDuration(0)
		self.progress.setWindowModality(Qt.WindowModal)
		self.progress.setRange(0,100) 
		#=====================================================Set progressBar
		#=====================================================
		if (self.text_T.text() ==""): self.text_T.setText("0.0")
		if (self.textbox0.text() == "" ):
			QMessageBox.about(self, "", "No Input Distribution!")
			self.progress.cancel()
		else:
			self.progress.cancel()

			NmacroSize = float(self.text_I0.text())/(self.freq*self.si_e_charge*float(self.text_PN.text())*1000)
			current_I = NmacroSize * float(self.text_att.text())
			f1 = open(self.textbox0.text(),'r+')
			f2 = open(self.defaults.defaultdict["BUNCHOUTDIR"]+'Input_distributin_external.txt',"w+")
			infos = f1.readlines()
			f1.seek(0,0)
			for line in infos:
				line_new = line.replace(str(self.macroSize),str(current_I))
				f2.write(line_new)
			f1.close()
			f2.close()
			
			if (float(self.sp0.text()) == 0.0 and float(self.sp1.text()) == 0.0 and float(self.sp2.text()) == 0.0 and float(self.sp3.text()) == 0.0 and float(self.sp4.text()) == 0.0 and\
				float(self.sp5.text()) == 0.0):
				QMessageBox.about(self, "", "No Gradients Entry!")
			else :
				if (self.elps_N.text() ==""):
						self.elps_N.setText("5")

				if (self.text_S.text()==""):             #textboxT0: Output at S position
					buttonReply = QMessageBox.question(self, '', "Not to set beam stop position?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)                    
					if buttonReply == QMessageBox.No:
						self.text_S.setText("7.500465174")

						self.thread1 = Lattice_Sim(self,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,self.sp0.text(),self.sp1.text(),self.sp2.text(),self.sp3.text(),self.sp4.text(), \
							self.sp5.text(),self.text_S.text(),current_I,self.text_PN.text(),self.text_T.text(),self.sc_cb.currentText(),self.elps_N.text())
						self.thread1.sim.connect(self.procBar_sim)
						self.thread1.Beamsizeplot.connect(self.plot_beam_size)
						self.thread1.twissCal.connect(self.calTws)
						self.thread1.start()

				else:
					latt_len = float(self.text_S.text())
					if ((1.06039< latt_len < 1.22239) or (5.04139< latt_len < 5.26139) or (5.40739< latt_len < 5.46739) or (5.61339< latt_len < 6.94189) or (latt_len ==7.500465174) ):

						self.thread1 = Lattice_Sim(self,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,self.sp0.text(),self.sp1.text(),self.sp2.text(),self.sp3.text(),self.sp4.text(), \
								self.sp5.text(),self.text_S.text(),current_I,self.text_PN.text(),self.text_T.text(),self.sc_cb.currentText(),self.elps_N.text())
						self.thread1.sim.connect(self.procBar_sim)
						self.thread1.Beamsizeplot.connect(self.plot_beam_size)
						self.thread1.twissCal.connect(self.calTws)
						self.thread1.start() 

					else:
						QMessageBox.about(self, "", "Position is inside magnets or in FODO lines, please try other positions! " + "\n" + "Available posotions are: \
							1.06039~1.22239, 5.04139~5.26139,5.40739~5.46739, 5.61339~6.94189")

	def procBar_sim(self,pos,ttL,step,Tr):
		if ((ttL-pos) > step):
			self.progress.setValue(pos /ttL*100)
			self.progress.setCancelButtonText(str('%.4f' %pos))
		else:
			if (Tr ==0):
				self.progress.setValue(100)
				QMessageBox.information(self,"Indicator","Finished!")
			else:
				self.progress.setValue(100)

	def calTws_click(self):
		PATH=self.defaults.defaultdict["BUNCHOUTDIR"]+'Distribution_Output_FODO.txt'
		if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
			print "File exists and is readable"
			self.calTws(PATH)
		else:
			QMessageBox.about(self, "", "No distribution files!")

	def calTws(self,optfilname):
		self.progress = QProgressDialog(self)
		self.progress.setLabelText(" Calculating of Twiss Parameters is in process...")
		self.progress.resize(500,100)
		self.progress.forceShow()     
		self.progress.setWindowModality(Qt.WindowModal)
		self.progress.setMinimum(0)
		self.progress.setMaximum(100)
		self.progress.setAutoClose(True) 

		if (self.text_T.text()==""):
			threshold= 0.0
			self.text_T.setText("0.0")
			self.text_T.repaint()
		else:
			threshold= float(self.text_T.text())/100

		grid_h = 30
		grid_v = 30
		self.thresh = threshold

		self.thread2 = Twiss_Cal(self,threshold,grid_h,grid_v,optfilname)
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
			self.progress.setValue(val /(2*float(Num_t))*100) 
			self.progress.setCancelButtonText("Running")
		else:
			if (n == 1):
				if ((Num_t-val) >20):
					self.progress.setValue((val+float(Num_t)) /(2*float(Num_t))*100)
					self.progress.setCancelButtonText("Running")
				else:
					self.progress.setValue(100)
					QMessageBox.information(self,"Indicator","Finished!")


	def twisscal(self,ax,bx,ex,ay,by,ey,az,bz,ez,new_index_counter,new_indey_counter,\
			xmin, xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax,Part_TN,grid_h,grid_v,xn,xpn,yn,ypn):
		self.text_xa_o.setText(str('%.4f' %ax))
		self.text_xb_o.setText(str('%.4f' %bx))
		self.text_xe_o.setText(str('%.4f' %ex))
		self.text_ya_o.setText(str('%.4f' %ay))
		self.text_yb_o.setText(str('%.4f' %by))
		self.text_ye_o.setText(str('%.4f' %ey))
		self.text_za_o.setText(str('%.4f' %az))
		self.text_zb_o.setText(str('%.4f' %bz))
		self.text_ze_o.setText(str('%.4f' %ez))
		self.text_FN.setText(str('%6.i' %Part_TN))

		self.xn = xn
		self.xpn = xpn
		self.yn = yn
		self.ypn = ypn

		self.TwoDx_ax.clear()
		self.TwoDy_ax.clear()

		outputx = np.reshape(new_index_counter, (grid_h,grid_v))
		self.TwoDx_ax.imshow(outputx,interpolation='lanczos',cmap='bone',origin='lower',aspect = 'auto', extent=[xmin, xmax, xpmin, xpmax],vmax=abs(outputx).max(), vmin=-abs(outputx).max())
		self.TwoDx_ax.set_title('x-xp (mm-mrad)')
		self.TwoDx_ax.figure.canvas.draw()

		outputy = np.reshape(new_indey_counter, (grid_h,grid_v))
		self.TwoDy_ax.imshow(outputy,interpolation='lanczos',cmap='bone',origin='lower',aspect = 'auto', extent=[ymin,ymax,ypmin,ypmax],vmax=abs(outputy).max(), vmin=-abs(outputy).max())
		self.TwoDy_ax.set_title('y-yp (mm-mrad)')
		self.TwoDy_ax.figure.canvas.draw()

		self.text_FN.repaint() 
	 
	def on_click8(self):              #============================================ Save data
		if (self.text_xa_o.text()==""):
			QMessageBox.about(self, "", "No Data!")
		else:
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
			if fileName:
				file_extrema_out = open(fileName,"w")
				file_extrema_out.write("     alpha    beta    emittance" + "\n")
				file_extrema_out.write("            (mm/mrad) (mm-mrad)" + "\n")
				file_extrema_out.write("x   " + "%6.4f   %6.4f   %6.4f"% (float(self.text_xa_o.text()),float(self.text_xb_o.text()),float(self.text_xe_o.text())) + '\n')
				file_extrema_out.write("y   " + "%6.4f   %6.4f   %6.4f"% (float(self.text_ya_o.text()),float(self.text_yb_o.text()),float(self.text_ye_o.text())) + '\n')
				file_extrema_out.write("            (deg/keV) (deg-keV)" + "\n")
				file_extrema_out.write("z   " + "%6.4f   %6.4f   %6.4f"% (float(self.text_za_o.text()),float(self.text_zb_o.text()),float(self.text_ze_o.text())) + '\n')
				# QMessageBox.about(self, "", "Save completed!")  

			dis_after = open (self.defaults.defaultdict["BUNCHOUTDIR"]+"Distribution_after_noise_removing_FODO.txt", 'w')
			dis_after.write("x(mm)  xp(mard)   y(mm)   yp(mrad)" + '\n')
			np.savetxt(dis_after,zip(self.xn,self.xpn,self.yn,self.ypn),fmt ="%.8f")

	def onclick_SaveBunchAs(self):              #============================================ Save bunch data under desired file name
		# This function saves current output bunch data (at designated endpoint)
		# It works by copying auto-generated bunch file from default name/location into specified name/location
		if (self.text_xa_o.text()==""):
			QMessageBox.about(self, "", "No Data!")
		else:
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
			if fileName:
				PATH=self.defaults.defaultdict["BUNCHOUTDIR"]+'Distribution_Output_FODO.txt'
				if (os.path.isfile(PATH) and os.access(PATH, os.R_OK))==True:
					shutil.copy2(PATH,fileName)
				else: QMessageBox.about(self, "", "User Error: You have to run the simulation first")    
	
	def savequads(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
		if fileName:
			FileforQuads = open(fileName,"w+")
			FileforQuads.write("FODO_Quads_Gradients (Unit:T)" +'\n')
			FileforQuads.write("QV10   " + "%6.5f"%float(self.sp0.text()) +'\n')
			FileforQuads.write("QH11   " + "%6.5f"%float(self.sp1.text()) +'\n')
			FileforQuads.write("QV12   " + "%6.5f"%float(self.sp2.text()) +'\n')
			FileforQuads.write("QH13   " + "%6.5f"%float(self.sp3.text()) +'\n')
			FileforQuads.write("QH33   " + "%6.5f"%float(self.sp4.text()) +'\n')
			FileforQuads.write("QV34   " + "%6.5f"%float(self.sp5.text()) +'\n')
			# QMessageBox.about(self, "", "Save completed!")

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ttt = WindowFODO()
	ttt.show()
	sys.exit(app.exec_())


