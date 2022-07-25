import sys
import time
import math
import numpy as np
import os
import os.path
import random
import errno
import shutil

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QMessageBox
from PyQt5.QtWidgets import QLabel, QMainWindow, QLineEdit, QAction,QTableWidget,QTableWidgetItem,QTextEdit,QFormLayout,QProgressDialog
from PyQt5.QtWidgets import  QInputDialog, QFileDialog,QSizePolicy, QTabWidget,QSpacerItem, QBoxLayout,QSpinBox,QDoubleSpinBox,QComboBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter,NullLocator

from btfsim.sim.forward_tracking_BTF import forward_track
from btfsim.gui.FODOgui import WindowFODO
import btfsim.util.Defaults as default # module for pointing to default data files
import btfsim.lattice.magUtilities as magutil # module for importing + converting magnet currents to gradients

# -- some packages still use Zhouli's source.
sys.path.append('/home/kruisard/Documents/BTF-simulations/BTF-GUI-zhouli/BTF_FODO_Dispersion_GUI/')
from btfsim.util.Twiss_cal_with_threshold import Twiss_cal_threshold
from btfsim.gui.Dispersion_curve_GUI import Window_dispersion



class Lattice_Sim(QtCore.QThread):
	sim = QtCore.pyqtSignal(float,float,float,float) 
	twissCal = QtCore.pyqtSignal(str) 
	Beamsizeplot = QtCore.pyqtSignal()

	def __init__(self, parent, ax,bx,ex,ay,by,ey,az,bz,ez,Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09,pos_cal,current,Part_Num,thresh,scSVR,elpsN):
		super(Lattice_Sim, self).__init__(parent)
		self.ax = ax;  self.bx = bx;  self.ex = ex;  self.ay = ay;  self.by = by;  self.ey = ey;  self.az = az;  self.bz = bz;  self.ez = ez;
		self.Q01 = Q01;   self.Q02 = Q02;    self.Q03 = Q03;   self.Q04 = Q04;   self.Q05 = Q05;    self.Q06 = Q06;  self.Q07 = Q07;   
		self.Q08 = Q08;    self.Q09 = Q09;      self.pos_cal = pos_cal;    self.current = current;    self.Part_Num = Part_Num; self.TR = thresh;
		self.scSVR = scSVR; self.elpsN = elpsN
		
		# -- get default directory structure
		self.defaults = default.getDefaults()
		
		
		# -- make data directories if they don't exist in homedir
		try:
			os.makedirs(self.defaults.defaultdict["OUTDIR"])
			print("Made folder %s in working directory"%self.defaults.defaultdict["OUTDIR"])
		except OSError, e:
			if e.errno != errno.EEXIST:
				raise
			else: print("Folder %s exists in working directory"%self.defaults.defaultdict["OUTDIR"]) 
		try:
			os.makedirs(self.defaults.defaultdict["OUTDIR"])
			print("Made folder %s in working directory"%self.defaults.defaultdict["OUTDIR"])
		except OSError, e:
			if e.errno != errno.EEXIST:
				raise
			else: print("Folder %s exists in working directory"%self.defaults.defaultdict["OUTDIR"])
				
		
				
	#====================================================Main function of a thread, executed by run() function
	def run(self):
		sss = forward_track(self.ax,self.bx,self.ex,self.ay,self.by,self.ey,self.az,self.bz,self.ez, self.Q01,self.Q02,\
			self.Q03,self.Q04,self.Q05,self.Q06,self.Q07,self.Q08,self.Q09, self.pos_cal, self.current, self.Part_Num, self.scSVR, self.elpsN)

		accLattice, bunch_in, paramsDict, actionContainer, ttlength, twiss_analysis, bunch_gen, AccActionsContainer,frequency,v_light= sss.sim()

		file_out = open(self.defaults.defaultdict["OUTDIR"] + "btf_output_twiss.txt","w")
		file_percent_particle_out = open(self.defaults.defaultdict["OUTDIR"] + "btf_output_percent_particles.txt","w")
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
			nn1 = np.floor(bunch.getSize() * 0.90)
			nn2 = np.floor(bunch.getSize() * 0.99)

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

		if (float(self.pos_cal) == 5.120950348):
			output_filename = self.defaults.defaultdict["OUTDIR"] + "Distribution_BTF_End.txt"                       
		else:
			output_filename = self.defaults.defaultdict["OUTDIR"] + "Distribution_Output_BTF.txt"
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
#=====================================================================================================================
#=====================================================================================================================
class Window(QWidget):
 
	def __init__(self):
		super(Window,self).__init__()
		self.freq = 402.5e+6
		self.si_e_charge = 1.6021773e-19 

		# -- get default directory structure
		self.defaults = default.getDefaults()
		
		# -- layout
		layout = QGridLayout()

		fodolayt = WindowFODO()
		dispersionlayt = Window_dispersion()

		self.tabs = QTabWidget()
		self.tab1 = QWidget()   
		self.tab2 = QWidget()
		self.tab3 = QWidget()

		self.tabs.addTab(self.tab1,"BTF_GUI")
		self.tabs.addTab(self.tab2,"FODO_GUI") 
		self.tabs.addTab(self.tab3,"Dispersion_curve_GUI")      

		BTFlayout = QGridLayout()
		FODOlayout = QGridLayout()
		Disp_layout = QGridLayout()

		mainlayout1 = QHBoxLayout()
		mainlayout1.addWidget(self.DisInputGroupBox())
		mainlayout1.addWidget(self.GradientGroupBox())
		mainlayout1.addWidget(self.CurrentGroupBox())
		mainlayout2 = QGridLayout()    #QHBoxLayout()
		mainlayout2.addWidget(self.OneDPlotGroupBox(),0,0)
		mainlayout2.addWidget(self.DisOutputGroupBox(),0,1)
		mainlayout2.addWidget(self.TwoDEmitplotGroupBox(),0,2)
		mainlayout2.setColumnStretch(0,2)
		mainlayout2.setColumnStretch(1,2)
		mainlayout2.setColumnStretch(2,4)
		BTFlayout.addLayout(mainlayout1,0,0)
		BTFlayout.addWidget(self.SizePlotGroupBox(),1,0)
		BTFlayout.addLayout(mainlayout2,2,0)

		FODOlayout.addWidget(fodolayt)
		FODOlayout.setContentsMargins(1,1, 1, 1)

		BTFlayout.setRowMinimumHeight(1,220)
		BTFlayout.setSpacing(4)
		BTFlayout.setContentsMargins(1,1, 1, 1)

		Disp_layout.addWidget(dispersionlayt)
		Disp_layout.setContentsMargins(20,1, 300, 80)

		self.tab1.setLayout(BTFlayout)
		self.tab2.setLayout(FODOlayout)
		self.tab3.setLayout(Disp_layout)

		layout.addWidget(self.tabs)
		layout.setContentsMargins(10,10,10,10)
		self.setLayout(layout)
		self.setWindowTitle("BTF_FODO_Experiment_Gui")
		self.show()

	def DisInputGroupBox(self):
		HgroupBox = QGroupBox("Select Input Distribution")
		HBox = QGridLayout()        #QVBoxLayout() QGridLayout()
		default_filename = self.defaults.defaultdict['BUNCH_IN'].split('/')[-1]
		self.label1 = QLabel("*Read from file('%s')"%(default_filename), self)
		self.button1 = QPushButton('Select',self)
		self.button1.clicked.connect(self.on_click1)
		self.label2 = QLabel('*6D Gauss Twiss parameters (nRMS)', self)
		self.label3 = QLabel('(mm/mrad)', self)
		self.label4 = QLabel('(mm-mrad)', self)
		self.labelx = QLabel('X', self)
		self.labely = QLabel('Y', self)
		self.labelz = QLabel('Z', self)
		self.textbox0 = QLineEdit(self)     #For input file
		self.text_xa = QLineEdit(self)
		self.text_xb = QLineEdit(self)
		self.text_xe = QLineEdit(self)
		self.text_ya = QLineEdit(self)
		self.text_yb = QLineEdit(self)
		self.text_ye = QLineEdit(self)
		self.text_za = QLineEdit(self)
		self.text_zb = QLineEdit(self)
		self.text_ze = QLineEdit(self) 
		self.labela = QLabel('alpha', self)
		self.labelb = QLabel('beta', self)
		self.labele = QLabel('emittance', self)

		select_file = QHBoxLayout()
		spacer1 = QSpacerItem(5, 1, QSizePolicy.Maximum)
		select_file.addItem(spacer1)
		select_file.addWidget(self.textbox0)
		select_file.addWidget(self.button1)
		rlayout = QGridLayout()
		rlayout.addWidget(self.labela,0,0)   #,Qt.AlignLeft)
		rlayout.addWidget(self.labelb,0,1,Qt.AlignCenter)
		rlayout.addWidget(self.labele,0,2,Qt.AlignCenter)
		rlayout.addWidget(self.label3,1,1,Qt.AlignCenter)
		rlayout.addWidget(self.label4,1,2,Qt.AlignCenter)
		rlayout.setVerticalSpacing(0)
		r_x = QHBoxLayout()
		r_x.addWidget(self.text_xa)
		r_x.addWidget(self.text_xb)
		r_x.addWidget(self.text_xe)
		r_y = QHBoxLayout()
		r_y.addWidget(self.text_ya)
		r_y.addWidget(self.text_yb)
		r_y.addWidget(self.text_ye)
		r_z = QHBoxLayout()
		r_z.addWidget(self.text_za)
		r_z.addWidget(self.text_zb)
		r_z.addWidget(self.text_ze)
		
		HBox.addWidget(self.label1,0,0,1,2)
		HBox.addLayout(select_file,1,1)
		HBox.addWidget(self.label2,2,0,1,2)
		HBox.addLayout(rlayout,3,1)
		HBox.addLayout(r_x,4,1)
		HBox.addLayout(r_y,5,1)
		HBox.addLayout(r_z,6,1)
		HBox.addWidget(self.labelx,4,0)
		HBox.addWidget(self.labely,5,0)
		HBox.addWidget(self.labelz,6,0)

		HBox.setSpacing(4)        
		HgroupBox.setLayout(HBox)
		return HgroupBox
   
	def GradientGroupBox(self):
		QGgroupBox = QGroupBox("Set Quads Gradients")
		G_layout = QVBoxLayout()        #QVBoxLayout()QGridLayout()
		default_filename = self.defaults.defaultdict['BTF_QUADS'].split('/')[-1]
		self.label_g0 = QLabel("*Read from file ('%s')"%(default_filename), self)
		self.button_g = QPushButton('Select',self)
		self.button_g.clicked.connect(self.on_click2)
		self.label_g10 = QLabel('*Manual Input (Unit: T)', self)

		self.quads_save = QPushButton('Save_Quads',self)
		self.quads_save.clicked.connect(self.savequads)

		self.label_Q1 = QLabel('QH01')
		self.label_Q2 = QLabel('QV02')
		self.label_Q3 = QLabel('QH03')
		self.label_Q4 = QLabel('QV04')
		self.label_Q5 = QLabel('QH05')
		self.label_Q6 = QLabel('QV06')
		self.label_Q7 = QLabel('QV07')
		self.label_Q8 = QLabel('QH08')
		self.label_Q9 = QLabel('QV09')
		
		self.text_Gfile = QLineEdit(self)         # for gradients input file
		Magstep = 0.002

		select_row =  QHBoxLayout()
		spacer = QSpacerItem(30, 1, QSizePolicy.Maximum)
		select_row.addItem(spacer)
		select_row.addWidget(self.text_Gfile)
		select_row.addWidget(self.button_g) 
		self.sp0 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp3 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp6 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		row036 = QHBoxLayout() 
		row036.addWidget(self.label_Q1)
		row036.addWidget(self.sp0)
		row036.addWidget(self.label_Q4)
		row036.addWidget(self.sp3)
		row036.addWidget(self.label_Q7)
		row036.addWidget(self.sp6)
		self.sp1 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp4 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp7 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		row147 = QHBoxLayout()
		row147.addWidget(self.label_Q2)
		row147.addWidget(self.sp1)
		row147.addWidget(self.label_Q5)
		row147.addWidget(self.sp4)
		row147.addWidget(self.label_Q8)
		row147.addWidget(self.sp7)
		self.sp2 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp5 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		self.sp8 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =-10,singleStep = Magstep)
		row258 = QHBoxLayout()
		row258.addWidget(self.label_Q3)
		row258.addWidget(self.sp2)
		row258.addWidget(self.label_Q6)
		row258.addWidget(self.sp5)
		row258.addWidget(self.label_Q9)
		row258.addWidget(self.sp8)
		row_manual = QHBoxLayout()
		row_manual.addWidget(self.label_g10)
		spacerSQ = QSpacerItem(150, 1, QSizePolicy.Maximum) 
		row_manual.addItem(spacerSQ)
		row_manual.addWidget(self.quads_save)
		spacer = QSpacerItem(200, 1, QSizePolicy.Maximum)
		row_manual.addItem(spacer)

		G_layout.addWidget(self.label_g0)
		G_layout.addLayout(select_row)
		G_layout.addLayout(row_manual)
		G_layout.addLayout(row036)
		G_layout.addLayout(row147)
		G_layout.addLayout(row258) 

		G_layout.setSpacing(2)   
		QGgroupBox.setLayout(G_layout)
		return QGgroupBox
		
	def CurrentGroupBox(self):
		CgroupBox = QGroupBox("Beam Current")
		clayout = QVBoxLayout()  #  QVBoxLayout()
		self.label_I0 = QLabel('I0=', self)
		self.label_I1 = QLabel('Att=', self)
		self.label_I2 = QLabel('mA', self)
		self.label_PN = QLabel('Initial particle number:       ', self)
		self.label_FN = QLabel('Final particle number:       ', self)
		self.text_I0 = QLineEdit(self)                 # for Current setting
		self.text_att = QLineEdit(self)                  # for current attenuation setting
		self.text_PN = QLineEdit(self)                 # Particle number presetting
		self.text_PN.setFixedWidth(60)
		self.text_FN = QLineEdit(self)                  # Final particle number
		self.text_FN.setReadOnly(True) 
		self.button_Ini = QPushButton('Default Settings',self)
		self.button_Ini.clicked.connect(self.on_click3)
		self.button_run = QPushButton('Run',self)
		self.button_run.clicked.connect(self.on_click4)
		self.text_space = QLabel('          ', self)
		self.sc_svr = QLabel('SC_Solvers:', self)
		self.sc_cb = QComboBox()
		self.sc_cb.addItem("Ellipse")
		self.sc_cb.addItem("FFT")
		self.sc_cb.currentIndexChanged.connect(self.selectionchange)
		self.elps = QLabel('Ellipse numbers:', self)
		self.elps_N = QLineEdit(self)

		I_layout = QHBoxLayout()
		I_layout.addWidget(self.label_I0)
		I_layout.addWidget(self.text_I0)
		I_layout.addWidget(self.label_I2)
		spacerI0 = QSpacerItem(100, 1, QSizePolicy.Maximum)
		I_layout.addItem(spacerI0)
		I_layout.addWidget(self.label_I1)
		I_layout.addWidget(self.text_att)

		P_layout = QHBoxLayout()
		P_layout.addWidget(self.label_PN)
		P_layout.addWidget(self.text_PN)

		F_layout = QHBoxLayout()
		F_layout.addWidget(self.label_FN)
		spacerF0 = QSpacerItem(70, 1, QSizePolicy.Maximum)
		F_layout.addItem(spacerF0)
		F_layout.addWidget(self.text_FN)

		SC_layout = QHBoxLayout()
		SC_layout.addWidget(self.sc_svr)
		spacerFt0 = QSpacerItem(60, 1, QSizePolicy.Maximum)
		SC_layout.addItem(spacerFt0)
		SC_layout.addWidget(self.sc_cb)

		elpsN_layout = QHBoxLayout()
		spacereps = QSpacerItem(200, 1, QSizePolicy.Maximum)
		elpsN_layout.addItem(spacereps)
		elpsN_layout.addWidget(self.elps)
		elpsN_layout.addWidget(self.elps_N)

		S_layout = QHBoxLayout()
		S_layout.addWidget(self.button_Ini)

		self.label_o1_1 = QLabel('Output at', self)
		self.label_o1_2 = QLabel('m', self)
		self.text_S = QLineEdit(self)
		out_layout = QHBoxLayout()
		out_layout.addWidget(self.label_o1_1)
		spacerOut1 = QSpacerItem(100, 1, QSizePolicy.Maximum)
		out_layout.addItem(spacerOut1)
		out_layout.addWidget(self.text_S)
		out_layout.addWidget(self.label_o1_2)

		clayout.addLayout(I_layout)
		clayout.addLayout(P_layout)
		clayout.addLayout(F_layout)
		clayout.addLayout(SC_layout)
		clayout.addLayout(elpsN_layout)
		clayout.addLayout(S_layout)
		clayout.addLayout(out_layout)
		clayout.addWidget(self.button_run)

		clayout.setSpacing(3)
		CgroupBox.setLayout(clayout)
		return CgroupBox

	def selectionchange(self):
		if (self.sc_cb.currentText()=="FFT"):
			self.elps_N.setReadOnly(True)
		else:
			self.elps_N.setReadOnly(False)

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
		labellayout.addWidget(self.label_p3,4,0,Qt.AlignLeft)
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
		data_extre = np.genfromtxt(self.defaults.defaultdict["OUTDIR"] + 'btf_output_percent_particles.txt', delimiter='', comments=None)
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
		OPgroupBox = QGroupBox("Output Setting and Results")
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

		self.label_o2_0 = QLabel('Threshold of Noise       ', self)
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
		Btnlayout.addWidget(self.button_o3)
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
					QMessageBox.about(self, "", "Wrong file!!! Header must read % PARTICLE_ATTRIBUTES_CONTROLLERS_NAMES")
				else:
					data_output = np.loadtxt(fileName,skiprows=(14))
					x1 = np.array(data_output[:,0])
					Particle_N = len(x1)

					self.macroSize = np.genfromtxt(fileName, dtype = float,delimiter='', skip_header=3,max_rows=1,comments=None,usecols = 3)
					cur_I = self.macroSize*self.freq*Particle_N*self.si_e_charge*1000 #Unit:mA

					self.text_I0.setText(str('%.2f' %cur_I))
					self.text_att.setText("1.0")
					self.text_PN.setText(str('%6.i' %Particle_N))
					self.text_PN.setReadOnly(True)
					self.text_xa.setText("")
					self.text_xb.setText("")
					self.text_xe.setText("")
					self.text_ya.setText("")
					self.text_yb.setText("")
					self.text_ye.setText("")
					self.text_za.setText("")
					self.text_zb.setText("")
					self.text_ze.setText("")
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
					self.sp0.setValue(float(self.gstate["QH01"]))
					self.sp1.setValue(-float(self.gstate["QV02"])) # need to hard-code polarity because .mstate doesn't save polarity for QH01,QV02
					self.sp2.setValue(float(self.gstate["QH03"]))
					self.sp3.setValue(float(self.gstate["QV04"]))
					self.sp4.setValue(float(self.gstate["QH05"]))
					self.sp5.setValue(float(self.gstate["QV06"]))
					self.sp6.setValue(float(self.gstate["QV07"]))
					self.sp7.setValue(float(self.gstate["QH08"]))
					self.sp8.setValue(float(self.gstate["QV09"]))
					
				elif fileext == 'txt':
					print('Read Gradients File.')
					fileRead = open(fileName,'r+')
					line1 = fileRead.readlines()[0]
					if ( line1.strip() != "BTF_Quads_Gradients (Unit:T)"):
						QMessageBox.about(self, "", "Wrong file!!! header must read: BTF_Quads_Gradients (Unit:T)")
					else:
						Grad = np.genfromtxt(fileName, dtype = float,delimiter='', skip_header=1,comments=None,usecols = (1))
						self.sp0.setValue(float(Grad[0]))   #'%.4f' %alphax))
						self.sp1.setValue(float(Grad[1]))
						self.sp2.setValue(float(Grad[2]))
						self.sp3.setValue(float(Grad[3]))
						self.sp4.setValue(float(Grad[4]))
						self.sp5.setValue(float(Grad[5]))
						self.sp6.setValue(float(Grad[6]))
						self.sp7.setValue(float(Grad[7]))
						self.sp8.setValue(float(Grad[8]))
				else: QMessageBox.about(self, "", "Wrong file!!! Expecting extension .txt or .mstate")
			else:
				print('No clicked.')

	def on_click3(self):   #======================================Defaut setting!
		PATH = self.defaults.defaultdict['TWISS_IN'] # path to default bunch distribution file
		if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
			print "File exists and is readable"

			data_output = np.genfromtxt(PATH, dtype = float,delimiter='', skip_header=2,comments=None,usecols = (1, 2, 3))
			self.text_xa.setText(str(data_output[0,0]))       
			self.text_xb.setText(str(data_output[0,1]))
			self.text_xe.setText(str(data_output[0,2]))
			self.text_ya.setText(str(data_output[1,0]))     
			self.text_yb.setText(str(data_output[1,1]))
			self.text_ye.setText(str(data_output[1,2]))
			self.text_za.setText(str(data_output[2,0]))      
			self.text_zb.setText(str(data_output[2,1]))
			self.text_ze.setText(str(data_output[2,2]))
			self.text_PN.setReadOnly(False)
		else:
			QMessageBox.about(self, "", "'%s' is missing or is not readable!"%(default_bunch_filename))
  
		PATH =  self.defaults.defaultdict['BTF_QUADS']  # path to default quad settings file
		if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
			print "File exists and is readable"

			Field = np.genfromtxt(PATH, dtype = float,delimiter='', skip_header=1,comments=None,usecols = (1))
			self.sp0.setValue(float(Field[0]))   #'%.4f' %alphax))
			self.sp1.setValue(float(Field[1]))
			self.sp2.setValue(float(Field[2]))
			self.sp3.setValue(float(Field[3]))
			self.sp4.setValue(float(Field[4]))            
			self.sp5.setValue(float(Field[5]))
			self.sp6.setValue(float(Field[6]))
			self.sp7.setValue(float(Field[7]))
			self.sp8.setValue(float(Field[8]))
		else:
			QMessageBox.about(self, "", "'%s' is missing or is not readable!"%(PATH))

		self.textbox0.setText("")
		self.text_Gfile.setText("")
		self.text_S.setText("5.120950348")
		self.text_T.setText("0.0")                  #("0.0000001")
		self.text_I0.setText("40.0")
		self.text_att.setText("1.0")
		self.text_PN.setText("20000")
		self.elps_N.setText("5")
		self.text_att.repaint()

	def on_click4(self):       #=================================================Run
		#===================================================Set progressBar
		self.progress = QProgressDialog(self)
		self.progress.setLabelText(" \n \n                                         Lattice Simulation is in process...                                               ")
		self.progress.setStyleSheet('font: bold;')
		self.progress.setMinimumDuration(0)
		self.progress.setWindowModality(Qt.WindowModal)
		self.progress.setRange(0,100) 
		#=====================================================Set progressBar
		if (self.text_T.text() ==""): self.text_T.setText("0.0")
		if (self.text_I0.text() == "" or self.text_att.text() == "" or self.text_PN.text() == ""):
			QMessageBox.about(self, "", "No beam input!")
			self.progress.cancel()
		else:
			if (float(self.sp0.text()) == 0.0 and float(self.sp1.text()) == 0.0 and float(self.sp2.text()) == 0.0 and float(self.sp3.text()) == 0.0 and float(self.sp4.text()) == 0.0 and\
				float(self.sp5.text()) == 0.0 and float(self.sp6.text()) == 0.0 and float(self.sp7.text()) == 0.0 and float(self.sp8.text()) == 0.0):
				QMessageBox.about(self, "", "No Gradients Entry!")
				self.progress.cancel()
			else :
				if (self.elps_N.text() ==""):
						self.elps_N.setText("5")

				if (self.textbox0.text() != "" ):                 #textbox0: Input file for input distribution
					NmacroSize = float(self.text_I0.text())/(self.freq*self.si_e_charge*float(self.text_PN.text())*1000)
					current_I = NmacroSize * float(self.text_att.text())
					f1 = open(self.textbox0.text(),'r+')
					f2 = open(self.defaults.defaultdict["OUTDIR"]+'Input_distributin_external.txt',"w+")
					infos = f1.readlines()
					f1.seek(0,0)
					for line in infos:
						line_new = line.replace(str(self.macroSize),str(current_I))
						f2.write(line_new)
					f1.close()
					f2.close()

					if (self.text_S.text()==""):             #textboxT0: Output at S position
						self.progress.cancel()
						buttonReply = QMessageBox.question(self, '', "Not to set beam stop position?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
						if buttonReply == QMessageBox.No:
							self.text_S.setText("5.120950348") 

							self.thread1 = Lattice_Sim(self,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,self.sp0.text(),self.sp1.text(),self.sp2.text(),self.sp3.text(),self.sp4.text(), \
									self.sp5.text(),self.sp6.text(),self.sp7.text(),self.sp8.text(),self.text_S.text(),current_I,self.text_PN.text(),self.text_T.text(),\
									self.sc_cb.currentText(),self.elps_N.text())
							self.thread1.sim.connect(self.procBar_sim)
							self.thread1.Beamsizeplot.connect(self.plot_beam_size)
							self.thread1.twissCal.connect(self.calTws)
							self.thread1.start()

					else:
						latt_len = float(self.text_S.text())
						if ((0.161< latt_len < 0.281) or (0.347< latt_len < 0.527) or (0.623< latt_len < 0.723) or (0.819< latt_len < 1.626) or (1.722< latt_len < 1.835) or\
						 (1.931< latt_len < 2.9105) or (3.469075174< latt_len < 3.607075174) or (3.703075174< latt_len < 3.853075174) or (3.949075174< latt_len < 4.299075174) or\
						  (4.395075174< latt_len < 4.533075174) or (5.091650348< latt_len <= 5.120950348)):

							self.thread1 = Lattice_Sim(self,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,self.sp0.text(),self.sp1.text(),self.sp2.text(),self.sp3.text(),self.sp4.text(), \
								self.sp5.text(),self.sp6.text(),self.sp7.text(),self.sp8.text(),self.text_S.text(),current_I,self.text_PN.text(),self.text_T.text(),\
								self.sc_cb.currentText(),self.elps_N.text())
							self.thread1.sim.connect(self.procBar_sim)
							self.thread1.Beamsizeplot.connect(self.plot_beam_size)
							self.thread1.twissCal.connect(self.calTws)
							self.thread1.start()
						else:
							self.progress.cancel()
							QMessageBox.about(self, "", "Position is inside magnets, please try other positions! " + "\n" + "Available posotions are: 0.161~0.281, 0.347~0.527,0.623~0.723,\
							 0.819~1.626, 1.722~1.835, 1.931~2.9105, 3.469~3.607, 3.703~3.853, 3.949~4.299, 4.395~4.533, 5.092~5.120950348")
				else:
					if (self.text_xa.text() == "" or self.text_xb.text() == "" or self.text_xe.text() == "" or self.text_ya.text() == "" or self.text_yb.text() == "" or \
					 self.text_ye.text() == "" or  self.text_za.text() == "" or self.text_zb.text() == "" or self.text_ze.text() == ""):
						self.progress.cancel()
						QMessageBox.about(self, "", "No Input Distribution!")                        
					else:
						current_I = float(self.text_I0.text()) * float(self.text_att.text())
						if (self.text_S.text()==""):
							self.progress.cancel()
							buttonReply = QMessageBox.question(self, '', "Not to set beam stop position?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
							if buttonReply == QMessageBox.No:
								self.text_S.setText("5.120950348") 
								# self.text_S.setText('%.5f' %float("5.120950348"))

								self.thread1 = Lattice_Sim(self,self.text_xa.text(),self.text_xb.text(),self.text_xe.text(),self.text_ya.text(),self.text_yb.text(),self.text_ye.text(), \
									self.text_za.text(),self.text_zb.text(),self.text_ze.text(),self.sp0.text(),self.sp1.text(),self.sp2.text(),self.sp3.text(),self.sp4.text(), \
									self.sp5.text(),self.sp6.text(),self.sp7.text(),self.sp8.text(),self.text_S.text(),current_I,self.text_PN.text(),self.text_T.text(),\
									self.sc_cb.currentText(),self.elps_N.text())
								self.thread1.sim.connect(self.procBar_sim)
								self.thread1.Beamsizeplot.connect(self.plot_beam_size)
								self.thread1.twissCal.connect(self.calTws)
								self.thread1.start()

						else:
							latt_len = float(self.text_S.text())
							if ((0.161< latt_len < 0.281) or (0.347< latt_len < 0.527) or (0.623< latt_len < 0.723) or (0.819< latt_len < 1.626) or (1.722< latt_len < 1.835) or\
						 (1.931< latt_len < 2.9105) or (3.469075174< latt_len < 3.607075174) or (3.703075174< latt_len < 3.853075174) or (3.949075174< latt_len < 4.299075174) or\
						  (4.395075174< latt_len < 4.533075174) or (5.091650348< latt_len <= 5.120950348)):                        

								self.thread1 = Lattice_Sim(self,self.text_xa.text(),self.text_xb.text(),self.text_xe.text(),self.text_ya.text(),self.text_yb.text(),self.text_ye.text(), \
									self.text_za.text(),self.text_zb.text(),self.text_ze.text(),self.sp0.text(),self.sp1.text(),self.sp2.text(),self.sp3.text(),self.sp4.text(), \
									self.sp5.text(),self.sp6.text(),self.sp7.text(),self.sp8.text(),self.text_S.text(),current_I,self.text_PN.text(),self.text_T.text(),\
									self.sc_cb.currentText(),self.elps_N.text())
								self.thread1.sim.connect(self.procBar_sim)
								self.thread1.Beamsizeplot.connect(self.plot_beam_size)
								self.thread1.twissCal.connect(self.calTws)
								self.thread1.start()                          
							else:
								self.progress.cancel()
								QMessageBox.about(self, "", "Position is inside magnets, please try other positions! " + "\n" + "Available posotions are: 0.161~0.281, 0.347~0.527, \
									0.623~0.723, 0.819~1.626, 1.722~1.835, 1.931~2.9105, 3.469~3.607, 3.703~3.853, 3.949~4.299, 4.395~4.533, 5.092~5.120950348")

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

		print "The file used for Twiss parameter calculation is:\n ", optfilname

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

			dis_after = open(self.defaults.defaultdict["OUTDIR"]+"Distribution_after_noise_removing_BTF.txt", 'w')
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
				PATH1=self.defaults.defaultdict["OUTDIR"]+'Distribution_Output_BTF.txt'
				PATH2=self.defaults.defaultdict["OUTDIR"]+'Distribution_BTF_End.txt'

				if (os.path.isfile(PATH1) and os.access(PATH1, os.R_OK))==True and (os.path.isfile(PATH2) and os.access(PATH2, os.R_OK))==False:
					print 'Distribution_Output_BTF.txt exists'
					shutil.copy2(PATH1,fileName)
				elif (os.path.isfile(PATH1) and os.access(PATH1, os.R_OK))==False and (os.path.isfile(PATH2) and os.access(PATH2, os.R_OK))==True:
					print 'Distribution_BTF_End.txt exists'
					shutil.copy2(PATH2,fileName)
				elif (os.path.isfile(PATH1) and os.access(PATH1, os.R_OK))==True and (os.path.isfile(PATH2) and os.access(PATH2, os.R_OK))==True:
					print "Two files exists"
					t1 = os.stat(PATH1).st_mtime
					t2 = os.stat(PATH2).st_mtime
					if (t1>t2):
						PATH = PATH1
					else:
						PATH = PATH2
					shutil.copy2(PATH,fileName)
				else: QMessageBox.about(self, "", "User Error: You have to run the simulation first")     

		
	def savequads(self):                   # Save Quads Gradients
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
		if fileName:
			FileforQuads = open(fileName,"w+")
			FileforQuads.write("BTF_Quads_Gradients (Unit:T)" +'\n')
			FileforQuads.write("QH01   " + "%6.5f"%float(self.sp0.text()) +'\n')
			FileforQuads.write("QV02   " + "%6.5f"%float(self.sp1.text()) +'\n')
			FileforQuads.write("QH03   " + "%6.5f"%float(self.sp2.text()) +'\n')
			FileforQuads.write("QV04   " + "%6.5f"%float(self.sp3.text()) +'\n')
			FileforQuads.write("QH05   " + "%6.5f"%float(self.sp4.text()) +'\n')
			FileforQuads.write("QV06   " + "%6.5f"%float(self.sp5.text()) +'\n')
			FileforQuads.write("QV07   " + "%6.5f"%float(self.sp6.text()) +'\n')
			FileforQuads.write("QH08   " + "%6.5f"%float(self.sp7.text()) +'\n')
			FileforQuads.write("QV09   " + "%6.5f"%float(self.sp8.text()) +'\n')
			# QMessageBox.about(self, "", "Save completed!")       



