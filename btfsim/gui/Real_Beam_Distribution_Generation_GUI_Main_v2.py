import sys
import time
import math
import numpy as np
import os
import os.path

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

# from Longitudinal_distributins_generation_class import dist_at_RFQ_exit
import random
from itertools import islice
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter,NullLocator
from Distribution_comparing_at_RFQexit_General import DistCompRFQexit
from btfsim.sim.backward_tracking import back_tracking
#from btf_linac_mebt_FODO_design_for_Z_generation import Generation_of_longitudinal_distribution
from Coordinates_centring_class_old import centered_cord
from Coordinates_centring_class_old import GridDist
from Difference import Window_Difference

class clearplot(QtCore.QThread):
	plotclear = QtCore.pyqtSignal()
	def __init__(self,parent = None):
		super(clearplot, self).__init__(parent)

	def run(self):
		self.plotclear.emit()

"""
class Zgeneration(QtCore.QThread):
	#zgene = QtCore.pyqtSignal(float,float,float)
	zdata = QtCore.pyqtSignal(str,float,float,float,float)

	def __init__(self,current,QH01,QH02,QH03,QH04,alpha_z,beta_z,emit_z,distrfilename,field_choice,Part_N):
		super(Zgeneration,self).__init__()
		self.current_forZ = current; self.Q01 = QH01; self.Q02 = QH02; self.Q03 = QH03; self.Q04 = QH04; self.alphaz = alpha_z; \
		self.betaz = beta_z; self.emitz = emit_z; self.distrfilename = distrfilename; self.field_choice = field_choice; self.Part_N = Part_N
	def run(self):
		# -- this line runs back-tracking simulation 
		self.zdata.emit(self.distrfilename,float(self.Q01),float(self.Q02),float(self.Q03),float(self.Q04))
		#zparameter = Generation_of_longitudinal_distribution(self.current_forZ,self.Q01,self.Q02,self.Q03,self.Q04,self.alphaz,self.betaz,self.emitz,self.field_choice,self.Part_N)
		#accLattice, bunch_in, paramsDict, actionContainer, ttlength, twiss_analysis, bunch_gen, AccActionsContainer,frequency,v_light=zparameter.sim()

		#file_out = open("pyorbit_btf_twiss_sizes_ekin.txt","w")
		#pos_start = 0.
		
		#def action_entrance(paramsDict):
			#self.zgene.emit(pos+pos_start,ttlength, paramsDict["pos_step"])
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
			s = " %35s  %4.5f "%(node.getName(),(pos+pos_start)*100.)
			s += "   %6.4f  %6.4f   %6.4f   %6.4f   "%(alphaX,betaX,emittX*5,norm_emittX)
			s += "   %6.4f  %6.4f   %6.4f  %6.4f   "%(alphaY,betaY,emittY*5,norm_emittY)
			s += "   %6.4f  %6.4f   %6.4f  %6.4f   "%(alphaZ,betaZ / ((1000/360.)*(v_light*beta/frequency)*1.0e+3),emittZ,phi_de_emittZ*5)
			s += "   %5.3f  %5.3f  %5.3f "%(x_rms/10.,y_rms/10.,z_rms_deg)                    #Units of x_rms and y_rms: cm
			s += "  %10.6f   %8d "%(eKin,nParts)
			file_out.write(s +"\n")
			file_out.flush()
			
		def action_exit(paramsDict):
			action_entrance(paramsDict)
		actionContainer.addAction(action_exit, AccActionsContainer.EXIT)
		accLattice.trackBunch(bunch_in, paramsDict = paramsDict, actionContainer = actionContainer)
		file_out.close()
		z,dE  = [0]*bunch_in.getSize(),[0]*bunch_in.getSize()
		for i in range(bunch_in.getSize()):
			z[i] = bunch_in.z(i)
			dE[i] = bunch_in.dE(i)
		print self.cname
		bunch_in.dumpBunch("Distribution_Output_new_for_Z.txt")
		"""

class back_tracking_thread(QtCore.QThread):
	backSim = QtCore.pyqtSignal(float,float,float)
	distatRFQ = QtCore.pyqtSignal(list,list,list,list,list,list,str)
	def __init__(self,parent,QH01,QH02,QH03,QH04,filename,field_choice):
		super(back_tracking_thread,self).__init__(parent)
		self.Q01 = QH01; self.Q02 = QH02; self.Q03 = QH03; self.Q04 = QH04; \
		self.filename = filename;self.field_choice=field_choice
	def run(self):
		backTrackSim = back_tracking(self.Q01,self.Q02,self.Q03,self.Q04,self.filename,self.field_choice)
		accLattice, bunch_in, paramsDict, actionContainer, ttlength,  AccActionsContainer, slit_1_Ind , BunchTransformerFunc =backTrackSim.sim()
		# pos_start = 0.
		def action_entrance(paramsDict):
			node = paramsDict["node"]
			bunch = paramsDict["bunch"]
			pos = paramsDict["path_length"]
			if(paramsDict["old_pos"] == pos): return
			if(paramsDict["old_pos"] + paramsDict["pos_step"] > pos): return
			paramsDict["old_pos"] = pos
			paramsDict["count"] += 1
			self.backSim.emit(pos,ttlength, paramsDict["pos_step"])
				
		def action_exit(paramsDict):
			action_entrance(paramsDict)
		actionContainer.addAction(action_entrance, AccActionsContainer.ENTRANCE)
		actionContainer.addAction(action_exit, AccActionsContainer.EXIT)
		# -- trackBunch
		print('Now tracking Bunch')
		accLattice.trackBunch(bunch_in, paramsDict = paramsDict, actionContainer = actionContainer, index_start = slit_1_Ind)
		BunchTransformerFunc(bunch_in)

		x_b,xp_b = [0]*bunch_in.getSize(),[0]*bunch_in.getSize()
		y_b,yp_b = [0]*bunch_in.getSize(),[0]*bunch_in.getSize()
		z_b,dE_b  = [0]*bunch_in.getSize(),[0]*bunch_in.getSize()
		print(bunch_in.getSize())
		for m in range(bunch_in.getSize()):
			x_b[m] = bunch_in.x(m)
			xp_b[m] = bunch_in.xp(m)
			y_b[m] = bunch_in.y(m)
			yp_b[m]= bunch_in.yp(m)
			z_b[m] = bunch_in.z(m)
			dE_b[m] = bunch_in.dE(m)
	  	#---- dump bunch at RFQ exit
		casename = self.filename.split('/')[-1]
		bunch_in.dumpBunch("Bunch_at_RFQ_exit_" + casename)
		self.distatRFQ.emit(x_b,xp_b,y_b,yp_b,z_b,dE_b,casename)


class cordCenterizingThread(QtCore.QThread):
	filtering = QtCore.pyqtSignal(float,float,float)
	PNinGrids = QtCore.pyqtSignal(float,float,list,list,str,list,list,list,list)
	file_final = QtCore.pyqtSignal(list,list,list,list,list,list,str)
	def __init__(self,parent,x,xp,y,yp,TR,GridN,cname,xmin0,xmax0,xpmin0,xpmax0,ymin0,ymax0,ypmin0,ypmax0):
		super(cordCenterizingThread,self).__init__(parent)
		self.x = x; self.xp = xp ; self.y = y; self.yp = yp;self.TR = TR; self.GridN = GridN; self.cname = cname;self.xmin = xmin0;\
		self.xmax = xmax0;self.xpmin = xpmin0;self.xpmax = xpmax0;self.ymin = ymin0;self.ymax = ymax0;self.ypmax = ypmax0;self.ypmin = ypmin0
	def run(self):
		xn = []     
		xpn = []
		yn = []
		ypn = []
		Grid_Dist = GridDist(self.x,self.xp,self.y,self.yp,float(self.GridN),self.xmin,self.xmax,self.xpmin,self.xpmax,self.ymin,self.ymax,self.ypmin,self.ypmax)
		index1,indey1,index_counter1,indey_counter1 = Grid_Dist.DistafterGrid()
	   
		new_index_counter = [0]*int(self.GridN)*int(self.GridN)
		new_indey_counter = [0]*int(self.GridN)*int(self.GridN)
		for i in range(int(self.GridN)*int(self.GridN)):
			if (i % 20 ==0):
				self.filtering.emit(i,int(self.GridN)*int(self.GridN), 0)
			if (float(index_counter1[i])/len(self.x) >= float(self.TR)/100 ) :
				new_index_counter[i] = index_counter1[i]
				for k in range(len(self.x)):               
					if (index1[k] == i):
						xn.append(self.x[k])
						xpn.append(self.xp[k])


		for j in range(int(self.GridN)*int(self.GridN)):
			if (j % 20 ==0):
				self.filtering.emit(j,int(self.GridN)*int(self.GridN), 1)
			if (float(indey_counter1[j])/len(self.y) >= float(self.TR)/100 ) :
				new_indey_counter[j] = indey_counter1[j]
				for h in range(len(self.y)):
					if (indey1[h] == j):
						yn.append(self.y[h])
						ypn.append(self.yp[h])
		# np.savetxt("x" +self.cname, np.array([xn,xpn]).T,fmt = "%12.7e")
		# np.savetxt("y" +self.cname, np.array([yn,ypn]).T,fmt = "%12.7e")
		self.PNinGrids.emit(sum(new_index_counter),sum(new_indey_counter),new_index_counter,new_indey_counter,self.cname,xn,xpn,yn,ypn)

class Window_dispersion(QWidget):
 
	def __init__(self):
		super(Window_dispersion,self).__init__()

		layout = QVBoxLayout()    #QVBoxLayout()   QGridLayout()

		dfrncelayt =  Window_Difference()
		diff_layout = QVBoxLayout()
		mainlayout = QVBoxLayout()

		self.tabs = QTabWidget()
		self.tab1 = QWidget()   
		self.tab2 = QWidget()

		self.tabs.addTab(self.tab1,"General_Comparing")
		self.tabs.addTab(self.tab2,"Difference_Comparing")

		uplayout = QHBoxLayout() 
		uplayout.addWidget(self.DistributionGroupBox())
		uplayout.addWidget(self.ComBox())

		mainlayout.addLayout(uplayout)
		mainlayout.addWidget(self.DistplotGroupBox())
		mainlayout.setSpacing(2)
		mainlayout.setContentsMargins(6,6, 6, 6)

		diff_layout.addWidget(dfrncelayt)
		diff_layout.setContentsMargins(0,0, 0, 0)

		self.tab1.setLayout(mainlayout)
		self.tab2.setLayout(diff_layout)

		layout.addWidget(self.tabs)

		self.setLayout(layout)
		self.setWindowTitle("Beam_Distribution_Generation_GUI")
		self.show()
   
	def DistributionGroupBox(self):
		QGgroupBox = QGroupBox("Distribution Selection")
		G_layout = QGridLayout()     #QVBoxLayout()   
		self.Dis_lbl = QLabel('Distribution Selection(/Cases_Measured/case#.txt)', self)
		self.text_Gfile = QLineEdit(self)
		self.Dis_btn = QPushButton('Select',self)
		self.Dis_btn.clicked.connect(self.on_click1)
		self.DisCSN_lbl = QLabel('Selected distributions are:', self)
		self.Dis1 = QLineEdit(self)
		self.Dis1.setReadOnly(True)
		self.Dis2 = QLineEdit(self)
		self.Dis2.setReadOnly(True)
		self.Dis3 = QLineEdit(self)
		self.Dis3.setReadOnly(True)
		self.Dis4 = QLineEdit(self)
		self.Dis4.setReadOnly(True)
		self.clear_btn = QPushButton('Clear',self)
		self.clear_btn.clicked.connect(self.on_click2)

		self.G_lbl = QLabel('Gradients:', self)

		self.quads_save = QPushButton('Save_Quads',self)
		self.quads_save.setFixedWidth(100)
		self.quads_save.clicked.connect(self.savequads)

		self.Q1 = QLabel('Q01')
		self.Q2 = QLabel('Q02')
		self.Q3 = QLabel('Q03')
		self.Q4 = QLabel('Q04')

		self.label_c1Qs = QLabel('Case1')
		self.label_c2Qs = QLabel('Case2')
		self.label_c3Qs = QLabel('Case3')
		self.label_c4Qs = QLabel('Case4')
		
		Magstep = 0.0005

		self.c1q1 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c1q2 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c1q3 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c1q4 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)

		self.c2q1 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c2q2 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c2q3 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c2q4 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)

		self.c3q1 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c3q2 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c3q3 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c3q4 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)

		self.c4q1 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c4q2 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c4q3 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
		self.c4q4 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)

		select_row = QHBoxLayout()
		select_row.addWidget(self.text_Gfile)
		select_row.addWidget(self.Dis_btn)

		dis_row = QHBoxLayout()
		dis_row.addWidget(self.Dis1)
		dis_row.addWidget(self.Dis2)
		dis_row.addWidget(self.Dis3)
		dis_row.addWidget(self.Dis4)
		dis_row.addWidget(self.clear_btn)

		saverow = QHBoxLayout()
		saverow.addWidget(self.G_lbl)
		saverow.addWidget(self.quads_save)

		label_rows = QHBoxLayout() 
		label_rows.addWidget(self.Q1)
		label_rows.addWidget(self.Q2)
		label_rows.addWidget(self.Q3)
		label_rows.addWidget(self.Q4)

		rowc1 =  QHBoxLayout() 
		rowc1.addWidget(self.c1q1)
		rowc1.addWidget(self.c1q2)
		rowc1.addWidget(self.c1q3)
		rowc1.addWidget(self.c1q4)

		rowc2 = QHBoxLayout() 
		rowc2.addWidget(self.c2q1)
		rowc2.addWidget(self.c2q2)
		rowc2.addWidget(self.c2q3)
		rowc2.addWidget(self.c2q4)

		rowc3 = QHBoxLayout() 
		rowc3.addWidget(self.c3q1)
		rowc3.addWidget(self.c3q2)
		rowc3.addWidget(self.c3q3)
		rowc3.addWidget(self.c3q4)

		rowc4 = QHBoxLayout() 
		rowc4.addWidget(self.c4q1)
		rowc4.addWidget(self.c4q2)
		rowc4.addWidget(self.c4q3)
		rowc4.addWidget(self.c4q4)

		G_layout.addWidget(self.Dis_lbl,0,0,1,2)
		G_layout.addLayout(select_row,1,1)
		G_layout.addWidget(self.DisCSN_lbl,2,0,1,2)
		G_layout.addLayout(dis_row,3,1,1,2)
		G_layout.addLayout(saverow,4,0,1,2)
		G_layout.addLayout(label_rows,5,1)
		G_layout.addLayout(rowc1,6,1)
		G_layout.addLayout(rowc2,7,1)
		G_layout.addLayout(rowc3,8,1)
		G_layout.addLayout(rowc4,9,1)
		G_layout.addWidget(self.label_c1Qs,6,0)
		G_layout.addWidget(self.label_c2Qs,7,0)
		G_layout.addWidget(self.label_c3Qs,8,0)
		G_layout.addWidget(self.label_c4Qs,9,0)

		G_layout.setSpacing(0)   
		QGgroupBox.setLayout(G_layout)
		return QGgroupBox
		
	def RunningGroupBox(self):
		CgroupBox = QGroupBox("Running Settings")
		clayout = QVBoxLayout()  #  QVBoxLayout()   QGridLayout()

		self.button_run = QPushButton('Run',self)
		self.button_run.clicked.connect(self.on_click4)
		PN_select = QLabel('Particle Number Selection:',self)

		self.Grid_N = QLineEdit(self)
		self.label_Grid_N = QLabel('Grid Number',self)
		self.label_Grid_N.setFixedWidth(80)
		self.Noise_TR = QLineEdit(self)
		self.Noise_TR.setToolTip('Relative to total particles number.') 
		self.label_Noise_TR1 = QLabel('Noise Threshold',self)
		self.label_Noise_TR1.setFixedWidth(100)
		self.label_Noise_TR1.setToolTip('Relative to total particles number.') 
		self.label_Noise_TR2 = QLabel('%',self)

		self.label_current1 = QLabel('Current',self)
		self.label_current1.setFixedWidth(60)
		self.current = QLineEdit(self)
		self.label_current2 = QLabel('mA',self)

		self.z_parameters = QLabel('Longitudianl Parameters:',self)
		self.label_a = QLabel('alpha',self)
		self.label_a.setFixedWidth(80)
		self.label_b = QLabel('beta',self)
		self.label_b.setFixedWidth(80)
		self.label_b1 = QLabel('mm/mrad',self)
		self.label_e = QLabel('emittance',self)
		self.label_e1 = QLabel('mm-mrad',self)
		self.label_e.setFixedWidth(80)
		self.a_z = QLineEdit(self)
		self.b_z = QLineEdit(self)
		self.e_z = QLineEdit(self)

		self.label_field = QLabel("Field Type",self)
		self.label_field.setFixedWidth(70)
		self.field = QComboBox()
		self.field.addItem("HardEdge")
		self.field.addItem("RealField")


		layout1 = QHBoxLayout()
		spacerQ1 = QSpacerItem(40, 1, QSizePolicy.Maximum) 
		layout1.addItem(spacerQ1)
		layout1.addWidget(self.label_Noise_TR1)
		layout1.addWidget(self.Noise_TR)
		layout1.addWidget(self.label_Noise_TR2)
		layout1.addWidget(self.label_Grid_N)
		layout1.addWidget(self.Grid_N)

		layout2 = QHBoxLayout()
		# spacerc1 = QSpacerItem(40, 1, QSizePolicy.Maximum) 
		# layout2.addItem(spacerc1)
		layout2.addWidget(self.label_current1)
		layout2.addWidget(self.current)
		layout2.addWidget(self.label_current2)
		spacerc3 = QSpacerItem(304, 1, QSizePolicy.Maximum) 
		layout2.addItem(spacerc3)

		ermslayout = QHBoxLayout()   
		spacere1 = QSpacerItem(70, 1, QSizePolicy.Maximum) 
		ermslayout.addItem(spacere1)
		ermslayout.addWidget(self.label_a)
		ermslayout.addWidget(self.a_z)
		ermslayout.addWidget(self.label_b)
		ermslayout.addWidget(self.b_z)
		ermslayout.addWidget(self.label_b1)
		spacere3 = QSpacerItem(62, 1, QSizePolicy.Maximum) 
		ermslayout.addItem(spacere3)
		
		ezlayout = QHBoxLayout()   
		spacerSQ4 = QSpacerItem(80, 1, QSizePolicy.Maximum) 
		ezlayout.addItem(spacerSQ4)
		ezlayout.addWidget(self.label_e)
		ezlayout.addWidget(self.e_z)
		ezlayout.addWidget(self.label_e1)
		spacerSQ5 = QSpacerItem(243, 1, QSizePolicy.Maximum) 
		ezlayout.addItem(spacerSQ5)
		

		layout3 = QHBoxLayout()
		# spacerSQ3 = QSpacerItem(40, 1, QSizePolicy.Maximum) 
		# layout3.addItem(spacerSQ3)
		layout3.addWidget(self.label_field)
		layout3.addWidget(self.field)
		spacerSQ4 = QSpacerItem(220, 1, QSizePolicy.Maximum) 
		layout3.addItem(spacerSQ4)

		clayout.addWidget(PN_select)
		clayout.addLayout(layout1)
		clayout.addLayout(layout2)
		clayout.addWidget(self.z_parameters)
		clayout.addLayout(ermslayout)
		clayout.addLayout(ezlayout)

		clayout.addLayout(layout3)
		clayout.addWidget(self.button_run)
		clayout.setSpacing(4)
		CgroupBox.setLayout(clayout)
		return CgroupBox

	def resultsBox(self):
		RegroupBox =  QGroupBox("Results")
		relayout = QVBoxLayout()

		self.dist_x_dscrpncy = QLineEdit(self)
		self.dist_y_dscrpncy = QLineEdit(self)
		self.label_x_dscrpncy = QLabel('Coordinates discrepency in x-plane:',self)
		self.label_x_dscrpncy.setFixedWidth(250)
		self.label_y_dscrpncy = QLabel('Coordinates discrepency in y-plane:',self)
		self.label_y_dscrpncy.setFixedWidth(250)
		self.PN_btn = QPushButton('PartNum and Comparing',self)
		self.resultsave = QPushButton('Save Distributions',self)
		self.resultsave.clicked.connect(self.saveresults)

		pslayout = QHBoxLayout()
		pslayout.addWidget(self.PN_btn)
		pslayout.addWidget(self.resultsave)

		xlayout = QHBoxLayout()
		xlayout.addWidget(self.label_x_dscrpncy)
		xlayout.addWidget(self.dist_x_dscrpncy)

		ylayout = QHBoxLayout()
		ylayout.addWidget(self.label_y_dscrpncy)
		ylayout.addWidget(self.dist_y_dscrpncy)

		relayout.addLayout(xlayout)
		relayout.addLayout(ylayout)
		relayout.addLayout(pslayout)

		self.PN_btn.clicked.connect(self.PNShow)
		self.P_x1 = -1

		relayout.setSpacing(2)
		RegroupBox.setLayout(relayout)
		return RegroupBox

	def ComBox(self):
		trygroupBox = QGroupBox("Running and Results")
		trylayout = QVBoxLayout()
		trylayout.addWidget(self.RunningGroupBox())
		trylayout.addWidget(self.resultsBox())
		trylayout.setSpacing(2)
		trygroupBox.setLayout(trylayout)
		return trygroupBox

	def DistplotGroupBox(self):
		SPgroupBox = QGroupBox("Distribution Plots at RFQ exit")
		playout = QGridLayout()   #QVBoxLayout() 
	   
		x_canvas = FigureCanvas(Figure(figsize=(1, 3)))
		playout.addWidget(x_canvas,1,1,30,1)
		y_canvas = FigureCanvas(Figure(figsize=(1, 3)))
		playout.addWidget(y_canvas,1,2,30,1)

		self.emit_x = x_canvas.figure.subplots()
		self.emit_x.set_position([0.1, 0.15, 0.85, 0.7])
		self.emit_y = y_canvas.figure.subplots()
		self.emit_y.set_position([0.1, 0.15, 0.85, 0.7])
		SPgroupBox.setLayout(playout)
		return SPgroupBox
	def PNShow(self):
		if self.P_x1 < 0:
			self.w3 = MyTableWidget('','','','','','','','',0,0,0,0,0,0,0,0,0,0,0,0)
			# self.w3 = MyTableWidget('','','','','','','','','','','','','','','','','','','','')
			self.w3.show()
		else:
			self.w3 = MyTableWidget(self.P_x1,self.P_y1,self.P_x2,self.P_y2,self.P_x3,self.P_y3,self.P_x4,self.P_y4,self.comp12x,self.comp13x ,self.comp14x,\
				self.comp23x,self.comp24x,self.comp34x,self.comp12y,self.comp13y ,self.comp14y,self.comp23y,self.comp24y,self.comp34y)
			self.w3.show()

	def on_click1(self):   #=================================Read gradients file
		self.openFileNameDialog1()

	def openFileNameDialog1(self):  
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
		if fileName:
			self.text_Gfile.setText(fileName)
			fileRead = open(fileName,'r+')
			line1 = fileRead.readlines()[0][0:2] 
			if line1 =="QH":
				QMessageBox.about(self, "", "Wrong file!!!")
			else:
				head, tail = os.path.split(fileName)
				self.filedir = head+'/' # directory where bunches are located
				N_part_file = np.loadtxt(fileName,skiprows=(14))
				self.N_part = len(np.array(N_part_file[:,0]))
				print("\n there are %i particles"%self.N_part)

				PATH1=head +"/G_" + tail
				if os.path.isfile(PATH1) and os.access(PATH1, os.R_OK):
					Grad = np.genfromtxt(PATH1, dtype = float,delimiter='', skip_header=0,comments=None,usecols = (1) )              
					if tail == 'case1.txt':
						self.Dis1.setText(tail)
						self.c1q1.setValue(float(Grad[0]))   #'%.4f' %alphax))
						self.c1q2.setValue(float(Grad[1]))
						self.c1q3.setValue(float(Grad[2]))
						self.c1q4.setValue(float(Grad[3]))
					if tail == 'case2.txt':
						self.Dis2.setText(tail)
						self.c2q1.setValue(float(Grad[0]))   #'%.4f' %alphax))
						self.c2q2.setValue(float(Grad[1]))
						self.c2q3.setValue(float(Grad[2]))
						self.c2q4.setValue(float(Grad[3]))
					if tail == 'case3.txt':
						self.Dis3.setText(tail)
						self.c3q1.setValue(float(Grad[0]))   #'%.4f' %alphax))
						self.c3q2.setValue(float(Grad[1]))
						self.c3q3.setValue(float(Grad[2]))
						self.c3q4.setValue(float(Grad[3]))
					if tail == 'case4.txt':
						self.Dis4.setText(tail)
						self.c4q1.setValue(float(Grad[0]))   #'%.4f' %alphax))
						self.c4q2.setValue(float(Grad[1]))
						self.c4q3.setValue(float(Grad[2]))
						self.c4q4.setValue(float(Grad[3]))

				else:
					QMessageBox.about(self, "", "G_" + tail +" is missing!!!")

	def on_click2(self):
		self.Dis1.setText("")
		self.c1q1.setValue(0)   
		self.c1q2.setValue(0)
		self.c1q3.setValue(0)
		self.c1q4.setValue(0)
		self.Dis2.setText("")
		self.c2q1.setValue(0)   
		self.c2q2.setValue(0)
		self.c2q3.setValue(0)
		self.c2q4.setValue(0)
		self.Dis3.setText("")
		self.c3q1.setValue(0)   
		self.c3q2.setValue(0)
		self.c3q3.setValue(0)
		self.c3q4.setValue(0)
		self.Dis4.setText("")
		self.c4q1.setValue(0)   
		self.c4q2.setValue(0)
		self.c4q3.setValue(0)
		self.c4q4.setValue(0)
		self.thread1 = clearplot(self)
		self.thread1.plotclear.connect(self.resetplot)
		self.thread1.start()

	def resetplot(self):
		self.emit_x.clear()
		self.emit_y.clear()
		self.emit_x.figure.canvas.draw()
		self.emit_y.figure.canvas.draw()

	def on_click4(self):       #=================================================Run
		#===================================================Set progressBar
		self.progress = QProgressDialog(self)
		self.progress.setLabelText(" \n \n                                         Lattice Simulation is in process...                                               ")
		self.progress.setStyleSheet('font: bold;')
		self.progress.setMinimumDuration(0)
		self.progress.setWindowModality(Qt.WindowModal)
		self.progress.setRange(0,100) 
		#=====================================================Set progressBar

		time1 = time.clock()
		if (self.Dis1.text()=="" and self.Dis2.text()=="" and self.Dis3.text()=="" and self.Dis4.text()==""):
			QMessageBox.about(self, "", "No files is chosen!")
			self.progress.cancel()
		else:
			if (self.Noise_TR.text() == ""):
				self.Noise_TR.setText("0.09")
			if (self.current.text() == ""):
				self.current.setText("20")
			if (self.Grid_N.text() == "" or self.Grid_N.text() == "0.0" or self.Grid_N.text() == "0"):
				self.Grid_N.setText("50")
			if (self.a_z.text() == ""):
				self.a_z.setText("0")
			if (self.b_z.text() == "" or self.b_z.text() == "0.0"   or self.b_z.text() == "0"):
				self.b_z.setText("0.6")
			if (self.e_z.text() == ""):
				self.e_z.setText("0.2")
				self.e_z.repaint()

			self.thread1 = clearplot(self)
			self.thread1.plotclear.connect(self.resetplot)
			self.thread1.start()

			gridx_n = gridy_n = int(self.Grid_N.text())

			if (self.Dis1.text()=="case1.txt"):
				#self.thread_z_generation = Zgeneration(self.current.text(),self.c1q1.text(),self.c1q2.text(),self.c1q3.text(),self.c1q4.text(),\
				#	self.a_z.text(),self.b_z.text(),self.e_z.text(),self.Dis1.text(),self.field.currentText(), self.N_part)
				#self.thread_z_generation.zgene.connect(self.proBar_z)
				#self.thread_z_generation.zdata.connect(self.back_track)
				#self.thread_z_generation.run()
				self.back_track(self.filedir+self.Dis1.text(),self.c1q1.text(),self.c1q2.text(),self.c1q3.text(),self.c1q4.text())
				self.D1 = 1
			else:
				self.D1 = 0
				self.x1_s,self.xp1_s,self.y1_s,self.yp1_s = [0]*gridx_n,[0]*gridx_n,[0]*gridy_n,[0]*gridy_n
				self.x1,self.xp1,self.y1,self.yp1 = [0]*gridx_n,[0]*gridx_n,[0]*gridy_n,[0]*gridy_n
				self.P_x1 = 0
				self.P_y1 = 0
				self.xarr1, self.yarr1 = 0,0

			if (self.Dis2.text()=="case2.txt"):
				#self.thread_z_generation = Zgeneration(self.current.text(),self.c2q1.text(),self.c2q2.text(),self.c2q3.text(),self.c2q4.text(),\
				#	self.a_z.text(),self.b_z.text(),self.e_z.text(),self.Dis2.text(),self.field.currentText(), self.N_part)
				#self.thread_z_generation.zgene.connect(self.proBar_z)
				#self.thread_z_generation.zdata.connect(self.back_track)
				#self.thread_z_generation.run()
				self.back_track(self.filedir+self.Dis2.text(),self.c2q1.text(),self.c2q2.text(),self.c2q3.text(),self.c2q4.text())
				self.D2 = 1
			else:
				self.D2 = 0
				self.x2_s,self.xp2_s,self.y2_s,self.yp2_s = [0]*gridx_n,[0]*gridx_n,[0]*gridy_n,[0]*gridy_n
				self.x2,self.xp2,self.y2,self.yp2 = [0]*gridx_n,[0]*gridx_n,[0]*gridy_n,[0]*gridy_n
				self.P_x2 = 0
				self.P_y2 = 0
				self.xarr2, self.yarr2 = 0,0

			if (self.Dis3.text()=="case3.txt"):
				#self.thread_z_generation = Zgeneration(self.current.text(),self.c3q1.text(),self.c3q2.text(),self.c3q3.text(),self.c3q4.text(),\
				#	self.a_z.text(),self.b_z.text(),self.e_z.text(),self.Dis3.text(),self.field.currentText(), self.N_part)
				#self.thread_z_generation.zgene.connect(self.proBar_z)
				#self.thread_z_generation.zdata.connect(self.back_track)
				#self.thread_z_generation.run()
				self.back_track(self.filedir+self.Dis3.text(),self.c3q1.text(),self.c3q2.text(),self.c3q3.text(),self.c3q4.text())
				self.D3 = 1
			else:
				self.D3 = 0
				self.x3_s,self.xp3_s,self.y3_s,self.yp3_s = [0]*gridx_n,[0]*gridx_n,[0]*gridy_n,[0]*gridy_n
				self.x3,self.xp3,self.y3,self.yp3 = [0]*gridx_n,[0]*gridx_n,[0]*gridy_n,[0]*gridy_n
				self.P_x3 = 0
				self.P_y3 = 0
				self.xarr3, self.yarr3 = 0,0

			if (self.Dis4.text()=="case4.txt"):
				#self.thread_z_generation = Zgeneration(self.current.text(),self.c4q1.text(),self.c4q2.text(),self.c4q3.text(),self.c4q4.text(),\
				#	self.a_z.text(),self.b_z.text(),self.e_z.text(),self.Dis4.text(),self.field.currentText(), self.N_part)
				#self.thread_z_generation.zgene.connect(self.proBar_z)
				#self.thread_z_generation.zdata.connect(self.back_track)
				#self.thread_z_generation.run()
				self.back_track(self.filedir+self.Dis4.text(),self.c4q1.text(),self.c4q2.text(),self.c4q3.text(),self.c4q4.text())
				self.D4 = 1
			else:
				self.D4 = 0
				self.x4_s,self.xp4_s,self.y4_s,self.yp4_s = [0]*gridx_n,[0]*gridx_n,[0]*gridy_n,[0]*gridy_n
				self.x4,self.xp4,self.y4,self.yp4 = [0]*gridx_n,[0]*gridx_n,[0]*gridy_n,[0]*gridy_n
				self.P_x4 = 0
				self.P_y4 = 0
				self.xarr4, self.yarr4 = 0,0

			if (self.D1 ==0 and self.D2 ==0 and self.D3 ==0 and self.D4 ==0):
				QMessageBox.about(self, "", "No data at RFQ exit!!!")
			else:
				xmin =  np.floor(np.min([min(self.x1_s),min(self.x2_s),min(self.x3_s),min(self.x4_s)]))   
				xmax =  np.ceil(np.max([max(self.x1_s),max(self.x2_s),max(self.x3_s),max(self.x4_s)]))   
				xpmin =  np.floor(np.min([min(self.xp1_s),min(self.xp2_s),min(self.xp3_s),min(self.xp4_s)]))
				xpmax =  np.ceil(np.max([max(self.xp1_s),max(self.xp2_s),max(self.xp3_s),max(self.xp4_s)])) 
				ymin =   np.floor(np.min([min(self.y1_s),min(self.y2_s),min(self.y3_s),min(self.y4_s)]))
				ymax =   np.ceil(np.max([max(self.y1_s),max(self.y2_s),max(self.y3_s),max(self.y4_s)]))
				ypmin =  np.floor(np.min([min(self.yp1_s),min(self.yp2_s),min(self.yp3_s),min(self.yp4_s)]))
				ypmax =  np.ceil(np.max([max(self.yp1_s),max(self.yp2_s),max(self.yp3_s),max(self.yp4_s)]))
				print xmin,xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax
				if (self.D1 ==1): self.cordCenterizing(self.x1_s,self.xp1_s,self.y1_s,self.yp1_s,"case1.txt",xmin,xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax)
				if (self.D2 ==1): self.cordCenterizing(self.x2_s,self.xp2_s,self.y2_s,self.yp2_s,"case2.txt",xmin,xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax)
				if (self.D3 ==1): self.cordCenterizing(self.x3_s,self.xp3_s,self.y3_s,self.yp3_s,"case3.txt",xmin,xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax)
				if (self.D4 ==1): self.cordCenterizing(self.x4_s,self.xp4_s,self.y4_s,self.yp4_s,"case4.txt",xmin,xmax,xpmin,xpmax,ymin,ymax,ypmin,ypmax)
				print self.P_x1,self.P_y1,self.P_x2,self.P_y2,self.P_x3,self.P_y3,self.P_x4,self.P_y4

				self.DisPlot(self.D1,self.D2,self.D3,self.D4,self.x1,self.xp1,self.y1,self.yp1,self.x2,self.xp2,self.y2,\
					self.yp2,self.x3,self.xp3,self.y3,self.yp3,self.x4,self.xp4,self.y4,self.yp4)

				comp_results = DistCompRFQexit(self.D1,self.D2,self.D3,self.D4,self.P_x1,self.P_y1,self.xarr1,self.yarr1,\
					self.P_x2,self.P_y2,self.xarr2,self.yarr2,self.P_x3,self.P_y3,self.xarr3,self.yarr3,self.P_x4,self.P_y4,self.xarr4,self.yarr4)
				self.discrepancy_x, self.discrepancy_y, self.comp12x,self.comp13x ,self.comp14x,self.comp23x,self.comp24x,self.comp34x,\
				self.comp12y,self.comp13y ,self.comp14y,self.comp23y,self.comp24y,self.comp34y= comp_results.ResultComp()
				print "Coordinates discrepencies in x-plane and y-plane are:",self.discrepancy_x, self.discrepancy_y
				self.dist_x_dscrpncy.setText(str('%.6f' %self.discrepancy_x))
				self.dist_y_dscrpncy.setText(str('%.6f' %self.discrepancy_y))
				self.dist_y_dscrpncy.repaint()

			time_exec = time.clock() - time1
			print "Execution time =", time_exec

	def back_track(self,distrfilename,Q01,Q02,Q03,Q04):
		"""
		# Combines transverse + longitudinal data
		filename = 'Distribution_at_measured_position.txt'
		distFile  = open (filename,"w")
		with open("Distribution_Output_new_for_Z.txt", "r") as oldinput:
			for line in islice(oldinput, 0,14):
				distFile.write(line)
		data1 = np.loadtxt("./cases_measured_190129_to_190201/" + cname,skiprows=(0))
		x = np.array(data1[:,0])/1000
		xp = np.array(data1[:,1])/1000
		y = np.array(data1[:,2])/1000
		yp = np.array(data1[:,3])/1000
		new = np.array([x,xp,y,yp,z,dE])
		np.savetxt(distFile,new.T,fmt = "%12.7e")
		distFile.flush()
		distFile.close()
		"""
		self.thread_backtracking = back_tracking_thread(self,Q01,Q02,Q03,Q04,distrfilename,self.field.currentText())
		self.thread_backtracking.backSim.connect(self.proBar_z)
		self.thread_backtracking.distatRFQ.connect(self.DistsSave)
		self.thread_backtracking.run() # run back_tracking 

	def proBar_z(self,pos,ttL,step):
		if ((ttL-pos) > step):
			self.progress.setValue(pos /ttL*100)
			self.progress.setCancelButtonText(str('%.4f' %pos))
		else:
			self.progress.setValue(100)

	def cordCenterizing(self,x,xp,y,yp,cname,xmin0,xmax0,xpmin0,xpmax0,ymin0,ymax0,ypmin0,ypmax0):
		self.thread_cordCenterizing = cordCenterizingThread(self,x,xp,y,yp,self.Noise_TR.text(),self.Grid_N.text(),cname,xmin0,xmax0,xpmin0,xpmax0,ymin0,ymax0,ypmin0,ypmax0)
		self.thread_cordCenterizing.filtering.connect(self.proBar_filter)
		self.thread_cordCenterizing.PNinGrids.connect(self.PN_in_grid)
		self.thread_cordCenterizing.run()

	def DistsSave(self,xs,xps,ys,yps,zs,dEs,cnames):
		cords_centered = centered_cord(xs,xps,ys,yps)
		if (cnames == "case1.txt"):
			self.x1_s,self.xp1_s,self.y1_s,self.yp1_s = cords_centered.CenteredParameters()
			self.z1_s, self.dE1_s = np.array(zs)*1000,np.array(dEs)*1000
		if (cnames == "case2.txt"):
			self.x2_s,self.xp2_s,self.y2_s,self.yp2_s = cords_centered.CenteredParameters()
			self.z2_s, self.dE2_s = np.array(zs)*1000,np.array(dEs)*1000
		if (cnames == "case3.txt"):
			self.x3_s,self.xp3_s,self.y3_s,self.yp3_s = cords_centered.CenteredParameters()
			self.z3_s, self.dE3_s = np.array(zs)*1000,np.array(dEs)*1000
		if (cnames == "case4.txt"):
			self.x4_s,self.xp4_s,self.y4_s,self.yp4_s = cords_centered.CenteredParameters()
			self.z4_s, self.dE4_s = np.array(zs)*1000,np.array(dEs)*1000

	def proBar_filter(self, val,Num_t,n):
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

	def PN_in_grid(self,sumx,sumy,x_arr,y_arr,cname_N,x_g,xp_g,y_g,yp_g):
		if (cname_N == "case1.txt"):
			self.x1,self.xp1,self.y1,self.yp1 = x_g,xp_g,y_g,yp_g
			self.P_x1 = float(sumx)
			self.P_y1 = float(sumy)
			self.xarr1 = x_arr
			self.yarr1 = y_arr

		if (cname_N == "case2.txt"):
			self.x2,self.xp2,self.y2,self.yp2 = x_g,xp_g,y_g,yp_g
			self.P_x2 = float(sumx)
			self.P_y2 = float(sumy)
			self.xarr2 = x_arr
			self.yarr2 = y_arr

		if (cname_N == "case3.txt"):
			self.x3,self.xp3,self.y3,self.yp3 = x_g,xp_g,y_g,yp_g
			self.P_x3 = float(sumx)
			self.P_y3 = float(sumy)
			self.xarr3 = x_arr
			self.yarr3 = y_arr

		if (cname_N == "case4.txt"):
			self.x4,self.xp4,self.y4,self.yp4 = x_g,xp_g,y_g,yp_g
			self.P_x4 = float(sumx)
			self.P_y4 = float(sumy)
			self.xarr4 = x_arr
			self.yarr4 = y_arr

	def savequads(self):                   # Save Quads Gradients
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
		if fileName:
			FileforQuads = open(fileName,"w+")     
			FileforQuads.write("Case1" +'\n')
			FileforQuads.write("  QH01   " + "%6.5f"%float(self.c1q1.text()) +'\n')
			FileforQuads.write("  QH02   " + "%6.5f"%float(self.c1q2.text()) +'\n')
			FileforQuads.write("  QH03   " + "%6.5f"%float(self.c1q3.text()) +'\n') 
			FileforQuads.write("  QH04   " + "%6.5f"%float(self.c1q4.text()) +'\n')
			FileforQuads.write("Case2" +'\n')
			FileforQuads.write("  QH01   " + "%6.5f"%float(self.c2q1.text()) +'\n')
			FileforQuads.write("  QH02   " + "%6.5f"%float(self.c2q2.text()) +'\n')
			FileforQuads.write("  QH03   " + "%6.5f"%float(self.c2q3.text()) +'\n') 
			FileforQuads.write("  QH04   " + "%6.5f"%float(self.c2q4.text()) +'\n')
			FileforQuads.write("Case3" +'\n')
			FileforQuads.write("  QH01   " + "%6.5f"%float(self.c3q1.text()) +'\n')
			FileforQuads.write("  QH02   " + "%6.5f"%float(self.c3q2.text()) +'\n')
			FileforQuads.write("  QH03   " + "%6.5f"%float(self.c3q3.text()) +'\n') 
			FileforQuads.write("  QH04   " + "%6.5f"%float(self.c3q4.text()) +'\n')
			FileforQuads.write("Case4" +'\n')
			FileforQuads.write("  QH01   " + "%6.5f"%float(self.c4q1.text()) +'\n')
			FileforQuads.write("  QH02   " + "%6.5f"%float(self.c4q2.text()) +'\n')
			FileforQuads.write("  QH03   " + "%6.5f"%float(self.c4q3.text()) +'\n') 
			FileforQuads.write("  QH04   " + "%6.5f"%float(self.c4q4.text()) +'\n')

	def saveresults(self):
		if (self.dist_x_dscrpncy.text() == ""):
			QMessageBox.about(self, "", "No Data!")
		else:            
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
			if fileName:
				self.DistsCombine(fileName)

	def DistsCombine(self,Distfile):
		x_f = []
		xp_f = []
		y_f = []
		yp_f = []
		z_f = []
		dE_f = []
		if (self.D1 == 1):
			x_f.extend(self.x1_s)
			xp_f.extend(self.xp1_s)
			y_f.extend(self.y1_s)
			yp_f.extend(self.yp1_s)
			z_f.extend(self.z1_s)
			dE_f.extend(self.dE1_s)

		if (self.D2 == 1):
			x_f.extend(self.x2_s)
			xp_f.extend(self.xp2_s)
			y_f.extend(self.y2_s)
			yp_f.extend(self.yp2_s)
			z_f.extend(self.z2_s)
			dE_f.extend(self.dE2_s) 

		if (self.D3 == 1):
			x_f.extend(self.x3_s)
			xp_f.extend(self.xp3_s)
			y_f.extend(self.y3_s)
			yp_f.extend(self.yp3_s)
			z_f.extend(self.z3_s)
			dE_f.extend(self.dE3_s) 

		if (self.D4 == 1):
			x_f.extend(self.x4_s)
			xp_f.extend(self.xp4_s)
			y_f.extend(self.y4_s)
			yp_f.extend(self.yp4_s)
			z_f.extend(self.z4_s)
			dE_f.extend(self.dE4_s)
		N_case =  self.D1 +self.D2 +self.D3 +self.D4

		distFile  = open (Distfile,"w")
		filename2 = "Distribution_Output_new_for_Z.txt"
		macroSize = np.genfromtxt(filename2, dtype = float,delimiter='', skip_header=3,max_rows=1,comments=None,usecols = 3)
		new_mcro = float(macroSize) / N_case
		with open(filename2, "r") as oldinput:
			for line in islice(oldinput, 0,14):
				line_new = line.replace(str(macroSize),str(new_mcro))
				distFile.write(line_new)

		new1 = np.array([x_f,xp_f,y_f,yp_f,z_f,dE_f])/1000.0
		np.savetxt(distFile,new1.T,fmt = "%12.7e")
		distFile.flush()
		distFile.close()

	def DisPlot(self,D1,D2,D3,D4,x1,xp1,y1,yp1,x2,xp2,y2,yp2,x3,xp3,y3,yp3,x4,xp4,y4,yp4):
		self.emit_x.clear()
		self.emit_y.clear()
		self.emit_x.set_title('x-xp (mm-mrad)')
		self.emit_y.set_title('y-yp (mm-mrad)')

		if (D1 == 1 and D2 ==0 and D3 ==0 and D4 ==0):
			self.emit_x.plot(x1,xp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_y.plot(y1,yp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
		if (D1 == 0 and D2 ==1 and D3 ==0 and D4 ==0):
			self.emit_x.plot(x2,xp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_y.plot(y2,yp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
		if (D1 == 0 and D2 ==0 and D3 ==1 and D4 ==0):
			self.emit_x.plot(x3,xp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_y.plot(y3,yp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
		if (D1 == 0 and D2 ==0 and D3 ==0 and D4 ==1):
			self.emit_x.plot(x4,xp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
			self.emit_y.plot(y4,yp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
		if (D1 == 1 and D2 ==1 and D3 ==0 and D4 ==0):
			self.emit_x.plot(x1,xp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_x.plot(x2,xp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_y.plot(y1,yp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_y.plot(y2,yp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
		if (D1 == 1 and D2 ==0 and D3 ==1 and D4 ==0):
			self.emit_x.plot(x1,xp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_x.plot(x3,xp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_y.plot(y1,yp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_y.plot(y3,yp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
		if (D1 == 1 and D2 ==0 and D3 ==0 and D4 ==1):
			self.emit_x.plot(x1,xp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_x.plot(x4,xp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
			self.emit_y.plot(y1,yp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_y.plot(y4,yp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
		if (D1 == 0 and D2 ==1 and D3 ==1 and D4 ==0):
			self.emit_x.plot(x2,xp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_x.plot(x3,xp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_y.plot(y2,yp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_y.plot(y3,yp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
		if (D1 == 0 and D2 ==1 and D3 ==0 and D4 ==1):
			self.emit_x.plot(x2,xp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_x.plot(x4,xp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
			self.emit_y.plot(y2,yp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_y.plot(y4,yp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
		if (D1 == 0 and D2 ==0 and D3 ==1 and D4 ==1):
			self.emit_x.plot(x3,xp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_x.plot(x4,xp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
			self.emit_y.plot(y3,yp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_y.plot(y4,yp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
		if (D1 == 1 and D2 ==1 and D3 ==1 and D4 ==0):
			self.emit_x.plot(x1,xp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_x.plot(x2,xp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_x.plot(x3,xp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_y.plot(y1,yp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_y.plot(y2,yp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_y.plot(y3,yp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
		if (D1 == 1 and D2 ==1 and D3 ==0 and D4 ==1):
			self.emit_x.plot(x1,xp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_x.plot(x2,xp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_x.plot(x4,xp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
			self.emit_y.plot(y1,yp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_y.plot(y2,yp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_y.plot(y4,yp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
		if (D1 == 1 and D2 ==0 and D3 ==1 and D4 ==1):
			self.emit_x.plot(x1,xp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_x.plot(x3,xp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_x.plot(x4,xp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
			self.emit_y.plot(y1,yp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_y.plot(y3,yp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_y.plot(y4,yp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
		if (D1 == 0 and D2 ==1 and D3 ==1 and D4 ==1):
			self.emit_x.plot(x2,xp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_x.plot(x3,xp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_x.plot(x4,xp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
			self.emit_y.plot(y2,yp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_y.plot(y3,yp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_y.plot(y4,yp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
		if (D1 == 1 and D2 ==1 and D3 ==1 and D4 ==1):
			self.emit_x.plot(x1,xp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_x.plot(x2,xp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_x.plot(x3,xp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_x.plot(x4,xp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')
			self.emit_y.plot(y1,yp1,linestyle='',marker='*',markerfacecolor='red',markeredgecolor='red',markersize=0.3, markeredgewidth=1.0,label='case1')
			self.emit_y.plot(y2,yp2,linestyle='',marker='*',markerfacecolor='green',markeredgecolor='green',markersize=0.3, markeredgewidth=1.0,label='case2')
			self.emit_y.plot(y3,yp3,linestyle='',marker='*',markerfacecolor='blue',markeredgecolor='blue',markersize=0.3, markeredgewidth=1.0,label='case3')
			self.emit_y.plot(y4,yp4,linestyle='',marker='*',markerfacecolor='cyan',markeredgecolor='cyan',markersize=0.3, markeredgewidth=1.0,label='case4')

		self.emit_x.legend(loc=2)
		self.emit_y.legend(loc=1)
		# self._static_ax.yaxis.set_major_locator(NullLocator())   # Hiding ticks and lable 
		self.emit_x.tick_params(direction='in')
		self.emit_y.tick_params(direction='in')
		self.emit_x.figure.canvas.draw()
		self.emit_y.figure.canvas.draw()
		self.Noise_TR.repaint()  

class MyTableWidget(QWidget):  
	def __init__(self,x1,y1,x2,y2,x3,y3,x4,y4,c12x,c13x,c14x,c23x,c24x,c34x,c12y,c13y,c14y,c23y,c24y,c34y):
		super(MyTableWidget,self).__init__()
		pnshowlayout = QGridLayout()    #QVBoxLayout()   QGridLayout() 

		self.label_pn =  QLabel(' =======Particle Numbers after Selection',self)
		self.xplane = QLabel('x_plane',self)
		self.yplane = QLabel('y_plane',self)
		self.label_c1 =  QLabel('case1',self)
		self.label_c2 =  QLabel('case2',self)
		self.label_c3 =  QLabel('case3',self)
		self.label_c4 =  QLabel('case4',self) 
		self.label_comp12 =  QLabel('Comp12',self) 
		self.label_comp13 =  QLabel('Comp13',self)
		self.label_comp14 =  QLabel('Comp14',self)
		self.label_comp23 =  QLabel('Comp23',self)
		self.label_comp24 =  QLabel('Comp24',self)
		self.label_comp34 =  QLabel('Comp34',self)
		self.label_comp =  QLabel(' =======Comparing Results',self)
		self.c1x = QLineEdit(self)
		self.c1y = QLineEdit(self)
		self.c2x = QLineEdit(self)
		self.c2y = QLineEdit(self)
		self.c3x = QLineEdit(self)
		self.c3y = QLineEdit(self)
		self.c4x = QLineEdit(self)
		self.c4y = QLineEdit(self)
		self.comp12x = QLineEdit(self)
		self.comp12y = QLineEdit(self)
		self.comp13x = QLineEdit(self)
		self.comp13y = QLineEdit(self)
		self.comp14x = QLineEdit(self)
		self.comp14y = QLineEdit(self)
		self.comp23x = QLineEdit(self)
		self.comp23y = QLineEdit(self)
		self.comp24x = QLineEdit(self)
		self.comp24y = QLineEdit(self)
		self.comp34x = QLineEdit(self)
		self.comp34y = QLineEdit(self)

		pnshowlayout.addWidget(self.label_pn,0,0,1,3)
		pnshowlayout.addWidget(self.xplane,1,1,1,1)
		pnshowlayout.addWidget(self.yplane,1,2,1,1)
		pnshowlayout.addWidget(self.label_c1,2,0,1,1)
		pnshowlayout.addWidget(self.label_c2,3,0,1,1)
		pnshowlayout.addWidget(self.label_c3,4,0,1,1)
		pnshowlayout.addWidget(self.label_c4,5,0,1,1)
		pnshowlayout.addWidget(self.label_comp12,7,0,1,1)
		pnshowlayout.addWidget(self.label_comp13,8,0,1,1)
		pnshowlayout.addWidget(self.label_comp14,9,0,1,1)
		pnshowlayout.addWidget(self.label_comp23,10,0,1,1)
		pnshowlayout.addWidget(self.label_comp24,11,0,1,1)
		pnshowlayout.addWidget(self.label_comp34,12,0,1,1)
		pnshowlayout.addWidget(self.c1x,2,1,1,1)
		pnshowlayout.addWidget(self.c1y,2,2,1,1)
		pnshowlayout.addWidget(self.c2x,3,1,1,1)
		pnshowlayout.addWidget(self.c2y,3,2,1,1)
		pnshowlayout.addWidget(self.c3x,4,1,1,1)
		pnshowlayout.addWidget(self.c3y,4,2,1,1)
		pnshowlayout.addWidget(self.c4x,5,1,1,1)
		pnshowlayout.addWidget(self.c4y,5,2,1,1)
		pnshowlayout.addWidget(self.label_comp,6,0,1,3)
		pnshowlayout.addWidget(self.comp12x,7,1,1,1)
		pnshowlayout.addWidget(self.comp12y,7,2,1,1)
		pnshowlayout.addWidget(self.comp13x,8,1,1,1)
		pnshowlayout.addWidget(self.comp13y,8,2,1,1)
		pnshowlayout.addWidget(self.comp14x,9,1,1,1)
		pnshowlayout.addWidget(self.comp14y,9,2,1,1)
		pnshowlayout.addWidget(self.comp23x,10,1,1,1)
		pnshowlayout.addWidget(self.comp23y,10,2,1,1)
		pnshowlayout.addWidget(self.comp24x,11,1,1,1)
		pnshowlayout.addWidget(self.comp24y,11,2,1,1)
		pnshowlayout.addWidget(self.comp34x,12,1,1,1)
		pnshowlayout.addWidget(self.comp34y,12,2,1,1)
		self.c1x.setText(str(x1))
		self.c1y.setText(str(y1))
		self.c2x.setText(str(x2))
		self.c2y.setText(str(y2))
		self.c3x.setText(str(x3))
		self.c3y.setText(str(y3))
		self.c4x.setText(str(x4))
		self.c4y.setText(str(y4))
		self.comp12x.setText(str('%.6f' %float(c12x)))
		self.comp12y.setText(str('%.6f' %float(c12y)))
		self.comp13x.setText(str('%.6f' %float(c13x)))
		self.comp13y.setText(str('%.6f' %float(c13y)))
		self.comp14x.setText(str('%.6f' %float(c14x)))
		self.comp14y.setText(str('%.6f' %float(c14y)))
		self.comp23x.setText(str('%.6f' %float(c23x)))
		self.comp23y.setText(str('%.6f' %float(c23y)))
		self.comp24x.setText(str('%.6f' %float(c24x)))
		self.comp24y.setText(str('%.6f' %float(c24y)))
		self.comp34x.setText(str('%.6f' %float(c34x)))
		self.comp34y.setText(str('%.6f' %float(c34y)))
		
		self.setLayout(pnshowlayout)
		self.setWindowTitle("Particle Numbers for Comparison ")

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ttt = Window_dispersion()
	ttt.show()
	sys.exit(app.exec_())


