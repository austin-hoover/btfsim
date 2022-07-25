from PyQt5.QtWidgets import QWidget, QTabWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QGroupBox
from PyQt5.QtWidgets import QLabel, QPushButton, QLineEdit, QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QDoubleSpinBox, QComboBox, QCheckBox
from PyQt5 import QtGui, QtCore
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter,NullLocator

# -- local
import btfsim.lattice.generate_btf_lattice as latticegenerator
import btfsim.util.Defaults as defaults

#from FODOgui import WindowFODO
#from Dispersion_curve_GUI import Window_dispersion

class MainWindow(QWidget):
	def __init__(self):
		super(MainWindow,self).__init__()
 
		layout = QGridLayout()

		# -- make tabs
		self.tabs = QTabWidget()
		self.tab1 = QWidget()   
		self.tab2 = QWidget()
		self.tab3 = QWidget()

		self.tabs.addTab(self.tab1,"BTF GUI")
		self.tabs.addTab(self.tab2,"BTF FODO GUI") 
		self.tabs.addTab(self.tab3,"Dispersion GUI")      


		BTFlayout = QGridLayout()
		FODOlayout = QGridLayout()
		DISPlayout = QGridLayout()

		btfwindow = WindowMEBT1MEBT2()
		BTFlayout.addWidget(btfwindow)
		BTFlayout.setContentsMargins(1,1, 1, 1)

		fodowindow = WindowFODO()
		FODOlayout.addWidget(fodowindow)
		FODOlayout.setContentsMargins(1,1, 1, 1)

		dispwindow = Window_dispersion()
		DISPlayout.addWidget(dispwindow)
		DISPlayout.setContentsMargins(20,1, 300, 80)

		self.tab1.setLayout(BTFlayout)
		self.tab2.setLayout(FODOlayout)
		self.tab3.setLayout(DISPlayout)

		layout.addWidget(self.tabs)
		layout.setContentsMargins(10,10,10,10)
		self.setLayout(layout)
		self.setWindowTitle("BTF_FODO_Experiment_Gui")
		self.show()

###############################################################################
## Single Pane Class for GUI design
###############################################################################					
class WindowPane(QWidget):
	def __init__(self,beamline=["MEBT1"]):
		super(WindowPane,self).__init__()

		# -- this will be stored and used later to define some layouts, defaults
		self.beamline = beamline
		self.default = defaults.getDefaults()
                
		mainlayout = QGridLayout()
 
		self.mainlayout1 = QHBoxLayout()
		self.mainlayout1.addWidget(self.DisInputGroupBox())
		self.gradientBox = self.GradientGroupBox()
		self.mainlayout1.addWidget(self.gradientBox)
		self.mainlayout1.addWidget(self.ParamsGroupBox())
		self.mainlayout2 = QGridLayout()    #QHBoxLayout()
		self.mainlayout2.addWidget(self.OneDPlotGroupBox(),0,0)
		self.mainlayout2.addWidget(self.DisOutputGroupBox(),0,1)
		self.mainlayout2.addWidget(self.TwoDEmitplotGroupBox(),0,2)
		self.mainlayout2.setColumnStretch(0,2)
		self.mainlayout2.setColumnStretch(1,2)
		self.mainlayout2.setColumnStretch(2,4)

		mainlayout.addLayout(self.mainlayout1,0,0)
		mainlayout.addWidget(self.SizePlotGroupBox(),1,0)
		mainlayout.addLayout(self.mainlayout2,2,0)
		mainlayout.setRowMinimumHeight(1,220)
		mainlayout.setSpacing(4)
		mainlayout.setContentsMargins(1,1,1,1)
		self.setLayout(mainlayout)
		self.setWindowTitle("BTF GUI")
		self.show()
 
	###############################################################################
	## Functions that create grouped boxes
	###############################################################################	
		
	def DisInputGroupBox(self):
		HgroupBox = QGroupBox("Select Input Distribution")
		HBox = QGridLayout()        #QVBoxLayout() QGridLayout()
		#default_filename = self.defaults.defaultdict['BUNCH_IN'].split('/')[-1]
		self.label1 = QLabel("Read from bunch file", self)
		self.button1 = QPushButton('Select',self)
#		self.button1.clicked.connect(self.on_click1)
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
		rlayout.addWidget(self.labela,0,0)   #,QtCore.Qt.AlignLeft)
		rlayout.addWidget(self.labelb,0,1,QtCore.Qt.AlignCenter)
		rlayout.addWidget(self.labele,0,2,QtCore.Qt.AlignCenter)
		rlayout.addWidget(self.label3,1,1,QtCore.Qt.AlignCenter)
		rlayout.addWidget(self.label4,1,2,QtCore.Qt.AlignCenter)
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
		self.G_layout = QVBoxLayout()        
		
		label_xmlinfo = QLabel("Lattice: %s"%(self.default.defaultdict["XML_FILE"]))
		
		label_beamline = QLabel("Beamlines:")
		beamline_list = ["MEBT1","MEBT2","MEBT3"]
		self.beamlineBoxes = []
		for beamline in beamline_list:
			cb = QCheckBox(beamline)
			if beamline in self.beamline:
				cb.setChecked(True)
			self.beamlineBoxes.append(cb)
		self.beamline_set_button = QPushButton("Set",self)
		self.beamline_set_button.clicked.connect(self.resetBeamlines)
		
		beamlines_layout = QHBoxLayout()
		beamlines_layout.addWidget(label_beamline)
		for cb in self.beamlineBoxes:
			beamlines_layout.addWidget(cb)
		beamlines_layout.addWidget(self.beamline_set_button)
		
		
		self.label_g0 = QLabel("*Read from file (.txt or .mstate)", self)
		self.button_g = QPushButton('Select',self)
		self.label_g10 = QLabel('*Manual Input, Unit:', self)
		self.quads_save = QPushButton('Save_Quads',self)
		self.checkbox_amps = QCheckBox('Amps',self)
		self.checkbox_amps.setChecked(True)
		self.checkbox_tesla = QCheckBox('Tesla',self)
		

		self.text_Gfile = QLineEdit(self)         # for gradients input file
		Magstep = 0.01

		select_row =  QHBoxLayout()
		spacer = QSpacerItem(30, 1, QSizePolicy.Maximum)
		select_row.addItem(spacer)
		select_row.addWidget(self.text_Gfile)
		select_row.addWidget(self.button_g) 

		row_manual = QHBoxLayout()
		row_manual.addWidget(self.label_g10)
		row_manual.addWidget(self.checkbox_amps)
		row_manual.addWidget(self.checkbox_tesla)
		spacerSQ = QSpacerItem(150, 1, QSizePolicy.Maximum) 
		row_manual.addItem(spacerSQ)
		row_manual.addWidget(self.quads_save)
		spacer = QSpacerItem(200, 1, QSizePolicy.Maximum)

		self.G_layout.addWidget(label_xmlinfo)
		self.G_layout.addLayout(beamlines_layout)
		self.G_layout.addWidget(self.label_g0)
		self.G_layout.addLayout(select_row)
		self.G_layout.addLayout(row_manual)

		self.gradientSubLayout = self.rowsFromLattice() # make rows from lattice xml file
		self.G_layout.addLayout(self.gradientSubLayout)
		#for i in range(len(self.g_layout_rows)):
		#		self.G_layout.addLayout(self.g_layout_rows[i])
            
		self.G_layout.setSpacing(2)   
		QGgroupBox.setLayout(self.G_layout)
		return QGgroupBox
	

		
	def ParamsGroupBox(self):
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
		#		self.button_Ini.clicked.connect(self.on_click3)
		self.button_run = QPushButton('Run',self)
		#		self.button_run.clicked.connect(self.on_click4)
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

		# -- output position (set to length of lattice)
		self.label_i1_1 = QLabel('Input at ', self)
		self.label_i1_2 = QLabel('m (or name)', self)
		self.text_S0 = QLineEdit(self)
		self.text_S0.setText("0.0")
		
		in_layout = QHBoxLayout()
		in_layout.addWidget(self.label_i1_1)
		spacerIn1 = QSpacerItem(100, 1, QSizePolicy.Maximum)
		in_layout.addItem(spacerIn1)
		in_layout.addWidget(self.text_S0)
		in_layout.addWidget(self.label_i1_2)
		
		# -- output position (set to length of lattice)
		self.label_o1_1 = QLabel('Output at', self)
		self.label_o1_2 = QLabel('m (or name)', self)
		self.text_S = QLineEdit(self)
		self.text_S.setText(str(self.max_sim_len))
		
		out_layout = QHBoxLayout()
		out_layout.addWidget(self.label_o1_1)
		spacerOut1 = QSpacerItem(100, 1, QSizePolicy.Maximum)
		out_layout.addItem(spacerOut1)
		out_layout.addWidget(self.text_S)
		out_layout.addWidget(self.label_o1_2)
		
		self.checkBox_reverse = QCheckBox('reverse',self)
		

		clayout.addLayout(I_layout)
		clayout.addLayout(P_layout)
		clayout.addLayout(F_layout)
		clayout.addLayout(SC_layout)
		clayout.addLayout(elpsN_layout)
		clayout.addLayout(S_layout)
		clayout.addLayout(in_layout)
		clayout.addLayout(out_layout)
		clayout.addWidget(self.checkBox_reverse)
		clayout.addWidget(self.button_run)

		clayout.setSpacing(3)
		CgroupBox.setLayout(clayout)
		return CgroupBox



	def SizePlotGroupBox(self):
		SPgroupBox = QGroupBox("Beam Sizes")
		playout = QGridLayout()   #  QHBoxLayout 

		self._static_fig = Figure(figsize=(5, 3))
		static_canvas = FigureCanvas(self._static_fig)
		playout.addWidget(static_canvas,0,0,8,1)
		self._static_ax = static_canvas.figure.subplots()
		self._static_ax.set_position([0.07, 0.2, 0.92, 0.7])
		self._static_ax.set_xlim([0,self.max_sim_len])

		playout.setColumnStretch(1,1)
		playout.setColumnStretch(0,20)
		
		# -- print-out for simulation time
		label = QLabel('Sim. time:')
		self.timer_out = QLabel('--')
		playout.addWidget(label,0,1)
		playout.addWidget(self.timer_out,1,1)
		
		# -- a save button
		self.save_hist_button = QPushButton("Save plot+data",self)
		#self.save_hist_button.setFixedWidth(160)
		#self.save_hist_button.setStyleSheet('font: 12px; min-width: 4.6em;')
		playout.addWidget(self.save_hist_button,2,1)
		
		# -- another save button
		self.save_bunch_button = QPushButton('Save bunch',self)
		#self.save_bunch_button.setFixedWidth(160)
		#self.save_bunch_button.setStyleSheet('font: 12px; min-width: 4.6em;')
		playout.addWidget(self.save_bunch_button,3,1)

		SPgroupBox.setLayout(playout)
		return SPgroupBox
 

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



	def DisOutputGroupBox(self):
		OPgroupBox = QGroupBox("Output Setting and Results")
		oplayout = QGridLayout()       #QGridLayout()   QVBoxLayout()     

		self.button_o1 = QPushButton('TwissCal',self)
		self.button_o1.setStyleSheet('font: 14px; min-width: 4.6em;')
		#		self.button_o1.clicked.connect(self.calTws_click)

		self.button_o2 = QPushButton('SaveTws',self)
		self.button_o2.setFixedWidth(120)
		self.button_o2.setStyleSheet('font: 14px; min-width: 4.6em;')
		#		self.button_o2.clicked.connect(self.on_click8)

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
		rowlayout2.addWidget(self.label_a,0,1,QtCore.Qt.AlignCenter)
		rowlayout2.addWidget(self.label_b,0,4,QtCore.Qt.AlignCenter)
		rowlayout2.addWidget(self.label_e,0,7,QtCore.Qt.AlignCenter)
		rowlayout2.addWidget(self.label_o3,1,4,QtCore.Qt.AlignCenter)
		rowlayout2.addWidget(self.label_o4,1,7,QtCore.Qt.AlignCenter)
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
		outerlayout = QVBoxLayout()
		
		optionslayout = QHBoxLayout()
		self.TwoD_replot_button = QPushButton('replot',self)
		self.TwoD_log_toggle = QCheckBox('log',self)
		label1 = QLabel('nx = ')
		self.TwoD_nx_entry =  QLineEdit('100',self)
		label2 = QLabel('nxp = ')
		self.TwoD_nxp_entry =  QLineEdit('100',self)
		label3 = QLabel('ny = ')
		self.TwoD_ny_entry =  QLineEdit('100',self)
		label4 = QLabel('nyp = ')
		self.TwoD_nyp_entry = QLineEdit('100',self)
		optionslayout.addWidget(self.TwoD_replot_button)
		optionslayout.addWidget(self.TwoD_log_toggle)
		optionslayout.addWidget(label1)
		optionslayout.addWidget(self.TwoD_nx_entry)
		optionslayout.addWidget(label2)
		optionslayout.addWidget(self.TwoD_nxp_entry)
		optionslayout.addWidget(label3)
		optionslayout.addWidget(self.TwoD_ny_entry)
		optionslayout.addWidget(label4)
		optionslayout.addWidget(self.TwoD_nyp_entry)
		
		plotlayout = QHBoxLayout()    #QGridLayout()

		TwoDx_canvas = FigureCanvas(Figure(figsize=(5, 3)))
		plotlayout.addWidget(TwoDx_canvas)
		TwoDy_canvas = FigureCanvas(Figure(figsize=(5, 3)))
		plotlayout.addWidget(TwoDy_canvas)
		self.TwoDx_ax = TwoDx_canvas.figure.subplots()
		self.TwoDx_ax.set_position([0.17, 0.17, 0.80, 0.80])
		self.TwoDy_ax = TwoDy_canvas.figure.subplots()
		self.TwoDy_ax.set_position([0.17, 0.17, 0.80, 0.80])
		
		outerlayout.addLayout(optionslayout)
		outerlayout.addLayout(plotlayout)
		
		onegroupBox.setLayout(outerlayout)
		return onegroupBox
	
	###############################################################################
	## sub-functions and helpers
	###############################################################################	

	def rowsFromLattice(self):
		lat = latticegenerator.genLattice(beamline=self.beamline)
		self.max_sim_len = lat.accLattice.getLength()
		magdict = lat.magdict
		magstep = 0.01
		print(magdict)
		
		# -- make list of widgets (boxes and labels)
		self.quadLabels = []
		self.quadBoxes = []
		for i in range(len(magdict)):
			quadname = magdict.keys()[i]
			if magdict[quadname]['coeff'] != [0,0]:# ignore PMQ quads
				current = magdict[quadname].get("current",0.0)
				cb = QDoubleSpinBox(decimals=5,maximum =1000,\
									minimum =-1000,\
									singleStep = magstep)
				cb.setValue(current)
				self.quadBoxes.append(cb)
				lbl = QLabel(quadname)
				self.quadLabels.append(lbl)
				
		gradientSubBox = QGridLayout()				
		ncol = 3 # there will be 3 columns
		nrow = np.int(np.ceil(len(self.quadBoxes)/float(ncol)))
		print(nrow)

		# -- assemble row by row
		rows = []
		i = 0
		for k in range(nrow):
			j = 0
			while i < len(self.quadBoxes) and j < ncol:
				minirow = QHBoxLayout()
				minirow.addWidget(self.quadLabels[i]) 
				minirow.addWidget(self.quadBoxes[i])
				gradientSubBox.addLayout(minirow,k,j)
				i += 1
				j += 1
				
		return gradientSubBox	
	
	def resetBeamlines(self):
		
		# -- make list of beamlines in use based on GUI checkboxes
		beamline_list = []
		for cb in self.beamlineBoxes:
			if cb.isChecked():
				beamline_list.append(cb.text())
		self.beamline = beamline_list
		
		# -- remove old widgets for quad current entry
		for i in reversed(range(self.gradientSubLayout.count())): 
			for j in reversed(range(self.gradientSubLayout.itemAt(i).count())):
				self.gradientSubLayout.itemAt(i).itemAt(j).widget().setParent(None)
			self.gradientSubLayout.itemAt(i).setParent(None)
			
		# -- add new rows for quad current entry appropriate to beamline
		self.gradientSubLayout = self.rowsFromLattice() # make rows from lattice xml file
		self.G_layout.addLayout(self.gradientSubLayout)
		
		# -- reset default length
		self.text_S0.setText("0.0")
		self.text_S.setText(str(self.max_sim_len))
		self.text_Gfile.setText("")
		
	def selectionchange(self):
		if (self.sc_cb.currentText()=="FFT"):
				self.elps_N.setReadOnly(True)
		else:
				self.elps_N.setReadOnly(False)
				
				
	###############################################################################
	## PLotters
	###############################################################################	
	
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

