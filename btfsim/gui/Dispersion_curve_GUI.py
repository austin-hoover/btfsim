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

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from btfsim.sim.pyorbit_1Particle_class import dispersion_curve_cal
import random
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter,NullLocator

class Window_dispersion(QWidget):
 
    def __init__(self):
        super(Window_dispersion,self).__init__()

        layout = QVBoxLayout()     

        Glayout = QGridLayout()
        Rlayout = QVBoxLayout()
        Uplayout = QHBoxLayout()

        Glayout.addWidget(self.GradientGroupBox())
        Rlayout.addWidget(self.RunningGroupBox())
        Uplayout.addLayout(Glayout)
        Uplayout.addLayout(Rlayout)
        layout.addLayout(Uplayout)
        layout.addWidget(self.DispersionGroupBox())
        layout.addWidget(self.Cord_after_bends())
        
        self.setLayout(layout)
        self.setWindowTitle("Achromatic_Lattice_Design_GUI")
        self.show()
   
    def GradientGroupBox(self):
        QGgroupBox = QGroupBox("Set Quads Gradients")
        G_layout = QVBoxLayout()   
        self.label_g0 = QLabel('*Read from file ("Achromatic_lattice_field.txt")', self)
        self.button_g = QPushButton('Select',self)
        self.button_g.clicked.connect(self.on_click2)
        self.label_g10 = QLabel('*Manual Input (Unit: T)', self)

        self.quads_save = QPushButton('Save_Quads',self)
        self.quads_save.clicked.connect(self.savequads)

        self.label_Q7 = QLabel('QV07')
        self.label_Q8 = QLabel('QH08')
        self.label_Q9 = QLabel('QV09')
        
        self.text_Gfile = QLineEdit(self)         # for gradients input file
        Magstep = 0.0001

        select_row =  QHBoxLayout()
        spacer = QSpacerItem(30, 1, QSizePolicy.Maximum)
        select_row.addItem(spacer)
        select_row.addWidget(self.text_Gfile)
        select_row.addWidget(self.button_g) 
        self.sp7 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
        self.sp8 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
        self.sp9 =  QDoubleSpinBox(decimals=5,maximum =10, minimum =0,singleStep = Magstep)
        row789 = QHBoxLayout() 
        row789.addWidget(self.label_Q7)
        row789.addWidget(self.sp7)
        row789.addWidget(self.label_Q8)
        row789.addWidget(self.sp8)
        row789.addWidget(self.label_Q9)
        row789.addWidget(self.sp9)
        
        row_manual = QHBoxLayout()
        row_manual.addWidget(self.label_g10)
        spacerSQ = QSpacerItem(100, 1, QSizePolicy.Maximum) 
        row_manual.addItem(spacerSQ)
        row_manual.addWidget(self.quads_save)
        spacer = QSpacerItem(60, 1, QSizePolicy.Maximum)
        row_manual.addItem(spacer)

        G_layout.addWidget(self.label_g0)
        G_layout.addLayout(select_row)
        G_layout.addLayout(row_manual)
        G_layout.addLayout(row789)

        G_layout.setSpacing(2)   
        QGgroupBox.setLayout(G_layout)
        return QGgroupBox
        
    def RunningGroupBox(self):
        CgroupBox = QGroupBox("Running Settings")
        clayout = QVBoxLayout()  #  QVBoxLayout()

        self.button_Ini = QPushButton('Default Gradients',self)
        self.button_Ini.clicked.connect(self.on_click3)
        self.button_run = QPushButton('Run',self)
        self.button_run.clicked.connect(self.on_click4)
        self.E_dis = QLineEdit(self)
        self.E_dis.setFixedWidth(50)
        self.label_Edis1 = QLabel('E_discrepancy:',self)
        self.label_Edis2 = QLabel('MeV',self)
        spacerE = QSpacerItem(60, 1, QSizePolicy.Maximum) 
        elayout = QHBoxLayout()
        elayout.addItem(spacerE)
        elayout.addWidget(self.E_dis)
        elayout.addWidget(self.label_Edis2)
        Etlayout = QVBoxLayout()
        Etlayout.addWidget(self.label_Edis1)
        Etlayout.addLayout(elayout)
        Etlayout.setSpacing(0)

        clayout.addLayout(Etlayout)
        clayout.addWidget(self.button_Ini)
        clayout.addWidget(self.button_run)

        CgroupBox.setLayout(clayout)
        return CgroupBox

    def DispersionGroupBox(self):
        SPgroupBox = QGroupBox("Single Particle Trajectories")
        playout = QGridLayout()   #  QHBoxLayout 
       
        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        playout.addWidget(static_canvas,1,1)
        self._static_ax = static_canvas.figure.subplots()
        self._static_ax.set_position([0.08, 0.22, 0.89, 0.7])
        self.toolbar = NavigationToolbar(static_canvas, self)

        playout.addWidget(self.toolbar,0,1)

        SPgroupBox.setLayout(playout)
        return SPgroupBox

    def plot_beam_size(self):
        self._static_ax.clear()
        data_extre = np.genfromtxt('Dispersion_cal_bunch_extrema.txt', delimiter='', comments=None)
        x_extre1 = map(float, zip(*data_extre)[1])
        x_extre2 = map(float, zip(*data_extre)[2])
        self._static_ax.plot(x_extre1,x_extre2,'r-')
        self._static_ax.set_xlabel('Position (cm)')
        self._static_ax.set_ylabel('x (cm)')
        # self._static_ax.yaxis.set_major_locator(NullLocator())   # Hiding ticks and lable 
        self._static_ax.tick_params(direction='in')
        self._static_ax.grid()    
        # self._static_ax.set_title('X,Y Beam Size')
        self._static_ax.figure.canvas.draw()
    def Cord_after_bends(self):
        CordgroupBox = QGroupBox("Single Particle Coordinates")
        corlayout = QGridLayout()  #  QVBoxLayout()
        self.label_i = QLabel('Cordinates before bends')
        self.label_o = QLabel('Cordinates after bends')
        self.label_x = QLabel('x (m)')
        self.label_xp = QLabel('xp (rad)')
        self.xi = QLineEdit(self)
        self.xpi = QLineEdit(self)
        self.x = QLineEdit(self)
        self.xp = QLineEdit(self)
        corlayout.addWidget(self.label_x,0,1,Qt.AlignCenter)
        corlayout.addWidget(self.label_xp,0,2,Qt.AlignCenter)
        corlayout.addWidget(self.label_i,1,0)
        corlayout.addWidget(self.label_o,2,0)
        corlayout.addWidget(self.xi,1,1)
        corlayout.addWidget(self.xpi,1,2)
        corlayout.addWidget(self.x,2,1)
        corlayout.addWidget(self.xp,2,2)
        CordgroupBox.setLayout(corlayout)
        return CordgroupBox

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
                print('Read Gradients File.')
                fileRead = open(fileName,'r+')
                line1 = fileRead.readlines()[0]
                if ( line1.strip() != "Achromatic_lattice_Gradients (Unit:T)"):
                    QMessageBox.about(self, "", "Wrong file!!!")
                else:
                    Grad = np.genfromtxt(fileName, dtype = float,delimiter='', skip_header=1,comments=None,usecols = (1))
                    self.sp7.setValue(float(Grad[0]))   #'%.4f' %alphax))
                    self.sp8.setValue(float(Grad[1]))
                    self.sp9.setValue(float(Grad[2]))
            else:
                print('No clicked.')

    def on_click3(self):   #======================================Defaut setting!
  
        PATH='./Achromatic_lattice_field_default.txt'
        if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
            print "File exists and is readable"

            Field = np.genfromtxt('Achromatic_lattice_field_default.txt', dtype = float,delimiter='', skip_header=1,comments=None,usecols = (1))
            self.sp7.setValue(float(Field[0]))   #'%.4f' %alphax))
            self.sp8.setValue(float(Field[1]))
            self.sp9.setValue(float(Field[2]))
        else:
            QMessageBox.about(self, "", "'Achromatic_lattice_field_default.txt' is missing or is not readable!")
        self.E_dis.setText("0.025")
        self.E_dis.repaint()
        self.text_Gfile.setText("")

    def on_click4(self):       #=================================================Run
        if (self.E_dis.text() == ""):
            self.E_dis.setText("0.025")
        if (float(self.sp7.text()) == 0 and float(self.sp8.text()) ==0 and float(self.sp9.text())==0):
            QMessageBox.about(self, "", "No Gradients Entry!")
        else:        
            a = dispersion_curve_cal(self.sp7.text(),self.sp8.text(),self.sp9.text(),self.E_dis.text())
            self.plot_beam_size()
            self.output()
            self.sp7.repaint()
    def output(self):
        data_output = np.loadtxt("Dispersion_cal_Output.txt",skiprows=(14))
        #print "==========================",len(data_output)
        if (len(data_output) ==0):
            QMessageBox.about(self, "", "Particle lost!")
            self.x.setText("")
            self.xp.setText("")

        else:
            x1 = data_output[0]   
            xp1= data_output[1]
            self.xi.setText("0") 
            self.xpi.setText("0") 
            self.x.setText(str(x1)) 
            self.xp.setText(str(xp1))  

    def savequads(self):                   # Save Quads Gradients
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            FileforQuads = open(fileName,"w+")
            FileforQuads.write("Achromatic_lattice_Gradients (Unit:T)" +'\n')
            FileforQuads.write("QV07   " + "%6.5f"%float(self.sp7.text()) +'\n')
            FileforQuads.write("QH08   " + "%6.5f"%float(self.sp8.text()) +'\n')
            FileforQuads.write("QV09   " + "%6.5f"%float(self.sp9.text()) +'\n')     

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ttt = Window_dispersion()
    ttt.show()
    sys.exit(app.exec_())


