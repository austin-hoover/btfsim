"""
Bunch Generation GUI
** Incomplete, just started setting layout **

This GUI facilitates the transformation of measured beam intensity data to macroparticle distributions.
Right now it is only set-up to read in 2D emittance scan data
As methods are developed for other dimensions, they should be added here.

K. Ruisard
2/18/19

"""

import sys
import time
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



class main_window_bunchgen(QWidget):
 
	def __init__(self):
		super(main_window_bunchgen,self).__init__()

		
		window = QWidget()
		lleftlayout = QVBoxLayout()    #QVBoxLayout()   QGridLayout()
		lrightlayout = QVBoxLayout()
		lowerlayout = QHBoxLayout()
		upperlayout = QHBoxLayout()
		mainlayout = QVBoxLayout()
		
		# add tabs here if needed
		
		upperlayout.addWidget(QPushButton('Generate'))
		
		
		
		rawplot = self.DistplotGroupBox("Intensity map from measured data")
		lrightlayout.addWidget(rawplot)
		lrightlayout.addWidget(QPushButton('Save Plot'))
		
		genplot = self.DistplotGroupBox("Distribution plots for generated bunch")
		lleftlayout.addWidget(genplot)
		lleftlayout.addWidget(QPushButton('Save Plot'))

		lowerlayout.addLayout(lleftlayout)
		lowerlayout.addLayout(lrightlayout)
		lowerlayout.setSpacing(2)
		lowerlayout.setContentsMargins(6,6, 6, 6)
		
		
		mainlayout.addLayout(upperlayout)
		mainlayout.addLayout(lowerlayout)
		self.setLayout(mainlayout)
		self.setWindowTitle("Bunch Generation GUI")
		self.show()
		
		
	def DistplotGroupBox(self,title):
		DistgroupBox = QGroupBox(title)
		plotlayout = QGridLayout()  
	   
		x_canvas = FigureCanvas(Figure(figsize=(2,2)))
		plotlayout.addWidget(x_canvas,0,0)
		y_canvas = FigureCanvas(Figure(figsize=(2,2)))
		plotlayout.addWidget(y_canvas,0,1)

		xp_canvas = FigureCanvas(Figure(figsize=(2,2)))
		plotlayout.addWidget(xp_canvas,1,0)
		yp_canvas = FigureCanvas(Figure(figsize=(2,2)))
		plotlayout.addWidget(yp_canvas,1,1)
		
		self.xplot = x_canvas.figure.subplots()
		self.xplot.set_position([.15, .15, 0.75, 0.75])
		self.yplot = y_canvas.figure.subplots()
		self.yplot.set_position([.15, .15, 0.75, 0.75])
		self.xpplot = xp_canvas.figure.subplots()
		self.xpplot.set_position([.15, .15, 0.75, 0.75])
		self.ypplot = yp_canvas.figure.subplots()
		self.ypplot.set_position([.15, .15, 0.75, 0.75])
		
		plotlayout.setContentsMargins(6,6,6,6)

		DistgroupBox.setLayout(plotlayout)
		return DistgroupBox
		
if __name__ == '__main__':
	app = QApplication(sys.argv)
	ttt = main_window_bunchgen()
	ttt.show()
	sys.exit(app.exec_())
				   
   