import sys
import math
import numpy as np
from PyQt5.QtWidgets import  QProgressBar,QProgressDialog

class Twiss_cal_threshold:
	def __init__(self,filename,threshold,g_h,g_v):

		e_kin_ini = 0.0025 # in [GeV]
		mass = 0.939294    # in [GeV]
		gamma = (mass + e_kin_ini)/mass
		beta = math.sqrt(gamma*gamma - 1.0)/gamma
		#print "relat.  beta=",beta
		frequency = 402.5e+6
		v_light = 2.99792458e+8  # in [m/sec]
		lamda = v_light/frequency
		betalamda = beta* lamda
		# print "################################################",filename



		TR = threshold		
		grid_h = g_h
		grid_v = g_v
		print "threshold =",TR
		

		data_output = np.loadtxt(filename,skiprows=(14))
		x1 = np.array(data_output[:,0])*1000    #Units of x and y are m in output file, here we transvert to mm
		xp1= np.array(data_output[:,1])*1000      #Units of xp and yp are rad in output file
		y1 = np.array(data_output[:,2])*1000
		yp1= np.array(data_output[:,3])*1000 
		dz = np.array(data_output[:,4])
		dE = np.array(data_output[:,5])*1000000


		dz2_avg=np.mean(dz**2)     # get column mean value
		dE2_avg=np.mean(dE**2)
		zE_avg=np.mean(dz*dE)
		ez_rms=math.sqrt(dz2_avg*dE2_avg-zE_avg**2)
		ez_phi_rms = (360/betalamda)*ez_rms              #Unit: deg-keV
		alphaz= -zE_avg/ez_rms
		betaz = (dz2_avg/ez_rms)*(360/betalamda)      #Unit:   deg/keV
		
		xmin = np.min(x1)-0.05
		xmax = np.max(x1)+0.05
		xpmin = np.min(xp1)-0.05
		xpmax = np.max(xp1)+0.05

		ymin = np.min(y1)-0.05
		ymax = np.max(y1)+0.05
		ypmin = np.min(yp1)-0.05
		ypmax = np.max(yp1)+0.05

		#=======================================================================================================================
		#================================================================  X-Plane calculation
		index_counter1 = [0]*grid_v*grid_h
		x1_n = (x1- xmin)/(xmax -xmin)*grid_h   # Normalize the coordinates in x plane
		xp1_n = (xp1- xpmin)/(xpmax -xpmin)*grid_v   # Normalize the coordinates in xp plane
		index1 = map(int, (grid_h*np.floor(xp1_n)+np.floor(x1_n)).tolist())  #Make the normalized coordinates equal to grids number ; There may be many coordinats of particles that in one grid, or may be zero particles.

		# #=========================================================================================
		# #=========================================== Y-plane calculation
		indey_counter1 = [0]*grid_v*grid_h
		y1_n = (y1- ymin)/(ymax -ymin)*grid_h   # Normalize the coordinates in x plane
		yp1_n = (yp1- ypmin)/(ypmax -ypmin)*grid_v   # Normalize the coordinates in xp plane
		indey1 = map(int, (grid_h*np.floor(yp1_n)+np.floor(y1_n)).tolist())  #Make the normalized coordinates equal to grids number ; There may be many coordinats of particles that in one grid, or may be zero particles.

		self.x1 = x1
		self.xp1 = xp1
		self.x1_n = x1_n
		self.xp1_n = xp1_n
		self.index1 = index1
		self.index_counter1 = index_counter1

		self.y1 = y1
		self.yp1 = yp1
		self.y1_n = y1_n
		self.yp1_n = yp1_n
		self.indey1 = indey1
		self.indey_counter1 = indey_counter1

		self.alphaz = alphaz
		self.betaz = betaz
		self.ez_phi_rms = ez_phi_rms

		self.xmin = xmin
		self.xmax =xmax
		self.xpmin = xpmin 
		self.xpmax = xpmax
		self.ymin = ymin
		self.ymax =ymax
		self.ypmin = ypmin 
		self.ypmax = ypmax
		self.part_N = len(x1)

	def data_particle(self):
		return self.x1, self.xp1,self.x1_n,self.xp1_n,self.index1,self.index_counter1, \
				self.y1, self.yp1,self.y1_n,self.yp1_n,self.indey1,self.indey_counter1,\
				self.alphaz,self.betaz,self.ez_phi_rms,self.xmin,self.xmax,self.xpmin,self.xpmax,\
				self.ymin,self.ymax,self.ypmin,self.ypmax,self.part_N


class OneD_data_cal:
	def __init__(self,filename,P_JD,tR):
		if (P_JD == True and tR == 0):		
			data_output = np.loadtxt(filename,skiprows=(14))
			x = np.array(data_output[:,0])*1000 
			y = np.array(data_output[:,2])*1000
		else:
			data_output = np.loadtxt(filename,skiprows=(1))
			x = np.array(data_output[:,0]) 
			y = np.array(data_output[:,2])
		#print "OK",len(x)

		dr = 0.3

		x_max=np.max(x)
		#print x_max
		x_min=np.min(x)
		tmpx = np.arange(x_min-0.1, x_max+0.1, dr)
		countx = [0]*len(tmpx)
		countx = np.histogram(x, tmpx)[0]

		y_max=np.max(y)
		y_min=np.min(y)
		tmpy = np.arange(y_min-0.1, y_max+0.1, dr)
		county = [0]*len(tmpy)
		county = np.histogram(y, tmpy)[0]

		x1 = np.array(zip(*zip(tmpx,countx))[0])
		x2 = np.array(zip(*zip(tmpx,countx))[1])
		y1 = np.array(zip(*zip(tmpy,county))[0])
		y2 = np.array(zip(*zip(tmpy,county))[1])

		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2

	def cords(self):
		return self.x1,self.x2,self.y1,self.y2







		





