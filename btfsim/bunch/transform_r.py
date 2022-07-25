# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import sys
sys.path.append('/media/sf_OracleShared/soft/btf-pyorbit-simulation/btfsim/util/')
import Defaults as Defaults
default = Defaults.getDefaults()

dr = 0.05
plotflag = 1
filename = '../../data/bunches/Distribution_Output.txt'
bb = np.loadtxt(filename,skiprows=14)

# Gaussian function for fitting
def gaus(gx,*p):
    A,sigma=p                                                             
    mu = 0.
    return A*np.exp(-(gx-mu)**2/(2.0*sigma**2))


x = np.array(bb[:,0])*10.  #first column,data unit in 'Input.txt' is centimeter
xp= np.array(bb[:,1])*1000.      #second column
y = np.array(bb[:,2])*10.   #third column,data unit in 'P_test.txt' is centimeter
yp= np.array(bb[:,3])*1000.       #fourth column


###############################################################################
## Calculate RMS Twiss params
###############################################################################

# -- RMS Twiss calculation 
x2 = x*x  
xp2 = xp*xp 
y2 = y*y  
yp2 = yp*yp 

x2_avg=np.mean(x2)     # get column mean value
xp2_avg=np.mean(xp2)
xxp_avg=np.mean(x*xp)
y2_avg=np.mean(y2)
yp2_avg=np.mean(yp2)
yyp_avg=np.mean(y*yp)

ex_rms=np.sqrt(x2_avg*xp2_avg-xxp_avg**2)
alphax= -xxp_avg/ex_rms
betax = x2_avg/ex_rms
print("ex=",ex_rms)
print("alphax=", alphax)
print("betax=", betax)

ey_rms=np.sqrt(y2_avg*yp2_avg-yyp_avg**2)
alphay= -yyp_avg/ey_rms
betay = y2_avg/ey_rms
print("ey=",ey_rms)
print("alphay=", alphay)
print("betay=", betay)

##
if plotflag: 
    CS1=plt.hist2d(x, xp, bins=100,norm=LogNorm())
    plt.savefig(default.outdir+'Hist_X')
    plt.show() # clf
    
    CS2=plt.hist2d(y, yp, bins=100,norm=LogNorm())
    plt.xlabel('y (mm)')
    plt.ylabel('yp (mrad)')
    plt.savefig(default.outdir+'Hist_Y') 
    plt.show() # clf  

###############################################################################
## Transform to normalized coodrinates by Twiss Params
###############################################################################

xn = x/np.sqrt(betax)
xnp = alphax*x/np.sqrt(betax)+xp*np.sqrt(betax)
rx = np.sqrt(xn**2+xnp**2)    #Radial x
rx_max = np.max(rx)
rx_min = np.min(rx)
print("rx_max=", rx_max)
print("rx_min=", rx_min)

if plotflag:
    CS1_1=plt.hist2d(xn, xnp, bins=100,norm=LogNorm())
    plt.xlabel('x')
    plt.ylabel('xp')
    plt.title('Transformed x-coordinates')
    plt.savefig(default.outdir+'Input_X_Trans')
    plt.show()#clf


yn =y/np.sqrt(betay)
ynp=alphay*y/np.sqrt(betay)+yp*np.sqrt(betay)
ry=np.sqrt(yn**2+ynp**2)    #Radial y
ry_max=np.max(ry)
ry_min=np.min(ry)
print("ry_max=", ry_max)
print("ry_min=", ry_min)

if plotflag:
    CS2_1=plt.hist2d(yn, ynp, bins=100,norm=LogNorm())
    plt.savefig(default.outdir+'Input_Y_Trans')
    plt.show() #clf

tr = np.array([xn,xnp,yn,ynp])
ma = tr.T     #Transformed_cood.
np.savetxt(default.outdir+"Transformed_cood.txt",ma)   #Transformed_cood.
Rxy=np.vstack([rx,ry]).T
np.savetxt(default.outdir+"R_xy.txt",Rxy)

##
###############################################################################
## Calculate density and Gaussian Fit
###############################################################################

tmpx = np.arange(0, rx_max+dr*2, dr)
circx = [0]*len(tmpx)
countx = np.histogram(rx, tmpx)[0]
densx = [countx[i]*1.0/(2*np.pi*(tmpx[i]+dr*0.5)*dr) for i in range(len(countx))]
densx_max=np.max(densx)

fx1 = np.delete(tmpx, 0)           
fx2= densx/densx_max
p0=[1.,.1]
coeff,pcov=curve_fit(gaus,fx1,fx2,p0=p0)
x_fit=gaus(fx1,*coeff)


tmpy = np.arange(0, ry_max+dr*2, dr)
circy = [0]*len(tmpy)
county = np.histogram(ry, tmpy)[0]
densy = [county[i]*1.0/(2*np.pi*(tmpy[i]+dr*0.5)*dr) for i in range(len(county))]
densy_max=np.max(densy)



fy1 = np.delete(tmpy, -1)           
fy2= densy/densy_max
p0=[1.,.01]
coeff,pcov=curve_fit(gaus,fy1,fy2,p0=p0)
y_fit=gaus(fy1,*coeff)

output_arrx = np.vstack([tmpx[1:],countx,densx/densx_max,x_fit]).T   # Particle number in per radial band
np.savetxt(default.outdir+'X_norm_density.txt', output_arrx, 
           fmt = '%.6f', 
           header = '## r [mm], count (btw r and r-dr), norm. density, Gauss fit')
output_arry = np.vstack([tmpy[1:],county,densy/densy_max,y_fit]).T   # Particle number in per radial band
np.savetxt(default.outdir+'Y_norm_density.txt', output_arry, 
           fmt = '%.6f', 
           header = '## r [mm], count (btw r and r-dr), norm. density, Gauss fit')
      
if plotflag: 
    plt.figure()
    plt.plot(fx1,fx2,'o-',fx1,x_fit)
    plt.yscale('log')
    plt.xlabel('Radius')
    plt.ylabel('Norm_density')
    plt.ylim([1e-6,1.1])
    plt.grid()
    plt.grid(b=True, which='minor') #, color='r', linestyle='--')
    plt.savefig('Norm_density_X')
    plt.show()
    
    plt.figure()
    plt.plot(fy1,fy2,'o-',fy1,y_fit)
    plt.yscale('log')
    plt.xlabel('Radius')
    plt.ylabel('Norm_density')
    plt.ylim([1e-6,1.1])
    plt.grid()
    plt.grid(b=True, which='minor') #, color='r', linestyle='--')
    plt.savefig('Norm_density_Y')
    plt.show()
            
