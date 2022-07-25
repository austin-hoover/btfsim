#!/usr/bin/env python
# coding: utf-8

# This translates A.Aleksandrov's 4D bunch generating script from matlab to python. 
# This is used to take measured distributions from independent 2D emittance measurements 
# to a macroparticle distribution for simulation.
#
# Commandline usage requires two input arguments: 
# generate4Dbunch.py -x <data file from x-scan> -y <data file from x-scan>
# optional: [-t <threshold (default 3e-4)>]
#
# K. Ruisard
#

import numpy as np
import argparse

################
## User Input ##
################

threshold_default = 3e-4 # default threshold value
filedir = '' # might be useful?

outfilename = filedir + 'particles_xy.txt' 

# -- define command line arguments
parser = argparse.ArgumentParser()
xhelpstring = 'name of text file containing measured x-xp intensity from 4D emittance GUI'
parser.add_argument('-x','--xscan-file',help=xhelpstring,required=True)
yhelpstring = 'name of text file containing measured y-yp intensity from 4D emittance GUI'
parser.add_argument('-y','--yscan-file',help=yhelpstring,required=True)
thelpstring = 'threshold (fraction of peak) for noise floor'
parser.add_argument('-t','--threshold',help=thelpstring,required=False,default=threshold_default)

# -- parse arguments
args = parser.parse_args()
xfilename = filedir + args.xscan_file
yfilename = filedir + args.yscan_file
threshold = args.threshold

###############################
## Import meas. distribution ##
###############################
#%%
# -- import data
xscandata = np.loadtxt(xfilename)
yscandata = np.loadtxt(yfilename)

# -- grab ranges from 1st row + 1st column
xaxis  = xscandata[0,1:]
xpaxis = xscandata[1:,0]

yaxis  = yscandata[0,1:]
ypaxis = yscandata[1:,0]

# -- trim 1st column + row from data
xscandata = xscandata[1:,1:]
yscandata = yscandata[1:,1:]

[XG,XPG] = np.meshgrid(xaxis,xpaxis)
[YG,YPG] = np.meshgrid(yaxis,ypaxis)

# -- remove bias
xscandata = xscandata - xscandata[-1,:].mean()
yscandata = yscandata - yscandata[-1,:].mean()

# -- set cutoff values based on peak
xcutoff = xscandata.max()*threshold
ycutoff = yscandata.max()*threshold

# -- set values below threshold to 0:
xscandata=xscandata.clip(xcutoff)-xcutoff
yscandata=yscandata.clip(ycutoff)-ycutoff

# -- define normalized distribution
xdistr = xscandata/xscandata.sum()
ydistr = yscandata/yscandata.sum()

# -- find centers
xavg = np.sum(np.sum(XG*xdistr));
xpavg = np.sum(np.sum(XPG*xdistr));

yavg = np.sum(np.sum(YG*ydistr));
ypavg = np.sum(np.sum(YPG*ydistr));



###########################
## Generate distribution ##
###########################

Nmax = 10000 # number of macro-particles where prob. distr = 1
Nxdistr = np.floor(xdistr*Nmax).astype(int)
Nxparts = Nxdistr.sum()
Nydistr = np.floor(ydistr*Nmax).astype(int)
Nyparts = Nydistr.sum()


# -- loop through grid + deposit Ndistr(i,j) particles at each point
# could be combined into 1 loop (do x and y simultaneously), 
# but left general in case x and y scans have different resolutions

XD,XPD = np.zeros([2,Nxparts])
counterx = 0
for i in range(np.shape(Nxdistr)[0]):
    for j in range(np.shape(Nxdistr)[1]):
        if ( Nxdistr[i,j] > 0 ):
            counterendx = counterx + Nxdistr[i,j]
            XD[counterx:counterendx] = XG[i,j]
            XPD[counterx:counterendx] = XPG[i,j]
            counterx = counterendx
            
YD,YPD = np.zeros([2,Nyparts])            
countery = 0
for i in range(np.shape(Nydistr)[0]):
    for j in range(np.shape(Nydistr)[1]):            
        if ( Nydistr[i,j] > 0 ):
            counterendy = countery + Nydistr[i,j]
            YD[countery:counterendy] = YG[i,j]
            YPD[countery:counterendy] = YPG[i,j]
            countery = counterendy
            
# -- spread particles out from gridpoint locations through a uniform kernel
# right now width of each kernel is twice grid spacing, is this intentional?
xstep  = XG[0,1]  - XG[0,0]
xpstep = XPG[1,0] - XPG[0,0]
xd  = XD + (np.random.rand(1,Nxparts) - 0.5)*2*xstep - xavg
xpd = XPD + (np.random.rand(1,Nxparts) - 0.5)*2*xpstep - xpavg

ystep  = YG[0,1]  - YG[0,0]
ypstep = YPG[1,0] - YPG[0,0]
yd  = YD + (np.random.rand(1,Nyparts) - 0.5)*2*ystep - yavg
ypd = YPD + (np.random.rand(1,Nyparts) - 0.5)*2*ypstep - ypavg


###################
## Save 4D bunch ##
###################

npdesired = 94500

# -- x and y distributions are different lengths; trim longer one
npmin = np.min([npdesired,np.shape(xd)[1],np.shape(yd)[1]])

# -- random sampling to remove correlations in x-y, xp-yp
xind = np.random.permutation(range(np.shape(xd)[1]))
xind = xind[0:npmin]

yind = np.random.permutation(range(np.shape(yd)[1]))
yind = yind[0:npmin]

# -- make numpy array
particle_array = np.vstack((xd[0,xind],xpd[0,xind],yd[0,yind],ypd[0,yind]))
particle_array = np.transpose(particle_array)

# -- save
np.savetxt(outfilename,particle_array,fmt='%.7f',delimiter=' ')
                           

