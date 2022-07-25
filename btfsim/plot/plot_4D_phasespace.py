import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

filedir = ''
filename = 'particles_xy_0131'



#%% Import
bunch1 = pd.read_csv(filedir+filename+'.txt',names=['x','xp','y','yp','z','de'],delimiter='\s+',index_col=False,skiprows=0)


print("N particles in bunch1: %i"%(len(bunch1)))

npmin = 92264; # number of particles in final bunch.
#%% plot initial distribution

# -- make histograms

# -- define bins based on both bunches
nbins=100
xmin = np.min([bunch1['x'].min(),bunch1['xp'].min()])
xpmin = xmin
ymin = np.min([bunch1['y'].min(),bunch1['yp'].min()])
ypmin = ymin
xmax = np.max([bunch1['x'].max(),bunch1['xp'].max()])
xpmax = xmax
ymax = np.max([bunch1['y'].max(),bunch1['yp'].max()])
ypmax = ymax

xbins = np.linspace(xmin,xmax,nbins)
xpbins = np.linspace(xpmin,xpmax,nbins)
ybins = np.linspace(ymin,ymax,nbins)
ypbins = np.linspace(ypmin,ypmax,nbins)

# x x'
[histxxp1,xedges,xpedges] = np.histogram2d(bunch1['x'],bunch1['xp'],bins=[xbins,xpbins])
# y y'
[histyyp1,yedges,ypedges] = np.histogram2d(bunch1['y'],bunch1['yp'],bins=[ybins,ypbins])
# x y
[histxy1,xedges,yedges] = np.histogram2d(bunch1['x'],bunch1['y'],bins=[xbins,ybins])
# x' y'
[histxpyp1,xpedges,ypedges] = np.histogram2d(bunch1['xp'],bunch1['yp'],bins=[xpbins,ypbins])

xbinwidth = (xedges[1]-xedges[0])/2.
ybinwidth = (yedges[1]-yedges[0])/2.
xpbinwidth = (xpedges[1]-xpedges[0])/2.
ypbinwidth = (ypedges[1]-ypedges[0])/2.

# -- set plot limits to +-10 mrad, +- 10 mm;
cmap = cm.get_cmap()
plxlim = 10; 
plylim = 10; 
plxplim= 10; 
plyplim = 10;


###############################################################################


# x y plots
fig = plt.figure(figsize=[10,8])

plt.subplot(2,2,1)
plt.pcolor(xedges[0:-1] + xbinwidth,yedges[0:-1] + ybinwidth ,histxy1.T); 
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-plxlim,plxlim]); plt.ylim([-plylim,plylim])
plt.colorbar()
ax = plt.gca()
ax.set_facecolor(cmap(0)) #  fill empty space with lower end of colormap

# x x' plots
plt.subplot(2,2,2)
plt.pcolor(xedges[0:-1] + xbinwidth,xpedges[0:-1] + xpbinwidth ,histxxp1.T); 
plt.xlabel('x')
plt.ylabel('xp')
plt.colorbar()
plt.xlim([-plxlim,plxlim]); plt.ylim([-plxplim,plxplim])
ax = plt.gca()
ax.set_facecolor(cmap(0))

# y y' plots
plt.subplot(2,2,3)
plt.pcolor(yedges[0:-1] + ybinwidth,ypedges[0:-1] + ypbinwidth ,histyyp1.T); 
plt.xlabel('y')
plt.ylabel('yp')
plt.colorbar()
plt.xlim([-plylim,plylim]); plt.ylim([-plyplim,plyplim])
ax = plt.gca()
ax.set_facecolor(cmap(0))

# x y plots
plt.subplot(2,2,4)
plt.pcolor(xpedges[0:-1] + xpbinwidth,ypedges[0:-1] + ypbinwidth ,histxpyp1.T); 
plt.xlabel('xp')
plt.ylabel('yp')
plt.colorbar()
plt.xlim([-plxplim,plxplim]); plt.ylim([-plyplim,plyplim])
ax = plt.gca()
ax.set_facecolor(cmap(0))

# -- savefig
fig.savefig(filename+'_initial.pdf',bbox_inches='tight')