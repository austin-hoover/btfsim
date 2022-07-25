import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
try: 
	from bunch import Bunch
	classBunchFlag = 1 
except: 
	print('Plotter enabled without support for PyOrbit Bunch() class')
	classBunchFlag = 0
	
class plotBase(object):
	"""
	Base class for plotting routines
	"""
	def __init__(self,**kwargs):
		self.fontsize = kwargs.get('fontsize',14)
		self.xl = kwargs.get('xlim',None)
		self.yl = kwargs.get('ylim',None)
			
	def show(self):
		plt.show()
			
	def freezelimits(self,axis=None):
		if axis == 'x':
			self.xl = plt.xlim()
		elif axis == 'y':
			self.yl = plt.xlim()
		elif not(axis):
			self.xl = plt.xlim()
			self.yl = plt.ylim()
			
	def unfreezelimits(self,axis=None):
		if axis == 'x':
			self.xl = None
		elif axis == 'y':
			self.yl = None
		elif not(axis):
			self.xl = None
			self.yl = None
	
class plotMoments(plotBase):
	"""
	Plot history data for bunch (saved in simulation_main.tracker.hist)
	Takes in dictionary data on init
	"""
	def __init__(self,hist,**kwargs):
		self.hist = hist
		super(plotMoments,self).__init__(**kwargs)
		
		
	def plot(self,xvar,yvar,figure=None,show=True,label='',axis=None,**kwargs):
		"""
		Input:
		x is key to x-axis variable
		y is /list/ of keys for y-axis variables (has to be in list format)
		fignum=None; if not specified makes new figure
		show=True; if false does not show plot
		label=''; add characters to legend label
		**kwargs are keyword arguments for matplotlib.pyplot.plot
		"""
		# -- get figure window
		if figure:
			if type(figure)==int:
				fig = plt.figure(figure)
		else: 
			fig = plt.figure()
		
		# -- get axis if passed
		if axis:
			ax = axis
		else:
			ax = plt.gca()
		
		
		# -- plot line for each key in list:
		if isinstance(yvar,list):
			for y in yvar:
				ax.plot(self.hist[xvar],self.hist[y],label=label+y,**kwargs)
		else: ax.plot(self.hist[xvar],self.hist[yvar],label=label+yvar,**kwargs)
		
		# -- make legend      
		ax.legend()
		
		# -- other stuff
		ax.set_xlabel(xvar)
		plt.grid(True)
		plt.tight_layout()
		
		# -- show plot
		if show:
			plt.show()

class makeHist(object):
	"""
	Base class to make histogram out of dictionary
	"""
	def __init__(self,thisdict={}):
		self.thisdict = thisdict
		self.nparts = len(thisdict['x'])
	
	def hist1d(self,var,nbins=100,xlim=None,xmin=None,xmax=None,partial=[None]):
		
		# -- assign limits based on which named args are passed
		if (None in (xmax,xmin)) & bool(xlim):
			xmax = xlim; xmin = -xlim
		
		if not(xmin):
			xmin = self.thisdict[var].min()
		if not(xmax):
			xmax = self.thisdict[var].max()
		
		xbins = np.linspace(xmin,xmax,nbins)
		
		# -- make cuts
		ind = list(range(self.nparts))
		if any(partial):
			# -- pop off planes to plot
			planes = list(self.planes)
			planes.pop(planes.index(var))
			
			for i in range(len(planes)):
				plane = planes[i]
				if partial[i]:
					pbin = int(np.round(1./partial[i]))
					edges = np.histogram_bin_edges(self.thisdict[plane],bins=pbin)
					i1 = int(pbin/2)
					i2 = i1+1
					theseind = list(np.where((self.thisdict[plane] > edges[i1]) & (self.thisdict[plane] < edges[i2]))[0])
					intersectind = [value for value in ind if value in theseind]
					ind = list(intersectind)
				else:
					pass 
		
		# -- make hist
		[thishist,xedges] = np.histogram(self.thisdict[var][ind],bins=xbins)
		
		return thishist,xedges
		
	def hist2d(self,xvar,yvar,nbins=100,xlim=None,ylim=None,xmin=None,xmax=None,ymin=None,ymax=None,partial=[None]):
		"""
		Partial = [None]; Partial is list of cuts to take in other projected planes (for the purpose of partial projection)
		ex: partial = [.3,.3,.3,None] will take middle third of beam in planes 0-3, and no cut in plane 4 (1 is equivalent to None)
		plane = ['x','xp','y','yp','z',de'].pop(plane1).pop(plane2)
		"""
		
		# -- assign limits based on which named args are passed
		if (None in (xmax,xmin)) & bool(xlim):
			xmax = xlim; xmin = -xlim
		if (None in (ymax,ymin)) & bool(ylim):
			ymax = ylim; ymin = -ylim
		
		if not(xmin):
			xmin = self.thisdict[xvar].min()
		if not(xmax):
			xmax = self.thisdict[xvar].max()
		if not(ymin):
			ymin = self.thisdict[yvar].min()
		if not(ymax):
			ymax = self.thisdict[yvar].max()
		
		# -- assign number of bins
		if isinstance(nbins,list):
			nxbins=nbins[0]
			nybins=nbins[1]
		else:
			nxbins=nbins
			nybins=nbins
		
		# -- make bins
		xbins = np.linspace(xmin,xmax,nxbins)
		ybins = np.linspace(ymin,ymax,nybins)
		
		# -- make cuts
		ind = np.arange(self.nparts)
		#print('plotting planes %s, %s'%(xvar,yvar))
		if any(partial):
			# -- only slice  in planes that aren't getting plotted
			planes = list(self.planes)
			planes.pop(planes.index(xvar))
			planes.pop(planes.index(yvar))
			
			for i in range(len(planes)):
				plane = planes[i]
				if partial[i]:
					pbin = int(np.round(1./partial[i]))
					#print('  slicing in plane %s, center of %i bins'%(plane,pbin))
					edges = np.histogram_bin_edges(self.thisdict[plane],bins=pbin)
					i1 = int(pbin/2)
					i2 = i1+1
					theseind = np.where((self.thisdict[plane][ind] > edges[i1]) & (self.thisdict[plane][ind] < edges[i2]))[0]
					ind = ind[theseind]
				else:
					pass 

		# -- make hist
		[thishist,xedges,yedges] = np.histogram2d(self.thisdict[xvar][ind],self.thisdict[yvar][ind],bins=[xbins,ybins])
		
		return thishist,xedges,yedges
			
class plotBunch(plotBase,makeHist):
	"""
	Plot histograms of bunch
	init takes bunch as argument
	"""
	def __init__(self,bunchInstance,**kwargs):
		
		self.planes = kwargs.pop('planes',['x','y'])
		
		# -- load data into dictionary 'bunch'
		if isinstance(bunchInstance,basestring): # if bunchObj is filename
			# should replace with non-pandas option
			bunch = np.genfromtxt(bunchInstance,skip_header=14,names='x,xp,y,yp,z,de')
			print("N particles in bunch: %i"%(len(bunch)))
		elif classBunchFlag:
			if isinstance(bunchInstance,Bunch):
				N = bunchInstance.getSizeGlobal()
				print("N particles in bunch: %i"%N)
				bunch = {'x':np.zeros(N),'y':np.zeros(N),'xp':np.zeros(N),'yp':np.zeros(N),'z':np.zeros(N),'de':np.zeros(N)}
				for i in range(N):
					bunch['x'][i] = bunchInstance.x(i)
					bunch['xp'][i] = bunchInstance.xp(i)
					bunch['y'][i] = bunchInstance.y(i)
					bunch['yp'][i] = bunchInstance.yp(i)
					bunch['z'][i] = bunchInstance.z(i)
					bunch['de'][i] = bunchInstance.dE(i)
			else: # raise error if bunch type is not known.
				raise TypeError('Do not recognize type %s for bunchInstance'%type(bunchInstance))
		else: # raise error if bunch type is not known.
			raise TypeError('Do not recognize type %s for bunchInstance'%type(bunchInstance))
			
		# -- inherit
		super(plotBunch,self).__init__(**kwargs)
		
		# -- adjust scale
		bunch['x'] *= 1e3 # m to mm
		bunch['xp'] *= 1e3 # rad to mrad
		bunch['y'] *= 1e3 # m to mm
		bunch['yp'] *= 1e3 # rad to mrad
		bunch['de'] *= 1e6 # GeV to keV
		
		self.thisdict = bunch
		self.nparts = len(bunch['x'])
		
	def plot1d(self,var,fignum=None,show=True,nbins=100,xlim=None,xmin=None,xmax=None,label='',logscale=None,axis=None):
		"""
		Make 1d histogram plot
		input:
		var = string indicating variable to plot. Right now can only be in group
		"""
		
		if (None in (xmax,xmin)) & bool(xlim):
			xmax = xlim; xmin = -xlim
		
		# -- get figure window
		if fignum:
			if type(fignum)==int:
				fig = plt.figure(fignum)
		else: 
			fig = plt.figure()
		
		# -- get axis if passed
		if axis:
			ax = axis
		else:
			ax = plt.gca()
			
		
		# - make histogram
		if isinstance(var,list):
			for x in var:
				hist,edges = self.hist1d(x,nbins=nbins,xmin=xmin,xmax=xmax)
				hist = np.hstack([[0],hist])
				if logscale: hist = np.log10(hist)
				# -- plot histogram
				plt.step(edges,hist,label=label+x)
		else:
			hist,edges = self.hist1d(var,nbins=nbins,xmin=xmin,xmax=xmax)
			hist = np.hstack([[0],hist])	
			if logscale: hist = np.log10(hist)
			# -- plot histogram
			plt.step(edges,hist,label=label+var)
		
		# -- formatting
		plt.xlabel(var)
		plt.ylabel('particle counts')	
		plt.legend()

		# -- set plot limits if specified
		if self.xl:
			plt.ylim(self.yl)
		if self.yl:
			plt.xlim(self.xl)
			
		plt.rcParams.update({'font.size': self.fontsize})
		
		if show:
			plt.show()

		
	def plot2d(self,xvar,yvar,figure=None,axis=None,show=True,nbins=100,xlim=None,ylim=None,xmin=None,xmax=None,ymin=None,ymax=None,logscale=False,colorbarFlag=True):	
		"""
		Make 1d histogram plot
		input:
		xvar = string indicating variable to plot. 
		yvar = ""
		"""
		# -- assign limits based on which named args are passed
		if (None in (xmax,xmin)) & bool(xlim):
			xmax = xlim; xmin = -xlim
		if (None in (ymax,ymin)) & bool(ylim):
			ymax = ylim; ymin = -ylim
		
		# -- get figure window
		if figure:
			if type(figure)==int:
				fig = plt.figure(figure)
		else: 
			fig = plt.figure()

		# -- get axis if passed
		if axis:
			ax = axis
		else:
			ax = plt.gca()

			
		# - make histogram
		hist,xedges,yedges = self.hist2d(xvar,yvar,nbins=nbins,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
		xbinwidth = (xedges[1]-xedges[0])/2.
		ybinwidth = (yedges[1]-yedges[0])/2.
		hist /= hist.max()

                # -- pcolor plot histogram
		x = xedges[0:-1] + xbinwidth
		y = yedges[0:-1] + ybinwidth
		if logscale: hist = np.log10(hist)
		imap = ax.pcolor(x,y,hist.T)
		
		ax.set_xlabel(xvar)
		ax.set_ylabel(yvar)
		
	
		# -- set plot limits if specified
		if self.xl:
			ax.set_xlim(self.xl)
		if self.yl:
			ax.set_ylim(self.yl)

		cmap = cm.get_cmap()
		ax.set_facecolor(cmap(0))
		if colorbarFlag:
			plt.colorbar(imap,ax=ax)

		plt.rcParams.update({'font.size': self.fontsize})
		plt.tight_layout()
		
		if show:
			plt.show()
		
		
