from btfsim.plot.Plotter import plotBunch
from bunch import Bunch
import numpy as np

class MovieBase():
	"""
	input arguments on init 
	
	(unnamed):
	savefolder 
	
	(named):
	planes=['x','y'], which 2D projections to plot. Must specify at least 2
	nbins = 50, resolution of histogram. 
	limits = None, None], symmetric limits to plots. (plane corresponds to list specified as 'planes')
			if None, limits are not held fixed between frames
	nbins = 50, number of bins. can be scalar or list of scalars

	"""
	def __init__(self,savefolder,planes=['x','y'],nbins=50,limits=None,partial=None):
		self.savefolder = savefolder
		self.planes = planes

		if not(type(limits) is list):
			self.limits = [limits for i in range(len(planes))]
		else:
			self.limits = limits
		
		if not(type(nbins) is list):
			self.nbins = [nbins for i in range(len(planes))]
		else:
			self.nbins = nbins
			
		if not(type(partial) is list):
			self.partial = [partial for i in range(len(planes))]
		else:
			self.partial = partial	
			
		self.counter = 0	
		
	def makeFrame(self,paramsDict):
		"""
		paramsDict must have entry "bunch"
		
		"""
		bunchInstance = paramsDict["bunch"]
		pos = paramsDict["path_length"]
		
		# -- make sure a valid bunch is passed
		if not(isinstance(bunchInstance,Bunch)):
			raise TypeError("bunch is type %s, expecting instance of class Bunch"%(type(bunchInstance))) 
			
		# -- initialize plotter for this bunch
		plotter = plotBunch(bunchInstance,planes=self.planes)
		
		# -- plot and save frames in all projections of named planes
		nplanes = len(self.planes)
		for i in range(nplanes):
			plane1 = self.planes[i]
			lim1 = self.limits[i]
			for j in range(i+1,nplanes):
				plane2 = self.planes[j]
				lim2 = self.limits[j]
				
				bin = [self.nbins[i],self.nbins[j]]
				
				# -- make list for slicing
				tmp_partial = list(self.partial)
				planes = list(self.planes)
				# -- pop off planes that will ge plotted this step
				tmp_partial.pop(planes.index(plane1))
				planes.pop(planes.index(plane1))
				tmp_partial.pop(planes.index(plane2))
				planes.pop(planes.index(plane2))
				slicestr = 'Slices: '
				for ii in range(len(planes)):
					if tmp_partial[ii]:
						slicestr += '%s: %.3f, '%(planes[ii],tmp_partial[ii])
				
				hist,bins1,bins2 = plotter.hist2d(plane1,plane2,nbins=bin,
												  xlim=lim1,ylim=lim2,
												 partial=tmp_partial)
				# -- save frame to file
				outfilename = 'frame_%s_%s_%i.txt'%(plane1,plane2,self.counter)
				header0 = 'pos=%.6f \r\n'%pos
				header1 = slicestr + '\r\n'
				header2 = plane1 + ': '+', '.join(['%.5f'%(bins1[k]) for k in range(len(bins1))]) +'\r\n'
				header3 = plane2 + ': '+', '.join(['%.5f'%(bins2[k]) for k in range(len(bins2))]) +'\r\n'
				header = header0+header1+header2+header3
				np.savetxt(self.savefolder + outfilename,hist,fmt='%.5f',header=header)
		
		# -- add 1 to counter
		self.counter += 1