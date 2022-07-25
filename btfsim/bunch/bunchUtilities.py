"""
Utilities for manipulation of bunches
"""
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
from bunch import Bunch, BunchTwissAnalysis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# Gaussian function for fitting
def gaus(gx,*p):
		A,sigma=p                                                             
		mu = 0.
		return A*np.exp(-(gx-mu)**2/(2.0*sigma**2))

class bunchCompare():
		def __init__(self,bunch1,bunch2):
			# -- save copy of bunch or try to load in bunch from file
			if isinstance(bunch1,Bunch):
				self.bunch1 = bunch1
			elif type(bunch1)==str: # load bunch from file
				self.bunch1 = Bunch()
				self.bunch1.readBunch(bunch1)
				print(" Read in bunch %s"%bunch1)
			if isinstance(bunch2,Bunch):
				self.bunch2 = bunch2
			elif type(bunch2)==str: # load bunch from file
				self.bunch2 = Bunch()
				self.bunch2.readBunch(bunch2)
				print(" Read in bunch %s"%bunch2)
				
				
			# -- make arrays of particle coordinates
			self.coord1 = np.zeros([self.bunch1.getSize(),6])
			for i in range(self.bunch1.getSize()):
				self.coord1[i,0] = self.bunch1.x(i)
				self.coord1[i,1] = self.bunch1.xp(i)
				self.coord1[i,2] = self.bunch1.y(i)
				self.coord1[i,3] = self.bunch1.yp(i)
				self.coord1[i,4] = self.bunch1.z(i)
				self.coord1[i,5] = self.bunch1.dE(i)
				
			self.coord2 = np.zeros([self.bunch2.getSize(),6])
			for i in range(self.bunch2.getSize()):
				self.coord2[i,0] = self.bunch2.x(i)
				self.coord2[i,1] = self.bunch2.xp(i)
				self.coord2[i,2] = self.bunch2.y(i)
				self.coord2[i,3] = self.bunch2.yp(i)
				self.coord2[i,4] = self.bunch2.z(i)
				self.coord2[i,5] = self.bunch2.dE(i)
				
			self.coordlist = ['x','xp','y','yp','z','dE']
			
			
			mins1 = np.min(self.coord1,axis=0)
			maxs1 = np.max(self.coord1,axis=0)
			mins2 = np.min(self.coord2,axis=0)
			maxs2 = np.max(self.coord2,axis=0)

			self.mins = np.min(np.vstack([mins1,mins2]),axis=0)
			self.maxs = np.max(np.vstack([maxs1,maxs2]),axis=0)
				
		def compare2D(self,d1='x',d2='xp',nbins=10):
			if type(nbins)==int:
				nbins = np.repeat(nbins,2)
				
			i1 = self.coordlist.index(d1)
			i2 = self.coordlist.index(d2)
			
			bins = []
			bins.append(np.linspace(self.mins[i1],self.maxs[i1],nbins[0]))
			bins.append(np.linspace(self.mins[i2],self.maxs[i2],nbins[1]))
				
			d = self.getHistDiff(bins,cols=[i1,i2])
			return d
			
			
		def compare6D(self,nbins=10):
			if type(nbins)==int:
				nbins = np.repeat(nbins,6)
			
			bins = []
			for i in range(6):
				bins.append(np.linspace(self.mins[i],self.maxs[i],nbins[i]))
				
			d = self.getHistDiff(bins)
			return d
		
		def getHistDiff(self,bins,cols=range(6)):
			H1,edges1 = np.histogramdd(self.coord1[:,cols],bins=bins,density=True)
			H2,edges2 = np.histogramdd(self.coord2[:,cols],bins=bins,density=True)
			
			diff = np.abs(H1.flatten() - H2.flatten())
			sum = H1.flatten()  + H2.flatten() 
			ind = np.where(sum!=0)[0]
			
			d = np.sum(diff[ind]/sum[ind])
			
			return d

			
	
class bunchCalculation():
		def __init__(self,bunch,file=None):
			# -- save copy of bunch or try to load in bunch from file

			if isinstance(bunch,Bunch):
				self.bunch = bunch
			elif type(bunch)==str: # load bunch from file
				self.bunch = Bunch()
				self.bunch.readBunch(bunch)
				
			self.filename = file
			
			# -- call BunchTwissAnalysis method
			self.twiss_analysis = BunchTwissAnalysis()
			# -- compute twiss (return twiss in mm, mrad units)
			self.twiss_analysis.analyzeBunch(bunch)

			self.nParts = bunch.getSizeGlobal() 
			self.gamma = bunch.getSyncParticle().gamma()
			self.beta = bunch.getSyncParticle().beta()
			self.mass = bunch.getSyncParticle().mass()
				

		def Twiss(self,plane='x',dispersionFlag = 0,emitnormFlag = 0):
				"""
				plane = 'x' (or 'y' or 'z')
				normalized=0; if enabled, gets normalized emittances
				"""
				
				if plane=='x': ind=0
				elif plane=='y': ind=1
				elif plane=='z': ind=2
				if dispersionFlag: # correct for dispersion in twiss calculation (for x and y)
					a = self.twiss_analysis.getAlpha(ind)
					b = self.twiss_analysis.getBeta(ind)
					e = self.twiss_analysis.getEmittance(ind)*1.0e+6
				else: # include dispersive spread in twiss calculation
					a = self.twiss_analysis.getEffectiveAlpha(ind)
					b = self.twiss_analysis.getEffectiveBeta(ind)
					e = self.twiss_analysis.getEffectiveEmittance(ind)*1.0e+6
					
				# also get dispersive terms
				d = self.twiss_analysis.getDispersion(ind)
				dp = self.twiss_analysis.getDispersionDerivative(ind)
				sigma_rms_cm = np.sqrt(b*e)/10. #cm

				# units change for z emittance
				#if plane=='z':
				#		# -- scale beta, emitt to mm, mrad units
				#		b = b * gamma**3 * beta**2 * mass
				#		e = e / (gamma**3 * beta**2 * mass )			
				
				# normalize emittances
				if emitnormFlag:
						if (plane=='x') | (plane=='y'):
							e = self.twiss_analysis.getEmittanceNormalized(ind)*1e6
						elif plane=='z':
							e = e * self.gamma**3 * self.beta


				# return dictionary of calculated values. 
				return {'beta':{'value':b,'unit':'mm/mrad'},
						'alpha':{'value':a,'unit':''},
						'emit':{'value':e,'unit':'mm-mrad'},
						'disp':{'value':d,'unit':'m'},
						'dispp':{'value':dp,'unit':''},
						'rms':{'value':sigma_rms_cm,'unit':'cm'}
					   }

		def Extent(self,fraction):
			"""
			Calculates max radial extent containing fraction of particles

			Input arguments:
			bunch: Bunch() instance
			fraction: fraction of particles to count. Eg, 0.90 for 90% extent.

			Returns:
			Radius [m]
			"""
			x_i, y_i = np.zeros([2,self.bunch.getSize()])
			for i in range(self.bunch.getSize()):               
				x_i[i] = self.bunch.x(i)
				y_i[i] = self.bunch.y(i)

			r = np.linalg.norm([x_i,y_i], axis=0)
			r.sort()
			if self.bunch.getSize()*fraction < 100: # if too few particles, just return n-1 radius
				nn = self.bunch.getSize() - 1
			else: # if enough particules, do 
				nn = np.round(self.bunch.getSize() * fraction)
			
			try:
				fractional_r = r[int(nn)]
			except:
				fractional_r = 0.
				
			return fractional_r		
		
		def Sigma4D(self):
			x_i,xp_i,yp_i,y_i= np.zeros([4,self.bunch.getSize()])
			for i in range(self.bunch.getSize()):               
				x_i[i] = self.bunch.x(i)
				y_i[i] = self.bunch.y(i)
				xp_i[i] = self.bunch.xp(i)
				yp_i[i] = self.bunch.yp(i)
				
			sig4D = np.zeros([4,4])
			sig4D[0,0] = np.mean(x_i**2)
			sig4D[1,1] = np.mean(xp_i**2)
			sig4D[2,2] = np.mean(y_i**2)
			sig4D[3,3] = np.mean(yp_i**2)
			
			sig4D[0,1] = np.mean(x_i*xp_i)
			sig4D[0,2] = np.mean(x_i*y_i)
			sig4D[0,3] = np.mean(x_i*yp_i)
			sig4D[1,2] = np.mean(xp_i*y_i)
			sig4D[1,3] = np.mean(xp_i*yp_i)
			sig4D[2,3] = np.mean(y_i*yp_i)
			
			return sig4D
				



		def NormCoord(self,plane='x',plotflag=0):
			"""
			Transform bunch coordinates into normalized coordinates.
			Normalization done according to rms Twiss parameters
			plane = 'x' (or 'y' or 'z')
			plotflag=0; generated plots are saved but will also show up in active window if =1
			"""
			

			x,xp = np.zeros([2,self.bunch.getSize()])
			for i in range(self.bunch.getSize()):
				if plane == 'x':
					x[i] = 1e3*self.bunch.x(i)
					xp[i] = 1e3*self.bunch.xp(i)
				elif plane =='y':
					x[i] = 1e3*self.bunch.y(i)
					xp[i] = 1e3*self.bunch.yp(i)
				elif plane == 'z':
					raise KeyError('plane=z norm. coordinates not defined')


			###############################################################################
			## Calculate RMS Twiss params
			###############################################################################
			twiss = self.Twiss(plane=plane,emitnormFlag=1)
			alpha = twiss['alpha']['value']
			beta = twiss['beta']['value']
			e_rms = twiss['emit']['value']
			print('%s plane:'%plane)
			print("emit=%.6f"%e_rms)
			print("alpha=%.6f"%alpha)
			print("beta=%.6f"%beta)


	#		# -- RMS Twiss calculation 
	#		x2 = x*x  
	#		xp2 = xp*xp 
	#		x2_avg=np.mean(x2)     # get column mean value
	#		xp2_avg=np.mean(xp2)
	#		xxp_avg=np.mean(x*xp)
	#
	#		ex_rms=np.sqrt(x2_avg*xp2_avg-xxp_avg**2)
	#		alphax= -xxp_avg/ex_rms
	#		betax = x2_avg/ex_rms
	#		print("ex=",ex_rms)
	#		print("alphax=", alphax)
	#		print("betax=", betax)

			plt.figure(figsize=[6,6])
			CS=plt.hist2d(x, xp, bins=100,norm=LogNorm())
			plt.xlabel('%s (mm)'%plane)
			plt.ylabel('%sp (mrad)'%plane)
			# -- set square plot limits
			yl = plt.ylim()
			xl = plt.xlim()
			pltlim = np.max(np.abs(np.hstack([xl,yl])))
			plt.xlim([-pltlim,pltlim])
			plt.ylim([-pltlim,pltlim])
			plt.savefig('%s_Hist_%s'%(self.filename,plane)) 
			if plotflag: 
				plt.show() # clf  

			###############################################################################
			## Transform to normalized coordinates by Twiss Params
			###############################################################################

			xn = x/np.sqrt(beta)
			xnp = alpha* x/np.sqrt(beta)+xp*np.sqrt(beta)

			
			plt.figure(figsize=[6,6])
			CS1_1=plt.hist2d(xn, xnp, bins=100,norm=LogNorm())
			plt.xlabel('%s'%plane)
			plt.ylabel('%sp'%plane)
			plt.title('Transformed %s-coordinates'%plane)
			# -- set square plot limits
			yl = plt.ylim()
			xl = plt.xlim()
			pltlim = np.max(np.abs(np.hstack([xl,yl])))
			plt.xlim([-pltlim,pltlim])
			plt.ylim([-pltlim,pltlim])
			# -- save and show
			plt.savefig('%s_Input_%s_Trans'%(self.filename,plane))
			if plotflag:
				plt.show()


			tr = np.array([xn,xnp])
			ma = tr.T     #Transformed_cood.
			#np.savetxt(default.outdir+"Transformed_coord_%s.txt"%plane,ma)   #Transformed_cood.

			return xn, xnp

		def RadialDensity(self,dr=0.1,plotflag=0,plane='x'):
			(xn,xnp) = self.NormCoord(plotflag=plotflag,plane=plane)

			r = np.sqrt(xn**2+xnp**2)  #Radial coordinate, unitless
			r_max = np.max(r)
			r_min = np.min(r)
			print("r_max=%.6f"%r_max)
			print("r_min=%.6f"%r_min)

			###############################################################################
			## Calculate density and Gaussian Fit
			###############################################################################

			tmpr = np.arange(0, r_max+dr*2, dr)
			countr = np.histogram(r, tmpr)[0]
			densr = [countr[i]*1.0/(2*np.pi*(tmpr[i]+dr*0.5)*dr) for i in range(len(countr))]
			densr_max=np.max(densr)

			fr1 = np.delete(tmpr, -1)           
			fr2= densr/densr_max
			# -- fit only region where fr2>=1e-2 to only fit core
			ind = np.argmin(np.abs(fr2 - .5))
			p0=[1.,dr]
			coeff,pcov=curve_fit(gaus,fr1[0:ind],fr2[0:ind],p0=p0)
			r_fit=gaus(fr1,*coeff)

			output_arr = np.vstack([tmpr[1:],countr,densr/densr_max,r_fit]).T   
			np.savetxt('%s_%s_norm_density.txt'%(self.filename,plane), output_arr, fmt = '%.7f', 
							   header = '## r [mm], count (btw r and r-dr), norm. density, Gauss fit')

			 
			plt.figure()
			plt.plot(fr1,fr2,'o-',fr1,r_fit)
			plt.xlabel('Radius [unitless]')
			plt.ylabel('Norm. density')
			plt.ylim([1e-6,1.1])
			plt.xlim([0,r_max])
			plt.grid()
			plt.grid(b=True, which='minor') 
			plt.savefig('%s_Norm_density_lin_%s'%(self.filename,plane))
			if plotflag:
				plt.show()

			plt.figure()
			plt.plot(fr1,fr2,'o-',fr1,r_fit)
			plt.yscale('log')
			plt.xlabel('Radius [unitless]')
			plt.ylabel('Norm. density')
			plt.ylim([1e-6,1.1])
			plt.xlim([0,r_max])
			plt.grid()
			plt.grid(b=True, which='minor') 
			plt.savefig('%s_Norm_density_log_%s'%(self.filename,plane))
			if plotflag:
				plt.show()

			return fr1, fr2, r_fit # return r, norm. density and Gaussian-fitted density


class bunchTrack():
	"""
		This class holds array with beam evolution data
		This class also has as a method the function that is called on action entrance to 
		add to the beam evolution array
		
		in init, 
		dispersionFlag = 0; set to 1 to subtract dispersive term from emittances
		emitnormFlag = 0; set to 1 fo calculate normalized emittances
		
		"""
	def __init__(self,dispersionFlag = 0,emitnormFlag = 0):

		# -- call BunchTwissAnalysis method
		self.twiss_analysis = BunchTwissAnalysis()
		self.dispersionFlag = dispersionFlag
		self.emitnormFlag = emitnormFlag
		
		# -- initialize history arrays in hist dict
		histkeys = ['s','npart','nlost','xrms','yrms','zrms','ax','bx','ex','ay','by','ey','az','bz','ez','sigx','sigy','sigxx','sigyy','sigxy','r90','r99','dx','dpx']
		histinitlen = 10000
		self.hist = dict((histkeys[k], np.zeros(histinitlen)) for k in range(len(histkeys)))
		self.hist["node"] = []
		

	def action_entrance(self,paramsDict):
		"""
		Executed at entrance of node
		"""
		node = paramsDict["node"]
		bunch = paramsDict["bunch"]
		pos = paramsDict["path_length"]
		if(paramsDict["old_pos"] == pos):return
		if(paramsDict["old_pos"] + paramsDict["pos_step"] > pos): return
		paramsDict["old_pos"] = pos
		paramsDict["count"] += 1

		# -- update statement
		nstep = paramsDict["count"]
		npart = bunch.getSize()
		print("Step %i, Nparts %i, s=%.3f m, node %s"%(nstep,npart,pos,node.getName()))

		
		bunchCalc = bunchCalculation(bunch)
		twissx = bunchCalc.Twiss(plane='x',dispersionFlag=self.dispersionFlag,emitnormFlag=self.emitnormFlag)
		(alphaX,betaX,emittX)  = (twissx['alpha']['value'],twissx['beta']['value'], twissx['emit']['value'])
		(dispX,disppX) = (twissx['disp']['value'], twissx['dispp']['value'])
		twissy = bunchCalc.Twiss(plane='y',dispersionFlag=self.dispersionFlag,emitnormFlag=self.emitnormFlag)
		(alphaY,betaY,emittY) = (twissy['alpha']['value'], twissy['beta']['value'],twissy['emit']['value'])
		twissz = bunchCalc.Twiss(plane='z',dispersionFlag=self.dispersionFlag,emitnormFlag=self.emitnormFlag)
		(alphaZ,betaZ,emittZ) = (twissz['alpha']['value'], twissz['beta']['value'],twissz['emit']['value'])
		
		
		nParts = bunch.getSizeGlobal() 
		gamma = bunch.getSyncParticle().gamma()
		beta = bunch.getSyncParticle().beta()

		## -- compute twiss (this is somehow more robust than above...but doesn't include dispersion flag..)
		#self.twiss_analysis.analyzeBunch(bunch)
		#(alphaX,betaX,emittX) = (self.twiss_analysis.getEffectiveAlpha(0),self.twiss_analysis.getEffectiveBeta(0),self.twiss_analysis.getEffectiveEmittance(0)*1.0e+6)
		#(alphaY,betaY,emittY) = (self.twiss_analysis.getEffectiveAlpha(1),self.twiss_analysis.getEffectiveBeta(1),self.twiss_analysis.getEffectiveEmittance(1)*1.0e+6)
		#(alphaZ,betaZ,emittZ) = (self.twiss_analysis.getTwiss(2)[0],self.twiss_analysis.getTwiss(2)[1],self.twiss_analysis.getTwiss(2)[3]*1.0e+6)
		
		
		## -- get rms from twiss
		x_rms = np.sqrt(betaX*emittX)/10. #cm /10
		y_rms = np.sqrt(betaY*emittY)/10. #cm /10
		z_rms = np.sqrt(betaZ*emittZ) #np.sqrt(self.twiss_analysis.getTwiss(2)[1]*self.twiss_analysis.getTwiss(2)[3])*1000.

		# -- compute sigma matrix to order 2
		order = 2; 
		self.twiss_analysis.computeBunchMoments(bunch,order,self.dispersionFlag,self.emitnormFlag)
		sigx	= self.twiss_analysis.getBunchMoment(1,0)*1e2 # avg x, in cm	
		sigy	= self.twiss_analysis.getBunchMoment(0,1)*1e2 # avg y, in cm
		s4D = bunchCalc.Sigma4D() # squared beam sizes
		sigxx	= s4D[0,0]*1e2
		sigyy	= s4D[2,2]*1e2
		sigxy	= s4D[0,2]*1e2

		# -- compute 90%, 99% extent
		#bunchCalc = bunchCalculation(bunch)
		r90 = bunchCalc.Extent(0.90)*1e2
		r99 = bunchCalc.Extent(0.99)*1e2
		
		# -- correctly assign nParticles for 0th step
		if paramsDict["count"]==1:
			self.hist["npart"][paramsDict["count"]-1] = nParts

				

		# -- assign history arrays in hist dict
		self.hist["s"][paramsDict["count"]] = pos
		self.hist["node"].append(node.getName())
		self.hist["npart"][paramsDict["count"]] = nParts
		self.hist["xrms"][paramsDict["count"]] = x_rms
		self.hist["yrms"][paramsDict["count"]] = y_rms
		self.hist["zrms"][paramsDict["count"]] = z_rms
		self.hist["ax"][paramsDict["count"]] = alphaX
		self.hist["bx"][paramsDict["count"]] = betaX
		self.hist["ex"][paramsDict["count"]] = emittX
		self.hist["dx"][paramsDict["count"]] = dispX
		self.hist["dpx"][paramsDict["count"]] = disppX 
		self.hist["ay"][paramsDict["count"]] = alphaY
		self.hist["by"][paramsDict["count"]] = betaY
		self.hist["ey"][paramsDict["count"]] = emittY
		self.hist["az"][paramsDict["count"]] = alphaZ
		self.hist["bz"][paramsDict["count"]] = betaZ
		self.hist["ez"][paramsDict["count"]] = emittZ
		self.hist["sigx"][paramsDict["count"]] = sigx
		self.hist["sigy"][paramsDict["count"]] = sigy
		self.hist["sigxx"][paramsDict["count"]] =sigxx
		self.hist["sigyy"][paramsDict["count"]] =sigyy
		self.hist["sigxy"][paramsDict["count"]] =sigxy
		self.hist["r90"][paramsDict["count"]] = r90
		self.hist["r99"][paramsDict["count"]] = r99
		self.hist["nlost"][paramsDict["count"]] = self.hist["npart"][0] - nParts

	def action_exit(self,paramsDict):
		"""
		Executed at exit of node
		"""
		self.action_entrance(paramsDict)
		
	def cleanup(self):
		# -- trim 0's from hist        
		ind = np.where(self.hist['xrms']==0)[0][1]
		for key, arr in self.hist.iteritems():
			self.hist[key] = arr[1:ind]

	def writehist(self,**kwargs):
		"""
		Save history data 
		optional argument:
		filename = location to save data
		"""

		# --- file name + location
		defaultfilename = 'btf_output_data.txt'
		filename = kwargs.get('filename',defaultfilename)

		# -- open files to write data
		file_out =  open(filename,"w")
		header = 's[m], nparts, xrms [cm], yrms [cm], zrms [cm], ax, bx, ex[mm-mrad], ay, by, ey[mm-mrad], az, bz, ez[m-GeV], sigx[cm], sigy[cm], sigxx[cm2], sigyy[cm2], sigxy[cm2], r90[cm], r99[cm], Dx [m], Dxp \n'
		file_out.write(header)


		for i in range(len(self.hist['s'])):
			line = "%.3f %i "%(self.hist["s"][i], self.hist["npart"][i])
			line+= "%.3f %.3f %.3f "%(self.hist["xrms"][i], self.hist["yrms"][i], self.hist["zrms"][i])
			line+="%.3f %.3f %.6f "%(self.hist["ax"][i], self.hist["bx"][i], self.hist["ex"][i])
			line+="%.3f %.3f %.6f "%(self.hist["ay"][i], self.hist["by"][i], self.hist["ey"][i])
			line+="%.3f %.3f %.6f "%(self.hist["az"][i], self.hist["bz"][i], self.hist["ez"][i])
			line+="%.5f %.5f %.6f "%(self.hist["sigx"][i], self.hist["sigy"][i],self.hist["sigxx"][i])
			line+="%.6f %.6f "%(self.hist["sigyy"][i], self.hist["sigxy"][i])
			line+="%.4f %.4f "%(self.hist["r90"][i], self.hist["r99"][i])
			line+="%.4f %.4f \n"%(self.hist["dx"][i], self.hist["dpx"][i])
			file_out.write(line)

		file_out.close()
		
		
class spTrack():
	"""
	This class holds array with beam evolution data
	Copy of bunchTrack class modified for single-particle tracking (no twiss/size data)
	"""
	def __init__(self):
		
		# -- initialize history arrays in hist dict
		histkeys = ['s','npart','x','xp','y','yp','z','dE']
		histinitlen = 10000
		self.hist = dict((histkeys[k], np.zeros(histinitlen)) for k in range(len(histkeys)))
		self.hist["node"] = []


	def action_entrance(self,paramsDict):
		"""
		Executed at entrance of node
		"""
		node = paramsDict["node"]
		bunch = paramsDict["bunch"]
		pos = paramsDict["path_length"]
		if(paramsDict["old_pos"] == pos):return
		if(paramsDict["old_pos"] + paramsDict["pos_step"] > pos): return
		paramsDict["old_pos"] = pos
		paramsDict["count"] += 1

		# -- update statement
		nstep = paramsDict["count"]
		npart = bunch.getSize()
		print("Step %i, Nparts %i, s=%.3f m, node %s"%(nstep,npart,pos,node.getName()))
		
		nParts = bunch.getSizeGlobal() 

		# -- get particle position, momenta
		x = bunch.x(0) * 1000.
		xp = bunch.xp(0) * 1000.
		y = bunch.y(0) * 1000.
		yp = bunch.yp(0) * 1000.
		z = bunch.z(0) * 1000.
		dE = bunch.dE(0) * 1000.

		# -- assign history arrays in hist dict
		self.hist["s"][paramsDict["count"]] = pos
		self.hist["node"].append(node.getName())
		self.hist["npart"][paramsDict["count"]] = nParts
		self.hist["x"][paramsDict["count"]] = x
		self.hist["y"][paramsDict["count"]] = y
		self.hist["z"][paramsDict["count"]] = z
		self.hist["xp"][paramsDict["count"]] = xp
		self.hist["yp"][paramsDict["count"]] = yp
		self.hist["dE"][paramsDict["count"]] = dE

	def action_exit(self,paramsDict):
		"""
		Executed at exit of node
		"""
		self.action_entrance(paramsDict)
		
	def cleanup(self):
		# -- trim 0's from hist        
		ind = np.where(self.hist['npart'][1:]==0)[0][0]+1
		for key, arr in self.hist.iteritems():
			self.hist[key] = arr[0:ind]

	def writehist(self,**kwargs):
		"""
		Save history data 
		optional argument:
		filename = location to save data
		"""

		# --- file name + location
		defaultfilename = 'btf_output_data.txt'
		filename = kwargs.get('filename',defaultfilename)

		# -- open files to write data
		file_out =  open(filename,"w")
		header = 's[m], nparts, x [mm], xp[mrad], y[mm], yp[mrad], z[mm?], dE[MeV?] \n'
		file_out.write(header)


		for i in range(len(self.hist['s'])-1):
			line = "%.3f %s %i %.6f %.6f %.6f %.6f %.6f %.6f \n"\
			%(self.hist["s"][i], self.hist["node"][i].split(':')[-1], self.hist["npart"][i], \
			  self.hist["x"][i], self.hist["xp"][i], self.hist["y"][i], \
			  self.hist["yp"][i], self.hist["z"][i], self.hist["dE"][i] )
			file_out.write(line)

		file_out.close()
	
															  

class Beamlet():
	"""
	Class to create beamlet out of specified bunch distribution.
	
	optional arguments:
	
	center of slice:
	x,y [mm]
	xp,yp [mrad]
	z [mm]
	dE [keV]
	
	slice width:
	xwidth,ywidth = .200 mm
	xpwidth,ypwidth = .200 mm/L
	L = 0.947 (HZ04-->HZ06 slit separation)
	zwidth = 2 deg. (~BSM resolution)
	dEwidth = 0.4 keV (~energy slit resolution
	
	"""
	def __init__(self,bunch_in,z2phase,**kwargs):
		self.z2phase = z2phase
		self.bunch_in = bunch_in
	
	
	def slice(self,**kwargs):
		
		# -- location of bunch slice
		xslice = kwargs.get('x',None)
		xpslice = kwargs.get('xp',None)
		yslice = kwargs.get('y',None)
		ypslice = kwargs.get('yp',None)
		zslice = kwargs.get('z',None)
		dEslice = kwargs.get('dE',None)
		
		# -- physical width of slits [mm]
		xw = kwargs.get('xwidth',.2)
		xpw = kwargs.get('xpwidth',.2)
		yw = kwargs.get('ywidth',.2)
		ypw = kwargs.get('ypwidth',.2)
		
		# -- width of z in deg.
		zw = kwargs.get('zwidth',0.4) # close to 1 pixel width
		
		# -- width of dE in keV
		dEw = kwargs.get('dEwidth',2)
		# per Cathey thesis, energy uncertainty is 
		# ~1.3 keV for 0.8 mm slit, 0.6 keV for screen
		
		# -- distance between transverse slits
		L = kwargs.get('L',0.947) # [m]
		#L = L*1e3 #[convert to mm]
		#Ldipoo2slit = 0.129 # distance dipole exit to VS06 (energy) slit
		#rho = 0.3556 # dipole bending radius
		#Lslit2dipo = 1.545 
		
		# -- convert to bunch units (meters, rad, GeV)
		xw = 0.5*xw 
		xpw =  0.5*xpw/L
		yw =  0.5*yw
		ypw =  0.5*ypw/L
		dEw =  0.5*dEw
		zw =  0.5*zw
		
		# -- be verbose, also convert to [m, rad, GeV]
		print("selecting in:")
		if not(xslice is None):
			print("%.6f < x < %.6f mm"%(xslice-xw,xslice+xw))
			xslice *= 1e-3
			xw *= 1e-3
		if not(xpslice is None):
			print("%.6f < x' < %.6f mrad"%(xpslice-xpw,xpslice+xpw))
			xpslice *= 1e-3
			xpw *= 1e-3
		if not(yslice is None):
			print("%.6f < y < %.6f mm"%(yslice-yw,yslice+yw))
			yslice *= 1e-3
			yw *= 1e-3
		if not(ypslice is None):
			print("%.6f < y' < %.6f mrad"%(ypslice-ypw,ypslice+ypw))
			ypslice *= 1e-3
			ypw *= 1e-3
		if not(zslice is None):
			print("%.6f < z < %.6f deg"%(zslice-zw,zslice+zw))
			zslice /= self.z2phase
			zw /= self.z2phase
		if not(dEslice is None):
			print("%.6f < dE < %.6f keV"%(dEslice-dEw,dEslice+dEw))
			dEslice *= 1e-6
			dEw *= 1e-6
		
		n =  self.bunch_in.getSizeGlobal()
		beamlet = Bunch() # make new empty
		for i in range(n):
			x = self.bunch_in.x(i)
			xp = self.bunch_in.xp(i)
			y = self.bunch_in.y(i)
			yp = self.bunch_in.yp(i)
			z = self.bunch_in.z(i)
			dE = self.bunch_in.dE(i)
			# -- check each dimension to see if particle is within slice,
			# if slice is specified. 
			if not(xslice is None):
				if not(x < xslice+xw and x > xslice-xw):
					continue
			if not(xpslice is None):
				if not(xp < xpslice+xpw and xp > xpslice-xpw):
					continue
			if not(yslice is None):
				if not(y < yslice+yw and y > yslice-yw):
					continue
			if not(ypslice is None): 
				if not(yp < ypslice+ypw and yp > ypslice-ypw):
					continue
			if not(zslice is None):
				if not(z < zslice+zw and z > zslice-zw):
					continue
			if not(dEslice is None):
				if not(dE < dEslice+dEw and dE > dEslice-dEw):
					continue
			beamlet.addParticle(x,xp,y,yp,z,dE)
			
		print("Beamlet has %i particles"%beamlet.getSizeGlobal())
		return beamlet
		
		
class adaptiveWeighting():
	"""
	This class adjusts macroparticle weight dynamically
	"""
	def __init__(self,z2phase,macrosize0):
		self.z2phase = z2phase
		self.macrosize0 = macrosize0
		
		# -- initialize history arrays in hist dict
		histkeys = ['s','macro']
		histinitlen = 10000
		self.hist = dict((histkeys[k], np.zeros(histinitlen)) for k in range(len(histkeys)))
		self.hist["node"] = []
		

	def action_entrance(self,paramsDict):
		"""
		Executed at entrance of node
		"""
		node = paramsDict["node"]
		bunch = paramsDict["bunch"]
		pos = paramsDict["path_length"]
		if(paramsDict["old_pos"] == pos):return
		if(paramsDict["old_pos"] + paramsDict["pos_step"] > pos): return
		paramsDict["old_pos"] = pos
		paramsDict["count"] += 1

	
		# -- count how many particles inside 1 RF period
		noutside = 0
		ntotal = bunch.getSize()
		for i in range(ntotal):
			phi = bunch.z(i) * self.z2phase
			if np.abs(phi) > 180.: # if outside 1 RF period
				noutside += 1
				
		macrosize = self.macrosize0 * ntotal/(ntotal - noutside)	
		bunch.macroSize(macrosize)

		# -- assign history arrays in hist dict
		self.hist["s"][paramsDict["count"]] = pos
		self.hist["node"].append(node.getName())
		self.hist["macro"][paramsDict["count"]] = macrosize
		
	def action_exit(self,paramsDict):
		self.action_entrance()
