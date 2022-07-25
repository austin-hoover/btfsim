print "Starting >>>>>>>"
#file_out = open("P_test.txt")
#print file_out.read()

aa = []
bb = []


import math
import numpy as np
import matplotlib.pyplot as plt



e_kin_ini = 0.0025 # in [GeV]
mass = 0.939294    # in [GeV]  mass = 0.939294
gamma = (mass + e_kin_ini)/mass
beta = math.sqrt(gamma*gamma - 1.0)/gamma

print "beta=", beta
print "gamma=", gamma

dr=0.005

#=================================================================================================
#=================================================================================================
#==========================================================

#bb = np.loadtxt("/Users/z2o/BTF-Simulations/BTF_and_BTF_extension/Distribution_Input_new.txt",skiprows=11)
bb = np.loadtxt("Distribution_Input.txt",skiprows=(11))
x_i = np.array(bb[:,0])*10  #first column,data unit in 'Input.txt' is centimeter
xp_i= np.array(bb[:,1])       #second column
y_i = np.array(bb[:,2])*10   #third column,data unit in 'P_test.txt' is centimeter
yp_i= np.array(bb[:,3])        #fourth column

x2_i = np.array(bb[:,0])**2*100
xp2_i = np.array(bb[:,1])**2
y2_i = np.array(bb[:,2])**2*100
yp2_i = np.array(bb[:,3])**2

x2_i_avg=np.mean(x2_i)     # get column mean value
xp2_i_avg=np.mean(xp2_i)
xxp_i_avg=np.mean(x_i*xp_i)
y2_i_avg=np.mean(y2_i)
yp2_i_avg=np.mean(yp2_i)
yyp_i_avg=np.mean(y_i*yp_i)
#print "x2_avg=",x2_avg
#print "xp2_avg=",xp2_avg
#print "xxp_avg=", xxp_avg

ex_i_rms=math.sqrt(x2_i_avg*xp2_i_avg-xxp_i_avg**2)
alphax_i= -xxp_i_avg/ex_i_rms
betax_i = x2_i_avg/ex_i_rms
print "ex_i=",ex_i_rms
print "alphax_i=", alphax_i
print "betax_i=", betax_i

#print "x_rms=",math.sqrt(x2_avg)

ey_i_rms=math.sqrt(y2_i_avg*yp2_i_avg-yyp_i_avg**2)
alphay_i= -yyp_i_avg/ey_i_rms
betay_i = y2_i_avg/ey_i_rms
print "ey_i=",ey_i_rms
print "alphay_i=", alphay_i
print "betay_i=", betay_i


xn_i =x_i/math.sqrt(betax_i)
xnp_i=alphax_i*x_i/math.sqrt(betax_i)+xp_i*math.sqrt(betax_i)
rx_i=np.sqrt(xn_i**2+xnp_i**2)    #Radial x
rx_i_max=np.max(rx_i)
rx_i_min=np.min(rx_i)
print "rx_i_max=", rx_i_max
print "rx_i_min=", rx_i_min


yn_i =y_i/math.sqrt(betay_i)
ynp_i=alphay_i*y_i/math.sqrt(betay_i)+yp_i*math.sqrt(betay_i)
ry_i=np.sqrt(yn_i**2+ynp_i**2)    #Radial y
ry_i_max=np.max(ry_i)
ry_i_min=np.min(ry_i)
print "ry_i_max=", ry_i_max
print "ry_i_min=", ry_i_min

tr_i = np.array([xn_i,xnp_i,yn_i,ynp_i])
ma_i = tr_i.T     #Transformed_cood.
#np.savetxt("Transformed_cood_i.txt",ma_i)   #Transformed_cood.
Rxy_i=zip(rx_i,ry_i)
#np.savetxt("R_xy_i.txt",Rxy_i)

tmpx_i = np.arange(0, rx_i_max+dr/0.5, dr)
countx_i = [0]*len(tmpx_i)
circx_i = [0]*len(tmpx_i)
countx_i = np.histogram(rx_i, tmpx_i)[0]
circx_i = [countx_i[i]*1.0/(2*math.pi*(tmpx_i[i]+dr*0.5)*dr) for i in range(len(countx_i))]
print np.sum(countx_i)
circx_i_max=np.max(circx_i)
#print "circx_i_max=",circx_i_max
#print "len(tmpx)=",len(tmpx)

N_rbandx_i = zip(tmpx_i,countx_i)    # Particle number in per radial band
Nor_rbandx_i = zip(tmpx_i,circx_i/circx_i_max)   # Particle number in normalized radial band
#np.savetxt('N_rbandx_i.txt', N_rbandx_i)
np.savetxt('Nor_rbandx_i.txt', Nor_rbandx_i)

tmpy_i = np.arange(0, ry_i_max+dr/0.5, dr)
county_i = [0]*len(tmpy_i)
circy_i = [0]*len(tmpy_i)
county_i = np.histogram(ry_i, tmpy_i)[0]
circy_i = [county_i[i]*1.0/(2*math.pi*(tmpy_i[i]+dr*0.5)*dr) for i in range(len(county_i))]
print np.sum(county_i)
circy_i_max=np.max(circy_i)
#print "circy_max=",circy_max
#print "len(tmpy)=",len(tmpy)

N_rbandy_i = zip(tmpy_i,county_i)    # Particle number in per radial band
Nor_rbandy_i = zip(tmpy_i,circy_i/circy_i_max)   # Particle number in normalized radial band
#np.savetxt('N_rbandy_i.txt', N_rbandy_i)
np.savetxt('Nor_rbandy_i.txt', Nor_rbandy_i)


#===================================================================Above dealing with input
#==============================================================================================
#==============================================================================================
aa = np.loadtxt("Distribution_Output.txt",skiprows=(14))
#print  "First row is:", aa[0]
#print  "First column is:", aa[:,0]

x = np.array(aa[:,0])*1000
#x = np.array(aa[:,0])*100  #first column,data unit in 'P_test.txt' is meter
xp= np.array(aa[:,1])       #second column
y = np.array(aa[:,2])*1000
#y = np.array(aa[:,2])*100   #third column,data unit in 'P_test.txt' is meter
yp= np.array(aa[:,3])        #fourth column

x2 = np.array(aa[:,0])**2*1000000
#x2 = np.array(aa[:,0])**2*10000
xp2= np.array(aa[:,1])**2
y2 = np.array(aa[:,2])**2*1000000
#y2 = np.array(aa[:,2])**2*10000
yp2= np.array(aa[:,3])**2

x2_avg=np.mean(x2)     # get column mean value
xp2_avg=np.mean(xp2)
xxp_avg=np.mean(x*xp)
y2_avg=np.mean(y2)
yp2_avg=np.mean(yp2)
yyp_avg=np.mean(y*yp)
#print "x2_avg=",x2_avg
#print "xp2_avg=",xp2_avg
#print "xxp_avg=", xxp_avg

ex_rms=math.sqrt(x2_avg*xp2_avg-xxp_avg**2)
alphax= -xxp_avg/ex_rms
betax = x2_avg/ex_rms
print "ex=",ex_rms
print "alphax=", alphax
print "betax=", betax

#print "x_rms=",math.sqrt(x2_avg)

ey_rms=math.sqrt(y2_avg*yp2_avg-yyp_avg**2)
alphay= -yyp_avg/ey_rms
betay = y2_avg/ey_rms
print "ey=",ey_rms
print "alphay=", alphay
print "betay=", betay

################################################# Original ellipse parameters calculation

xn =x/math.sqrt(betax)
xnp=alphax*x/math.sqrt(betax)+xp*math.sqrt(betax)
rx=np.sqrt(xn**2+xnp**2)    #Radial x
rx_max=np.max(rx)
rx_min=np.min(rx)
print "rx_max=", rx_max
print "rx_min=", rx_min


yn =y/math.sqrt(betay)
ynp=alphay*y/math.sqrt(betay)+yp*math.sqrt(betay)
ry=np.sqrt(yn**2+ynp**2)    #Radial y
ry_max=np.max(ry)
ry_min=np.min(ry)
print "ry_max=", ry_max
print "ry_min=", ry_min

#tr = np.array([x,xp,y,yp])
tr = np.array([xn,xnp,yn,ynp])

ma_o = tr.T     #Transformed_cood.

#np.savetxt("Transformed_cood.txt",ma_o)   #Transformed_cood.
Rxy=zip(rx,ry)
#np.savetxt("R_xy.txt",Rxy)                # X,Y coordinate of particles after transformation

#######################################################  Coordinates transformation

tmpx = np.arange(0, rx_max+dr/0.5, dr)
countx = [0]*len(tmpx)
circx = [0]*len(tmpx)
countx = np.histogram(rx, tmpx)[0]
circx = [countx[i]*1.0/(2*math.pi*(tmpx[i]+dr*0.5)*dr) for i in range(len(countx))]
print np.sum(countx)
circx_max=np.max(circx)
#print "circx_max=",circx_max
#print "len(tmpx)=",len(tmpx)


N_rbandx = zip(tmpx,countx)    # Particle number in per radial band
Nor_rbandx = zip(tmpx,circx/circx_i_max)   # Particle number in normalized radial band
#np.savetxt('N_rbandx.txt', N_rbandx)
np.savetxt('Nor_rbandx.txt', Nor_rbandx)


#==========================================================  X-plane analysis

tmpy = np.arange(0, ry_max+dr/0.5, dr)
county = [0]*len(tmpy)
circy = [0]*len(tmpy)
county = np.histogram(ry, tmpy)[0]
circy = [county[i]*1.0/(2*math.pi*(tmpy[i]+dr*0.5)*dr) for i in range(len(county))]
print np.sum(county)
circy_max=np.max(circy)
#print "circy_max=",circy_max
#print "len(tmpy)=",len(tmpy)

N_rbandy = zip(tmpy,county)    # Particle number in per radial band
Nor_rbandy = zip(tmpy,circy/circy_i_max)   # Particle number in normalized radial band
# np.savetxt('N_rbandy.txt', N_rbandy)
np.savetxt('Nor_rbandy.txt', Nor_rbandy)


#==========================================================  Y-plane analysis
#================================================================================================= Above dealing with Output
#=================================================================================================

from scipy.optimize import curve_fit

px_i = zip(tmpx_i, circx_i/circx_i_max)
py_i = zip(tmpy_i, circy_i/circy_i_max)

px = zip(tmpx, circx/circx_i_max)
py = zip(tmpy, circy/circy_i_max)
print '======================'
print '======================'
print circx_i_max,circx_max
#--------------------------------------------------------------------------
fx1 = np.delete(tmpx_i, -1)           # delete the last element of an array
fx1_1=fx1[0:16]
fx2=circx_i/circx_i_max
fx2_1=fx2[0:16]

"""
def gaus(gx,*p):
    A,mu,sigma=p                                                             #Gauss fit
    return A*np.exp(-(gx-mu)**2/(2.0*sigma**2))
p0=[1.,0.0,1.]
coeff,pcov=curve_fit(gaus,fx1_1,fx2_1,p0=p0)
x_fit1=gaus(fx1,*coeff)
x_fit_input = zip(fx1,x_fit1)
np.savetxt('x_fit_input.txt', x_fit_input)



fx3 = np.delete(tmpx, -1)           # delete the last element of an array
fx4=circx/circx_i_max

def gaus(gx,*p):
    A,mu,sigma=p                                                             #Gauss fit
    return A*np.exp(-(gx-mu)**2/(2.0*sigma**2))
p0=[1.,0.0,1.]
coeff,pcov=curve_fit(gaus,fx3,fx4,p0=p0)
x_fit3=gaus(fx3,*coeff)
"""
#--------------------------------------------------------------------------


plt.plot(zip(*px_i)[0], zip(*px_i)[1],linestyle='',marker='o',markersize=4.0,markerfacecolor='red',label='x_input')
plt.legend(loc=1)
plt.plot(zip(*px)[0], zip(*px)[1],linestyle='',marker='*',markersize=4.0,markerfacecolor='blue',label='x_output')
plt.legend(loc=1)
# plt.plot(fx1,x_fit1,color='green', linestyle='dashed',label='x_input_fit')  #Fitting output
# plt.legend(loc=1)
#plt.plot(fx3,x_fit3,color='m', linestyle='dashed',label='x_output_fit')  #Fitting output
#plt.legend(loc=1)

plt.yscale('log')
plt.xlabel('Radius')
plt.ylabel('Norm_density')
plt.grid()
plt.grid(b=True, which='minor') #, color='r', linestyle='--')
plt.savefig('Norm_density_X')
plt.show()

#--------------------------------------------------------------------------
fy1 = np.delete(tmpy_i, -1)
fy1_1 = fy1[0:18]
fy2=circy_i/circy_i_max
fy2_1=fy2[0:18]          # delete the last element of an array


def gaus(gy,*p):                                                               #Gauss fit
    A,mu,sigma=p
    return A*np.exp(-(gy-mu)**2/(2.0*sigma**2))
p0=[1.,0.0,1.]
coeff,pcov=curve_fit(gaus,fy1_1,fy2_1,p0=p0)
y_fit1=gaus(fy1,*coeff)
y_fit_input = zip(fy1,y_fit1)
# np.savetxt('y_fit_input.txt', y_fit_input)
"""
fy3 = np.delete(tmpy, -1)           # delete the last element of an array
fy4=circy/circy_i_max

def gaus(gy,*p):                                                               #Gauss fit
    A,mu,sigma=p
    return A*np.exp(-(gy-mu)**2/(2.0*sigma**2))
p0=[1.,0.0,1.]
coeff,pcov=curve_fit(gaus,fy3,fy4,p0=p0)
y_fit3=gaus(fy3,*coeff)
"""

#z = np.polyfit(fy1,fy2,2)
#p = np.poly1d(z)
#print len(p(fy1))
#print len(fy1)
#----------------------------------------------------------------------------
plt.plot(zip(*py_i)[0], zip(*py_i)[1], linestyle='',marker='o',markersize=4.0,markerfacecolor='red',label='y_input')
plt.legend(loc=1)
plt.plot(zip(*py)[0], zip(*py)[1], linestyle='',marker='*',markersize=4.0,markerfacecolor='blue',label='y_output')
plt.legend(loc=1)
plt.plot(fy1,y_fit1,color='green', linestyle='dashed',label='y_input_fit')    #Fitting output
plt.legend(loc=1)
#plt.plot(fy3,y_fit3,color='m', linestyle='dashed',label='y_output_fit')    #Fitting output
#plt.legend(loc=1)


plt.yscale('log')
plt.xlabel('Radius')
plt.ylabel('Norm_density')
plt.grid()
plt.grid(b=True, which='minor') #, color='r', linestyle='--')
plt.savefig('Norm_density_Y')
plt.show()

print "The End"
#===========================================================  Print_out
