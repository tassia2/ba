
from re import X
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Expansion Coefficient
#values from https://www.engineeringtoolbox.com/water-density-specific-weight-d_595.html
temps = np.array([10,15,20,25,30,35,40])
thermal_exp_coeff = np.array([0.88,1.51,2.07,2.57,3.03,3.45,3.84])*10**(-4)#/1K

def f(x,a,b,c,d,e):
    return a*x+b*x**2+c*x**3+d*x**4+e

popt, pcov = curve_fit(f,temps,thermal_exp_coeff)
print("Expansion Coefficient parameter : \n",popt)

x = np.linspace(5,45,1001)
plt.title("Expansion Coefficient")
plt.scatter(temps,thermal_exp_coeff)
plt.plot(x,f(x,*popt))
plt.show()


# therefore
[ 1.80060020e-05, -2.79388322e-07,  3.75732212e-09, -2.42369956e-11, -4.43056317e-17, -6.76426287e-05]

# Kinematic viscosity
#https://www.engineeringtoolbox.com/water-dynamic-kinematic-viscosity-d_596.html

kin_vis = np.array([1.3065,1.0035,0.8927,0.8007,0.6579])*10**(-6) #m2/s
temps_vis = np.array([10,20,25,30,40])

popt_vis, pcov_vis = curve_fit(f,temps_vis,kin_vis)
print("Kinematic Viscosity parameter : \n",popt_vis)

x = np.linspace(5,45,1001)
plt.title("Kinematic Viscosity")
plt.scatter(temps_vis,kin_vis)
plt.plot(x,f(x,*popt_vis))
plt.show()
