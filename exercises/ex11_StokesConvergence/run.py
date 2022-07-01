import os
import math
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt

mpi_proc = 4
exec_name = 'ex11_StokesConvergence'

os.system("mpirun -np " + str(mpi_proc) + " ./" + exec_name)

data = loadtxt("errors.csv", comments="#", delimiter=",", unpack=False)
m,n = np.shape(data)

# plot with various axes scales
plt.figure()

e_l2_vel = (math.log(data[1,0]) - math.log(data[1,-1])) / (math.log(data[0,0]) - math.log(data[0,-1]))
e_h1_vel = (math.log(data[2,0]) - math.log(data[2,-1])) / (math.log(data[0,0]) - math.log(data[0,-1]))
e_l2_pre = (math.log(data[3,0]) - math.log(data[3,-1])) / (math.log(data[0,0]) - math.log(data[0,-1]))

print (e_l2_vel)
print (e_h1_vel)
print (e_l2_pre)

plt.plot(data[0,:], data[1,:], label = "L2 velocity")
plt.text(data[0,-2]*1.1, data[1,-2], r'$\gamma=$' + '{:.2f}'.format( e_l2_vel ) )

plt.plot(data[0,:], data[2,:], label = "H1 velocity")
plt.text(data[0,-2]*1.1, data[2,-2], r'$\gamma=$' + '{:.2f}'.format( e_h1_vel ) )

plt.plot(data[0,:], data[3,:], label = "L2 pressure")
plt.text(data[0,-2]*1.1, data[3,-2], r'$\gamma=$' + '{:.2f}'.format( e_l2_pre ) )

plt.plot(data[0,:], data[4,:], 'k', label = "||div(u)||")


plt.ylabel('errors')
plt.xlabel('h / nu')
plt.yscale('log')
plt.xscale('log')
plt.title('Error Stokes')
plt.legend(loc="upper left")
plt.grid(True)

plt.show()
