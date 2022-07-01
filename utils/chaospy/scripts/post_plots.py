import numpoly
import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib import cm

poly = numpoly.load('/home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq/poly.npy')

ray = np.linspace(1e4, 1e6,100)
prandl = np.linspace(2, 13.5,100)

arr = []
for ra in ray:
    arr.append(poly(6,ra))

plt.title("Pressure for constant Prandl-number=6 \nat point=(64,32) with varying Rayleigh-number\nby evaluating chaos polynomial")
plt.xlabel("rayleigh number")
plt.ylabel("pressure")
plt.plot(ray,arr)
plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X, Y = np.meshgrid(prandl,ray)
row, col = X.shape
input = np.vstack((X.reshape((row * row)),Y.reshape((row * row))))

Z = numpoly.call(poly,input)
Z = Z.reshape((row,col))


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
print("hi")
