import numpy as np
import chaospy as cp
import scipy.special as sp
import math
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath("scripts"))
import PC_expansion as PCE
import distributions as di
import poly_basis as pb

mu = [2.,0.,-1.]
sigma = [2.,4.,0.5]
N = len(mu)
poly_degree = 3
PT = pb.hermite_basis ( poly_degree, mu, sigma )

print PT

print len(PT)

dist = []
for n in range(N):
  dist.append(cp.Normal(mu[n],sigma[n]))

dist = di.combine_dist(dist)

norms = pb.basis_norms(PT, dist)

print norms

print pb.test_orth(PT, dist)

bnd = [-3.,6.]
x_val = np.linspace(bnd[0],bnd[1],100)
y_val = np.transpose(PT(x_val,0,0))

plt.plot(x_val, y_val)
plt.ylim(-20, 20)
plt.show()









