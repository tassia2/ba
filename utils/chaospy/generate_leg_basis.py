import numpy as np
import chaospy as cp
import scipy.special as sp
import math
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath("scripts"))
import poly_basis as pb
import distributions as di
import integration as integ


U_bnd = [[-3.,2.],[-1.,0.],[0., 5.]]
poly_degree = 3

PT = pb.legendre_basis ( poly_degree, U_bnd )

#print (PT)

#print (len(PT))

dist = []
for bnd in U_bnd:
  dist.append(cp.Uniform(bnd[0],bnd[1]))

dist = di.combine_dist(dist)

norms = pb.basis_norms(PT, dist)

#print (norms)

#print (pb.test_orth(PT, dist))

ten3 = integ.tensor3(PT, dist)

x_val = np.linspace(U_bnd[0][0],U_bnd[0][1],100)
y_val = np.transpose(PT(x_val,0,0))

plt.plot(x_val, y_val)
plt.show()


