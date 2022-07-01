import numpy as np
import chaospy as cp
import math
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath("scripts"))
import poly_basis as pb
import distributions as di
import integration as integ
import randomn_var as rv

mu = [2.,0.,-1.]
sigma = [2.,4.,0.5]
N = len(mu)

poly_degree = 3

PT = pb.hermite_basis ( poly_degree,  mu, sigma )

print PT

print len(PT)

dist = []
for n in range(N):
  dist.append(cp.Normal(mu[n],sigma[n]))

dist = di.combine_dist(dist)

norms = pb.basis_norms(PT, dist)

print norms

print pb.test_orth(PT, dist)

const = rv.constant( [10, 3, 2] )

print const

print integ.project_RV(const,PT,dist)

