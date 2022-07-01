import numpy as np
import chaospy as cp

import os
import sys
sys.path.append(os.path.abspath("scripts"))
import poly_basis as pb
import distributions as di
import integration as integ
import randomn_var as rv

U_bnd = [[-2.,4.],[0.,1.]]
poly_degree = 3

PT = pb.legendre_basis ( poly_degree, U_bnd )

print PT

print len(PT)

dist = []
for bnd in U_bnd:
  dist.append(cp.Uniform(bnd[0],bnd[1]))

dist = di.combine_dist(dist)

norms = pb.basis_norms(PT, dist)

print norms

print pb.test_orth(PT, dist)

lin = rv.linear( U_bnd, [10, 3] , [0, 2] )

print lin

print integ.project_RV(lin,PT,dist)

