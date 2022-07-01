console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import os
import sys
import shutil
import numpy as np
import chaospy as cp
import subprocess as proc

###-------------------------------------------------------------------------###
### Import UQ scripts ###
###-------------------------------------------------------------------------###

import PC_expansion as PCE
import distributions as di
import poly_basis as pb
import quadrature as quad
import solver_input as si

def create_PC_collocation ( pc_deg, quad_rule, sparse_flag, dist_list ):
  """
  Create collocation rule based on PC and a list of distributions
  input:
    pc_deg: Degree of PC expansion
    quad_rule: Identifier for quadrature rule, see quadrature.py
    sparse_flag: Sets whether sparse grid should be used
    dist_list: List of parameter distributions
  return:
    basis, basis_norms, nodes, weights, dist, num_nodes
  """

  ### Define random input distributions
  dist = di.combine_dist ( dist_list )

  if console_info:
    print ( "Distribution:\n", dist )

  ### Generate orthogonal polynomial basis
  basis, basis_norms = pb.poly_basis ( pc_deg, dist )

  if console_info:
    print ( "Basis:\n", basis )
    print ( "Norms:\n", basis_norms )

  ### Generate quadrature rule
  nodes, weights = quad.quadrature( pc_deg, dist, quad_rule, sparse_flag )
  
  num_nodes = len( weights )

  if console_info:
    print ( "Nodes:\n", nodes )
    print ( "Weights:\n", weights )
    print ( "Number of nodes:\n", num_nodes )
    print ( "Sum of weights:\n", sum( weights ) )
    print ( "Valid quadrature:\n", quad.check_weights( weights ) )

  return basis, basis_norms, nodes, weights, dist, num_nodes
