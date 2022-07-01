console_info = False

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import numpy as np
import chaospy as cp

###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

import poly_basis as pb

def mean ( RV, dist ):
  """
  Compute mean value
  input:
    RV: Random variable or surrogate model
    dist: Corresponding distribution
  return:
    Mean value
  """
  return cp.E( RV, dist )

def var ( RV, dist ):
  """
  Compute variance
  input:
    RV: Random variable or surrogate model
    dist: Corresponding distribution
  return:
    variance
  """
  return cp.Var( RV, dist )

def stddev ( RV, dist ):
  """
  Compute standard deviation
  input:
    RV: Random variable or surrogate model
    dist: Corresponding distribution
  return:
    standard deviation
  """
  return cp.Std( RV, dist )

def skew ( RV, dist ):
  """
  Compute skewness (element by element 3rd order statistics)
  input:
    RV: Random variable or surrogate model
    dist: Corresponding distribution
  return:
    skewness
  """
  return cp.Skew( RV, dist )

def kurt ( RV, dist ):
  """
  Compute kurtosis (element by element 4rd order statistics)
  input:
    RV: Random variable or surrogate model
    dist: Corresponding distribution
  return:
    kurtosis
  """
  return cp.Kurt( RV, dist )

def sens_m ( RV, dist ):
  """
  Compute main variance-based Sobol sensitivity index
  input:
    RV: Random variable or surrogate model
    dist: Corresponding distribution
  return:
    First order main Sobol index
  """
  return cp.Sens_m( RV, dist )

def sens_t ( RV, dist ):
  """
  Compute total variance-based Sobol sensitivity index
  input:
    RV: Random variable or surrogate model
    dist: Corresponding distribution
  return:
    First order total Sobol index
  """
  return cp.Sens_t( RV, dist )

def project_RV ( RV, PB, dist ):
  """
  Project a random variable to a polynomial basis
  input:
    RV: Random variable
    PB: polynomial basis
    dist: Corresponding distribution
  return:
    projection coefficients
  """
  projection = [ cp.E( rv * PB, dist ) / cp.E( PB * PB, dist ) for rv in RV ]

  if console_info:
    print ( "Projection:\n", projection )

  return np.array( projection )

def tensor3 ( PB, dist ):
  """
  Compute third order tensor
  input:
    PB: polynomial basis
    dist: Corresponding distribution
  return:
    third oder tensor
  """
  norms = pb.basis_norms( PB, dist )

  c_kij = [[cp.E( bk * bi * PB, dist ) for bi in PB] for bk in ( PB / norms )]

  if console_info:
    #print ( "3rd order Tensor:\n", c_kij )
    visu_tensor3( c_kij )

  return np.array( c_kij )

def visu_tensor3 ( c_kij ):
  """
  Visualize third order tensor
  input:
    c_kij: third oder tensor
  prints:
    Non-zero-entry visualization
  """
  acc = 1e-10

  k = 0
  for kij in c_kij:
    print( " k =", k ) 
    for ij in kij:
      line = ""
      for j in ij:
        if abs( j ) > acc:
          line += " X"
        else:
          line += " ."
      print( line )
    k += 1

