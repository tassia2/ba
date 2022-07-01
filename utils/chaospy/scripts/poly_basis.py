console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import numpy as np
import chaospy as cp
import PC_expansion as PCE

def poly_basis ( P, dist ):
  """
  Generate orthogonal polynomial basis
  input:
    P: max PC degree
    dist: distribution
  return:
    basis and norms
  """
  pb = cp.orth_ttr( P, dist, retall=True )

  if console_info:
    print ( "Polynomial basis:" )
    for p in pb:
      print( p )

  return pb

def multidim_basis ( PL ):
  """
  Function for tensor product:
  Construct multidimensional basis from list of 1D bases
  input:
    PL: list of 1D bases
  return:
    multidimensional basis
  """
  N = len( PL )
  P = len( PL[0] ) - 1
  
  multii = PCE.multiindex( P, N )

  ### Multiplication of the 1D polynomials
  PLm = []
  for m in multii:
    pl = cp.Poly( 1 )
    for i in range( len( m ) ):
      pl = pl * PL[i][int( m[i] )]

    PLm.append( pl )

  mb = cp.Poly( PLm )

  if console_info:
    print ( "Multidimensional polynomial basis:\n", mb )

  return mb

def legendre_basis ( P, ext ):
  """
  Construct Legendre polynomial basis
  input:
    P: max PC degree
    ext: Extensions of the uniform distributions
  return:
    basis
  """
  N = len( ext )
  q = cp.variable( N )

  ext = np.array( ext )
  ext = np.transpose( ext )

  ### Rescale to standard interval: [-1,1]
  q = 2. * ( q - ext[0] ) / ( ext[1] - ext[0] ) - 1.

  ### List of polynomials for each dimension
  PL = []
  if N == 1:
    pl = [cp.Poly( 1 ), q]

    for d in range( 2, P + 1 ):
      pl.append( ( ( 2 * d - 1 ) * q * pl[d-1] \
                     - ( d - 1 ) * pl[d-2] ) / d )

    PL.append( cp.Poly( pl ) )

  else:
    for n in range( N ):
      pl = [cp.Poly( 1 ), q[n]]

      for d in range( 2, P + 1 ):
        pl.append( ( ( 2 * d - 1 ) * q[n] * pl[d-1] \
                       - ( d - 1 ) * pl[d-2] ) / d )

      PL.append( cp.Poly( pl ) )

  if console_info:
    print ( "Legendre polynomials:" )
    for pl in PL:
      print( pl )

  ### Get multidimensional basis from the 1D polynomials
  mb = multidim_basis( PL )

  return mb

def hermite_basis ( P, mu, sigma ):
  """
  Hermite
  input:
    P: max PC degree
    mu: Mean values of the corresponding normal dists
    sigma: Standard deviation of the normal dists
  return:
    basis
  """
  N = len( mu )
  if N != len( sigma ):
    print ("Error: mu and sigma should have the same length.")
    quit( )
  q = cp.variable( N )

  ### Rescale to centered normal distribution: mu = 0, sigma = 1
  q = ( q - np.array( mu ) ) / np.array( sigma )

  ### List of polynomials for each dimension
  PL = []
  if N == 1:
    pl = [cp.Poly( 1 ), q]

    for d in range( 2, P + 1 ):
      pl.append( q * pl[d-1] - ( d - 1 ) * pl[d-2] )

    PL.append( cp.Poly( pl ) )

  else:
    for n in range( N ):
      pl = [cp.Poly( 1 ), q[n]]

      for d in range( 2, P + 1 ):
        pl.append( q[n] * pl[d-1] - ( d - 1 ) * pl[d-2] )

      PL.append( cp.Poly( pl ) )

  if console_info:
    print ( "Hermite polynomials:" )
    for pl in PL:
      print( pl )

  ### Get multidimensional basis from the 1D polynomials
  mb = multidim_basis( PL )

  return mb

def basis_norms ( PB, dist ):
  """
  Calculate basis norms
  input:
    PB: Polynomial basis
    dist: Corresponding distribution
  return:
    norms
  """
  norms = cp.E( PB * PB, dist ) 

  if console_info:
    print ( "Basis norms:\n", norms )

  return norms

def test_orth ( PB, dist, acc = 1e-10 ):
  """
  Test orthogonality of polynomial basis
  input:
    PB: Polynomial basis
    dist: Corresponding distribution
    acc: Tolerance for machine accuracy
  return:
    true if orthogonal
  """
  for p in range( len( PB ) ):
    SP = cp.E( PB[p] * PB[:], dist )

    if console_info:
      print ("Scalar product:\n", SP)

    ### Norm should be non-zero
    if abs( SP[p] ) < acc:
      return false

    ### All other scalar products should be zero
    SP = np.delete( SP, p )
    for sp in SP:
      if abs( sp ) > acc:
        return False

  return True


