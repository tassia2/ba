console_info = False

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import numpy as np
from math import factorial as fct

def num_modes ( P, N ):
  """
  Calculate the number of modes
  input:
    P: Maximal PC degree
    N: Dimension of stochastical space
  return:
    Number of modes
  """
  nm = int( ( fct( P + N ) / ( fct( P ) * fct( N ) ) ) - 1 )

  if console_info:
    print ( "Number of modes:\n", nm )

  return nm

def multiindex ( P, N ):
  """
  multiindex for the chaos polynomial basis
  Adapted from Le Maitre2010, p.517
  input:
    P: max PC degree
    N: Dimension of stochastical space
  return:
    Multiindex matrix
  """
  M = num_modes( P, N )

  ### Allocation of integer matrix
  multii = np.zeros( shape = ( M + 1, N ), dtype=int )

  if P > 0:
    for i in range( 1, N + 1 ):
      multii[i][i-1] = 1

  if P > 1:
    R = N
    mat = np.zeros( shape = ( N, P ), dtype=int )

    for i in range( N ):
      mat[i][0] = 1

    for k in range( 1, P ):
      L = R

      for i in range( N ):
        s = 0
        for l in range( i, N ):
          s = s + mat[l][k-1]

        mat[i][k] = s

      for j in range( N ):
        for l in range( L - int( mat[j][k] ) + 1, L + 1 ):
          R = R + 1

          for i in range( N ):
            multii[R][i] = multii[l][i]

          multii[R][j] = multii[R][j] + 1

  if console_info:
    print ( "Multi index:\n", multii )

  return multii

