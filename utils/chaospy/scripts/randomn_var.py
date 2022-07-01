console_info = False

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import chaospy as cp
import numpy as np

def linear ( ext, mean, pm ):
  """
  Get linear function
  input:
    ext: Support
    mean: Value in the middle of the support
    pm: Deviation at the limits of the support
  return:
    linear polynomial
  """
  dim = len( ext )
  if dim != len( mean ) or dim != len( pm ):
    print "Error: inputs for linear function should have the same length"
    quit()

  ext = np.array( ext )
  ext = np.transpose( ext )

  q = cp.variable( dim )

  mid = ( ext[0] + ext[1] ) / 2.
  dist = ( ext[1] - ext[0] ) / 2.

  q = pm / dist * ( q - mid ) + mean

  poly = cp.Poly( q )

  if console_info:
    print ( "Linear randomn variable:\n", poly )

  return poly

def constant ( val ):
  """
  Get constant function
  input:
    val: value of the function
  return:
    constant polynomial
  """
  dim = len( val )

  q = cp.variable( dim )

  q = q * np.array(val)

  poly = cp.Poly( q )

  if console_info:
    print ( "Constant randomn variable:\n", poly )

  return poly

