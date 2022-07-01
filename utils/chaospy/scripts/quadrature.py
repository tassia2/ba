###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###
import numpy as np
import chaospy as cp

console_info = False

def quadrature ( P, dist, quad_rule, sparse_flag=False ):
  """
  Set up quadrature rule
  input:
    P: max PC degree
    dist: distribution
    quad_rule: Quadrature type
      'c': 'clenshaw_curtis'
      'e': 'Gaussian with gauss_legendre'
      'g': 'Gaussian with golub_welsch'
      'j': 'leja' ###
    spars_flag: Usage of Smolyak sparse grid
  return:
    quadrature nodes and weights
  """

  ### Quadrature for the scalar product of chaos polynomials
  ### is exact with order
  ### 'c': PC_degree * 2
  ### 'e': PC_degree
  ### 'g': PC_degree
  ### 'j': PC_degree * 2 ?

  if quad_rule == 'c' or quad_rule == 'j':
    P *= 2

  nodes, weights = cp.generate_quadrature( P, dist, \
                                 sparse=sparse_flag, \
                                 rule=quad_rule )

  if console_info:
    print ( "Nodes:\n", nodes )
    print ( "Weights:\n", weights )

  if not check_weights( weights ):
    print ("Error: Sum of weights is", np.sum( weights ))
    quit( )

  return nodes, weights

def check_weights ( weights, acc = 1e-10 ):
  """
  Check quadrature weights
  input:
    weights: quadrature weights
    acc: Tolerance for machine accuracy
  return:
    True if sum of weights is 1
  """
  err = abs( np.sum( weights ) - 1 )

  if console_info:
    print ( "Weights error:\n", err )

  return err < acc

def equidist_nodes ( intervals, num_sub_intervals ):
  """
  Set up an equidistant grid
  input:
    intervals: list of intervals for a number of input parameters
    num_sub_intervals: number of subintervals for each parameter
  return:
    nodes: nodes of the equidistant grid
    num_nodes: number of nodes in the grid
  """

  ### Get the dimension of the grid
  dimension = len( intervals )
  if console_info:
    print ( "Equidistant grid dimension:\n", dimension )

  if ( dimension != len( num_sub_intervals ) ):
    print ( "Error: dimension of intervals", intervals,
            "does not match number of subintervals", num_sub_intervals )
  
  ### Calculate the total number of nodes
  num_nodes = 1
  for nsi in num_sub_intervals:
    num_nodes *= nsi + 1

  if console_info:
    print ( "Number of nodes in equidistant grid:\n", num_nodes )

  ### Divide the intervals in equidistant sub-intervals
  nodes_1D = [np.linspace( intervals[i][0],
                           intervals[i][1],
                           num=num_sub_intervals[i] + 1 ) \
              for i in range( dimension )]

  ### Tensor grid creation
  grid = np.meshgrid( *nodes_1D )
  nodes = [ g.reshape( num_nodes ) for g in grid ]

  return nodes, num_nodes

