###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###
from __future__ import print_function
from os import path
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt

console_info = True
###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

import IO_basics as io

def read_node_qoi_postime ( work_folder, qoi_filename, num_nodes ):
  """
  Read QOI data from all collocation nodes
  input:
    work_folder: Folder with collocation node folders
    qoi_filename: File name of QOI files
    num_nodes: Number of collocation nodes
  return:
    QOI data
  """
  qoi_data = []
  input_folder_prefix = path.join( work_folder, 'node_' )
  num_time_steps = 0
  ### for each collocation point 
  for n in range( num_nodes ):
  
    ### open results file
    node_folder = input_folder_prefix + str( n )
    results_file_name = path.join( node_folder,'/results_points', qoi_filename )
    if console_info:
      print ( "read data from\n", results_file_name )
      
    read_data = io.read_numbers( results_file_name )
    
    nb_t = np.shape(read_data)[0]

    if console_info:
      print ( "Number of lines in data:\n", nb_t )

    if n == 0:
      num_time_steps = nb_t
    else:
      num_time_steps = min( num_time_steps, nb_t )
        
    qoi_data.append( read_data )
      
  # qoi_data is a list of the node results
  # this
  for n in range( num_nodes ):
    qoi_data[n] = qoi_data[n][:num_time_steps].flatten()

  qoi_data = np.array(qoi_data)

  if console_info:
    print ( "Number of time steps:\n", num_time_steps )
    print ( "Shape of qoi data:\n", np.shape( qoi_data ) )

  return qoi_data, num_time_steps

def read_node_qoi ( work_folder, qoi_filename, num_nodes ):
  """
  Read QOI data from all collocation nodes
  input:
    work_folder: Folder with collocation node folders
    qoi_filename: File name of QOI files
    num_nodes: Number of collocation nodes
  return:
    QOI data
  """
  qoi_data = []
  input_folder_prefix = path.join( work_folder, 'node_' )
  num_time_steps = 0
  ### for each collocation point 
  for n in range( num_nodes ):
  
    ### open results file
    node_folder = input_folder_prefix + str( n )
    results_file_name = path.join( node_folder, qoi_filename )
    if console_info:
      print ( "read data from\n", results_file_name )
      
    read_data = io.read_numbers( results_file_name )
    
    nb_t = np.shape(read_data)[0]

    if console_info:
      print ( "Number of lines in data:\n", nb_t )

    if n == 0:
      num_time_steps = nb_t
    else:
      num_time_steps = min( num_time_steps, nb_t )
        
    qoi_data.append( read_data )
      
  # qoi_data is a list of the node results
  # this
  for n in range( num_nodes ):
    qoi_data[n] = qoi_data[n][:num_time_steps].flatten()

  qoi_data = np.array(qoi_data)

  if console_info:
    print ( "Number of time steps:\n", num_time_steps )
    print ( "Shape of qoi data:\n", np.shape( qoi_data ) )

  return qoi_data, num_time_steps

def create_surrogate_from_qoi ( basis, norms, nodes, weights, qoi_data, console_info=False ):
  """
  Create surrogate model from QOIs
  input:
    basis: CP basis
    norms: CP basis norms
    nodes: Quadrature nodes
    weights: Quadrature weights
  return:
    surrogate model
  """
  ### create cp surroagte
  surr = cp.fit_quadrature( basis, nodes, weights, qoi_data, norms=norms )

  if console_info:
    print ( "Surrogate model:\n", surr )

  return surr

def create_surrogate_from_qoi_files ( work_folder, qoi_filename,
                                      basis, norms, nodes, weights, type, position=0.25, time=1):
  """
  Create surrogate model from QOI data files
  Combines read_node_qoi and create_surrogate_from_qoi
  input:
    work_folder: Folder with collocation node folders
    qoi_filename: File name of QOI files
    basis: CP basis
    norms: CP basis norms
    nodes: Quadrature nodes
    weights: Quadrature weights
  return:
    surrogate model, number of time steps
  """
  ### read node qoi data
  nn = len( weights )
  if(type == 'position and time'):
    qoi_data, num_time_steps = read_node_qoi_postime ( work_folder, qoi_filename, nn )
  elif (type == 'position'):
    qoi_data, num_time_steps = read_node_qoi_pos ( work_folder, qoi_filename, nn )
  else:
    qoi_data, num_time_steps = read_node_qoi ( work_folder, qoi_filename, nn )

  ### create cp surroagte
  surr = create_surrogate_from_qoi ( basis, norms, nodes, weights, qoi_data )

  return surr, num_time_steps

def compute_percentiles ( surr, dist, perc_list, s_num=10**4, sampling='L' ):
  """
  Compute percentiles
  Based on percentiles, confidence intervals can be given
  input:
    dist: Corresponding input distribution
    surr: Surrogate model or random variable
    perc_list: List of percentiles to be evaluated
    s_num: Number of samples to compute the percentiles
    sampling: Sampling method
    Available sampling methods:
      'C':   Chebyshev nodes
      'NC':  Nested Chebyshev
      'K':   Korobov
      'R':   (Pseudo-)Random
      'RG':  Regular grid
      'NG':  Nested grid
      'L':   Latin hypercube
      'S':   Sobol
      'H':   Halton
      'M':   Hammersley
  return:
    List of percentile values
  """
  ### Take samples from corresponding distribution
  samples_dist = dist.sample(s_num, rule=sampling)

  ### Evaluate surrogate model
  samples_surr = surr(*samples_dist)

  ### Compute percentiles
  perc = [np.percentile( samples_surr, pl, axis=1 ) for pl in perc_list]

  return perc

def construct_dist_from_surr ( surr, dist, s_num=10**4, sampling='L' ):
  """
  Construct a probability distribution from a QOI surrogate
  input:
    dist: Corresponding input distribution
    surr: Surrogate model or random variable
    s_num: Number of samples to compute the percentiles
    sampling: Sampling method
    Available sampling methods:
      'C':   Chebyshev nodes
      'NC':  Nested Chebyshev
      'K':   Korobov
      'R':   (Pseudo-)Random
      'RG':  Regular grid
      'NG':  Nested grid
      'L':   Latin hypercube
      'S':   Sobol
      'H':   Halton
      'M':   Hammersley
  return:
    Probability distribution
  """
  return cp.QoI_Dist( surr, dist, s_num, rule=sampling )

def extract_single_qoi ( qoi_data, num_time_steps, ind_qoi ):
  """
  Extract single QOI time range from time range of multipe QOIs data
  input:
    qoi_data: multipe QOIs data
    num_time_steps: Number of time steps in the data
    ind_qoi: Index of the QOI in the data
  return:
    time range of the QOI value
  """
  ### Determine number of QOIs in the data
  num_qois = int( len( qoi_data ) / num_time_steps )
  
  if console_info:
    print ( "Number of QOIs:\n", num_qois )

  ### Reording to one row per time step
  qoi_data = np.reshape( qoi_data, ( num_time_steps, num_qois ) )

  ### Return the wanted QOI column, only
  return qoi_data[:num_time_steps,ind_qoi]

def plot_transient_qoi ( time_steps, mean, lower, upper, 
                         out_folder, out_filename, 
                         labelname='Data', lowername='lower', uppername='upper',
                         title='UQ PP', xlabel='Time', ylabel='Data',
                         width=16, height=9, line_width=2, font_size=12 ):
  """
  Plot a time-dependent QOI with a mean value and lower and upper bounds
  input:
    time_steps: 1D array with time steps
    mean: 1D array with mean values
    lower: 1D array with lower bound values
    upper: 1D array with upper bound values
    out_folder: the folder the plot is saved to
    out_filename: filename of the plot, including the extension
    further arguments: Configuration of the plot
  Shows and saves:
    plot
  """
  ### Check consistency of input data
  if ( np.shape( time_steps ) != np.shape( mean )
       or np.shape( time_steps ) != np.shape( lower )
       or np.shape( time_steps ) != np.shape( upper ) ):
    print ( "Array sizes missmatch!" )
    print ( "Time step shape:\n", np.shape( time_steps ) )
    print ( "Mean value shape:\n", np.shape( mean ) )
    print ( "Lower shape:\n", np.shape( lower ) )
    print ( "Upper shape:\n", np.shape( upper ) )
    quit( )

  ### Create a plot
  fig=plt.figure( )
  fig.set_size_inches( width, height )
  plt.grid( True )
  ### Plot the mean value
  plt.plot( time_steps, mean,
            label=labelname + ': mean',
            color='k', linestyle='-', linewidth=line_width )
  ### Plot the lower bound value
  plt.plot( time_steps, lower,
            label=labelname + ': ' + lowername,
            color='b', linestyle=':', linewidth=line_width )
  ### Plot the upper bound value
  plt.plot( time_steps, upper,
            label=labelname + ': ' + uppername,
            color='g', linestyle='--', linewidth=line_width )
  ### Set titles and labels
  plt.suptitle( title, fontsize=font_size )
  plt.ylabel( ylabel, fontsize=font_size )
  plt.xlabel( xlabel, fontsize=font_size )
  plt.legend( loc='best' )
  ### Save plot
  file_format = out_filename.split( '.' )[-1]
  plt.savefig( path.join( out_folder, out_filename ),
               format=file_format, dpi=fig.dpi, bbox_inches='tight' )
  ### Show interactive plot
  plt.show( )

