#!/usr/bin/env python3

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###
from __future__ import print_function

import os
import sys
import numpy as np
import chaospy as cp

console_info = True
###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

sys.path.append(os.path.abspath("scripts"))
import integration as integ
import qoi_postprocessing as qoipp
import PC_collocation as pcc

###-------------------------------------------------------------------------###
### Input and parameters 
###-------------------------------------------------------------------------###

### work directory
work_folder = '/home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq2d'
# work_folder = '/home/philipp/Programme/Hiflow/hiflow_dev/build_debug_gcc/examples/poisson_chaospy'

### location where to write the postprocessed output
pp_folder = work_folder
# pp_folder = '/home/philipp/Programme/Hiflow/hiflow_dev/build_debug_gcc/examples/poisson_chaospy'

qoi_output_name = '0.5_0.5.cvs'

### Define UQ parameters
PC_degree = 8
dist_list = [cp.Uniform(2, 13.5), cp.Uniform(1e4, 1e6)]
quadrature_rule = 'g'
sparse_flag = False

basis, basis_norms, nodes, weights, dist, num_nodes = pcc.create_PC_collocation (PC_degree, quadrature_rule, sparse_flag, dist_list)

###-------------------------------------------------------------------------###
### Create surrogate model from node results 
###-------------------------------------------------------------------------###

surr_qoi, num_time_steps = qoipp.create_surrogate_from_qoi_files ( work_folder,
         qoi_output_name, basis, basis_norms, nodes, weights )

###-------------------------------------------------------------------------###
### Calculate statistics with chaospy 
###-------------------------------------------------------------------------###

### QoI Postprocessing

### Mean values
mean_qoi = integ.mean( surr_qoi, dist )
  
### Variances
var_qoi = integ.var( surr_qoi, dist )

### Standard deviations
std_dev_qoi = integ.stddev( surr_qoi, dist )

### Higher order moments
skew_qoi = integ.skew(surr_qoi,dist)
kurt_qoi = integ.kurt(surr_qoi,dist)

### Sobol indices
main_si_qoi = integ.sens_m(surr_qoi,dist)
total_si_qoi = integ.sens_t(surr_qoi,dist)

### Percentiles
### Based on percentiles, confidence intervals can be given
p = qoipp.compute_percentiles( surr_qoi, dist, [10,90] )

### Construction of the distribution of the results
surr_dist_qoi = qoipp.construct_dist_from_surr( surr_qoi, dist)

### Probability density function evaluation
####Didn't work
#pdf_val_qoi = [surr_dist_qoi[l].pdf( mean_qoi[l] ) for l in range(len(surr_dist_qoi))]
#pdf_val_qoi = np.array( pdf_val_qoi )

if console_info:
  print ( "Means:\n", mean_qoi )
  print ( "Variances:\n", var_qoi )
  print ( "Standard deviations:\n", std_dev_qoi )
  print ( "Skewness:\n", skew_qoi )
  print ( "Kurtosis:\n", kurt_qoi )
  print ( "Main Sobol indices:\n", main_si_qoi )
  print ( "Total Sobol indices:\n", total_si_qoi )
  print ( "10th Percentiles:\n", p[0] )
  print ( "90th Percentiles:\n", p[1] )
  #print ( "pdf at mean:\n", pdf_val_qoi )
  
qoi2 = qoipp.extract_single_qoi( mean_qoi, 1, 1 )

if console_info:
  print ( "QOI2:\n", qoi2 )

time_steps = np.array(range(3))

qoipp.plot_transient_qoi( time_steps, mean_qoi,
                          mean_qoi - std_dev_qoi, mean_qoi + std_dev_qoi,
                          pp_folder, 'plot_stddev.png' )

qoipp.plot_transient_qoi( time_steps, mean_qoi, p[0], p[1], pp_folder, 'plot_10perc.png' )

