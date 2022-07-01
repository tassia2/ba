#!/usr/bin/env python3
console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import os
import sys

###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

sys.path.append(os.path.abspath("chaospy/scripts"))

import solver_input as si
import numpy as np

###-------------------------------------------------------------------------###
### Input and parameters 
###-------------------------------------------------------------------------###

# work directory
work_folder = '/home/philipp/Programme/Hiflow/hiflow_public/build_release/examples/poisson_chaospy'

# script that should be executed for each node
per_node_submit_script = 'run_job.sh'

# parameters for per_node_submit_script
per_node_submit_args = work_folder

# mounted template directory 
template_folder = os.path.join(work_folder,'template')

num_threads = 2;

# define varying input parameters
# each row defines a sequence of values for one specific parameter
# -> each column corresponds to one job
params = np.matrix([[1, 2], [3, 4]])

num_params, num_nodes = np.shape(params)

###-------------------------------------------------------------------------###
### Adapt application input files 
###-------------------------------------------------------------------------###

### copy template dir

state = si.create_node_folders( template_folder, work_folder, params )

if console_info:
  print ("Created", num_nodes, "node input folders:", state)

###-------------------------------------------------------------------------###
### Test submission script creation 
###-------------------------------------------------------------------------###

si.create_job_submission_script( os.path.join( work_folder, "submission.sh" ), work_folder, num_nodes, per_node_submit_script, per_node_submit_args)

###-------------------------------------------------------------------------###
### Run application 
###-------------------------------------------------------------------------###

si.run_jobs_locally (work_folder, per_node_submit_script, per_node_submit_args, num_threads, num_nodes)

