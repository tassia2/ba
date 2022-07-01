#!/usr/bin/env python3
console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import os
import sys
#import chaospy as cp

###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

sys.path.append(os.path.abspath("scripts"))

import solver_input as si
#import PC_collocation as pcc
#-----------------------------------------------------------#
### Input and parameters ###
#-----------------------------------------------------------#

### Get template file names
# mounted work directory
#mount_work_folder = '/home/philipp/RemoteDir/bwforDev_home/TEHD/PhDThesis/ConvTest/InstatPeriodic'
mount_work_folder = '/home/gerstner/RemoteDir/emclL1_home/scratch/TEHD/ConvTest/InstatAlgebraic'

# work directory on server
#server_work_folder = '/home/hd/hd_hd/hd_cf131/TEHD/PhDThesis/ConvTest/InstatPeriodic'
server_work_folder = '/home/gerstner/RemoteDir/emclL1_home/scratch/TEHD/ConvTest/InstatAlgebraic'

# script that should be executed for each node
per_node_submit_script = 'chain_job_bwfordev.sh'

# parameters for per_node_submit_script
per_node_submit_args = '1 1'

# name of created complete submission script
submit_script_name = mount_work_folder + '/submit_uq_jobs.sh'

# mounted template directory 
template_folder = os.path.join(mount_work_folder,'template')

### Define UQ parameters

# define varying input parameters
# each row defines a sequence of values for one specific parameter
# -> each column corresponds to one job
params = np.matrix([[1, 2], [3, 4]])

num_params, num_nodes = np.shape(params)

#-----------------------------------------------------------#
### Adapt application input files ###
#-----------------------------------------------------------#

### copy template dir

state = si.create_node_folders( template_folder, mount_work_folder, params )

if console_info:
  print ("Created", num_nodes, "node input folders:", state)

#-----------------------------------------------------------#
### Create job submisson script ###
#-----------------------------------------------------------# 

si.create_job_submission_script (submit_script_name, server_work_folder, num_nodes, per_node_submit_script, per_node_submit_args)

