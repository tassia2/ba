#!/usr/bin/env python3
console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import os
import sys
import chaospy as cp

###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

sys.path.append(os.path.abspath("scripts"))

import solver_input as si
import PC_collocation as pcc
#-----------------------------------------------------------#
### Input and parameters ###
#-----------------------------------------------------------#

### Get template file names
# mounted work directory
mount_work_folder = '/home/philipp/RemoteDir/bwforDev_home/TEHD/Projekt/AK5_7K_0kV_3D_stat_meso_finite_dT_uniform_1'

# work directory on server
server_work_folder = '/home/hd/hd_hd/hd_cf131/TEHD/Projekt/AK5_7K_0kV_3D_stat_meso_finite_dT_uniform_1'

# script that should be executed for each node
per_node_submit_script = 'chain_job_bwfordev.sh'

# parameters for per_node_submit_script
per_node_submit_args = '1 1'

# name of created complete submission script
submit_script_name = mount_work_folder + '/submit_uq_jobs.sh'

# mounted template directory 
template_folder = os.path.join(mount_work_folder,'template')

### Define UQ parameters

PC_degree = 3
dist_list = [cp.Uniform(6.5, 7.5)]
quadrature_rule = 'g'
sparse_flag = False

basis, basis_norms, nodes, weights, dist, num_nodes = pcc.create_PC_collocation (PC_degree, quadrature_rule, sparse_flag, dist_list)

#-----------------------------------------------------------#
### Adapt application input files ###
#-----------------------------------------------------------#

### copy template dir

state = si.create_node_folders( template_folder, mount_work_folder, nodes )

if console_info:
  print ("Created", num_nodes, "node input folders:", state)

#-----------------------------------------------------------#
### Create job submisson script ###
#-----------------------------------------------------------# 

si.create_job_submission_script (submit_script_name, server_work_folder, num_nodes, per_node_submit_script, per_node_submit_args)

