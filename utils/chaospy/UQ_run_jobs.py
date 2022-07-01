#!/usr/python3
console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

#from asyncio import subprocess
import subprocess
import os
import sys
import chaospy as cp
from configparser import ConfigParser

###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

sys.path.append(os.path.abspath("scripts"))

import solver_input as si
import PC_collocation as pcc

###-------------------------------------------------------------------------###
### Input and parameters 
###-------------------------------------------------------------------------###

# work directory
#work_folder = '/home/philipp/Programme/Hiflow/hiflow_dev/build_gcc_relwithdebinfo/examples/poisson_chaospy'
#work_folder = '/home/kratzke/gitlab/hiflow_build/examples/poisson_chaospy'
work_folder = '/home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq2d' 
#server_work_folder = work_folder

# script that should be executed for each node
per_node_submit_script = 'run_uq.sh'

# parameters for per_node_submit_script
per_node_submit_args = work_folder

# mounted template directory 
template_folder = os.path.join(work_folder,'template')

### Define UQ parameters

PC_degree = 12
quadrature_rule = 'g'
sparse_flag = False
dist_list = [cp.Uniform(2, 13.5), cp.Uniform(1*1e4, 1*1e6)]
comment = "variing Prandtl number and Rayleigh number from "

# save configuraiton
config = ConfigParser()
config_file = os.path.join(work_folder,'PC_collocation_config.ini')
config.add_section('main')
config.set('main', 'PC_degree', str(PC_degree))
config.set('main', 'quadrature_rule', quadrature_rule)
config.set('main', 'sparse_flag', str(sparse_flag))
config.set('main', 'dist_list', str(dist_list))
config.set('main', 'comment', comment)
with open(config_file, 'w') as f:
    config.write(f)

# every calculation of the bossinesq2d should be executed with as many cores as PC_degree (for max speed)
# as an example for a PC_degree of 8 the run_u.sh file in the examples/boussinesq2d/template foulder should read  "mpirun -np 8 $BASE_DIR/node_id/... "
# total_threads states the number of threads (likely 2xnumber of cores) of your machine. num_threads specifies how many nodes should be calculated in parallel.
# please remember to adjust the run_uq.sh file manually (in the template folder, bevor running this script)
total_threads = 24
threads_per_node = PC_degree
num_threads =total_threads//threads_per_node

basis, basis_norms, nodes, weights, dist, num_nodes = pcc.create_PC_collocation (PC_degree, quadrature_rule, sparse_flag, dist_list)

###-------------------------------------------------------------------------###
### Adapt application input files 
###-------------------------------------------------------------------------###

### copy template dir

state = si.create_node_folders( template_folder, work_folder, nodes,threads_per_node )
#state = si.create_node_folders( template_folder, work_folder, server_work_folder, nodes )

if console_info:
  print ("Created", num_nodes, "node input folders:", state)

###-------------------------------------------------------------------------###
### Test submission script creation 
###-------------------------------------------------------------------------###

si.create_job_submission_script( os.path.join( work_folder, "submission.sh" ), work_folder, num_nodes, per_node_submit_script, per_node_submit_args)

###-------------------------------------------------------------------------###
### Run application 
###-------------------------------------------------------------------------###

# linux command to stop ubuntu 20.04 to enter suspend and stop calculation
#command = 'sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target'
#process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()
#print("Output stopping sleep ",output, ", Error: ",error)
#print (error)

si.run_jobs_locally (work_folder, per_node_submit_script, per_node_submit_args, num_threads, num_nodes)

# reset suspend
#command = 'sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target'
#process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()
#print("Output stopping sleep ",output, ", Error: ",error)
#print (error)


