###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import os
import sys
import numpy as np
import subprocess as proc
import time
import datetime

console_info = True
###-------------------------------------------------------------------------###
### Import UQ scripts ###
###-------------------------------------------------------------------------###

import IO_basics as io

def create_job_submission_script ( submission_script_name, server_work_folder,
                                   num_nodes, per_node_submit_script,
                                   per_node_submit_args ):
  """
  Create job submission script
  input:
    submission_script_name: Path and name of submission script
    server_work_folder: Name of work folder
    num_nodes: Number of collocation nodes
    per_node_submit_script: submission script for a single node
    per_node_submit_args: arguments for single node submission
  writes:
    job submission script
  """
  print ('Create submission script:', submission_script_name )

  ### Define the script's content
  content =   '#!/bin/bash\n' \
            + '#$ -S /bin/bash\n' \
            + 'CUR_DIR=\''+ server_work_folder + '\'\n'

  ### Extent content for each collocation point 
  for n in range(num_nodes):
    ### add call to per_node_submit_script
    content +=   '\n ' \
               + 'cd ${CUR_DIR}/node_' + str(n) + '\n' \
               + './' + per_node_submit_script + ' ' + per_node_submit_args

  io.write_file( content, submission_script_name )

  ### Make the script executable
  os.system( 'chmod +x ' + submission_script_name )

def run_jobs_locally ( work_folder, per_node_submit_script,
                       per_node_submit_args, num_threads, num_nodes ):
  """
  Run jobs locally 
  input:
    work_folder: Folder with collocation point folders
    per_node_submit_script: submission script name
    per_node_submit_args: submission arguments
    num_threads: number of threads to be utilised
    num_nodes: number of collocation nodes
  return:
    success state
  """
  process = []
  input_folder_prefix = os.path.join( work_folder, 'node_' )

  start_time = time.time()

  ### for each collocation point 
  for n in range(num_nodes):

    ### run application with arguments
    node_folder = input_folder_prefix + str(n)
    print ( [os.path.join( node_folder, per_node_submit_script ), per_node_submit_args] )
    process.append( proc.Popen( [os.path.join( node_folder, per_node_submit_script ),
                                per_node_submit_args] ) )
  
    ### avoid spawning too many processes
    if (n+1)%num_threads == 0:
      process[-1].communicate()

  ### Wait for all processes to finish
  for p in process:
    p.communicate()


  if console_info:
    print ("Finished running the application.")
    print("duration: ", str(datetime.timedelta(seconds=(time.time() - start_time))))

def adapt_node_files ( node_folder, node, node_id, threads_per_node ):
  """
  Read, adapt and write template files into node folder
  Target values should have the placeholder "paramXYZ"
  input:
    template_folder: Folder with template files
    node_folder: folder to copy files to
    node: values of the collocation point
  return:
    success state
  """
  ### Set of file types to be adapted.
  template_file_extensions = {'xml', 'sh', 'conf', 'ev'}

  ### for each template file 
  for root, dirs, files in os.walk(node_folder, topdown=False):
    for tf in files:
      ### read template file
      cur_path = os.path.join( root, tf )
      ext = cur_path.split( '.' )[-1]


      # set number of threads in run_uq.sh
      if(ext == "sh"):
        run_sh_content = io.read_file(cur_path)
        run_sh_content = run_sh_content.replace( 'num_threads', str(threads_per_node) ) 
        io.write_file( run_sh_content, cur_path )

    
      if not ( ext in template_file_extensions ):
        continue
        
      f_content = io.read_file( cur_path )

      ### for each uncertain parameter
      dim = len( node )
          
      for d in range( dim ):

        # ## recalculate node tempature value to to Rayleigh Prandtl number
        # f_content = f_content.replace( 'param_rayleigh', temp_to_rayleigh( node[d] ) )
        # f_content = f_content.replace( 'param_prandtl', temp_to_prandtl( node[d] ) )

        ### -old way - replace placeholder with collocation node value
        pholder = 'param' + str( d + 1 )
        f_content = f_content.replace( pholder, str( node[d] ) )

      ### also replace node numbers
      f_content = f_content.replace( 'node_id', 'node_' + str( node_id ) )
        
      ### copy file to node folder
      io.write_file( f_content, cur_path )

  return True

def create_node_folders ( template_folder, work_folder, nodes,threads_per_node=2 ):
  """
  Create node folders from template folder
  input:
    template_folder: Folder with template files
    work_folder: folder to create node folders in
    nodes: number of collocation points
  return:
    success state
  """
  ### rearrange matrix of collocation point values
  nodes = [*zip( *nodes )]

  ### for each collocation point
  for n in range( len( nodes ) ):

    node_folder = os.path.join( work_folder, "node_" + str( n ) )
    
    ### delete old node folders
    os.system( "rm -rf " + node_folder )
    
    ### copy template folder -> new node folder
    state = io.copy_folder ( template_folder, node_folder )
    if not state:
      print ( "Error while creating folder" + node_folder )
      quit( ) 

    ### adapt template files in the node folders
    state = adapt_node_files( node_folder, nodes[n], n ,threads_per_node)
    if not state:
      print ( "Error while copying templates to" + node_folder )
      quit( ) 
    
    ### create output folder
    output_folder = os.path.join( work_folder, "node_" + str( n ) + "/results_points")
    os.system( "mkdir " + output_folder )

  return True


def temp_to_rayleigh(temp):
  Grashof = 9.81 * Thermal_Expansion(temp) * 2 / (Kinematic_Viscosity(temp))**2
  return temp_to_prandtl(temp) * Grashof

def temp_to_prandtl(temp):
  return 50000/(temp**2 + 155*temp + 3700)

def Thermal_Expansion(temp):
  params = [ 1.80060020e-05, -2.79388322e-07,  3.75732212e-09, -2.42369956e-11, -4.43056317e-17, -6.76426287e-05]
  return params[0]*temp+params[1]*temp**2+params[2]*temp**3+params[3]*temp**4+params[4]*temp**5+params[5]

def Kinematic_Viscosity(temp):
  params = [-5.81444444e-08,  1.28411111e-09, -1.75888889e-11,  1.08888889e-13, 1.77603333e-06]
  return params[0]*temp+params[1]*temp**2+params[2]*temp**3+params[3]*temp**4+params[4]