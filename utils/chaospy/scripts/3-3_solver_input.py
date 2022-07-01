###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import os
import sys
import numpy as np
import subprocess as proc
import config

console_info = False
###-------------------------------------------------------------------------###
### Import UQ scripts ###
###-------------------------------------------------------------------------###

import IO_basics as io

def setup_jobs (mount_work_folder, server_work_folder, template_folder, 
                per_node_submit_script, per_node_submit_args, per_node_procs, 
                nodes, uq_ind = [], fixed_params = [], fixed_ind = [], 
                solver_params = [], series_params = []):
                  
  """
  input
  
  return
  
  """

  tmp, num_nodes = np.shape(nodes)
  lvl_num_nodes = []
  lvl_num_nodes.append(num_nodes)
      
  state = create_node_folders( template_folder, mount_work_folder, server_work_folder,
                               nodes, uq_ind, fixed_params, fixed_ind, 
                               solver_params, 0, series_params )
  
  if console_info:
    print ("Created", num_nodes, "node input folders:", state)

  #submission script creation 
  create_job_submission_script( os.path.join( mount_work_folder, "submission" ), server_work_folder, 
                                per_node_submit_script, per_node_submit_args, per_node_procs,
                                lvl_num_nodes)
  return num_nodes

def setup_multilevel_jobs (mount_work_folder, server_work_folder, template_folder, 
                           per_node_submit_script, per_node_submit_args, per_node_procs,
                           lvl_nodes, uq_ind = [], fixed_params = [], fixed_ind = [],
                           lvl_solver_params = [], series_params = []):

  """
  input
  
  return
  
  """
  assert(len(lvl_nodes) == len(lvl_solver_params))
  
  num_dirs = 0
  num_lvl = len(lvl_nodes)
  lvl_num_nodes = []
  
  # for each level copy template dir
  for l in range(num_lvl):   
    state = create_node_folders( template_folder, mount_work_folder, server_work_folder,
                                 lvl_nodes[l], uq_ind, fixed_params, fixed_ind,
                                 lvl_solver_params[l], l, series_params )
                                 
    tmp, num_nodes = np.shape(lvl_nodes[l])
    lvl_num_nodes.append(num_nodes)
    num_dirs += num_nodes

  #create_quad_folder (mount_work_folder, lvl_nodes)

  if console_info:
    print ("Created", num_dirs, "node input folders:", state)

  #submission script creation 
  create_job_submission_script( os.path.join( mount_work_folder, "submission" ), server_work_folder, 
                                per_node_submit_script, per_node_submit_args, per_node_procs,
                                lvl_num_nodes)
                                   
  return lvl_num_nodes
  
def create_job_submission_script ( submission_script_name, server_work_folder,
                                   per_node_submit_script, per_node_submit_args, per_node_procs,
                                   lvl_num_nodes):
  """
  Create job submission script
  input:
    submission_script_name: Path and name of submission script
    server_work_folder: Name of work folder
    per_node_submit_script: submission script for a single node
    per_node_submit_args: arguments for single node submission
    num_nodes: Number of collocation nodes
    
  writes:
    job submission script
  """
  if console_info:
    print ('Create submission script:', submission_script_name )

  ### Define the script's content
  base_content =   '#!/bin/bash\n' \
            + '#$ -S /bin/bash\n' \
            + 'CUR_DIR=\''+ server_work_folder + '\'\n'

  content = base_content
  
  num_lvl = len(lvl_num_nodes)

  cur_ctr = 0
  cur_num_procs = 0
  
  ### on each level
  for l in range(num_lvl):
    num_nodes = lvl_num_nodes[l]
    
    ### Extent content for each collocation point
    for n in range(num_nodes):
      
      ### add call to per_node_submit_script
      content +=   '\n ' \
               + 'cd ${CUR_DIR}/node_' + str(l) + '_' + str(n) + '\n' \
               + './' + per_node_submit_script + ' ' + per_node_submit_args

      cur_num_procs += per_node_procs
      if cur_num_procs >= config.available_procs - per_node_procs:
        io.write_file( content, submission_script_name + "_" + str(cur_ctr) + ".sh")
        os.system( 'chmod +x ' + submission_script_name + "_" + str(cur_ctr) + ".sh" )
        cur_num_procs = 0
        cur_ctr += 1
        content = base_content
        
  io.write_file( content, submission_script_name + "_" + str(cur_ctr) + ".sh")
  os.system( 'chmod +x ' + submission_script_name + "_" + str(cur_ctr) + ".sh" )  
  

def run_jobs_locally ( work_folder, per_node_submit_script,
                       per_node_submit_args, num_threads, lvl_num_nodes):
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
  num_lvl = len(lvl_num_nodes)

  ### for each collocation point
  for l in range(num_lvl):
    num_nodes = lvl_num_nodes[l]
    for n in range(num_nodes):

      ### run application with arguments
      node_folder = input_folder_prefix + str(l) + '_' + str(n)
      if console_info:
        print ( [os.path.join( node_folder, per_node_submit_script ), per_node_submit_args] )
      process.append( proc.Popen( [os.path.join( node_folder, per_node_submit_script ), per_node_submit_args] ) )
  
      ### avoid spawning too many processes
      if (l*num_nodes + (n+1))%num_threads == 0:
        process[-1].communicate()

  ### Wait for all processes to finish
  for p in process:
    p.communicate()

  if console_info:
    print ("Finished running the application.")

            
def adapt_node_files ( node_folder, node_id,
                       node, uq_ind = [], fixed_params = [], fixed_ind = [],
                       cur_level_params = [], cur_level = 0, series_params = [],
                       work_folder = ""):
  """
  Read, adapt and write template files into node folder
  Target values should have the placeholder "paramXYZ"
  input:
    template_folder: Folder with template files
    node_folder: folder to copy files to
    node: values of the collocation point
    cur_level_params: parameters needed for defining determininstic solver on current level
    e.g. refinement level, time step size, solver accuracy
    cur_level: level index for multilevel methods
  return:
    success state
  """
  ### Set of file types to be adapted.
  template_file_extensions = {'xml', 'sh', 'conf', 'ev'}

  # number of uncertain parameters
  uq_dim = len( node )
    
  print (node)
  print (uq_dim)
  print (uq_ind)
  
  # number of fixed parameters
  fixed_dim = len( fixed_params )
  
  # default case: uq params have are numbered from 1 to uq_dim
  if len( uq_ind ) == 0:
    uq_ind = range(1,uq_dim+1)
  
  ### for each template file
  for root, dirs, files in os.walk(node_folder, topdown=False):
    for tf in files:
      ### read template file
      cur_path = os.path.join( root, tf )
      ext = cur_path.split( '.' )[-1]
    
      if not ( ext in template_file_extensions ):
        continue
        
      f_content = io.read_file( cur_path )

      pholder = 'param' + str( 0 ) + '.'
      f_content = f_content.replace( pholder, str( node_id ) )
        
      ### for each uncertain parameter          
      for i in range( uq_dim ):

        param_i = node[i]
        ind_i = uq_ind[i]
        
        ### replace placeholder with collocation node value
        pholder = 'param' + str( ind_i ) + '.'
        f_content = f_content.replace( pholder, str( param_i ) )

      ### for each fixed parameter          
      for i in range( fixed_dim ):
        param_i = fixed_params[i]
        ind_i = fixed_ind[i]
        
        ### replace placeholder with collocation node value
        pholder = 'param' + str( ind_i ) + '.'
        f_content = f_content.replace( pholder, str( param_i ) )

      ### for each solver level parameter
      dim_lvl = len( cur_level_params )
      
      # level0 denotes current level
      pholder = 'level' + str( 0 ) + '.'
      f_content = f_content.replace( pholder, str( cur_level ) )

      pholder = 'WORKSPACE.' 
      f_content = f_content.replace( pholder, work_folder )
              
      for d in range( dim_lvl ):

        ### replace placeholder with collocation node value
        pholder = 'level' + str( d + 1 ) + '.'
        f_content = f_content.replace( pholder, str( cur_level_params[d] ) )
        
        #if type(cur_level_params[d]).__name__ == 'int':
        #  f_content = f_content.replace( pholder, str( float(cur_level_params[d] )) )
        #else:
        #  f_content = f_content.replace( pholder, str( int(cur_level_params[d] )) )
      
      for d in range( len(series_params) ):

        ### replace placeholder with collocation node value
        pholder = 'series' + str( d + 1 ) + '.'
        #print ("replace", pholder, " by ", str( series_params[d] ))
        f_content = f_content.replace( pholder, str( series_params[d] ) )
        
      ### also replace node numbers
      f_content = f_content.replace( 'node_id', 'node_' + str( cur_level ) + '_' + str( node_id ) )
        
      ### copy file to node folder
      io.write_file( f_content, cur_path )

  return True
  
def create_node_folders ( template_folder, work_folder, server_work_folder, 
                          nodes, uq_ind = [], fixed_params = [], fixed_ind = [],
                          cur_level_params = [], cur_level = 0, series_params = [] ):
  """
  Create node folders from template folder
  input:
    template_folder: Folder with template files
    work_folder: folder to create node folders in
    nodes: number of collocation points
    cur_level_params: deterministic solver parameters for current level
    cur_level: current level index for multilevel methods
  return:
    success state
  """
  ### rearrange matrix of collocation point values
  nodes = [*zip( *nodes )]

  ### for each collocation point
  for n in range( len( nodes ) ):

    node_folder = os.path.join( work_folder, "node_" + str( cur_level ) + '_' + str( n ) )
    #print (node_folder)
    
    ### delete old node folders
    os.system( "rm -rf " + node_folder )
    
    ### copy template folder -> new node folder
    state = io.copy_folder ( template_folder, node_folder )
    if not state:
      print ( "Error while creating folder" + node_folder )
      quit( ) 

    ### adapt template files in the node folders   
    state = adapt_node_files( node_folder, n, 
                              nodes[n], uq_ind, fixed_params, fixed_ind,
                              cur_level_params, cur_level, series_params,
                              server_work_folder )
    if not state:
      print ( "Error while copying templates to" + node_folder )
      quit( ) 

  return True
  
def create_quad_folder (work_folder, lvl_nodes ):
      
  quad_folder = os.path.join(work_folder,'quadrature')
  
  ### delete old quad folder
  os.system( "rm -rf " + quad_folder )
  
  ### copy template folder -> new node folder
  state = io.make_folder ( quad_folder )

  ### write quad nodes and weights into file
  num_lvl = len (lvl_nodes)
  
  for l in range(num_lvl):
    
    node_file = os.path.join(quad_folder,'nodes_' + str(l) + '.txt')
    #weight_file = os.path.join(quad_folder,'weights_' + str(l) + '.txt')
    
    np.savetxt(node_file, lvl_nodes[l])
    #np.savetxt(weight_file, lvl_weights[l])
    
  return True
  
  
  
  
