#!/usr/bin/env python3

# call this program with mpiexec -n 20 python UQ_postprocess_vtk_par.py

console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import os
from pathlib import Path
import sys
from unittest import result
import chaospy as cp
import numpoly
from configparser import ConfigParser

###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

sys.path.append(os.path.abspath("scripts"))
import integration as integ
import qoi_postprocessing as qoipp
import PC_collocation as pcc
import vtk_postprocessing as vtkpp
import IO_vtk as iovtk
import parallel as par
import numpy as np

###-------------------------------------------------------------------------###
### Input and parameters 
###-------------------------------------------------------------------------###


# mounted work directory
work_folder = '/home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq2d' 
# create Folder to save results
result_folder_name = 'postprocess'
result_folder = os.path.join(work_folder,result_folder_name)
Path(result_folder).mkdir(exist_ok=True)

# path to pvtk and vtk within node folder
# use './' if there is no additional folder
node_2_pvtu_path = ''
node_2_vtu_path = 'vtus'
pvtu_2_vtu_path = "vtus"

# pvtu filename - defines the Timestep
vtk_output_name = 'ChannelBenchmark_solution0050.pvtu'

# defines what quantity we're interested in : p for pressure, v and u for velocity and q for density
qoi_names = ["p"]

### Define UQ parameters - manually
# PC_degree = 8
# quadrature_rule = 'g'
# sparse_flag = False
# dist_list = [cp.Uniform(2, 13.5), cp.Uniform(1e4, 1e6)]

### Get UQ parameters from Parameter file
config_r = ConfigParser()
config_r.read(os.path.join(work_folder,'PC_collocation_config.ini'))
PC_degree = int(config_r.get('main', 'PC_degree'))
quadrature_rule = config_r.get('main', 'quadrature_rule')
sparse_flag = config_r.get('main', 'sparse_flag') == 'True'
dist_str = config_r.get('main', 'dist_list')
dist_array = dist_str[1:-1].split(",") # convert string to cp.Distribution
dist_list = [None,None]
exec("dist_list[0]=cp."+dist_array[0]+","+dist_array[1])
exec("dist_list[1]=cp."+dist_array[2]+","+dist_array[3])

basis, basis_norms, nodes, weights, dist, num_nodes = pcc.create_PC_collocation (PC_degree, quadrature_rule, sparse_flag, dist_list)

# Only use certain points like points=[[64,64],[20,20]]
# leave empty to calculate a PC for all points joint
# type in ['all'] to create a folder with polynomials for all points
points = ['all']
print("only looking at points: ",points)

# save configuraiton
config_file = os.path.join(result_folder,'parameters_that_generated_poly.ini')
if not os.path.isfile(config_file):
  config = ConfigParser()
  config.read(config_file)
  config.add_section('main')
  config.set('main', 'points', str(points))
  config.set('main', 'values', str(qoi_names))
  config.set('main', 'comment', '"points" state for which points the Chaos Polynomials were fitted. They are indexed from 0-128 in the square cell. If empty all points where used.\n"values" states which physical quantity was fitted.')
  with open(config_file, 'w') as f:
      config.write(f)


pvtu_dummy = os.path.join( work_folder, "node_0", node_2_pvtu_path, vtk_output_name )
print("dummpy name: ",pvtu_dummy)
node_2_vtu_filenames, pvtu_2_vtu_filenames = iovtk.read_pvtu_structure( pvtu_dummy, node_2_pvtu_path )

print ("node_2_vtu_filenames",node_2_vtu_filenames)
print ("pvtu_2_vtu_filenames",pvtu_2_vtu_filenames)

# if all points should get a own polynomial:
if(len(points) > 0 and points[0] == 'all'):
  
  # # When some part of the polynomials are already calculated and we want to continue, not start from the beginning
  # continue_calculation = True
  
  if len(node_2_vtu_filenames) > 1:
    print("multi poly not implemented for multiple vtus (see line 109 in UQ_postprocess_vtk_par.py")
  poly_save_path = os.path.join(result_folder,'polynomials')
  Path(poly_save_path).mkdir(exist_ok=True)

  # loc_chunk = 1
  # if(continue_calculation):
  #   def cutstr(s):
  #     return int(s[:-4])
  #   # automatically find highest polynomial file, 
  #   # but if the process was interrupted, there may be missing polynomials
  #   # better use a fixed startpoint then
  #   # high = sorted(list(map(cutstr,os.listdir(poly_save_path))))[-1]
  #   high = 5990
  #   num_tasks =  128*128 
  #   loc_chunk = np.array(par.local_chunk( num_tasks ))
  #   loc_chunk = loc_chunk[loc_chunk > high]
    
  # else:
  #   num_tasks = 128*128
  #   loc_chunk = par.local_chunk( num_tasks )
  # # for i in range(128):
  # #   for j in range(128):
  # for num in loc_chunk:
  #   i = num//128
  #   j = num - i*128
  #   points = [[i,j]]
  #   # print("fitting : ",points)
  #   vtk_surr, vtk_offsets = vtkpp.create_surrogate_from_vtk( work_folder, node_2_vtu_filenames[0], qoi_names, 
  #                                   basis, basis_norms, nodes, weights, points, console_info=False )
  #   numpoly.save(os.path.join(poly_save_path,str(i*128+j)+'.npy'),vtk_surr)
  vtkpp.create_PCs_from_vtk( work_folder, node_2_vtu_filenames[0], qoi_names, \
                            basis, basis_norms, nodes, weights, poly_save_path, [], console_info=False )
  print("done")
  # exit program
  exit()

###-------------------------------------------------------------------------###
### Create surrogate model from node results 
###-------------------------------------------------------------------------###

num_tasks = len( node_2_vtu_filenames )

loc_chunk = par.local_chunk( num_tasks )

if node_2_vtu_path != "./":
  os.system("mkdir " + os.path.join( result_folder, node_2_vtu_path ))

if node_2_pvtu_path != "./":
  os.system("mkdir " + os.path.join( result_folder, node_2_pvtu_path ))
  
for rank_task in loc_chunk:

  print ( "Task:\n", rank_task )

  node_2_vtu_filename = node_2_vtu_filenames[rank_task]

  vtk_surr, vtk_offsets = \
     vtkpp.create_surrogate_from_vtk( work_folder, node_2_vtu_filename, qoi_names, 
                                      basis, basis_norms, nodes, weights, points )

###-------------------------------------------------------------------------###
### Calculate statistics with chaospy 
###-------------------------------------------------------------------------###

### VTK postprocessing
### mean
  numpoly.save(os.path.join( result_folder,"poly.npy"),vtk_surr)
  mean_vtk = integ.mean( vtk_surr, dist )
  np.save(os.path.join( result_folder,"mean.npy"),mean_vtk)
### Standard deviations
  std_dev_vtk = integ.stddev( vtk_surr, dist )
  np.save(os.path.join( result_folder,"std.npy"),std_dev_vtk)

### Main Sobol indices
  main1_vtk, main2_vtk = integ.sens_m( vtk_surr, dist )

### Total Sobol indices
  total1_vtk, total2_vtk = integ.sens_t( vtk_surr, dist )

### Percentiles
  perc_vtk = qoipp.compute_percentiles( vtk_surr, dist, [10,90] )

###-------------------------------------------------------------------------###
### Write results 
###-------------------------------------------------------------------------###
    
  dummy_file = os.path.join( work_folder, "node_0", node_2_vtu_filename )
  mean_file = os.path.join( result_folder, node_2_vtu_path, "mean_" + str(rank_task) + ".vtu" )
  stddev_file = os.path.join( result_folder, node_2_vtu_path, "std_dev_" + str(rank_task) + ".vtu" )
  main1_file = os.path.join( result_folder, node_2_vtu_path, "main1_" + str(rank_task) + ".vtu" )
  main2_file = os.path.join( result_folder, node_2_vtu_path, "main2_" + str(rank_task) + ".vtu" )
  total1_file = os.path.join( result_folder, node_2_vtu_path, "total1_" + str(rank_task) + ".vtu" )
  total2_file = os.path.join( result_folder, node_2_vtu_path, "total2_" + str(rank_task) + ".vtu" )
  p10_file = os.path.join( result_folder, node_2_vtu_path, "perc10_" + str(rank_task) + ".vtu" )
  p90_file = os.path.join( result_folder, node_2_vtu_path, "perc90_" + str(rank_task) + ".vtu" )
  iovtk.write_vtu (dummy_file, mean_file, mean_vtk, vtk_offsets, qoi_names)
  iovtk.write_vtu (dummy_file, stddev_file, std_dev_vtk, vtk_offsets, qoi_names)
  iovtk.write_vtu (dummy_file, main1_file, main1_vtk, vtk_offsets, qoi_names)
  iovtk.write_vtu (dummy_file, main2_file, main2_vtk, vtk_offsets, qoi_names)
  iovtk.write_vtu (dummy_file, total1_file, total1_vtk, vtk_offsets, qoi_names)
  iovtk.write_vtu (dummy_file, total2_file, total2_vtk, vtk_offsets, qoi_names)
  iovtk.write_vtu (dummy_file, p10_file, perc_vtk[0], vtk_offsets, qoi_names)
  iovtk.write_vtu (dummy_file, p90_file, perc_vtk[1], vtk_offsets, qoi_names)


if ( par.rank() == 0 ):
  print ("Rank 0 writes pvtu.")
  mean_file_list = [pvtu_2_vtu_path + "mean_" + str( p ) + ".vtu" for p in range( num_tasks )]
  stddev_file_list = [pvtu_2_vtu_path + "std_dev_" + str( p ) + ".vtu" for p in range( num_tasks )]
  main1_file_list = [pvtu_2_vtu_path + "main1_" + str( p ) + ".vtu" for p in range( num_tasks )]
  main2_file_list = [pvtu_2_vtu_path + "main2_" + str( p ) + ".vtu" for p in range( num_tasks )]
  total1_file_list = [pvtu_2_vtu_path + "total1_" + str( p ) + ".vtu" for p in range( num_tasks )]
  total2_file_list = [pvtu_2_vtu_path + "total2_" + str( p ) + ".vtu" for p in range( num_tasks )]
  p10_file_list = [pvtu_2_vtu_path + "perc10_" + str( p ) + ".vtu" for p in range( num_tasks )]
  p90_file_list = [pvtu_2_vtu_path + "perc90_" + str( p ) + ".vtu" for p in range( num_tasks )]
  iovtk.adapt_pvtu( pvtu_dummy, os.path.join( result_folder, node_2_pvtu_path, "mean.pvtu" ),
                    pvtu_2_vtu_filenames, mean_file_list )
  iovtk.adapt_pvtu( pvtu_dummy, os.path.join( result_folder, node_2_pvtu_path, "std_dev.pvtu" ),
                    pvtu_2_vtu_filenames, stddev_file_list )
  iovtk.adapt_pvtu( pvtu_dummy, os.path.join( result_folder, node_2_pvtu_path, "main1.pvtu" ),
                    pvtu_2_vtu_filenames, main1_file_list )
  iovtk.adapt_pvtu( pvtu_dummy, os.path.join( result_folder, node_2_pvtu_path, "main2.pvtu" ),
                    pvtu_2_vtu_filenames, main2_file_list )
  iovtk.adapt_pvtu( pvtu_dummy, os.path.join( result_folder, node_2_pvtu_path, "total1.pvtu" ),
                    pvtu_2_vtu_filenames, total1_file_list )
  iovtk.adapt_pvtu( pvtu_dummy, os.path.join( result_folder, node_2_pvtu_path, "total2.pvtu" ),
                    pvtu_2_vtu_filenames, total2_file_list )
  iovtk.adapt_pvtu( pvtu_dummy, os.path.join( result_folder, node_2_pvtu_path, "perc10.pvtu" ),
                    pvtu_2_vtu_filenames, p10_file_list )
  iovtk.adapt_pvtu( pvtu_dummy, os.path.join( result_folder, node_2_pvtu_path, "perc90.pvtu" ),
                    pvtu_2_vtu_filenames, p90_file_list )

