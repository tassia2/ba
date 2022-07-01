console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

import IO_vtk
import qoi_postprocessing as qoipp
import parallel as par

def create_surrogate_from_vtk ( work_folder, vtk_filename, qoi_names,
                                basis, norms, nodes, weights, points=[],console_info=False  ):
  """
  Create surrogate model from vtk node files
  Combines read_vtk_nodes and create_surrogate_from_qoi
  input:
    work_folder: Folder with collocation node folders
    vtk_filename: vtk file within a respective node folder
    qoi_names: names of the QOIs in the vtk files
    basis: CP basis
    norms: CP basis norms
    nodes: Quadrature nodes
    weights: Quadrature weights
  return:
    surr: surrogate model
    offsets: offsets for indexing original data arrays
  """
  ### read node qoi data
  nn = len( weights )
  nodes_data, offsets = \
    IO_vtk.read_vtk_nodes ( work_folder, vtk_filename, qoi_names, nn, points,console_info )

  ### create cp surroagte
  surr = qoipp.create_surrogate_from_qoi ( basis, norms, nodes, weights, nodes_data,console_info )

  return surr, offsets

def create_PCs_from_vtk ( work_folder, vtk_filename, qoi_names,
                                basis, norms, nodes, weights,poly_save_path, points=[],console_info=False  ):
  """
  Create surrogate model from vtk node files
  Combines read_vtk_nodes and create_surrogate_from_qoi
  input:
    work_folder: Folder with collocation node folders
    vtk_filename: vtk file within a respective node folder
    qoi_names: names of the QOIs in the vtk files
    basis: CP basis
    norms: CP basis norms
    nodes: Quadrature nodes
    weights: Quadrature weights
  return:
    surr: surrogate model
    offsets: offsets for indexing original data arrays
  """
  import numpoly
  import os
  ### read node qoi data
  nn = len( weights )
  nodes_data, offsets = \
    IO_vtk.read_vtk_nodes ( work_folder, vtk_filename, qoi_names, nn, [],console_info )


  loc_chunk = par.local_chunk( len(nodes_data[0] ))
  for point in loc_chunk:
    print(point)
    data = nodes_data[:,point]
    surr = qoipp.create_surrogate_from_qoi ( basis, norms, nodes, weights, data,console_info )
    numpoly.save(os.path.join(poly_save_path,str(point)+'.npy'),surr)
  return surr, offsets
