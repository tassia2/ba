// Copyright (C) 2011-2020 Vincent Heuveline
//
// HiFlow3 is free software: you can redistribute it and/or modify it under the
// terms of the European Union Public Licence (EUPL) v1.2 as published by the
// European Union or (at your option) any later version.
//
// HiFlow3 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the European Union Public Licence (EUPL) v1.2 for
// more details.
//
// You should have received a copy of the European Union Public Licence (EUPL)
// v1.2 along with HiFlow3.  If not, see
// <https://joinup.ec.europa.eu/page/eupl-text-11-12>.

#ifndef HIFLOW_VISU_XDMF_WRITER
#define HIFLOW_VISU_XDMF_WRITER


/// This class provides functions to visualize FeFunctions using only
/// values saved in the corresponding CoupledVector. The generated outputs
/// are a HDF5 file storing the "heavy data" and a XDMF file that can be
/// opened with e.g. ParaView (www.paraview.org/).
/// It is not (yet) possible to use this class to append the visualization
/// data to an existing one instead it will overwrite the previous.

/// concerning the mesh data strucutre, the XDFM file provides the possibility
/// to set the coordinates (Geometry in XDFM file) based on the geometrical
/// dimension (2D or 3D). However, in order to keep consistent with
/// CellVisualization, the current implementation write the coordinates of the
/// mesh directly in 3D.

#include "config.h"


#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <mpi.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <iterator>

#include "common/log.h"
#include "visualization/visualization_data.h"
#include "space/vector_space.h"
#include "mesh/mesh.h"
#include "mesh/attributes.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"

#ifdef WITH_HDF5
#include "common/hdf5_tools.h"
#else
#define ERROR                                                                  \
  LOG_ERROR("HiFlow was not compiled with HDF5 support!");                     \
  exit(-1);
#endif

#include "linear_algebra/coupled_vector.h"
#ifdef WITH_HYPRE
#include "linear_algebra/hypre_vector.h"
#endif

#include <tinyxml2.h>

namespace hiflow {

template <class DataType, int DIM>
class XDMFWriter 
{

public:

  explicit XDMFWriter(VisualizationData<DataType, DIM>& visu, 
                      const MPI_Comm &mpi_comm, 
                      const int master_rank)
    : comm_(mpi_comm), master_rank_(master_rank), xdmf_file_(true), visu_(visu),
    functions_ (visu.get_point_values()),
    functions_grad_(visu.get_grad_values()),
    functions_cell_(visu.get_cell_values()),
    mapped_pts_(visu.get_point_coords()),
    verts_(visu.get_point_ind()),
    cell_offsets_(visu.get_cell_offsets()),
    cell_types_(visu.get_cell_types())
  {}

  void write(const std::string& filename, 
             const std::string& filepath = "",
             const std::string& filename_mesh = "", 
             const bool& write_mesh = true);

  void clear();
  void clear_topology_data();

private:

  void get_global_dimensions();
  void get_offsets();

  void create_xdmf_topology(const int& offset) const;

  void write_xdmf_dataitem(tinyxml2::XMLElement* xdmf_element, const int& dim,
                           const std::string& nb_type, const std::string& format,
                           const std::string& filename, const std::string& groupname,
                           const std::string& datasetname, const int& precision = -1);

  /// The MPI Communicator.
  MPI_Comm comm_;
  const int master_rank_;

  /// xdmf topology
  mutable std::vector<int> xdmf_topology_;

  /// global dimensions
  int global_element_, global_topology_, global_geometry_;
  std::map<std::string, int> global_attributes_;
  std::map<std::string, int> global_attributes_grad_;
  std::map<std::string, int> global_attributes_cell_;

  /// offsets
  int offset_topology_, offset_geometry_;
  std::map<std::string, int> offset_attributes_;
  std::map<std::string, int> offset_attributes_grad_;
  std::map<std::string, int> offset_attributes_cell_;

  /// xdmf file
  tinyxml2::XMLDocument xdmf_file_;
  
  VisualizationData<DataType, DIM>& visu_;
  
  std::map< std::string, std::vector< DataType > >& functions_;
  std::map< std::string, std::vector< DataType > > & functions_grad_;
  std::map< std::string, std::vector< DataType > >& functions_cell_;
  
  std::vector<DataType>& mapped_pts_;
  std::vector<int>& verts_;
  std::vector<size_t>& cell_offsets_;
  std::vector<int>& cell_types_;
};

} // namespace hiflow
#endif
