// Copyright (C) 2011-2017 Vincent Heuveline
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

#ifndef HIFLOW_VISU_VTK_WRITER
#define HIFLOW_VISU_VTK_WRITER

/// \author Staffan Ronnas, Martin Baumann, Teresa Beck, Simon Gawlok, Jonas
/// Kratzke, Philipp Gerstner
///
/// \brief Visualization of finite element functions.
///
/// Using this class a Vtk (http://www.vtk.org/) unstructured grid visualization
/// file can be created. Please find detailed information about Vtk's file
/// formats at http://www.vtk.org/VTK/img/file-formats.pdf.
/// This type of visualization writes out every cell and with function values
/// provided by a user-defined evaluation function.
///
/// Please note for simulations with multiple visualization calls, that this
/// class is NOT ment to be initialized once for several visualization calls.
/// Please construct a new instantiation of the CellVisualization every single
/// time you want to visualize your data.
///


#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <mpi.h>

#include "visualization/visualization_data.h"

namespace hiflow {

/// \brief Writer for Pvtk files.
/// \details Write PVtk files and also the corresponding Vtk files.

template < class DataType, int DIM>
class VTKWriter 
{
public:
  /// \brief Ctor for PVtkWriter.
  /// \param [in] mpi_comm MPI Communicator.
  
  explicit VTKWriter(VisualizationData<DataType, DIM>& visu, 
                     const MPI_Comm &mpi_comm, 
                     const int master_rank)
      : 
        comm_(mpi_comm), master_rank_(master_rank), 
        visu_(visu),
        functions_ (visu.get_point_values()),
        functions_grad_(visu.get_grad_values()),
        functions_cell_(visu.get_cell_values()),
        mapped_pts_(visu.get_point_coords()),
        verts_(visu.get_point_ind()),
        cell_offsets_(visu.get_cell_offsets()),
        cell_types_(visu.get_cell_types())
  {}

  /// \brief Writes a single vtk unstructured grid.
  void write_vtu(const std::string &filename) const;
  
  /// \brief Writes a parallel vtk unstructured grid.
  void write_pvtu(const std::string &filename, 
                  const std::string &path_pvtu,
                  const std::string &path_pvtu2path_vtu, 
                  int num_writers) const;
                  
  /// \brief dummy functions for backward compatibility
  void write(const std::string &filename ) const 
  {
    this->write_pvtu(filename, "", "", -1);
  }
             
  void write(const std::string &filename, 
             const std::string &path_pvtu,
             const std::string &path_pvtu2path_vtu) const
  {
    this->write_pvtu(filename, path_pvtu, path_pvtu2path_vtu, -1);
  }
             
  void write(const std::string &filename, int num_writers)  const
  {
    this->write_pvtu(filename, "", "", num_writers);
  }

protected:
 
  /// The MPI Communicator.
  MPI_Comm comm_;
  const int master_rank_;
  const VisualizationData<DataType, DIM>& visu_;
  
  std::map< std::string, std::vector< DataType > >& functions_;
  std::map< std::string, std::vector< DataType > >& functions_grad_;
  std::map< std::string, std::vector< DataType > >& functions_cell_;
  
  std::vector<DataType>& mapped_pts_;
  std::vector<int>& verts_;
  std::vector<size_t>& cell_offsets_;
  std::vector<int>& cell_types_;
};

} // namespace hiflow

#endif
