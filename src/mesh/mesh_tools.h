// Copyright (C) 2011-2021 Vincent Heuveline
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

#ifndef _MESH_TOOLS_H_
#define _MESH_TOOLS_H_

#include "mesh.h"

#include <string>

#include <mpi.h>

#include "common/parcom.h"
#include "mesh/boundary_domain_descriptor.h"
#include "mesh/periodicity_tools.h"
#include "mesh/iterator.h"
#include "mesh/types.h"
#include "mesh/geometric_tools.h"

/// @brief High-level tools for dealing with meshes
/// @author Staffan Ronnas, Thomas Gengenbach, Jonathan Schwegler, Simon
/// Gawlok, Philipp Gerstner
namespace hiflow {
namespace mesh {
class GraphPartitioner;
class MeshPartitioner;
class SharedVertexTable;

} // namespace mesh
//////////////// General interface /////////////////////
/// \brief Read in a mesh from a file.

mesh::MeshPtr read_mesh_from_file(const std::string &filename, mesh::TDim tdim,
                                  mesh::GDim gdim, const MPI_Comm *comm);

mesh::MeshPtr read_mesh_from_file(const std::string &filename, mesh::TDim tdim,
                                  mesh::GDim gdim, const MPI_Comm *comm,
                                  mesh::IMPL impl);

mesh::MeshPtr read_mesh_from_file(const std::string &filename, mesh::TDim tdim,
                                  mesh::GDim gdim, const MPI_Comm *comm,
                                  mesh::IMPL impl,
                                  std::vector< mesh::MasterSlave > period);

mesh::MeshPtr read_mesh_from_file(const std::string &filename, mesh::TDim tdim,
                                  mesh::GDim gdim, const MPI_Comm *comm,
                                  mesh::IMPL impl,
                                  std::vector< mesh::MasterSlave > period,
                                  bool is_rectangular);

/// \brief Save a mesh in a hdf5 file
void save_mesh(std::string filename, mesh::MeshPtr mesh, const MPI_Comm &comm);

/// \brief Load a mesh from a hdf5 file
mesh::MeshPtr load_mesh(std::string filename, const MPI_Comm &comm,
                        mesh::IMPL impl);

/// \brief creates a distributed mesh from sequential master_mesh for a given
/// partitioning
mesh::MeshPtr create_distributed_mesh(const mesh::MeshPtr master_mesh,
                                      const int master_rank,
                                      const MPI_Comm &comm,
                                      std::vector< int > &partitioning,
                                      mesh::IMPL impl);

/// \brief extract local mesh from sequential master_mesh for a given
/// partitioning
mesh::MeshPtr create_distributed_mesh(const mesh::MeshPtr master_mesh,
                                      const int master_rank,
                                      const MPI_Comm &comm,
                                      std::vector< int > &partitioning,
                                      mesh::IMPL impl);

/// \brief Partition and distribute mesh from one process to all processes.
/// Uses a \brief MetisGraphPartitioner if available, otherwise a
/// NaiveGraphPartitioner.
mesh::MeshPtr partition_and_distribute(const mesh::MeshPtr master_mesh,
                                       const int master_rank,
                                       const MPI_Comm &comm,
                                       int &uniform_ref_steps,
                                       mesh::IMPL impl = mesh::IMPL_DBVIEW);

mesh::MeshPtr partition_and_distribute(const mesh::MeshPtr master_mesh,
                                       const int master_rank,
                                       const MPI_Comm &comm,
                                       const std::vector<int>& cell_weights,
                                       int &uniform_ref_steps,
                                       mesh::IMPL impl = mesh::IMPL_DBVIEW);

/// \brief Partition and distribute mesh from one process to all processes
/// using \brief a graph partitioner acting on the dual graph of the mesh.
mesh::MeshPtr partition_and_distribute( const mesh::MeshPtr master_mesh, 
                                        const int master_rank,
                                        const MPI_Comm &comm, 
                                        const std::vector<int>& cell_weights,
                                        const mesh::GraphPartitioner *gpartitioner,
                                        int &uniform_ref_steps, 
                                        mesh::IMPL impl = mesh::IMPL_DBVIEW);

mesh::MeshPtr partition_and_distribute( const mesh::MeshPtr master_mesh, 
                                        const int master_rank,
                                        const MPI_Comm &comm, 
                                        const mesh::GraphPartitioner *gpartitioner,
                                        int &uniform_ref_steps, 
                                        mesh::IMPL impl = mesh::IMPL_DBVIEW);

/// \brief Partition and distribute mesh from one process to all processes
/// using \brief a mesh partitioner.
mesh::MeshPtr partition_and_distribute( const mesh::MeshPtr master_mesh, 
                                        const int master_rank,
                                        const MPI_Comm &comm, 
                                        const mesh::MeshPartitioner *mpartitioner,
                                        int &uniform_ref_steps, 
                                        mesh::IMPL impl = mesh::IMPL_DBVIEW);

/// \brief Create new mesh containing the ghost cells from neighboring
/// sub-domains. <br> CAUTION: In order to refine the mesh locally,
/// layer_width has to be at least 2.
mesh::MeshPtr compute_ghost_cells(const mesh::Mesh &local_mesh,
                                  const MPI_Comm &comm,
                                  mesh::SharedVertexTable &shared_verts,
                                  mesh::IMPL, int layer_width);

mesh::MeshPtr compute_ghost_cells(const mesh::Mesh &local_mesh,
                                  const MPI_Comm &comm,
                                  mesh::SharedVertexTable &shared_verts,
                                  mesh::IMPL);

mesh::MeshPtr compute_ghost_cells(const mesh::Mesh &local_mesh,
                                  const MPI_Comm &comm,
                                  mesh::SharedVertexTable &shared_verts);

/// \brief Interpolate attribute from parent mesh to child mesh
/// (new attributes in neighborhood are interpolated if possible
/// by neighbor values)
bool interpolate_attribute(const mesh::MeshPtr parent_mesh,
                           const std::string parent_attribute_name,
                           mesh::MeshPtr child_mesh);

/// \brief Read a mesh partitioned using the partition_mesh utility.
///
/// \details Reads volume mesh from PVtk file and boundary mesh from
/// corresponding Vtk file with ending '_bdy.vtu', in order to obtain the
/// material ids for the boundary facets. Returns the volume mesh, which does
/// not include ghost cells.
mesh::MeshPtr read_partitioned_mesh(const std::string &filename,
                                    mesh::TDim tdim, mesh::GDim gdim,
                                    const MPI_Comm &comm);

/// \brief Sets a default value for the material number of all boundary
/// facets with material number -1. Useful to distinguish between
/// boundary facets created by the partion-process and "real"/physical
/// boundary facets.
void set_default_material_number_on_bdy(mesh::MeshPtr mesh,
                                        mesh::MaterialNumber default_value = 1);
void set_default_material_number_on_bdy(mesh::MeshPtr mesh,
                                        mesh::MeshPtr bdy_mesh,
                                        mesh::MaterialNumber default_value = 1);
/// \brief: for exchanging ghost data.
/// recv_cells[p] = indices of cells on local subdomain, which are owned by process p and which receives data from p   
/// send_cells[p] = indices of cells on local subdomain, which are in ghost layer of process p 
void prepare_cell_exchange_requests(mesh::ConstMeshPtr mesh,
                                    const ParCom& parcom, 
                                    std::vector<int>& num_recv_cells,
                                    std::vector<int>& num_send_cells,
                                    std::vector< std::vector< int > >& recv_cells,
                                    std::vector< std::vector< int > >& send_cells);

template <class T>
void exchange_cell_data(const ParCom& parcom, 
                        const std::vector< int >& num_recv_cells,
                        const std::vector< std::vector< int > >& send_cells,
                        const std::vector< T >& my_cell_data,
                        std::vector< std::vector< T > >& recv_cell_data);
                        
void prepare_broadcast_cell_data(mesh::ConstMeshPtr mesh,
                                 const ParCom& parcom, 
                                 std::vector<int>& num_recv_cells,
                                 std::vector<int>& num_send_cells);
                                 
template <class T>
void broadcast_cell_data(const ParCom& parcom, 
                         const size_t data_size_per_cell,
                         const std::vector< int >& num_recv_cells,
                         const std::vector< int >& num_send_cells,
                         const std::vector< T >& my_cell_data,
                         std::vector< std::vector< T > >& recv_cell_data);
                                 
/// \brief Project the boundary of a mesh onto a domain given by the
/// BoundaryDomainDescriptor.
template <class DataType, int DIM>
void adapt_boundary_to_function(mesh::MeshPtr mesh,
                                const mesh::BoundaryDomainDescriptor<DataType, DIM> &bdd)
{
  assert(mesh != 0);
  mesh::MeshPtr bdry_mesh = mesh->extract_boundary_mesh();
  // go through the boundary mesh and project every point on the
  // boundary given through the BoundaryDomainDescriptor
  // Then replace the initial vertex with the projected vertex
  for (mesh::EntityIterator it = bdry_mesh->begin(bdry_mesh->tdim());
       it != bdry_mesh->end(bdry_mesh->tdim()); ++it) {
    mesh::MaterialNumber mat = it->get_material_number();

    for (mesh::IncidentEntityIterator it_vert = it->begin_incident(0);
         it_vert != it->end_incident(0); ++it_vert) {
      std::vector< mesh::Coordinate > vert(mesh->gdim());
      it_vert->get_coordinates(vert);
        
      Vec<DIM, DataType> pt;
      for (int d=0; d<DIM; ++d) 
      {
        pt[d] = static_cast<DataType> (vert[d]);
      }
      std::vector< DataType > new_vert = mesh::project_point<DataType, DIM>(bdd, pt, mat);
      int vert_id = it_vert->index();
      std::vector< mesh::Coordinate > cast_vert (mesh->gdim());
      for (int d=0; d<DIM; ++d)
      {
        cast_vert[d] = static_cast<mesh::Coordinate> (new_vert[d]);
      }
      bdry_mesh->replace_vertex(cast_vert, vert_id);
    }
  }
}

//////////////// MeshDbView Implementation /////////////
/// \brief Partition and distribute mesh from one process to all processes
/// using \brief a graph partitioner acting on the dual graph of the mesh.
mesh::MeshPtr
partition_and_distribute_dbview(const mesh::MeshPtr master_mesh,
                                const int master_rank, 
                                const MPI_Comm &comm,
                                const std::vector<int>& cell_weights,
                                const mesh::GraphPartitioner *gpartitioner);

/// \brief Partition and distribute mesh from one process to all processes
/// using \brief a mesh partitioner.
mesh::MeshPtr
partition_and_distribute_dbview(const mesh::MeshPtr master_mesh,
                                const int master_rank, 
                                const MPI_Comm &comm,
                                const mesh::MeshPartitioner *mpartitioner);

/// \brief Create new mesh containing the ghost cells from neighboring
/// sub-domains. Implementation for MeshDbView
mesh::MeshPtr compute_ghost_cells_dbview(const mesh::Mesh &local_mesh,
                                         const MPI_Comm &comm,
                                         mesh::SharedVertexTable &shared_verts, int layer_width);

//////////////// MeshP4est Implementation //////////////
/// \brief Partition and distribute coarse mesh. Initially, all processes
/// contain the coarse mesh and the p4est structures p4est_t and
/// p4est_connectivity_t. Note that p4est uses its own partitioner based on
/// z-curves
mesh::MeshPtr partition_and_distribute_pXest(const mesh::MeshPtr master_mesh,
                                             const int master_rank,
                                             const MPI_Comm &comm,
                                             const std::vector<int>& cell_weights,
                                             int &uniform_ref_steps);

/// \brief Create new mesh containing the ghost cells from neighboring
/// sub-domains. Implementation for MeshP4est
mesh::MeshPtr compute_ghost_cells_pXest(const mesh::Mesh &local_mesh,
                                        const MPI_Comm &comm, int layer_width);
} // namespace hiflow
#endif /* _MESH_TOOLS_H_ */
