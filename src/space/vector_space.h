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

#ifndef HIFLOW_VECTOR_SPACE_H_
#define HIFLOW_VECTOR_SPACE_H_

#include <vector>
#include "mpi.h"
#include "common/vector_algebra_descriptor.h"
#include "dof/dof_fem_types.h"
#include "fem/fe_manager.h"
#include "mesh/types.h"
#include "space/space_types.h"

#include "dof/dof_partition.h"
#include "fem/fe_reference.h"
#include "fem/fe_mapping.h"
#include "linear_algebra/vector.h"
#include "linear_algebra/la_couplings.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"
#include "mesh/mesh.h"
#include "linear_algebra/la_couplings.h"


namespace hiflow {

/// @brief A HiFlow vector space.
/// @author Martin Baumann, Chandramowli Subramanian, Michael Schick, Philipp Gerstner
///
/// Contains FEM and DOF data.

namespace mesh{
class Mesh;
class Entity;
class EntityIterator;
}

namespace doffem {
template <class DataType, int DIM> class DofPartition;
template <class DataType, int DIM> class FEManager;
template <class DataType, int DIM> class RefElement;
template <class DataType, int DIM> class CellTransformation;
}
 
template < class DataType, int DIM > 
class VectorSpace {
public:
  using Coord = Vec<DIM, DataType>;

  typedef mesh::Mesh Mesh;
  typedef mesh::Entity MeshEntity;
  typedef mesh::EntityIterator MeshEntityIterator;
  typedef mesh::EntityNumber MeshEntityNumber;
  typedef mesh::Coordinate Coordinate;

  typedef doffem::gDofId gDofId;
  typedef doffem::lDofId lDofId;
  typedef doffem::cDofId cDofId;
  typedef doffem::DofPartition< DataType, DIM > DofPartition;
  typedef doffem::FEManager< DataType, DIM > FEManager;
  typedef doffem::RefElement< DataType, DIM > RefElement;
  typedef doffem::FEType FEType;
  typedef doffem::CellTransformation< DataType, DIM > CellTransformation;
  typedef doffem::CCellTrafoSPtr< DataType, DIM> CCellTrafoSPtr;
  typedef la::Vector<DataType> Vector;
  typedef la::LaCouplings LaCouplings;
  
  VectorSpace();
  VectorSpace(const MPI_Comm &comm);
  virtual ~VectorSpace()
  {
    this->clear();
  }

  inline const MPI_Comm &get_mpi_comm() const 
  { 
    return comm_; 
  }
  
  inline int rank() const 
  {
    return this->rank_; 
  }
  
  /// @return dof
  inline const DofPartition &dof() const 
  { 
    return *(this->dof_); 
  }

  inline DofPartition &dof() 
  { 
    return *(this->dof_); 
  }

  /// @return fe manager
  inline const FEManager &fe_manager() const 
  { 
    return *(this->fe_manager_); 
  }

  inline FEManager &fe_manager() 
  { 
    return *(this->fe_manager_); 
  }

  inline const LaCouplings &la_couplings() const
  {
    return this->couplings_;
  }
 
  /// @return mesh
  inline const Mesh &mesh() const 
  { 
    return *(this->mesh_); 
  }

  inline mesh::MeshPtr meshPtr() const 
  { 
    return this->mesh_; 
  }

  /// @return topological dimension
  inline size_t tdim() const
  { 
    return this->fe_manager_->tdim(); 
  }

  /// @return number of variables
  inline size_t nb_var() const
  { 
    return this->fe_manager_->nb_var(); 
  }

  inline size_t nb_fe() const
  { 
    return this->fe_manager_->nb_fe(); 
  }

  inline size_t var_2_fe (size_t var) const
  {
    return this->fe_manager_->var_2_fe(var);
  }

  inline size_t var_2_comp (size_t var) const
  {
    return this->fe_manager_->var_2_comp(var);
  }

  inline std::vector<size_t> fe_2_var (size_t fe_ind) const
  {
    return this->fe_manager_->fe_2_var(fe_ind);
  }

  inline size_t fe_comp_2_var (size_t fe_ind, size_t comp) const
  {
    assert (fe_ind < this->nb_fe());
    assert (comp < DIM);
    assert (comp < this->fe_2_var(fe_ind).size());
    return this->fe_comp_2_var_[fe_ind * DIM + comp];
  }

  inline size_t is_dg (size_t fe_ind) const
  {
    assert (fe_ind < this->nb_fe());
    return this->is_dg_[fe_ind];
  }
  
  inline doffem::FEConformity fe_conformity (size_t fe_ind) const
  {
    assert (fe_ind < this->nb_fe());
    return this->fe_manager_->fe_conformity(fe_ind);
  }
  
  inline size_t nb_subdom () const
  {
    return this->dof_->nb_subdom();
  }
   
  /// @return number of DOFs on subentity of cell
  /// @param cell cell
  inline cDofId nb_dof_on_subentity(int cell_index, int tdim, int subindex) const
  {
    return this->dof_->nb_dofs_on_subentity(cell_index, tdim, subindex);
  }

  /// @return number of DOFs on @em cell
  /// @param cell cell
  inline cDofId nb_dof_on_cell(int cell_index) const
  {
    return this->dof_->nb_dofs_on_cell(cell_index);
  }

  inline cDofId nb_dof_on_cell(size_t fe_ind, int cell_index) const
  {
    return this->dof_->nb_dofs_on_cell(fe_ind, cell_index);
  }

  /// Returns the number of dofs on the whole global computational domain
  /// (including the other subdomains)
  inline gDofId nb_dofs_global() const
  {
    return this->dof_->nb_dofs_global();
  }

  inline gDofId nb_dofs_global(size_t fe_ind) const
  {
    return this->dof_->nb_dofs_global(fe_ind);
  }

  inline gDofId my_dof_offset() const
  {
    return this->dof_->my_dof_offset();
  }

  inline bool is_local_dof(gDofId global_id) const
  {
    return this->dof_->is_dof_on_subdom(global_id);
  }

  /// Get the number of dofs on my subdomain for a specific variable (without ghost dofs)
  inline lDofId nb_dofs_local(size_t fe_ind) const
  {
    return this->dof_->nb_dofs_local(fe_ind);
  }

  /// Get the number of dofs on my subdomain for all variables (without ghost dofs)
  inline lDofId nb_dofs_local() const
  {
    return this->dof_->nb_dofs_local();
  }
  
  /// Get the number of ghost dofs on my subdomain for a specific variable
  inline lDofId nb_dofs_ghost(size_t fe_ind) const
  {
    return this->nb_ghost_dofs_per_fe_[fe_ind];
  }

  /// Get the number of ghost dofs on my subdomain for all variables 
  inline lDofId nb_dofs_ghost() const
  {
    return this->nb_ghost_dofs_;
  }

  inline gDofId ownership_begin() const {
    return this->ownership_begin_;
  }

  /// @return One more than the global index of the last local entry

  inline gDofId ownership_end() const {
    return this->ownership_end_;
  }

  /// Print status information
  inline void print() const
  {
    dof_->print_numer();
  }

  /// Clear space (DoF, Fem, ...)
  void clear();

  /// Initialize vector valued space (DoF, Fem, ...)
  /// @param[in] mesh : underlying triangulation
  /// @param[in] fe_ansatz : vector containing specifier for FE ansatz space for each variable. <br>
  /// possible choices defined in dof_fem_types.h
  /// @param[in] is_cg : is_cg[i] = false <=> interface dofs if fe ansatz i are NOT identified
  /// @param[in] param : param[i] set of parameters needed for defining Fe ansatz i. For most cases, degree[i] is a vector of size 1, 
  /// determining the local polynomial degree (fixed across all cells)
  /// @param[in] order : type of DOF ordering
  void Init(Mesh &mesh, 
            const std::vector< FEType > & fe_ansatz,
            const std::vector< bool > & is_cg,
            const std::vector< int > &degrees,
            hiflow::doffem::DOF_ORDERING order = hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC,
            bool order_fe_before_cell = true);

  void Init(Mesh &mesh, 
            const std::vector< FEType > & fe_ansatz,
            const std::vector< bool > & is_cg,
            const std::vector<std::vector< std::vector< int > > > &params,
            hiflow::doffem::DOF_ORDERING order = hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC,
            bool order_fe_before_cell = true);
            
  /// Get dof indices on the boundary of a cell of one variable.
  /// @param var variable number
  /// @param cell cell
  /// @param tdim dimension of subentity
  /// @param sindex number of subentity
  /// @param inddof dof indices
  inline void get_dof_indices_on_subentity(size_t fe_ind , 
                                           int cell_index, 
                                           int tdim, 
                                           int sindex, 
                                           std::vector< gDofId >& inddof) const
  {
    this->dof_->get_dofs_on_subentity(fe_ind, cell_index, tdim, sindex, inddof);
  }

  /// Get all boundary dof indices on a cell.
  /// @param cell cell
  /// @param inddof dof indices
  void get_dof_indices_on_subentity(int cell_index, int tdim, int sindex, 
                                    std::vector< gDofId >& inddof) const;

  /// Get dof indices on a cell of one variable.
  /// @param var variable number
  /// @param cell cell
  /// @param inddof dof indices
  inline void get_dof_indices(size_t fe_ind, int cell_index, 
                              std::vector< gDofId >& inddof) const
  {
    assert (cell_index >= 0);
    assert (cell_index < this->mesh_->num_entities(DIM));
    this->dof_->get_dofs_on_cell(fe_ind, cell_index, inddof);
  }

  inline void get_dof_indices_local(size_t fe_ind, int cell_index, 
                                    std::vector< lDofId >& inddof) const
  {
    assert (cell_index >= 0);
    assert (cell_index < this->mesh_->num_entities(DIM));
    this->dof_->get_dofs_on_cell_local(fe_ind, cell_index, inddof);
  }

  /// Get all dof indices on a cell.
  /// @param cell cell
  /// @param inddof dof indices
  void get_dof_indices(int cell_index, std::vector< gDofId >& inddof) const;
 
  void get_local_dof_indices(const std::vector< size_t > &fe_inds,
                             std::vector<gDofId>& dof_indices) const;

  /// @return cell transformation of @em cell
  ///         only dependent on mesh
  inline CCellTrafoSPtr get_cell_transformation(int cell_index/*, size_t fe_ind=0*/) const
  {
    assert (cell_index >= 0);
    return this->fe_manager().get_cell_transformation(cell_index/*, fe_ind*/);
  }

  /// Get mapping of dofs to finite element function for variables defined in
  /// vars. Output vector is ordered by global indices of the specific dofs.
  /// This functionality is needed/useful if a system of PDE is solved and the
  /// BoomerAMG preconditioner of the hypre software package is used.
  /// @param[in] vars Variables/solution function of which the mapping global-
  /// dof-id -> var is needed
  std::vector< gDofId > get_dof_func(std::vector< size_t > fe_inds = std::vector< size_t >(0)) const;
  std::vector< gDofId > get_dof_func_for_var(std::vector< size_t > vars = std::vector< size_t >(0)) const;

  /// Get all dofs including ghost dofs for a range of variables.
  /// The returned vector of dofs can be used to initialize block-LA objects.
  /// @param[in] vars Variables for which the dofs are needed
  /// @return vector of global dof ids
  std::vector< gDofId > get_dofs(const std::vector< size_t > &fe_inds) const;
  std::vector< gDofId > get_dofs_for_var(const std::vector< size_t > &vars) const;

  // for a given vector of global dof ids, return the corresponding local or ghost index
  // if global_ids[i] is a local dof, then ghost_ids[i] = -1
  // if global_ids[i] is a ghost dof, then local_ids[i] = -1
  
  void global_2_local_and_ghost (const std::vector<gDofId>& global_ids,
                                 std::vector<lDofId>& local_ids,
                                 std::vector<lDofId>& ghost_ids) const;
  
  void extract_dof_values (size_t fe_ind, 
                           int cell_index,
                           const Vector &sol,
                           std::vector<DataType> &dof_values) const;
  
  void insert_dof_values (size_t fe_ind, 
                          int cell_index,
                          Vector &sol,
                          const std::vector<DataType> &dof_values) const;

  void set_print (bool flag) const
  {
    this->print_ = flag;
  }

private:

  // no implementation of copy constructor or assignement operator
  VectorSpace(const VectorSpace &);
  VectorSpace operator=(const VectorSpace &);
  
  /// Get data to initialize LaCouplings, i.e. needed ghost and border indices.
  /// The global order of variables must be sliced by owner.
  /// @param global_offsets Offsets specifying the owner process rank of the
  /// variables
  /// @param ghost_dofs Global (DoF) indices of the ghost entries
  /// (must be sliced by owner; may contain duplicates)
  /// @param ghost_offsets Offsets specifying the owner process rank of the
  /// ghost variables
  void setup_la_couplings();
                        
  // DegreeOfFreedom*  dof_;
  doffem::DofPartition<DataType, DIM> * dof_;
  doffem::FEManager<DataType, DIM> * fe_manager_;

  Mesh *mesh_; // only a reference

  LaCouplings couplings_;
  
  /// MPI Communicator
  MPI_Comm comm_;
  
  int rank_;
  
  mutable bool print_;
  
  la::LaCouplings la_couplings_;
  gDofId ownership_begin_ = -1;
  gDofId ownership_end_ = -1;
  
  std::vector<bool> is_dg_;  

  std::vector<gDofId> nb_ghost_dofs_per_fe_;
  gDofId nb_ghost_dofs_; 

  std::vector<size_t> fe_comp_2_var_;
  mutable std::vector< gDofId > _global_dof_ids;
  mutable std::vector< DataType > _dof_factors;
  mutable std::vector< DataType > _vector_values;
  mutable std::vector< gDofId > _inddof_var;
};

} // namespace hiflow

#endif // HIFLOW_VECTOR_SPACE_H_
