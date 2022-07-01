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

#ifndef _DOF_DOF_PARTITION_H_
#define _DOF_DOF_PARTITION_H_

#include <map>
#include <set>
#include <vector>
#include "mpi.h"

#include "common/log.h"
#include "dof/dof_interpolation.h"

namespace hiflow {
namespace mesh {
class Mesh;
}

namespace doffem {

template < class DataType, int DIM > class FEManager;
template < class DataType, int DIM > class RefElement;
template < class DataType > class DofInterpolation;
template < class DataType, int DIM> class NumberingStrategy;

/// \author Michael Schick<br>Martin Baumann<br>Philipp Gerstner

template < class DataType, int DIM > 
class DofPartition
{
public:
  /// Constructor
  DofPartition();

  /// Destructor
  virtual ~DofPartition() {}

  //////////////////////////////////////////////////////////////////////
  ////// simpe set functions ///////////////////////////////////////////

  /// Set MPI Communicator if needed, default value is MPI_COMM_WORLD
  void set_mpi_comm(const MPI_Comm &comm);

  /// Set given mesh to a constant pointer
  void set_mesh(const mesh::Mesh *mesh);

  /// Setting the FEManager for the given mesh
  void set_fe_manager(FEManager< DataType, DIM > const *manager);
  
  void set_numbering_strategy (NumberingStrategy< DataType, DIM> * number_strategy);

  //////////////////////////////////////////////////////////////////////
  ////// simple get functions //////////////////////////////////////////
  
  /// Get MPI communicator
  inline const MPI_Comm &get_mpi_comm() const
  {
    return comm_;
  }
  
  /// Get the FEManager for the given mesh
  inline FEManager< DataType, DIM > const &get_fe_manager() const
  {
    return *fe_manager_;
  }

  /// Get the mesh
  inline const mesh::Mesh &get_mesh() const 
  { 
    return *mesh_; 
  }

  /// Get the DoF Interpolation
  /// handles DoF interpolation for hanging nodes (h- and p-refinement)
  inline DofInterpolation<DataType> &dof_interpolation()
  {
    return dof_interpolation_;
  }
  
  inline const DofInterpolation<DataType> &dof_interpolation() const
  {
    return dof_interpolation_;
  }
  
  //////////////////////////////////////////////////////////////////////
  ////// initialize numbering //////////////////////////////////////////
  void number (DOF_ORDERING order, bool ordering_fe_cell);
    
  //////////////////////////////////////////////////////////////////////
  ////// number of dofs, subdomains, etc. //////////////////////////////
  
  /// Check if global dof is on subdomain (only if dof is owned)
  inline bool is_dof_on_subdom(gDofId global_id) const
  {
    return global_id >= my_dof_offset_ &&
         global_id < my_dof_offset_ + nb_dofs_on_subdom_[my_subdom_];
  }

  /// Returns the lowest subdomain index, which shares the global dof ID
  int owner_of_dof(gDofId global_id) const;

  /// Get dof offset of this subdomain
  inline int my_dof_offset() const
  {
    return my_dof_offset_;
  }

  /// Get subdomain index (subdomain which owns dof)
  inline int my_subdom() const
  {
    return my_subdom_;
  }
    
  /// Returns the number of dofs on the whole global computational domain
  /// (including the other subdomains)
  inline size_t nb_dofs_global() const
  {
    return gl_nb_dofs_total_;
  }
  
  inline size_t nb_dofs_global(size_t fe_ind) const
  {
    interminable_assert(fe_ind <= fe_manager_->nb_fe());
    return global_nb_dofs_for_fe_[fe_ind];
  }

  /// Get the number of dofs on my subdomain for a specific variable
  inline size_t nb_dofs_local(size_t fe_ind) const
  {
    interminable_assert(fe_ind <= fe_manager_->nb_fe());
    return local_nb_dofs_for_fe_[fe_ind];
  }

  /// Get the number of dofs on my subdomain for all variables
  inline size_t nb_dofs_local() const
  {
    return local_nb_dofs_total_;
  }

  /// Return number of subdomains (= number of MPI processes)
  inline size_t nb_subdom () const
  {
    return this->nb_subdom_;
  }
  
  /// Returns the number of dofs on a specific subdomain owned by the subdomain
  inline size_t nb_dofs_on_subdom(int subdomain) const
  {
    assert(subdomain >= 0 && subdomain < nb_subdom_);
    return nb_dofs_on_subdom_[subdomain];
  }

  /// Get the number of dofs for a specific variable on a specific cell
  inline size_t nb_dofs_on_cell(size_t fe_ind, int cell_index) const
  {
    assert (fe_ind < this->nb_dofs_on_cell_for_fe_.size());
    assert (cell_index < this->nb_dofs_on_cell_for_fe_[fe_ind].size());
    return this->nb_dofs_on_cell_for_fe_[fe_ind][cell_index];
    //return fe_manager_->get_fe(cell_index, fe_ind)->nb_dof_on_cell();
  }

  /// Get the total number of dofs on a specific cell (including all variables)
  inline size_t nb_dofs_on_cell(int cell_index) const
  {
    assert (cell_index < this->nb_dofs_on_cell_.size());
    return this->nb_dofs_on_cell_[cell_index];
    
    /*
    int result = 0;
    for (size_t fe_ind = 0; fe_ind < fe_manager_->nb_fe(); ++fe_ind)  
    {
     result += this->nb_dofs_on_cell(fe_ind, cell_index);
    }
    return result;
    */
  }

  /// Get the number of dofs on a specific subentity for a specific variable on a
  /// specific cell
  size_t nb_dofs_on_subentity(size_t fe_ind, int cell_index, int tdim, int sub_index) const;
  
  /// Get the total number of dofs on a specific subentity on a specific cell (including
  /// all variables)
  size_t nb_dofs_on_subentity(int cell_index, int tdim, int sub_index) const;

  /// Returns numbre of Elements
  inline size_t nb_fe () const 
  {
    return this->nb_fe_;
  }

  /// Returns number of physical variables
  inline size_t nb_var () const 
  {
    return this->nb_var_;
  }

  /// Returns topological dimension of underlying mesh
  inline size_t tdim() const 
  {
    return this->tdim_;
  }
  
  //////////////////////////////////////////////////////////////////////
  ////// get dof indices ///////////////////////////////////////////////
  
  /// Get the global DofIds (w.r.t. complete mesh) on a specific mesh cell and variable
  void get_dofs_on_cell(size_t fe_ind, int cell_index, std::vector< gDofId > &ids) const;
  void get_dofs_on_cell(int cell_index, std::vector< gDofId > &ids) const;
    
  /// Returns dofs on cell for a specific variable. The lDofIds are local
  /// w.r.t. subdomain (Including the dofs which are not owned)
  void get_dofs_on_cell_local(size_t fe_ind, int cell_index, std::vector< lDofId > &ids) const;
  void get_dofs_on_cell_local(int cell_index, std::vector< lDofId > &ids) const;

  /// Get the factors with which dof values have to be multiplied when evaluating 
  /// an FE function (+- 1, due to orientation of cell normal in case of Hdiv conforming elements)
  void get_dof_factors_on_cell(int cell_index, std::vector< DataType > &ids) const;

  /// Get the global DofIds on a subentity (point, edge, fase) of a cell
  void get_dofs_on_subentity(size_t fe_ind, int cell_index, int tdim, int sindex, std::vector< gDofId > &ids) const;

  //////////////////////////////////////////////////////////////////////
  ////// numbering mappings ////////////////////////////////////////////
  
  /// For a given local DofId on a subdomain, this routine computes the global
  /// DofId (including the local dofs which are not owned have local number on
  /// subdomain)
  void local2global(lDofId local_id, gDofId *global_id) const;
  
  /// For a given global DofId, this routine computes the local DofId on the
  /// current subdomain (including the dofs which are not owned have local
  /// number on subdomain)
  void global2local(gDofId global_id, lDofId *local_id) const;

  /// \brief Mapping local 2 global. Local DofId is local to a specific
  ///        cell and a specific variable. 
  /// Global DofId is the DOF Id on the local mesh, unless DofPartitonGlobal::renumber is called.
  /// Then, global id refers to the whole mesh
  /// cell2local = cell2global before renumber() is called
  inline gDofId cell2global(size_t fe_ind, int cell_index, cDofId local_on_cell_fe_id) const
  {
    return this->numer_cell_2_global_[numer_cell_2_global_offsets_[fe_ind][cell_index] + local_on_cell_fe_id];
  }
  
  /*
  inline gDofId cell2global(int cell_index, cDofId local_on_cell_id) const
  {
    return this->numer_cell_2_global_[numer_cell_2_global_offsets_per_cell_[cell_index] + local_on_cell_id];
  }
  */
  
  inline lDofId cell2local (size_t fe_ind, int cell_index, cDofId local_on_cell_fe_id) const
  {
    return this->numer_cell_2_local_[numer_cell_2_global_offsets_[fe_ind][cell_index] + local_on_cell_fe_id];
  }

  /*
  inline lDofId cell2local (int cell_index, cDofId local_on_cell_id) const
  {
    return this->numer_cell_2_local_[numer_cell_2_global_offsets_per_cell_[cell_index] + local_on_cell_id];
  }
  */
  
  inline DataType cell2factor(int cell_index, cDofId local_on_cell_id) const
  {
    return this->cell_2_dof_factor_[cell_index][local_on_cell_id];
  }
  
  /// Returns index of Element for given physical variable
  inline size_t var_2_fe(size_t var) const
  {
    assert (this->fe_manager_ != nullptr);
    return this->fe_manager_->var_2_fe (var);
  }
  
  void global2cells (gDofId gl, std::vector<int>& cells) const;
  void global2cells (gDofId gl, 
                     std::vector<int>::iterator insert_pos,
                     std::vector<int>& cells, 
                     std::vector<int>::iterator& end_pos) const;
                     
  void fe2global (size_t fe_ind, std::vector<gDofId>& ids) const;
  
  //////////////////////////////////////////////////////////////////////
  ////// misc //////////////////////////////////////////////////////////
  
  /// Apply permutation of DoF IDs
  /// \param description is an optional parameter that should describe what
  ///                    the permutation represents
  void apply_permutation(const std::vector< DofID > &permutation, const std::string & = "");
  
  /// Printing information about the numbering field
  void print_numer() const;

  /// Check that ordering of vertices of mesh entity is valid.
  bool check_mesh();

private:
  template <class Scalar, int DIMENSION> friend class NumberingLagrange;
  template <class Scalar, int DIMENSION> friend class NumberingStrategy;
  
  ////// routines accessed only by numbergin strategies ////////////////
  /// Set whether numbering was performed
  inline void set_applied_number_strategy (bool flag) 
  {
    this->applied_number_strategy_ = flag;
  }

  inline std::vector< gDofId > * numer_cell_2_global() 
  {
    return &numer_cell_2_global_;
  }

  inline std::vector< lDofId > * numer_cell_2_local() 
  {
    return &numer_cell_2_local_;
  }
    
  inline std::vector< std::vector<DataType > > * cell_2_dof_factor() 
  {
    return &cell_2_dof_factor_;
  }

  inline std::vector< std::vector< int > > * numer_cell_2_global_offsets()
  {
    return &numer_cell_2_global_offsets_;
  }

  /*
  inline std::vector< int > * numer_cell_2_global_offsets_per_cell()
  {
    return &numer_cell_2_global_offsets_per_cell_;
  }
  */
  
  /// Create ownerships of dofs (needed for identification of sceleton dofs)
  void create_ownerships();
  
  /// Renumber dofs according to subdomains to achive global unique numbering
  void renumber();
  
  /// In parallel world, dof factors of cells shared by several subdomains
  /// may differ. By this routine, dof factors for each dof on a shared cell
  /// are set to those factors, that corresponds to the owner subdomain
  void correct_dof_factors();
  
#if 0
  /// Create local and global correspondences
  void consecutive_numbering();

  /// Permute the local lDofIds w.r.t. the subdomain by a given permutation
  /// (Including the dofs which are not owned)
  void apply_permutation_local(const std::vector< lDofId > &permutation);
#endif

  /// Update number_of_dofs_total_ and number_of_dofs_for_var_
  /// \param description is an optional parameter that should describe what
  ///                    the permutation represents
  void update_number_of_dofs(DofID& total_nb_dof,
                             std::vector<DofID>& nb_dof_for_fe,
                             const std::string & = "") const;
  
  /// Check, if dof factors are consistent over all processes
  void check_dof_factors();
  
  //////////////////////////////////////////////////////////////////////
  ////// Reference and pointers to complex objects /////////////////////
  
  /// MPI Communicator
  MPI_Comm comm_;
  
  /// Const pointer to mesh
  const mesh::Mesh *mesh_;

  /// FEManager on the given mesh
  FEManager< DataType, DIM > const *fe_manager_;

  /// Interpolation Container, which stores the interpolation weights
  DofInterpolation<DataType> dof_interpolation_;
  
  // underlying numbering strategy
  NumberingStrategy< DataType, DIM> * number_strategy_;
  
  //////////////////////////////////////////////////////////////////////
  ////// simple numbers ////////////////////////////////////////////////
  
  /// Topological dimension
  size_t tdim_;

  /// Total number of variables
  size_t nb_var_;
  
  /// number of elements
  size_t nb_fe_;
  
  /// Subdomain index
  int my_subdom_;
  
  /// Offset for this subdomain
  gDofId my_dof_offset_;
  
  /// Number of dofs including ghost layer
  lDofId nb_dofs_incl_ghost_;
  
  /// Total number of subdomains
  int nb_subdom_;
  
  // Number of dofs on the global computational domain
  /// (including the other subdomains)
  gDofId gl_nb_dofs_total_;
  
  /// Total number of dofs for all variables
  lDofId local_nb_dofs_total_;
  
  /// Total number of dofs per variable
  std::vector< lDofId > local_nb_dofs_for_fe_;
  std::vector< gDofId > global_nb_dofs_for_fe_;
  
  /// Number of dofs for a specific subdomain owned by the subdomain
  std::vector< lDofId > nb_dofs_on_subdom_;
  
  std::vector< std::vector< int > > nb_dofs_on_cell_for_fe_;
  std::vector< int > nb_dofs_on_cell_;
  
  //////////////////////////////////////////////////////////////////////
  ////// mappings //////////////////////////////////////////////////////
  
  /// Vector storing information for the mapping local 2 global
  std::vector< gDofId > local2global_;
  
  /// Map storing information for the mapping global 2 local
  std::map< gDofId, lDofId > global2local_;
  
  /// Map storing all cells that are attached to a given global DoF
  std::map< gDofId, std::set<int> > global2cells_;
  std::vector< std::set<gDofId> > fe2global_;
  
  /// Vector storing local (w.r.t. subdomain) numer_ field
  /// in sequential: numer_cell_2_local_ = numer_cell_2_global_
  /// in parallel: numer_cell_2_local_ = numer_cell_2_global_ before renumber() is called
  std::vector< lDofId > numer_cell_2_local_;

  /// Holds DoF IDs, needed for cell2global
  std::vector< gDofId > numer_cell_2_global_;
  
  /// Offset container for numer_, needed for cell2global
  /// [fe_ind][cell_index] = offset in numer_cell_2_global for given fe index and cell index
  
  std::vector< std::vector< gDofId > > numer_cell_2_global_offsets_;
  //std::vector< int > numer_cell_2_global_offsets_per_cell_;
  
  std::vector< std::vector<DataType > > cell_2_dof_factor_;

  /// Ghost subdomain indices, which are needed in case of point to point
  /// communication
  std::vector< bool > shared_subdomains_;

  /// Vector of ownership
  std::vector< int > ownership_;

  //////////////////////////////////////////////////////////////////////
  ////// some flags ////////////////////////////////////////////////////
  
  /// numbering strategy was applied
  bool applied_number_strategy_;
  
  /// Check if mesh is set
  bool mesh_flag_;
  
  /// Check if fe_manager is set
  bool fe_manager_flag_;
  
  bool number_strategy_flag_;
  
  bool called_renumber_;
  
  bool ordering_fe_cell_;
};


} // namespace doffem
} // namespace hiflow
#endif
