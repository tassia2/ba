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

/// @author Philipp Gerstner

#ifndef HIFLOW_LINEAR_ALGEBRA_INDEX_HANDLER_H
#define HIFLOW_LINEAR_ALGEBRA_INDEX_HANDLER_H

#include "assembly/assembly_types.h"
#include "common/log.h"
#include "common/vector_algebra.h"
#include "linear_algebra/linear_operator.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "space/vector_space.h"
#include "linear_algebra/lmp/lvector.h"
#include "dof/dof_fem_types.h"
#include <assert.h>
#include <cstddef>

namespace hiflow {
namespace la {

/// \brief class for handlig dof indices, use for matrix-free stencil operator
/// this one only works if only one type of FE is used

template < class LAD, int DIM> 
class IndexFunctionUniformFE  
{
public:
  typedef typename LAD::DataType DataType;
  typedef typename LAD::VectorType GlobalVector;

  IndexFunctionUniformFE()
  {}

  ~IndexFunctionUniformFE() 
  {
    this->clear();
  }

  std::vector<int>::const_iterator first_local_cell() const 
  {
    return this->local_cells_.begin();
  } 

  std::vector<int>::const_iterator last_local_cell() const 
  {
    return this->local_cells_.end();
  }

  std::vector<int>::const_iterator first_ghost_cell() const 
  {
    return this->ghost_cells_.begin();
  } 

  std::vector<int>::const_iterator last_ghost_cell() const 
  {
    return this->ghost_cells_.end();
  }

  std::vector<lDofId>::const_iterator first_local_fixed_dof() const 
  {
    return this->local_fixed_dofs_.begin();
  }

  std::vector<lDofId>::const_iterator last_local_fixed_dof() const 
  {
    return this->local_fixed_dofs_.end();
  }

  void clear() 
  {
    this->is_initialized_ = false;
  
    this->tdim_ = -1;
    this->num_cells_ = 0;
    this->N_ = 0;
    this->ownership_begin_ = 0;
    this->ownership_end_ = 0;

    this->dof_ids_per_cell_.clear();
    this->dof_local_per_cell_.clear();
    this->local_cells_.clear();
    this->ghost_cells_.clear();
    this->local_fixed_dofs_.clear();
  }

  /// cell_factors: set of cell_factors for each cell in mesh 
  /// stencil_matrices: set of stencils 

  void Init (const VectorSpace<DataType, DIM>& space, 
             bool consider_facet_integrals = true)
  {
    this->clear();
    const LaCouplings& couplings = space.la_couplings();

    ConstMeshPtr mesh = space.meshPtr();
    this->tdim_ = mesh->tdim();
    this->num_cells_ = mesh->num_entities(tdim_);
    this->N_ = space.nb_dof_on_cell(0);
    

    this->dof_ids_per_cell_.resize(num_cells_ * this->N_);
    this->dof_local_per_cell_.resize(num_cells_ * this->N_);

    auto cell_begin = mesh->begin(tdim_);
    auto cell_end = mesh->end(tdim_);

    this->ownership_begin_ = space.ownership_begin();
    this->ownership_end_ = space.ownership_end();
  
    std::vector<gDofId> cell_dof_ids;

    this->local_cells_.reserve(this->num_cells_);
    this->ghost_cells_.reserve(this->num_cells_);

    // 0: all dofs on cell and iface neighbot cell are local 
    // 1: all dofs on cell are local but some dofs on iface neighbor cell are non-local
    // 2: some dofs on cell are non-local
    std::vector<int> cell_status(this->num_cells_, 0);

    // ---------------------------------------------
    // Step 1:
    // loop through cells and get dof ids on cell
    for (auto cell_it = cell_begin; cell_it != cell_end; ++cell_it)
    {
      const auto cell_index = cell_it->index();

      // get global dof ids from cell 
      cell_dof_ids.clear();
      space.get_dof_indices(cell_index, cell_dof_ids);

      // check if global dofs on cell are local 
      int ctr = 0;
      int offset = cell_index * this->N_;
      bool all_dofs_are_local = true;
      for (auto dof_id : cell_dof_ids)
      {
        if (dof_id >= ownership_begin_ && dof_id < ownership_end_)
        {
          // dof is local
          this->dof_ids_per_cell_[offset+ctr] = dof_id - ownership_begin_;
          this->dof_local_per_cell_[offset+ctr] = 1;
        }
        else if (couplings.global2offdiag().find(dof_id) !=
                 couplings.global2offdiag().end()) 
        {
          // dof is not local
          this->dof_ids_per_cell_[offset+ctr] = couplings.Global2Offdiag(dof_id);         
          this->dof_local_per_cell_[offset+ctr] = 0;
          all_dofs_are_local = false;
        }
        else 
        {
          assert (false);
        }
        ctr++;
      }
      if (!all_dofs_are_local)
      {
        cell_status[cell_index] = 2;
      }
    }
    
    // ---------------------------------------
    // Step 2:
    // loop through iface list
    if (consider_facet_integrals)
    {
      mesh::ConstMeshPtr mesh = &space.mesh(); 
      mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);
      const int tdim =  mesh->tdim();

      for (mesh::InterfaceList::const_iterator it = if_list.begin(),
           end_it = if_list.end(); it != end_it; ++it) 
      {
        bool if_is_local = true;

        const auto master_cell_index = it->master_index();
        const bool master_is_local = all_dofs_are_local (space, master_cell_index, 
                                                         ownership_begin_, ownership_end_, cell_dof_ids);
        if_is_local = if_is_local && master_is_local;
        
        const int num_slaves = it->num_slaves(); 
        assert (num_slaves >= 0);
  
        // Loop over slaves
        for (int s = 0; s < num_slaves; ++s) 
        {
          const int slave_cell_index = it->slave_index(s);
          const bool slave_is_local = all_dofs_are_local (space, slave_cell_index, 
                                                          ownership_begin_, ownership_end_, cell_dof_ids);
          if_is_local = if_is_local && slave_is_local;
        }

        if (if_is_local)
        {
          continue;
        }

        // at least one  of the adjacent cells contains non-local dofs
        // ->mark all adjacent cells as ghost
        cell_status[master_cell_index] = std::max(cell_status[master_cell_index], 1);
        for (int s = 0; s < num_slaves; ++s) 
        {
          const int slave_cell_index = it->slave_index(s);
          cell_status[slave_cell_index] = std::max(cell_status[slave_cell_index], 1);
        }
      }
    }

    // ----------------------------------
    // Step 3 summarize cell status 
    for (int c=0; c!=this->num_cells_; ++c)
    {
      if (cell_status[c] == 0)
      {
        this->local_cells_.push_back(c);
      }
      else
      {
        this->ghost_cells_.push_back(c);
      }
    }

    this->is_initialized_ = true;
  }

  void set_unit_rows (const std::vector<gDofId>& fixed_dofs)
  {
    // -----------------------------------
    // Step 4
    // take care of rows which are set to unit vector e_i
    const int num_fixed = fixed_dofs.size();
    this->local_fixed_dofs_.clear();
    this->local_fixed_dofs_.reserve(num_fixed);

    for (int i=0; i!=num_fixed; ++i)
    {
      const doffem::gDofId fixed_dof = fixed_dofs[i];
      if ( (fixed_dof >= ownership_begin_) && (fixed_dof < ownership_end_) )
      {
        this->local_fixed_dofs_.push_back(fixed_dof - ownership_begin_);
      }
    }
  }


  void get_dof_indices (int N, int cell_index, lDofId* dof_ids, int* is_local) const 
  {
    assert (this->is_initialized_);
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    //assert (this->dof_ids_per_cell_[cell_index].size() == N);
    //assert (this->dof_local_per_cell_[cell_index].size() == N);
    
    const lDofId* src_id = &(this->dof_ids_per_cell_[cell_index * this->N_]);
    const int*  src_loc = &(this->dof_local_per_cell_[cell_index * this->N_]);
    
    std::memcpy(dof_ids, src_id, this->N_ * sizeof(lDofId));
    std::memcpy(is_local, src_loc, this->N_ * sizeof(int));
  }

  inline lDofId get_dof_index (int cell_index, int i) const 
  {
    assert (this->is_initialized_);
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    assert (i>=0);
    assert (i < N_);

    return this->dof_ids_per_cell_[cell_index * this->N_+i];
  } 

  inline int dof_is_local (int cell_index, int i) const 
  {
    assert (this->is_initialized_);
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
     assert (i>=0);
    assert (i < N_);

    return this->dof_local_per_cell_[cell_index * this->N_+i];
  } 

  bool IsInitialized() const 
  { 
    return this->is_initialized_; 
  }

private:

  bool all_dofs_are_local (const VectorSpace<DataType, DIM>& space,
                           int cell_index, 
                           int ownership_begin, 
                           int ownership_end,
                           std::vector<gDofId>& cell_dof_ids)
  {
    // get global dof ids from cell 
    cell_dof_ids.clear();
    space.get_dof_indices(cell_index, cell_dof_ids);

    // check if global dofs on cell are local 
    for (auto dof_id : cell_dof_ids)
    {
      if (!(dof_id >= ownership_begin && dof_id < ownership_end) )
      {
        return false;
      }
    }
    return true; 
  }

  bool is_initialized_ = false;
  
  int tdim_ = -1;
  int num_cells_ = 0;
  int N_ = 0;

  gDofId ownership_begin_ = 0;
  gDofId ownership_end_ = 0;
  
  std::vector< lDofId > dof_ids_per_cell_;
  std::vector< int > dof_local_per_cell_;

  std::vector<int> local_cells_;
  std::vector<int> ghost_cells_;

  std::vector<lDofId> local_fixed_dofs_;
};

}
}

#endif