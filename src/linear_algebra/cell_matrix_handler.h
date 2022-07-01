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

#ifndef HIFLOW_LINEAR_ALGEBRA_CELL_MATRIX_HANDLER_H
#define HIFLOW_LINEAR_ALGEBRA_CELL_MATRIX_HANDLER_H

#include "assembly/assembly_types.h"
#include "common/log.h"
#include "common/vector_algebra_descriptor.h"
#include "linear_algebra/linear_operator.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "space/vector_space.h"
#include "linear_algebra/lmp/lvector.h"
#include "dof/dof_fem_types.h"
#include <assert.h>
#include <cstddef>

namespace hiflow {
namespace la {

/// \brief class for handling cell matrices, to be used inside user-defined local assembler 
/// -> needed for matrix-free stencil operator

template < class DataType, int DIM> 
class CellMatrixHandler  
{
public:
  typedef SeqDenseMatrix< DataType > LocalMatrix;
  typedef int LMIndex;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using rmat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  CellMatrixHandler()
  {}

  ~CellMatrixHandler() 
  {
    this->clear();
  }

  void clear()
  {
    this->matrices_.clear();
    this->cell_factors_.clear();
    this->cell_2_lm_.clear();
    this->cross_matrices_.clear();
    this->cell_cross_factors_.clear();
    this->cell_2_cross_lm_.clear();
    this->cell_2_cross_index_.clear();

    this->num_cells_ = 0;
    this->initialized_ = false;
  } 

  void init (int num_cells, int max_num_lm)
  {
    this->clear();
    this->cell_factors_.resize (num_cells);
    this->cell_2_lm_.resize (num_cells);
    this->cell_cross_factors_.resize (num_cells);
    this->cell_2_cross_lm_.resize (num_cells);
    this->cell_2_cross_index_.resize(num_cells);

    this->num_cells_ = num_cells;

    for (int c=0; c!= num_cells; ++c)
    {
      this->cell_factors_[c].reserve(max_num_lm);
      this->cell_2_lm_[c].reserve(max_num_lm);
      this->cell_cross_factors_[c].reserve(max_num_lm);
      this->cell_2_cross_lm_[c].reserve(max_num_lm);
      this->cell_2_cross_index_.reserve(max_num_lm);
    }
    this->initialized_ = true;
  }

  void push_back (int cell_index, LMIndex lm_index, DataType cell_factor)
  {
    assert (initialized_);
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    assert (lm_index >= 0);
    assert (lm_index < this->num_matrices());

    this->cell_2_lm_[cell_index].push_back(lm_index);
    this->cell_factors_[cell_index].push_back(cell_factor);
  }

  void push_back_cross (int cell_index, LMIndex lm_index, DataType cell_factor, int in_cell_index)
  {
    assert (initialized_);
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    assert (lm_index >= 0);
    assert (lm_index < this->num_cross_matrices());

    this->cell_2_cross_lm_[cell_index].push_back(lm_index);
    this->cell_cross_factors_[cell_index].push_back(cell_factor);
    this->cell_2_cross_index_[cell_index].push_back(in_cell_index);
  }

  void print () const 
  {
    //std::cout << string_from_range(this->cell_offsets_.begin(), this->cell_offsets_.end()) << std::endl;
  }

  inline int num_matrices() const 
  {
    return this->matrices_.size();
  }
  inline int num_matrices(int cell_index) const 
  {
    return this->cell_2_lm_[cell_index].size();
  }

  inline int num_cross_matrices() const 
  {
    return this->cross_matrices_.size();
  }
  inline int num_cross_matrices(int cell_index) const 
  {
    return this->cell_2_cross_lm_[cell_index].size();
  }

  inline LMIndex register_matrix (const LocalMatrix& lm) 
  {
    this->matrices_.push_back(lm);
    return this->matrices_.size() - 1;
  }
  inline LMIndex add_to_matrix (const LocalMatrix& lm, LMIndex lm_index) 
  {
    assert (lm_index >= 0);
    assert (lm_index < this->matrices_.size());
    this->matrices_[lm_index].Add(lm);
    return lm_index;
  }

  inline LMIndex register_cross_matrix (const LocalMatrix& lm) 
  {
    this->cross_matrices_.push_back(lm);
    return this->cross_matrices_.size() - 1;
  }

  inline const LocalMatrix& get_matrix(LMIndex lm_index) const 
  {
    assert (lm_index >= 0);
    assert (lm_index < this->num_matrices());
    return this->matrices_[lm_index];
  }
  inline const LocalMatrix& get_cross_matrix(LMIndex lm_index) const 
  {
    assert (lm_index >= 0);
    assert (lm_index < this->num_cross_matrices());
    return this->cross_matrices_[lm_index];
  }

  inline typename std::vector<LMIndex>::const_iterator begin_lmindex (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_2_lm_[cell_index].begin();
  }

  inline typename std::vector<LMIndex>::const_iterator end_lmindex (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_2_lm_[cell_index].end();
  }

  inline typename std::vector<LMIndex>::const_iterator begin_cross_lmindex (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_2_cross_lm_[cell_index].begin();
  }

  inline typename std::vector<LMIndex>::const_iterator end_cross_lmindex (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_2_cross_lm_[cell_index].end();
  }

  inline typename std::vector<DataType>::const_iterator begin_factor (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_factors_[cell_index].begin();
  }

  inline typename std::vector<DataType>::const_iterator end_factor (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_factors_[cell_index].end();
  }

  inline typename std::vector<DataType>::const_iterator begin_cross_factor (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_cross_factors_[cell_index].begin();
  }

  inline typename std::vector<DataType>::const_iterator end_cross_factor (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_cross_factors_[cell_index].end();
  }

  inline typename std::vector<int>::const_iterator begin_cross_index (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_2_cross_index_[cell_index].begin();
  }

  inline typename std::vector<int>::const_iterator end_cross_index (int cell_index) const 
  {
    assert (cell_index >= 0);
    assert (cell_index < this->num_cells_);
    return this->cell_2_cross_index_[cell_index].end();
  }

  int get_iface_orientation(const vec& normal ) const 
  {
    for (int d = 0; d!=DIM; ++d )
    {
      if ( (std::abs( normal[d] - 1.) < 1e-12) || (std::abs( normal[d] + 1.) < 1e-12) ) 
      {
        return d;
      }
    }
    return -1;
  }

  int get_bface_orientation(const vec& normal ) const 
  {
    int ctr = 0;
    for (int d = 0; d!=DIM; ++d )
    {
      if (std::abs( normal[d] - 1) < 1e-12) 
      {
        return ctr;
      }
      ctr++; 
      if (std::abs( normal[d] + 1) < 1e-12) 
      {
        return ctr;
      }
      ctr++;
    }
    return -1; 
  }

  int get_DG_combi_index(InterfaceSide trial_if_side,
                         InterfaceSide test_if_side) const 
  {
    if (test_if_side == InterfaceSide::BOUNDARY)
    {
      return 0;
    }
    else if (trial_if_side == InterfaceSide::MASTER && test_if_side == InterfaceSide::MASTER)
    {
      return 0;
    }
    else if (trial_if_side == InterfaceSide::SLAVE && test_if_side == InterfaceSide::MASTER)
    {
      return 1;
    }
    else if (trial_if_side == InterfaceSide::MASTER && test_if_side == InterfaceSide::SLAVE)
    {
      return 2;
    }
    else if (trial_if_side == InterfaceSide::SLAVE && test_if_side == InterfaceSide::SLAVE)
    {
      return 3;
    }
    assert (false);
    return -1;
  }

  inline bool is_cross_lm (int combi_index) const 
  {
    if (combi_index == 1 || combi_index == 2)
    {
      return true;
    }
    return false;
  }

  inline int out_cell_index (int combi_index, int master_index, int slave_index) const 
  {
    if (combi_index == 0 || combi_index == 1)
    {
      return master_index;
    }
    return slave_index;
  }

  inline int in_cell_index (int combi_index, int master_index, int slave_index) const 
  {
    if (combi_index == 0 || combi_index == 2)
    {
      return master_index;
    }
    return slave_index;
  }

private:
  int num_cells_ = 0;
  bool initialized_ = false;

  // local matrices with in_dofs = out_dofs
  std::vector< LocalMatrix > matrices_;
  std::vector< std::vector<DataType> > cell_factors_;
  std::vector< std::vector<LMIndex> >  cell_2_lm_;  

  // local matrices with in_dofs != out_dofs 
  // e.g. in case of interface integrals
  std::vector< LocalMatrix > cross_matrices_;
  std::vector< std::vector<DataType> > cell_cross_factors_;
  std::vector< std::vector<LMIndex> >  cell_2_cross_lm_;  
  std::vector< std::vector<int> > cell_2_cross_index_;
  
};

}
}

#endif