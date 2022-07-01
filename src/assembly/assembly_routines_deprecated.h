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

/// \author Staffan Ronnas, Simon Gawlok, Philipp Gerstner

#ifndef _ASSEMBLY_ROUTINES_DEPRECATED_H_
#define _ASSEMBLY_ROUTINES_DEPRECATED_H_

#include <vector>
#include "assembly/assembly_routines.h"
#include "assembly/assembly_utils.h"
#include "assembly/global_assembler_deprecated.h"
#include "assembly/generic_assembly_algorithm.h"
#include "assembly/quadrature_selection.h"
#include "common/pointers.h"
#include "common/array_tools.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "mesh/attributes.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "mesh/interface.h"
#include "space/element.h"
#include "space/vector_space.h"

namespace hiflow {

// deprecated
template < class DataType, int DIM >
void add_global_dg(const VectorSpace< DataType, DIM > &space,
                   const int test_cell_index,
                   const int trial_cell_index,
                   const std::vector< int > &row_dofs,
                   const std::vector< int > &col_dofs,
                   const typename GlobalAssembler< DataType, DIM >::LocalMatrix &lm,
                   typename GlobalAssembler< DataType, DIM >::GlobalMatrix &gm);

// deprecated
template < class DataType, int DIM >
void add_global_dg(const VectorSpace< DataType, DIM > &space,
                   const int test_cell_index,
                   const std::vector< int > &dofs,
                   const typename GlobalAssembler< DataType, DIM >::LocalVector &lv,
                   typename GlobalAssembler< DataType, DIM >::GlobalVector &vec);

//////////////// StandardAssembly helper functions ////////////////
// deprecated
template < template < class, class, int > class AlgorithmType, class DataType, int DIM >
class StandardMatrixAssembly
    : public AssemblyAlgorithmBase<AlgorithmType, StandardMatrixAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > 
{
public:

  typedef la::SeqDenseMatrix< DataType > LocalObjectType;
  typedef Quadrature< DataType > QuadratureType;
  typedef AssemblyAlgorithmBase<AlgorithmType, StandardMatrixAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > Base;

  // Name resolution does not manage to get the base members, so
  // we must do it ourselves.
  using Base::dof_;
  using Base::space_;
  using Base::traversal_;

  StandardMatrixAssembly(
      const VectorSpace< DataType, DIM > &space,
      typename GlobalAssembler< DataType, DIM >::GlobalMatrix &matrix)
      : Base(space), matrix_(matrix) {
    sort_elements(space, this->traversal_);
  }

  void add(const Element< DataType, DIM > &element,
           const LocalObjectType &local_mat) {
    
    // skip zero contributions       
    DataType local_mat_abs = local_mat.abs();
    if (local_mat_abs == 0.)
    {
      return;
    }
    
    const size_t num_dofs = this->dof_.size();

    std::vector< int > dofs_sort_permutation;
    std::vector< int > dofs_sorted(num_dofs);

    // get permutation for sorting dofs
    compute_sort_permutation_stable(this->dof_, dofs_sort_permutation);

    // fill sorted dof array
    for (size_t i = 0; i != num_dofs; ++i) {
      dofs_sorted[i] = this->dof_[dofs_sort_permutation[i]];
    }

    std::vector< DataType > dof_factors;
    this->space_.dof().get_dof_factors_on_cell (element.cell_index(), dof_factors);
    assert (dof_factors.size() == num_dofs);
    
    // create row array
    std::vector< int > row_indices;
    std::vector< int > row_permutation;
    row_indices.reserve(num_dofs);
    row_permutation.reserve(num_dofs);
    for (size_t i = 0; i != num_dofs; ++i) 
    {
      const int dof_sort_perm = dofs_sort_permutation[i];
      const int dof_ind = this->dof_[dof_sort_perm];

#ifdef ADD_NONLOCAL_MATRIX_ROWS
      if (this->matrix_.has_ghost_rows())
      {
        row_indices.push_back(dof_ind);
        row_permutation.push_back(dof_sort_perm);
      }
#else
      if (this->space_.dof().is_dof_on_subdom(dof_ind)) 
      {
        row_indices.push_back(dof_ind);
        row_permutation.push_back(dof_sort_perm);
      }
#endif
    }

    // fill reduced and sorted local matrix
    // TODO: make local_mat_sorted_reduced as class member for performance issues
    LocalObjectType local_mat_sorted_reduced;
    if (!row_indices.empty() && num_dofs > 0) 
    {
      local_mat_sorted_reduced.Resize(row_indices.size(), num_dofs);
      for (size_t i = 0, i_e = row_indices.size(); i != i_e; ++i) 
      {
        const int row_ind = row_permutation[i];

        //std::cout << " CG " << row_ind << " <-> " << i << " <-> " << dofs_sort_permutation[i] << std::endl;
        for (size_t j = 0, j_e = num_dofs; j != j_e; ++j) 
        {
          const int col_ind = dofs_sort_permutation[j];
          local_mat_sorted_reduced(i, j) =  dof_factors[row_ind] 
                                          * dof_factors[col_ind] 
                                          * local_mat(row_ind, col_ind);
        }
      }

      // Add local to global matrix
      assert (row_indices.size() > 0);
      assert (dofs_sorted.size() > 0);
      
      this->matrix_.Add(vec2ptr(row_indices), row_indices.size(),
                        vec2ptr(dofs_sorted), dofs_sorted.size(),
                        &local_mat_sorted_reduced(0, 0));
    }
  }

private:
  typename GlobalAssembler< DataType, DIM >::GlobalMatrix &matrix_;
};

// deprecated
template < class DataType, int DIM >
class HpMatrixAssembly
    : public AssemblyAlgorithmBase< InteriorAssemblyAlgorithm,
                                    HpMatrixAssembly< DataType, DIM >, DataType, DIM > 
{
public:
  typedef la::SeqDenseMatrix< DataType > LocalObjectType;
  typedef Quadrature< DataType > QuadratureType;
    typedef std::vector< std::pair< int, double > >::const_iterator
        ConstraintIterator;

  HpMatrixAssembly(const VectorSpace< DataType, DIM > &space,
                   typename GlobalAssembler< DataType, DIM >::GlobalMatrix &matrix)
      : AssemblyAlgorithmBase< hiflow::InteriorAssemblyAlgorithm, HpMatrixAssembly, DataType, DIM >(space),
        matrix_(matrix), interp_(space.dof().dof_interpolation()) 
  {
#ifdef OCTAVE_OUTPUT

    octave_.open("check_assembly.m", std::ios_base::app);
    octave_.precision(16);
    octave_ << "% ==== Global matrix assembly ====\n";
    octave_ << "A = zeros(" << matrix.nrows_global() << ");\n";
#endif
    matrix_.Zeros();
  }

  ~HpMatrixAssembly() {
    // Set rows of constrained dofs to identity to obtain non-singular
    // matrix
    SortedArray< int > constrained_dofs;
    for (typename DofInterpolation<DataType>::const_iterator it = interp_.begin(),
                                          end = interp_.end();
         it != end; ++it) {

      if (this->space_.dof().is_dof_on_subdom(it->first)) {
        constrained_dofs.find_insert(it->first);
      }
    }

    if (!constrained_dofs.empty()) {
      matrix_.diagonalize_rows(&constrained_dofs.front(),
                               constrained_dofs.size(), 1.);
    }
#ifdef OCTAVE_OUTPUT
    // Close octave stream
    octave_ << "\n\n";
    octave_.close();
#endif
  }

  void add(const Element< DataType, DIM > &element, const LocalObjectType &local_mat) 
  {
    // skip zero contributions       
    DataType local_mat_abs = local_mat.abs();
    if (local_mat_abs == 0.)
    {
      return;
    }
    
    // Assemble into global system.  Only add entries to
    // unconstrained rows and columns, and only if the dof
    // corresponding to the row belongs to the local subdomain.
    const int num_dofs = this->dof_.size();
    for (size_t i = 0; i != num_dofs; ++i) {
      typename DofInterpolation<DataType>::const_iterator it_i = this->interp_.find(this->dof_[i]);

      if (it_i != this->interp_.end()) {
        // dof[i] is constrained -> add contributions to dependent rows
        for (ConstraintIterator c_it = it_i->second.begin(),
                                c_end = it_i->second.end();
             c_it != c_end; ++c_it) {
          if (this->space_.dof().is_dof_on_subdom(c_it->first)) {

            for (size_t j = 0; j != num_dofs; ++j) {
              typename DofInterpolation<DataType>::const_iterator it_j =
                  this->interp_.find(this->dof_[j]);

              if (it_j != this->interp_.end()) {
                // dof[j] is constrained -> add attributions to dependent
                // columns
                // TODO: are these not cleared at the end anyway?
                for (ConstraintIterator c2_it = it_j->second.begin(),
                                        c2_end = it_j->second.end();
                     c2_it != c2_end; ++c2_it) {
                  matrix_.Add(c_it->first, c2_it->first,
                              c_it->second * c2_it->second * local_mat(i, j));
                }
              } else {
                // dof[j] unconstrained -> add contribution to dof[j] column
                matrix_.Add(c_it->first, this->dof_[j],
                            c_it->second * local_mat(i, j));
              }
            }
          }
        }
      } else {
        // dof[i] is unconstrained
        if (this->space_.dof().is_dof_on_subdom(this->dof_[i])) {
          for (size_t j = 0; j != num_dofs; ++j) {
            typename DofInterpolation<DataType>::const_iterator it_j =
                this->interp_.find(this->dof_[j]);
            if (it_j != this->interp_.end()) {
              for (ConstraintIterator c_it = it_j->second.begin(),
                                      c_end = it_j->second.end();
                   c_it != c_end; ++c_it) {
                // dof[j] is constrained -> add attributions to dependent
                // columns
                matrix_.Add(this->dof_[i], c_it->first,
                            c_it->second * local_mat(i, j));
              }
            } else {
              // dof[j] unconstrained - assemble normally
              matrix_.Add(this->dof_[i], this->dof_[j], local_mat(i, j));
            }
          }
        }
      }
    }
#ifdef OCTAVE_OUTPUT
    octave_ << "% Element " << element.get_cell_index() << "\n";
    octave_ << "dof = [" << string_from_range(dof_.begin(), dof_.end())
            << "] + 1;\n"
            << "A_local = " << local_mat << ";\n"
            << "A(dof, dof) += A_local;\n";
#endif
  }

private:
  typename GlobalAssembler< DataType, DIM >::GlobalMatrix &matrix_;
  const DofInterpolation<DataType> &interp_;
#ifdef OCTAVE_OUTPUT
  std::ofstream octave_;
#endif
};

// deprecated
template < template < class, class, int > class AlgorithmType, class DataType, int DIM >
class StandardVectorAssembly
    : public AssemblyAlgorithmBase<AlgorithmType, StandardVectorAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > 
{
public:
  typedef std::vector< DataType > LocalObjectType;
  typedef hiflow::Quadrature< DataType > QuadratureType;
  typedef AssemblyAlgorithmBase< AlgorithmType, StandardVectorAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > Base;

  // Name resolution does not manage to get the base members, so
  // we must do it ourselves.
  using Base::dof_;
  using Base::space_;
  using Base::traversal_;

  StandardVectorAssembly(
      const VectorSpace< DataType, DIM > &space,
      typename GlobalAssembler< DataType, DIM >::GlobalVector &vec)
      : Base(space), vector_(vec) {

    sort_elements(space, this->traversal_);
  }

  void add(const Element< DataType, DIM > &element,
           const LocalObjectType &local_vec) {

    assert (!contains_nan(local_vec));
    
    // skip zero contributions       
    DataType local_vec_abs = norm1(local_vec);
    if (local_vec_abs == 0.)
    {
      return;
    }
    
    const size_t num_dofs = this->dof_.size();

    std::vector< int > dofs_sort_permutation;

    // get permutation for sorting dofs
    compute_sort_permutation_stable(this->dof_, dofs_sort_permutation);

    // create row array
    std::vector< int > row_indices;
    row_indices.reserve(num_dofs);

    LocalObjectType local_vec_sorted;
    local_vec_sorted.reserve(num_dofs);

    std::vector< DataType > dof_factors;
    this->space_.dof().get_dof_factors_on_cell (element.cell_index(), dof_factors);
    assert (dof_factors.size() == num_dofs);
    
    for (size_t i = 0; i != num_dofs; ++i) 
    {
      const int dof_sort_perm = dofs_sort_permutation[i];
      const int dof_ind = this->dof_[dof_sort_perm];
      if (this->space_.dof().is_dof_on_subdom(dof_ind)) 
      {
        row_indices.push_back(dof_ind);
        local_vec_sorted.push_back(dof_factors[dof_sort_perm] * local_vec[dof_sort_perm]);
      }
    }
            
    // Add local to global vector
    if (!row_indices.empty()) {
      this->vector_.Add(vec2ptr(row_indices), row_indices.size(),
                        vec2ptr(local_vec_sorted));
    }
  }

private:
  typename GlobalAssembler< DataType, DIM >::GlobalVector &vector_;
};

// deprecated
template < class DataType, int DIM >
class HpVectorAssembly
    : public AssemblyAlgorithmBase< InteriorAssemblyAlgorithm,
                                    HpVectorAssembly< DataType, DIM >, DataType, DIM > 
{
public:
  typedef typename GlobalAssembler< DataType, DIM >::LocalVector LocalObjectType;
  typedef hiflow::Quadrature< DataType > QuadratureType;
  typedef std::vector< std::pair< int, double > >::const_iterator
    ConstraintIterator;

  HpVectorAssembly(const VectorSpace< DataType, DIM > &space,
                   typename GlobalAssembler< DataType, DIM >::GlobalVector &vec)
      : AssemblyAlgorithmBase< hiflow::InteriorAssemblyAlgorithm,
                               HpVectorAssembly, DataType, DIM >(space),
        vector_(vec), interp_(space.dof().dof_interpolation()) {
#ifdef OCTAVE_OUTPUT
    octave_.open("check_assembly.m", std::ios_base::app);
    octave_.precision(16);
    octave_ << "% Global vector assembly\n";
    octave_ << "b = zeros(" << vector_.size_global() << ", 1);\n";
#endif
    vector_.Zeros();
  }

  ~HpVectorAssembly() {
#ifdef OCTAVE_OUTPUT
    octave_ << "\n\n";
    octave_.close();
#endif
  }

  void add(const Element< DataType, DIM > &element, const LocalObjectType &local_vec) 
  {
    DataType local_vec_abs = norm1(local_vec);
    if (local_vec_abs == 0.)
    {
      return;
    }
    std::cout << "------------------" << std::endl;
    const int num_dofs = this->dof_.size();
    for (size_t i = 0; i != num_dofs; ++i) 
    {
      typename DofInterpolation<DataType>::const_iterator it = this->interp_.find(this->dof_[i]);
      if (it != this->interp_.end()) 
      {
        // dof[i] is constrained -> add contributions to dependent dofs
        for (ConstraintIterator c_it = it->second.begin(),
                                c_end = it->second.end();
             c_it != c_end; ++c_it) 
          {
          if (this->space_.dof().is_dof_on_subdom(c_it->first)) 
          {
            vector_.Add(c_it->first, c_it->second * local_vec[i]);
            std::cout << "i = " << i << " gl_i = " << this->dof_[i] 
                      << " gl_ii = " << c_it->first 
                      << " weight = " << c_it->second << " locv = " << local_vec[i] << std::endl;
          }
        }
      } 
      else 
      {
        // dof[i] is unconstrained -> add contribution to this dof
        if (this->space_.dof().is_dof_on_subdom(this->dof_[i])) 
        {
          vector_.Add(this->dof_[i], local_vec[i]);
        }
      }
    }
#ifdef OCTAVE_OUTPUT
    octave_ << "% Element " << element.get_cell_index() << "\n";
    octave_ << "dof = [" << string_from_range(dof_.begin(), dof_.end())
            << "] + 1;\n"
            << "b_local = ["
            << precise_string_from_range(local_vec.begin(), local_vec.end())
            << "]';\n"
            << "b(dof) += b_local;\n";
#endif
  }

private:
  typename GlobalAssembler< DataType, DIM >::GlobalVector &vector_;
  const DofInterpolation<DataType> &interp_;

#ifdef OCTAVE_OUTPUT
  std::ofstream octave_;
#endif
};

// deprecated
template < class DataType, int DIM >
void add_global_dg(const VectorSpace< DataType, DIM > &space,
                   const int test_cell_index,
                   const int trial_cell_index,
                   const std::vector< int > &row_dofs,
                   const std::vector< int > &col_dofs,
                   const typename GlobalAssembler< DataType, DIM >::LocalMatrix &lm,
                   typename GlobalAssembler< DataType, DIM >::GlobalMatrix &gm) 
{
  // skip zero contributions       
  DataType local_mat_abs = lm.abs();
  if (local_mat_abs == 0.)
  {
    return;
  }
    
  const size_t num_dofs_row = row_dofs.size();
  const size_t num_dofs_col = col_dofs.size();

  std::vector< int > row_dofs_sort_permutation;
  std::vector< int > col_dofs_sort_permutation;
  std::vector< int > row_dofs_sorted(num_dofs_row);
  std::vector< int > col_dofs_sorted(num_dofs_col);

  // get permutation for sorting dofs
  compute_sort_permutation_stable(row_dofs, row_dofs_sort_permutation);
  compute_sort_permutation_stable(col_dofs, col_dofs_sort_permutation);

  // fill sorted dof array
  for (size_t i = 0; i != num_dofs_row; ++i) {
    row_dofs_sorted[i] = row_dofs[row_dofs_sort_permutation[i]];
  }
  for (size_t i = 0; i != num_dofs_col; ++i) {
    col_dofs_sorted[i] = col_dofs[col_dofs_sort_permutation[i]];
  }

  std::vector< DataType > dof_factors_trial;
  std::vector< DataType > dof_factors_test;
  
  space.dof().get_dof_factors_on_cell (trial_cell_index, dof_factors_trial);
  space.dof().get_dof_factors_on_cell (test_cell_index, dof_factors_test);
  
  assert (dof_factors_trial.size() == num_dofs_col);
  assert (dof_factors_test.size() == num_dofs_row);
    
  // create row array
  std::vector< int > row_indices;
  std::vector< int > row_permutation;
  row_indices.reserve(num_dofs_row);
  row_permutation.reserve(num_dofs_row);
  for (size_t i = 0; i != num_dofs_row; ++i) {
    if (space.dof().is_dof_on_subdom(row_dofs_sorted[i])) {
      row_indices.push_back(row_dofs_sorted[i]);
      row_permutation.push_back(row_dofs_sort_permutation[i]);
    }
  }

  // fill reduced and sorted local matrix
  typename GlobalAssembler< DataType, DIM >::LocalMatrix local_mat_sorted_reduced;
  if (!row_indices.empty() && col_dofs_sorted.size() > 0) 
  {
    local_mat_sorted_reduced.Resize(row_indices.size(), col_dofs_sorted.size());
    for (size_t i = 0; i != row_indices.size(); ++i) 
    {
      const int test_ind = row_permutation[i];
      for (size_t j = 0; j != col_dofs_sorted.size(); ++j) 
      {
        const int trial_ind = col_dofs_sort_permutation[j];
        local_mat_sorted_reduced(i, j) = dof_factors_trial[trial_ind]
                                       * dof_factors_test[test_ind]
                                       * lm(test_ind, trial_ind);
      }
    }

    // Add local to global matrix
    gm.Add(vec2ptr(row_indices), row_indices.size(), vec2ptr(col_dofs_sorted),
           col_dofs_sorted.size(), &local_mat_sorted_reduced(0, 0));
  }
}

// deprecated
template < class DataType, int DIM >
void add_global_dg(const VectorSpace< DataType, DIM > &space,
                   const int test_cell_index,
                   const std::vector< int > &dofs,
                   const typename GlobalAssembler< DataType, DIM >::LocalVector &lv,
                   typename GlobalAssembler< DataType, DIM >::GlobalVector &vec) 
{

  // skip zero contributions       
  DataType local_vec_abs = norm1(lv);
  if (local_vec_abs == 0.)
  {
    return;
  }
    
  const size_t num_dofs = dofs.size();

  std::vector< int > dofs_sort_permutation;
  std::vector< int > dofs_sorted(num_dofs);

  // get permutation for sorting dofs
  compute_sort_permutation_stable(dofs, dofs_sort_permutation);

  // fill sorted dof array
  for (size_t i = 0; i != num_dofs; ++i) {
    dofs_sorted[i] = dofs[dofs_sort_permutation[i]];
  }

  std::vector< DataType > dof_factors_test;
  space.dof().get_dof_factors_on_cell (test_cell_index, dof_factors_test);
  
  assert (dof_factors_test.size() == num_dofs);


  // create row array
  std::vector< int > row_indices;
  row_indices.reserve(num_dofs);

  typename GlobalAssembler< DataType, DIM >::LocalVector local_vec_sorted;
  local_vec_sorted.reserve(num_dofs);

  for (size_t i = 0; i != num_dofs; ++i) 
  {
    if (space.dof().is_dof_on_subdom(dofs_sorted[i])) 
    {
      const int test_ind = dofs_sort_permutation[i];
      row_indices.push_back(dofs_sorted[i]);
      local_vec_sorted.push_back(dof_factors_test[test_ind] * lv[test_ind]);
    }
  }

  // Add local to global vector
  if (!row_indices.empty()) {
    vec.Add(vec2ptr(row_indices), row_indices.size(),
            vec2ptr(local_vec_sorted));
  }
}

} // namespace hiflow
#endif
