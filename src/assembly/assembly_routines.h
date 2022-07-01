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

#ifndef _ASSEMBLY_ROUTINES_H_
#define _ASSEMBLY_ROUTINES_H_

#include <vector>
#include "assembly/assembly_types.h"
#include "assembly/assembly_utils.h"
#include "assembly/global_assembler_deprecated.h"
#include "assembly/generic_assembly_algorithm.h"
#include "assembly/quadrature_selection.h"
#include "common/pointers.h"
#include "common/array_tools.h"
#include "linear_algebra/matrix.h"
#include "mesh/attributes.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "mesh/interface.h"
#include "space/element.h"
#include "space/vector_space.h"

// used to get non-local rows
#define ADD_NONLOCAL_MATRIX_ROWS
#define ADD_NONLOCAL_MATRIX_HANGING_ROWS

// this seems to be necessary, at least when using Elements based on facet moments (e.g. BDM, RT)
#define INCLUDE_GHOST_INTERFACES

#define NO_SINGLE_ADD
#define nMAP

const int PRINT_RANK = 0;

namespace hiflow {

// TODO: use this function in MatrixAssembly
template < class DataType, int DIM >
class Local2GlobalInserter 
{
public:

  typedef la::Matrix< DataType > GlobalMatrix;
  typedef la::Vector< DataType > GlobalVector;
  typedef la::SeqDenseMatrix< DataType > LocalMatrix;
  typedef std::vector< DataType > LocalVector;
  typedef VectorSpace< DataType, DIM > VecSpace;
  typedef int DofId;

  Local2GlobalInserter() 
  {
  }

  void add_2_global_matrix (const VectorSpace< DataType, DIM > &space,
                            const int test_cell_index,
                            const int trial_cell_index,
                            const std::vector< DofId > &row_dofs,
                            const std::vector< DofId > &col_dofs,
                            const LocalMatrix &lm,
                            GlobalMatrix &gm) const
  {
    const int DBG_LVL = 5;

    // skip zero contributions       
    DataType local_mat_abs = lm.abs();
    if (local_mat_abs == 0.)
    {
      return;
    }

    const DofInterpolation<DataType> &interp = space.dof().dof_interpolation();

    const size_t num_dofs_row = row_dofs.size();
    const size_t num_dofs_col = col_dofs.size();

    this->row_dofs_sort_permutation.clear();
    this->col_dofs_sort_permutation.clear();
    this->row_dofs_sorted.clear();  
    this->col_dofs_sorted.clear();  

#ifdef NO_SINGLE_ADD
    this->add_row_ind.clear();
    this->add_col_ind.clear();
    this->add_val.clear();
#endif 

    // get permutation for sorting dofs
    compute_sort_permutation_quick(row_dofs, row_dofs_sorted, row_dofs_sort_permutation);
    compute_sort_permutation_quick(col_dofs, col_dofs_sort_permutation);

    // fill sorted dof array
    this->col_dofs_sorted.reserve(num_dofs_col);
    for (size_t i = 0; i != num_dofs_col; ++i) {
      //col_dofs_sorted[i] = col_dofs[col_dofs_sort_permutation[i]];
      col_dofs_sorted.find_insert(col_dofs[col_dofs_sort_permutation[i]]);
    }

    this->dof_factors_trial.clear();
    this->dof_factors_test.clear();

    space.dof().get_dof_factors_on_cell (trial_cell_index, dof_factors_trial);
    space.dof().get_dof_factors_on_cell (test_cell_index, dof_factors_test);

    assert (dof_factors_trial.size() == num_dofs_col);
    assert (dof_factors_test.size() == num_dofs_row);

    // create row array
    this->srow_indices.clear();
    this->row_permutation.clear();
    srow_indices.reserve(num_dofs_row);
    row_permutation.reserve(num_dofs_row);
    for (size_t i = 0; i != num_dofs_row; ++i) 
    {
      const int dof_sort_perm = row_dofs_sort_permutation[i];
      const int dof_ind = row_dofs_sorted[i];

  #ifdef ADD_NONLOCAL_MATRIX_ROWS
      if (gm.has_ghost_rows())
      {
        assert (dof_ind >= 0);
        //srow_indices.data().push_back(dof_ind);
        srow_indices.find_insert(dof_ind);
        row_permutation.push_back(dof_sort_perm);
      }
      else 
      {
        if (space.dof().is_dof_on_subdom(dof_ind)) 
        {
          //srow_indices.data().push_back(dof_ind);
          srow_indices.find_insert(dof_ind);
          row_permutation.push_back(dof_sort_perm);
        }
      }
  #else
      if (space.dof().is_dof_on_subdom(dof_ind)) 
      {
        //srow_indices.data().push_back(dof_ind);
        srow_indices.find_insert(dof_ind);
        row_permutation.push_back(dof_sort_perm);
      }
  #endif
    }

    
    assert (srow_indices.check_sorting());
    assert (col_dofs_sorted.check_sorting());

    // fill reduced and sorted local matrix
    // TODO: check whether interpolation and dof_factors do interfere
    LOG_DEBUG(DBG_LVL, "====================================");
    LOG_DEBUG(DBG_LVL, "====================================");
    LOG_DEBUG(DBG_LVL, "test cell " << test_cell_index << " trial cell " << trial_cell_index);

    if (!srow_indices.empty() && col_dofs_sorted.size() > 0) 
    {
      const size_t num_rows = srow_indices.size();
      const size_t num_cols = col_dofs_sorted.size();

#ifdef NO_SINGLE_ADD
#ifdef MAP
      std::map< std::pair<int, int>, DataType> irreg_vals;
#else 
      this->add_row_ind.reserve(num_rows * num_cols);
      this->add_col_ind.reserve(num_rows * num_cols);
      this->add_val.reserve(num_rows * num_cols);
#endif
#endif
      if (num_rows != local_mat_sorted_reduced.nrows() || num_cols != local_mat_sorted_reduced.ncols())
      {
        local_mat_sorted_reduced.Resize(num_rows, col_dofs_sorted.size());
      }
      else 
      {
        local_mat_sorted_reduced.Zeros();
      }
      for (size_t i = 0; i != num_rows; ++i) 
      {
        const int loc_i = row_permutation[i]; // index for local object
        const int gl_i = srow_indices[i];      // global dof index
        typename DofInterpolation<DataType>::const_iterator it_i = interp.find(gl_i);

        if (it_i != interp.end()) 
        {
          // dof[i] is constrained -> add contributions to dependent rows
          for (auto c_it = it_i->second.begin(), c_end = it_i->second.end();
               c_it != c_end; ++c_it) 
          {
            // check which local dof corresponds to global dof c_it->first
            int ii = -1;
            bool found_ii = srow_indices.find(c_it->first, &ii);

            for (size_t j = 0; j != num_cols; ++j) 
            {
              const int loc_j = col_dofs_sort_permutation[j]; // index for local object
              const int gl_j = col_dofs_sorted[j];            // global dof index
              typename DofInterpolation<DataType>::const_iterator it_j = interp.find(gl_j);
              if (it_j != interp.end()) 
              {
                // dof[j] is constrained -> add attributions to dependent columns
                // TODO: are these not cleared at the end anyway?
                for (auto c2_it = it_j->second.begin(), c2_end = it_j->second.end();
                     c2_it != c2_end; ++c2_it) 
                {
                  // check which local dof corresponds to global dof c2_it->first
                  // TODO: more efficient search
                  int jj = -1;
                  bool found_jj = col_dofs_sorted.find(c2_it->first, &jj);

                  if (found_jj && found_ii)
                  {
                    LOG_DEBUG(DBG_LVL, "add matrix entry cc1 (" << c_it->first << " , " << c2_it->first << ")");
                    // dependent dof c_it->first is on local cell (i.e. contained in vector dofs) 
                    // -> add to list for insertion 
                    // note: no direct insertion for performance reasons
                    local_mat_sorted_reduced(ii, jj) += dof_factors_test[loc_i]
                                                     * dof_factors_trial[loc_j]
                                                     * c_it->second 
                                                     * c2_it->second
                                                     * lm(loc_i, loc_j);  
                  }
                  else 
                  {
                    // dependent dof c_it->first is NOT on local cell (i.e. not contained in vector dofs)
                    // add directly to global vector 
#ifdef ADD_NONLOCAL_MATRIX_HANGING_ROWS
#else
                    if (space.dof().is_dof_on_subdom(c_it->first))
#endif
                    {
                      LOG_DEBUG(DBG_LVL, "add matrix entry cc2 (" << c_it->first << " , " << c2_it->first << ")");
                      
#ifdef NO_SINGLE_ADD
#ifdef MAP                                             
                      DataType val = c_it->second  * c2_it->second 
                                     * dof_factors_test[loc_i] * dof_factors_trial[loc_j]
                                     * lm(loc_i, loc_j);
                      
                      int row = c_it->first;
                      int col =  c2_it->first;
                      auto key = std::make_pair<int, int>(std::move(row), std::move(col));
                      auto it = irreg_vals.find(key);
                      if (it != irreg_vals.end())
                      {
                        it->second += val;
                      }
                      else 
                      {
                        irreg_vals[key] = val;
                      }
#else 
                      this->add_row_ind.push_back(c_it->first);
                      this->add_col_ind.push_back(c2_it->first);
                      this->add_val.push_back(c_it->second  * c2_it->second 
                                              * dof_factors_test[loc_i] * dof_factors_trial[loc_j]
                                              * lm(loc_i, loc_j));
#endif
#else
                      gm.Add(c_it->first, c2_it->first,   c_it->second 
                                                        * c2_it->second 
                                                        * dof_factors_test[loc_i]
                                                        * dof_factors_trial[loc_j]
                                                        * lm(loc_i, loc_j));
#endif
                    }
                  }
                }
              }
              else 
              {
                // dof[j] unconstrained -> add contribution to dof[j] column
                if (found_ii)
                {
                  LOG_DEBUG(DBG_LVL, "add matrix entry cu1 (" << c_it->first << " , " << gl_j << ")"); 
                  local_mat_sorted_reduced(ii, j) += dof_factors_test[loc_i]
                                                   * dof_factors_trial[loc_j]
                                                   * c_it->second 
                                                   * lm(loc_i, loc_j);
                }
                else 
                {
#ifdef ADD_NONLOCAL_MATRIX_HANGING_ROWS
#else
                  if (space.dof().is_dof_on_subdom(c_it->first))
#endif
                  {
                    LOG_DEBUG(DBG_LVL, "add matrix entry cu2 (" << c_it->first << " , " << gl_j << ")");
#ifdef NO_SINGLE_ADD
#ifdef MAP                                             
                      DataType val = c_it->second  
                                            * dof_factors_test[loc_i] * dof_factors_trial[loc_j]
                                            * lm(loc_i, loc_j);
                      
                      int row = c_it->first;
                      int col =  gl_j;
                      auto key = std::make_pair<int, int>(std::move(row), std::move(col));
                      auto it = irreg_vals.find(key);
                      if (it != irreg_vals.end())
                      {
                        it->second += val;
                      }
                      else 
                      {
                        irreg_vals[key] = val;
                      }
#else 
                    this->add_row_ind.push_back(c_it->first);
                    this->add_col_ind.push_back(gl_j);
                    this->add_val.push_back(c_it->second  
                                            * dof_factors_test[loc_i] * dof_factors_trial[loc_j]
                                            * lm(loc_i, loc_j));
#endif
#else
                                             
                    gm.Add(c_it->first, gl_j,  c_it->second 
                                           * dof_factors_test[loc_i]
                                           * dof_factors_trial[loc_j]
                                           * lm(loc_i, loc_j));
#endif
                  }
                }
              }
            }
          }
        }
        else 
        {
          // dof[i] is unconstrained
          for (size_t j = 0; j != num_cols; ++j) 
          {
            const int loc_j = col_dofs_sort_permutation[j]; // index for local object
            const int gl_j = col_dofs_sorted[j];            // global dof index

            typename DofInterpolation<DataType>::const_iterator it_j = interp.find(gl_j);
            if (it_j != interp.end()) 
            {
              for (auto c_it = it_j->second.begin(), c_end = it_j->second.end();
                   c_it != c_end; ++c_it) 
              {
                // check which local dof corresponds to global dof c_it->first
                int jj = -1;
                bool found_jj = col_dofs_sorted.find(c_it->first, &jj);

                if (found_jj)
                {
                  LOG_DEBUG(DBG_LVL, "add matrix entry uc1 (" << gl_i << " , " << c_it->first << ")");
                  // dof[j] is constrained -> add attributions to dependent columns
                  local_mat_sorted_reduced(i, jj) += dof_factors_test[loc_i]
                                                  * dof_factors_trial[loc_j]
                                                  * c_it->second 
                                                  * lm(loc_i, loc_j);     
                }
                else 
                {
                  LOG_DEBUG(DBG_LVL, "add matrix entry uc2 (" << gl_i << " , " << c_it->first << ")");
#ifdef NO_SINGLE_ADD
#ifdef MAP                                             
                      DataType val = c_it->second  
                                          * dof_factors_test[loc_i] * dof_factors_trial[loc_j]
                                          * lm(loc_i, loc_j);
                                          
                      int row = gl_i;
                      int col =  c_it->first;
                      auto key = std::make_pair<int, int>(std::move(row), std::move(col));

                      auto it = irreg_vals.find(key);
                      if (it != irreg_vals.end())
                      {
                        it->second += val;
                      }
                      else 
                      {
                        irreg_vals[key] = val;
                      }
#else 
                  this->add_row_ind.push_back(gl_i);
                  this->add_col_ind.push_back(c_it->first);
                  this->add_val.push_back(c_it->second  
                                          * dof_factors_test[loc_i] * dof_factors_trial[loc_j]
                                          * lm(loc_i, loc_j));
#endif
#else                        
                  gm.Add(gl_i, c_it->first,  c_it->second 
                                           * dof_factors_test[loc_i]
                                           * dof_factors_trial[loc_j]
                                           * lm(loc_i, loc_j));
#endif
                }
              }
            }
            else 
            { 
              LOG_DEBUG(DBG_LVL, "add matrix entry uu (" << gl_i << " , " << gl_j << ")");
              // dof[j] unconstrained - assemble normally
              local_mat_sorted_reduced(i, j) += dof_factors_test[loc_i]
                                              * dof_factors_trial[loc_j]
                                              * lm(loc_i, loc_j);
            }
          }
        }
      }
      // Add local to global matrix
      gm.Add(vec2ptr(srow_indices.data()), num_rows, vec2ptr(col_dofs_sorted.data()),
             num_cols, &local_mat_sorted_reduced(0, 0));

      // add dependent entries
#ifdef NO_SINGLE_ADD
#ifdef MAP
      for (auto it = irreg_vals.begin(), e_it = irreg_vals.end(); it != e_it; ++it)
      {
        gm.Add(it->first.first, it->first.second, it->second);
      }
#else
      gm.Add(this->add_row_ind, this->add_col_ind, this->add_val);
#endif
#endif
    }
  }


  void add_2_global_vector(const VectorSpace< DataType, DIM > &space,
                    const int test_cell_index,
                    const std::vector< int > &dofs,
                    const LocalVector &lv,
                    GlobalVector &vec) const
  {
    assert (!contains_nan(lv));

    // skip zero contributions   
    /*    
    DataType local_vec_abs = norm1(lv);
    if (local_vec_abs == 0.)
    {
      return;
    }
    */

    //std::cout << "------------------" << std::endl;
    const DofInterpolation<DataType> &interp = space.dof().dof_interpolation();

    const size_t num_dofs = dofs.size();

    // get permutation for sorting dofs
    this->dofs_sort_permutation.clear();
    this->dofs_sorted.clear();
    this->dofs_sorted.reserve(num_dofs);

    compute_sort_permutation_quick(dofs, dofs_sort_permutation);

    
    for (size_t i = 0; i != num_dofs; ++i) {
      dofs_sorted.find_insert(dofs[dofs_sort_permutation[i]]);
    }

    this->dof_factors_test.clear();
    space.dof().get_dof_factors_on_cell (test_cell_index, dof_factors_test);

    assert (dof_factors_test.size() == num_dofs);

    // create row array
    set_to_value(num_dofs, -1, this->row_indices);
    set_to_value(num_dofs, 0, local_vec_sorted);

    for (size_t i = 0; i != num_dofs; ++i) 
    {
      const int loc_i = dofs_sort_permutation[i]; // index for local object
      const int gl_i = dofs_sorted[i];            // global dof index

      typename DofInterpolation<DataType>::const_iterator it = interp.find(gl_i);
      if (it != interp.end()) 
      {
        // dof[i] is constrained -> add contributions to dependent dofs
        for (auto c_it = it->second.begin(), c_end = it->second.end();
             c_it != c_end; ++c_it) 
        {
          if (space.dof().is_dof_on_subdom(c_it->first)) 
          {
            int ii = -1;
            bool found_ii = dofs_sorted.find(c_it->first, &ii);
            
            if (found_ii)
            {
              // dependent dof c_it->first is on local cell (i.e. contained in vector dofs) 
              // -> add to list for insertion 
              // note: no direct insertion for performance reasons
              if (row_indices[ii] >= 0)
              {
                assert (row_indices[ii] == c_it->first);
              }
              row_indices[ii] = c_it->first;
              local_vec_sorted[ii] += dof_factors_test[loc_i] * c_it->second * lv[loc_i];
              LOG_DEBUG(3, "i = " << i << " gl_i = " << gl_i << " loc_i = " << loc_i 
                            << " ii = " << ii << "  gl_ii = " << row_indices[ii] 
                            << " : weight = " << c_it->second << " dfac = " << dof_factors_test[loc_i] 
                            << " loc_v = " << lv[loc_i]);
            }
            else
            {
              // dependent dof c_it->first is NOT on local cell (i.e. not contained in vector dofs)
              // add directly to global vector 
              vec.Add(c_it->first, dof_factors_test[loc_i] * c_it->second * lv[loc_i]);
              LOG_DEBUG(3, "i = " << i << " gl_i = " << gl_i << " loc_i = " << loc_i 
                            << " ii = -1 gl_ii = " << c_it->first 
                            << " : weight = " << c_it->second << " dfac = " << dof_factors_test[loc_i] 
                            << " loc_v = " << lv[loc_i]);
            }
          }
        }
      }
      else 
      {
        if (space.dof().is_dof_on_subdom(gl_i)) 
        {   
          row_indices[i] = gl_i;
          local_vec_sorted[i] += dof_factors_test[loc_i] * lv[loc_i];
        }
      }
    }

    row_insert_dofs.clear();
    row_insert_values.clear();
    row_insert_dofs.reserve(num_dofs);
    row_insert_values.reserve(num_dofs);

    for (int i = 0; i!=num_dofs; ++i)
    {
      if (row_indices[i] >= 0)
      {
        row_insert_dofs.push_back(row_indices[i]);
        row_insert_values.push_back(local_vec_sorted[i]);
      }
    }

    // Add local to global vector
    if (!row_indices.empty()) 
    {
      vec.Add(vec2ptr(row_insert_dofs), row_insert_dofs.size(), vec2ptr(row_insert_values));
    }
  }

  // NOTE: only works on meshes without hanging nodes and Lagrange elements
  inline void add_2_global_vector_fast(const VectorSpace< DataType, DIM > &space,
                                       const int test_cell_index,
                                       const std::vector< int > &dofs,
                                       const LocalVector &lv,
                                       GlobalVector &vec) const
  {
    assert (!contains_nan(lv));

    // TODO: hanging node assert and Lagrange assert
    vec.Add(vec2ptr(dofs), dofs.size(), vec2ptr(lv));
  }

private:

  mutable std::vector< int > row_dofs_sort_permutation;
  mutable std::vector< int > col_dofs_sort_permutation;
  mutable std::vector< int > dofs_sort_permutation;
  mutable std::vector< int > row_permutation;
  mutable std::vector< int > add_row_ind;
  mutable std::vector< int > add_col_ind;
  mutable std::vector< DataType > add_val;
  mutable std::vector< DofId > row_dofs_sorted;
  mutable SortedArray< DofId > col_dofs_sorted;
  mutable SortedArray< DofId > dofs_sorted;
  mutable SortedArray< DofId > srow_indices;
  mutable std::vector< DofId > row_indices;
  mutable std::vector< DofId >  row_insert_dofs;
  mutable std::vector< DataType > row_insert_values;
  mutable std::vector< DataType > dof_factors_trial;
  mutable std::vector< DataType > dof_factors_test;
  mutable LocalMatrix local_mat_sorted_reduced;
  mutable LocalVector local_vec_sorted;
};


//////////////// StandardAssembly helper functions ////////////////
template < template < class, class, int > class AlgorithmType, class DataType, int DIM >
class CellMatrixAssembly
    : public AssemblyAlgorithmBase<AlgorithmType, CellMatrixAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > 
{
public:

  typedef la::SeqDenseMatrix< DataType > LocalObjectType;
  typedef Quadrature< DataType > QuadratureType;
  typedef AssemblyAlgorithmBase<AlgorithmType, CellMatrixAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > Base;

  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalMatrix GlobalMatrix;
  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalVector GlobalVector;

  // Name resolution does not manage to get the base members, so
  // we must do it ourselves.
  using Base::dof_;
  using Base::space_;
  using Base::traversal_;

  CellMatrixAssembly( const VectorSpace< DataType, DIM > &space)
  : Base(space), matrix_(nullptr) 
  {
    sort_elements(space, this->traversal_);
  }

  CellMatrixAssembly( const VectorSpace< DataType, DIM > &space,
                      const std::vector<int>& traversal)
  : Base(space, traversal), matrix_(nullptr) 
  {
  }

  CellMatrixAssembly( const VectorSpace< DataType, DIM > &space,
                      GlobalMatrix &matrix)
  : Base(space), matrix_(&matrix) 
  {
    sort_elements(space, this->traversal_);
  }

  CellMatrixAssembly( const VectorSpace< DataType, DIM > &space,
                      const std::vector<int>& traversal,
                      GlobalMatrix &matrix)
  : Base(space, traversal), matrix_(&matrix) 
  {
  }

  void add(const Element< DataType, DIM > &element,
           const LocalObjectType &local_mat) 
  {
    if (this->matrix_ != nullptr)
    {
      inserter_.add_2_global_matrix(this->space_, 
                                    element.cell_index(), 
                                    element.cell_index(), 
                                    this->dof_, 
                                    this->dof_, 
                                    local_mat, 
                                    *this->matrix_);
    }
  }

private:
  GlobalMatrix * matrix_;
  Local2GlobalInserter<DataType, DIM> inserter_;
};

template < class DataType, int DIM >
class InterfaceMatrixAssembly
{
public:

  // TODO: this could be optimized by splitting into separate cell/facet
  // selection functions.
  typedef std::function < void ( const Element< DataType, DIM > &, const Element< DataType, DIM > &, 
                                 int, int,
                                 Quadrature< DataType > &, Quadrature< DataType > &) >
                                 IFQuadratureSelectionFun;

  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalMatrix GlobalMatrix;
  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalVector GlobalVector;
  typedef typename Local2GlobalInserter<DataType, DIM>::LocalMatrix LocalMatrix;
  typedef typename Local2GlobalInserter<DataType, DIM>::LocalVector LocalVector;

  InterfaceMatrixAssembly()
  {}

  template< class LocalAssembler>
  void assemble( const VectorSpace< DataType, DIM > &space,
                 const mesh::InterfaceList& if_list,
                 IFQuadratureSelectionFun if_q_select,
                 LocalAssembler& local_asm, 
                 GlobalMatrix* matrix
                 ) const 
  {
    const bool insert_into_matrix = (matrix != nullptr);

    mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive

    LocalMatrix L_MM, L_MS, L_SM, L_SS;

    std::vector< doffem::gDofId > master_dofs, slave_dofs;

    Quadrature< DataType > master_quadrature, slave_quadrature;

    // Loop over interfaces

    for (mesh::InterfaceList::const_iterator it = if_list.begin(),
        end_it = if_list.end();
        it != end_it; ++it) 
    {
        int remote_index_master = -10;
        mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                                it->master_index(), &remote_index_master);

        // Master dofs
        const auto master_cell_index = it->master_index();
        Element< DataType, DIM > master_elem(space, master_cell_index);
        
        space.get_dof_indices(master_cell_index, master_dofs);
        const auto master_facet_number = it->master_facet_number();

        L_MM.Resize(master_dofs.size(), master_dofs.size());
        L_MM.Zeros();

        // Initialize master quadrature
        if_q_select(master_elem, master_elem, master_facet_number,
                    master_facet_number, master_quadrature, master_quadrature);

        const int num_slaves = it->num_slaves(); 
        assert (num_slaves >= 0);
    
        // subdom              = localdom + ghostdom
        // num_slaves = 0      -> bdy of subdom
        // num_slaves > 0      -> interior of subdom
        // remote_index == -1  -> cell \in localdom
        // remote_index >= 0   -> cell \in ghostdom
        // num_slaves = 0 / remote_index_master == -1 -> bdy of localdom = physical bdy
        // num_slaves = 0 / remote_index_master >= 0  -> bdy of ghost    = physical interior
    
        // treat boundary facet
        if (remote_index_master == -1) 
        {
          if (num_slaves == 0) 
          {
              local_asm(master_elem, master_elem, 
                        master_quadrature, master_quadrature, 
                        master_facet_number, master_facet_number,
                        InterfaceSide::MASTER, InterfaceSide::BOUNDARY, 
                        -1, num_slaves,
                        L_MM);
              assert (!L_MM.contains_nan());
          }

          if (insert_into_matrix)
          {
            inserter_.add_2_global_matrix(space, master_cell_index, master_cell_index, master_dofs, master_dofs, L_MM, *matrix);
          }
        }
        
        // treat interior facets
        // Loop over slaves
        for (int s = 0; s < num_slaves; ++s) 
        {
          const int slave_cell_index = it->slave_index(s);
          const int slave_facet_number = it->slave_facet_number(s);

          int remote_index_slave = -10;
          mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                                      slave_cell_index, &remote_index_slave);
                                      
          Element< DataType, DIM > slave_elem(space, slave_cell_index);
          space.get_dof_indices(slave_cell_index, slave_dofs);
                  

          // Initialize slave quadrature. NB: only called once per slave.
          // default quad selection: all quadrature points lie on that part of the interface,
          // which has non-empty intersection with slave cell
          if_q_select(master_elem, slave_elem, 
                      master_facet_number, slave_facet_number, 
                      master_quadrature, slave_quadrature);

  #ifdef INCLUDE_GHOST_INTERFACES
          constexpr bool add_to_master = true;
          constexpr bool add_to_slave = true;
  #else
          const bool add_to_master = (remote_index_master == -1); 
          const bool add_to_slave = (remote_index_slave == -1); 
  #endif

          if (add_to_master) 
          {

            // master / slave
            L_MS.Resize(master_dofs.size(), slave_dofs.size());
            L_MS.Zeros();
            local_asm(master_elem, slave_elem, 
                      master_quadrature, slave_quadrature,
                      master_facet_number, slave_facet_number, 
                      InterfaceSide::SLAVE, InterfaceSide::MASTER, 
                      s, num_slaves,
                      L_MS);
            assert (!L_MS.contains_nan());

            if (insert_into_matrix)
            {
              inserter_.add_2_global_matrix(space, master_cell_index, slave_cell_index, master_dofs, slave_dofs, L_MS, *matrix);
            }

            // master / master
            // Note: in case of hanging nodes, there holds num_slaves > 1, 
            // i.e. the combination master / master is called more then once.
            // Use the slave index s passed to your local assembler implementation 
            // to decide what to do in this case.
            // Why do we do that? Because for evaluating facet integrals, it might be necessary to
            // have both master and slave element at hand, even if the combination master / master is considered. 
            // Example: need to compute jump and average of discontinuous function, which is neither ansatz, nor 
            // test function
            //
            // Note: actually, calling master / master more than once should be fine,
            // since the quadrature is restricted to the slave portion of the interface
            
            L_MM.Resize(master_dofs.size(), master_dofs.size());
            L_MM.Zeros();
            local_asm(master_elem, slave_elem, 
                      master_quadrature, slave_quadrature,
                      master_facet_number, slave_facet_number, 
                      InterfaceSide::MASTER, InterfaceSide::MASTER, 
                      s, num_slaves,
                      L_MM);
            assert (!L_MM.contains_nan());
            if (insert_into_matrix)
            {
              inserter_.add_2_global_matrix(space, master_cell_index, master_cell_index, master_dofs, master_dofs, L_MM, *matrix);
            }
          }
          if (add_to_slave) 
          {
            // slave / master

            L_SM.Resize(slave_dofs.size(), master_dofs.size());
            L_SM.Zeros();
            local_asm(master_elem, slave_elem,  
                      master_quadrature, slave_quadrature,
                      master_facet_number, slave_facet_number, 
                      InterfaceSide::MASTER, InterfaceSide::SLAVE, 
                      s, num_slaves,
                      L_SM);
            assert (!L_SM.contains_nan());
            if (insert_into_matrix)
            {
              inserter_.add_2_global_matrix(space, slave_cell_index, master_cell_index, slave_dofs, master_dofs, L_SM, *matrix);
            }

            // slave / slave
            L_SS.Resize(slave_dofs.size(), slave_dofs.size());
            L_SS.Zeros();
            local_asm(master_elem, slave_elem, 
                      master_quadrature, slave_quadrature,
                      master_facet_number, slave_facet_number, 
                      InterfaceSide::SLAVE, InterfaceSide::SLAVE, 
                      s, num_slaves,
                      L_SS);
            assert (!L_SS.contains_nan());
            if (insert_into_matrix)
            {
              inserter_.add_2_global_matrix(space, slave_cell_index, slave_cell_index, slave_dofs, slave_dofs, L_SS, *matrix);
            }
          }
        }
      }
    }
  
  Local2GlobalInserter<DataType, DIM> inserter_;

};

template < template < class, class, int > class AlgorithmType, class DataType, int DIM >
class CellVectorAssembly
    : public AssemblyAlgorithmBase<AlgorithmType, CellVectorAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > 
{
public:
  typedef std::vector< DataType > LocalObjectType;
  typedef hiflow::Quadrature< DataType > QuadratureType;
  typedef AssemblyAlgorithmBase< AlgorithmType, CellVectorAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > Base;

  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalMatrix GlobalMatrix;
  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalVector GlobalVector;

  // Name resolution does not manage to get the base members, so
  // we must do it ourselves.
  using Base::dof_;
  using Base::space_;
  using Base::traversal_;

  CellVectorAssembly(const VectorSpace< DataType, DIM > &space,
                     const std::vector<int>& traversal,
                     GlobalVector &vec)
  : Base(space, traversal), vector_(&vec) 
  {
  }

  CellVectorAssembly(const VectorSpace< DataType, DIM > &space,
                     GlobalVector &vec)
  : Base(space), vector_(&vec) 
  {
    sort_elements(space, this->traversal_);
  }

  CellVectorAssembly(const VectorSpace< DataType, DIM > &space,
                     const std::vector<int>& traversal)
  : Base(space, traversal), vector_(nullptr) 
  {
  }

  CellVectorAssembly(const VectorSpace< DataType, DIM > &space)
  : Base(space), vector_(nullptr) 
  {
    sort_elements(space, this->traversal_);
  }

  inline void set_vector(GlobalVector &vec)
  {
    this->vector_ = &vec;
  }

  void set_you_know_what_you_are_doing_flags(bool use_insertion_for_lagrange_without_hanging_nodes = false)
  {
    use_fast_insertion_ = use_insertion_for_lagrange_without_hanging_nodes;
  }

  void add(const Element< DataType, DIM > &element,
           const LocalObjectType &local_vec) 
  {
    assert (this->vector_ != nullptr);
    if (!use_fast_insertion_)
    {
      inserter_.add_2_global_vector(this->space_, 
                                    element.cell_index(),  
                                    this->dof_,  
                                    local_vec, *this->vector_);
    }
    else 
    {
      inserter_.add_2_global_vector_fast(this->space_, 
                                         element.cell_index(),  
                                         this->dof_,  
                                         local_vec, *this->vector_);
    }
  }

private:
  GlobalVector *vector_;
  Local2GlobalInserter<DataType, DIM> inserter_;
  bool use_fast_insertion_ = false;
};

template < class DataType, int DIM >
class InterfaceVectorAssembly
{
public:

  // TODO: this could be optimized by splitting into separate cell/facet
  // selection functions.
  typedef std::function < void ( const Element< DataType, DIM > &, const Element< DataType, DIM > &, 
                                 int, int,
                                 Quadrature< DataType > &, Quadrature< DataType > &) >
                                 IFQuadratureSelectionFun;

  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalMatrix GlobalMatrix;
  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalVector GlobalVector;
  typedef typename Local2GlobalInserter<DataType, DIM>::LocalMatrix LocalMatrix;
  typedef typename Local2GlobalInserter<DataType, DIM>::LocalVector LocalVector;

  InterfaceVectorAssembly()
  {}

  template< class LocalAssembler>
  void assemble( const VectorSpace< DataType, DIM > &space,
                 const mesh::InterfaceList& if_list,
                 IFQuadratureSelectionFun if_q_select,
                 LocalAssembler& local_asm, 
                 GlobalVector &vec
                 ) const 
  {
    // Create interface list from mesh
    mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  
    LocalVector L_M, L_S;
  
    std::vector< doffem::gDofId > master_dofs, slave_dofs;
  
    Quadrature< DataType > master_quadrature, slave_quadrature;
  
    // Loop over interfaces
  
    for (mesh::InterfaceList::const_iterator it = if_list.begin(),
         end_it = if_list.end();
         it != end_it; ++it) 
    {
      const int master_cell_index = it->master_index();
      const int master_facet_number = it->master_facet_number();
      int remote_index_master = -10;
      mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                                master_cell_index, &remote_index_master);
  
      // Master dofs
      Element< DataType, DIM > master_elem(space, master_cell_index);
      space.get_dof_indices(master_cell_index, master_dofs);
      
      set_to_value(master_dofs.size(), 0., L_M);
  
      // Initialize master quadrature
      if_q_select(master_elem, master_elem, 
                   master_facet_number, master_facet_number, 
                   master_quadrature, master_quadrature);
  
      const int num_slaves = it->num_slaves();
      if (remote_index_master == -1) 
      {
        if (num_slaves == 0) 
        {
          // boundary facet
          local_asm(master_elem, master_elem, 
                    master_quadrature, master_quadrature, 
                    master_facet_number, master_facet_number,
                    InterfaceSide::BOUNDARY, 
                    -1, num_slaves,
                    L_M);
          assert (!contains_nan(L_M));
        }
        inserter_.add_2_global_vector(space, master_cell_index, master_dofs, L_M, vec);
      }
      // Loop over slaves
      for (int s = 0; s < num_slaves; ++s) 
      {
        const int slave_facet_number = it->slave_facet_number(s);
        const int slave_cell_index = it->slave_index(s);
        
        int remote_index_slave = -10;
        mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                                  slave_cell_index, &remote_index_slave);
                                  
        Element< DataType, DIM > slave_elem(space, slave_cell_index);
        space.get_dof_indices(slave_cell_index, slave_dofs);
  
        
        // Initialize slave quadrature. NB: only called once per slave.
        // default quad selection: all quadrature points lie on that part of the interface,
        // which has non-empty intersection with slave cell
        if_q_select(master_elem, slave_elem, 
                     master_facet_number, slave_facet_number, 
                     master_quadrature, slave_quadrature);
  
  #ifdef INCLUDE_GHOST_INTERFACES
        bool add_to_master = true;
        bool add_to_slave = true;
  #else
        bool add_to_master = (remote_index_master == -1); 
        bool add_to_slave = (remote_index_slave == -1); 
  #endif
  
        if (add_to_master) 
        {
          // master
          set_to_value(master_dofs.size(), 0., L_M);
          local_asm(master_elem, slave_elem, 
                    master_quadrature, slave_quadrature, 
                    master_facet_number, slave_facet_number,
                    InterfaceSide::MASTER, 
                    s, num_slaves,
                    L_M);
          assert (!contains_nan(L_M));
          inserter_.add_2_global_vector(space, master_cell_index, master_dofs, L_M, vec);
        }
        if (add_to_slave) 
        {
          // slave
          set_to_value(slave_dofs.size(), 0., L_S);
          local_asm(master_elem, slave_elem, 
                    master_quadrature, slave_quadrature,
                    master_facet_number, slave_facet_number, 
                    InterfaceSide::SLAVE, 
                    s, num_slaves,
                    L_S);
          assert (!contains_nan(L_S));
          inserter_.add_2_global_vector(space, slave_cell_index, slave_dofs, L_S, vec);
        }
      }
    }
  }

private:
  Local2GlobalInserter<DataType, DIM> inserter_;

};

template < template < class, class, int > class AlgorithmType, class DataType, int DIM >
class StandardScalarAssembly
    : public AssemblyAlgorithmBase<AlgorithmType, StandardScalarAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > 
{
public:
  typedef DataType LocalObjectType;
  typedef hiflow::Quadrature< DataType > QuadratureType;
  typedef AssemblyAlgorithmBase< AlgorithmType, StandardScalarAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > Base;

  // Name resolution does not manage to get the base members, so
  // we must do it ourselves.
  using Base::curr_;
  using Base::elem_;
  using Base::has_next;
  using Base::space_;
  using Base::traversal_;

  StandardScalarAssembly(const VectorSpace< DataType, DIM > &space,
                         std::vector< DataType > &values)
      : Base(space), values_(values) {
    const size_t num_elements = this->traversal_.size();
    this->values_.resize(num_elements, 0.);

    this->remove_non_local_elements();
    sort_elements(space, this->traversal_);
  }

  /// The has_next() and next() functions are overloaded in
  /// order to skip elements that do not belong to the local
  /// subdomain. This is done by setting the corresponding
  /// entries in the traversal_ array to -1, and later skipping
  /// those items.

  const Element< DataType, DIM > &next() {
    assert(this->has_next());

    this->elem_ = Element< DataType, DIM >(this->space_, this->traversal_[this->curr_]);

    ++(this->curr_);

    return this->elem_;
  }

  void add(const Element< DataType, DIM > &element,
           const LocalObjectType &local_val) {
    this->values_[element.cell_index()] += local_val;
  }

private:
  /// Remove non_local elements

  void remove_non_local_elements() {

    const mesh::Mesh &mesh = this->space_.mesh();

    if (!mesh.has_attribute("_remote_index_", mesh.tdim())) {
      // If the "_remote_index_" attribute does not exist, we
      // assume that there are no ghost cells.
      return;
    }

    int remote_index;
    int index;
    std::vector< int > traversal_tmp;
    traversal_tmp.reserve(mesh.num_entities(mesh.tdim()));

    for (mesh::EntityIterator it_cell = mesh.begin(mesh.tdim()),
                              e_it_cell = mesh.end(mesh.tdim());
         it_cell != e_it_cell; ++it_cell) {
      // test if cell on subdomain
      it_cell->get("_remote_index_", &remote_index);
      if (remote_index == -1) {
        index = it_cell->index();
        traversal_tmp.push_back(index);
      }
    }
    this->traversal_ = traversal_tmp;
  }

  std::vector< DataType > &values_;
};

template < class DataType, int DIM >
class InterfaceScalarAssembly
{
public:

  // TODO: this could be optimized by splitting into separate cell/facet
  // selection functions.
  typedef std::function < void ( const Element< DataType, DIM > &, const Element< DataType, DIM > &, 
                                 int, int,
                                 Quadrature< DataType > &, Quadrature< DataType > &) >
                                 IFQuadratureSelectionFun;

  InterfaceScalarAssembly()
  {}

  template< class LocalAssembler>
  void assemble( const VectorSpace< DataType, DIM > &space,
                 const mesh::InterfaceList& if_list,
                 IFQuadratureSelectionFun if_q_select,
                 LocalAssembler& local_asm, 
                 std::vector< DataType > &values
                 ) const
  {
    // Create interface list from mesh
    mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
    
    DataType L_M, L_S;
  
    Quadrature< DataType > master_quadrature, slave_quadrature;
  
    // Loop over interfaces
    size_t i = 0;
    for (mesh::InterfaceList::const_iterator it = if_list.begin(),
         end_it = if_list.end();
         it != end_it; ++it) 
    {
      int remote_index_master = -10;
      mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                                it->master_index(), &remote_index_master);
  
      // Master dofs
      Element< DataType, DIM > master_elem(space, it->master_index());
  
      const int master_facet_number = it->master_facet_number();
  
      L_M = 0.;
  
      // Initialize master quadrature
      if_q_select(master_elem, master_elem, 
                   master_facet_number, master_facet_number, 
                   master_quadrature, master_quadrature);
  
      const int num_slaves = it->num_slaves();
      if (remote_index_master == -1) 
      {
        if (num_slaves == 0) 
        {
          // boundary facet
          local_asm(master_elem, master_elem, 
                    master_quadrature, master_quadrature, 
                    master_facet_number, master_facet_number,
                    InterfaceSide::BOUNDARY, 
                    -1, 0,
                    L_M);
        }
        values[i] += L_M;
      }
      // Loop over slaves
      for (int s = 0; s < num_slaves; ++s) 
      {
        int remote_index_slave = -10;
        mesh->get_attribute_value("_remote_index_", mesh->tdim(), it->slave_index(s), &remote_index_slave);
        Element< DataType, DIM > slave_elem(space, it->slave_index(s));
        const int slave_facet_number = it->slave_facet_number(s);
  
        // Initialize slave quadrature. NB: only called once per slave.
        // default quad selection: all quadrature points lie on that part of the interface,
        // which has non-empty intersection with slave cell
        if_q_select(master_elem, slave_elem, 
                     master_facet_number, slave_facet_number, 
                     master_quadrature, slave_quadrature);
  
        if (remote_index_master == -1) 
        {
          // master / slave
          L_S = 0.;
          local_asm(master_elem, slave_elem, 
                    master_quadrature, slave_quadrature,
                    master_facet_number, slave_facet_number, 
                    InterfaceSide::SLAVE, 
                    s, num_slaves,
                    L_S);
          values[i] += L_S;
        }
      }
      ++i;
    }
  }
};

template < class DataType, int DIM >
class InterfaceCellScalarAssembly
{
public:

  // TODO: this could be optimized by splitting into separate cell/facet
  // selection functions.
  typedef std::function < void ( const Element< DataType, DIM > &, const Element< DataType, DIM > &, 
                                 int, int,
                                 Quadrature< DataType > &, Quadrature< DataType > &) >
                                 IFQuadratureSelectionFun;

  InterfaceCellScalarAssembly()
  {}

  template< class LocalAssembler>
  void assemble( const VectorSpace< DataType, DIM > &space,
                 const mesh::InterfaceList& if_list,
                 IFQuadratureSelectionFun if_q_select,
                 LocalAssembler& local_asm, 
                 std::vector< DataType > &values
                 ) const
  {
    mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive

    int rank = space.dof().my_subdom();
  
    // Loop over interfaces
    for (mesh::InterfaceList::const_iterator it = if_list.begin(),
         end_it = if_list.end();
         it != end_it; ++it) 
    {
      int remote_index_master = -10;
      mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                                it->master_index(), &remote_index_master);
  
      // Master dofs
      Element< DataType, DIM > master_elem(space, it->master_index());
  
      const int master_facet_number = it->master_facet_number();
  
      DataType L_M, L_S;
  
      Quadrature< DataType > master_master_quadrature;
  
      L_M = 0.;
  
      // Initialize master quadrature
      if_q_select(master_elem, master_elem, master_facet_number,
                         master_facet_number, master_master_quadrature,
                         master_master_quadrature);
  
      const int num_slaves = it->num_slaves();
  
      if (remote_index_master == -1) 
      {
        if (num_slaves == 0) 
        {
          // boundary facet
          local_asm(master_elem, master_elem, 
                    master_master_quadrature, master_master_quadrature, 
                    master_facet_number, master_facet_number, 
                    InterfaceSide::BOUNDARY, 
                    -1, 0,
                    L_M);
        }
  
        LOG_DEBUG(3, "[" << rank << "] Master index: " << it->master_index()
                         << " with remote index " << remote_index_master
                         << ", add to master_cell L_MM=" << L_M);
  
        values[it->master_index()] += L_M;
      }
  
      // Loop over slaves
      for (int s = 0; s < num_slaves; ++s) 
      {
        int remote_index_slave = -10;
        mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                                  it->slave_index(s), &remote_index_slave);
        Element< DataType, DIM > slave_elem(space, it->slave_index(s));
        const int slave_facet_number = it->slave_facet_number(s);
  
        Quadrature< DataType > master_quadrature, slave_quadrature;
  
        // Initialize slave quadrature. NB: only called once per slave.
        // default quad selection: all quadrature points lie on that part of the interface,
        // which has non-empty intersection with slave cell
        if_q_select(master_elem, slave_elem, 
                           master_facet_number, slave_facet_number, 
                           master_quadrature, slave_quadrature);
  
        if (remote_index_master == -1 || remote_index_slave == -1) 
        {
          // master / slave
          L_S = 0.;
          local_asm(master_elem, slave_elem, 
                    master_quadrature, slave_quadrature,
                    master_facet_number, slave_facet_number, 
                    InterfaceSide::SLAVE, 
                    s, num_slaves,
                    L_S);
  
          if (num_slaves > 1 && rank == PRINT_RANK) 
          {
            LOG_DEBUG(2, "[" << rank << "] Master index: " << it->master_index()
                             << " with remote index " << remote_index_master
                             << " and slave index " << it->slave_index(s)
                             << " with remote index " << remote_index_slave
                             << ", add to master_cell L_S=" << 0.5 * L_S);
          }
  
          // contribution of interface is shared between master and slave cells
          if (remote_index_master == -1) 
          {
            values[it->master_index()] += 0.5 * L_S;
          }
          if (remote_index_slave == -1) 
          {
            values[it->slave_index(s)] += 0.5 * L_S;
          }
        }
      }
    }
  }
};

template < template < class, class, int > class AlgorithmType, class DataType, int DIM >
class StandardBoundaryScalarAssembly
    : public AssemblyAlgorithmBase<AlgorithmType, StandardBoundaryScalarAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > 
{
public:
  typedef std::vector< DataType > LocalObjectType;
  typedef hiflow::Quadrature< DataType > QuadratureType;
  typedef AssemblyAlgorithmBase< AlgorithmType, StandardBoundaryScalarAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > Base;

  // Name resolution does not manage to get the base members, so
  // we must do it ourselves.
  using Base::curr_;
  using Base::elem_;
  using Base::has_next;
  using Base::space_;
  using Base::traversal_;

  StandardBoundaryScalarAssembly(const VectorSpace< DataType, DIM > &space,
                                 std::vector< DataType > &values)
      : Base(space), values_(values) {
    remove_non_local_elements();
    sort_elements(space, this->traversal_);
  }

  /// The has_next() and next() functions are overloaded in
  /// order to skip elements that do not belong to the local
  /// subdomain. This is done by setting the corresponding
  /// entries in the traversal_ array to -1, and later skipping
  /// those items.

  const Element< DataType, DIM > &next() {
    assert(this->has_next());

    this->elem_ = Element< DataType, DIM >(this->space_, this->traversal_[this->curr_]);

    ++(this->curr_);

    return this->elem_;
  }

  void add(const Element< DataType, DIM > &element, const LocalObjectType &local_val) 
  {
    mesh::TDim tdim = this->space_.mesh().tdim();
    mesh::IncidentEntityIterator iter = element.get_cell().begin_incident(tdim - 1);
    mesh::IncidentEntityIterator end = element.get_cell().end_incident(tdim - 1);
    int facet_number = 0;
    for (; iter != end; iter++) 
    {
      this->values_[iter->index()] += local_val[facet_number];
      ++facet_number;
    }
  }

  void reset(typename GlobalAssembler< DataType, DIM >::LocalVector &local_vec) 
  {
    mesh::TDim tdim = this->space_.mesh().tdim();
    local_vec.clear();
    local_vec.resize(this->elem_.get_cell().num_incident_entities(tdim - 1), 0.);
  }

private:
  /// Remove non_local elements

  void remove_non_local_elements() {

    const mesh::Mesh &mesh = this->space_.mesh();

    if (!mesh.has_attribute("_remote_index_", mesh.tdim())) {
      // If the "_remote_index_" attribute does not exist, we
      // assume that there are no ghost cells.
      return;
    }

    int remote_index;
    int index;
    std::vector< int > traversal_tmp;
    traversal_tmp.reserve(mesh.num_entities(mesh.tdim()));

    for (mesh::EntityIterator it_cell = mesh.begin(mesh.tdim()),
                              e_it_cell = mesh.end(mesh.tdim());
         it_cell != e_it_cell; ++it_cell) {
      // test if cell on subdomain
      it_cell->get("_remote_index_", &remote_index);
      if (remote_index == -1) {
        index = it_cell->index();
        traversal_tmp.push_back(index);
      }
    }
    this->traversal_ = traversal_tmp;
  }

  std::vector< DataType > &values_;
};

template < template < class, class, int > class AlgorithmType, class DataType, int DIM >
class StandardMultipleScalarAssembly
    : public AssemblyAlgorithmBase<AlgorithmType,
                                   StandardMultipleScalarAssembly< AlgorithmType, DataType, DIM >,
                                   DataType, DIM > 
{
public:
  typedef std::vector< DataType > LocalObjectType;
  typedef hiflow::Quadrature< DataType > QuadratureType;
  typedef AssemblyAlgorithmBase< AlgorithmType, StandardMultipleScalarAssembly< AlgorithmType, DataType, DIM >, DataType, DIM > Base;

  // Name resolution does not manage to get the base members, so
  // we must do it ourselves.
  using Base::curr_;
  using Base::elem_;
  using Base::has_next;
  using Base::space_;
  using Base::traversal_;

  StandardMultipleScalarAssembly(const VectorSpace< DataType, DIM > &space,
                                 std::vector< std::vector< DataType > > &values,
                                 const size_t num_scalars)
      : Base(space), num_scalars_(num_scalars), values_(values) {
    const size_t num_elements = this->traversal_.size();
    this->values_.resize(num_elements);
    for (size_t l = 0; l < num_elements; ++l) {
      this->values_[l].resize(this->num_scalars_, 0.);
    }
    this->remove_non_local_elements();
    sort_elements(space, this->traversal_);
  }

  /// The has_next() and next() functions are overloaded in
  /// order to skip elements that do not belong to the local
  /// subdomain. This is done by setting the corresponding
  /// entries in the traversal_ array to -1, and later skipping
  /// those items.

  const Element< DataType, DIM > &next() {
    assert(this->has_next());

    this->elem_ = Element< DataType, DIM >(this->space_, this->traversal_[this->curr_]);

    ++(this->curr_);

    return this->elem_;
  }

  void add(const Element< DataType, DIM > &element, const LocalObjectType &local_val) 
  {
    for (size_t l = 0; l < this->num_scalars_; ++l) 
    {
      this->values_[element.cell_index()][l] += local_val[l];
    }
  }

  void reset(typename GlobalAssembler< DataType, DIM >::LocalVector &local_vec) 
  {
    local_vec.clear();
    local_vec.resize(this->num_scalars_, 0.);
  }
  
private:
  /// Remove non_local elements

  void remove_non_local_elements() {

    const mesh::Mesh &mesh = this->space_.mesh();

    if (!mesh.has_attribute("_remote_index_", mesh.tdim())) {
      // If the "_remote_index_" attribute does not exist, we
      // assume that there are no ghost cells.
      return;
    }

    int remote_index;
    int index;
    std::vector< int > traversal_tmp;
    traversal_tmp.reserve(mesh.num_entities(mesh.tdim()));

    for (mesh::EntityIterator it_cell = mesh.begin(mesh.tdim()),
                              e_it_cell = mesh.end(mesh.tdim());
         it_cell != e_it_cell; ++it_cell) {
      // test if cell on subdomain
      it_cell->get("_remote_index_", &remote_index);
      if (remote_index == -1) {
        index = it_cell->index();
        traversal_tmp.push_back(index);
      }
    }
    this->traversal_ = traversal_tmp;
  }
  size_t num_scalars_;
  std::vector< std::vector< DataType > > &values_;
};

template < class DataType, int DIM >
class InterfaceCellMultipleScalarAssembly
{
public:

  // TODO: this could be optimized by splitting into separate cell/facet
  // selection functions.
  typedef std::function < void ( const Element< DataType, DIM > &, const Element< DataType, DIM > &, 
                                 int, int,
                                 Quadrature< DataType > &, Quadrature< DataType > &) >
                                 IFQuadratureSelectionFun;

  typedef typename Local2GlobalInserter<DataType, DIM>::LocalVector LocalVector;

  InterfaceCellMultipleScalarAssembly()
  {}

  template< class LocalAssembler>
  void assemble( const VectorSpace< DataType, DIM > &space,
                 const mesh::InterfaceList& if_list,
                 IFQuadratureSelectionFun if_q_select,
                 LocalAssembler& local_asm, 
                 std::vector< LocalVector > &values
                 ) const
  {
    // Create interface list from mesh
    mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive

    #ifndef NDEBUG
    int rank = space.dof().my_subdom();
    #endif

    // Loop over interfaces
    for (mesh::InterfaceList::const_iterator it = if_list.begin(), end_it = if_list.end(); it != end_it; ++it) 
    {
        int remote_index_master = -10;
        mesh->get_attribute_value("_remote_index_", mesh->tdim(), it->master_index(), &remote_index_master);

        // Master dofs
        Element< DataType, DIM > master_elem(space, it->master_index());

        const int master_facet_number = it->master_facet_number();

        LocalVector L_M, L_S;

        Quadrature< DataType > master_master_quadrature;

        // Initialize master quadrature
        if_q_select(master_elem, master_elem, master_facet_number,
                        master_facet_number, master_master_quadrature,
                        master_master_quadrature);

        const int num_slaves = it->num_slaves();

        // Boundary integral
        if (remote_index_master == -1) 
        {
        if (num_slaves == 0) {
            // boundary facet
            local_asm(master_elem, master_elem, 
                    master_master_quadrature, master_master_quadrature, 
                    master_facet_number, master_facet_number, 
                    InterfaceSide::BOUNDARY, 
                    -1, 0,
                    L_M);

            LOG_DEBUG(3, "[" << rank << "] Master index: " << it->master_index()
                            << " with remote index " << remote_index_master
                            << ", add to master_cell L_MM="
                            << string_from_range(L_M.begin(), L_M.end()));

            assert(values[it->master_index()].size() == L_M.size());
            for (size_t l = 0; l < L_M.size(); ++l) 
            {
            values[it->master_index()][l] += L_M[l];
            }
        }
        }

        // Interface integrals
        // Loop over slaves
        for (int s = 0; s < num_slaves; ++s) 
        {
        int remote_index_slave = -10;
        mesh->get_attribute_value("_remote_index_", mesh->tdim(), it->slave_index(s), &remote_index_slave);
        Element< DataType, DIM > slave_elem(space, it->slave_index(s));
        const int slave_facet_number = it->slave_facet_number(s);

        Quadrature< DataType > master_quadrature, slave_quadrature;

        // Initialize slave quadrature. NB: only called once per slave.
        // default quad selection: all quadrature points lie on that part of the interface,
        // which has non-empty intersection with slave cell
        if_q_select(master_elem, slave_elem, 
                            master_facet_number, slave_facet_number, 
                            master_quadrature, slave_quadrature);

        if (remote_index_master == -1 || remote_index_slave == -1) 
        {
            local_asm(master_elem, slave_elem, 
                    master_quadrature, slave_quadrature,
                    master_facet_number, slave_facet_number, 
                    InterfaceSide::SLAVE, 
                    s, num_slaves, 
                    L_S);

            assert(values[it->master_index()].size() == L_S.size());
            assert(values[it->slave_index(s)].size() == L_S.size());

            // contribution of interface is shared between master and slave cells
            if (remote_index_master == -1) 
            {
            for (size_t l = 0; l < L_S.size(); ++l) 
            {
                values[it->master_index()][l] += 0.5 * L_S[l];
            }
            }
            if (remote_index_slave == -1) 
            {
            for (size_t l = 0; l < L_S.size(); ++l) 
            {
                values[it->slave_index(s)][l] += 0.5 * L_S[l];
            }
            }
        }
        }
    }
    }
};

template < template < class, class, int > class AlgorithmType, class DataType, int DIM >
class StandardBoundaryMultipleScalarAssembly
    : public AssemblyAlgorithmBase<
          AlgorithmType,
          StandardBoundaryMultipleScalarAssembly< AlgorithmType, DataType, DIM >,
          DataType,
          DIM > {
public:
  typedef std::vector< std::vector<DataType> > LocalObjectType;
  typedef hiflow::Quadrature< DataType > QuadratureType;
  typedef AssemblyAlgorithmBase<
      AlgorithmType, StandardBoundaryMultipleScalarAssembly< AlgorithmType, DataType, DIM >,
      DataType,
      DIM >
      Base;

  typedef typename Local2GlobalInserter<DataType, DIM>::GlobalVector GlobalVector;
  typedef typename Local2GlobalInserter<DataType, DIM>::LocalVector LocalVector;

  // Name resolution does not manage to get the base members, so
  // we must do it ourselves.
  using Base::curr_;
  using Base::elem_;
  using Base::has_next;
  using Base::space_;
  using Base::traversal_;

  StandardBoundaryMultipleScalarAssembly(const VectorSpace< DataType, DIM > &space,
                                         std::vector< std::vector<DataType> > &values,
                                         size_t num_scalars)
      : Base(space), values_(values), num_scalars_(num_scalars) 
  {
    this->remove_non_local_elements();
    sort_elements(space, this->traversal_);
  }

  /// The has_next() and next() functions are overloaded in
  /// order to skip elements that do not belong to the local
  /// subdomain. This is done by setting the corresponding
  /// entries in the traversal_ array to -1, and later skipping
  /// those items.

  const Element< DataType, DIM > &next() {
    assert(this->has_next());

    this->elem_ = Element< DataType, DIM >(this->space_, this->traversal_[this->curr_]);

    ++(this->curr_);

    return this->elem_;
  }

  void add(const Element< DataType, DIM > &element,
           const LocalObjectType &local_val) 
  {
    mesh::TDim tdim = this->space_.mesh().tdim();
    mesh::IncidentEntityIterator iter = element.get_cell().begin_incident(tdim - 1);
    mesh::IncidentEntityIterator end = element.get_cell().end_incident(tdim - 1);

    int facet_number = 0;
    for (; iter != end; iter++) 
    {
      const int facet_id = iter->index();

      assert (facet_id < this->values_.size() );
      assert (facet_number < local_val.size());
      assert (local_val[facet_number].size() == this->num_scalars_);
      assert (this->values_[facet_id].size() == this->num_scalars_);
      
      for (size_t l=0; l<this->num_scalars_; ++l)
      {
        this->values_[facet_id][l] += local_val[facet_number][l];
      }
      ++facet_number;
    }
  }

  void reset(std::vector<LocalVector> &local_vec) 
  {
    mesh::TDim tdim = this->space_.mesh().tdim();
    local_vec.clear();
    local_vec.resize(this->elem_.get_cell().num_incident_entities(tdim - 1));
    for (int f=0; f<local_vec.size(); ++f)
    {
      local_vec[f].clear();
      local_vec[f].resize(this->num_scalars_, 0.);
    }
  }

private:
  /// Remove non_local elements

  void remove_non_local_elements() {

    const mesh::Mesh &mesh = this->space_.mesh();

    if (!mesh.has_attribute("_remote_index_", mesh.tdim())) {
      // If the "_remote_index_" attribute does not exist, we
      // assume that there are no ghost cells.
      return;
    }

    int remote_index;
    int index;
    std::vector< int > traversal_tmp;
    traversal_tmp.reserve(mesh.num_entities(mesh.tdim()));

    for (mesh::EntityIterator it_cell = mesh.begin(mesh.tdim()),
                              e_it_cell = mesh.end(mesh.tdim());
         it_cell != e_it_cell; ++it_cell) {
      // test if cell on subdomain
      it_cell->get("_remote_index_", &remote_index);
      if (remote_index == -1) {
        index = it_cell->index();
        traversal_tmp.push_back(index);
      }
    }
    this->traversal_ = traversal_tmp;
  }
  
  std::vector< std::vector<DataType> > &values_;
  size_t num_scalars_;
};

} // namespace hiflow
#endif
