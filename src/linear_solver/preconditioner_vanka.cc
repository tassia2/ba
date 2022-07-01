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

/// \author Simon Gawlok

#include "linear_solver/preconditioner_vanka.h"
#include "dof/dof_partition.h"
#include "space/vector_space.h"

namespace hiflow {
namespace la {

template < class LAD, int DIM >
PreconditionerVanka< LAD, DIM >::PreconditionerVanka()
    : LinearSolver<LAD, LAD >() 
{   
}

template < class LAD, int DIM > PreconditionerVanka< LAD, DIM >::~PreconditionerVanka() 
{
  this->Clear();
}

template < class LAD, int DIM >
void PreconditionerVanka< LAD, DIM >::InitIluppPrecond(
    int prepro_type, 
    int precond_no, 
    int max_levels, 
    DataType mem_factor,
    DataType threshold, 
    DataType min_pivot) 
{
  this->precond_.InitParameter(prepro_type, precond_no, max_levels, mem_factor,
                               threshold, min_pivot);
  this->use_preconditioner_ = true;
}

template < class LAD, int DIM >
void PreconditionerVanka< LAD, DIM >::SetupOperator(OperatorType &op) 
{
  this->op_ = &op;
  this->SetModifiedOperator(true);

  if (this->use_preconditioner_) 
  {
    this->precond_.SetupOperator(*(this->op_));
    this->precond_.SetModifiedOperator(true);
  }
}

template < class LAD, int DIM >
void PreconditionerVanka< LAD, DIM >::InitParameter(
    const hiflow::VectorSpace< DataType, DIM > &space, 
    const DataType damping_param,
    const int num_iter, 
    const bool use_preconditioner,
    const VankaPatchMode patch_mode,
    const bool prebuild_local_matrices,
    const int local_LU_block_size) 
{
  this->space_ = &space;
  this->damping_param_ = damping_param;
  this->maxits_ = num_iter;
  this->use_preconditioner_ = use_preconditioner;
  this->local_LU_block_size_ = local_LU_block_size;
  this->prebuild_matrices_ = prebuild_local_matrices;
  this->patch_mode_ = patch_mode;
  if (this->print_level_ > 2) {
    LOG_INFO("Damping parameter", this->damping_param_);
    LOG_INFO("Number of iterations", this->maxits_);
  }

  switch (patch_mode) 
  {
    case VankaPatchMode::SingleCell:
      this->create_index_sets_singlecell();
      break; 
    case VankaPatchMode::VertexPatch:
      this->create_index_sets_patch(0);
      break;
    case VankaPatchMode::FacetPatch:
      this->create_index_sets_patch(DIM-1);
      break;  
    case VankaPatchMode::CellPatch:
      this->create_index_sets_patch(DIM);
      break;
    default:
      assert (false);
      break;
  }

  if (this->prebuild_matrices_)
  {
    this->create_local_matrices();
  }
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::Clear() 
{
  this->sorted_dofs_diag_.clear();
  this->clear_local_matrices();

  if (this->use_preconditioner_) 
  {
    this->precond_.Clear();
  }
  Preconditioner< LAD >::Clear();
}

template < class LAD, int DIM >
void PreconditionerVanka< LAD, DIM >::BuildImpl(VectorType const *b, VectorType *x) 
{
  assert(this->op_ != nullptr);

  if (this->prebuild_matrices_)
  {
    this->fill_local_matrices_from_operator();
    this->factorize_local_matrices();
  }

  if (this->use_preconditioner_) 
  {
    if (!this->precond_.GetReuse() || !this->precond_.GetState()) 
    {
      this->precond_.Build(b, x);
      this->precond_.SetState(true);
      this->precond_.SetModifiedOperator(false);
    }
  }
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::sort_and_reduce_index_sets() 
{
  const int num_sets = this->sorted_dofs_diag_.size();
  for (int i=0; i!=num_sets; ++i)
  {
    // sort dofs and make them unique
    std::sort(sorted_dofs_diag_[i].begin(), sorted_dofs_diag_[i].end());
    sorted_dofs_diag_[i].erase( unique(sorted_dofs_diag_[i].begin(),
                                       sorted_dofs_diag_[i].end()),
                                       sorted_dofs_diag_[i].end());

    // reduce dof indices to only subdomain indices
    bool found = true;
    while (found) 
    {
      found = false;
      for (int j = 0, j_e = sorted_dofs_diag_[i].size(); j != j_e && !found; ++j) 
      {
        if (!this->space_->dof().is_dof_on_subdom(sorted_dofs_diag_[i][j])) 
        {
          sorted_dofs_diag_[i].erase(sorted_dofs_diag_[i].begin() + j);
          found = true;
        }
      }
    }
  }

  // remove cells with empty dof list
  bool found = true;
  while (found) 
  {
    found = false;
    for (int i = 0, i_e = sorted_dofs_diag_.size(); i != i_e && !found; ++i) 
    {
      if (sorted_dofs_diag_[i].size() == 0) 
      {
        sorted_dofs_diag_.erase(sorted_dofs_diag_.begin() + i);
        found = true;
      }
    }
  }
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::create_index_sets_singlecell() 
{
  // Get pointer to finite element mesh
  hiflow::mesh::ConstMeshPtr mesh = &(space_->mesh());

  // Get number of elements in (local) mesh
  const auto num_elements = mesh->num_entities(mesh->tdim());

  // prepare indices vectors
  sorted_dofs_diag_.clear();
  sorted_dofs_diag_.resize(num_elements);

  std::vector< int > var_indices;

  // collect indices 
  for (hiflow::mesh::EntityIterator it = mesh->begin(mesh->tdim()),
                                    e_it = mesh->end(mesh->tdim());
       it != e_it; ++it) 
  {
    // get global dof indices on current cell
    for (int fe_ind = 0, var_e = space_->nb_fe(); fe_ind != var_e; ++fe_ind) 
    {    
      var_indices.clear();
      space_->dof().get_dofs_on_cell(fe_ind, it->index(), var_indices);
      sorted_dofs_diag_[it->index()].insert(sorted_dofs_diag_[it->index()].end(), 
                                            var_indices.begin(),
                                            var_indices.end());
    }
  }
  this->sort_and_reduce_index_sets();
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::create_index_sets_patch(int iter_entity_tdim) 
{
  // Get pointer to finite element mesh
  hiflow::mesh::ConstMeshPtr mesh = &(space_->mesh());

  // Get number of elements in (local) mesh
  const auto num_elements = mesh->num_entities(iter_entity_tdim);

  // prepare indices vectors
  sorted_dofs_diag_.clear();
  sorted_dofs_diag_.resize(num_elements);

  std::vector< int > var_indices;

  // loop over base entities 
  for (hiflow::mesh::EntityIterator it = mesh->begin(iter_entity_tdim),
                                    e_it = mesh->end(iter_entity_tdim);
       it != e_it; ++it) 
  {
    // get global dof indices on current cell
    for (int fe_ind = 0, var_e = space_->nb_fe(); fe_ind != var_e; ++fe_ind) 
    {  
      // loop over cells atached to base entity  
      for (auto it_inc = it->begin_incident(mesh->tdim()),
           e_it_inc = it->end_incident(mesh->tdim());
           it_inc != e_it_inc; ++it_inc) 
      {
        var_indices.clear();
        space_->dof().get_dofs_on_cell(fe_ind, it_inc->index(), var_indices);
        sorted_dofs_diag_[it->index()].insert(sorted_dofs_diag_[it->index()].end(), 
                                              var_indices.begin(),
                                              var_indices.end());
      }
    }
  }
  this->sort_and_reduce_index_sets();
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::clear_local_matrices() 
{
  const int old_size = this->local_mat_diag_.size();
  for (int i=0; i!=old_size; ++i)
  {
    if (this->local_mat_diag_[i] != nullptr)
    {
      this->local_mat_diag_[i]->Clear();
      delete this->local_mat_diag_[i];
    }
  }
  this->local_mat_diag_.clear();
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::create_local_matrices() 
{
  const int num_local_cells = sorted_dofs_diag_.size();
  assert (num_local_cells > 0);

  this->clear_local_matrices();
  
  // prepare local matrices
  local_mat_diag_.resize(num_local_cells, nullptr);
  for (int i = 0, i_e = local_mat_diag_.size(); i != i_e; ++i) 
  {
    // Resize local matrix to correct size
    const int N = sorted_dofs_diag_[i].size();
    local_mat_diag_[i] = new SeqDenseMatrix< DataType >;
    local_mat_diag_[i]->Resize(N, N);

    // Set block-size for LU decomposition
    if (this->local_LU_block_size_ == -1)
    {
      local_mat_diag_[i]->set_blocksize(N);
    }
    else 
    {
      local_mat_diag_[i]->set_blocksize(this->local_LU_block_size_);
    }
  }
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::fill_local_matrices_from_operator() 
{
  assert (this->local_mat_diag_.size() == sorted_dofs_diag_.size());
  assert (this->op_ != nullptr);
  const int num_local_mat = sorted_dofs_diag_.size();
  assert (num_local_mat > 0);

  for (int i = 0, i_e = num_local_mat; i != i_e; ++i) 
  {
    const int N = sorted_dofs_diag_[i].size();
    // Get local matrix from global operator
    this->op_->GetValues(
          vec2ptr(sorted_dofs_diag_[i]), N,
          vec2ptr(sorted_dofs_diag_[i]), N,
          &((*local_mat_diag_[i])(0, 0)));
  }
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::factorize_local_matrices() 
{
  const int num_local_mat = sorted_dofs_diag_.size();
  for (int i = 0; i < num_local_mat; ++i) 
  {
    assert (local_mat_diag_[i] != nullptr);
    local_mat_diag_[i]->Factorize();
  }
}

template < class LAD, int DIM > 
void PreconditionerVanka< LAD, DIM >::build_tmp_matrix(int i)
{
  const int N = sorted_dofs_diag_[i].size();
  // Resize local matrix to correct size
  local_mat_diag_block_mode_.Resize(N, N);

  // Get local matrix from global operator
  this->op_->GetValues(
            vec2ptr(sorted_dofs_diag_[i]), N,
            vec2ptr(sorted_dofs_diag_[i]), N,
            &((local_mat_diag_block_mode_)(0, 0)));

  // Set block-size for LU decomposition
  if (this->local_LU_block_size_ == -1)
  {
    local_mat_diag_block_mode_.set_blocksize(N);
  }
  else 
  {
    local_mat_diag_block_mode_.set_blocksize(this->local_LU_block_size_);
  }
}

template < class LAD, int DIM >
LinearSolverState PreconditionerVanka< LAD, DIM >::SolveImpl(const VectorType &b,
                                                             VectorType *x) 
{
  if (this->use_preconditioner_) 
  {
    this->precond_.Solve(b, x);
  }

  for (int k = 0, k_e = this->maxits_; k != k_e; ++k) 
  {
    x->Update();

    // Loop over all patches
    for (size_t i = 0, i_e = sorted_dofs_diag_.size(); i != i_e; ++i) 
    {
      // COMPUTE CURRENT RESIDUAL
      const int N = sorted_dofs_diag_[i].size();

      res_loc.clear();
      res_loc_2.clear();
      x_temp_loc.clear();

      res_loc.resize(N, 0.);
      res_loc_2.resize(N, 0.);
      x_temp_loc.resize(N, 0.);

      this->op_->VectorMult_submatrix_vanka(vec2ptr(sorted_dofs_diag_[i]),
                                            N, *x,
                                            vec2ptr(res_loc));

      x->GetValues(vec2ptr(sorted_dofs_diag_[i]), N, vec2ptr(x_temp_loc));

      if (this->prebuild_matrices_) 
      {
        local_mat_diag_[i]->VectorMult(x_temp_loc, res_loc_2);
      } 
      else 
      {
        this->build_tmp_matrix(i);
        local_mat_diag_block_mode_.VectorMult(x_temp_loc, res_loc_2);
      }

      // Get local rhs
      b_loc.resize(N, 0.);
      b.GetValues(vec2ptr(sorted_dofs_diag_[i]), N, vec2ptr(b_loc));

      for (size_t j = 0, j_e = b_loc.size(); j != j_e; ++j) 
      {
        b_loc[j] = b_loc[j] - res_loc[j] + res_loc_2[j];
      }

      // Solve local problem
      x_loc.resize(N, 0.);
      if (this->prebuild_matrices_) 
      {
        local_mat_diag_[i]->ForwardBackward(b_loc, x_loc);
      } 
      else 
      {
        local_mat_diag_block_mode_.Solve(b_loc, x_loc);
      }

      // add local contribution to solution vector
      for (size_t j = 0, j_e = x_loc.size(); j != j_e; ++j) 
      {
        x_temp_loc[j] = (1. - this->damping_param_) * x_temp_loc[j] + this->damping_param_ * x_loc[j];
      }

      x->SetValues(vec2ptr(sorted_dofs_diag_[i]), N, vec2ptr(x_temp_loc));
    }

    // Loop backward over all "cells"
    for (int i = sorted_dofs_diag_.size(); i--;) 
    {
      // COMPUTE CURRENT RESIDUAL
      const int N = sorted_dofs_diag_[i].size();
      res_loc.resize(N, 0.);
      res_loc_2.resize(N, 0.);
      x_temp_loc.resize(N, 0.);

      this->op_->VectorMult_submatrix_vanka(vec2ptr(sorted_dofs_diag_[i]),
                                            N, *x,
                                            vec2ptr(res_loc));

      x->GetValues(vec2ptr(sorted_dofs_diag_[i]), N, vec2ptr(x_temp_loc));

      if (this->prebuild_matrices_) 
      {
        local_mat_diag_[i]->VectorMult(x_temp_loc, res_loc_2);
      } 
      else 
      {
        this->build_tmp_matrix(i);
        local_mat_diag_block_mode_.VectorMult(x_temp_loc, res_loc_2);
      }

      // Get local rhs
      b_loc.resize(N, 0.);
      b.GetValues(vec2ptr(sorted_dofs_diag_[i]), N, vec2ptr(b_loc));

      for (size_t j = 0, j_e = b_loc.size(); j != j_e; ++j) 
      {
        b_loc[j] = b_loc[j] - res_loc[j] + res_loc_2[j];
      }

      // Solve local problem
      x_loc.resize(N, 0.);
      if (this->prebuild_matrices_) 
      {
        local_mat_diag_[i]->ForwardBackward(b_loc, x_loc);
      } 
      else 
      {
        local_mat_diag_block_mode_.Solve(b_loc, x_loc);
      }

      // add local contribution to solution vector
      for (size_t j = 0, j_e = x_loc.size(); j != j_e; ++j) 
      {
        x_temp_loc[j] = (1. - this->damping_param_) * x_temp_loc[j] + this->damping_param_ * x_loc[j];
      }

      x->SetValues(vec2ptr(sorted_dofs_diag_[i]), N, vec2ptr(x_temp_loc));
    }
  }
  x->Update();

  return kSolverSuccess;
}



/// template instantiation
template class PreconditionerVanka< hiflow::la::LADescriptorCoupledD, 2 >;
template class PreconditionerVanka< hiflow::la::LADescriptorCoupledD, 3 >;


} // namespace la
} // namespace hiflow
