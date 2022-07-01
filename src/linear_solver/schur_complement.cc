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

/// @author Simon Gawlok

#include "linear_solver/schur_complement.h"
#include <nonlinear/nonlinear_problem.h>
#include "space/vector_space.h"
#include "dof/dof_partition.h"

namespace hiflow {
namespace la {

template < class LAD, int DIM >
SchurComplement< LAD, DIM >::SchurComplement() : LinearSolver< LAD >() {
  this->comm_ = MPI_COMM_NULL;
  this->nb_procs_ = -1;
  this->my_rank_ = -1;
  this->dpx_precond_type_ = DPX_NOT_SET;
  this->precond_ = nullptr;
  this->is_simple_precond_ = false;
  this->filter_solution_ = false;
  this->scaling_block_two_matrix_ = 1.;
  this->print_level_ = 0;
  this->solver_A_ = nullptr;
  this->solver_D_ = nullptr;
  this->solver_Q_ = nullptr;
  this->solver_H_ = nullptr;

  this->Q_is_initialized_ = false;
  this->F_is_initialized_ = false;
  this->H_is_initialized_ = false;
  this->Q_is_set_ = false;
  this->F_is_set_ = false;
  this->H_is_set_ = false;
  this->block_two_matrix_is_initialized_ = false;
  this->block_two_matrix_is_set_ = false;
  this->solver_precond_block_two_ = nullptr;
  this->nested_level_ = 0;
  this->scaling_D_ = 0.0;
  this->scaling_S_ = 0.0;

  this->A_modified_ = false;
  this->A_passed2solver_ = false;
  this->Q_modified_ = false;
  this->Q_passed2solver_ = false;
  this->H_modified_ = false;
  this->H_passed2solver_ = false;
  this->D_modified_ = false;
  this->D_passed2solver_ = false;

  this->precond_two_modified_ = false;
  this->precond_two_passed2solver_ = false;
  this->use_schur_precond_only_ = false;
}

template < class LAD, int DIM > SchurComplement< LAD, DIM >::~SchurComplement() {
  this->Clear();
}

template < class LAD, int DIM > void SchurComplement< LAD, DIM >::Clear() {
  LinearSolver< LAD >::Clear();
  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
    assert(this->comm_ == MPI_COMM_NULL);
  }

  this->my_rank_ = -1;
  this->nb_procs_ = -1;
  this->dpx_precond_type_ = DPX_NOT_SET;
  this->couplings_A_.clear();
  this->couplings_B_.clear();
  this->couplings_C_.clear();
  this->couplings_D_.clear();
  this->sparsity_A_.clear();
  this->sparsity_B_.clear();
  this->sparsity_C_.clear();
  this->sparsity_D_.clear();
  this->offsets_block_one_.clear();
  this->offsets_block_two_.clear();
  this->mapb2s_one_.clear();
  this->mapb2s_two_.clear();
  this->maps2b_one_.clear();
  this->maps2b_two_.clear();
  this->la_c_one_.Clear();
  this->la_c_two_.Clear();
  this->indexset_one_.clear();
  this->indexset_two_.clear();
  this->val_temp_one_.clear();
  this->val_temp_two_.clear();
  this->A_.Clear();
  this->B_.Clear();
  this->C_.Clear();
  this->D_.Clear();
  this->Q_.Clear();
  this->H_.Clear();
  this->F_.Clear();
  this->dInvAB_.Clear();
  this->Q_is_initialized_ = false;
  this->F_is_initialized_ = false;
  this->H_is_initialized_ = false;
  this->Q_is_set_ = false;
  this->F_is_set_ = false;
  this->H_is_set_ = false;
  this->block_two_matrix_is_initialized_ = false;
  this->block_two_matrix_is_set_ = false;
  this->block_two_matrix_.Clear();
  this->precond_two_.Clear();
  this->f_.Clear();
  this->g_.Clear();
  this->x_.Clear();
  this->y_.Clear();
  this->h_one_1_.Clear();
  this->rhs_x_.Clear();
  this->h1_.Clear();
  this->h2_.Clear();
  this->h_two_1_.Clear();
  this->w_.Clear();
  this->h3_.Clear();

  for (size_t i = 0; i != this->V_.size(); ++i) {
    delete this->V_[i];
  }
  this->V_.clear();

  // preconditioned Krylov subspace basis
  for (size_t i = 0; i != this->Z_.size(); ++i) {
    delete this->Z_[i];
  }
  this->Z_.clear();
  if (this->precond_ != nullptr) {
    this->precond_->Clear();
    this->precond_ = nullptr;
  }
  this->diag_A_.clear();
  this->non_lin_op_ = nullptr;
  if (this->solver_A_ != nullptr) {
    this->solver_A_->Clear();
    this->solver_A_ = nullptr;
  }
  if (this->solver_D_ != nullptr) {
    this->solver_D_->Clear();
    this->solver_D_ = nullptr;
  }
  if (this->solver_Q_ != nullptr) {
    this->solver_Q_->Clear();
    this->solver_Q_ = nullptr;
  }
  if (this->solver_H_ != nullptr) {
    this->solver_H_->Clear();
    this->solver_H_ = nullptr;
  }
  if (this->solver_precond_block_two_ != nullptr) {
    this->solver_precond_block_two_->Clear();
    this->solver_precond_block_two_ = nullptr;
  }
  this->scaling_D_ = 0.0;
  this->scaling_S_ = 0.0;
  this->nested_level_ = 0;
  this->use_schur_precond_only_ = false;
}

template < class LAD, int DIM > void SchurComplement< LAD, DIM >::DestroySolver() {
  if (this->solver_A_ != nullptr) {
    if (this->solver_A_->IsCritical()) {
      this->solver_A_->DestroySolver();
    }
  }
  if (this->solver_D_ != nullptr) {
    if (this->solver_D_->IsCritical()) {
      this->solver_D_->DestroySolver();
    }
  }
  if (this->solver_Q_ != nullptr) {
    if (this->solver_Q_->IsCritical()) {
      this->solver_Q_->DestroySolver();
    }
  }
  if (this->solver_H_ != nullptr) {
    if (this->solver_H_->IsCritical()) {
      this->solver_H_->DestroySolver();
    }
  }
  if (this->solver_precond_block_two_ != nullptr) {
    if (this->solver_precond_block_two_->IsCritical()) {
      this->solver_precond_block_two_->DestroySolver();
    }
  }
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::Init(const hiflow::VectorSpace< ValueType, DIM > &space,
                                  const std::vector< int > &block_one_variables,
                                  const std::vector< int > &block_two_variables,
                                  const PrecondTwoType precond_two_type,
                                  const DPXPrecondType dpx_precond_type) {
  // clear member variables
  this->Clear();

  //*****************************************************************
  // Set Schur preconditioning type
  //*****************************************************************
  precond_two_type_ = precond_two_type;
  dpx_precond_type_ = dpx_precond_type;

  if (precond_two_type_ == DPXPrecond && dpx_precond_type_ == DPX_NOT_SET) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Using DPXPrecond, the DPXPrecondType has to be set!!!");
    quit_program();
  }

  //*****************************************************************
  // Set number of variables in the blocks
  //*****************************************************************
  num_var_one_ = block_one_variables.size();
  num_var_two_ = block_two_variables.size();

  //*****************************************************************
  // Create own duplicate of MPI communicator of space
  //*****************************************************************
  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
  }

  assert(space.get_mpi_comm() != MPI_COMM_NULL);
  // MPI communicator

  // determine nb. of processes
  int info = MPI_Comm_size(space.get_mpi_comm(), &nb_procs_);
  assert(info == MPI_SUCCESS);
  assert(nb_procs_ > 0);

  // retrieve my rank
  info = MPI_Comm_rank(space.get_mpi_comm(), &my_rank_);
  assert(info == MPI_SUCCESS);
  assert(my_rank_ >= 0);
  assert(my_rank_ < nb_procs_);

  info = MPI_Comm_split(space.get_mpi_comm(), 0, my_rank_, &(this->comm_));
  assert(info == MPI_SUCCESS);

  //*****************************************************************
  // Create couplings/sparsity structure in system numbering
  //*****************************************************************
  couplings_A_.clear();
  couplings_B_.clear();
  couplings_C_.clear();
  couplings_D_.clear();

  typename VectorSpace< ValueType, DIM >::MeshEntityIterator mesh_it =
      space.mesh().begin(space.tdim());
  typename VectorSpace< ValueType, DIM >::MeshEntityIterator e_mesh_it =
      space.mesh().end(space.tdim());

  while (mesh_it != e_mesh_it) {

    std::vector< doffem::gDofId > dof_ind;

    std::vector< std::vector< doffem::lDofId > > dof_ind_block_one;
    dof_ind_block_one.resize(block_one_variables.size());
    std::vector< std::vector< doffem::lDofId > > dof_ind_block_two;
    dof_ind_block_two.resize(block_two_variables.size());

    for (int k = 0, k_e = block_one_variables.size(); k != k_e; ++k) {
      const int var = block_one_variables[k];
      // get dof indices for variable
      space.get_dof_indices(var, mesh_it->index(), dof_ind);
      dof_ind_block_one[k] = dof_ind;

      // clear dof_ind vector
      dof_ind.clear();
    }

    for (int k = 0, k_e = block_two_variables.size(); k != k_e; ++k) {
      const int var = block_two_variables[k];
      // get dof indices for variable
      space.get_dof_indices(var, mesh_it->index(), dof_ind);
      dof_ind_block_two[k] = dof_ind;

      // clear dof_ind vector
      dof_ind.clear();
    }

    // sub matrix A, sub matrix B
    for (int i = 0, i_e = dof_ind_block_one.size(); i != i_e; ++i) {

      // detect couplings
      for (int ii = 0, ii_e = dof_ind_block_one[i].size(); ii != ii_e; ++ii) {
        // dof indice of test variable
        const auto di_i = dof_ind_block_one[i][ii];

        // if my row
        if (space.dof().is_dof_on_subdom(di_i)) {
          // matrix A
          SortedArray< int > *temp = &(couplings_A_[di_i]);

          for (size_t j = 0, j_e = dof_ind_block_one.size(); j != j_e; ++j) {
            for (size_t jj = 0, jj_e = dof_ind_block_one[j].size(); jj != jj_e;
                 ++jj) {
              temp->find_insert(dof_ind_block_one[j][jj]);
            }
          }

          // matrix B
          temp = &(couplings_B_[di_i]);

          for (size_t j = 0, j_e = dof_ind_block_two.size(); j != j_e; ++j) {
            for (size_t jj = 0, jj_e = dof_ind_block_two[j].size(); jj != jj_e;
                 ++jj) {
              temp->find_insert(dof_ind_block_two[j][jj]);
            }
          }
        } // if
      }   // for (int ii = 0 ...)
    }     // for (int i = 0 ...)

    // sub matrix C, sub matrix D
    for (int i = 0, i_e = dof_ind_block_two.size(); i != i_e; ++i) {

      // detect couplings
      for (int ii = 0, ii_e = dof_ind_block_two[i].size(); ii != ii_e; ++ii) {
        // dof indice of test variable
        const int di_i = dof_ind_block_two[i][ii];

        // if my row
        if (space.dof().is_dof_on_subdom(di_i)) {
          // matrix C
          SortedArray< int > *temp = &(couplings_C_[di_i]);

          for (size_t j = 0, j_e = dof_ind_block_one.size(); j != j_e; ++j) {
            for (size_t jj = 0, jj_e = dof_ind_block_one[j].size(); jj != jj_e;
                 ++jj) {
              temp->find_insert(dof_ind_block_one[j][jj]);
            }
          }

          // matrix D
          temp = &(couplings_D_[di_i]);

          for (size_t j = 0, j_e = dof_ind_block_two.size(); j != j_e; ++j) {
            for (size_t jj = 0, jj_e = dof_ind_block_two[j].size(); jj != jj_e;
                 ++jj) {
              temp->find_insert(dof_ind_block_two[j][jj]);
            }
          }
        } // if
      }   // for (int ii = 0 ...)
    }     // for (int i = 0 ...)
    // next cell
    ++mesh_it;
  } // while (mesh_it != ...)

  //*****************************************************************
  // Compute offsets for variables in block one and variables in
  // block two
  //*****************************************************************

  offsets_block_one_.clear();
  offsets_block_two_.clear();
  offsets_block_one_.resize(this->nb_procs_ + 1, 0);
  offsets_block_two_.resize(this->nb_procs_ + 1, 0);

  std::vector< int > local_dof_numbers_one;
  local_dof_numbers_one.reserve(this->nb_procs_);
  std::vector< int > local_dof_numbers_two;
  local_dof_numbers_two.reserve(this->nb_procs_);

  {
    std::vector< int > num_local_dofs(2, -1);
    num_local_dofs[0] = couplings_A_.size();
    num_local_dofs[1] = couplings_C_.size();

    std::vector< int > local_dof_numbers_temp(2 * this->nb_procs_, -1);

    MPI_Allgather(vec2ptr(num_local_dofs), 2, MPI_INT,
                  vec2ptr(local_dof_numbers_temp), 2, MPI_INT, this->comm_);

    // Unpack received data
    for (int i = 0; i < local_dof_numbers_temp.size(); i = i + 2) {
      local_dof_numbers_one.push_back(local_dof_numbers_temp[i]);
      local_dof_numbers_two.push_back(local_dof_numbers_temp[i + 1]);
    }
  }

#ifndef NDEBUG
  for (int i = 0; i < this->nb_procs_; ++i) {
    assert(local_dof_numbers_one[i] > -1);
    assert(local_dof_numbers_two[i] > -1);
  }
#endif

  for (int i = 1; i < this->nb_procs_ + 1; ++i) {
    offsets_block_one_[i] =
        offsets_block_one_[i - 1] + local_dof_numbers_one[i - 1];
    offsets_block_two_[i] =
        offsets_block_two_[i - 1] + local_dof_numbers_two[i - 1];
  }

  //*****************************************************************
  // Compute mappings: system->block and block->system
  //*****************************************************************
  mapb2s_one_.clear();
  mapb2s_two_.clear();

  maps2b_one_.clear();
  maps2b_two_.clear();
  int index = 0;
  mapb2s_one_.resize(local_dof_numbers_one[this->my_rank_], 0);
  for (auto it = couplings_A_.begin(),
           e_it = couplings_A_.end();
       it != e_it; ++it) {
    mapb2s_one_[index] = it->first;
    maps2b_one_[it->first] = index + offsets_block_one_[this->my_rank_];
    ++index;
  }
  index = 0;
  mapb2s_two_.resize(local_dof_numbers_two[this->my_rank_], 0);
  for (auto it = couplings_C_.begin(),
           e_it = couplings_C_.end();
       it != e_it; ++it) {
    mapb2s_two_[index] = it->first;
    maps2b_two_[it->first] = index + offsets_block_two_[this->my_rank_];
    ++index;
  }

  local_dof_numbers_one.clear();
  local_dof_numbers_two.clear();

  //*****************************************************************
  // Compute mappings: system->block of ghost dofs
  //*****************************************************************

  // determine which dofs are ghost and who is owner

  std::vector< SortedArray< int > > ghost_dofs_one;
  std::vector< SortedArray< int > > ghost_dofs_two;

  ghost_dofs_one.resize(this->nb_procs_);
  ghost_dofs_two.resize(this->nb_procs_);

  // sub matrix A
  for (auto it = couplings_A_.begin(),
           e_it = couplings_A_.end();
       it != e_it; ++it) {
    for (int i = 0, e_i = it->second.size(); i != e_i; ++i) {
      if (!space.dof().is_dof_on_subdom(it->second[i])) {
        ghost_dofs_one[space.dof().owner_of_dof(it->second[i])].find_insert(
            it->second[i]);
      }
    }
  }
  // sub matrix C
  for (auto it = couplings_C_.begin(),
           e_it = couplings_C_.end();
       it != e_it; ++it) {
    for (int i = 0, e_i = it->second.size(); i != e_i; ++i) {
      if (!space.dof().is_dof_on_subdom(it->second[i])) {
        ghost_dofs_one[space.dof().owner_of_dof(it->second[i])].find_insert(
            it->second[i]);
      }
    }
  }
  // sub matrix B
  for (auto it = couplings_B_.begin(),
           e_it = couplings_B_.end();
       it != e_it; ++it) {
    for (int i = 0, e_i = it->second.size(); i != e_i; ++i) {
      if (!space.dof().is_dof_on_subdom(it->second[i])) {
        ghost_dofs_two[space.dof().owner_of_dof(it->second[i])].find_insert(
            it->second[i]);
      }
    }
  }
  // sub matrix D
  for (auto it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    for (int i = 0, e_i = it->second.size(); i != e_i; ++i) {
      if (!space.dof().is_dof_on_subdom(it->second[i])) {
        ghost_dofs_two[space.dof().owner_of_dof(it->second[i])].find_insert(
            it->second[i]);
      }
    }
  }

  // Each process exchanges with each process number of shared dofs and
  // determines block-dof-ids of ghost dofs
  std::vector< std::vector< int > > ghost_dofs_block_one;
  std::vector< std::vector< int > > ghost_dofs_block_two;

  ghost_dofs_block_one.resize(this->nb_procs_);
  ghost_dofs_block_two.resize(this->nb_procs_);

  {
    // BLOCK ONE FIRST

    // numbers of DoFs which I request of other processes
    std::vector< int > num_dofs_requested_block_one(this->nb_procs_, 0);
    int total_number_dofs_requested = 0;
    for (int i = 0; i < this->nb_procs_; ++i) {
      num_dofs_requested_block_one[i] = ghost_dofs_one[i].size();
      total_number_dofs_requested += ghost_dofs_one[i].size();
    }

    // numbers of DoFs which other processes request from me
    std::vector< int > num_dofs_requested_block_one_others(this->nb_procs_, 0);

    // Exchange requested numbers of DoFs
    MPI_Alltoall(vec2ptr(num_dofs_requested_block_one), 1, MPI_INT,
                 vec2ptr(num_dofs_requested_block_one_others), 1, MPI_INT,
                 this->comm_);

    // DoF numbers which I need from others
    std::vector< int > dofs_requested_block_one;
    dofs_requested_block_one.reserve(total_number_dofs_requested);

    std::vector< int > offsets_dofs_requested_block_one(this->nb_procs_ + 1, 0);
    for (int i = 0; i < this->nb_procs_; ++i) {
      dofs_requested_block_one.insert(dofs_requested_block_one.end(),
                                      ghost_dofs_one[i].begin(),
                                      ghost_dofs_one[i].end());
      offsets_dofs_requested_block_one[i + 1] =
          offsets_dofs_requested_block_one[i] + ghost_dofs_one[i].size();
    }

    // DoF numbers which others request from me
    std::vector< int > dofs_requested_block_one_others;
    int total_number_dofs_requested_others = 0;

    std::vector< int > offsets_dofs_requested_block_one_others(
        this->nb_procs_ + 1, 0);
    for (int i = 0; i < this->nb_procs_; ++i) {
      offsets_dofs_requested_block_one_others[i + 1] =
          offsets_dofs_requested_block_one_others[i] +
          num_dofs_requested_block_one_others[i];
      total_number_dofs_requested_others +=
          num_dofs_requested_block_one_others[i];
    }
    dofs_requested_block_one_others.resize(total_number_dofs_requested_others,
                                           0);

    // Exchange requested DoF IDs
    MPI_Alltoallv(vec2ptr(dofs_requested_block_one),
                  vec2ptr(num_dofs_requested_block_one),
                  vec2ptr(offsets_dofs_requested_block_one), MPI_INT,
                  vec2ptr(dofs_requested_block_one_others),
                  vec2ptr(num_dofs_requested_block_one_others),
                  vec2ptr(offsets_dofs_requested_block_one_others), MPI_INT,
                  this->comm_);

    // Prepare DoF IDs which others need from me
    for (int k = 0; k < total_number_dofs_requested_others; ++k) {
      assert(maps2b_one_.find(dofs_requested_block_one_others[k]) !=
             maps2b_one_.end());
      dofs_requested_block_one_others[k] =
          maps2b_one_[dofs_requested_block_one_others[k]];
    }

    // Exchange mapped DoF IDs
    MPI_Alltoallv(vec2ptr(dofs_requested_block_one_others),
                  vec2ptr(num_dofs_requested_block_one_others),
                  vec2ptr(offsets_dofs_requested_block_one_others), MPI_INT,
                  vec2ptr(dofs_requested_block_one),
                  vec2ptr(num_dofs_requested_block_one),
                  vec2ptr(offsets_dofs_requested_block_one), MPI_INT,
                  this->comm_);

    // Unpack received data
    for (int i = 0; i < this->nb_procs_; ++i) {
      ghost_dofs_block_one[i].resize(num_dofs_requested_block_one[i], -1);
      for (int j = 0; j < num_dofs_requested_block_one[i]; ++j) {
        ghost_dofs_block_one[i][j] =
            dofs_requested_block_one[offsets_dofs_requested_block_one[i] + j];
      }
    }
  }

  {
    // BLOCK TWO SECOND

    // numbers of DoFs which I request of other processes
    std::vector< int > num_dofs_requested_block_two(this->nb_procs_, 0);
    int total_number_dofs_requested = 0;
    for (int i = 0; i < this->nb_procs_; ++i) {
      num_dofs_requested_block_two[i] = ghost_dofs_two[i].size();
      total_number_dofs_requested += ghost_dofs_two[i].size();
    }

    // numbers of DoFs which other processes request from me
    std::vector< int > num_dofs_requested_block_two_others(this->nb_procs_, 0);

    // Exchange requested numbers of DoFs
    MPI_Alltoall(vec2ptr(num_dofs_requested_block_two), 1, MPI_INT,
                 vec2ptr(num_dofs_requested_block_two_others), 1, MPI_INT,
                 this->comm_);

    // DoF numbers which I need from others
    std::vector< int > dofs_requested_block_two;
    dofs_requested_block_two.reserve(total_number_dofs_requested);

    std::vector< int > offsets_dofs_requested_block_two(this->nb_procs_ + 1, 0);
    for (int i = 0; i < this->nb_procs_; ++i) {
      dofs_requested_block_two.insert(dofs_requested_block_two.end(),
                                      ghost_dofs_two[i].begin(),
                                      ghost_dofs_two[i].end());
      offsets_dofs_requested_block_two[i + 1] =
          offsets_dofs_requested_block_two[i] + ghost_dofs_two[i].size();
    }

    // DoF numbers which others request from me
    std::vector< int > dofs_requested_block_two_others;
    int total_number_dofs_requested_others = 0;

    std::vector< int > offsets_dofs_requested_block_two_others(
        this->nb_procs_ + 1, 0);
    for (int i = 0; i < this->nb_procs_; ++i) {
      offsets_dofs_requested_block_two_others[i + 1] =
          offsets_dofs_requested_block_two_others[i] +
          num_dofs_requested_block_two_others[i];
      total_number_dofs_requested_others +=
          num_dofs_requested_block_two_others[i];
    }
    dofs_requested_block_two_others.resize(total_number_dofs_requested_others,
                                           0);

    // Exchange requested DoF IDs
    MPI_Alltoallv(vec2ptr(dofs_requested_block_two),
                  vec2ptr(num_dofs_requested_block_two),
                  vec2ptr(offsets_dofs_requested_block_two), MPI_INT,
                  vec2ptr(dofs_requested_block_two_others),
                  vec2ptr(num_dofs_requested_block_two_others),
                  vec2ptr(offsets_dofs_requested_block_two_others), MPI_INT,
                  this->comm_);

    // Prepare DoF IDs which others need from me
    for (int k = 0; k < total_number_dofs_requested_others; ++k) {
      assert(maps2b_two_.find(dofs_requested_block_two_others[k]) !=
             maps2b_two_.end());
      dofs_requested_block_two_others[k] =
          maps2b_two_[dofs_requested_block_two_others[k]];
    }

    // Exchange mapped DoF IDs
    MPI_Alltoallv(vec2ptr(dofs_requested_block_two_others),
                  vec2ptr(num_dofs_requested_block_two_others),
                  vec2ptr(offsets_dofs_requested_block_two_others), MPI_INT,
                  vec2ptr(dofs_requested_block_two),
                  vec2ptr(num_dofs_requested_block_two),
                  vec2ptr(offsets_dofs_requested_block_two), MPI_INT,
                  this->comm_);

    // Unpack received data
    for (int i = 0; i < this->nb_procs_; ++i) {
      ghost_dofs_block_two[i].resize(num_dofs_requested_block_two[i], -1);
      for (int j = 0; j < num_dofs_requested_block_two[i]; ++j) {
        ghost_dofs_block_two[i][j] =
            dofs_requested_block_two[offsets_dofs_requested_block_two[i] + j];
      }
    }
  }

  // build maps system->block
  std::map< int, int > maps2b_one_offdiag;
  std::map< int, int > maps2b_two_offdiag;

  for (int p = 0; p < this->nb_procs_; ++p) {
    for (int i = 0, e_i = ghost_dofs_one[p].data().size(); i != e_i; ++i) {
      maps2b_one_offdiag[ghost_dofs_one[p].data()[i]] =
          ghost_dofs_block_one[p][i];
    }

    for (int i = 0, e_i = ghost_dofs_two[p].data().size(); i != e_i; ++i) {
      maps2b_two_offdiag[ghost_dofs_two[p].data()[i]] =
          ghost_dofs_block_two[p][i];
    }
  }

  //*****************************************************************
  // Translate couplings from system to block numbering
  //*****************************************************************
  sparsity_A_.clear();
  sparsity_B_.clear();
  sparsity_C_.clear();
  sparsity_D_.clear();

  sparsity_A_.resize(couplings_A_.size());
  sparsity_B_.resize(couplings_B_.size());
  sparsity_C_.resize(couplings_C_.size());
  sparsity_D_.resize(couplings_D_.size());

  // sub matrix A
  index = 0;
  for (auto it = couplings_A_.begin(),
           e_it = couplings_A_.end();
       it != e_it; ++it) {
    sparsity_A_[index].resize(it->second.size());
    for (int i = 0; i < it->second.size(); ++i) {
      if (maps2b_one_.find(it->second[i]) != maps2b_one_.end()) {
        sparsity_A_[index][i] = maps2b_one_[it->second[i]];
      } else {
        assert(maps2b_one_offdiag.find(it->second[i]) !=
               maps2b_one_offdiag.end());
        sparsity_A_[index][i] = maps2b_one_offdiag[it->second[i]];
      }
    }
    ++index;
  }
  // sub matrix B
  index = 0;
  for (auto it = couplings_B_.begin(),
           e_it = couplings_B_.end();
       it != e_it; ++it) {
    sparsity_B_[index].resize(it->second.size());
    for (int i = 0; i < it->second.size(); ++i) {
      if (maps2b_two_.find(it->second[i]) != maps2b_two_.end()) {
        sparsity_B_[index][i] = maps2b_two_[it->second[i]];
      } else {
        assert(maps2b_two_offdiag.find(it->second[i]) !=
               maps2b_two_offdiag.end());
        sparsity_B_[index][i] = maps2b_two_offdiag[it->second[i]];
      }
    }
    ++index;
  }
  // sub matrix C
  index = 0;
  for (auto it = couplings_C_.begin(),
           e_it = couplings_C_.end();
       it != e_it; ++it) {
    sparsity_C_[index].resize(it->second.size());
    for (int i = 0; i < it->second.size(); ++i) {
      if (maps2b_one_.find(it->second[i]) != maps2b_one_.end()) {
        sparsity_C_[index][i] = maps2b_one_[it->second[i]];
      } else {
        assert(maps2b_one_offdiag.find(it->second[i]) !=
               maps2b_one_offdiag.end());
        sparsity_C_[index][i] = maps2b_one_offdiag[it->second[i]];
      }
    }
    ++index;
  }
  // sub matrix D
  index = 0;
  for (auto it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    sparsity_D_[index].resize(it->second.size());
    for (int i = 0; i < it->second.size(); ++i) {
      if (maps2b_two_.find(it->second[i]) != maps2b_two_.end()) {
        sparsity_D_[index][i] = maps2b_two_[it->second[i]];
      } else {
        assert(maps2b_two_offdiag.find(it->second[i]) !=
               maps2b_two_offdiag.end());
        sparsity_D_[index][i] = maps2b_two_offdiag[it->second[i]];
      }
    }
    ++index;
  }

  //*****************************************************************
  // Setup LaCouplings for different blocks
  //*****************************************************************

  std::vector< int > offdiag_offsets_block_one(this->nb_procs_ + 1, 0);
  std::vector< int > offdiag_cols_block_one;

  for (size_t i = 0; i < ghost_dofs_block_one.size(); ++i) {
    offdiag_cols_block_one.insert(offdiag_cols_block_one.end(),
                                  ghost_dofs_block_one[i].begin(),
                                  ghost_dofs_block_one[i].end());
    offdiag_offsets_block_one[i + 1] =
        offdiag_offsets_block_one[i] + ghost_dofs_block_one[i].size();
  }

  std::vector< int > offdiag_offsets_block_two(this->nb_procs_ + 1, 0);
  std::vector< int > offdiag_cols_block_two;

  for (size_t i = 0; i < ghost_dofs_block_two.size(); ++i) {
    offdiag_cols_block_two.insert(offdiag_cols_block_two.end(),
                                  ghost_dofs_block_two[i].begin(),
                                  ghost_dofs_block_two[i].end());
    offdiag_offsets_block_two[i + 1] =
        offdiag_offsets_block_two[i] + ghost_dofs_block_two[i].size();
  }

  la_c_one_.Init(this->comm_);
  la_c_one_.InitializeCouplings(offsets_block_one_, offdiag_cols_block_one,
                                offdiag_offsets_block_one);
  la_c_two_.Init(this->comm_);
  la_c_two_.InitializeCouplings(offsets_block_two_, offdiag_cols_block_two,
                                offdiag_offsets_block_two);

  //*****************************************************************
  // Initialize matrices and vectors
  //*****************************************************************
  A_.Init(this->comm_, la_c_one_, la_c_one_);
  B_.Init(this->comm_, la_c_one_, la_c_two_);
  C_.Init(this->comm_, la_c_two_, la_c_one_);
  D_.Init(this->comm_, la_c_two_, la_c_two_);
  if (this->precond_two_type_ == PCDPrecond) {
    Q_.Init(this->comm_, la_c_two_, la_c_two_);
    F_.Init(this->comm_, la_c_two_, la_c_two_);
    H_.Init(this->comm_, la_c_two_, la_c_two_);
  }
  if (this->dpx_precond_type_ == DPX_DPX) {
    block_two_matrix_.Init(this->comm_, la_c_two_, la_c_two_);
    precond_two_.Init(this->comm_, la_c_two_, la_c_two_);
  }
  if (this->dpx_precond_type_ == DPX_SIMPLE) {
    dInvAB_.Init(this->comm_, la_c_one_, la_c_two_);
  }

  f_.Init(this->comm_, la_c_one_);
  g_.Init(this->comm_, la_c_two_);
  x_.Init(this->comm_, la_c_one_);
  y_.Init(this->comm_, la_c_two_);

  h_one_1_.Init(this->comm_, la_c_one_);
  rhs_x_.Init(this->comm_, la_c_one_);
  h1_.Init(this->comm_, la_c_one_);
  h2_.Init(this->comm_, la_c_one_);
  h_two_1_.Init(this->comm_, la_c_two_);
  w_.Init(this->comm_, la_c_two_);
  h3_.Init(this->comm_, la_c_two_);

  //*****************************************************************
  // Initialize structures of matrices
  //*****************************************************************
  std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;

  // sub matrix A
  for (int i = 0; i < sparsity_A_.size(); ++i) {
    for (int j = 0; j < sparsity_A_[i].size(); ++j) {
      if ((sparsity_A_[i][j] >= offsets_block_one_[this->my_rank_]) &&
          (sparsity_A_[i][j] < offsets_block_one_[this->my_rank_ + 1])) {
        rows_diag.push_back(offsets_block_one_[this->my_rank_] + i);
        cols_diag.push_back(sparsity_A_[i][j]);
      } else {
        rows_offdiag.push_back(offsets_block_one_[this->my_rank_] + i);
        cols_offdiag.push_back(sparsity_A_[i][j]);
      }
    }
  }

  A_.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                   vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                   cols_offdiag.size());

  // sub matrix B
  rows_diag.clear();
  cols_diag.clear();
  rows_offdiag.clear();
  cols_offdiag.clear();
  for (int i = 0; i < sparsity_B_.size(); ++i) {
    for (int j = 0; j < sparsity_B_[i].size(); ++j) {
      if ((sparsity_B_[i][j] >= offsets_block_two_[this->my_rank_]) &&
          (sparsity_B_[i][j] < offsets_block_two_[this->my_rank_ + 1])) {
        rows_diag.push_back(offsets_block_one_[this->my_rank_] + i);
        cols_diag.push_back(sparsity_B_[i][j]);
      } else {
        rows_offdiag.push_back(offsets_block_one_[this->my_rank_] + i);
        cols_offdiag.push_back(sparsity_B_[i][j]);
      }
    }
  }

  B_.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                   vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                   cols_offdiag.size());

  if (this->dpx_precond_type_ == DPX_SIMPLE) {
    dInvAB_.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag),
                          cols_diag.size(), vec2ptr(rows_offdiag),
                          vec2ptr(cols_offdiag), cols_offdiag.size());
  }

  // sub matrix C
  rows_diag.clear();
  cols_diag.clear();
  rows_offdiag.clear();
  cols_offdiag.clear();
  for (int i = 0; i < sparsity_C_.size(); ++i) {
    for (int j = 0; j < sparsity_C_[i].size(); ++j) {
      if ((sparsity_C_[i][j] >= offsets_block_one_[this->my_rank_]) &&
          (sparsity_C_[i][j] < offsets_block_one_[this->my_rank_ + 1])) {
        rows_diag.push_back(offsets_block_two_[this->my_rank_] + i);
        cols_diag.push_back(sparsity_C_[i][j]);
      } else {
        rows_offdiag.push_back(offsets_block_two_[this->my_rank_] + i);
        cols_offdiag.push_back(sparsity_C_[i][j]);
      }
    }
  }

  C_.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                   vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                   cols_offdiag.size());

  // sub matrix D
  rows_diag.clear();
  cols_diag.clear();
  rows_offdiag.clear();
  cols_offdiag.clear();
  for (int i = 0; i < sparsity_D_.size(); ++i) {
    for (int j = 0; j < sparsity_D_[i].size(); ++j) {
      if ((sparsity_D_[i][j] >= offsets_block_two_[this->my_rank_]) &&
          (sparsity_D_[i][j] < offsets_block_two_[this->my_rank_ + 1])) {
        rows_diag.push_back(offsets_block_two_[this->my_rank_] + i);
        cols_diag.push_back(sparsity_D_[i][j]);
      } else {
        rows_offdiag.push_back(offsets_block_two_[this->my_rank_] + i);
        cols_offdiag.push_back(sparsity_D_[i][j]);
      }
    }
  }

  D_.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                   vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                   cols_offdiag.size());

  if (this->dpx_precond_type_ == DPX_DPX) {
    this->precond_two_.InitStructure(
        vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
        vec2ptr(rows_offdiag), vec2ptr(cols_offdiag), cols_offdiag.size());
  }

  //*****************************************************************
  // Create block index sets
  //*****************************************************************
  this->indexset_one_.resize(this->mapb2s_one_.size());
  for (size_t i = 0, e_i = this->indexset_one_.size(); i != e_i; ++i) {
    this->indexset_one_[i] = i + offsets_block_one_[this->my_rank_];
  }

  this->indexset_two_.resize(this->mapb2s_two_.size());
  for (size_t i = 0, e_i = this->indexset_two_.size(); i != e_i; ++i) {
    this->indexset_two_[i] = i + offsets_block_two_[this->my_rank_];
  }

  this->val_temp_one_.resize(this->mapb2s_one_.size());
  this->val_temp_two_.resize(this->mapb2s_two_.size());
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupBlockTwoMatrix(const OperatorType &op) {
  if (this->dpx_precond_type_ != DPX_DPX) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Schur complement Solver is not of type DPX_DPX!!!");
    quit_program();
  }

  //*****************************************************************
  // Operator setup means copying entries from the system matrix op to
  // the submatrix block_two_matrix_
  //*****************************************************************

  if (!this->block_two_matrix_is_initialized_) {
    std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;
    for (int i = 0; i < sparsity_D_.size(); ++i) {
      for (int j = 0; j < sparsity_D_[i].size(); ++j) {
        if ((sparsity_D_[i][j] >= offsets_block_two_[this->my_rank_]) &&
            (sparsity_D_[i][j] < offsets_block_two_[this->my_rank_ + 1])) {
          rows_diag.push_back(offsets_block_two_[this->my_rank_] + i);
          cols_diag.push_back(sparsity_D_[i][j]);
        } else {
          rows_offdiag.push_back(offsets_block_two_[this->my_rank_] + i);
          cols_offdiag.push_back(sparsity_D_[i][j]);
        }
      }
    }

    this->block_two_matrix_.InitStructure(
        vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
        vec2ptr(rows_offdiag), vec2ptr(cols_offdiag), cols_offdiag.size());
    this->block_two_matrix_is_initialized_ = true;
  }

  // this->block_two_matrix_.Zeros ( );
  // submatrix matrix M
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< ValueType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    for (int i = 0, e_i = vals.size(); i != e_i; ++i) {
      vals[i] *= this->scaling_block_two_matrix_;
    }

    // Set values in submatrix M
    int block_index_system =
        row_index_block + offsets_block_two_[this->my_rank_];
    this->block_two_matrix_.SetValues(
        &block_index_system, 1, vec2ptr(this->sparsity_D_[row_index_block]),
        it->second.size(), vec2ptr(vals));
    ++row_index_block;
  }
  this->block_two_matrix_is_set_ = true;
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupOperator(OperatorType &op) {
  Timer timer;
  if (this->info_ != nullptr) {
    timer.reset();
    timer.start();
  }
  //*****************************************************************
  // Operator setup means copying entries from system matrix op to
  // submatrices A, B, C and D
  //*****************************************************************

  // A_.Zeros ( );
  // B_.Zeros ( );
  // C_.Zeros ( );
  // D_.Zeros ( );
  // precond_two_.Zeros ( );

  // sub matrix A
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_A_.begin(),
           e_it = couplings_A_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< ValueType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix A
    int block_index_system =
        row_index_block + offsets_block_one_[this->my_rank_];
    A_.SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_A_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }
  this->A_modified_ = true;
  this->A_passed2solver_ = false;

  // sub matrix B
  row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_B_.begin(),
           e_it = couplings_B_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< ValueType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix B
    int block_index_system =
        row_index_block + offsets_block_one_[this->my_rank_];
    B_.SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_B_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }

  // sub matrix C
  row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_C_.begin(),
           e_it = couplings_C_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< ValueType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix C
    int block_index_system =
        row_index_block + offsets_block_two_[this->my_rank_];
    C_.SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_C_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }

  // sub matrix D and precond_two
  row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< ValueType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix D
    int block_index_system =
        row_index_block + offsets_block_two_[this->my_rank_];
    D_.SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_D_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    if (this->dpx_precond_type_ == DPX_DPX) {
      precond_two_.SetValues(&block_index_system, 1,
                             vec2ptr(this->sparsity_D_[row_index_block]),
                             it->second.size(), vec2ptr(vals));
    }
    ++row_index_block;
  }
  this->D_modified_ = true;
  this->D_passed2solver_ = false;

  if (this->dpx_precond_type_ == DPX_DPX) {
    this->precond_two_modified_ = true;
    this->precond_two_passed2solver_ = false;
  }

  // Add block_two_matrix_ to precond_two_ if present
  if (this->block_two_matrix_is_initialized_ &&
      this->block_two_matrix_is_set_ && this->dpx_precond_type_ == DPX_DPX) {
    row_index_block = 0;

    for (typename std::map< int, SortedArray< int > >::const_iterator
             it = couplings_D_.begin(),
             e_it = couplings_D_.end();
         it != e_it; ++it) {
      // Get values from system matrix block_two_matrix_
      int block_index_system =
          row_index_block + offsets_block_two_[this->my_rank_];
      std::vector< ValueType > vals(it->second.size(), 0.);
      block_two_matrix_.GetValues(&block_index_system, 1,
                                  vec2ptr(this->sparsity_D_[row_index_block]),
                                  it->second.size(), vec2ptr(vals));

      // Set values in block matrix precond_two
      precond_two_.Add(&block_index_system, 1,
                       vec2ptr(this->sparsity_D_[row_index_block]),
                       it->second.size(), vec2ptr(vals));
      ++row_index_block;
    }
  }

  //*****************************************************************
  // Initialize diag_A_
  //*****************************************************************
  if (this->is_simple_precond_ || this->dpx_precond_type_ == DPX_SIMPLE) {
    diag_A_.clear();
    diag_A_.resize(sparsity_A_.size(), 0.);

    const int offset_A = this->offsets_block_one_[this->my_rank_];
    for (size_t i = 0, e_i = diag_A_.size(); i != e_i; ++i) {
      int diag_num = offset_A + i;
      A_.GetValues(&diag_num, 1, &diag_num, 1, &(diag_A_[i]));
    }
  }

  if (this->dpx_precond_type_ == DPX_SIMPLE) {
    int row_index_block = 0;
    for (typename std::map< int, SortedArray< int > >::const_iterator
             it = couplings_B_.begin(),
             e_it = couplings_B_.end();
         it != e_it; ++it) {
      // Get values from system matrix op
      int block_index_system =
          row_index_block + offsets_block_one_[this->my_rank_];
      std::vector< ValueType > vals(it->second.size(), 0.);
      B_.GetValues(&block_index_system, 1,
                   vec2ptr(this->sparsity_B_[row_index_block]),
                   it->second.size(), vec2ptr(vals));

      for (int i = 0; i < vals.size(); ++i) {
        vals[i] /= -diag_A_[row_index_block];
      }

      // Set values in block matrix B
      dInvAB_.SetValues(&block_index_system, 1,
                        vec2ptr(this->sparsity_B_[row_index_block]),
                        it->second.size(), vec2ptr(vals));
      ++row_index_block;
    }

    Timer timerMM;
    if (this->info_ != nullptr) {
      timerMM.reset();
      timerMM.start();
    }

    precond_two_.MatrixMult(C_, dInvAB_);

    if (this->info_ != nullptr) {
      timerMM.stop();
      this->info_->add("MatrixMultWCT", timerMM.get_duration());
    }

    // sub matrix D and precond_two
    row_index_block = 0;
    for (typename std::map< int, SortedArray< int > >::const_iterator
             it = couplings_D_.begin(),
             e_it = couplings_D_.end();
         it != e_it; ++it) {
      // Get values from block D matrix
      int block_index_system =
          row_index_block + offsets_block_two_[this->my_rank_];
      std::vector< ValueType > vals(it->second.size(), 0.);
      D_.GetValues(&block_index_system, 1,
                   vec2ptr(this->sparsity_D_[row_index_block]),
                   it->second.size(), vec2ptr(vals));

      // Set values in block matrix precond_two
      precond_two_.Add(&block_index_system, 1,
                       vec2ptr(this->sparsity_D_[row_index_block]),
                       it->second.size(), vec2ptr(vals));
      ++row_index_block;
    }

    this->precond_two_modified_ = true;
    this->precond_two_passed2solver_ = false;
  }

  //*****************************************************************
  // Ensure consistency with general preconditioner data structure
  //*****************************************************************
  this->op_ = &op;
  this->SetModifiedOperator(true);
  this->SetUseSolverOperator(false);

  if (this->info_ != nullptr) {
    timer.stop();
    this->info_->add("SetupOpWCT", timer.get_duration());
  }
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupOperatorQ(const OperatorType &op) {
  if (this->precond_two_type_ != PCDPrecond) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Schur complement Solver is not of type PCDPrecond!!!");
    quit_program();
  }

  if (!this->Q_is_initialized_) {
    std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;
    for (int i = 0; i < sparsity_D_.size(); ++i) {
      for (int j = 0; j < sparsity_D_[i].size(); ++j) {
        if ((sparsity_D_[i][j] >= offsets_block_two_[this->my_rank_]) &&
            (sparsity_D_[i][j] < offsets_block_two_[this->my_rank_ + 1])) {
          rows_diag.push_back(offsets_block_two_[this->my_rank_] + i);
          cols_diag.push_back(sparsity_D_[i][j]);
        } else {
          rows_offdiag.push_back(offsets_block_two_[this->my_rank_] + i);
          cols_offdiag.push_back(sparsity_D_[i][j]);
        }
      }
    }

    Q_.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                     vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                     cols_offdiag.size());
    this->Q_is_initialized_ = true;
  }

  // Q_.Zeros ( );
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< ValueType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix Q
    int block_index_system =
        row_index_block + offsets_block_two_[this->my_rank_];
    Q_.SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_D_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }
  this->Q_is_set_ = true;
  this->Q_modified_ = true;
  this->Q_passed2solver_ = false;

  //*****************************************************************
  // Ensure consistency with general preconditioner data structure
  //*****************************************************************
  this->SetState(false);
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupOperatorF(const OperatorType &op) {
  if (this->precond_two_type_ != PCDPrecond) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Schur complement Solver is not of type PCDPrecond!!!");
    quit_program();
  }

  if (!this->F_is_initialized_) {
    std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;
    for (int i = 0; i < sparsity_D_.size(); ++i) {
      for (int j = 0; j < sparsity_D_[i].size(); ++j) {
        if ((sparsity_D_[i][j] >= offsets_block_two_[this->my_rank_]) &&
            (sparsity_D_[i][j] < offsets_block_two_[this->my_rank_ + 1])) {
          rows_diag.push_back(offsets_block_two_[this->my_rank_] + i);
          cols_diag.push_back(sparsity_D_[i][j]);
        } else {
          rows_offdiag.push_back(offsets_block_two_[this->my_rank_] + i);
          cols_offdiag.push_back(sparsity_D_[i][j]);
        }
      }
    }

    F_.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                     vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                     cols_offdiag.size());
    this->F_is_initialized_ = true;
  }

  // F_.Zeros ( );
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< ValueType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix F
    int block_index_system =
        row_index_block + offsets_block_two_[this->my_rank_];
    F_.SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_D_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }
  this->F_is_set_ = true;
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupOperatorH(const OperatorType &op) {
  if (this->precond_two_type_ != PCDPrecond) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Schur complement Solver is not of type PCDPrecond!!!");
    quit_program();
  }

  if (!this->H_is_initialized_) {
    std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;
    for (int i = 0; i < sparsity_D_.size(); ++i) {
      for (int j = 0; j < sparsity_D_[i].size(); ++j) {
        if ((sparsity_D_[i][j] >= offsets_block_two_[this->my_rank_]) &&
            (sparsity_D_[i][j] < offsets_block_two_[this->my_rank_ + 1])) {
          rows_diag.push_back(offsets_block_two_[this->my_rank_] + i);
          cols_diag.push_back(sparsity_D_[i][j]);
        } else {
          rows_offdiag.push_back(offsets_block_two_[this->my_rank_] + i);
          cols_offdiag.push_back(sparsity_D_[i][j]);
        }
      }
    }

    H_.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                     vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                     cols_offdiag.size());
    this->H_is_initialized_ = true;
  }

  // H_.Zeros ( );
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< ValueType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix H
    int block_index_system =
        row_index_block + offsets_block_two_[this->my_rank_];
    H_.SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_D_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }
  this->H_is_set_ = true;
  this->H_modified_ = true;
  this->H_passed2solver_ = false;

  //*****************************************************************
  // Ensure consistency with general preconditioner data structure
  //*****************************************************************
  this->SetState(false);
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupPreconditioner(
    Preconditioner< LAD > &precond) {
  if (this->precond_two_type_ != externalPrecond) {
    LOG_ERROR(
        "[" << this->nested_level_
            << "] Schur complement Solver is not of type externalPrecond!!!");
    quit_program();
  }

  this->precond_ = &precond;
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupSolverBlockA(LinearSolver< LAD > &solver_A) {
  this->solver_A_ = &solver_A;
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupSolverQ(LinearSolver< LAD > &solver) {
  if (this->precond_two_type_ != PCDPrecond) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Schur complement Solver is not of type PCDPrecond!!!");
    quit_program();
  }

  this->solver_Q_ = &solver;
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupSolverH(LinearSolver< LAD > &solver) {
  if (this->precond_two_type_ != PCDPrecond) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Schur complement Solver is not of type PCDPrecond!!!");
    quit_program();
  }

  this->solver_H_ = &solver;
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupSolverD(LinearSolver< LAD > &solver) {
  if (this->precond_two_type_ != PCDPrecond) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Schur complement Solver is not of type PCDPrecond!!!");
    quit_program();
  }

  this->solver_D_ = &solver;
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SetupSolverPrecondBlockTwo(
    LinearSolver< LAD > &solver_precond_block_two) {
  if (this->precond_two_type_ != DPXPrecond) {
    LOG_ERROR("[" << this->nested_level_
                  << "] Schur complement Solver is not of type DPXPrecond!!!");
    quit_program();
  }

  this->solver_precond_block_two_ = &solver_precond_block_two;
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::BuildImpl(VectorType const *b, VectorType *x) {

  if (this->solver_A_ != nullptr) {
    if (this->A_modified_ || !(this->A_passed2solver_)) {
      if (this->solver_A_->GetPreconditioner() != nullptr) {
        this->solver_A_->GetPreconditioner()->SetupOperator(this->A_);
      }
      this->solver_A_->SetupOperator(this->A_);
      this->solver_A_->Build(b, x);

      this->A_modified_ = false;
      this->A_passed2solver_ = true;
    }
  } else {
    LOG_ERROR("[" << this->nested_level_
                  << "] No solver for submatrix A defined!!!");
    quit_program();
  }

  // Setup operators for solver objects
  if (precond_two_type_ == externalPrecond) {
    if (this->precond_ == nullptr) {
      LOG_ERROR("[" << this->nested_level_ << "] No precond_ defined!!!");
      quit_program();
    }
    this->precond_->Build(b, x);
  } else if (precond_two_type_ == DPXPrecond) {
    if (this->solver_precond_block_two_ != nullptr) {
      if (this->precond_two_modified_ || !(this->precond_two_passed2solver_)) {
        if (this->solver_precond_block_two_->GetPreconditioner() != nullptr) {
          if (dpx_precond_type_ == DPX_D) {
            this->solver_precond_block_two_->GetPreconditioner()->SetupOperator(
                this->D_);
          } else {
            this->solver_precond_block_two_->GetPreconditioner()->SetupOperator(
                this->precond_two_);
          }
        }
        if (dpx_precond_type_ == DPX_D) {
          this->solver_precond_block_two_->SetupOperator(this->D_);
        } else {
          this->solver_precond_block_two_->SetupOperator(this->precond_two_);
        }
        this->solver_precond_block_two_->Build(b, x);

        this->precond_two_modified_ = false;
        this->precond_two_passed2solver_ = true;
      }
    } else {
      LOG_ERROR("[" << this->nested_level_
                    << "] No solver for matrix precond_two_ defined!!!");
      quit_program();
    }
  } else if (precond_two_type_ == PCDPrecond) {
    if (this->scaling_S_ != 0.0) {
      if (this->solver_Q_ != nullptr) {
        if (this->Q_modified_ || !(this->Q_passed2solver_)) {
          if (this->solver_Q_->GetPreconditioner() != nullptr) {
            this->solver_Q_->GetPreconditioner()->SetupOperator(this->Q_);
          }
          this->solver_Q_->SetupOperator(this->Q_);
          this->solver_Q_->Build(b, x);
          this->Q_modified_ = false;
          this->Q_passed2solver_ = true;
        }
      } else {
        LOG_ERROR("[" << this->nested_level_
                      << "] No solver for operator Q defined!!!");
        quit_program();
      }

      if (this->solver_H_ != nullptr) {
        if (this->H_modified_ || !(this->H_passed2solver_)) {
          if (this->solver_H_->GetPreconditioner() != nullptr) {
            this->solver_H_->GetPreconditioner()->SetupOperator(this->H_);
          }
          this->solver_H_->SetupOperator(this->H_);
          this->solver_H_->Build(b, x);
          this->H_modified_ = false;
          this->H_passed2solver_ = true;
        }
      } else {
        LOG_ERROR("[" << this->nested_level_
                      << "] No solver for operator H defined !!!");
        quit_program();
      }

      if (!this->Q_is_set_) {
        LOG_ERROR("[" << this->nested_level_ << "] No operator Q defined!!!");
        quit_program();
      }
      if (!this->F_is_set_) {
        LOG_ERROR("[" << this->nested_level_ << "] No operator F defined!!!");
        quit_program();
      }
      if (!this->H_is_set_) {
        LOG_ERROR("[" << this->nested_level_ << "] No operator H defined!!!");
        quit_program();
      }
    }
    if (this->solver_D_ != nullptr) {
      if (this->D_modified_ || !(this->D_passed2solver_)) {
        if (this->solver_D_->GetPreconditioner() != nullptr) {
          this->solver_D_->GetPreconditioner()->SetupOperator(this->D_);
        }
        this->solver_D_->SetupOperator(this->D_);
        this->solver_D_->Build(b, x);
        this->D_modified_ = false;
        this->D_passed2solver_ = true;
      }
    } else {
      if (this->scaling_D_ != 0.0) {
        LOG_ERROR("[" << this->nested_level_
                      << "] No solver for operator D defined !!!");
        quit_program();
      }
    }
  }
}

template < class LAD, int DIM >
LinearSolverState SchurComplement< LAD, DIM >::SolveImpl(const VectorType &b,
                                                    VectorType *x) {
  Timer timer;
  if (this->info_ != nullptr) {
    timer.reset();
    timer.start();
  }

  x_.Zeros();
  y_.Zeros();

  // Objects for temporary data
  VectorType rhs_temp, sol_temp;
  rhs_temp.CloneFromWithoutContent(b);
  sol_temp.CloneFromWithoutContent(b);

  //*****************************************************************
  // Get block parts of vector b
  //*****************************************************************
  b.GetValues(vec2ptr(this->mapb2s_one_), this->mapb2s_one_.size(),
              vec2ptr(this->val_temp_one_));
  f_.SetValues(vec2ptr(this->indexset_one_), this->indexset_one_.size(),
               vec2ptr(this->val_temp_one_));

  b.GetValues(vec2ptr(this->mapb2s_two_), this->mapb2s_two_.size(),
              vec2ptr(this->val_temp_two_));
  g_.SetValues(vec2ptr(this->indexset_two_), this->indexset_two_.size(),
               vec2ptr(this->val_temp_two_));

  //*****************************************************************
  // Prepare right hand side for y block: g <- g - C*A^{-1}*f
  //*****************************************************************
  // this->h_one_1_ = A^{-1}*f
  this->h_one_1_.Zeros();
  // pass on information object
  if (this->info_ != nullptr) {
    this->info_->add("solverA1");
    this->solver_A_->SetInfo(this->info_->get_child("solverA1"));
  }
  this->SolveBlockA(f_, this->h_one_1_);

  // this->h_two_1_ = C*this->h_one_1_
  /*C_.VectorMult ( this->h_one_1_, &this->h_two_1_ );

  // g <- g - this->h_two_1_
  g_.Axpy ( this->h_two_1_, static_cast < ValueType > ( -1. ) );*/

  C_.VectorMultAdd(-1.0, this->h_one_1_, 1.0, &(this->g_));

  //*****************************************************************
  // FGMRES in y_ block, solving the Schur complement equation
  //*****************************************************************
  IterateControl::State conv = IterateControl::kIterate;

  // compute really used basis size as minimum of maximum iterations and
  // given basis size
  const int basis_size_actual = std::min(10000, this->control().maxits());

  // Hessenberg matrix
  SeqDenseMatrix< ValueType > H;
  H.Resize(basis_size_actual, basis_size_actual + 1);

  // Allocate array of pointer for Krylov subspace basis
  if (this->V_.size() != basis_size_actual + 1) {
    for (size_t i = 0; i != this->V_.size(); ++i) {
      delete this->V_[i];
    }
    this->V_.clear();

    this->V_.resize(basis_size_actual + 1, nullptr);
    for (size_t i = 0; i != this->V_.size(); ++i) {
      this->V_[i] = new VectorType;
      this->V_[i]->Init(this->comm_, this->la_c_two_);
    }
  }

  // preconditioned Krylov subspace basis
  if (this->Z_.size() != basis_size_actual + 1) {
    for (size_t i = 0; i != this->Z_.size(); ++i) {
      delete this->Z_[i];
    }
    this->Z_.clear();

    this->Z_.resize(basis_size_actual + 1, nullptr);
    for (size_t i = 0; i != this->V_.size(); ++i) {
      this->Z_[i] = new VectorType;
      this->Z_[i]->Init(this->comm_, this->la_c_two_);
    }
  }

  std::vector< ValueType > g(basis_size_actual + 1,
                             0.); // rhs of least squares problem
  std::vector< ValueType > cs(basis_size_actual + 1, 0.); // Givens rotations
  std::vector< ValueType > sn(basis_size_actual + 1, 0.); // Givens rotations

  int iter = 0;

  // compute initial residual norm: y_=0, therefore residual equals
  /// right hand side g_
  V_[0]->CopyFrom(g_);

  this->FilterVector(*(V_[0]), rhs_temp);

  this->res_ = V_[0]->Norm2();
  conv = this->control().Check(iter, this->res_);
  switch (this->print_level_) {
  case 0: {
    break;
  }
  case 2: {
    LOG_INFO("[" << this->nested_level_ << "] FGMRES (Schur complement solver)",
             " with right preconditioning");
    LOG_INFO(
        "[" << this->nested_level_
            << "] FGMRES (Schur complement solver) starts with residual norm",
        this->res_);
    break;
  }
  case 3: {
    LOG_INFO(
        "[" << this->nested_level_
            << "] FGMRES (Schur complement solver) starts with residual norm ",
        this->res_);
    break;
  }
  }

  Timer timerS;
  if (this->info_ != nullptr) {
    timerS.reset();
    timerS.start();
  }
  // main loop
  while (conv == IterateControl::kIterate) {
    g.assign(g.size(), static_cast< ValueType >(0.)); // g = 0
    H.Zeros();

    assert(this->res_ != static_cast< ValueType >(0.));
    V_[0]->Scale(static_cast< ValueType >(1.) / this->res_); // norm residual
    g[0] = this->res_;

    for (int j = 0; j != basis_size_actual && conv == IterateControl::kIterate;
         ++j) {
      ++iter;
      this->iter_ = iter;
      Z_[j]->Zeros();

      // Apply preconditioner
      this->SchurPrecond(*(V_[j]), *Z_[j], rhs_temp, sol_temp);

      this->FilterVector(*(Z_[j]), rhs_temp);

      // w.Zeros ( );
      this->SchurVectorMult(*(Z_[j]), this->w_); // w = S Z[j]

      this->FilterVector(this->w_, rhs_temp);

      // -- start building Hessenberg matrix H --
      // vectors in V are ONB of Krylov subspace K_i(A,V[0])
      for (int i = 0; i <= j; ++i) {
        H(j, i) = this->w_.Dot(*(V_[i]));
        this->w_.Axpy(*(V_[i]), static_cast< ValueType >(-1.) * H(j, i));
      }

      H(j, j + 1) = this->w_.Norm2();
      assert(H(j, j + 1) != static_cast< ValueType >(0.));

      this->w_.Scale(static_cast< ValueType >(1.) / H(j, j + 1));

      V_[j + 1]->CopyFrom(this->w_);
      // V[j + 1]->Update( );
      // -- end building Hessenberg matrix H --

      // apply old Givens rotation on old H entries
      for (int k = 0; k < j; ++k)
        this->ApplyPlaneRotation(cs[k], sn[k], &H(j, k), &H(j, k + 1));

      // determine new Givens rotation for actual iteration i
      this->GeneratePlaneRotation(H(j, j), H(j, j + 1), &cs[j], &sn[j]);

      // apply Givens rotation on new H element
      this->ApplyPlaneRotation(cs[j], sn[j], &H(j, j), &H(j, j + 1));

      // update g for next dimension -> g[j+1] is norm of actual residual
      this->ApplyPlaneRotation(cs[j], sn[j], &g[j], &g[j + 1]);

      this->res_ = std::abs(g[j + 1]);
      switch (this->print_level_) {
      case 0: {
        break;
      }
      case 2: {
        LOG_INFO(
            "[" << this->nested_level_
                << "] FGMRES (Schur complement solver) residual (iteration "
                << iter << ")     ",
            this->res_);
      }
      }
      conv = this->control().Check(iter, this->res_);

      switch (conv) {
      case IterateControl::kIterate: {
        break;
      }
      default: {
        this->UpdateSolution(H, g, j, &y_); // x = x + Zy
        break;
      }
      }
    } // for (int j = 0; j < this->size_basis(); ++j)

    // setup for restart
    switch (conv) {
    case IterateControl::kIterate: {
      this->UpdateSolution(H, g, basis_size_actual - 1, &y_); // x = x + Zy

      this->SchurVectorMult(y_, *(V_[0]));
      V_[0]->Scale(static_cast< ValueType >(-1.));
      V_[0]->Axpy(g_, static_cast< ValueType >(1.));

      this->FilterVector(*(V_[0]), rhs_temp);

      break;
    }
    default: { break; }
    }
  } // while (conv == IterateControl::kIterate)
  if (this->info_ != nullptr)
    timerS.stop();

  switch (this->print_level_) {
  case 0: {
    break;
  }
  case 2: {
    LOG_INFO("[" << this->nested_level_ << "] FGMRES (Schur complement)",
             " with right preconditioning ended after " << iter
                                                        << " iterations ");
    LOG_INFO("[" << this->nested_level_ << "] FGMRES (Schur complement)",
             " with residual norm " << this->res_);
    break;
  }
  }

  //*****************************************************************
  // Solve A*x = f - B*y
  //*****************************************************************
  /*B_.VectorMult ( y_, &( this->rhs_x_ ) );
  f_.Axpy ( this->rhs_x_, static_cast < ValueType > ( -1. ) );*/
  B_.VectorMultAdd(-1.0, y_, 1.0, &(this->f_));

  if (this->info_ != nullptr) {
    this->info_->add("solverA2");
    this->solver_A_->SetInfo(this->info_->get_child("solverA2"));
  }
  this->SolveBlockA(f_, x_);

  //*****************************************************************
  // Copy block results x_ and y_ to system solution vector x
  //*****************************************************************
  x_.GetValues(vec2ptr(this->indexset_one_), this->indexset_one_.size(),
               vec2ptr(this->val_temp_one_));
  x->SetValues(vec2ptr(this->mapb2s_one_), this->mapb2s_one_.size(),
               vec2ptr(this->val_temp_one_));

  y_.GetValues(vec2ptr(this->indexset_two_), this->indexset_two_.size(),
               vec2ptr(this->val_temp_two_));
  x->SetValues(vec2ptr(this->mapb2s_two_), this->mapb2s_two_.size(),
               vec2ptr(this->val_temp_two_));

  if (this->info_ != nullptr) {
    timer.stop();
    this->info_->add("Schur_iter", iter);
    this->info_->add("total_time", timer.get_duration());
    this->info_->add("Schur_time", timerS.get_duration());
  }

  // Cleanup
  rhs_temp.Clear();
  sol_temp.Clear();

  g.clear();
  cs.clear();
  sn.clear();

  this->DestroySolver();

  switch (conv) {
  case IterateControl::kFailureDivergenceTol: {
    return kSolverExceeded;
    break;
  }
  case IterateControl::kFailureMaxitsExceeded: {
    return kSolverExceeded;
    break;
  }
  default: {
    return kSolverSuccess;
    break;
  }
  }
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SchurVectorMult(VectorType &x, VectorType &r) {
  //*****************************************************************
  // Compute r = D*y
  //*****************************************************************
  D_.VectorMult(x, &r);

  //*****************************************************************
  // Compute C*A^{-1}*B*y
  //*****************************************************************
  // h1 = B*y
  B_.VectorMult(x, &(this->h1_));

  // h2 = A^{-1}h1
  this->h2_.Zeros();
  if (this->is_simple_precond_) {
    this->SolveSimpleBlockA(this->h1_, this->h2_);
  } else {
    this->SolveBlockA(this->h1_, this->h2_);
  }

  // h3 = C*h2
  /*C_.VectorMult ( this->h2_, &( this->h3_ ) );

  // r = r - h3
  r.Axpy ( this->h3_, static_cast < ValueType > ( -1. ) );*/

  // r -= C * h2
  C_.VectorMultAdd(-1.0, this->h2_, 1.0, &r);
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SchurPrecond(const VectorType &b, VectorType &x,
                                          VectorType &rhs_temp,
                                          VectorType &sol_temp) {
  Timer timer;

  if (precond_two_type_ == DPXPrecond) {
    // pass on information object
    if (this->info_ != nullptr) {
      std::stringstream pre_str;
      pre_str << "precond"
              << std::setw(1 + std::log10(this->control().maxits()))
              << std::setfill('0') << this->iter_;
      this->info_->add(pre_str.str());
      this->solver_precond_block_two_->SetInfo(
          this->info_->get_child(pre_str.str()));
    }

    timer.reset();
    timer.start();
    this->solver_precond_block_two_->Solve(b, &x);
    timer.stop();
    switch (this->print_level_) {
    case 0: {
      break;
    }
    case 1: {
      break;
    }
    case 2: {
      LOG_INFO("[" << this->nested_level_
                   << "] Iterations preconditioner/solver (block two)",
               this->solver_precond_block_two_->iter());
      LOG_INFO(
          "[" << this->nested_level_
              << "] Final relative residual preconditioner/solver (block two)",
          this->solver_precond_block_two_->res());
      break;
    }
    default: {
      LOG_INFO("[" << this->nested_level_
                   << "] preconditioner/solver (block two): CPU time           "
                      "     ",
               timer.get_duration());
      LOG_INFO("[" << this->nested_level_
                   << "] preconditioner/solver (block two): Iter               "
                      "     ",
               this->solver_precond_block_two_->iter());
      LOG_INFO("[" << this->nested_level_
                   << "] preconditioner/solver (block two): final rel. res.    "
                      "     ",
               this->solver_precond_block_two_->res());
      break;
    }
    }
  } else if (precond_two_type_ == PCDPrecond) {
    timer.reset();
    timer.start();
    this->ApplyPressureConvDiff(b, &x);
    timer.stop();

    if (this->print_level_ > 2) {
      LOG_INFO("[" << this->nested_level_
                   << "] PressConvDiff: CPU time                               "
                      "     ",
               timer.get_duration());
    }
    if (this->info_ != nullptr) {
      std::stringstream pre_str;
      pre_str << "precond"
              << std::setw(1 + std::log10(this->control().maxits()))
              << std::setfill('0') << this->iter_;
      this->info_->add(pre_str.str());
      this->info_->get_child(pre_str.str())->add("time", timer.get_duration());
    }
  } else if (precond_two_type_ == externalPrecond) {
    b.GetValues(vec2ptr(this->indexset_two_), this->indexset_two_.size(),
                vec2ptr(this->val_temp_two_));
    rhs_temp.SetValues(vec2ptr(this->mapb2s_two_), this->mapb2s_two_.size(),
                       vec2ptr(this->val_temp_two_));
    sol_temp.Zeros();
    // pass on information object
    if (this->info_ != nullptr) {
      std::stringstream pre_str;
      pre_str << "precond"
              << std::setw(1 + std::log10(this->control().maxits()))
              << std::setfill('0') << this->iter_;
      this->info_->add(pre_str.str());
      this->precond_->SetInfo(this->info_->get_child(pre_str.str()));
    }
    timer.reset();
    timer.start();
    precond_->Solve(rhs_temp, &sol_temp);
    timer.stop();

    if (this->print_level_ > 2) {
      LOG_INFO("[" << this->nested_level_
                   << "] preconditioner/solver (block two): CPU time           "
                      "     ",
               timer.get_duration());
    }

    sol_temp.GetValues(vec2ptr(this->mapb2s_two_), this->mapb2s_two_.size(),
                       vec2ptr(this->val_temp_two_));
    x.SetValues(vec2ptr(this->indexset_two_), this->indexset_two_.size(),
                vec2ptr(this->val_temp_two_));
  }
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SolveBlockA(const VectorType &b, VectorType &x) {
  Timer timer;

  timer.reset();
  timer.start();

  this->solver_A_->Solve(b, &x);
  switch (this->print_level_) {
  default:
    break;
  case 2: {
    LOG_INFO("[" << this->nested_level_ << "] Iterations solver for block A",
             this->solver_A_->iter());
    LOG_INFO("[" << this->nested_level_
                 << "] Final relative residual of solver for block A",
             this->solver_A_->res());
    break;
  }
  case 3: {
    LOG_INFO(
        "[" << this->nested_level_
            << "] Solver A: iter                                             ",
        this->solver_A_->iter());
    LOG_INFO(
        "[" << this->nested_level_
            << "] Solver A: final re. res.                                   ",
        this->solver_A_->res());
    break;
  }
  }

  timer.stop();
  if (this->print_level_ > 2)
    LOG_INFO(
        "[" << this->nested_level_
            << "] Solver A: CPU time                                         ",
        timer.get_duration());
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::SolveSimpleBlockA(const VectorType &b,
                                               VectorType &x) {
  std::vector< ValueType > b_loc(diag_A_.size(), 0.);
  std::vector< ValueType > x_loc(diag_A_.size(), 0.);

  const int offset_A = this->offsets_block_one_[this->my_rank_];

  std::vector< int > indices_A(diag_A_.size(), 0);
  for (size_t i = 0, e_i = diag_A_.size(); i != e_i; ++i) {
    indices_A[i] = offset_A + i;
  }

  b.GetValues(vec2ptr(indices_A), indices_A.size(), vec2ptr(b_loc));

  for (size_t i = 0, e_i = diag_A_.size(); i != e_i; ++i) {
    x_loc[i] = b_loc[i] / diag_A_[i];
  }

  x.SetValues(vec2ptr(indices_A), indices_A.size(), vec2ptr(x_loc));

  b_loc.clear();
  x_loc.clear();
}

/// Updates solution: x = x + Vy with y solution of least squares problem.
/// @param V Krylov subspace basis
/// @param H Hessenberg matrix
/// @param g rhs of least squares problem
/// @param k iteration step
/// @param x solution vector

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::UpdateSolution(
    const SeqDenseMatrix< ValueType > &H, const std::vector< ValueType > &g,
    int k, VectorType *x) const {
  std::vector< ValueType > y(g);

  // back substitution
  for (int i = k + 1; i--;) {
    assert(H(i, i) != static_cast< ValueType >(0.));
    y[i] /= H(i, i);

    const ValueType temp = y[i];
    for (int j = 0; j < i; ++j)
      y[j] -= H(i, j) * temp;
  }

  // compute solution
  for (int j = 0; j <= k; ++j)
    x->Axpy(*(Z_[j]), y[j]);

  y.clear();
}

template < class LAD, int DIM >
void SchurComplement< LAD, DIM >::ApplyPressureConvDiff(const VectorType &b,
                                                   VectorType *x) {
  VectorType y, z;
  x->Zeros();
  y.CloneFromWithoutContent(*x);
  z.CloneFromWithoutContent(*x);
  Timer timer;

  // compute -scaling_S_ * Q^{-1} * F * H^{-1}b
  if (this->scaling_S_ != 0.0) {
    // y =     H^{-1}b
    timer.reset();
    timer.start();
    this->solver_H_->Solve(b, &y);
    timer.stop();

    if (this->print_level_ > 2)
      LOG_INFO("[" << this->nested_level_
                   << "] Solver H: CPU time                                    "
                      "     ",
               timer.get_duration());
    // if ( print_level_ > 1 ) LOG_INFO( "[" << this->nested_level_ << "] Solver
    // H: iter                                             ",
    // this->solver_H_->iter( ) ); if ( print_level_ > 1 ) LOG_INFO( "[" <<
    // this->nested_level_ << "] Solver H: final re. res. ",
    // this->solver_H_->res( ) );

    // z = F * y
    timer.reset();
    timer.start();
    F_.VectorMult(y, &z);
    timer.stop();
    if (this->print_level_ > 2)
      LOG_INFO("[" << this->nested_level_
                   << "] Op F: CPU time                                        "
                      "     ",
               timer.get_duration());

    // x = Q^{-1} z
    timer.reset();
    timer.start();
    this->solver_Q_->Solve(z, x);
    timer.stop();
    if (this->print_level_ > 2)
      LOG_INFO("[" << this->nested_level_
                   << "] Solver Q: CPU time                                    "
                      "     ",
               timer.get_duration());
    // if ( print_level_ > 1 ) LOG_INFO( "[" << this->nested_level_ << "] Solver
    // Q: iter                                             ",
    // this->solver_Q_->iter( ) ); if ( print_level_ > 1 ) LOG_INFO( "[" <<
    // this->nested_level_ << "] Solver Q: final re. res. ",
    // this->solver_Q_->res( ) );

    // x = -scaling_S * x
    x->Scale(static_cast< ValueType >(-1.) * this->scaling_S_);
  }

  // compute x = scaling_D * D^{-1}b + x
  if (this->scaling_D_ != 0.0 && this->solver_D_ != nullptr) {
    // y = D^{-1} b
    y.Zeros();
    timer.reset();
    timer.start();
    this->solver_D_->Solve(b, &y);
    timer.stop();
    if (this->print_level_ > 2)
      LOG_INFO("[" << this->nested_level_
                   << "] Solver D: CPU time                                    "
                      "     ",
               timer.get_duration());

    // x = x + scaling_D * y
    x->Axpy(y, static_cast< ValueType >(1.) * this->scaling_D_);
  }
  z.Clear();
  y.Clear();
}

// template instantiation
#ifdef WITH_HYPRE
template class SchurComplement< LADescriptorHypreD, 2 >;
template class SchurComplement< LADescriptorHypreD, 3 >;
#endif
template class SchurComplement< LADescriptorCoupledD, 2 >;
template class SchurComplement< LADescriptorCoupledD, 3 >;
template class SchurComplement< LADescriptorCoupledS, 2 >;
template class SchurComplement< LADescriptorCoupledS, 3 >;
} // namespace la
} // namespace hiflow
