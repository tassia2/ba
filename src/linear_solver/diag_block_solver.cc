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

#include <time.h>

#include "linear_solver/diag_block_solver.h"

namespace hiflow {
namespace la {

template < class LAD > void DiagBlockSolver< LAD >::Clear() {
  this->block_op_ = nullptr;

  this->initialized_ = false;
  this->override_operator_.clear();
  this->op_modified_.clear();
  this->op_passed2solver_.clear();
  this->active_blocks_.clear();
  this->use_explicit_solver_.clear();

  this->expl_solver_.clear();
  this->impl_solver_.clear();

  this->level_ = 0;
  this->name_ = "DiagBlockSolver";
}

template < class LAD >
void DiagBlockSolver< LAD >::Init(
    int num_blocks, const std::vector< std::vector< int > > &blocks) {
  assert(!blocks.empty());
  this->Clear();

#ifndef NDEBUG
  for (size_t l = 0; l < blocks.size(); ++l) {
    assert(!blocks[l].empty());
    for (size_t k = 0; k < blocks[l].size(); ++k) {
      assert(blocks[l][k] >= 0);
      assert(blocks[l][k] < num_blocks);
    }
  }
#endif
  this->num_sub_solvers_ = blocks.size();
  this->blocks_ = blocks;

  this->active_blocks_.resize(this->num_sub_solvers_);
  for (int l = 0; l < num_sub_solvers_; ++l) {
    this->active_blocks_[l].resize(num_blocks, false);
  }

  for (size_t cur_sub_solver = 0; cur_sub_solver < blocks.size();
       ++cur_sub_solver) {
    for (size_t k = 0; k < blocks[cur_sub_solver].size(); ++k) {
      int cur_block = blocks[cur_sub_solver][k];

      this->active_blocks_[cur_sub_solver][cur_block] = true;
    }
  }

  this->expl_solver_.resize(this->num_sub_solvers_, nullptr);
  this->impl_solver_.resize(this->num_sub_solvers_, nullptr);

  this->override_operator_.resize(this->num_sub_solvers_, false);

  this->op_modified_.resize(this->num_sub_solvers_, false);
  this->op_passed2solver_.resize(this->num_sub_solvers_, false);
  this->use_explicit_solver_.resize(this->num_sub_solvers_, true);

  this->initialized_ = true;

  if (this->print_level_ >= 1) {
    for (int l = 0; l < this->num_sub_solvers_; ++l) {
      LOG_INFO("Active Blocks for sub solver "
                   << l << ": "
                   << string_from_range(active_blocks_[l].begin(),
                                        active_blocks_[l].end()),
               "");
    }
  }
}

template < class LAD >
void DiagBlockSolver< LAD >::BuildImpl(BlockVector< LAD > const *b,
                                       BlockVector< LAD > *x) {
  assert(this->op_ != nullptr);

  // cast to block matrix
  this->block_op_ = dynamic_cast< BlockMatrix< LAD > * >(this->op_);

  if (this->block_op_ == 0) {
    LOG_ERROR("[" << this->level_
                  << "] Called DiagBlockSolver::Build with incompatible input "
                     "vector type.");
    quit_program();
  }

  // pass operator to sub solver
  for (int l = 0; l < this->num_sub_solvers_; ++l) {
    if (this->use_explicit_solver_[l]) {
      assert(this->expl_solver_[l] != nullptr);
      if (!this->override_operator_[l]) {
        if (this->op_modified_[l] || !(this->op_passed2solver_[l])) {
          this->expl_solver_[l]->SetupOperator(*this->block_op_);
          this->expl_solver_[l]->Build(b, x);

          this->op_modified_[l] = false;
          this->op_passed2solver_[l] = true;
        }
      }
    }
  }

  this->SetState(true);
  this->SetModifiedOperator(false);
}

template < class LAD >
LinearSolverState
DiagBlockSolver< LAD >::SolveImpl(const BlockVector< LAD > &in,
                                  BlockVector< LAD > *out) {
  assert(this->initialized_);
  assert(out->is_initialized());
  assert(in.is_initialized());

  std::vector< bool > active_blocks_out = out->get_active_blocks();
  LinearSolverState state = kSolverError;

  for (int l = 0; l < this->num_sub_solvers_; ++l) {
    // out[block_[l]] = A_l^{-1}*f_l
    out->set_active_blocks(this->active_blocks_[l]);
    state = this->SolveBlock(l, in, out);
  }

  // reset active blocks
  out->set_active_blocks(active_blocks_out);

  return state;
}

template < class LAD >
LinearSolverState
DiagBlockSolver< LAD >::SolveBlock(int solver_nr, const BlockVector< LAD > &in,
                                   BlockVector< LAD > *out) {
  assert(solver_nr >= 0);
  assert(solver_nr < this->num_sub_solvers_);

  if (this->use_explicit_solver_[solver_nr]) {
    if (this->print_level_ > 2) {
      LOG_INFO("[" << this->level_ << "] Solving with explicitly given matrix ",
               solver_nr);
    }
    assert(this->expl_solver_[solver_nr] != nullptr);
    return this->expl_solver_[solver_nr]->Solve(in, out);
  }
  if (this->print_level_ > 2) {
    LOG_INFO("[" << this->level_ << "] Solving with implicitly given matrix ",
             solver_nr);
  }
  assert(this->impl_solver_[solver_nr] != nullptr);
  return this->impl_solver_[solver_nr]->Solve(in, out);
}

// template instantiation
template class DiagBlockSolver< LADescriptorCoupledD >;
template class DiagBlockSolver< LADescriptorCoupledS >;
#ifdef WITH_HYPRE
template class DiagBlockSolver< LADescriptorHypreD >;
#endif
} // namespace la
} // namespace hiflow
