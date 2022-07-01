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

#include <ctime>

#include "linear_solver/single_subblock_solver.h"

namespace hiflow {
namespace la {

template < class LAD > void SingleSubBlockSolver< LAD >::Clear() {
  LinearSolver< LADescriptorBlock< LAD > >::Clear();
  this->block_op_ = nullptr;
  this->sub_solver_ = nullptr;
  this->block_nr_ = 0;
  this->initialized_ = false;
  this->override_operator_ = false;
  this->name_ = "SingleSubBlockSolver";
}

template < class LAD >
void SingleSubBlockSolver< LAD >::BuildImpl(BlockVector< LAD > const *b,
                                            BlockVector< LAD > *x) {

  // pass operator to sub solver
  if (!this->override_operator_) {
    assert(this->op_ != nullptr);

    // cast to block matrix
    this->block_op_ = dynamic_cast< BlockMatrix< LAD > * >(this->op_);

    if (this->block_op_ == 0) {
      LOG_ERROR("Called SingleSubBlockSolver::Build with incompatible input "
                "vector type.");
      quit_program();
    }

    this->sub_solver_->SetupOperator(
        this->block_op_->GetBlock(this->block_nr_, this->block_nr_));
    this->sub_solver_->Build(&(b->GetBlock(this->block_nr_)),
                             &(x->GetBlock(this->block_nr_)));
  }
}

template < class LAD >
LinearSolverState
SingleSubBlockSolver< LAD >::SolveImpl(const BlockVector< LAD > &in,
                                       BlockVector< LAD > *out) {
  assert(this->initialized_);
  assert(this->sub_solver_ != nullptr);
  assert(this->block_nr_ < in.num_blocks());
  assert(out->is_initialized());
  assert(in.is_initialized());

  assert (in.block_is_active(this->block_nr_));
  assert (out->block_is_active(this->block_nr_));
  
  // pass on information object
  if (this->info_ != nullptr) {
    this->sub_solver_->SetInfo(this->info_);
  }

  LinearSolverState conv = this->sub_solver_->Solve(
      in.GetBlock(this->block_nr_), &(out->GetBlock(this->block_nr_)));

  return conv;
}

// template instantiation
template class SingleSubBlockSolver< LADescriptorCoupledD >;
template class SingleSubBlockSolver< LADescriptorCoupledS >;
#ifdef WITH_HYPRE
template class SingleSubBlockSolver< LADescriptorHypreD >;
#endif
} // namespace la
} // namespace hiflow
