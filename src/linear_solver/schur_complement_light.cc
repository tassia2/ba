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

#include "linear_solver/schur_complement_light.h"

namespace hiflow {
namespace la {

template < class LAD > void SchurComplementLight< LAD >::Clear() {
  this->block_op_ = nullptr;
  this->block_one_.clear();
  this->block_two_.clear();
  this->initialized_ = false;
  this->active_blocks_B_.clear();
  this->active_blocks_C_.clear();
  this->override_operator_A_ = false;
  this->A_modified_ = false;
  this->A_passed2solver_ = false;
  this->active_blocks_one_.clear();
  this->active_blocks_two_.clear();
  this->use_explicit_A_ = true;
  this->use_explicit_S_ = false;

  this->expl_solver_A_ = nullptr;
  this->expl_solver_S_ = nullptr;
  this->impl_solver_A_ = nullptr;
  this->impl_solver_S_ = nullptr;

  this->level_ = 0;

  this->name_ = "SchurComplementSolver";
}

template < class LAD >
void SchurComplementLight< LAD >::Init(const size_t num_blocks,
                                       std::vector< size_t > &block_one,
                                       std::vector< size_t > &block_two) {
  assert(!block_one.empty());
  assert(!block_two.empty());

  // this->Clear ( );

#ifndef NDEBUG
  for (size_t l = 0; l < block_one.size(); ++l) {
    assert(block_one[l] < num_blocks);
  }
  for (size_t l = 0; l < block_two.size(); ++l) {
    assert(block_two[l] < num_blocks);
  }
#endif
  this->block_one_ = block_one;
  this->block_two_ = block_two;

  this->active_blocks_B_.clear();
  this->active_blocks_B_.resize(num_blocks);
  this->active_blocks_C_.clear();
  this->active_blocks_C_.resize(num_blocks);

  this->active_blocks_one_.resize(num_blocks, false);
  this->active_blocks_two_.resize(num_blocks, false);

  for (size_t l = 0; l < num_blocks; ++l) {
    this->active_blocks_B_[l].resize(num_blocks, false);
    this->active_blocks_C_[l].resize(num_blocks, false);
  }

  for (size_t k = 0; k < block_one_.size(); ++k) {
    size_t kk = this->block_one_[k];
    this->active_blocks_one_[kk] = true;

    for (size_t l = 0; l < block_two_.size(); ++l) {
      size_t ll = this->block_two_[l];

      this->active_blocks_B_[kk][ll] = true;
      this->active_blocks_C_[ll][kk] = true;

      this->active_blocks_two_[ll] = true;
    }
  }
  this->initialized_ = true;
}

template < class LAD >
void SchurComplementLight< LAD >::BuildImpl(BlockVector< LAD > const *b,
                                            BlockVector< LAD > *x) {
  assert(this->op_ != nullptr);

  // cast to block matrix
  this->block_op_ = dynamic_cast< BlockMatrix< LAD > * >(this->op_);

  if (this->block_op_ == 0) {
    LOG_ERROR(this->name_ << " Called SchurComplementLight::Build with "
                             "incompatible input vector type.");
    quit_program();
  }

  // pass operator to sub solver
  assert(this->block_op_ != nullptr);
  assert(this->expl_solver_A_ != nullptr);

  if (this->use_explicit_A_) {
    if (!this->override_operator_A_) {
      if (this->A_modified_ || !(this->A_passed2solver_)) {
        this->expl_solver_A_->SetupOperator(*this->block_op_);
        this->expl_solver_A_->Build(b, x);

        this->A_modified_ = false;
        this->A_passed2solver_ = true;
      }
    }
  }
}

template < class LAD >
LinearSolverState
SchurComplementLight< LAD >::SolveImpl(const BlockVector< LAD > &in,
                                       BlockVector< LAD > *out) {
  assert(this->initialized_);
  assert(this->block_op_ != nullptr);
  assert(out->is_initialized());
  assert(in.is_initialized());

#ifndef NDEBUG
  for (int i=0; i<this->block_one_.size(); ++i) 
  {
    assert (in.block_is_active(this->block_one_[i]));
    assert (out->block_is_active(this->block_one_[i]));
  }
  for (int i=0; i<this->block_two_.size(); ++i) 
  {
    assert (in.block_is_active(this->block_two_[i]));
    assert (out->block_is_active(this->block_two_[i]));
  }
#endif

  if (this->print_level_ > 2) {
    LOG_INFO(this->name_,
             " Preparing right hand side for Schur complement equation");
  }

  // tmp[block_one] = A^{-1}*f
  // tmp[block_two] = 0
  this->AllocateVectors(in);
  this->SetVectorsToZero();

  this->tmp_.set_active_blocks(this->active_blocks_one_);

  // pass on information object
  if (this->info_ != nullptr) {
    std::string str = "BlockA1";
    this->info_->add(str);
    if (this->use_explicit_A_) {
      assert(this->expl_solver_A_ != nullptr);
      this->expl_solver_A_->SetInfo(this->info_->get_child(str));
    } else {
      assert(this->impl_solver_A_ != nullptr);
      this->impl_solver_A_->SetInfo(this->info_->get_child(str));
    };
  }

  this->SolveBlockA(in, &(this->tmp_));

  std::vector<bool> all_active (this->active_blocks_one_.size(), true);
  this->tmp_.set_active_blocks (all_active);
  
  // pass on information object
  if (this->info_ != nullptr) {
    std::string str = "BlockA2";
    this->info_->add(str);
    if (this->use_explicit_A_) {
      assert(this->expl_solver_A_ != nullptr);
      this->expl_solver_A_->SetInfo(this->info_->get_child(str));
    } else {
      assert(this->impl_solver_A_ != nullptr);
      this->impl_solver_A_->SetInfo(this->info_->get_child(str));
    };
  }

  this->block_op_->SubmatrixVectorMult(this->active_blocks_C_, this->tmp_,
                                       &(this->tmp_));

  // tmp[block_two] = in[block_two] - tmp[block_two] = g - C * A^{-1} f
  this->tmp_.set_active_blocks(this->active_blocks_two_);
  this->tmp_.ScaleAdd(in, static_cast< BDataType >(-1.));

  // out[block_two] = S^{-1} tmp[block_two] = S^{-1} (g - C * A^{-1} f) = y
  std::vector< bool > active_blocks_out = out->get_active_blocks();
  out->set_active_blocks(this->active_blocks_two_);

  // pass on information object
  if (this->info_ != nullptr) {
    std::string str = "BlockS";
    this->info_->add(str);
    if (this->use_explicit_S_) {
      assert(this->expl_solver_S_ != nullptr);
      this->expl_solver_S_->SetInfo(this->info_->get_child(str));
    } else {
      assert(this->impl_solver_S_ != nullptr);
      this->impl_solver_S_->SetInfo(this->info_->get_child(str));
    };
  }

  this->SolveBlockS(this->tmp_, out);

  // tmp[block_one] = in[Block_one] = f
  // tmp[block_two] = in[block_two] = g
  this->tmp_.CopyFrom(in);

  // tmp[block_one] = tmp[block_one] - B * out[block_two] = f - B * y
  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, " Backward substitution");
  }

  this->tmp_.set_active_blocks(this->active_blocks_one_);
  this->block_op_->SubmatrixVectorMultAdd(this->active_blocks_B_, -1., *out, 1.,
                                          &(this->tmp_));

  // out[block_one] = A^{-1} * tmp[block_one] = A^{-1} * (f - B*y)
  out->set_active_blocks(this->active_blocks_one_);
  this->tmp_.set_active_blocks(this->active_blocks_one_);

  // pass on information object
  if (this->info_ != nullptr) {
    std::string str = "BlockA3";
    this->info_->add(str);
    if (this->use_explicit_A_) {
      assert(this->expl_solver_A_ != nullptr);
      this->expl_solver_A_->SetInfo(this->info_->get_child(str));
    } else {
      assert(this->impl_solver_A_ != nullptr);
      this->impl_solver_A_->SetInfo(this->info_->get_child(str));
    };
  }

  this->SolveBlockA(this->tmp_, out);

  // reset active blocks
  out->set_active_blocks(active_blocks_out);

  // deallocate Krylov subspace basis V
  if (!this->reuse_vectors_) {
    this->FreeVectors();
  }
  return hiflow::la::kSolverSuccess;
}

template < class LAD >
LinearSolverState
SchurComplementLight< LAD >::SolveBlockA(const BlockVector< LAD > &in,
                                         BlockVector< LAD > *out) {
  LinearSolverState state;
  Timer timer;
  timer.start();
  if (this->use_explicit_A_) {
    if (this->print_level_ > 2) {
      LOG_INFO(this->name_, " Solving with explicitly given matrix A");
    }
    assert(this->expl_solver_A_ != nullptr);
    state = this->expl_solver_A_->Solve(in, out);
  } else {
    if (this->print_level_ > 2) {
      LOG_INFO(this->name_, " Solving with implicitly given matrix A");
    }
    assert(this->impl_solver_A_ != nullptr);
    state = this->impl_solver_A_->Solve(in, out);
  };
  timer.stop();
  BDataType time = timer.get_duration();

  if (this->print_level_ >= 2) {
    LOG_INFO(this->name_, " Solver A: CPU time " << time);
  }

  return state;
}

template < class LAD >
LinearSolverState
SchurComplementLight< LAD >::SolveBlockS(const BlockVector< LAD > &in,
                                         BlockVector< LAD > *out) {
  LinearSolverState state;
  Timer timer;
  timer.start();
  if (this->use_explicit_S_) {
    if (this->print_level_ > 2) {
      LOG_INFO(this->name_, " Solving with explicitly given matrix S");
    }
    assert(this->expl_solver_S_ != nullptr);
    state = this->expl_solver_S_->Solve(in, out);
  } else {
    if (this->print_level_ > 2) {
      LOG_INFO(this->name_, " Solving with implicitly given matrix S");
    }
    assert(this->impl_solver_S_ != nullptr);
    state = this->impl_solver_S_->Solve(in, out);
  }
  timer.stop();
  BDataType time = timer.get_duration();

  if (this->print_level_ >= 2) {
    LOG_INFO(this->name_, " Solver S: CPU time " << time);
  }

  return state;
}

// template instantiation
template class SchurComplementLight< LADescriptorCoupledD >;
template class SchurComplementLight< LADescriptorCoupledS >;
#ifdef WITH_HYPRE
template class SchurComplementLight< LADescriptorHypreD >;
#endif
} // namespace la
} // namespace hiflow
