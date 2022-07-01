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

#include "common/log.h"
#include "common/timer.h"
#include "linear_algebra/schur_operator.h"

namespace hiflow {
namespace la {

template < class LAD > void SchurOperator< LAD >::Clear() {
  this->block_op_ = nullptr;
  this->solver_A_ = nullptr;
  this->block_one_.clear();
  this->block_two_.clear();
  this->called_init_ = false;
  this->active_blocks_B_.clear();
  this->active_blocks_C_.clear();
  this->active_blocks_D_.clear();
  this->override_operator_A_ = false;
  this->A_modified_ = false;
  this->A_passed2solver_ = false;
  this->active_blocks_one_.clear();
  this->active_blocks_two_.clear();

  this->aux_vec_init_ = false;
  this->reuse_vectors_ = false;

  this->num_A_ = 0;
  this->iter_A_ = 0;
  this->time_A_ = 0.;

  this->print_level_ = 0;
}

template < class LAD >
void SchurOperator< LAD >::Init(int num_blocks,
                                std::vector< size_t > &block_one,
                                std::vector< size_t > &block_two) {
  assert(block_one.size() > 0);
  assert(block_two.size() > 0);

  // this->Clear ( );
#ifndef NDEBUG
  for (int l = 0; l < block_one.size(); ++l) {
    assert(block_one[l] < num_blocks);
  }
  for (int l = 0; l < block_two.size(); ++l) {
    assert(block_two[l] < num_blocks);
  }
#endif
  this->block_one_ = block_one;
  this->block_two_ = block_two;

  this->active_blocks_B_.clear();
  this->active_blocks_B_.resize(num_blocks);
  this->active_blocks_C_.clear();
  this->active_blocks_C_.resize(num_blocks);
  this->active_blocks_D_.clear();
  this->active_blocks_D_.resize(num_blocks);

  this->active_blocks_one_.clear();
  this->active_blocks_one_.resize(num_blocks, false);
  this->active_blocks_two_.clear();
  this->active_blocks_two_.resize(num_blocks, false);

  for (int l = 0; l < num_blocks; ++l) {
    this->active_blocks_B_[l].resize(num_blocks, false);
    this->active_blocks_C_[l].resize(num_blocks, false);
    this->active_blocks_D_[l].resize(num_blocks, false);
  }

  for (int k = 0; k < block_one_.size(); ++k) {
    int kk = this->block_one_[k];
    this->active_blocks_one_[kk] = true;

    for (int l = 0; l < block_two_.size(); ++l) {
      int ll = this->block_two_[l];

      this->active_blocks_B_[kk][ll] = true;
      this->active_blocks_C_[ll][kk] = true;

      this->active_blocks_two_[ll] = true;
    }
  }
  for (int l = 0; l < block_two_.size(); ++l) {
    int ll = this->block_two_[l];

    for (int k = 0; k < block_two_.size(); ++k) {
      int kk = this->block_two_[k];
      this->active_blocks_D_[ll][kk] = true;
    }
  }

  this->called_init_ = true;
}

template < class LAD > 
void SchurOperator< LAD >::PassOpA2Solver() {
  if (!this->override_operator_A_) {
    if (this->A_modified_ || !(this->A_passed2solver_)) {
      if (this->print_level_ > 2) {
        LOG_INFO(" Pass operator solver A", " ");
      }
      this->solver_A_->SetupOperator(*this->block_op_);
      // this->solver_A_->Build ( );
      this->A_modified_ = false;
      this->A_passed2solver_ = true;
    }
  }
}

template < class LAD >
void SchurOperator< LAD >::VectorMult(Vector< BDataType > &in,
                                      Vector< BDataType > *out) const {
  BlockVector< LAD > *bv_in, *bv_out;

  bv_in = dynamic_cast< BlockVector< LAD > * >(&in);
  bv_out = dynamic_cast< BlockVector< LAD > * >(out);

  if ((bv_in != 0) && (bv_out != 0)) {
    this->VectorMult(*bv_in, bv_out);
  } else {
    if (bv_in == 0) {
      LOG_ERROR("Called SchurOperator::VectorMult with incompatible input "
                "vector type.");
    }
    if (bv_out == 0) {
      LOG_ERROR("Called SchurOperator::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class LAD >
void SchurOperator< LAD >::VectorMult(BlockVector< LAD > &in,
                                      BlockVector< LAD > *out) const {
  assert(this->IsInitialized());
  assert(this->solver_A_ != nullptr);
  assert(this->block_op_ != nullptr);
  assert(out->is_initialized());
  assert(in.is_initialized());
#ifndef NDEBUG
  for (int i=0; i<this->block_two_.size(); ++i) 
  {   
    assert (in.block_is_active(this->block_two_[i]));
    assert (out->block_is_active(this->block_two_[i]));
  }
#endif

  // Allocate array of pointer for Krylov subspace basis
  this->AllocateVectors(in);
  this->SetVectorsToZero();

  // tmp[block_one] = B * in [block_two] = B*g
  this->tmp_->set_active_blocks(this->active_blocks_one_);
  this->block_op_->SubmatrixVectorMult(this->active_blocks_B_, in, this->tmp_);

  // tmp2[block_one] = A^{-1} * tmp[block_one] = A^{-1} * B * g
  this->tmp2_->set_active_blocks(this->active_blocks_one_);

  Timer timer;
  timer.reset();
  timer.start();
  this->solver_A_->Solve(*(this->tmp_), this->tmp2_);
  timer.stop();

  if (this->print_level_ >= 1) {
    LOG_INFO(" Solver A: iter      ", this->solver_A_->iter());
    LOG_INFO(" Solver A: res.      ", this->solver_A_->res());

    if (this->print_level_ >= 2) {
      LOG_INFO(" Solver A: CPU time ", timer.get_duration());
    }
  }

  // statistics
  this->num_A_++;
  this->time_A_ += timer.get_duration();
  this->iter_A_ += this->solver_A_->iter();

  // out[block_two] = C * tmp2[block_one] = C * A^{-1} * B * g
  this->block_op_->SubmatrixVectorMult(this->active_blocks_C_, *(this->tmp2_), out);

  // out[block_two] = D * in[block_two] - out[block_two] = D*g - C * A^{-1} * B * g
  this->block_op_->SubmatrixVectorMultAdd(this->active_blocks_D_, 1., in, -1., out);

  // deallocate auxiliary vectors
  if (!this->reuse_vectors_) {
    this->FreeVectors();
  }
}

// template instantiation
template class SchurOperator< LADescriptorCoupledD >;
template class SchurOperator< LADescriptorCoupledS >;
#ifdef WITH_HYPRE
template class SchurOperator< LADescriptorHypreD >;
#endif
} // namespace la
} // namespace hiflow
