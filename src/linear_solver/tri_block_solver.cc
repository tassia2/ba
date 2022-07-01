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

#include "linear_solver/tri_block_solver.h"

namespace hiflow {
namespace la {

template < class LAD > void TriBlockSolver< LAD >::Clear() {
  this->block_op_ = nullptr;
  this->block_one_.clear();
  this->block_two_.clear();
  this->initialized_ = false;
  this->active_blocks_B_.clear();
  this->active_blocks_C_.clear();
  this->override_operator_A_ = false;
  this->A_modified_ = false;
  this->A_passed2solver_ = false;
  this->D_modified_ = false;
  this->D_passed2solver_ = false;
  this->active_blocks_one_.clear();
  this->active_blocks_two_.clear();
  this->use_explicit_A_ = true;
  this->use_explicit_D_ = false;

  this->expl_solver_A_ = nullptr;
  this->expl_solver_D_ = nullptr;
  this->impl_solver_A_ = nullptr;
  this->impl_solver_D_ = nullptr;

  this->level_ = 0;

  this->name_ = "TriBlockSolver";
}

template < class LAD >
void TriBlockSolver< LAD >::Init(int num_blocks,
                                 const std::vector< int > &block_one,
                                 const std::vector< int > &block_two,
                                 TriangularStructure type) {
  assert(!block_one.empty());
  assert(!block_two.empty());

  // this->Clear ( );

#ifndef NDEBUG
  for (size_t l = 0; l < block_one.size(); ++l) {
    assert(block_one[l] >= 0);
    assert(block_one[l] < num_blocks);
  }
  for (size_t l = 0; l < block_two.size(); ++l) {
    assert(block_two[l] >= 0);
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
    int kk = this->block_one_[k];
    this->active_blocks_one_[kk] = true;

    for (size_t l = 0; l < block_two_.size(); ++l) {
      int ll = this->block_two_[l];

      this->active_blocks_B_[kk][ll] = true;
      this->active_blocks_C_[ll][kk] = true;

      this->active_blocks_two_[ll] = true;
    }
  }
  this->initialized_ = true;
  this->structure_ = type;
}

template < class LAD >
void TriBlockSolver< LAD >::PassOpA2Solver(BlockVector< LAD > const *b,
                                           BlockVector< LAD > *x) {
  assert(this->block_op_ != nullptr);

  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, " PassOpA2Solver");
    LOG_INFO(this->name_, " Use explicit solver " << this->use_explicit_A_);
    LOG_INFO(this->name_, " Override solver " << this->override_operator_A_);
    LOG_INFO(this->name_, " Modified operator " << this->A_modified_);
    LOG_INFO(this->name_, " Already passed operator to solver " << this->A_passed2solver_);
  }

  if (this->use_explicit_A_) {
    assert(this->expl_solver_A_ != nullptr);
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
void TriBlockSolver< LAD >::PassOpD2Solver(BlockVector< LAD > const *b,
                                           BlockVector< LAD > *x) {
  assert(this->block_op_ != nullptr);

  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, " PassOpD2Solver");
    LOG_INFO(this->name_, " Use explicit solver " << this->use_explicit_D_);
    LOG_INFO(this->name_, " Override solver " << this->override_operator_D_);
    LOG_INFO(this->name_, " Modified operator " << this->D_modified_);
    LOG_INFO(this->name_, " Already passed operator to solver " << this->D_passed2solver_);
  }

  if (this->use_explicit_D_) {
    assert(this->expl_solver_D_ != nullptr);
    if (!this->override_operator_D_) {
      if (this->D_modified_ || !(this->D_passed2solver_)) {
        this->expl_solver_D_->SetupOperator(*this->block_op_);
        this->expl_solver_D_->Build(b, x);

        this->D_modified_ = false;
        this->D_passed2solver_ = true;
      }
    }
  }
}

template < class LAD >
void TriBlockSolver< LAD >::BuildImpl(BlockVector< LAD > const *b,
                                      BlockVector< LAD > *x) {
  assert(this->op_ != nullptr);

  // cast to block matrix
  this->block_op_ = dynamic_cast< BlockMatrix< LAD > * >(this->op_);

  if (this->block_op_ == 0) {
    LOG_ERROR(this->name_ << " Called TriBlockSolver::Build with incompatible "
                             "input vector type.");
    quit_program();
  }

  // pass operator to sub solver
  this->PassOpA2Solver(b, x);
  this->PassOpD2Solver(b, x);
}
/*
        template<class LAD>
        LinearSolverState TriBlockSolver<LAD>::SolveImpl ( const
   Vector<BDataType>& in, Vector<BDataType>* out ) { BlockVector<LAD> const
   *bv_in = dynamic_cast < BlockVector<LAD> const * > ( &in ); BlockVector<LAD>*
   bv_out = dynamic_cast < BlockVector<LAD>* > ( out );

            if ( ( bv_in != 0 ) && ( bv_out != 0 ) )
            {
                this->SolveImpl ( *bv_in, bv_out );
            }
            else
            {
                if ( bv_in == 0 )
                {
                    LOG_ERROR ( this->name_ << " Called TriBlockSolver::Solve
   with incompatible input vector type." );
                }
                if ( bv_out == 0 )
                {
                    LOG_ERROR ( this->name_ << " Called TriBlockSolver::Solve
   with incompatible output vector type." );
                }
                exit ( -1 );
            }
        }
*/
template < class LAD >
LinearSolverState TriBlockSolver< LAD >::SolveImpl(const BlockVector< LAD > &in,
                                                   BlockVector< LAD > *out) {
  assert(this->initialized_);
  assert(this->block_op_ != nullptr);
  assert(in.is_initialized());
  assert(out->is_initialized());
  LinearSolverState state = kSolverError; 

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

  this->AllocateVectors(in);

  // tmp[block_one] = tmp[block_two] = 0
  this->SetVectorsToZero();
  std::vector< bool > active_blocks_out = out->get_active_blocks();
  std::vector< bool > active_blocks_in = in.get_active_blocks();

  // #####################################
  // UPPER_TRIANGULAR
  // | A  B |  |x_1|  = |f|
  // | 0  D |  |x_2|  = |g|
  //LOG_INFO(this->name_, " Start SolveImpl");
  
  if (this->structure_ == UPPER_TRIANGULAR) 
  {
    // out[block_two] = D^{-1}*g
    out->set_active_blocks(this->active_blocks_two_);

    this->SolveBlockD(in, out);

    // tmp[block_one] =  B * out[block_two] =  B * D^{-1} g
    this->tmp_.set_active_blocks(this->active_blocks_one_);
    this->block_op_->SubmatrixVectorMult(this->active_blocks_B_, *out,
                                         &(this->tmp_));

    // tmp[block_one] = in[block_one] - tmp[block_one] = f - B * D^{-1} g
    this->tmp_.ScaleAdd(in, static_cast< BDataType >(-1.));

    // out[block_one] = A^{-1} tmp[block_one] = A^{-1} (f - B * D^{-1} g)
    out->set_active_blocks(this->active_blocks_one_);

    state = this->SolveBlockA(this->tmp_, out);
  }
  // #####################################
  // LOWER_TRIANGULAR
  // | A  0 |  |x_1|  = |f|
  // | C  D |  |x_2|  = |g|
  if (this->structure_ == LOWER_TRIANGULAR) 
  {
    // out[block_one] = A^{-1}*f
    out->set_active_blocks(this->active_blocks_one_);
    this->SolveBlockA(in, out);

    // tmp[block_two] =  C * out[block_one] =  C * A^{-1} f
    this->tmp_.set_active_blocks(this->active_blocks_two_);

    this->block_op_->SubmatrixVectorMult(this->active_blocks_C_, *out,
                                         &(this->tmp_));

    // tmp[block_two] = in[block_two] - tmp[block_two] = g - C * A^{-1} f
    this->tmp_.ScaleAdd(in, static_cast< BDataType >(-1.));

    // out[block_two] = D^{-1} tmp[block_two] = D^{-1} (g - C * A^{-1} f)
    out->set_active_blocks(this->active_blocks_two_);

    state = this->SolveBlockD(this->tmp_, out);
  }

  // #####################################
  // DIAGONAL
  if (this->structure_ == DIAGONAL) 
  {
    // out[block_one] = A^{-1}*f
    out->set_active_blocks(this->active_blocks_one_);
    this->SolveBlockA(in, out);

    // out[block_two] = D^{-1} g
    out->set_active_blocks(this->active_blocks_two_);
    state = this->SolveBlockD(this->tmp_, out);
  }

  // reset active blocks
  out->set_active_blocks(active_blocks_out);

  // deallocate Krylov subspace basis V
  if (!this->reuse_vectors_) 
  {
    this->FreeVectors();
  }
  return state;
}

template < class LAD >
LinearSolverState
TriBlockSolver< LAD >::SolveBlockA(const BlockVector< LAD > &in,
                                   BlockVector< LAD > *out) {
  LinearSolverState state;
  Timer timer;
  timer.start();
  if (this->use_explicit_A_) 
  {
    if (this->print_level_ > 2) 
    {
      LOG_INFO(this->name_, " Solving with explicitly given matrix A");
    }
    assert(this->expl_solver_A_ != nullptr);
    state = this->expl_solver_A_->Solve(in, out);
  } 
  else 
  {
    if (this->print_level_ > 2) 
    {
      LOG_INFO(this->name_, " Solving with implicitly given matrix A");
    }
    assert(this->impl_solver_A_ != nullptr);
    state = this->impl_solver_A_->Solve(in, out);
  }
  timer.stop();
  BDataType time = timer.get_duration();

  if (this->print_level_ >= 2) 
  {
    LOG_INFO(this->name_, " Solver A: CPU time " << time);
  }

  return state;
}

template < class LAD >
LinearSolverState
TriBlockSolver< LAD >::SolveBlockD(const BlockVector< LAD > &in,
                                   BlockVector< LAD > *out) 
{  
  LinearSolverState state;
  Timer timer;
  timer.start();
  if (this->use_explicit_D_) 
  {
    if (this->print_level_ > 2) 
    {
      LOG_INFO(this->name_, " Solving with explicitly given matrix D");
    }

    assert(this->expl_solver_D_ != nullptr);
    state = this->expl_solver_D_->Solve(in, out);
  } 
  else 
  {
    if (this->print_level_ > 2) 
    {
      LOG_INFO(this->name_, " Solving with implicitly given matrix D");
    }
    assert(this->impl_solver_D_ != nullptr);
    state = this->impl_solver_D_->Solve(in, out);
  }
  timer.stop();
  BDataType time = timer.get_duration();

  if (this->print_level_ >= 2) 
  {
    LOG_INFO(this->name_, " Solver D: CPU time " << time);
  }

  return state;
}

// template instantiation
template class TriBlockSolver< LADescriptorCoupledD >;
template class TriBlockSolver< LADescriptorCoupledS >;
#ifdef WITH_HYPRE
template class TriBlockSolver< LADescriptorHypreD >;
#endif
} // namespace la
} // namespace hiflow
