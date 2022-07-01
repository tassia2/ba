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

#include "linear_solver/pcd_preconditioner.h"

namespace hiflow {
namespace la {

template < class LAD > void PCDPreconditioner< LAD >::Clear() {
  LinearSolver< LAD >::Clear();
  this->op_F_ = nullptr;
  this->solver_Q_ = nullptr;
  this->solver_H_ = nullptr;
  this->solver_D_ = nullptr;
  this->s_D_ = 0.;
  this->s_QFH_ = 1.;
  this->aux_vec_init_ = false;
  this->reuse_vectors_ = false;

  this->name_ = "PCD";
}

template < class LAD >
void PCDPreconditioner< LAD >::BuildImpl(VectorType const *b, VectorType *x) {
  assert(this->op_F_ != nullptr);
  assert(this->solver_Q_ != nullptr);
  assert(this->solver_H_ != nullptr);

  this->solver_Q_->Build(b, x);
  this->solver_H_->Build(b, x);

  if (this->solver_D_ != nullptr) {
    this->solver_D_->Build(b, x);
  }
}

template < class LAD >
LinearSolverState
PCDPreconditioner< LAD >::SolveImpl(const Vector< DataType > &in,
                                    Vector< DataType > *out) {
  VectorType const *v_in = dynamic_cast< VectorType const * >(&in);
  VectorType *v_out = dynamic_cast< VectorType * >(out);

  if ((v_in != 0) && (v_out != 0)) {
    return this->SolveImpl(*v_in, v_out);
  } else {
    if (v_in == 0) {
      LOG_ERROR("Called PCDPreconditioner::Solve with incompatible input "
                "vector type.");
    }
    if (v_out == 0) {
      LOG_ERROR("Called PCDPreconditioner::Solve with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class LAD >
LinearSolverState PCDPreconditioner< LAD >::SolveImpl(const VectorType &in,
                                                      VectorType *out) {
  assert(this->op_F_ != nullptr);
  assert(this->solver_Q_ != nullptr);
  assert(this->solver_H_ != nullptr);
  assert(out->is_initialized());
  assert(in.is_initialized());

  this->AllocateVectors(in);
  this->SetVectorsToZero();

  out->Zeros();
  Timer timer;

  // out = H^{-1} in
  timer.reset();
  timer.start();
  LinearSolverState conv = this->solver_H_->Solve(in, out);
  timer.stop();

  if (this->print_level_ >= 1) {
    LOG_INFO(this->name_, " Solver H: iter     : " << this->solver_H_->iter());
    LOG_INFO(this->name_, " Solver H: res.     : " << this->solver_H_->res());

    if (this->print_level_ >= 2) {
      LOG_INFO(this->name_, " Solver H: CPU time : " << timer.get_duration());
    }
  }

  // tmp_ = F * out
  timer.reset();
  timer.start();

  this->op_F_->VectorMult(*out, &tmp_);
  timer.stop();
  if (this->print_level_ >= 2) {
    LOG_INFO(this->name_, " Op F: CPU time     : " << timer.get_duration());
  }

  // out = Q^{-1} tmp_
  out->Zeros();
  timer.reset();
  timer.start();
  conv = this->solver_Q_->Solve(tmp_, out);
  timer.stop();

  if (this->print_level_ >= 1) {
    LOG_INFO(this->name_, " Solver Q: iter     : " << this->solver_Q_->iter());
    LOG_INFO(this->name_, " Solver Q: res.     : " << this->solver_Q_->res());
    if (this->print_level_ >= 2) {
      LOG_INFO(this->name_, " Solver Q: CPU time : " << timer.get_duration());
    }
  }

  // out = s_QFH * out
  out->Scale(this->s_QFH_);

  // compute out = s_D * D^{-1} in + out
  if (this->s_D_ != 0.0 && this->solver_D_ != nullptr) {
    // y = D^{-1} in
    tmp_.Zeros();
    timer.reset();
    timer.start();
    conv = this->solver_D_->Solve(in, &tmp_);
    timer.stop();

    if (this->print_level_ >= 1) {
      LOG_INFO(this->name_,
               " Solver D: iter     : " << this->solver_D_->iter());
      LOG_INFO(this->name_, " Solver D: res.     : " << this->solver_D_->res());
      if (this->print_level_ >= 2) {
        LOG_INFO(this->name_, " Solver D: CPU time : " << timer.get_duration());
      }
    }

    // out = out + s_D * tmp_
    out->Axpy(tmp_, this->s_D_);
  }

  if (!this->reuse_vectors_) {
    this->FreeVectors();
  }

  return conv;
}

// template instantiation
template class PCDPreconditioner< LADescriptorCoupledD >;
template class PCDPreconditioner< LADescriptorCoupledS >;
template class PCDPreconditioner< LADescriptorBlock< LADescriptorCoupledD > >;
template class PCDPreconditioner< LADescriptorBlock< LADescriptorCoupledS > >;
#ifdef WITH_HYPRE
template class PCDPreconditioner< LADescriptorHypreD >;
template class PCDPreconditioner< LADescriptorBlock< LADescriptorHypreD > >;
#endif
} // namespace la
} // namespace hiflow
