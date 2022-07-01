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

#ifndef HIFLOW_LINEARSOLVER_RICHARDSON_H_
#define HIFLOW_LINEARSOLVER_RICHARDSON_H_

#include "linear_solver/linear_solver.h"
#include "linear_solver/linear_solver_creator.h"
#include <string>
#include <cassert>
#include <cmath>

#include "common/log.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/la_descriptor.h"
#include <iomanip>

namespace hiflow {
namespace la {

/// @brief Richarson iterative solver
///

// no   precond, no   parallel naive: xnew = xold + omega * (b - A*xold)
// with precond, no   parallel naive: xnew = xold + omega * P * (b - A*xold), with P ~ A^{-1}
// with precond, with parallel naive: xnew = P(b - OD*xold)                   with P ~ D^{-1}, A = D + OD

template < class LAD, class PreLAD = LAD > 
class Richardson : public LinearSolver< LAD, PreLAD > 
{
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  Richardson() 
  : LinearSolver<LAD, PreLAD>(),
  do_preconditioning_(false),
  do_parallel_naive_(false),
  fixed_maxits_(true),
  omega_(1.)
  {
    this->SetMethod("NoPreconditioning");
    if (this->print_level_ > 2) 
    {
      LOG_INFO("Linear solver", "Richarson");
      LOG_INFO("Preconditioning", this->Method());
    }
    this->name_ = "Richardson";
  }
  
  virtual ~Richardson() {}

  void InitParameter(std::string const &method, 
                     DataType omega,
                     bool do_parallel_naive,
                     bool fixed_maxits) 
  {
    assert (omega > 0.);
    assert (omega <= 1.);
    
    this->SetMethod(method);
    this->fixed_maxits_ = fixed_maxits;
    this->omega_ = omega;
    
    assert((this->Method() == "NoPreconditioning") 
        || (this->Method() == "Preconditioning") 
        || (this->Method() == "RightPreconditioning")
        || (this->Method() == "LeftPreconditioning"));
        
    this->do_preconditioning_ = (this->Method() != "NoPreconditioning");
    this->do_parallel_naive_ = do_parallel_naive;
  }

protected:
  LinearSolverState SolveImpl(const VectorType &b, VectorType *x)
  {
    LinearSolverState state = kSolverSuccess;
    if (this->do_parallel_naive_)
    {
      state = this->SolveImplApproxResidual(b, x);
    }
    else
    {
      state = this->SolveImplFullResidual(b, x);
    }
    return state;
  }
  
  void BuildImpl(VectorType const *b, VectorType *x) 
  {
    assert (b != nullptr);
    LinearSolver<LAD, PreLAD>::BuildImpl(b,x);
    this->r_.Clear();
    this->r_.CloneFromWithoutContent(*b);
    this->r_.Zeros();
    
    this->z_.Clear();
    this->z_.CloneFromWithoutContent(*b);
    this->z_.Zeros();
    
    this->xold_.Clear();
    this->xold_.CloneFromWithoutContent(*b);
    this->xold_.Zeros();
  }

  LinearSolverState SolveImplApproxResidual(const VectorType &b, VectorType *x);  
  LinearSolverState SolveImplFullResidual(const VectorType &b, VectorType *x);
  
  bool fixed_maxits_;
  bool do_preconditioning_;
  bool do_parallel_naive_;
  DataType omega_;
  
  VectorType r_;
  VectorType z_;
  VectorType xold_;

};

template <class LAD, class PreLAD>
LinearSolverState Richardson<LAD, PreLAD>::SolveImplFullResidual(const VectorType &b, VectorType *x) 
{
  assert(x->is_initialized());
  assert(b.is_initialized());
  assert(this->r_.is_initialized());
  assert(this->z_.is_initialized());
  
  assert((this->Method() == "NoPreconditioning") 
        || (this->Method() == "Preconditioning") 
        || (this->Method() == "RightPreconditioning")
        || (this->Method() == "LeftPreconditioning"));
        
  assert((this->Method() == "NoPreconditioning") || this->precond_ != nullptr); 
  assert(this->op_ != nullptr);

  if (this->print_level_ > 1) 
  {
    LOG_INFO(this->name_, "solve with " << this->Method());
  }

  if (this->fixed_maxits_)
  {
    // only number of iterations matters
    this->InitControl(this->maxits_, 1e-40, 1e-20, 1e10);
  }
    
  IterateControl::State conv = IterateControl::kIterate;

  x->Update();
  
  // initialization step
  this->iter_ = 0;

  // r = b - Ax
  this->r_.CopyFrom(b); 
  this->op_->VectorMultAdd(-1, *x, 1., &this->r_);
    
  if (!this->fixed_maxits_)
  {
    this->res_ = r_.Norm2();
  }
  else
  {
    this->res_ = 1.;
  }
  
  this->res_init_ = this->res_;
  this->res_rel_ = 1.;
  conv = this->control().Check(this->iter_, this->res_);

  if (this->print_level_ > 1) 
  {
    LOG_INFO(this->name_, "initial res norm   =  " << this->res_);
  }

  // main loop  
  while (conv == IterateControl::kIterate) 
  {
    ++(this->iter_);
    
    if (this->do_preconditioning_)
    {
      //x_new = x_old + omega * P * r
      this->ApplyPreconditioner(this->r_, &this->z_);
      x->Axpy(this->z_, this->omega_);
    }
    else
    {
      //x_new = x_old + omega * r
      x->Axpy(this->r_, this->omega_);
    }
    
    // update residual
    this->r_.CopyFrom(b); 
    this->op_->VectorMultAdd(-1, *x, 1., &this->r_);
    
    if (!this->fixed_maxits_)
    {
      this->res_ = this->r_.Norm2();
    }
    else
    {
      this->res_ = 1.; 
    }
    
    this->res_rel_ = this->res_ / this->res_init_;
    
    if (this->print_level_ > 2) 
    {
      if (!this->fixed_maxits_)
      {
        LOG_INFO(this->name_, "residual (iteration " << this->iter_ << "): " << this->res_);
      }
      else
      {
        LOG_INFO(this->name_, "residual (iteration " << this->iter_ 
                              << "): residual not computed, since fixed number of iterations is set" );
      }
    }

    conv = this->control().Check(this->iter_, this->res_);
    if (conv != IterateControl::kIterate) 
    {
      break;
    }
  }

  if (this->print_level_ > 1) 
  {
    LOG_INFO(this->name_, "final iterations   = " << this->iter_);
    LOG_INFO(this->name_, "final abs res norm = " << this->res_);
    LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_)
  } 

  // fixed number of iterations -> always success
  return kSolverSuccess;
}

template <class LAD, class PreLAD>
LinearSolverState Richardson<LAD, PreLAD>::SolveImplApproxResidual(const VectorType &b, VectorType *x) 
{
  assert(x->is_initialized());
  assert(b.is_initialized());
  assert(this->r_.is_initialized());
  assert(this->z_.is_initialized());
  
  assert(  (this->Method() == "Preconditioning") 
        || (this->Method() == "RightPreconditioning")
        || (this->Method() == "LeftPreconditioning"));
        
  assert(this->precond_ != nullptr); 
  assert(this->op_ != nullptr);

  if (this->print_level_ > 1) 
  {
    LOG_INFO(this->name_, "solve with " << this->Method());
  }

  // only number of iterations matters
  this->InitControl(this->maxits_, 1e-40, 1e-20, 1e10);
    
  x->Update();
  
  // initialization step
  this->iter_ = 0;
  
  this->res_init_ = 1.;
  this->res_rel_ = 1.;

  for (int i = 0; i < this->maxits_; ++i) 
  {
    this->xold_.CopyFrom(*x);

    // Compute z = op.offdiag * x
    this->z_.Zeros();
    this->op_->VectorMultOffdiag(*x, &(this->z_));

    // Compute r = b - z
    this->r_.CopyFrom(b);
    this->r_.Axpy(this->z_, static_cast< DataType >(-1.0));

    // Apply Preconditioner of diagonal block
    this->ApplyPreconditioner(this->r_, x);

    x->Scale(this->omega_);
    x->Axpy(this->xold_, static_cast< DataType >(1. - this->omega_));
    x->Update();
  }
  return kSolverSuccess;
}

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_CG_H_
