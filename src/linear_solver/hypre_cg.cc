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

#include "linear_solver/hypre_cg.h"

namespace hiflow {
namespace la {

template < class LAD > HypreCG< LAD >::HypreCG() : HypreLinearSolver< LAD >() {
  this->is_critical_hypre_solver_ = true;
  this->name_ = "Hypre_CG";
}

template < class LAD > HypreCG< LAD >::~HypreCG() { this->Clear(); }

template < class LAD > void HypreCG< LAD >::Init() {
  assert(this->op_ != nullptr);

#ifdef WITH_HYPRE
  if (this->initialized_) {
    HYPRE_ParCSRPCGDestroy(this->solver_);
  }
  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
  }
  MPI_Comm_dup(this->op_->comm(), &this->comm_);
  HYPRE_ParCSRPCGCreate(this->comm_, &(this->solver_));
  this->SetInitialized(true);
#endif
}

template < class LAD >
void HypreCG< LAD >::BuildImpl(VectorType const *b, VectorType *x) {
#ifdef WITH_HYPRE
  assert(this->op_ != nullptr);
  assert(b != nullptr);
  assert(x != nullptr);
  assert(x->is_initialized());
  assert(b->is_initialized());

  HYPRE_PCGSetMaxIter(this->solver_, this->maxits_); /* max iterations */
  if (this->print_level_ > 2) {
    LOG_INFO("Maximum iterations", this->maxits_);
  }
  HYPRE_PCGSetTol(this->solver_, this->reltol_); /* relative conv. tolerance */
  if (this->print_level_ > 2) {
    LOG_INFO("Relative tolerance [convergence]", this->reltol_);
  }
  HYPRE_PCGSetAbsoluteTol(this->solver_,
                          this->abstol_); /* absolute conv. tolerance */
  if (this->print_level_ > 2) {
    LOG_INFO("Absolute tolerance [convergence]", this->abstol_);
  }
  HYPRE_PCGSetTwoNorm(this->solver_,
                      1); /* use the two norm as the stopping criteria */

  // HYPRE_PCGSetPrintLevel(solver_, 2); /* print solve info */
  HYPRE_PCGSetLogging(this->solver_, 1); /* needed to get run info later */

  if (this->hypre_precond_ == nullptr) {
    HYPRE_ParCSRPCGSetup(this->solver_, *(this->op_->GetParCSRMatrix()),
                         *(b->GetParVector()), *(x->GetParVector()));
  } else {
    HYPRE_PCGSetPrecond(this->solver_,
                        this->hypre_precond_->get_solve_function(),
                        this->hypre_precond_->get_setup_function(),
                        this->hypre_precond_->get_solver());
    HYPRE_ParCSRPCGSetup(this->solver_, *(this->op_->GetParCSRMatrix()),
                         *(b->GetParVector()), *(x->GetParVector()));

    // TODO is the preconditioenr really already set up at this point?
    this->hypre_precond_->SetState(true);
    this->hypre_precond_->SetModifiedOperator(false);
  }
#endif
}

template < class LAD >
LinearSolverState HypreCG< LAD >::SolveImpl(const VectorType &b,
                                            VectorType *x) {
#ifdef WITH_HYPRE
  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "starts with residual norm " << this->res_);
  }
  HYPRE_ParCSRPCGSolve(this->solver_, *(this->op_->GetParCSRMatrix()),
                       *(b.GetParVector()), *(x->GetParVector()));
  HYPRE_Int iter_temp;
  HYPRE_Real res_temp, res_rel_temp;
  HYPRE_PCGGetNumIterations(this->solver_, &(iter_temp));
  HYPRE_PCGGetFinalRelativeResidualNorm(this->solver_, &(res_temp));
  HYPRE_PCGGetFinalRelativeResidualNorm(this->solver_, &(res_rel_temp));
  this->iter_ = static_cast< int >(iter_temp);
  this->res_ = static_cast< DataType >(res_temp);
  this->res_rel_ = static_cast< DataType >(res_rel_temp);

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "final iterations   = " << this->iter_);
    LOG_INFO(this->name_, "final abs res norm = " << this->res_);
    LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_);
  } 

  return kSolverSuccess;
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class LAD > void HypreCG< LAD >::DestroySolver() {
#ifdef WITH_HYPRE
  HYPRE_ParCSRPCGDestroy(this->solver_);

  this->SetInitialized(false);
  if (this->hypre_precond_ != nullptr) {
    if (this->hypre_precond_->IsCritical()) {
      this->hypre_precond_->DestroySolver();
    }
  }
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class LAD > void HypreCG< LAD >::Clear() {
#ifdef WITH_HYPRE
  if (this->initialized_) {
    HYPRE_ParCSRPCGDestroy(this->solver_);
  }
  HypreLinearSolver< LAD >::Clear();
  this->is_critical_hypre_solver_ = true;
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template class HypreCG< LADescriptorHypreD >;
} // namespace la
} // namespace hiflow
