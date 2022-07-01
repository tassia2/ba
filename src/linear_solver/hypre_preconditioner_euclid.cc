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

#include "linear_solver/hypre_preconditioner_euclid.h"

namespace hiflow {
namespace la {

template < class LAD >
HyprePreconditionerEuclid< LAD >::HyprePreconditionerEuclid(MPI_Comm &comm)
    : HyprePreconditioner< LAD >() 
{
  this->Init(comm);
}

template < class LAD >
HyprePreconditionerEuclid< LAD >::HyprePreconditionerEuclid()
    : HyprePreconditioner< LAD >() 
{
  this->SetInitialized(false);
}

template < class LAD >
HyprePreconditionerEuclid< LAD >::~HyprePreconditionerEuclid() {
  this->Clear();
}

template < class LAD >
void HyprePreconditionerEuclid< LAD >::Init(MPI_Comm &comm) {
  assert(this->op_ != NULL);

#ifdef WITH_HYPRE
  if (this->initialized_) {
    HYPRE_EuclidDestroy(this->solver_);
  }
  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
  }
  MPI_Comm_dup(this->op_->comm(), &this->comm_);
  HYPRE_EuclidCreate(this->comm_, &(this->solver_));
  HYPRE_EuclidSetStats(this->solver_, 100);
  this->SetInitialized(true);
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  exit(-1);
#endif

}

template < class LAD >
LinearSolverState HyprePreconditionerEuclid< LAD >::SolveImpl(const VectorType &b,
                                                              VectorType *x) {
#ifdef WITH_HYPRE
  HYPRE_EuclidSolve(this->solver_, *(this->op_->GetParCSRMatrix()),
                    *(b.GetParVector()), *(x->GetParVector()));
  return kSolverSuccess;
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class LAD > 
void HyprePreconditionerEuclid< LAD >::Clear() {
#ifdef WITH_HYPRE
  if (this->initialized_) {
    HYPRE_EuclidDestroy(this->solver_);
  }
  HyprePreconditioner< LAD >::Clear();
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template class HyprePreconditionerEuclid< LADescriptorHypreD >;
} // namespace la
} // namespace hiflow
