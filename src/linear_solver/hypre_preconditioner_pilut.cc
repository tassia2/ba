// Copyright (C) 2011-2020 Vincent Heuveline
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

#include "linear_solver/hypre_preconditioner_pilut.h"

namespace hiflow {
namespace la {

template < class LAD >
HyprePreconditionerPILUT< LAD >::HyprePreconditionerPILUT()
    : HyprePreconditioner< LAD >() {}

template < class LAD >
HyprePreconditionerPILUT< LAD >::HyprePreconditionerPILUT(MPI_Comm &comm)
    : HyprePreconditioner< LAD >() {

#ifdef WITH_HYPRE
  MPI_Comm_dup(comm, &this->comm_);
  HYPRE_ParCSRPilutCreate(this->comm_, &(this->solver_));
  this->SetInitialized(true);
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class LAD >
HyprePreconditionerPILUT< LAD >::~HyprePreconditionerPILUT() {
  this->Clear();
}

template < class LAD >
void HyprePreconditionerPILUT< LAD >::Init(MPI_Comm &comm) {
  this->comm_ = MPI_COMM_NULL;
  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
  }
  this->SetInitialized(false);
#ifdef WITH_HYPRE
  MPI_Comm_dup(comm, &this->comm_);
  HYPRE_ParCSRPilutCreate(this->comm_, &(this->solver_));
  this->SetInitialized(true);
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

/*
template<class LAD>
void HyprePreconditionerPILUT<LAD>::SetupOperator(OperatorType& op) {
    LOG_ERROR("Operator is set by solver!");
    quit_program();
}
 */

template < class LAD >
LinearSolverState
HyprePreconditionerPILUT< LAD >::SolveImpl(const VectorType &b, VectorType *x) {
#ifdef WITH_HYPRE
  HYPRE_ParCSRPilutSolve(this->solver_, *(this->op_->GetParCSRMatrix()),
                         *(b.GetParVector()), *(x->GetParVector()));
  return kSolverSuccess;
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class LAD > void HyprePreconditionerPILUT< LAD >::Clear() {
#ifdef WITH_HYPRE
  if (this->initialized_) {
    HYPRE_ParCSRPilutDestroy(this->solver_);
  }
  HyprePreconditioner< LAD >::Clear();
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template class HyprePreconditionerPILUT< LADescriptorHypreD >;
} // namespace la
} // namespace hiflow
