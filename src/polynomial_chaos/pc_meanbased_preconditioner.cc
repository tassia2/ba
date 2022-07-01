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

/// \author Michael Schick

#include "polynomial_chaos/pc_meanbased_preconditioner.h"
#include "common/log.h"
#include <cassert>

namespace hiflow {
namespace polynomialchaos {

template < class LAD >
MeanbasedPreconditioner< LAD >::MeanbasedPreconditioner()
    : la::Preconditioner< LAD >() {}

template < class LAD >
MeanbasedPreconditioner< LAD >::~MeanbasedPreconditioner() {}

template < class LAD >
void MeanbasedPreconditioner< LAD >::BuildImpl(VectorType const *b,
                                               VectorType *x) {
  assert(this->op_ != NULL);

  // set the matrix to be used as the operator
  matrix_.CloneFrom(*this->op_->GetModes()->at(0));

#ifdef WITH_UMFPACK
  linear_solver_.Init_LU_umfpack();
#endif
  linear_solver_.SetupOperator(matrix_);
  linear_solver_.Build(b->Mode(0), x->Mode(0));

  this->SetModifiedOperator(false);
  this->SetState(true);
}

template < class LAD >
la::LinearSolverState
MeanbasedPreconditioner< LAD >::SolveImpl(const VectorType &b, VectorType *x) {
  for (int mode = 0; mode < b.NModes(); ++mode) {
    la::LinearSolverState state = this->linear_solver_.Solve(*b.Mode(mode), x->Mode(mode));
  
    if (state != la::kSolverSuccess)
    {
      return state;
    }
  }
  return la::kSolverSuccess;
}

/// template instantiation
template class MeanbasedPreconditioner< la::LADescriptorPolynomialChaosD >;

} // namespace polynomialchaos
} // namespace hiflow
