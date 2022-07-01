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

#include "linear_solver/regularized_matrix_solver.h"

namespace hiflow {
namespace la {

template < class LAD > void RegularizedMatrixSolver< LAD >::Clear() {
  SingleSubBlockSolver< LAD >::Clear();
  this->regularization_matrix_.Clear();
  this->regularized_matrix_.Clear();

  this->auxiliary_matrices_initialized_ = false;
  this->regularization_parameter_ = static_cast< BDataType >(1);
}

template < class LAD >
void RegularizedMatrixSolver< LAD >::SetRegularizationMatrixAndParameter(
    const BMatrix &mat, const BDataType regularization_parameter) {
  this->regularization_matrix_.CloneFromWithoutContent(mat);
  this->regularized_matrix_.CloneFromWithoutContent(mat);

  this->regularization_parameter_ = regularization_parameter;

  this->regularization_matrix_.Axpy(mat, this->regularization_parameter_);

  this->auxiliary_matrices_initialized_ = true;
}

template < class LAD >
void RegularizedMatrixSolver< LAD >::BuildImpl(BlockVector< LAD > const *b,
                                               BlockVector< LAD > *x) {
  assert(this->op_ != nullptr);

  // cast to block matrix
  this->block_op_ = dynamic_cast< BlockMatrix< LAD > * >(this->op_);

  if (this->block_op_ == 0) {
    LOG_ERROR("Called RegularizedMatrixSolver::Build with incompatible input "
              "operator type.");
    quit_program();
  }

  // pass operator to sub solver
  if (!this->override_operator_) {
    // Build regularized matrix
    assert(this->auxiliary_matrices_initialized_);

    this->regularized_matrix_.Zeros();
    this->regularized_matrix_.Axpy(this->regularization_matrix_,
                                   static_cast< BDataType >(1));
    this->regularized_matrix_.Axpy(
        this->block_op_->GetBlock(this->block_nr_, this->block_nr_),
        static_cast< BDataType >(1));

    if (this->sub_solver_->GetPreconditioner() != nullptr) {
      this->sub_solver_->GetPreconditioner()->SetupOperator(
          this->regularized_matrix_);
    }
    this->sub_solver_->SetupOperator(this->regularized_matrix_);
    this->sub_solver_->Build(&(b->GetBlock(this->block_nr_)),
                             &(x->GetBlock(this->block_nr_)));
  }
}

// template instantiation
#ifdef WITH_HYPRE
template class RegularizedMatrixSolver< LADescriptorHypreD >;
#endif
} // namespace la
} // namespace hiflow
