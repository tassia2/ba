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

/// \author Simon Gawlok

#ifndef HIFLOW_LINEARSOLVER_REGULARIZED_MATRIX_SOLVER_H_
#define HIFLOW_LINEARSOLVER_REGULARIZED_MATRIX_SOLVER_H_

#include <map>
#include <mpi.h>
#include <vector>

#include "common/log.h"
#include "common/timer.h"
#include "config.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/block_vector.h"
#include "linear_solver/linear_solver.h"

#include "linear_solver/single_subblock_solver.h"

namespace hiflow {
namespace la {

template < class LAD >
class RegularizedMatrixSolver : public SingleSubBlockSolver< LAD > {
public:
  typedef typename LAD::MatrixType BMatrix;
  typedef typename LAD::VectorType BVector;
  typedef typename LAD::DataType BDataType;

  /// standard constructor

  RegularizedMatrixSolver() : SingleSubBlockSolver< LAD >() {
    this->Clear();
    this->name_ = "RegMatrixSolver";
  }

  /// destructor

  virtual ~RegularizedMatrixSolver() { this->Clear(); }

  void
  SetRegularizationMatrixAndParameter(const BMatrix &mat,
                                      const BDataType regularization_parameter);

  virtual void Clear();

protected:
  /// Build the preconditioner, i.e. pass the operators to the subsolvers and
  /// build the subsolvers
  virtual void BuildImpl(BlockVector< LAD > const *b, BlockVector< LAD > *x);

  BMatrix regularization_matrix_;
  BMatrix regularized_matrix_;
  bool auxiliary_matrices_initialized_;
  BDataType regularization_parameter_;
};
} // namespace la
} // namespace hiflow

#endif /* HIFLOW_LINEARSOLVER_REGULARIZED_MATRIX_SOLVER_H_ */
