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

/// \author Philipp Gerstner

#ifndef HIFLOW_LINEARSOLVER_SINGLE_SUBBLOCK_SOLVER_H_
#define HIFLOW_LINEARSOLVER_SINGLE_SUBBLOCK_SOLVER_H_

#include <map>
#include <mpi.h>
#include <vector>

#include "common/log.h"
#include "common/timer.h"
#include "config.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/block_vector.h"
#include "linear_solver/linear_solver.h"

namespace hiflow {
namespace la {

template < class LAD >
class SingleSubBlockSolver : public LinearSolver< LADescriptorBlock< LAD > > {
public:
  typedef typename LAD::MatrixType BMatrix;
  typedef typename LAD::VectorType BVector;
  typedef typename LAD::DataType BDataType;

  /// standard constructor

  SingleSubBlockSolver() : LinearSolver< LADescriptorBlock< LAD > >() {
    this->Clear();
  }

  /// destructor

  virtual ~SingleSubBlockSolver() { this->Clear(); }

  virtual void Clear();

  virtual void Init(size_t block_number) {
    this->block_nr_ = block_number;
    this->initialized_ = true;
  }

  virtual void SetSubSolver(LinearSolver< LAD > *solver,
                            bool override_operator) {
    this->sub_solver_ = solver;
    this->SetState(false);
    this->override_operator_ = override_operator;
  }

  /// @return residual

  virtual BDataType res() const {
    if (this->sub_solver_ != nullptr) {
      return this->sub_solver_->res();
    }
    return -1.;
  }

  /// @return Number of iterations for last solve.

  virtual int iter() const {
    if (this->sub_solver_ != nullptr) {
      return this->sub_solver_->iter();
    }

    return 0.;
  }

  LinearSolver< LAD > *GetSubSolver() { return this->sub_solver_; }

protected:
  /// Build the preconditioner, i.e. pass the operators to the subsolvers and
  /// build the subsolvers
  virtual void BuildImpl(BlockVector< LAD > const *b, BlockVector< LAD > *x);

  /// Applies the Schur complement solver.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if preconditioning succeeded
  virtual LinearSolverState SolveImpl(const BlockVector< LAD > &b,
                                      BlockVector< LAD > *x);

  BlockMatrix< LAD > *block_op_;

  LinearSolver< LAD > *sub_solver_;

  size_t block_nr_;
  bool override_operator_;
  bool initialized_;
};
} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_SCHUR_COMPLEMENT_LIGHT_H_
