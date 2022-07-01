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

#ifndef HIFLOW_LINEARSOLVER_PCD_PRECONDITIONER_H_
#define HIFLOW_LINEARSOLVER_PCD_PRECONDITIONER_H_

#include <map>
#include <mpi.h>
#include <vector>

#include "common/log.h"
#include "common/timer.h"
#include "config.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/linear_operator.h"
#include "linear_solver/linear_solver.h"

namespace hiflow {
namespace la {
/// \brief Implementation of abstract pressure convection diffusion
/// preconditioner <br> PCD(x) = s_D * D^{1} (x) + s_QFH * Q^{-1} F H^{-1} x

template < class LAD > class PCDPreconditioner : public LinearSolver< LAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  /// standard constructor

  PCDPreconditioner() : LinearSolver< LAD >() { this->Clear(); }

  /// destructor

  virtual ~PCDPreconditioner() { this->Clear(); }

  virtual void Clear();

  virtual void set_scaling(DataType s_D, DataType s_QFH) {
    this->s_D_ = s_D;
    this->s_QFH_ = s_QFH;
  }

  /// Set operator F
  /// @param

  virtual void SetOperatorF(LinearOperator< DataType > const *op) {
    assert(op != nullptr);
    this->op_F_ = op;
    this->SetState(false);
  }

  /// Set solver Q

  virtual void SetSolverQ(LinearSolver< LAD > *solver) {
    assert(solver != nullptr);
    this->solver_Q_ = solver;
    this->SetState(false);
  }

  /// Set solver H

  virtual void SetSolverH(LinearSolver< LAD > *solver) {
    assert(solver != nullptr);
    this->solver_H_ = solver;
    this->SetState(false);
  }

  /// Set solver D

  virtual void SetSolverD(LinearSolver< LAD > *solver) {
    assert(solver != nullptr);
    this->solver_D_ = solver;
    this->SetState(false);
  }

  /// Set flag whether or not auxiliary vectors should be reused

  void SetReuseVectors(bool flag) { this->reuse_vectors_ = flag; }

  /// Deallocate auxiliary vectors

  inline void FreeVectors() {
    this->tmp_.Clear();
    this->aux_vec_init_ = false;
  }

  LinearSolver< LAD > *GetSolverQ() { return this->solver_Q_; }

  LinearSolver< LAD > *GetSolverH() { return this->solver_H_; }

  LinearSolver< LAD > *GetSolverD() { return this->solver_D_; }

protected:
  /// Build the preconditioner, i.e. pass the operators to the subsolvers and
  /// build the subsolvers
  virtual void BuildImpl(VectorType const *b, VectorType *x);

  /// Applies the Schur complement solver.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if preconditioning succeeded
  virtual LinearSolverState SolveImpl(const Vector< DataType > &b,
                                      Vector< DataType > *x);

  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x);

  /// Allocate auxiliary vectors

  inline void AllocateVectors(const VectorType &ref_vec) {
    if (!this->aux_vec_init_) {
      this->tmp_.CloneFromWithoutContent(ref_vec);
      this->aux_vec_init_ = true;
    }
  }

  /// Set all auxiliary vectors to zero

  inline void SetVectorsToZero() { this->tmp_.Zeros(); }

  LinearOperator< DataType > const *op_F_;

  LinearSolver< LAD > *solver_Q_;
  LinearSolver< LAD > *solver_H_;
  LinearSolver< LAD > *solver_D_;

  DataType s_D_;
  DataType s_QFH_;

  VectorType tmp_;
  bool reuse_vectors_;
  bool aux_vec_init_;
};
} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_PCD_PRECONDITIONER_H_
