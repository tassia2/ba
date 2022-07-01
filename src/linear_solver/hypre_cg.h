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

#ifndef HIFLOW_LINEARSOLVER_HYPRE_CG_H_
#define HIFLOW_LINEARSOLVER_HYPRE_CG_H_

#include <cstdlib>
#include <mpi.h>

#include "common/log.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_solver/hypre_linear_solver.h"

namespace hiflow {
namespace la {
/// @author Simon Gawlok

/// @brief Wrapper class for CG implementation of Hypre
/// A linear solver is in particular a preconditioner.

template < class LAD > class HypreCG : public HypreLinearSolver< LAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  HypreCG();

  ~HypreCG();

  /// Clear allocated data
  void Clear();

  /// Initialize solver object
  void Init();

  /// Destroy solver object
  void DestroySolver();

#ifdef WITH_HYPRE
  /// Get pointer to solve function of preconditioner

  HYPRE_PtrToSolverFcn get_solve_function() {
    return (HYPRE_PtrToSolverFcn)HYPRE_ParCSRPCGSolve;
  }

  /// Get pointer to setup function of preconditioner

  HYPRE_PtrToSolverFcn get_setup_function() {
    return (HYPRE_PtrToSolverFcn)HYPRE_ParCSRPCGSetup;
  }

  /// Get hypre preconditioner object

  HYPRE_Solver &get_solver() { return this->solver_; }
#endif
protected:
  /// Build solver + preconditioner
  void BuildImpl(VectorType const *b, VectorType *x);

  /// Solves a linear system.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if solver succeeded
  LinearSolverState SolveImpl(const VectorType &b, VectorType *x);
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_HYPRE_CG_H_
