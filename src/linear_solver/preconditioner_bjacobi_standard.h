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

/// @author Dimitar Lukarski, Chandramowli Subramanian

#ifndef HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_STANDARD_H_
#define HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_STANDARD_H_

#include "linear_algebra/lmp/lpreconditioner.h"
#include "linear_solver/preconditioner_bjacobi.h"
#include "linear_solver/preconditioner.h"

namespace hiflow {
namespace la {

/// @brief Standard Block Jacobi preconditioners (Gauss-Seidel, SOR, and etc.)

template < class LAD >
class PreconditionerBlockJacobiStand : public PreconditionerBlockJacobi< LAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  PreconditionerBlockJacobiStand();
  virtual ~PreconditionerBlockJacobiStand();

  /// Setup the local operator for the local preconditioner
  virtual void SetupOperator(OperatorType &op);

  /// Sets up paramaters
  virtual void Init_Jacobi();
  virtual void Init_GaussSeidel();
  virtual void Init_SymmetricGaussSeidel();
  virtual void Init_SOR(DataType omega);
  virtual void Init_SSOR(DataType omega);
  virtual void Init_ILUp(int ilu_p);
  virtual void Init_FSAI(const int solver_max_iter, 
                         const DataType solver_rel_eps,
                         const DataType solver_abs_eps, 
                         const int matrix_power);
                         
  virtual void Init(const std::string& local_solver_type, 
                    const PropertyTree &params);
                                                 
  /// Erase possibly allocated data.
  virtual void Clear();

  virtual void Print(std::ostream &out = std::cout) const;

protected:
  /// Applies the preconditioner on the diagonal block.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if preconditioning succeeded
  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x);

  /// Build the preconditioner
  virtual void BuildImpl(VectorType const *b, VectorType *x);

  lPreconditioner< typename LAD::DataType > *localPrecond_;
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_STANDARD_H_
