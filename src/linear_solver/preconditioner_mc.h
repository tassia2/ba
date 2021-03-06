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

/// \author Dimitar Lukarski, Chandramowli Subramanian

#ifndef HIFLOW_LINEARSOLVER_PRECONDITIONER_MC_H_
#define HIFLOW_LINEARSOLVER_PRECONDITIONER_MC_H_


#include "linear_solver/preconditioner_bjacobi.h"

namespace hiflow {
namespace doffem {
template <class DataType, int DIM> class DofPartition;
}

namespace la {

template < class DataType > class lPreconditioner_MultiColoring;

enum MultiColoringType {
  kMultiColoring0 = 0,
  kMultiColoring1
  // etc
};

/// \brief Block Jacobi preconditioner with multi coloring techniques.
///

template < class LAD, int DIM >
class PreconditionerMultiColoring : public PreconditionerBlockJacobi< LAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  /// standard constructor
  PreconditionerMultiColoring();
  /// destructor
  virtual ~PreconditionerMultiColoring();

  /// Creates the local multi coloring preconditioner.
  void Init_GaussSeidel();
  void Init_SymmetricGaussSeidel();
  //  void Init_SOR(const DataType omega);
  //  void Init_SSOR(const DataType omega);
  void Init_ILU(int fillins);
  void Init_ILU(int fillins, int power);

  /// Preprocessing, i.e. permute dof according to multi coloring algorithm
  /// @param op matrix which only needs to hold the structure of the diagonal
  /// blocks
  void Preprocess(const OperatorType &op, const VectorType &x,
                  doffem::DofPartition< DataType, DIM > *dof);

  /// Inits the operator
  /// @param op linear operator (now already permuted)
  void SetupOperator(OperatorType &op);

  /// Clears allocated data.
  void Clear();

  virtual void Print(std::ostream &out = std::cout) const;

private:
  /// Build the precond
  void BuildImpl(VectorType const *b, VectorType *x);

  /// Applies the multi coloring preconditioner.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if preconditioning succeeded
  LinearSolverState SolveImpl(const VectorType &b, VectorType *x);

  /// Permutes DoF numbering.
  void PermuteDofNumbering(const doffem::lDofId *perm,
                           doffem::DofPartition< DataType, DIM > *dof);

  OperatorType *op_; // associated linear operator
  lPreconditioner_MultiColoring< DataType > *lprecond_mc_;

  MultiColoringType mc_type_;
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_PRECONDITIONER_MC_H_
