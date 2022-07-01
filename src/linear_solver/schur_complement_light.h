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

#ifndef HIFLOW_LINEARSOLVER_SCHUR_COMPLEMENT_LIGHT_H_
#define HIFLOW_LINEARSOLVER_SCHUR_COMPLEMENT_LIGHT_H_

#include <map>
#include <mpi.h>
#include <vector>

#include "common/log.h"
#include "common/timer.h"
#include "config.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/block_vector.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_solver/linear_solver.h"

namespace hiflow {
namespace la {

/// This class provides functionality to precondition a two-block linear
/// system of equations by the Schur complement method: \n
/// Given \f$\left(\begin{array}{cc} A & B \\ C & D \end{array}\right)
/// \left(\begin{array}{cc} x \\ y \end{array}\right)
/// = \left(\begin{array}{cc} f \\ g \end{array}\right) \f$,
/// the system is solved by \n
/// 1. \f$Sy = g - CA^{-1}f\f$ \n
/// 2. \f$Ax = f - By\f$ \n
/// with the Schur complement matrix \f$S = D-CA^{-1}B\f$. \n
///
/// The user has to provide LinearOperators for B and C, as well as
/// LinearSolvers for A and S. \brief Schur complement solver interface

template < class LAD >
class SchurComplementLight : public LinearSolver< LADescriptorBlock< LAD > > {
public:
  typedef typename LAD::MatrixType BMatrix;
  typedef typename LAD::VectorType BVector;
  typedef typename LAD::DataType BDataType;

  typedef LADescriptorBlock< LAD > BlockLAD;
  typedef LADescriptorGeneral< BlockLAD > FreeBlockLAD;

  /// standard constructor

  SchurComplementLight() : LinearSolver< BlockLAD >() {
    this->Clear();
    this->aux_vec_init_ = false;
    this->reuse_vectors_ = false;
  }

  /// destructor

  virtual ~SchurComplementLight() {
    this->Clear();
    this->aux_vec_init_ = false;
    this->reuse_vectors_ = false;
  }

  virtual void Clear();

  virtual void SetupOperator(typename BlockLAD::MatrixType &op) {
    this->op_ = &op;
    this->SetModifiedOperator(true);

    this->A_modified_ = true;
    this->A_passed2solver_ = false;
  }

  virtual void Init(size_t num_blocks, std::vector< size_t > &block_one,
                    std::vector< size_t > &block_two);

  /// Setup solver for submatrix A_
  /// @param solver_A Solver object to solve with submatrix A_

  virtual void SetSolverA(LinearSolver< BlockLAD > *solver,
                          bool override_operator) {
    this->expl_solver_A_ = solver;
    this->override_operator_A_ = override_operator;
    this->SetState(false);
    this->use_explicit_A_ = true;
  }

  virtual void SetSolverA(LinearSolver< FreeBlockLAD > *solver) {
    this->impl_solver_A_ = solver;
    this->use_explicit_A_ = false;
  }

  /// Setup solver for schur complement matrix S
  /// @param solver Solver object to solve Schur complement operator S

  virtual void SetSolverS(LinearSolver< BlockLAD > *solver) {
    this->expl_solver_S_ = solver;
    this->use_explicit_S_ = true;
  }

  virtual void SetSolverS(LinearSolver< FreeBlockLAD, BlockLAD > *solver) {
    this->impl_solver_S_ = solver;
    this->use_explicit_S_ = false;
  }

  void SetLevel(int level) { this->level_ = level; }

  /// Set flag whether or not auxiliary vectors should be reused

  void SetReuseVectors(bool flag) { this->reuse_vectors_ = flag; }

  /// Deallocate auxiliary vectors

  inline void FreeVectors() {
    this->tmp_.Clear();
    this->aux_vec_init_ = false;
  }

  LinearSolver< BlockLAD > *GetExplSolverA() { return this->expl_solver_A_; }

  LinearSolver< BlockLAD > *GetExplSolverS() { return this->expl_solver_S_; }

  LinearSolver< FreeBlockLAD > *GetImplSolverA() {
    return this->impl_solver_A_;
  }

  LinearSolver< FreeBlockLAD, BlockLAD > *GetImplSolverS() {
    return this->impl_solver_S_;
  }

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

  /// Allocate auxiliary vectors

  inline void AllocateVectors(const BlockVector< LAD > &ref_vec) {
    if (!this->aux_vec_init_) {
      this->tmp_.CloneFromWithoutContent(ref_vec);
      this->aux_vec_init_ = true;
    }
  }

  /// Set all auxiliary vectors to zero

  inline void SetVectorsToZero() { this->tmp_.Zeros(); }

  virtual LinearSolverState SolveBlockA(const BlockVector< LAD > &b,
                                        BlockVector< LAD > *x);
  virtual LinearSolverState SolveBlockS(const BlockVector< LAD > &b,
                                        BlockVector< LAD > *x);

  BlockMatrix< LAD > *block_op_;

  BlockVector< LAD > tmp_;
  bool reuse_vectors_;
  bool aux_vec_init_;

  LinearSolver< BlockLAD > *expl_solver_A_;
  LinearSolver< BlockLAD > *expl_solver_S_;

  LinearSolver< FreeBlockLAD > *impl_solver_A_;
  LinearSolver< FreeBlockLAD, BlockLAD > *impl_solver_S_;

  bool override_operator_A_;

  std::vector< size_t > block_one_;
  std::vector< size_t > block_two_;

  bool initialized_;

  bool A_modified_;
  bool A_passed2solver_;

  bool use_explicit_A_;
  bool use_explicit_S_;

  std::vector< std::vector< bool > > active_blocks_B_;
  std::vector< std::vector< bool > > active_blocks_C_;

  std::vector< bool > active_blocks_two_;
  std::vector< bool > active_blocks_one_;

  int level_;
};
} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_SCHUR_COMPLEMENT_LIGHT_H_
