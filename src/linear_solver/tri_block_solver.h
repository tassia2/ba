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

#ifndef HIFLOW_LINEARSOLVER_TRIBLOCK_LIGHT_H_
#define HIFLOW_LINEARSOLVER_TRIBLOCK_LIGHT_H_

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

enum TriangularStructure {
  UPPER_TRIANGULAR = 0,
  LOWER_TRIANGULAR = 1,
  DIAGONAL = 2
};

/// This class provides functionality to precondition a two-block linear
/// system of equations in triangular form: \n
/// Given \f$\left(\begin{array}{cc} A & B \\ C & D \end{array}\right)
/// \left(\begin{array}{cc} x \\ y \end{array}\right)
/// = \left(\begin{array}{cc} f \\ g \end{array}\right) \f$,
/// where either B (case 1) or C (case 2) or both (case 3) are zero,
/// the system is solved by \n
/// case 1: \n
/// 1. \f$Ax = f \f$ \n
/// 2. \f$Dy = g - Cx\f$ \n
/// case 2: \n
/// 1. \f$Dy = g \f$ \n
/// 2. \f$Ax = f - By\f$ \n
/// case 3: \n
/// 1. \f$Ax = f \f$ \n
/// 2. \f$Dy = g \f$ \n
/// The user has to provide LinearOperators for B and C, as well as
/// LinearSolvers for A and D. \brief TriBlock solver interface

template < class LAD >
class TriBlockSolver : public LinearSolver< LADescriptorBlock< LAD > > {
public:
  typedef typename LAD::MatrixType BMatrix;
  typedef typename LAD::VectorType BVector;
  typedef typename LAD::DataType BDataType;

  typedef LADescriptorBlock< LAD > BlockLAD;
  typedef LADescriptorGeneral< BlockLAD > FreeBlockLAD;

  /// standard constructor

  TriBlockSolver() : LinearSolver< LADescriptorBlock< LAD > >() {
    this->Clear();
    this->aux_vec_init_ = false;
    this->reuse_vectors_ = false;
  }

  /// destructor

  virtual ~TriBlockSolver() {
    this->Clear();
    this->aux_vec_init_ = false;
    this->reuse_vectors_ = false;
  }

  virtual void Clear();

  virtual void Init(int num_blocks, const std::vector< int > &block_one,
                    const std::vector< int > &block_two,
                    TriangularStructure type);

  virtual void SetupOperator(typename BlockLAD::MatrixType &op) {
    this->op_ = &op;
    this->SetModifiedOperator(true);

    this->A_modified_ = true;
    this->D_modified_ = true;

    this->A_passed2solver_ = false;
    this->D_passed2solver_ = false;
  }

  /// Setup solver for submatrix A_
  /// @param solver_A Solver object to solve with submatrix A_

  virtual void SetSolverA(LinearSolver< BlockLAD > *solver,
                          bool override_operator) {
    assert (solver != nullptr);
    this->expl_solver_A_ = solver;
    this->override_operator_A_ = override_operator;
    this->SetState(false);
    this->use_explicit_A_ = true;
  }

  virtual void SetSolverA(LinearSolver< FreeBlockLAD > *solver) {
    assert (solver != nullptr);
    this->impl_solver_A_ = solver;
    this->use_explicit_A_ = false;
  }

  /// Setup solver for submatrix D_
  /// @param solver_D Solver object to solve with submatrix D_

  virtual void SetSolverD(LinearSolver< BlockLAD > *solver,
                          bool override_operator) {
    assert (solver != nullptr);
    this->expl_solver_D_ = solver;
    this->override_operator_D_ = override_operator;
    this->SetState(false);
    this->use_explicit_D_ = true;
  }

  virtual void SetSolverD(LinearSolver< FreeBlockLAD > *solver) {
    assert (solver != nullptr);
    this->impl_solver_D_ = solver;
    this->use_explicit_D_ = false;
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

  LinearSolver< BlockLAD > *GetExplSolverD() { return this->expl_solver_D_; }

  LinearSolver< FreeBlockLAD > *GetImplSolverA() {
    return this->impl_solver_A_;
  }

  LinearSolver< FreeBlockLAD > *GetImplSolverD() {
    return this->impl_solver_D_;
  }

protected:
  /// Build the preconditioner, i.e. pass the operators to the subsolvers and
  /// build the subsolvers
  virtual void BuildImpl(BlockVector< LAD > const *b, BlockVector< LAD > *x);

  /// Applies the Schur complement solver.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if preconditioning succeeded
  //            virtual LinearSolverState SolveImpl ( const Vector<BDataType>&
  //            b, Vector<BDataType>* x );

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

  virtual void PassOpA2Solver(BlockVector< LAD > const *b,
                              BlockVector< LAD > *x);
  virtual void PassOpD2Solver(BlockVector< LAD > const *b,
                              BlockVector< LAD > *x);

  virtual LinearSolverState SolveBlockA(const BlockVector< LAD > &b,
                                        BlockVector< LAD > *x);
  virtual LinearSolverState SolveBlockD(const BlockVector< LAD > &b,
                                        BlockVector< LAD > *x);

  BlockMatrix< LAD > *block_op_;

  BlockVector< LAD > tmp_;
  bool reuse_vectors_;
  bool aux_vec_init_;

  LinearSolver< BlockLAD > *expl_solver_A_;
  LinearSolver< BlockLAD > *expl_solver_D_;

  LinearSolver< FreeBlockLAD > *impl_solver_A_;
  LinearSolver< FreeBlockLAD > *impl_solver_D_;

  bool override_operator_A_;
  bool override_operator_D_;

  std::vector< int > block_one_;
  std::vector< int > block_two_;

  bool initialized_;

  bool A_modified_;
  bool A_passed2solver_;

  bool D_modified_;
  bool D_passed2solver_;

  bool use_explicit_A_;
  bool use_explicit_D_;

  std::vector< std::vector< bool > > active_blocks_B_;
  std::vector< std::vector< bool > > active_blocks_C_;

  std::vector< bool > active_blocks_two_;
  std::vector< bool > active_blocks_one_;

  int level_;
  TriangularStructure structure_;
};
} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_TRIBLOCK_LIGHT_H_
