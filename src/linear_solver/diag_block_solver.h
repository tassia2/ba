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

#ifndef HIFLOW_LINEARSOLVER_DIAGBLOCK_H_
#define HIFLOW_LINEARSOLVER_DIAGBLOCK_H_

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
#include "linear_solver/single_subblock_solver.h"

namespace hiflow {
namespace la {

template < class LAD >
class DiagBlockSolver : public LinearSolver< LADescriptorBlock< LAD > > {
public:
  typedef typename LAD::MatrixType BMatrix;
  typedef typename LAD::VectorType BVector;
  typedef typename LAD::DataType BDataType;

  typedef LADescriptorBlock< LAD > BlockLAD;
  typedef LADescriptorGeneral< BlockLAD > FreeBlockLAD;

  /// standard constructor

  DiagBlockSolver() : LinearSolver< LADescriptorBlock< LAD > >() {
    this->Clear();
  }

  /// destructor

  virtual ~DiagBlockSolver() { this->Clear(); }

  virtual void Clear();

  virtual void Init(int num_blocks,
                    const std::vector< std::vector< int > > &block_ids);

  /// Setup solver for submatrix A_
  /// @param solver_A Solver object to solve with submatrix A_

  virtual void SetSingleBlockSolver(int block_nr, LinearSolver< LAD > *solver,
                                    bool override_operator) {
    assert(this->initialized_);
    assert(block_nr >= 0);
    assert(block_nr < num_sub_solvers_);

    assert(this->blocks_[block_nr].size() == 1);

    SingleSubBlockSolver< LAD > *block_solver = new SingleSubBlockSolver< LAD >;
    block_solver->Init(this->blocks_[block_nr][0]);
    block_solver->SetSubSolver(solver, override_operator);

    this->expl_solver_[block_nr] = block_solver;
    this->override_operator_[block_nr] = override_operator;
    this->use_explicit_solver_[block_nr] = true;

    // TODO delete block_solver object

    this->SetState(false);
  }

  virtual void SetExplicitBlockSolver(int block_nr,
                                      LinearSolver< BlockLAD > *solver,
                                      bool override_operator) {
    assert(this->initialized_);
    assert(block_nr >= 0);
    assert(block_nr < num_sub_solvers_);

    this->expl_solver_[block_nr] = solver;
    this->override_operator_[block_nr] = override_operator;
    this->use_explicit_solver_[block_nr] = true;

    this->SetState(false);
  }

  virtual void SetImplicitBlockSolver(int block_nr,
                                      LinearSolver< FreeBlockLAD > *solver) {
    assert(this->initialized_);
    assert(block_nr >= 0);
    assert(block_nr < num_sub_solvers_);

    this->impl_solver_[block_nr] = solver;
    this->use_explicit_solver_[block_nr] = false;
  }

  void SetLevel(int level) { this->level_ = level; }

  LinearSolver< BlockLAD > *GetExplSubSolver(int block_nr) {
    assert(block_nr >= 0);
    assert(block_nr < num_sub_solvers_);

    return this->expl_solver_[block_nr];
  }

  LinearSolver< FreeBlockLAD > *GetImplSubSolver(int block_nr) {
    assert(block_nr >= 0);
    assert(block_nr < num_sub_solvers_);

    return this->impl_solver_[block_nr];
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

  virtual LinearSolverState
  SolveBlock(int block_nr, const BlockVector< LAD > &b, BlockVector< LAD > *x);

  BlockMatrix< LAD > *block_op_;

  std::vector< LinearSolver< BlockLAD > * > expl_solver_;

  std::vector< LinearSolver< FreeBlockLAD > * > impl_solver_;

  std::vector< bool > override_operator_;

  bool initialized_;

  std::vector< bool > op_modified_;
  std::vector< bool > op_passed2solver_;

  std::vector< std::vector< int > > blocks_;

  std::vector< bool > use_explicit_solver_;

  int num_sub_solvers_;
  std::vector< std::vector< bool > > active_blocks_;

  int level_;
};
} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_TRIBLOCK_LIGHT_H_
