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

#ifndef HIFLOW_LINEAR_ALGEBRA_SCHUR_OPERATOR_H
#define HIFLOW_LINEAR_ALGEBRA_SCHUR_OPERATOR_H

#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/block_vector.h"
#include "linear_algebra/linear_operator.h"
#include "linear_solver/linear_solver.h"
#include "linear_algebra/vector.h"

namespace hiflow {
namespace la {

/// \brief Implementation of matrix-vector product of Schur operator : D  - C *
/// A^{-1} * B

template < class LAD >
class SchurOperator : public LinearOperator< typename LAD::DataType > {
public:
  typedef typename LAD::MatrixType BMatrix;
  typedef typename LAD::VectorType BVector;
  typedef typename LAD::DataType BDataType;

  typedef LADescriptorBlock< LAD > BlockLAD;

  /// Standard Constructor

  SchurOperator() {
    this->Clear();

    this->aux_vec_init_ = false;
    this->reuse_vectors_ = false;

    this->tmp_ = new BlockVector< LAD >();
    this->tmp2_ = new BlockVector< LAD >();
  };
  /// Destructor

  virtual ~SchurOperator() {
    this->Clear();
    this->FreeVectors();

    delete this->tmp_;
    delete this->tmp2_;
  }

  virtual void Clear();

  virtual void Init(int num_blocks, std::vector< size_t > &block_one,
                    std::vector< size_t > &block_two);

  virtual void VectorMult(Vector< BDataType > &in,
                          Vector< BDataType > *out) const;

  /// out = this * in
  virtual void VectorMult(BlockVector< LAD > &in,
                          BlockVector< LAD > *out) const;

  /// set solver for A^{-1}

  virtual void SetSolverA(LinearSolver< BlockLAD > *solver,
                          bool override_operator) {
    this->solver_A_ = solver;
    this->override_operator_A_ = override_operator;

    if (this->block_op_ != nullptr) {
      this->PassOpA2Solver();
    }
  }

  virtual void SetBlockOperator(BlockMatrix< LAD > &op) {
    this->block_op_ = &op;
    this->A_modified_ = true;
    this->A_passed2solver_ = false;

    if (this->solver_A_ != nullptr) {
      this->PassOpA2Solver();
    }
  }

  /// Set flag whether or not auxiliary vectors should be reused

  virtual void SetReuseVectors(bool flag) { this->reuse_vectors_ = flag; }

  virtual void SetPrintLevel(int level) { this->print_level_ = level; }

  inline void FreeVectors() const {
    this->tmp_->Clear();
    this->tmp2_->Clear();
    this->aux_vec_init_ = false;
  }

  virtual LinearSolver< BlockLAD > *GetSolverA() { return this->solver_A_; }

  virtual bool IsInitialized() const {
    if (this->solver_A_ != nullptr) {
      if (this->block_op_ != nullptr) {
        if (this->block_op_->IsInitialized()) {
          if (this->called_init_) {
            return true;
          } else {
            LOG_DEBUG(0, "Init not called");
          }
        } else {
          LOG_DEBUG(0, "block_op not initialized");
        }
      } else {
        LOG_DEBUG(0, "block_op = nullptr");
      }
    } else {
      LOG_DEBUG(0, "solver_A = nullptr");
    }
    return false;
  }

protected:
  /// Allocate auxiliary vectors

  inline void AllocateVectors(const BlockVector< LAD > &ref_vec) const {
    if (!this->aux_vec_init_) {
      this->tmp_->CloneFromWithoutContent(ref_vec);
      this->tmp2_->CloneFromWithoutContent(ref_vec);
      this->aux_vec_init_ = true;
    }
  }

  /// Set all basis vectors to zero

  inline void SetVectorsToZero() const {
    this->tmp_->Zeros();
    this->tmp2_->Zeros();
  }

  virtual void PassOpA2Solver();

  BlockMatrix< LAD > *block_op_;

  mutable BlockVector< LAD > *tmp_;
  mutable BlockVector< LAD > *tmp2_;

  LinearSolver< BlockLAD > *solver_A_;

  bool override_operator_A_;

  std::vector< size_t > block_one_;
  std::vector< size_t > block_two_;

  bool called_init_;
  bool A_modified_;
  bool A_passed2solver_;

  mutable bool aux_vec_init_;
  bool reuse_vectors_;

  std::vector< bool > active_blocks_one_;
  std::vector< bool > active_blocks_two_;

  std::vector< std::vector< bool > > active_blocks_B_;
  std::vector< std::vector< bool > > active_blocks_C_;
  std::vector< std::vector< bool > > active_blocks_D_;

  int print_level_;

  mutable int num_A_;
  mutable int iter_A_;
  mutable double time_A_;
};

} // namespace la
} // namespace hiflow

#endif
