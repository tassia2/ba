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

/// @author Simon Gawlok, Martin Wlotzka

#ifndef HIFLOW_LINEAR_ALGEBRA_MATRIX_H
#define HIFLOW_LINEAR_ALGEBRA_MATRIX_H

#include "linear_algebra/linear_operator.h"
#include "linear_algebra/vector.h"

namespace hiflow {
namespace la {

/// \brief Abstract base class for distributed matrix implementations.

template < class DataType > class Matrix : public LinearOperator< DataType > {
public:
  /// Standard Constructor
  Matrix() { this->print_level_ = 0; };

  /// Destructor
  virtual ~Matrix() {}

  /// Clone this Matrix
  virtual Matrix< DataType > *Clone() const = 0;

  /// Global number of rows
  virtual int num_rows_global() const = 0;
  /// Global number of columns
  virtual int num_cols_global() const = 0;
  /// Local number of rows
  virtual int num_rows_local() const = 0;
  /// Local number of columns
  virtual int num_cols_local() const = 0;

  /// out = this * in
  virtual void VectorMult(Vector< DataType > &in,
                          Vector< DataType > *out) const = 0;

  /// this = inA * inB
  virtual void MatrixMult(Matrix< DataType > &inA, Matrix< DataType > &inB) = 0;

  /// out = beta * out + alpha * this * in
  virtual void VectorMultAdd(DataType alpha, Vector< DataType > &in,
                             DataType beta, Vector< DataType > *out) const = 0;

  /// Get values at specified indices
  virtual void GetValues(const int *row_indices, const int num_rows,
                         const int *col_indices, const int num_cols,
                         DataType *values) const = 0;

  // Mutating functions: after calling any of these, a call to
  // begin_update()/end_update() or update() must be made before
  // any other function can be called. It is, however, possible
  // to call the same mutating function several times in a row,
  // without calling update() in between.

  /// Add value to given indices
  virtual void Add(const int global_row_id, const int global_col_id,
                   const DataType value) = 0;

  /// \brief Add submatrix of values at positions (rows x cols).
  /// The row and column numbers are assumed to correspond to global dof ids.
  /// Size of values is assumed to be |rows| x |cols|.
  virtual void Add(const int *rows, const int num_rows, const int *cols,
                   const int num_cols, const DataType *values) = 0;

  /// \brief Add arbitrary values (in COO format)
  virtual void Add(const std::vector<int>& row_ind, 
                   const std::vector<int>& col_ind,
                   const std::vector<DataType>& values) {NOT_YET_IMPLEMENTED;};

  /// Set value at given indices
  virtual void SetValue(const int row, const int col, const DataType value) = 0;
  /// Set submatrix of values
  virtual void SetValues(const int *row_indices, const int num_rows,
                         const int *col_indices, const int num_cols,
                         const DataType *values) = 0;
  /// Set Matrix to zero
  virtual void Zeros() = 0;

  /// Sets rows to zero except the diagonal element to alpha.
  /// @param row_indices Global row indices (must be owned by this process)
  /// @param num_rows Size of array @em row_indices
  /// @param diagonal_value Value to be set for diagonal element
  virtual void diagonalize_rows(const int *row_indices, const int num_rows,
                                const DataType diagonal_value) = 0;

  virtual void set_unit_rows(const int *row_indices, const int num_rows)
  {
    this->diagonalize_rows(row_indices, num_rows, 1. );
  }

  /// Scale Matrix: this = alpha * this
  /// @param alpha Scaling factor
  virtual void Scale(const DataType alpha) = 0;

  /// Update values between different processes
  virtual void Update() = 0;
  /// Initiate update
  virtual void begin_update() = 0;
  /// Finalize update
  virtual void end_update() = 0;

  virtual void SetPrintLevel(int level) { this->print_level_ = level; }

  virtual bool has_ghost_rows() {return false;};
protected:
  int print_level_;
};

} // namespace la
} // namespace hiflow

#endif
