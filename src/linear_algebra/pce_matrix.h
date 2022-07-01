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

#ifndef HIFLOW_LINEARALGEBRA_PCE_MATRIX_H_
#define HIFLOW_LINEARALGEBRA_PCE_MATRIX_H_

#include "config.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/pce_vector.h"
#include "linear_algebra/matrix.h"
#include "linear_algebra/vector.h"
#include "polynomial_chaos/pc_tensor.h"

using namespace hiflow::polynomialchaos;

namespace hiflow {
namespace la {

template < class LAD >
class PCEMatrix : public Matrix< typename LAD::DataType > {
public:
  typedef typename LAD::MatrixType PMatrix;
  typedef typename LAD::VectorType PVector;
  typedef typename LAD::DataType PDataType;

  // constructor
  PCEMatrix();

  // destructor
  ~PCEMatrix();

  // Inititialize
  void Init(PCTensor& pctensor, const MPI_Comm& comm, const LaCouplings& cp);

  // define operator to access member of basis_matrix_
  PMatrix& BasisMode(const int i);
  const PMatrix& GetBasisMode(const int i) const;

  // number of basis matrices
  int nb_basis() const;

  // Zeros
  void Zeros();
  void Zeros(const int i);

  // Clear
  void Clear();

  // Clone
  virtual Matrix< PDataType > *Clone() const;

  // VectorMult
  void VectorMult(PCEVector< LAD >& in, PCEVector< LAD >* out) const;
  void VectorMult(PCEVector< LAD >& in, PCEVector< LAD >* out,
                  const int l) const;
  void VectorMult(const int i, PVector& in, PVector* out) const;
  void VectorMult(Vector< PDataType > &in, Vector< PDataType > *out) const;

  // VectorMultAdd
  void VectorMultAdd(PDataType alpha, PCEVector< LAD >& in, PDataType beta,
                     PCEVector< LAD >* out) const;
  void VectorMultAdd(PDataType alpha, PCEVector< LAD >& in, PDataType beta,
                     PCEVector< LAD >* out, const int l) const;
  void VectorMultAdd(PDataType alpha, Vector< PDataType >& in, PDataType beta,
                     Vector< PDataType >* out) const;

  /// @return Global number of rows
  int num_rows_global() const;
  /// @return Global number of columns
  int num_cols_global() const;
  /// @return Local number of rows
  int num_rows_local() const;
  /// @return Local number of columns
  int num_cols_local() const;

  /// this = inA * inB
  void MatrixMult(Matrix< PDataType > &inA, Matrix< PDataType > &inB);

  /// Get values at specified indices
  void GetValues(const int *row_indices, const int num_rows,
                 const int *col_indices, const int num_cols,
                 PDataType *values) const;

  /// Add value to given indices
  void Add(const int global_row_id, const int global_col_id,
           const PDataType value);

  void Add(const int *rows, const int num_rows, const int *cols,
           const int num_cols, const PDataType *values);

  /// Set value at given indices
  void SetValue(const int row, const int col, const PDataType value);
  /// Set submatrix of values
  void SetValues(const int *row_indices, const int num_rows,
                 const int *col_indices, const int num_cols,
                 const PDataType *values);

  /// Sets rows to zero except the diagonal element to alpha.
  void diagonalize_rows(const int *row_indices, const int num_rows,
                        const PDataType diagonal_value);

  /// Scale Matrix: this = alpha * this
  /// @param alpha Scaling factor
  void Scale(const PDataType alpha);

  // Update matrix entries
  void Update();

  /// Initiate update
  void begin_update();

  /// Finalize update
  void end_update();

private:
  // a set of basis matrices
  std::vector< PMatrix > basis_matrix_;

  // pc tensor
  PCTensor pctensor_;

  // size of basis
  int nbasis_;
};

template < class LAD >
class LADescriptorPCE {
public:
  typedef PCEMatrix< LAD > MatrixType;
  typedef PCEVector< LAD > VectorType;
  typedef typename LAD::DataType DataType;
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARALGEBRA_PCE_MATRIX_H_
