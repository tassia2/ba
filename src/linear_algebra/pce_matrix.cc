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

#include "linear_algebra/pce_matrix.h"

namespace hiflow {
namespace la {

// constructor

template < class LAD >
PCEMatrix< LAD >::PCEMatrix() {

  this->basis_matrix_.clear();
  this->nbasis_ = -1;

  // continue filling
}

// destructor

template < class LAD >
PCEMatrix< LAD >::~PCEMatrix() {

  this->Clear();
  this->basis_matrix_.clear();
}

// Inititialize

template < class LAD >
void PCEMatrix< LAD >::Init(PCTensor& pctensor, const MPI_Comm& comm,
                            const LaCouplings& cp) {
  // assign pctensor
  this->pctensor_ = pctensor;

  // calculate size of basis
  std::vector< int > bk = pctensor_.k_breakpoints();
  this->nbasis_ = bk[1] - bk[0];
  assert(this->nbasis_ > 0);

  // initialize the size of basis_matrix_
  this->basis_matrix_.resize(this->nbasis_);

  // initialize each HypreMatrix in basis_matrix_
  // currently, only use the mean_matrix to initialize the set of matrices,
  // it can be extended with a set of matrices by further implementation
  for (int i = 0; i != this->nbasis_; ++i) {
    this->basis_matrix_[i].Init(comm, cp);
  }
}

// accessing the member of basis_matrix_

template < class LAD >
typename LAD::MatrixType&
PCEMatrix< LAD >::BasisMode(const int i) {
  return this->basis_matrix_[i];
}

template < class LAD >
const typename LAD::MatrixType&
PCEMatrix< LAD >::GetBasisMode(const int i) const {
  return this->basis_matrix_[i];
}

// number of basis

template < class LAD >
int PCEMatrix< LAD >::nb_basis() const {
  assert(this->nbasis_ >= 0);
  return this->nbasis_;
}

// Zeros

template < class LAD >
void PCEMatrix< LAD >::Zeros() {
  assert(this->nbasis_ > 0);
  for (int i = 0; i != this->nbasis_; ++i) {
    this->basis_matrix_[i].Zeros();
  }
}

template < class LAD >
void PCEMatrix< LAD >::Zeros(const int i) {
  assert(this->nbasis_ > 0);
  this->basis_matrix_[i].Zeros();
}

// Clear
template < class LAD >
void PCEMatrix< LAD >::Clear() {
  assert(this->nbasis_ > 0);
  for (int i = 0; i != this->nbasis_; ++i) {
    this->basis_matrix_[i].Clear();
  }
}

// Clone
template < class LAD >
Matrix< typename LAD::DataType >* PCEMatrix< LAD >::Clone() const {
  LOG_ERROR("PCEMatrix::Clone not yet implemented!!!");
  exit(-1);
  return nullptr;
}

// VectorMult
template < class LAD >
void PCEMatrix< LAD >::VectorMult(PCEVector< LAD >& in,
                                  PCEVector< LAD >* out) const {
  assert(this->nbasis_ > 0);
  assert(in.nb_mode() == out->nb_mode());
  assert(this->pctensor_.Size() == in.nb_mode());

  out->Zeros();

  std::vector< int > bk = this->pctensor_.k_breakpoints();

  PVector vec_tmp;
  vec_tmp.CloneFromWithoutContent(in.Mode(0));
  vec_tmp.Zeros();

  for (int mode = 0; mode != this->pctensor_.Size(); ++mode) {

    for (int pos = bk[mode]; pos != bk[mode + 1]; ++pos) {

      vec_tmp.Zeros();

      std::vector< int > mode_idx = this->pctensor_.IndicesGlobal(pos);

      this->VectorMult(mode_idx[0], in.Mode(mode_idx[1]), &vec_tmp);

      out->Mode(mode).Axpy(vec_tmp, this->pctensor_.Val(pos));
    }
  }
}

template < class LAD >
void PCEMatrix< LAD >::VectorMult(PCEVector< LAD >& in,
                                  PCEVector< LAD >* out,
                                  const int l) const {
  assert(this->nbasis_ > 0);
  assert(in.nb_mode() == out->nb_mode());

  out->Zeros();

  const int mode_total = this->pctensor_.Size(l);

  assert(in.nb_mode() == mode_total);
  assert(out->nb_mode() == mode_total);

  std::vector< int > bk = this->pctensor_.k_breakpoints();

  PVector vec_tmp;
  vec_tmp.CloneFromWithoutContent(in.Mode(0));
  vec_tmp.Zeros();

  for (int mode = 0; mode != this->pctensor_.Size(); ++mode) {

    for (int pos = bk[mode]; pos != bk[mode + 1]; ++pos) {

      std::vector< int > mode_idx = this->pctensor_.IndicesGlobal(pos);

      if (mode_idx[0] < mode_total && mode_idx[1] < mode_total &&
          mode_idx[2] < mode_total) {

        vec_tmp.Zeros();

        this->VectorMult(mode_idx[0], in.Mode(mode_idx[1]), &vec_tmp);

        out->Mode(mode).Axpy(vec_tmp, this->pctensor_.Val(pos));
      }
    }
  }
}

template < class LAD >
void PCEMatrix< LAD >::VectorMult(const int i,
                                  PVector& in,
                                  PVector* out) const {
  assert(this->nbasis_ > 0);
  assert(this->nbasis_ > i);

  this->basis_matrix_[i].VectorMult(in, out);
}

template < class LAD >
void PCEMatrix< LAD >::VectorMult(Vector< PDataType > &in,
                                  Vector< PDataType > *out) const {
  assert(this->nbasis_ > 0);

  PCEVector< LAD >* hv_in;
  PCEVector< LAD >* hv_out;

  hv_in = dynamic_cast< PCEVector< LAD > * >(&in);
  hv_out = dynamic_cast< PCEVector< LAD > * >(out);

  if ((hv_in != 0) && (hv_out != 0)) {
    this->VectorMult(*hv_in, hv_out);
  } else {
    if (hv_in == 0) {
      LOG_ERROR("Called PCEMatrix::VectorMult with incompatible input vector "
                "type.");
    }
    if (hv_out == 0) {
      LOG_ERROR("Called PCEMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    exit(-1);
  }

}

// VectorMultAdd
template< class LAD >
void PCEMatrix< LAD >::VectorMultAdd(PDataType alpha, PCEVector<LAD>& in,
                                     PDataType beta, PCEVector<LAD>* out) const {


}

template < class LAD >
void PCEMatrix< LAD >::VectorMultAdd(PDataType alpha, Vector< PDataType > &in,
                                     PDataType beta, Vector< PDataType > *out) const {

  PCEVector<LAD>* hv_in;
  PCEVector<LAD>* hv_out;

  hv_in = dynamic_cast< PCEVector<LAD>* >(&in);
  hv_out = dynamic_cast< PCEVector<LAD>* >(out);

  if ((hv_in != 0) && (hv_out != 0)) {
    this->VectorMultAdd(alpha, *hv_in, beta, hv_out);
  } else {
    if (hv_in == 0) {
      LOG_ERROR("Called PCEMatrix::VectorMult with incompatible input vector "
                "type.");
    }
    if (hv_out == 0) {
      LOG_ERROR("Called PCEMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    exit(-1);
  }
}

// Global number of rows
template < class LAD >
int PCEMatrix< LAD >::num_rows_global() const {
  LOG_ERROR("PCEMatrix::num_rows_global() not yet implemented!!!");
  exit(-1);
  return -1;
}

// Global number of cols
template < class LAD >
int PCEMatrix< LAD >::num_cols_global() const {
  LOG_ERROR("PCEMatrix::num_cols_global() not yet implemented!!!");
  exit(-1);
  return -1;
}

// Local number of rows
template < class LAD >
int PCEMatrix< LAD >::num_rows_local() const {
  LOG_ERROR("PCEMatrix::num_rows_local() not yet implemented!!!");
  exit(-1);
  return -1;
}

// Local number of cols
template < class LAD >
int PCEMatrix< LAD >::num_cols_local() const {
  LOG_ERROR("PCEMatrix::num_cols_local() not yet implemented!!!");
  exit(-1);
  return -1;
}

// matrix-matrix mult
template < class LAD >
void PCEMatrix< LAD >::MatrixMult(Matrix< PDataType > &inA,
                                  Matrix< PDataType > &inB) {
  LOG_ERROR("PCEMatrix::MatrixMult not yet implemented!!!");
  exit(-1);
}

// GetValues
template < class LAD >
void PCEMatrix< LAD >::GetValues(const int *row_indices, const int num_rows,
                                 const int *col_indices, const int num_cols,
                                 PDataType *values) const {
  LOG_ERROR("PCEMatrix::GetValues not yet implemented!!!");
  exit(-1);
}

// Add
template < class LAD >
void PCEMatrix< LAD >::Add(const int global_row_id, const int global_col_id,
                           const PDataType value) {
  LOG_ERROR("PCEMatrix::Add not yet implemented!!!");
  exit(-1);
}

template < class LAD >
void PCEMatrix< LAD >::Add(const int *rows, const int num_rows, const int *cols,
                           const int num_cols, const PDataType *values) {
  LOG_ERROR("PCEMatrix::Add not yet implemented!!!");
  exit(-1);
}

// SetValue
template < class LAD >
void PCEMatrix< LAD >::SetValue(const int row, const int col,
                                const PDataType value) {
  LOG_ERROR("PCEMatrix::SetValue not yet implemented!!!");
  exit(-1);
}

// SetValues
template < class LAD >
void PCEMatrix< LAD >::SetValues(const int *row_indices, const int num_rows,
                                 const int *col_indices, const int num_cols,
                                 const PDataType *values) {
  LOG_ERROR("PCEMatrix::SetValues not yet implemented!!!");
  exit(-1);
}

// diagonalize_row
template < class LAD >
void PCEMatrix< LAD >::diagonalize_rows(const int *row_indices,
                                        const int num_rows,
                                        const PDataType diagonal_value) {
  for (int i = 0; i < this->nbasis_; ++i) {
    this->basis_matrix_[i].diagonalize_rows(row_indices,
                                            num_rows, diagonal_value);
  }
}

// Scale
template < class LAD >
void PCEMatrix< LAD >::Scale(const PDataType alpha) {
  for (int i = 0; i < this->nbasis_; ++i) {
    this->basis_matrix_[i].Scale(alpha);
  }
}

// Update
template < class LAD >
void PCEMatrix< LAD >::Update() {
  for (int i = 0; i < this->nbasis_; ++i) {
    this->basis_matrix_[i].Update();
  }
}

// begin_undate
template < class LAD >
void PCEMatrix< LAD >::begin_update() {
  for (int i = 0; i < this->nbasis_; ++i) {
    this->basis_matrix_[i].begin_update();
  }
}

// end_undate
template < class LAD >
void PCEMatrix< LAD >::end_update() {
  for (int i = 0; i < this->nbasis_; ++i) {
    this->basis_matrix_[i].end_update();
  }
}

// template instantiation
template class PCEMatrix< LADescriptorCoupledD >;
template class PCEMatrix< LADescriptorCoupledS >;

#ifdef WITH_HYPRE
template class PCEMatrix< LADescriptorHypreD >;
#endif

//#ifdef WITH_PETSC
//template class PCEMatrix< LADescriptorPETScD >;
//#endif

} // namespace la
} // namespace hiflow
