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

/// @author Dimitar Lukarski

#include "config.h"

#include "CPUmkl_coo_lmatrix.h"
#include "CPUmkl_blas_routines.h"
#include "../lmp_log.h"

#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <typeinfo>

#ifdef WITH_MKL
#include <mkl.h>
#include <mkl_spblas.h>
#endif

using namespace hiflow::la;

template < typename ValueType >
CPUmkl_COO_lMatrix< ValueType >::CPUmkl_COO_lMatrix(int init_nnz,
                                                    int init_num_row,
                                                    int init_num_col,
                                                    std::string init_name) {
#ifdef WITH_MKL
  this->Init(init_nnz, init_num_row, init_num_col, init_name);
  this->implementation_name_ = "Intel MKL";
  this->implementation_id_ = MKL;
  this->set_num_threads();
#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
CPUmkl_COO_lMatrix< ValueType >::CPUmkl_COO_lMatrix() {
#ifdef WITH_MKL
  this->implementation_name_ = "Intel MKL";
  this->implementation_id_ = MKL;
  this->set_num_threads();
#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
CPUmkl_COO_lMatrix< ValueType >::~CPUmkl_COO_lMatrix() {}

template < typename ValueType >
void CPUmkl_COO_lMatrix< ValueType >::CloneFrom(
    const hiflow::la::lMatrix< ValueType > &other) {
  if (this != &other) {
    // if it is not empty init() will clean it
    this->Init(other.get_nnz(), other.get_num_row(), other.get_num_col(),
               other.get_name());

    this->CopyStructureFrom(other);

    this->CopyFrom(other);
  }

  const CPUmkl_COO_lMatrix< ValueType > *mkl_other =
      dynamic_cast< const CPUmkl_COO_lMatrix< ValueType > * >(&other);
  if (mkl_other != 0) {
    this->set_num_threads(mkl_other->num_threads());
  } else {
    const CPUopenmp_COO_lMatrix< ValueType > *omp_other =
        dynamic_cast< const CPUopenmp_COO_lMatrix< ValueType > * >(&other);
    if (omp_other != 0) {
      this->set_num_threads(omp_other->num_threads());
    }
    // no else
  }
}

template < typename ValueType >
void CPUmkl_COO_lMatrix< ValueType >::set_num_threads(void) {
#ifdef WITH_MKL
  // default value
  this->set_num_threads(8);
#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
void CPUmkl_COO_lMatrix< ValueType >::set_num_threads(int num_thread) {
#ifdef WITH_MKL
  this->num_threads_ = num_thread;
#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
void CPUmkl_COO_lMatrix< ValueType >::VectorMult(
    const lVector< ValueType > &invec, lVector< ValueType > *outvec) const {

  const CPU_lVector< ValueType > *casted_invec =
      dynamic_cast< const CPU_lVector< ValueType > * >(&invec);
  CPU_lVector< ValueType > *casted_outvec =
      dynamic_cast< CPU_lVector< ValueType > * >(outvec);

  if ((casted_invec == NULL) && (casted_outvec == NULL)) {
    LOG_ERROR("CPUmkl_COO_lMatrix<ValueType>::VectorMult unsupported in or out "
              "vector");
    this->print();
    invec.print();
    outvec->print();
    exit(-1);
  }

  this->VectorMult(*casted_invec, casted_outvec);
}

template <>
void CPUmkl_COO_lMatrix< double >::VectorMult(
    const CPU_lVector< double > &invec, CPU_lVector< double > *outvec) const {
#ifdef WITH_MKL

  assert(invec.get_size() > 0);
  assert(outvec->get_size() > 0);
  assert(invec.get_size() == this->get_num_col());
  assert(outvec->get_size() == this->get_num_row());

  char transa = 'N';

  int nrow = this->get_num_row();

  mkl_set_num_threads(this->num_threads_);

  mkl_cspblas_dcoogemv(&transa, &nrow, this->matrix.val, this->matrix.row,
                       this->matrix.col, (int *)&(this->nnz_), invec.buffer,
                       outvec->buffer);

#else
  NO_MKL_ERROR;
#endif
}

template <>
void CPUmkl_COO_lMatrix< float >::VectorMult(
    const CPU_lVector< float > &invec, CPU_lVector< float > *outvec) const {
#ifdef WITH_MKL

  assert(invec.get_size() > 0);
  assert(outvec->get_size() > 0);
  assert(invec.get_size() == this->get_num_col());
  assert(outvec->get_size() == this->get_num_row());

  LOG_ERROR("CPUmkl_coo_matrix::VectorMult there is no float sparse "
            "matrix-vector multiplication function");
  exit(-1);

#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
void CPUmkl_COO_lMatrix< ValueType >::CastFrom(
    const CPU_COO_lMatrix< double > &other) {
  assert(this->get_num_row() == other.get_num_row());
  assert(this->get_num_col() == other.get_num_col());
  assert(this->get_nnz() == other.get_nnz());

  for (int i = 0, e_i = this->get_nnz(); i != e_i; ++i) {
    this->matrix.val[i] = static_cast< ValueType >(other.matrix.val[i]);
  }
}

template < typename ValueType >
void CPUmkl_COO_lMatrix< ValueType >::CastFrom(
    const CPU_COO_lMatrix< float > &other) {
  assert(this->get_num_row() == other.get_num_row());
  assert(this->get_num_col() == other.get_num_col());
  assert(this->get_nnz() == other.get_nnz());

  for (int i = 0, e_i = this->get_nnz(); i != e_i; ++i) {
    this->matrix.val[i] = static_cast< ValueType >(other.matrix.val[i]);
  }
}

template < typename ValueType >
void CPUmkl_COO_lMatrix< ValueType >::CastTo(
    CPU_COO_lMatrix< double > &other) const {
  assert(this->get_num_row() == other.get_num_row());
  assert(this->get_num_col() == other.get_num_col());
  assert(this->get_nnz() == other.get_nnz());

  for (int i = 0, e_i = this->get_nnz(); i != e_i; ++i) {
    other.matrix.val[i] = static_cast< double >(this->matrix.val[i]);
  }
}

template < typename ValueType >
void CPUmkl_COO_lMatrix< ValueType >::CastTo(
    CPU_COO_lMatrix< float > &other) const {
  assert(this->get_num_row() == other.get_num_row());
  assert(this->get_num_col() == other.get_num_col());
  assert(this->get_nnz() == other.get_nnz());

  for (int i = 0, e_i = this->get_nnz(); i != e_i; ++i) {
    other.matrix.val[i] = static_cast< float >(this->matrix.val[i]);
  }
}

template class CPUmkl_COO_lMatrix< float >;
template class CPUmkl_COO_lMatrix< double >;
