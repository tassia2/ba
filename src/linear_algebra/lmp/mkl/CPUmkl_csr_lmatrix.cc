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

/// @author Dimitar Lukarski, Martin Wlotzka

#include "config.h"

#include "common/macros.h"
#include "CPUmkl_csr_lmatrix.h"
#include "CPUmkl_blas_routines.h"
#include "../lmp_log.h"

#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <typeinfo>
#include <type_traits>

#define nBSR_HANDLE
#define nUPDATE_HANDLE
#define nMKL_UPDATE_VAL_SUPPORT

using namespace hiflow::la;

template < typename ValueType >
CPUmkl_CSR_lMatrix< ValueType >::CPUmkl_CSR_lMatrix(int init_nnz,
                                                    int init_num_row,
                                                    int init_num_col,
                                                    std::string init_name) {
#ifdef WITH_MKL
  this->Init(init_nnz, init_num_row, init_num_col, init_name);
  this->implementation_name_ = "Intel MKL";
  this->implementation_id_ = MKL;
  this->set_num_threads();
  
  this->A_descr_.type = SPARSE_MATRIX_TYPE_GENERAL;
  this->A_descr_.mode = SPARSE_FILL_MODE_LOWER;
  this->A_descr_.diag = SPARSE_DIAG_NON_UNIT;
  this->handle_created_ = false;
#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
CPUmkl_CSR_lMatrix< ValueType >::CPUmkl_CSR_lMatrix() {
#ifdef WITH_MKL
  this->implementation_name_ = "Intel MKL";
  this->implementation_id_ = MKL;
  this->set_num_threads();
  this->handle_created_ = false;
  
  this->A_descr_.type = SPARSE_MATRIX_TYPE_GENERAL;
  this->A_descr_.mode = SPARSE_FILL_MODE_LOWER;
  this->A_descr_.diag = SPARSE_DIAG_NON_UNIT;
#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
CPUmkl_CSR_lMatrix< ValueType >::~CPUmkl_CSR_lMatrix() {
#ifdef WITH_MKL 
  if (this->handle_is_created())
  {
    mkl_sparse_destroy(this->A_);
    this->handle_created_ = false;
  }
  // TODO: check whether this->matrix is deleted 
#else
  NO_MKL_ERROR;
#endif
}

template <>
void CPUmkl_CSR_lMatrix< double >::update_matrix_handle() const
{
#ifdef WITH_MKL
  // size of MKL_INT is defined in mkl/include/mkl_types.h  
  // -> depends on the way MKL is compiled   
  assert (sizeof(MKL_INT) == sizeof(int));

  if (this->handle_is_created())
  {
    mkl_sparse_destroy(this->A_);
    this->handle_created_ = false;
  }
#ifdef BSR_HANDLE
  sparse_status_t state = mkl_sparse_d_create_bsr(&this->A_, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, 
                                                   this->num_row_, this->num_col_, 1,
                                                   this->matrix.row , this->matrix.row+1, 
                                                   this->matrix.col, 
                                                   this->matrix.val);
#else
  sparse_status_t state = mkl_sparse_d_create_csr(&this->A_, SPARSE_INDEX_BASE_ZERO, 
                                                   this->num_row_, this->num_col_, 
                                                   this->matrix.row , this->matrix.row+1, 
                                                   this->matrix.col, 
                                                   this->matrix.val);
#endif

#ifndef NDEBUG
  if (state != SPARSE_STATUS_SUCCESS)
  {
    /*
    for (int r=0; r != this->num_row_+1; ++r)
    {
      std::cout << this->matrix.row[r] << " ";
    }
    */
    LOG_ERROR("Error code" << state);
    quit_program();
  }
#endif
  this->updated_structure_ = false;
  this->updated_values_ = false;
  this->handle_created_ = true;
#else
  NO_MKL_ERROR
#endif
}

template <>
void CPUmkl_CSR_lMatrix< float >::update_matrix_handle() const
{
#ifdef WITH_MKL
  // size of MKL_INT is defined in mkl/include/mkl_types.h  
  // -> depends on the way MKL is compiled   
  assert (sizeof(MKL_INT) == sizeof(int));
  
  if (this->handle_is_created())
  {
    mkl_sparse_destroy(this->A_);
    this->handle_created_ = false;
  }
#ifdef BSR_HANDLE
  sparse_status_t state = mkl_sparse_s_create_bsr(&this->A_, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, 
                                                   this->num_row_, this->num_col_, 1,
                                                   this->matrix.row , this->matrix.row+1, 
                                                   this->matrix.col, 
                                                   this->matrix.val);
#else
  sparse_status_t state = mkl_sparse_s_create_csr(&this->A_, SPARSE_INDEX_BASE_ZERO, 
                                                   this->num_row_, this->num_col_, 
                                                   this->matrix.row , this->matrix.row+1, 
                                                   this->matrix.col, 
                                                   this->matrix.val);
#endif

#ifndef NDEBUG
  if (state != SPARSE_STATUS_SUCCESS)
  {
    LOG_ERROR("Error code" << state);
    quit_program();
  }
#endif
 
  this->updated_structure_ = false;
  this->updated_values_ = false;
  this->handle_created_ = true;
#else
  NO_MKL_ERROR
#endif
}

template <>
void CPUmkl_CSR_lMatrix< double >::update_matrix_handle_values() const
{
#ifdef WITH_MKL
  assert (this->handle_is_created());
#ifdef MKL_UPDATE_VAL_SUPPORT
  sparse_status_t state = mkl_sparse_d_update_values(this->A_, this->nnz_, NULL, NULL, this->matrix.val);
  if (state != SPARSE_STATUS_SUCCESS)
  {
    LOG_ERROR("Error code" << state);
    quit_program();
  }
#endif

  this->updated_values_ = false;
#else
  NO_MKL_ERROR
#endif
}

template <>
void CPUmkl_CSR_lMatrix< float >::update_matrix_handle_values() const
{
#ifdef WITH_MKL
  assert (this->handle_is_created());
#ifdef MKL_UPDATE_VAL_SUPPORT
  sparse_status_t state = mkl_sparse_s_update_values(this->A_, this->nnz_, NULL, NULL, this->matrix.val);

  if (state != SPARSE_STATUS_SUCCESS)
  {
    LOG_ERROR("Error code" << state);
    quit_program();
  }
#endif

  this->updated_values_ = false;
#else
  NO_MKL_ERROR
#endif
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::Compress() 
{
  CPU_CSR_lMatrix<ValueType>::Compress();
  this->update_matrix_handle();
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::Init(const int init_nnz,
                                           const int init_num_row,
                                           const int init_num_col,
                                           const std::string init_name) 
{
  CPU_CSR_lMatrix<ValueType>::Init(init_nnz, init_num_row, init_num_col, init_name);
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::init_structure(const int *rows,
                                                     const int *cols) 
{
  CPU_CSR_lMatrix<ValueType>::init_structure(rows, cols);
  this->update_matrix_handle();
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::CloneFrom(const hiflow::la::lMatrix< ValueType > &other)
{
  if (this != &other) 
  {
    // if it is not empty init() will clean it
    this->Init(other.get_nnz(), other.get_num_row(), other.get_num_col(),
               other.get_name());

    this->CopyStructureFrom(other);

    this->CopyFrom(other);
  }

  const CPUmkl_CSR_lMatrix< ValueType > *mkl_other =
      dynamic_cast< const CPUmkl_CSR_lMatrix< ValueType > * >(&other);
  if (mkl_other != 0) 
  {
    this->set_num_threads(mkl_other->num_threads());
  } 
  else 
  {
    const CPUopenmp_CSR_lMatrix< ValueType > *omp_other =
        dynamic_cast< const CPUopenmp_CSR_lMatrix< ValueType > * >(&other);
    if (omp_other != 0) 
    {
      this->set_num_threads(omp_other->num_threads());
    }
    // no else
  }
  
  this->set_updated_values(true);
  this->set_updated_structure(true);
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::set_num_threads(void) 
{
#ifdef WITH_MKL
  // default value
  this->set_num_threads(1);
#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::set_num_threads(int num_thread) 
{
#ifdef WITH_MKL
  this->num_threads_ = num_thread;
#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::VectorMult(const lVector< ValueType > &invec, 
                                                 lVector< ValueType > *outvec) const 
{

  const CPU_lVector< ValueType > *casted_invec =
      dynamic_cast< const CPU_lVector< ValueType > * >(&invec);
  
  CPU_lVector< ValueType > *casted_outvec =
      dynamic_cast< CPU_lVector< ValueType > * >(outvec);

  if ((casted_invec == NULL) || (casted_outvec == NULL)) {
    LOG_ERROR("CPUmkl_CSR_lMatrix<ValueType>::VectorMult unsupported in or out vector");
    this->print();
    invec.print();
    outvec->print();
    quit_program();
  }

  this->VectorMult(*casted_invec, casted_outvec);
}

template <>
void CPUmkl_CSR_lMatrix< double >::VectorMult(const CPU_lVector< double > &invec, 
                                              CPU_lVector< double > *outvec) const 
{
#ifdef WITH_MKL
  assert(invec.get_size() > 0);
  assert(outvec->get_size() > 0);
  assert(invec.get_size() == this->get_num_col());
  assert(outvec->get_size() == this->get_num_row());
  assert (invec.buffer != nullptr);
  assert (outvec->buffer != nullptr);
  
  mkl_set_num_threads(this->num_threads_);

#ifdef UPDATE_HANDLE
  if (this->updated_structure())
  {
    this->update_matrix_handle();
  }
  else if (this->updated_values())
  {
    this->update_matrix_handle_values();
  }
#endif
  sparse_status_t state = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 
                                          1., this->A_, this->A_descr_,  
                                          invec.buffer, 0., outvec->buffer);

#ifndef NDEBUG
  if (state != SPARSE_STATUS_SUCCESS)
  {
    LOG_ERROR("Error code " << state);
    quit_program();
  }
#endif

#else
  NO_MKL_ERROR;
#endif
}

template <>
void CPUmkl_CSR_lMatrix< float >::VectorMult(const CPU_lVector< float > &invec, 
                                             CPU_lVector< float > *outvec) const 
{
#ifdef WITH_MKL
  assert(invec.get_size() > 0);
  assert(outvec->get_size() > 0);
  assert(invec.get_size() == this->get_num_col());
  assert(outvec->get_size() == this->get_num_row());
  
  mkl_set_num_threads(this->num_threads_);

#ifdef UPDATE_HANDLE
  if (this->updated_structure())
  {
    this->update_matrix_handle();
  }
  else if (this->updated_values())
  {
    this->update_matrix_handle_values();
  }
#endif
  sparse_status_t state = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 
                                          1., this->A_, this->A_descr_,  
                                          invec.buffer, 0., outvec->buffer);

#ifndef NDEBUG
  if (state != SPARSE_STATUS_SUCCESS)
  {
    LOG_ERROR("Error code" << state);
    quit_program();
  }
#endif

#else
  NO_MKL_ERROR;
#endif
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::CastFrom(const CPU_CSR_lMatrix< double > &other) 
{
  assert(this->get_num_row() == other.get_num_row());
  assert(this->get_num_col() == other.get_num_col());
  assert(this->get_nnz() == other.get_nnz());

  for (int i = 0, e_i = this->get_nnz(); i != e_i; ++i) {
    this->matrix.val[i] = static_cast< ValueType >(other.matrix.val[i]);
  }
  this->set_updated_values(true);
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::CastFrom(const CPU_CSR_lMatrix< float > &other) 
{
  assert(this->get_num_row() == other.get_num_row());
  assert(this->get_num_col() == other.get_num_col());
  assert(this->get_nnz() == other.get_nnz());

  for (int i = 0, e_i = this->get_nnz(); i != e_i; ++i) {
    this->matrix.val[i] = static_cast< ValueType >(other.matrix.val[i]);
  }
  
  this->set_updated_values(true);
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::CastTo(CPU_CSR_lMatrix< double > &other) const 
{
  assert(this->get_num_row() == other.get_num_row());
  assert(this->get_num_col() == other.get_num_col());
  assert(this->get_nnz() == other.get_nnz());

  for (int i = 0, e_i = this->get_nnz(); i != e_i; ++i) {
    other.matrix.val[i] = static_cast< double >(this->matrix.val[i]);
  }
}

template < typename ValueType >
void CPUmkl_CSR_lMatrix< ValueType >::CastTo(CPU_CSR_lMatrix< float > &other) const 
{
  assert(this->get_num_row() == other.get_num_row());
  assert(this->get_num_col() == other.get_num_col());
  assert(this->get_nnz() == other.get_nnz());

  for (int i = 0, e_i = this->get_nnz(); i != e_i; ++i) {
    other.matrix.val[i] = static_cast< float >(this->matrix.val[i]);
  }
}

template class CPUmkl_CSR_lMatrix< float >;
template class CPUmkl_CSR_lMatrix< double >;
