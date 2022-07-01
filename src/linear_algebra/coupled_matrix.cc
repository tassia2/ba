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

/// @author Chandramowli Subramanian, Nico Trost, Dimitar Lukarski, Martin
/// Wlotzka, Simon Gawlok

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "common/log.h"
#include "common/pointers.h"
#include "linear_algebra/coupled_matrix.h"
#include "linear_algebra/coupled_vector.h"
#include "linear_algebra/lmp/lmatrix.h"
#include "linear_algebra/lmp/lmatrix_csr_cpu.h"
#include "lmp/init_vec_mat.h"
#include "lmp/lmp_mem.h"
#include "tools/mpi_tools.h"

#define COL_OFFDIAG
namespace hiflow {
namespace la {

template < class DataType > CoupledMatrix< DataType >::CoupledMatrix() {
  this->row_couplings_ = nullptr;
  this->col_couplings_ = nullptr;
  this->comm_ = MPI_COMM_NULL;
  this->comm_size_ = -1;
  this->my_rank_ = -1;
  this->diagonal_ = nullptr;
  this->offdiagonal_ = nullptr;
  this->col_offdiagonal_ = nullptr;
  this->row_ownership_begin_ = -1;
  this->row_ownership_end_ = -1;
  this->col_ownership_begin_ = -1;
  this->col_ownership_end_ = -1;
  this->called_init_ = false;
  this->initialized_structure_ = false;
}

template < class DataType > CoupledMatrix< DataType >::~CoupledMatrix() {
  this->Clear();
  int is_finalized;
  MPI_Finalized(&is_finalized);
  if (!is_finalized) {
    if (this->comm_ != MPI_COMM_NULL)
      MPI_Comm_free(&this->comm_);
  }
}

template < class DataType >
void CoupledMatrix< DataType >::Init(const MPI_Comm &comm,
                                     const LaCouplings &row_cp,
                                     const LaCouplings &col_cp, 
                                     PLATFORM plat,
                                     IMPLEMENTATION impl, 
                                     MATRIX_FORMAT format,
                                     const SYSTEM &my_system) 
{
  if (this->print_level_ > 0) {
    LOG_INFO("Platform", plat);
    LOG_INFO("Implementation", impl);
    LOG_INFO("Matrix format", format);
  }

  if (this->comm_ != MPI_COMM_NULL)
    MPI_Comm_free(&this->comm_);

  assert(comm != MPI_COMM_NULL);
  // MPI communicator

  // determine nb. of processes
  int info = MPI_Comm_size(comm, &(this->comm_size_));
  assert(info == MPI_SUCCESS);
  assert(this->comm_size_ > 0);

  // retrieve my rank
  info = MPI_Comm_rank(comm, &(this->my_rank_));
  assert(info == MPI_SUCCESS);
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->comm_size_);

  info = MPI_Comm_split(comm, 0, this->my_rank_, &(this->comm_));
  assert(info == MPI_SUCCESS);
  // couplings
  this->row_couplings_ = &row_cp;
  assert(this->row_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  this->col_couplings_ = &col_cp;
  assert(this->col_couplings_ != nullptr);
  assert(this->col_couplings_->initialized());

  // clear old data
  this->Clear();
  if (plat == OPENCL) 
  {
    this->diagonal_ = init_matrix< DataType >(0, 0, 0, "diagonal", plat, impl, format, my_system);
    assert(this->diagonal_ != nullptr);

    this->offdiagonal_ = init_matrix< DataType >(0, 0, 0, "offdiagonal", plat, impl, format, my_system);
    assert(this->offdiagonal_ != nullptr);
#ifdef COL_OFFDIAG
    this->col_offdiagonal_ = init_matrix< DataType >(0, 0, 0, "col_offdiagonal", plat, impl, format, my_system);
    assert(this->col_offdiagonal_ != nullptr);
#endif
  } 
  else 
  {
    this->diagonal_ = init_matrix< DataType >(0, 0, 0, "diagonal", plat, impl, format);
    assert(this->diagonal_ != nullptr);

    this->offdiagonal_ = init_matrix< DataType >(0, 0, 0, "offdiagonal", plat, impl, format);
    assert(this->offdiagonal_ != nullptr);
#ifdef COL_OFFDIAG
    this->col_offdiagonal_ = init_matrix< DataType >(0, 0, 0, "col_offdiagonal", plat, impl, format);
    assert(this->col_offdiagonal_ != nullptr);
#endif
  }

  this->called_init_ = true;
}

// TODO: use row_couplings for row instead of col_cpuplings
template < class DataType >
void CoupledMatrix< DataType >::InitStructure(
  const int *rows_diag, const int *cols_diag, int nnz_diag,
  const int *rows_offdiag, const int *cols_offdiag, int nnz_offdiag) {
  assert(this->called_init_);
  assert(this->comm_ != MPI_COMM_NULL);
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
#ifdef COL_OFFDIAG
  assert(this->col_offdiagonal_ != nullptr);
#endif
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());

  assert(rows_diag != nullptr);
  assert(cols_diag != nullptr);
  assert(nnz_diag > 0);

  int num_rows_local = this->row_couplings_->nb_dofs(this->my_rank_);
  assert(num_rows_local > 0);
  int num_cols_local = this->col_couplings_->nb_dofs(this->my_rank_);
  assert(num_cols_local > 0);

  // compute ownership range
  this->ComputeOwnershipRange();

  LOG_DEBUG(2, "num_rows_local() = "
            << num_rows_local << ", num_cols_local() = " << num_cols_local
            << " row_os_begin = " << this->row_ownership_begin_
            << " col_os_begin = " << this->col_ownership_begin_);

  // shift rows_diag, cols_diag to local numbering
  std::vector< int > rows_shifted(nnz_diag);
  std::vector< int > cols_shifted(nnz_diag);

  for (int i = 0; i < nnz_diag; ++i) {
    rows_shifted[i] = rows_diag[i] - this->row_ownership_begin_;
    cols_shifted[i] = cols_diag[i] - this->col_ownership_begin_;
  }

  if (DEBUG_LEVEL >= 2) {
    std::vector< int > tmp_row;
    std::vector< int > tmp_col;
    for (int i = 0; i < nnz_diag; ++i) {
      tmp_row.push_back(rows_diag[i]);
      tmp_col.push_back(cols_diag[i]);
    }
    LOG_DEBUG(
      2, "diag_rows: " << string_from_range(tmp_row.begin(), tmp_row.end()));
    LOG_DEBUG(
      2, "diag_cols: " << string_from_range(tmp_col.begin(), tmp_col.end()));
  }

  // init structure of diagonal block
  this->diagonal_->Clear();
  this->diagonal_->Init(nnz_diag, num_rows_local, num_cols_local, "diagonal");
  this->diagonal_->init_structure(&(rows_shifted[0]), &(cols_shifted[0]));

  // init structure of offdiagonal block
  if (nnz_offdiag > 0) {
    assert(rows_offdiag != nullptr);
    assert(cols_offdiag != nullptr);

    // shift rows_offdiag, cols_offdiag
    rows_shifted.resize(nnz_offdiag);
    cols_shifted.resize(nnz_offdiag);

    for (int k = 0; k < nnz_offdiag; ++k) {
      rows_shifted[k] = rows_offdiag[k] - this->row_ownership_begin();
      cols_shifted[k] = this->col_couplings_->Global2Offdiag(cols_offdiag[k]);
    }

    this->offdiagonal_->Clear();
    this->offdiagonal_->Init(nnz_offdiag, num_rows_local,
                             this->col_couplings_->size_ghost(), "offdiagonal");
    this->offdiagonal_->init_structure(&(rows_shifted[0]), &(cols_shifted[0]));
    
    // assume symmetric sparsity structure
#ifdef COL_OFFDIAG
    this->col_offdiagonal_->Clear();
    this->col_offdiagonal_->Init(nnz_offdiag, this->col_couplings_->size_ghost(), num_rows_local, "col_offdiagonal");
    this->col_offdiagonal_->init_structure(&(cols_shifted[0]), &(rows_shifted[0]));
#endif
  }

  int my_nnz = this->diagonal_->get_nnz() + this->offdiagonal_->get_nnz();
  MPI_Allreduce(&my_nnz, &nnz_global_, 1, MPI_INT, MPI_SUM, comm_);
  assert(my_nnz <= nnz_global_);

  this->initialized_structure_ = true;
}

template < class DataType > 
void CoupledMatrix< DataType >::Compress() {
  assert(this->IsInitialized());

  this->diagonal_->Compress();

  if (this->offdiagonal_->get_nnz() > 0)
  {
    this->offdiagonal_->Compress();
  }
#ifdef COL_OFFDIAG
  if (this->col_offdiagonal_->get_nnz() > 0)
  {
    this->col_offdiagonal_->Compress();
  }
#endif
  int my_nnz = this->diagonal_->get_nnz() + this->offdiagonal_->get_nnz();
  MPI_Allreduce(&my_nnz, &nnz_global_, 1, MPI_INT, MPI_SUM, comm_);
  assert(my_nnz <= nnz_global_);
}

template < class DataType >
void CoupledMatrix< DataType >::VectorMultOffdiag(
  Vector< DataType > &in, Vector< DataType > *out) const {
  CoupledVector< DataType > *cv_in, *cv_out;

  cv_in = dynamic_cast< CoupledVector< DataType > * >(&in);
  cv_out = dynamic_cast< CoupledVector< DataType > * >(out);

  if ((cv_in != 0) && (cv_out != 0)) {
    this->VectorMultOffdiag(*cv_in, cv_out);
  } else {
    if (cv_in == 0) {
      LOG_ERROR("Called CoupledMatrix::VectorMult with incompatible input "
                "vector type.");
    }
    if (cv_out == 0) {
      LOG_ERROR("Called CoupledMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class DataType >
void CoupledMatrix< DataType >::VectorMultOffdiag(
  CoupledVector< DataType > &in, CoupledVector< DataType > *out) const {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());
  assert(this->comm_size_ > 0);

  // check sizes
  assert(this->num_cols_global() == in.size_global());
  assert(this->num_cols_local() == in.size_local());
  assert(this->num_rows_global() == out->size_global());
  assert(this->num_rows_local() == out->size_local());

  in.ReceiveGhost();
  in.SendBorder();

  in.WaitForRecv();

  // add ghost contributions
  this->offdiagonal_->VectorMultAdd(in.ghost(), &(out->interior()));

  in.WaitForSend();
}

template < class DataType >
void CoupledMatrix< DataType >::VectorMult(Vector< DataType > &in,
    Vector< DataType > *out) const {
  CoupledVector< DataType > *cv_in, *cv_out;

  cv_in = dynamic_cast< CoupledVector< DataType > * >(&in);
  cv_out = dynamic_cast< CoupledVector< DataType > * >(out);

  if ((cv_in != 0) && (cv_out != 0)) {
    this->VectorMult(*cv_in, cv_out);
  } else {
    if (cv_in == 0) {
      LOG_ERROR("Called CoupledMatrix::VectorMult with incompatible input "
                "vector type.");
    }
    if (cv_out == 0) {
      LOG_ERROR("Called CoupledMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class DataType >
void CoupledMatrix< DataType >::VectorMult(
  CoupledVector< DataType > &in, CoupledVector< DataType > *out) const {
  assert(this->IsInitialized());

  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());
  assert(this->comm_size_ > 0);

  // check sizes
  assert(this->num_cols_global() == in.size_global());
  assert(this->num_cols_local() == in.size_local());
  assert(this->num_rows_global() == out->size_global());
  assert(this->num_rows_local() == out->size_local());

  in.ReceiveGhost();
  in.SendBorder();

  // multiply diagonal block with interior of vector
  /*
  LOG_DEBUG(2," Diagonal Matrix: dimension = " << this->diagonal_->get_num_row()
  << " x " << this->diagonal_->get_num_col()
  << " Interior vector in: dimension = " << in.interior().get_size()
  << " Interior vector out: dimension = " << out->interior().get_size() );
   */
  this->diagonal_->VectorMult(in.interior(), &(out->interior()));

  in.WaitForRecv();

  // add ghost contributions
  this->offdiagonal_->VectorMultAdd(in.ghost(), &(out->interior()));

  in.WaitForSend();
}

template < class DataType >
void CoupledMatrix< DataType >::VectorMultAdd(DataType alpha,
    Vector< DataType > &in,
    DataType beta,
    Vector< DataType > *out) const {
  CoupledVector< DataType > *cv_in, *cv_out;

  cv_in = dynamic_cast< CoupledVector< DataType > * >(&in);
  cv_out = dynamic_cast< CoupledVector< DataType > * >(out);

  if ((cv_in != 0) && (cv_out != 0)) {
    this->VectorMultAdd(alpha, *cv_in, beta, cv_out);
  } else {
    if (cv_in == 0) {
      LOG_ERROR("Called CoupledMatrix::VectorMult with incompatible input "
                "vector type.");
    }
    if (cv_out == 0) {
      LOG_ERROR("Called CoupledMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class DataType >
void CoupledMatrix< DataType >::VectorMultAdd(
  DataType alpha, CoupledVector< DataType > &in, DataType beta,
  CoupledVector< DataType > *out) const {

  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());
  assert(this->comm_size_ > 0);

  // check sizes
  assert(this->num_cols_global() == in.size_global());
  assert(this->num_cols_local() == in.size_local());
  assert(this->num_rows_global() == out->size_global());
  assert(this->num_rows_local() == out->size_local());

  if (beta != 0.) {
    out->Scale(beta);
  }

  if (alpha != 0.) {
    CoupledVector< DataType > temp;
    temp.CloneFromWithoutContent(*out);

    in.ReceiveGhost();
    in.SendBorder();

    // multiply diagonal block with interior of vector
    this->diagonal_->VectorMult(in.interior(), &(temp.interior()));

    in.WaitForRecv();

    // add ghost contributions
    this->offdiagonal_->VectorMultAdd(in.ghost(), &(temp.interior()));

    in.WaitForSend();

    out->Axpy(temp, alpha);
  }
}

template < class DataType > void CoupledMatrix< DataType >::Zeros() {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
#ifdef COL_OFFDIAG
  assert(this->col_offdiagonal_ != nullptr);
#endif

  this->diagonal_->Zeros();
  this->offdiagonal_->Zeros();
#ifdef COL_OFFDIAG
  this->col_offdiagonal_->Zeros();
#endif
}

template < class DataType >
void CoupledMatrix< DataType >::diagonalize_rows(const int *row_indices,
    int num_rows,
    DataType diagonal_value) {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);

#ifndef NDEBUG
  for (int i = 0; i < num_rows; i++) {
    assert(row_ownership_begin_ <= row_indices[i]);
    assert(row_indices[i] < row_ownership_end_);
  }
#endif // NDEBUG

  // shift indices to local numbering
  std::vector< int > indices_shifted(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    indices_shifted[i] = row_indices[i] - this->row_ownership_begin_;
  }

  this->diagonal_->ZeroRows(&(indices_shifted[0]), num_rows, diagonal_value);
  if (this->offdiagonal().get_num_row() > 0) {
    this->offdiagonal_->ZeroRows(&(indices_shifted[0]), num_rows,
                                 static_cast< DataType >(0.));
  }
}

// TODO: use row_couplings for row instead of col_cpuplings
template < class DataType >
void CoupledMatrix< DataType >::Add(const int global_row_id,
                                    const int global_col_id,
                                    const DataType val) {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
#ifdef COL_OFFDIAG
  assert(this->col_offdiagonal_ != nullptr);
#endif
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());
  //assert(this->row_ownership_begin_ <= global_row_id);
  //assert(global_row_id < this->row_ownership_end_);

  if (this->col_ownership_begin_ <= global_col_id &&
      global_col_id < this->col_ownership_end_) 
  {
    if (this->row_ownership_begin_ <= global_row_id &&
        global_row_id < this->row_ownership_end_)
    {
      // diagonal block
      this->diagonal_->add_value(global_row_id - this->row_ownership_begin_,
                                 global_col_id - this->col_ownership_begin_, 
                                 val);
    }
#ifdef COL_OFFDIAG
    else
    {

      // column offdiagonal block
      this->col_offdiagonal_->add_value( this->col_couplings_->Global2Offdiag(global_row_id),
                                         global_col_id - this->col_ownership_begin_,
                                         val);
    }
#endif
  }
  else 
  {
    if (this->row_ownership_begin_ <= global_row_id &&
        global_row_id < this->row_ownership_end_)
    {
      // row offdiagonal block
      this->offdiagonal_->add_value( global_row_id - this->row_ownership_begin_,
                                   this->col_couplings_->Global2Offdiag(global_col_id), 
                                   val);
    }
    else 
    {
      // TODO: make assert or do nothing??
    }
  }
}

// TODO: col_offdiagonal
template < class DataType >
void CoupledMatrix< DataType >::GetValues(const int *row_indices, int num_rows,
    const int *col_indices, int num_cols,
    DataType *values) const {
  assert(this->IsInitialized());

  // clear target array
  memsethost(values, 0, num_rows * num_cols, sizeof(DataType));

  // prepare row indices
  std::vector< int > row_ids(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    row_ids[i] = row_indices[i] - this->row_ownership_begin_;
  }

  // split columns in diagonal and offdiagonal part
  std::vector< int > col_ids_diag, col_ind_diag;
  std::vector< int > col_ids_offdiag, col_ind_offdiag;

  col_ids_diag.reserve(num_cols);
  col_ind_diag.reserve(num_cols);
  col_ids_offdiag.reserve(num_cols);
  col_ind_offdiag.reserve(num_cols);

  for (int i = 0; i < num_cols; ++i) {
    // diagonal block
    if (this->col_ownership_begin_ <= col_indices[i] &&
        col_indices[i] < this->col_ownership_end_) {
      col_ids_diag.push_back(col_indices[i] - this->col_ownership_begin_);
      col_ind_diag.push_back(i);
    }
    // offdiagonal block
    else {
      col_ids_offdiag.push_back(
        this->col_couplings_->Global2Offdiag(col_indices[i]));
      col_ind_offdiag.push_back(i);
    }
  }

  // Get values from diagonal part
  this->diagonal().get_add_values(vec2ptr(row_ids), row_ids.size(),
                                  vec2ptr(col_ids_diag), col_ids_diag.size(),
                                  vec2ptr(col_ind_diag), num_cols, values);

  // Get values from off-diagonal part
  if (this->offdiagonal().get_num_row() > 0 && col_ids_offdiag.size() > 0) {
    this->offdiagonal().get_add_values(
      vec2ptr(row_ids), row_ids.size(), 
      vec2ptr(col_ids_offdiag), col_ids_offdiag.size(), 
      vec2ptr(col_ind_offdiag), num_cols, values);
  }
}

template < class DataType >
void CoupledMatrix< DataType >::VectorMult_submatrix(
  const int *row_indices, int num_rows, const int *col_indices, int num_cols,
  const DataType *in_values, DataType *out_values) const {
  assert(this->IsInitialized());

  // clear target array
  memsethost(out_values, 0, num_rows, sizeof(DataType));

  // prepare row indices
  std::vector< int > row_ids(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    row_ids[i] = row_indices[i] - this->row_ownership_begin_;
  }

  // split columns in diagonal and offdiagonal part
  std::vector< int > col_ids_diag, col_ind_diag;
  std::vector< int > col_ids_offdiag, col_ind_offdiag;

  col_ids_diag.reserve(num_cols);
  col_ind_diag.reserve(num_cols);
  col_ids_offdiag.reserve(num_cols);
  col_ind_offdiag.reserve(num_cols);

  for (int i = 0; i < num_cols; ++i) {
    // diagonal block
    if (this->col_ownership_begin_ <= col_indices[i] &&
        col_indices[i] < this->col_ownership_end_) {
      col_ids_diag.push_back(col_indices[i] - this->col_ownership_begin_);
      col_ind_diag.push_back(i);
    }
    // offdiagonal block
    else {
      col_ids_offdiag.push_back(
        this->col_couplings_->Global2Offdiag(col_indices[i]));
      col_ind_offdiag.push_back(i);
    }
  }

  // Get values from diagonal part
  this->diagonal().VectorMultAdd_submatrix(
    vec2ptr(row_ids), row_ids.size(), vec2ptr(col_ids_diag),
    col_ids_diag.size(), vec2ptr(col_ind_diag), in_values, out_values);

  // Get values from off-diagonal part
  if (this->offdiagonal().get_num_row() > 0 && col_ids_offdiag.size() > 0) {
    this->offdiagonal().VectorMultAdd_submatrix(
      vec2ptr(row_ids), row_ids.size(), vec2ptr(col_ids_offdiag),
      col_ids_offdiag.size(), vec2ptr(col_ind_offdiag), in_values,
      out_values);
  }
}

template < class DataType >
void CoupledMatrix< DataType >::VectorMult_submatrix_vanka(
  const int *row_indices, int num_rows, CoupledVector< DataType > &in,
  DataType *out_values) const {
  assert(this->IsInitialized());

  // clear target array
  memsethost(out_values, 0, num_rows, sizeof(DataType));

  // prepare row indices
  std::vector< int > row_ids(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    row_ids[i] = row_indices[i] - this->row_ownership_begin_;
  }

  // Get values from diagonal part
  this->diagonal().VectorMultAdd_submatrix_vanka(
    vec2ptr(row_ids), row_ids.size(), in.interior(), out_values);

  // Get values from off-diagonal part
  if (this->offdiagonal().get_num_row() > 0) {
    this->offdiagonal().VectorMultAdd_submatrix_vanka(
      vec2ptr(row_ids), row_ids.size(), in.ghost(), out_values);
  }
}

// TODO: use row_couplings for row instead of col_cpuplings
template < class DataType >
void CoupledMatrix< DataType >::Add(const int *rows, int num_rows,
                                    const int *cols, int num_cols,
                                    const DataType *values) {
  assert(this->IsInitialized());
  assert (rows != nullptr);
  assert (cols != nullptr);
  assert (values != nullptr);

  // prepare row indices
  /*
  std::vector< int > row_ids(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    row_ids[i] = rows[i] - this->row_ownership_begin_;
  }
  */
  
  row_ids_diag.clear();
  row_ind_diag.clear();
  row_ids_offdiag.clear();
  row_ind_offdiag.clear();

  row_ids_diag.reserve(num_rows);
  row_ind_diag.reserve(num_rows);
  row_ids_offdiag.reserve(num_rows);
  row_ind_offdiag.reserve(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    // diagonal block
    if (this->row_ownership_begin_ <= rows[i] &&
        rows[i] < this->row_ownership_end_) {
      row_ids_diag.push_back(rows[i] - this->row_ownership_begin_);
      row_ind_diag.push_back(i);
    }
    // offdiagonal block
    else {
      // std::cout << cols[i] << " not in ( " <<  this->col_ownership_begin_ <<
      // " , " <<  this->col_ownership_end_ << " ) "<< std::endl;
      row_ids_offdiag.push_back(this->col_couplings_->Global2Offdiag(rows[i]));
      row_ind_offdiag.push_back(i);
    }
  }
  
  
  // split columns in diagonal and offdiagonal part
  col_ids_diag.clear();
  col_ind_diag.clear();
  col_ids_offdiag.clear();
  col_ind_offdiag.clear();

  col_ids_diag.reserve(num_cols);
  col_ind_diag.reserve(num_cols);
  col_ids_offdiag.reserve(num_cols);
  col_ind_offdiag.reserve(num_cols);

  for (int i = 0; i < num_cols; ++i) {
    // diagonal block
    if (this->col_ownership_begin_ <= cols[i] &&
        cols[i] < this->col_ownership_end_) {
      col_ids_diag.push_back(cols[i] - this->col_ownership_begin_);
      col_ind_diag.push_back(i);
    }
    // offdiagonal block
    else {
      // std::cout << cols[i] << " not in ( " <<  this->col_ownership_begin_ <<
      // " , " <<  this->col_ownership_end_ << " ) "<< std::endl;
      col_ids_offdiag.push_back(this->col_couplings_->Global2Offdiag(cols[i]));
      col_ind_offdiag.push_back(i);
    }
  }

  // split values
  diag_values.clear();
  diag_values.resize(row_ids_diag.size() * col_ids_diag.size());
  offdiag_values.clear();
  offdiag_values.resize(row_ids_diag.size() * col_ids_offdiag.size());
#ifdef COL_OFFDIAG
  col_offdiag_values.clear();
  col_offdiag_values.resize(row_ids_offdiag.size() * col_ids_diag.size());
#endif
  for (size_t i = 0, e_i = row_ids_diag.size(); i != e_i; ++i) {
    const int row_offset = i * col_ids_diag.size();
    const int row_offset_old = row_ind_diag[i] * num_cols;
    for (size_t j = 0, e_j = col_ids_diag.size(); j != e_j; ++j) {
      diag_values[row_offset + j] = values[row_offset_old + col_ind_diag[j]];
    }
  }

  for (size_t i = 0, e_i = row_ids_diag.size(); i != e_i; ++i) {
    const int row_offset = i * col_ids_offdiag.size();
    const int row_offset_old = row_ind_diag[i] * num_cols;
    for (size_t j = 0, e_j = col_ids_offdiag.size(); j != e_j; ++j) {
      offdiag_values[row_offset + j] = values[row_offset_old + col_ind_offdiag[j]];
    }
  }

#ifdef COL_OFFDIAG
  for (size_t i = 0, e_i = row_ids_offdiag.size(); i != e_i; ++i) {
    const int row_offset = i * col_ids_diag.size();
    const int row_offset_old = row_ind_offdiag[i] * num_cols;
    for (size_t j = 0, e_j = col_ids_diag.size(); j != e_j; ++j) {
      col_offdiag_values[row_offset + j] = values[row_offset_old + col_ind_diag[j]];
    }
  }
#endif
  // add values to submatrices
  if (row_ids_diag.size() > 0 && col_ids_diag.size() > 0)
  {
    this->diagonal_->add_values(vec2ptr(row_ids_diag), row_ids_diag.size(),
                                vec2ptr(col_ids_diag), col_ids_diag.size(),
                                vec2ptr(diag_values));
  }
  if (row_ids_diag.size() > 0 && col_ids_offdiag.size() > 0) 
  {
    this->offdiagonal_->add_values(vec2ptr(row_ids_diag), row_ids_diag.size(), 
                                   vec2ptr(col_ids_offdiag), col_ids_offdiag.size(),
                                   vec2ptr(offdiag_values));
  }
#ifdef COL_OFFDIAG
  if (row_ids_offdiag.size() > 0 && col_ids_diag.size() > 0) 
  {
    this->col_offdiagonal_->add_values(vec2ptr(row_ids_offdiag), row_ids_offdiag.size(), 
                                       vec2ptr(col_ids_diag), col_ids_diag.size(),
                                       vec2ptr(col_offdiag_values));
  }
#endif
}

template < class DataType >
void CoupledMatrix< DataType >::Add(const std::vector<int>& row_ind, 
                                    const std::vector<int>& col_ind,
                                    const std::vector<DataType>& values) 
{
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
#ifdef COL_OFFDIAG
  assert(this->col_offdiagonal_ != nullptr);
#endif
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());
  //assert(this->row_ownership_begin_ <= global_row_id);
  //assert(global_row_id < this->row_ownership_end_);

  row_ids_diag.clear();
  row_ids_offdiag.clear();
  col_ids_diag.clear();
  col_ids_offdiag.clear();
  diag_values.clear();
  offdiag_values.clear();
  
  const size_t size = row_ind.size();

  row_ids_diag.reserve(size);
  col_ids_diag.reserve(size);
  diag_values.reserve(size);

  row_ids_offdiag.reserve(size);
  col_ids_offdiag.reserve(size);
  offdiag_values.reserve(size);

  for (size_t i = 0; i != size; ++i)
  {
    const int row = row_ind[i];
    const int col = col_ind[i];
    if (this->col_ownership_begin_ <= col && col < this->col_ownership_end_
        && this->row_ownership_begin_ <= row && row < this->row_ownership_end_)
    {
      row_ids_diag.push_back(row-this->row_ownership_begin_);
      col_ids_diag.push_back(col-this->col_ownership_begin_);
      diag_values.push_back(values[i]);
    }
    else if (this->row_ownership_begin_ <= row && row < this->row_ownership_end_)
    {
      row_ids_offdiag.push_back(row- this->row_ownership_begin_);
      col_ids_offdiag.push_back(this->col_couplings_->Global2Offdiag(col));
      offdiag_values.push_back(values[i]);
    }
  }

  const size_t size_diag = row_ids_diag.size();
  const size_t size_offdiag = row_ids_offdiag.size();
  
  if (size_diag > 0)
  {
    this->diagonal_->add_values(size_diag, vec2ptr(row_ids_diag), vec2ptr(col_ids_diag), vec2ptr(diag_values));
  }
  if (size_offdiag > 0)
  {
    this->offdiagonal_->add_values(size_offdiag, vec2ptr(row_ids_offdiag), vec2ptr(col_ids_offdiag), vec2ptr(offdiag_values));
  }

#if 0
  const size_t num_entries = row_ind.size();

  assert(num_entries == col_ind.size());
  assert(num_entries == values.size());

  if (num_entries == 0)
  {
    return;
  }
  if (num_entries == 1)
  {
    this->Add(row_ind[0], col_ind[0], values[0]);
    return;
  }

  this->create_reduced_matrix(row_ind, col_ind, values, 
                              this->row_ind_reduced_, 
                              this->col_ind_reduced_,
                              this->tmp_vals_);

  this->Add(vec2ptr(this->row_ind_reduced_), this->row_ind_reduced_.size(), 
            vec2ptr(this->col_ind_reduced_), this->col_ind_reduced_.size(),
            vec2ptr(this->tmp_vals_));
#endif
}

template < class DataType >
void CoupledMatrix< DataType >::create_reduced_matrix(const std::vector<int>& row_ind, 
                                                      const std::vector<int>& col_ind,
                                                      const std::vector<DataType>& values,
                                                      std::vector<int>& row_ind_red, 
                                                      std::vector<int>& col_ind_red,
                                                      std::vector<DataType>& values_red) const
{
  const size_t num_entries = row_ind.size();
  assert(num_entries == col_ind.size());
  assert(num_entries == values.size());

  row_ind_red.clear();  
  col_ind_red.clear();  
  this->perm_tmp1_.clear();
  this->perm_tmp2_.clear();
  this->perm_row_.clear();
  this->perm_col_.clear();

  sort_and_reduce(row_ind, row_ind_red, this->perm_tmp1_, this->perm_tmp2_, this->perm_row_);
  sort_and_reduce(col_ind, col_ind_red, this->perm_tmp1_, this->perm_tmp2_, this->perm_col_);

  const size_t num_rows = row_ind_red.size();
  const size_t num_cols = col_ind_red.size();

  assert (num_entries == this->perm_row_.size());
  assert (num_entries == this->perm_col_.size());

  values_red.clear();
  values_red.resize(num_rows * num_cols, 0.);
  for (size_t i = 0; i != num_entries; ++i)
  {
    const int row = this->perm_row_[i];
    const int col = this->perm_col_[i];

    values_red[row * num_cols + col] += values[i];
  }
}


template < class DataType >
void CoupledMatrix< DataType >::SetValue(int row, int col, DataType value) {
  assert(this->IsInitialized());
  this->SetValues(&row, 1, &col, 1, &value);
};

template < class DataType >
void CoupledMatrix< DataType >::SetValues(const int *rows, int num_rows,
                                          const int *cols, int num_cols,
                                          const DataType *values) {
  assert(this->IsInitialized());
  assert (rows != nullptr);
  assert (cols != nullptr);
  assert (values != nullptr);

  // prepare row indices
  /*
  std::vector< int > row_ids(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    row_ids[i] = rows[i] - this->row_ownership_begin_;
  }
  */
  std::vector< int > row_ids_diag;
  std::vector< int > row_ind_diag;
  std::vector< int > row_ids_offdiag;
  std::vector< int > row_ind_offdiag;

  row_ids_diag.reserve(num_rows);
  row_ind_diag.reserve(num_rows);
  row_ids_offdiag.reserve(num_rows);
  row_ind_offdiag.reserve(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    // diagonal block
    if (this->row_ownership_begin_ <= rows[i] &&
        rows[i] < this->row_ownership_end_) {
      row_ids_diag.push_back(rows[i] - this->row_ownership_begin_);
      row_ind_diag.push_back(i);
    }
    // offdiagonal block
    else {
      // std::cout << cols[i] << " not in ( " <<  this->col_ownership_begin_ <<
      // " , " <<  this->col_ownership_end_ << " ) "<< std::endl;
      row_ids_offdiag.push_back(this->col_couplings_->Global2Offdiag(rows[i]));
      row_ind_offdiag.push_back(i);
    }
  }
  
  
  // split columns in diagonal and offdiagonal part
  std::vector< int > col_ids_diag;
  std::vector< int > col_ind_diag;
  std::vector< int > col_ids_offdiag;
  std::vector< int > col_ind_offdiag;

  col_ids_diag.reserve(num_cols);
  col_ind_diag.reserve(num_cols);
  col_ids_offdiag.reserve(num_cols);
  col_ind_offdiag.reserve(num_cols);

  for (int i = 0; i < num_cols; ++i) {
    // diagonal block
    if (this->col_ownership_begin_ <= cols[i] &&
        cols[i] < this->col_ownership_end_) {
      col_ids_diag.push_back(cols[i] - this->col_ownership_begin_);
      col_ind_diag.push_back(i);
    }
    // offdiagonal block
    else {
      // std::cout << cols[i] << " not in ( " <<  this->col_ownership_begin_ <<
      // " , " <<  this->col_ownership_end_ << " ) "<< std::endl;
      col_ids_offdiag.push_back(this->col_couplings_->Global2Offdiag(cols[i]));
      col_ind_offdiag.push_back(i);
    }
  }

  // split values
  std::vector< DataType > diag_values(row_ids_diag.size() * col_ids_diag.size());
  std::vector< DataType > offdiag_values(row_ids_diag.size() * col_ids_offdiag.size());
#ifdef COL_OFFDIAG
  std::vector< DataType > col_offdiag_values(row_ids_offdiag.size() * col_ids_diag.size());
#endif
  for (size_t i = 0, e_i = row_ids_diag.size(); i != e_i; ++i) {
    const int row_offset = i * col_ids_diag.size();
    const int row_offset_old = row_ind_diag[i] * num_cols;
    for (size_t j = 0, e_j = col_ids_diag.size(); j != e_j; ++j) {
      diag_values[row_offset + j] = values[row_offset_old + col_ind_diag[j]];
    }
  }

  for (size_t i = 0, e_i = row_ids_diag.size(); i != e_i; ++i) {
    const int row_offset = i * col_ids_offdiag.size();
    const int row_offset_old = row_ind_diag[i] * num_cols;
    for (size_t j = 0, e_j = col_ids_offdiag.size(); j != e_j; ++j) {
      offdiag_values[row_offset + j] = values[row_offset_old + col_ind_offdiag[j]];
    }
  }

#ifdef COL_OFFDIAG
  for (size_t i = 0, e_i = row_ids_offdiag.size(); i != e_i; ++i) {
    const int row_offset = i * col_ids_diag.size();
    const int row_offset_old = row_ind_offdiag[i] * num_cols;
    for (size_t j = 0, e_j = col_ids_diag.size(); j != e_j; ++j) {
      col_offdiag_values[row_offset + j] = values[row_offset_old + col_ind_diag[j]];
    }
  }
#endif
  // set values to submatrices
  if (row_ids_diag.size() > 0 && col_ids_diag.size() > 0)
  {
    this->diagonal_->set_values(vec2ptr(row_ids_diag), row_ids_diag.size(),
                                vec2ptr(col_ids_diag), col_ids_diag.size(),
                                vec2ptr(diag_values));
  }
  if (row_ids_diag.size() > 0 && col_ids_offdiag.size() > 0) 
  {
    this->offdiagonal_->set_values(vec2ptr(row_ids_diag), row_ids_diag.size(), 
                                   vec2ptr(col_ids_offdiag), col_ids_offdiag.size(),
                                   vec2ptr(offdiag_values));
  }
#ifdef COL_OFFDIAG
  if (row_ids_offdiag.size() > 0 && col_ids_diag.size() > 0) 
  {
    this->col_offdiagonal_->set_values(vec2ptr(row_ids_offdiag), row_ids_offdiag.size(), 
                                       vec2ptr(col_ids_diag), col_ids_diag.size(),
                                       vec2ptr(col_offdiag_values));
  }
#endif
}

template < class DataType >
void CoupledMatrix< DataType >::ExtractCSR(int *ia, int *ja,
    DataType *val) const {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);

  if (this->diagonal().get_platform() == CPU) {

    if (this->diagonal().get_matrix_format() == CSR) {
      // dynamic cast to CPU matrix in order to access ia, ja, val
      const CPU_CSR_lMatrix< DataType > *casted_diag =
        dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(this->diagonal_);
      assert(casted_diag != nullptr);

      const CPU_CSR_lMatrix< DataType > *casted_offdiag =
        dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(
          this->offdiagonal_);
      assert(casted_offdiag != nullptr);

      int counter = 0;
      for (int ind_ia = 0; ind_ia < this->num_rows_local(); ++ind_ia) {
        ia[ind_ia] = counter;

        // diagonal block
        for (int ind_ja_diag = casted_diag->matrix.row[ind_ia];
             ind_ja_diag < casted_diag->matrix.row[ind_ia + 1]; ++ind_ja_diag) {
          ja[counter] =
            casted_diag->matrix.col[ind_ja_diag] + this->col_ownership_begin_;
          val[counter] = casted_diag->matrix.val[ind_ja_diag];
          ++counter;
        }
        // offdiagonal block
        if (this->offdiagonal().get_num_row() > 0) {
          for (int ind_ja_offdiag = casted_offdiag->matrix.row[ind_ia];
               ind_ja_offdiag < casted_offdiag->matrix.row[ind_ia + 1];
               ++ind_ja_offdiag) {
            ja[counter] = this->col_couplings_->Offdiag2Global(
                            casted_offdiag->matrix.col[ind_ja_offdiag]);
            val[counter] = casted_offdiag->matrix.val[ind_ja_offdiag];
            ++counter;
          }
        }
      }
      ia[this->num_rows_local()] = counter;

      assert(counter == this->nnz_local());
    }

    else {
      LOG_ERROR(
          "CoupledMatrix::ExtractCSR: not supported for non CSR formats.");
      quit_program();
    }
  } else {
    LOG_ERROR("CoupledMatrix::ExtractCSR: not supported on non CPU platforms.");
    quit_program();
  }
}

template < class DataType >
void CoupledMatrix< DataType >::ExtractDiagonalCSR(int *ia, int *ja,
    DataType *val) const {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);

  if (this->diagonal().get_platform() == CPU) {
    if (this->diagonal().get_matrix_format() == CSR) {
      // dynamic cast to CPU matrix in order to access ia, ja, val
      const CPU_CSR_lMatrix< DataType > *casted_diag =
        dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(this->diagonal_);
      assert(casted_diag != nullptr);

      for (int i = 0; i < this->num_rows_local() + 1; i++) {
        ia[i] = casted_diag->matrix.row[i];
      }
      for (int j = 0; j < this->nnz_local_diag(); j++) {
        ja[j] = casted_diag->matrix.col[j];
        val[j] = casted_diag->matrix.val[j];
      }
    } else {
      LOG_ERROR("CoupledMatrix::ExtractDiagonalCSR: not supported for non CSR "
                "formats.");
      quit_program();
    }
  } else {
    LOG_ERROR("CoupledMatrix::ExtractDiagonalCSR: not supported on non CPU "
              "platforms.");
    quit_program();
  }
}

template < class DataType >
void CoupledMatrix< DataType >::GetDiagonalCSR(int *& ia, int *& ja, DataType *& val)
{
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);

  if (this->diagonal().get_platform() == CPU) {
    if (this->diagonal().get_matrix_format() == CSR) {
      // dynamic cast to CPU matrix in order to access ia, ja, val
      const CPU_CSR_lMatrix< DataType > *casted_diag =
        dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(this->diagonal_);
      assert(casted_diag != nullptr);

      ia = casted_diag->matrix.row;
      ja = casted_diag->matrix.col;
      val = casted_diag->matrix.val;
    } else {
      LOG_ERROR("CoupledMatrix::ExtractDiagonalCSR: not supported for non CSR "
                "formats.");
      quit_program();
    }
  } else {
    LOG_ERROR("CoupledMatrix::ExtractDiagonalCSR: not supported on non CPU "
              "platforms.");
    quit_program();
  }
}

template < class DataType >
void CoupledMatrix< DataType >::SetCSR(int *ia, int *ja,
                                       DataType *val) {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);

  if (this->diagonal().get_platform() == CPU) {

    if (this->diagonal().get_matrix_format() == CSR) {

      // dynamic cast to CPU matrix in order to access ia, ja, val
      const CPU_CSR_lMatrix< DataType > *casted_diag =
        dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(this->diagonal_);
      assert(casted_diag != nullptr);

      const CPU_CSR_lMatrix< DataType > *casted_offdiag =
        dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(
          this->offdiagonal_);
      assert(casted_offdiag != nullptr);

      int counter = 0;
      for (int ind_ia = 0; ind_ia < this->num_rows_local(); ++ind_ia) {

        // diagonal block
        for (int ind_ja_diag = casted_diag->matrix.row[ind_ia];
             ind_ja_diag < casted_diag->matrix.row[ind_ia + 1]; ind_ja_diag++) {

          casted_diag->matrix.val[ind_ja_diag] = val[counter];

          ++counter;
        }
        // offdiagonal block
        if (this->offdiagonal_->get_nnz() > 0) {
          for (int ind_ja_offdiag = casted_offdiag->matrix.row[ind_ia];
               ind_ja_offdiag < casted_offdiag->matrix.row[ind_ia + 1];
               ind_ja_offdiag++) {

            casted_offdiag->matrix.val[ind_ja_offdiag] = val[counter];

            ++counter;
          }
        }
      }

      assert(counter == this->nnz_local());

    } else {
      LOG_ERROR(
        "CoupledMatrix::ExtractCSR: not supported for non CSR formats.");
      exit(-1);
    }
  } else {
    LOG_ERROR("CoupledMatrix::ExtractCSR: not supported on non CPU platforms.");
    exit(-1);
  }
}


template < class DataType >
Matrix< DataType > *CoupledMatrix< DataType >::Clone() const {
  CoupledMatrix< DataType > *clone = new CoupledMatrix< DataType >();
  clone->CloneFrom(*this);
  return clone;
}

template < class DataType >
void CoupledMatrix< DataType >::CloneFrom(
  const CoupledMatrix< DataType > &mat) {
  if (this != &mat) {
    // clone the matrix and copy the structure
    this->CloneFromWithoutContent(mat);
    // copy entries
    this->diagonal_->CopyFrom(mat.diagonal());
    this->offdiagonal_->CopyFrom(mat.offdiagonal());
    this->col_offdiagonal_->CopyFrom(mat.col_offdiagonal());
  }
}

template < class DataType >
void CoupledMatrix< DataType >::CopyFrom(const CoupledMatrix< DataType > &mat) {
  if (this != &mat) {
    assert(this->comm_size_ == mat.comm_size());
    assert(this->my_rank_ == mat.my_rank());
    assert(this->row_ownership_begin_ == mat.row_ownership_begin());
    assert(this->row_ownership_end_ == mat.row_ownership_end());
    assert(this->col_ownership_begin_ == mat.col_ownership_begin());
    assert(this->col_ownership_end_ == mat.col_ownership_end());

    this->diagonal_->CopyFrom(mat.diagonal());
    this->offdiagonal_->CopyFrom(mat.offdiagonal());
    this->col_offdiagonal_->CopyFrom(mat.col_offdiagonal());
  }
}

template < class DataType >
void CoupledMatrix< DataType >::CastFrom(const CoupledMatrix< double > &other) {

  assert(this->comm_size_ == other.comm_size());
  assert(this->my_rank_ == other.my_rank());
  assert(this->row_ownership_begin_ == other.row_ownership_begin());
  assert(this->row_ownership_end_ == other.row_ownership_end());
  assert(this->col_ownership_begin_ == other.col_ownership_begin());
  assert(this->col_ownership_end_ == other.col_ownership_end());

  this->diagonal_->CastFrom(other.diagonal());
  this->offdiagonal_->CastFrom(other.offdiagonal());
  this->col_offdiagonal_->CastFrom(other.col_offdiagonal());
}

template < class DataType >
void CoupledMatrix< DataType >::CastFrom(const CoupledMatrix< float > &other) {

  assert(this->comm_size_ == other.comm_size());
  assert(this->my_rank_ == other.my_rank());
  assert(this->row_ownership_begin_ == other.row_ownership_begin());
  assert(this->row_ownership_end_ == other.row_ownership_end());
  assert(this->col_ownership_begin_ == other.col_ownership_begin());
  assert(this->col_ownership_end_ == other.col_ownership_end());

  this->diagonal_->CastFrom(other.diagonal());
  this->offdiagonal_->CastFrom(other.offdiagonal());
  this->col_offdiagonal_->CastFrom(other.col_offdiagonal());
}

template < class DataType >
void CoupledMatrix< DataType >::CopyStructureFrom(
  const CoupledMatrix< DataType > &mat) {
  if (this != &mat) {
    // no Clear() !
    this->comm_size_ = mat.comm_size();
    this->my_rank_ = mat.my_rank();
    int info = 0;
    if (this->comm_ != MPI_COMM_NULL) {
      info = MPI_Comm_free(&this->comm_);
      assert(info == MPI_SUCCESS);
    }
    info = MPI_Comm_split(mat.comm(), 0, this->my_rank_, &(this->comm_));
    assert(info == MPI_SUCCESS);
    this->row_couplings_ = &(mat.row_couplings());
    this->col_couplings_ = &(mat.col_couplings());
    this->row_ownership_begin_ = mat.row_ownership_begin();
    this->row_ownership_end_ = mat.row_ownership_end();
    this->col_ownership_begin_ = mat.col_ownership_begin();
    this->col_ownership_end_ = mat.col_ownership_end();
    this->diagonal_->CopyStructureFrom(mat.diagonal());
    this->offdiagonal_->CopyStructureFrom(mat.offdiagonal());
    this->col_offdiagonal_->CopyStructureFrom(mat.col_offdiagonal());
  }
}

template < class DataType >
void CoupledMatrix< DataType >::CloneFromWithoutContent(
  const CoupledMatrix< DataType > &mat) {
  if (this != &mat) {
    this->Clear();
    this->comm_size_ = mat.comm_size();
    this->my_rank_ = mat.my_rank();
    int info = 0;
    if (this->comm_ != MPI_COMM_NULL) {
      info = MPI_Comm_free(&this->comm_);
      assert(info == MPI_SUCCESS);
    }
    info = MPI_Comm_split(mat.comm(), 0, this->my_rank_, &(this->comm_));
    assert(info == MPI_SUCCESS);
    this->row_couplings_ = &(mat.row_couplings());
    this->col_couplings_ = &(mat.col_couplings());
    this->row_ownership_begin_ = mat.row_ownership_begin();
    this->row_ownership_end_ = mat.row_ownership_end();
    this->col_ownership_begin_ = mat.col_ownership_begin();
    this->col_ownership_end_ = mat.col_ownership_end();

    // clone the matrix
    this->diagonal_ = mat.diagonal().CloneWithoutContent();
    this->offdiagonal_ = mat.offdiagonal().CloneWithoutContent();
    this->col_offdiagonal_ = mat.col_offdiagonal().CloneWithoutContent();

    // copy structure
    this->diagonal_->CopyStructureFrom(mat.diagonal());
    this->offdiagonal_->CopyStructureFrom(mat.offdiagonal());
    this->col_offdiagonal_->CopyStructureFrom(mat.col_offdiagonal());

    this->called_init_ = mat.called_init_;
    this->initialized_structure_ = mat.initialized_structure_;
  }
}

template < class DataType >
void CoupledMatrix< DataType >::ConvertFromCSR2COO(
  const CoupledMatrix< DataType > &mat) {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());
  assert(this->diagonal().get_matrix_format() == COO);
  assert(this->offdiagonal().get_matrix_format() == COO);
  assert(this->col_offdiagonal().get_matrix_format() == COO);

  if (this->diagonal().get_platform() == CPU) {
    int nnz_diag = mat.diagonal().get_nnz();
    int nnz_offdiag = mat.offdiagonal().get_nnz();
    int num_rows_local = this->row_couplings_->nb_dofs(this->my_rank_);
    int num_cols_local = this->col_couplings_->nb_dofs(this->my_rank_);

    // init structure of diagonal block
    this->diagonal_->Clear();
    this->diagonal_->Init(nnz_diag, num_rows_local, num_cols_local, "diagonal");
    this->diagonal_->ConvertFrom(mat.diagonal());

    // init structure offdiagonal block
    if (nnz_offdiag > 0) {
      this->offdiagonal_->Clear();
      this->offdiagonal_->Init(nnz_offdiag, num_rows_local,
                               this->col_couplings_->size_ghost(),
                               "offdiagonal");
      this->offdiagonal_->ConvertFrom(mat.offdiagonal());
#ifdef COL_OFFDIAG
      this->col_offdiagonal_->Clear();
      this->col_offdiagonal_->Init(nnz_offdiag, this->col_couplings_->size_ghost(), num_rows_local,
                               "col_offdiagonal");
      this->col_offdiagonal_->ConvertFrom(mat.col_offdiagonal());
#endif
    }
  } else {
    LOG_ERROR(
        "CoupledMatrix::ConvertFromCSR2COO: not supported for GPU matrices.");
    quit_program();
  }
}

template < class DataType >
void CoupledMatrix< DataType >::ConvertFromCOO2CSR(
  const CoupledMatrix< DataType > &mat) {
  assert(this->IsInitialized());
  assert(this->diagonal_ != nullptr);
  assert(this->offdiagonal_ != nullptr);
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());
  assert(this->diagonal().get_matrix_format() == CSR);
  assert(this->offdiagonal().get_matrix_format() == CSR);
  assert(this->col_offdiagonal().get_matrix_format() == CSR);

  int nnz_diag = mat.diagonal().get_nnz();
  int nnz_offdiag = mat.offdiagonal().get_nnz();
  int num_rows_local = this->row_couplings_->nb_dofs(this->my_rank_);
  int num_cols_local = this->col_couplings_->nb_dofs(this->my_rank_);

  // init structure of diagonal block
  this->diagonal_->Clear();
  this->diagonal_->Init(nnz_diag, num_rows_local, num_cols_local, "diagonal");
  this->diagonal_->ConvertFrom(mat.diagonal());

  // init structure offdiagonal block
  if (nnz_offdiag > 0) {
    this->offdiagonal_->Clear();
    this->offdiagonal_->Init(nnz_offdiag, num_rows_local,
                             this->col_couplings_->size_ghost(), "offdiagonal");
    this->offdiagonal_->ConvertFrom(mat.offdiagonal());
#ifdef COL_OFFDIAG
    this->col_offdiagonal_->Clear();
    this->col_offdiagonal_->Init(nnz_offdiag, this->col_couplings_->size_ghost(), num_rows_local, "col_offdiagonal");
    this->col_offdiagonal_->ConvertFrom(mat.col_offdiagonal());
#endif
  }
}

template < class DataType >
void CoupledMatrix< DataType >::CreateTransposedFrom(
  const CoupledMatrix< DataType > &other) {
  if (this != &other) {
    this->Clear();
    this->comm_size_ = other.comm_size();
    this->my_rank_ = other.my_rank();
    int info = 0;
    if (this->comm_ != MPI_COMM_NULL) {
      info = MPI_Comm_free(&this->comm_);
      assert(info == MPI_SUCCESS);
    }
    info = MPI_Comm_split(other.comm(), 0, this->my_rank_, &(this->comm_));
    assert(info == MPI_SUCCESS);
    this->row_couplings_ = &(other.col_couplings());
    this->col_couplings_ = &(other.row_couplings());
    this->row_ownership_begin_ = other.col_ownership_begin();
    this->row_ownership_end_ = other.col_ownership_end();
    this->col_ownership_begin_ = other.row_ownership_begin();
    this->col_ownership_end_ = other.row_ownership_end();

    // clone diagonal block and transpose it
    this->diagonal_ = other.diagonal().CloneWithoutContent();
    this->diagonal_->CopyStructureFrom(other.diagonal());
    this->diagonal_->CopyFrom(other.diagonal());
    this->diagonal_->transpose_me();
    this->diagonal_->Compress();

    // all-to-all offdiagonal coo data
    std::vector< DataType > val;
    std::vector< int > col;
    std::vector< int > row;
    other.offdiagonal().get_as_coo(val, col, row);
    // adjust for global row/col
    for (int i = 0; i < col.size(); ++i) {
      col[i] = this->col_couplings_->Offdiag2Global(col[i]);
      row[i] += this->row_ownership_begin_;
    }

    std::vector< int > offs(comm_size_ + 1, 0);
    int size = val.size();
    MPI_Allgather(&size, 1, MPI_INT, &(offs[1]), 1, MPI_INT, comm_);

    for (int r = 0; r < comm_size_; ++r)
      offs[r + 1] += offs[r];

    int total_size = offs.back();
    std::vector< DataType > total_val(total_size);
    std::vector< int > total_col(total_size);
    std::vector< int > total_row(total_size);

    memcpy(&(total_val[offs[my_rank_]]), &(val[0]),
           (offs[my_rank_ + 1] - offs[my_rank_]) * sizeof(DataType));
    memcpy(&(total_col[offs[my_rank_]]), &(col[0]),
           (offs[my_rank_ + 1] - offs[my_rank_]) * sizeof(int));
    memcpy(&(total_row[offs[my_rank_]]), &(row[0]),
           (offs[my_rank_ + 1] - offs[my_rank_]) * sizeof(int));

    for (int r = 0; r < comm_size_; ++r) {
      MPI_Bcast(&(total_val[offs[r]]), offs[r + 1] - offs[r],
                mpi_data_type< DataType >::get_type(), r, comm_);
      MPI_Bcast(&(total_col[offs[r]]), offs[r + 1] - offs[r], MPI_INT, r,
                comm_);
      MPI_Bcast(&(total_row[offs[r]]), offs[r + 1] - offs[r], MPI_INT, r,
                comm_);
    }

    // crunch offdiagonal coo data to figure out my transpose offdiagonal
    val.clear();
    col.clear();
    row.clear();

    for (int r = 0; r < comm_size_; ++r) {
      for (int i = offs[r]; i < offs[r + 1]; ++i) {
        // find transpose entries
        if (total_col[i] >= this->row_ownership_begin_ &&
            total_col[i] < this->row_ownership_end_) {
          val.push_back(total_val[i]);
          col.push_back(total_row[i]);
          row.push_back(total_col[i]);
        }
      }
    }

    for (int i = 0; i < col.size(); ++i) {
      col[i] = this->col_couplings_->Global2Offdiag(col[i]);
      row[i] -= this->row_ownership_begin_;
    }

    if (val.size() > 0) {
      // create empty offdiagonal block
      this->offdiagonal_ = init_matrix< DataType >(
                             val.size(), this->row_couplings_->nb_dofs(this->my_rank_),
                             this->col_couplings_->size_ghost(), "offdiagonal",
                             other.offdiagonal().get_platform(),
                             other.offdiagonal().get_implementation(),
                             other.offdiagonal().get_matrix_format());

      this->offdiagonal_->init_structure(&(row[0]), &(col[0]));
      this->offdiagonal_->Zeros();

      for (int i = 0; i < val.size(); ++i) {
        this->offdiagonal_->add_value(row[i], col[i], val[i]);
      }
      this->offdiagonal_->Compress();
    } else {
      this->offdiagonal_ = init_matrix< DataType >(
                             0, 0, 0, "offdiagonal",
                             other.offdiagonal().get_platform(),
                             other.offdiagonal().get_implementation(),
                             other.offdiagonal().get_matrix_format());
    }
    this->called_init_ = true;
    this->initialized_structure_ = true;
  } else {
    LOG_ERROR("CoupledMatrix::CreateTransposeFrom called on itself.");
    quit_program();
  }
}

template < class DataType > void CoupledMatrix< DataType >::Clear() {
  // clear diagonal block
  if (this->diagonal_ != nullptr)
    delete this->diagonal_;
  this->diagonal_ = nullptr;

  // clear offdiagonal block
  if (this->offdiagonal_ != nullptr)
    delete this->offdiagonal_;
  this->offdiagonal_ = nullptr;

  if (this->col_offdiagonal_ != nullptr)
    delete this->col_offdiagonal_;
  this->col_offdiagonal_ = nullptr;
  
  this->called_init_ = false;
  this->initialized_structure_ = false;
}

template < class DataType >
int CoupledMatrix< DataType >::num_rows_global() const {
  assert(this->called_init_);
  assert(this->row_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());

  return this->row_couplings_->nb_total_dofs();
}

template < class DataType >
int CoupledMatrix< DataType >::num_cols_global() const {
  assert(this->called_init_);
  assert(this->col_couplings_ != nullptr);
  assert(this->col_couplings_->initialized());

  return this->col_couplings_->nb_total_dofs();
}

template < class DataType >
void CoupledMatrix< DataType >::Print(std::ostream &out) const {
  assert(this->IsInitialized());
  this->diagonal_->print(out);
  this->offdiagonal_->print(out);
}

template < class DataType >
void CoupledMatrix< DataType >::ComputeOwnershipRange() {
  assert(this->called_init_);
  assert(this->row_couplings_ != nullptr);
  assert(this->col_couplings_ != nullptr);
  assert(this->row_couplings_->initialized());
  assert(this->col_couplings_->initialized());
  assert(this->my_rank_ >= 0);

  this->row_ownership_begin_ = this->row_couplings_->dof_offset(this->my_rank_);
  this->row_ownership_end_ = this->row_ownership_begin_ +
                             this->row_couplings_->nb_dofs(this->my_rank_);

  this->col_ownership_begin_ = this->col_couplings_->dof_offset(this->my_rank_);
  this->col_ownership_end_ = this->col_ownership_begin_ +
                             this->col_couplings_->nb_dofs(this->my_rank_);
}

template < class DataType >
void CoupledMatrix< DataType >::Scale(const DataType scalar) {
  assert(this->IsInitialized());
  this->diagonal_->Scale(scalar);
  this->offdiagonal_->Scale(scalar);
  this->col_offdiagonal_->Scale(scalar);
}

/// template instantiation
template class CoupledMatrix< double >;
template class CoupledMatrix< float >;

} // namespace la
} // namespace hiflow
