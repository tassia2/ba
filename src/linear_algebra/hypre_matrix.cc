// Copyright (C) 2011-2020 Vincent Heuveline
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

/// @author Simon Gawlok

#include "linear_algebra/hypre_matrix.h"
#include "common/log.h"
#include "common/pointers.h"
#include <cstdlib>
#include <vector>

namespace hiflow {

namespace la {

template < class DataType > HypreMatrix< DataType >::HypreMatrix() {
  this->initialized_ = false;
  this->comm_ = MPI_COMM_NULL;
  this->cp_row_ = nullptr;
  this->cp_col_ = nullptr;

  this->called_init_ = false;
  this->initialized_structure_ = false;
}

template < class DataType > HypreMatrix< DataType >::~HypreMatrix() {

  this->Clear();

  int is_finalized;
  MPI_Finalized(&is_finalized);
  if (is_finalized == 0) {
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
    }
  }
  this->called_init_ = false;
  this->initialized_structure_ = false;
}

template < class DataType >
Matrix< DataType > *HypreMatrix< DataType >::Clone() const {
  LOG_ERROR("HypreMatrix::Clone not yet implemented!!!");
  quit_program();
  return nullptr;
}

template < class DataType >
void HypreMatrix< DataType >::Init(const MPI_Comm &comm,
                                   const LaCouplings &cp_row,
                                   const LaCouplings &cp_col) {
  // clear possibly existing data
  if (this->called_init_) {
    this->Clear();
  }

  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
  }

  assert(comm != MPI_COMM_NULL);

  MPI_Comm_dup(comm, &this->comm_);
  // MPI communicator

  // determine nb. of processes
  int info = MPI_Comm_size(this->comm_, &comm_size_);
  assert(info == MPI_SUCCESS);
  assert(comm_size_ > 0);

  // retrieve my rank
  info = MPI_Comm_rank(this->comm_, &my_rank_);
  assert(info == MPI_SUCCESS);
  assert(my_rank_ >= 0);
  assert(my_rank_ < comm_size_);

  this->cp_row_ = &cp_row;
  this->cp_col_ = &cp_col;
  assert(this->cp_row_ != nullptr);
  assert(this->cp_col_ != nullptr);
  assert(this->cp_row_->initialized());
  assert(this->cp_col_->initialized());

  this->called_init_ = true;
}

template < class DataType >
void HypreMatrix< DataType >::InitStructure(
  const int *rows_diag, const int *cols_diag, const int nnz_diag,
  const int *rows_offdiag, const int *cols_offdiag, const int nnz_offdiag) 
{
  assert(this->called_init_);
  // Get information about size of local matrix
  ilower_ = this->cp_row_->dof_offset(my_rank_);
  iupper_ = ilower_ + this->cp_row_->nb_dofs(my_rank_) - 1;
  jlower_ = this->cp_col_->dof_offset(my_rank_);
  jupper_ = jlower_ + this->cp_col_->nb_dofs(my_rank_) - 1;
  size_local_ = static_cast< size_t >(this->cp_row_->nb_dofs(my_rank_));
  nnz_local_ = static_cast< size_t >(nnz_diag + nnz_offdiag);

  //structure_.resize(size_local_);
  structure_diag_.resize(this->num_rows_local());
  structure_offdiag_.resize(this->num_rows_local());
  row_length_diag_.resize(size_local_, 0);
  row_length_offdiag_.resize(size_local_, 0);

  // 1. step: fill structure_ and row_length_diag_ with diagonal part
  for (size_t i = 0; i < static_cast< size_t >(nnz_diag); ++i) {
    bool found = structure_diag_[rows_diag[i] - ilower_].find_insert(cols_diag[i]);
    if (!found) {
      row_length_diag_[rows_diag[i] - ilower_] += 1;
    }
  }

  // 2. step: fill structure_ and row_length_offdiag_ with offdiagonal part
  for (size_t i = 0; i < static_cast< size_t >(nnz_offdiag); ++i) {
    bool found =
      structure_offdiag_[rows_offdiag[i] - ilower_].find_insert(cols_offdiag[i]);
    if (!found) {
      row_length_offdiag_[rows_offdiag[i] - ilower_] += 1;
    }
  }

  // Create the HYPRE matrix
  HYPRE_IJMatrixCreate(comm_, ilower_, iupper_, jlower_, jupper_, &A_);

  // Use parallel csr format
  HYPRE_IJMatrixSetObjectType(A_, HYPRE_PARCSR);

  HYPRE_IJMatrixSetPrintLevel(A_, 100);

  HYPRE_IJMatrixSetDiagOffdSizes(A_, vec2ptr(row_length_diag_),
                                 vec2ptr(row_length_offdiag_));

  // Tell HYPRE that no matrix entries need to be communicated to other
  // processors
  HYPRE_IJMatrixSetMaxOffProcElmts(A_, 0);

  // Initialize
  HYPRE_IJMatrixInitialize(A_);

  // Now initialize exact structure of matrix. To achieve this we set every
  // element to zero
  this->Zeros();

  HYPRE_IJMatrixAssemble(A_);
  HYPRE_IJMatrixGetObject(A_, (void **)&parcsr_A_);
  this->initialized_ = true;
  this->initialized_structure_ = true;
  this->called_init_ = true;

  if (this->print_level_ >= 0) {
    HYPRE_Int m, n;
    HYPRE_ParCSRMatrixGetDims(parcsr_A_, &m, &n);
    LOG_INFO("Global number of rows", m);
    LOG_INFO("Global number of columns", n);
    LOG_INFO("nnz diag ", nnz_diag << " <> " << hypre_CSRMatrixNumNonzeros(this->parcsr_A_->diag));
    LOG_INFO("nnz offdiag ", nnz_offdiag);
    LOG_INFO("nnz local", nnz_local_);
  }
}

template < class DataType >
void HypreMatrix< DataType >::Init(const MPI_Comm &comm,
                                   HYPRE_ParCSRMatrix *parcsr_temp) {
  assert(parcsr_temp != nullptr);

  if (this->called_init_) {
    this->Clear();
  }

  this->cp_row_ = nullptr;
  this->cp_col_ = nullptr;
  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
    assert(this->comm_ == MPI_COMM_NULL);
  }

  assert(comm != MPI_COMM_NULL);

  // MPI communicator
  MPI_Comm_dup(comm, &this->comm_);

  assert(comm_ != MPI_COMM_NULL);

  // determine nb. of processes
  int info = MPI_Comm_size(this->comm_, &comm_size_);
  assert(info == MPI_SUCCESS);
  assert(comm_size_ > 0);

  // retrieve my rank
  info = MPI_Comm_rank(this->comm_, &my_rank_);
  assert(info == MPI_SUCCESS);
  assert(my_rank_ >= 0);
  assert(my_rank_ < comm_size_);

  // Construct IJMatrix from the ParCSRMatrix
  HYPRE_ParCSRMatrixGetLocalRange(*parcsr_temp, &ilower_, &iupper_, &jlower_,
                                  &jupper_);
  size_local_ = static_cast< size_t >(iupper_ - ilower_ + 1);

  // Create the HYPRE matrix
  HYPRE_IJMatrixCreate(comm_, ilower_, iupper_, jlower_, jupper_, &A_);

  // Use parallel csr format
  HYPRE_IJMatrixSetObjectType(A_, HYPRE_PARCSR);

  HYPRE_IJMatrixSetPrintLevel(A_, 100);

  // Tell HYPRE that no matrix entries need to be communicated to other
  // processors
  HYPRE_IJMatrixSetMaxOffProcElmts(A_, 0);

  // Initialize
  HYPRE_IJMatrixInitialize(A_);

  // set structure
  structure_diag_.clear();
  structure_diag_.resize(this->num_rows_local());
  structure_offdiag_.clear();
  structure_offdiag_.resize(this->num_rows_local());
  nnz_local_ = 0;

  // Get each row of the ParCSRMatrix and set it to the IJMatrix
  HYPRE_Int i, size, *col_ind;
  HYPRE_Complex *values;
  for (i = ilower_; i <= iupper_; i++) {
    // Get the row
    HYPRE_ParCSRMatrixGetRow(*parcsr_temp, i, &size, &col_ind, &values);

    // Fill the structure
    for (HYPRE_Int k = 0; k < size; ++k) {

      if ((jlower_ <= col_ind[k]) && (col_ind[k] <= jupper_)) {
        structure_diag_[i - ilower_].find_insert(col_ind[k]);
      } else {
        structure_offdiag_[i - ilower_].find_insert(col_ind[k]);
      }

    }

    nnz_local_ += (structure_diag_[i - ilower_].size() + structure_offdiag_[i -
                   ilower_].size());
    // Set the row
    HYPRE_IJMatrixSetValues(A_, 1, &size, &i, col_ind, values);

    // Prepare Hypre for the next operation
    HYPRE_ParCSRMatrixRestoreRow(*parcsr_temp, i, &size, &col_ind, &values);
  }

  // Finalize the matrix construction
  HYPRE_IJMatrixAssemble(A_);
  HYPRE_IJMatrixGetObject(A_, (void **)&parcsr_A_);

  this->initialized_ = true;
  this->called_init_ = true;
  this->initialized_structure_ = true;
}

template < class DataType >
void HypreMatrix< DataType >::CloneFromWithoutContent(
  const HypreMatrix< DataType > &mat) {
  if (this != &mat) {
    // clear possibly existing data
    if (this->initialized_) {
      this->Clear();
    }

    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
    }

    assert(mat.comm() != MPI_COMM_NULL);

    MPI_Comm_dup(mat.comm(), &(this->comm_));
    // MPI communicator

    // determine nb. of processes
    int info = MPI_Comm_size(this->comm_, &comm_size_);
    assert(info == MPI_SUCCESS);
    assert(comm_size_ > 0);

    // retrieve my rank
    info = MPI_Comm_rank(this->comm_, &my_rank_);
    assert(info == MPI_SUCCESS);
    assert(my_rank_ >= 0);
    assert(my_rank_ < comm_size_);

    this->cp_row_ = &(mat.row_couplings());
    this->cp_col_ = &(mat.col_couplings());
    assert(this->cp_row_ != nullptr);
    assert(this->cp_col_ != nullptr);
    assert(this->cp_row_->initialized());
    assert(this->cp_col_->initialized());

    // Get information about size of local matrix
    ilower_ = this->cp_row_->dof_offset(my_rank_);
    iupper_ = ilower_ + this->cp_row_->nb_dofs(my_rank_) - 1;
    jlower_ = this->cp_col_->dof_offset(my_rank_);
    jupper_ = jlower_ + this->cp_col_->nb_dofs(my_rank_) - 1;
    size_local_ = static_cast< size_t >(this->cp_row_->nb_dofs(my_rank_));
    nnz_local_ = mat.nnz_local();

    // diag
    this->structure_diag_.resize(mat.Structure_diag().size());
    for (int i = 0; i < mat.Structure_diag().size(); ++i) {
      this->structure_diag_[i].data() = mat.Structure_diag()[i].data();
    }

    // off-diag
    this->structure_offdiag_.resize(mat.Structure_offdiag().size());
    for (int i = 0; i < mat.Structure_offdiag().size(); ++i) {
      this->structure_offdiag_[i].data() = mat.Structure_offdiag()[i].data();
    }

    // 1. step: fill structure_ and row_length_diag_ with diagonal part
    this->row_length_diag_ = mat.RowLengthDiag();

    // 2. step: fill structure_ and row_length_offdiag_ with offdiagonal part
    this->row_length_offdiag_ = mat.RowLengthOffDiag();

    // Create the HYPRE matrix
    HYPRE_IJMatrixCreate(comm_, ilower_, iupper_, jlower_, jupper_, &A_);

    // Use parallel csr format
    HYPRE_IJMatrixSetObjectType(A_, HYPRE_PARCSR);

    HYPRE_IJMatrixSetPrintLevel(A_, 100);

    HYPRE_IJMatrixSetDiagOffdSizes(A_, vec2ptr(row_length_diag_),
                                   vec2ptr(row_length_offdiag_));

    // Tell HYPRE that no matrix entries need to be communicated to other
    // processors
    HYPRE_IJMatrixSetMaxOffProcElmts(A_, 0);

    // Initialize
    HYPRE_IJMatrixInitialize(A_);

    // Now initialize exact structure of matrix. To achieve this we set every
    // element to zero
    this->Zeros();

    HYPRE_IJMatrixAssemble(A_);
    HYPRE_IJMatrixGetObject(A_, (void **)&parcsr_A_);
    this->called_init_ = mat.called_init_;
    this->initialized_structure_ = mat.initialized_structure_;

    if (this->print_level_ > 0) {
      HYPRE_Int m, n;
      HYPRE_ParCSRMatrixGetDims(parcsr_A_, &m, &n);
      LOG_INFO("Global number of rows", m);
      LOG_INFO("Global number of columns", n);
    }

  }

}

template < class DataType >
void HypreMatrix< DataType >::CloneFrom(const HypreMatrix< DataType > &mat) {
  if (this != &mat) {
    this->CloneFromWithoutContent(mat);

    // diag
    for (size_t i = 0; i < this->structure_diag_.size(); ++i) {
      const int current_row = this->ilower_ + i;

      std::vector< int > cols_temp(this->structure_diag_[i].size());
      for (size_t j = 0; j < this->structure_diag_[i].size(); ++j) {
        cols_temp[j] = static_cast< int >(this->structure_diag_[i][j]);
      }

      // Array for copying values
      std::vector< DataType > values(this->structure_diag_[i].size(),
                                     static_cast< DataType >(0));

      // Get values of current row from mat
      mat.GetValues(&current_row, 1, vec2ptr(cols_temp),
                    static_cast< int >(this->structure_diag_[i].size()),
                    vec2ptr(values));

      // Add values to this
      this->Add(&current_row, 1, vec2ptr(cols_temp),
                static_cast< int >(this->structure_diag_[i].size()),
                vec2ptr(values));
    }

    // offdiag
    for (size_t i = 0; i < this->structure_offdiag_.size(); ++i) {
      const int current_row = this->ilower_ + i;

      std::vector< int > cols_temp(this->structure_offdiag_[i].size());
      for (size_t j = 0; j < this->structure_offdiag_[i].size(); ++j) {
        cols_temp[j] = static_cast< int >(this->structure_offdiag_[i][j]);
      }

      // Array for copying values
      std::vector< DataType > values(this->structure_offdiag_[i].size(),
                                     static_cast< DataType >(0));

      // Get values of current row from mat
      mat.GetValues(&current_row, 1, vec2ptr(cols_temp),
                    static_cast< int >(this->structure_offdiag_[i].size()),
                    vec2ptr(values));

      // Add values to this
      this->Add(&current_row, 1, vec2ptr(cols_temp),
                static_cast< int >(this->structure_offdiag_[i].size()),
                vec2ptr(values));
    }

  }

}

template < class DataType > void HypreMatrix< DataType >::Clear() {
  structure_.clear();
  structure_diag_.clear();
  structure_offdiag_.clear();
  
  row_length_diag_.clear();
  row_length_offdiag_.clear();
  if (this->initialized_structure_) {
    HYPRE_IJMatrixDestroy(A_);
  }

  // destroy ?
  // HYPRE_ParCSRMatrix parcsr_A_;
      
  this->cp_row_ = nullptr;
  this->cp_col_ = nullptr;
  this->ilower_ = -1;
  this->iupper_ = -1;
  this->jlower_ = -1;
  this->jupper_ = -1;
  this->size_local_ = 0;
  this->nnz_local_ = 0;
  
  this->initialized_ = false;
  this->initialized_structure_ = false;
  this->called_init_ = false;
}

template < class DataType >
void HypreMatrix< DataType >::print_statistics() const {

  HYPRE_Int ilower, iupper, jlower, jupper;
  HYPRE_IJMatrixGetLocalRange(this->A_, &ilower, &iupper, &jlower, &jupper);

  std::vector< HYPRE_Int > rows(iupper - ilower + 1, 0);
PRAGMA_LOOP_VEC
  for (int i = 0; i < iupper - ilower + 1; ++i) {
    rows[i] = ilower + i;
  }

  std::vector< HYPRE_Int > nnz_count(iupper - ilower + 1, 0);
  HYPRE_IJMatrixGetRowCounts(A_, iupper - ilower + 1, vec2ptr(rows),
                             vec2ptr(nnz_count));

  // print statistics
  for (int i = 0; i < comm_size_; ++i) {
    MPI_Barrier(comm_);
    if (i == my_rank_) {
      std::cout << "HypreMatrix on process " << my_rank_ << ":" << std::endl;
      // print size information
      std::cout << "\t ilower: " << ilower << std::endl;
      std::cout << "\t iupper: " << iupper << std::endl;
      std::cout << "\t jlower: " << jlower << std::endl;
      std::cout << "\t jupper: " << jupper << std::endl;
      std::cout << "\t Nonzero elements (row: nnz)" << std::endl;
      for (int j = 0; j < iupper - ilower + 1; ++j) {
        std::cout << "\t\t " << ilower + j << ": " << nnz_count[j] << std::endl;
      }
    }
    MPI_Barrier(comm_);
  }
}

template < class DataType >
void HypreMatrix< DataType >::Axpy(const HypreMatrix< DataType > &mat,
                                   const DataType alpha) {
  assert(this->num_cols_global() == mat.num_cols_global());
  assert(this->num_cols_local() == mat.num_cols_local());
  assert(this->num_rows_global() == mat.num_rows_global());
  assert(this->num_rows_local() == mat.num_rows_local());

  // diag
  for (size_t i = 0; i < this->structure_diag_.size(); ++i) {
    const int current_row = this->ilower_ + i;

    std::vector< int > cols_temp(this->structure_diag_[i].size());
    for (size_t j = 0; j < this->structure_diag_[i].size(); ++j) {
      cols_temp[j] = static_cast< int >(this->structure_diag_[i][j]);
    }

    // Array for copying values
    std::vector< DataType > values(this->structure_diag_[i].size(),
                                   static_cast< DataType >(0));

    // Get values of current row from mat
    mat.GetValues(&current_row, 1, vec2ptr(cols_temp),
                  static_cast< int >(this->structure_diag_[i].size()),
                  vec2ptr(values));

    // Scale values by alpha
    for (size_t j = 0; j < values.size(); ++j) {
      values[j] *= alpha;
    }

    // Add values to this
    this->Add(&current_row, 1, vec2ptr(cols_temp),
              static_cast< int >(this->structure_diag_[i].size()), vec2ptr(values));
  }

  // offdiag
  for (size_t i = 0; i < this->structure_offdiag_.size(); ++i) {
    const int current_row = this->ilower_ + i;

    std::vector< int > cols_temp(this->structure_offdiag_[i].size());
    for (size_t j = 0; j < this->structure_offdiag_[i].size(); ++j) {
      cols_temp[j] = static_cast< int >(this->structure_offdiag_[i][j]);
    }

    // Array for copying values
    std::vector< DataType > values(this->structure_offdiag_[i].size(),
                                   static_cast< DataType >(0));

    // Get values of current row from mat
    mat.GetValues(&current_row, 1, vec2ptr(cols_temp),
                  static_cast< int >(this->structure_offdiag_[i].size()),
                  vec2ptr(values));

    // Scale values by alpha
    for (size_t j = 0; j < values.size(); ++j) {
      values[j] *= alpha;
    }

    // Add values to this
    this->Add(&current_row, 1, vec2ptr(cols_temp),
              static_cast< int >(this->structure_offdiag_[i].size()), vec2ptr(values));
  }
}

template < class DataType >
void HypreMatrix< DataType >::VectorMult(Vector< DataType > &in,
    Vector< DataType > *out) const {
  HypreVector< DataType > *hv_in, *hv_out;

  hv_in = dynamic_cast< HypreVector< DataType > * >(&in);
  hv_out = dynamic_cast< HypreVector< DataType > * >(out);

  if ((hv_in != 0) && (hv_out != 0)) {
    this->VectorMult(*hv_in, hv_out);
  } else {
    if (hv_in == 0) {
      LOG_ERROR("Called HypreMatrix::VectorMult with incompatible input vector "
                "type.");
    }
    if (hv_out == 0) {
      LOG_ERROR("Called HypreMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class DataType >
void HypreMatrix< DataType >::VectorMult(HypreVector< DataType > &in,
    HypreVector< DataType > *out) const {

  assert(this->IsInitialized());
  assert(in.is_initialized());
  assert(out->is_initialized());

  HYPRE_ParCSRMatrixMatvec(
    static_cast< HYPRE_Complex >(1), parcsr_A_, *(in.GetParVector()),
    static_cast< HYPRE_Complex >(0), *(out->GetParVector()));
}

template < class DataType >
void HypreMatrix< DataType >::MatrixMult(Matrix< DataType > &inA,
    Matrix< DataType > &inB) {
  HypreMatrix< DataType > *hv_inA, *hv_inB;

  hv_inA = dynamic_cast< HypreMatrix< DataType > * >(&inA);
  hv_inB = dynamic_cast< HypreMatrix< DataType > * >(&inB);

  if ((hv_inA != 0) && (hv_inB != 0)) {
    this->MatrixMult(*hv_inA, *hv_inB);
  } else {
    if (hv_inA == 0) {
      LOG_ERROR("Called HypreMatrix::MatrixMult with incompatible input matrix "
                "A type.");
    }
    if (hv_inB == 0) {
      LOG_ERROR("Called HypreMatrix::MatrixMult with incompatible input matrix "
                "B type.");
    }
    quit_program();
  }
}

template < class DataType >
void HypreMatrix< DataType >::MatrixMult(HypreMatrix< DataType > &inA,
    HypreMatrix< DataType > &inB) {
  assert(inA.IsInitialized());
  assert(inB.IsInitialized());

  // Matrix-Matrix multiplication on temporary parcsr object
  HYPRE_ParCSRMatrix parcsr_temp =
    hypre_ParMatmul(*(inA.GetParCSRMatrix()), *(inB.GetParCSRMatrix()));

  // initialize from parcsr object
  this->Init(inA.comm(), &parcsr_temp);

  // Clean up temp data
  HYPRE_ParCSRMatrixDestroy(parcsr_temp);
}

template < class DataType >
void HypreMatrix< DataType >::VectorMultAdd(DataType alpha,
    Vector< DataType > &in,
    DataType beta,
    Vector< DataType > *out) const {
  HypreVector< DataType > *hv_in, *hv_out;

  hv_in = dynamic_cast< HypreVector< DataType > * >(&in);
  hv_out = dynamic_cast< HypreVector< DataType > * >(out);

  if ((hv_in != 0) && (hv_out != 0)) {
    this->VectorMultAdd(alpha, *hv_in, beta, hv_out);
  } else {
    if (hv_in == 0) {
      LOG_ERROR("Called HypreMatrix::VectorMult with incompatible input vector "
                "type.");
    }
    if (hv_out == 0) {
      LOG_ERROR("Called HypreMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }

}

template < class DataType >
void HypreMatrix< DataType >::VectorMultAdd(
  DataType alpha, HypreVector< DataType > &in, DataType beta,
  HypreVector< DataType > *out) const {
  assert(this->IsInitialized());
  assert(out != nullptr);
  assert(out->is_initialized());
  assert(in.is_initialized());
  HYPRE_ParCSRMatrixMatvec(
    static_cast< HYPRE_Complex >(alpha), parcsr_A_, *(in.GetParVector()),
    static_cast< HYPRE_Complex >(beta), *(out->GetParVector()));
}

template < class DataType >
void HypreMatrix< DataType >::GetValues(const int *row_indices, int num_rows,
                                        const int *col_indices, int num_cols,
                                        DataType *values) const {
  assert(this->IsInitialized());

  for (size_t i = 0; i < num_rows; ++i) {
    std::vector< HYPRE_Int > col_ind_row, col_map_row;
    std::vector< HYPRE_Complex > val_row;
    col_ind_row.reserve(num_cols);
    col_map_row.reserve(num_cols);

    HYPRE_Int current_row_global = row_indices[i];
    const HYPRE_Int current_row_local = current_row_global - ilower_;

    // diag
    if (!this->structure_diag_[current_row_local].empty()) {

      int k = 0;
      const HYPRE_Int *const struct_curr_row =
        &(structure_diag_[current_row_local][0]);

      for (size_t j = 0, j_e = structure_diag_[current_row_local].size();
           k < num_cols && j != j_e; ++j) {

        if (col_indices[k] == struct_curr_row[j]) {

          col_ind_row.push_back(col_indices[k]);
          col_map_row.push_back(k);
          ++k;

        } else if (col_indices[k] < struct_curr_row[j]) {

          while (col_indices[k] < struct_curr_row[j] && k < num_cols) {
            ++k;
          }

          if (k >= num_cols) {
            break;
          }

          if (col_indices[k] == struct_curr_row[j]) {
            col_ind_row.push_back(col_indices[k]);
            col_map_row.push_back(k);
            ++k;
          }
        }
      }

      HYPRE_Int ncols = col_ind_row.size();
      val_row.resize(col_ind_row.size(), 0.);

      if (ncols > 0) {
        HYPRE_IJMatrixGetValues(A_, 1, &ncols, &current_row_global,
                                vec2ptr(col_ind_row), vec2ptr(val_row));

        const size_t offset = i * num_cols;
        for (size_t j = 0; j < ncols; ++j) {
          values[offset + col_map_row[j]] = val_row[j];
        }
      }

      col_ind_row.clear();
      col_map_row.clear();
      val_row.clear();
    }

    // offdiag
    if (!this->structure_offdiag_[current_row_local].empty()) {

      int k = 0;
      const HYPRE_Int *const struct_curr_row =
        &(structure_offdiag_[current_row_local][0]);

      for (size_t j = 0, j_e = structure_offdiag_[current_row_local].size();
           k < num_cols && j != j_e; ++j) {

        if (col_indices[k] == struct_curr_row[j]) {

          col_ind_row.push_back(col_indices[k]);
          col_map_row.push_back(k);
          ++k;

        } else if (col_indices[k] < struct_curr_row[j]) {

          while (col_indices[k] < struct_curr_row[j] && k < num_cols) {
            ++k;
          }

          if (k >= num_cols) {
            break;
          }

          if (col_indices[k] == struct_curr_row[j]) {
            col_ind_row.push_back(col_indices[k]);
            col_map_row.push_back(k);
            ++k;
          }

        }
      }

      HYPRE_Int ncols = col_ind_row.size();
      val_row.resize(col_ind_row.size(), 0.);

      if (ncols > 0) {
        HYPRE_IJMatrixGetValues(A_, 1, &ncols, &current_row_global,
                                vec2ptr(col_ind_row), vec2ptr(val_row));

        const size_t offset = i * num_cols;
        for (size_t j = 0; j < ncols; ++j) {
          values[offset + col_map_row[j]] = val_row[j];
        }
      }

      col_ind_row.clear();
      col_map_row.clear();
      val_row.clear();
    }

  }

}

template < class DataType >
void HypreMatrix< DataType >::Add(int global_row_id, int global_col_id,
                                  DataType value) {
  this->Add(&global_row_id, 1, &global_col_id, 1, &value);
}

template < class DataType >
void HypreMatrix< DataType >::Add(const int *rows, int num_rows,
                                  const int *cols, int num_cols,
                                  const DataType *values) {
  assert(this->IsInitialized());

  if (num_rows == 1 && num_cols == 1) {
    const int current_row_local = rows[0] - ilower_;
    int pos = -1;

    // diag
    if (this->structure_diag_[current_row_local].find(cols[0], &pos)) {
      assert(pos >= 0);
      HYPRE_Int ncols_add = 1;
      HYPRE_Int rows_temp = static_cast< HYPRE_Int >(rows[0]);
      HYPRE_Int cols_temp = static_cast< HYPRE_Int >(cols[0]);
      HYPRE_Complex val_temp = static_cast< HYPRE_Complex >(values[0]);
      HYPRE_IJMatrixAddToValues(A_, 1, &ncols_add, &rows_temp, &cols_temp,
                                &val_temp);
    }

    // offdiag
    if (this->structure_offdiag_[current_row_local].find(cols[0], &pos)) {
      assert(pos >= 0);
      HYPRE_Int ncols_add = 1;
      HYPRE_Int rows_temp = static_cast< HYPRE_Int >(rows[0]);
      HYPRE_Int cols_temp = static_cast< HYPRE_Int >(cols[0]);
      HYPRE_Complex val_temp = static_cast< HYPRE_Complex >(values[0]);
      HYPRE_IJMatrixAddToValues(A_, 1, &ncols_add, &rows_temp, &cols_temp,
                                &val_temp);
    }

  } else {

    int nrows_add = 0;
    std::vector< HYPRE_Int > ncols_add;
    ncols_add.reserve(num_rows);
    std::vector< HYPRE_Int > rows_add;
    rows_add.reserve(num_rows);
    std::vector< HYPRE_Int > cols_add;
    cols_add.reserve(num_rows * num_cols);
    std::vector< HYPRE_Complex > vals_add;
    vals_add.reserve(num_rows * num_cols);

    for (size_t i = 0; i != static_cast< size_t >(num_rows); ++i) {
      int cols_in_row = 0;

      const int current_row_global = rows[i];
      const int current_row_local = current_row_global - ilower_;
      int k = 0;
      const HYPRE_Int row_offset = i * num_cols;

      // diag
      if (!this->structure_diag_[current_row_local].empty()) {

        const HYPRE_Int *const struct_curr_row =
          &(structure_diag_[current_row_local][0]);

        for (size_t j = 0, j_e = structure_diag_[current_row_local].size();
             k < num_cols && j != j_e; ++j) {

          if (cols[k] == struct_curr_row[j]) {
            cols_add.push_back(cols[k]);
            vals_add.push_back(
              static_cast< HYPRE_Complex >(values[row_offset + k]));
            ++cols_in_row;
            ++k;

          } else if (cols[k] < struct_curr_row[j]) {

            //while (cols[k] < struct_curr_row[j] && k < num_cols) {
            //  ++k;
            //}
            while (k < num_cols)
            {
              if (cols[k] >= struct_curr_row[j])
              {
                break;
              }
              ++k;
            }
            
            if (k >= num_cols) {
              break;
            }

            if (cols[k] == struct_curr_row[j]) {
              cols_add.push_back(cols[k]);
              vals_add.push_back(
                static_cast< HYPRE_Complex >(values[row_offset + k]));
              ++cols_in_row;
              ++k;
            }

          }

        }

        ncols_add.push_back(cols_in_row);
        rows_add.push_back(current_row_global);
        ++nrows_add;
      }

      k = 0;
      cols_in_row = 0;
      // offdiag
      if (!this->structure_offdiag_[current_row_local].empty()) {

        const HYPRE_Int *const struct_curr_row =
          &(structure_offdiag_[current_row_local][0]);

        for (size_t j = 0, j_e = structure_offdiag_[current_row_local].size();
             k < num_cols && j != j_e; ++j) {

          if (cols[k] == struct_curr_row[j]) {

            cols_add.push_back(cols[k]);
            vals_add.push_back(
              static_cast< HYPRE_Complex >(values[row_offset + k]));
            ++cols_in_row;
            ++k;

          } else if (cols[k] < struct_curr_row[j]) {

            //while (cols[k] < struct_curr_row[j] && k < num_cols) {
            //   ++k;
            // }
            while (k < num_cols)
            {
              if (cols[k] >= struct_curr_row[j])
              {
                break;
              }
              ++k;
            }
    
            if (k >= num_cols) {
              break;
            }

            if (cols[k] == struct_curr_row[j]) {
              cols_add.push_back(cols[k]);
              vals_add.push_back(
                static_cast< HYPRE_Complex >(values[row_offset + k]));
              ++cols_in_row;
              ++k;
            }
          }

        }

        ncols_add.push_back(cols_in_row);
        rows_add.push_back(current_row_global);
        ++nrows_add;
      }
    }

    HYPRE_IJMatrixAddToValues(A_, nrows_add, vec2ptr(ncols_add),
                              vec2ptr(rows_add), vec2ptr(cols_add),
                              vec2ptr(vals_add));

  }
}

template < class DataType >
void HypreMatrix< DataType >::Add(const std::vector<int>& row_ind, 
                                  const std::vector<int>& col_ind,
                                  const std::vector<DataType>& values) 
{
  const size_t num_entries = row_ind.size();

  assert(num_entries == col_ind.size());
  assert(num_entries == values.size());

  for (int i = 0; i != num_entries; ++i)
  {
    this->Add(row_ind[i], col_ind[i], values[i]);
  }
}

template < class DataType >
void HypreMatrix< DataType >::SetValue(int row, int col, DataType value) {
  this->SetValues(&row, 1, &col, 1, &value);
}

// TODO: skip offdiagonal rows
template < class DataType >
void HypreMatrix< DataType >::SetValues(const int *row_indices, int num_rows,
                                        const int *col_indices, int num_cols,
                                        const DataType *values) {

  assert(this->IsInitialized());

  HYPRE_Int nrows_add = 0;
  std::vector< HYPRE_Int > ncols_add;
  ncols_add.reserve(num_rows);
  std::vector< HYPRE_Int > rows_add;
  rows_add.reserve(num_rows);
  std::vector< HYPRE_Int > cols_add;
  cols_add.reserve(num_rows * num_cols);
  std::vector< HYPRE_Complex > vals_add;
  vals_add.reserve(num_rows * num_cols);

  for (size_t i = 0; i != static_cast< size_t >(num_rows); ++i) {
    int cols_in_row = 0;
    int k = 0;

    const int current_row_global = row_indices[i];
    const int current_row_local = current_row_global - ilower_;
    const int row_offset = i * num_cols;

    // diag
    if (!this->structure_diag_[current_row_local].empty()) {

      const HYPRE_Int *const struct_curr_row =
        &(structure_diag_[current_row_local][0]);

      for (size_t j = 0, j_e = structure_diag_[current_row_local].size();
           k < num_cols && j != j_e; ++j) {

        if (col_indices[k] == struct_curr_row[j]) {

          cols_add.push_back(col_indices[k]);
          vals_add.push_back(
            static_cast< HYPRE_Complex >(values[row_offset + k]));
          ++cols_in_row;
          ++k;

        } else if (col_indices[k] < struct_curr_row[j]) {

          while (col_indices[k] < struct_curr_row[j] && k < num_cols) {
            ++k;
          }

          if (k >= num_cols) {
            break;
          }

          if (col_indices[k] == struct_curr_row[j]) {
            cols_add.push_back(col_indices[k]);
            vals_add.push_back(
              static_cast< HYPRE_Complex >(values[row_offset + k]));
            ++cols_in_row;
            ++k;
          }
        }

      }

      ncols_add.push_back(cols_in_row);
      rows_add.push_back(current_row_global);
      ++nrows_add;
    }

    k = 0;
    cols_in_row = 0;
    // offdiag
    if (!this->structure_offdiag_[current_row_local].empty()) {

      const HYPRE_Int *const struct_curr_row =
        &(structure_offdiag_[current_row_local][0]);

      for (size_t j = 0, j_e = structure_offdiag_[current_row_local].size();
           k < num_cols && j != j_e; ++j) {

        if (col_indices[k] == struct_curr_row[j]) {

          cols_add.push_back(col_indices[k]);
          vals_add.push_back(
            static_cast< HYPRE_Complex >(values[row_offset + k]));
          ++cols_in_row;
          ++k;

        } else if (col_indices[k] < struct_curr_row[j]) {

          while (col_indices[k] < struct_curr_row[j] && k < num_cols) {
            ++k;
          }

          if (k >= num_cols) {
            break;
          }

          if (col_indices[k] == struct_curr_row[j]) {
            cols_add.push_back(col_indices[k]);
            vals_add.push_back(
              static_cast< HYPRE_Complex >(values[row_offset + k]));
            ++cols_in_row;
            ++k;
          }
        }

      }

      ncols_add.push_back(cols_in_row);
      rows_add.push_back(current_row_global);
      ++nrows_add;
    }
  }

  HYPRE_IJMatrixSetValues(A_, nrows_add, vec2ptr(ncols_add), vec2ptr(rows_add),
                          vec2ptr(cols_add), vec2ptr(vals_add));
}

template < class DataType > void HypreMatrix< DataType >::Zeros() {

  // diag
  for (size_t i = 0, e_i = structure_diag_.size(); i != e_i; ++i) {
    HYPRE_Int row_index_global = this->ilower_ + static_cast< HYPRE_Int >(i);
    HYPRE_Int ncols_zero = static_cast< HYPRE_Int >
                           (this->structure_diag_[i].size());
    std::vector< HYPRE_Complex > zero_val(this->structure_diag_[i].size(), 0.);

    if (ncols_zero > 0) {
      HYPRE_IJMatrixSetValues(this->A_, 1, &ncols_zero, &row_index_global,
                              vec2ptr(this->structure_diag_[i].data()),
                              vec2ptr(zero_val));
    }
  }

  // offdiag
  for (size_t i = 0, e_i = structure_offdiag_.size(); i != e_i; ++i) {
    HYPRE_Int row_index_global = this->ilower_ + static_cast< HYPRE_Int >(i);
    HYPRE_Int ncols_zero = static_cast< HYPRE_Int >
                           (this->structure_offdiag_[i].size());
    std::vector< HYPRE_Complex > zero_val(this->structure_offdiag_[i].size(), 0.);

    if (ncols_zero > 0) {
      HYPRE_IJMatrixSetValues(this->A_, 1, &ncols_zero, &row_index_global,
                              vec2ptr(this->structure_offdiag_[i].data()),
                              vec2ptr(zero_val));
    }
  }
}

template < class DataType >
void HypreMatrix< DataType >::SetCSR(const int* ia, const int* ja,
                                     const DataType* val) {
  NOT_YET_IMPLEMENTED; // buggy
  assert(this->IsInitialized());

  // get diagonal and off-diagonal par_csr object
  hypre_CSRMatrix const *diag = this->parcsr_A_->diag;
  hypre_CSRMatrix const *off_diag = this->parcsr_A_->offd;

  // get row offsets: 1. diag + off diag
  assert(hypre_CSRMatrixNumRows(diag) == hypre_CSRMatrixNumRows(off_diag));

  int counter = 0;
  for (int i = 0; i < diag->num_rows; ++i) {

    // diagonal part
    for (int j = diag->i[i]; j < diag->i[i+1]; ++j) {

      hypre_CSRMatrixData(diag)[j] = static_cast< HYPRE_Complex >(val[counter]);
      ++counter;
    }

    // off-diagonal part
    for (int j = off_diag->i[i]; j < off_diag->i[i+1]; ++j) {

      hypre_CSRMatrixData(off_diag)[j] = static_cast< HYPRE_Complex >(val[counter]);
      ++counter;
    }

  } // i

  assert(counter == this->nnz_local());
}

template < class DataType >
void HypreMatrix< DataType >::diagonalize_rows(const int *row_indices,
    int num_rows,
    DataType diagonal_value) {
  assert(this->IsInitialized());

  for (size_t i = 0; i != num_rows; ++i) {
    assert(row_indices[i] >= ilower_ && row_indices[i] <= iupper_);
    const HYPRE_Int row_index_global = row_indices[i];
    const HYPRE_Int row_index_local = row_index_global - ilower_;
    HYPRE_Int ncols = structure_diag_[row_index_local].size();
    std::vector< HYPRE_Complex > val(ncols, 0.);

    int pos = -1;
    bool found = structure_diag_[row_index_local].find(row_index_global, &pos);
    if (found) {
      assert(pos >= 0);
      val[pos] = static_cast< HYPRE_Complex >(diagonal_value);
    }

    HYPRE_IJMatrixSetValues(A_, 1, &ncols, &row_index_global,
                            &(structure_diag_[row_index_local].front()),
                            vec2ptr(val));

    val.clear();
  }
}

template < class DataType >
void HypreMatrix< DataType >::Scale(const DataType alpha) {
  assert(this->IsInitialized());

  // diag
  for (size_t i = 0, e_i = structure_diag_.size(); i != e_i; ++i) {

    HYPRE_Int ncols = static_cast< HYPRE_Int >(structure_diag_[i].size());

    if (ncols > 0) {
      HYPRE_Int global_row_index = ilower_ + i;
      std::vector< HYPRE_Complex > val(ncols, 0.);
      HYPRE_IJMatrixGetValues(A_, 1, &ncols, &global_row_index,
                              &(structure_diag_[i].front()), vec2ptr(val));

      for (size_t j = 0; j != static_cast< size_t >(ncols); ++j) {
        val[j] *= static_cast< HYPRE_Complex >(alpha);
      }
      HYPRE_IJMatrixSetValues(A_, 1, &ncols, &global_row_index,
                              &(structure_diag_[i].front()), vec2ptr(val));

      val.clear();
    }
  }

  // offdiag
  for (size_t i = 0, e_i = structure_offdiag_.size(); i != e_i; ++i) {
    HYPRE_Int ncols = static_cast< HYPRE_Int >(structure_offdiag_[i].size());
    if (ncols > 0) {
      HYPRE_Int global_row_index = ilower_ + i;
      std::vector< HYPRE_Complex > val(ncols, 0.);
      HYPRE_IJMatrixGetValues(A_, 1, &ncols, &global_row_index,
                              &(structure_offdiag_[i].front()), vec2ptr(val));

      for (size_t j = 0; j != static_cast< size_t >(ncols); ++j) {
        val[j] *= static_cast< HYPRE_Complex >(alpha);
      }
      HYPRE_IJMatrixSetValues(A_, 1, &ncols, &global_row_index,
                              &(structure_offdiag_[i].front()), vec2ptr(val));

      val.clear();
    }
  }

}

template < class DataType >
void HypreMatrix< DataType >::ScaleRows(const int *row_indices, int num_rows,
                                        const DataType *alphas) {
  assert(this->IsInitialized());

  // diag
  for (size_t i = 0; i != static_cast< size_t >(num_rows); ++i) {
    assert(row_indices[i] >= ilower_ && row_indices[i] <= iupper_);

    HYPRE_Int row_index_global = static_cast< HYPRE_Int >(row_indices[i]);
    const int row_index_local = row_index_global - ilower_;
    HYPRE_Int ncols =
      static_cast< HYPRE_Int >(structure_diag_[row_index_local].size());

    std::vector< HYPRE_Complex > val(ncols, 0.);

    HYPRE_IJMatrixGetValues(A_, 1, &ncols, &row_index_global,
                            &(structure_diag_[i].front()), vec2ptr(val));
    for (size_t j = 0; j != ncols; ++j) {
      val[j] *= static_cast< HYPRE_Complex >(alphas[i]);
    }

    HYPRE_IJMatrixSetValues(A_, 1, &ncols, &row_index_global,
                            &(structure_diag_[i].front()), vec2ptr(val));
    val.clear();
  }

  // offdiag
  for (size_t i = 0; i != static_cast< size_t >(num_rows); ++i) {
    assert(row_indices[i] >= ilower_ && row_indices[i] <= iupper_);

    HYPRE_Int row_index_global = static_cast< HYPRE_Int >(row_indices[i]);
    const int row_index_local = row_index_global - ilower_;
    HYPRE_Int ncols =
      static_cast< HYPRE_Int >(structure_offdiag_[row_index_local].size());

    std::vector< HYPRE_Complex > val(ncols, 0.);

    HYPRE_IJMatrixGetValues(A_, 1, &ncols, &row_index_global,
                            &(structure_offdiag_[i].front()), vec2ptr(val));
    for (size_t j = 0; j != ncols; ++j) {
      val[j] *= static_cast< HYPRE_Complex >(alphas[i]);
    }

    HYPRE_IJMatrixSetValues(A_, 1, &ncols, &row_index_global,
                            &(structure_offdiag_[i].front()), vec2ptr(val));
    val.clear();
  }

}

template < class DataType >
void HypreMatrix< DataType >::ExtractDiagValues(
  std::vector< DataType > &vals) const {

  assert(this->IsInitialized());
  HYPRE_Int num_diag = static_cast< HYPRE_Int >(this->num_rows_local());

  std::vector< HYPRE_Complex > vals_temp(num_diag, 0.);
  vals.clear();
  vals.resize(num_diag, 0.);

  std::vector< HYPRE_Int > rows(num_diag, 0);
  std::vector< HYPRE_Int > ncols(num_diag, 1);
  std::vector< HYPRE_Int > cols(num_diag, 0);

  for (size_t i = this->ilower_; i != this->iupper_ + 1; ++i) {
    rows[i - this->ilower_] = i;
    cols[i - this->ilower_] = i;
  }

  HYPRE_IJMatrixGetValues(A_, num_diag, vec2ptr(ncols), vec2ptr(rows),
                          vec2ptr(cols), vec2ptr(vals_temp));

  for (size_t i = 0; i < vals_temp.size(); ++i) {
    vals[i] = static_cast< DataType >(vals_temp[i]);
  }
}

template < class DataType >
void HypreMatrix< DataType >::CreateTransposedFrom(
  const HypreMatrix< DataType > &other) {

  if (this != &other) {
    this->Clear();
    this->comm_size_ = other.comm_size();
    this->my_rank_ = other.my_rank();
    int info = 0;
    if (this->comm_ != MPI_COMM_NULL) {
      info = MPI_Comm_free(&this->comm_);
      assert(info == MPI_SUCCESS);
    }

    assert(other.comm() != MPI_COMM_NULL);

    MPI_Comm_dup(other.comm(), &(this->comm_));

    //info = MPI_Comm_split(other.comm(), 0, this->my_rank_, &(this->comm_));
    info = MPI_Comm_size(this->comm_, &comm_size_);
    assert(info == MPI_SUCCESS);
    assert(comm_size_ > 0);

    // retrieve my rank
    info = MPI_Comm_rank(this->comm_, &my_rank_);
    assert(info == MPI_SUCCESS);
    assert(my_rank_ >= 0);
    assert(my_rank_ < comm_size_);

    this->cp_row_ = &(other.col_couplings());
    this->cp_col_ = &(other.row_couplings());
    this->ilower_ = this->cp_row_->dof_offset(my_rank_);
    this->iupper_ = ilower_ + this->cp_row_->nb_dofs(my_rank_) - 1;
    this->jlower_ = this->cp_col_->dof_offset(my_rank_);
    this->jupper_ = jlower_ + this->cp_col_->nb_dofs(my_rank_) - 1;
    this->size_local_ = static_cast< size_t >(this->cp_row_->nb_dofs(my_rank_));

    hypre_ParCSRMatrix * parcsr_temp;
    hypre_ParCSRMatrixTranspose(*(other.GetParCSRMatrix()), &parcsr_temp, 1);

    hypre_CSRMatrix const* other_diagT = parcsr_temp->diag;
    hypre_CSRMatrix const* other_off_diagT = parcsr_temp->offd;

    const int nnz_diag = hypre_CSRMatrixNumNonzeros(other_diagT);
    const int nnz_offdiag = hypre_CSRMatrixNumNonzeros(other_off_diagT);

    this->nnz_local_ = nnz_diag + nnz_offdiag;

    //structure_.resize(this->size_local_);
    structure_diag_.resize(this->num_rows_local());
    structure_offdiag_.resize(this->num_rows_local());
    row_length_diag_.resize(this->size_local_, 0);
    row_length_offdiag_.resize(this->size_local_, 0);

    // check the row size of diagT and off_diagT
    assert(this->size_local_ ==
           hypre_CSRMatrixNumRows(other_off_diagT));

    int counter = 0;
    // diag
    for (int i = 0; i < other_diagT->num_rows; ++i) {
      this->row_length_diag_[i] = other_diagT->i[i+1] - other_diagT->i[i];

      // filling structure_diag_
      for (int j = 0; j < this->row_length_diag_[i]; ++j) {

        const int g_col = other_diagT->j[counter] + this->jlower_;
        this->structure_diag_[i].find_insert(g_col);
        ++counter;

      }
    }

    counter = 0;
    // off diag
    for (int i = 0; i < other_off_diagT->num_rows; ++i) {
      this->row_length_offdiag_[i] = other_off_diagT->i[i+1] - other_off_diagT->i[i];

      // filling structure_offdiag_
      for (int j = 0; j < this->row_length_offdiag_[i]; ++j) {

        const int g_col = other_off_diagT->j[counter] + this->jlower_;
        this->structure_offdiag_[i].find_insert(g_col);
        ++counter;

      }
    }

    // Create the HYPRE matrix
    HYPRE_IJMatrixCreate(comm_, ilower_, iupper_, jlower_, jupper_, &(this->A_));

    // Use parallel csr format
    HYPRE_IJMatrixSetObjectType(this->A_, HYPRE_PARCSR);

    HYPRE_IJMatrixSetPrintLevel(this->A_, 100);

    HYPRE_IJMatrixSetDiagOffdSizes(this->A_, vec2ptr(this->row_length_diag_),
                                   vec2ptr(this->row_length_offdiag_));

    // Tell HYPRE that no matrix entries need to be communicated to other
    // processors
    HYPRE_IJMatrixSetMaxOffProcElmts(A_, 0);
    // Initialize
    HYPRE_IJMatrixInitialize(A_);

    // set flags
    this->called_init_ = true;
    this->initialized_structure_ = true;
    this->initialized_ = true;

    // Get each row of the ParCSRMatrix and set it to the IJMatrix
    HYPRE_Int size, *col_ind;
    HYPRE_Complex *values;
    for (int i = this->ilower_; i <= this->iupper_; ++i) {
      // Get the row
      HYPRE_ParCSRMatrixGetRow(parcsr_temp, i, &size, &col_ind, &values);
      // Set the row
      HYPRE_IJMatrixSetValues(A_, 1, &size, &i, col_ind, values);
      // Prepare Hypre for the next operation
      HYPRE_ParCSRMatrixRestoreRow(parcsr_temp, i, &size, &col_ind, &values);
    }

    HYPRE_IJMatrixAssemble(A_);

    HYPRE_IJMatrixGetObject(A_, (void **)&parcsr_A_);

    hypre_ParCSRMatrixDestroy(parcsr_temp);

  } else {
    LOG_ERROR("HypreMatrix::CreateTransposeFrom called on itself.");
    exit(-1);
  }

}

template < class DataType >
void HypreMatrix< DataType >::ExtractInvDiagValues(
  const DataType eps, const DataType default_val,
  std::vector< DataType > &vals) const {

  assert(this->IsInitialized());
  std::vector< DataType > diag_vals;
  this->ExtractDiagValues(diag_vals);
  size_t num_diag = diag_vals.size();

  vals.clear();
  vals.resize(num_diag, default_val);

  for (size_t l = 0; l != num_diag; ++l) {
    if (std::abs(diag_vals[l]) > eps) {
      assert(diag_vals[l] != 0.);
      vals[l] = 1. / diag_vals[l];
    }
  }
}

template < class DataType >
void HypreMatrix< DataType >::ExtractCSR(int* ia, int* ja,
    DataType* val) const {
  assert(this->IsInitialized());

  // get diagonal and off-diagonal par_csr object
  hypre_CSRMatrix const *diag = this->parcsr_A_->diag;
  hypre_CSRMatrix const *off_diag = this->parcsr_A_->offd;

  // get map of colums of off-diagonal to global colums
  int const *col_map_offd = this->parcsr_A_->col_map_offd;

  // get row offsets: 1. diag + off diag
  assert(hypre_CSRMatrixNumRows(diag) == hypre_CSRMatrixNumRows(off_diag));
  assert(structure_diag_.size() == structure_offdiag_.size());

  int counter = 0;
  for (int i = 0; i < diag->num_rows; ++i) {
    ia[i] = counter;

    // diagonal part
    for (int j = diag->i[i]; j < diag->i[i+1]; ++j) {

      ja[counter] = static_cast< int >(hypre_CSRMatrixJ(diag)[j]) + this->jlower_;
      //ja[counter] = diag->j[j] + this->jlower_;
      //val[counter] = diag->data[j];
      val[counter] = static_cast< DataType >(hypre_CSRMatrixData(diag)[j]);
      ++counter;
    }

    // off diagonal part
    for (int j = off_diag->i[i]; j < off_diag->i[i+1]; ++j) {

      //ja[counter] = static_cast< int >(hypre_CSRMatrixJ(off_diag)[j]) + this->jlower_;
      //ja[counter] = off_diag->j[j] + this->jlower_;
      ja[counter] = col_map_offd[off_diag->j[j]];
      val[counter] = static_cast< DataType >(hypre_CSRMatrixData(off_diag)[j]);
      ++counter;
    }

  } // i

  ia[structure_diag_.size()] = counter;

}

template < class DataType >
void HypreMatrix< DataType >::ExtractDiagonalCSR(int *ia, int *ja,
    DataType *val) const {
  NOT_YET_IMPLEMENTED;// buggy
  assert(this->IsInitialized());

  // get diagonal par_csr object
  hypre_CSRMatrix const *diag = this->parcsr_A_->diag;

  // get row offsets
  for (int i = 0; i < hypre_CSRMatrixNumRows(diag) + 1; i++) {
    ia[i] = static_cast< int >(hypre_CSRMatrixI(diag)[i]);
  }

  // get column indices and values
  for (int j = 0; j < hypre_CSRMatrixNumNonzeros(diag); j++) {
    ja[j] = static_cast< int >(hypre_CSRMatrixJ(diag)[j]);
    val[j] = static_cast< DataType >(hypre_CSRMatrixData(diag)[j]);
  }
}

template < class DataType >
void HypreMatrix< DataType >::GetDiagonalCSR(int *& ia, int *& ja, DataType *& val) 
{
  NOT_YET_IMPLEMENTED; // buggy
  assert(this->IsInitialized());

  // get diagonal par_csr object
  hypre_CSRMatrix const *diag = hypre_ParCSRMatrixDiag(this->parcsr_A_);

  // get row offsets
  ia = hypre_CSRMatrixI(diag);
  ja = hypre_CSRMatrixJ(diag);
  val = hypre_CSRMatrixData(diag);
}

template < class DataType >
HYPRE_ParCSRMatrix *HypreMatrix< DataType >::GetParCSRMatrix() {
  assert(this->IsInitialized());
  HYPRE_IJMatrixGetObject(A_, (void **)&parcsr_A_);
  return &parcsr_A_;
}

template < class DataType >
const HYPRE_ParCSRMatrix *HypreMatrix< DataType >::GetParCSRMatrix() const {
  return &parcsr_A_;
}

template < class DataType >
int HypreMatrix< DataType >::nnz_global() const {
  int nnz_global;
  MPI_Allreduce(&(this->nnz_local_), &nnz_global, 1, MPI_INT, MPI_SUM,
                this->comm_);
  return nnz_global;
}

template class HypreMatrix< double >;
} // namespace la
} // namespace hiflow
