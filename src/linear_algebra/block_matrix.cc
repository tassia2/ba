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

/// @author Jonas Kratzke

#include "linear_algebra/block_matrix.h"
#include "common/log.h"
#include "common/pointers.h"
#include <cstdlib>
#include <vector>

namespace hiflow {

namespace la {

template < class LAD > 
BlockMatrix< LAD >::BlockMatrix() {
  this->initialized_structure_ = false;
  this->called_init_ = false;
  this->comm_ = MPI_COMM_NULL;
  this->num_blocks_ = -1;
  this->block_manager_ = nullptr;
  this->nnz_local_diag_ = 0;
  this->nnz_local_offdiag_ = 0;
  this->nnz_local_ = 0;
  this->nnz_global_ = 0;
}

template < class LAD > 
BlockMatrix< LAD >::~BlockMatrix() {
  if (this->called_init_) {
    this->Clear();
  }

  int is_finalized;
  MPI_Finalized(&is_finalized);
  if (!is_finalized) {
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
      assert(this->comm_ == MPI_COMM_NULL);
    }
  }
}

template < class LAD >
Matrix< typename LAD::DataType > *BlockMatrix< LAD >::Clone() const {
  LOG_ERROR("BlockMatrix::Clone not yet implemented!!!");
  quit_program();
  return nullptr;
}

template < class LAD >
void BlockMatrix< LAD >::Init(const MPI_Comm &comm,
                              PLATFORM plat,
                              IMPLEMENTATION impl,
                              MATRIX_FORMAT format,
                              CBlockManagerSPtr block_manager) {

  // clear possibly existing DataType
  if (this->called_init_) {
    this->Clear();
  }

  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
    assert(this->comm_ == MPI_COMM_NULL);
  }

  assert(comm != MPI_COMM_NULL);

  MPI_Comm_dup(comm, &this->comm_);
  assert(this->comm_ != MPI_COMM_NULL);

  assert (block_manager != 0);
  assert (block_manager.get() != nullptr);
  this->block_manager_ = block_manager;

  // Set number of blocks
  this->num_blocks_ = this->block_manager_->num_blocks();
  assert(num_blocks_ > 0);

  //*****************************************************************
  // Initialize block vectors
  //*****************************************************************

  this->mat_.resize(this->num_blocks_);
  for (int i = 0; i < this->num_blocks_; ++i) {
    this->mat_[i].resize(this->num_blocks_);
    for (int j = 0; j < this->num_blocks_; ++j) {
      this->mat_[i][j] = new BMatrix();
      this->mat_[i][j]->Init(this->comm_,
                             *(this->block_manager_->la_c_blocks()[i]),
                             *(this->block_manager_->la_c_blocks()[j]),
                             plat, impl, format);
    }
  }
  this->called_init_ = true;
}

template < class LAD >
void BlockMatrix< LAD >::InitStructure(const int *rows_diag,
                                       const int *cols_diag, const int nnz_diag,
                                       const int *rows_offdiag,
                                       const int *cols_offdiag,
                                       const int nnz_offdiag) {
  // Make sure that Init(..) has already been called
#ifndef NDEBUG
  assert(rows_diag != nullptr);
  assert(cols_diag != nullptr);
  assert(nnz_diag > 0);
  if (nnz_offdiag > 0) {
    assert(rows_offdiag != nullptr);
    assert(cols_offdiag != nullptr);
  }
  assert(this->called_init_);
  assert(this->comm_ != MPI_COMM_NULL);
  for (int i = 0; i < this->num_blocks_; ++i) {
    assert(this->block_manager_->la_c_blocks()[i] != nullptr);
    assert(this->block_manager_->la_c_blocks()[i]->initialized());
  }

  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      assert(this->mat_[i][j] != nullptr);
    }
  }
#endif
  //*****************************************************************
  // Split diagonal part into blocks
  //*****************************************************************
  std::vector< std::vector< int > > rows_diag_block(
      this->num_blocks_ * this->num_blocks_, std::vector< int >(0, 0));
  std::vector< std::vector< int > > cols_diag_block(
      this->num_blocks_ * this->num_blocks_, std::vector< int >(0, 0));
  for (int i = 0; i < nnz_diag; ++i) {
    // Determine block of row index
    int block_num_row = -1;
    int block_index_row = -1;
    this->block_manager_->map_system2block(rows_diag[i], block_num_row,
                                           block_index_row);

    // Determine block of col index
    int block_num_col = -1;
    int block_index_col = -1;
    this->block_manager_->map_system2block(cols_diag[i], block_num_col,
                                           block_index_col);

    rows_diag_block[block_num_row * this->num_blocks_ + block_num_col]
        .push_back(block_index_row);
    cols_diag_block[block_num_row * this->num_blocks_ + block_num_col]
        .push_back(block_index_col);
  }

  // Sanity checks
#ifndef NDEBUG
  int nnz_diag_block = 0;
  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      assert(rows_diag_block[i * this->num_blocks_ + j].size() ==
             cols_diag_block[i * this->num_blocks_ + j].size());
      nnz_diag_block += rows_diag_block[i * this->num_blocks_ + j].size();
    }
  }
  assert(nnz_diag_block == nnz_diag);
#endif

  //*****************************************************************
  // Split offdiagonal part into blocks
  //*****************************************************************
  std::vector< std::vector< int > > rows_offdiag_block(
      this->num_blocks_ * this->num_blocks_, std::vector< int >(0, 0));
  std::vector< std::vector< int > > cols_offdiag_block(
      this->num_blocks_ * this->num_blocks_, std::vector< int >(0, 0));

  for (int i = 0; i < nnz_offdiag; ++i) {
    // Determine block of row index
    int block_num_row = -1;
    int block_index_row = -1;
    this->block_manager_->map_system2block(rows_offdiag[i], block_num_row,
                                           block_index_row);

    // Determine block of col index
    int block_num_col = -1;
    int block_index_col = -1;
    this->block_manager_->map_system2block(cols_offdiag[i], block_num_col,
                                           block_index_col);

    rows_offdiag_block[block_num_row * this->num_blocks_ + block_num_col]
        .push_back(block_index_row);
    cols_offdiag_block[block_num_row * this->num_blocks_ + block_num_col]
        .push_back(block_index_col);
  }

  // Sanity checks
#ifndef NDEBUG
  int nnz_offdiag_block = 0;
  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      assert(rows_offdiag_block[i * this->num_blocks_ + j].size() ==
             cols_offdiag_block[i * this->num_blocks_ + j].size());
      nnz_offdiag_block += rows_offdiag_block[i * this->num_blocks_ + j].size();
    }
  }
  assert(nnz_offdiag_block == nnz_offdiag);
#endif

  // Initialize structure of individual block matrices
  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      LOG_DEBUG(2, "Block "
                       << i << " , " << j << ": "
                       << rows_diag_block[i * this->num_blocks_ + j].size()
                       << " "
                       << cols_diag_block[i * this->num_blocks_ + j].size()
                       << " "
                       << rows_offdiag_block[i * this->num_blocks_ + j].size()
                       << " "
                       << cols_offdiag_block[i * this->num_blocks_ + j].size())

      // TODO: Initialize only those blocks, that are nonzero, which is defined
      // by couplings->space->coupling_vars
      //                  if ( rows_diag_block[i * this->num_blocks_ + j].size()
      //                  > 0 )
      //                  {
      this->mat_[i][j]->InitStructure(
          vec2ptr(rows_diag_block[i * this->num_blocks_ + j]),
          vec2ptr(cols_diag_block[i * this->num_blocks_ + j]),
          static_cast< int >(rows_diag_block[i * this->num_blocks_ + j].size()),
          vec2ptr(rows_offdiag_block[i * this->num_blocks_ + j]),
          vec2ptr(cols_offdiag_block[i * this->num_blocks_ + j]),
          static_cast< int >(
              rows_offdiag_block[i * this->num_blocks_ + j].size()));
      //                  }
    }
  }

  this->nnz_local_diag_ = 0;
  this->nnz_local_offdiag_ = 0;
  this->nnz_local_ = 0;
  this->nnz_global_ = 0;
  
  // get some nnz statistics
  for (int i = 0; i < this->num_blocks_; ++i) 
  {
    for (int j = 0; j < this->num_blocks_; ++j) 
    {
      this->nnz_local_diag_ += this->mat_[i][j]->nnz_local_diag();
      this->nnz_local_offdiag_ += this->mat_[i][j]->nnz_local_offdiag();
      this->nnz_local_ += this->mat_[i][j]->nnz_local();
      this->nnz_global_ += this->mat_[i][j]->nnz_global();
    }
  }
  

  this->initialized_structure_ = true;
}

template < class LAD > 
void BlockMatrix< LAD >::Clear() {
  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      if (this->mat_[i][j] != nullptr) {
        delete this->mat_[i][j];
      }
    }
    this->mat_[i].clear();
  }
  this->mat_.clear();

  this->num_blocks_ = -1;

  this->called_init_ = false;
  this->initialized_structure_ = false;
  
  this->nnz_local_diag_ = 0;
  this->nnz_local_offdiag_ = 0;
  this->nnz_local_ = 0;
  this->nnz_global_ = 0;
}

template < class LAD > 
int BlockMatrix< LAD >::num_rows_global() const {
  assert(this->called_init_);
  int num = 0;
  // iterate over block rows
  for (int i = 0; i < this->num_blocks_; ++i) {
    int max_num = 0;
    // iterate over block columns
    for (int j = 0; j < this->num_blocks_; ++j) {
      // a block could be undefined, i.e. having zero rows
      max_num = std::max(max_num, this->mat_[i][j]->num_rows_global());
    }
    num += max_num;
  }
  return num;
}

template < class LAD > 
int BlockMatrix< LAD >::num_cols_global() const {
  assert(this->called_init_);
  int num = 0;
  // iterate over block columns
  for (int j = 0; j < this->num_blocks_; ++j) {
    int max_num = 0;
    // iterate over block rows
    for (int i = 0; i < this->num_blocks_; ++i) {
      // a block could be undefined, i.e. having zero columns
      max_num = std::max(max_num, this->mat_[i][j]->num_cols_global());
    }
    num += max_num;
  }
  return num;
}

template < class LAD > 
int BlockMatrix< LAD >::num_rows_local() const {
  assert(this->called_init_);
  int num = 0;
  // iterate over block rows
  for (int i = 0; i < this->num_blocks_; ++i) {
    int max_num = 0;
    // iterate over block columns
    for (int j = 0; j < this->num_blocks_; ++j) {
      // a block could be undefined, i.e. having zero rows
      max_num = std::max(max_num, this->mat_[i][j]->num_rows_local());
    }
    num += max_num;
  }
  return num;
}

template < class LAD > 
int BlockMatrix< LAD >::num_cols_local() const {
  assert(this->called_init_);
  int num = 0;
  // iterate over block columns
  for (int j = 0; j < this->num_blocks_; ++j) {
    int max_num = 0;
    // iterate over block rows
    for (int i = 0; i < this->num_blocks_; ++i) {
      // a block could be undefined, i.e. having zero columns
      max_num = std::max(max_num, this->mat_[i][j]->num_cols_local());
    }
    num += max_num;
  }
  return num;
}

template < class LAD >
void BlockMatrix< LAD >::CloneFromWithoutContent(const BlockMatrix< LAD > &mat) 
{
  if (this != &mat) 
  {
    this->Clear();
    int info = 0;
    if (this->comm_ != MPI_COMM_NULL) {
      info = MPI_Comm_free(&this->comm_);
      assert(info == MPI_SUCCESS);
    }

    info = MPI_Comm_dup(mat.comm_, &(this->comm_));
    assert(info == MPI_SUCCESS);
    
    num_blocks_ = mat.num_blocks_;
    this->block_manager_ = mat.block_manager_;

    this->mat_.resize(this->num_blocks_);
    for (int i = 0; i < this->num_blocks_; ++i) 
    {
        this->mat_[i].resize(this->num_blocks_);
        for (int j = 0; j < this->num_blocks_; ++j) 
        {
            this->mat_[i][j] = new BMatrix();
            this->mat_[i][j]->CloneFromWithoutContent(mat.GetBlock(i,j));
        }
    }
    this->called_init_ = mat.called_init_;
    this->initialized_structure_ = mat.initialized_structure_;
  }
}

template < class LAD >
void BlockMatrix< LAD >::CloneFrom(const BlockMatrix< LAD > &mat) 
{
  if (this != &mat) 
  {
    this->Clear();
    int info = 0;
    if (this->comm_ != MPI_COMM_NULL) {
      info = MPI_Comm_free(&this->comm_);
      assert(info == MPI_SUCCESS);
    }

    info = MPI_Comm_dup(mat.comm_, &(this->comm_));
    assert(info == MPI_SUCCESS);
    
    num_blocks_ = mat.num_blocks_;
    this->block_manager_ = mat.block_manager_;

    this->mat_.resize(this->num_blocks_);
    for (int i = 0; i < this->num_blocks_; ++i) 
    {
        this->mat_[i].resize(this->num_blocks_);
        for (int j = 0; j < this->num_blocks_; ++j) 
        {
            this->mat_[i][j] = new BMatrix();
            this->mat_[i][j]->CloneFrom(mat.GetBlock(i,j));
        }
    }
    this->called_init_ = mat.called_init_;
    this->initialized_structure_ = mat.initialized_structure_;
  }
}

template < class LAD >
const typename LAD::MatrixType &
BlockMatrix< LAD >::GetBlock(const int row_block_number,
                             const int col_block_number) const {
  assert(this->called_init_);
  assert(row_block_number >= 0);
  assert(row_block_number < this->num_blocks_);
  assert(col_block_number >= 0);
  assert(col_block_number < this->num_blocks_);
  return *(this->mat_[row_block_number][col_block_number]);
}

template < class LAD >
typename LAD::MatrixType &
BlockMatrix< LAD >::GetBlock(const int row_block_number,
                             const int col_block_number) {
  assert(this->called_init_);
  assert(row_block_number >= 0);
  assert(row_block_number < this->num_blocks_);
  assert(col_block_number >= 0);
  assert(col_block_number < this->num_blocks_);
  return *(this->mat_[row_block_number][col_block_number]);
}

template < class LAD >
void BlockMatrix< LAD >::VectorMult(Vector< BDataType > &in,
                                    Vector< BDataType > *out) const {
  BlockVector< LAD > *hv_in, *hv_out;

  hv_in = dynamic_cast< BlockVector< LAD > * >(&in);
  hv_out = dynamic_cast< BlockVector< LAD > * >(out);

  if ((hv_in != 0) && (hv_out != 0)) {
    this->VectorMult(*hv_in, hv_out);
  } else {
    if (hv_in == 0) {
      LOG_ERROR("Called BlockMatrix::VectorMult with incompatible input vector "
                "type.");
    }
    if (hv_out == 0) {
      LOG_ERROR("Called BlockMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class LAD >
void BlockMatrix< LAD >::VectorMult(BlockVector< LAD > &in,
                                    BlockVector< LAD > *out) const {
  assert(this->IsInitialized());
  assert(out->num_blocks() == this->num_blocks_);
  assert(in.num_blocks() == this->num_blocks_);
  out->Zeros();

  // iterate over block rows
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (!out->block_is_active(i)) {
       continue;
    }
    
    // iterate over block columns
    for (int j = 0; j < this->num_blocks_; ++j) {
      if (!in.block_is_active(j)) {
        continue;
      }
    
      assert(this->mat_[i][j]->num_rows_global() ==
             out->GetBlock(i).size_global());
      assert(this->mat_[i][j]->num_cols_global() ==
             in.GetBlock(j).size_global());
      this->mat_[i][j]->VectorMultAdd(1., in.GetBlock(j), 1.,
                                      &(out->GetBlock(i)));
    }
  }
}

template < class LAD >
void BlockMatrix< LAD >::SubmatrixVectorMult(
    const std::vector< std::vector< bool > > &active_blocks,
    Vector< BDataType > &in, Vector< BDataType > *out) const {
  BlockVector< LAD > *hv_in, *hv_out;

  hv_in = dynamic_cast< BlockVector< LAD > * >(&in);
  hv_out = dynamic_cast< BlockVector< LAD > * >(out);

  if ((hv_in != 0) && (hv_out != 0)) {
    this->SubmatrixVectorMult(active_blocks, *hv_in, hv_out);
  } else {
    if (hv_in == 0) {
      LOG_ERROR("Called BlockMatrix::VectorMult with incompatible input vector "
                "type.");
    }
    if (hv_out == 0) {
      LOG_ERROR("Called BlockMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class LAD >
void BlockMatrix< LAD >::SubmatrixVectorMult(
    const std::vector< std::vector< bool > > &active_blocks,
    BlockVector< LAD > &in, BlockVector< LAD > *out) const {
  assert(this->IsInitialized());
  assert(out->num_blocks() == this->num_blocks_);
  assert(in.num_blocks() == this->num_blocks_);

  assert(active_blocks.size() == this->num_blocks_);
#ifndef NDEBUG
  for (int i = 0; i < active_blocks.size(); ++i) {
    assert(active_blocks[i].size() == this->num_blocks_);
  }
#endif
  // Determine touched blocks in output vector
  std::vector< bool > active_blocks_out(this->num_blocks_, false);
  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      if (active_blocks[i][j]) {
        active_blocks_out[i] = true;
      }
    }
  }

  for (int i = 0; i < this->num_blocks_; ++i) {
    if (active_blocks_out[i]) {
      assert (out->block_is_active(i));
      out->GetBlock(i).Zeros();
    }
  }

  // iterate over block rows
  for (int i = 0; i < this->num_blocks_; ++i) {
    // iterate over block columns
    for (int j = 0; j < this->num_blocks_; ++j) {
      if (active_blocks[i][j]) {
        assert (in.block_is_active(j));
        assert (out->block_is_active(i));
        
        this->mat_[i][j]->VectorMultAdd(1., in.GetBlock(j), 1.,
                                        &(out->GetBlock(i)));
      }
    }
  }
}

template < class LAD >
void BlockMatrix< LAD >::MatrixMult(Matrix< BDataType > &inA,
                                    Matrix< BDataType > &inB) {
  LOG_ERROR("Called BlockMatrix::MatrixMult not yet implemented!!!");
  quit_program();
}

template < class LAD >
void BlockMatrix< LAD >::VectorMultAdd(BDataType alpha, Vector< BDataType > &in,
                                       BDataType beta,
                                       Vector< BDataType > *out) const {
  BlockVector< LAD > *hv_in, *hv_out;

  hv_in = dynamic_cast< BlockVector< LAD > * >(&in);
  hv_out = dynamic_cast< BlockVector< LAD > * >(out);

  if ((hv_in != 0) && (hv_out != 0)) {
    this->VectorMultAdd(alpha, *hv_in, beta, hv_out);
  } else {
    if (hv_in == 0) {
      LOG_ERROR("Called BlockMatrix::VectorMult with incompatible input vector "
                "type.");
    }
    if (hv_out == 0) {
      LOG_ERROR("Called BlockMatrix::VectorMult with incompatible output "
                "vector type.");
    }
    quit_program();
  }
}

template < class LAD >
void BlockMatrix< LAD >::VectorMultAdd(BDataType alpha, BlockVector< LAD > &in,
                                       BDataType beta,
                                       BlockVector< LAD > *out) const {
  assert(this->IsInitialized());
  assert(out->num_blocks() == this->num_blocks_);
  assert(in.num_blocks() == this->num_blocks_);

  // Scale out vector first
  out->Scale(beta);

  // iterate over block rows
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (!out->block_is_active(i)) {
      continue;
    }
    
    // iterate over block columns
    for (int j = 0; j < this->num_blocks_; ++j) {
      if (!in.block_is_active(j)) {
        continue;
      }
      this->mat_[i][j]->VectorMultAdd(alpha, in.GetBlock(j), 1.,
                                      &(out->GetBlock(i)));
    }
  }
}

template < class LAD >
void BlockMatrix< LAD >::SubmatrixVectorMultAdd(
    const std::vector< std::vector< bool > > &active_blocks, BDataType alpha,
    Vector< BDataType > &in, BDataType beta, Vector< BDataType > *out) const {
  BlockVector< LAD > *hv_in, *hv_out;

  hv_in = dynamic_cast< BlockVector< LAD > * >(&in);
  hv_out = dynamic_cast< BlockVector< LAD > * >(out);

  if ((hv_in != 0) && (hv_out != 0)) {
    this->SubmatrixVectorMultAdd(active_blocks, alpha, *hv_in, beta, hv_out);
  } else {
    if (hv_in == 0) {
      LOG_ERROR("Called BlockMatrix::SubmatrixVectorMult with incompatible "
                "input vector type.");
    }
    if (hv_out == 0) {
      LOG_ERROR("Called BlockMatrix::SubmatrixVectorMult with incompatible "
                "output vector type.");
    }
    quit_program();
  }
}

template < class LAD >
void BlockMatrix< LAD >::SubmatrixVectorMultAdd(
    const std::vector< std::vector< bool > > &active_blocks, BDataType alpha,
    BlockVector< LAD > &in, BDataType beta, BlockVector< LAD > *out) const {
  assert(this->IsInitialized());
  assert(out->num_blocks() == this->num_blocks_);
  assert(in.num_blocks() == this->num_blocks_);

  assert(active_blocks.size() == this->num_blocks_);
#ifndef NDEBUG
  for (int i = 0; i < active_blocks.size(); ++i) {
    assert(active_blocks[i].size() == this->num_blocks_);
  }
#endif

  // Determine touched blocks in output vector
  std::vector< bool > active_blocks_out(this->num_blocks_, false);
  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      if (active_blocks[i][j]) {
        active_blocks_out[i] = true;
      }
    }
  }

  // Scale out vector first
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (active_blocks_out[i]) {
      assert (out->block_is_active(i));
      out->GetBlock(i).Scale(beta);
    }
  }

  // iterate over block rows
  for (int i = 0; i < this->num_blocks_; ++i) {   
    // iterate over block columns
    for (int j = 0; j < this->num_blocks_; ++j) {
      if (active_blocks[i][j]) {
        assert (out->block_is_active(i));
        assert (in.block_is_active(j));    
        this->mat_[i][j]->VectorMultAdd(alpha, in.GetBlock(j), 1.,
                                        &(out->GetBlock(i)));
      }
    }
  }
}

template < class LAD >
void BlockMatrix< LAD >::GetValues(const int *row_indices, const int num_rows,
                                   const int *col_indices, const int num_cols,
                                   BDataType *values) const {
  assert(this->IsInitialized());
  //*****************************************************************
  // Map indices to block numbers and block indices
  //*****************************************************************
  for (int i = 0; i < num_rows; ++i) {
    // Determine block of row index
    int block_num_row = -1;
    int block_index_row = -1;
    this->block_manager_->map_system2block(row_indices[i], block_num_row,
                                           block_index_row);

    for (int j = 0; j < num_cols; ++j) {
      // Determine block of col index
      int block_num_col = -1;
      int block_index_col = -1;
      this->block_manager_->map_system2block(col_indices[j], block_num_col,
                                             block_index_col);

      // Get value in individual block matrix
      this->mat_[block_num_row][block_num_col]->GetValues(
          &(block_index_row), 1, &(block_index_col), 1,
          &(values[i * num_cols + j]));
    }
  }
}

template < class LAD >
void BlockMatrix< LAD >::Add(const int global_row_id, const int global_col_id,
                             const BDataType value) {
  assert(this->IsInitialized());
            
  int block_num_row = -1;
  int block_index_row = -1;
            
  int block_num_col = -1;
  int block_index_col = -1;
            
  this->block_manager_->map_system2block(global_row_id, block_num_row, block_index_row);
  this->block_manager_->map_system2block(global_col_id, block_num_col, block_index_col);
            
  this->mat_[block_num_row][block_num_col]->Add(block_index_row, block_index_col, value);
}

template < class LAD >
void BlockMatrix< LAD >::Add(const int *rows, const int num_rows,
                             const int *cols, const int num_cols,
                             const BDataType *values) {
  assert(this->IsInitialized());

  //*****************************************************************
  // Map indices to block numbers and block indices
  //*****************************************************************

  std::vector< int > block_num_rows(num_rows, -1);
  std::vector< int > block_index_rows(num_rows, -1);

  std::vector< int > block_num_cols(num_cols, -1);
  std::vector< int > block_index_cols(num_cols, -1);

  for (int i = 0; i < num_rows; ++i) {
    this->block_manager_->map_system2block(rows[i], block_num_rows[i],
                                           block_index_rows[i]);
  }

  for (int i = 0; i < num_cols; ++i) {
    this->block_manager_->map_system2block(cols[i], block_num_cols[i],
                                           block_index_cols[i]);
  }

  std::vector< std::vector< int > > rows_block(this->num_blocks_);
  std::vector< std::vector< int > > cols_block(this->num_blocks_);

  std::vector< std::vector< std::vector< BDataType > > > values_block(
      this->num_blocks_);

  for (int i = 0; i < this->num_blocks_; ++i) {
    rows_block[i].reserve(num_rows);
    cols_block[i].reserve(num_cols);
    values_block[i].resize(this->num_blocks_);
    for (int j = 0; j < this->num_blocks_; ++j) {
      values_block[i][j].reserve(num_rows * num_cols);
    }
  }

  for (int i = 0; i < num_rows; ++i) {
    rows_block[block_num_rows[i]].push_back(block_index_rows[i]);
  }

  for (int i = 0; i < num_cols; ++i) {
    cols_block[block_num_cols[i]].push_back(block_index_cols[i]);
  }

  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      values_block[block_num_rows[i]][block_num_cols[j]].push_back(
          values[i * num_cols + j]);
    }
  }
  // Add values in individual block matrices
  for (int i = 0; i < this->num_blocks_; ++i) {
    const int num_rows_block = rows_block[i].size();
    if (num_rows_block > 0) {
      for (int j = 0; j < this->num_blocks_; ++j) {
        const int num_cols_block = cols_block[j].size();
        if (num_cols_block > 0) {
          this->mat_[i][j]->Add(vec2ptr(rows_block[i]), num_rows_block,
                                vec2ptr(cols_block[j]), num_cols_block,
                                vec2ptr(values_block[i][j]));
        }
      }
    }
  }
}

template < class LAD >
void BlockMatrix< LAD >::SetValue(const int row, const int col,
                                  const BDataType value) {
  this->SetValues(&row, 1, &col, 1, &value);
}

template < class LAD >
void BlockMatrix< LAD >::SetValues(const int *row_indices, const int num_rows,
                                   const int *col_indices, const int num_cols,
                                   const BDataType *values) {
  assert(this->IsInitialized());
  //*****************************************************************
  // Map indices to block numbers and block indices
  //*****************************************************************
  std::vector< int > block_num_rows(num_rows, -1);
  std::vector< int > block_index_rows(num_rows, -1);

  std::vector< int > block_num_cols(num_cols, -1);
  std::vector< int > block_index_cols(num_cols, -1);

  for (int i = 0; i < num_rows; ++i) {
    this->block_manager_->map_system2block(row_indices[i], block_num_rows[i],
                                           block_index_rows[i]);
  }

  for (int i = 0; i < num_cols; ++i) {
    this->block_manager_->map_system2block(col_indices[i], block_num_cols[i],
                                           block_index_cols[i]);
  }

  std::vector< std::vector< int > > rows_block(this->num_blocks_);
  std::vector< std::vector< int > > cols_block(this->num_blocks_);

  std::vector< std::vector< std::vector< BDataType > > > values_block(
      this->num_blocks_);

  for (int i = 0; i < this->num_blocks_; ++i) {
    rows_block[i].reserve(num_rows);
    cols_block[i].reserve(num_cols);
    values_block[i].resize(this->num_blocks_);
    for (int j = 0; j < this->num_blocks_; ++j) {
      values_block[i][j].reserve(num_rows * num_cols);
    }
  }

  for (int i = 0; i < num_rows; ++i) {
    rows_block[block_num_rows[i]].push_back(block_index_rows[i]);
  }

  for (int i = 0; i < num_cols; ++i) {
    cols_block[block_num_cols[i]].push_back(block_index_cols[i]);
  }

  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      values_block[block_num_rows[i]][block_num_cols[j]].push_back(
          values[i * num_cols + j]);
    }
  }

  // Set values in individual block matrices
  for (int i = 0; i < this->num_blocks_; ++i) {
    const int num_rows_block = rows_block[i].size();
    if (num_rows_block > 0) {
      for (int j = 0; j < this->num_blocks_; ++j) {
        const int num_cols_block = cols_block[j].size();
        if (num_cols_block > 0) {
          this->mat_[i][j]->SetValues(vec2ptr(rows_block[i]), num_rows_block,
                                      vec2ptr(cols_block[j]), num_cols_block,
                                      vec2ptr(values_block[i][j]));
        }
      }
    }
  }
}

template < class LAD > 
void BlockMatrix< LAD >::Zeros() {
  assert(this->IsInitialized());

  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      this->mat_[i][j]->Zeros();
    }
  }
}

template < class LAD >
void BlockMatrix< LAD >::diagonalize_rows(const int *row_indices,
                                          const int num_rows,
                                          const BDataType diagonal_value) {
  assert(this->IsInitialized());

  //*****************************************************************
  // Map indices to block numbers and block indices
  //*****************************************************************
  std::vector< std::vector< int > > rows_block(this->num_blocks_);
  for (int i = 0; i < num_rows; ++i) {
    // Determine block of row index
    int block_num_row = -1;
    int block_index_row = -1;
    this->block_manager_->map_system2block(row_indices[i], block_num_row,
                                           block_index_row);

    rows_block[block_num_row].push_back(block_index_row);
  }

  //*****************************************************************
  // Set values to the block matrices
  //*****************************************************************
  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      if (i != j) {
        // in non-diagonal blocks, the rows are set to zero
        this->mat_[i][j]->diagonalize_rows(
            vec2ptr(rows_block[i]), static_cast< int >(rows_block[i].size()),
            static_cast< BDataType >(0));
      } else {
        // in diagonal blocks, the diagonal element is set to given value
        this->mat_[i][j]->diagonalize_rows(
            vec2ptr(rows_block[i]), static_cast< int >(rows_block[i].size()),
            diagonal_value);
      }
    }
  }
}

template < class LAD > 
void BlockMatrix< LAD >::Scale(const BDataType alpha) {
  assert(this->IsInitialized());

  for (int i = 0; i < this->num_blocks_; ++i) {
    for (int j = 0; j < this->num_blocks_; ++j) {
      this->mat_[i][j]->Scale(alpha);
    }
  }
}

template < class LAD > 
void BlockMatrix< LAD >::Update() {
  assert(this->IsInitialized());
  for (int i = 0; i < num_blocks_; ++i) {
    for (int j = 0; j < num_blocks_; ++j) {
      this->mat_[i][j]->Update();
    }
  }
}

template < class LAD > 
void BlockMatrix< LAD >::begin_update() {
  assert(this->IsInitialized());
  for (int i = 0; i < num_blocks_; ++i) {
    for (int j = 0; j < num_blocks_; ++j) {
      this->mat_[i][j]->begin_update();
    }
  }
}

template < class LAD > 
void BlockMatrix< LAD >::end_update() {
  assert(this->IsInitialized());
  for (int i = 0; i < num_blocks_; ++i) {
    for (int j = 0; j < num_blocks_; ++j) {
      this->mat_[i][j]->end_update();
    }
  }
}

template < class LAD >
void BlockMatrix< LAD >::ExtractDiagonalCSR(int *ia, int *ja, BDataType *val) const 
{
  assert(this->IsInitialized());
  int nz_counter = 0;
  int row_counter = 1;
  ia[0] = 0;
  
  // assume major row order
  // loop over block rows
  for (int k = 0; k < num_blocks_; ++k)
  {
    // get sparse indices of all block in current row
    std::vector< std::vector< int > > ib(num_blocks_);
    std::vector< std::vector< int > > jb(num_blocks_);
    std::vector< std::vector< BDataType > > vb(num_blocks_);
    
    for (int l = 0; l < num_blocks_; ++l)
    {
      ib[l].resize(this->mat_[k][l]->num_rows_local() + 1);
      jb[l].resize(this->mat_[k][l]->nnz_local_diag());
      vb[l].resize(this->mat_[k][l]->nnz_local_diag());
  
      this->mat_[k][l]->ExtractDiagonalCSR(&(ib[l].front()), &(jb[l].front()), &(vb[l].front()));
    }
    
    // put block sparse indices in global index vectors
    // ia[0] = 0 
    // ia[m] = ia[m-1] + nnz[row m-1]
    // nnz[row m-1] = sum_l nnz[block l, row m-1]
    for (int m = 1; m < ib[0].size(); ++m)
    {
      ia[row_counter] = ia[row_counter - 1];
      for (int l = 0; l < num_blocks_; ++l)
      {
        ia[row_counter] += (ib[l][m] - ib[l][m-1]);
      }
      row_counter++;
    }
    
    // va = vb[0][ib[0][0]], ... , vb[0][ib[0][1]-1], vb[1][ib[1][0]], ... vb[1][ib[1][1]-1], ....
    //      vb[0][ib[0][1]], ... , vb[0][ib[0][2]-1], vb[1][ib[1][1]], ... vb[1][ib[1][2]-1], ....  
    
    // loop over local rows
    for (int m = 0; m < this->mat_[k][0]->num_rows_local(); ++m)
    {
      // looper over block columns
      for (int l = 0; l < num_blocks_; ++l)
      {
        // loop over nonzeros in current block in current row
        const int start_index = ib[l][m];
        const int end_index = ib[l][m+1];
        for (int q = start_index; q < end_index; ++q)
        {
          ja[nz_counter] = jb[l][q];
          val[nz_counter] = vb[l][q];
          nz_counter++;
        }
      }
    }
  }
  assert (nz_counter == this->nnz_local_diag());
  assert (row_counter == this->num_rows_local() + 1);
}


template < class LAD > void BlockMatrix< LAD >::print_statistics() const {}

// template instantiation
template class BlockMatrix< LADescriptorCoupledD >;
template class BlockMatrix< LADescriptorCoupledS >;
#ifdef WITH_HYPRE
template class BlockMatrix< LADescriptorHypreD >;
#endif
#if defined(WITH_PETSC)
template class BlockMatrix< LADescriptorPETScD >;
#endif
} // namespace la
} // namespace hiflow
