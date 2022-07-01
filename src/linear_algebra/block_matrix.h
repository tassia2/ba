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

#ifndef HIFLOW_LINEARALGEBRA_BLOCK_MATRIX_H_
#define HIFLOW_LINEARALGEBRA_BLOCK_MATRIX_H_

#include "config.h"
#include "linear_algebra/block_utilities.h"
#include "linear_algebra/block_vector.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/matrix.h"
#include "linear_algebra/vector.h"
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <vector>

namespace hiflow {
namespace la {
/// @author Jonas Kratzke, Simon Gawlok

/// @brief Block matrix

template < class LAD >
class BlockMatrix : public Matrix< typename LAD::DataType > {
public:
  typedef typename LAD::MatrixType BMatrix;
  typedef typename LAD::VectorType BVector;
  typedef typename LAD::DataType BDataType;

  /// Standard constructor
  BlockMatrix();
  /// Destructor
  ~BlockMatrix();

  virtual Matrix< BDataType > *Clone() const;

  // TODO: Inits have to be well defined

  /// Initialize matrix blocks
  void Init(const MPI_Comm &comm,
            PLATFORM plat,
            IMPLEMENTATION impl,
            MATRIX_FORMAT format,
            CBlockManagerSPtr block_manager);
                              
  /// Additional init interface which is compatible to both block and non-block
  /// matrix
  void Init(const MPI_Comm &comm,
            const LaCouplings &cp,
            PLATFORM plat,
            IMPLEMENTATION impl,
            MATRIX_FORMAT format,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, CPU, NAIVE, CSR, block_manager);
  }
            
  void Init(const MPI_Comm &comm, 
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, CPU, NAIVE, CSR, block_manager);
  }
  
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, CPU, NAIVE, CSR, block_manager);
  }

  /// Initializes the structure of the matrix with pairs of indices.
  /// @param rows_diag Global row indices of diagonal block
  /// @param cols_diag Global column indices of diagonal block
  /// @param nnz_diag Size of @em rows_diag and @em cols_diag arrays
  /// @param rows_offdiag Global row indices of offdiagonal block
  /// @param cols_offdiag Global column indices of offdiagonal block
  /// @param nnz_offdiag Size of @em rows_offdiag and @em cols_offdiag arrays
  void InitStructure(const int *rows_diag, const int *cols_diag,
                     const int nnz_diag, const int *rows_offdiag,
                     const int *cols_offdiag, const int nnz_offdiag);

  void InitStructure(const SparsityStructure& sparsity)
  {
    this->InitStructure( vec2ptr(sparsity.diagonal_rows), 
                         vec2ptr(sparsity.diagonal_cols),
                         sparsity.diagonal_rows.size(), 
                         vec2ptr(sparsity.off_diagonal_rows),
                         vec2ptr(sparsity.off_diagonal_cols), 
                         sparsity.off_diagonal_rows.size() );
  }
  
  /// Clear all allocated data
  void Clear();

  /// Clones the whole matrix (everything).
  /// @param mat Matrix to copy
  void CloneFromWithoutContent(const BlockMatrix< LAD > &mat);
  void CloneFrom(const BlockMatrix< LAD > &mat);
  
  /// @return Global number of rows
  int num_rows_global() const;
  /// @return Global number of columns
  int num_cols_global() const;
  /// @return Local number of rows
  int num_rows_local() const;
  /// @return Local number of columns
  int num_cols_local() const;

  /// @return Local number of nonzero elements of diagonal block
  inline int nnz_local_diag() const
  {
    return this->nnz_local_diag_;
  }

  /// @return Local number of nonzero elements of offdiagonal block
  inline int nnz_local_offdiag() const
  {
    return this->nnz_local_offdiag_;
  }
  
  /// @return Local number of nonzero elements
  inline int nnz_local() const
  {
    return this->nnz_local_;
  }
  
  /// @return Global number of nonzero elements
  inline int nnz_global() const
  {
    return this->nnz_global_;
  }
  
  int num_blocks() const {
    assert(this->called_init_);
    return this->num_blocks_;
  }

  /// out = this * in
  void VectorMult(Vector< BDataType > &in, Vector< BDataType > *out) const;
  void VectorMult(BlockVector< LAD > &in, BlockVector< LAD > *out) const;

  /// out = submatrix_of_this * in
  void
  SubmatrixVectorMult(const std::vector< std::vector< bool > > &active_blocks,
                      Vector< BDataType > &in, Vector< BDataType > *out) const;
  void
  SubmatrixVectorMult(const std::vector< std::vector< bool > > &active_blocks,
                      BlockVector< LAD > &in, BlockVector< LAD > *out) const;

  /// this = inA * inB
  void MatrixMult(Matrix< BDataType > &inA, Matrix< BDataType > &inB);

  /// out = beta * out + alpha * this * in
  void VectorMultAdd(BDataType alpha, Vector< BDataType > &in, BDataType beta,
                     Vector< BDataType > *out) const;
  void VectorMultAdd(BDataType alpha, BlockVector< LAD > &in, BDataType beta,
                     BlockVector< LAD > *out) const;

  /// out = beta * out + alpha * submatrix_of_this * in
  void SubmatrixVectorMultAdd(
      const std::vector< std::vector< bool > > &active_blocks, BDataType alpha,
      Vector< BDataType > &in, BDataType beta, Vector< BDataType > *out) const;
  void SubmatrixVectorMultAdd(
      const std::vector< std::vector< bool > > &active_blocks, BDataType alpha,
      BlockVector< LAD > &in, BDataType beta, BlockVector< LAD > *out) const;

  /// Get values at specified indices
  void GetValues(const int *row_indices, const int num_rows,
                 const int *col_indices, const int num_cols,
                 BDataType *values) const;

  /// Non-Const access to blocks
  BMatrix &GetBlock(const int row_block_number, const int col_block_number);

  /// Non-Const access to blocks
  const BMatrix &GetBlock(const int row_block_number,
                          const int col_block_number) const;

  // Mutating functions: after calling any of these, a call to
  // begin_update()/end_update() or update() must be made before
  // any other function can be called. It is, however, possible
  // to call the same mutating function several times in a row,
  // without calling update() in between.

  /// Add value to given indices
  void Add(const int global_row_id, const int global_col_id,
           const BDataType value);

  /// \brief Add submatrix of values at positions (rows x cols).
  /// The row and column numbers are assumed to correspond to global dof ids.
  /// Size of values is assumed to be |rows| x |cols|.
  void Add(const int *rows, const int num_rows, const int *cols,
           const int num_cols, const BDataType *values);
  /// Set value at given indices
  void SetValue(const int row, const int col, const BDataType value);
  /// Set submatrix of values
  void SetValues(const int *row_indices, const int num_rows,
                 const int *col_indices, const int num_cols,
                 const BDataType *values);
  /// Set Matrix to zero
  void Zeros();

  /// Sets rows to zero except the diagonal element to alpha.
  /// @param row_indices Global row indices (must be owned by this process)
  /// @param num_rows Size of array @em row_indices
  /// @param diagonal_value Value to be set for diagonal element
  void diagonalize_rows(const int *row_indices, const int num_rows,
                        const BDataType diagonal_value);

  /// Scale Matrix: this = alpha * this
  /// @param alpha Scaling factor
  void Scale(const BDataType alpha);

  void ExtractDiagonalCSR(int *ia, int *ja, BDataType *val) const;

  // Update matrix entries
  void Update();

  /// Initiate update
  void begin_update();

  /// Finalize update
  void end_update();

  /// Print statistical data
  void print_statistics() const;

  virtual bool IsInitialized() const {
    if (this->called_init_) {
      if (this->initialized_structure_) {
        for (int l = 0; l < this->num_blocks_; ++l) {
          for (int k = 0; k < this->num_blocks_; ++k) {
            if (!this->mat_[l][k]->IsInitialized()) {
              LOG_DEBUG(0,
                        "Submatrix " << l << " , " << k << " not initialized");
              return false;
            }
          }
        }
        return true;
      } else {
        LOG_DEBUG(0, "InitializeStructure not called");
      }
    } else {
      LOG_DEBUG(0, "Init not called");
    }
    return false;
  }

private:
  /// MPI communicator
  MPI_Comm comm_;

  int num_blocks_;
  int nnz_local_diag_;
  int nnz_local_offdiag_;
  int nnz_local_;
  int nnz_global_;
  
  bool called_init_;
  bool initialized_structure_;

  CBlockManagerSPtr block_manager_;

  // Matrices of individual blocks
  std::vector< std::vector< BMatrix * > > mat_;
};

template < class LAD > 
class LADescriptorBlock {
public:
  typedef BlockMatrix< LAD > MatrixType;
  typedef BlockVector< LAD > VectorType;
  typedef typename LAD::DataType DataType;
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARALGEBRA_BLOCK_MATRIX_H_
