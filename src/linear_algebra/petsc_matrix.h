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

/// @author Bernd Doser, HITS gGmbH
/// @date 2015-11-26

#ifndef HIFLOW_LINEARALGEBRA_PETSC_MATRIX_H_
#define HIFLOW_LINEARALGEBRA_PETSC_MATRIX_H_

#include <numeric>

#include "common/log.h"
#include "common/sorted_array.h"
#include "config.h"
#include "linear_algebra/block_utilities.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/matrix.h"
#include "linear_algebra/petsc_vector.h"
#include "mpi.h"

// TODO: Fix order dependency of next inclusion
#include "common/smart_pointers.h"

namespace hiflow {
namespace la {

/// Forwarding PETSc matrix wrapper class
namespace petsc {
struct Mat_wrapper;
}

/// Forwarding PETSc linear solver
template < class LAD >
class PETScCG;

/// @brief Wrapper class to PETSc matrix

template < class DataType >
class PETScMatrix : public Matrix< DataType > {
public:
  /// Default constructor
  PETScMatrix();

  /// Destructor
  virtual ~PETScMatrix();

  virtual Matrix< DataType > *Clone() const;

  void CloneFromWithoutContent(const PETScMatrix< DataType > &mat);

  void CloneFrom(const PETScMatrix< DataType > &mat);

  /// Initialize matrix
  /// @param[in] comm MPI communicator to be used by matrix
  void Init(const MPI_Comm &comm, const LaCouplings &cp);

  /// Initialize matrix
  void Init(const MPI_Comm &comm, const LaCouplings &cp_row,
            const LaCouplings &cp_col);

  /// Additional init interface which is compatible to both block and non-block
  /// matrix
  void Init(const MPI_Comm &comm, const LaCouplings &cp,
            const BlockManager &block_manager);

  void Init(const MPI_Comm &comm, 
			      const LaCouplings &row_cp,
            const LaCouplings &col_cp, 
            PLATFORM plat, 
            IMPLEMENTATION impl,
            MATRIX_FORMAT format)
  {
    this->Init(comm, row_cp, col_cp);
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

  /// Print statistical data
  void print_statistics() const;

  /// out = this * in
  void VectorMult(Vector< DataType > &in, Vector< DataType > *out) const;

  void VectorMult(PETScVector< DataType > &in,
                  PETScVector< DataType > *out) const;

  /// out = beta * out + alpha * this * in

  void VectorMultAdd(DataType alpha, Vector< DataType > &in, DataType beta,
                     Vector< DataType > *out) const;

  void VectorMultAdd(DataType alpha, PETScVector< DataType > &in, DataType beta,
                     PETScVector< DataType > *out) const;

  virtual void MatrixMult(Matrix< DataType > &inA, Matrix< DataType > &inB) {
    LOG_INFO("PetscMatrix::MatrixMult", "not implemented");
  };

  /// Get value at specified indices
  DataType GetValue(int row, int col) const;
  /// Get values at specified indices
  void GetValues(const int *row_indices, int num_rows, const int *col_indices,
                 int num_cols, DataType *values) const;

  /// Euclidean length of vector
  DataType NormFrobenius() const;

  // Mutating functions: after calling any of these, a call to
  // begin_update()/end_update() or update() must be made before
  // any other function can be called. It is, however, possible
  // to call the same mutating function several times in a row,
  // without calling update() in between.

  /// Add value to given indices
  void Add(int global_row_id, int global_col_id, DataType value);

  /// \brief Add submatrix of values at positions (rows x cols).
  /// The row and column numbers are assumed to correspond to global dof ids.
  /// Size of values is assumed to be |rows| x |cols|.
  void Add(const int *rows, int num_rows, const int *cols, int num_cols,
           const DataType *values);

  void Add(const std::vector<int>& row_ind, 
           const std::vector<int>& col_ind,
           const std::vector<DataType>& values);
           
  /// Set value at given indices
  void SetValue(int row, int col, DataType value);
  /// Set submatrix of values
  void SetValues(const int *row_indices, int num_rows, const int *col_indices,
                 int num_cols, const DataType *values);
  /// Set Matrix to zero
  void Zeros();

  /// Sets rows to zero except the diagonal element to alpha.
  /// @param row_indices Global row indices (must be owned by this process)
  /// @param num_rows Size of array @em row_indices
  /// @param diagonal_value Value to be set for diagonal element
  void diagonalize_rows(const int *row_indices, int num_rows,
                        DataType diagonal_value);

  /// Scale Matrix: this = alpha * this
  /// @param alpha Scaling factor
  void Scale(const DataType alpha);

  /// Extracts CSR structure of matrix (only CPU matrices with CSR format)
  /// @param ia Indices of local row pointers (needs to be allocated)
  /// @param ja Global column indices (needs to be allocated)
  /// @param val Values (needs to be allocated)
  void ExtractDiagonalCSR(int *ia, int *ja, DataType *val) const;

  /// Initiate update

  void begin_update() {}
  /// Finalize update

  void end_update() {}

  // Update matrix entries

  void Update() {}

  /// Global number of rows

  int num_rows_global() const {
    int nrows_local = this->num_rows_local();
    int nrows_global = 0;
    MPI_Allreduce(&nrows_local, &nrows_global, 1, MPI_INT, MPI_SUM,
                  this->comm_);
    return nrows_global;
  }
  /// Global number of columns

  int num_cols_global() const {
    int ncols_local = this->num_cols_local();
    int ncols_global = 0;
    MPI_Allreduce(&ncols_local, &ncols_global, 1, MPI_INT, MPI_SUM,
                  this->comm_);
    return ncols_global;
  }
  /// Local number of rows

  int num_rows_local() const {
    return this->size_local_;
  }
  /// Local number of columns

  int num_cols_local() const {
    return this->size_local_;
  }

  /// Get MPI communicator

  const MPI_Comm &comm() const {
    return comm_;
  }

  /// Get MPI communicator

  MPI_Comm &comm() {
    return comm_;
  }

  /// Get LaCouplings for rows

  const LaCouplings &row_couplings() const {
    return *cp_row_;
  }

  /// Get LaCouplings for columns

  const LaCouplings &col_couplings() const {
    return *cp_col_;
  }

  const std::vector< int > &RowLengthDiag() const {
    return this->row_length_diag_;
  }

  const std::vector< int > &RowLengthOffDiag() const {
    return this->row_length_offdiag_;
  }

  const std::vector< SortedArray< int > > &Structure() const {
    return this->structure_;
  }

  size_t nnz_local() const {
    return this->nnz_local_;
  }

  /// @return Global index of the first local row

  int row_ownership_begin() const {
    return static_cast< int >(this->ilower_);
  }

  /// @return One more than the global index of the last local row

  int row_ownership_end() const {
    return static_cast< int >(this->iupper_ + 1);
  }

  /// @return Global index of the first local column

  int col_ownership_begin() const {
    return static_cast< int >(this->jlower_);
  }

  /// @return One more than the global index of the last local column

  int col_ownership_end() const {
    return static_cast< int >(this->jupper_ + 1);
  }

  /// number of rows in diagonal part

  int nrows_local() const {
    return this->size_local_;
  }

  /// number of nonzeros in diagonal part

  int nnz_local_diag() const {
    return std::accumulate(this->row_length_diag_.begin(),
                           this->row_length_diag_.end(), 0);
  }

  /// number of nonzeros in off-diagonal part

  int nnz_local_offdiag() const {
    return std::accumulate(this->row_length_offdiag_.begin(),
                           this->row_length_offdiag_.end(), 0);
  }

  /// number of nonzeros in global matrix
  int nnz_global() const;
  

private:
  /// Friends
  template < class LAD > friend class PETScCG;
  template < class LAD > friend class PETScGeneralKSP;

  /// Final assemble of cached values. Must be called before using the matrix.
  void Assembly() const;

  /// MPI communicator
  MPI_Comm comm_;
  /// Rank of current process
  int my_rank_;
  /// Global number of processes
  int nb_procs_;
  /// Linear algebra couplings describing global row dof distribution
  const LaCouplings *cp_row_;
  /// Linear algebra couplings describing global column dof distribution
  const LaCouplings *cp_col_;

  /// Global number of first row owned by this process
  int ilower_;
  /// Global number of last row owned by this process
  int iupper_;
  /// Global number of first column owned by this process
  int jlower_;
  /// Global number of last column owned by this process
  int jupper_;
  /// Number of rows owned by this process
  int size_local_;
  /// Number of non-zero entries on this process
  int nnz_local_;

  /// Flag if matrix is initialized
  bool initialized_;

  /// Flag if matrix is assembled
  mutable bool assembled_;

  /// Structure of the matrix
  std::vector< SortedArray< int > > structure_;
  std::vector< int > row_length_diag_;
  std::vector< int > row_length_offdiag_;

  /// Pointer to PETSc matrix object
  hiflow::scoped_ptr< petsc::Mat_wrapper > ptr_mat_wrapper_;
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARALGEBRA_PETSC_MATRIX_H_
