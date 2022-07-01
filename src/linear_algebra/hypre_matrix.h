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

/// @author Simon Gawlok

#ifndef HIFLOW_LINEARALGEBRA_HYPRE_MATRIX_H_
#define HIFLOW_LINEARALGEBRA_HYPRE_MATRIX_H_

#include "common/log.h"
#include "common/sorted_array.h"
#include "config.h"
#include "linear_algebra/block_utilities.h"
#include "linear_algebra/hypre_vector.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/matrix.h"
#include <cstdlib>
#include <iostream>
#include <map>
#include <mpi.h>
#include <set>
#ifdef WITH_HYPRE
extern "C" {
  #include "HYPRE.h"
  #include "HYPRE_parcsr_ls.h"
  #include "HYPRE_parcsr_mv.h"
  #include "_hypre_parcsr_mv.h"
  #include "_hypre_utilities.h"
}
#endif

namespace hiflow {
namespace la {

/// @author Simon Gawlok

/// @brief Wrapper to HYPRE matrix

template < class DataType > class HypreMatrix : public Matrix< DataType > {
public:
  /// Standard constructor
  HypreMatrix();
  /// Destructor
  ~HypreMatrix();

  virtual Matrix< DataType > *Clone() const;

  void CloneFromWithoutContent(const HypreMatrix< DataType > &mat);

  void CloneFrom(const HypreMatrix< DataType > &mat);

  /// Initialize matrix
  /// @param[in] comm MPI communicator to be used by matrix
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp)
  {
    this->Init(comm, cp, cp);  
  }

  /// Initialize matrix
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp_row,
            const LaCouplings &cp_col);

  /// Additional init interface which is compatible to both block and non-block
  /// matrix
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, cp, cp);
  }
          
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            PLATFORM plat, 
            IMPLEMENTATION impl,
            MATRIX_FORMAT format,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, cp, cp);  
  }

  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp_row,
            const LaCouplings &cp_col,
            PLATFORM plat, 
            IMPLEMENTATION impl,
            MATRIX_FORMAT format)
  {
    this->Init(comm, cp_row, cp_col);  
  }            
                        
#ifdef WITH_HYPRE
  /// Initialize matrix from Hypre ParCSR object
  void Init(const MPI_Comm &comm, HYPRE_ParCSRMatrix *parcsr_temp);
#endif

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

  void VectorMult(HypreVector< DataType > &in,
                  HypreVector< DataType > *out) const;

  /// out = this * in
  void MatrixMult(Matrix< DataType > &inA, Matrix< DataType > &inB);

  void MatrixMult(HypreMatrix< DataType > &inA, HypreMatrix< DataType > &inB);

  /// out = beta * out + alpha * this * in
  void VectorMultAdd(DataType alpha, Vector< DataType > &in, DataType beta,
                     Vector< DataType > *out) const;

  /// out = beta * out + alpha * this * in
  void VectorMultAdd(DataType alpha, HypreVector< DataType > &in, DataType beta,
                     HypreVector< DataType > *out) const;

  /// Get values at specified indices
  void GetValues(const int *row_indices, int num_rows, const int *col_indices,
                 int num_cols, DataType *values) const;

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

  /// Set Matrix CSR data
  void SetCSR(const int* ia, const int* ja, const DataType* val);

  /// Sets rows to zero except the diagonal element to alpha.
  /// @param row_indices Global row indices (must be owned by this process)
  /// @param num_rows Size of array @em row_indices
  /// @param diagonal_value Value to be set for diagonal element
  void diagonalize_rows(const int *row_indices, int num_rows,
                        DataType diagonal_value);

  /// Scale Matrix: this = alpha * this
  /// @param alpha Scaling factor
  void Scale(const DataType alpha);

  void ScaleRows(const int *row_indices, int num_rows, const DataType *alphas);

  /// Extracts CSR structure of matrix (only CPU matrices with CSR format)
  /// @param ia Indices of local row pointers (needs to be allocated)
  /// @param ja Global column indices (needs to be allocated)
  /// @param val Values (needs to be allocated)
  void ExtractCSR(int *ia, int *ja, DataType *val) const;

  void ExtractDiagValues(std::vector< DataType > &vals) const;

  void ExtractInvDiagValues(const DataType eps, const DataType default_val,
                            std::vector< DataType > &vals) const;

  /// Add multiple of another HypreMatrix. Only entries in the sparsity
  /// structure of this matrix are taken into account
  void Axpy(const HypreMatrix< DataType > &mat, const DataType alpha);

#ifdef WITH_HYPRE
  /// Get pointer to HXPRE_ParCSRMatrix objects
  HYPRE_ParCSRMatrix *GetParCSRMatrix();

  /// Get pointer to HXPRE_ParCSRMatrix objects
  const HYPRE_ParCSRMatrix *GetParCSRMatrix() const;
#endif

  /// Extracts CSR structure of matrix (only CPU matrices with CSR format)
  /// @param ia Indices of local row pointers (needs to be allocated)
  /// @param ja Global column indices (needs to be allocated)
  /// @param val Values (needs to be allocated)
  void ExtractDiagonalCSR(int *ia, int *ja, DataType *val) const;

  void GetDiagonalCSR(int *& ia, int *& ja, DataType *& val);

  /// Creates the transpose of a square matrix.
  /// @param other Matrix to create the transpose from.
  void CreateTransposedFrom(const HypreMatrix< DataType > &other);

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
    return static_cast< int >(this->iupper_ - this->ilower_ + 1);
  }

  /// Local number of columns
  int num_cols_local() const {
    return static_cast< int >(this->jupper_ - this->jlower_ + 1);
  }

  /// Get MPI communicator
  const MPI_Comm &comm() const { return comm_; }

  /// Get MPI communicator
  MPI_Comm &comm() { return comm_; }

  /// Get LaCouplings for rows
  const LaCouplings &row_couplings() const { return *cp_row_; }

  /// Get LaCouplings for columns
  const LaCouplings &col_couplings() const { return *cp_col_; }

  const std::vector< HYPRE_Int > &RowLengthDiag() const {
    return this->row_length_diag_;
  }

  const std::vector< HYPRE_Int > &RowLengthOffDiag() const {
    return this->row_length_offdiag_;
  }

  const std::vector< SortedArray< HYPRE_Int > > &Structure() const {
    return this->structure_;
  }

  const std::vector< SortedArray< HYPRE_Int > > &Structure_diag() const {
    return this->structure_diag_;
  }

  const std::vector< SortedArray< HYPRE_Int > > &Structure_offdiag() const {
    return this->structure_offdiag_;
  }

  size_t nnz_local() const { return this->nnz_local_; }

  /// @return Size of communicator
  int comm_size() const { return this->comm_size_; }

  /// @return Rank of this process
  int my_rank() const { return this->my_rank_; }

  /// @return Global index of the first local row

  int row_ownership_begin() const { return static_cast< int >(this->ilower_); }

  /// @return One more than the global index of the last local row

  int row_ownership_end() const {
    return static_cast< int >(this->iupper_ + 1);
  }

  /// @return Global index of the first local column

  int col_ownership_begin() const { return static_cast< int >(this->jlower_); }

  /// @return One more than the global index of the last local column

  int col_ownership_end() const {
    return static_cast< int >(this->jupper_ + 1);
  }

  /// number of rows in diagonal part

  int nrows_local() const {
#ifdef WITH_HYPRE
    return hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(this->parcsr_A_));
#endif
  }

  /// number of nonzeros in diagonal part

  int nnz_local_diag() const {
    //hypre_CSRMatrix const *diag = hypre_ParCSRMatrixDiag(*this->GetParCSRMatrix());

    //auto nrows = hypre_CSRMatrixNumRows(diag);
    
    /*
    std::cout << "hypre " << nrows 
              << " " << static_cast< int >(hypre_CSRMatrixI(diag)[nrows])
              << " " << hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(this->parcsr_A_)) << std::endl;

    */
    NOT_YET_IMPLEMENTED;  // there is something wrong
    return hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(this->parcsr_A_));
  }

  /// number of nonzeros in off-diagonal part
  int nnz_local_offdiag() const {
    return (this->nnz_local() - this->nnz_local_diag());
  }
  
  /// number of nonzeros in global matrix
  int nnz_global() const;
  
  virtual bool IsInitialized() const {
    if (this->called_init_) {
      if (this->initialized_structure_) {
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
  /// Rank of current process
  int my_rank_;
  /// Global number of processes
  int comm_size_;
  /// Linear algebra couplings describing global row dof distribution
  const LaCouplings *cp_row_;
  /// Linear algebra couplings describing global column dof distribution
  const LaCouplings *cp_col_;

  /// Global number of first row owned by this process
  HYPRE_Int ilower_;
  /// Global number of last row owned by this process
  HYPRE_Int iupper_;
  /// Global number of first column owned by this process
  HYPRE_Int jlower_;
  /// Global number of last column owned by this process
  HYPRE_Int jupper_;

  /// Number of rows owned by this process
  size_t size_local_;
  /// Number of non-zero entries on this process
  size_t nnz_local_;

  /// Flag if vector is already initialized
  bool initialized_;
  bool initialized_structure_;
  bool called_init_;

  /// Structure of the matrix
  std::vector< SortedArray< HYPRE_Int > > structure_;
  std::vector< SortedArray< HYPRE_Int > > structure_diag_;
  std::vector< SortedArray< HYPRE_Int > > structure_offdiag_;
  std::vector< HYPRE_Int > row_length_diag_;
  std::vector< HYPRE_Int > row_length_offdiag_;

  HYPRE_IJMatrix A_;
  HYPRE_ParCSRMatrix parcsr_A_;

  mutable std::vector<int> perm_tmp1_;
  mutable std::vector<int> perm_tmp2_;
  mutable std::vector<int> perm_row_;
  mutable std::vector<int> perm_col_;
  mutable std::vector<int> row_ind_reduced_;
  mutable std::vector<int> col_ind_reduced_;
  mutable std::vector<DataType> tmp_vals_;

};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARALGEBRA_HYPRE_MATRIX_H_
