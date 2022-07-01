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

#ifndef __LMATRIX_CSR_CPU_H
#define __LMATRIX_CSR_CPU_H

#include <iostream>
#include <stdlib.h>

#include "lmatrix_csr.h"
#include "lmatrix_formats.h"
#include "lvector_cpu.h"

/// @brief Provides the base matrix (CSR) class for CPU
/// @author Dimitar Lukarski
///
/// CPU_CSR_lMatrix maintains the matrix;
/// provides access functions, copy operators,
/// transformation from different cpu matrix types (COO),
/// basic implementation of matrix-free-preconditioner,
/// sequential implementation of the VectorMultAdd fct

template < typename ValueType >
class CPU_CSR_lMatrix : public CSR_lMatrix< ValueType > {
public:
  /// Similar to standard library
  typedef ValueType value_type;

  CPU_CSR_lMatrix();
  virtual ~CPU_CSR_lMatrix();

  // in side:
  //  ValueType *val; // value
  //  int *col; // col index
  //  int *row; // row pointer

  virtual void get_as_coo(std::vector< ValueType > &vals,
                          std::vector< int > &rows,
                          std::vector< int > &cols) const;

  // copy and converting
  virtual hiflow::la::lMatrix< ValueType > &
  operator=(const hiflow::la::lMatrix< ValueType > &mat2);
  virtual void CopyFrom(const hiflow::la::lMatrix< ValueType > &mat2);
  virtual void CopyTo(hiflow::la::lMatrix< ValueType > &mat2) const;
  virtual void CopyStructureFrom(const hiflow::la::lMatrix< ValueType > &mat2);
  virtual void CopyStructureTo(hiflow::la::lMatrix< ValueType > &mat2) const;
  virtual void ConvertFrom(const hiflow::la::lMatrix< ValueType > &mat2);

  virtual void CastFrom(const hiflow::la::lMatrix< double > &other);
  virtual void CastFrom(const hiflow::la::lMatrix< float > &other);

  virtual void CastFrom(const CPU_CSR_lMatrix< double > &other) = 0;
  virtual void CastFrom(const CPU_CSR_lMatrix< float > &other) = 0;

  virtual void CastTo(hiflow::la::lMatrix< double > &other) const;
  virtual void CastTo(hiflow::la::lMatrix< float > &other) const;

  virtual void CastTo(CPU_CSR_lMatrix< double > &other) const = 0;
  virtual void CastTo(CPU_CSR_lMatrix< float > &other) const = 0;

  virtual void SwapDiagElementsToRowFront(void);

  virtual void VectorMultNoDiag(const hiflow::la::lVector< ValueType > &in,
                                hiflow::la::lVector< ValueType > *out) const;
  virtual void VectorMultNoDiag(const CPU_lVector< ValueType > &in,
                                CPU_lVector< ValueType > *out) const;

  void Init(const int init_nnz, const int init_num_row,
            const int init_num_col, const std::string init_name);
  
  virtual void Clear(void);

  virtual void ZeroRows(const int *index_set, const int size,
                        const ValueType alpha);

  virtual void ZeroCols(const int *index_set, const int size,
                        const ValueType alpha);

  virtual void Zeros(void);

  /// Compress matrix by removing zero elements
  virtual void Compress(void);

  virtual void Reorder(const int *index);

  virtual void Multicoloring(int &ncolors, int **color_sizes,
                             int **permut_index) const;

  virtual void Levelscheduling(int &nlevels, int **level_sizes,
                               int **permut_index) const;

  virtual void ilu0(void);

  virtual void ilup(const int p);
  virtual void ilup(const hiflow::la::lMatrix< ValueType > &mat, const int p);

  virtual void ilu_solve(const hiflow::la::lVector< ValueType > &invec,
                         hiflow::la::lVector< ValueType > *outvec) const;

  virtual void Scale(const ValueType alpha);

  virtual void ScaleOffdiag(const ValueType alpha);

  virtual void init_structure(const int *rows, const int *cols);
  virtual void add_value(const int row, const int col, const ValueType val);
  virtual void get_value(const int row, const int col, ValueType *val) const;

  virtual int get_row_nnz (const int row) const;
  virtual void get_row_values(const int row, ValueType *val) const;
  virtual int get_actual_nnz () const;
  
  virtual void add_values(const int *rows, int num_rows, const int *cols,
                          int num_cols, const ValueType *values);

  virtual void add_values(int num_entries, 
                          const int *rows, const int *cols,
                          const ValueType *vals);
                          
  virtual void set_values(const int *rows, int num_rows, const int *cols,
                          int num_cols, const ValueType *values);

  virtual void get_add_values(const int *rows, int num_rows, const int *cols,
                              int num_cols, const int *cols_target,
                              int num_cols_target, ValueType *values) const;

  virtual void get_add_row_sums (ValueType *values) const;
  
  virtual void VectorMultAdd_submatrix(const int *rows, int num_rows,
                                       const int *cols, int num_cols,
                                       const int *cols_input,
                                       const ValueType *in_values,
                                       ValueType *out_values) const;

  virtual void
  VectorMultAdd_submatrix_vanka(const int *rows, int num_rows,
                                const CPU_lVector< ValueType > &invec,
                                ValueType *out_values) const;

  virtual void
  VectorMultAdd_submatrix_vanka(const int *rows, int num_rows,
                                const hiflow::la::lVector< ValueType > &invec,
                                ValueType *out_values) const;

  virtual void transpose_me(void);
  virtual void compress_me(void);
  virtual bool issymmetric(void);

  virtual void delete_diagonal(void);
  virtual void delete_offdiagonal(void);
  virtual void delete_lower_triangular(void);
  virtual void delete_strictly_lower_triangular(void);
  virtual void delete_upper_triangular(void);
  virtual void delete_strictly_upper_triangular(void);

  virtual hiflow::la::lMatrix< ValueType > *
  extract_submatrix(const int start_row, const int start_col, const int end_row,
                    const int end_col) const;

  virtual void
  extract_diagelements(const int start_i, const int end_i,
                       hiflow::la::lVector< ValueType > *vec) const;

  virtual void
  extract_diagelements(const int start_i, const int end_i,
                       ValueType* vals) const;
                       
  virtual void
  extract_invdiagelements(const int start_i, const int end_i,
                          hiflow::la::lVector< ValueType > *vec) const;

  virtual void 
  set_diagelements(const ValueType* vals);

  virtual void 
  get_diagelements(ValueType* vals) const;
  
  virtual void VectorMultAdd(const hiflow::la::lVector< ValueType > &invec,
                             hiflow::la::lVector< ValueType > *outvec) const;
  virtual void VectorMultAdd(const CPU_lVector< ValueType > &invec,
                             CPU_lVector< ValueType > *outvec) const;

  virtual hiflow::la::lMatrix< ValueType > *
  MatrixMult(const hiflow::la::lMatrix< ValueType > &inmat) const;
  virtual hiflow::la::lMatrix< ValueType > *
  MatrixMult(const CPU_CSR_lMatrix< ValueType > &inmat) const;

  virtual hiflow::la::lMatrix< ValueType > *
  MatrixMultSupStructure(const hiflow::la::lMatrix< ValueType > &inmat) const;
  virtual hiflow::la::lMatrix< ValueType > *
  MatrixMultSupStructure(const CPU_CSR_lMatrix< ValueType > &inmat) const;

  virtual hiflow::la::lMatrix< ValueType > *MatrixSupSPower(const int p) const;

  virtual void MatrixAdd(const hiflow::la::lMatrix< ValueType > &inmat);
  virtual void MatrixAdd(const CPU_CSR_lMatrix< ValueType > &inmat);

  virtual void GershgorinSpectrum(ValueType *lambda_min,
                                  ValueType *lambda_max) const;

  virtual void ReadFile(const char *filename);
  virtual void WriteFile(const char *filename) const;

  virtual void Pjacobi(const hiflow::la::lVector< ValueType > &invec,
                       hiflow::la::lVector< ValueType > *outvec) const;
  virtual void Pgauss_seidel(const hiflow::la::lVector< ValueType > &invec,
                             hiflow::la::lVector< ValueType > *outvec) const;
  virtual void Psgauss_seidel(const hiflow::la::lVector< ValueType > &invec,
                              hiflow::la::lVector< ValueType > *outvec) const;
  virtual void
  BlockPsgauss_seidel(const hiflow::la::lVector< ValueType > &invec,
                      hiflow::la::lVector< ValueType > *outvec,
                      const int start_i, const int end_i) const;

  virtual void
  BlocksPsgauss_seidel(const hiflow::la::lVector< ValueType > &invec,
                       hiflow::la::lVector< ValueType > *outvec,
                       const int num_blocks) const;

  virtual void Psor(const ValueType omega,
                    const hiflow::la::lVector< ValueType > &invec,
                    hiflow::la::lVector< ValueType > *outvec) const;
  virtual void Pssor(const ValueType omega,
                     const hiflow::la::lVector< ValueType > &invec,
                     hiflow::la::lVector< ValueType > *outvec) const;

  /// Transform a local COO matrix to a local CSR matrix;
  /// the local matrix should be initialized;
  /// The elements afterward are sorted
  /// @param rows - the row index set
  /// @param cols - the column index set
  /// @param data - the values of the coo matrix
  /// @param num_rows - the number of rows in the coo matrix
  /// @param num_cols - the number of cols in the coo matrix
  /// @param num_nnz - the number of nnz in the coo matrix
  virtual void TransformFromCOO(const int *rows, const int *cols,
                                const ValueType *data, const int num_rows,
                                const int num_cols, const int num_nonzeros);
                                
  virtual bool found_entry(const int row, const int col, ValueType val) const;

  virtual bool found_entries(const int *rows, int num_rows,
                             const int *cols, int num_cols,
                             const ValueType *values) const; 

  inline ValueType& matrix_val(int i)
  {
    return this->matrix.val[i];
  }
    
  inline const ValueType& matrix_val(int i) const 
  {
    return this->matrix.val[i];
  }

  inline ValueType const * matrix_val() const 
  {
    return this->matrix.val;
  }

  inline ValueType * matrix_val()  
  {
    return this->matrix.val;
  }
  
  inline int& matrix_row(int i)
  {
    return this->matrix.row[i];
  }
    
  inline const int& matrix_row(int i) const 
  {
    return this->matrix.row[i];
  }

  inline int const * matrix_row() const 
  {
    return this->matrix.row;
  }

  inline int * matrix_row()  
  {
    return this->matrix.row;
  }
    
  inline int& matrix_col(int i)
  {
    return this->matrix.col[i];
  }
    
  inline const int& matrix_col(int i) const 
  {
    return this->matrix.col[i];
  }

  inline int const * matrix_col() const 
  {
    return this->matrix.col;
  }

  inline int * matrix_col()  
  {
    return this->matrix.col;
  }
    
  inline int * diag_ptr()
  {
    return this->diag_ptr_;
  }

  inline int const * diag_ptr() const
  {
    return this->diag_ptr_;
  }
  
  // @@ Bernd Doser, 2015-10-16: Data should be private.
  hiflow::la::CSR_lMatrixType< ValueType > matrix;
    
  inline void set_updated_values(bool flag)
  {
    this->updated_values_ = flag;
  }
  
  inline bool updated_values() const 
  {
    return updated_values_;
  }

  inline void set_updated_structure(bool flag)
  {
    this->updated_structure_ = flag;
  }
  
  inline bool updated_structure() const 
  {
    return updated_structure_;
  }
protected:
  mutable bool updated_values_;
  mutable bool updated_structure_;  
  
  int* diag_ptr_;
};

/// @brief Provides CPU naive/simple sequential implementation
/// @author Dimitar Lukarski

template < typename ValueType >
class CPUsimple_CSR_lMatrix final: public CPU_CSR_lMatrix< ValueType > {
public:
  CPUsimple_CSR_lMatrix();
  /// The constructor call the Init() function to do the allocation
  /// of the matrix
  CPUsimple_CSR_lMatrix(int init_nnz, int init_num_row, int init_num_col,
                        std::string init_name);
  ~CPUsimple_CSR_lMatrix();

  void VectorMult(const hiflow::la::lVector< ValueType > &invec,
                          hiflow::la::lVector< ValueType > *outvec) const;
  void VectorMult(const CPU_lVector< ValueType > &invec,
                          CPU_lVector< ValueType > *outvec) const;

  void CastFrom(const CPU_CSR_lMatrix< double > &other);
  void CastFrom(const CPU_CSR_lMatrix< float > &other);

  void CastTo(CPU_CSR_lMatrix< double > &other) const;
  void CastTo(CPU_CSR_lMatrix< float > &other) const;
};

/// @brief Provides CPU OpenMP parallel implementation
/// @author Dimitar Lukarski

template < typename ValueType >
class CPUopenmp_CSR_lMatrix final : public CPU_CSR_lMatrix< ValueType > {
public:
  CPUopenmp_CSR_lMatrix();
  CPUopenmp_CSR_lMatrix(int init_nnz, int init_num_row, int init_num_col,
                        std::string init_name);
  ~CPUopenmp_CSR_lMatrix();

  void CloneFrom(const hiflow::la::lMatrix< ValueType > &other);

  void VectorMultAdd(const hiflow::la::lVector< ValueType > &invec,
                             hiflow::la::lVector< ValueType > *outvec) const;
  void VectorMultAdd(const CPU_lVector< ValueType > &invec,
                             CPU_lVector< ValueType > *outvec) const;

  void VectorMult(const hiflow::la::lVector< ValueType > &invec,
                          hiflow::la::lVector< ValueType > *outvec) const;
  void VectorMult(const CPU_lVector< ValueType > &invec,
                          CPU_lVector< ValueType > *outvec) const;

  void
  BlocksPsgauss_seidel(const hiflow::la::lVector< ValueType > &invec,
                       hiflow::la::lVector< ValueType > *outvec,
                       const int num_blocks) const;

  void add_values(const int *rows, int num_rows, const int *cols,
                          int num_cols, const ValueType *values);

  void set_values(const int *rows, int num_rows, const int *cols,
                          int num_cols, const ValueType *values);

  void get_add_values(const int *rows, int num_rows, const int *cols,
                              int num_cols, const int *cols_target,
                              int num_cols_target, ValueType *values) const;

  void VectorMultAdd_submatrix(const int *rows, int num_rows,
                                       const int *cols, int num_cols,
                                       const int *cols_input,
                                       const ValueType *in_values,
                                       ValueType *out_values) const;

  void
  VectorMultAdd_submatrix_vanka(const int *rows, int num_rows,
                                const CPU_lVector< ValueType > &invec,
                                ValueType *out_values) const;

  void
  VectorMultAdd_submatrix_vanka(const int *rows, int num_rows,
                                const hiflow::la::lVector< ValueType > &invec,
                                ValueType *out_values) const;

  void set_num_threads(void);
  void set_num_threads(int num_thread);

  int num_threads(void) const { return this->num_threads_; }

  void CastFrom(const CPU_CSR_lMatrix< double > &other);
  void CastFrom(const CPU_CSR_lMatrix< float > &other);

  void CastTo(CPU_CSR_lMatrix< double > &other) const;
  void CastTo(CPU_CSR_lMatrix< float > &other) const;

  void VectorMultNoDiag(const CPU_lVector< ValueType > &in,
                        CPU_lVector< ValueType > *out) const;

protected:
  int num_threads_;
};

#endif
