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

#ifndef __LMATRIX_MKL_CSR_CPU_H
#define __LMATRIX_MKL_CSR_CPU_H

#include <iostream>
#include <stdlib.h>

#include "linear_algebra/lmp/lmatrix_csr_cpu.h"

#ifdef WITH_MKL

#include <mkl.h>
#include <mkl_spblas.h>

#endif

template < typename ValueType >
class CPUmkl_CSR_lMatrix final : public CPU_CSR_lMatrix< ValueType > {
public:
  CPUmkl_CSR_lMatrix();
  /// The constructor call the Init() function to the do allocation
  /// of the matrix
  CPUmkl_CSR_lMatrix(int init_nnz, int init_num_row, int init_num_col,
                     std::string init_name);
  ~CPUmkl_CSR_lMatrix();

  void Init(const int init_nnz, const int init_num_row,
            const int init_num_col, const std::string init_name);
                    
  void init_structure(const int *rows, const int *cols);
  
  void CloneFrom(const hiflow::la::lMatrix< ValueType > &other);

  void VectorMult(const hiflow::la::lVector< ValueType > &invec,
                  hiflow::la::lVector< ValueType > *outvec) const;
  void VectorMult(const CPU_lVector< ValueType > &invec,
                  CPU_lVector< ValueType > *outvec) const;

  void set_num_threads(void);
  void set_num_threads(int num_thread);

  int num_threads(void) const { return this->num_threads_; }

  void CastFrom(const CPU_CSR_lMatrix< double > &other);
  void CastFrom(const CPU_CSR_lMatrix< float > &other);

  void CastTo(CPU_CSR_lMatrix< double > &other) const;
  void CastTo(CPU_CSR_lMatrix< float > &other) const;
  
  virtual void Compress(void) override;

protected:
  void update_matrix_handle() const;
  void update_matrix_handle_values() const;
  bool handle_is_created() const 
  {
    return this->handle_created_;
  }
  
  void set_handle_created(bool flag)
  {
    handle_created_ = flag;
  }
  
#ifdef WITH_MKL
  matrix_descr A_descr_;
  mutable sparse_matrix_t A_;
#endif
  int num_threads_;
  mutable bool handle_created_;
};

#endif
