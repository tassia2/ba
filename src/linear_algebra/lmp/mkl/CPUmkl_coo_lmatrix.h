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

#ifndef __LMATRIX_MKL_COO_CPU_H
#define __LMATRIX_MKL_COO_CPU_H

#include <iostream>
#include <stdlib.h>

#include "linear_algebra/lmp/lmatrix_coo_cpu.h"

// @brief Provides wrapper to CPU Intel/MKL implementation
/// @author Dimitar Lukarski

template < typename ValueType >
class CPUmkl_COO_lMatrix : public CPU_COO_lMatrix< ValueType > {
public:
  CPUmkl_COO_lMatrix();
  CPUmkl_COO_lMatrix(int init_nnz, int init_num_row, int init_num_col,
                     std::string init_name);
  virtual ~CPUmkl_COO_lMatrix();

  virtual void CloneFrom(const hiflow::la::lMatrix< ValueType > &other);

  virtual void VectorMult(const hiflow::la::lVector< ValueType > &invec,
                          hiflow::la::lVector< ValueType > *outvec) const;
  virtual void VectorMult(const CPU_lVector< ValueType > &invec,
                          CPU_lVector< ValueType > *outvec) const;

  virtual void set_num_threads(void);
  virtual void set_num_threads(int num_thread);

  virtual int num_threads(void) const { return this->num_threads_; }

  virtual void CastFrom(const CPU_COO_lMatrix< double > &other);
  virtual void CastFrom(const CPU_COO_lMatrix< float > &other);

  virtual void CastTo(CPU_COO_lMatrix< double > &other) const;
  virtual void CastTo(CPU_COO_lMatrix< float > &other) const;

protected:
  int num_threads_;
};

#endif
