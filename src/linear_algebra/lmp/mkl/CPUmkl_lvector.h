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

#ifndef __LVECTOR_MKL_CPU_H
#define __LVECTOR_MKL_CPU_H

#include <cstring>
#include <iostream>

#include "linear_algebra/lmp/lvector_cpu.h"

/// @brief Provides wrapper to CPU Intel/MKL implementation
/// of the Blas 1 routines
/// @author Dimitar Lukarski

template < typename ValueType >
class CPUmkl_lVector : public CPU_lVector< ValueType > {
public:
  CPUmkl_lVector();
  CPUmkl_lVector(const int size, const std::string name);
  virtual ~CPUmkl_lVector();

  virtual void CloneFrom(const hiflow::la::lVector< ValueType > &vec);

  virtual int ArgMin(void) const;
  virtual int ArgMax(void) const;
  virtual ValueType Norm1(void) const;
  virtual ValueType Norm2(void) const;
  virtual ValueType NormMax(void) const;
  virtual ValueType Dot(const hiflow::la::lVector< ValueType > &vec) const;
  virtual ValueType Dot(const CPU_lVector< ValueType > &vec) const;
  virtual void Axpy(const hiflow::la::lVector< ValueType > &vec,
                    const ValueType scalar);
  virtual void Axpy(const CPU_lVector< ValueType > &vec,
                    const ValueType scalar);
  virtual void ScaleAdd(const ValueType scalar,
                        const hiflow::la::lVector< ValueType > &vec);
  virtual void ScaleAdd(const ValueType scalar,
                        const CPU_lVector< ValueType > &vec);
  virtual void Scale(const ValueType scalar);
  virtual void Rot(hiflow::la::lVector< ValueType > *vec, const ValueType &sc,
                   const ValueType &ss);
  virtual void Rot(CPU_lVector< ValueType > *vec, const ValueType &sc,
                   const ValueType &ss);
  virtual void Rotg(ValueType *sa, ValueType *sb, ValueType *sc,
                    ValueType *ss) const;
  virtual void Rotm(hiflow::la::lVector< ValueType > *vec,
                    const ValueType &sparam);
  virtual void Rotm(CPU_lVector< ValueType > *vec, const ValueType &sparam);
  virtual void Rotmg(ValueType *sd1, ValueType *sd2, ValueType *x1,
                     const ValueType &x2, ValueType *sparam) const;

  virtual void set_num_threads(void);
  virtual void set_num_threads(int num_thread);

  int num_threads(void) const { return this->num_threads_; }

  virtual void CastFrom(const CPU_lVector< double > &other);
  virtual void CastFrom(const CPU_lVector< float > &other);

  virtual void CastTo(CPU_lVector< double > &vec) const;
  virtual void CastTo(CPU_lVector< float > &vec) const;

protected:
  int num_threads_;
};

#endif
