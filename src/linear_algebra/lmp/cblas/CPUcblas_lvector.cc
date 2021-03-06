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

#include "config.h"

#include "../lvector_cpu.h"

#include <cassert>
#include <iostream>
#include <cstdlib>
#include <typeinfo>

#include "../lmp_log.h"

#ifdef WITH_CBLAS

extern "C" {
#include <cblas.h>
}

#else

#define ERROR                                                                  \
  LOG_ERROR("no Cblas support");                                               \
  exit(-1);

#endif

using namespace hiflow::la;

template < typename ValueType >
CPUcblas_lVector< ValueType >::CPUcblas_lVector(const int size,
                                                const std::string name) {
#ifdef WITH_CBLAS
  this->Init(size, name);
  this->implementation_name_ = "CBLAS";
  this->implementation_id_ = BLAS;
#else
  ERROR;
#endif
}

template < typename ValueType >
CPUcblas_lVector< ValueType >::CPUcblas_lVector() {
#ifdef WITH_CBLAS
  this->implementation_name_ = "CBLAS";
  this->implementation_id_ = BLAS;
#else
  ERROR;
#endif
}

template < typename ValueType >
CPUcblas_lVector< ValueType >::~CPUcblas_lVector() {}

template <> int CPUcblas_lVector< double >::ArgMin(void) const {
#ifdef WITH_CBLAS
  int minimum = -1;

  assert(this->get_size() > 0);

  minimum = cblas_idamax(this->get_size(), this->buffer, 1);

  return minimum;
#else
  ERROR;
  return -1;
#endif
}

template <> int CPUcblas_lVector< float >::ArgMin(void) const {
#ifdef WITH_CBLAS
  int minimum = -1;

  assert(this->get_size() > 0);

  minimum = cblas_isamax(this->get_size(), this->buffer, 1);

  return minimum;
#else
  ERROR;
  return -1;
#endif
}

template <> int CPUcblas_lVector< double >::ArgMax(void) const {
#ifdef WITH_CBLAS
  int maximum = -1;

  assert(this->get_size() > 0);

  maximum = cblas_idamax(this->get_size(), this->buffer, 1);

  return maximum;
#else
  ERROR;
  return -1;
#endif
}

template <> int CPUcblas_lVector< float >::ArgMax(void) const {
#ifdef WITH_CBLAS
  int maximum = -1;

  assert(this->get_size() > 0);

  maximum = cblas_isamax(this->get_size(), this->buffer, 1);

  return maximum;
#else
  ERROR;
  return -1;
#endif
}

template <> double CPUcblas_lVector< double >::Norm1(void) const {
#ifdef WITH_CBLAS
  double nrm1 = -1;

  assert(this->get_size() > 0);

  nrm1 = cblas_dasum(this->get_size(), this->buffer, 1);

  return nrm1;
#else
  ERROR;
  return -1.0;
#endif
}

template <> float CPUcblas_lVector< float >::Norm1(void) const {
#ifdef WITH_CBLAS
  float nrm1 = -1;

  assert(this->get_size() > 0);

  nrm1 = cblas_sasum(this->get_size(), this->buffer, 1);

  return nrm1;
#else
  ERROR;
  return -1.0;
#endif
}

template <> double CPUcblas_lVector< double >::Norm2(void) const {
#ifdef WITH_CBLAS
  double nrm2 = -1;

  assert(this->get_size() > 0);

  nrm2 = cblas_dnrm2(this->get_size(), this->buffer, 1);

  return nrm2;
#else
  ERROR;
  return -1.0;
#endif
}

template <> float CPUcblas_lVector< float >::Norm2(void) const {
#ifdef WITH_CBLAS
  float nrm2 = -1;

  assert(this->get_size() > 0);

  nrm2 = cblas_snrm2(this->get_size(), this->buffer, 1);

  return nrm2;
#else
  ERROR;
  return -1.0;
#endif
}

template < typename ValueType >
ValueType CPUcblas_lVector< ValueType >::NormMax(void) const {
#ifdef WITH_CBLAS
  return std::abs(this->buffer[this->ArgMax()]);
#else
  ERROR;
  return -1.0;
#endif
}

template < typename ValueType >
ValueType
CPUcblas_lVector< ValueType >::Dot(const lVector< ValueType > &vec) const {
  ValueType dot = 0.0;

  if (const CPU_lVector< ValueType > *casted_vec =
          dynamic_cast< const CPU_lVector< ValueType > * >(&vec)) {

    dot = this->Dot(*casted_vec);

  } else {
    LOG_ERROR("CPUcblas_lVector::Dot unsupported vectors");
    this->print();
    vec.print();
    exit(-1);
  }

  return dot;
}

template <>
double CPUcblas_lVector< double >::Dot(const CPU_lVector< double > &vec) const {
#ifdef WITH_CBLAS
  double dot = 0.0;

  assert(this->get_size() > 0);
  assert(this->get_size() == vec.get_size());

  dot = cblas_ddot(this->get_size(), this->buffer, 1, vec.buffer, 1);

  return dot;
#else
  ERROR;
  return -1.0;
#endif
}

template <>
float CPUcblas_lVector< float >::Dot(const CPU_lVector< float > &vec) const {
#ifdef WITH_CBLAS
  float dot = 0.0;

  assert(this->get_size() > 0);
  assert(this->get_size() == vec.get_size());

  dot = cblas_sdot(this->get_size(), this->buffer, 1, vec.buffer, 1);

  return dot;
#else
  ERROR;
  return -1.0;
#endif
}

template < typename ValueType >
void CPUcblas_lVector< ValueType >::Axpy(const lVector< ValueType > &vec,
                                         const ValueType scalar) {
  if (const CPU_lVector< ValueType > *casted_vec =
          dynamic_cast< const CPU_lVector< ValueType > * >(&vec)) {

    this->Axpy(*casted_vec, scalar);

  } else {
    LOG_ERROR("CPUcblas_lVector::Axpy unsupported vectors")
    this->print();
    vec.print();
    exit(-1);
  }
}

template <>
void CPUcblas_lVector< double >::Axpy(const CPU_lVector< double > &vec,
                                      const double scalar) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);
  assert(this->get_size() == vec.get_size());

  cblas_daxpy(this->get_size(), scalar, vec.buffer, 1, this->buffer, 1);

#else
  ERROR;
#endif
}

template <>
void CPUcblas_lVector< float >::Axpy(const CPU_lVector< float > &vec,
                                     const float scalar) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);
  assert(this->get_size() == vec.get_size());

  cblas_saxpy(this->get_size(), scalar, vec.buffer, 1, this->buffer, 1);

#else
  ERROR;
#endif
}

template < typename ValueType >
void CPUcblas_lVector< ValueType >::ScaleAdd(const ValueType scalar,
                                             const lVector< ValueType > &vec) {

  if (const CPU_lVector< ValueType > *casted_vec =
          dynamic_cast< const CPU_lVector< ValueType > * >(&vec)) {

    this->ScaleAdd(scalar, *casted_vec);

  } else {
    LOG_ERROR("CPUcblas_lVector::ScaleAdd unsupported vectors");
    this->print();
    vec.print();
    exit(-1);
  }
}

template <>
void CPUcblas_lVector< double >::ScaleAdd(const double scalar,
                                          const CPU_lVector< double > &vec) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);
  assert(this->get_size() == vec.get_size());

  cblas_dscal(this->get_size(), scalar, this->buffer, 1);
  cblas_daxpy(this->get_size(), (double)(1.0), vec.buffer, 1, this->buffer, 1);

#else
  ERROR;
#endif
}

template <>
void CPUcblas_lVector< float >::ScaleAdd(const float scalar,
                                         const CPU_lVector< float > &vec) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);
  assert(this->get_size() == vec.get_size());

  cblas_sscal(this->get_size(), scalar, this->buffer, 1);
  cblas_saxpy(this->get_size(), (float)(1.0), vec.buffer, 1, this->buffer, 1);

#else
  ERROR;
#endif
}

template <> void CPUcblas_lVector< double >::Scale(const double scalar) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);

  cblas_dscal(this->get_size(), scalar, this->buffer, 1);

#else
  ERROR;
#endif
}

template <> void CPUcblas_lVector< float >::Scale(const float scalar) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);

  cblas_sscal(this->get_size(), scalar, this->buffer, 1);

#else
  ERROR;
#endif
}

template < typename ValueType >
void CPUcblas_lVector< ValueType >::Rot(lVector< ValueType > *vec,
                                        const ValueType &sc,
                                        const ValueType &ss) {

  if (CPU_lVector< ValueType > *casted_vec =
          dynamic_cast< CPU_lVector< ValueType > * >(vec)) {

    this->Rot(casted_vec, sc, ss);

  } else {
    LOG_ERROR("CPUcblas_lVector::Rot unsupported vectors");
    this->print();
    vec->print();
    exit(-1);
  }
}

template <>
void CPUcblas_lVector< double >::Rot(CPU_lVector< double > *vec,
                                     const double &sc, const double &ss) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);
  assert(this->get_size() == vec->get_size());

  cblas_drot(this->get_size(), vec->buffer, 1, this->buffer, 1, sc, ss);

#else
  ERROR;
#endif
}

template <>
void CPUcblas_lVector< float >::Rot(CPU_lVector< float > *vec, const float &sc,
                                    const float &ss) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);
  assert(this->get_size() == vec->get_size());

  cblas_srot(this->get_size(), vec->buffer, 1, this->buffer, 1, sc, ss);

#else
  ERROR;
#endif
}

template <>
void CPUcblas_lVector< double >::Rotg(double *sa, double *sb, double *sc,
                                      double *ss) const {
#ifdef WITH_CBLAS

  cblas_drotg(sa, sb, sc, ss);

#else
  ERROR;
#endif
}

template <>
void CPUcblas_lVector< float >::Rotg(float *sa, float *sb, float *sc,
                                     float *ss) const {
#ifdef WITH_CBLAS

  cblas_srotg(sa, sb, sc, ss);

#else
  ERROR;
#endif
}

template < typename ValueType >
void CPUcblas_lVector< ValueType >::Rotm(lVector< ValueType > *vec,
                                         const ValueType &sparam) {

  if (CPU_lVector< ValueType > *casted_vec =
          dynamic_cast< CPU_lVector< ValueType > * >(vec)) {

    this->Rotm(casted_vec, sparam);

  } else {
    LOG_ERROR("CPUcblas_lVector::Rotm unsupported vectors");
    this->print();
    vec->print();
    exit(-1);
  }
}

template <>
void CPUcblas_lVector< double >::Rotm(CPU_lVector< double > *vec,
                                      const double &sparam) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);
  assert(this->get_size() == vec->get_size());

  cblas_drotm(this->get_size(), vec->buffer, 1, this->buffer, 1, &sparam);

#else
  ERROR;
#endif
}

template <>
void CPUcblas_lVector< float >::Rotm(CPU_lVector< float > *vec,
                                     const float &sparam) {
#ifdef WITH_CBLAS

  assert(this->get_size() > 0);
  assert(this->get_size() == vec->get_size());

  cblas_srotm(this->get_size(), vec->buffer, 1, this->buffer, 1, &sparam);

#else
  ERROR;
#endif
}

template <>
void CPUcblas_lVector< double >::Rotmg(double *sd1, double *sd2, double *x1,
                                       const double &x2, double *sparam) const {
#ifdef WITH_CBLAS

  cblas_drotmg(sd1, sd2, x1, x2, sparam);

#else
  ERROR;
#endif
}

template <>
void CPUcblas_lVector< float >::Rotmg(float *sd1, float *sd2, float *x1,
                                      const float &x2, float *sparam) const {
#ifdef WITH_CBLAS

  cblas_srotmg(sd1, sd2, x1, x2, sparam);

#else
  ERROR;
#endif
}

template < typename ValueType >
void CPUcblas_lVector< ValueType >::CastFrom(const CPU_lVector< double > &vec) {
  assert(this->get_size() == vec.get_size());

  for (int i = 0; i < this->get_size(); ++i)
    (this->buffer)[i] = static_cast< ValueType >(vec.buffer[i]);
}

template < typename ValueType >
void CPUcblas_lVector< ValueType >::CastFrom(const CPU_lVector< float > &vec) {
  assert(this->get_size() == vec.get_size());

  for (int i = 0; i < this->get_size(); ++i)
    (this->buffer)[i] = static_cast< ValueType >(vec.buffer[i]);
}

template < typename ValueType >
void CPUcblas_lVector< ValueType >::CastTo(CPU_lVector< double > &vec) const {
  assert(this->get_size() == vec.get_size());

  for (int i = 0; i < this->get_size(); ++i)
    vec.buffer[i] = static_cast< double >((this->buffer)[i]);
}

template < typename ValueType >
void CPUcblas_lVector< ValueType >::CastTo(CPU_lVector< float > &vec) const {
  assert(this->get_size() == vec.get_size());

  for (int i = 0; i < this->get_size(); ++i)
    vec.buffer[i] = static_cast< float >((this->buffer)[i]);
}

//
//
//
//

template class CPUcblas_lVector< double >;
template class CPUcblas_lVector< float >;
