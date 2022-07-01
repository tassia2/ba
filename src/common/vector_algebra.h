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

#ifndef HIFLOW_VECTOR_ALGEBRA_H
#define HIFLOW_VECTOR_ALGEBRA_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include "common/macros.h"
#include "common/log.h"

/// @brief This file contains template classes for representing small
/// floating-point vectors and matrices with sizes fixed at
/// compile-time; as well as common mathematical operations for these
/// objects.

/// @author Staffan Ronnas, Simon Gawlok

namespace {
// Tolerance for comparing elements of the vector.
const double COMPARISON_TOL = 1.e-14;
} // namespace

namespace hiflow {

template < size_t M, size_t N, class DataType > class sMat;

/// \brief Class representing a floating-point vector of size N.
///
/// \details The class also supports common mathematical operations.

template < size_t N, class DataType > class sVec {
public:
  typedef DataType value_type;

  // Constructors

  sVec() {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] = value_type(0.);
    }
  }

  sVec(const sVec< N, DataType > &v) {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] = v.v_[i];
    }
  }

  explicit sVec(const value_type *values) {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] = values[i];
    }
  }

  explicit sVec(const std::vector< value_type > values) {
    assert(values.size() <= N);
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      if (values.size() > i) {
        this->v_[i] = values[i];
      } else {
        this->v_[i] = value_type(0.);
      }
    }
  }

  explicit sVec(const std::vector< value_type > values, size_t begin) {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      if (values.size() > i) {
        this->v_[i] = values[begin+i];
      } else {
        this->v_[i] = value_type(0.);
      }
    }
  }

  // Assignment

  sVec< N, DataType > &operator=(const sVec< N, DataType > &v) {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] = v.v_[i];
    }
    return *this;
  }

  template < size_t M >
  sVec< N, DataType > &operator=(const sVec< M, DataType > &v) {
    size_t len = std::min(N,M);
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < len; ++i) {
      this->v_[i] = v.v_[i];
    }
    return *this;
  }

  // Access operator

  inline value_type operator[](size_t i) const {
    // prevent i < 0 with i unsigned warning
    assert(i < N);
    return v_[i];
  }

  inline value_type &operator[](size_t i) {
    // prevent i < 0 with i unsigned warning
    assert(i < N);
    return v_[i];
  }

  inline void set (const size_t i, const DataType val)
  {
    // prevent i < 0 with i unsigned warning
    assert(i < N);
    this->v_[i] = val;
  }

  inline void add (const size_t i, const DataType val)
  {
    // prevent i < 0 with i unsigned warning
    assert(i < N);
    this->v_[i] += val;
  }

  // lexicographic comparison

  inline bool operator< (const sVec< N, DataType >rhs) const {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) 
    {
      if (this->v_[i] < rhs[i])
      {
        return true;
      }
    }
    return false;
  }

  inline bool operator<= (const sVec< N, DataType >rhs) const {
    if ((*this)<rhs)
    {
      return true;
    }
    return ((*this)==rhs);
  }
  
  // Multiplication by scalar

  inline sVec< N, DataType > &operator*=(const value_type s) {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] = this->v_[i] * s;
    }
    return *this;
  }

  // Division by scalar

  inline sVec< N, DataType > &operator/=(const value_type s) {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] = this->v_[i] / s;
    }
    return *this;
  }

  // Addition

  inline sVec< N, DataType > &operator+=(sVec< N, DataType > v) {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] += v.v_[i];
    }
    return *this;
  }

  // Subtraction

  inline sVec< N, DataType > &operator-=(sVec< N, DataType > v) {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] -= v.v_[i];
    }
    return *this;
  }

  // Comparison

  inline bool operator==(sVec< N, DataType > v) const {
    for (size_t i = 0; i < N; ++i) {
      if (std::abs(v_[i] - v[i]) > static_cast< DataType >(COMPARISON_TOL)) {
        return false;
      }
    }
    return true;
  }

  inline bool eq(sVec< N, DataType > v, DataType eps) const {
    for (size_t i = 0; i < N; ++i) {
      if (std::abs(v_[i] - v[i]) > eps) {
        return false;
      }
    }
    return true;
  }
  
  inline bool operator!=(sVec< N, DataType > v) const {
    return  !(*this == v);
  }

  inline bool neq(sVec< N, DataType > v, DataType eps) const {
    return !(this->eq(v,eps));
  }
  
  // Size

  inline size_t size() const { return N; }

  // Add multiple of second vector

  inline void Axpy(const sVec< N, DataType > &vec, const DataType alpha) 
  {
    if constexpr (N == 1)
    {
      this->v_[0] += vec.v_[0] * alpha;
    }
    else if constexpr (N == 2)
    {
      this->v_[0] += vec.v_[0] * alpha;
      this->v_[1] += vec.v_[1] * alpha;
    }
    else if constexpr (N == 3)
    {
      this->v_[0] += vec.v_[0] * alpha;
      this->v_[1] += vec.v_[1] * alpha;
      this->v_[2] += vec.v_[2] * alpha;
    }
    else 
    {
PRAGMA_LOOP_VEC
      for (size_t i = 0; i < N; ++i) {
        this->v_[i] += vec.v_[i] * alpha;
      }
    }
  }

  operator std::vector< value_type >() const {
    return std::vector< value_type >(&v_[0], &v_[N]);
  }

  value_type const * begin() const {
    return &v_[0];
  }

  value_type const * end() const {
    return &v_[N];
  }

  void Zeros() {
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < N; ++i) {
      this->v_[i] = 0;
    }
  }

private:
  value_type v_[N];
};

// Binary vector operations (do not modify the arguments).

template < size_t N, class DataType >
inline sVec< N, DataType >
operator*(const sVec< N, DataType > &v,
          const typename sVec< N, DataType >::value_type s) {
  sVec< N, DataType > tmp(v);
  tmp *= s;
  return tmp;
}

template < size_t N, class DataType >
inline sVec< N, DataType >
operator*(const typename sVec< N, DataType >::value_type s,
          const sVec< N, DataType > &v) {
  return v * s;
}

template < size_t N, class DataType >
inline sVec< N, DataType > operator+(const sVec< N, DataType > &v1,
                                    const sVec< N, DataType > &v2) {
  sVec< N, DataType > tmp(v1);
  tmp += v2;
  return tmp;
}

template < size_t N, class DataType >
inline sVec< N, DataType > operator-(const sVec< N, DataType > &v1,
                                    const sVec< N, DataType > &v2) {
  sVec< N, DataType > tmp(v1);
  tmp -= v2;
  return tmp;
}

/// \brief Computes the euclidean distance of two vectors.

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type distance(const sVec< N, DataType > &v1, 
                                                        const sVec< N, DataType > &v2) 
{
  typename sVec< N, DataType >::value_type res = typename sVec< N, DataType >::value_type();
  for (size_t i = 0; i < N; ++i) 
  {
    res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  }

  return std::sqrt(res);
}

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type distance(const sVec< N, DataType > &v1, 
                                                        const sVec< N, DataType > &v2,
                                                        const DataType p) 
{
  typename sVec< N, DataType >::value_type res = typename sVec< N, DataType >::value_type();
  for (size_t i = 0; i < N; ++i) 
  {
    res += std::pow(std::abs(v1[i] - v2[i]), p); 
  }

  return std::pow(res, 1. / p);
}

/// \brief Computes the scalar product of two vectors.

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type dot(const sVec< N, DataType > &v1, 
                                                   const sVec< N, DataType > &v2) 
{
  typename sVec< N, DataType >::value_type res = typename sVec< N, DataType >::value_type();
  for (size_t i = 0; i < N; ++i) 
  {
    res += v1[i] * v2[i];
  }

  return res;
}

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type dot(const sVec< N, DataType > &v1, 
                                                   const sVec< N, DataType > &v2,
                                                   const sVec< N, DataType > &v3) 
{
  typename sVec< N, DataType >::value_type res = typename sVec< N, DataType >::value_type();
  for (size_t i = 0; i < N; ++i) 
  {
    res += v1[i] * v2[i] * v3[i];
  }

  return res;
}

/// \brief Computes the cross product of two 3d vectors.

template < class DataType >
inline sVec< 3, DataType > cross(const sVec< 3, DataType > &v1,
                                const sVec< 3, DataType > &v2) {
  sVec< 3, DataType > v3;
  v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
  v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
  v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
  return v3;
}

/// \brief this is just a dummy function
template < class DataType >
inline sVec< 2, DataType > cross(const sVec< 2, DataType > &v1,
                                const sVec< 2, DataType > &v2) {
  assert (false);
  sVec< 2, DataType > v3;
  return v3;
}

/// \brief this is just a dummy function
template < class DataType >
inline sVec< 1, DataType > cross(const sVec< 1, DataType > &v1,
                                const sVec< 1, DataType > &v2) {
  assert (false);
  sVec< 1, DataType > v3;
  return v3;
}

/// \brief Computes the sum of the vector components.

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type
sum(const sVec< N, DataType > &v) {
  typename sVec< N, DataType >::value_type res =
      typename sVec< N, DataType >::value_type();
  for (size_t i = 0; i < N; ++i) {
    res += v[i];
  }
  return res;
}

/// \brief Computes the Euclidean norm of a vector.

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type
norm(const sVec< N, DataType > &v1) {
  typename sVec< N, DataType >::value_type res =
      typename sVec< N, DataType >::value_type();
  res = std::sqrt(dot(v1, v1));
  return res;
}

/// \brief Computes the L1 norm of a vector.

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type
norm1(const sVec< N, DataType > &v) {
  typename sVec< N, DataType >::value_type res =
      typename sVec< N, DataType >::value_type();
      
  for (size_t i = 0; i != N; ++i) {
    res += std::abs(v[i]);
  }
  return res;
}

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type
max(const sVec< N, DataType > &v1) 
{
  typename sVec< N, DataType >::value_type max_val = std::numeric_limits<DataType>::min();

  for (int d=0; d!=N; ++d)
  {
    max_val = std::max(max_val, v1[d]);
  }
  return max_val;
}

template < size_t N, class DataType >
inline typename sVec< N, DataType >::value_type
min(const sVec< N, DataType > &v1) 
{
  typename sVec< N, DataType >::value_type min_val = std::numeric_limits<DataType>::max();

  for (int d=0; d!=N; ++d)
  {
    min_val = std::min(min_val, v1[d]);
  }
  return min_val;
}

/// \brief Computes the normed normal of a 2d vector.

template < class DataType >
inline sVec< 2, DataType > normal(const sVec< 2, DataType > &v) {
  sVec< 2, DataType > v2;
  v2[0] = -v[1];
  v2[1] = v[0];
  DataType v2norm = norm(v2);
  v2 /= v2norm;
  return v2;
}

/// \brief Computes the normed normal of two 3d vectors.

template < class DataType >
inline sVec< 3, DataType > normal(const sVec< 3, DataType > &v1,
                                 const sVec< 3, DataType > &v2) {
  sVec< 3, DataType > v3 = cross(v1, v2);
  DataType v3norm = norm(v3);
  v3 /= v3norm;
  return v3;
}

/// \brief Computes the normed normal of two 3d vectors.

template < class DataType >
inline sVec< 3, DataType > normal(const sVec< 6, DataType > &v1v2) {
  DataType v1_array[3] = {v1v2[0], v1v2[1], v1v2[2]};
  DataType v2_array[3] = {v1v2[3], v1v2[4], v1v2[5]};
  sVec< 3, DataType > v1(v1_array);
  sVec< 3, DataType > v2(v2_array);
  sVec< 3, DataType > v3 = cross(v1, v2);
  DataType v3norm = norm(v3);
  v3 /= v3norm;
  return v3;
}

/// \brief Checks if two 2d vectors are parallel

template < class DataType >
inline bool is_parallel(const sVec<2, DataType> &v1, const sVec<2, DataType> &v2) {
  const DataType EPS = 1.0e3 * std::numeric_limits<DataType>::epsilon();
  return std::abs(v1[0] * v2[1] - v1[1] * v2[0]) < EPS;
}

/// \brief Checks if two 3d vectors are parallel

template < class DataType >
inline bool is_parallel(const sVec<3, DataType> &v1, const sVec<3, DataType> &v2) {
  const DataType EPS = 1.0e3 * std::numeric_limits<DataType>::epsilon();
  sVec<3, DataType> w = cross(v1, v2);
  return (std::abs(w[0]) < EPS) && (std::abs(w[1]) < EPS) && (std::abs(w[2]) < EPS);
}

/// \brief Output operator for vectors.

template < size_t N, class DataType >
std::ostream &operator<<(std::ostream &os, const sVec< N, DataType > &v) {
  os << "[ ";
  for (size_t i = 0; i < N; ++i) {
    os << v[i] << " ";
  }
  os << "]";
  return os;
}

/// \brief Class representing a floating-point matrix with M rows
/// and N columns.

template < size_t M, size_t N, class DataType > class sMat {
public:
  typedef DataType value_type;

  // Constructors

  sMat() {
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] = value_type(0.);
    }
  }

  sMat(const sMat< M, N, DataType > &m) {
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] = m.m_[i];
    }
  }

  // second argument for compatibility reasons wth class pMat
  explicit sMat(const value_type *values, bool input_is_col_major = false) {
    assert (input_is_col_major == false);
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] = values[i];
    }
  }

  // second argument for compatibility reasons wth class pMat
  explicit sMat(const std::vector<value_type> & values, bool input_is_col_major = false) {
    assert (input_is_col_major == false);
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] = values[i];
    }
  }

  inline sMat< M, N, DataType > &operator=(const sMat< M, N, DataType > &m) {
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] = m.m_[i];
    }
    return *this;
  }

  inline void Zeros() {
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] = 0.;
    }
  }

  constexpr inline size_t num_row() const
  {
    return M;
  }

  constexpr inline size_t num_col() const 
  {
    return N;
  }

  // Element access

  inline value_type operator()(size_t i, size_t j) const {
    // prevent i < 0 with i unsigned warning
    assert(j < N);
    assert(i < M);

    return m_[i * N + j];
  }

  inline value_type &operator()(size_t i, size_t j) {
    // prevent i < 0 with i unsigned warning
    assert(j < N);
    assert(i < M);

    return m_[i * N + j];
  }

  inline void set (const size_t i, const size_t j, const value_type val)
  {
    // prevent i < 0 with i unsigned warning
    assert(j < N);
    assert(i < M);

    this->m_[i * N + j] = val;
  }

  inline void add (const size_t i, const size_t j, const value_type val)
  {
    // prevent i < 0 with i unsigned warning
    assert(j < N);
    assert(i < M);

    this->m_[i * N + j] += val;
  }

  // Multiplication by scalar

  inline sMat< M, N, DataType > &operator*=(const value_type s) {
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      m_[i] *= s;
    }
    return *this;
  }

  // Division by scalar

  inline sMat< M, N, DataType > &operator/=(const value_type s) {
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] /= s;
    }
    return *this;
  }

  // sMatrix addition

  inline sMat< M, N, DataType > &operator+=(const sMat< M, N, DataType > &m) {
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] += m.m_[i];
    }
    return *this;
  }

  // sMatrix subtraction

  inline sMat< M, N, DataType > &operator-=(const sMat< M, N, DataType > &m) {
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] -= m.m_[i];
    }
    return *this;
  }

  // sMatrix multiplication with square matrix

  inline sMat< M, N, DataType > &operator*=(const sMat< N, N, DataType > &m) {
    // copy values of this to new array
    value_type cpy[M * N];
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      cpy[i] = this->m_[i];
      this->m_[i] = 0;
    }

    // perform matrix-matrix multiplication
    for (size_t i = 0; i < M; ++i) { // loop over rows
      const size_t i_offset = i * N;
      for (size_t k = 0; k < N; ++k) { // inner loop
        const size_t k_offset = k * N;
PRAGMA_LOOP_VEC
        for (size_t j = 0; j < N; ++j) { // loop over columns
          this->m_[i_offset + j] += cpy[i_offset + k] * m.m_[k_offset + j];
        }
      }
    }
    return *this;
  }

  // sMatrix comparison

  inline bool operator==(const sMat< M, N, DataType > &m) {
    const size_t len = M * N;
    for (size_t i = 0; i != len; ++i) {
        if (std::abs(this->m_[i] - m.m_[i]) > static_cast<DataType>(COMPARISON_TOL))
          return false;
    }
    return true;
  }

  // sMatrix comparison

  inline bool operator!=(const sMat< M, N, DataType > &m) {
    return !(*this == m);
  }

  // sVector multiplication

  inline void VectorMult(const sVec< N, DataType > &in,
                         sVec< M, DataType > &out) const {
    for (size_t i = 0, e_i = M; i < e_i; ++i) {
      out[i] = static_cast< DataType >(0);
      const size_t i_ind = i * N;
//PRAGMA_LOOP_VEC
      for (size_t j = 0, e_j = N; j < e_j; ++j) {
        out[i] += this->m_[i_ind + j] * in[j];
      }
    }
  }

  // Add multiple of second matrix

  inline void Axpy(const sMat< M, N, DataType > &mat, const DataType alpha) {
PRAGMA_LOOP_VEC
    for (size_t i = 0, e_i = M * N; i < e_i; ++i) {
      this->m_[i] += mat.m_[i] * alpha;
    }
  }

  inline void SetRow(int row, const sVec< N, DataType > &row_vec) {
    assert (row >= 0);
    assert (row < M);
PRAGMA_LOOP_VEC
    const size_t i_ind = row * N;
    for (size_t i = 0; i < N; ++i) {
      this->m_[i_ind + i] = row_vec[i];
    }
  }

  inline void SetCol(int col, const sVec< M, DataType > &col_vec) {
    assert (col >= 0);
    assert (col < N);
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < M; ++i) {
      this->m_[i * N + col] = col_vec[i];
    }
  }

  template < size_t _M, size_t _N, class _ScalarType >
  friend class sMat;

private:
  value_type m_[M * N];
};

// Specializations for (unused) cases M = 0 or N = 0.
/*
template < size_t M, class DataType > class sMat< M, 0, DataType > {};

template < size_t N, class DataType > class sMat< 0, N, DataType > {};
*/
// sMatrix-sMatrix multiplication: out = mat1 * mat2

template < size_t M, size_t N, size_t P, class DataType >
inline void MatrixMatrixMult(sMat< M, P, DataType > &out,
                             const sMat< M, N, DataType > &mat1,
                             const sMat< N, P, DataType > &mat2) {
  out.Zeros();
  for (size_t i = 0; i < M; ++i) {   // loop over rows
    for (size_t k = 0; k < N; ++k) { // inner loop
PRAGMA_LOOP_VEC
      for (size_t j = 0; j < P; ++j) { // loop over columns
        out(i, j) += mat1(i, k) * mat2(k, j);
      }
    }
  }
}

// Binary matrix operations (do not modify the arguments).

template < size_t M, size_t N, class DataType >
inline sMat< M, N, DataType >
operator*(const sMat< M, N, DataType > &m,
          const typename sMat< M, N, DataType >::value_type s) {
  sMat< M, N, DataType > tmp(m);
  tmp *= s;
  return tmp;
}

template < size_t M, size_t N, class DataType >
inline sMat< M, N, DataType >
operator*(const typename sMat< M, N, DataType >::value_type s,
          const sMat< M, N, DataType > &m) {
  sMat< M, N, DataType > tmp(m);
  tmp *= s;
  return tmp;
}

template < size_t M, size_t N, class DataType >
inline sMat< M, N, DataType >
operator/(const sMat< M, N, DataType > &m,
          typename sMat< M, N, DataType >::value_type s) {
  sMat< M, N, DataType > tmp(m);
  tmp /= s;
  return tmp;
}

template < size_t M, size_t N, class DataType >
inline sMat< M, N, DataType > operator+(const sMat< M, N, DataType > &m1,
                                       const sMat< M, N, DataType > &m2) {
  sMat< M, N, DataType > tmp(m1);
  tmp += m2;
  return tmp;
}

template < size_t M, size_t N, class DataType >
inline sMat< M, N, DataType > operator-(const sMat< M, N, DataType > &m1,
                                       const sMat< M, N, DataType > &m2) {
  sMat< M, N, DataType > tmp(m1);
  tmp -= m2;
  return tmp;
}

// sMatrix-matrix multiplication with square matrix.

template < size_t M, size_t N, class DataType >
inline sMat< M, N, DataType > operator*(const sMat< M, N, DataType > &m1,
                                       const sMat< N, N, DataType > &m2) {
  sMat< M, N, DataType > tmp(m1);
  tmp *= m2;
  return tmp;
}

// General matrix-matrix multiplication.

template < size_t M, size_t N, size_t P, class DataType >
inline sMat< M, P, DataType > operator*(const sMat< M, N, DataType > &A,
                                       const sMat< N, P, DataType > &B) {
  sMat< M, P, DataType > C;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < P; ++k) {
        C(i, k) += A(i, j) * B(j, k);
      }
    }
  }
  return C;
}

// sMatrix-vector multiplication

template < size_t M, size_t N, class DataType >
inline sVec< M, DataType > operator*(const sMat< M, N, DataType > &m,
                                    const sVec< N, DataType > &v) {
  sVec< M, DataType > mv;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      mv[i] += m(i, j) * v[j];
    }
  }
  return mv;
}

// sVector-matrix multiplication

template < size_t M, size_t N, class DataType >
inline sVec< N, DataType > operator*(const sVec< M, DataType > &v,
                                    const sMat< M, N, DataType > &m) {
  sVec< N, DataType > mv;
  for (size_t i = 0; i < M; ++i) {
PRAGMA_LOOP_VEC
    for (size_t j = 0; j < N; ++j) {
      mv[j] += m(i, j) * v[i];
    }
  }
  return mv;
}

// matrix-matrix inner product
template < size_t M, size_t N, class DataType >
inline typename sMat< M, N, DataType >::value_type dot(const sMat< M, N, DataType > &m1, 
                                                      const sMat< M, N, DataType > &m2) 
{
  typename sMat< M, N, DataType >::value_type res = typename sMat< M, N, DataType >::value_type();
  for (size_t i = 0; i < M; ++i) 
  {
    for (size_t j = 0; j < N; ++j) 
    {
      res += m1(i,j) * m2(i,j);
    }
  }
  return res;
}

// inner product of two vectors with product defined by matrix : res = v1^T M v2
template < size_t M, size_t N, class DataType >
inline typename sMat< M, N, DataType >::value_type inner(const sVec< M, DataType > &v1, 
                                                        const sMat< M, N, DataType > &m,
                                                        const sVec< N, DataType > &v2) 
{
  typename sMat< M, N, DataType >::value_type res = typename sMat< M, N, DataType >::value_type();
  for (size_t i = 0; i < M; ++i) 
  {
    for (size_t j = 0; j < N; ++j) 
    {
      res += v1[i] * m(i,j) * v2[j];
    }
  }
  return res;
}

// inner product of two vectors with product defined by transposed matrix : res = v2^T M v1
template < size_t M, size_t N, class DataType >
inline typename sMat< M, N, DataType >::value_type innerT(const sVec< N, DataType > &v1, 
                                                         const sMat< M, N, DataType > &m,
                                                         const sVec< M, DataType > &v2) 
{
  typename sMat< M, N, DataType >::value_type res = typename sMat< M, N, DataType >::value_type();
  for (size_t i = 0; i < M; ++i) 
  {
    for (size_t j = 0; j < N; ++j) 
    {
      res += v2[i] * m(i,j) * v1[j];
    }
  }
  return res;
}


// sMatrix minor computation (helper function for det and inv of 3x3 matrix)

template < class DataType >
inline typename sMat< 3, 3, DataType >::value_type
matrix_minor(const sMat< 3, 3, DataType > &m, size_t i, size_t j) {
  const int indI0 = (i + 1) % 3;
  const int indI1 = (i + 2) % 3;
  const int indJ0 = (j + 1) % 3;
  const int indJ1 = (j + 2) % 3;
  return m(indI0, indJ0) * m(indI1, indJ1) - m(indI0, indJ1) * m(indI1, indJ0);
}

// Sign of sum of two integers.

template < class DataType >
inline typename sMat< 3, 3, DataType >::value_type sign(size_t i, size_t j) {
  return (i + j) % 2 == 0 ? 1. : -1.;
}

// Determinant for 1x1 matrix.

template < class DataType >
inline DataType det(const sMat< 1, 1, DataType > &m) {
  return m(0, 0);
}

///\brief Determinant for 2x2 matrix.

template < class DataType >
inline DataType det(const sMat< 2, 2, DataType > &m) {
  return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}

///\brief Determinant for 3x3 matrix.

template < class DataType >
inline DataType det(const sMat< 3, 3, DataType > &m) {
  return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
         m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
         m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
}

/// \brief Inverse of 1x1 matrix.

template <size_t M, size_t N, class DataType> 
inline void inv(const sMat< M, N, DataType > &m, sMat< M, N, DataType > &m_inv) 
{
  LOG_ERROR("sMatrix inv not defined for M = " << M << ", N = " << N);
  quit_program();
}

template <>
inline void inv(const sMat< 1, 1, float > &m, sMat< 1, 1, float > &m_inv) 
{
  m_inv(0, 0) = 1. / m(0, 0);
}

template <>
inline void inv(const sMat< 1, 1, double > &m, sMat< 1, 1, double > &m_inv) 
{
  m_inv(0, 0) = 1. / m(0, 0);
}

/// \brief Inverse of 2x2 matrix.

template <>
inline void inv(const sMat< 2, 2, float > &m, sMat< 2, 2, float > &m_inv) 
{
  typename sMat< 2, 2, float >::value_type d = det(m);
  assert(d != 0.);
  m_inv(0, 0) = m(1, 1) / d;
  m_inv(0, 1) = -m(0, 1) / d;
  m_inv(1, 0) = -m(1, 0) / d;
  m_inv(1, 1) = m(0, 0) / d;
}

template <>
inline void inv(const sMat< 2, 2, double > &m, sMat< 2, 2, double > &m_inv) 
{
  typename sMat< 2, 2, double >::value_type d = det(m);
  assert(d != 0.);
  m_inv(0, 0) = m(1, 1) / d;
  m_inv(0, 1) = -m(0, 1) / d;
  m_inv(1, 0) = -m(1, 0) / d;
  m_inv(1, 1) = m(0, 0) / d;
}

/// \brief Inverse of 3x3 matrix.

template <>
inline void inv(const sMat< 3, 3, float > &m, sMat< 3, 3, float > &m_inv) 
{
  // compute determinant
  const float d = det(m);
  assert(d != static_cast< float >(0));

  m_inv(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / d;
  m_inv(0, 1) = -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)) / d;
  m_inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / d;

  m_inv(1, 0) = -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) / d;
  m_inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / d;
  m_inv(1, 2) = -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) / d;

  m_inv(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) / d;
  m_inv(2, 1) = -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0)) / d;
  m_inv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) / d;
}

template <>
inline void inv(const sMat< 3, 3, double > &m, sMat< 3, 3, double > &m_inv) 
{
  // compute determinant
  const double d = det(m);
  assert(d != static_cast< double >(0));

  m_inv(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / d;
  m_inv(0, 1) = -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)) / d;
  m_inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / d;

  m_inv(1, 0) = -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) / d;
  m_inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / d;
  m_inv(1, 2) = -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) / d;

  m_inv(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) / d;
  m_inv(2, 1) = -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0)) / d;
  m_inv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) / d;
}

template < size_t M, class ScalarType >
inline sMat< M, M, ScalarType> 
inv(const sMat< M, M, ScalarType> &m)
{
  sMat< M, M, ScalarType> m_inv;
  inv(m, m_inv);
  return m_inv;
}

/// \brief Inverse-Transpose of 1x1 matrix.

template < class DataType >
inline void invTransp(const sMat< 1, 1, DataType > &m,
                      sMat< 1, 1, DataType > &m_inv) {
  m_inv(0, 0) = 1. / m(0, 0);
}

/// \brief Inverse-Transpose of 2x2 matrix.

template < class DataType >
inline void invTransp(const sMat< 2, 2, DataType > &m,
                      sMat< 2, 2, DataType > &m_inv) {
  typename sMat< 2, 2, DataType >::value_type d = det(m);
  assert(d != 0.);
  m_inv(0, 0) = m(1, 1) / d;
  m_inv(0, 1) = -m(1, 0) / d;
  m_inv(1, 0) = -m(0, 1) / d;
  m_inv(1, 1) = m(0, 0) / d;
}

/// \brief Inverse-Transpose of 3x3 matrix.

template < class DataType >
inline void invTransp(const sMat< 3, 3, DataType > &m,
                      sMat< 3, 3, DataType > &m_inv) {
  // copy into  inverse matrix
  // compute determinant
  const DataType d = det(m);
  assert(d != static_cast< DataType >(0));

  m_inv(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / d;
  m_inv(0, 1) = -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) / d;
  m_inv(0, 2) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) / d;

  m_inv(1, 0) = -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)) / d;
  m_inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / d;
  m_inv(1, 2) = -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0)) / d;

  m_inv(2, 0) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / d;
  m_inv(2, 1) = -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) / d;
  m_inv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) / d;
}

// Pseudo Inverse A^# = (A^T * A)^-1 * A^T 
// for injective A
template < size_t M, size_t N, class DataType >
inline void invPseudoInj(const sMat< M, N, DataType > &m,
                         sMat< N, M, DataType > &m_inv) 
{
  assert (N <= M);
  sMat< N, M, DataType> mT;
  trans(m, mT);

  const sMat<N, N, DataType> mTm = mT * m;

  sMat<N, N, DataType> mTm_inv;
  inv (mTm, mTm_inv);

  m_inv = mTm_inv * mT;
}

template < size_t M, size_t N, class ScalarType >
inline sMat< N, M, ScalarType> 
invPseudoInj(const sMat< M, N, ScalarType> &m)
{
  sMat< N, M, ScalarType> m_inv;
  invPseudoInj(m, m_inv);
  return m_inv;
}


// Pseudo Inverse A^# = A^T * (A * A^T)^-1 
// for surjective A
template < size_t M, size_t N, class DataType >
inline void invPseudoSur(const sMat< M, N, DataType > &m,
                         sMat< N, M, DataType > &m_inv) 
{
  static_assert (M <= N);
  sMat< N, M, DataType> mT;
  trans(m, mT);

  const sMat<M, M, DataType> mmT = m * mT;

  sMat<M, M, DataType> mmT_inv;
  inv (mmT, mmT_inv);

  m_inv = mT * mmT_inv;
}

template < size_t M, size_t N, class ScalarType >
inline sMat< N, M, ScalarType> 
invPseudoSur(const sMat< M, N, ScalarType> &m)
{
  sMat< N, M, ScalarType> m_inv;
  invPseudoSur(m, m_inv);
  return m_inv;
}

/// \brief Gaussian elimination of a linear system.

template < size_t M, size_t N, class DataType >
inline bool gauss(sMat< M, N, DataType > &mat, sVec< M, DataType > &vec) {
  // Gaussian elimination with pivoting of a linear system of equations
  // transforms the given sMatrix and vector
  // if the submatrix m[1:M,1:M] is regular, the solution of m[1:M,1:M] x = v
  // ist stored into vec
  // if the matrix consists of m = [m[1:M,1:M] Id_M], the invers of m
  // is stored in the second half of the matrix

  // the current version needs to get equal or more columns than rows
  assert(N >= M);

  for (int i = 0; i < M; ++i) {
    // find pivot row
    int pivot = i;
    for (int p = i; p < M; ++p) {
      if (std::abs(mat(pivot, i)) < std::abs(mat(p, i))) {
        pivot = p;
      }
    }
    // check if system is solvable
    if (std::abs(mat(pivot, i)) < COMPARISON_TOL) {
      return false;
    }
    // swap rows
    if (pivot != i) {
      for (int n = i; n < N; ++n) {
        std::swap(mat(pivot, n), mat(i, n));
      }
      std::swap(vec[pivot], vec[i]);
    }
    // reduce
    vec[i] /= mat(i, i);
    for (int  n = N - 1; n >= i; --n) {
      mat(i, n) /= mat(i, i);
    }
    // elimination forwards
    for (int m = i + 1; m < M; ++m) {
      vec[m] -= vec[i] * mat(m, i);
      for (int n = N - 1; n >= i; --n) {
        mat(m, n) -= mat(i, n) * mat(m, i);
      }
    }
  }

  // elimination backwards
  for (int i = M - 1; i > 0; --i) {
    for (int m = static_cast< int >(i - 1); m >= 0; --m) {
      vec[m] -= vec[i] * mat(m, i);
      for (int n = N - 1; n >= i; --n) {
        mat(m, n) -= mat(i, n) * mat(m, i);
      }
    }
  }
  return true;
}

/// \brief Transpose of a general matrix.

template < size_t M, size_t N, class DataType >
inline void trans(const sMat< M, N, DataType > &m,
                  sMat< N, M, DataType > &m_trans) {
PRAGMA_LOOP_VEC
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      m_trans(i, j) = m(j, i);
    }
  }
}

template < size_t M, size_t N, class ScalarType >
inline sMat< N, M, ScalarType> 
trans(const sMat< M, N, ScalarType> &m)
{
  sMat< N, M, ScalarType> m_trans;
  trans(m, m_trans);
  return m_trans;
}

/// \brief Computes the Frobenius product of two quadratic matrices.

template < size_t N, class DataType >
inline DataType frob(const sMat< N, N, DataType > &m1,
                     const sMat< N, N, DataType > &m2) {

  DataType res = 0.0;
  assert(N > 0);
  for (size_t i = 0; i < N; ++i) {
//PRAGMA_LOOP_VEC
    for (size_t k = 0; k < N; ++k) {
      res += m1(i, k) * m2(i, k);
    }
  }
  return res;
}

template < size_t N, class DataType >
inline DataType frob(const sMat< N, N, DataType > &m1) 
{
  return std::sqrt(frob(m1,m1));
}
/// \brief Computes the sum of all absolute matrix entries 

template < size_t N, size_t M, class DataType >
inline DataType abs(const sMat< M, N, DataType > &m) {

  DataType res = 0.0;
  assert(N > 0);
  assert(M > 0); 
  for (size_t i = 0; i != M; ++i) {
//PRAGMA_LOOP_VEC
    for (size_t k = 0; k != N; ++k) {
      res += std::abs(m(i, k));
    }
  }
  return res;
}

/// \brief Trace of a quadratic matrix.

template < size_t M, class DataType >
inline DataType trace(const sMat< M, M, DataType > &m) {
  DataType trace = 0;
//PRAGMA_LOOP_VEC
  for (size_t i = 0; i < M; ++i) {
    trace += m(i, i);
  }
  return trace;
}

/// \brief Output operator for a general matrix.

template < size_t M, size_t N, class DataType >
std::ostream &operator<<(std::ostream &os, const sMat< M, N, DataType > &m) {
  os << "[";
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      os << " " << m(i, j);
    }
    if (i < M - 1) {
      os << ";";
    }
  }
  os << " ]";
  return os;
}

} // namespace hiflow

#endif /* _VECTOR_ALGEBRA_H_ */
