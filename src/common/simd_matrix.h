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

#ifndef HIFLOW_SIMD_MATRIX_ALGEBRA_H
#define HIFLOW_SIMD_MATRIX_ALGEBRA_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include "common/macros.h"
#include "common/log.h"
#include "common/simd_types.h"
#include "common/simd_functions.h"
#include "common/simd_vector.h"


/// @brief This file contains template classes for representing small
/// floating-point vectors and matrices with sizes fixed at
/// compile-time; as well as common mathematical operations for these
/// objects.

/// @author Philipp Gerstner

namespace {

} // namespace

namespace hiflow 
{

/// \brief Class representing a floating-point vector of size N.
///
/// \details The class also supports common mathematical operations.

template < size_t M, size_t N, class ScalarType, 
           bool force_scalar = false, 
           typename SimdType = typename VCLType<ScalarType, force_scalar>::vec_type > 
class pMat 
{
  using NVEC = pVec<N, ScalarType, force_scalar, SimdType>;
  using MVEC = pVec<M, ScalarType, force_scalar, SimdType>;

public:
  // number of scalar values contained in single SIMD type
  static constexpr size_t L = SIMDInfo<SimdType>::NumScalar;

  static constexpr bool is_scalar = (L == 1);

  // number of simd-variables need for one column
  static constexpr size_t K = compute_nb_vector (M, L);

  // remainder of integer division N / P
  static constexpr size_t R = compute_remainder (M, L);


  pMat() 
  {
    this->Zeros();
  }

  pMat(const pMat &m) 
  {
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      this->m_[k] = m.m_[k];
    }
  }

  explicit pMat(const std::vector<ScalarType>& values, bool input_is_col_major = false) 
  {
    this->Zeros();
    if (input_is_col_major)
    {
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t m = 0; m < M; ++m)
        {
          this->set(m,n, values[n*M+m]);
        }  
      } 
    }
    else 
    {
      for (size_t m = 0; m < M; ++m)
      {
        for (size_t n = 0; n < N; ++n)
        {
          this->set(m,n, values[m*N+n]);
        }  
      } 
    }
  }

  explicit pMat(const ScalarType *values, bool input_is_col_major = false) 
  {
    this->Zeros();
    if (input_is_col_major)
    {
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t m = 0; m < M; ++m)
        {
          this->set(m,n, values[n*M+m]);
        }  
      } 
    }
    else 
    {
      for (size_t m = 0; m < M; ++m)
      {
        for (size_t n = 0; n < N; ++n)
        {
          this->set(m,n, values[m*N+n]);
        }  
      } 
    }
  }

  inline pMat &operator=(const pMat &m) 
  {
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      this->m_[k] = m.m_[k];
    }
    return *this;
  }

  inline void Zeros() 
  {
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      this->m_[k] = ScalarType(0.);
    }
  }

  // Element access

  inline ScalarType operator()(size_t i, size_t j) const 
  {
    // prevent i < 0 with i unsigned warning
    assert(j < N);
    assert(i < M);

    if constexpr (is_scalar)
    {
      return m_[j * M + i];
    }
    else 
    {
      const size_t k = i / L;
      const size_t l = i % L;
      return m_[j * K + k].extract(l);
    }
  }

  inline void set (const size_t i, const size_t j, const ScalarType val)
  {
    // prevent i < 0 with i unsigned warning
    assert(j < N);
    assert(i < M);

    if constexpr (is_scalar)
    {
      this->m_[j * M + i] = val;
    }
    else 
    {
      const size_t k = i / L;
      const size_t l = i % L;
      this->m_[j * K + k].insert(l, val);
    }
  }

  inline void add (const size_t i, const size_t j, const ScalarType val)
  {
    // prevent i < 0 with i unsigned warning
    assert(j < N);
    assert(i < M);

    if constexpr (is_scalar)
    {
      this->m_[j * M + i] += val;
    }
    else 
    {
      const size_t k = i / L;
      const size_t l = i % L;
      const ScalarType tmp = this->m_[j * K + k][l];
      this->m_[j * K + k].insert(l, tmp+val);
    }
  }

  /* only works for scalar
  inline ScalarType &operator()(size_t i, size_t j) {
    // prevent i < 0 with i unsigned warning
    assert(j < N);
    assert(i < M);

    return m_[i * N + j];
  }
  */ 

  constexpr inline size_t num_row() const
  {
    return M;
  }

  constexpr inline size_t num_col() const 
  {
    return N;
  }

  // Multiplication by scalar
  template <typename DataType>
  inline pMat &operator*=(const DataType s) 
  {
    const SimdType vs = static_cast<ScalarType>(s);
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      m_[k] *= vs;
    }
    return *this;
  }

  // Division by scalar
  template <typename DataType>
  inline pMat &operator/=(const DataType s) 
  {
    const SimdType vs = static_cast<ScalarType>(s);
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      m_[k] /= vs;
    }
    return *this;
  }

  // pMatrix addition
  inline pMat &operator+=(const pMat &m) 
  {
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      this->m_[k] += m.m_[k];
    }
    return *this;
  }

  // pMatrix subtraction
  inline pMat &operator-=(const pMat &m) 
  {
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      this->m_[k] -= m.m_[k];
    }
    return *this;
  }

  // pMatrix multiplication with square matrix
  inline pMat &operator*=(const pMat< N, N, ScalarType, force_scalar, SimdType > &m) 
  {
    // copy values of this to new array
    SimdType cpy[K * N];
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      cpy[k] = this->m_[k];
      this->m_[k] = 0;
    }
    
    // perform matrix-matrix multiplication
    for (size_t n=0; n < N; ++n)
    {
      const size_t n_offset = n * K;
      for (size_t nn=0; nn < N; ++nn)
      {
        const size_t nn_offset = nn * K;
        const SimdType rhs = m(nn,n);
        for (size_t k = 0; k != K; ++k)
        {
          this->m_[n_offset+k] = mul_add(cpy[nn_offset+k], rhs, this->m_[n_offset+k]);
        }
      } 
    }
    return *this;
  }

  // pMatrix comparison

  inline bool operator==(const pMat &m)
  {
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      for (size_t l = 0; l != L; ++l)
      {
        if constexpr (!is_scalar)
        {
          if (std::abs(m_[k][l] - m.m_[k][l]) > static_cast< ScalarType >(COMPARISON_TOL)) 
          {
            return false;
          }
        }
        else 
        {
          if (std::abs(m_[k] - m.m_[k]) > static_cast< ScalarType >(COMPARISON_TOL)) 
          {
            return false;
          }
        }
      }
    }
    return true;
  }

  // pMatrix comparison

  inline bool operator!=(const pMat &m) {
    return !(*this == m);
  }

  // Vector multiplication

  inline void VectorMult(const NVEC &in, MVEC &out) const 
  {
    for (size_t k = 0; k < K; ++k) 
    {
      out.v_[k] = ScalarType(0.);
    }

    for (size_t n=0; n < N; ++n)
    {
      const size_t n_offset = n * K;
      const SimdType rhs = in[n];

      for (size_t k = 0; k < K; ++k) 
      {
        out.v_[k] = mul_add(this->m_[n_offset+k], rhs, out.v_[k]);
      }
    }
  }

  // Add multiple of second matrix
  template <typename DataType>
  inline void Axpy(const pMat &mat, const DataType alpha) 
  {
    const SimdType av = static_cast<ScalarType>(alpha);
    for (size_t k = 0, e_k = K * N; k < e_k; ++k) 
    {
      this->m_[k] = mul_add(mat.m_[k], av, this->m_[k]);
    }
  }

  template< bool _force_scalar, typename _SimdType >
  inline void SetRow(int row, const pVec< N, ScalarType, _force_scalar, _SimdType > &row_vec) 
  {
    assert (row >= 0);
    assert (row < M);
    for (size_t i = 0; i < N; ++i) 
    {
      this->set(row, i, row_vec[i]);
    }
  }

  inline void SetCol(int col, const pVec< M, ScalarType, force_scalar, SimdType > &col_vec) 
  {
    assert (col >= 0);
    assert (col < N);

    const size_t offset = col * K;
    for (size_t k = 0; k < K; ++k) 
    {
      this->m_[offset + k] = col_vec.v_[k];
    }
  }

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend class pMat;

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType dot(const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &, 
                                const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType abs(const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _M, size_t _N, size_t _P, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline void MatrixMatrixMult(pMat< _M, _P, _ScalarType, _force_scalar, _SimdType > &,
                                      const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &,
                                      const pMat< _N, _P, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline pVec< _N, _ScalarType, _force_scalar, _SimdType > 
  operator*(const pVec< _M, _ScalarType, _force_scalar, _SimdType > &,
            const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType dot(const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &,
                                const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &); 

private:
  SimdType m_[K * N];
};

// Matrix-Matrix Multiplication
// out = mat1 * mat2
template < size_t M, size_t N, size_t P, class ScalarType, bool force_scalar, typename SimdType >
inline void MatrixMatrixMult(pMat< M, P, ScalarType, force_scalar, SimdType > &out,
                             const pMat< M, N, ScalarType, force_scalar, SimdType > &mat1,
                             const pMat< N, P, ScalarType, force_scalar, SimdType > &mat2) 
{
  out.Zeros();
  const size_t K = pMat< M, P, ScalarType, force_scalar, SimdType >::K;

  // loop over columns
  for (size_t p=0; p < P; ++p)
  {
    const size_t p_offset = p * K;

    // loop over rows
    for (size_t n=0; n < N; ++n)
    {
      const size_t n_offset = n * K;
      const SimdType rhs = mat2(n,p);
      for (size_t k = 0; k != K; ++k)
      {
        out.m_[p_offset+k] = mul_add(mat1.m_[n_offset+k], rhs, out.m_[p_offset+k]);
      }
    } 
  }
}

// Matrix-matrix multiplication with square matrix.
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pMat< M, N, ScalarType, force_scalar, SimdType > 
operator*(const pMat< M, N, ScalarType, force_scalar, SimdType > &m1,
          const pMat< N, N, ScalarType, force_scalar, SimdType > &m2) 
{
  pMat< M, N, ScalarType, force_scalar, SimdType > tmp(m1);
  tmp *= m2;
  return tmp;
}

// General matrix-matrix multiplication.
template < size_t M, size_t N, size_t P, class ScalarType, bool force_scalar, typename SimdType >
inline pMat< M, P, ScalarType, force_scalar, SimdType > 
operator*(const pMat< M, N, ScalarType, force_scalar, SimdType > &A,
          const pMat< N, P, ScalarType, force_scalar, SimdType > &B) 
{
  pMat< M, P, ScalarType, force_scalar, SimdType > C;
  MatrixMatrixMult(C, A, B);
  return C;
}

// Vector-matrix multiplication
// out = v * m
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pVec< N, ScalarType, force_scalar, SimdType > 
operator*(const pVec< M, ScalarType, force_scalar, SimdType > &v,
          const pMat< M, N, ScalarType, force_scalar, SimdType > &m) 
{
  pVec< N, ScalarType, force_scalar, SimdType > mv;
  constexpr size_t K = pMat< M, N, ScalarType, force_scalar, SimdType >::K;

  // loop over columns
  for (size_t n = 0; n < N; ++n) 
  {
    const size_t mat_offset = n * K;
    const ScalarType res = dot<M, K, SimdType>(v.v_, m.m_, 0, mat_offset);
    mv.set(n, res);
  }
  return mv;
}


// Binary matrix operations (do not modify the arguments).

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType, typename DataType >
inline pMat< M, N, ScalarType, force_scalar, SimdType >
operator*(const pMat< M, N, ScalarType, force_scalar, SimdType > &m,
          const DataType s) 
{
  pMat< M, N, ScalarType, force_scalar, SimdType > tmp(m);
  tmp *= s;
  return tmp;
}

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType, typename DataType >
inline pMat< M, N, ScalarType, force_scalar, SimdType >
operator*(const DataType s,
          const pMat< M, N, ScalarType, force_scalar, SimdType > &m) 
{
  pMat< M, N, ScalarType, force_scalar, SimdType > tmp(m);
  tmp *= s;
  return tmp;
}

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType, typename DataType >
inline pMat< M, N, ScalarType, force_scalar, SimdType >
operator/(const pMat< M, N, ScalarType, force_scalar, SimdType > &m,
          DataType s) 
{
  pMat< M, N, ScalarType, force_scalar, SimdType > tmp(m);
  tmp /= s;
  return tmp;
}

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pMat< M, N, ScalarType, force_scalar, SimdType > 
operator+(const pMat< M, N, ScalarType, force_scalar, SimdType > &m1,
          const pMat< M, N, ScalarType, force_scalar, SimdType > &m2) 
{
  pMat< M, N, ScalarType, force_scalar, SimdType > tmp(m1);
  tmp += m2;
  return tmp;
}

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pMat< M, N, ScalarType, force_scalar, SimdType > 
operator-(const pMat< M, N, ScalarType, force_scalar, SimdType > &m1,
          const pMat< M, N, ScalarType, force_scalar, SimdType > &m2) 
{
  pMat< M, N, ScalarType, force_scalar, SimdType > tmp(m1);
  tmp -= m2;
  return tmp;
}

// Matrix-vector multiplication
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pVec< M, ScalarType, force_scalar, SimdType > 
operator*(const pMat< M, N, ScalarType, force_scalar, SimdType > &m,
          const pVec< N, ScalarType, force_scalar, SimdType > &v) 
{
  pVec< M, ScalarType, force_scalar, SimdType > mv;
  m.VectorMult(v, mv);
  return mv;
}

/// \brief Transpose of a general matrix.
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline void trans(const pMat< M, N, ScalarType, force_scalar, SimdType > &m,
                  pMat< N, M, ScalarType, force_scalar, SimdType > &m_trans) 
{
  for (size_t i = 0; i < N; ++i) 
  {
    for (size_t j = 0; j < M; ++j) 
    {
      m_trans.set(i, j, m(j, i));
    }
  }
}

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pMat< N, M, ScalarType, force_scalar, SimdType > 
trans(const pMat< M, N, ScalarType, force_scalar, SimdType > &m)
{
  pMat< N, M, ScalarType, force_scalar, SimdType > m_trans;
  trans(m, m_trans);
  return m_trans;
}

// Determinant for 1x1, 2x2, 3x3 matrix
template < size_t M, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType det(const pMat< M, M, ScalarType, force_scalar, SimdType > &m)
{
  static_assert( M <= 3);

  if constexpr (M == 1)
  {
    return m(0,0);
  }
  else if constexpr (M==2)
  {
    return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
  }
  else if constexpr (M==3)
  {
    return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
           m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
           m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
  }
}

template < size_t M, class ScalarType, bool force_scalar, typename SimdType >
inline void inv(const pMat< M, M, ScalarType, force_scalar, SimdType > &m,
                pMat< M, M, ScalarType, force_scalar, SimdType > &m_inv)
{
  static_assert (M <= 3);
  if constexpr (M == 1)
  {
    m_inv.set(0, 0, 1. / m(0, 0)); 
  }
  else if constexpr (M==2)
  {
    auto d = det(m);
    assert(d != 0.);
    m_inv.set(0, 0, m(1, 1) / d);
    m_inv.set(0, 1, -m(0, 1) / d);
    m_inv.set(1, 0, -m(1, 0) / d);
    m_inv.set(1, 1, m(0, 0) / d);
  }
  else if constexpr (M==3)
  {
    auto d = det(m);
    assert(d != 0.);

    m_inv.set(0, 0,  (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / d);
    m_inv.set(0, 1, -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)) / d);
    m_inv.set(0, 2,  (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / d);
    m_inv.set(1, 0, -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) / d);
    m_inv.set(1, 1,  (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / d);
    m_inv.set(1, 2, -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) / d);
    m_inv.set(2, 0,  (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) / d);
    m_inv.set(2, 1, -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0)) / d);
    m_inv.set(2, 2,  (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) / d);
  }
}

template < size_t M, class ScalarType, bool force_scalar, typename SimdType >
inline pMat< M, M, ScalarType, force_scalar, SimdType > 
inv(const pMat< M, M, ScalarType, force_scalar, SimdType > &m)
{
  pMat< M, M, ScalarType, force_scalar, SimdType > m_inv;
  inv(m, m_inv);
  return m_inv;
}

template < size_t M, class ScalarType, bool force_scalar, typename SimdType >
inline void invTransp(const pMat< M, M, ScalarType, force_scalar, SimdType > &m,
                      pMat< M, M, ScalarType, force_scalar, SimdType > &m_inv)
{
  static_assert (M <= 3);

  if constexpr (M == 1)
  {
    m_inv.set(0, 0, 1. / m(0, 0)); 
  }
  else if constexpr (M==2)
  {
    auto d = det(m);
    assert(d != 0.);
    m_inv.set(0, 0,  m(1, 1) / d);
    m_inv.set(0, 1, -m(1, 0) / d);
    m_inv.set(1, 0, -m(0, 1) / d);
    m_inv.set(1, 1,  m(0, 0) / d);
  }
  else if constexpr (M==3)
  {
    auto d = det(m);
    assert(d != 0.);

    m_inv.set(0, 0,  (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / d);
    m_inv.set(0, 1, -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) / d);
    m_inv.set(0, 2,  (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) / d);
    m_inv.set(1, 0, -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)) / d);
    m_inv.set(1, 1,  (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / d);
    m_inv.set(1, 2, -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0)) / d);
    m_inv.set(2, 0,  (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / d);
    m_inv.set(2, 1, -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) / d);
    m_inv.set(2, 2,  (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) / d);
  } 
}

// Pseudo Inverse A^# = (A^T * A)^-1 * A^T 
// for injective A
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline void invPseudoInj(const pMat< M, N, ScalarType, force_scalar, SimdType > &m,
                         pMat< N, M, ScalarType, force_scalar, SimdType > &m_inv)
{
  static_assert (N <= M);
  pMat< N, M, ScalarType, force_scalar, SimdType > mT;
  trans(m, mT);

  const pMat< N, N, ScalarType, force_scalar, SimdType > mTm = mT * m;

  pMat< N, N, ScalarType, force_scalar, SimdType > mTm_inv;
  inv (mTm, mTm_inv);

  m_inv = mTm_inv * mT;
}

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pMat< N, M, ScalarType, force_scalar, SimdType > 
invPseudoInj(const pMat< M, N, ScalarType, force_scalar, SimdType > &m)
{
  pMat< N, M, ScalarType, force_scalar, SimdType > m_invPseudoInj;
  invPseudoInj(m, m_invPseudoInj);
  return m_invPseudoInj;
}

// Pseudo Inverse A^# = A^T * (A * A^T)^-1 
// for surjective A
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline void invPseudoSur(const pMat< M, N, ScalarType, force_scalar, SimdType > &m,
                         pMat< N, M, ScalarType, force_scalar, SimdType > &m_inv)
{
  static_assert (M <= N);
  pMat< N, M, ScalarType, force_scalar, SimdType > mT;
  trans(m, mT);

  const pMat< M, M, ScalarType, force_scalar, SimdType > mmT = m * mT;

  pMat< M, M, ScalarType, force_scalar, SimdType > mmT_inv;
  inv (mmT, mmT_inv);

  m_inv = mT * mmT_inv;
}

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pMat< N, M, ScalarType, force_scalar, SimdType > 
invPseudoSur(const pMat< M, N, ScalarType, force_scalar, SimdType > &m)
{
  pMat< N, M, ScalarType, force_scalar, SimdType > m_invPseudoSur;
  invPseudoSur(m, m_invPseudoSur);
  return m_invPseudoSur;
}

// matrix-matrix inner product
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType dot(const pMat< M, N, ScalarType, force_scalar, SimdType > &m1,
                      const pMat< M, N, ScalarType, force_scalar, SimdType > &m2) 
{
  return dot<M * N, pMat< M, N, ScalarType, force_scalar, SimdType >::K * N, SimdType>(m1.m_, m2.m_, 0, 0);
}

// inner product of two vectors with product defined by matrix : res = v1^T M v2
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType inner(const pVec< M, ScalarType, force_scalar, SimdType > &v1, 
                        const pMat< M, N, ScalarType, force_scalar, SimdType > &m,
                        const pVec< N, ScalarType, force_scalar, SimdType > &v2) 
{
  pVec< M, ScalarType, force_scalar, SimdType > tmp;
  m.VectorMult(v2, tmp);
  return dot<M, pVec< M, ScalarType, force_scalar, SimdType >::K, SimdType>(v1.v_, tmp.v_, 0, 0);
}

// inner product of two vectors with product defined by transposed matrix : res = v2^T M v1
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType innerT(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                        const pMat< M, N, ScalarType, force_scalar, SimdType > &m,
                        const pVec< M, ScalarType, force_scalar, SimdType > &v2) 
{
  pVec< M, ScalarType, force_scalar, SimdType > tmp;
  m.VectorMult(v1, tmp);
  return dot<M, pVec< M, ScalarType, force_scalar, SimdType >::K, SimdType>(v2.v_, tmp.v_, 0, 0);
}

// frobenius norm of matrix
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType frob(const pMat< M, N, ScalarType, force_scalar, SimdType > &m) 
{
  return std::sqrt(dot(m,m));
}

/// \brief Trace of a quadratic matrix.
template < size_t M, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType trace(const pMat< M, M, ScalarType, force_scalar, SimdType > &m)
{
  ScalarType trace = 0;
  for (size_t i = 0; i < M; ++i) 
  {
    trace += m(i, i);
  }
  return trace;
}

/// \brief Computes the sum of all absolute matrix entries 

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType abs(const pMat< M, N, ScalarType, force_scalar, SimdType > &m)
{
  ScalarType res = 0.;
  for (size_t k = 0, e_k = pMat< M, N, ScalarType, force_scalar, SimdType >::K * N; k != e_k; ++k) 
  {
    SimdType tmp = abs(m.m_[k]);
    res += horizontal_add(tmp);
  }
  return res;
}


/// \brief Output operator for a general matrix.
template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType >
std::ostream &operator<<(std::ostream &os, pMat< M, N, ScalarType, force_scalar, SimdType > &m) 
{
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


/// \brief Gaussian elimination of a linear system.
/*
template < size_t M, size_t N, class DataType >
inline bool gauss(Mat< M, N, DataType > &mat, Vec< M, DataType > &vec) {
  // Gaussian elimination with pivoting of a linear system of equations
  // transforms the given Matrix and vector
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
*/

/* not needed?
// Matrix minor computation (helper function for det and inv of 3x3 matrix)

template < class DataType >
inline typename Mat< 3, 3, DataType >::value_type
matrix_minor(const Mat< 3, 3, DataType > &m, size_t i, size_t j) {
  const int indI0 = (i + 1) % 3;
  const int indI1 = (i + 2) % 3;
  const int indJ0 = (j + 1) % 3;
  const int indJ1 = (j + 2) % 3;
  return m(indI0, indJ0) * m(indI1, indJ1) - m(indI0, indJ1) * m(indI1, indJ0);
}

// Sign of sum of two integers.

template < class DataType >
inline typename Mat< 3, 3, DataType >::value_type sign(size_t i, size_t j) {
  return (i + j) % 2 == 0 ? 1. : -1.;
}
*/




} // namespace hiflow

#endif /* _VECTOR_ALGEBRA_H_ */
