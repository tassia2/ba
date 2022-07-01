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

#ifndef HIFLOW_SIMD_VECTOR_ALGEBRA_H
#define HIFLOW_SIMD_VECTOR_ALGEBRA_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include "common/macros.h"
#include "common/log.h"
#include "common/simd_types.h"
#include "common/simd_functions.h"

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

template < size_t M, size_t N, class ScalarType, bool force_scalar, typename SimdType > class pMat;

template < size_t N, class ScalarType, 
           bool force_scalar = false, 
           typename SimdType = typename VCLType<ScalarType, force_scalar>::vec_type > 
class pVec 
{

public:
  // number of scalar values contained in single SIMD type
  static constexpr size_t L = SIMDInfo<SimdType>::NumScalar;

  static constexpr bool is_scalar = (L == 1);

  // number of simd-variables need for one  vector
  static constexpr size_t K = compute_nb_vector (N, L);

  // remainder of integer division N / P
  static constexpr size_t R = compute_remainder (N, L);

  // Constructors

  pVec() 
  {
    this->Zeros();
  }

  pVec(const pVec &v) 
  {
    for (size_t k = 0; k < K; ++k)  
    {
      this->v_[k] = v.v_[k];
    }
  }

  pVec(const sVec< N, ScalarType > &v) 
  {
    this->Zeros();
    for (size_t i = 0; i < N; ++i)  
    {
      this->set(i, v[i]);
    }
  }

  explicit pVec(const ScalarType *values) 
  {
    this->Zeros();
    for (size_t i = 0; i < N; ++i)  
    {
      this->set(i, values[i]);
    }
  }

  explicit pVec(const std::vector< ScalarType >& values) 
  {
    assert(values.size() <= N);
    for (size_t i = 0; i < N; ++i) {
      if (values.size() > i) {
        this->set(i, values[i]);
      } else {
        this->set(i, 0.);
      }
    }
  }

  explicit pVec(const std::vector< ScalarType >& values, const size_t begin) 
  {
    for (size_t i = 0; i < N; ++i) {
      if (values.size() > i) {
        this->set(i, values[begin+i]);
      } else {
        this->set(i, 0.);
      }
    }
  }

  inline void Zeros() 
  {
    for (size_t k = 0; k != K; ++k) 
    {
      this->v_[k] = ScalarType(0.);
    }
  }

  // Size
  constexpr inline size_t size() const
  { 
    return N; 
  }

  // Assignment

  pVec &operator=(const pVec &v) 
  {
    for (size_t k = 0; k < K; ++k)  
    {
      this->v_[k] = v.v_[k];
    }
    return *this;
  }

  pVec &operator=(const sVec< N, ScalarType > &v) 
  {
    for (size_t i = 0; i < N; ++i)  
    {
      this->set(i, v[i]);
    }
    return *this;
  }

  template < size_t _N >
  pVec &operator=(const pVec< _N, ScalarType, force_scalar, SimdType > &v) 
  {
    constexpr size_t len = std::min(pVec::L, pVec< _N, ScalarType, force_scalar, SimdType >::L);

    for (size_t k = 0; k < len; ++k) 
    {
      this->v_[k] = v.v_[k];
    }
    return *this;
  }

  // Access operator
  inline ScalarType operator[](const size_t i) const 
  {
    // prevent i < 0 with i unsigned warning
    assert(i < N);

    if constexpr (is_scalar)
    {
      return this->v_[i];
    }
    else 
    {
      const size_t k = i / L;
      const size_t l = i % L;
      return v_[k].extract(l);
    }
  }

  /* only works for scalar
  inline ScalarType &operator[](size_t i) 
  {
    assert(is_scalar);
    
    // prevent i < 0 with i unsigned warning
    assert(i < N);
    
    if constexpr (is_scalar)
    {
      return v_[i];
    }
    else 
    {
      return 
    }
  }

  operator std::vector< ScalarType >() const {
    return std::vector< ScalarType >(&v_[0], &v_[N]);
  }

  ScalarType const * begin() const {
    return &v_[0];
  }

  ScalarType const * end() const {
    return &v_[N];
  }
*/

  inline void set (const size_t i, const ScalarType val)
  {
    // prevent i < 0 with i unsigned warning
    assert(i < N);

    if constexpr (is_scalar)
    {
      this->v_[i] = val;
    }
    else 
    {
      const size_t k = i / L;
      const size_t l = i % L;
      this->v_[k].insert(l, val);
    }
  }

  inline void add (const size_t i, const ScalarType val)
  {
    // prevent i < 0 with i unsigned warning
    assert(i < N);

    if constexpr (is_scalar)
    {
      this->v_[i] += val;
    }
    else 
    {
      const size_t k = i / L;
      const size_t l = i % L;
      const ScalarType tmp = this->v_[k][l];
      this->v_[k].insert(l, tmp+val);
    }
  }

  // lexicographic comparison
  inline bool operator< (const pVec& rhs) const 
  {
    for (size_t i = 0; i < N; ++i) 
    {
      if (operator[](i) < rhs[i])
      {
        return true;
      }
    }
    return false;
  }

  inline bool operator<= (const pVec& rhs) const 
  {
    if ((*this)<rhs)
    {
      return true;
    }
    return ((*this)==rhs);
  }

  // Comparison
  // Comparison
  inline bool operator==(const pVec& v) const 
  {
    for (size_t k = 0; k != K; ++k) 
    {
      for (size_t l = 0; l != L; ++l)
      {
        if constexpr (!is_scalar)
        {
          if (std::abs(v_[k][l] - v.v_[k][l]) > static_cast< ScalarType >(COMPARISON_TOL)) 
          {
            return false;
          }
        }
        else 
        {
          if (std::abs(v_[k] - v.v_[k]) > static_cast< ScalarType >(COMPARISON_TOL)) 
          {
            return false;
          }
        }
      }
    }
    return true;
  }

  inline bool eq(const pVec& v, ScalarType eps) const 
  {
    for (size_t k = 0; k != K; ++k) 
    {
      for (size_t l = 0; l != L; ++l)
      {
        if constexpr (!is_scalar)
        {
          if (std::abs(v_[k][l] - v.v_[k][l]) > eps) 
          {
            return false;
          }
        }
        else 
        {
          if (std::abs(v_[k] - v.v_[k]) > eps) 
          {
            return false;
          }
        }
      }
    }
    return true;
  }
  
  inline bool operator!=(const pVec& v) const 
  {
    return  !(*this == v);
  }

  inline bool neq(const pVec& v, ScalarType eps) const 
  {
    return !(this->eq(v,eps));
  }

  // Multiplication by scalar
  template <typename DataType>
  inline pVec &operator*=(const DataType s) 
  {
    const SimdType vs = static_cast<ScalarType>(s);
    for (size_t k = 0; k != K; ++k) 
    {
      this->v_[k] *= vs;
    }
    return *this;
  }

  // Division by scalar
  template <typename DataType>
  inline pVec &operator/=(const DataType s) 
  {
    const SimdType vs = static_cast<ScalarType>(s);
    for (size_t k = 0; k != K; ++k) 
    {
      this->v_[k] /= vs;
    }
    return *this;
  }

  // Addition
  inline pVec &operator+=(const pVec& v) 
  {
    for (size_t k = 0; k != K; ++k) 
    {
      this->v_[k] += v.v_[k];
    }
    return *this;
  }

  // Subtraction
  inline pVec &operator-=(const pVec& v) 
  {
    for (size_t k = 0; k != K; ++k) 
    {
      this->v_[k] -= v.v_[k];
    }
    return *this;
  }

  // Add multiple of second vector
  template <typename DataType>
  inline void Axpy(const pVec &vec, const DataType alpha) 
  {
    const SimdType av = static_cast<ScalarType> (alpha);
    for (size_t k = 0; k != K; ++k)
    {
      this->v_[k] = mul_add(vec.v_[k], av, this->v_[k]);
    }
  }

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend class pMat;

  template < size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType dot(const pVec< _N, _ScalarType, _force_scalar, _SimdType > &, 
                                const pVec< _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType sum(const pVec< _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType max(const pVec< _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType min(const pVec< _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType norm1(const pVec< _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType norm(const pVec< _N, _ScalarType, _force_scalar, _SimdType > &, 
                                 const ScalarType);

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline pVec< _N, _ScalarType, _force_scalar, _SimdType > 
  operator*(const pVec< _M, _ScalarType, _force_scalar, _SimdType > &,
            const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &);

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType inner(const pVec< _M, _ScalarType, _force_scalar, _SimdType > &, 
                                  const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &,
                                  const pVec< _N, _ScalarType, _force_scalar, _SimdType > &); 

  template < size_t _M, size_t _N, class _ScalarType, bool _force_scalar, typename _SimdType >
  friend inline _ScalarType innerT(const pVec< _N, _ScalarType, _force_scalar, _SimdType > &, 
                                   const pMat< _M, _N, _ScalarType, _force_scalar, _SimdType > &,
                                   const pVec< _M, _ScalarType, _force_scalar, _SimdType > &); 

  private:
   SimdType v_[K]; 

};

// Binary vector operations (do not modify the arguments).
template < size_t N, class ScalarType, bool force_scalar, typename SimdType, typename DataType >
inline pVec< N, ScalarType, force_scalar, SimdType > operator*(const pVec< N, ScalarType, force_scalar, SimdType > &v,
                                                               const DataType s)
{
  pVec< N, ScalarType, force_scalar, SimdType > tmp(v);
  tmp *= s;
  return tmp;
}

template < size_t N, class ScalarType, bool force_scalar, typename SimdType, typename DataType >
inline pVec< N, ScalarType, force_scalar, SimdType > operator*(const DataType s,
                                                               const pVec< N, ScalarType, force_scalar, SimdType > &v)
{
  pVec< N, ScalarType, force_scalar, SimdType > tmp(v);
  tmp *= s;
  return tmp;
}

template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pVec< N, ScalarType, force_scalar, SimdType > operator+(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                                                               const pVec< N, ScalarType, force_scalar, SimdType > &v2)
{
  pVec< N, ScalarType, force_scalar, SimdType > tmp(v1);
  tmp += v2;
  return tmp;
}

template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pVec< N, ScalarType, force_scalar, SimdType > operator-(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                                                               const pVec< N, ScalarType, force_scalar, SimdType > &v2)
{
  pVec< N, ScalarType, force_scalar, SimdType > tmp(v1);
  tmp -= v2;
  return tmp;
}

/// \brief Computes the euclidean distance of two vectors.
template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType distance(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                           const pVec< N, ScalarType, force_scalar, SimdType > &v2)
{
  auto diff = v1 - v2;
  return norm(diff);
}

template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType distance(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                           const pVec< N, ScalarType, force_scalar, SimdType > &v2,
                           const ScalarType p)
{
  auto diff = v1 - v2;
  return norm(diff, p);
}

/// \brief Computes the scalar product of two vectors.

template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType dot(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                      const pVec< N, ScalarType, force_scalar, SimdType > &v2) 
{
  return dot<N, pVec< N, ScalarType, force_scalar, SimdType >::K, SimdType> (v1.v_, v2.v_, 0, 0);
}

/*
template < size_t N, class ScalarType >
inline typename pVec< N, ScalarType >::ScalarType dot(const pVec< N, ScalarType > &v1, 
                                                   const pVec< N, ScalarType > &v2,
                                                   const pVec< N, ScalarType > &v3) 
{
  typename pVec< N, ScalarType >::ScalarType res = typename pVec< N, ScalarType >::ScalarType();
  for (size_t i = 0; i < N; ++i) 
  {
    res += v1[i] * v2[i] * v3[i];
  }

  return res;
}
*/


/// \brief Computes the cross product of two 3d vectors.

template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pVec< N, ScalarType, force_scalar, SimdType > cross(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                                                           const pVec< N, ScalarType, force_scalar, SimdType > &v2) 
{
  if constexpr (N != 3)
  {
    assert (false);
    return pVec< N, ScalarType, force_scalar, SimdType >();
  }
  else 
  {
    pVec< N, ScalarType, force_scalar, SimdType > v3;
    v3.set(0, v1[1] * v2[2] - v1[2] * v2[1]);
    v3.set(1, v1[2] * v2[0] - v1[0] * v2[2]);
    v3.set(2, v1[0] * v2[1] - v1[1] * v2[0]);
    return v3;
  }
}

/// \brief Computes the sum of the vector components.
template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType sum(const pVec< N, ScalarType, force_scalar, SimdType > &v) 
{
  SimdType vres (v.v_[0]);

  for (size_t k = 1; k != pVec< N, ScalarType, force_scalar, SimdType >::K; ++k) 
  {
    vres += v.v_[k];
  }
  
  return horizontal_add(vres);
}

template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType max(const pVec< N, ScalarType, force_scalar, SimdType > &v) 
{
  ScalarType max_val = std::numeric_limits<ScalarType>::min();

  for (size_t k = 1; k != pVec< N, ScalarType, force_scalar, SimdType >::K; ++k) 
  {
    max_val = std::max(max_val, horizontal_max(v.v_[k]));
  }
  
  return max_val;
}

template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType min(const pVec< N, ScalarType, force_scalar, SimdType > &v) 
{
  ScalarType min_val = std::numeric_limits<ScalarType>::max();

  for (size_t k = 1; k != pVec< N, ScalarType, force_scalar, SimdType >::K; ++k) 
  {
    min_val = std::min( min_val, horizontal_min(v.v_[k]));
  }
  
  return min_val;
}

/// \brief Computes the Euclidean norm of a vector.
template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType norm(const pVec< N, ScalarType, force_scalar, SimdType > &v)
{ 
  ScalarType res = std::sqrt(dot(v, v));
  return res;
}

/// \brief Computes the L1 norm of a vector.
template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType norm1(const pVec< N, ScalarType, force_scalar, SimdType > &v)
{
  SimdType vres (abs(v.v_[0]));

  for (size_t k = 1; k != pVec< N, ScalarType, force_scalar, SimdType >::K; ++k) 
  {
    vres += abs(v.v_[k]);
  }
  
  return horizontal_add(vres);
}

template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline ScalarType norm(const pVec< N, ScalarType, force_scalar, SimdType > &v,
                       const ScalarType p)
{ 
  SimdType vres (pow(abs(v.v_[0]), p));

  for (size_t k = 1; k != pVec< N, ScalarType, force_scalar, SimdType >::K; ++k) 
  {
    vres += pow(abs(v.v_[k]), p);
  }
  return std::pow(horizontal_add(vres), 1. / p);
}


/// \brief Computes the normed normal of a 2d / 3d vector.
template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pVec< N, ScalarType, force_scalar, SimdType > normal(const pVec< N, ScalarType, force_scalar, SimdType > &v1)
{
  if constexpr (N == 2)
  {
    pVec< N, ScalarType, force_scalar, SimdType > v2;
    v2.set(0, -v1[1]);
    v2.set(1,  v1[0]);
    const ScalarType n = norm(v2);
    return v2 * (1. / n);
  }
  else 
  {
    assert (false);
    return pVec< N, ScalarType, force_scalar, SimdType >();
  }
}

/// \brief Computes the normed normal of a 2d / 3d vector.
template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline pVec< N, ScalarType, force_scalar, SimdType > normal(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                                                            const pVec< N, ScalarType, force_scalar, SimdType > &v2)
{
  if constexpr (N == 3)
  {
    pVec< N, ScalarType, force_scalar, SimdType > v3 = cross (v1, v2);
    const ScalarType n = norm(v3);
    return v3 * (1. / n);
  }
  else 
  {
    assert (false);
    return pVec< N, ScalarType, force_scalar, SimdType >();
  }
}

/// \brief Computes the normed normal of two 3d vectors.
/*
template < class ScalarType >
inline pVec< 3, ScalarType > normal(const pVec< 6, ScalarType > &v1v2) {
  ScalarType v1_array[3] = {v1v2[0], v1v2[1], v1v2[2]};
  ScalarType v2_array[3] = {v1v2[3], v1v2[4], v1v2[5]};
  pVec< 3, ScalarType > v1(v1_array);
  pVec< 3, ScalarType > v2(v2_array);
  pVec< 3, ScalarType > v3 = cross(v1, v2);
  ScalarType v3norm = norm(v3);
  v3 /= v3norm;
  return v3;
}
*/

/// \brief Checks if two 2d vectors are parallel
template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
inline bool is_parallel(const pVec< N, ScalarType, force_scalar, SimdType > &v1, 
                        const pVec< N, ScalarType, force_scalar, SimdType > &v2)
{
  const ScalarType EPS = 1.0e3 * std::numeric_limits<ScalarType>::epsilon();
  if constexpr (N == 2)
  {
    return std::abs(v1[0] * v2[1] - v1[1] * v2[0]) < EPS;
  }
  else if constexpr (N == 3)
  {
    pVec< N, ScalarType, force_scalar, SimdType > v3 = cross (v1, v2);
    return (norm1(v3) < EPS);
  }
}

/// \brief Output operator for vectors.
template < size_t N, class ScalarType, bool force_scalar, typename SimdType >
std::ostream &operator<<(std::ostream &os, const pVec< N, ScalarType, force_scalar, SimdType > &v)
{
  os << "[ ";
  for (size_t i = 0; i < N; ++i) {
    os << v[i] << " ";
  }
  os << "]";
  return os;
}

} // namespace hiflow

#endif /* _VECTOR_ALGEBRA_H_ */
