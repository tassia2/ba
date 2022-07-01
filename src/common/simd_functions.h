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

#ifndef __COMMON_SIMD_FUNCTIONS_H
#define __COMMON_SIMD_FUNCTIONS_H

#include <cmath>
#include "simd_types.h"

namespace hiflow {
/// @author Philipp Gerstner

// for compatibility reasons with VCL functions
inline float horizontal_add(const float val)
{
  return val;
} 
inline double horizontal_add(const double val)
{
  return val;
}
inline int horizontal_add(const int val)
{
  return val;
}

inline float mul_add(const float a, const float b, const float c)
{
  return c + b * a;
}
inline double mul_add(const double a, const double b, const double c)
{
  return c + b * a;
}
inline int mul_add(const int a, const int b, const int c)
{
  return c + b * a;
}

inline float abs(const float a)
{
  return std::abs(a);
}
inline double abs(const double a)
{
  return std::abs(a);
}
inline int abs(const int a)
{
  return std::abs(a);
}

inline double horizontal_max(const double val)
{
  return val;
}
inline float horizontal_max(const float val)
{
  return val;
}
inline int horizontal_max(const int val)
{
  return val;
}

inline double horizontal_min(const double val)
{
  return val;
}
inline float horizontal_min(const float val)
{
  return val;
}
inline int horizontal_min(const int val)
{
  return val;
}

// ************************************************
// load aligned 
inline void load_a (double& out, const double* in)
{
  out = *in;
}
inline void load_a (float& out, const float* in)
{
  out = *in;
}

#ifdef WITH_VCL 
inline void load_a (vcl::Vec16f& out, const float* in)
{
  out.load_a(in);
}
inline void load_a (vcl::Vec8f& out, const float* in)
{
  out.load_a(in);
}
inline void load_a (vcl::Vec4f& out, const float* in)
{
  out.load_a(in);
}
inline void load_a (vcl::Vec8d& out, const double* in)
{
  out.load_a(in);
}
inline void load_a (vcl::Vec4d& out, const double* in)
{
  out.load_a(in);
}
inline void load_a (vcl::Vec2d& out, const double* in)
{
  out.load_a(in);
}
#endif

// ************************************************
// store
inline void store (const double& in, double* out)
{
  *out = in;
}
inline void store (const float& in, float* out)
{
  *out = in;
}

#ifdef WITH_VCL 
inline void store (const vcl::Vec16f& in, float* out)
{
  in.store(out);
}
inline void store (const vcl::Vec8f& in, float* out)
{
  in.store(out);
}
inline void store (const vcl::Vec4f& in, float* out)
{
  in.store(out);
}
inline void store (const vcl::Vec8d& in, double* out)
{
  in.store(out);
}
inline void store (const vcl::Vec4d& in, double* out)
{
  in.store(out);
}
inline void store (const vcl::Vec2d& in, double* out)
{
  in.store(out);
}
#endif

// ************************************************
// store partial 
inline void store_partial (const double& in, const int r, double* out) 
{
  *out = in;
}
inline void store_partial (const float& in, const int r, float* out) 
{
  *out = in;
}

#ifdef WITH_VCL 
inline void store_partial (const vcl::Vec16f& in, const int r, float* out)
{
  in.store_partial(r, out);
}
inline void store_partial (const vcl::Vec8f& in, const int r, float* out)
{
  in.store_partial(r, out);
}
inline void store_partial (const vcl::Vec4f& in, const int r, float* out)
{
  in.store_partial(r, out);
}
inline void store_partial (const vcl::Vec8d& in, const int r, double* out)
{
  in.store_partial(r, out);
}
inline void store_partial (const vcl::Vec4d& in, const int r, double* out)
{
  in.store_partial(r, out);
}
inline void store_partial (const vcl::Vec2d& in, const int r, double* out)
{
  in.store_partial(r, out);
}
#endif

// dot product of two simd types
template < typename SimdType >
inline typename SIMDInfo<SimdType>::scalar_type 
dot(const SimdType& v1, 
    const SimdType& v2) 
{
  if constexpr (SIMDInfo<SimdType>::NumScalar == 1)
  {
    return v1 * v2; 
  }
  else if constexpr (SIMDInfo<SimdType>::NumScalar == 2)
  {
    SimdType tmp = v1 * v2;
    return tmp.extract(0) + tmp.extract(1);
  }
  else 
  {
    SimdType tmp = v1 * v2;
    return horizontal_add(tmp);
  }
}

// N = total number of scalar values 
// K = number of simd values 
// L = number of scalar values per Simd type
// (K-1) * L < N <= K * L 

template < size_t N, size_t K, typename SimdType >
inline typename SIMDInfo<SimdType>::scalar_type dot(const SimdType* v1, 
                                                    const SimdType* v2,
                                                    const size_t offset1,
                                                    const size_t offset2) 
{
  //std::cout << N << " " << K << std::endl;
  static_assert(N <= K * SIMDInfo<SimdType>::NumScalar);
  //static_assert(N > (K-1) * SIMDInfo<SimdType>::NumScalar);
  static_assert(K >= 1);

  if constexpr (N == 1)
  {
    if constexpr (SIMDInfo<SimdType>::NumScalar == 1)
    {
      return v1[offset1] * v2[offset2]; 
    }
    else 
    {
      return v1[offset1].extract(0) * v2[offset2].extract(0); 
    }
  }
  else if constexpr (N == 2)
  {
    if constexpr (SIMDInfo<SimdType>::NumScalar >= 2)
    {
      SimdType tmp = v1[offset1] * v2[offset2];
      return tmp.extract(0) + tmp.extract(1);
    }
    else 
    {
      typename SIMDInfo<SimdType>::scalar_type res = 0.;
      for (int i = 0; i != N; ++i)
      {
        res += v1[offset1+i] * v2[offset2+i];
      }
      return res;
    } 
  }
  else if constexpr (N == 3)
  {
    if constexpr (SIMDInfo<SimdType>::NumScalar >= 3)
    {
      SimdType tmp = v1[offset1] * v2[offset2];
      return horizontal_add(tmp);//(tmp.extract(0) + tmp.extract(1)) + tmp.extract(2);
    }
    else if constexpr (SIMDInfo<SimdType>::NumScalar == 2)
    {
      SimdType tmp0 = v1[offset1] * v2[offset2];
      SimdType tmp1 = v1[offset1+1] * v2[offset2+1];
      return (tmp0.extract(0) + tmp0.extract(1)) + tmp1.extract(0);
    }
    else 
    {
      typename SIMDInfo<SimdType>::scalar_type res = 0.;
      for (int i = 0; i != N; ++i)
      {
        res += v1[offset1+i] * v2[offset2+i];
      }
      return res;
    }
  }
  else 
  {
    typename SIMDInfo<SimdType>::scalar_type res = 0.;
    for (size_t k = 0; k != K; ++k) 
    {
      SimdType tmp = v1[offset1+k] * v2[offset2+k];
      res += horizontal_add(tmp);
      //std::cout << k << ":    " << res << " " << v1 << " <-> " << v2 << std::endl;
    }
    return res;
  }
}


}
#endif