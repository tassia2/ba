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

#ifndef __COMMON_SIMD_TYPES_H
#define __COMMON_SIMD_TYPES_H


#define VCL_NAMESPACE vcl
#ifdef WITH_VCL
#include "vectorclass.h"
#endif

#include <immintrin.h>

#if ( defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64) || ( _M_IX86_FP > 1 ) )
#  define WITH_SEE  1 
#else 
#  define WITH_SEE  0
#endif

#if defined(__AVX__)
#  define WITH_AVX  1 
#else 
#  define WITH_AVX  0
#endif

#if defined(__AVX512__)
#  define WITH_AVX512  1 
#else 
#  define WITH_AVX512  0
#endif

#if defined(__FMA__)
#  define WITH_FMA  1 
#else 
#  define WITH_FMA  0
#endif

namespace hiflow {

/// @author Philipp Gerstner

enum class InstructionSet 
{
    None = 0,
    SSE2 = 1,
    SSE3 = 2,
    SSE4 = 3,
    AVX = 4,
    AVX2 = 5,
    AVX512 = 6
};

// ///////////////////////////////////////////////////////////////
// VCLType: struct for selecting appropriate VCL SIMD type
// ///////////////////////////////////////////////////////////////
template <typename ScalarType, bool force_scalar> struct VCLType;

#ifdef WITH_VCL
template <>
struct VCLType<float, false>
{
#if WITH_AVX512 == 1
    using vec_type = vcl::Vec16f;
#elif WITH_AVX == 1
    using vec_type = vcl::Vec8f;
#elif WITH_SSE == 1
    using vec_type = vcl::Vec4f;
#else
    using vec_type = float;
#endif
}; 

template <>
struct VCLType<float, true>
{
    using vec_type = float;
};

template <>
struct VCLType<double, false>
{
#if WITH_AVX512 == 1
    using vec_type = vcl::Vec8d;
#elif WITH_AVX == 1
    using vec_type = vcl::Vec4d;
#elif WITH_SSE == 1
    using vec_type = vcl::Vec2d;
#else
    using vec_type = double;
#endif
};

template <>
struct VCLType<double, true>
{
    using vec_type = double;
};


template <>
struct VCLType<int, true>
{
    using vec_type = int;
};

// TODO
template <>
struct VCLType<int, false>
{
    using vec_type = int;
};

#else 

template <>
struct VCLType<float, true>
{
    using vec_type = float;
}; 

template <>
struct VCLType<float, false>
{
    using vec_type = float;
};

template <>
struct VCLType<double, true>
{
    using vec_type = double;
};

template <>
struct VCLType<double, false>
{
    using vec_type = double;
};

template <>
struct VCLType<int, true>
{
    using vec_type = int;
};

template <>
struct VCLType<int, false>
{
    using vec_type = int;
};

#endif

// ///////////////////////////////////////////////////////////////
// SIMDInfo: struct for collecting data for respective SIMD types
// ///////////////////////////////////////////////////////////////
template <typename SimdType> struct SIMDInfo;

template <>
struct SIMDInfo<double> 
{
    using scalar_type = double;
    using intrinsic_type = double;
    static constexpr InstructionSet IS = InstructionSet::None;
    static constexpr int NumScalar = 1;
    static constexpr int NumBit = 64;
    static constexpr int NumByte = 8; 
    static constexpr int Alignment = 8;
}; 

template <>
struct SIMDInfo<float> 
{
    using scalar_type = float;
    using intrinsic_type = float;
    static constexpr InstructionSet IS = InstructionSet::None;
    static constexpr int NumScalar = 1;
    static constexpr int NumBit = 32;
    static constexpr int NumByte = 4; 
    static constexpr int Alignment = 8;
}; 

template <>
struct SIMDInfo<int> 
{
    using scalar_type = int;
    using intrinsic_type = int;
    static constexpr InstructionSet IS = InstructionSet::None;
    static constexpr int NumScalar = 1;
    static constexpr int NumBit = 32;
    static constexpr int NumByte = 4; 
    static constexpr int Alignment = 8;
}; 

// vector types contained in Agner Fog's vectorclass
#ifdef WITH_VCL
template <>
struct SIMDInfo<vcl::Vec4f> 
{
    using scalar_type = float;
    using intrinsic_type = __m128;
    static constexpr InstructionSet IS = InstructionSet::SSE2;
    static constexpr int NumScalar = 4;
    static constexpr int NumBit = 128;
    static constexpr int NumByte = 16; 
    static constexpr int Alignment = 16;
}; 
template <>
struct SIMDInfo<vcl::Vec8f> 
{
    using scalar_type = float;
    using intrinsic_type = __m256;
    static constexpr InstructionSet IS = InstructionSet::AVX;
    static constexpr int NumScalar = 8;
    static constexpr int NumBit = 256;
    static constexpr int NumByte = 32; 
    static constexpr int Alignment = 32;
};
template <>
struct SIMDInfo<vcl::Vec16f> 
{
    using scalar_type = float;
    using intrinsic_type = __m512;
    static constexpr InstructionSet IS = InstructionSet::AVX512;
    static constexpr int NumScalar = 16;
    static constexpr int NumBit = 512;
    static constexpr int NumByte = 64; 
    static constexpr int Alignment = 64;
};
template <>
struct SIMDInfo<vcl::Vec2d> 
{
    using scalar_type = double;
    using intrinsic_type = __m128d;
    static constexpr InstructionSet IS = InstructionSet::SSE2;
    static constexpr int NumScalar = 2;
    static constexpr int NumBit = 128;
    static constexpr int NumByte = 16; 
    static constexpr int Alignment = 16;
};
template <>
struct SIMDInfo<vcl::Vec4d> 
{
    using scalar_type = double;
    using intrinsic_type = __m256d;
    static constexpr InstructionSet IS = InstructionSet::AVX;
    static constexpr int NumScalar = 4;
    static constexpr int NumBit = 256;
    static constexpr int NumByte = 32; 
    static constexpr int Alignment = 32;
};
template <>
struct SIMDInfo<vcl::Vec8d> 
{
    using scalar_type = double;
    using intrinsic_type = __m512d;
    static constexpr InstructionSet IS = InstructionSet::AVX512;
    static constexpr int NumScalar = 8;
    static constexpr int NumBit = 512;
    static constexpr int NumByte = 64; 
    static constexpr int Alignment = 64;
};
#endif

constexpr int compute_remainder (int N, int P) 
{
    return N - (N/P) * P; 
}

constexpr int compute_nb_vector (int N, int P) 
{
    if (compute_remainder(N,P) > 0 )
    {
        return N / P + 1;
    }
    return N / P; 
}

}
#endif