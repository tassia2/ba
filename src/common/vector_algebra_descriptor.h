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

#ifndef __COMMON_VECTOR_ALGEBRA_DESCRIPTOR_H
#define __COMMON_VECTOR_ALGEBRA_DESCRIPTOR_H

#include "common/vector_algebra.h"
#include "common/simd_types.h"
#include "common/simd_vector.h"
#include "common/simd_matrix.h"

namespace hiflow {
/// @author Philipp Gerstner

#ifdef WITH_SIMD_VECTOR_ALGEBRA 
template <size_t DIM, typename ScalarType> 
using Vec = pVec<DIM, ScalarType, false, typename VCLType<ScalarType, false>::vec_type >;

template <size_t M, size_t N, typename ScalarType> 
using Mat = pMat<M, N, ScalarType, false, typename VCLType<ScalarType, false>::vec_type >;

#else 

template <size_t DIM, typename ScalarType> 
using Vec = sVec<DIM, ScalarType>;

template <size_t M, size_t N, typename ScalarType> 
using Mat = sMat<M, N, ScalarType>;

#endif


template <size_t M, size_t N, typename ScalarType> 
struct StaticLA 
{
#ifdef WITH_SIMD_VECTOR_ALGEBRA
  using MatrixType      = pMat<M, N, ScalarType, false, typename VCLType<ScalarType, false>::vec_type >;
  using ColVectorType   = pVec<M, ScalarType, false, typename VCLType<ScalarType, false>::vec_type >;
  using RowVectorType   = pVec<N, ScalarType, false, typename VCLType<ScalarType, false>::vec_type >;
#else 
  using MatrixType      = sMat<M, N, ScalarType >;
  using ColVectorType   = sVec<M, ScalarType >;
  using RowVectorType   = sVec<N, ScalarType >;
#endif
};


}
#endif