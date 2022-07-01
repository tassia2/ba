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

#ifndef __LMP_CPU_MKL_BLAS_H
#define __LMP_CPU_MKL_BLAS_H

#ifdef WITH_MKL
#include <mkl.h>
#endif

/// @author Philipp Gerstner

namespace hiflow {
namespace la {
namespace mkl {

#ifdef WITH_MKL
using INDEX_INT = MKL_INT;
#else 
using INDEX_INT = int;
#endif

#define NO_MKL_ERROR                     \
  LOG_ERROR("no Intel MKL support");     \
  exit(-1);



// compute y:= alpha * Ax + beta *y
// A stored in row major order
// underlying blas routine:

// float / double
// void cblas_[s/d]]gemv ( const CBLAS_LAYOUT Layout , 
//                    const CBLAS_TRANSPOSE trans , 
//                    const MKL_INT m , const MKL_INT n , 
//                    const float alpha , const float *a , const MKL_INT lda , 
//                    const float *x , const MKL_INT incx , const float beta , float *y , const MKL_INT incy );

inline void blas_MV_row_major (INDEX_INT nrow, INDEX_INT ncol, const float * A, 
                               float alpha, const float * x, 
                               float beta, float * y)
{
#ifdef WITH_MKL
  cblas_sgemv(CblasRowMajor, CblasNoTrans, nrow, ncol, alpha , A, ncol , x, 1, beta , y , 1);
#else 
  NO_MKL_ERROR
#endif
}

inline void blas_MV_row_major (INDEX_INT nrow, INDEX_INT ncol, const double * A, 
                               double alpha, const double * x, 
                               double beta, double * y)
{
#ifdef WITH_MKL
  cblas_dgemv(CblasRowMajor, CblasNoTrans, nrow, ncol, alpha , A, ncol , x, 1, beta , y , 1);
#else 
  NO_MKL_ERROR
#endif
}

// compute y:= alpha * Ax + beta *y
// A stored in column major order
inline void blas_MV_col_major (INDEX_INT nrow, INDEX_INT ncol, const float * A, 
                               float alpha, const float * x, 
                               float beta, float * y)
{
#ifdef WITH_MKL
  cblas_sgemv(CblasColMajor, CblasNoTrans, nrow, ncol, alpha , A, nrow , x, 1, beta , y , 1);
#else 
  NO_MKL_ERROR
#endif
}

inline void blas_MV_col_major (INDEX_INT nrow, INDEX_INT ncol, const double * A, 
                               double alpha, const double * x, 
                               double beta, double * y)
{
#ifdef WITH_MKL
  cblas_dgemv(CblasColMajor, CblasNoTrans, nrow, ncol, alpha , A, nrow , x, 1, beta , y , 1);
#else 
  NO_MKL_ERROR
#endif
}


}
}
}

#endif