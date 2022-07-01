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

#ifndef __LMP_CPU_MKL_STENCIL_H
#define __LMP_CPU_MKL_STENCIL_H

#include "linear_algebra/seq_dense_matrix.h"
#include "CPUmkl_blas_routines.h"

#ifdef WITH_MKL

#include <mkl.h>
#endif

#define CPUmkl_STENCIL_ROW_MAJOR

namespace hiflow {
namespace la {

/// @brief The MKL stencil class
/// @author Philipp Gerstner

template < typename DataType, int N >
class CPUmkl_Stencil
{
public:
    typedef SeqDenseMatrix< DataType > LocalMatrix;

    CPUmkl_Stencil()
    {}

    ~CPUmkl_Stencil()
    {}

    void init (const LocalMatrix& lm)
    {
        assert (lm.nrows() == N);
        assert (lm.ncols() == N);

#ifdef CPUmkl_STENCIL_ROW_MAJOR
        for (int i=0; i!=N; ++i)
        {
            for (int j=0; j!=N; ++j)
            {
                vals_[i*N+j] = lm(i,j);
            }
        }
#else 
        for (int i=0; i!=N; ++i)
        {
            for (int j=0; j!=N; ++j)
            {
                vals_[i*N+j] = lm(j,i);
            }
        }
#endif
    }

    void VectorMult(DataType alpha, DataType const * in, DataType * out) const
    {
#ifdef WITH_MKL
#ifdef CPUmkl_STENCIL_ROW_MAJOR
      mkl::blas_MV_row_major (N, N, vals_, alpha, in,  0., out);
#else 
      mkl::blas_MV_col_major (N, N, vals_, alpha, in,  0., out);
#endif
#else 
      NO_MKL_ERROR
#endif
    }

private:
    alignas(DataType) DataType vals_[N*N];

};


}
}


#endif