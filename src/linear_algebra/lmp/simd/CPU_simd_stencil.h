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

#ifndef __LMP_CPU_SIMD_DENSE_OPERATOR_H
#define __LMP_CPU_SIMD_DENSE_OPERATOR_H

#include "common/array_tools.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "common/simd_types.h"
#include "common/simd_functions.h"

namespace hiflow {
namespace la {

/// @brief MxN dense matrix using vectorization provided by VCL 
/// @author Philipp Gerstner

template <int M, int N, class SimdType> 
class CPUsimdStencil
{
public:
    // DataType: scalar data type
    using ScalarType = typename SIMDInfo<SimdType>::scalar_type ;   
    using LocalMatrix =  SeqDenseMatrix< ScalarType >;
    
    CPUsimdStencil()
    {
    }

    ~CPUsimdStencil()
    {
        if (this->delete_ptr_ != nullptr)
        {
            delete[] this->delete_ptr_;
        }
    }

    static constexpr int get_alignment()
    {
        return SIMDInfo<SimdType>::Alignment;
    }

    // initialize buffer containing matrix entries in column major order
    // size of buffer is Mp * N, with  M <= Mp <= M + p - 1
    // Mp is a multiple of p (p = nb scalars inside SimdType)   
    void init (const LocalMatrix& lm)
    {
        assert (lm.nrows() == M);
        assert (lm.ncols() == N);

        // allocate aligned memory
        this->buffer_ = allocate_aligned<ScalarType, SIMDInfo<SimdType>::Alignment>(Mp * N, this->delete_ptr_);

        for (int i=0; i!=N; ++i)
        {
            for (int j=0; j!=M; ++j)
            {
                buffer_[i*Mp+j] = lm(j,i);
            }
            for (int j=M; j!=Mp; ++j)
            {
                buffer_[i*Mp+j] = 0.;
            }
        }
        this->initialized_ = true;
    }


    // compute y = alpha * Ax, 
    void VectorMult(ScalarType const * x, ScalarType* y) const
    {
        assert (this->initialized_);

        SimdType y_v[m];
        for (int i = 0; i != m; ++i)
        {
            y_v[i] = 0.;
        }

        // loop over columns 
        for (int col = 0; col != N; ++col)
        {
            // load x[col] and broadcast into vector
            const SimdType xc_v(x[col]);

            // loop over row vectors 
            for (int r = 0; r != m; ++r)
            {
                // load chunk of current column
                SimdType Ar_v;
                load_a(Ar_v, this->buffer_ + col * Mp + r * p);
                
                // y += A[:,col] * x[col]
                //y_v[r] += Ar_v * xc_v;
                y_v[r] = mul_add(Ar_v, xc_v, y_v[r]);
            }
        }

        // store into output buffer
        for (int i = 0; i != fm; ++i)
        {
            store(y_v[i], y + i * p);
        }

        if (rm > 0)
        {
            assert (fm * p + rm == M);
            store_partial(y_v[m-1], rm, y + fm * p);
        }
    }

private:

    // number of scalar values contained in single SIMD type
    const int p = SIMDInfo<SimdType>::NumScalar;
    
    // number of vectors need for one column
    const int m = compute_nb_vector (M, p);

    // remainder of integer division M / P
    const int rm = compute_remainder (M, p);

    // number of full vectors needed for one column
    const int fm = (rm == 0) ? m : (m - 1); 

    // column length after padding 
    const int Mp = m * p;

    ScalarType* buffer_ = nullptr;
    ScalarType* delete_ptr_ = nullptr;

    bool initialized_ = false;
};


}
}


#endif