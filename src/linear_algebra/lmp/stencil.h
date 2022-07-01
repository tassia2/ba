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

#ifndef __LMP_STENCIL_H
#define __LMP_STENCIL_H

#include "linear_algebra/seq_dense_matrix.h"
#include "linear_algebra/lmp/avx_types.h"

namespace hiflow {
namespace la {

/// @brief The naive stencil class
/// @author Philipp Gerstner

template < typename DataType, int N >
class CPU_simple_Stencil
public:
    typedef SeqDenseMatrix< DataType > LocalMatrix;

    CPU_simple_Stencil()
    {}

    ~CPU_simple_Stencil()
    {}

    void init (const LocalMatrix& lm)
    {
        assert (lm.nrows() == N);
        assert (lm.ncols() == N);

        for (int i=0; i!=N; ++i)
        {
            for (int j=0; j!=N; ++j)
            {
                vals_[i*N+j] = lm(i,j);
            }
        }
    }

    void VectorMult(DataType alpha, DataType const * in, DataType* out) const 
    {
        for (int i=0; i!=N; ++i)
        {
            DataType sum = 0.;
            for (int j=0; j!=N; ++j)
            {
                sum += this->vals_[i*N+j] * in[j];
            }
            out[i] = alpha * sum;
        }
    }

private:
    alignas(DataType) DataType vals_[N*N];

};

constexpr int compute_n (int N, int p) 
{
    if (N - (N/p) * p > 0 )
    {
        return N / p + 1;
    }
    return N / p; 
}

constexpr int compute_r (int N, int p) 
{
    return N - (N/p) * p; 
}

constexpr int compute_m (int N, int p) 
{
    return N / p; 
}

template < typename DataType, int N >
class CPU_avx2_Stencil
public:
    typedef SeqDenseMatrix< DataType > LocalMatrix;
    typedef typename avx256<DataType>::type avx_type;

    CPU_avx2_Stencil()
    {
        int mask_array[p_];
        //int* mask_array = (int*)aligned_alloc(4, p_ * sizeof(int));

        for (int i = 0; i != r_; ++i)
        {
            mask_array[i] = 1;
        }
        for (int i = r_; i != p_; ++i)
        {
            mask_array[i] = 0;
        }
        mask_ = _mm256_load_si256(mask_array);
    }

    ~CPU_avx2_Stencil()
    {}

    void init (const LocalMatrix& lm)
    {
        assert (lm.nrows() == N);
        assert (lm.ncols() == N);

        for (int i=0; i!=N; ++i)
        {
            for (int j=0; j!=N; ++j)
            {
                vals_[i*N+j] = lm(i,j);
            }
        }
    }


    void VectorMult(DataType alpha, DataType const * in, DataType* out) const;
     
    {
        // load data from in vector
        DataType* pos = in;
        for (int i=0; i!= m_; ++i)
        {
            rhs_[i] = _mm256_load_pd(pos);
            pos += p_;
        }
        if (r_ > 0)
        {
            rhs_[m_] = _mm256_maskload_pd(pos, mask_);
        }

        // loop through rows 
        _mm256_fmadd_pd
        for (int i=0; i!=N; ++i)
        {
            DataType sum = 0.;
            for (int j=0; j!=N; ++j)
            {
                sum += this->vals_[i*N+j] * in[j];
            }
            out[i] = alpha * sum;
        }
    }

private:
    alignas(DataType) DataType vals_[N*N];

    // p_ : number of DataTypes contained in one avx type
    // n_ : number of avx_types required for one matrix row / rhs vector 
    // r_ : remainder 
    // m_ : number of full avx vectors (=n_ if <=> r_ = 0)
    const int p_ = avx256<DataType>::n;
    const int n_ = compute_n(N, p_);
    const int m_ = compute_m(N, p_);
    const int r_ = compute_r(N, p_);
    
    avx_type rhs_[n_];
    __m256i mask_;
};

}
}


#endif