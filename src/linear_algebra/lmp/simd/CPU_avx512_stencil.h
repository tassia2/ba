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

#ifndef __LMP_CPU_AVX512_STENCIL_H
#define __LMP_CPU_AVX512_STENCIL_H


#include <memory>
#include "linear_algebra/seq_dense_matrix.h"
#include "common/simd_types.h"

namespace hiflow {
namespace la {

/// @brief 8x8 double Stencil class using avx2 IS
/// @author Philipp Gerstner

template <int N>
class CPUavx512Stencil
{
public:
    typedef double DataType;
    typedef SeqDenseMatrix< double > LocalMatrix ;
#if WITH_AVX512 == 1
    typedef __m512d avxType;
#else    
    typedef double avxType;
#endif 

    CPUavx512Stencil()
    {
    }

    ~CPUavx512Stencil()
    {
        if (this->delete_ptr_ != nullptr)
        {
            delete[] this->delete_ptr_;
        }
    }

    static constexpr int get_alignment()
    {
        return 64;
    }

    void init (const LocalMatrix& lm)
    {
        assert (lm.nrows() == N);
        assert (lm.ncols() == N);

        // create memory for matrix values, starting at an adress that is a multiple of 32
        this->vals_ = allocate_aligned<DataType, 64>(N * N, this->delete_ptr_);

        for (int i=0; i!=N; ++i)
        {
            for (int j=0; j!=N; ++j)
            {
                vals_[i*N+j] = lm(j,i);
            }
        }
    }

    // compute y = Ax, 
    void VectorMult(DataType const * x, DataType* y) const
    {
#if WITH_AVX512 == 1
        if constexpr (N == 16)
        {
            // set y = 0
            auto y_07 = _mm512_setzero_pd();                   // y0 - y7
            auto y_815 = _mm512_setzero_pd();                   // y8 - y15

            for (int i=0; i!= 16; ++i)
            {
                const auto x_i = _mm512_set1_pd(x[i]);
                const auto A_07_i = _mm512_load_pd(vals_+i*16); 
                y_07 = _mm512_fmadd_pd(A_07_i, x_i, y_07);
                const auto A_815_i = _mm512_load_pd(vals_+i*16+8); 
                y_815 = _mm512_fmadd_pd(A_815_i, x_i, y_815);
            }

            // stores result in output vector 
            _mm512_store_pd(y, y_07);
            _mm512_store_pd(y+8, y_815);
        }
        else if constexpr (N == 8)
        {
            // set y = 0
            auto y_07_1 = _mm512_setzero_pd();                   // y0 - y7
            auto y_07_2 = _mm512_setzero_pd();

            const auto x_0 = _mm512_set1_pd(x[0]);
            const auto A_07_0 = _mm512_load_pd(vals_); 
            y_07_1 = _mm512_fmadd_pd(A_07_0, x_0, y_07_1);

            const auto x_1 = _mm512_set1_pd(x[1]);
            const auto A_07_1 = _mm512_load_pd(vals_+8);
            y_07_2 = _mm512_fmadd_pd(A_07_1, x_1, y_07_2);

            const auto x_2 = _mm512_set1_pd(x[2]);
            const auto A_07_2 = _mm512_load_pd(vals_+16);
            y_07_1 = _mm512_fmadd_pd(A_07_2, x_2, y_07_1);

            const auto x_3 = _mm512_set1_pd(x[3]);
            const auto A_07_3 = _mm512_load_pd(vals_+24);
            y_07_2 = _mm512_fmadd_pd(A_07_3, x_3, y_07_2);  

            const auto x_4 = _mm512_set1_pd(x[4]);
            const auto A_07_4 = _mm512_load_pd(vals_+32);
            y_07_1 = _mm512_fmadd_pd(A_07_4, x_4, y_07_1);

            const auto x_5 = _mm512_set1_pd(x[5]);
            const auto A_07_5 = _mm512_load_pd(vals_+40);             
            y_07_2 = _mm512_fmadd_pd(A_07_5, x_5, y_07_2);

            const auto x_6 = _mm512_set1_pd(x[6]);
            const auto A_07_6 = _mm512_load_pd(vals_+48);             
            y_07_1 = _mm512_fmadd_pd(A_07_6, x_6, y_07_1);

            const auto x_7 = _mm512_set1_pd(x[7]);
            const auto A_07_7 = _mm512_load_pd(vals_+56);             
            y_07_2 = _mm512_fmadd_pd(A_07_7, x_7, y_07_2);

            const auto y_07 = _mm512_add_pd(y_07_1, y_07_2);

            // stores result in output vector 
            _mm512_store_pd(y, y_07);
        }
        else 
        {
            assert (false);
            quit_program();
        }
#else 
        LOG_ERROR("need to activate BUILD_AVX512 in cmake");
        assert (false);
        quit_program();
#endif
    }

private:
    DataType* vals_ = nullptr;
    DataType* delete_ptr_ = nullptr;
  
};


}
}


#endif