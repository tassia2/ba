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

#ifndef __LMP_CPU_AVX_STENCIL_H
#define __LMP_CPU_AVX_STENCIL_H

#include <memory>
#include "linear_algebra/seq_dense_matrix.h"
#include "common/simd_types.h"

namespace hiflow {
namespace la {

/// @brief NxN double Stencil class using avx2 IS
/// @author Philipp Gerstner

template <int N>
class CPUavx256Stencil
{
public:
    typedef double DataType;
    typedef SeqDenseMatrix< double > LocalMatrix ;
#if WITH_AVX == 1
    typedef __m256d avxType;
#else 
    typedef double avxType;
#endif

    CPUavx256Stencil()
    {
    }

    ~CPUavx256Stencil()
    {
        if (this->delete_ptr_ != nullptr)
        {
            delete[] this->delete_ptr_;
        }
    }

    static constexpr int get_alignment()
    {
        return 32;
    }

    void init (const LocalMatrix& lm)
    {
        assert (lm.nrows() == N);
        assert (lm.ncols() == N);

        // create memory for matrix values, starting at an adress that is a multiple of 32
        this->vals_ = allocate_aligned<DataType, 32>(N * N, this->delete_ptr_);

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
        // p_ : number of DataType's per avxType 

        //const int p_ = 4;
        //const int m_ = 2;
#if WITH_AVX == 1
        if constexpr (N == 16)
        {
            // set y = 0
            auto y_03 = _mm256_setzero_pd();                   // y0 - y3
            auto y_47 = _mm256_setzero_pd();                   // y4 - y7
            auto y_811 = _mm256_setzero_pd();                   // y8 - y11
            auto y_1215 = _mm256_setzero_pd();                   // y8 - y11

            for (int i=0; i!= 16; ++i)
            {
                const auto x_i = _mm256_set1_pd(x[i]);
                const auto A_03_i = _mm256_load_pd(vals_+i*16); 
                y_03 = _mm256_fmadd_pd(A_03_i, x_i, y_03);
                const auto A_47_i = _mm256_load_pd(vals_+i*16+4); 
                y_47 = _mm256_fmadd_pd(A_47_i, x_i, y_47);
                const auto A_811_i = _mm256_load_pd(vals_+i*16+8); 
                y_811 = _mm256_fmadd_pd(A_811_i, x_i, y_811);
                const auto A_1215_i = _mm256_load_pd(vals_+i*16+12); 
                y_1215 = _mm256_fmadd_pd(A_1215_i, x_i, y_1215);
            }

            // stores result in output vector 
            _mm256_store_pd(y, y_03);
            _mm256_store_pd(y+4, y_47);
            _mm256_store_pd(y+8, y_811);
            _mm256_store_pd(y+12, y_1215);
        }
        else if constexpr (N == 8)
        {
            // set y = 0
            auto y_03 = _mm256_setzero_pd();                   // y0 - y3
            auto y_47 = _mm256_setzero_pd();                   // y4 - y7

            const auto x_0 = _mm256_set1_pd(x[0]);
            const auto A_03_0 = _mm256_load_pd(vals_); 
            y_03 = _mm256_fmadd_pd(A_03_0, x_0, y_03);
            const auto A_47_0 = _mm256_load_pd(vals_+4); 
            y_47 = _mm256_fmadd_pd(A_47_0, x_0, y_47);

            const auto x_1 = _mm256_set1_pd(x[1]);
            const auto A_03_1 = _mm256_load_pd(vals_+8);
            y_03 = _mm256_fmadd_pd(A_03_1, x_1, y_03);
            const auto A_47_1 = _mm256_load_pd(vals_+12); 
            y_47 = _mm256_fmadd_pd(A_47_1, x_1, y_47);

            const auto x_2 = _mm256_set1_pd(x[2]);
            const auto A_03_2 = _mm256_load_pd(vals_+16);
            y_03 = _mm256_fmadd_pd(A_03_2, x_2, y_03);
            const auto A_47_2 = _mm256_load_pd(vals_+20);
            y_47 = _mm256_fmadd_pd(A_47_2, x_2, y_47);

            const auto x_3 = _mm256_set1_pd(x[3]);
            const auto A_03_3 = _mm256_load_pd(vals_+24);
            y_03 = _mm256_fmadd_pd(A_03_3, x_3, y_03);  
            const auto A_47_3 = _mm256_load_pd(vals_+28);            
            y_47 = _mm256_fmadd_pd(A_47_3, x_3, y_47);

            const auto x_4 = _mm256_set1_pd(x[4]);
            const auto A_03_4 = _mm256_load_pd(vals_+32);
            y_03 = _mm256_fmadd_pd(A_03_4, x_4, y_03);
            const auto A_47_4 = _mm256_load_pd(vals_+36); 
            y_47 = _mm256_fmadd_pd(A_47_4, x_4, y_47);

            const auto x_5 = _mm256_set1_pd(x[5]);
            const auto A_03_5 = _mm256_load_pd(vals_+40);             
            y_03 = _mm256_fmadd_pd(A_03_5, x_5, y_03);
            const auto A_47_5 = _mm256_load_pd(vals_+44);    
            y_47 = _mm256_fmadd_pd(A_47_5, x_5, y_47);

            const auto x_6 = _mm256_set1_pd(x[6]);
            const auto A_03_6 = _mm256_load_pd(vals_+48);             
            y_03 = _mm256_fmadd_pd(A_03_6, x_6, y_03);
            const auto A_47_6 = _mm256_load_pd(vals_+52);       
            y_47 = _mm256_fmadd_pd(A_47_6, x_6, y_47);

            const auto x_7 = _mm256_set1_pd(x[7]);
            const auto A_03_7 = _mm256_load_pd(vals_+56);             
            y_03 = _mm256_fmadd_pd(A_03_7, x_7, y_03);
            const auto A_47_7 = _mm256_load_pd(vals_+60);             
            y_47 = _mm256_fmadd_pd(A_47_7, x_7, y_47);

            // stores result in output vector 
            _mm256_store_pd(y, y_03);
            _mm256_store_pd(y+4, y_47);
        }
        else if constexpr (N == 4)
        {
            // set y = 0
            auto y1_03 = _mm256_setzero_pd();                   // y0 - y3
            auto y2_03 = _mm256_setzero_pd();
            
            const auto x_0 = _mm256_set1_pd(x[0]);
            const auto A_03_0 = _mm256_load_pd(vals_); 
            y1_03 = _mm256_fmadd_pd(A_03_0, x_0, y1_03);

            const auto x_1 = _mm256_set1_pd(x[1]);
            const auto A_03_1 = _mm256_load_pd(vals_+4);
            y2_03 = _mm256_fmadd_pd(A_03_1, x_1, y2_03);

            const auto x_2 = _mm256_set1_pd(x[2]);
            const auto A_03_2 = _mm256_load_pd(vals_+8);
            y1_03 = _mm256_fmadd_pd(A_03_2, x_2, y1_03);

            const auto x_3 = _mm256_set1_pd(x[3]);
            const auto A_03_3 = _mm256_load_pd(vals_+12);
            y2_03 = _mm256_fmadd_pd(A_03_3, x_3, y2_03);  

            auto y_03 = _mm256_add_pd(y1_03, y2_03);

            // stores result in output vector 
            _mm256_store_pd(y, y_03);
        }
        else 
        {
            assert (false);
            quit_program();
        }
#else 
        LOG_ERROR("need to compile with -mavx");
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