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

#ifndef __LMP_CPU_SIMPLE_STENCIL_H
#define __LMP_CPU_SIMPLE_STENCIL_H

#include "linear_algebra/seq_dense_matrix.h"

namespace hiflow {
namespace la {

/// @brief The naive stencil class
/// @author Philipp Gerstner

template < typename DataType, int M, int N >
class CPUsimpleStencil
{
public:
    typedef SeqDenseMatrix< DataType > LocalMatrix;

    CPUsimpleStencil()
    {}

    ~CPUsimpleStencil()
    {}

    static constexpr int get_alignment ()
    {
        return 8;
    }

    void init (const LocalMatrix& lm)
    {
        assert (lm.nrows() == M);
        assert (lm.ncols() == N);

        for (int i=0; i!=M; ++i)
        {
            for (int j=0; j!=N; ++j)
            {
                vals_[i*N+j] = lm(i,j);
                //valsC_[i*N+j] = lm(j,i);
            }
        }
    }

    inline void VectorMult(DataType const * in, DataType * out) const 
    {
        constexpr int CHUNK = 4;
        constexpr int R = N % CHUNK;
        constexpr int I = N / CHUNK;
        constexpr int J = I * CHUNK;
        
        if constexpr (I > 0)
        {
            for (int i=0; i!=M; ++i)
            {
                DataType sum1 = 0.;
                DataType sum2 = 0.;
                DataType sum3 = 0.;
                DataType sum4 = 0.;

                for (int j=0; j+3<N; j+=CHUNK)
                {
                    sum1 += this->vals_[i*N+j] * in[j];
                    sum2 += this->vals_[i*N+j+1] * in[j+1];
                    sum3 += this->vals_[i*N+j+2] * in[j+2];
                    sum4 += this->vals_[i*N+j+3] * in[j+3];
                }
                
                for (int j=J; j<N; ++j)
                {
                    sum1 += this->vals_[i*N+j] * in[j];
                }
                out[i] = ((sum1 + sum2) + (sum3 + sum4));
            }
        }
        else 
        {
            for (int i=0; i!=M; ++i)
            {
                DataType sum = 0.;
                for (int j=0; j<N; ++j)
                {
                    sum += this->vals_[i*N+j] * in[j];
                }
                out[i] = sum;
            }
        }
    }

private:
    alignas(8) DataType vals_[M*N];
    //alignas(8) DataType valsC_[M*N];
};


}
}


#endif