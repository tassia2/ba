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

/// \author Philipp Gerstner

#define BOOST_TEST_MODULE simd

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <fstream>

#include "hiflow.h"

using namespace hiflow;
using namespace hiflow::la;

template <class ScalarType>
void build_system (int M, int N, 
                   SeqDenseMatrix< ScalarType >& matrix, 
                   std::vector<ScalarType>& x)
{
    matrix.Clear();
    matrix.Resize(M,N);
    x.clear();
    x.resize(N,0.);

    for (int i = 0; i!=M; ++i)
    {
        for (int j=0; j!=N; ++j)
        {
            matrix(i,j) = (i+1)*(j+1);
        }
    }
    for (int j=0; j!=N; ++j)
    {
        x[j] = j;
    }
}

template <class ScalarType>
void VectorMult (int M, int N, 
                 const SeqDenseMatrix< ScalarType >& matrix, 
                 const std::vector<ScalarType>& x,
                 std::vector<ScalarType>& y)
{
    y.clear();
    y.resize(M, 0.);
    for (int i = 0; i!=M; ++i)
    {
        ScalarType yi = 0.;
        for (int j=0; j!=N; ++j)
        {
            yi += matrix(i,j) * x[j];
        }
        y[i] =yi;
    }
}

template <class ScalarType>
void compare_vectors (const std::vector<ScalarType>& y1,
                      const std::vector<ScalarType>& y2)
{
    assert (y1.size() == y2.size());
    const int N = y1.size();
    for (int i=0; i!=N; ++i)
    {
        BOOST_TEST(std::abs(y1[i]- y2[i]) < 1e-12);
    }
}



BOOST_AUTO_TEST_CASE(simd) 
{
    std::ofstream debug_file("simd_test_output.log");
    LogKeeper::get_log("debug").set_target(&debug_file);

    const int M = 25;
    const int N = 8;

    typedef double ScalarType;

    const ScalarType alpha = 1.;
    SeqDenseMatrix< ScalarType > A1;
    SeqDenseMatrix< ScalarType > A2;
    std::vector<ScalarType> x1;
    std::vector<ScalarType> x2;

    build_system<ScalarType>(M, N, A1, x1);
    build_system<ScalarType>(N, N, A2, x2);
    
    std::vector<ScalarType> y1_ref(M, 0.);
    std::vector<ScalarType> y2_ref(N, 0.);

    VectorMult<ScalarType> (M, N, A1, x1, y1_ref);
    VectorMult<ScalarType> (N, N, A2, x2, y2_ref);
    
    // auto-vec for general size
    CPUsimpleStencil<ScalarType, M, N> A1_simple;
    CPUsimpleStencil<ScalarType, N, N> A2_simple;

    A1_simple.init(A1);
    A2_simple.init(A2);
    
    std::vector<ScalarType> y1_simple(M, 0.);
    std::vector<ScalarType> y2_simple(N, 0.);

    A1_simple.VectorMult(&(x1[0]), &(y1_simple[0]));
    compare_vectors(y1_ref, y1_simple);

    A2_simple.VectorMult(&(x2[0]), &(y2_simple[0]));
    compare_vectors(y2_ref, y2_simple);

    // AVX for 8d
#if WITH_AVX == 1
    std::vector<ScalarType> y_8d(8, 0.);
    CPUavx256Stencil8d mat_8d;
    mat_8d.init(A2);
    mat_8d.VectorMult(&(x2[0]), &(y_8d[0]));
    compare_vectors(y2_ref, y_8d);
#endif

#if WITH_AVX512 == 1
    // AVX512 for 8d
    std::vector<ScalarType> y512_8d(8, 0.);
    CPUavx512Stencil8d mat512_8d;
    mat512_8d.init(A2);
    mat512_8d.VectorMult(&(x2[0]), &(y512_8d[0]));
    compare_vectors(y2_ref, y512_8d);
#endif

#ifdef WITH_VCL
    // simd for general size
    CPUsimdStencil<M, N, vcl::Vec4d> A1_simd;
    CPUsimdStencil<N, N, vcl::Vec4d> A2_simd;

    A1_simd.init(A1);
    A2_simd.init(A2);
    
    std::vector<ScalarType> y1_simd(M, 0.);
    std::vector<ScalarType> y2_simd(N, 0.);

    A1_simd.VectorMult(&(x1[0]), &(y1_simd[0]));
    compare_vectors(y1_ref, y1_simd);

    A2_simd.VectorMult(&(x2[0]), &(y2_simd[0]));
    compare_vectors(y2_ref, y2_simd);
#endif


}


