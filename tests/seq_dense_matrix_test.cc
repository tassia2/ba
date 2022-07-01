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

/// @author Simon Gawlok

#define BOOST_TEST_MODULE seq_dense_matrix

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/string.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/set.hpp>

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#include <string>
#include <iostream>

#include "hiflow.h"


using namespace std;
using namespace hiflow::la;

//typedef boost::mpl::string<
//'NAIVE'
////#ifdef WITH_OPENMP
////,
////OPENMP
////#endif
////#ifdef WITH_MKL
////,
////MKL
////#endif
////#ifdef WITH_CLAPACK
////,
////BLAS
////#endif
//> aaMyTypeList;

//typedef boost::mpl::list<
//0
////#ifdef WITH_OPENMP
////,
////OPENMP
////#endif
////#ifdef WITH_MKL
////,
////MKL
////#endif
////#ifdef WITH_CLAPACK
////,
////BLAS
////#endif
//> MyTypeList;

void VectorMult(IMPLEMENTATION impl) {
  int n = 3;
  int m = 16;

  SeqDenseMatrix< double > a_;

  a_.set_implementation_and_platform(impl, CPU);

  a_.Resize(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a_(i, j) = i * n + j + 1;
    }
  }

  std::vector< double > ins(n, 1), outs(m, 0);

  // perform multiplication
  a_.VectorMult(ins, outs);

  BOOST_REQUIRE_CLOSE(outs[0] - 6, 0.0, 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[1], 15., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[2], 24., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[3], 33., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[4], 42., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[5], 51., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[6], 60., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[7], 69., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[8], 78., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[9], 87., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[10], 96., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[11], 105., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[12], 114., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[13], 123., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[14], 132., 1.e-16);
  BOOST_REQUIRE_CLOSE(outs[15], 141., 1.e-16);
}


void MatrixMult(IMPLEMENTATION impl) {
  int n = 3;
  int m = 2;

  SeqDenseMatrix< double > a_;
  a_.set_implementation_and_platform(impl, CPU);

  a_.Resize(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a_(i, j) = i * n + j + 1;
    }
  }

  SeqDenseMatrix< double > inMat, outMat;
  inMat.set_implementation_and_platform(impl, CPU);
  outMat.set_implementation_and_platform(impl, CPU);

  // assemble inMat
  inMat.Resize(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      inMat(i, j) = i * n + j + 7;
    }
  }

  // prepare outMat
  outMat.Resize(m, n);

  // perform multiplication
  a_.MatrixMult(inMat, outMat);

  // Verify result
  BOOST_REQUIRE_CLOSE(outMat(0, 0), 66., 1.e-16);
  BOOST_REQUIRE_CLOSE(outMat(0, 1), 72., 1.e-16);
  BOOST_REQUIRE_CLOSE(outMat(0, 2), 78., 1.e-16);

  BOOST_REQUIRE_CLOSE(outMat(1, 0), 156., 1.e-16);
  BOOST_REQUIRE_CLOSE(outMat(1, 1), 171., 1.e-16);
  BOOST_REQUIRE_CLOSE(outMat(1, 2), 186., 1.e-16);
}

void LU(IMPLEMENTATION impl) {
  int n = 3;
  int m = 3;

  SeqDenseMatrix< double > a_;
  a_.set_implementation_and_platform(impl, CPU);

  a_.Resize(m, n);
  a_(0, 0) = a_(1, 1) = a_(2, 2) = 1;
  a_(0, 1) = 2;
  a_(0, 2) = 4;
  a_(1, 0) = 3;
  a_(1, 2) = 6;
  a_(2, 0) = 5;
  a_(2, 1) = 7;

  std::vector< double > bs(n, 1.), xs(m, 0.);

  // perform decomposition
  a_.Solve(bs, xs);

  // Correctness check
  std::vector< double > res(n, 0);
  a_.VectorMult(xs, res);
  double res_norm = 0;
  for (int i = 0; i < n; ++i) {
    res_norm += (res[i] - bs[i]) * (res[i] - bs[i]);
  }
  BOOST_TEST(std::sqrt(res_norm) < 1.e-12);

  // Scaling test
  n = 3;
  for (int i = 0; i < 3; ++i) {
    // assemble matrix
    a_.Resize(n, n);
    for (int k = 0; k < n; ++k) {
      for (int l = 0; l < n; ++l) {
        a_(k, l) = static_cast< double >(std::rand());
      }
    }
    // cout << a_ << endl;
    bs.resize(n, 1.);
    xs.resize(n, 0.);

    // solve linear system
    a_.Solve(bs, xs);

    // Correctness check

    res.resize(n, 0);
    a_.VectorMult(xs, res);
    res_norm = 0;
    for (int i = 0; i < n; ++i) {
      res_norm += (res[i] - bs[i]) * (res[i] - bs[i]);
    }
    BOOST_TEST(std::sqrt(res_norm) < 1.e-12);
    n *= 2;
  }
}

void QR(IMPLEMENTATION impl) {
  int n = 3;
  int m = 4;

  SeqDenseMatrix< double > a_;
  a_.set_implementation_and_platform(impl, CPU);

  a_.Resize(m, n);
  a_(0, 0) = a_(1, 1) = a_(2, 2) = 1.;
  a_(0, 1) = 2.;
  a_(0, 2) = 4.;
  a_(1, 0) = 3.;
  a_(1, 2) = 6.;
  a_(2, 0) = 5.;
  a_(2, 1) = 7.;
  a_(3, 0) = 10.;
  a_(3, 1) = 11.;
  a_(3, 2) = 12.;

  SeqDenseMatrix< double > Q, R;
  Q.set_implementation_and_platform(impl, CPU);
  R.set_implementation_and_platform(impl, CPU);
  Q.Resize(m, n);
  R.Resize(n, n);

  Q.Zeros();
  R.Zeros();

  // perform decomposition
  a_.QRFactorize(Q, R);

  // Correctness check
  SeqDenseMatrix< double > V;
  V.Resize(m, n);
  V.Zeros();

  // V = Q*R
  Q.MatrixMult(R, V);

  // compute error between A and V
  double residual = 0.;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const double loc = a_(i, j) - V(i, j);
      residual += loc * loc;
    }
  }

  residual = std::sqrt(residual);
  BOOST_TEST(residual < 1.e-12);

  // compute orthonormality of Q
  // DEBUG: Check for orthogonality of Q
  V.Resize(m, m);
  V.Zeros();
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      for (int l = 0; l < m; ++l) {
        V(j, k) += Q(l, j) * Q(l, k);
      }
    }
  }
  residual = 0.;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double loc = 0.;
      if (i == j) {
        loc = V(i, j) - 1.;
      } else {
        loc = V(i, j);
      }
      residual += loc * loc;
    }
  }
  residual = std::sqrt(residual);
  BOOST_TEST(residual < 1.e-12);

  // Scaling test
  n = 3;
  m = 4;
  for (int i = 0; i < 3; ++i) {
    // assemble matrix
    a_.Resize(m, n);
    for (int k = 0; k < m; ++k) {
      for (int l = 0; l < n; ++l) {
        a_(k, l) = static_cast< double >(rand());
      }
    }
    Q.Resize(m, n);
    R.Resize(n, n);

    Q.Zeros();
    R.Zeros();

    // perform decomposition
    a_.QRFactorize(Q, R);

    // Correctness check
    V.Resize(m, n);
    V.Zeros();
    V.set_implementation_and_platform(impl, CPU);

    // V = Q*R
    Q.MatrixMult(R, V);

    // compute error between A and V
    double residual = 0.;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        const double loc = a_(i, j) - V(i, j);
        residual += loc * loc;
      }
    }

    residual = std::sqrt(residual);
    //BOOST_REQUIRE_CLOSE(residual, 0., 5.e-5);

    // compute orthonormality of Q
    // DEBUG: Check for orthogonality of Q: V = Q'*Q
    V.Resize(m, m);
    V.Zeros();
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < m; ++l) {
        const double temp = Q(l, j);
        for (int k = 0; k < n; ++k) {
          V(j, k) += temp * Q(l, k);
        }
      }
    }
    residual = 0.;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double loc = 0.;
        if (i == j) {
          loc = V(i, j) - 1.;
        } else {
          loc = V(i, j);
        }
        residual += loc * loc;
      }
    }
    residual = std::sqrt(residual);
    //BOOST_REQUIRE_CLOSE(residual, 0., 5.e-14);

    // increase dimension for next run
    n *= 2;
    m *= 2;
  }
}

void Compute_Eigenvalues_And_Vectors(IMPLEMENTATION impl) {
  int n = 3;

  SeqDenseMatrix< double > a_;
  a_.set_implementation_and_platform(impl, CPU);

  a_.Resize(n, n);
  // assemble matrix
  a_(0, 0) = 1.0;
  a_(0, 1) = a_(1, 0) = 2.0;
  a_(1, 1) = 1.0;
  a_(0, 2) = a_(2, 0) = 3.0;
  a_(1, 2) = a_(2, 1) = 2.0;
  a_(2, 2) = 1.0;

  // compute eigenvalues and eigenvectors
  a_.compute_eigenvalues_and_vectors();

  std::vector< double > evs = a_.get_eigenvalues();
  std::vector< double > evecs = a_.get_eigenvectors();

  // check, if eigenvalues and vectors fulfil respective condition
  for (int j = 0; j < n; ++j) {
    std::vector< double > in(n, 0.);
    for (int k = 0; k < n; ++k) {
      in[k] = evecs[j * n + k];
    }

    std::vector< double > v1(n, 0.), v2(n, 0.);

    // v1 = A * in
    a_.VectorMult(in, v1);

    // v2 = lambda * in
    for (int k = 0; k < n; ++k) {
      v2[k] = evs[j] * in[k];
    }

    // Now check v1 == v2
    for (int k = 0; k < n; ++k) {
      BOOST_REQUIRE_CLOSE(v1[k], v2[k], 1.e-5);
    }
  }

  // scaling test
  for (int i = 0; i < 3; ++i) {
    n *= 2;
    a_.Resize(n, n);
    for (int k = 0; k < n; ++k) {
      for (int l = k; l < n; ++l) {
        a_(k, l) = a_(l, k) = k * l * rand();
      }
    }

    // compute eigenvalues and eigenvectors
    a_.compute_eigenvalues_and_vectors();

    evs.clear();
    evs = a_.get_eigenvalues();

    evecs.clear();
    evecs = a_.get_eigenvectors();

    // check, if eigenvalues and vectors fulfil respective condition
    for (int j = 0; j < n; ++j) {
      std::vector< double > in(n, 0.);
      for (int k = 0; k < n; ++k) {
        in[k] = evecs[j * n + k];
      }

      std::vector< double > v1(n, 0.), v2(n, 0.);

      // v1 = A * in
      a_.VectorMult(in, v1);

      // v2 = lambda * in
      for (int k = 0; k < n; ++k) {
        v2[k] = evs[j] * in[k];
      }

      // Now check v1 == v2
      for (int k = 0; k < n; ++k) {
        BOOST_REQUIRE_CLOSE(v1[k], v2[k], 1.e-5);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(vector_mult) {
  VectorMult(NAIVE);

#ifdef WITH_OPENMP
  VectorMult(OPENMP);
#endif
#ifdef WITH_MKL
  VectorMult(MKL);
#endif
#ifdef WITH_CLAPACK
  VectorMult(BLAS);
#endif
}

BOOST_AUTO_TEST_CASE(matrix_mult) {
  MatrixMult(NAIVE);

#ifdef WITH_OPENMP
  MatrixMult(OPENMP);
#endif
#ifdef WITH_MKL
  MatrixMult(MKL);
#endif
#ifdef WITH_CLAPACK
  MatrixMult(BLAS);
#endif
}

BOOST_AUTO_TEST_CASE(lu) {
  LU(NAIVE);

#ifdef WITH_OPENMP
  LU(OPENMP);
#endif
#ifdef WITH_MKL
  LU(MKL);
#endif
#ifdef WITH_CLAPACK
  LU(BLAS);
#endif
}

BOOST_AUTO_TEST_CASE(qr) {
  QR(NAIVE);

#ifdef WITH_OPENMP
  QR(OPENMP);
#endif
#ifdef WITH_MKL
  QR(MKL);
#endif
#ifdef WITH_CLAPACK
  QR(BLAS);
#endif
}

//BOOST_AUTO_TEST_CASE(compute_eigenvalues_and_vectors) {
//  Compute_Eigenvalues_And_Vectors(NAIVE);
//
//#ifdef WITH_OPENMP
//  Compute_Eigenvalues_And_Vectors(OPENMP);
//#endif
//#ifdef WITH_MKL
//  Compute_Eigenvalues_And_Vectors(MKL);
//#endif
//#ifdef WITH_CLAPACK
//  Compute_Eigenvalues_And_Vectors(BLAS);
//#endif
//}
