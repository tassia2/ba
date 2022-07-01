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

#define BOOST_TEST_MODULE vector_algebra

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include "common/vector_algebra.h"
#include "common/simd_vector.h"
#include "common/simd_matrix.h"
#include "common/log.h"

using namespace hiflow;
using ScalarType = double;

ScalarType TOLERANCE = 1e-12;

template <class ScalarType>
bool compare_scalars (const ScalarType y1,
                      const ScalarType y2)
{
  return (std::abs(y1- y2) < TOLERANCE);
}

template <class VectorTypeA, class VectorTypeB>
bool compare_vectors (const VectorTypeA y1,
                      const VectorTypeB y2)
{
  BOOST_TEST(y1.size() == y2.size());

  const int N = y1.size();
  for (int i=0; i!=N; ++i)
  {
      if (std::abs(y1[i]- y2[i]) >= TOLERANCE)
      {
        return false;
      }
  }
  return true;
}

template <class MatrixTypeA, class MatrixTypeB>
bool compare_matrices (const MatrixTypeA A,
                       const MatrixTypeB B)
{
  BOOST_TEST(A.num_row() == B.num_row());
  BOOST_TEST(A.num_col() == B.num_col());

  const int M = A.num_row();
  const int N = A.num_col();
  for (int i=0; i!=M; ++i)
  {
    for (int j=0; j!=N; ++j)
    {
      if (std::abs(A(i,j)- B(i,j)) >= TOLERANCE)
      {
        return false;
      }
    }
  }
  return true;
}

template <class MatrixType, class ScalarType>
bool compare_matrices (const std::vector<ScalarType> A,
                       const MatrixType B,
                       const size_t M,
                       const size_t N)
{
  BOOST_TEST(B.num_row() == M);
  BOOST_TEST(B.num_col() == N);

  for (int i=0; i!=M; ++i)
  {
    for (int j=0; j!=N; ++j)
    {
      if (std::abs(A[i*N+j]- B(i,j)) >= TOLERANCE)
      {
        return false;
      }
    }
  }
  return true;
}


BOOST_AUTO_TEST_CASE(vector_algebra) 
{
  ScalarType s = 3.;

  std::vector<ScalarType> a1_val{1.};
  std::vector<ScalarType> b1_val{2.};
  std::vector<ScalarType> a2_val{1., -1.};
  std::vector<ScalarType> b2_val{3., 2.};
  std::vector<ScalarType> a3_val{1., -1., 2.};
  std::vector<ScalarType> b3_val{3., 2., -3.};

  std::vector<ScalarType> A11_val{2.};
  std::vector<ScalarType> B11_val{3.};

  std::vector<ScalarType> C12_val{-1.4, 3.1};

  std::vector<ScalarType> A22_val{2.4, -1.6, 3.1, 5.2};
  std::vector<ScalarType> B22_val{-1.7, 3.6, -3.1, 10.3};

 std::vector<ScalarType> C23_val{-1.4, 4.5, 7.8, -4.3, 8.9, 10.2};

  std::vector<ScalarType> A33_val{2.3, -4.5, 3.9, 4.1, -7.5, 8.9, 5.4, -6.3, 4.8};
  std::vector<ScalarType> B33_val{-3.2, -1.3, -3.9, 4.7, -8.5, 10.9, 50.4, 67.3, 3.8};

   pVec<1, ScalarType> a1_svec(a1_val);
   pVec<1, ScalarType> b1_svec(b1_val);

   pVec<2, ScalarType> a2_svec(a2_val);
  pVec<2, ScalarType> b2_svec(b2_val);

  pVec<3, ScalarType> a3_svec(a3_val);
  pVec<3, ScalarType> b3_svec(b3_val);

  sVec<1, ScalarType> a1_vec(a1_val);
  sVec<1, ScalarType> b1_vec(b1_val);

  sVec<2, ScalarType> a2_vec(a2_val);
  sVec<2, ScalarType> b2_vec(b2_val);

  sVec<3, ScalarType> a3_vec(a3_val);
  sVec<3, ScalarType> b3_vec(b3_val);

  pMat<1,1, ScalarType> A11_smat(A11_val, false);
  pMat<1,1, ScalarType> B11_smat(B11_val, false);
  pMat<1,2, ScalarType> C12_smat(C12_val, false);
  pMat<2,2, ScalarType> A22_smat(A22_val, false);
  pMat<2,2, ScalarType> B22_smat(B22_val, false);
  pMat<2,3, ScalarType> C23_smat(C23_val, false);
  pMat<3,3, ScalarType> A33_smat(A33_val, false);
  pMat<3,3, ScalarType> B33_smat(B33_val, false);

  sMat<1,1, ScalarType> A11_mat(A11_val);
  sMat<1,1, ScalarType> B11_mat(B11_val);
  sMat<1,2, ScalarType> C12_mat(C12_val);
  sMat<2,2, ScalarType> A22_mat(A22_val);
  sMat<2,2, ScalarType> B22_mat(B22_val);
  sMat<2,3, ScalarType> C23_mat(C23_val);
  sMat<3,3, ScalarType> A33_mat(A33_val);
  sMat<3,3, ScalarType> B33_mat(B33_val);


  const size_t L = pVec<1, ScalarType>::L;
  std::cout << "SIMD L = " << L;

  BOOST_TEST (compare_vectors(a1_vec, a1_svec) );
  BOOST_TEST (compare_vectors(a2_vec, a2_svec));
  BOOST_TEST (compare_vectors(a3_vec, a3_svec));
  
  BOOST_TEST (compare_vectors(a1_vec + b1_vec, a1_svec + b1_svec));
  BOOST_TEST (compare_vectors(a2_vec + b2_vec, a2_svec + b2_svec));
  BOOST_TEST (compare_vectors(a3_vec + b3_vec, a3_svec + b3_svec));

  BOOST_TEST (compare_vectors(a1_vec - b1_vec, a1_svec - b1_svec));
  BOOST_TEST (compare_vectors(a2_vec - b2_vec, a2_svec - b2_svec));
  BOOST_TEST (compare_vectors(a3_vec - b3_vec, a3_svec - b3_svec));

  BOOST_TEST (compare_vectors(a1_vec * s, a1_svec * s));
  BOOST_TEST (compare_vectors(a2_vec * s, a2_svec * s));
  BOOST_TEST (compare_vectors(a3_vec * s, a3_svec * s));

  BOOST_TEST (compare_scalars (dot(a1_vec, b1_vec), dot(a1_svec, b1_svec)));
  BOOST_TEST (compare_scalars (dot(a2_vec, b2_vec), dot(a2_svec, b2_svec)));
  BOOST_TEST (compare_scalars (dot(a3_vec, b3_vec), dot(a3_svec, b3_svec)));

  BOOST_TEST (compare_scalars (distance(a1_vec, b1_vec), distance(a1_svec, b1_svec)));
  BOOST_TEST (compare_scalars (distance(a2_vec, b2_vec), distance(a2_svec, b2_svec)));
  BOOST_TEST (compare_scalars (distance(a3_vec, b3_vec), distance(a3_svec, b3_svec)));

  BOOST_TEST (compare_vectors (cross(a3_vec, b3_vec), cross(a3_svec, b3_svec)));

  BOOST_TEST (compare_scalars (sum(a1_vec), sum(a1_svec)));
  BOOST_TEST (compare_scalars (sum(a2_vec), sum(a2_svec)));
  BOOST_TEST (compare_scalars (sum(a3_vec), sum(a3_svec)));

  BOOST_TEST (compare_scalars (norm(a1_vec), norm(a1_svec)));
  BOOST_TEST (compare_scalars (norm(a2_vec), norm(a2_svec)));
  BOOST_TEST (compare_scalars (norm(a3_vec), norm(a3_svec)));

  BOOST_TEST (compare_vectors (normal(a2_vec), normal(a2_svec)));
  BOOST_TEST (compare_vectors (normal(a3_vec, b3_vec), normal(a3_svec, b3_svec)));

  BOOST_TEST (compare_matrices(A11_mat, A11_smat));
  BOOST_TEST (compare_matrices(A22_mat, A22_smat));
  BOOST_TEST (compare_matrices(A33_mat, A33_smat));  

  BOOST_TEST (compare_matrices(A11_mat + B11_mat, A11_smat + B11_smat));
  BOOST_TEST (compare_matrices(A22_mat + B22_mat, A22_smat + B22_smat));
  BOOST_TEST (compare_matrices(A33_mat + B33_mat, A33_smat + B33_smat));  

  BOOST_TEST (compare_matrices(A11_mat * s, A11_smat * s));
  BOOST_TEST (compare_matrices(A22_mat * s, A22_smat * s));
  BOOST_TEST (compare_matrices(A33_mat * s, A33_smat * s));  

  BOOST_TEST (compare_scalars(frob(A11_mat), frob(A11_smat)));
  BOOST_TEST (compare_scalars(frob(A22_mat), frob(A22_smat)));
  BOOST_TEST (compare_scalars(frob(A33_mat), frob(A33_smat)));  

  BOOST_TEST (compare_scalars(det(A11_mat), det(A11_smat)));
  BOOST_TEST (compare_scalars(det(A22_mat), det(A22_smat)));
  BOOST_TEST (compare_scalars(det(A33_mat), det(A33_smat)));  

  BOOST_TEST (compare_matrices(A11_mat * B11_mat, A11_smat * B11_smat));
  BOOST_TEST (compare_matrices(A22_mat * B22_mat, A22_smat * B22_smat));
  BOOST_TEST (compare_matrices(A33_mat * B33_mat, A33_smat * B33_smat));  

  BOOST_TEST (compare_scalars(abs(A11_mat), abs(A11_smat)));
  BOOST_TEST (compare_scalars(abs(A22_mat), abs(A22_smat)));
  BOOST_TEST (compare_scalars(abs(A33_mat), abs(A33_smat)));  

  BOOST_TEST (compare_matrices(A11_mat * C12_mat, A11_smat * C12_smat));
  BOOST_TEST (compare_matrices(A22_mat * C23_mat, A22_smat * C23_smat));
  BOOST_TEST (compare_matrices(C12_mat * B22_mat, C12_smat * B22_smat));
  BOOST_TEST (compare_matrices(C23_mat * B33_mat, C23_smat * B33_smat));

  BOOST_TEST (compare_vectors(A11_mat * a1_vec, A11_smat * a1_svec));
  BOOST_TEST (compare_vectors(A22_mat * a2_vec, A22_smat * a2_svec));
  BOOST_TEST (compare_vectors(A33_mat * a3_vec, A33_smat * a3_svec));

  BOOST_TEST (compare_scalars(dot(A11_mat, B11_mat), dot(A11_smat, B11_smat)));
  BOOST_TEST (compare_scalars(dot(A22_mat, B22_mat), dot(A22_smat, B22_smat)));
  BOOST_TEST (compare_scalars(dot(A33_mat, B33_mat), dot(A33_smat, B33_smat)));  

  BOOST_TEST (compare_scalars(inner(a1_vec, A11_mat, b1_vec), inner(a1_svec, A11_smat, b1_svec)));
  BOOST_TEST (compare_scalars(inner(a2_vec, A22_mat, b2_vec), inner(a2_svec, A22_smat, b2_svec)));
  BOOST_TEST (compare_scalars(inner(a3_vec, A33_mat, b3_vec), inner(a3_svec, A33_smat, b3_svec)));

  BOOST_TEST (compare_scalars(innerT(a1_vec, A11_mat, b1_vec), innerT(a1_svec, A11_smat, b1_svec)));
  BOOST_TEST (compare_scalars(innerT(a2_vec, A22_mat, b2_vec), innerT(a2_svec, A22_smat, b2_svec)));
  BOOST_TEST (compare_scalars(innerT(a3_vec, A33_mat, b3_vec), innerT(a3_svec, A33_smat, b3_svec)));

  BOOST_TEST (compare_matrices(trans(A11_mat), trans(A11_smat)));
  BOOST_TEST (compare_matrices(trans(A22_mat), trans(A22_smat)));
  BOOST_TEST (compare_matrices(trans(A33_mat), trans(A33_smat))); 

  BOOST_TEST (compare_matrices(inv(A11_mat), inv(A11_smat)));
  BOOST_TEST (compare_matrices(inv(A22_mat), inv(A22_smat)));
  BOOST_TEST (compare_matrices(inv(A33_mat), inv(A33_smat))); 

  BOOST_TEST (compare_matrices(invPseudoSur(C12_mat), invPseudoSur(C12_smat)));
  BOOST_TEST (compare_matrices(invPseudoSur(C23_mat), invPseudoSur(C23_smat)));

}

