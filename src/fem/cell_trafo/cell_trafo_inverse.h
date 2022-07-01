// Copyright (C) 2011-2017 Vincent Heuveline
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

#ifndef __FEM_CELL_TRANSFORMATION_INVERSE_H_
#define __FEM_CELL_TRANSFORMATION_INVERSE_H_

#include <cassert>
#include <vector>
#include <limits>
#include <cmath>

#include "common/log.h"
#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"

namespace hiflow {
namespace doffem {

template < class DataType, int DIM, class Transformation >
bool inverse_transformation_newton(const typename StaticLA<DIM, DIM, DataType>::ColVectorType &co_phy, 
                                   typename StaticLA<DIM, DIM, DataType>::ColVectorType &co_ref );

template < class DataType, int DIM, class Transformation >
bool inverse_transformation_newton(const typename StaticLA<DIM, DIM, DataType>::ColVectorType &co_phy, 
                                   typename StaticLA<DIM, DIM, DataType>::ColVectorType &co_ref, 
                                   typename StaticLA<DIM, DIM, DataType>::ColVectorType &co_ref_0);

/// \details The Inverse of the mapping (reference cell to physical cell)
///          is computed in 3D via a Newton-scheme with Armijo updates. The
///          Problem, which is to be solved writes<br>
///          G([x_ref, y_ref, z_ref]) := [x_phy - x(coord_ref),
///                                       y_phy - y(coord_ref),
///                                       z_phy - z(coord_ref)]<br>
///          solve G([x_ref, y_ref, z_ref]) = 0 via Newton-scheme
/// \param[in] x_phy x coordinate on physical cell
/// \param[in] y_phy y coordinate on physical cell
/// \param[in] z_phy z coordinate on physical cell
/// \param[out] x_ref x coordinate on reference cell
/// \param[out] y_ref y coordinate on reference cell
/// \param[out] z_ref z coordinate on reference cell
template < class DataType, int RDIM, int PDIM, class Transformation >
bool inverse_transformation_newton(const Transformation* trafo,
                                   const typename StaticLA<PDIM, RDIM, DataType>::ColVectorType &co_phy, 
                                   typename StaticLA<PDIM, RDIM, DataType>::RowVectorType &co_ref)
{
  typedef typename StaticLA<PDIM, RDIM, DataType>::RowVectorType RCoord;

  RCoord co_ref_0;
  if constexpr (RDIM == 1)      
  {
    co_ref_0.set(0, 0.5);
  }
  else if constexpr (RDIM == 2)
  {
    co_ref_0.set(0, 0.55);
    co_ref_0.set(1, 0.55);
  }
  else if constexpr (RDIM == 3)
  {
    co_ref_0.set(0, 0.1154);
    co_ref_0.set(1, 0.1832);
    co_ref_0.set(2, 0.1385);
  }
  else 
  {
    assert (false);
  }

  const DataType atol = 1000. * std::numeric_limits< double >::epsilon();
  const DataType stol = 10. * std::numeric_limits< DataType >::epsilon();
  const DataType ctol = 1.e3 * std::numeric_limits< DataType >::epsilon();
  const int max_it = 100;
  const int armijo_max_it = 10;

  return inverse_transformation_newton<DataType, RDIM, PDIM, Transformation> (trafo, co_phy, co_ref, co_ref_0,
                                                                              atol, stol, ctol, max_it, armijo_max_it);
}

template < class DataType, int RDIM, int PDIM, class Transformation >
bool inverse_transformation_newton(const Transformation* trafo,
                                   const typename StaticLA<PDIM, RDIM, DataType>::ColVectorType &co_phy,
                                   typename StaticLA<PDIM, RDIM, DataType>::RowVectorType &co_ref, 
                                   typename StaticLA<PDIM, RDIM, DataType>::RowVectorType &co_ref_0, 
                                   const DataType atol = 1000. * std::numeric_limits< double >::epsilon(),
                                   const DataType stol = 10. * std::numeric_limits< DataType >::epsilon(),  // stagnation tolerance
                                   const DataType ctol = 1.e3 * std::numeric_limits< DataType >::epsilon(), // tolerance for cell bounds
                                   const int max_it = 100,
                                   const int armijo_max_it = 10)
{
  using RCoord = typename StaticLA<PDIM, RDIM, DataType>::RowVectorType;
  using PCoord = typename StaticLA<PDIM, RDIM, DataType>::ColVectorType;

  using RRmat = typename StaticLA<RDIM, RDIM, DataType>::MatrixType;
  using PPmat = typename StaticLA<PDIM, PDIM, DataType>::MatrixType;
  using RPmat = typename StaticLA<RDIM, PDIM, DataType>::MatrixType;
  using PRmat = typename StaticLA<PDIM, RDIM, DataType>::MatrixType;

  // Initialisation
  PCoord pt_phy = co_phy;

  RCoord ref_k1, ref_k;
  ref_k = co_ref_0;

  int iter = 0;

  // Residual
  PCoord pt_k;
  trafo->transform(ref_k, pt_k);
  
  DataType residual = norm(pt_phy - pt_k);
  DataType progress = norm(ref_k1 - ref_k);

  // Newton

  // Jacobian Matrix (grad G)
  PRmat J;

  // (Pseudo-) Inverse of the jacobian
  RPmat J_inv;

  while (residual > atol) 
  {
    trafo->J(ref_k, J);
  
    if constexpr (RDIM == PDIM)
    {
      inv(J, J_inv);
    }
    else 
    {
      invPseudoInj(J, J_inv);
    }

    // Armijo parameter
    int iter_armijo = 0;

    DataType residual_armijo = 2. * residual;

    DataType omega = 1.;

    const RCoord update_vec = J_inv * (pt_phy - pt_k);

    // Start Armijo

    while ((iter_armijo <= armijo_max_it) && (residual_armijo > residual)) 
    {

      ref_k1 = ref_k + omega * update_vec;

      for (int d=0; d<RDIM; ++d) 
      {
        if ((ref_k1[d] >= -ctol) && (ref_k1[d] <= ctol)) 
        {
          ref_k1.set(d, 0.);
        }
        else if ((ref_k1[d] - 1. >= -ctol) && (ref_k1[d] - 1. <= ctol)) 
        {
          ref_k1.set(d, 1.);
        }
      }

      while (!(trafo->contains_reference_point(ref_k1))) 
      {
        omega /= 2.0;
        ref_k1 = ref_k + omega * update_vec;

        for (int d=0; d<RDIM; ++d) 
        {
          if ((ref_k1[d] >= -ctol) && (ref_k1[d] <= ctol)) 
          {
            ref_k1.set(d, 0.);
          }
          else if ((ref_k1[d] - 1. >= -ctol) && (ref_k1[d] - 1. <= ctol)) 
          {
            ref_k1.set(d, 1.);
          }
        }
      }

      PCoord F_k1;
      trafo->transform(ref_k1, F_k1);

      residual_armijo = norm(pt_phy - F_k1);

      ++iter_armijo;
      omega /= 2.;
    }

    progress = norm(ref_k1 - ref_k);
    ref_k = ref_k1;

    trafo->transform(ref_k, pt_k);

    residual = norm(pt_phy - pt_k);

    ++iter;
    if (iter > max_it) 
    {
      break;
    }
    if (progress < stol)
    {
      break;
    }
  } // end newton

  LOG_DEBUG(2, "Inverse cell-trafo ended after "
                   << iter << " Newton iterations with residual = " << residual
                   << ", |x_k - x_{k-1}| = " << progress);
  // Set values ...
  co_ref = ref_k;

  return residual < atol;
} 


template < class DataType, int RDIM, int PDIM, class Transformation, class SubTransformation >
bool inverse_transformation_decomposition(const Transformation* trafo,
                                          SubTransformation* sub_trafo,
                                          const typename StaticLA<PDIM, RDIM, DataType>::ColVectorType& co_phy,   
                                          typename StaticLA<PDIM, RDIM, DataType>::RowVectorType &co_ref) 
{
  using RCoord = typename StaticLA<PDIM, RDIM, DataType>::RowVectorType;
  using PCoord = typename StaticLA<PDIM, RDIM, DataType>::ColVectorType;
  
  RCoord co_ref_sub;

  for (int d=0; d!=RDIM; ++d)
  {
    co_ref_sub.set(d, -1.);
  }

  int sub_ind = -1;
  bool found_pt_in_sub = false;
  const int num_sub_vertices = sub_trafo->num_vertices();
  const int num_sub = trafo->num_sub_decomposition();

  std::vector<PCoord> sub_coord(num_sub_vertices);
    
  // loop through tetrahedron decomposition
  for (int t = 0; t < num_sub; ++t) 
  {
    // build linear tetrahedron transformation
    sub_coord.clear();
    sub_coord.resize(num_sub_vertices);
    for (int v = 0; v < num_sub_vertices; ++v) 
    {
      sub_coord[v] = trafo->get_coordinate(trafo->get_subtrafo_decomposition(t,v));
    }
    sub_trafo->reinit(sub_coord);

    // compute reference coordinates w.r.t. tetrahedron
    bool sub_success = sub_trafo->inverse(co_phy, co_ref_sub);
    if (!sub_success) 
    {
      continue;
    }

    // check whether reference coordinates are contained in tetrahedron

    // if yes: convert reference coordinates
    if (sub_trafo->contains_reference_point(co_ref_sub)) 
    {
      trafo->decompose_2_ref(t, co_ref_sub, co_ref);
      sub_ind = t;
      found_pt_in_sub = true;
      break;
    }
  }

  LOG_DEBUG(2, "found point by decomposition " << found_pt_in_sub);

  return found_pt_in_sub;
}

//////////////////////////////////////////////////////////////////////OLD implementations//////////////////////////////////////////////////////////////////////////////
#if 0
template < class DataType, int DIM, class Transformation >
bool inverse_transformation_newton(const Transformation* trafo,
                                   const Vec<DIM, DataType> &co_phy, 
                                   Vec<DIM, DataType> &co_ref, 
                                   Vec<DIM, DataType> &co_ref_0)
{
  typedef Vec<DIM, DataType> Coord;

  // Initialisation
  Coord pt_phy = co_phy;

  Coord ref_k1, ref_k;
  ref_k = co_ref_0;

  const DataType tol_eps = 1.e3 * std::numeric_limits< DataType >::epsilon();
  const DataType stagnation_tol = 10. * std::numeric_limits< DataType >::epsilon();

  // Some general parameters

  const int iter_max = 100;
  const int iter_max_armijo = 10;
  int iter = 0;

  // Residual

  Coord coord_ref_k = ref_k;
  Coord pt_k;
  trafo->transform(coord_ref_k, pt_k);
  
  DataType residual = norm(pt_phy - pt_k);
  DataType abserr = norm(ref_k1 - ref_k);

  // Newton

  // Jacobian Matrix (grad G)
  Mat< DIM, DIM, DataType > G;
  // Inverse of the jacobian
  Mat< DIM, DIM, DataType > B;

  while (residual > INVERSE_RESIDUAL_TOL) 
  {
    trafo->J(coord_ref_k, G);
  
#ifndef NDEBUG
    const DataType detG = det(G);
    assert(detG != 0.);
#endif

    inv(G, B);

    // Armijo parameter
    int iter_armijo = 0;

    DataType residual_armijo = 2. * residual;

    DataType omega = 1.;

    const Coord update_vec = B * (pt_phy - pt_k);

    // Start Armijo

    while ((iter_armijo <= iter_max_armijo) && (residual_armijo > residual)) {

      ref_k1 = ref_k + omega * update_vec;

      for (int d=0; d<DIM; ++d) 
      {
        if ((ref_k1[d] >= -tol_eps) && (ref_k1[d] <= tol_eps)) {
          ref_k1[d] = 0.;
        }
        if ((ref_k1[d] - 1. >= -tol_eps) && (ref_k1[d] - 1. <= tol_eps)) {
          ref_k1[d] = 1.;
        }
      }

      Coord ref_check = ref_k1;

      while (!(trafo->contains_reference_point(ref_check))) 
      {
        omega /= 2.0;
        ref_k1 = ref_k + omega * update_vec;

        for (int d=0; d<DIM; ++d) 
        {
          if ((ref_k1[d] >= -tol_eps) && (ref_k1[d] <= tol_eps)) 
          {
            ref_k1[d] = 0.;
          }
          if ((ref_k1[d] - 1. >= -tol_eps) && (ref_k1[d] - 1. <= tol_eps)) 
          {
            ref_k1[d] = 1.;
          }
        }
        ref_check = ref_k1;
      }

      Coord coord_ref_k1 = ref_k1;

      Coord F_k1;
      trafo->transform(coord_ref_k1, F_k1);

      residual_armijo = norm(pt_phy - F_k1);

      ++iter_armijo;
      omega /= 2.;
    }

    abserr = norm(ref_k1 - ref_k);
    ref_k = ref_k1;

    coord_ref_k = ref_k;
    trafo->transform(coord_ref_k, pt_k);

    residual = norm(pt_phy - pt_k);

    ++iter;
    if (iter > iter_max) {
      break;
    }
    if (abserr < stagnation_tol)
      break;

  } // end newton

  LOG_DEBUG(2, "Inverse cell-trafo ended after "
                   << iter << " Newton iterations with residual = " << residual
                   << ", |x_k - x_{k-1}| = " << abserr);
  // Set values ...
  co_ref = ref_k;

  return residual < INVERSE_RESIDUAL_TOL;
} 
#endif

#if 0

template < class DataType, int DIM >
bool inverse_newton_2Dto3D( const Vec<3, DataType> &co_phy, 
                            Vec<2, DataType> &co_ref, 
                            Vec<2, DataType> &co_ref_0) const 
{
/* needs to be debugged 
  // Initialisation

  Vec<3, DataType> pt_phy = co_phy;

  Vec<2, DataType> ref_k1, ref_k;
  ref_k = co_ref_0;

  const DataType tol_eps = 1.e3 * std::numeric_limits< DataType >::epsilon();

  // Some general parameters

  const int iter_max = 1000;
  int iter = 0;

  // Residual

  Vec<2, DataType> coord_ref_k = ref_k;
  Vec<3, DataType> pt_k;

   pt_k[0] = this->x(coord_ref_k);
   pt_k[1] = this->y(coord_ref_k);
   pt_k[2] = this->z(coord_ref_k);

  
  DataType residual = norm(pt_phy - pt_k);
  DataType abserr = norm(ref_k1 - ref_k);

  // Newton

  // Jacobian Matrix (grad G)
  Mat< 3, 2, DataType > G;
  // Pseudo-Inverse of the jacobian
  Mat< 2, 3, DataType > B;

  while (residual > 10. * std::numeric_limits< DataType >::epsilon()) 
  {
    G.set(0, 0, this->x_x(coord_ref_k));
    G.set(0, 1, this->x_y(coord_ref_k));

    G.set(1, 0, this->y_x(coord_ref_k));
    G.set(1, 1, this->y_y(coord_ref_k));

    G.set(2, 0, this->z_x(coord_ref_k));
    G.set(2, 1, this->z_y(coord_ref_k));

    //inverse of G^T * G
    
    Mat< 2, 2, DataType > res;

#ifndef NDEBUG
    const DataType detres = det(res);

    assert(det(G.transpose_me() * G) != 0.);
#endif

    inv(G.transpose_me() * G, res);

    B = res * G.transpose_me();

    // Armijo parameter

    const int iter_max_armijo = 500;
    int iter_armijo = 0;

    DataType residual_armijo = 2. * residual;

    DataType omega = 1.;

    const Vec<2, DataType> update_vec = B * (pt_phy - pt_k);

    // Start Armijo

    while ((iter_armijo <= iter_max_armijo) && (residual_armijo > residual)) {

      ref_k1 = ref_k + omega * update_vec;

      for (int d=0; d<2; ++d) 
      {
        if ((ref_k1[d] >= -tol_eps) && (ref_k1[d] <= tol_eps)) {
          ref_k1[d] = 0.;
        }
        if ((ref_k1[d] - 1. >= -tol_eps) && (ref_k1[d] - 1. <= tol_eps)) {
          ref_k1[d] = 1.;
        }
      }

      Vec<2, DataType> ref_check = ref_k1;

      while (!(this->contains_reference_point(ref_check))) {
        omega /= 2.0;
        ref_k1 = ref_k + omega * update_vec;

        for (int d=0; d<2; ++d) 
        {
          if ((ref_k1[d] >= -tol_eps) && (ref_k1[d] <= tol_eps)) {
            ref_k1[d] = 0.;
          }
          if ((ref_k1[d] - 1. >= -tol_eps) && (ref_k1[d] - 1. <= tol_eps)) {
            ref_k1[d] = 1.;
          }
        }
        ref_check = ref_k1;
      }

      Vec<2, DataType> coord_ref_k1 = ref_k1;

      Coord F_k1;

      F_k1[0] = this->x(coord_ref_k1);
      F_k1[1] = this->y(coord_ref_k1);
      F_k1[2] = this->z(coord_ref_k1);


      residual_armijo = norm(pt_phy - F_k1);

      ++iter_armijo;
      omega /= 2.;
    }

    abserr = norm(ref_k1 - ref_k);
    ref_k = ref_k1;

    coord_ref_k = ref_k;

    pt_k[0] = this->x(coord_ref_k);
    pt_k[1] = this->y(coord_ref_k);
    pt_k[2] = this->z(coord_ref_k);


    residual = norm(pt_phy - pt_k);

    ++iter;
    if (iter > iter_max) {
      break;
    }
    if (abserr < 10. * std::numeric_limits< DataType >::epsilon())
      break;

  } // end newton

  LOG_DEBUG(1, "Inverse cell-trafo ended after "
                   << iter << " Newton iterations with residual = " << residual
                   << ", |x_k - x_{k-1}| = " << abserr);
  // Set values ...
  co_ref = ref_k;

  return residual < 10. * std::numeric_limits< DataType >::epsilon();
*/
  return false;
} 
#endif

#if 0

//to invert the matrix, we have to solve a 3 x 2 linear system. Since one row depends linearly on the other two rows,
//we indentify two linearly independent rows and solve the remaining 2x2 system.
template < class DataType, int DIM >
bool LinearTriangleTransformation< DataType, DIM >::inverse_2Dto3D(Vec< 3, DataType > co_phy, Coord &co_ref) const {
  
  DataType a11 = this->coord_vtx_[1][0] -
                 this->coord_vtx_[0][0];
  DataType a12 = this->coord_vtx_[2][0] -
                 this->coord_vtx_[0][0];
  DataType a21 = this->coord_vtx_[1][1] -
                 this->coord_vtx_[0][1];
  DataType a22 = this->coord_vtx_[2][1] -
                 this->coord_vtx_[0][1];
  DataType a31 = this->coord_vtx_[1][2] -
                 this->coord_vtx_[0][2];
  DataType a32 = this->coord_vtx_[2][2] -
                 this->coord_vtx_[0][2];
 

  DataType det1 = a11 * a22 - a21 * a12;
  DataType det2 = a11 * a32 - a31 * a12;
  DataType det3 = a21 * a32 - a31 * a22;
  
  if (det1 != 0.0) 
  {
    co_ref[0] = (1.0 / det1) * (a22 * (co_phy[0] - this->coord_vtx_[0][0]) -
                           a12 * (co_phy[1] - this->coord_vtx_[0][1]));
    co_ref[1] = (1.0 / det1) * (-a21 * (co_phy[0] - this->coord_vtx_[0][0]) +
                           a11 * (co_phy[1] - this->coord_vtx_[0][1]));
  }
  else if (det2 != 0) 
  {
    co_ref[0] = (1.0 / det2) * (a32 * (co_phy[0] - this->coord_vtx_[0][0]) -
                           a12 * (co_phy[2] - this->coord_vtx_[0][2]));
    co_ref[1] = (1.0 / det2) * (-a31 * (co_phy[0] - this->coord_vtx_[0][0]) +
                           a11 * (co_phy[2] - this->coord_vtx_[0][2]));
  }
  else if (det3 != 0) 
  {
    co_ref[0] = (1.0 / det3) * (a32 * (co_phy[1] - this->coord_vtx_[0][1]) -
                           a22 * (co_phy[2] - this->coord_vtx_[0][2]));
    co_ref[1] = (1.0 / det3) * (-a31 * (co_phy[1] - this->coord_vtx_[0][1]) +
                           a21 * (co_phy[2] - this->coord_vtx_[0][2]));
  }
   
  else 
  {
    return false; //triangle is degenerate
  }
  
  return this->contains_reference_point(co_ref);
}
#endif
} // namespace doffem
} // namespace hiflow

#endif
