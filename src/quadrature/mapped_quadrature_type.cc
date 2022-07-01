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

#include "quadrature/mapped_quadrature_type.h"
#include "quadrature/quadrature.h"

#include "fem/cell_trafo/cell_transformation.h"
#include "fem/cell_trafo/surface_transformation.h"

#include <cassert>
#include <cmath>

namespace hiflow {

template < class DataType >
MappedQuadratureType< DataType >::MappedQuadratureType(
    const Quadrature< DataType > &base_quadrature,
    const QuadratureMapping< DataType > &mapping) {
  const int size = base_quadrature.size();
  // std::cout << "size=" << size << std::endl;
  assert(size > 0);
  // create quadrature type with one size only
  x_.resize(1);
  y_.resize(1);
  z_.resize(1);
  w_.resize(1);
  // map points and weights
  mapping.map_quadrature_data(base_quadrature, x_[0], y_[0], z_[0], w_[0]);
  // register quadrature in index_field
  index_field_[size] = 0;
}

template < class DataType >
MappedQuadratureType< DataType > *
MappedQuadratureType< DataType >::clone() const {
  return new MappedQuadratureType< DataType >(*this);
}

template < class DataType >
QuadrilateralQuadratureMapping< DataType >::QuadrilateralQuadratureMapping(
    int facet_number)
    : facet_number_(facet_number) {}

template < class DataType >
void QuadrilateralQuadratureMapping< DataType >::map_quadrature_data(
    const Quadrature< DataType > &base_quadrature,
    std::vector< DataType > &target_x, std::vector< DataType > &target_y,
    std::vector< DataType > &target_z,
    std::vector< DataType > &target_w) const {
  // NB some assumptions are made here:
  //
  // base_quadrature is a 1d quadrature with coordinates in
  // x-vector. It is a quadrature rule for the domain [0,1]. The
  // target quadrature is for the domain [0,1] x [0,1], with the following
  // facets:
  //        2
  //   +----------+
  //   |          |
  // 3 |          | 1
  //   |          |
  //   |          |
  //   +----------+
  //        0
  //  y
  //  ^
  //  |
  //  ---> x

  const int size = base_quadrature.size();
  target_x.resize(size);
  target_y.resize(size);
  target_z.resize(size, 0.);
  target_w.resize(size);

  // points
  switch (facet_number_) {
  case 0:
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
    }
    std::fill(target_y.begin(), target_y.end(), 0.);
    break;

  case 1:

    for (int q = 0; q < size; ++q) {
      target_y[q] = base_quadrature.x(q);
    }
    std::fill(target_x.begin(), target_x.end(), 1.);
    break;

  case 2:
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
    }
    std::fill(target_y.begin(), target_y.end(), 1.);
    break;

  case 3:
    for (int q = 0; q < size; ++q) {
      target_y[q] = base_quadrature.x(q);
    }
    std::fill(target_x.begin(), target_x.end(), 0.);
    break;

  default:
    // TODO handle sub-facets correctly, with scaling and offset
    assert(false);
  }

  // weights
  switch (facet_number_) {
  case 0:
  case 1:
  case 2:
  case 3:
    for (int q = 0; q < size; ++q) {
      target_w[q] = base_quadrature.w(q);
    }
    break;
  default:
    // TODO handle sub-facets correctly, with scaling
    assert(false);
  }
}

template < class DataType >
TriangleQuadratureMapping< DataType >::TriangleQuadratureMapping(
    int facet_number)
    : facet_number_(facet_number) {}

template < class DataType >
void TriangleQuadratureMapping< DataType >::map_quadrature_data(
    const Quadrature< DataType > &base_quadrature,
    std::vector< DataType > &target_x, std::vector< DataType > &target_y,
    std::vector< DataType > &target_z,
    std::vector< DataType > &target_w) const {
  /*
  // NB some assumptions are made here:
  //
  // base_quadrature is a 1d quadrature with coordinates in
  // x-vector. It is a quadrature rule for the domain [0,1]. The
  // target quadrature is for the domain [0,1] x [0,1], with the following
  facets:
  //
  //   +\
  //   | -\
  // 2 |  --\   1
  //   |     -\
  //   |      --\
  //   +----------+
  //        0
  //  y
  //  ^
  //  |
  //  ---> x
   */

  const int size = base_quadrature.size();
  target_x.resize(size);
  target_y.resize(size);
  target_z.resize(size, 0.);
  target_w.resize(size);

  // points
  switch (facet_number_) {
  case 0:
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
    }
    std::fill(target_y.begin(), target_y.end(), 0.);
    break;

  case 1:
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      // target_y[q] = base_quadrature.x(size - 1 - q);
      target_y[q] = 1. - base_quadrature.x(q);
    }
    break;

  case 2:
    for (int q = 0; q < size; ++q) {
      target_y[q] = base_quadrature.x(q);
    }
    std::fill(target_x.begin(), target_x.end(), 0.);
    break;

  default:
    // TODO handle sub-facets correctly, with scaling and offset
    assert(false);
  }

  // weights
  switch (facet_number_) {
  case 0:
    for (int q = 0; q < size; ++q) {
      target_w[q] = base_quadrature.w(q);
    }
    break;
  case 1: // this is correct! even if it seems odd.
    for (int q = 0; q < size; ++q) {
      target_w[q] = base_quadrature.w(q);
    }
    break;
  case 2:
    for (int q = 0; q < size; ++q) {
      target_w[q] = base_quadrature.w(q);
    }
    break;
  case 3:
  default:
    // TODO handle sub-facets correctly, with scaling
    assert(false);
  }
}

template < class DataType >
HexahedralQuadratureMapping< DataType >::HexahedralQuadratureMapping(
    int facet_number)
    : facet_number_(facet_number) {}

template < class DataType >
void HexahedralQuadratureMapping< DataType >::map_quadrature_data(
    const Quadrature< DataType > &base_quadrature,
    std::vector< DataType > &target_x, std::vector< DataType > &target_y,
    std::vector< DataType > &target_z,
    std::vector< DataType > &target_w) const {
  // NB some assumptions are made here:
  //
  // base_quadrature is a 2d quadrature with coordinates in x-vector
  // and y-vector. It is a quadrature rule for the domain
  // [0,1]x[0,1]. The target quadrature is for the domain
  // [0,1]x[0,1]x[0,1], with the following facets:
  //
  //                      f5 (top) 
  //                4 --------------- 7
  //               /|                /|
  //              / | f2 (left)     / |
  //             /  |              /  |                   ^ z
  //            5 --------------- 6   |                   |
  //            |   |             |   |  f4 (back)        | 
  // f1 (front) |   |             |   |                   |
  //            |   0 ------------|-- 3                   0-------->y
  //            |  /              |  /                   /
  //            | /  f3 (right)   | /                   /
  //            |/                |/                   / x
  //            1 --------------- 2                   v
  //                f0 (bottom)

  const int size = base_quadrature.size();
  target_x.resize(size, 0.);
  target_y.resize(size, 0.);
  target_z.resize(size, 0.);
  target_w.resize(size, 0.);
  // TODO

  // points
  switch (facet_number_) {
  case 0: // bottom
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
    }
    std::fill(target_z.begin(), target_z.end(), 0.);
    break;

  case 1: // front
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_z[q] = base_quadrature.y(q);
    }
    std::fill(target_y.begin(), target_y.end(), 0.);
    break;

  case 2: // left
    for (int q = 0; q < size; ++q) {
      target_y[q] = base_quadrature.y(q);
      target_z[q] = base_quadrature.x(q);
    }
    std::fill(target_x.begin(), target_x.end(), 0.);
    break;

  case 3: // right
    for (int q = 0; q < size; ++q) {
      target_y[q] = base_quadrature.y(q);
      target_z[q] = base_quadrature.x(q);
    }
    std::fill(target_x.begin(), target_x.end(), 1.);
    break;

  case 4: // back
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_z[q] = base_quadrature.y(q);
    }
    std::fill(target_y.begin(), target_y.end(), 1.);
    break;

  case 5: // top
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
    }
    std::fill(target_z.begin(), target_z.end(), 1.);
    break;

  default:
    // TODO handle sub-facets correctly, with scaling and offset
    assert(false);
  }

  // weights
  switch (facet_number_) {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
    for (int q = 0; q < size; ++q) {
      target_w[q] = base_quadrature.w(q);
    }
    break;
  default:
    // TODO handle sub-facets correctly, with scaling
    assert(false);
  }
}

template < class DataType >
TetrahedralQuadratureMapping< DataType >::TetrahedralQuadratureMapping(
    int facet_number)
    : facet_number_(facet_number) {}

template < class DataType >
void TetrahedralQuadratureMapping< DataType >::map_quadrature_data(
    const Quadrature< DataType > &base_quadrature,
    std::vector< DataType > &target_x, std::vector< DataType > &target_y,
    std::vector< DataType > &target_z,
    std::vector< DataType > &target_w) const {
  // NB some assumptions are made here:
  //
  // base_quadrature is a 2d quadrature with coordinates in x-vector
  // and y-vector. It is a quadrature rule for the domain
  // [0,1]x[0,1]. The target quadrature is for the domain
  // [0,1]x[0,1]x[0,1], with the following facets:
  //
  // facet number 0: bottom (+,#,$)
  // facet number 1: left-front (+,$,*)
  // facet number 2: back (+,#,*)
  // facet number 3: front-right ($,#,*)
  //
  //  *
  //  |\
    //  |\ \
    //  | |  \
    //  | |    \
    //  |  \     \
    //  |   |      \
    //  +---|-------#
  //   \  |     /
  //    \ |   /
  //     \| /
  //      $
  //
  //
  //   ^
  //   |
  // z |
  //   |  y
  //   +----->
  //    \
    //   x \
    //      \>
  //

  const int size = base_quadrature.size();
  target_x.resize(size, 0.);
  target_y.resize(size, 0.);
  target_z.resize(size, 0.);
  target_w.resize(size, 0.);
  // TODO

  // points
  switch (facet_number_) {
  case 0: // bottom
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
    }
    std::fill(target_z.begin(), target_z.end(), 0.);
    break;

  case 1: // right
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_z[q] = base_quadrature.y(q);
    }
    std::fill(target_y.begin(), target_y.end(), 0.);
    break;

  case 2: // left
    for (int q = 0; q < size; ++q) {
      target_y[q] = base_quadrature.y(q);
      target_z[q] = base_quadrature.x(q);
    }
    std::fill(target_x.begin(), target_x.end(), 0.);
    break;

  case 3: // front
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
      target_z[q] = 1. - base_quadrature.x(q) - base_quadrature.y(q);
    }
    break;

  default:
    std::cerr << "facet_number_ == " << facet_number_ << "\n";
    // TODO handle sub-facets correctly, with scaling and offset
    assert(false);
  }

  // weights
  switch (facet_number_) {
  case 0:
  case 1:
  case 2:
  case 3:
    for (int q = 0; q < size; ++q) {
      target_w[q] = base_quadrature.w(q);
    }
    break;
  default:
    std::cerr << "facet_number_ == " << facet_number_ << "\n";
    // TODO handle sub-facets correctly, with scaling
    assert(false);
  }
}

template < class DataType >
PyramidQuadratureMapping< DataType >::PyramidQuadratureMapping(int facet_number)
    : facet_number_(facet_number) {}

template < class DataType >
void PyramidQuadratureMapping< DataType >::map_quadrature_data(
    const Quadrature< DataType > &base_quadrature,
    std::vector< DataType > &target_x, std::vector< DataType > &target_y,
    std::vector< DataType > &target_z,
    std::vector< DataType > &target_w) const {

  const int size = base_quadrature.size();
  target_x.resize(size, 0.);
  target_y.resize(size, 0.);
  target_z.resize(size, 0.);
  target_w.resize(size, 0.);

  // points
  switch (facet_number_) {
  case 0: // bottom
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
    }
    std::fill(target_z.begin(), target_z.end(), 0.);
    break;

  case 1: // front
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
      target_z[q] = 2.0 * base_quadrature.y(q);
    }
    break;

  case 2: // right
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
      target_z[q] = 2.0 - 2.0 * base_quadrature.x(q);
    }
    break;

  case 3: // back
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
      target_z[q] = 2.0 - 2.0 * base_quadrature.y(q);
    }
    break;

  case 4: // left
    for (int q = 0; q < size; ++q) {
      target_x[q] = base_quadrature.x(q);
      target_y[q] = base_quadrature.y(q);
      target_z[q] = 2.0 * base_quadrature.x(q);
    }
    break;

  default:
    std::cerr << "facet_number_ == " << facet_number_ << "\n";
    // TODO handle sub-facets correctly, with scaling and offset
    assert(false);
  }

  // weights
  switch (facet_number_) {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
    for (int q = 0; q < size; ++q) {
      target_w[q] = base_quadrature.w(q);
    }
    break;
  default:
    std::cerr << "facet_number_ == " << facet_number_ << "\n";
    // TODO handle sub-facets correctly, with scaling
    assert(false);
  }
}

template < class DataType, int DIM >
SubCellQuadratureMapping< DataType, DIM >::SubCellQuadratureMapping(doffem::CCellTrafoSPtr<DataType, DIM> c_trafo_1,
                                                                    doffem::CCellTrafoSPtr<DataType, DIM> c_trafo_2)
: c_trafo_1_(c_trafo_1),
c_trafo_2_(c_trafo_2) 
{}

template < class DataType, int DIM >
void SubCellQuadratureMapping< DataType, DIM >::map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                                                                    std::vector< DataType > &target_x, 
                                                                    std::vector< DataType > &target_y,
                                                                    std::vector< DataType > &target_z,
                                                                    std::vector< DataType > &target_w) const 
{
  assert (c_trafo_1_ != 0);
  assert (c_trafo_2_ != 0);
  
  const int size = base_quadrature.size();
  target_x.resize(size, 0.);
  target_y.resize(size, 0.);
  target_z.resize(size, 0.);
  target_w.resize(size, 0.);
  
  // loop over quad points in base quad
  for (int q=0; q != size; ++q)
  {
    target_w[q] = base_quadrature.w(q);
    
    // get base quad point
    Coord xq;
    xq.set(0, base_quadrature.x(q));
    
    if constexpr (DIM >= 2)
    {
      xq.set(1, base_quadrature.y(q));
      if constexpr (DIM >= 3)
      {
        xq.set(2, base_quadrature.z(q));
      }
    }
    
    // map to physical cell
    Coord xp;
    c_trafo_1_->transform(xq, xp);
    
    // map to reference cell
    Coord xr;
    bool found = c_trafo_2_->inverse(xp, xr);
    assert (found);
    
    target_x[q] = xr[0];
    if constexpr (DIM >= 2)
    {
      target_y[q] = xr[1];
      if constexpr (DIM >=3)
      {
        target_z[q] = xr[2];
      }
    }
  }
}  

template < class DataType, int RDIM, int PDIM >
SubEntityQuadratureMapping< DataType, RDIM, PDIM >::SubEntityQuadratureMapping(doffem::CSurfaceTrafoSPtr<DataType, RDIM, PDIM> s_trafo,
                                                                               doffem::CCellTrafoSPtr<DataType, PDIM> c_trafo)
: s_trafo_(s_trafo),
c_trafo_(c_trafo) 
{}
    
template < class DataType, int RDIM, int PDIM >
void SubEntityQuadratureMapping< DataType, RDIM, PDIM >::map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                                                                             std::vector< DataType > &target_x, 
                                                                             std::vector< DataType > &target_y,
                                                                             std::vector< DataType > &target_z,
                                                                             std::vector< DataType > &target_w) const 
{
  assert (s_trafo_ != 0);
  assert (c_trafo_ != 0);
  
  const int size = base_quadrature.size();
  target_x.resize(size, 0.);
  target_y.resize(size, 0.);
  target_z.resize(size, 0.);
  target_w.resize(size, 0.);
  
  // loop over quad points in base quad
  for (int q=0; q != size; ++q)
  {
    target_w[q] = base_quadrature.w(q);
    
    // get base quad point
    RCoord xq;
    xq.set(0, base_quadrature.x(q));
    
    if constexpr (RDIM == 2)
    {
      xq.set(1, base_quadrature.y(q));
    }
    
    // map to physical cell
    PCoord xp;
    s_trafo_->transform(xq, xp);
    
    // map to reference cell
    PCoord xr;
    bool found = c_trafo_->inverse(xp, xr);
    assert (found);
    
    target_x[q] = xr[0];
    target_y[q] = xr[1];
    if constexpr (PDIM==3)
    {
      target_z[q] = xr[2];
    }
  }
}    
    
// template instantiations
template class MappedQuadratureType< float >;
template class MappedQuadratureType< double >;
template class QuadrilateralQuadratureMapping< float >;
template class QuadrilateralQuadratureMapping< double >;
template class TriangleQuadratureMapping< float >;
template class TriangleQuadratureMapping< double >;
template class HexahedralQuadratureMapping< float >;
template class HexahedralQuadratureMapping< double >;
template class TetrahedralQuadratureMapping< float >;
template class TetrahedralQuadratureMapping< double >;
template class PyramidQuadratureMapping< float >;
template class PyramidQuadratureMapping< double >;

template class SubCellQuadratureMapping< double, 1 >;
template class SubCellQuadratureMapping< double, 2 >;
template class SubCellQuadratureMapping< double, 3 >;

template class SubCellQuadratureMapping< float, 1 >;
template class SubCellQuadratureMapping< float, 2 >;
template class SubCellQuadratureMapping< float, 3 >;

template class SubEntityQuadratureMapping< double, 1, 2 >;
template class SubEntityQuadratureMapping< double, 1, 3 >;
template class SubEntityQuadratureMapping< double, 2, 3 >;

template class SubEntityQuadratureMapping< float, 1, 2 >;
template class SubEntityQuadratureMapping< float, 1, 3 >;
template class SubEntityQuadratureMapping< float, 2, 3 >;

} // namespace hiflow
