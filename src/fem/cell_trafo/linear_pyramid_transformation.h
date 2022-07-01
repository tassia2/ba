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

#ifndef __FEM_LINEAR_PYRAMID_TRANSFORMATION_H_
#define __FEM_LINEAR_PYRAMID_TRANSFORMATION_H_

#include "fem/cell_trafo/cell_transformation.h"
#include <cassert>
#include <cmath>
#include <iomanip>

namespace hiflow {
namespace doffem {

template < class DataType, int DIM >
class LinearPyramidTransformation final: public CellTransformation< DataType, DIM > {
public:
  using Coord = typename CellTransformation< DataType, DIM >::Coord;
  using mat = typename CellTransformation< DataType, DIM >::mat;

  explicit LinearPyramidTransformation(CRefCellSPtr<DataType, DIM> ref_cell);
  explicit LinearPyramidTransformation(CRefCellSPtr<DataType, DIM> ref_cell, 
                                       const std::vector< mesh::MasterSlave >& period);

  bool differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const
  {
    // TODO: implement
    return false;
  }
    
  bool inverse_impl(const Coord& co_phy, Coord &co_ref) const;

  inline DataType x(const Coord &coord_ref) const;
  inline DataType x_x(const Coord &coord_ref) const;
  inline DataType x_y(const Coord &coord_ref) const;
  inline DataType x_z(const Coord &coord_ref) const;
  inline DataType y(const Coord &coord_ref) const;
  inline DataType y_x(const Coord &coord_ref) const;
  inline DataType y_y(const Coord &coord_ref) const;
  inline DataType y_z(const Coord &coord_ref) const;
  inline DataType z(const Coord &coord_ref) const;
  inline DataType z_x(const Coord &coord_ref) const;
  inline DataType z_y(const Coord &coord_ref) const;
  inline DataType z_z(const Coord &coord_ref) const;
};

template < class DataType, int DIM >
LinearPyramidTransformation< DataType, DIM >::LinearPyramidTransformation(CRefCellSPtr<DataType, DIM> ref_cell)
    : CellTransformation< DataType, DIM >(ref_cell) 
{
  this->order_ = 1;  
  this->fixed_ref_cell_type_ = RefCellType::PYR_STD;
  this->name_ = "Pyramid";
  this->my_valid_dim_ = 3;
  this->my_nb_vertices_ = 5;
}

template < class DataType, int DIM >
LinearPyramidTransformation< DataType, DIM >::LinearPyramidTransformation(CRefCellSPtr<DataType, DIM> ref_cell,
                                                                          const std::vector< mesh::MasterSlave >& period)
    : CellTransformation< DataType, DIM >(ref_cell, period) 
{
  this->order_ = 1;  
  this->fixed_ref_cell_type_ = RefCellType::PYR_STD;
  this->name_ = "Pyramid";
  this->my_valid_dim_ = 3;
  this->my_nb_vertices_ = 5;
}

template < class DataType, int DIM >
bool LinearPyramidTransformation< DataType, DIM >::inverse_impl(const Coord& co_phy, Coord &co_ref ) const 
{
    // x = x0 + (x1 - x0) \xi + (x3 - x0) \eta + (x4 - 0.5*x1 - 0.5*x3) \zeta
  // y = y0 + (y1 - y0) \xi + (y3 - y0) \eta + (y4 - 0.5*y1 - 0.5*y3) \zeta
  // z = z0 + (z1 - z0) \xi + (z3 - z0) \eta + (z4 - 0.5*z1 - 0.5*z3) \zeta
  // only works for crystalline element

  DataType a11 = this->coord_vtx_[1][0] -
                 this->coord_vtx_[0][0];
  DataType a12 = this->coord_vtx_[3][0] -
                 this->coord_vtx_[0][0];
  DataType a13 = this->coord_vtx_[4][0] -
                 0.5 * this->coord_vtx_[1][0] -
                 0.5 * this->coord_vtx_[3][0];

  DataType a21 = this->coord_vtx_[1][1] -
                 this->coord_vtx_[0][1];
  DataType a22 = this->coord_vtx_[3][1] -
                 this->coord_vtx_[0][1];
  DataType a23 = this->coord_vtx_[4][1] -
                 0.5 * this->coord_vtx_[1][1] -
                 0.5 * this->coord_vtx_[3][1];

  DataType a31 = this->coord_vtx_[1][2] -
                 this->coord_vtx_[0][2];
  DataType a32 = this->coord_vtx_[3][2] -
                 this->coord_vtx_[0][2];
  DataType a33 = this->coord_vtx_[4][2] -
                 0.5 * this->coord_vtx_[1][2] -
                 0.5 * this->coord_vtx_[3][2];

  DataType det = a11 * (a33 * a22 - a32 * a23) - a21 * (a33 * a12 - a32 * a13) +
                 a31 * (a23 * a12 - a22 * a13);

  assert(det != 0.0);

  co_ref.set(0, (1.0 / det) * ((a33 * a22 - a32 * a23) *
                             (co_phy[0] - this->coord_vtx_[0][0]) -
                         (a33 * a12 - a32 * a13) *
                             (co_phy[1] - this->coord_vtx_[0][1]) +
                         (a23 * a12 - a22 * a13) *
                             (co_phy[2] - this->coord_vtx_[0][2])));

  co_ref.set(1, (1.0 / det) * (-(a33 * a21 - a31 * a23) *
                             (co_phy[0] - this->coord_vtx_[0][0]) +
                         (a33 * a11 - a31 * a13) *
                             (co_phy[1] - this->coord_vtx_[0][1]) -
                         (a23 * a11 - a21 * a13) *
                             (co_phy[2] - this->coord_vtx_[0][2])));

  co_ref.set(2, (1.0 / det) * ((a32 * a21 - a31 * a22) *
                             (co_phy[0] - this->coord_vtx_[0][0]) -
                         (a32 * a11 - a31 * a12) *
                             (co_phy[1] - this->coord_vtx_[0][1]) +
                         (a22 * a11 - a21 * a12) *
                             (co_phy[2] - this->coord_vtx_[0][2])));

  return true;
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::x(const Coord &coord_ref) const {
  
  return this->coord_vtx_[0][0] 
      + (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]) * coord_ref[0] 
      + (this->coord_vtx_[3][0] - this->coord_vtx_[0][0]) * coord_ref[1] 
      + (this->coord_vtx_[4][0] - 0.5 * this->coord_vtx_[1][0] - 0.5 * this->coord_vtx_[3][0]) * coord_ref[2];
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::x_x(const Coord &coord_ref) const {
  
  return (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::x_y(const Coord &coord_ref) const {
  
  return (this->coord_vtx_[3][0] - this->coord_vtx_[0][0]);
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::x_z(const Coord &coord_ref) const {
  
  return (this->coord_vtx_[4][0] - 0.5 * this->coord_vtx_[1][0] - 0.5 * this->coord_vtx_[3][0]);
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::y(const Coord &coord_ref) const {
  
  return   this->coord_vtx_[0][1] 
        + (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]) * coord_ref[0] 
        + (this->coord_vtx_[3][1] - this->coord_vtx_[0][1]) * coord_ref[1] 
        + (this->coord_vtx_[4][1] - 0.5 * this->coord_vtx_[1][1] - 0.5 * this->coord_vtx_[3][1]) * coord_ref[2];
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::y_x(const Coord &coord_ref) const {
  
  return (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::y_y(const Coord &coord_ref) const {
  
  return (this->coord_vtx_[3][1] - this->coord_vtx_[0][1]);
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::y_z(const Coord &coord_ref) const {
  
  return (this->coord_vtx_[4][1] - 0.5 * this->coord_vtx_[1][1] - 0.5 * this->coord_vtx_[3][1]);
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::z(const Coord &coord_ref) const {
  
  return   this->coord_vtx_[0][2] 
        + (this->coord_vtx_[1][2] - this->coord_vtx_[0][2]) * coord_ref[0] 
        + (this->coord_vtx_[3][2] - this->coord_vtx_[0][2]) * coord_ref[1] 
        + (this->coord_vtx_[4][2] - 0.5 * this->coord_vtx_[1][2] - 0.5 * this->coord_vtx_[3][2]) * coord_ref[2];
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::z_x(const Coord &coord_ref) const {
  
  return (this->coord_vtx_[1][2] - this->coord_vtx_[0][2]);
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::z_y(const Coord &coord_ref) const {
    return (this->coord_vtx_[3][2] - this->coord_vtx_[0][2]);
}

template < class DataType, int DIM >
DataType
LinearPyramidTransformation< DataType, DIM >::z_z(const Coord &coord_ref) const {
  
  return (this->coord_vtx_[4][2] - 0.5 * this->coord_vtx_[1][2] - 0.5 * this->coord_vtx_[3][2]);
}

} // namespace doffem
} // namespace hiflow

#endif
