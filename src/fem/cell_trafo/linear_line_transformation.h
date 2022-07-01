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

#ifndef __FEM_LINEAR_LINE_TRANSFORMATION_H_
#define __FEM_LINEAR_LINE_TRANSFORMATION_H_

#include "fem/cell_trafo/cell_transformation.h"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace hiflow {
namespace doffem {

///
/// \class LinearLineTransformation linear_line_transformation.h
/// \brief Linear transformation mapping from reference to physical cell for a
/// Line \author Michael Schick<br>Martin Baumann<br>Julian Kraemer
///

template < class DataType, int DIM >
class LinearLineTransformation final : public CellTransformation< DataType, DIM > {
public:
  using Coord = typename CellTransformation< DataType, DIM >::Coord;
  using mat = typename CellTransformation< DataType, DIM >::mat;


  explicit LinearLineTransformation(CRefCellSPtr<DataType, DIM> ref_cell);
  explicit LinearLineTransformation(CRefCellSPtr<DataType, DIM> ref_cell, const std::vector< mesh::MasterSlave >& period);

  bool differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const;
   
  bool inverse_impl(const Coord& co_phy, Coord &co_ref) const;

  inline DataType x(const Coord &coord_ref) const;
  inline DataType x_x(const Coord &coord_ref) const;
  inline DataType y(const Coord & coord_ref) const;
  inline DataType y_x(const Coord & coord_ref) const;
  inline DataType z(const Coord & coord_ref) const;
  inline DataType z_x(const Coord & coord_ref) const;

};

template < class DataType, int DIM >
LinearLineTransformation< DataType, DIM >::LinearLineTransformation(CRefCellSPtr<DataType, DIM> ref_cell)
  : CellTransformation< DataType, DIM >(ref_cell) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::LINE_STD;
  this->name_ = "Line";
  this->my_valid_dim_ = 1;
  this->my_nb_vertices_ = 2;
}

template < class DataType, int DIM >
LinearLineTransformation< DataType, DIM >::LinearLineTransformation(CRefCellSPtr<DataType, DIM> ref_cell,
                                                                    const std::vector< mesh::MasterSlave >& period)
  : CellTransformation< DataType, DIM >(ref_cell, period) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::LINE_STD;
  this->name_ = "Line";
  this->my_valid_dim_ = 1;
  this->my_nb_vertices_ = 2;
}

template < class DataType, int DIM >
bool LinearLineTransformation< DataType, DIM >::differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const
{
  assert(this->ref_cell_);
  
  if (this->ref_cell_->type() != rhs->get_ref_cell()->type())
  {
    return false;
  }
  
  std::vector< Coord > ref_coords = this->ref_cell_->get_coords();
  assert (ref_coords.size() == 2);
  
  Coord my_p01 = this->transform(ref_coords[0]) - this->transform(ref_coords[1]); 
  Coord rhs_p01 = rhs->transform(ref_coords[0]) - rhs->transform(ref_coords[1]); 
  if (my_p01 != rhs_p01)
  {
    return false;
  }
  return true;
}

template < class DataType, int DIM >
bool LinearLineTransformation< DataType, DIM >::inverse_impl(const Coord& co_phy, Coord &co_ref) const {

  co_ref.set(0, (co_phy[0] - this->coord_vtx_[0][0]) / (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]));
  return true;
}

template < class DataType, int DIM >
DataType LinearLineTransformation< DataType, DIM >::x(const Coord &coord_ref) const {
  return coord_ref[0] * (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]) + this->coord_vtx_[0][0];
}


template < class DataType, int DIM >
DataType
LinearLineTransformation< DataType, DIM >::x_x(const Coord &coord_ref) const {
  return this->coord_vtx_[1][0] - this->coord_vtx_[0][0];
}

// begin preliminary test functions when the dimension of the physical point is > 1

template < class DataType, int DIM >
DataType
LinearLineTransformation< DataType, DIM >:: y(const Coord & coord_ref) const {
  return coord_ref[0] * (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]) 
        + this->coord_vtx_[0][1];
}

template < class DataType, int DIM >
DataType
LinearLineTransformation< DataType, DIM >:: z(const Coord & coord_ref) const {
  return coord_ref[0] * (this->coord_vtx_[1][2] - this->coord_vtx_[0][2]) 
        + this->coord_vtx_[0][2];
}

template < class DataType, int DIM >
DataType
LinearLineTransformation< DataType, DIM >::y_x(const Coord &coord_ref) const {
  return this->coord_vtx_[1][1] - this->coord_vtx_[0][1];
}

template < class DataType, int DIM >
DataType
LinearLineTransformation< DataType, DIM >::z_x(const Coord &coord_ref) const {
  return this->coord_vtx_[1][2] - this->coord_vtx_[0][2];
}

} // namespace doffem
} // namespace hiflow

#endif
