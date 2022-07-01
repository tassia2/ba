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

#ifndef __FEM_LINEAR_TETRAHEDRON_TRANSFORMATION_H_
#define __FEM_LINEAR_TETRAHEDRON_TRANSFORMATION_H_

#include "fem/cell_trafo/cell_transformation.h"
#include <cmath>
#include <iomanip>

namespace hiflow {
namespace doffem {

///
/// \class LinearTetrahedronTransformation linear_tetrahedron_transformation.h
/// \brief Linear transformation mapping from reference to physical cell for a
/// Tetrahedron \author Michael Schick<br>Martin Baumann
///

template < class DataType, int DIM >
class LinearTetrahedronTransformation final: public CellTransformation< DataType, DIM > 
{
public:
  using Coord = typename CellTransformation< DataType, DIM >::Coord;
  using mat = typename CellTransformation< DataType, DIM >::mat;

  explicit LinearTetrahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell);
  
  explicit LinearTetrahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell, 
                                           const std::vector< mesh::MasterSlave >& period);

  bool differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const;
    
  bool inverse_impl(const Coord& co_phy, Coord &co_ref) const;

  void reinit(const std::vector<DataType> &coord_vtx);
  void reinit(const std::vector<Coord >&coord_vtx);

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

private:
  void setup_transformation_matrix();
  mat A_;
  mat Ainv_;
};

template < class DataType, int DIM >
LinearTetrahedronTransformation< DataType, DIM >::LinearTetrahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell)
    : CellTransformation< DataType, DIM >(ref_cell) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::TET_STD; 
  this->name_ = "Tet";
  this->my_valid_dim_ = 3;
  this->my_nb_vertices_ = 4;
}

template < class DataType, int DIM >
LinearTetrahedronTransformation< DataType, DIM >::LinearTetrahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell,
                                                                                  const std::vector< mesh::MasterSlave >& period)
    : CellTransformation< DataType, DIM >(ref_cell, period) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::TET_STD; 
  this->name_ = "Tet";
  this->my_valid_dim_ = 3;
  this->my_nb_vertices_ = 4;
}

template < class DataType, int DIM >
bool LinearTetrahedronTransformation< DataType, DIM >::differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const
{
  assert(this->ref_cell_);
  
  if (this->ref_cell_->type() != rhs->get_ref_cell()->type())
  {
    return false;
  }
  
  std::vector< Coord > ref_coords = this->ref_cell_->get_coords();
  assert (ref_coords.size() == 4);
  
  Coord my_p01 = this->transform(ref_coords[0]) - this->transform(ref_coords[1]); 
  Coord rhs_p01 = rhs->transform(ref_coords[0]) - rhs->transform(ref_coords[1]); 
  if (my_p01 != rhs_p01)
  {
    return false;
  }
  Coord my_p02 = this->transform(ref_coords[0]) - this->transform(ref_coords[2]); 
  Coord rhs_p02 = rhs->transform(ref_coords[0]) - rhs->transform(ref_coords[2]); 
  if (my_p02 != rhs_p02)
  {
    return false;
  }
  Coord my_p03 = this->transform(ref_coords[0]) - this->transform(ref_coords[3]); 
  Coord rhs_p03 = rhs->transform(ref_coords[0]) - rhs->transform(ref_coords[3]); 
  if (my_p03 != rhs_p03)
  {
    return false;
  }
  return true;
}

template < class DataType, int DIM >
void LinearTetrahedronTransformation< DataType, DIM >::setup_transformation_matrix() 
{ 
  this->A_.set(0, 0, this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
  this->A_.set(1, 0, this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
  this->A_.set(2, 0, this->coord_vtx_[1][2] - this->coord_vtx_[0][2]);
  this->A_.set(0, 1, this->coord_vtx_[2][0] - this->coord_vtx_[0][0]);
  this->A_.set(1, 1, this->coord_vtx_[2][1] - this->coord_vtx_[0][1]);
  this->A_.set(2, 1, this->coord_vtx_[2][2] - this->coord_vtx_[0][2]);
  this->A_.set(0, 2, this->coord_vtx_[3][0] - this->coord_vtx_[0][0]);
  this->A_.set(1, 2, this->coord_vtx_[3][1] - this->coord_vtx_[0][1]);
  this->A_.set(2, 2, this->coord_vtx_[3][2] - this->coord_vtx_[0][2]);

#ifndef NDEBUG
  DataType determ = det(this->A_);
  if (std::abs(determ) < 1e-16)
  {
    std::cout << "very small determinant of linear cell trafo jacobian detected: " << determ << std::endl;
    //assert (false);
  }
  
#endif

  inv(this->A_, this->Ainv_);
}

template < class DataType, int DIM >
void LinearTetrahedronTransformation< DataType, DIM >::reinit(const std::vector<DataType> &coord_vtx) 
{
  CellTransformation< DataType, DIM >::reinit(coord_vtx);
  this->setup_transformation_matrix();
}

template < class DataType, int DIM >
void LinearTetrahedronTransformation< DataType, DIM >::reinit(const std::vector<Coord >&coord_vtx) 
{
  CellTransformation< DataType, DIM >::reinit(coord_vtx);
  this->setup_transformation_matrix();
}

template < class DataType, int DIM >
bool LinearTetrahedronTransformation< DataType, DIM >::inverse_impl(const Coord& co_phy, Coord &co_ref) const {
    
  Coord rhs = co_phy - this->coord_vtx_[0];
  this->Ainv_.VectorMult(rhs, co_ref);

  return true;
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::x(const Coord &coord_ref) const {
    return (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]) * coord_ref[0] +
         (this->coord_vtx_[2][0] - this->coord_vtx_[0][0]) * coord_ref[1] +
         (this->coord_vtx_[3][0] - this->coord_vtx_[0][0]) * coord_ref[2] +
          this->coord_vtx_[0][0];
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::x_x(const Coord &coord_ref) const {
    return (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::x_y(const Coord &coord_ref) const {
    return (this->coord_vtx_[2][0] - this->coord_vtx_[0][0]);
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::x_z(const Coord &coord_ref) const {
    return (this->coord_vtx_[3][0] - this->coord_vtx_[0][0]);
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::y(const Coord &coord_ref) const {
    return (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]) * coord_ref[0] +
         (this->coord_vtx_[2][1] - this->coord_vtx_[0][1]) * coord_ref[1] +
         (this->coord_vtx_[3][1] - this->coord_vtx_[0][1]) * coord_ref[2] +
         this->coord_vtx_[0][1];
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::y_x(const Coord &coord_ref) const {
    return (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::y_y(const Coord &coord_ref) const {
    return (this->coord_vtx_[2][1] - this->coord_vtx_[0][1]);
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::y_z(const Coord &coord_ref) const {
    return (this->coord_vtx_[3][1] - this->coord_vtx_[0][1]);
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::z(const Coord &coord_ref) const {
    return (this->coord_vtx_[1][2] - this->coord_vtx_[0][2]) * coord_ref[0] +
         (this->coord_vtx_[2][2] - this->coord_vtx_[0][2]) * coord_ref[1] +
         (this->coord_vtx_[3][2] - this->coord_vtx_[0][2]) * coord_ref[2] +
         this->coord_vtx_[0][2];
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::z_x(const Coord &coord_ref) const {
    return (this->coord_vtx_[1][2] - this->coord_vtx_[0][2]);
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::z_y(const Coord &coord_ref) const {
    return (this->coord_vtx_[2][2] - this->coord_vtx_[0][2]);
}

template < class DataType, int DIM >
DataType LinearTetrahedronTransformation< DataType, DIM >::z_z(const Coord &coord_ref) const {
    return (this->coord_vtx_[3][2] - this->coord_vtx_[0][2]);
}

} // namespace doffem
} // namespace hiflow

#endif
