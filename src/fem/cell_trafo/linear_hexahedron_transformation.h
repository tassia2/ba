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

#ifndef __FEM_ALIGNED_HEXAHEDRON_TRANSFORMATION_H_
#define __FEM_ALIGNED_HEXAHEDRON_TRANSFORMATION_H_

#include "fem/cell_trafo/cell_transformation.h"
//#include "fem/cell_trafo/trilinear_hexahedron_transformation.h"
#include "common/log.h"
#include "common/vector_algebra.h"
#include "mesh/geometric_tools.h"
#include <cmath>
#include <iomanip>
#include <cassert>

namespace hiflow {
namespace doffem {

///
/// \class LinearHexahedronTransformation LinearHexahedronTransformation.h
/// \brief Trilinear transformation mapping from reference to physical cell for
/// a Hexahedron which is assumed to be axis aligned \author Philipp Gerstner
///

template < class DataType, int DIM >
class LinearHexahedronTransformation final : public CellTransformation< DataType, DIM > {
public:
  using Coord = typename CellTransformation< DataType, DIM >::Coord;
  using mat = typename CellTransformation< DataType, DIM >::mat;

  explicit LinearHexahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell);
                                           
  explicit LinearHexahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell, 
                                           const std::vector< mesh::MasterSlave >& period);

  ~ LinearHexahedronTransformation()
  {
  }

  DataType cell_diameter () const override;
  
  bool differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const;
  
  void reinit(const std::vector<DataType> &coord_vtx);

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

protected:
  mat A_;
  mat Ainv_;
  

};

// Reordering of vertices to make transformation coorespond to mesh
// ordering, with (0,0,0) mapped to vertex 0, and (1,1,1) mapped to vertex 7.

template < class DataType, int DIM >
LinearHexahedronTransformation< DataType, DIM >::LinearHexahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell)
    : CellTransformation< DataType, DIM >(ref_cell)
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::HEX_STD;
  this->name_ = "AlignedHex";
  this->my_valid_dim_ = 3;
  this->my_nb_vertices_ = 8;
  
  //
  //        4 --------------- 7
  //       /|                /|
  //      / |               / |
  //     /  |z             /  |
  //    5 --------------- 6   |
  //    |   |             |   |
  //    |   |       y     |   |
  //    |   0 ------------|-- 3
  //    |  /              |  /
  //    | /x              | /
  //    |/                |/
  //    1 --------------- 2
}

template < class DataType, int DIM >
LinearHexahedronTransformation< DataType, DIM >::LinearHexahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell,
                                                                                  const std::vector< mesh::MasterSlave >& period)
    : CellTransformation< DataType, DIM >(ref_cell, period)
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::HEX_STD;
  this->name_ = "AlignedHex";
  this->my_valid_dim_ = 3;
  this->my_nb_vertices_ = 8;
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::cell_diameter () const 
{
  DataType diag_length[4];
  
  diag_length[0] = distance(this->coord_vtx_[0], this->coord_vtx_[6]);
  diag_length[1] = distance(this->coord_vtx_[1], this->coord_vtx_[7]);
  diag_length[2] = distance(this->coord_vtx_[2], this->coord_vtx_[4]);
  diag_length[3] = distance(this->coord_vtx_[3], this->coord_vtx_[5]);
  
  DataType h = 0;
  for (int i=0; i!=4; ++i)
  {
    h = std::max(h, diag_length[i]);
  }
  return h;
}

template < class DataType, int DIM >
bool LinearHexahedronTransformation< DataType, DIM >::differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const
{
  assert (this->ref_cell_);
  
  if (this->ref_cell_->type() != rhs->get_ref_cell()->type())
  {
    return false;
  }
  
  std::vector< Coord > ref_coords = this->ref_cell_->get_coords();
  assert (ref_coords.size() == 8);
  
  Coord my_p01 = this->transform(ref_coords[0]) - this->transform(ref_coords[1]); 
  Coord rhs_p01 = rhs->transform(ref_coords[0]) - rhs->transform(ref_coords[1]); 
  if (my_p01 != rhs_p01)
  {
    return false;
  }
  Coord my_p12 = this->transform(ref_coords[1]) - this->transform(ref_coords[2]); 
  Coord rhs_p12 = rhs->transform(ref_coords[1]) - rhs->transform(ref_coords[2]); 
  if (my_p12 != rhs_p12)
  {
    return false;
  }
  Coord my_p23 = this->transform(ref_coords[2]) - this->transform(ref_coords[3]); 
  Coord rhs_p23 = rhs->transform(ref_coords[2]) - rhs->transform(ref_coords[3]); 
  if (my_p23 != rhs_p23)
  {
    return false;
  }
  Coord my_p30 = this->transform(ref_coords[3]) - this->transform(ref_coords[0]); 
  Coord rhs_p30 = rhs->transform(ref_coords[3]) - rhs->transform(ref_coords[0]); 
  if (my_p30 != rhs_p30)
  {
    return false;
  }
  Coord my_p45 = this->transform(ref_coords[4]) - this->transform(ref_coords[5]); 
  Coord rhs_p45 = rhs->transform(ref_coords[4]) - rhs->transform(ref_coords[5]); 
  if (my_p45 != rhs_p45)
  {
    return false;
  }
  Coord my_p56 = this->transform(ref_coords[5]) - this->transform(ref_coords[6]); 
  Coord rhs_p56 = rhs->transform(ref_coords[5]) - rhs->transform(ref_coords[6]); 
  if (my_p56 != rhs_p56)
  {
    return false;
  }
  Coord my_p67 = this->transform(ref_coords[6]) - this->transform(ref_coords[7]); 
  Coord rhs_p67 = rhs->transform(ref_coords[6]) - rhs->transform(ref_coords[7]); 
  if (my_p67 != rhs_p67)
  {
    return false;
  }
  Coord my_p74 = this->transform(ref_coords[7]) - this->transform(ref_coords[4]); 
  Coord rhs_p74 = rhs->transform(ref_coords[7]) - rhs->transform(ref_coords[4]); 
  if (my_p74 != rhs_p74)
  {
    return false;
  }
  Coord my_p04 = this->transform(ref_coords[0]) - this->transform(ref_coords[4]); 
  Coord rhs_p04 = rhs->transform(ref_coords[0]) - rhs->transform(ref_coords[4]); 
  if (my_p04 != rhs_p04)
  {
    return false;
  }
  Coord my_p15 = this->transform(ref_coords[1]) - this->transform(ref_coords[5]); 
  Coord rhs_p15 = rhs->transform(ref_coords[1]) - rhs->transform(ref_coords[5]); 
  if (my_p15 != rhs_p15)
  {
    return false;
  }
  Coord my_p26 = this->transform(ref_coords[2]) - this->transform(ref_coords[6]); 
  Coord rhs_p26 = rhs->transform(ref_coords[2]) - rhs->transform(ref_coords[6]); 
  if (my_p26 != rhs_p26)
  {
    return false;
  }
  Coord my_p37 = this->transform(ref_coords[3]) - this->transform(ref_coords[7]); 
  Coord rhs_p37 = rhs->transform(ref_coords[3]) - rhs->transform(ref_coords[7]); 
  if (my_p37 != rhs_p37)
  {
    return false;
  }
  return true;
}

template < class DataType, int DIM >
void LinearHexahedronTransformation< DataType, DIM >::reinit(const std::vector<DataType> &coord_vtx) 
{
  assert (DIM == this->my_valid_dim_);
  
  assert (coord_vtx.size() == DIM * 8);
  
  this->coord_vtx_.clear();
  this->coord_vtx_.resize(8);
  
  for (int i=0; i<8; ++i)
  {
    for (int d=0; d<3; ++d)
    {
      this->coord_vtx_[i].set(d, coord_vtx[this->ij2ind(i,d)]);
    }
  }

  //assert (mesh::is_aligned_rectangular_cuboid(coord_vtx));
  assert (mesh::is_parallelepiped(coord_vtx));
  
  this->A_.set(0, 0, this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
  this->A_.set(1, 0, this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
  this->A_.set(2, 0, this->coord_vtx_[1][2] - this->coord_vtx_[0][2]);
  this->A_.set(0, 1, this->coord_vtx_[3][0] - this->coord_vtx_[0][0]);
  this->A_.set(1, 1, this->coord_vtx_[3][1] - this->coord_vtx_[0][1]);
  this->A_.set(2, 1, this->coord_vtx_[3][2] - this->coord_vtx_[0][2]);
  this->A_.set(0, 2, this->coord_vtx_[4][0] - this->coord_vtx_[0][0]);
  this->A_.set(1, 2, this->coord_vtx_[4][1] - this->coord_vtx_[0][1]);
  this->A_.set(2, 2, this->coord_vtx_[4][2] - this->coord_vtx_[0][2]);

#ifndef NDEBUG
  DataType determ = det(this->A_);
  DataType sca_1 = 0.;
  DataType sca_2 = 0.;
  DataType sca_3 = 0.;
  for (size_t d = 0; d<DIM; ++d)
  {
    sca_1 += this->A_(d,0) * this->A_(d,1);
    sca_2 += this->A_(d,0) * this->A_(d,2);
    sca_3 += this->A_(d,1) * this->A_(d,2);
  }
    
  assert (std::abs(determ) > 1e-6);
  //assert (std::abs(sca_1) < 1e-10);
  //assert (std::abs(sca_2) < 1e-10);
  //assert (std::abs(sca_3) < 1e-10);
#endif

  inv(this->A_, this->Ainv_);
}

template < class DataType, int DIM >
bool LinearHexahedronTransformation< DataType, DIM >::inverse_impl(const Coord& co_phy, Coord &co_ref) const {

  // Note (Philipp G): I know, y_ref = (x_phy ... and x_ref = (y_phy ... looks
  // wrong and the other way round should be more reasonable. However, there's a
  // problem somewhere in the DOF/FEM module related to checking which (local)
  // DOFs lie on a specific interface. Unfortunately, it seems that this
  // procedure does NOT take into account the underlying cell transformation.
  // Instead, this procedure assumes, for example, that those dofs that lie on
  // the reference facet with normal, say n = (0, 1, 0), do also lie on the
  // physical facet with normal n = (0, 1, 0), even if the underlying cell
  // transformation performs a rotation in the xy-plane. To make it even worse,
  // the DOF/FEM module assumes that a change in the x-axis in the reference
  // cell leads to a change in the y -axis in the physical cell. I guess this is
  // related to the internal dof ordering. To make a long story short: the
  // assumed DOF ordering is different from the assumed cell vertex ordering by
  // a rotation of 90 degress in the xy-plane.

  Coord rhs = co_phy - this->coord_vtx_[0];
  this->Ainv_.VectorMult(rhs, co_ref);

  return true;
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::x(const Coord &coord_ref) const {

  return this->coord_vtx_[0][0] + this->A_(0,0) * coord_ref[0] + this->A_(0,1) * coord_ref[1] + this->A_(0,2) * coord_ref[2]; 
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::x_x(const Coord &coord_ref) const {
    return this->A_(0,0); 
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::x_y(const Coord &coord_ref) const {
    return this->A_(0,1); 
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::x_z(const Coord &coord_ref) const {
    return this->A_(0,2); 
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::y(const Coord &coord_ref) const {
    return this->coord_vtx_[0][1] + this->A_(1,0) * coord_ref[0] + this->A_(1,1) * coord_ref[1] + this->A_(1,2) * coord_ref[2];
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::y_x(const Coord &coord_ref) const {
    return this->A_(1,0);
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::y_y(const Coord &coord_ref) const {
    return this->A_(1,1);
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::y_z(const Coord &coord_ref) const {
    return this->A_(1,2);
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::z(const Coord &coord_ref) const {
    return this->coord_vtx_[0][2] + this->A_(2,0) * coord_ref[0] + this->A_(2,1) * coord_ref[1] + this->A_(2,2) * coord_ref[2];
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::z_x(const Coord &coord_ref) const {
    return this->A_(2,0);
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::z_y(const Coord &coord_ref) const {
    return this->A_(2,1);
}

template < class DataType, int DIM >
DataType LinearHexahedronTransformation< DataType, DIM >::z_z(const Coord &coord_ref) const {
    return this->A_(2,2);
}

} // namespace doffem
} // namespace hiflow

#endif
