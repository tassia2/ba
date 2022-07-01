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

#ifndef __FEM_BILINEAR_QUAD_TRANSFORMATION_H_
#define __FEM_BILINEAR_QUAD_TRANSFORMATION_H_

#include "fem/cell_trafo/cell_transformation.h"
#include "common/log.h"
#include "fem/cell_trafo/linear_triangle_transformation.h"
#include <cmath>
#include <iomanip>

namespace hiflow {
namespace doffem {

///
/// \class BiLinearQuadTransformation bilinear_quad_transformation.h
/// \brief Bilinear transformation mapping from reference to physical cell for a
/// Quadrilateral /// \author Michael Schick<br>Martin Baumann<br>Simon Gawlok<br>Philipp Gerstner
///

template < class DataType, int DIM >
class BiLinearQuadTransformation final: public CellTransformation< DataType, DIM > {
public:
  using Coord = typename CellTransformation< DataType, DIM >::Coord;
  using mat = typename CellTransformation< DataType, DIM >::mat;
  
  explicit BiLinearQuadTransformation(CRefCellSPtr<DataType, DIM> ref_cell);
  explicit BiLinearQuadTransformation(CRefCellSPtr<DataType, DIM> ref_cell,
                                      const std::vector< mesh::MasterSlave >& period);
  ~ BiLinearQuadTransformation ()
  {
  }

  bool differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const;
   
  bool inverse_impl(const Coord& co_phy, Coord &co_ref) const;
  
  DataType cell_diameter() const override;

  inline int num_sub_decomposition() const 
  {
    return 2;
  }

  inline int get_subtrafo_decomposition(int t, int v) const 
  {
    assert (t >= 0);
    assert (t < 2);
    assert (v >= 0);
    assert (v < 3);
    return this->triangle_decomp_ind_[t][v];
  }

  void decompose_2_ref (int t, const Coord& decomp_co_ref, Coord& co_ref) const 
  {
    switch (t) 
    {
      case 0:
        co_ref = decomp_co_ref;
        break;
      case 1:
        co_ref.set(0, 1. - decomp_co_ref[0]);
        co_ref.set(1, 1. - decomp_co_ref[1]);
        break;
    }
  }

  inline DataType x(const Coord &coord_ref) const;
  inline DataType x_x(const Coord &coord_ref) const;
  inline DataType x_y(const Coord &coord_ref) const;
  inline DataType x_xy(const Coord &coord_ref) const;
  inline DataType y(const Coord &coord_ref) const;
  inline DataType y_x(const Coord &coord_ref) const;
  inline DataType y_y(const Coord &coord_ref) const;
  inline DataType y_xy(const Coord &coord_ref) const;
  inline DataType z(const Coord &coord_ref) const;
  inline DataType z_x(const Coord &coord_ref) const;
  inline DataType z_y(const Coord &coord_ref) const;
  inline DataType z_xy(const Coord &coord_ref) const;

protected:
  void init();
  
  bool inverse_by_decomposition(const Coord& co_phy, Coord &co_ref) const;

  int triangle_decomp_ind_[2][3];
};

template < class DataType, int DIM >
BiLinearQuadTransformation< DataType, DIM >::BiLinearQuadTransformation(CRefCellSPtr<DataType, DIM> ref_cell)
    : CellTransformation< DataType, DIM >(ref_cell) 
{
  this->my_valid_dim_ = 2;
  this->my_nb_vertices_ = 4;
  this->init();
}

template < class DataType, int DIM >
BiLinearQuadTransformation< DataType, DIM >::BiLinearQuadTransformation(CRefCellSPtr<DataType, DIM> ref_cell,  
                                                                        const std::vector< mesh::MasterSlave >& period)
    : CellTransformation< DataType, DIM >(ref_cell, period) 
{
  this->my_valid_dim_ = 2;
  this->my_nb_vertices_ = 4;
  this->init();
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::cell_diameter () const 
{
  DataType diag_length[2];
  
  diag_length[0] = distance(this->coord_vtx_[0], this->coord_vtx_[2]);
  diag_length[1] = distance(this->coord_vtx_[1], this->coord_vtx_[3]);
  
  DataType h = 0;
  for (int i=0; i!=2; ++i)
  {
    h = std::max(h, diag_length[i]);
  }
  return h;
}

template < class DataType, int DIM >
void BiLinearQuadTransformation< DataType, DIM >::init()
{
  this->order_ = 2;
  this->fixed_ref_cell_type_ = RefCellType::QUAD_STD;
  this->name_ = "Quad";
    
  // define decomposition of quadrilateral into 2 triangle
  //
  //        0 --------------- 3
  //       /        y        /
  //      /x                /
  //     /                 /
  //    1 --------------- 2

  // triangle 0:
  //        0 --------------- 3
  //       /        y
  //      /x
  //     /
  //    1

  triangle_decomp_ind_[0][0] = 0;
  triangle_decomp_ind_[0][1] = 1;
  triangle_decomp_ind_[0][2] = 3;

  // triangle 1:
  //                          3
  //                         /
  //                       x/
  //            y          /
  //    1 --------------- 2

  triangle_decomp_ind_[1][0] = 2;
  triangle_decomp_ind_[1][1] = 3;
  triangle_decomp_ind_[1][2] = 1;
}

template < class DataType, int DIM >
bool BiLinearQuadTransformation< DataType, DIM >::differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const
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
  return true;
}

template < class DataType, int DIM >
bool BiLinearQuadTransformation< DataType, DIM >::inverse_by_decomposition(const Coord& co_phy, Coord &co_ref) const 
{
  RefCellSPtr <DataType, DIM> ref_tri = RefCellSPtr<DataType, DIM>(new RefCellTriStd<DataType, DIM>);
  LinearTriangleTransformation< DataType, DIM > tria_trafo(ref_tri);

  bool found_pt_in_tria = inverse_transformation_decomposition
                            <DataType, DIM, DIM, BiLinearQuadTransformation< DataType, DIM >, LinearTriangleTransformation< DataType, DIM > >
                              (this, &tria_trafo, co_phy, co_ref);

#ifdef NEW_INVERSE
  return found_pt_in_tria;
#else 
  // old impl
  if (found_pt_in_tria)
  {
    return this->residual_inverse_check(co_phy, co_ref);
  }
  else
  {
    return false;
  }
#endif
}

template < class DataType, int DIM >
bool BiLinearQuadTransformation< DataType, DIM >::inverse_impl(const Coord& co_phy, Coord &co_ref) const 
{
  // if true: point is contained in sub-triangle
  bool found_pt_in_triangle = this->inverse_by_decomposition(co_phy, co_ref);

  // if true: reference point obtained by decomposition is sufficiently accurate
  bool passed_residual_check = this->residual_inverse_check(co_phy, co_ref);

  if (found_pt_in_triangle && passed_residual_check) 
  {
    return true;
  } 
  else 
  {
#ifdef NEW_INVERSE
    if (!found_pt_in_triangle)
    {
      // if point is not contained in sub-triangle, it is also not contained in quad
      // -> no need to call newton
      return false;
    }
#endif
    // point is contained in quad but
    // reference point obtained by decomposition is not accurate enough
    // -> use as initial value for Newton method
    //LOG_INFO("found pt in dec with residual ", this->residual_inverse(co_phy, co_ref));
    bool newton_success = inverse_transformation_newton<DataType, DIM, DIM, BiLinearQuadTransformation<DataType, DIM> >
                            (this, co_phy, co_ref, co_ref);
    return newton_success;
  }
}

template < class DataType, int DIM >
DataType
BiLinearQuadTransformation< DataType, DIM >::x(const Coord &coord_ref) const 
{
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][0] * (1. - coord_0) * (1. - coord_1) +
         this->coord_vtx_[1][0] * coord_0 * (1. - coord_1) +
         this->coord_vtx_[2][0] * coord_0 * coord_1 +
         this->coord_vtx_[3][0] * (1. - coord_0) * coord_1;
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::x_x(const Coord &coord_ref) const 
{
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][0] * (-1. + coord_1) +
         this->coord_vtx_[1][0] * (1. - coord_1) +
         this->coord_vtx_[2][0] * coord_1 -
         this->coord_vtx_[3][0] * coord_1;
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::x_y(const Coord &coord_ref) const {
  const DataType coord_0 = coord_ref[0];

  return this->coord_vtx_[0][0] * (-1. + coord_0) -
         this->coord_vtx_[1][0] * coord_0 +
         this->coord_vtx_[2][0] * coord_0 +
         this->coord_vtx_[3][0] * (1. - coord_0);
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::x_xy(const Coord &coord_ref) const {
  return this->coord_vtx_[0][0] -
         this->coord_vtx_[1][0] +
         this->coord_vtx_[2][0] -
         this->coord_vtx_[3][0];
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::y(const Coord &coord_ref) const {
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][1] * (1. - coord_0) * (1. - coord_1) +
         this->coord_vtx_[1][1] * coord_0 * (1. - coord_1) +
         this->coord_vtx_[2][1] * coord_0 * coord_1 +
         this->coord_vtx_[3][1] * (1. - coord_0) * coord_1;
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::y_x(const Coord &coord_ref) const {
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][1] * (-1. + coord_1) +
         this->coord_vtx_[1][1] * (1. - coord_1) +
         this->coord_vtx_[2][1] * coord_1 -
         this->coord_vtx_[3][1] * coord_1;
}

template < class DataType, int DIM >
DataType
BiLinearQuadTransformation< DataType, DIM >::y_y(const Coord &coord_ref) const {
  const DataType coord_0 = coord_ref[0];

  return this->coord_vtx_[0][1] * (-1. + coord_0) -
         this->coord_vtx_[1][1] * coord_0 +
         this->coord_vtx_[2][1] * coord_0 +
         this->coord_vtx_[3][1] * (1. - coord_0);
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::y_xy(const Coord &coord_ref) const {
  return this->coord_vtx_[0][1] -
         this->coord_vtx_[1][1] +
         this->coord_vtx_[2][1] -
         this->coord_vtx_[3][1];
}

// begin preliminary test functions when the dimension of the physical point is > 1
template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::z(const Coord &coord_ref) const {
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][2] * (1. - coord_0) * (1. - coord_1) +
         this->coord_vtx_[1][2] * coord_0 * (1. - coord_1) +
         this->coord_vtx_[2][2] * coord_0 * coord_1 +
         this->coord_vtx_[3][2] * (1. - coord_0) * coord_1;
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::z_x(const Coord &coord_ref) const {
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][2] * (coord_1 - 1.) +
         this->coord_vtx_[1][2] * (1. - coord_1) +
         this->coord_vtx_[2][2] * coord_1 -
         this->coord_vtx_[3][2] * coord_1;
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::z_y(const Coord &coord_ref) const {
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][2] * (coord_0 - 1.) -
         this->coord_vtx_[1][2] * coord_0  +
         this->coord_vtx_[2][2] * coord_0 +
         this->coord_vtx_[3][2] * (1. - coord_0);
}

template < class DataType, int DIM >
DataType BiLinearQuadTransformation< DataType, DIM >::z_xy(const Coord &coord_ref) const {
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][2] -
         this->coord_vtx_[1][2] +
         this->coord_vtx_[2][2] -
         this->coord_vtx_[3][2];
}


} // namespace doffem
} // namespace hiflow

#endif
