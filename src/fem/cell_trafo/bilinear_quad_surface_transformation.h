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

#ifndef __FEM_BILINEAR_QUAD_SURFACE_TRANSFORMATION_H_
#define __FEM_BILINEAR_QUAD_SURFACE_TRANSFORMATION_H_

#include "common/vector_algebra_descriptor.h"
#include "fem/cell_trafo/surface_transformation.h"
#include "fem/cell_trafo/linear_triangle_surface_transformation.h"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace hiflow {
namespace doffem {

///
/// \class BiLinearQuadSurfaceTransformation linear_line_transformation.h
/// \brief Linear transformation mapping from reference to physical cell for a
/// Line \author Philipp Gerstner
///

template < class DataType, int RDIM, int PDIM >
class BiLinearQuadSurfaceTransformation final : public SurfaceTransformation< DataType, RDIM, PDIM > {
public:
  using RCoord = typename SurfaceTransformation< DataType, RDIM, PDIM >::RCoord;
  using PCoord = typename SurfaceTransformation< DataType, RDIM, PDIM >::PCoord;
  using RRmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::RRmat;
  using PPmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::PPmat;
  using RPmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::RPmat;
  using PRmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::PRmat;

  template <int N, typename ScalarType>
  using Coord = typename StaticLA<N, N, ScalarType>::RowVectorType;

  explicit BiLinearQuadSurfaceTransformation(CRefCellSPtr<DataType, RDIM> ref_cell);

  void reinit(const std::vector<DataType> &coord_vtx);
  void reinit(const std::vector<PCoord> &coord_vtx);

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

  void decompose_2_ref (int t, const RCoord& decomp_co_ref, RCoord& co_ref) const 
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

  inline DataType x  (const RCoord &coord_ref) const override;
  inline DataType x_x(const RCoord &coord_ref) const override;
  inline DataType x_y(const RCoord &coord_ref) const override;
  inline DataType x_xy(const RCoord &coord_ref) const override;
  inline DataType y  (const RCoord &coord_ref) const override;
  inline DataType y_x(const RCoord &coord_ref) const override;
  inline DataType y_y(const RCoord &coord_ref) const override;
  inline DataType y_xy(const RCoord &coord_ref) const override;
  inline DataType z  (const RCoord &coord_ref) const override;
  inline DataType z_x(const RCoord &coord_ref) const override;
  inline DataType z_y(const RCoord &coord_ref) const override;
  inline DataType z_xy(const RCoord &coord_ref) const override;

protected:
  bool inverse_impl(const PCoord& co_phy, RCoord &co_ref) const;

private:
  bool inverse_2Dto3D(const Coord<3, DataType>& co_phy, RCoord &co_ref) const;
  int triangle_decomp_ind_[2][3];
};

template < class DataType, int RDIM, int PDIM >
BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::BiLinearQuadSurfaceTransformation(CRefCellSPtr<DataType, RDIM> ref_cell)
    : SurfaceTransformation< DataType, RDIM, PDIM >(ref_cell) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::QUAD_STD;
  this->name_ = "Quad";
  this->my_valid_rdim_ = 2;

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

template < class DataType, int RDIM, int PDIM >
void BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::reinit(const std::vector<DataType> &coord_vtx) 
{  
  SurfaceTransformation< DataType, RDIM, PDIM >::reinit(coord_vtx);

}

template < class DataType, int RDIM, int PDIM >
void BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::reinit(const std::vector<PCoord> &coord_vtx) 
{  
  SurfaceTransformation< DataType, RDIM, PDIM >::reinit(coord_vtx);

}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::x(const RCoord &coord_ref) const 
{
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][0] * (1. - coord_0) * (1. - coord_1) +
         this->coord_vtx_[1][0] * coord_0 * (1. - coord_1) +
         this->coord_vtx_[2][0] * coord_0 * coord_1 +
         this->coord_vtx_[3][0] * (1. - coord_0) * coord_1;
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::x_x(const RCoord &coord_ref) const 
{
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][0] * (-1. + coord_1) +
         this->coord_vtx_[1][0] * (1. - coord_1) +
         this->coord_vtx_[2][0] * coord_1 -
         this->coord_vtx_[3][0] * coord_1;
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::x_y(const RCoord &coord_ref) const 
{
  const DataType coord_0 = coord_ref[0];

  return this->coord_vtx_[0][0] * (-1. + coord_0) -
         this->coord_vtx_[1][0] * coord_0 +
         this->coord_vtx_[2][0] * coord_0 +
         this->coord_vtx_[3][0] * (1. - coord_0);
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::x_xy(const RCoord &coord_ref) const 
{
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][0] -
         this->coord_vtx_[1][0] +
         this->coord_vtx_[2][0] -
         this->coord_vtx_[3][0] ;
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >:: y(const RCoord & coord_ref) const 
{
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][1] * (1. - coord_0) * (1. - coord_1) +
         this->coord_vtx_[1][1] * coord_0 * (1. - coord_1) +
         this->coord_vtx_[2][1] * coord_0 * coord_1 +
         this->coord_vtx_[3][1] * (1. - coord_0) * coord_1;
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::y_x(const RCoord &coord_ref) const 
{
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][1] * (-1. + coord_1) +
         this->coord_vtx_[1][1] * (1. - coord_1) +
         this->coord_vtx_[2][1] * coord_1 -
         this->coord_vtx_[3][1] * coord_1;
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::y_y(const RCoord &coord_ref) const 
{
  const DataType coord_0 = coord_ref[0];

  return this->coord_vtx_[0][1] * (-1. + coord_0) -
         this->coord_vtx_[1][1] * coord_0 +
         this->coord_vtx_[2][1] * coord_0 +
         this->coord_vtx_[3][1] * (1. - coord_0);
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::y_xy(const RCoord &coord_ref) const 
{
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][1] -
         this->coord_vtx_[1][1] +
         this->coord_vtx_[2][1] -
         this->coord_vtx_[3][1] ;
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >:: z(const RCoord & coord_ref) const 
{
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][2] * (1. - coord_0) * (1. - coord_1) +
         this->coord_vtx_[1][2] * coord_0 * (1. - coord_1) +
         this->coord_vtx_[2][2] * coord_0 * coord_1 +
         this->coord_vtx_[3][2] * (1. - coord_0) * coord_1;
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::z_x(const RCoord &coord_ref) const 
{
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][2] * (-1. + coord_1) +
         this->coord_vtx_[1][2] * (1. - coord_1) +
         this->coord_vtx_[2][2] * coord_1 -
         this->coord_vtx_[3][2] * coord_1;
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::z_y(const RCoord &coord_ref) const 
{
  const DataType coord_0 = coord_ref[0];

  return this->coord_vtx_[0][2] * (-1. + coord_0) -
         this->coord_vtx_[1][2] * coord_0 +
         this->coord_vtx_[2][2] * coord_0 +
         this->coord_vtx_[3][2] * (1. - coord_0);
}

template < class DataType, int RDIM, int PDIM >
DataType BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::z_xy(const RCoord &coord_ref) const 
{
  const DataType coord_1 = coord_ref[1];

  return this->coord_vtx_[0][2] -
         this->coord_vtx_[1][2] +
         this->coord_vtx_[2][2] -
         this->coord_vtx_[3][2] ;
}

template < class DataType, int RDIM, int PDIM >
bool BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >::inverse_2Dto3D(const Coord<3, DataType>& co_phy, RCoord &co_ref) const 
{
  RefCellSPtr <DataType, RDIM> ref_tri = RefCellSPtr<DataType, RDIM>(new RefCellTriStd<DataType, RDIM>);
  LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM > tria_trafo(ref_tri);

  bool found_pt_in_tria = inverse_transformation_decomposition
                            <DataType, RDIM, PDIM, 
                              BiLinearQuadSurfaceTransformation< DataType, RDIM, PDIM >, 
                              LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM > >
                              (this, &tria_trafo, co_phy, co_ref);

  // if true: reference point obtained by decomposition is sufficiently accurate
  bool passed_residual_check = this->residual_inverse_check(co_phy, co_ref);

  if (found_pt_in_tria && passed_residual_check) 
  {
    return true;
  } 
  else 
  {
    // point is contained in quad but
    // reference point obtained by decomposition is not accurate enough
    // -> use as initial value for Newton method
    //LOG_INFO("found pt in dec with residual ", this->residual_inverse(co_phy, co_ref));
    bool newton_success = inverse_transformation_newton<DataType, RDIM, PDIM, 
                                                        BiLinearQuadSurfaceTransformation<DataType, RDIM, PDIM> >
                            (this, co_phy, co_ref, co_ref);
    return newton_success;
  }
  return false;
}


} // namespace doffem
} // namespace hiflow

#endif
