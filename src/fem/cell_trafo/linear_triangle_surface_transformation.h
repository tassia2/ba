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

#ifndef __FEM_LINEAR_TRIANGLE_SURFACE_TRANSFORMATION_H_
#define __FEM_LINEAR_TRIANGLE_SURFACE_TRANSFORMATION_H_

#include "fem/cell_trafo/surface_transformation.h"
#include "common/vector_algebra_descriptor.h"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace hiflow {
namespace doffem {

///
/// \class LinearTriangleSurfaceTransformation linear_line_transformation.h
/// \brief Linear transformation mapping from reference to physical cell for a
/// Line \author Philipp Gerstner
///

template < class DataType, int RDIM, int PDIM >
class LinearTriangleSurfaceTransformation final : public SurfaceTransformation< DataType, RDIM, PDIM > {
public:
  using RCoord = typename SurfaceTransformation< DataType, RDIM, PDIM >::RCoord;
  using PCoord = typename SurfaceTransformation< DataType, RDIM, PDIM >::PCoord;
  using RRmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::RRmat;
  using PPmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::PPmat;
  using RPmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::RPmat;
  using PRmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::PRmat;

  template <int N, typename ScalarType>
  using Coord = typename StaticLA<N, N, ScalarType>::RowVectorType;


  explicit LinearTriangleSurfaceTransformation(CRefCellSPtr<DataType, RDIM> ref_cell);

  void reinit(const std::vector<DataType> &coord_vtx);
  void reinit(const std::vector<PCoord> &coord_vtx);
  
  inline DataType x  (const RCoord &coord_ref) const override;
  inline DataType x_x(const RCoord &coord_ref) const override;
  inline DataType x_y(const RCoord &coord_ref) const override;
  inline DataType y  (const RCoord &coord_ref) const override;
  inline DataType y_x(const RCoord &coord_ref) const override;
  inline DataType y_y(const RCoord &coord_ref) const override;
  inline DataType z  (const RCoord &coord_ref) const override;
  inline DataType z_x(const RCoord &coord_ref) const override;
  inline DataType z_y(const RCoord &coord_ref) const override;

protected:
  bool inverse_impl(const PCoord& co_phy, RCoord &co_ref) const;

private:
  bool inverse_2Dto3D(const Coord<3, DataType>& co_phy, RCoord &co_ref) const;
  void setup_transformation_matrix();

  RRmat A_;
  RRmat A_inv_; 

  int co0_;
  int co1_;

#ifndef NDEBUG
  RRmat Axy_;
  RRmat Axy_inv_; 
  RRmat Axz_;
  RRmat Axz_inv_;
  RRmat Ayz_;
  RRmat Ayz_inv_;
#endif
};

template < class DataType, int RDIM, int PDIM >
LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::LinearTriangleSurfaceTransformation(CRefCellSPtr<DataType, RDIM> ref_cell)
    : SurfaceTransformation< DataType, RDIM, PDIM >(ref_cell) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::TRI_STD;
  this->name_ = "Triangle";
  this->my_valid_rdim_ = 2;
  this->my_nb_vertices_ = 3;
}

template < class DataType, int RDIM, int PDIM >
void LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::setup_transformation_matrix()
{
  // equation system for inverse transformation is overdetermined
  // -> skip one equation
  // -> selected by using most regular 2x2 submatrix of jacobian
  RRmat Axy;
  RRmat Axz;
  RRmat Ayz;

  Axy.set(0, 0, this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
  Axy.set(1, 0, this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
  Axy.set(0, 1, this->coord_vtx_[2][0] - this->coord_vtx_[0][0]);
  Axy.set(1, 1, this->coord_vtx_[2][1] - this->coord_vtx_[0][1]);

  if (PDIM > 2)
  {
    Axz.set(0, 0, this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
    Axz.set(1, 0, this->coord_vtx_[1][2] - this->coord_vtx_[0][2]);
    Axz.set(0, 1, this->coord_vtx_[2][0] - this->coord_vtx_[0][0]);
    Axz.set(1, 1, this->coord_vtx_[2][2] - this->coord_vtx_[0][2]);

    Ayz.set(0, 0, this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
    Ayz.set(1, 0, this->coord_vtx_[1][2] - this->coord_vtx_[0][2]);
    Ayz.set(0, 1, this->coord_vtx_[2][1] - this->coord_vtx_[0][1]);
    Ayz.set(1, 1, this->coord_vtx_[2][2] - this->coord_vtx_[0][2]);
  }

  DataType det_xy = std::abs(det(Axy));
  DataType det_xz = std::abs(det(Axz));
  DataType det_yz = std::abs(det(Ayz));

  if ( (det_xz >= det_xy) && (det_xz >= det_yz) )
  {
    co0_ = 0;
    co1_ = 2;
    this->A_ = Axz;
  }
  else if ( (det_yz >= det_xy) && (det_yz >= det_xz) )
  {
    co0_ = 1;
    co1_ = 2;
    this->A_ = Ayz;
  }
  else 
  {
    co0_ = 0;
    co1_ = 1;
    this->A_ = Axy;
  }

  assert (det(A_) != 0.);
  inv(this->A_, this->A_inv_);

#ifndef NDEBUG 
  this->Axy_ = Axy;
  this->Axz_ = Axz;
  this->Ayz_ = Ayz;

  if (det_xy > 0.)
    inv(this->Axy_, this->Axy_inv_);
  if (det_xz > 0.)
    inv(this->Axz_, this->Axz_inv_);  
  if (det_yz > 0.)
    inv(this->Ayz_, this->Ayz_inv_); 

#endif 
}

template < class DataType, int RDIM, int PDIM >
void LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::reinit(const std::vector<DataType> &coord_vtx) 
{  
  SurfaceTransformation< DataType, RDIM, PDIM >::reinit(coord_vtx);
  this->setup_transformation_matrix();
}

template < class DataType, int RDIM, int PDIM >
void LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::reinit(const std::vector<PCoord> &coord_vtx) 
{  
  SurfaceTransformation< DataType, RDIM, PDIM >::reinit(coord_vtx);
  this->setup_transformation_matrix();
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::x(const RCoord &coord_ref) const 
{
  return (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]) * coord_ref[0] 
        +(this->coord_vtx_[2][0] - this->coord_vtx_[0][0]) * coord_ref[1] 
        + this->coord_vtx_[0][0];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::x_x(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[1][0] - this->coord_vtx_[0][0];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::x_y(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[2][0] - this->coord_vtx_[0][0];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >:: y(const RCoord & coord_ref) const 
{
  return (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]) * coord_ref[0] 
        +(this->coord_vtx_[2][1] - this->coord_vtx_[0][1]) * coord_ref[1] 
        + this->coord_vtx_[0][1];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::y_x(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[1][1] - this->coord_vtx_[0][1];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::y_y(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[2][1] - this->coord_vtx_[0][1];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >:: z(const RCoord & coord_ref) const 
{
  return (this->coord_vtx_[1][2] - this->coord_vtx_[0][2]) * coord_ref[0] 
        +(this->coord_vtx_[2][2] - this->coord_vtx_[0][2]) * coord_ref[1] 
        + this->coord_vtx_[0][2];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::z_x(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[1][2] - this->coord_vtx_[0][2];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::z_y(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[2][2] - this->coord_vtx_[0][2];
}

template < class DataType, int RDIM, int PDIM >
bool LinearTriangleSurfaceTransformation< DataType, RDIM, PDIM >::inverse_2Dto3D(const Coord<3, DataType>& co_phy, RCoord &co_ref) const 
{
  Coord<2, DataType> rhs;
  rhs.set(0, co_phy[co0_] - this->coord_vtx_[0][co0_]);
  rhs.set(1, co_phy[co1_] - this->coord_vtx_[0][co1_]);
  this->A_inv_.VectorMult(rhs, co_ref);

#ifndef NDEBUG
  const auto geom_tol = SurfaceTransformation< DataType, RDIM, PDIM >::GEOM_TOL;
  if (det(Axy_) != 0.)
  {
    Coord<2, DataType> rhs_xy;
    Coord<2, DataType> sol_xy;
    rhs_xy.set(0, co_phy[0] - this->coord_vtx_[0][0]);
    rhs_xy.set(1, co_phy[1] - this->coord_vtx_[0][1]);
    this->Axy_inv_.VectorMult(rhs_xy, sol_xy);
    //std::cout << "xy " << det(Axy_) << " : " << co0_ << " " << co1_ << " : " << co_ref << " - " << sol_xy << " = " << norm(co_ref - sol_xy) << std::endl;
    assert (norm(co_ref - sol_xy) < geom_tol || !this->contains_reference_point(sol_xy));
  }

  if (det(Axz_) != 0.)
  {
    Coord<2, DataType> rhs_xz;
    Coord<2, DataType> sol_xz;
    rhs_xz.set(0, co_phy[0] - this->coord_vtx_[0][0]);
    rhs_xz.set(1, co_phy[2] - this->coord_vtx_[0][2]);
    this->Axz_inv_.VectorMult(rhs_xz, sol_xz);
    //std::cout << "xz " << det(Axy_) << " : " << co0_ << " " << co1_ << " : " << co_ref << " - " << sol_xz << " = " << norm(co_ref - sol_xz) << std::endl;
    assert (norm(co_ref - sol_xz) < geom_tol || !this->contains_reference_point(sol_xz));
  }

  if (det(Ayz_) != 0.)
  {
    Coord<2, DataType> rhs_yz;
    Coord<2, DataType> sol_yz;
    rhs_yz.set(0, co_phy[1] - this->coord_vtx_[0][1]);
    rhs_yz.set(1, co_phy[2] - this->coord_vtx_[0][2]);
    this->Ayz_inv_.VectorMult(rhs_yz, sol_yz);
    //std::cout << "yz " << det(Axy_) << " : " << co0_ << " " << co1_ << " : " << co_ref << " - " << sol_yz << " = " << norm(co_ref - sol_yz) << std::endl;
    assert (norm(co_ref - sol_yz) < geom_tol || !this->contains_reference_point(sol_yz));
  }
#endif
  return true;
}

} // namespace doffem
} // namespace hiflow

#endif
