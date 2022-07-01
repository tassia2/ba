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

#ifndef __FEM_LINEAR_LINE_SURFACE_TRANSFORMATION_H_
#define __FEM_LINEAR_LINE_SURFACE_TRANSFORMATION_H_

#include "fem/cell_trafo/surface_transformation.h"
#include "common/vector_algebra_descriptor.h"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace hiflow {
namespace doffem {

///
/// \class LinearLineTransformation linear_line_transformation.h
/// \brief Linear transformation mapping from reference to physical cell for a
/// Line \author Philipp Gerstner
///

template < class DataType, int RDIM, int PDIM >
class LinearLineSurfaceTransformation final : public SurfaceTransformation< DataType, RDIM, PDIM > {
public:
  using RCoord = typename SurfaceTransformation< DataType, RDIM, PDIM >::RCoord;
  using PCoord = typename SurfaceTransformation< DataType, RDIM, PDIM >::PCoord;
  using RRmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::RRmat;
  using PPmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::PPmat;
  using RPmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::RPmat;
  using PRmat = typename SurfaceTransformation< DataType, RDIM, PDIM >::PRmat;

  template <int N, typename ScalarType>
  using Coord = typename StaticLA<N, N, ScalarType>::RowVectorType;

  explicit LinearLineSurfaceTransformation(CRefCellSPtr<DataType, RDIM> ref_cell);
      
  bool inverse_1Dto2D(const Coord<2, DataType>& co_phy, RCoord &co_ref) const;
  bool inverse_1Dto3D(const Coord<3, DataType>& co_phy, RCoord &co_ref) const;

  inline DataType x  (const RCoord &coord_ref) const override;
  inline DataType x_x(const RCoord &coord_ref) const override;
  inline DataType y  (const RCoord &coord_ref) const override;
  inline DataType y_x(const RCoord &coord_ref) const override;
  inline DataType z  (const RCoord &coord_ref) const override;
  inline DataType z_x(const RCoord &coord_ref) const override;

protected:
  bool inverse_impl(const PCoord& co_phy, RCoord &co_ref) const;

};

template < class DataType, int RDIM, int PDIM >
LinearLineSurfaceTransformation< DataType, RDIM, PDIM >::LinearLineSurfaceTransformation(CRefCellSPtr<DataType, RDIM> ref_cell)
    : SurfaceTransformation< DataType, RDIM, PDIM >(ref_cell) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::LINE_STD;
  this->name_ = "Line";
  this->my_valid_rdim_ = 1;
  this->my_nb_vertices_ = 2;
}

template < class DataType, int RDIM, int PDIM >
DataType LinearLineSurfaceTransformation< DataType, RDIM, PDIM >::x(const RCoord &coord_ref) const 
{
  return coord_ref[0] * (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]) + this->coord_vtx_[0][0];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearLineSurfaceTransformation< DataType, RDIM, PDIM >::x_x(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[1][0] - this->coord_vtx_[0][0];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearLineSurfaceTransformation< DataType, RDIM, PDIM >:: y(const RCoord & coord_ref) const 
{
  return coord_ref[0] * (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]) + this->coord_vtx_[0][1];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearLineSurfaceTransformation< DataType, RDIM, PDIM >::y_x(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[1][1] - this->coord_vtx_[0][1];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearLineSurfaceTransformation< DataType, RDIM, PDIM >:: z(const RCoord & coord_ref) const 
{
  return coord_ref[0] * (this->coord_vtx_[1][2] - this->coord_vtx_[0][2]) + this->coord_vtx_[0][2];
}

template < class DataType, int RDIM, int PDIM >
DataType LinearLineSurfaceTransformation< DataType, RDIM, PDIM >::z_x(const RCoord &coord_ref) const 
{
  return this->coord_vtx_[1][2] - this->coord_vtx_[0][2];
}

//The inverse function can actually remain the same. Because all the rows in the linear system are pairwise linearly dependent, it suffices to solve one
//However, one can check if the physial point actually lies in the span of the physical cell by checking if the solutions provided by each row individually coincide.

template < class DataType, int RDIM, int PDIM >
bool LinearLineSurfaceTransformation< DataType, RDIM, PDIM >::inverse_1Dto2D(const Coord<2, DataType>& co_phy, RCoord &co_ref) const 
{
  DataType sol_0 = (co_phy[0] - this->coord_vtx_[0][0]) / (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
  DataType sol_1 = (co_phy[1] - this->coord_vtx_[0][1]) / (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
  
  if (std::abs(sol_0 - sol_1) > SurfaceTransformation< DataType, RDIM, PDIM >::GEOM_TOL) 
  {
    return false;
  }
  co_ref.set(0, sol_0);
  return true;
}

template < class DataType, int RDIM, int PDIM >
bool LinearLineSurfaceTransformation< DataType, RDIM, PDIM >::inverse_1Dto3D(const Coord<3, DataType>& co_phy, RCoord &co_ref) const 
{
  DataType sol_0 = (co_phy[0] - this->coord_vtx_[0][0]) / (this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
  DataType sol_1 = (co_phy[1] - this->coord_vtx_[0][1]) / (this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
  DataType sol_2 = (co_phy[2] - this->coord_vtx_[0][2]) / (this->coord_vtx_[1][2] - this->coord_vtx_[0][2]);
  
  if ((std::abs(sol_0 - sol_1) > SurfaceTransformation< DataType, RDIM, PDIM >::GEOM_TOL) 
    || (std::abs(sol_0 - sol_2) > SurfaceTransformation< DataType, RDIM, PDIM >::GEOM_TOL) 
    || (std::abs(sol_1 - sol_2) > SurfaceTransformation< DataType, RDIM, PDIM >::GEOM_TOL) )
  { 
    return false;
  }
  co_ref.set(0, sol_0);
  return true;
}


} // namespace doffem
} // namespace hiflow

#endif
