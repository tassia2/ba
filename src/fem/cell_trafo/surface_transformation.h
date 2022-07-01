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

#ifndef __FEM_SURFACE_TRANSFORMATION_H_
#define __FEM_SURFACE_TRANSFORMATION_H_

#include <cassert>
#include <vector>
#include <limits>
#include <cmath>

#include "common/log.h"
#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"
#include "dof/dof_fem_types.h"
#include "fem/reference_cell.h"
#include "fem/cell_trafo/cell_trafo_inverse.h"
//#include "mesh/geometric_tools.h"

namespace hiflow {
namespace doffem {

enum SurfaceTrafoType
{
  SURFACE_TRAFO_NONE = -1,
  SURFACE_TRAFO_LINEARLINE = 0,
  SURFACE_TRAFO_LINEARTRI = 1,
  SURFACE_TRAFO_LINEARQUAD = 2,
  SURFACE_TRAFO_BILINEARQUAD = 3
};

using STrafoString = StaticString<50>;

//const double GEOM_TOL = 1.e-14;

///
/// \class SurfaceTransformation surface_transformation.h
/// \brief Ancestor class of all transformation mappings from reference to
/// physical cells in different spaces
/// \author Philipp Gerstner
///

template < class DataType, int RDIM, int PDIM > 
class SurfaceTransformation 
{

public:
  using RCoord = typename StaticLA<RDIM, PDIM, DataType>::ColVectorType;
  using PCoord = typename StaticLA<RDIM, PDIM, DataType>::RowVectorType;
  template <int N>
  using Coord = typename StaticLA<N, N, DataType>::RowVectorType;

  using RRmat = typename StaticLA<RDIM, RDIM, DataType>::MatrixType;
  using PPmat = typename StaticLA<PDIM, PDIM, DataType>::MatrixType;
  using RPmat = typename StaticLA<RDIM, PDIM, DataType>::MatrixType;
  using PRmat = typename StaticLA<PDIM, RDIM, DataType>::MatrixType;

  static constexpr double INVERSE_RESIDUAL_TOL = 1000. * std::numeric_limits< double >::epsilon();
  static constexpr const double GEOM_TOL = 1.e-14;  
  
  /// Use this constructor which needs the geometrical dimension as input
  explicit SurfaceTransformation(CRefCellSPtr<DataType, RDIM> ref_cell);

  virtual ~SurfaceTransformation() 
  {
  }

  STrafoString name() const 
  {
    return this->name_;
  }
   
  /// Reinitialization of the transformation via coordinates of physical cell
  virtual void reinit(const std::vector<DataType> &coord_vtx);
  virtual void reinit(const std::vector<PCoord> &coord_vtx);

  CRefCellSPtr<DataType, RDIM> get_ref_cell() const
  {
    return this->ref_cell_;
  }

  inline int num_vertices() const 
  {
    return my_nb_vertices_;
  }

  std::vector< RCoord > get_reference_coordinates() const 
  { 
    assert(this->ref_cell_);
    return this->ref_cell_->get_coords();
  }

  std::vector<PCoord> get_coordinates() const 
  { 
    return this->coord_vtx_;
  }

  PCoord get_coordinate(int vertex_id) const 
  {
    assert (vertex_id >= 0);
    assert (vertex_id < this->coord_vtx_.size());
    return this->coord_vtx_[vertex_id];
  }

  /// \brief Check whether a given point is contained in the closure of the
  /// cell.
  ///
  /// \param[in] coord_ref    reference coordinates of the point
  /// \returns  True if reference coordinates are contained in the cell.
  bool contains_reference_point(const RCoord &coord_ref) const
  {
    assert(this->ref_cell_);
    return this->ref_cell_->contains_point (coord_ref);
  }

  /// \brief Check whether a given point is contained in the closure of the
  /// cell.
  /// \param[in]  coord_phys   Physical coordinates of the point.
  /// \param[out] coord_ref    Optional output parameter, where reference
  /// coordinates in the cell are stored, if the function returns true. \returns
  /// True if reference coordinates are contained in the cell.
  virtual bool contains_physical_point(const PCoord &coord_phys, RCoord &coord_ref) const;

  bool residual_inverse_check (const PCoord &co_phy, const RCoord &co_ref ) const;

  void transform ( const RCoord& coord_ref, PCoord& coord_mapped ) const;
  
  PCoord transform ( const RCoord& coord_ref ) const
  {
    PCoord coord_mapped;
    this->transform(coord_ref, coord_mapped);
    return coord_mapped;
  }

  void J (const RCoord &coord_ref, PRmat& J) const;

  void H (const RCoord &coord_ref, size_t d, RRmat& H) const;

  void normal (const RCoord &coord_ref, PCoord& n) const;

  /// \brief Given physical cell coordinates in 1D,
  ///        this routine computes the corresponding reference cell coordinates
  /// @return true, if inverse computation was successful
  virtual bool inverse(const PCoord& co_phy, RCoord &co_ref) const;

  /// Given reference coordinates, this routine computes the physical x
  /// coordinates
  virtual DataType x(const RCoord &coord_ref) const { return 0.; }
  
  /// \brief Given reference coordinates, this routine computes the derivatives
  /// of the mapping (ref_coordinates to physical x value)
  virtual DataType x_x(const RCoord &coord_ref) const { return 0.; }
  virtual DataType x_y(const RCoord &coord_ref) const { return 0.; }
  
  /// \brief Given reference coordinates, these routine compute the second
  /// derivatives of the mapping (ref_coordinates to physical x value). 
  virtual DataType x_xx(const RCoord &coord_ref) const { return 0.; }
  virtual DataType x_xy(const RCoord &coord_ref) const { return 0.; }
  virtual DataType x_yy(const RCoord &coord_ref) const { return 0.; }
  
  /// Given reference coordinates, this computes the physical y coordinates
  virtual DataType y(const RCoord &coord_ref) const { return 0.; }
  
  /// \brief Given reference coordinates, this routine computes the derivatives
  /// of the mapping (ref_coordinates to physical y value)
  virtual DataType y_x(const RCoord &coord_ref) const { return 0.; }
  virtual DataType y_y(const RCoord &coord_ref) const { return 0.; }

  /// \brief Given reference coordinates, this routine computes the second
  /// derivatives of the mapping (ref_coordinates to physical y value)
  virtual DataType y_xx(const RCoord &coord_ref) const { return 0.; }
  virtual DataType y_xy(const RCoord &coord_ref) const { return 0.; }
  virtual DataType y_yy(const RCoord &coord_ref) const { return 0.; }

  /// Given reference coordinates, this computes the physical z coordinates
  virtual DataType z(const RCoord &coord_ref) const { return 0.; }
  /// \brief Given reference coordinates, this routine computes the derivatives
  /// in of the mapping (ref_coordinates to physical z value).
  /// Return value is a dummy value for 2D problems, but neccessary for UnitIntegrator

  /// \brief Given reference coordinates, this routine computes the derivatives
  /// of the mapping (ref_coordinates to physical z value).
  /// Return value is a dummy value for 2D problems, but neccessary for
  /// UnitIntegrator
  virtual DataType z_x(const RCoord &coord_ref) const { return 0.; }
  virtual DataType z_y(const RCoord &coord_ref) const { return 0.; }

  /// \brief Given reference coordinates, this routine computes the second
  /// derivatives  of the mapping (ref_coordinates to physical z
  /// value). Return value is a dummy value for 2D problems, but
  /// neccessary for UnitIntegrator
  virtual DataType z_xx(const RCoord &coord_ref) const { return 0.; }
  virtual DataType z_xy(const RCoord &coord_ref) const { return 0.; }
  virtual DataType z_yy(const RCoord &coord_ref) const { return 0.; }

protected:
  virtual bool inverse_impl(const PCoord& co_phy, RCoord &co_ref) const = 0;
  int my_valid_rdim_ = -1;
  int my_nb_vertices_ = -1;

  /// \details The vector index is calculated by an offset of the
  ///          magnitude i * geometrical dimension
  /// \param[in] i index of vertex id
  /// \param[in] j index of coordinate id (0 for x, 1 for y and 2 for z)
  /// \return index for vector of coordinates coord_vtx_

  inline int ij2ind(int i, int j) const 
  { 
    return i * PDIM + j; 
  }

  CRefCellSPtr<DataType, RDIM> ref_cell_;
  RefCellType fixed_ref_cell_type_;

  SurfaceTrafoType trafo_type_;
  
  /// Vector, which holds the coordinates of every vertex of the physical surface
  std::vector<PCoord> coord_vtx_;
  
  /// highest polynomial order 
  int order_;
  
  STrafoString name_;
};

template < class DataType, int RDIM, int PDIM >
SurfaceTransformation< DataType, RDIM, PDIM >::SurfaceTransformation(CRefCellSPtr<DataType, RDIM> ref_cell) 
: ref_cell_(ref_cell), 
order_(10),
trafo_type_(SURFACE_TRAFO_NONE) 
{
  assert (RDIM > 0);
  assert (PDIM > 0);
  assert (RDIM <= 2);
  assert (PDIM <= 3);
  assert (RDIM <= PDIM);
}

/// \details Given vector of coordinates on physical cell, the are
///          stored the protected member variable coord_vtx_

template < class DataType, int RDIM, int PDIM >
void SurfaceTransformation< DataType, RDIM, PDIM >::reinit(const std::vector<DataType> &coord_vtx) 
{  
  assert (RDIM == this->my_valid_rdim_);
  assert (this->fixed_ref_cell_type_ == this->ref_cell_->type());
  
  int num_points = coord_vtx.size() / PDIM;
  coord_vtx_.clear();
  coord_vtx_.resize(num_points);
  for (int i=0; i<num_points; ++i) 
  {
    for (int d=0; d<PDIM; ++d) 
    {
      coord_vtx_[i].set(d, coord_vtx[i*PDIM + d]);
    } 
  }
}

template < class DataType, int RDIM, int PDIM >
void SurfaceTransformation< DataType, RDIM, PDIM >::reinit(const std::vector<PCoord> &coord_vtx) 
{  
  assert (this->fixed_ref_cell_type_ == this->ref_cell_->type());
  this->coord_vtx_ = coord_vtx;
}

template < class DataType, int RDIM, int PDIM >
bool SurfaceTransformation< DataType, RDIM, PDIM >::contains_physical_point(const PCoord &coord_phys, RCoord& cr) const 
{
  bool found_ref_point = this->inverse(coord_phys, cr);

  if (!found_ref_point) 
  {
    return false;
  }
  return this->contains_reference_point(cr);
}

template<class DataType, int RDIM, int PDIM>
bool SurfaceTransformation<DataType, RDIM, PDIM>::inverse(const PCoord &co_phy, RCoord &co_ref ) const
{  
  // compute inverse cell trafo
  // true if inverse compuation was successfull. 
  // However, this does not mean that the co_phy is really conatined in the physical cell
  for (int d=0; d!= RDIM; ++d)
    co_ref.set(d, -1.);

  bool found_ref_pt = this->inverse_impl(co_phy, co_ref);

  // avoid rounding errors on ref cell bdy 
  const DataType eps = this->ref_cell_->eps();
  for (int i = 0; i != RDIM; ++i)
  {
    if (std::abs(co_ref[i]) < eps) 
    {
      co_ref.set(i, 0.);
    } 
    else if (std::abs(co_ref[i] - 1.) < eps) 
    {
      co_ref.set(i, 1.);
    }
  }
  
  // check if ref point is in ref cell
  bool found = found_ref_pt && this->contains_reference_point(co_ref);

  if (found)
  {
    return true;
  }
  return false;
}

template<class DataType, int RDIM, int PDIM>
bool SurfaceTransformation<DataType, RDIM, PDIM>::residual_inverse_check (const PCoord &co_phy, const RCoord &co_ref ) const
{ 
  DataType residual = norm(this->transform(co_ref) - co_phy);
  LOG_DEBUG(2, "residual of inverse compuation " << residual << " vs tolerance " << INVERSE_RESIDUAL_TOL);

  if (residual < INVERSE_RESIDUAL_TOL)
  {
    return true;
  }
  return false;
}

template<class DataType, int RDIM, int PDIM>
void SurfaceTransformation<DataType, RDIM, PDIM>::transform ( const RCoord& coord_ref, PCoord& coord_mapped ) const 
{
  if constexpr (PDIM == 1)
  {
    coord_mapped.set(0, this->x ( coord_ref ));
  }
  else if constexpr (PDIM == 2)
  {
    coord_mapped.set(0, this->x ( coord_ref ));
    coord_mapped.set(1, this->y ( coord_ref ));
  }
  else if constexpr (PDIM == 3)
  {
    coord_mapped.set(0, this->x ( coord_ref ));
    coord_mapped.set(1, this->y ( coord_ref ));
    coord_mapped.set(2, this->z ( coord_ref ));
  }
  else 
  {
    assert(false);
  }
}

template<class DataType, int RDIM, int PDIM>
void SurfaceTransformation<DataType, RDIM, PDIM>::J ( const RCoord& coord_ref, PRmat& mat) const 
{
  if constexpr (PDIM == 1)
  {
    mat.set(0, 0, this->x_x(coord_ref));
  }
  else if constexpr (PDIM == 2)
  {
    if constexpr (RDIM == 1)
    {
      mat.set(0, 0, this->x_x(coord_ref));
      mat.set(1, 0, this->y_x(coord_ref));
    }
    else if constexpr (RDIM == 2)
    {
      mat.set(0, 0, this->x_x(coord_ref));
      mat.set(0, 1, this->x_y(coord_ref));
      mat.set(1, 0, this->y_x(coord_ref));
      mat.set(1, 1, this->y_y(coord_ref));
    }
    else 
    {
      assert (false);
    }
  }
  else if constexpr (PDIM == 3)
  {
    if constexpr (RDIM == 1)
    {
      mat.set(0, 0, this->x_x(coord_ref));
      mat.set(1, 0, this->y_x(coord_ref));
      mat.set(2, 0, this->z_x(coord_ref));
    }
    else if constexpr (RDIM == 2)
    {
      mat.set(0, 0, this->x_x(coord_ref));
      mat.set(0, 1, this->x_y(coord_ref));
      
      mat.set(1, 0, this->y_x(coord_ref));
      mat.set(1, 1, this->y_y(coord_ref));

      mat.set(2, 0, this->z_x(coord_ref));
      mat.set(2, 1, this->z_y(coord_ref));
    }
    else 
    {
      assert (false);
    }
  }
  else 
  {
    assert(false);
  }
}

template<class DataType, int RDIM, int PDIM>
void SurfaceTransformation<DataType, RDIM, PDIM>::normal ( const RCoord& coord_ref, PCoord& normal) const 
{
  if constexpr (PDIM == 2)
  {
    if constexpr (RDIM == 1)
    {
      normal.set(0, - this->y_x(coord_ref));
      normal.set(1, this->x_x(coord_ref));

      DataType scale = 1. / norm(normal);
      normal *= scale;
    }
    else 
    {
      assert (false);
    }
  }
  else if constexpr (PDIM == 3)
  {
    if constexpr (RDIM == 1)
    {
      NOT_YET_IMPLEMENTED;
    }
    else if constexpr (RDIM == 2)
    {
      PCoord tangent_1;
      PCoord tangent_2;

      tangent_1.set(0, this->x_x(coord_ref));
      tangent_1.set(1, this->y_x(coord_ref));
      tangent_1.set(2, this->z_x(coord_ref));

      tangent_2.set(0, this->x_y(coord_ref));
      tangent_2.set(1, this->y_y(coord_ref));
      tangent_2.set(2, this->z_y(coord_ref));

      normal = cross(tangent_1, tangent_2);
      DataType scale = 1. / norm(normal);
      normal *= scale;
    }
    else 
    {
      assert (false);
    }
  }
  else 
  {
    assert (false);
  }
}

template<class DataType, int RDIM, int PDIM>
void SurfaceTransformation<DataType, RDIM, PDIM>::H ( const RCoord& coord_ref, size_t d, RRmat& mat) const 
{
  assert (d<PDIM);
  
  if constexpr (RDIM == 1)
  {
    switch (d)
    {
      case 0:
        mat.set(0, 0, this->x_xx(coord_ref));
        break;
      case 1:
        mat.set(0, 0, this->y_xx(coord_ref));
        break;
      case 2:
        mat.set(0, 0, this->z_xx(coord_ref));
        break;
      default:
        assert(0);
        break;
    }
  }
  else if constexpr (RDIM == 2)
  {
    switch (d)
    {
      case 0:
        mat.set(0, 0, this->x_xx(coord_ref));
        mat.set(0, 1, this->x_xy(coord_ref));
        mat.set(1, 0, mat(0,1));
        mat.set(1, 1, this->x_yy(coord_ref));
        break;
      case 1:
        mat.set(0, 0, this->y_xx(coord_ref));
        mat.set(0, 1, this->y_xy(coord_ref));
        mat.set(1, 0, mat(0,1));
        mat.set(1, 1, this->y_yy(coord_ref));
        break;
      case 2:
        mat.set(0, 0, this->z_xx(coord_ref));
        mat.set(0, 1, this->z_xy(coord_ref));
        mat.set(1, 0, mat(0,1));
        mat.set(1, 1, this->z_yy(coord_ref));
        break;
      default:
        assert(0);
        break;
    }
  }
  else 
  {
    assert (false);
  }
}

} // namespace doffem
} // namespace hiflow

#endif
