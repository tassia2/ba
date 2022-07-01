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

#ifndef __FEM_ALIGNED_QUAD_TRANSFORMATION_H_
#define __FEM_ALIGNED_QUAD_TRANSFORMATION_H_

//#include "fem/cell_trafo/bilinear_quad_transformation.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "mesh/geometric_tools.h"
#include "common/log.h"
#include "common/vector_algebra.h"
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
class LinearQuadTransformation final : public CellTransformation< DataType, DIM > {
public:
  using Coord = typename CellTransformation< DataType, DIM >::Coord;
  using mat = typename CellTransformation< DataType, DIM >::mat;

  explicit LinearQuadTransformation(CRefCellSPtr<DataType, DIM> ref_cell)
  ;
  explicit LinearQuadTransformation(CRefCellSPtr<DataType, DIM> ref_cell, 
                                     const std::vector< mesh::MasterSlave >& period);

  bool differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const;
  
  ~ LinearQuadTransformation()
  {
  }

  DataType cell_diameter() const override;
  
  void reinit(const std::vector<DataType> &coord_vtx);

  bool inverse_impl(const Coord& co_phy, Coord &co_ref) const;

  inline DataType x(const Coord &coord_ref) const;
  inline DataType x_x(const Coord &coord_ref) const;
  inline DataType x_y(const Coord &coord_ref) const;
  inline DataType y(const Coord &coord_ref) const;
  inline DataType y_x(const Coord &coord_ref) const;
  inline DataType y_y(const Coord &coord_ref) const;

protected:
  mat A_;
  mat Ainv_; 
};

// Reordering of vertices to make transformation coorespond to mesh
// ordering, with (0,0,0) mapped to vertex 0, and (1,1,1) mapped to vertex 7.
template < class DataType, int DIM >
LinearQuadTransformation< DataType, DIM >::LinearQuadTransformation(CRefCellSPtr<DataType, DIM> ref_cell)
  : CellTransformation< DataType, DIM >(ref_cell) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::QUAD_STD;
  this->name_ = "AlignedQuad";
  this->my_valid_dim_ = 2;
  this->my_nb_vertices_ = 4;
  
  //
  //        0 --------------- 3
  //       /        y        /
  //      /x                /
  //     /                 /
  //    1 --------------- 2

}

template < class DataType, int DIM >
LinearQuadTransformation< DataType, DIM >::LinearQuadTransformation(CRefCellSPtr<DataType, DIM> ref_cell, 
                                                                      const std::vector< mesh::MasterSlave >& period)
  : CellTransformation< DataType, DIM >(ref_cell, period) 
{
  this->order_ = 1;
  this->fixed_ref_cell_type_ = RefCellType::QUAD_STD;
  this->name_ = "AlignedQuad";
  this->my_valid_dim_ = 2;
  this->my_nb_vertices_ = 4;
}

template < class DataType, int DIM >
DataType LinearQuadTransformation< DataType, DIM >::cell_diameter () const 
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
bool LinearQuadTransformation< DataType, DIM >::differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const
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
void LinearQuadTransformation< DataType, DIM >::reinit(const std::vector<DataType> &coord_vtx) 
{
  assert (DIM == this->my_valid_dim_);
  
  assert (coord_vtx.size() == DIM * 4);
  
  this->coord_vtx_.clear();
  this->coord_vtx_.resize(4);
  
  for (int i=0; i<4; ++i)
  {
    for (int d=0; d<2; ++d)
    {
      this->coord_vtx_[i].set(d, coord_vtx[this->ij2ind(i,d)]);
    }
  }

  //assert (mesh::is_aligned_rectangular_quad(coord_vtx));
  assert (mesh::is_parallelogram(coord_vtx));
  
  this->A_.set(0, 0, this->coord_vtx_[1][0] - this->coord_vtx_[0][0]);
  this->A_.set(1, 0, this->coord_vtx_[1][1] - this->coord_vtx_[0][1]);
  this->A_.set(0, 1, this->coord_vtx_[3][0] - this->coord_vtx_[0][0]);
  this->A_.set(1, 1, this->coord_vtx_[3][1] - this->coord_vtx_[0][1]);

#ifndef NDEBUG   
  DataType determ = this->A_(0,0) * this->A_(1,1) - this->A_(0,1) * this->A_(1,0);
  DataType sca = this->A_(0,0) * this->A_(0,1) + this->A_(1,0) * this->A_(1,1); 
  
  //std::cout << this->A_ << std::endl;
  assert (std::abs(determ) > 1e-12);
  //assert (std::abs(sca) < 1e-10);
#endif

  inv(this->A_, this->Ainv_);
}

template < class DataType, int DIM >
bool LinearQuadTransformation< DataType, DIM >::inverse_impl(const Coord& co_phy, Coord &co_ref) const 
{
  
  Coord rhs = co_phy - this->coord_vtx_[0];
  this->Ainv_.VectorMult(rhs, co_ref);
  
  return true;
}

template < class DataType, int DIM >
DataType LinearQuadTransformation< DataType, DIM >::x(const Coord &coord_ref) const {
    return this->coord_vtx_[0][0] + this->A_(0,0) * coord_ref[0] + this->A_(0,1) * coord_ref[1]; 
}

template < class DataType, int DIM >
DataType LinearQuadTransformation< DataType, DIM >::x_x(const Coord &coord_ref) const {
    return this->A_(0,0);
}

template < class DataType, int DIM >
DataType LinearQuadTransformation< DataType, DIM >::x_y(const Coord &coord_ref) const {
    return this->A_(0,1);
}

template < class DataType, int DIM >
DataType LinearQuadTransformation< DataType, DIM >::y(const Coord &coord_ref) const {
    return this->coord_vtx_[0][1] + this->A_(1,0) * coord_ref[0] + this->A_(1,1) * coord_ref[1]; 
}

template < class DataType, int DIM >
DataType LinearQuadTransformation< DataType, DIM >::y_x(const Coord &coord_ref) const {
    return this->A_(1,0); 
}

template < class DataType, int DIM >
DataType LinearQuadTransformation< DataType, DIM >::y_y(const Coord &coord_ref) const {
    return this->A_(1,1); 
}

} // namespace doffem
} // namespace hiflow

#endif
