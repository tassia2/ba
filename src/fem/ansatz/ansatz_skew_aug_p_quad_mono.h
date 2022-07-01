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
// <https://joinup.ec.europa.eu/page/eupl-text-11-12>.dfdfdfdfdf

#ifndef __ANSATZ_SKEW_AUG_PQUAD_MONO_H_
#define __ANSATZ_SKEW_AUG_PQUAD_MONO_H_

#include "fem/ansatz/ansatz_space.h"
#include "common/vector_algebra_descriptor.h"
#include <cassert>
#include <cmath>

namespace hiflow {
namespace doffem {

///
/// \class SkewAugPQuadMono ansatz_skew_aug_p_quad_mono.h
/// \brief r * curl[x^(k+1)y] + s * curl[xy^(k+1)], needed
/// for BDM elements on quadrilaterals
/// \author Jonas Roller
///

template < class DataType, int DIM >
class SkewAugPQuadMono final : public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  SkewAugPQuadMono(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~SkewAugPQuadMono()
  {
  }

  void init (size_t degree);

  void N(const Coord &pt, std::vector< DataType > &weight) const override;

  void N_x(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_y(const Coord &pt, std::vector< DataType > &weight) const override;

  void N_xx(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_xy(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_yy(const Coord &pt, std::vector< DataType > &weight) const override;

private:
  void compute_degree_hash () const
  {
    NOT_YET_IMPLEMENTED;
  }

  void set_xk_yk (const Coord& pt) const;
  
  mutable std::vector<DataType> xk_;
  mutable std::vector<DataType> yk_;
  
  size_t deg_;
};


template < class DataType, int DIM > 
SkewAugPQuadMono< DataType, DIM >::SkewAugPQuadMono(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 2;
  this->name_ = "Skew_Aug_Q_Quad_Mono";
  this->type_ = AnsatzSpaceType::SKEW_P_AUG;

  assert (this->ref_cell_->type() == RefCellType::QUAD_STD);
}

template < class DataType, int DIM > 
void SkewAugPQuadMono< DataType, DIM >::init ( size_t degree )
{
  assert (DIM == 2);
  this->tdim_ = 2;
  this->nb_comp_ = 2;
  this->max_deg_ = degree+1;
  this->dim_ = 2;
  this->deg_ = degree;

  this->comp_weight_size_.clear();
  this->comp_weight_size_.resize(this->nb_comp_, 2);

  this->comp_offset_.clear();
  this->comp_offset_.resize(this->nb_comp_, 0);
  this->comp_offset_[1] = this->dim_;
  
  this->weight_size_ = 2 * this->dim_;
  this->xk_.clear();
  this->yk_.clear();
  this->xk_.resize(this->deg_ + 2, 1.);
  this->yk_.resize(this->deg_ + 2, 1.);
}

template < class DataType, int DIM > 
void SkewAugPQuadMono< DataType, DIM >::set_xk_yk ( const Coord &pt ) const
{
  const DataType x = pt[0];
  const DataType y = pt[1];
  
  // x = pt[0], y = pt[1]
  // xk = [1, x, x², x³, ....., x^(deg+1) ]
  // yk = [1, y, y², y³, ....., y^(deg+1) ]

  this->xk_[0] = 1.;
  this->yk_[0] = 1.;
  
  for (size_t k = 1; k <= this->deg_+1; ++k)
  {
    this->xk_[k] = this->xk_[k-1] * x;
    this->yk_[k] = this->yk_[k-1] * y;
  }  
}

template < class DataType, int DIM > 
void SkewAugPQuadMono< DataType, DIM >::N(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk(pt);
  
  weight[0] = this->xk_[this->deg_+1];
  weight[this->comp_offset_[1]] = -1. * (deg+1.) * this->xk_[this->deg_] * this->yk_[1];

  weight[1] = (deg+1.) * this->xk_[1] * this->yk_[this->deg_];
  weight[1 + this->comp_offset_[1]] = -1. * this->yk_[this->deg_+1];
}

template < class DataType, int DIM > 
void SkewAugPQuadMono< DataType, DIM >::N_x(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk(pt);
  
  weight[0] = (deg+1.) * this->xk_[this->deg_];
  if (this->deg_ > 0)
    weight[this->comp_offset_[1]] = -1. * (deg+1.) * deg * this->xk_[this->deg_-1] * this->yk_[1];
  else  
    weight[this->comp_offset_[1]] = 0.0;

  weight[1] = (deg+1.) * this->yk_[this->deg_];
  weight[1 + this->comp_offset_[1]] = 0.0;
}

template < class DataType, int DIM > 
void SkewAugPQuadMono< DataType, DIM >::N_y(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk(pt);
  
  weight[0] = 0.0;
  weight[this->comp_offset_[1]] = -1. * (deg+1.) * this->xk_[this->deg_];

  if (this->deg_ > 0)
    weight[1] = (deg+1.) * deg * this->xk_[1] * this->yk_[this->deg_-1];
  else
    weight[1] = 0.0;
  weight[1 + this->comp_offset_[1]] = -1. * (deg+1.) * this->yk_[this->deg_];
}

template < class DataType, int DIM > 
void SkewAugPQuadMono< DataType, DIM >::N_xx(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk(pt);
 
  if (this->deg_ > 0) 
    weight[0] = (deg+1.) * deg * this->xk_[this->deg_-1];
  else
    weight[0] = 0.0;

  if(this->deg_ > 1)
    weight[this->comp_offset_[1]] = -1. * (deg+1.) * deg * (deg-1.) * this->xk_[this->deg_-2] * this->yk_[1];
  else  
    weight[this->comp_offset_[1]] = 0.0;

  weight[1] = 0.0;
  weight[1 + this->comp_offset_[1]] = 0.0;
}

template < class DataType, int DIM > 
void SkewAugPQuadMono< DataType, DIM >::N_xy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk(pt);
  
  weight[0] = 0.0;
  
  if (this->deg_ > 0)
    weight[this->comp_offset_[1]] = -1. * (deg+1.) * deg * this->xk_[this->deg_-1];
  else  
    weight[this->comp_offset_[1]] = 0.0;

  if (this->deg_ > 0)
    weight[1] = (deg+1.) * deg * this->yk_[this->deg_-1];
  else
    weight[1] = 0.0;
  weight[1 + this->comp_offset_[1]] = 0.0;
}

template < class DataType, int DIM > 
void SkewAugPQuadMono< DataType, DIM >::N_yy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk(pt);
  
  weight[0] = 0.0;
  weight[this->comp_offset_[1]] = 0.0;

  if (this->deg_ > 1)
    weight[1] = (deg+1.) * deg * (deg-1.) * this->xk_[1] * this->yk_[this->deg_-2];
  else
    weight[1] = 0.0;

  if (this->deg_ > 0)
    weight[1 + this->comp_offset_[1]] = -1. * (deg+1.) * deg * this->yk_[this->deg_-1];
  else  
    weight[1 + this->comp_offset_[1]] = 0.0;
}

} // namespace doffem
} // namespace hiflow
#endif
