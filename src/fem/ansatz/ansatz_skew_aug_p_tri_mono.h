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

#ifndef __ANSATZ_SKEW_AUG_PTRI_MONO_H_
#define __ANSATZ_SKEW_AUG_PTRI_MONO_H_

#include "fem/ansatz/ansatz_space.h"
#include "common/vector_algebra_descriptor.h"
#include <cassert>
#include <cmath>

namespace hiflow {
namespace doffem {

///
/// \class SkewAugPTriMono ansatz_skew_aug_p_tri_mono.h
/// \brief (y, -x) * \bar{P}_k with \bar{P}_k = [x^k, x^(k-1)y, x^(k-2)y^2, .... , x, y^(k-1), y^k] 
/// \author Philipp Gerstner
///

template < class DataType, int DIM >
class SkewAugPTriMono final: public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  SkewAugPTriMono(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~SkewAugPTriMono()
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
SkewAugPTriMono< DataType, DIM >::SkewAugPTriMono(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 2;
  this->name_ = "Skew_Aug_P_Tri_Mono";
  this->type_ = AnsatzSpaceType::SKEW_P_AUG;

  assert (this->ref_cell_->type() == RefCellType::TRI_STD);
}

template < class DataType, int DIM > 
void SkewAugPTriMono< DataType, DIM >::init ( size_t degree )
{
  assert (DIM == 2);
  this->tdim_ = 2;
  this->nb_comp_ = 2;
  this->max_deg_ = degree+1;
  this->dim_ = degree+1;
  this->deg_ = degree;

  this->comp_weight_size_.clear();
  this->comp_weight_size_.resize(this->nb_comp_, degree+1);

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
void SkewAugPTriMono< DataType, DIM >::set_xk_yk ( const Coord &pt ) const
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
void SkewAugPTriMono< DataType, DIM >::N(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  this->set_xk_yk(pt);

  // loop over ansatz functions (dim = deg+1)
  for (size_t j=0; j<this->deg_+1; ++j)
  {
    // x component: x^(deg - j) * y^(j+1) 
    weight[j] = this->xk_[this->deg_-j] * this->yk_[j+1];     
   
    // y component: -x^(deg+1-j) * y^(j)
    weight[this->dim_ + j] = - this->xk_[this->deg_+1-j] * this->yk_[j];  
  }
}

template < class DataType, int DIM > 
void SkewAugPTriMono< DataType, DIM >::N_x(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  this->set_xk_yk(pt);
  const DataType deg = static_cast<DataType> (this->deg_);

  // loop over ansatz functions (dim = deg+1)
  for (size_t j=0; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType>(j);

    // x component: (deg-j) * x^(deg - 1 - j) * y^(j+1) 
    weight[j] = (deg-jv) * this->xk_[this->deg_-j-1] * this->yk_[j+1];     
   
    // y component: -(deg-j+1) * x^(deg-j) * y^(j)
    weight[this->dim_ + j] = - (deg+1-jv) * this->xk_[this->deg_-j] * this->yk_[j];  
  }
  weight[this->deg_] = 0.;
  weight[this->dim_+this->deg_] = - this->yk_[this->deg_];
}

template < class DataType, int DIM > 
void SkewAugPTriMono< DataType, DIM >::N_y(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  this->set_xk_yk(pt);
  
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=1; j<=this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType>(j);

    // x component: (j+1) * x^(deg - j) * y^(j) 
    weight[j] = (jv + 1.) * this->xk_[this->deg_-j] * this->yk_[j];     
   
    // y component: - j * x^(deg+1-j) * y^(j-1)
    weight[this->dim_ + j] = - jv * this->xk_[this->deg_-j+1] * this->yk_[j-1];  
  }
  weight[0] = this->xk_[this->deg_];
  weight[this->dim_] = 0.;
}

template < class DataType, int DIM > 
void SkewAugPTriMono< DataType, DIM >::N_xx(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  this->set_xk_yk(pt);
  const DataType deg = static_cast<DataType> (this->deg_);
 
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=0; j<this->deg_-1; ++j)
  {
    const DataType jv = static_cast<DataType>(j);
    // x component: (deg-j) * (deg - 1 - j) * x^(deg - 2 - j) * y^(j+1) 
    weight[j] = (deg-jv) * (deg - 1 - jv) * this->xk_[this->deg_-j-2] * this->yk_[j+1];     
   
    // y component: -(deg-j+1) * (deg - j) * x^(deg-1-j) * y^(j)
    weight[this->dim_ + j] = - (deg+1-jv) * (deg - jv) * this->xk_[this->deg_-j-1] * this->yk_[j];  
  }

  weight[this->deg_-1] = 0.;
  weight[this->deg_] = 0.;
  weight[this->dim_+this->deg_-1] = - 2. * this->yk_[this->deg_-1];
  weight[this->dim_+this->deg_] = 0.;
}

template < class DataType, int DIM > 
void SkewAugPTriMono< DataType, DIM >::N_xy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  this->set_xk_yk(pt);
  const DataType deg = static_cast<DataType> (this->deg_);
 
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=1; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j); 
    // x component: (deg-j) * (j+1) * x^(deg - 1 - j) * y^(j) 
    weight[j] = (deg-jv) * (jv+1.) * this->xk_[this->deg_-j-1] * this->yk_[j];     
   
    // y component: -(deg-j+1) * j * x^(deg-j) * y^(j-1)
    weight[this->dim_ + j] = - (deg+1-jv) * jv * this->xk_[this->deg_-j] * this->yk_[j-1];  
  }
  weight[0] = deg * this->xk_[this->deg_-1];
  weight[this->dim_] = 0.;

  weight[this->deg_] = 0.;
  weight[this->dim_+this->deg_] = - deg * this->yk_[this->deg_];
}

template < class DataType, int DIM > 
void SkewAugPTriMono< DataType, DIM >::N_yy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  this->set_xk_yk(pt);
  
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=2; j<=this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType>(j);

    // x component: (j+1) * j * x^(deg - j) * y^(j-1) 
    weight[j] = (jv + 1.) * jv * this->xk_[this->deg_-j] * this->yk_[j-1];     
   
    // y component: - j * (j-1) * x^(deg+1-j) * y^(j-1)
    weight[this->dim_ + j] = - jv * (jv -1.) * this->xk_[this->deg_-j+1] * this->yk_[j-2];  
  }
  weight[0] = 0.;
  weight[this->dim_] = 0.;
  
  weight[1] = 2. * this->xk_[this->deg_-1];
  weight[this->dim_+1] = 0.;
}

} // namespace doffem
} // namespace hiflow
#endif
