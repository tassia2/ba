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

#ifndef __ANSATZ_AUG_PTRI_MONO_H_
#define __ANSATZ_AUG_PTRI_MONO_H_

#include "fem/ansatz/ansatz_space.h"
#include "common/vector_algebra_descriptor.h"
#include <cassert>
#include <cmath>

namespace hiflow {
namespace doffem {

///
/// \class AugPTriMono ansatz_skew_aug_p_tri_mono.h
/// \brief (x, y) * \bar{P}_k with \bar{P}_k = [x^k, x^(k-1)y, x^(k-2)y^2, .... , x, y^(k-1), y^k] 
/// \author Philipp Gerstner
///

template < class DataType, int DIM >
class AugPTriMono final : public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  AugPTriMono(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~AugPTriMono()
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
AugPTriMono< DataType, DIM >::AugPTriMono(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 2;
  this->name_ = "Aug_P_Tri_Mono";
  this->type_ = AnsatzSpaceType::P_AUG;

  assert (this->ref_cell_->type() == RefCellType::TRI_STD);
}

template < class DataType, int DIM > 
void AugPTriMono< DataType, DIM >::init ( size_t degree )
{
  assert (DIM == 2);
  this->tdim_ = 2;
  this->nb_comp_ = 2;
  this->max_deg_ = degree+1;
  this->dim_ = degree+1;
  this->deg_ = degree;

  this->comp_weight_size_.clear();
  this->comp_weight_size_.resize(this->nb_comp_, this->dim_);

  this->comp_offset_.clear();
  this->comp_offset_.resize(this->nb_comp_, 0);
  this->comp_offset_[1] = this->dim_;
  
  this->weight_size_ = this->nb_comp_ * this->dim_;

  this->xk_.clear();
  this->yk_.clear();
  this->xk_.resize(this->deg_ + 2, 1.);
  this->yk_.resize(this->deg_ + 2, 1.);
}

template < class DataType, int DIM > 
void AugPTriMono< DataType, DIM >::set_xk_yk ( const Coord &pt ) const
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
void AugPTriMono< DataType, DIM >::N(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  this->set_xk_yk(pt);
  const size_t offset = this->comp_offset_[1];
  
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=0; j<this->deg_+1; ++j)
  {
    // x component: x^(deg+1 - j) * y^(j) 
    weight[j] = this->xk_[this->deg_+1-j] * this->yk_[j];     
   
    // y component: x^(deg-j) * y^(j+1)
    weight[offset + j] = this->xk_[this->deg_-j] * this->yk_[j+1];  
    
//  std::cout << this->xk_[this->deg_+1-j] * this->yk_[j] << " " << this->xk_[this->deg_-j] * this->yk_[j+1] << std::endl;
  }
}

template < class DataType, int DIM > 
void AugPTriMono< DataType, DIM >::N_x(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  const size_t offset = this->comp_offset_[1];
  
  this->set_xk_yk(pt);
  
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=0; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType>(j);

    weight[j]              = (deg-jv+1.) * this->xk_[this->deg_-j]   * this->yk_[j];      
    weight[offset + j] = (deg-jv)    * this->xk_[this->deg_-j-1] * this->yk_[j+1];  
  
  }
  weight[this->deg_]            = this->yk_[this->deg_];
  weight[offset+this->deg_] = 0;
}

template < class DataType, int DIM > 
void AugPTriMono< DataType, DIM >::N_y(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const size_t offset = this->comp_offset_[1];
  this->set_xk_yk(pt);
  
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=1; j<=this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType>(j);
    weight[j]              = jv     * this->xk_[this->deg_-j+1] * this->yk_[j-1];     
    weight[offset + j] = (jv+1) * this->xk_[this->deg_-j]   * this->yk_[j];  
  }
  weight[0]          = 0.;
  weight[offset] = this->xk_[this->deg_];
}

template < class DataType, int DIM > 
void AugPTriMono< DataType, DIM >::N_xx(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  this->set_xk_yk(pt);
  const DataType deg = static_cast<DataType> (this->deg_);
  const size_t offset = this->comp_offset_[1];
  
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=0; j<this->deg_-1; ++j)
  {
    const DataType jv = static_cast<DataType>(j);

    weight[j]              = (deg+1-jv) * (deg - jv) * this->xk_[this->deg_-j-1] * this->yk_[j];     
    weight[offset + j] = (deg-jv) * (deg-jv-1)   * this->xk_[this->deg_-j-2] * this->yk_[j+1];  
  }

  weight[this->deg_-1] = 2. * this->yk_[this->deg_-1];
  weight[this->deg_] = 0.;
  
  weight[offset+this->deg_-1] = 0.;
  weight[offset+this->deg_] = 0.;
}

template < class DataType, int DIM > 
void AugPTriMono< DataType, DIM >::N_xy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  this->set_xk_yk(pt);
  const size_t offset = this->comp_offset_[1];
  
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=1; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j); 
    weight[j]              = (deg-jv+1) * jv * this->xk_[this->deg_-j] * this->yk_[j-1];     
    weight[offset + j] = (deg-jv) * (jv+1.) * this->xk_[this->deg_-j-1] * this->yk_[j];  
  }
  
  weight[0] = 0.;
  weight[this->deg_] = deg * this->yk_[this->deg_-1];
  
  weight[offset] = deg * this->xk_[this->deg_-1]; 
  weight[offset+this->deg_] = 0.;
}

template < class DataType, int DIM > 
void AugPTriMono< DataType, DIM >::N_yy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 2);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  this->set_xk_yk(pt);
  const size_t offset = this->comp_offset_[1];
  
  // loop over ansatz functions (dim = deg+1)
  for (size_t j=2; j<=this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType>(j);

    weight[j]              = jv * (jv-1.) * this->xk_[this->deg_-j+1] * this->yk_[j-2];     
    weight[offset + j] = (jv+1.) * jv * this->xk_[this->deg_-j] * this->yk_[j-1];  
  }
  
  weight[0] = 0.;
  weight[1] = 0.;
  
  weight[offset] = 0.;
  weight[offset+1] = 2. * this->xk_[this->deg_-1];
}

} // namespace doffem
} // namespace hiflow
#endif
