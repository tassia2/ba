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

#ifndef __ANSATZ_SKEW_AUG_PTET_MONO_H_
#define __ANSATZ_SKEW_AUG_PTET_MONO_H_

#include "fem/ansatz/ansatz_space.h"
#include "common/vector_algebra_descriptor.h"
#include <cassert>
#include <cmath>

namespace hiflow {
namespace doffem {

///
/// \class SkewAugPTetMono ansatz_skew_aug_p_tet_mono.h
/// \brief (x, y, z) \cross (\bar{P}_k)^3 with \bar{P}_k = [x^k, x^(k-1)y, x^{k-1}z, x^(k-2)y^2, .... , x, y^(k-1)x, y^{k-1}z, y^k] 
/// \author Jonas Roller
///

template < class DataType, int DIM >
class SkewAugPTetMono final: public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  SkewAugPTetMono(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~SkewAugPTetMono()
  {
  }

  void init (size_t degree);

  void N(const Coord &pt, std::vector< DataType > &weight) const override;

  void N_x(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_y(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_z(const Coord &pt, std::vector< DataType > &weight) const override;

  void N_xx(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_xy(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_xz(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_yy(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_yz(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_zz(const Coord &pt, std::vector< DataType > &weight) const override;

private:
  void compute_degree_hash () const
  {
    NOT_YET_IMPLEMENTED;
  }

  void set_xk_yk_zk (const Coord& pt) const;
  
  mutable std::vector<DataType> xk_;
  mutable std::vector<DataType> yk_;
  mutable std::vector<DataType> zk_;
  
  size_t deg_;
};


template < class DataType, int DIM > 
SkewAugPTetMono< DataType, DIM >::SkewAugPTetMono(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 3;
  this->name_ = "Skew_Aug_P_Tet_Mono";
  this->type_ = AnsatzSpaceType::P_AUG;
   

  assert (this->ref_cell_->type() == RefCellType::TET_STD);
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::init ( size_t degree )
{
  assert (DIM == 3 && degree > 0);
  this->tdim_ = 3;
  this->nb_comp_ = 3;
  this->max_deg_ = degree+1;
  this->dim_ = degree * (degree+2);
  this->deg_ = degree;

  this->comp_weight_size_.clear();
  this->comp_weight_size_.resize(this->nb_comp_, this->dim_);

  this->comp_offset_.clear();
  this->comp_offset_.resize(this->nb_comp_, 0);
  this->comp_offset_[1] = this->dim_;
  this->comp_offset_[2] = 2 * this->dim_;
  
  this->weight_size_ = this->nb_comp_ * this->dim_;

  this->xk_.clear();
  this->yk_.clear();
  this->zk_.clear();
  this->xk_.resize(this->deg_ + 2, 1.);
  this->yk_.resize(this->deg_ + 2, 1.);
  this->zk_.resize(this->deg_ + 2, 1.);
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::set_xk_yk_zk ( const Coord &pt ) const
{
  const DataType x = pt[0];
  const DataType y = pt[1];
  const DataType z = pt[2];
  
  // x = pt[0], y = pt[1], z = pt[2]
  // xk = [1, x, x², x³, ....., x^(deg+1) ]
  // yk = [1, y, y², y³, ....., y^(deg+1) ]
  // zk = [1, z, z², z³, ....., z^(deg+1) ]
  
  this->xk_[0] = 1.;
  this->yk_[0] = 1.;
  this->zk_[0] = 1.;
  
  for (size_t k = 1; k <= this->deg_+1; ++k)
  {
    this->xk_[k] = this->xk_[k-1] * x;
    this->yk_[k] = this->yk_[k-1] * y;
    this->zk_[k] = this->zk_[k-1] * z;
  }  
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  this->set_xk_yk_zk(pt);

  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  for(size_t j = 1; j < this->deg_+1; ++j)
  {
    for(size_t i = 0; i < this->deg_+1-j; ++i) 
    {
      weight[i + temp_offset] = this->xk_[j-1] * this->yk_[i] * this->zk_[this->deg_+1-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * this->xk_[j] * this->yk_[i] * this->zk_[this->deg_-j-i];
    } 
    temp_offset += this->deg_+1-j;
  }

  for (size_t j=0; j<this->deg_; ++j)
  {
    for(size_t i=1; i<this->deg_+1-j; ++i)
    {
      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = this->xk_[j] * this->yk_[i-1] * this->zk_[this->deg_+1-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * this->xk_[j] * this->yk_[i] * this->zk_[this->deg_-j-i];
    }
    temp_offset += this->deg_-j;
  }

  for(size_t j = 1; j < this->deg_+1; ++j)
  {
    size_t i = this->deg_+1-j;
   
    weight[temp_offset] = this->xk_[j-1] * this->yk_[i];
    
    weight[this->comp_offset_[1] + temp_offset] = -1. * this->xk_[j] * this->yk_[i-1];
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_x(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk_zk(pt);
   
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  // case j = 1
  for(size_t i = 0; i < this->deg_; ++i) 
  {
    weight[i + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * this->yk_[i] * this->zk_[this->deg_-1-i];
  } 
  temp_offset += this->deg_;

  for(size_t j = 2; j < this->deg_+1; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i = 0; i < this->deg_+1-j; ++i) 
    {
      weight[i + temp_offset] = (jv-1.) * this->xk_[j-2] * this->yk_[i] * this->zk_[this->deg_+1-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * jv *  this->xk_[j-1] * this->yk_[i] * this->zk_[this->deg_-j-i];
    } 
    temp_offset += this->deg_+1-j;
  }

  // case j = 0
  for(size_t i=1; i<this->deg_+1; ++i)
  {
    weight[i-1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i-1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i-1 + temp_offset] = 0.0;
  }
  temp_offset += this->deg_;

  for (size_t j=1; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i=1; i<this->deg_+1-j; ++i)
    {
      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = jv * this->xk_[j-1] * this->yk_[i-1] * this->zk_[this->deg_+1-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * jv * this->xk_[j-1] * this->yk_[i] * this->zk_[this->deg_-j-i];
    }
    temp_offset += this->deg_-j;
  }

  // case j = 1
  size_t i = this->deg_;
   
  weight[temp_offset] = 0.0;
    
  weight[this->comp_offset_[1] + temp_offset] = -1. * this->yk_[i-1];
    
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  ++temp_offset;

  for(size_t j = 2; j < this->deg_+1; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    
    size_t i = this->deg_+1-j;
   
    weight[temp_offset] = (jv-1.) * this->xk_[j-2] * this->yk_[i];
    
    weight[this->comp_offset_[1] + temp_offset] = -1. * jv * this->xk_[j-1] * this->yk_[i-1];
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_y(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  for(size_t j = 1; j < this->deg_+1; ++j)
  {
    // case i = 0
    weight[temp_offset] = 0.0;

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    for(size_t i = 1; i < this->deg_+1-j; ++i) 
    {
      const DataType iv = static_cast<DataType> (i);

      weight[i + temp_offset] = iv * this->xk_[j-1] * this->yk_[i-1] * this->zk_[this->deg_+1-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * iv * this->xk_[j] * this->yk_[i-1] * this->zk_[this->deg_-j-i];
    } 
    temp_offset += this->deg_+1-j;
  }

  for (size_t j=0; j<this->deg_; ++j)
  {
    // case i = 1
    weight[temp_offset] = 0.0;

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = -1. * this->xk_[j] * this->zk_[this->deg_-j-1];
    
    for(size_t i=2; i<this->deg_+1-j; ++i)
    {
      const DataType iv = static_cast<DataType> (i);
      
      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = (iv-1.) * this->xk_[j] * this->yk_[i-2] * this->zk_[this->deg_+1-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * iv * this->xk_[j] * this->yk_[i-1] * this->zk_[this->deg_-j-i];
    }
    temp_offset += this->deg_-j;
  }

  for(size_t j = 1; j < this->deg_; ++j)
  {
    size_t i = this->deg_+1-j;
    
    const DataType iv = static_cast<DataType> (i);
   
    weight[temp_offset] = iv * this->xk_[j-1] * this->yk_[i-1];
    
    weight[this->comp_offset_[1] + temp_offset] = -1. * (iv-1.) * this->xk_[j] * this->yk_[i-2];
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
  // case j = deg --> i = 1
  weight[temp_offset] = this->xk_[this->deg_-1];
    
  weight[this->comp_offset_[1] + temp_offset] = 0.0; 
    
  weight[this->comp_offset_[2] + temp_offset] = 0.0;
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_z(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  for(size_t j = 1; j < this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i = 0; i < this->deg_-j; ++i) 
    {
      const DataType iv = static_cast<DataType> (i);

      weight[i + temp_offset] = (deg+1.-jv-iv) * this->xk_[j-1] * this->yk_[i] * this->zk_[this->deg_-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * (deg-jv-iv) * this->xk_[j] * this->yk_[i] * this->zk_[this->deg_-1-j-i];
    }
    // case i = deg-j
    weight[this->deg_-j + temp_offset] = this->xk_[j-1] * this->yk_[this->deg_-j];

    weight[this->comp_offset_[1] + this->deg_-j + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-j + temp_offset] = 0.0;

    temp_offset += this->deg_+1-j;
  }
  // case j = deg --> i = 0
  weight[temp_offset] = this->xk_[this->deg_-1];

  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  ++temp_offset;

  for (size_t j=0; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i=1; i<this->deg_-j; ++i)
    {
      const DataType iv = static_cast<DataType> (i);

      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = (deg+1-jv-iv) * this->xk_[j] * this->yk_[i-1] * this->zk_[this->deg_-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * (deg-jv-iv) * this->xk_[j] * this->yk_[i] * this->zk_[this->deg_-1-j-i];
    }
    // case i = deg-j
    weight[this->deg_-j-1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + this->deg_-j-1 + temp_offset] =  this->xk_[j] * this->yk_[this->deg_-j-1];
      
    weight[this->comp_offset_[2] + this->deg_-j-1 + temp_offset] = 0.0;

    temp_offset += this->deg_-j;
  }

  for(size_t j = 1; j < this->deg_+1; ++j)
  {
    size_t i = this->deg_+1-j;
   
    weight[temp_offset] = 0.0;
    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;

    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
}


template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_xx(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  // case j = 1
  for(size_t i = 0; i < this->deg_; ++i) 
  {
    weight[i + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  } 
  temp_offset += this->deg_;
 
  if(this->deg_ > 1)
  { 
    // case j = 2
    for(size_t i = 0; i < this->deg_-1; ++i) 
    {
      weight[i + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -2. * this->yk_[i] * this->zk_[this->deg_-2-i];
    } 
    temp_offset += this->deg_-1;
  }
  
  for(size_t j = 3; j < this->deg_+1; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i = 0; i < this->deg_+1-j; ++i) 
    {
      weight[i + temp_offset] = (jv-1.) * (jv-2.) * this->xk_[j-3] * this->yk_[i] * this->zk_[this->deg_+1-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * jv * (jv-1.) * this->xk_[j-2] * this->yk_[i] * this->zk_[this->deg_-j-i];
    } 
    temp_offset += this->deg_+1-j;
  }

  // case j = 0,1
  for(size_t i=1; i<this->deg_+1; ++i)
  {
    weight[i-1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i-1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i-1 + temp_offset] = 0.0;
  }
  temp_offset += this->deg_;

  for(size_t i=1; i<this->deg_; ++i)
  {
    weight[i-1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i-1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i-1 + temp_offset] = 0.0;
  }
  temp_offset += this->deg_-1;

  for (size_t j=2; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i=1; i<this->deg_+1-j; ++i)
    {
      const DataType iv = static_cast<DataType> (i);
      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = jv * (jv-1.) * this->xk_[j-2] * this->yk_[i-1] * this->zk_[this->deg_+1-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * jv * (jv-1.) * this->xk_[j-2] * this->yk_[i] * this->zk_[this->deg_-j-i];
    }
    temp_offset += this->deg_-j;
  }

  // case j = 1
  weight[temp_offset] = 0.0;
    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
    
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  ++temp_offset;

  // case j = 2
  if(this->deg_ > 1)
  {
    size_t i = this->deg_-1;
  
    weight[temp_offset] = 0.0;
    
    weight[this->comp_offset_[1] + temp_offset] = -2. * this->yk_[i-1];
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;
  }

  for(size_t j = 3; j < this->deg_+1; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    
    size_t i = this->deg_+1-j;
   
    weight[temp_offset] = (jv-1.) * (jv-2.) * this->xk_[j-3] * this->yk_[i];
    
    weight[this->comp_offset_[1] + temp_offset] = -1. * jv * (jv-1.) * this->xk_[j-2] * this->yk_[i-1];
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_yy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const DataType deg = static_cast<DataType>(this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  for(size_t j = 1; j < this->deg_+1; ++j)
  {
    // case i = 0
    weight[temp_offset] = 0.0;

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    // case i = 1
    weight[1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + 1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + 1 + temp_offset] = 0.0;

    for(size_t i = 2; i < this->deg_+1-j; ++i) 
    {
      const DataType iv = static_cast<DataType> (i);

      weight[i + temp_offset] = iv * (iv-1.) * this->xk_[j-1] * this->yk_[i-2] * this->zk_[this->deg_+1-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * iv * (iv-1.) * this->xk_[j] * this->yk_[i-2] * this->zk_[this->deg_-j-i];
    } 
    temp_offset += this->deg_+1-j;
  }

  for (size_t j=0; j<this->deg_-1; ++j)
  {
    // case i = 1
    weight[temp_offset] = 0.0;

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    // case i = 2
    weight[1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + 1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + 1 + temp_offset] = -2. * this->xk_[j] * this->zk_[this->deg_-j-2];

    for(size_t i=3; i<this->deg_+1-j; ++i)
    {
      const DataType iv = static_cast<DataType> (i);
      
      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = (iv-1.) * (iv-2.) * this->xk_[j] * this->yk_[i-3] * this->zk_[this->deg_+1-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * iv * (iv-1.) * this->xk_[j] * this->yk_[i-2] * this->zk_[this->deg_-j-i];
    }
    temp_offset += this->deg_-j;
  }
  // case j = deg-1 --> i = 1
  weight[temp_offset] = 0.0;

  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  ++temp_offset;

  for(size_t j = 1; j < this->deg_-1; ++j)
  {
    size_t i = this->deg_+1-j;
   
    const DataType iv = static_cast<DataType> (i);
    
    weight[temp_offset] = iv * (iv-1.) * this->xk_[j-1] * this->yk_[i-2];
    
    weight[this->comp_offset_[1] + temp_offset] = -1. * (iv-1.) * (iv-2.) * this->xk_[j] * this->yk_[i-3];
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
  // case j = deg-1
  if (this->deg_ > 1)
  {
    size_t i = 2;
   
    const DataType iv = static_cast<DataType> (i);
    
    weight[temp_offset] = iv * (iv-1.) * this->xk_[this->deg_-2];
    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
  // case j = deg
  size_t i = 1;
   
  const DataType iv = static_cast<DataType> (i);
    
  weight[temp_offset] = 0.0;
    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
    
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  ++temp_offset;
}
template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_zz(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const DataType deg = static_cast<DataType> (this->deg_);

  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  for(size_t j = 1; j < this->deg_-1; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i = 0; i < this->deg_-1-j; ++i) 
    {
      const DataType iv = static_cast<DataType> (i);

      weight[i + temp_offset] = (deg+1.-jv-iv) * (deg-jv-iv) * this->xk_[j-1] * this->yk_[i] * this->zk_[this->deg_-1-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * (deg-jv-iv) * (deg-1.-jv-iv) * this->xk_[j] * this->yk_[i] * this->zk_[this->deg_-2-j-i];
    } 
    // case i = deg-1-j
    weight[this->deg_-1-j + temp_offset] = 2. * this->xk_[j-1] * this->yk_[this->deg_-1-j];

    weight[this->comp_offset_[1] + this->deg_-1-j + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-1-j + temp_offset] = 0.0;

    // case i = deg-j
    weight[this->deg_-j + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + this->deg_-j + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-j + temp_offset] = 0.0;

    temp_offset += this->deg_+1-j;
  }

  if (this->deg_ > 1)
  {
    // case j = deg-1, i = 0
    weight[temp_offset] = 2. * this->xk_[deg-2]; 

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0;
    
    // case j = deg-1, i = 1
    weight[1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + 1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + 1 + temp_offset] = 0.0;
  
    temp_offset += 2;
  }
 
  // case j = deg, i = 0
  weight[temp_offset] = 0.0;

  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;
 
  ++temp_offset;

  for (size_t j=0; j<this->deg_-1; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i=1; i<this->deg_-1-j; ++i)
    {
      const DataType iv = static_cast<DataType> (i);

      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = (deg+1.-jv-iv) * (deg-jv-iv) * this->xk_[j] * this->yk_[i-1] * this->zk_[this->deg_-1-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * (deg-jv-iv) * (deg-1.-jv-iv) * this->xk_[j] * this->yk_[i] * this->zk_[this->deg_-2-j-i];
    }
    // case i = deg-1-j
    if (this->deg_ > 1)
    {
      weight[this->deg_-2-j + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + this->deg_-2-j + temp_offset] = 2. * this->xk_[j] * this->yk_[this->deg_-2-j];
      
      weight[this->comp_offset_[2] + this->deg_-2-j + temp_offset] = 0.0;
    }
    // case i = deg-j
    weight[this->deg_-1-j + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + this->deg_-1-j + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-1-j + temp_offset] = 0.0;

    temp_offset += this->deg_-j;
  }
  // case j = deg-1 --> i = 1
  weight[temp_offset] = 0.0;

  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  for(size_t j = 1; j < this->deg_+1; ++j)
  {
    size_t i = this->deg_+1-j;
   
    weight[temp_offset] = 0.0;
    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_xy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const DataType deg = static_cast<DataType>(this->deg_);
  
  this->set_xk_yk_zk(pt);

  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  // case j = 1, i = 0
  weight[temp_offset] = 0.0;

  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;
  for(size_t i = 1; i < this->deg_; ++i) 
  {
    const DataType iv = static_cast<DataType>(i);

    weight[i + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * iv * this->yk_[i-1] * this->zk_[this->deg_-1-i];
  }
  temp_offset += this->deg_;

  for(size_t j = 2; j < this->deg_+1; ++j)
  {
    const DataType jv = static_cast<DataType>(j);
    // case i = 0
    weight[temp_offset] = 0.0;

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    for(size_t i = 1; i < this->deg_+1-j; ++i) 
    {
      const DataType iv = static_cast<DataType>(i);

      weight[i + temp_offset] = (jv-1.) * iv * this->xk_[j-2] * this->yk_[i-1] * this->zk_[this->deg_+1-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * jv * iv * this->xk_[j-1] * this->yk_[i-1] * this->zk_[this->deg_-j-i];
    } 
    temp_offset += this->deg_+1-j;
  }

  // case j = 0
  for(size_t i=1; i<this->deg_+1; ++i)
  {
    weight[i-1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i-1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i-1 + temp_offset] = 0.0;
  }
  temp_offset += this->deg_;
  
  for (size_t j=1; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType>(j);
    
    // case i = 1 
    weight[temp_offset] = 0.0;

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = -1. * jv * this->xk_[j-1] * this->zk_[this->deg_-j-1];

    for(size_t i=2; i<this->deg_+1-j; ++i)
    {
      const DataType iv = static_cast<DataType>(i);

      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = jv * (iv-1.) * this->xk_[j-1] * this->yk_[i-2] * this->zk_[this->deg_+1-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * jv * iv * this->xk_[j-1] * this->yk_[i-1] * this->zk_[this->deg_-j-i];
    }
    temp_offset += this->deg_-j;
  }

  if(this->deg_ > 1)
  {	  
    // case j = 1 --> i = deg
    weight[temp_offset] = 0.0;
    
    weight[this->comp_offset_[1] + temp_offset] = -1. * (deg-1.) * this->yk_[this->deg_-2];
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
 

    for(size_t j = 1; j < this->deg_; ++j)
    {
      const DataType jv = static_cast<DataType> (j);
      
      size_t i = this->deg_+1-j;
    
      const DataType iv = static_cast<DataType> (i);
   
      weight[temp_offset] = (jv-1.) * iv * this->xk_[j-2] * this->yk_[i-1];
    
      weight[this->comp_offset_[1] + temp_offset] = -1. * jv * (iv-1.) * this->xk_[j-1] * this->yk_[i-2];
    
      weight[this->comp_offset_[2] + temp_offset] = 0.0;

      ++temp_offset;
    }
    // case j = deg --> i = 1
    weight[temp_offset] = (deg-1.) * this->xk_[this->deg_-2];
    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;
  }
  else
  {
    weight[temp_offset] = 0.0;
    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;
  }
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_xz(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const DataType deg = static_cast<DataType>(this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  // case j = 1
  for(size_t i = 0; i < this->deg_-1; ++i) 
  {
    const DataType iv = static_cast<DataType> (i);
    
    weight[i + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * (deg-1.-iv) * this->yk_[i] * this->zk_[this->deg_-2-i];
  }
  // case j = 1, i = deg-1
  weight[this->deg_-1 + temp_offset] = 0.0;

  weight[this->comp_offset_[1] + this->deg_-1 + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_-1 + temp_offset] = 0.0;

  temp_offset += this->deg_;
  for(size_t j = 2; j < this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i = 0; i < this->deg_-j; ++i) 
    {
      const DataType iv = static_cast<DataType> (i);
      
      weight[i + temp_offset] = (jv-1.) * (deg+1.-jv-iv) * this->xk_[j-2] * this->yk_[i] * this->zk_[this->deg_-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * jv * (deg-jv-iv) * this->xk_[j-1] * this->yk_[i] * this->zk_[this->deg_-1-j-i];
    } 
    // case i = deg-j
    weight[this->deg_-j + temp_offset] = (jv-1.) * this->xk_[j-2] * this->yk_[this->deg_-j];

    weight[this->comp_offset_[1] + this->deg_-j + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-j + temp_offset] = 0.0;

    temp_offset += this->deg_+1-j;
  }
  // case j = deg --> i = 0
  if(this->deg_ > 1)
  {
    weight[temp_offset] = (deg-1.) * this->xk_[this->deg_-2];

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }

  // case j = 0
  for(size_t i=1; i<this->deg_+1; ++i)
  {
    weight[i-1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + i-1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i-1 + temp_offset] = 0.0;
  }
  temp_offset += this->deg_;

  for (size_t j=1; j<this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    for(size_t i=1; i<this->deg_-j; ++i)
    {
      const DataType iv = static_cast<DataType> (i);
      
      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = jv * (deg+1.-jv-iv) * this->xk_[j-1] * this->yk_[i-1] * this->zk_[this->deg_-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * jv * (deg-jv-iv) * this->xk_[j-1] * this->yk_[i] * this->zk_[this->deg_-1-j-i];
    }
    // case i = deg-j
    weight[this->deg_-j-1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + this->deg_-j-1 + temp_offset] = jv * this->xk_[j-1] * this->yk_[this->deg_-j-1];
      
    weight[this->comp_offset_[2] + this->deg_-j-1 + temp_offset] = 0.0;

    temp_offset += this->deg_-j;
  }

  for(size_t j = 1; j < this->deg_+1; ++j)
  {
    weight[temp_offset] = 0.0;
    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
}

template < class DataType, int DIM > 
void SkewAugPTetMono< DataType, DIM >::N_yz(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);

  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = deg*(deg+2))
  for(size_t j = 1; j < this->deg_; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    // case i = 0
    weight[temp_offset] = 0.0;

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0; 

    for(size_t i = 1; i < this->deg_-j; ++i) 
    {
      const DataType iv = static_cast<DataType> (i);
      
      weight[i + temp_offset] = iv * (deg+1.-jv-iv) * this->xk_[j-1] * this->yk_[i-1] * this->zk_[this->deg_-j-i];

      weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
      weight[this->comp_offset_[2] + i + temp_offset] = -1. * iv * (deg-jv-iv) * this->xk_[j] * this->yk_[i-1] * this->zk_[this->deg_-1-j-i];
    } 
    // case i = deg-j
    weight[this->deg_-j + temp_offset] = (deg-jv) * this->xk_[j-1] * this->yk_[this->deg_-j-1];

    weight[this->comp_offset_[1] + this->deg_-j + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-j + temp_offset] = 0.0;

    temp_offset += this->deg_+1-j;
  }

  for (size_t j=0; j<this->deg_-1; ++j)
  {
    const DataType jv = static_cast<DataType> (j);
    // case i = 1
    weight[temp_offset] = 0.0;

    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = -1. * (deg-jv-1) * this->xk_[j] * this->zk_[this->deg_-2-j];

    for(size_t i=2; i<this->deg_-j; ++i)
    {
      const DataType iv = static_cast<DataType> (i);
      
      weight[i-1 + temp_offset] = 0.0;

      weight[this->comp_offset_[1] + i-1 + temp_offset] = (iv-1.) * (deg+1.-jv-iv) * this->xk_[j] * this->yk_[i-2] * this->zk_[this->deg_-j-i];
      
      weight[this->comp_offset_[2] + i-1 + temp_offset] = -1. * iv * (deg-jv-iv) * this->xk_[j] * this->yk_[i-1] * this->zk_[this->deg_-1-j-i];
    }
    // case i = deg-j
    weight[this->deg_-j-1 + temp_offset] = 0.0;

    weight[this->comp_offset_[1] + this->deg_-j-1 + temp_offset] = (deg-jv-1.) * this->xk_[j] * this->yk_[this->deg_-j-2];
      
    weight[this->comp_offset_[2] + this->deg_-j-1 + temp_offset] = 0.0;

    temp_offset += this->deg_-j;
  }
  // case j = deg-1 --> i = 1
  weight[temp_offset] = 0.0;

  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  ++temp_offset;

  for(size_t j = 1; j < this->deg_+1; ++j)
  {
    weight[temp_offset] = 0.0;
    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;
    
    weight[this->comp_offset_[2] + temp_offset] = 0.0;

    ++temp_offset;
  }
}

} // namespace doffem
} // namespace hiflow
#endif
