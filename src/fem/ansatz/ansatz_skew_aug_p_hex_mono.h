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

#ifndef __ANSATZ_SKEW_AUG_PHEX_MONO_H_
#define __ANSATZ_SKEW_AUG_PHEX_MONO_H_

#include "fem/ansatz/ansatz_space.h"
#include "common/vector_algebra_descriptor.h"
#include <cassert>
#include <cmath>

namespace hiflow {
namespace doffem {

///
/// \class SkewAugPTetMono ansatz_skew_aug_p_hex_mono.h
/// \sum_{i=0]^k r_i * curl{0,0,x^(i+1)z^(k-i)} + s_i * curl{yz^(i+1)x^(k-i),0,0} + t_i * curl{0,zx^(i+1)y^(k-i),0},
/// needed for BDM elements on hexahedra
/// \author Jonas Roller
///

template < class DataType, int DIM >
class SkewAugPHexMono final: public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  
  /// Default Constructor
  SkewAugPHexMono(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~SkewAugPHexMono()
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
SkewAugPHexMono< DataType, DIM >::SkewAugPHexMono(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 3;
  this->name_ = "Skew_Aug_P_Hex_Mono";
  this->type_ = AnsatzSpaceType::P_AUG;

  assert (this->ref_cell_->type() == RefCellType::HEX_STD);
}

template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::init ( size_t degree )
{
  assert (DIM == 3);
  this->tdim_ = 3;
  this->nb_comp_ = 3;
  this->max_deg_ = degree+1;
  this->dim_ = 3*(degree+1);
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
void SkewAugPHexMono< DataType, DIM >::set_xk_yk_zk ( const Coord &pt ) const
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
void SkewAugPHexMono< DataType, DIM >::N(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  this->set_xk_yk_zk(pt);

  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * this->xk_[1] * this->yk_[i] * this->zk_[this->deg_-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = -1. * this->yk_[i+1] * this->zk_[this->deg_-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }      
  temp_offset += this->deg_+1;

  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * this->yk_[1] * this->zk_[i] * this->xk_[this->deg_-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * this->zk_[i+1] * this->xk_[this->deg_-i];
  } 
  temp_offset += this->deg_+1;
 
  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = -1. * this->xk_[i+1] * this->yk_[this->deg_-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * this->zk_[1] * this->xk_[i] * this->yk_[this->deg_-i];
  } 
}

template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_x(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk_zk(pt);
   
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * this->yk_[i] * this->zk_[this->deg_-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }      
  temp_offset += this->deg_+1;

  for(size_t i = 0; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * (deg-iv) * this->yk_[1] * this->zk_[i] * this->xk_[this->deg_-1-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * (deg-iv) * this->zk_[i+1] * this->xk_[this->deg_-1-i];
  }

  // case i = deg 
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;

  temp_offset += this->deg_+1;


  // case i = 0 
  weight[temp_offset] = -1. * this->yk_[this->deg_];
	    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  for(size_t i = 1; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = -1. * (iv+1.) * this->xk_[i] * this->yk_[this->deg_-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * iv * this->zk_[1] * this->xk_[i-1] * this->yk_[this->deg_-i];
  }
}

template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_y(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  // case i = 0
  weight[temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + temp_offset] = -1. * this->zk_[this->deg_];
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  for(size_t i = 1; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * iv * this->xk_[1] * this->yk_[i-1] * this->zk_[this->deg_-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = -1. * (iv+1.) * this->yk_[i] * this->zk_[this->deg_-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }      
  temp_offset += this->deg_+1;

  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * this->zk_[i] * this->xk_[this->deg_-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  } 
  temp_offset += this->deg_+1;
 
  for(size_t i = 0; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = -1. * (deg-iv) * this->xk_[i+1] * this->yk_[this->deg_-1-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * (deg-iv) * this->zk_[1] * this->xk_[i] * this->yk_[this->deg_-1-i];
  }
  // case i = deg
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;
}

template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_z(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  for(size_t i = 0; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * (deg-iv) * this->xk_[1] * this->yk_[i] * this->zk_[this->deg_-1-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = -1. * (deg-iv) * this->yk_[i+1] * this->zk_[this->deg_-1-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }     
  // case i = deg 
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;

  temp_offset += this->deg_+1;

  // case i = 0
  weight[temp_offset] = 0.0;

  weight[this->comp_offset_[1] + temp_offset] = 0.0;

  weight[this->comp_offset_[2] + temp_offset] = -1. * this->xk_[this->deg_];

  for(size_t i = 1; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * iv * this->yk_[1] * this->zk_[i-1] * this->xk_[this->deg_-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * (iv+1.) * this->zk_[i] * this->xk_[this->deg_-i];
  } 
  temp_offset += this->deg_+1;
 
  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * this->xk_[i] * this->yk_[this->deg_-i];
  }
}


template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_xx(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }      
  temp_offset += this->deg_+1;

  for(size_t i = 0; i < this->deg_-1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * (deg-iv) * (deg-1.-iv) * this->yk_[1] * this->zk_[i] * this->xk_[this->deg_-2-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * (deg-iv) * (deg-1.-iv) * this->zk_[i+1] * this->xk_[this->deg_-2-i];
  } 
  // case i = deg-1
  if(this->deg_ > 0)
  {
    weight[this->deg_-1 + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + this->deg_-1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-1 + temp_offset] = 0.0;
  }
  // case i = deg
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;

  temp_offset += this->deg_+1;

  // case i = 0 
  weight[temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  if (this->deg_ > 0)
  {	  
    // case i = 1
    weight[1 + temp_offset] = -2. * this->yk_[this->deg_-1];	    
    
    weight[this->comp_offset_[1] + 1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + 1 + temp_offset] = 0.0;
  }

  for(size_t i = 2; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = -1. * (iv+1.) * iv * this->xk_[i-1] * this->yk_[this->deg_-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * iv * (iv-1.) * this->zk_[1] * this->xk_[i-2] * this->yk_[this->deg_-i];
  } 
}

template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_yy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const DataType deg = static_cast<DataType>(this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  // case i = 0
  weight[temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;
  // case i = 1 
  if (this->deg_ > 0)
  {
    weight[1 + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + 1 + temp_offset] = -2. * this->zk_[this->deg_-1];
      
    weight[this->comp_offset_[2] + 1 + temp_offset] = 0.0;
  }
  for(size_t i = 2; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * iv * (iv-1.) * this->xk_[1] * this->yk_[i-2] * this->zk_[this->deg_-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = -1. * (iv+1.) * iv * this->yk_[i-1] * this->zk_[this->deg_-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }      
  temp_offset += this->deg_+1;

  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  } 
  temp_offset += this->deg_+1;
 
  for(size_t i = 0; i < this->deg_-1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = -1. * (deg-iv) * (deg-1.-iv) * this->xk_[i+1] * this->yk_[this->deg_-2-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * (deg-iv) * (deg-1.-iv) * this->zk_[1] * this->xk_[i] * this->yk_[this->deg_-2-i];
  }
  // case i = deg-1
  if(this->deg_ > 0)
  {
    weight[this->deg_-1 + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + this->deg_-1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-1 + temp_offset] = 0.0;
  }
  // case i = deg 
  weight[this->deg_ + temp_offset] = 0.0;

  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;
}
template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_zz(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const DataType deg = static_cast<DataType> (this->deg_);

  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  for(size_t i = 0; i < this->deg_-1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * (deg-iv) * (deg-1.-iv) * this->xk_[1] * this->yk_[i] * this->zk_[this->deg_-2-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = -1. * (deg-iv) * (deg-1.-iv) * this->yk_[i+1] * this->zk_[this->deg_-2-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }     
  // case i = deg-1
  if (this->deg_ > 0)
  {
    weight[this->deg_-1 + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + this->deg_-1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + this->deg_-1 + temp_offset] = 0.0;
  }
  // case i = deg
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;

  temp_offset += this->deg_+1;

  // case i = 0
  weight[temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  // case i = 1
  if (this->deg_ > 0)
  {
    weight[1 + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + 1 + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + 1 + temp_offset] = -2. * this->xk_[this->deg_-1];
  }

  for(size_t i = 2; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * iv * (iv-1.) * this->yk_[1] * this->zk_[i-2] * this->xk_[this->deg_-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * (iv+1.) * iv * this->zk_[i-1] * this->xk_[this->deg_-i];
  } 
  temp_offset += this->deg_+1;
 
  for(size_t i = 0; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  } 
}

template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_xy(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const DataType deg = static_cast<DataType>(this->deg_);
  
  this->set_xk_yk_zk(pt);

  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  // case i = 0
  weight[temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  for(size_t i = 1; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * iv * this->yk_[i-1] * this->zk_[this->deg_-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }      
  temp_offset += this->deg_+1;

  for(size_t i = 0; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * (deg-iv) * this->zk_[i] * this->xk_[this->deg_-1-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }
  // case i = deg 
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;

  temp_offset += this->deg_+1;

  // case i = 0
  if (this->deg_ > 0)
  {
    weight[temp_offset] = -1. * deg * this->yk_[this->deg_-1];
	    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0;
  }
  for(size_t i = 1; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = -1. * (iv+1.) * (deg-iv) * this->xk_[i] * this->yk_[this->deg_-1-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * iv * (deg-iv) * this->zk_[1] * this->xk_[i-1] * this->yk_[this->deg_-1-i];
  } 
  // case i = deg
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;
}

template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_xz(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());

  const DataType deg = static_cast<DataType>(this->deg_);
  
  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  for(size_t i = 0; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * (deg-iv) * this->yk_[i] * this->zk_[this->deg_-1-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }     
  // case i = deg 
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;

  temp_offset += this->deg_+1;

  // case i = 0
  if (this->deg_ > 0)
  {
    weight[temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + temp_offset] = -1. * deg * this->xk_[this->deg_-1];
  }

  for(size_t i = 1; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * iv * (deg-iv) * this->yk_[1] * this->zk_[i-1] * this->xk_[this->deg_-1-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = -1. * (iv+1.) * (deg-iv) * this->zk_[i] * this->xk_[this->deg_-1-i];
  } 

  // case i = deg
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;

  temp_offset += this->deg_+1;
 
  // case i = 0
  weight[temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  for(size_t i = 1; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * iv * this->xk_[i-1] * this->yk_[this->deg_-i];
  } 
}

template < class DataType, int DIM > 
void SkewAugPHexMono< DataType, DIM >::N_yz(const Coord &pt, std::vector< DataType > &weight) const
{
  assert (DIM == 3);
  assert (this->dim_ * this->nb_comp_ == weight.size());
  
  const DataType deg = static_cast<DataType> (this->deg_);

  this->set_xk_yk_zk(pt);
  
  size_t temp_offset = 0;
  // loop over ansatz functions (dim = 3*(deg+1))
  // case i = 0
  if (this->deg_ > 0)
  {
    weight[temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + temp_offset] = -1. * deg * this->zk_[this->deg_-1];
      
    weight[this->comp_offset_[2] + temp_offset] = 0.0;
  }
  for(size_t i = 1; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = (iv+1.) * iv * (deg-iv) * this->xk_[1] * this->yk_[i-1] * this->zk_[this->deg_-1-i];
	    
    weight[this->comp_offset_[1] + i + temp_offset] = -1. * (iv+1.) * (deg-iv) * this->yk_[i] * this->zk_[this->deg_-1-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  }      
  // case i = deg
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;

  temp_offset += this->deg_+1;

  // case i = 0
  weight[temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + temp_offset] = 0.0;

  for(size_t i = 1; i < this->deg_+1; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = (iv+1.) * iv * this->zk_[i-1] * this->xk_[this->deg_-i];
      
    weight[this->comp_offset_[2] + i + temp_offset] = 0.0;
  } 
  temp_offset += this->deg_+1;
 
  for(size_t i = 0; i < this->deg_; ++i)
  {
    const DataType iv = static_cast<DataType> (i);

    weight[i + temp_offset] = 0.0;
	    
    weight[this->comp_offset_[1] + i + temp_offset] = 0.0;
      
    weight[this->comp_offset_[2] + i + temp_offset] = (iv+1.) * (deg-iv) * this->xk_[i] * this->yk_[this->deg_-1-i];
  } 
  // case i = deg
  weight[this->deg_ + temp_offset] = 0.0;
	    
  weight[this->comp_offset_[1] + this->deg_ + temp_offset] = 0.0;
      
  weight[this->comp_offset_[2] + this->deg_ + temp_offset] = 0.0;
}

} // namespace doffem
} // namespace hiflow
#endif
