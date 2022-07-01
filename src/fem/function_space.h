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

#ifndef __FEM_FUNCTION_SPACE_H_
#define __FEM_FUNCTION_SPACE_H_

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <cmath>

#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"
#include "common/log.h"
#include "fem/reference_cell.h"


namespace hiflow {
namespace doffem {

// TODO: use templates instead of this function if possible

/// RefCellFunction evaluates a number nb_func of functions at a given point on the reference cell 
/// It is assumed that all functions have the same number of components nb_comp
template < class DataType, int DIM > 
class RefCellFunction {

public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  
  RefCellFunction(size_t nb_comp, size_t nb_func)
  : nb_comp_ (nb_comp),
  nb_func_ (nb_func)
  {}
  
  virtual ~RefCellFunction()
  {}
  
  /// perform evaluation at given point on reference cell
  /// structure of values: val_ij <-> value of j-th component of function i
  /// [val_00, ... , val_0n, val_10, ... , val_1n, ....... , val_m0, ..., val_mn]
  virtual void evaluate (const Coord &ref_pt, std::vector<DataType>& values) const = 0;
  
  /// return number of components of each function
  size_t nb_comp () const 
  {
    return this->nb_comp_;
  }
  
  /// return number of functions
  size_t nb_func () const 
  {
    return this->nb_func_;
  }
  
  virtual size_t weight_size() const = 0;
  
  virtual size_t iv2ind (size_t i, size_t var ) const = 0;
  
protected:
  size_t nb_func_;
  size_t nb_comp_;
};

///
/// \class FunctionSpace function_space.h
/// \brief Abstract base class for vector spaces of functions defined on some reference cell
/// \author Philipp Gerstner


template < class DataType, int DIM > 
class FunctionSpace {

public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  /// Default Constructor
  FunctionSpace()
  : ref_cell_(0), 
    tdim_(0), 
    nb_comp_(0), 
    dim_(0), 
    weight_size_(0),
    max_deg_(0),
    deg_hash_(0)
  {}
    
  /// Default Destructor
  virtual ~FunctionSpace();

  /// Get information (name) about the used FE Type
  inline std::string name() const;

  /// Get the polynomial degree of the ansatz.
  inline size_t max_deg() const;

  /// Get hash value for polynomial degrees
  inline size_t deg_hash() const; 

  /// Dimension of Ansatz space
  inline size_t dim() const;

  /// Tpological dimension of cell
  inline size_t tdim() const;
  
  /// get required size to store the values of all components of all basis functions evaluated at one specified point 
  inline size_t weight_size() const;

  /// get required size to store the values of specific component of all basis functions evaluated at one specified point
  inline size_t weight_size( size_t comp ) const;

  /// Number of components
  inline size_t nb_comp() const;
 
  inline CRefCellSPtr<DataType, DIM> ref_cell() const;

  inline RefCellType ref_cell_type() const; 

  inline std::vector< Coord > ref_cell_coords () const;

  /// Indexing return array weight of N_*(pt, weight) routines to find corresponding value
  /// for basis function i and component var
  virtual size_t iv2ind (size_t i, size_t var ) const = 0;
 
  /// For given point, get values of all shapefunctions on reference cell
  virtual void N(const Coord &ref_pt, std::vector< DataType > &weight) const = 0;
  virtual void grad_N (const Coord &ref_pt, std::vector< Coord > &gradients) const;
  virtual void hessian_N (const Coord &ref_pt, std::vector< mat > &hessians) const;

  virtual void N_x(const Coord &ref_pt, std::vector< DataType > &weight) const = 0;
  virtual void N_y(const Coord &ref_pt, std::vector< DataType > &weight) const = 0 ;
  virtual void N_z(const Coord &ref_pt, std::vector< DataType > &weight) const = 0;

  virtual void N_xx(const Coord &ref_pt, std::vector< DataType > &weight) const  = 0;
  virtual void N_xy(const Coord &ref_pt, std::vector< DataType > &weight) const = 0 ;
  virtual void N_xz(const Coord &ref_pt, std::vector< DataType > &weight) const = 0;
  virtual void N_yy(const Coord &ref_pt, std::vector< DataType > &weight) const = 0 ;
  virtual void N_yz(const Coord &ref_pt, std::vector< DataType > &weight) const = 0 ;
  virtual void N_zz(const Coord &ref_pt, std::vector< DataType > &weight) const  = 0;

  /// for compatibility reasons 
  void evaluate (const Coord &ref_pt, std::vector< DataType > &weight) const 
  {
    this->N(ref_pt, weight);
  }

  /// check whether coordinates of underlying reference cell coincide with 
  /// given set of points
  bool ref_coord_match( const std::vector<Coord> test_coord) const;

protected:

  virtual void compute_degree_hash () const = 0 ;

  /// Status information if fetype was initialized
  bool init_status_;

  /// Storing an instance of the reference cell
  CRefCellSPtr<DataType, DIM>ref_cell_;

  std::string name_;

  /// Topological dimension
  size_t tdim_;

  /// Number of components
  size_t nb_comp_;

  /// dimension of ansatz space
  size_t dim_;

  std::vector<size_t> comp_weight_size_;
  
  size_t weight_size_;

  /// Maximal polynomial degree
  size_t max_deg_;

  /// Hash value for unique degree identification
  mutable size_t deg_hash_;
  
  mutable std::vector< DataType > gradient_component_;
  mutable std::vector< DataType > hessian_component_;
  
};

template < class DataType, int DIM > 
FunctionSpace< DataType, DIM>::~FunctionSpace()
{}

template < class DataType, int DIM > 
size_t FunctionSpace< DataType, DIM >::max_deg() const 
{
  return max_deg_;
}

template < class DataType, int DIM > 
size_t FunctionSpace< DataType, DIM >::deg_hash() const 
{
  return this->deg_hash_;
}

template < class DataType, int DIM > 
size_t FunctionSpace< DataType, DIM >::nb_comp() const 
{
  return this->nb_comp_;
}

template < class DataType, int DIM > 
size_t FunctionSpace< DataType, DIM >::dim() const 
{
  return this->dim_;
}

template < class DataType, int DIM > 
size_t FunctionSpace< DataType, DIM >::tdim() const 
{
  return this->tdim_;
}

template < class DataType, int DIM > 
size_t FunctionSpace< DataType, DIM >::weight_size() const 
{
  return this->weight_size_;
}


template < class DataType, int DIM > 
size_t FunctionSpace< DataType, DIM >::weight_size(size_t comp) const 
{
  assert (comp < this->comp_weight_size_.size());
  return this->comp_weight_size_[comp];
}


template < class DataType, int DIM > 
std::string FunctionSpace< DataType, DIM >::name() const 
{
  return this->name_;
}

template < class DataType, int DIM > 
CRefCellSPtr<DataType, DIM> FunctionSpace< DataType, DIM >::ref_cell() const 
{
  return this->ref_cell_;
}

template < class DataType, int DIM > 
RefCellType FunctionSpace< DataType, DIM >::ref_cell_type() const 
{
  assert(this->ref_cell_);
  return this->ref_cell_->type();
}

template < class DataType, int DIM > 
std::vector< typename FunctionSpace<DataType, DIM>::Coord > FunctionSpace< DataType, DIM >::ref_cell_coords() const 
{
  assert(this->ref_cell_);
  return this->ref_cell_->get_coords();
}

template < class DataType, int DIM > 
bool FunctionSpace< DataType, DIM >::ref_coord_match( const std::vector<Coord> test_coord) const 
{
  assert(this->ref_cell_);
  return this->ref_cell_->ref_coord_match (test_coord);
}

template < class DataType, int DIM > 
void FunctionSpace< DataType, DIM >::grad_N (const Coord &pt, std::vector< Coord > &gradients) const
{
  assert (gradients.size() == this->weight_size_);
 
  for (size_t c = 0; c < DIM; ++c) 
  {
    this->gradient_component_.clear();
    this->gradient_component_.resize(weight_size_, 0.);
    switch (c) 
    {
      case 0:
        this->N_x(pt, gradient_component_);
        break;
      case 1:
        this->N_y(pt, gradient_component_);
        break;
      case 2:
        this->N_z(pt, gradient_component_);
        break;
      default:
        assert(false);
        break;
    };
    PRAGMA_LOOP_VEC
    for (size_t k=0; k<weight_size_; ++k)
    {
      gradients[k].set(c, gradient_component_[k]);
    }
  }
}

template < class DataType, int DIM > 
void FunctionSpace< DataType, DIM >::hessian_N (const Coord &pt, std::vector< mat > &hessians) const
{
  assert (hessians.size() == this->weight_size_);

  for (size_t c1 = 0; c1 < DIM; ++c1) 
  {
    for (size_t c2 = c1; c2 < DIM; ++c2) 
    {
      this->hessian_component_.clear();
      this->hessian_component_.resize(weight_size_, 0.);
      switch (c1) 
      {
        case 0:
          switch (c2) 
          {
            case 0:
              this->N_xx(pt, hessian_component_);
              break;
            case 1:
              this->N_xy(pt, hessian_component_);
              break;
            case 2:
              this->N_xz(pt, hessian_component_);
              break;
            default:
              assert(false);
              break;
          }
          break;
        case 1:
          switch (c2) 
          {
            case 1:
              this->N_yy(pt, hessian_component_);
              break;
            case 2:
              this->N_yz(pt, hessian_component_);
              break;
            default:
              assert(false);
              break;
          }
          break;
        case 2:
          if (c2 == 2) 
          {
            this->N_zz(pt, hessian_component_);
          } 
          else 
          {
            assert(false);
            break;
          }
          break;
        default:
          assert(false);
          break;
      };
      PRAGMA_LOOP_VEC
      for (size_t k=0; k<weight_size_; ++k)
      {
        hessians[k].set(c1,c2, hessian_component_[k]);
        if (c1 != c2)
        {
          hessians[k].set(c2,c1, hessian_component_[k]);
        }
      }
    }
  }
}

}
}
#endif
