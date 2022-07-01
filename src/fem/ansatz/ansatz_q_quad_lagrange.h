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

#ifndef __ANSATZ_QQUAD_LAGRANGE_H_
#define __ANSATZ_QQUAD_LAGRANGE_H_

#include "fem/ansatz/ansatz_space.h"
#include "polynomials/lagrangepolynomial.h"
#include "common/vector_algebra_descriptor.h"
#include <cassert>

namespace hiflow {
namespace doffem {

///
/// \class QQuadLag ansatz_space_hex.h
/// \brief Q Lagrange polynomials on reference Hexahedron [0,1]x[0,1]x[0,1]
/// \author Michael Schick<br>Martin Baumann<br>Philipp Gerstner<br>Simon Gawlok
///

template < class DataType, int DIM >
class QQuadLag final: public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  QQuadLag (CRefCellSPtr<DataType, DIM> ref_cell);

  void init (size_t degree);
  void init (size_t degree, size_t nb_comp);
  void init (const std::vector< size_t > &degrees);
  void init (const std::vector< std::vector<size_t> > &degrees); 

  /// Default Destructor
  virtual ~QQuadLag()
  {
  }

private:
  void compute_degree_hash () const 
  {
    NOT_YET_IMPLEMENTED;
  }

  void N   (const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_x (const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_y (const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_xx(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_xy(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_yy(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;

  /// Index ordering form space (x=i,y=j,z=k) to vector index
  inline size_t ij2ind(size_t i, size_t j, size_t comp) const;

  /// Lagrange polynomials which are used for evaluating shapefunctions
  LagrangePolynomial< DataType > lp_;

  std::vector< std::vector<size_t> > my_degrees_;
  std::vector< std::vector<size_t> > nb_dof_on_line_;
};

template < class DataType, int DIM > 
QQuadLag< DataType, DIM >::QQuadLag(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 2;
  this->name_ = "Q_Quad_Lagrange";
  this->type_ = AnsatzSpaceType::Q_LAGRANGE;

  assert (this->ref_cell_->type() == RefCellType::QUAD_STD);
}

template < class DataType, int DIM > 
void QQuadLag< DataType, DIM >::init ( size_t degree )
{
  this->init(degree, 1);
}

template < class DataType, int DIM > 
void QQuadLag< DataType, DIM >::init ( size_t degree, size_t nb_comp )
{
  std::vector< std::vector< size_t > > degrees (nb_comp);
  for (size_t l=0; l<nb_comp; ++l)
  {
    degrees[l].resize(DIM, degree);
  }
  this->init(degrees);
}

template < class DataType, int DIM > 
void QQuadLag< DataType, DIM >::init ( const std::vector< size_t > &degrees )
{
  std::vector< std::vector< size_t > > new_degrees (degrees.size());
  for (size_t l=0; l<degrees.size(); ++l)
  {
    new_degrees[l].resize(DIM, degrees[l]);
  }
  this->init(new_degrees);
}

template < class DataType, int DIM > 
void QQuadLag< DataType, DIM >::init ( const std::vector< std::vector<size_t> > &degrees )
{
  assert (DIM == 2);
  assert (degrees.size() > 0);
  this->tdim_ = 2;
  this->nb_comp_ = degrees.size();
  this->my_degrees_ = degrees;
  this->comp_weight_size_.clear();
  this->comp_weight_size_.resize(this->nb_comp_, 0);
  this->comp_offset_.clear();
  this->comp_offset_.resize(this->nb_comp_, 0);
  
  this->dim_ = 0;
  this->max_deg_ = 0;

  this->nb_dof_on_line_.clear();
  this->nb_dof_on_line_.resize(this->nb_comp_);
  size_t sum = 0; 
  
  for (int l=0; l<this->nb_comp_; ++l)
  {
    // number of space directions has to equal 2 (dimension of quad)
    assert (degrees[l].size() == DIM);
    this->nb_dof_on_line_[l].resize(2,0.);
    
    this->comp_weight_size_[l] = (degrees[l][0] + 1) * (degrees[l][1] + 1); 
    this->comp_offset_[l] = sum;
    sum += this->comp_weight_size_[l];
   
    this->nb_dof_on_line_[l][0] = degrees[l][0] + 1;
    this->nb_dof_on_line_[l][1] = degrees[l][1] + 1;

    this->max_deg_ = std::max(this->max_deg_, degrees[l][0]);
    this->max_deg_ = std::max(this->max_deg_, degrees[l][1]);
  }
  // since this is a tensor product space
  this->dim_ = sum;
  this->weight_size_ = this->dim_ * this->nb_comp_;
}

template < class DataType, int DIM >
size_t QQuadLag< DataType, DIM >::ij2ind(size_t i, size_t j, size_t comp) const 
{
  assert (DIM == 2);
  return i + j * this->nb_dof_on_line_[comp][0];
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < this->my_deg_. All possible combinations are
///          multiplied and considered in the sum.

template < class DataType, int DIM >
void QQuadLag< DataType, DIM >::N(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert(DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());

  if (this->comp_weight_size_[comp] > 1) 
  {
    for (int j = 0; j <= this->my_degrees_[comp][1]; ++j) {
      const DataType lp_j = this->lp_.poly(this->my_degrees_[comp][1], j, pt[1]);
      for (int i = 0; i <= this->my_degrees_[comp][0]; ++i) {
        weight[offset + ij2ind(i, j, comp)] = this->lp_.poly(this->my_degrees_[comp][0], i, pt[0]) * lp_j;
      }
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 1.0;
  }
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < this->my_deg_. All possible combinations are
///          multiplied and considered in the sum, w.r.t. the derivatives for
///          the x - variable.

template < class DataType, int DIM >
void QQuadLag< DataType, DIM >::N_x(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert(DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());

  if (this->comp_weight_size_[comp] > 1) 
  {
    for (int j = 0; j <= this->my_degrees_[comp][1]; ++j) {
      const DataType lp_j = this->lp_.poly(this->my_degrees_[comp][1], j, pt[1]);
      for (int i = 0; i <= this->my_degrees_[comp][0]; ++i) {
        weight[offset + ij2ind(i, j, comp)] = this->lp_.poly_x(this->my_degrees_[comp][0], i, pt[0]) * lp_j;
      }
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 0.0;
  }
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < this->my_deg_. All possible combinations are
///          multiplied and considered in the sum, w.r.t. the derivatives for
///          the y - variable.

template < class DataType, int DIM >
void QQuadLag< DataType, DIM >::N_y(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert(DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());

  if (this->comp_weight_size_[comp] > 1) 
  {
    for (int j = 0; j <= this->my_degrees_[comp][1]; ++j) {
      const DataType lp_j = this->lp_.poly_x(this->my_degrees_[comp][1], j, pt[1]);
      for (int i = 0; i <= this->my_degrees_[comp][0]; ++i) {
        weight[offset + ij2ind(i, j, comp)] = this->lp_.poly(this->my_degrees_[comp][0], i, pt[0]) * lp_j;
      }
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 0.0;
  }
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < this->my_deg_. All possible combinations are
///          multiplied and considered in the sum, w.r.t. the second derivatives
///          for the xx - variable.

template < class DataType, int DIM >
void QQuadLag< DataType, DIM >::N_xx(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert(DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());

  if (this->comp_weight_size_[comp] > 1) 
  {
    for (int j = 0; j <= this->my_degrees_[comp][1]; ++j) {
      const DataType lp_j = this->lp_.poly(this->my_degrees_[comp][1], j, pt[1]);
      for (int i = 0; i <= this->my_degrees_[comp][0]; ++i) {
        weight[offset + ij2ind(i, j, comp)] =
            this->lp_.poly_xx(this->my_degrees_[comp][0], i, pt[0]) * lp_j;
      }
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 0.0;
  }
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < this->my_deg_. All possible combinations are
///          multiplied and considered in the sum, w.r.t. the second derivatives
///          for the xy - variable.

template < class DataType, int DIM >
void QQuadLag< DataType, DIM >::N_xy(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert(DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());

  if (this->comp_weight_size_[comp] > 1) 
  {
    for (int j = 0; j <= this->my_degrees_[comp][1]; ++j) {
      const DataType lp_j = this->lp_.poly_x(this->my_degrees_[comp][1], j, pt[1]);
      for (int i = 0; i <= this->my_degrees_[comp][0]; ++i) {
        weight[offset + ij2ind(i, j, comp)] = this->lp_.poly_x(this->my_degrees_[comp][0], i, pt[0]) * lp_j;
      }
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 0.0;
  }
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < this->my_deg_. All possible combinations are
///          multiplied and considered in the sum, w.r.t. the second derivatives
///          for the yy - variable.

template < class DataType, int DIM >
void QQuadLag< DataType, DIM >::N_yy(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert(DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());

  if (this->comp_weight_size_[comp] > 1) 
  {
    for (int j = 0; j <= this->my_degrees_[comp][1]; ++j) {
      const double lp_j = this->lp_.poly_xx(this->my_degrees_[comp][1], j, pt[1]);
      for (int i = 0; i <= this->my_degrees_[comp][0]; ++i) {
        weight[offset + ij2ind(i, j, comp)] = this->lp_.poly(this->my_degrees_[comp][0], i, pt[0]) * lp_j;
      }
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 0.0;
  }
}

} // namespace doffem
} // namespace hiflow
#endif
