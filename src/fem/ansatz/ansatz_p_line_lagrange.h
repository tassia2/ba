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

#ifndef __ANSATZ_PLINE_LAGRANGE_H_
#define __ANSATZ_PLINE_LAGRANGE_H_

#include "fem/ansatz/ansatz_space.h"
#include "polynomials/lagrangepolynomial.h"
#include "common/vector_algebra_descriptor.h"
#include <cassert>

namespace hiflow {
namespace doffem {

///
/// \class PLineLag felagrange_line.h
/// \brief Lagrangian Finite Element on a line
/// \author Michael Schick<br>Martin Baumann<br>Julian Kraemer<br>Simon Gawlok<br>Philipp Gerstner
///

template < class DataType, int DIM >
class PLineLag final : public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  PLineLag(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~PLineLag()
  {
  }

  void init (size_t degree);
  void init (size_t degree, size_t nb_comp);
  void init (const std::vector< size_t > &degrees);
  void init (const std::vector< std::vector<size_t> > &degrees);

private:
  void compute_degree_hash () const
  {
    NOT_YET_IMPLEMENTED;
  }
  
  void N(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_x(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_xx(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;

  /// Index ordering form space (x=i,y=j,z=k) to vector index
  inline size_t ij2ind(size_t i, size_t j, size_t comp) const;

  /// Lagrange polynomials which are used for evaluating shapefunctions
  LagrangePolynomial< DataType > lp_;

  std::vector< size_t > my_degrees_;
  std::vector< size_t > nb_dof_on_line_;
};

template < class DataType, int DIM > 
PLineLag<DataType, DIM>::PLineLag (CRefCellSPtr<DataType, DIM> ref_cell)
 : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 1;
  this->name_ = "P_Line_Lagrange";
  this->type_ = AnsatzSpaceType::P_LAGRANGE;

  assert (this->ref_cell_->type() == RefCellType::LINE_STD);
}

template < class DataType, int DIM > 
void PLineLag< DataType, DIM >::init ( size_t degree )
{
  this->init(degree, 1);
}

template < class DataType, int DIM > 
void PLineLag< DataType, DIM >::init ( size_t degree, size_t nb_comp )
{
  std::vector< size_t > degrees (nb_comp);
  for (size_t l=0; l<nb_comp; ++l)
  {
    degrees[l] = degree;
  }
  this->init(degrees);
}

template < class DataType, int DIM > 
void PLineLag< DataType, DIM >::init ( const std::vector< std::vector<size_t> > &degrees )
{
  std::vector<size_t> new_deg(degrees.size(), 0);
  for (size_t l=0; l<degrees.size(); ++l)
  {
    // number of space directions has to equal 1 (dimension of line)
    assert (degrees[l].size() == 1);
    new_deg[l] = degrees[l][0];
  }
  this->init (new_deg);
}

template < class DataType, int DIM > 
void PLineLag< DataType, DIM >::init ( const std::vector< size_t > &degrees )
{
  assert (DIM == 1);
  assert (degrees.size() > 0);
  this->tdim_ = 1;
  this->nb_comp_ = degrees.size();
  this->my_degrees_ = degrees;
  this->comp_weight_size_.clear();
  this->comp_weight_size_.resize(this->nb_comp_, 0);
  this->comp_offset_.resize(this->nb_comp_, 0);
  
  this->dim_ = 0;
  this->max_deg_ = 0;

  this->nb_dof_on_line_.clear();
  this->nb_dof_on_line_.resize(this->nb_comp_);
  size_t sum = 0; 
  
  for (size_t l=0; l<this->nb_comp_; ++l)
  {
    size_t deg = degrees[l];
    
    this->comp_weight_size_[l] = deg + 1; 
    this->comp_offset_[l] = sum;
    sum += this->comp_weight_size_[l];
   
    this->nb_dof_on_line_[l] = deg + 1;

    this->max_deg_ = std::max(this->max_deg_, deg);
  }
  // since this is a tensor product space
  this->dim_ = sum;
  this->weight_size_ = this->dim_ * this->nb_comp_;
}

template < class DataType, int DIM >
size_t PLineLag< DataType, DIM >::ij2ind(size_t i, size_t j, size_t comp) const 
{
  assert (DIM == 1);
  return (j);
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < comp_deg. All possible combinations are
///          multiplied and considered in the sum.

template < class DataType, int DIM >
void PLineLag< DataType, DIM >::N(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 1);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];

  if (comp_deg > 0) {
    for (int j = 0; j <= comp_deg; ++j) {
      weight[offset + ij2ind(0, j, comp)] = this->lp_.poly(comp_deg, j, pt[0]);
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 1.0;
  }
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < comp_deg. All possible combinations are
///          multiplied and considered in the sum, w.r.t. the derivatives for
///          the x - variable.

template < class DataType, int DIM >
void PLineLag< DataType, DIM >::N_x(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 1);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];

  if (comp_deg > 0) {
    for (int j = 0; j <= comp_deg; ++j)
    {
      weight[offset + ij2ind(0, j, comp)] = this->lp_.poly_x(comp_deg, j, pt[0]);
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 0.0;
  }
}

/// \details Every degree of a used lagrangian polynomial has to satisfy the
///          condition degree < comp_deg. All possible combinations are
///          multiplied and considered in the sum, w.r.t. the second derivatives
///          for the xx - variable.

template < class DataType, int DIM >
void PLineLag< DataType, DIM >::N_xx(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 1);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];

  if (comp_deg > 0) {
    for (int j = 0; j <= comp_deg; ++j) {
      weight[offset + ij2ind(0, j, comp)] = this->lp_.poly_xx(comp_deg, j, pt[0]);
    }
  } else {
    weight[offset + ij2ind(0, 0, comp)] = 0.0;
  }
}

} // namespace doffem
} // namespace hiflow
#endif
