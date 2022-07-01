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

#ifndef __ANSATZ_PTRI_LAGRANGE_H_
#define __ANSATZ_PTRI_LAGRANGE_H_

#include "common/vector_algebra_descriptor.h"
#include "fem/ansatz/ansatz_space.h"
#include "polynomials/lagrangepolynomial.h"
#include <cassert>
#include <cmath>

namespace hiflow {
namespace doffem {

///
/// \class QHexLag ansatz_space_hex.h
/// \brief Q Lagrange polynomials on reference Hexahedron [0,1]x[0,1]x[0,1]
/// \author Michael Schick<br>Martin Baumann<br> Philipp Gerstner<br>Simon Gawlok
///

template < class DataType, int DIM >
class PTriLag final: public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  PTriLag(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~PTriLag()
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

  std::vector< size_t > my_degrees_;
  std::vector< size_t > nb_dof_on_line_;
};


template < class DataType, int DIM > 
PTriLag< DataType, DIM >::PTriLag(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 2;
  this->name_ = "P_Tri_Lagrange";
  this->type_ = AnsatzSpaceType::P_LAGRANGE;

  //assert (this->ref_cell_->type() == RefCellType::TRI_STD);
}

template < class DataType, int DIM > 
void PTriLag< DataType, DIM >::init ( size_t degree )
{
  this->init(degree, 1);
}

template < class DataType, int DIM > 
void PTriLag< DataType, DIM >::init ( size_t degree, size_t nb_comp )
{
  std::vector< size_t > degrees (nb_comp);
  for (size_t l=0; l<nb_comp; ++l)
  {
    degrees[l] = degree;
  }
  this->init(degrees);
}

template < class DataType, int DIM > 
void PTriLag< DataType, DIM >::init ( const std::vector< std::vector<size_t> > &degrees )
{
  std::vector<size_t> new_deg(degrees.size(), 0);
  for (size_t l=0; l<degrees.size(); ++l)
  {
    // different polynomial degrees for different space directions are not allowed for P elements
    assert (degrees[l].size() == 1);
    new_deg[l] = degrees[l][0];
  }
  this->init (new_deg);
}

template < class DataType, int DIM > 
void PTriLag< DataType, DIM >::init ( const std::vector< size_t > &degrees )
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
  
  for (size_t l=0; l<this->nb_comp_; ++l)
  {
    size_t deg = degrees[l];
    
    this->comp_weight_size_[l] = (deg + 1) * (deg + 2) / 2; 
    this->comp_offset_[l] = sum;
    sum += this->comp_weight_size_[l];
   
    this->nb_dof_on_line_[l] = deg + 1;

    this->max_deg_ = std::max(this->max_deg_, deg);
  }
  // since this is a tensor product space
  this->dim_ = sum;
  this->weight_size_ = this->dim_ * this->nb_comp_;
}

/// \details The counting of the dofs on the reference cell is done by the
///          lexicographical numbering strategy. Therefore, beginning on the
///          x coordinate, then continuing on the y coordinate this is achieved
///          by computing the corresponding offsets to consider the restriction
///          given by the triangle which reads y in [0,1], x < 1 - y

template < class DataType, int DIM >
size_t PTriLag< DataType, DIM >::ij2ind(size_t i, size_t j, size_t comp) const 
{
  assert (DIM == 2);
  assert (comp < this->nb_comp_);

  size_t offset = 0;
  const size_t nb_dof_line = this->nb_dof_on_line_[comp];

  for (size_t n = 0; n < j; ++n)
  {
    offset += nb_dof_line - n;
  }
  
  return (i + offset);
}

/// \details The restriction of lagrangian finite elements on a triangle reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree this->my_deg_". Since this->my_deg_ = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y in cartesian sense, the barycentric
///          coordinates read (1-x-y, x, y). Also, they need to be scaled by the
///          factor (this->my_deg_ / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j}_{d-i-j}
///          ((d/(d-i-j)^*(1-x-y))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)\f$

template < class DataType, int DIM >
void PTriLag< DataType, DIM >::N(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 2);
  assert(comp < this->nb_comp_);

  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);

  const DataType help1 = 1.0 - pt[0] - pt[1];
  const DataType dh_1 = deg * help1;

  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];

  if (comp_deg > 0)
  {
    weight[offset + this->ij2ind(0, 0, comp)] = this->lp_.poly(comp_deg, comp_deg, help1);
  }
  else
  {
    weight[offset + this->ij2ind(0, 0, comp)] = 1.0;
  }
  for (size_t i = 1; i < comp_deg; ++i)
  {
    weight[offset + this->ij2ind(i, 0, comp)] =
        this->lp_.poly(comp_deg - i, comp_deg - i, dh_1 / (deg - i)) *
        this->lp_.poly(i, i, dp_0 / i);
  }
  
  if (comp_deg > 0)
  {
    weight[offset + this->ij2ind(comp_deg, 0, comp)] =
        this->lp_.poly(comp_deg, comp_deg, pt[0]);
  }
  else
  {
    weight[offset + this->ij2ind(comp_deg, 0, comp)] = 1.0;
  }
  for (size_t j = 1; j < comp_deg; ++j)
  {
    weight[offset + this->ij2ind(0, j, comp)] =
        this->lp_.poly(comp_deg - j, comp_deg - j, dh_1 / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);
  }
  if (comp_deg > 0)
  {
    weight[offset + this->ij2ind(0, comp_deg, comp)] =
        this->lp_.poly(comp_deg, comp_deg, pt[1]);
  }
  else
  {
    weight[offset + this->ij2ind(0, comp_deg, comp)] = 1.0;
  }
  
  // Main "for" loop
  for (size_t j = 1; j < comp_deg; ++j) 
  {
    const DataType lp_j = this->lp_.poly(j, j, dp_1 / j);
    const size_t lp_offset = comp_deg - j;
    const DataType offset_double = static_cast< DataType >(deg - j);
    for (size_t i = 1; i < comp_deg - j; ++i) 
    {
      weight[offset + this->ij2ind(i, j, comp)] =
          this->lp_.poly(lp_offset - i, lp_offset - i, dh_1 / (offset_double - i)) *
          this->lp_.poly(i, i, dp_0 / i) * lp_j;
    }
  }

  for (size_t j = 1; j < comp_deg; ++j)
  {
    weight[offset + this->ij2ind(comp_deg - j, j, comp)] =
        this->lp_.poly(comp_deg - j, comp_deg - j, dp_0 / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);
  }
}

/// \details The restriction of lagrangian finite elements on a triangle reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y) in cartesian sense, the barycentric
///          coordinates read (1-x-y, x, y). Also, they need to be scaled by the
///          factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j}_{d-i-j}
///          ((d/(d-i-j)^*(1-x-y))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)\f$ Here, the
///          derivatives for x are considered via the chain rule.

template < class DataType, int DIM >
void PTriLag< DataType, DIM >::N_x(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);

  const DataType help1 = 1.0 - pt[0] - pt[1];
  const DataType dh_1 = deg * help1;

  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];

  if (comp_deg > 0)
    weight[offset + ij2ind(0, 0, comp)] =
        -this->lp_.poly_x(comp_deg, comp_deg, help1);
  else
    weight[offset + ij2ind(0, 0, comp)] = 0.0;

  for (size_t i = 1; i < comp_deg; ++i)
    weight[offset + ij2ind(i, 0, comp)] =
        (-deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             dh_1 / (deg - i)) *
            this->lp_.poly(i, i, dp_0 / i) +
        this->lp_.poly(comp_deg - i, comp_deg - i, dh_1 / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i);

  if (comp_deg > 0)
    weight[offset + ij2ind(comp_deg, 0, comp)] =
        this->lp_.poly_x(comp_deg, comp_deg, pt[0]);
  else
    weight[offset + ij2ind(comp_deg, 0, comp)] = 0.0;

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(0, j, comp)] =
        -(deg / (deg - j)) *
        this->lp_.poly_x(comp_deg - j, comp_deg - j,
                         dh_1 / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);

  weight[offset + ij2ind(0, comp_deg, comp)] = 0.0;

  // Main "for" loop
  for (size_t j = 1; j < comp_deg; ++j) {
    const DataType lp_j = this->lp_.poly(j, j, dp_1 / j);
    const size_t lp_offset = comp_deg - j;
    const DataType offset_double = static_cast< DataType >(deg - j);
    for (size_t i = 1; i < comp_deg - j; ++i) {
      weight[offset + ij2ind(i, j, comp)] =
          -(deg / (offset_double - i)) *
              this->lp_.poly_x(lp_offset - i, lp_offset - i,
                               dh_1 / (offset_double - i)) *
              this->lp_.poly(i, i, dp_0 / i) * lp_j +
          this->lp_.poly(lp_offset - i, lp_offset - i, dh_1 / (offset_double - i)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) * lp_j;
    }
  }

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(comp_deg - j, j, comp)] =
        (deg / (deg - j)) *
        this->lp_.poly_x(comp_deg - j, comp_deg - j,
                         dp_0 / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);
}

/// \details The restriction of lagrangian finite elements on a triangle reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y) in cartesian sense, the barycentric
///          coordinates read (1-x-y, x, y). Also, they need to be scaled by the
///          factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j}_{d-i-j}
///          ((d/(d-i-j)^*(1-x-y))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)\f$ Here, the
///          derivatives for y are considered via the chain rule.

template < class DataType, int DIM >
void PTriLag< DataType, DIM >::N_y(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);

  const DataType help1 = 1.0 - pt[0] - pt[1];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];

  if (comp_deg > 0)
    weight[offset + ij2ind(0, 0, comp)] =
        -this->lp_.poly_x(comp_deg, comp_deg, help1);
  else
    weight[offset + ij2ind(0, 0, comp)] = 0.0;

  for (size_t i = 1; i < comp_deg; ++i)
    weight[offset + ij2ind(i, 0, comp)] =
        -(deg / (deg - i)) *
        this->lp_.poly_x(comp_deg - i, comp_deg - i,
                         deg * help1 / (deg - i)) *
        this->lp_.poly(i, i, dp_0 / i);

  weight[offset + ij2ind(comp_deg, 0, comp)] = 0.0;

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(0, j, comp)] =
        -(deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help1 / (deg - j)) *
            this->lp_.poly(j, j, dp_1 / j) +
        this->lp_.poly(comp_deg - j, comp_deg - j,
                       deg * help1 / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);

  if (comp_deg > 0)
    weight[offset + ij2ind(0, comp_deg, comp)] =
        this->lp_.poly_x(comp_deg, comp_deg, pt[1]);
  else
    weight[offset + ij2ind(0, comp_deg, comp)] = 0.0;

  // Main "for" loop
  for (size_t j = 1; j < comp_deg; ++j)
    for (size_t i = 1; i < comp_deg - j; ++i) {
      weight[offset + ij2ind(i, j, comp)] =
          -(deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) *
              this->lp_.poly(j, j, deg * pt[1] / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, deg * pt[1] / j);
    }

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(comp_deg - j, j, comp)] =
        this->lp_.poly(comp_deg - j, comp_deg - j, dp_0 / (deg - j)) *
        (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);
}

/// \details The restriction of lagrangian finite elements on a triangle reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y) in cartesian sense, the barycentric
///          coordinates read (1-x-y, x, y). Also, they need to be scaled by the
///          factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j}_{d-i-j}
///          ((d/(d-i-j)^*(1-x-y))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)\f$ Here, the
///          derivatives for xx are considered via the chain rule.

template < class DataType, int DIM >
void PTriLag< DataType, DIM >::N_xx(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);

  const DataType help1 = 1.0 - pt[0] - pt[1];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];

  if (comp_deg > 0)
    weight[offset + ij2ind(0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help1);
  else
    weight[offset + ij2ind(0, 0, comp)] = 0.0;

  for (size_t i = 1; i < comp_deg; ++i)
    weight[offset + ij2ind(i, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
            this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                              deg * help1 / (deg - i)) *
            this->lp_.poly(i, i, dp_0 / i) -
        (deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             deg * help1 / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) -
        (deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             deg * help1 / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) +
        this->lp_.poly(comp_deg - i, comp_deg - i,
                       deg * help1 / (deg - i)) *
            (deg / i) * (deg / i) * this->lp_.poly_xx(i, i, dp_0 / i);

  if (comp_deg > 0)
    weight[offset + ij2ind(comp_deg, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, pt[0]);
  else
    weight[offset + ij2ind(comp_deg, 0, comp)] = 0.0;

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(0, j, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
        this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                          deg * help1 / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);

  weight[offset + ij2ind(0, comp_deg, comp)] = 0.0;

  // Main "for" loop
  for (size_t j = 1; j < comp_deg; ++j)
    for (size_t i = 1; i < comp_deg - j; ++i) {
      weight[offset + ij2ind(i, j, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
              this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                                deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) +
          (-deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help1 / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j) +
          (-deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help1 / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help1 / (deg - i - j)) *
              (deg / i) * (deg / i) * this->lp_.poly_xx(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j);
    }

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(comp_deg - j, j, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
        this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                          dp_0 / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);
}

/// \details The restriction of lagrangian finite elements on a triangle reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y) in cartesian sense, the barycentric
///          coordinates read (1-x-y, x, y). Also, they need to be scaled by the
///          factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j}_{d-i-j}
///          ((d/(d-i-j)^*(1-x-y))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)\f$ Here, the
///          derivatives for xy are considered via the chain rule.

template < class DataType, int DIM >
void PTriLag< DataType, DIM >::N_xy(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);

  const DataType help1 = 1.0 - pt[0] - pt[1];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];

  if (comp_deg > 0)
    weight[offset + ij2ind(0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help1);
  else
    weight[offset + ij2ind(0, 0, comp)] = 0.0;

  for (size_t i = 1; i < comp_deg; ++i)
    weight[offset + ij2ind(i, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
            this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                              deg * help1 / (deg - i)) *
            this->lp_.poly(i, i, dp_0 / i) -
        (deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             deg * help1 / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i);

  weight[offset + ij2ind(comp_deg, 0, comp)] = 0.0;

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(0, j, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
            this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                              deg * help1 / (deg - j)) *
            this->lp_.poly(j, j, dp_1 / j) -
        (deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help1 / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);

  weight[offset + ij2ind(0, comp_deg, comp)] = 0.0;

  // Main "for" loop
  for (size_t j = 1; j < comp_deg; ++j)
    for (size_t i = 1; i < comp_deg - j; ++i) {
      weight[offset + ij2ind(i, j, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
              this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                                deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help1 / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help1 / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j);
    }

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(comp_deg - j, j, comp)] =
        (deg / (deg - j)) *
        this->lp_.poly_x(comp_deg - j, comp_deg - j,
                         dp_0 / (deg - j)) *
        (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);
}

/// \details The restriction of lagrangian finite elements on a triangle reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y) in cartesian sense, the barycentric
///          coordinates read (1-x-y, x, y). Also, they need to be scaled by the
///          factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j}_{d-i-j}
///          ((d/(d-i-j)^*(1-x-y))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)\f$ Here, the
///          derivatives for yy are considered via the chain rule.

template < class DataType, int DIM >
void PTriLag< DataType, DIM >::N_yy(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 2);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);

  const DataType help1 = 1.0 - pt[0] - pt[1];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];

  if (comp_deg > 0)
    weight[offset + ij2ind(0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help1);
  else
    weight[offset + ij2ind(0, 0, comp)] = 0.0;

  for (size_t i = 1; i < comp_deg; ++i)
    weight[offset + ij2ind(i, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
        this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                          comp_deg * help1 / (deg - i)) *
        this->lp_.poly(i, i, dp_0 / i);

  weight[offset + ij2ind(comp_deg, 0, comp)] = 0.0;

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(0, j, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
            this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                              deg * help1 / (deg - j)) *
            this->lp_.poly(j, j, dp_1 / j) -
        (deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help1 / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) -
        (deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help1 / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) +
        this->lp_.poly(comp_deg - j, comp_deg - j,
                       deg * help1 / (deg - j)) *
            (deg / j) * (deg / j) * this->lp_.poly_xx(j, j, dp_1 / j);

  if (comp_deg > 0)
    weight[offset + ij2ind(0, comp_deg, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, pt[1]);
  else
    weight[offset + ij2ind(0, comp_deg, comp)] = 0.0;

  // Main "for" loop
  for (size_t j = 1; j < comp_deg; ++j)
    for (size_t i = 1; i < comp_deg - j; ++i) {
      weight[offset + ij2ind(i, j, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
              this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                                deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help1 / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) * (deg / j) *
              this->lp_.poly_xx(j, j, dp_1 / j);
    }

  for (size_t j = 1; j < comp_deg; ++j)
    weight[offset + ij2ind(comp_deg - j, j, comp)] =
        this->lp_.poly(comp_deg - j, comp_deg - j, dp_0 / (deg - j)) *
        (deg / j) * (deg / j) * this->lp_.poly_xx(j, j, dp_1 / j);
}

} // namespace doffem
} // namespace hiflow
#endif
