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

#ifndef __ANSATZ_PTET_LAGRANGE_H_
#define __ANSATZ_PTET_LAGRANGE_H_

#include "fem/ansatz/ansatz_space.h"
#include "polynomials/lagrangepolynomial.h"
#include "common/vector_algebra_descriptor.h"
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
class PTetLag final : public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  PTetLag(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~PTetLag()
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
  void N_z (const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_xx(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_xy(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_xz(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_yy(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_yz(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  void N_zz(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const override;
  
  /// Index ordering form space (x=i,y=j,z=k) to vector index
  inline size_t ijk2ind(size_t i, size_t j, size_t k, size_t comp) const;

  /// Lagrange polynomials which are used for evaluating shapefunctions
  LagrangePolynomial< DataType > lp_;

  std::vector< size_t > my_degrees_;
  std::vector< size_t > nb_dof_on_line_;
};

template < class DataType, int DIM > 
PTetLag< DataType, DIM >::PTetLag(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 3;
  this->name_ = "P_Tet_Lagrange";
  this->type_ = AnsatzSpaceType::P_LAGRANGE;

  //assert (this->ref_cell_->type() == RefCellType::TET_STD);
}

template < class DataType, int DIM > 
void PTetLag< DataType, DIM >::init ( size_t degree )
{
  this->init(degree, 1);
}

template < class DataType, int DIM > 
void PTetLag< DataType, DIM >::init ( size_t degree, size_t nb_comp )
{
  std::vector< size_t > degrees (nb_comp);
  for (size_t l=0; l<nb_comp; ++l)
  {
    degrees[l] = degree;
  }
  this->init(degrees);
}

template < class DataType, int DIM > 
void PTetLag< DataType, DIM >::init ( const std::vector< std::vector<size_t> > &degrees )
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
void PTetLag< DataType, DIM >::init ( const std::vector< size_t > &degrees )
{
  assert (DIM == 3);
  assert (degrees.size() > 0);
  this->tdim_ = 3;
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
    
    this->comp_weight_size_[l] = (deg + 1) * (deg + 2) * (deg + 3) / 6;
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
///          x coordinate, then continuing on the y coordinate and last
///          continuing on the z coordinate this is achieved by computing the
///          corresponding offsets to consider the restriction given by the
///          tetrahedron which reads z in [0,1], y < 1 - z, x < 1 - y - z
template < class DataType, int DIM >
size_t PTetLag< DataType, DIM >::ijk2ind(size_t i, size_t j, size_t k, size_t comp) const 
{
  assert (DIM == 3);
  assert (comp < this->nb_comp_);

  // x component = i, y component = j, z component = k

  size_t offset = 0;
  const size_t nb_dof_line = this->nb_dof_on_line_[comp];

  // First: offset z axis

  for (size_t m = 0; m < k; ++m) 
  {
    const int help = nb_dof_line - m;
    for (size_t dof = 0; dof < nb_dof_line - m; ++dof) 
    {
      offset += help - dof;
    }
  }
  
  for (size_t n = 0; n < j; ++n) 
  {
    offset += nb_dof_line - n - k;
  }
  
  return (i + offset);
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree my_deg_". Since my_deg_ = 0 is also allowed, there are
///          several distinctions to be done. For performance reasons, the code
///          becomes a little bit trenched. But the main "for(int ...)",
///          representing what is really happening is found at the end of the
///          function. The values for the coordinates are transformed from the
///          cartesian system to the barycentric system. This means, given
///          (x,y,z) in cartesian sense, the barycentric coordinates read
///          (1-x-y-z, x, y, z). Also, they need to be scaled by the factor
///          (my_deg_ / polynomial degree). The resulting combination of the
///          polynomials which is computed is given by \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);

  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);

  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0) {
    weight[offset + ijk2ind(0, 0, 0, comp)] = this->lp_.poly(comp_deg, comp_deg, help);
  } else {
    weight[offset + ijk2ind(0, 0, 0, comp)] = 1.0;
  }

  for (size_t i = 1; i < comp_deg; ++i) {
    weight[offset + ijk2ind(i, 0, 0, comp)] = this->lp_.poly(comp_deg - i, comp_deg - i, deg * help / (deg - i)) 
                                            * this->lp_.poly(i, i, dp_0 / i);
  }

  if (comp_deg > 0) {
    weight[offset + ijk2ind(comp_deg, 0, 0, comp)] = this->lp_.poly(comp_deg, comp_deg, pt[0]);
  }

  if (comp_deg > 0) {
    weight[offset + ijk2ind(0, comp_deg, 0, comp)] = this->lp_.poly(comp_deg, comp_deg, pt[1]);
  }

  for (size_t j = 1; j < comp_deg; ++j) {
    weight[offset + ijk2ind(0, j, 0, comp)] = this->lp_.poly(comp_deg - j, comp_deg - j, deg * help / (deg - j)) 
                                            * this->lp_.poly(j, j, dp_1 / j);
  }

  for (size_t j = 1; j < comp_deg; ++j) {
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] = this->lp_.poly(comp_deg - j, comp_deg - j, dp_0 / (deg - j))   
                                                       * this->lp_.poly(j, j, deg * pt[1] / j);
  }

  for (size_t j = 1; j < comp_deg; ++j) {
    for (size_t i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] = this->lp_.poly(comp_deg - i - j, comp_deg - i - j, deg * help / (deg - i - j)) 
                                              * this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j);
    }
  }

  if (comp_deg > 0) {
    weight[offset + ijk2ind(0, 0, comp_deg, comp)] = this->lp_.poly(comp_deg, comp_deg, pt[2]);
  }

  for (size_t k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] = this->lp_.poly(comp_deg - k, comp_deg - k, deg * help / (deg - k)) 
                                            * this->lp_.poly(k, k, deg * pt[2] / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] = this->lp_.poly(comp_deg - k, comp_deg - k, dp_0 / (deg - k)) 
                                                       * this->lp_.poly(k, k, dp_2 / k);

    for (size_t i = 1; i < comp_deg - k; ++i) {
      weight[offset + ijk2ind(i, 0, k, comp)] = this->lp_.poly(comp_deg - i - k, comp_deg - i - k, deg * help / (deg - i - k)) 
                                              * this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k);
    }

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] = this->lp_.poly(comp_deg - k, comp_deg - k, dp_1 / (deg - k)) 
                                                       * this->lp_.poly(k, k, dp_2 / k);
  }

  for (size_t k = 1; k < comp_deg; ++k) {
    for (size_t j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] = this->lp_.poly(comp_deg - j - k, comp_deg - j - k, deg * help / (deg - j - k)) 
                                              * this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] = this->lp_.poly(comp_deg - k - j, comp_deg - k - j, dp_0 / (deg - k - j)) 
                                                             * this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k);
    }
  }

  // Main "for" loop
  for (size_t k = 1; k < comp_deg; ++k) {
    for (size_t j = 1; j < comp_deg - k; ++j) {
      for (size_t i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] = this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k, deg * help / (deg - i - j - k)) 
                                                * this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) 
                                                * this->lp_.poly(k, k, dp_2 / k);
      }
    }
  }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree my_deg_". Since my_deg_ = 0 is also allowed, there are
///          several distinctions to be done. For performance reasons, the code
///          becomes a little bit trenched. But the main "for(int ...)",
///          representing what is really happening is found at the end of the
///          function. The values for the coordinates are transformed from the
///          cartesian system to the barycentric system. This means, given
///          (x,y,z) in cartesian sense, the barycentric coordinates read
///          (1-x-y-z, x, y, z). Also, they need to be scaled by the factor
///          (my_deg_ / polynomial degree). The resulting combination of the
///          polynomials which is computed is given by \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for x are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_x(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
 
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0) {
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        -this->lp_.poly_x(comp_deg, comp_deg, help);
  } else {
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;
  }

  for (int i = 1; i < comp_deg; ++i) {
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        -(deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             deg * help / (deg - i)) *
            this->lp_.poly(i, i, dp_0 / i) +
        this->lp_.poly(comp_deg - i, comp_deg - i,
                       deg * help / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i);
  }

  if (comp_deg > 0) {
    weight[offset + ijk2ind(comp_deg, 0, 0, comp)] =
        this->lp_.poly_x(comp_deg, comp_deg, pt[0]);
  }

  weight[offset + ijk2ind(0, comp_deg, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j) {
    weight[offset + ijk2ind(0, j, 0, comp)] =
        -(deg / (deg - j)) *
        this->lp_.poly_x(comp_deg - j, comp_deg - j,
                         deg * help / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);
  }

  for (int j = 1; j < comp_deg; ++j) {
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] =
        (deg / (deg - j)) *
        this->lp_.poly_x(comp_deg - j, comp_deg - j,
                         dp_0 / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);
  }

  for (int j = 1; j < comp_deg; ++j) {
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          -(deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j);
    }
  }

  weight[offset + ijk2ind(0, 0, comp_deg, comp)] = 0.0;

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        -(deg / (deg - k)) *
        this->lp_.poly_x(comp_deg - k, comp_deg - k,
                         deg * help / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] =
        (deg / (deg - k)) *
        this->lp_.poly_x(comp_deg - k, comp_deg - k,
                         dp_0 / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);

    for (int i = 1; i < comp_deg - k; ++i) {
      weight[offset + ijk2ind(i, 0, k, comp)] =
          -(deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - i - k, comp_deg - i - k,
                         deg * help / (deg - i - k)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(k, k, dp_2 / k);
    }

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] = 0.0;
  }

  for (int k = 1; k < comp_deg; ++k) {
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          -(deg / (deg - j - k)) *
          this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                           deg * help / (deg - j - k)) *
          this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          (deg / (deg - k - j)) *
          this->lp_.poly_x(comp_deg - k - j, comp_deg - k - j,
                           dp_0 / (deg - k - j)) *
          this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k);
    }
  }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k) {
    for (int j = 1; j < comp_deg - k; ++j) {
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            -(deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k);
      }
    }
  }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y,z) in cartesian sense, the barycentric
///          coordinates read (1-x-y-z, x, y, z). Also, they need to be scaled
///          by the factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for y are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_y(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
  
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        -this->lp_.poly_x(comp_deg, comp_deg, help);
  else
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;

  for (int i = 1; i < comp_deg; ++i)
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        -(deg / (deg - i)) *
        this->lp_.poly_x(comp_deg - i, comp_deg - i,
                         deg * help / (deg - i)) *
        this->lp_.poly(i, i, dp_0 / i);

  weight[offset + ijk2ind(comp_deg, 0, 0, comp)] = 0.0;

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, comp_deg, 0, comp)] =
        this->lp_.poly_x(comp_deg, comp_deg, pt[1]);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(0, j, 0, comp)] =
        -(deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help / (deg - j)) *
            this->lp_.poly(j, j, dp_1 / j) +
        this->lp_.poly(comp_deg - j, comp_deg - j,
                       deg * help / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] =
        this->lp_.poly(comp_deg - j, comp_deg - j, dp_0 / (deg - j)) *
        (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          -(deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j);
    }

  weight[offset + ijk2ind(0, 0, comp_deg, comp)] = 0.0;

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        -(deg / (deg - k)) *
        this->lp_.poly_x(comp_deg - k, comp_deg - k,
                         deg * help / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] = 0.0;

    for (int i = 1; i < comp_deg - k; ++i)
      weight[offset + ijk2ind(i, 0, k, comp)] =
          -(deg / (deg - i - k)) *
          this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                           deg * help / (deg - i - k)) *
          this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] =
        (deg / (deg - k)) *
        this->lp_.poly_x(comp_deg - k, comp_deg - k,
                         deg * pt[1] / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);
  }

  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          -(deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - j - k, comp_deg - j - k,
                         deg * help / (deg - j - k)) *
              (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) *
              this->lp_.poly(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          this->lp_.poly(comp_deg - k - j, comp_deg - k - j,
                         dp_0 / (deg - k - j)) *
          (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) *
          this->lp_.poly(k, k, dp_2 / k);
    }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j)
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            -(deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
                this->lp_.poly_x(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k);
      }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y,z) in cartesian sense, the barycentric
///          coordinates read (1-x-y-z, x, y, z). Also, they need to be scaled
///          by the factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for z are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_z(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
  
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        -this->lp_.poly_x(comp_deg, comp_deg, help);
  else
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;

  for (int i = 1; i < comp_deg; ++i)
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        -(deg / (deg - i)) *
        this->lp_.poly_x(comp_deg - i, comp_deg - i,
                         deg * help / (deg - i)) *
        this->lp_.poly(i, i, dp_0 / i);

  weight[offset + ijk2ind(comp_deg, 0, 0, comp)] = 0.0;

  weight[offset + ijk2ind(0, comp_deg, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(0, j, 0, comp)] =
        -(deg / (deg - j)) *
        this->lp_.poly_x(comp_deg - j, comp_deg - j,
                         deg * help / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          -(deg / (deg - i - j)) *
          this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                           deg * help / (deg - i - j)) *
          this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j);
    }

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, comp_deg, comp)] =
        this->lp_.poly_x(comp_deg, comp_deg, pt[2]);

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        -(deg / (deg - k)) *
            this->lp_.poly_x(comp_deg - k, comp_deg - k,
                             deg * help / (deg - k)) *
            this->lp_.poly(k, k, dp_2 / k) +
        this->lp_.poly(comp_deg - k, comp_deg - k,
                       deg * help / (deg - k)) *
            (deg / k) * this->lp_.poly_x(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] =
        this->lp_.poly(comp_deg - k, comp_deg - k, dp_0 / (deg - k)) *
        (deg / k) * this->lp_.poly_x(k, k, dp_2 / k);

    for (int i = 1; i < comp_deg - k; ++i)
      weight[offset + ijk2ind(i, 0, k, comp)] =
          -(deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - i - k, comp_deg - i - k,
                         deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k);

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] =
        this->lp_.poly(comp_deg - k, comp_deg - k, dp_1 / (deg - k)) *
        (deg / k) * this->lp_.poly_x(k, k, dp_2 / k);
  }

  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          -(deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - j - k, comp_deg - j - k,
                         deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          this->lp_.poly(comp_deg - k - j, comp_deg - k - j,
                         dp_0 / (deg - k - j)) *
          this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
          this->lp_.poly_x(k, k, dp_2 / k);
    }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j)
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            -(deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
                this->lp_.poly_x(k, k, dp_2 / k);
      }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y,z) in cartesian sense, the barycentric
///          coordinates read (1-x-y-z, x, y, z). Also, they need to be scaled
///          by the factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for xx are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_xx(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
  
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help);
  else
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;

  for (int i = 1; i < comp_deg; ++i)
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
            this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                              deg * help / (deg - i)) *
            this->lp_.poly(i, i, dp_0 / i) -
        (deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             deg * help / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) -
        (deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             deg * help / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) +
        this->lp_.poly(comp_deg - i, comp_deg - i,
                       deg * help / (deg - i)) *
            (deg / i) * (deg / i) * this->lp_.poly_xx(i, i, dp_0 / i);

  if (comp_deg > 0)
    weight[offset + ijk2ind(comp_deg, 0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, pt[0]);

  weight[offset + ijk2ind(0, comp_deg, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(0, j, 0, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
        this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                          deg * help / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
        this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                          dp_0 / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
              this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                                deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help / (deg - i - j)) *
              (deg / i) * (deg / i) * this->lp_.poly_xx(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j);
    }

  weight[offset + ijk2ind(0, 0, comp_deg, comp)] = 0.0;

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        (deg / (deg - k)) * (deg / (deg - k)) *
        this->lp_.poly_xx(comp_deg - k, comp_deg - k,
                          deg * help / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] =
        (deg / (deg - k)) * (deg / (deg - k)) *
        this->lp_.poly_xx(comp_deg - k, comp_deg - k,
                          dp_0 / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);

    for (int i = 1; i < comp_deg - k; ++i)
      weight[offset + ijk2ind(i, 0, k, comp)] =
          (deg / (deg - i - k)) * (deg / (deg - i - k)) *
              this->lp_.poly_xx(comp_deg - i - k, comp_deg - i - k,
                                deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - i - k, comp_deg - i - k,
                         deg * help / (deg - i - k)) *
              (deg / i) * (deg / i) * this->lp_.poly_xx(i, i, dp_0 / i) *
              this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] = 0.0;
  }

  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          (deg / (deg - j - k)) * (deg / (deg - j - k)) *
          this->lp_.poly_xx(comp_deg - j - k, comp_deg - j - k,
                            deg * help / (deg - j - k)) *
          this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          (deg / (deg - k - j)) * (deg / (deg - k - j)) *
          this->lp_.poly_xx(comp_deg - k - j, comp_deg - k - j,
                            dp_0 / (deg - k - j)) *
          this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k);
    }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j)
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            (deg / (deg - i - j - k)) * (deg / (deg - i - j - k)) *
                this->lp_.poly_xx(comp_deg - i - j - k,
                                  comp_deg - i - j - k,
                                  deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                (deg / i) * (deg / i) * this->lp_.poly_xx(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k);
      }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y,z) in cartesian sense, the barycentric
///          coordinates read (1-x-y-z, x, y, z). Also, they need to be scaled
///          by the factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for xy are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_xy(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
  
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help);
  else
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;

  for (int i = 1; i < comp_deg; ++i)
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
            this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                              deg * help / (deg - i)) *
            this->lp_.poly(i, i, dp_0 / i) -
        (deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             deg * help / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i);

  weight[offset + ijk2ind(comp_deg, 0, 0, comp)] = 0.0;

  weight[offset + ijk2ind(0, comp_deg, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(0, j, 0, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
            this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                              deg * help / (deg - j)) *
            this->lp_.poly(j, j, dp_1 / j) -
        (deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] =
        (deg / (deg - j)) *
        this->lp_.poly_x(comp_deg - j, comp_deg - j,
                         dp_0 / (deg - j)) *
        (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
              this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                                deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j);
    }

  weight[offset + ijk2ind(0, 0, comp_deg, comp)] = 0.0;

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        (deg / (deg - k)) * (deg / (deg - k)) *
        this->lp_.poly_xx(comp_deg - k, comp_deg - k,
                          deg * help / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] = 0.0;

    for (int i = 1; i < comp_deg - k; ++i)
      weight[offset + ijk2ind(i, 0, k, comp)] =
          (deg / (deg - i - k)) * (deg / (deg - i - k)) *
              this->lp_.poly_xx(comp_deg - i - k, comp_deg - i - k,
                                deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] = 0.0;
  }

  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          (deg / (deg - j - k)) * (deg / (deg - j - k)) *
              this->lp_.poly_xx(comp_deg - j - k, comp_deg - j - k,
                                deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) *
              this->lp_.poly(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          (deg / (deg - k - j)) *
          this->lp_.poly_x(comp_deg - k - j, comp_deg - k - j,
                           dp_0 / (deg - k - j)) *
          (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) *
          this->lp_.poly(k, k, dp_2 / k);
    }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j)
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            (deg / (deg - i - j - k)) * (deg / (deg - i - j - k)) *
                this->lp_.poly_xx(comp_deg - i - j - k,
                                  comp_deg - i - j - k,
                                  deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
                this->lp_.poly_x(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) * (deg / j) *
                this->lp_.poly_x(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k);
      }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y,z) in cartesian sense, the barycentric
///          coordinates read (1-x-y-z, x, y, z). Also, they need to be scaled
///          by the factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for xz are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_xz(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
  
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help);
  else
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;

  for (int i = 1; i < comp_deg; ++i)
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
            this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                              deg * help / (deg - i)) *
            this->lp_.poly(i, i, dp_0 / i) -
        (deg / (deg - i)) *
            this->lp_.poly_x(comp_deg - i, comp_deg - i,
                             deg * help / (deg - i)) *
            (deg / i) * this->lp_.poly_x(i, i, dp_0 / i);

  weight[offset + ijk2ind(comp_deg, 0, 0, comp)] = 0.0;

  weight[offset + ijk2ind(0, comp_deg, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(0, j, 0, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
        this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                          deg * help / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
              this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                                deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
              this->lp_.poly(j, j, dp_1 / j);
    }

  weight[offset + ijk2ind(0, 0, comp_deg, comp)] = 0.0;

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        (deg / (deg - k)) * (deg / (deg - k)) *
            this->lp_.poly_xx(comp_deg - k, comp_deg - k,
                              deg * help / (deg - k)) *
            this->lp_.poly(k, k, dp_2 / k) -
        (deg / (deg - k)) *
            this->lp_.poly_x(comp_deg - k, comp_deg - k,
                             deg * help / (deg - k)) *
            (deg / k) * this->lp_.poly_x(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] =
        (deg / (deg - k)) *
        this->lp_.poly_x(comp_deg - k, comp_deg - k,
                         dp_0 / (deg - k)) *
        (deg / k) * this->lp_.poly_x(k, k, dp_2 / k);

    for (int i = 1; i < comp_deg - k; ++i)
      weight[offset + ijk2ind(i, 0, k, comp)] =
          (deg / (deg - i - k)) * (deg / (deg - i - k)) *
              this->lp_.poly_xx(comp_deg - i - k, comp_deg - i - k,
                                deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k)

          - (deg / (deg - i - k)) *
                this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                                 deg * help / (deg - i - k)) *
                (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
                this->lp_.poly(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - i - k, comp_deg - i - k,
                         deg * help / (deg - i - k)) *
              (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k);

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] = 0.0;
  }

  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          (deg / (deg - j - k)) * (deg / (deg - j - k)) *
              this->lp_.poly_xx(comp_deg - j - k, comp_deg - j - k,
                                deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          (deg / (deg - k - j)) *
          this->lp_.poly_x(comp_deg - k - j, comp_deg - k - j,
                           dp_0 / (deg - k - j)) *
          this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
          this->lp_.poly_x(k, k, dp_2 / k);
    }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j)
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            (deg / (deg - i - j - k)) * (deg / (deg - i - j - k)) *
                this->lp_.poly_xx(comp_deg - i - j - k,
                                  comp_deg - i - j - k,
                                  deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
                this->lp_.poly_x(k, k, dp_2 / k)

            - (deg / (deg - i - j - k)) *
                  this->lp_.poly_x(comp_deg - i - j - k,
                                   comp_deg - i - j - k,
                                   deg * help / (deg - i - j - k)) *
                  (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
                  this->lp_.poly(j, j, dp_1 / j) *
                  this->lp_.poly(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                (deg / i) * this->lp_.poly_x(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
                this->lp_.poly_x(k, k, dp_2 / k);
      }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y,z) in cartesian sense, the barycentric
///          coordinates read (1-x-y-z, x, y, z). Also, they need to be scaled
///          by the factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for yy are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_yy(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
  
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help);
  else
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;

  for (int i = 1; i < comp_deg; ++i)
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
        this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                          deg * help / (deg - i)) *
        this->lp_.poly(i, i, dp_0 / i);

  weight[offset + ijk2ind(comp_deg, 0, 0, comp)] = 0.0;

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, comp_deg, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, pt[1]);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(0, j, 0, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
            this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                              deg * help / (deg - j)) *
            this->lp_.poly(j, j, dp_1 / j) -
        (deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) -
        (deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) +
        this->lp_.poly(comp_deg - j, comp_deg - j,
                       deg * help / (deg - j)) *
            (deg / j) * (deg / j) * this->lp_.poly_xx(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] =
        this->lp_.poly(comp_deg - j, comp_deg - j, dp_0 / (deg - j)) *
        (deg / j) * (deg / j) * this->lp_.poly_xx(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
              this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                                deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j) +
          this->lp_.poly(comp_deg - i - j, comp_deg - i - j,
                         deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) * (deg / j) *
              this->lp_.poly_xx(j, j, dp_1 / j);
    }

  weight[offset + ijk2ind(0, 0, comp_deg, comp)] = 0.0;

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        (deg / (deg - k)) * (deg / (deg - k)) *
        this->lp_.poly_xx(comp_deg - k, comp_deg - k,
                          deg * help / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] = 0.0;

    for (int i = 1; i < comp_deg - k; ++i)
      weight[offset + ijk2ind(i, 0, k, comp)] =
          (deg / (deg - i - k)) * (deg / (deg - i - k)) *
          this->lp_.poly_xx(comp_deg - i - k, comp_deg - i - k,
                            deg * help / (deg - i - k)) *
          this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k);

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] =
        (deg / (deg - k)) * (deg / (deg - k)) *
        this->lp_.poly_xx(comp_deg - k, comp_deg - k,
                          dp_1 / (deg - k)) *
        this->lp_.poly(k, k, dp_2 / k);
  }

  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          (deg / (deg - j - k)) * (deg / (deg - j - k)) *
              this->lp_.poly_xx(comp_deg - j - k, comp_deg - j - k,
                                deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) *
              this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) *
              this->lp_.poly(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - j - k, comp_deg - j - k,
                         deg * help / (deg - j - k)) *
              (deg / j) * (deg / j) * this->lp_.poly_xx(j, j, dp_1 / j) *
              this->lp_.poly(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          this->lp_.poly(comp_deg - k - j, comp_deg - k - j,
                         dp_0 / (deg - k - j)) *
          (deg / j) * (deg / j) * this->lp_.poly_xx(j, j, dp_1 / j) *
          this->lp_.poly(k, k, dp_2 / k);
    }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j)
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            (deg / (deg - i - j - k)) * (deg / (deg - i - j - k)) *
                this->lp_.poly_xx(comp_deg - i - j - k,
                                  comp_deg - i - j - k,
                                  deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
                this->lp_.poly_x(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
                this->lp_.poly_x(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) * (deg / j) * (deg / j) *
                this->lp_.poly_xx(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k);
      }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y,z) in cartesian sense, the barycentric
///          coordinates read (1-x-y-z, x, y, z). Also, they need to be scaled
///          by the factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for yz are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_yz(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
  
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help);
  else
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;

  for (int i = 1; i < comp_deg; ++i)
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
        this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                          deg * help / (deg - i)) *
        this->lp_.poly(i, i, dp_0 / i);

  weight[offset + ijk2ind(comp_deg, 0, 0, comp)] = 0.0;

  weight[offset + ijk2ind(0, comp_deg, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(0, j, 0, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
            this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                              deg * help / (deg - j)) *
            this->lp_.poly(j, j, dp_1 / j) -
        (deg / (deg - j)) *
            this->lp_.poly_x(comp_deg - j, comp_deg - j,
                             deg * help / (deg - j)) *
            (deg / j) * this->lp_.poly_x(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
              this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                                deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j) -
          (deg / (deg - i - j)) *
              this->lp_.poly_x(comp_deg - i - j, comp_deg - i - j,
                               deg * help / (deg - i - j)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
              this->lp_.poly_x(j, j, dp_1 / j);
    }

  weight[offset + ijk2ind(0, 0, comp_deg, comp)] = 0.0;

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        (deg / (deg - k)) * (deg / (deg - k)) *
            this->lp_.poly_xx(comp_deg - k, comp_deg - k,
                              deg * help / (deg - k)) *
            this->lp_.poly(k, k, dp_2 / k) -
        (deg / (deg - k)) *
            this->lp_.poly_x(comp_deg - k, comp_deg - k,
                             deg * help / (deg - k)) *
            (deg / k) * this->lp_.poly_x(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] = 0.0;

    for (int i = 1; i < comp_deg - k; ++i)
      weight[offset + ijk2ind(i, 0, k, comp)] =
          (deg / (deg - i - k)) * (deg / (deg - i - k)) *
              this->lp_.poly_xx(comp_deg - i - k, comp_deg - i - k,
                                deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k);

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] =
        (deg / (deg - k)) *
        this->lp_.poly_x(comp_deg - k, comp_deg - k,
                         dp_1 / (deg - k)) *
        (deg / k) * this->lp_.poly_x(k, k, dp_2 / k);
  }

  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          (deg / (deg - j - k)) * (deg / (deg - j - k)) *
              this->lp_.poly_xx(comp_deg - j - k, comp_deg - j - k,
                                deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k) -
          (deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) *
              this->lp_.poly(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - j - k, comp_deg - j - k,
                         deg * help / (deg - j - k)) *
              (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          this->lp_.poly(comp_deg - k - j, comp_deg - k - j,
                         dp_0 / (deg - k - j)) *
          (deg / j) * this->lp_.poly_x(j, j, dp_1 / j) * (deg / k) *
          this->lp_.poly_x(k, k, dp_2 / k);
    }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j)
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            (deg / (deg - i - j - k)) * (deg / (deg - i - j - k)) *
                this->lp_.poly_xx(comp_deg - i - j - k,
                                  comp_deg - i - j - k,
                                  deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
                this->lp_.poly_x(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
                this->lp_.poly_x(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) * (deg / j) *
                this->lp_.poly_x(j, j, dp_1 / j) * (deg / k) *
                this->lp_.poly_x(k, k, dp_2 / k);
      }
}

/// \details The restriction of lagrangian finite elements on a tetrahedron
/// reads
///          "sum of all multiplied polynomial degrees is less or equal to the
///          total degree comp_deg". Since comp_deg = 0 is also
///          allowed, there are several distinctions to be done. For performance
///          reasons, the code becomes a little bit trenched. But the main
///          "for(int ...)", representing what is really happening is found at
///          the end of the function. The values for the coordinates are
///          transformed from the cartesian system to the barycentric system.
///          This means, given (x,y,z) in cartesian sense, the barycentric
///          coordinates read (1-x-y-z, x, y, z). Also, they need to be scaled
///          by the factor (comp_deg / polynomial degree). The resulting
///          combination of the polynomials which is computed is given by
///          \f$L^{d-i-j-k}_{d-i-j-k}
///          ((d/(d-i-j-k)^*(1-x-y-z))^*L^i_i(d/i^*x)^*L^j_j(d/j^*y)^*L^k_k(d/k^*z)\f$
///          Here, the derivatives for zz are considered via the chain rule.

template < class DataType, int DIM >
void PTetLag< DataType, DIM >::N_zz(const Coord &pt, size_t comp, size_t offset, std::vector< DataType > &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  const size_t comp_deg = this->my_degrees_[comp];
  const DataType deg = static_cast< DataType >(comp_deg);
  
  const DataType help = 1.0 - pt[0] - pt[1] - pt[2];
  const DataType dp_0 = deg * pt[0];
  const DataType dp_1 = deg * pt[1];
  const DataType dp_2 = deg * pt[2];

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, 0, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, help);
  else
    weight[offset + ijk2ind(0, 0, 0, comp)] = 0.0;

  for (int i = 1; i < comp_deg; ++i)
    weight[offset + ijk2ind(i, 0, 0, comp)] =
        (deg / (deg - i)) * (deg / (deg - i)) *
        this->lp_.poly_xx(comp_deg - i, comp_deg - i,
                          deg * help / (deg - i)) *
        this->lp_.poly(i, i, dp_0 / i);

  weight[offset + ijk2ind(comp_deg, 0, 0, comp)] = 0.0;

  weight[offset + ijk2ind(0, comp_deg, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(0, j, 0, comp)] =
        (deg / (deg - j)) * (deg / (deg - j)) *
        this->lp_.poly_xx(comp_deg - j, comp_deg - j,
                          deg * help / (deg - j)) *
        this->lp_.poly(j, j, dp_1 / j);

  for (int j = 1; j < comp_deg; ++j)
    weight[offset + ijk2ind(comp_deg - j, j, 0, comp)] = 0.0;

  for (int j = 1; j < comp_deg; ++j)
    for (int i = 1; i < comp_deg - j; ++i) {
      weight[offset + ijk2ind(i, j, 0, comp)] =
          (deg / (deg - i - j)) * (deg / (deg - i - j)) *
          this->lp_.poly_xx(comp_deg - i - j, comp_deg - i - j,
                            deg * help / (deg - i - j)) *
          this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(j, j, dp_1 / j);
    }

  if (comp_deg > 0)
    weight[offset + ijk2ind(0, 0, comp_deg, comp)] =
        this->lp_.poly_xx(comp_deg, comp_deg, pt[2]);

  for (int k = 1; k < comp_deg; ++k) {
    weight[offset + ijk2ind(0, 0, k, comp)] =
        (deg / (deg - k)) * (deg / (deg - k)) *
            this->lp_.poly_xx(comp_deg - k, comp_deg - k,
                              deg * help / (deg - k)) *
            this->lp_.poly(k, k, dp_2 / k) -
        (deg / (deg - k)) *
            this->lp_.poly_x(comp_deg - k, comp_deg - k,
                             deg * help / (deg - k)) *
            (deg / k) * this->lp_.poly_x(k, k, dp_2 / k) -
        (deg / (deg - k)) *
            this->lp_.poly_x(comp_deg - k, comp_deg - k,
                             deg * help / (deg - k)) *
            (deg / k) * this->lp_.poly_x(k, k, dp_2 / k) +
        this->lp_.poly(comp_deg - k, comp_deg - k,
                       deg * help / (deg - k)) *
            (deg / k) * (deg / k) * this->lp_.poly_xx(k, k, dp_2 / k);

    weight[offset + ijk2ind(comp_deg - k, 0, k, comp)] =
        this->lp_.poly(comp_deg - k, comp_deg - k, dp_0 / (deg - k)) *
        (deg / k) * (deg / k) * this->lp_.poly_xx(k, k, dp_2 / k);

    for (int i = 1; i < comp_deg - k; ++i)
      weight[offset + ijk2ind(i, 0, k, comp)] =
          (deg / (deg - i - k)) * (deg / (deg - i - k)) *
              this->lp_.poly_xx(comp_deg - i - k, comp_deg - i - k,
                                deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k) -
          (deg / (deg - i - k)) *
              this->lp_.poly_x(comp_deg - i - k, comp_deg - i - k,
                               deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - i - k, comp_deg - i - k,
                         deg * help / (deg - i - k)) *
              this->lp_.poly(i, i, dp_0 / i) * (deg / k) * (deg / k) *
              this->lp_.poly_xx(k, k, dp_2 / k);

    weight[offset + ijk2ind(0, comp_deg - k, k, comp)] =
        this->lp_.poly(comp_deg - k, comp_deg - k, dp_1 / (deg - k)) *
        (deg / k) * (deg / k) * this->lp_.poly_xx(k, k, dp_2 / k);
  }

  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j) {
      weight[offset + ijk2ind(0, j, k, comp)] =
          (deg / (deg - j - k)) * (deg / (deg - j - k)) *
              this->lp_.poly_xx(comp_deg - j - k, comp_deg - j - k,
                                deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * this->lp_.poly(k, k, dp_2 / k) -
          (deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k) -
          (deg / (deg - j - k)) *
              this->lp_.poly_x(comp_deg - j - k, comp_deg - j - k,
                               deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
              this->lp_.poly_x(k, k, dp_2 / k) +
          this->lp_.poly(comp_deg - j - k, comp_deg - j - k,
                         deg * help / (deg - j - k)) *
              this->lp_.poly(j, j, dp_1 / j) * (deg / k) * (deg / k) *
              this->lp_.poly_xx(k, k, dp_2 / k);

      weight[offset + ijk2ind(comp_deg - k - j, j, k, comp)] =
          this->lp_.poly(comp_deg - k - j, comp_deg - k - j,
                         dp_0 / (deg - k - j)) *
          this->lp_.poly(j, j, dp_1 / j) * (deg / k) * (deg / k) *
          this->lp_.poly_xx(k, k, dp_2 / k);
    }

  // Main "for" loop
  for (int k = 1; k < comp_deg; ++k)
    for (int j = 1; j < comp_deg - k; ++j)
      for (int i = 1; i < comp_deg - k - j; ++i) {
        weight[offset + ijk2ind(i, j, k, comp)] =
            (deg / (deg - i - j - k)) * (deg / (deg - i - j - k)) *
                this->lp_.poly_xx(comp_deg - i - j - k,
                                  comp_deg - i - j - k,
                                  deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) *
                this->lp_.poly(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
                this->lp_.poly_x(k, k, dp_2 / k) -
            (deg / (deg - i - j - k)) *
                this->lp_.poly_x(comp_deg - i - j - k,
                                 comp_deg - i - j - k,
                                 deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * (deg / k) *
                this->lp_.poly_x(k, k, dp_2 / k) +
            this->lp_.poly(comp_deg - i - j - k, comp_deg - i - j - k,
                           deg * help / (deg - i - j - k)) *
                this->lp_.poly(i, i, dp_0 / i) *
                this->lp_.poly(j, j, dp_1 / j) * (deg / k) * (deg / k) *
                this->lp_.poly_xx(k, k, dp_2 / k);
      }
}

} // namespace doffem
} // namespace hiflow
#endif
