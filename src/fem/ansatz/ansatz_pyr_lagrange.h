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

#ifndef __ANSATZ_PPYR_LAGRANGE_H_
#define __ANSATZ_PPYR_LAGRANGE_H_

#include "fem/ansatz/ansatz_space.h"
#include "polynomials/lagrangepolynomial.h"
#include "common/vector_algebra_descriptor.h"
#include <cassert>
#include <cmath>
#include <iomanip>

namespace hiflow {
namespace doffem {

template < class DataType, int DIM >
class PyrLag final: public virtual AnsatzSpace< DataType, DIM > 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Default Constructor
  PyrLag(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~PyrLag()
  {}
  
  void init (size_t degree, size_t nb_comp)
  {
    assert (nb_comp == 1);
    this->init(degree);
  }
  
  void init (size_t degree);

private:
  void init_coord();

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
  
  /// Index ordering from 3 dimensional coordinates (x=i,y=j,z=k) to one vector
  inline int ijk2ind(int i, int j, int k) const;
  int my_deg_;
};

template < class DataType, int DIM > 
PyrLag< DataType, DIM >::PyrLag(CRefCellSPtr<DataType, DIM> ref_cell)
  : AnsatzSpace<DataType, DIM>(ref_cell)
{
  this->tdim_ = 3;
  this->name_ = "P_Pyr_Lagrange";
  this->type_ = AnsatzSpaceType::P_LAGRANGE;
  this->my_deg_ = -1;
  assert (this->ref_cell_->type() == RefCellType::PYR_STD);
}

template < class DataType, int DIM > 
void PyrLag< DataType, DIM >::init ( size_t degree )
{
  assert (DIM == 3);
  assert (degree <= 2);
  if (degree > 2) 
  {
    std::cerr << "Only support up to degree = 2 !" << std::endl;
    quit_program();
  }
  
  this->tdim_ = 3;
  this->nb_comp_ = 1;
  this->my_deg_ = degree;
  this->comp_weight_size_.clear();
  this->comp_weight_size_.resize(this->nb_comp_, 0);
  this->comp_offset_.clear();
  this->comp_offset_.resize(this->nb_comp_, 0);
  
  this->max_deg_ = degree;
  this->dim_ = 0;

  if (degree == 0)
  {
    this->dim_ = 1;
  }
  else if (degree == 1)
  {
    this->dim_ = 5;
  }
  else if (degree == 2)
  {
    this->dim_ = 14;
  }
  else
  {
    assert (false);
  }  
  
  this->comp_weight_size_[0] = this->dim_;
  this->weight_size_ = this->dim_;
}

template < class DataType, int DIM >
int PyrLag< DataType, DIM >::ijk2ind(int i, int j, int k) const 
{
  assert (DIM == 3);
  // x component = i, y component = j, z component = k

  int offset = 0;
  const int nb_dof_line = this->my_deg_ + 1;

  // First: offset z axis

  for (int m = 0; m < k; ++m) 
  {
    const int help = nb_dof_line - m;
    offset += help * help;
  }

  // Second: increasing offset by y axis on current z axis

  for (int n = 0; n < j; ++n) 
  {
    offset += nb_dof_line - k;
  }

  return (i + offset);
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));

  if (this->my_deg_ == 0) 
  {
    weight[offset+0] = 1.0;
  } 
  else if (this->my_deg_ == 1) 
  {
    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = -0.5 * z + (0.25 * x - 0.25) * (y - z - 1.0);
      weight[offset+1] = -0.5 * z + (0.25 * x + 0.25) * (-y + z + 1.0);
      weight[offset+2] = (-0.25 * x + 0.25) * (y - z + 1.0);
      weight[offset+3] = (0.25 * x + 0.25) * (y - z + 1.0);
      weight[offset+4] = z;
      break;
    case 2:
      weight[offset+0] = 0.25 * z * (x + y - 2.0) + (0.25 * x - 0.25) * (y - z - 1.0);
      weight[offset+1] = -0.25 * z * (x + y + 2.0) + (0.25 * x + 0.25) * (-y + z + 1.0);
      weight[offset+2] = -0.25 * z * (x + y) + (-0.25 * x + 0.25) * (y - z + 1.0);
      weight[offset+3] = 0.25 * z * (x + y) + (0.25 * x + 0.25) * (y - z + 1.0);
      weight[offset+4] = z;
      break;
    case 3:
      weight[offset+0] = (0.25 * x - 0.25) * (y + z - 1.0);
      weight[offset+1] = (0.25 * x + 0.25) * (-y - z + 1.0);
      weight[offset+2] = -0.5 * x * z + (-0.25 * x + 0.25) * (y - z + 1.0);
      weight[offset+3] = 0.5 * x * z + (0.25 * x + 0.25) * (y - z + 1.0);
      weight[offset+4] = z;
      break;
    case 4:
      weight[offset+0] = 0.25 * z * (x - y - 2.0) + (0.25 * x - 0.25) * (y - z - 1.0);
      weight[offset+1] = -0.25 * z * (x - y + 2.0) + (0.25 * x + 0.25) * (-y + z + 1.0);
      weight[offset+2] = -0.25 * z * (x - y) + (-0.25 * x + 0.25) * (y - z + 1.0);
      weight[offset+3] = 0.25 * z * (x - y) + (0.25 * x + 0.25) * (y - z + 1.0);
      weight[offset+4] = z;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 1.0 * weight[offset+0];
    weight[offset+1] = 1.0 * weight[offset+1];
    weight[offset+2] = 1.0 * weight[offset+2];
    weight[offset+3] = 1.0 * weight[offset+3];
    weight[offset+4] = 1.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0.5 * z * (x + y) -
                  0.5 * (0.25 * x - 0.25 * z) * (y - z) *
                      (4.0 * z - (-x + z + 1.0) * (-y + z + 1.0)) -
                  0.125 * (x + z) * (y - z) * (x + z - 1.0) * (-y + z + 1.0);
      weight[offset+1] = -0.5 * x2 * y2 + 0.5 * x2 * y * z +
                  0.5 * x2 * y - 1.0 * x2 * z -
                  0.5 * y2 * z + 0.5 * y2 + 0.5 * y * z2 -
                  0.5 * y;
      weight[offset+2] = -0.5 * z * (x - y) -
                  0.5 * (0.25 * x - 0.25 * z) * (y - z) * (x - z + 1.0) *
                      (-y + z + 1.0) +
                  0.125 * (x + z) * (y - z) *
                      (4.0 * z - (x + z + 1.0) * (-y + z + 1.0));
      weight[offset+3] =
          (0.5 * y - 0.5 * z + 0.5) * (0.5 * x * (y - 1.0) * (-x + z + 1.0) -
                                       0.5 * x * (y - 1.0) * (x + z - 1.0) -
                                       0.5 * z * (2.0 * y + 1.0) + 0.5 * z);
      weight[offset+4] =
          ((z * (x - y + z + 1.0) + (x + 1.0) * (y - 1.0)) * (x + z - 1.0) +
           (z * (x + y - z - 1.0) + (x - 1.0) * (y - 1.0)) * (x - z + 1.0)) *
          (0.5 * y - 0.5 * z + 0.5);
      weight[offset+5] = (-0.25 * y + 0.25 * z - 0.25) *
                  (x * (y - 1.0) * (x - z + 1.0) +
                   x * (y - 1.0) * (x + z + 1.0) + z * (2.0 * y + 1.0) - z);
      weight[offset+6] = (-0.5 * (x - z) * (0.25 * y - 0.25 * z) * (-x + z + 1.0) +
                   0.125 * (x + z) * (y - z) * (x + z - 1.0)) *
                  (y - z + 1.0);
      weight[offset+7] = -0.5 * y *
                  ((x - 1.0) * (x - z + 1.0) + (x + 1.0) * (x + z - 1.0)) *
                  (0.5 * y - 0.5 * z + 0.5);
      weight[offset+8] = ((0.25 * x - 0.25 * z) * (y - z) * (x - z + 1.0) +
                   (x + z) * (0.25 * y - 0.25 * z) * (x + z + 1.0)) *
                  (0.5 * y - 0.5 * z + 0.5);
      weight[offset+9] = 1.0 * z * (x * y - x * z - x - y - z + 1);
      weight[offset+10] = 1.0 * z * (-x * y + x * z + x - y - z + 1);
      weight[offset+11] = 1.0 * z * (-x + 1) * (y - z + 1.0);
      weight[offset+12] = 0.5 * z * (2 * x + 2.0) * (y - z + 1.0);
      weight[offset+13] = z * (2.0 * z - 1.0);
      break;
    case 2:
      weight[offset+0] = (x + z) *
                  (-0.125 * (y - z) * (-y + z + 1.0) +
                   0.125 * (y + z) * (y + z - 1.0)) *
                  (x + z - 1.0);
      weight[offset+1] = (x + z - 1) * (0.25 * y * (x + 1.0) * (-y + z + 1.0) -
                                 0.25 * y * (x + 1.0) * (y + z - 1.0) -
                                 0.25 * z * (2.0 * x + 1.0) + 0.25 * z);
      weight[offset+2] = -0.5 * z * (x - y) +
                  0.125 * (x + z) * (y - z) *
                      (4.0 * z - (x + z + 1.0) * (-y + z + 1.0)) +
                  0.125 * (x + z) * (y + z) * (x + z + 1.0) * (y + z - 1.0);
      weight[offset+3] =
          -x * (0.5 * (y - 1.0) * (x + z - 1.0) * (0.5 * y - 0.5 * z + 0.5) +
                0.25 * (y + 1.0) * (x + z - 1) * (y + z - 1.0));
      weight[offset+4] = (0.5 * (z * (x - y + z + 1.0) + (x + 1.0) * (y - 1.0)) *
                       (y - z + 1.0) -
                   0.5 * (z * (x + y + z + 1.0) - (x + 1.0) * (y + 1.0)) *
                       (y + z - 1.0)) *
                  (x + z - 1.0);
      weight[offset+5] = -0.5 * x2 * y2 - 0.5 * x2 * z +
                  0.5 * x2 - 0.5 * x * y2 * z -
                  0.5 * x * y2 - 0.5 * x * z2 + 0.5 * x -
                  1.0 * y2 * z;
      weight[offset+6] = 0.125 * (x + z) *
                  ((y - z) * (y - z + 1.0) + (y + z) * (y + z + 1.0)) *
                  (x + z - 1.0);
      weight[offset+7] =
          -0.5 * y * (x + 1.0) * (x + z - 1.0) * (0.5 * y - 0.5 * z + 0.5) -
          0.25 * (x + z - 1) *
              (y * (x + 1.0) * (y + z + 1.0) + z * (2.0 * x - 1.0) + z);
      weight[offset+8] =
          -0.5 * z * (x + y) +
          0.5 * (x + z) * (0.25 * y - 0.25 * z) * (x + z + 1.0) *
              (y - z + 1.0) -
          0.125 * (x + z) * (y + z) * (4.0 * z - (x + z + 1.0) * (y + z + 1.0));
      weight[offset+9] = 1.0 * z * (y - 1) * (x + z - 1.0);
      weight[offset+10] = 1.0 * z * (-x * y + x - y * z - y - z + 1);
      weight[offset+11] = -0.5 * z * (2 * y + 2.0) * (x + z - 1.0);
      weight[offset+12] = 1.0 * z * (x * y + x + y * z + y - z + 1);
      weight[offset+13] = z * (2.0 * z - 1.0);
      break;
    case 3:
      weight[offset+0] = (y + z) *
                  (-0.125 * (x - z) * (-x + z + 1.0) +
                   0.125 * (x + z) * (x + z - 1.0)) *
                  (y + z - 1.0);
      weight[offset+1] = -y *
                  (0.5 * (x - 1.0) * (0.5 * x - 0.5 * z + 0.5) +
                   0.25 * (x + 1.0) * (x + z - 1)) *
                  (y + z - 1.0);
      weight[offset+2] = (y + z) *
                  (0.5 * (0.25 * x - 0.25 * z) * (x - z + 1.0) +
                   0.125 * (x + z) * (x + z + 1.0)) *
                  (y + z - 1.0);
      weight[offset+3] =
          -0.25 * x * (y + 1.0) * (x + z - 1) * (y + z - 1.0) +
          0.25 * (y + z - 1) *
              (x * (y + 1.0) * (-x + z + 1.0) - z * (2.0 * y + 1.0) + z);
      weight[offset+4] = (0.5 * (z * (-x + y + z + 1.0) + (x - 1.0) * (y + 1.0)) *
                       (x - z + 1.0) -
                   0.5 * (z * (x + y + z + 1.0) - (x + 1.0) * (y + 1.0)) *
                       (x + z - 1.0)) *
                  (y + z - 1.0);
      weight[offset+5] =
          -0.5 * x * (y + 1.0) * (0.5 * x - 0.5 * z + 0.5) * (y + z - 1.0) -
          0.25 * (y + z - 1) *
              (x * (y + 1.0) * (x + z + 1.0) + z * (2.0 * y + 1.0) - z);
      weight[offset+6] = 0.5 * z * (x - y) +
                  0.5 * (0.25 * x - 0.25 * z) * (y + z) *
                      (4.0 * z - (-x + z + 1.0) * (y + z + 1.0)) +
                  0.125 * (x + z) * (y + z) * (x + z - 1.0) * (y + z + 1.0);
      weight[offset+7] = -0.5 * x2 * y2 - 0.5 * x2 * y * z -
                  0.5 * x2 * y - 1.0 * x2 * z -
                  0.5 * y2 * z + 0.5 * y2 - 0.5 * y * z2 +
                  0.5 * y;
      weight[offset+8] =
          -0.5 * z * (x + y) +
          0.125 * (x - z) * (y + z) * (x - z + 1.0) * (y + z + 1.0) -
          0.125 * (x + z) * (y + z) * (4.0 * z - (x + z + 1.0) * (y + z + 1.0));
      weight[offset+9] = 1.0 * z * (x - 1) * (y + z - 1.0);
      weight[offset+10] = -0.5 * z * (2 * x + 2.0) * (y + z - 1.0);
      weight[offset+11] = 1.0 * z * (-x * y - x * z - x + y - z + 1);
      weight[offset+12] = 1.0 * z * (x * y + x * z + x + y - z + 1);
      weight[offset+13] = z * (2.0 * z - 1.0);
      break;
    case 4:
      weight[offset+0] = 0.5 * z * (x + y) -
                  0.5 * (0.25 * x - 0.25 * z) * (y - z) *
                      (4.0 * z - (-x + z + 1.0) * (-y + z + 1.0)) -
                  0.125 * (x - z) * (y + z) * (-x + z + 1.0) * (y + z - 1.0);
      weight[offset+1] = (-0.25 * x + 0.25 * z - 0.25) *
                  (-y * (x - 1.0) * (-y + z + 1.0) +
                   y * (x - 1.0) * (y + z - 1.0) + z * (2.0 * x - 1.0) + z);
      weight[offset+2] =
          (0.25 * x - 0.25 * z) *
          (-0.5 * (y - z) * (-y + z + 1.0) + 0.5 * (y + z) * (y + z - 1.0)) *
          (x - z + 1.0);
      weight[offset+3] = -0.5 * x2 * y2 - 0.5 * x2 * z +
                  0.5 * x2 + 0.5 * x * y2 * z +
                  0.5 * x * y2 + 0.5 * x * z2 - 0.5 * x -
                  1.0 * y2 * z;
      weight[offset+4] =
          ((z * (-x + y + z + 1.0) + (x - 1.0) * (y + 1.0)) * (y + z - 1.0) +
           (z * (x + y - z - 1.0) + (x - 1.0) * (y - 1.0)) * (y - z + 1.0)) *
          (0.5 * x - 0.5 * z + 0.5);
      weight[offset+5] = -0.5 * x *
                  ((y - 1.0) * (x - z + 1.0) * (0.5 * y - 0.5 * z + 0.5) +
                   (y + 1.0) * (0.5 * x - 0.5 * z + 0.5) * (y + z - 1.0));
      weight[offset+6] = 0.5 * z * (x - y) +
                  0.5 * (0.25 * x - 0.25 * z) * (y + z) *
                      (4.0 * z - (-x + z + 1.0) * (y + z + 1.0)) -
                  0.5 * (x - z) * (0.25 * y - 0.25 * z) * (-x + z + 1.0) *
                      (y - z + 1.0);
      weight[offset+7] =
          -0.5 * y * (x - 1.0) * (x - z + 1.0) * (0.5 * y - 0.5 * z + 0.5) -
          0.5 * (0.5 * x - 0.5 * z + 0.5) *
              (y * (x - 1.0) * (y + z + 1.0) + z * (2.0 * x + 1.0) - z);
      weight[offset+8] = (0.5 * (0.25 * x - 0.25 * z) * (y - z) * (y - z + 1.0) +
                   0.125 * (x - z) * (y + z) * (y + z + 1.0)) *
                  (x - z + 1.0);
      weight[offset+9] = 1.0 * z * (x * y - x - y * z - y - z + 1);
      weight[offset+10] = 1.0 * z * (-y + 1) * (x - z + 1.0);
      weight[offset+11] = 1.0 * z * (-x * y - x + y * z + y - z + 1);
      weight[offset+12] = 0.5 * z * (2 * y + 2.0) * (x - z + 1.0);
      weight[offset+13] = z * (2.0 * z - 1.0);
      break;
    default:
      std::cout << x << " " << y << std::endl;
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 1.0 * weight[offset+0];
    weight[offset+1] = 1.0 * weight[offset+1];
    weight[offset+2] = 1.0 * weight[offset+2];
    weight[offset+3] = 1.0 * weight[offset+3];
    weight[offset+4] = 1.0 * weight[offset+4];
    weight[offset+5] = 1.0 * weight[offset+5];
    weight[offset+6] = 1.0 * weight[offset+6];
    weight[offset+7] = 1.0 * weight[offset+7];
    weight[offset+8] = 1.0 * weight[offset+8];
    weight[offset+9] = 1.0 * weight[offset+9];
    weight[offset+10] = 1.0 * weight[offset+10];
    weight[offset+11] = 1.0 * weight[offset+11];
    weight[offset+12] = 1.0 * weight[offset+12];
    weight[offset+13] = 1.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_x(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));

  if (this->my_deg_ == 0) {
    weight[offset+0] = 0.0;
  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0.25 * y - 0.25 * z - 0.25;
      weight[offset+1] = -0.25 * y + 0.25 * z + 0.25;
      weight[offset+2] = -0.25 * y + 0.25 * z - 0.25;
      weight[offset+3] = 0.25 * y - 0.25 * z + 0.25;
      weight[offset+4] = 0;
      break;
    case 2:
      weight[offset+0] = 0.25 * y - 0.25;
      weight[offset+1] = -0.25 * y + 0.25;
      weight[offset+2] = -0.25 * y - 0.25;
      weight[offset+3] = 0.25 * y + 0.25;
      weight[offset+4] = 0;
      break;
    case 3:
      weight[offset+0] = 0.25 * y + 0.25 * z - 0.25;
      weight[offset+1] = -0.25 * y - 0.25 * z + 0.25;
      weight[offset+2] = -0.25 * y - 0.25 * z - 0.25;
      weight[offset+3] = 0.25 * y + 0.25 * z + 0.25;
      weight[offset+4] = 0;
      break;
    case 4:
      weight[offset+0] = 0.25 * y - 0.25;
      weight[offset+1] = -0.25 * y + 0.25;
      weight[offset+2] = -0.25 * y - 0.25;
      weight[offset+3] = 0.25 * y + 0.25;
      weight[offset+4] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 2.0 * weight[offset+0];
    weight[offset+1] = 2.0 * weight[offset+1];
    weight[offset+2] = 2.0 * weight[offset+2];
    weight[offset+3] = 2.0 * weight[offset+3];
    weight[offset+4] = 2.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0.5 * x * y2 - 1.0 * x * y * z - 0.5 * x * y +
                  0.5 * x * z2 + 0.5 * x * z - 0.25 * y2 +
                  0.25 * y + 0.25 * z2 + 0.25 * z;
      weight[offset+1] = x * (-1.0 * y2 + 1.0 * y * z + 1.0 * y - 2.0 * z);
      weight[offset+2] = 0.5 * x * y2 - 1.0 * x * y * z - 0.5 * x * y +
                  0.5 * x * z2 + 0.5 * x * z + 0.25 * y2 -
                  0.25 * y - 0.25 * z2 - 0.25 * z;
      weight[offset+3] = -0.5 * (4 * x - 2.0) * (y - 1.0) * (0.5 * y - 0.5 * z + 0.5);
      weight[offset+4] = x * (2.0 * y2 - 2.0 * z2 + 4.0 * z - 2.0);
      weight[offset+5] = -0.5 * (4 * x + 2.0) * (y - 1.0) * (0.5 * y - 0.5 * z + 0.5);
      weight[offset+6] =
          (y - z + 1.0) *
          (0.5 * (x - z) * (0.25 * y - 0.25 * z) + 0.125 * (x + z) * (y - z) -
           0.5 * (0.25 * y - 0.25 * z) * (-x + z + 1.0) +
           0.125 * (y - z) * (x + z - 1.0));
      weight[offset+7] = 1.0 * x * y * (-y + z - 1);
      weight[offset+8] = (y - z + 1.0) * (0.5 * (0.25 * x - 0.25 * z) * (y - z) +
                                   0.5 * (x + z) * (0.25 * y - 0.25 * z) +
                                   0.5 * (0.25 * y - 0.25 * z) * (x + z + 1.0) +
                                   0.125 * (y - z) * (x - z + 1.0));
      weight[offset+9] = 1.0 * z * (y - z - 1);
      weight[offset+10] = 1.0 * z * (-y + z + 1.0);
      weight[offset+11] = 1.0 * z * (-y + z - 1);
      weight[offset+12] = 1.0 * z * (y - z + 1.0);
      weight[offset+13] = 0;
      break;
    case 2:
      weight[offset+0] = 0.5 * x * y2 - 0.5 * x * y + 0.5 * x * z2 +
                  0.5 * y2 * z - 0.25 * y2 - 0.5 * y * z +
                  0.25 * y + 0.5 * pow(z, 3) - 0.25 * z2;
      weight[offset+1] = -1.0 * x * y2 + 1.0 * x * y - 1.0 * x * z -
                  0.5 * y2 * z + 0.5 * y * z - 0.5 * z2 + 0.5 * z;
      weight[offset+2] = 0.5 * x * y2 - 0.5 * x * y + 0.5 * x * z2 +
                  0.5 * y2 * z + 0.25 * y2 - 0.25 * y +
                  0.5 * pow(z, 3) - 0.25 * z2 - 0.5 * z;
      weight[offset+3] = -1.0 * x * y2 - 1.0 * x * z + 1.0 * x -
                  0.5 * y2 * z + 0.5 * y2 - 0.5 * z2 +
                  1.0 * z - 0.5;
      weight[offset+4] = 2.0 * x * y2 - 2.0 * x * z2 + 4.0 * x * z -
                  2.0 * x - 2.0 * pow(z, 3) + 3.0 * z2 - 1.0 * z;
      weight[offset+5] = -(0.5 * (y - 1.0) * (0.5 * y - 0.5 * z + 0.5) +
                    0.25 * (y + 1.0) * (y + z - 1)) *
                  (2 * x + z + 1.0);
      weight[offset+6] = 0.5 * x * y2 + 0.5 * x * y + 0.5 * x * z2 +
                  0.5 * y2 * z - 0.25 * y2 + 0.5 * y * z -
                  0.25 * y + 0.5 * pow(z, 3) - 0.25 * z2;
      weight[offset+7] = -1.0 * x * y2 - 1.0 * x * y - 1.0 * x * z -
                  0.5 * y2 * z - 0.5 * y * z - 0.5 * z2 + 0.5 * z;
      weight[offset+8] = 0.5 * x * y2 + 0.5 * x * y + 0.5 * x * z2 +
                  0.5 * y2 * z + 0.25 * y2 + 0.25 * y +
                  0.5 * pow(z, 3) - 0.25 * z2 - 0.5 * z;
      weight[offset+9] = 1.0 * z * (y - 1);
      weight[offset+10] = 1.0 * z * (-y + 1);
      weight[offset+11] = -1.0 * z * (y + 1);
      weight[offset+12] = 1.0 * z * (y + 1);
      weight[offset+13] = 0;
      break;
    case 3:
      weight[offset+0] = (0.5 * x - 0.25) * (y + z) * (y + z - 1.0);
      weight[offset+1] = 1.0 * x * y * (-y - z + 1);
      weight[offset+2] = (0.5 * x + 0.25) * (y + z) * (y + z - 1.0);
      weight[offset+3] = -0.25 * (y + 1) *
                  (x * (y + z - 1.0) - (-2 * x + z + 1.0) * (y + z - 1) +
                   (x + z - 1) * (y + z - 1.0));
      weight[offset+4] = x * (2.0 * y2 - 2.0 * z2 + 4.0 * z - 2.0);
      weight[offset+5] = -1.0 * x * y2 - 1.0 * x * y * z - 1.0 * x * z +
                  1.0 * x - 0.5 * y2 - 0.5 * y * z - 0.5 * z + 0.5;
      weight[offset+6] = 0.5 * x * y2 + 1.0 * x * y * z + 0.5 * x * y +
                  0.5 * x * z2 + 0.5 * x * z - 0.25 * y2 -
                  0.25 * y + 0.25 * z2 + 0.25 * z;
      weight[offset+7] = -x * (1.0 * y2 + 1.0 * y * z + 1.0 * y + 2.0 * z);
      weight[offset+8] = 0.5 * x * y2 + 1.0 * x * y * z + 0.5 * x * y +
                  0.5 * x * z2 + 0.5 * x * z + 0.25 * y2 +
                  0.25 * y - 0.25 * z2 - 0.25 * z;
      weight[offset+9] = 1.0 * z * (y + z - 1.0);
      weight[offset+10] = 1.0 * z * (-y - z + 1);
      weight[offset+11] = -1.0 * z * (y + z + 1.0);
      weight[offset+12] = 1.0 * z * (y + z + 1.0);
      weight[offset+13] = 0;
      break;
    case 4:
      weight[offset+0] = 0.5 * x * y2 - 0.5 * x * y + 0.5 * x * z2 -
                  0.5 * y2 * z - 0.25 * y2 + 0.25 * y -
                  0.5 * pow(z, 3) + 0.25 * z2 + 0.5 * z;
      weight[offset+1] = -1.0 * x * y2 + 1.0 * x * y - 1.0 * x * z +
                  0.5 * y2 * z - 0.5 * y * z + 0.5 * z2 - 0.5 * z;
      weight[offset+2] = 0.5 * x * y2 - 0.5 * x * y + 0.5 * x * z2 -
                  0.5 * y2 * z + 0.25 * y2 + 0.5 * y * z -
                  0.25 * y - 0.5 * pow(z, 3) + 0.25 * z2;
      weight[offset+3] = (-0.5 * (y - 1.0) * (0.5 * y - 0.5 * z + 0.5) -
                   0.25 * (y + 1.0) * (y + z - 1)) *
                  (2 * x - z - 1.0);
      weight[offset+4] = 2.0 * x * y2 - 2.0 * x * z2 + 4.0 * x * z -
                  2.0 * x + 2.0 * pow(z, 3) - 3.0 * z2 + 1.0 * z;
      weight[offset+5] = -1.0 * x * y2 - 1.0 * x * z + 1.0 * x +
                  0.5 * y2 * z - 0.5 * y2 + 0.5 * z2 -
                  1.0 * z + 0.5;
      weight[offset+6] = 0.5 * x * y2 + 0.5 * x * y + 0.5 * x * z2 -
                  0.5 * y2 * z - 0.25 * y2 - 0.25 * y -
                  0.5 * pow(z, 3) + 0.25 * z2 + 0.5 * z;
      weight[offset+7] = -1.0 * x * y2 - 1.0 * x * y - 1.0 * x * z +
                  0.5 * y2 * z + 0.5 * y * z + 0.5 * z2 - 0.5 * z;
      weight[offset+8] = 0.5 * x * y2 + 0.5 * x * y + 0.5 * x * z2 -
                  0.5 * y2 * z + 0.25 * y2 - 0.5 * y * z +
                  0.25 * y - 0.5 * pow(z, 3) + 0.25 * z2;
      weight[offset+9] = 1.0 * z * (y - 1);
      weight[offset+10] = 1.0 * z * (-y + 1);
      weight[offset+11] = -1.0 * z * (y + 1);
      weight[offset+12] = 1.0 * z * (y + 1);
      weight[offset+13] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    if (xx == 0 && yy == 0) {
      weight[offset+2] = 0.5 * x * y2 - 0.5 * x * y + 0.5 * x * z2 -
                  0.5 * y2 * z + 0.25 * y2 + 0.5 * y * z -
                  0.25 * y - 0.5 * pow(z, 3) + 0.25 * z2;
      weight[offset+3] = -0.25 * (y + 1) *
                  (x * (y + z - 1.0) - (-2 * x + z + 1.0) * (y + z - 1) +
                   (x + z - 1) * (y + z - 1.0));
      weight[offset+5] = -1.0 * x * y2 - 1.0 * x * y * z - 1.0 * x * z +
                  1.0 * x - 0.5 * y2 - 0.5 * y * z - 0.5 * z + 0.5;
      weight[offset+8] = 0.5 * x * y2 + 0.5 * x * y + 0.5 * x * z2 -
                  0.5 * y2 * z + 0.25 * y2 - 0.5 * y * z +
                  0.25 * y - 0.5 * pow(z, 3) + 0.25 * z2;
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 2.0 * weight[offset+0];
    weight[offset+1] = 2.0 * weight[offset+1];
    weight[offset+2] = 2.0 * weight[offset+2];
    weight[offset+3] = 2.0 * weight[offset+3];
    weight[offset+4] = 2.0 * weight[offset+4];
    weight[offset+5] = 2.0 * weight[offset+5];
    weight[offset+6] = 2.0 * weight[offset+6];
    weight[offset+7] = 2.0 * weight[offset+7];
    weight[offset+8] = 2.0 * weight[offset+8];
    weight[offset+9] = 2.0 * weight[offset+9];
    weight[offset+10] = 2.0 * weight[offset+10];
    weight[offset+11] = 2.0 * weight[offset+11];
    weight[offset+12] = 2.0 * weight[offset+12];
    weight[offset+13] = 2.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_y(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));

  if (this->my_deg_ == 0) {
    weight[offset+0] = 0.0;
  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx >= yy && xx <= -yy) {
      indicator = 1;
    } else if (xx > yy && xx > -yy) {
      indicator = 2;
    } else if (xx <= yy && xx >= -yy) {
      indicator = 3;
    } else if (xx < yy && xx < -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0.25 * x - 0.25;
      weight[offset+1] = -0.25 * x - 0.25;
      weight[offset+2] = -0.25 * x + 0.25;
      weight[offset+3] = 0.25 * x + 0.25;
      weight[offset+4] = 0;
      break;
    case 2:
      weight[offset+0] = 0.25 * x + 0.25 * z - 0.25;
      weight[offset+1] = -0.25 * x - 0.25 * z - 0.25;
      weight[offset+2] = -0.25 * x - 0.25 * z + 0.25;
      weight[offset+3] = 0.25 * x + 0.25 * z + 0.25;
      weight[offset+4] = 0;
      break;
    case 3:
      weight[offset+0] = 0.25 * x - 0.25;
      weight[offset+1] = -0.25 * x - 0.25;
      weight[offset+2] = -0.25 * x + 0.25;
      weight[offset+3] = 0.25 * x + 0.25;
      weight[offset+4] = 0;
      break;
    case 4:
      weight[offset+0] = 0.25 * x - 0.25 * z - 0.25;
      weight[offset+1] = -0.25 * x + 0.25 * z - 0.25;
      weight[offset+2] = -0.25 * x + 0.25 * z + 0.25;
      weight[offset+3] = 0.25 * x - 0.25 * z + 0.25;
      weight[offset+4] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 2.0 * weight[offset+0];
    weight[offset+1] = 2.0 * weight[offset+1];
    weight[offset+2] = 2.0 * weight[offset+2];
    weight[offset+3] = 2.0 * weight[offset+3];
    weight[offset+4] = 2.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx >= yy && xx <= -yy) {
      indicator = 1;
    } else if (xx > yy && xx > -yy) {
      indicator = 2;
    } else if (xx <= yy && xx >= -yy) {
      indicator = 3;
    } else if (xx < yy && xx < -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0.5 * x2 * y - 0.5 * x2 * z - 0.25 * x2 -
                  0.5 * x * y + 0.25 * x + 0.5 * y * z2 -
                  0.5 * pow(z, 3) + 0.25 * z2 + 0.5 * z;
      weight[offset+1] = (-0.5 * (x - 1.0) * (0.5 * x - 0.5 * z + 0.5) -
                   0.25 * (x + 1.0) * (x + z - 1)) *
                  (2 * y - z - 1.0);
      weight[offset+2] = 0.5 * x2 * y - 0.5 * x2 * z - 0.25 * x2 +
                  0.5 * x * y - 0.25 * x + 0.5 * y * z2 -
                  0.5 * pow(z, 3) + 0.25 * z2 + 0.5 * z;
      weight[offset+3] = -1.0 * x2 * y + 0.5 * x2 * z + 1.0 * x * y -
                  0.5 * x * z - 1.0 * y * z + 0.5 * z2 - 0.5 * z;
      weight[offset+4] = 2.0 * x2 * y - 2.0 * y * z2 + 4.0 * y * z -
                  2.0 * y + 2.0 * pow(z, 3) - 3.0 * z2 + 1.0 * z;
      weight[offset+5] = -1.0 * x2 * y + 0.5 * x2 * z - 1.0 * x * y +
                  0.5 * x * z - 1.0 * y * z + 0.5 * z2 - 0.5 * z;
      weight[offset+6] = 0.5 * x2 * y - 0.5 * x2 * z + 0.25 * x2 -
                  0.5 * x * y + 0.5 * x * z - 0.25 * x + 0.5 * y * z2 -
                  0.5 * pow(z, 3) + 0.25 * z2;
      weight[offset+7] = -1.0 * x2 * y + 0.5 * x2 * z - 0.5 * x2 -
                  1.0 * y * z + 1.0 * y + 0.5 * z2 - 1.0 * z + 0.5;
      weight[offset+8] = 0.5 * x2 * y - 0.5 * x2 * z + 0.25 * x2 +
                  0.5 * x * y - 0.5 * x * z + 0.25 * x + 0.5 * y * z2 -
                  0.5 * pow(z, 3) + 0.25 * z2;
      weight[offset+9] = 1.0 * z * (x - 1);
      weight[offset+10] = -1.0 * z * (x + 1);
      weight[offset+11] = 1.0 * z * (-x + 1);
      weight[offset+12] = 1.0 * z * (x + 1);
      weight[offset+13] = 0;
      break;
    case 2:
      weight[offset+0] = (x + z) * (0.5 * y - 0.25) * (x + z - 1.0);
      weight[offset+1] = -0.25 * (x + 1.0) * (4 * y - 2.0) * (x + z - 1);
      weight[offset+2] = 0.5 * x2 * y - 0.25 * x2 + 1.0 * x * y * z +
                  0.5 * x * y - 0.25 * x + 0.5 * y * z2 + 0.5 * y * z +
                  0.25 * z2 + 0.25 * z;
      weight[offset+3] = 1.0 * x * y * (-x - z + 1);
      weight[offset+4] = y * (2.0 * x2 - 2.0 * z2 + 4.0 * z - 2.0);
      weight[offset+5] = -y * (1.0 * x2 + 1.0 * x * z + 1.0 * x + 2.0 * z);
      weight[offset+6] = 0.125 * (x + z) * (4 * y + 2.0) * (x + z - 1.0);
      weight[offset+7] = -1.0 * x2 * y - 0.5 * x2 - 1.0 * x * y * z -
                  0.5 * x * z - 1.0 * y * z + 1.0 * y - 0.5 * z + 0.5;
      weight[offset+8] = 0.5 * x2 * y + 0.25 * x2 + 1.0 * x * y * z +
                  0.5 * x * y + 0.25 * x + 0.5 * y * z2 + 0.5 * y * z -
                  0.25 * z2 - 0.25 * z;
      weight[offset+9] = 1.0 * z * (x + z - 1.0);
      weight[offset+10] = -1.0 * z * (x + z + 1.0);
      weight[offset+11] = 1.0 * z * (-x - z + 1);
      weight[offset+12] = 1.0 * z * (x + z + 1.0);
      weight[offset+13] = 0;
      break;
    case 3:
      weight[offset+0] = 0.5 * x2 * y + 0.5 * x2 * z - 0.25 * x2 -
                  0.5 * x * y - 0.5 * x * z + 0.25 * x + 0.5 * y * z2 +
                  0.5 * pow(z, 3) - 0.25 * z2;
      weight[offset+1] = -1.0 * x2 * y - 0.5 * x2 * z + 0.5 * x2 -
                  1.0 * y * z + 1.0 * y - 0.5 * z2 + 1.0 * z - 0.5;
      weight[offset+2] = 0.5 * x2 * y + 0.5 * x2 * z - 0.25 * x2 +
                  0.5 * x * y + 0.5 * x * z - 0.25 * x + 0.5 * y * z2 +
                  0.5 * pow(z, 3) - 0.25 * z2;
      weight[offset+3] = -1.0 * x2 * y - 0.5 * x2 * z + 1.0 * x * y +
                  0.5 * x * z - 1.0 * y * z - 0.5 * z2 + 0.5 * z;
      weight[offset+4] = 2.0 * x2 * y - 2.0 * y * z2 + 4.0 * y * z -
                  2.0 * y - 2.0 * pow(z, 3) + 3.0 * z2 - 1.0 * z;
      weight[offset+5] = -1.0 * x2 * y - 0.5 * x2 * z - 1.0 * x * y -
                  0.5 * x * z - 1.0 * y * z - 0.5 * z2 + 0.5 * z;
      weight[offset+6] = 0.5 * x2 * y + 0.5 * x2 * z + 0.25 * x2 -
                  0.5 * x * y - 0.25 * x + 0.5 * y * z2 +
                  0.5 * pow(z, 3) - 0.25 * z2 - 0.5 * z;
      weight[offset+7] = -(0.5 * (x - 1.0) * (0.5 * x - 0.5 * z + 0.5) +
                    0.25 * (x + 1.0) * (x + z - 1)) *
                  (2 * y + z + 1.0);
      weight[offset+8] = 0.5 * x2 * y + 0.5 * x2 * z + 0.25 * x2 +
                  0.5 * x * y + 0.25 * x + 0.5 * y * z2 +
                  0.5 * pow(z, 3) - 0.25 * z2 - 0.5 * z;
      weight[offset+9] = 1.0 * z * (x - 1);
      weight[offset+10] = -1.0 * z * (x + 1);
      weight[offset+11] = 1.0 * z * (-x + 1);
      weight[offset+12] = 1.0 * z * (x + 1);
      weight[offset+13] = 0;
      break;
    case 4:
      weight[offset+0] = 0.5 * x2 * y - 0.25 * x2 - 1.0 * x * y * z -
                  0.5 * x * y + 0.25 * x + 0.5 * y * z2 + 0.5 * y * z +
                  0.25 * z2 + 0.25 * z;
      weight[offset+1] = -0.5 * (x - 1.0) * (4 * y - 2.0) * (0.5 * x - 0.5 * z + 0.5);
      weight[offset+2] = (0.25 * x - 0.25 * z) * (2.0 * y - 1.0) * (x - z + 1.0);
      weight[offset+3] = y * (-1.0 * x2 + 1.0 * x * z + 1.0 * x - 2.0 * z);
      weight[offset+4] = y * (2.0 * x2 - 2.0 * z2 + 4.0 * z - 2.0);
      weight[offset+5] = 1.0 * x * y * (-x + z - 1);
      weight[offset+6] = 0.5 * x2 * y + 0.25 * x2 - 1.0 * x * y * z -
                  0.5 * x * y - 0.25 * x + 0.5 * y * z2 + 0.5 * y * z -
                  0.25 * z2 - 0.25 * z;
      weight[offset+7] = -1.0 * x2 * y - 0.5 * x2 + 1.0 * x * y * z +
                  0.5 * x * z - 1.0 * y * z + 1.0 * y - 0.5 * z + 0.5;
      weight[offset+8] = (x - z + 1.0) *
                  (0.5 * (0.25 * x - 0.25 * z) * (y - z) +
                   0.5 * (0.25 * x - 0.25 * z) * (y - z + 1.0) +
                   0.125 * (x - z) * (y + z) + 0.125 * (x - z) * (y + z + 1.0));
      weight[offset+9] = 1.0 * z * (x - z - 1);
      weight[offset+10] = 1.0 * z * (-x + z - 1);
      weight[offset+11] = 1.0 * z * (-x + z + 1.0);
      weight[offset+12] = 1.0 * z * (x - z + 1.0);
      weight[offset+13] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    if (xx == 0 && yy == 0) {
      weight[offset+0] = 0.5 * x2 * y + 0.5 * x2 * z - 0.25 * x2 -
                  0.5 * x * y - 0.5 * x * z + 0.25 * x + 0.5 * y * z2 +
                  0.5 * pow(z, 3) - 0.25 * z2;
      weight[offset+1] = -0.5 * (x - 1.0) * (4 * y - 2.0) * (0.5 * x - 0.5 * z + 0.5);
      weight[offset+7] = -1.0 * x2 * y - 0.5 * x2 + 1.0 * x * y * z +
                  0.5 * x * z - 1.0 * y * z + 1.0 * y - 0.5 * z + 0.5;
      weight[offset+2] = 0.5 * x2 * y + 0.5 * x2 * z - 0.25 * x2 +
                  0.5 * x * y + 0.5 * x * z - 0.25 * x + 0.5 * y * z2 +
                  0.5 * pow(z, 3) - 0.25 * z2;
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 2.0 * weight[offset+0];
    weight[offset+1] = 2.0 * weight[offset+1];
    weight[offset+2] = 2.0 * weight[offset+2];
    weight[offset+3] = 2.0 * weight[offset+3];
    weight[offset+4] = 2.0 * weight[offset+4];
    weight[offset+5] = 2.0 * weight[offset+5];
    weight[offset+6] = 2.0 * weight[offset+6];
    weight[offset+7] = 2.0 * weight[offset+7];
    weight[offset+8] = 2.0 * weight[offset+8];
    weight[offset+9] = 2.0 * weight[offset+9];
    weight[offset+10] = 2.0 * weight[offset+10];
    weight[offset+11] = 2.0 * weight[offset+11];
    weight[offset+12] = 2.0 * weight[offset+12];
    weight[offset+13] = 2.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_z(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));
  
  if (this->my_deg_ == 0) {
    weight[offset+0] = 0.0;
  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = -0.25 * x - 0.25;
      weight[offset+1] = 0.25 * x - 0.25;
      weight[offset+2] = 0.25 * x - 0.25;
      weight[offset+3] = -0.25 * x - 0.25;
      weight[offset+4] = 1;
      break;
    case 2:
      weight[offset+0] = 0.25 * y - 0.25;
      weight[offset+1] = -0.25 * y - 0.25;
      weight[offset+2] = -0.25 * y - 0.25;
      weight[offset+3] = 0.25 * y - 0.25;
      weight[offset+4] = 1;
      break;
    case 3:
      weight[offset+0] = 0.25 * x - 0.25;
      weight[offset+1] = -0.25 * x - 0.25;
      weight[offset+2] = -0.25 * x - 0.25;
      weight[offset+3] = 0.25 * x - 0.25;
      weight[offset+4] = 1;
      break;
    case 4:
      weight[offset+0] = -0.25 * y - 0.25;
      weight[offset+1] = 0.25 * y - 0.25;
      weight[offset+2] = 0.25 * y - 0.25;
      weight[offset+3] = -0.25 * y - 0.25;
      weight[offset+4] = 1;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 1.0 * weight[offset+0];
    weight[offset+1] = 1.0 * weight[offset+1];
    weight[offset+2] = 1.0 * weight[offset+2];
    weight[offset+3] = 1.0 * weight[offset+3];
    weight[offset+4] = 1.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = -0.5 * x2 * y + 0.5 * x2 * z +
                  0.25 * x2 + 0.5 * x * z + 0.25 * x +
                  0.5 * y2 * z - 1.5 * y * z2 + 0.5 * y * z +
                  0.5 * y + 1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+1] =
          0.5 * x2 * y - 1.0 * x2 - 0.5 * y2 + 1.0 * y * z;
      weight[offset+2] = -0.5 * x2 * y + 0.5 * x2 * z +
                  0.25 * x2 - 0.5 * x * z - 0.25 * x +
                  0.5 * y2 * z - 1.5 * y * z2 + 0.5 * y * z +
                  0.5 * y + 1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+3] = 0.5 * x2 * y - 0.5 * x2 - 0.5 * x * y +
                  0.5 * x - 0.5 * y2 + 1.0 * y * z - 0.5 * y;
      weight[offset+4] = -2.0 * x2 * z + 2.0 * x2 - 2.0 * y2 * z +
                  2.0 * y2 + 6.0 * y * z2 - 6.0 * y * z +
                  1.0 * y - 4.0 * pow(z, 3) + 3.0 * z2 + 4.0 * z - 3.0;
      weight[offset+5] = 0.5 * x2 * y - 0.5 * x2 + 0.5 * x * y -
                  0.5 * x - 0.5 * y2 + 1.0 * y * z - 0.5 * y;
      weight[offset+6] = -0.5 * x2 * y + 0.5 * x2 * z -
                  0.25 * x2 + 0.5 * x * y - 0.5 * x * z + 0.25 * x +
                  0.5 * y2 * z - 1.5 * y * z2 + 0.5 * y * z +
                  1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+7] = y * (0.5 * x2 - 0.5 * y + 1.0 * z - 1.0);
      weight[offset+8] = -0.5 * x2 * y + 0.5 * x2 * z -
                  0.25 * x2 - 0.5 * x * y + 0.5 * x * z - 0.25 * x +
                  0.5 * y2 * z - 1.5 * y * z2 + 0.5 * y * z +
                  1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+9] = 1.0 * x * y - 2.0 * x * z - 1.0 * x - 1.0 * y - 2.0 * z + 1.0;
      weight[offset+10] =
          -1.0 * x * y + 2.0 * x * z + 1.0 * x - 1.0 * y - 2.0 * z + 1.0;
      weight[offset+11] =
          -1.0 * x * y + 2.0 * x * z - 1.0 * x + 1.0 * y - 2.0 * z + 1.0;
      weight[offset+12] =
          1.0 * x * y - 2.0 * x * z + 1.0 * x + 1.0 * y - 2.0 * z + 1.0;
      weight[offset+13] = 4.0 * z - 1.0;
      break;
    case 2:
      weight[offset+0] = 0.5 * x2 * z + 0.5 * x * y2 - 0.5 * x * y +
                  1.5 * x * z2 - 0.5 * x * z + 0.5 * y2 * z -
                  0.25 * y2 - 0.5 * y * z + 0.25 * y + 1.0 * pow(z, 3) -
                  0.75 * z2;
      weight[offset+1] = -0.5 * x2 - 0.5 * x * y2 + 0.5 * x * y -
                  1.0 * x * z + 0.5 * x - 0.5 * y2 + 0.5 * y;
      weight[offset+2] = 0.5 * x2 * z + 0.5 * x * y2 +
                  1.5 * x * z2 - 0.5 * x * z - 0.5 * x +
                  0.5 * y2 * z + 0.25 * y2 + 0.5 * y * z +
                  0.25 * y + 1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+3] = x * (-0.5 * x - 0.5 * y2 - 1.0 * z + 1.0);
      weight[offset+4] = -2.0 * x2 * z + 2.0 * x2 - 6.0 * x * z2 +
                  6.0 * x * z - 1.0 * x - 2.0 * y2 * z +
                  2.0 * y2 - 4.0 * pow(z, 3) + 3.0 * z2 +
                  4.0 * z - 3.0;
      weight[offset+5] = -0.5 * x2 - 0.5 * x * y2 - 1.0 * x * z -
                  1.0 * y2;
      weight[offset+6] = 0.5 * x2 * z + 0.5 * x * y2 + 0.5 * x * y +
                  1.5 * x * z2 - 0.5 * x * z + 0.5 * y2 * z -
                  0.25 * y2 + 0.5 * y * z - 0.25 * y + 1.0 * pow(z, 3) -
                  0.75 * z2;
      weight[offset+7] = -0.5 * x2 - 0.5 * x * y2 - 0.5 * x * y -
                  1.0 * x * z + 0.5 * x - 0.5 * y2 - 0.5 * y;
      weight[offset+8] = 0.5 * x2 * z + 0.5 * x * y2 +
                  1.5 * x * z2 - 0.5 * x * z - 0.5 * x +
                  0.5 * y2 * z + 0.25 * y2 - 0.5 * y * z -
                  0.25 * y + 1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+9] = 1.0 * x * y - 1.0 * x + 2.0 * y * z - 1.0 * y - 2.0 * z + 1.0;
      weight[offset+10] =
          -1.0 * x * y + 1.0 * x - 2.0 * y * z - 1.0 * y - 2.0 * z + 1.0;
      weight[offset+11] =
          -1.0 * x * y - 1.0 * x - 2.0 * y * z + 1.0 * y - 2.0 * z + 1.0;
      weight[offset+12] =
          1.0 * x * y + 1.0 * x + 2.0 * y * z + 1.0 * y - 2.0 * z + 1.0;
      weight[offset+13] = 4.0 * z - 1.0;
      break;
    case 3:
      weight[offset+0] = 0.5 * x2 * y + 0.5 * x2 * z - 0.25 * x2 -
                  0.5 * x * y - 0.5 * x * z + 0.25 * x + 0.5 * y2 * z +
                  1.5 * y * z2 - 0.5 * y * z + 1.0 * pow(z, 3) -
                  0.75 * z2;
      weight[offset+1] = y * (-0.5 * x2 - 0.5 * y - 1.0 * z + 1.0);
      weight[offset+2] = 0.5 * x2 * y + 0.5 * x2 * z - 0.25 * x2 +
                  0.5 * x * y + 0.5 * x * z - 0.25 * x + 0.5 * y2 * z +
                  1.5 * y * z2 - 0.5 * y * z + 1.0 * pow(z, 3) -
                  0.75 * z2;
      weight[offset+3] = -0.5 * x2 * y - 0.5 * x2 + 0.5 * x * y +
                  0.5 * x - 0.5 * y2 - 1.0 * y * z + 0.5 * y;
      weight[offset+4] = -2.0 * x2 * z + 2.0 * x2 - 2.0 * y2 * z +
                  2.0 * y2 - 6.0 * y * z2 + 6.0 * y * z -
                  1.0 * y - 4.0 * pow(z, 3) + 3.0 * z2 + 4.0 * z - 3.0;
      weight[offset+5] = -0.5 * x2 * y - 0.5 * x2 - 0.5 * x * y -
                  0.5 * x - 0.5 * y2 - 1.0 * y * z + 0.5 * y;
      weight[offset+6] = 0.5 * x2 * y + 0.5 * x2 * z + 0.25 * x2 +
                  0.5 * x * z + 0.25 * x + 0.5 * y2 * z +
                  1.5 * y * z2 - 0.5 * y * z - 0.5 * y +
                  1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+7] = -0.5 * x2 * y - 1.0 * x2 - 0.5 * y2 -
                  1.0 * y * z;
      weight[offset+8] = 0.5 * x2 * y + 0.5 * x2 * z + 0.25 * x2 -
                  0.5 * x * z - 0.25 * x + 0.5 * y2 * z +
                  1.5 * y * z2 - 0.5 * y * z - 0.5 * y +
                  1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+9] = 1.0 * x * y + 2.0 * x * z - 1.0 * x - 1.0 * y - 2.0 * z + 1.0;
      weight[offset+10] =
          -1.0 * x * y - 2.0 * x * z + 1.0 * x - 1.0 * y - 2.0 * z + 1.0;
      weight[offset+11] =
          -1.0 * x * y - 2.0 * x * z - 1.0 * x + 1.0 * y - 2.0 * z + 1.0;
      weight[offset+12] =
          1.0 * x * y + 2.0 * x * z + 1.0 * x + 1.0 * y - 2.0 * z + 1.0;
      weight[offset+13] = 4.0 * z - 1.0;
      break;
    case 4:
      weight[offset+0] = 0.5 * x2 * z - 0.5 * x * y2 -
                  1.5 * x * z2 + 0.5 * x * z + 0.5 * x +
                  0.5 * y2 * z + 0.25 * y2 + 0.5 * y * z +
                  0.25 * y + 1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+1] = -0.5 * x2 + 0.5 * x * y2 - 0.5 * x * y +
                  1.0 * x * z - 0.5 * x - 0.5 * y2 + 0.5 * y;
      weight[offset+2] = 0.5 * x2 * z - 0.5 * x * y2 + 0.5 * x * y -
                  1.5 * x * z2 + 0.5 * x * z + 0.5 * y2 * z -
                  0.25 * y2 - 0.5 * y * z + 0.25 * y + 1.0 * pow(z, 3) -
                  0.75 * z2;
      weight[offset+3] = -0.5 * x2 + 0.5 * x * y2 + 1.0 * x * z -
                  1.0 * y2;
      weight[offset+4] = -2.0 * x2 * z + 2.0 * x2 + 6.0 * x * z2 -
                  6.0 * x * z + 1.0 * x - 2.0 * y2 * z +
                  2.0 * y2 - 4.0 * pow(z, 3) + 3.0 * z2 +
                  4.0 * z - 3.0;
      weight[offset+5] = x * (-0.5 * x + 0.5 * y2 + 1.0 * z - 1.0);
      weight[offset+6] = 0.5 * x2 * z - 0.5 * x * y2 -
                  1.5 * x * z2 + 0.5 * x * z + 0.5 * x +
                  0.5 * y2 * z + 0.25 * y2 - 0.5 * y * z -
                  0.25 * y + 1.0 * pow(z, 3) - 0.75 * z2;
      weight[offset+7] = -0.5 * x2 + 0.5 * x * y2 + 0.5 * x * y +
                  1.0 * x * z - 0.5 * x - 0.5 * y2 - 0.5 * y;
      weight[offset+8] = 0.5 * x2 * z - 0.5 * x * y2 - 0.5 * x * y -
                  1.5 * x * z2 + 0.5 * x * z + 0.5 * y2 * z -
                  0.25 * y2 + 0.5 * y * z - 0.25 * y + 1.0 * pow(z, 3) -
                  0.75 * z2;
      weight[offset+9] = 1.0 * x * y - 1.0 * x - 2.0 * y * z - 1.0 * y - 2.0 * z + 1.0;
      weight[offset+10] =
          -1.0 * x * y + 1.0 * x + 2.0 * y * z - 1.0 * y - 2.0 * z + 1.0;
      weight[offset+11] =
          -1.0 * x * y - 1.0 * x + 2.0 * y * z + 1.0 * y - 2.0 * z + 1.0;
      weight[offset+12] =
          1.0 * x * y + 1.0 * x - 2.0 * y * z + 1.0 * y - 2.0 * z + 1.0;
      weight[offset+13] = 4.0 * z - 1.0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 1.0 * weight[offset+0];
    weight[offset+1] = 1.0 * weight[offset+1];
    weight[offset+2] = 1.0 * weight[offset+2];
    weight[offset+3] = 1.0 * weight[offset+3];
    weight[offset+4] = 1.0 * weight[offset+4];
    weight[offset+5] = 1.0 * weight[offset+5];
    weight[offset+6] = 1.0 * weight[offset+6];
    weight[offset+7] = 1.0 * weight[offset+7];
    weight[offset+8] = 1.0 * weight[offset+8];
    weight[offset+9] = 1.0 * weight[offset+9];
    weight[offset+10] = 1.0 * weight[offset+10];
    weight[offset+11] = 1.0 * weight[offset+11];
    weight[offset+12] = 1.0 * weight[offset+12];
    weight[offset+13] = 1.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_xx(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));

  if (this->my_deg_ == 0) {
    weight[offset+0] = 0.0;
  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 2:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 3:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 4:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 4.0 * weight[offset+0];
    weight[offset+1] = 4.0 * weight[offset+1];
    weight[offset+2] = 4.0 * weight[offset+2];
    weight[offset+3] = 4.0 * weight[offset+3];
    weight[offset+4] = 4.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = -0.5 * (y - z) * (-y + z + 1.0);
      weight[offset+1] = 1.0 * y * (-y + z + 1.0) - 2.0 * z;
      weight[offset+2] = -0.5 * (y - z) * (-y + z + 1.0);
      weight[offset+3] = -1.0 * y2 + 1.0 * y * z - 1.0 * z + 1.0;
      weight[offset+4] = 2.0 * (y - z + 1.0) * (y + z - 1.0);
      weight[offset+5] = -1.0 * y2 + 1.0 * y * z - 1.0 * z + 1.0;
      weight[offset+6] = (0.5 * y - 0.5 * z) * (y - z + 1.0);
      weight[offset+7] = 1.0 * y * (-y + z - 1);
      weight[offset+8] = (0.5 * y - 0.5 * z) * (y - z + 1.0);
      weight[offset+9] = 0;
      weight[offset+10] = 0;
      weight[offset+11] = 0;
      weight[offset+12] = 0;
      weight[offset+13] = 0;
      break;
    case 2:
      weight[offset+0] = 0.5 * y2 - 0.5 * y + 0.5 * z2;
      weight[offset+1] = -1.0 * y2 + 1.0 * y - 1.0 * z;
      weight[offset+2] = 0.5 * y2 - 0.5 * y + 0.5 * z2;
      weight[offset+3] = -1.0 * y2 - 1.0 * z + 1.0;
      weight[offset+4] = 2.0 * (y - z + 1.0) * (y + z - 1.0);
      weight[offset+5] = -1.0 * y2 - 1.0 * z + 1.0;
      weight[offset+6] = 0.5 * y2 + 0.5 * y + 0.5 * z2;
      weight[offset+7] = -1.0 * y2 - 1.0 * y - 1.0 * z;
      weight[offset+8] = 0.5 * y2 + 0.5 * y + 0.5 * z2;
      weight[offset+9] = 0;
      weight[offset+10] = 0;
      weight[offset+11] = 0;
      weight[offset+12] = 0;
      weight[offset+13] = 0;
      break;
    case 3:
      weight[offset+0] = 0.5 * (y + z) * (y + z - 1.0);
      weight[offset+1] = 1.0 * y * (-y - z + 1);
      weight[offset+2] = 0.5 * (y + z) * (y + z - 1.0);
      weight[offset+3] = -1.0 * y2 - 1.0 * y * z - 1.0 * z + 1.0;
      weight[offset+4] = 2.0 * (y - z + 1.0) * (y + z - 1.0);
      weight[offset+5] = -1.0 * y2 - 1.0 * y * z - 1.0 * z + 1.0;
      weight[offset+6] = 0.5 * (y + z) * (y + z + 1.0);
      weight[offset+7] = -1.0 * y * (y + z + 1.0) - 2.0 * z;
      weight[offset+8] = 0.5 * (y + z) * (y + z + 1.0);
      weight[offset+9] = 0;
      weight[offset+10] = 0;
      weight[offset+11] = 0;
      weight[offset+12] = 0;
      weight[offset+13] = 0;
      break;
    case 4:
      weight[offset+0] = 0.5 * y2 - 0.5 * y + 0.5 * z2;
      weight[offset+1] = -1.0 * y2 + 1.0 * y - 1.0 * z;
      weight[offset+2] = 0.5 * y2 - 0.5 * y + 0.5 * z2;
      weight[offset+3] = -1.0 * y2 - 1.0 * z + 1.0;
      weight[offset+4] = 2.0 * (y - z + 1.0) * (y + z - 1.0);
      weight[offset+5] = -1.0 * y2 - 1.0 * z + 1.0;
      weight[offset+6] = 0.5 * y2 + 0.5 * y + 0.5 * z2;
      weight[offset+7] = -1.0 * y2 - 1.0 * y - 1.0 * z;
      weight[offset+8] = 0.5 * y2 + 0.5 * y + 0.5 * z2;
      weight[offset+9] = 0;
      weight[offset+10] = 0;
      weight[offset+11] = 0;
      weight[offset+12] = 0;
      weight[offset+13] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 4.0 * weight[offset+0];
    weight[offset+1] = 4.0 * weight[offset+1];
    weight[offset+2] = 4.0 * weight[offset+2];
    weight[offset+3] = 4.0 * weight[offset+3];
    weight[offset+4] = 4.0 * weight[offset+4];
    weight[offset+5] = 4.0 * weight[offset+5];
    weight[offset+6] = 4.0 * weight[offset+6];
    weight[offset+7] = 4.0 * weight[offset+7];
    weight[offset+8] = 4.0 * weight[offset+8];
    weight[offset+9] = 4.0 * weight[offset+9];
    weight[offset+10] = 4.0 * weight[offset+10];
    weight[offset+11] = 4.0 * weight[offset+11];
    weight[offset+12] = 4.0 * weight[offset+12];
    weight[offset+13] = 4.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_xy(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));
  
  if (this->my_deg_ == 0) {

    weight[offset+0] = 0.0;

  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0.250000000000000;
      weight[offset+1] = -0.250000000000000;
      weight[offset+2] = -0.250000000000000;
      weight[offset+3] = 0.250000000000000;
      weight[offset+4] = 0;
      break;
    case 2:
      weight[offset+0] = 0.250000000000000;
      weight[offset+1] = -0.250000000000000;
      weight[offset+2] = -0.250000000000000;
      weight[offset+3] = 0.250000000000000;
      weight[offset+4] = 0;
      break;
    case 3:
      weight[offset+0] = 0.250000000000000;
      weight[offset+1] = -0.250000000000000;
      weight[offset+2] = -0.250000000000000;
      weight[offset+3] = 0.250000000000000;
      weight[offset+4] = 0;
      break;
    case 4:
      weight[offset+0] = 0.250000000000000;
      weight[offset+1] = -0.250000000000000;
      weight[offset+2] = -0.250000000000000;
      weight[offset+3] = 0.250000000000000;
      weight[offset+4] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 4.0 * weight[offset+0];
    weight[offset+1] = 4.0 * weight[offset+1];
    weight[offset+2] = 4.0 * weight[offset+2];
    weight[offset+3] = 4.0 * weight[offset+3];
    weight[offset+4] = 4.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 1.0 * x * y - 1.0 * x * z - 0.5 * x - 0.5 * y + 0.25;
      weight[offset+1] = x * (-2.0 * y + 1.0 * z + 1.0);
      weight[offset+2] = 1.0 * x * y - 1.0 * x * z - 0.5 * x + 0.5 * y - 0.25;
      weight[offset+3] = -2.0 * x * y + 1.0 * x * z + 1.0 * y - 0.5 * z;
      weight[offset+4] = 4.0 * x * y;
      weight[offset+5] = -2.0 * x * y + 1.0 * x * z - 1.0 * y + 0.5 * z;
      weight[offset+6] =
          1.0 * x * y - 1.0 * x * z + 0.5 * x - 0.5 * y + 0.5 * z - 0.25;
      weight[offset+7] = x * (-2.0 * y + 1.0 * z - 1.0);
      weight[offset+8] =
          1.0 * x * y - 1.0 * x * z + 0.5 * x + 0.5 * y - 0.5 * z + 0.25;
      weight[offset+9] = 1.0 * z;
      weight[offset+10] = -1.0 * z;
      weight[offset+11] = -1.0 * z;
      weight[offset+12] = 1.0 * z;
      weight[offset+13] = 0;
      break;
    case 2:
      weight[offset+0] =
          1.0 * x * y - 0.5 * x + 1.0 * y * z - 0.5 * y - 0.5 * z + 0.25;
      weight[offset+1] = -2.0 * x * y + 1.0 * x - 1.0 * y * z + 0.5 * z;
      weight[offset+2] = 1.0 * x * y - 0.5 * x + 1.0 * y * z + 0.5 * y - 0.25;
      weight[offset+3] = y * (-2.0 * x - 1.0 * z + 1.0);
      weight[offset+4] = 4.0 * x * y;
      weight[offset+5] = -y * (2.0 * x + 1.0 * z + 1.0);
      weight[offset+6] =
          1.0 * x * y + 0.5 * x + 1.0 * y * z - 0.5 * y + 0.5 * z - 0.25;
      weight[offset+7] = -2.0 * x * y - 1.0 * x - 1.0 * y * z - 0.5 * z;
      weight[offset+8] = 1.0 * x * y + 0.5 * x + 1.0 * y * z + 0.5 * y + 0.25;
      weight[offset+9] = 1.0 * z;
      weight[offset+10] = -1.0 * z;
      weight[offset+11] = -1.0 * z;
      weight[offset+12] = 1.0 * z;
      weight[offset+13] = 0;
      break;
    case 3:
      weight[offset+0] =
          1.0 * x * y + 1.0 * x * z - 0.5 * x - 0.5 * y - 0.5 * z + 0.25;
      weight[offset+1] = x * (-2.0 * y - 1.0 * z + 1.0);
      weight[offset+2] =
          1.0 * x * y + 1.0 * x * z - 0.5 * x + 0.5 * y + 0.5 * z - 0.25;
      weight[offset+3] = -2.0 * x * y - 1.0 * x * z + 1.0 * y + 0.5 * z;
      weight[offset+4] = 4.0 * x * y;
      weight[offset+5] = -2.0 * x * y - 1.0 * x * z - 1.0 * y - 0.5 * z;
      weight[offset+6] = 1.0 * x * y + 1.0 * x * z + 0.5 * x - 0.5 * y - 0.25;
      weight[offset+7] = -x * (2.0 * y + 1.0 * z + 1.0);
      weight[offset+8] = 1.0 * x * y + 1.0 * x * z + 0.5 * x + 0.5 * y + 0.25;
      weight[offset+9] = 1.0 * z;
      weight[offset+10] = -1.0 * z;
      weight[offset+11] = -1.0 * z;
      weight[offset+12] = 1.0 * z;
      weight[offset+13] = 0;
      break;
    case 4:
      weight[offset+0] = 1.0 * x * y - 0.5 * x - 1.0 * y * z - 0.5 * y + 0.25;
      weight[offset+1] = -2.0 * x * y + 1.0 * x + 1.0 * y * z - 0.5 * z;
      weight[offset+2] =
          1.0 * x * y - 0.5 * x - 1.0 * y * z + 0.5 * y + 0.5 * z - 0.25;
      weight[offset+3] = y * (-2.0 * x + 1.0 * z + 1.0);
      weight[offset+4] = 4.0 * x * y;
      weight[offset+5] = y * (-2.0 * x + 1.0 * z - 1.0);
      weight[offset+6] = 1.0 * x * y + 0.5 * x - 1.0 * y * z - 0.5 * y - 0.25;
      weight[offset+7] = -2.0 * x * y - 1.0 * x + 1.0 * y * z + 0.5 * z;
      weight[offset+8] =
          1.0 * x * y + 0.5 * x - 1.0 * y * z + 0.5 * y - 0.5 * z + 0.25;
      weight[offset+9] = 1.0 * z;
      weight[offset+10] = -1.0 * z;
      weight[offset+11] = -1.0 * z;
      weight[offset+12] = 1.0 * z;
      weight[offset+13] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    if (xx == 0 && yy == 0) {
      weight[offset+2] =
          1.0 * x * y - 0.5 * x - 1.0 * y * z + 0.5 * y + 0.5 * z - 0.25;
      weight[offset+3] = -2.0 * x * y - 1.0 * x * z + 1.0 * y + 0.5 * z;
      weight[offset+5] = -2.0 * x * y - 1.0 * x * z - 1.0 * y - 0.5 * z;
      weight[offset+8] =
          1.0 * x * y + 0.5 * x - 1.0 * y * z + 0.5 * y - 0.5 * z + 0.25;
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 4.0 * weight[offset+0];
    weight[offset+1] = 4.0 * weight[offset+1];
    weight[offset+2] = 4.0 * weight[offset+2];
    weight[offset+3] = 4.0 * weight[offset+3];
    weight[offset+4] = 4.0 * weight[offset+4];
    weight[offset+5] = 4.0 * weight[offset+5];
    weight[offset+6] = 4.0 * weight[offset+6];
    weight[offset+7] = 4.0 * weight[offset+7];
    weight[offset+8] = 4.0 * weight[offset+8];
    weight[offset+9] = 4.0 * weight[offset+9];
    weight[offset+10] = 4.0 * weight[offset+10];
    weight[offset+11] = 4.0 * weight[offset+11];
    weight[offset+12] = 4.0 * weight[offset+12];
    weight[offset+13] = 4.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_xz(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));
  
  if (this->my_deg_ == 0) {

    weight[offset+0] = 0.0;

  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = -0.250000000000000;
      weight[offset+1] = 0.250000000000000;
      weight[offset+2] = 0.250000000000000;
      weight[offset+3] = -0.250000000000000;
      weight[offset+4] = 0;
      break;
    case 2:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 3:
      weight[offset+0] = 0.250000000000000;
      weight[offset+1] = -0.250000000000000;
      weight[offset+2] = -0.250000000000000;
      weight[offset+3] = 0.250000000000000;
      weight[offset+4] = 0;
      break;
    case 4:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 2.0 * weight[offset+0];
    weight[offset+1] = 2.0 * weight[offset+1];
    weight[offset+2] = 2.0 * weight[offset+2];
    weight[offset+3] = 2.0 * weight[offset+3];
    weight[offset+4] = 2.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = -1.0 * x * y + 1.0 * x * z + 0.5 * x + 0.5 * z + 0.25;
      weight[offset+1] = x * (1.0 * y - 2.0);
      weight[offset+2] = -1.0 * x * y + 1.0 * x * z + 0.5 * x - 0.5 * z - 0.25;
      weight[offset+3] = (1.0 * x - 0.5) * (y - 1.0);
      weight[offset+4] = x * (-4.0 * z + 4.0);
      weight[offset+5] = (1.0 * x + 0.5) * (y - 1.0);
      weight[offset+6] =
          -1.0 * x * y + 1.0 * x * z - 0.5 * x + 0.5 * y - 0.5 * z + 0.25;
      weight[offset+7] = x * y;
      weight[offset+8] =
          -1.0 * x * y + 1.0 * x * z - 0.5 * x - 0.5 * y + 0.5 * z - 0.25;
      weight[offset+9] = 1.0 * y - 2.0 * z - 1.0;
      weight[offset+10] = -1.0 * y + 2.0 * z + 1.0;
      weight[offset+11] = -1.0 * y + 2.0 * z - 1.0;
      weight[offset+12] = 1.0 * y - 2.0 * z + 1.0;
      weight[offset+13] = 0;
      break;
    case 2:
      weight[offset+0] =
          1.0 * x * z + 0.5 * y2 - 0.5 * y + 1.5 * z2 - 0.5 * z;
      weight[offset+1] = -1.0 * x - 0.5 * y2 + 0.5 * y - 1.0 * z + 0.5;
      weight[offset+2] =
          1.0 * x * z + 0.5 * y2 + 1.5 * z2 - 0.5 * z - 0.5;
      weight[offset+3] = -1.0 * x - 0.5 * y2 - 1.0 * z + 1.0;
      weight[offset+4] = -4.0 * x * z + 4.0 * x - 6.0 * z2 + 6.0 * z - 1.0;
      weight[offset+5] = -1.0 * x - 0.5 * y2 - 1.0 * z;
      weight[offset+6] =
          1.0 * x * z + 0.5 * y2 + 0.5 * y + 1.5 * z2 - 0.5 * z;
      weight[offset+7] = -1.0 * x - 0.5 * y2 - 0.5 * y - 1.0 * z + 0.5;
      weight[offset+8] =
          1.0 * x * z + 0.5 * y2 + 1.5 * z2 - 0.5 * z - 0.5;
      weight[offset+9] = 1.0 * y - 1.0;
      weight[offset+10] = -1.0 * y + 1.0;
      weight[offset+11] = -1.0 * y - 1.0;
      weight[offset+12] = 1.0 * y + 1.0;
      weight[offset+13] = 0;
      break;
    case 3:
      weight[offset+0] =
          1.0 * x * y + 1.0 * x * z - 0.5 * x - 0.5 * y - 0.5 * z + 0.25;
      weight[offset+1] = -1.0 * x * y;
      weight[offset+2] =
          1.0 * x * y + 1.0 * x * z - 0.5 * x + 0.5 * y + 0.5 * z - 0.25;
      weight[offset+3] = (-1.0 * x + 0.5) * (y + 1.0);
      weight[offset+4] = x * (-4.0 * z + 4.0);
      weight[offset+5] = -(1.0 * x + 0.5) * (y + 1.0);
      weight[offset+6] = 1.0 * x * y + 1.0 * x * z + 0.5 * x + 0.5 * z + 0.25;
      weight[offset+7] = -x * (1.0 * y + 2.0);
      weight[offset+8] = 1.0 * x * y + 1.0 * x * z + 0.5 * x - 0.5 * z - 0.25;
      weight[offset+9] = 1.0 * y + 2.0 * z - 1.0;
      weight[offset+10] = -1.0 * y - 2.0 * z + 1.0;
      weight[offset+11] = -1.0 * y - 2.0 * z - 1.0;
      weight[offset+12] = 1.0 * y + 2.0 * z + 1.0;
      weight[offset+13] = 0;
      break;
    case 4:
      weight[offset+0] =
          1.0 * x * z - 0.5 * y2 - 1.5 * z2 + 0.5 * z + 0.5;
      weight[offset+1] = -1.0 * x + 0.5 * y2 - 0.5 * y + 1.0 * z - 0.5;
      weight[offset+2] =
          1.0 * x * z - 0.5 * y2 + 0.5 * y - 1.5 * z2 + 0.5 * z;
      weight[offset+3] = -1.0 * x + 0.5 * y2 + 1.0 * z;
      weight[offset+4] = -4.0 * x * z + 4.0 * x + 6.0 * z2 - 6.0 * z + 1.0;
      weight[offset+5] = -1.0 * x + 0.5 * y2 + 1.0 * z - 1.0;
      weight[offset+6] =
          1.0 * x * z - 0.5 * y2 - 1.5 * z2 + 0.5 * z + 0.5;
      weight[offset+7] = -1.0 * x + 0.5 * y2 + 0.5 * y + 1.0 * z - 0.5;
      weight[offset+8] =
          1.0 * x * z - 0.5 * y2 - 0.5 * y - 1.5 * z2 + 0.5 * z;
      weight[offset+9] = 1.0 * y - 1.0;
      weight[offset+10] = -1.0 * y + 1.0;
      weight[offset+11] = -1.0 * y - 1.0;
      weight[offset+12] = 1.0 * y + 1.0;
      weight[offset+13] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    if (xx == 0 && yy == 0) {
      weight[offset+2] =
          1.0 * x * z - 0.5 * y2 + 0.5 * y - 1.5 * z2 + 0.5 * z;
      weight[offset+3] = -1.0 * x + 0.5 * y2 + 1.0 * z;
      weight[offset+8] =
          1.0 * x * z - 0.5 * y2 - 0.5 * y - 1.5 * z2 + 0.5 * z;
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 2.0 * weight[offset+0];
    weight[offset+1] = 2.0 * weight[offset+1];
    weight[offset+2] = 2.0 * weight[offset+2];
    weight[offset+3] = 2.0 * weight[offset+3];
    weight[offset+4] = 2.0 * weight[offset+4];
    weight[offset+5] = 2.0 * weight[offset+5];
    weight[offset+6] = 2.0 * weight[offset+6];
    weight[offset+7] = 2.0 * weight[offset+7];
    weight[offset+8] = 2.0 * weight[offset+8];
    weight[offset+9] = 2.0 * weight[offset+9];
    weight[offset+10] = 2.0 * weight[offset+10];
    weight[offset+11] = 2.0 * weight[offset+11];
    weight[offset+12] = 2.0 * weight[offset+12];
    weight[offset+13] = 2.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_yy(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));

  if (this->my_deg_ == 0) {

    weight[offset+0] = 0.0;

  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 2:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 3:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 4:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 4.0 * weight[offset+0];
    weight[offset+1] = 4.0 * weight[offset+1];
    weight[offset+2] = 4.0 * weight[offset+2];
    weight[offset+3] = 4.0 * weight[offset+3];
    weight[offset+4] = 4.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0.5 * x2 - 0.5 * x + 0.5 * z2;
      weight[offset+1] = -1.0 * x2 - 1.0 * z + 1.0;
      weight[offset+2] = 0.5 * x2 + 0.5 * x + 0.5 * z2;
      weight[offset+3] = -1.0 * x2 + 1.0 * x - 1.0 * z;
      weight[offset+4] = 2.0 * (x - z + 1.0) * (x + z - 1.0);
      weight[offset+5] = -1.0 * x2 - 1.0 * x - 1.0 * z;
      weight[offset+6] = 0.5 * x2 - 0.5 * x + 0.5 * z2;
      weight[offset+7] = -1.0 * x2 - 1.0 * z + 1.0;
      weight[offset+8] = 0.5 * x2 + 0.5 * x + 0.5 * z2;
      weight[offset+9] = 0;
      weight[offset+10] = 0;
      weight[offset+11] = 0;
      weight[offset+12] = 0;
      weight[offset+13] = 0;
      break;
    case 2:
      weight[offset+0] = 0.5 * (x + z) * (x + z - 1.0);
      weight[offset+1] = -1.0 * (x + 1) * (x + z - 1);
      weight[offset+2] = 0.5 * (x + z) * (x + z + 1.0);
      weight[offset+3] = 1.0 * x * (-x - z + 1);
      weight[offset+4] = 2.0 * (x - z + 1.0) * (x + z - 1.0);
      weight[offset+5] = -1.0 * x * (x + z + 1.0) - 2.0 * z;
      weight[offset+6] = 0.5 * (x + z) * (x + z - 1.0);
      weight[offset+7] = -1.0 * x2 - 1.0 * x * z - 1.0 * z + 1.0;
      weight[offset+8] = 0.5 * (x + z) * (x + z + 1.0);
      weight[offset+9] = 0;
      weight[offset+10] = 0;
      weight[offset+11] = 0;
      weight[offset+12] = 0;
      weight[offset+13] = 0;
      break;
    case 3:
      weight[offset+0] = 0.5 * x2 - 0.5 * x + 0.5 * z2;
      weight[offset+1] = -1.0 * x2 - 1.0 * z + 1.0;
      weight[offset+2] = 0.5 * x2 + 0.5 * x + 0.5 * z2;
      weight[offset+3] = -1.0 * x2 + 1.0 * x - 1.0 * z;
      weight[offset+4] = 2.0 * (x - z + 1.0) * (x + z - 1.0);
      weight[offset+5] = -1.0 * x2 - 1.0 * x - 1.0 * z;
      weight[offset+6] = 0.5 * x2 - 0.5 * x + 0.5 * z2;
      weight[offset+7] = -1.0 * x2 - 1.0 * z + 1.0;
      weight[offset+8] = 0.5 * x2 + 0.5 * x + 0.5 * z2;
      weight[offset+9] = 0;
      weight[offset+10] = 0;
      weight[offset+11] = 0;
      weight[offset+12] = 0;
      weight[offset+13] = 0;
      break;
    case 4:
      weight[offset+0] = (-0.5 * x + 0.5 * z) * (-x + z + 1.0);
      weight[offset+1] = -1.0 * x2 + 1.0 * x * z - 1.0 * z + 1.0;
      weight[offset+2] = 2.0 * (0.25 * x - 0.25 * z) * (x - z + 1.0);
      weight[offset+3] = 1.0 * x * (-x + z + 1.0) - 2.0 * z;
      weight[offset+4] = 2.0 * (x - z + 1.0) * (x + z - 1.0);
      weight[offset+5] = 1.0 * x * (-x + z - 1);
      weight[offset+6] = (-0.5 * x + 0.5 * z) * (-x + z + 1.0);
      weight[offset+7] = -1.0 * x2 + 1.0 * x * z - 1.0 * z + 1.0;
      weight[offset+8] = (0.5 * x - 0.5 * z) * (x - z + 1.0);
      weight[offset+9] = 0;
      weight[offset+10] = 0;
      weight[offset+11] = 0;
      weight[offset+12] = 0;
      weight[offset+13] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    if (xx == 0 && yy == 0) {
      weight[offset+2] = 2.0 * (0.25 * x - 0.25 * z) * (x - z + 1.0);
      weight[offset+3] = 1.0 * x * (-x + z + 1.0) - 2.0 * z;
      weight[offset+8] = (0.5 * x - 0.5 * z) * (x - z + 1.0);
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 4.0 * weight[offset+0];
    weight[offset+1] = 4.0 * weight[offset+1];
    weight[offset+2] = 4.0 * weight[offset+2];
    weight[offset+3] = 4.0 * weight[offset+3];
    weight[offset+4] = 4.0 * weight[offset+4];
    weight[offset+5] = 4.0 * weight[offset+5];
    weight[offset+6] = 4.0 * weight[offset+6];
    weight[offset+7] = 4.0 * weight[offset+7];
    weight[offset+8] = 4.0 * weight[offset+8];
    weight[offset+9] = 4.0 * weight[offset+9];
    weight[offset+10] = 4.0 * weight[offset+10];
    weight[offset+11] = 4.0 * weight[offset+11];
    weight[offset+12] = 4.0 * weight[offset+12];
    weight[offset+13] = 4.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_yz(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());

  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));

  if (this->my_deg_ == 0) {

    weight[offset+0] = 0.0;

  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 2:
      weight[offset+0] = 0.250000000000000;
      weight[offset+1] = -0.250000000000000;
      weight[offset+2] = -0.250000000000000;
      weight[offset+3] = 0.250000000000000;
      weight[offset+4] = 0;
      break;
    case 3:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 4:
      weight[offset+0] = -0.250000000000000;
      weight[offset+1] = 0.250000000000000;
      weight[offset+2] = 0.250000000000000;
      weight[offset+3] = -0.250000000000000;
      weight[offset+4] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 2.0 * weight[offset+0];
    weight[offset+1] = 2.0 * weight[offset+1];
    weight[offset+2] = 2.0 * weight[offset+2];
    weight[offset+3] = 2.0 * weight[offset+3];
    weight[offset+4] = 2.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx >= yy && xx <= -yy) {
      indicator = 1;
    } else if (xx > yy && xx > -yy) {
      indicator = 2;
    } else if (xx <= yy && xx >= -yy) {
      indicator = 3;
    } else if (xx < yy && xx < -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] =
          -0.5 * x2 + 1.0 * y * z - 1.5 * z2 + 0.5 * z + 0.5;
      weight[offset+1] = 0.5 * x2 - 1.0 * y + 1.0 * z;
      weight[offset+2] =
          -0.5 * x2 + 1.0 * y * z - 1.5 * z2 + 0.5 * z + 0.5;
      weight[offset+3] = 0.5 * x2 - 0.5 * x - 1.0 * y + 1.0 * z - 0.5;
      weight[offset+4] = -4.0 * y * z + 4.0 * y + 6.0 * z2 - 6.0 * z + 1.0;
      weight[offset+5] = 0.5 * x2 + 0.5 * x - 1.0 * y + 1.0 * z - 0.5;
      weight[offset+6] =
          -0.5 * x2 + 0.5 * x + 1.0 * y * z - 1.5 * z2 + 0.5 * z;
      weight[offset+7] = 0.5 * x2 - 1.0 * y + 1.0 * z - 1.0;
      weight[offset+8] =
          -0.5 * x2 - 0.5 * x + 1.0 * y * z - 1.5 * z2 + 0.5 * z;
      weight[offset+9] = 1.0 * x - 1.0;
      weight[offset+10] = -1.0 * x - 1.0;
      weight[offset+11] = -1.0 * x + 1.0;
      weight[offset+12] = 1.0 * x + 1.0;
      weight[offset+13] = 0;
      break;
    case 2:
      weight[offset+0] =
          1.0 * x * y - 0.5 * x + 1.0 * y * z - 0.5 * y - 0.5 * z + 0.25;
      weight[offset+1] = (x + 1.0) * (-1.0 * y + 0.5);
      weight[offset+2] = 1.0 * x * y + 1.0 * y * z + 0.5 * y + 0.5 * z + 0.25;
      weight[offset+3] = -1.0 * x * y;
      weight[offset+4] = y * (-4.0 * z + 4.0);
      weight[offset+5] = -y * (1.0 * x + 2.0);
      weight[offset+6] =
          1.0 * x * y + 0.5 * x + 1.0 * y * z - 0.5 * y + 0.5 * z - 0.25;
      weight[offset+7] = -(x + 1.0) * (1.0 * y + 0.5);
      weight[offset+8] = 1.0 * x * y + 1.0 * y * z + 0.5 * y - 0.5 * z - 0.25;
      weight[offset+9] = 1.0 * x + 2.0 * z - 1.0;
      weight[offset+10] = -1.0 * x - 2.0 * z - 1.0;
      weight[offset+11] = -1.0 * x - 2.0 * z + 1.0;
      weight[offset+12] = 1.0 * x + 2.0 * z + 1.0;
      weight[offset+13] = 0;
      break;
    case 3:
      weight[offset+0] =
          0.5 * x2 - 0.5 * x + 1.0 * y * z + 1.5 * z2 - 0.5 * z;
      weight[offset+1] = -0.5 * x2 - 1.0 * y - 1.0 * z + 1.0;
      weight[offset+2] =
          0.5 * x2 + 0.5 * x + 1.0 * y * z + 1.5 * z2 - 0.5 * z;
      weight[offset+3] = -0.5 * x2 + 0.5 * x - 1.0 * y - 1.0 * z + 0.5;
      weight[offset+4] = -4.0 * y * z + 4.0 * y - 6.0 * z2 + 6.0 * z - 1.0;
      weight[offset+5] = -0.5 * x2 - 0.5 * x - 1.0 * y - 1.0 * z + 0.5;
      weight[offset+6] =
          0.5 * x2 + 1.0 * y * z + 1.5 * z2 - 0.5 * z - 0.5;
      weight[offset+7] = -0.5 * x2 - 1.0 * y - 1.0 * z;
      weight[offset+8] =
          0.5 * x2 + 1.0 * y * z + 1.5 * z2 - 0.5 * z - 0.5;
      weight[offset+9] = 1.0 * x - 1.0;
      weight[offset+10] = -1.0 * x - 1.0;
      weight[offset+11] = -1.0 * x + 1.0;
      weight[offset+12] = 1.0 * x + 1.0;
      weight[offset+13] = 0;
      break;
    case 4:
      weight[offset+0] = -1.0 * x * y + 1.0 * y * z + 0.5 * y + 0.5 * z + 0.25;
      weight[offset+1] = (x - 1.0) * (1.0 * y - 0.5);
      weight[offset+2] =
          -1.0 * x * y + 0.5 * x + 1.0 * y * z - 0.5 * y - 0.5 * z + 0.25;
      weight[offset+3] = y * (1.0 * x - 2.0);
      weight[offset+4] = y * (-4.0 * z + 4.0);
      weight[offset+5] = 1.0 * x * y;
      weight[offset+6] = -1.0 * x * y + 1.0 * y * z + 0.5 * y - 0.5 * z - 0.25;
      weight[offset+7] = (x - 1.0) * (1.0 * y + 0.5);
      weight[offset+8] =
          -1.0 * x * y - 0.5 * x + 1.0 * y * z - 0.5 * y + 0.5 * z - 0.25;
      weight[offset+9] = 1.0 * x - 2.0 * z - 1.0;
      weight[offset+10] = -1.0 * x + 2.0 * z - 1.0;
      weight[offset+11] = -1.0 * x + 2.0 * z + 1.0;
      weight[offset+12] = 1.0 * x - 2.0 * z + 1.0;
      weight[offset+13] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    if (xx == 0 && yy == 0) {
      weight[offset+0] =
          0.5 * x2 - 0.5 * x + 1.0 * y * z + 1.5 * z2 - 0.5 * z;
      weight[offset+2] =
          0.5 * x2 + 0.5 * x + 1.0 * y * z + 1.5 * z2 - 0.5 * z;
      weight[offset+7] = -0.5 * x2 - 1.0 * y - 1.0 * z;
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 2.0 * weight[offset+0];
    weight[offset+1] = 2.0 * weight[offset+1];
    weight[offset+2] = 2.0 * weight[offset+2];
    weight[offset+3] = 2.0 * weight[offset+3];
    weight[offset+4] = 2.0 * weight[offset+4];
    weight[offset+5] = 2.0 * weight[offset+5];
    weight[offset+6] = 2.0 * weight[offset+6];
    weight[offset+7] = 2.0 * weight[offset+7];
    weight[offset+8] = 2.0 * weight[offset+8];
    weight[offset+9] = 2.0 * weight[offset+9];
    weight[offset+10] = 2.0 * weight[offset+10];
    weight[offset+11] = 2.0 * weight[offset+11];
    weight[offset+12] = 2.0 * weight[offset+12];
    weight[offset+13] = 2.0 * weight[offset+13];
  }
}

template < class DataType, int DIM >
void PyrLag< DataType, DIM >::N_zz(const Coord &pt, size_t comp, size_t offset, std::vector<DataType> &weight) const 
{
  assert (DIM == 3);
  assert(comp < this->nb_comp_);
  assert(offset + this->comp_weight_size_[comp] <= weight.size());
  assert (this->my_deg_ >= 0);
  
  const DataType x = 2.0 * pt[0] - 1.0;
  const DataType y = 2.0 * pt[1] - 1.0;
  const DataType z = pt[2];
  const DataType x2 = std::pow(x,2);
  const DataType y2 = std::pow(y,2);
  const DataType z2 = std::pow(z,2);
  
  std::setprecision(2 * sizeof(DataType));

  if (this->my_deg_ == 0) {

    weight[offset+0] = 0.0;

  } else if (this->my_deg_ == 1) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 2:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 3:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    case 4:
      weight[offset+0] = 0;
      weight[offset+1] = 0;
      weight[offset+2] = 0;
      weight[offset+3] = 0;
      weight[offset+4] = 0;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 1.0 * weight[offset+0];
    weight[offset+1] = 1.0 * weight[offset+1];
    weight[offset+2] = 1.0 * weight[offset+2];
    weight[offset+3] = 1.0 * weight[offset+3];
    weight[offset+4] = 1.0 * weight[offset+4];

  } else if (this->my_deg_ == 2) {

    int indicator = -1;

    const DataType xx = trunc(x * 1e10);
    const DataType yy = trunc(y * 1e10);

    if (xx > yy && xx < -yy) {
      indicator = 1;
    } else if (xx >= yy && xx >= -yy) {
      indicator = 2;
    } else if (xx < yy && xx > -yy) {
      indicator = 3;
    } else if (xx <= yy && xx <= -yy) {
      indicator = 4;
    }

    switch (indicator) {
    case 1:
      weight[offset+0] = 0.5 * x2 + 0.5 * x + 0.5 * y2 - 3.0 * y * z +
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+1] = 1.0 * y;
      weight[offset+2] = 0.5 * x2 - 0.5 * x + 0.5 * y2 - 3.0 * y * z +
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+3] = 1.0 * y;
      weight[offset+4] = -2.0 * x2 - 2.0 * y2 + 12.0 * y * z - 6.0 * y -
                  12.0 * z2 + 6.0 * z + 4.0;
      weight[offset+5] = 1.0 * y;
      weight[offset+6] = 0.5 * x2 - 0.5 * x + 0.5 * y2 - 3.0 * y * z +
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+7] = 1.0 * y;
      weight[offset+8] = 0.5 * x2 + 0.5 * x + 0.5 * y2 - 3.0 * y * z +
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+9] = -2.0 * x - 2.0;
      weight[offset+10] = 2.0 * x - 2.0;
      weight[offset+11] = 2.0 * x - 2.0;
      weight[offset+12] = -2.0 * x - 2.0;
      weight[offset+13] = 4.00000000000000;
      break;
    case 2:
      weight[offset+0] = 0.5 * x2 + 3.0 * x * z - 0.5 * x + 0.5 * y2 -
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+1] = -1.0 * x;
      weight[offset+2] = 0.5 * x2 + 3.0 * x * z - 0.5 * x + 0.5 * y2 +
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+3] = -1.0 * x;
      weight[offset+4] = -2.0 * x2 - 12.0 * x * z + 6.0 * x - 2.0 * y2 -
                  12.0 * z2 + 6.0 * z + 4.0;
      weight[offset+5] = -1.0 * x;
      weight[offset+6] = 0.5 * x2 + 3.0 * x * z - 0.5 * x + 0.5 * y2 +
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+7] = -1.0 * x;
      weight[offset+8] = 0.5 * x2 + 3.0 * x * z - 0.5 * x + 0.5 * y2 -
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+9] = 2.0 * y - 2.0;
      weight[offset+10] = -2.0 * y - 2.0;
      weight[offset+11] = -2.0 * y - 2.0;
      weight[offset+12] = 2.0 * y - 2.0;
      weight[offset+13] = 4.00000000000000;
      break;
    case 3:
      weight[offset+0] = 0.5 * x2 - 0.5 * x + 0.5 * y2 + 3.0 * y * z -
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+1] = -1.0 * y;
      weight[offset+2] = 0.5 * x2 + 0.5 * x + 0.5 * y2 + 3.0 * y * z -
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+3] = -1.0 * y;
      weight[offset+4] = -2.0 * x2 - 2.0 * y2 - 12.0 * y * z + 6.0 * y -
                  12.0 * z2 + 6.0 * z + 4.0;
      weight[offset+5] = -1.0 * y;
      weight[offset+6] = 0.5 * x2 + 0.5 * x + 0.5 * y2 + 3.0 * y * z -
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+7] = -1.0 * y;
      weight[offset+8] = 0.5 * x2 - 0.5 * x + 0.5 * y2 + 3.0 * y * z -
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+9] = 2.0 * x - 2.0;
      weight[offset+10] = -2.0 * x - 2.0;
      weight[offset+11] = -2.0 * x - 2.0;
      weight[offset+12] = 2.0 * x - 2.0;
      weight[offset+13] = 4.00000000000000;
      break;
    case 4:
      weight[offset+0] = 0.5 * x2 - 3.0 * x * z + 0.5 * x + 0.5 * y2 +
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+1] = 1.0 * x;
      weight[offset+2] = 0.5 * x2 - 3.0 * x * z + 0.5 * x + 0.5 * y2 -
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+3] = 1.0 * x;
      weight[offset+4] = -2.0 * x2 + 12.0 * x * z - 6.0 * x - 2.0 * y2 -
                  12.0 * z2 + 6.0 * z + 4.0;
      weight[offset+5] = 1.0 * x;
      weight[offset+6] = 0.5 * x2 - 3.0 * x * z + 0.5 * x + 0.5 * y2 -
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+7] = 1.0 * x;
      weight[offset+8] = 0.5 * x2 - 3.0 * x * z + 0.5 * x + 0.5 * y2 +
                  0.5 * y + 3.0 * z2 - 1.5 * z;
      weight[offset+9] = -2.0 * y - 2.0;
      weight[offset+10] = 2.0 * y - 2.0;
      weight[offset+11] = 2.0 * y - 2.0;
      weight[offset+12] = -2.0 * y - 2.0;
      weight[offset+13] = 4.00000000000000;
      break;
    default:
      std::cerr << "Unexpected coordinate " << std::endl;
      quit_program();
    }

    // rescaling due to changing coordinates
    weight[offset+0] = 1.0 * weight[offset+0];
    weight[offset+1] = 1.0 * weight[offset+1];
    weight[offset+2] = 1.0 * weight[offset+2];
    weight[offset+3] = 1.0 * weight[offset+3];
    weight[offset+4] = 1.0 * weight[offset+4];
    weight[offset+5] = 1.0 * weight[offset+5];
    weight[offset+6] = 1.0 * weight[offset+6];
    weight[offset+7] = 1.0 * weight[offset+7];
    weight[offset+8] = 1.0 * weight[offset+8];
    weight[offset+9] = 1.0 * weight[offset+9];
    weight[offset+10] = 1.0 * weight[offset+10];
    weight[offset+11] = 1.0 * weight[offset+11];
    weight[offset+12] = 1.0 * weight[offset+12];
    weight[offset+13] = 1.0 * weight[offset+13];
  }
}

} // namespace doffem
} // namespace hiflow
#endif
