#include "pdf.h"
#include <cassert>
#include <cmath> // for abs
#include <iostream>
#include <math.h> // for pow

namespace hiflow {
namespace polynomialchaos {

template < class DataType >
PDF< DataType >::PDF()
    : a_(1.), b_(-1.), c_(1.), d_(-1.), norm_const_(1.), num_quad_points_(1),
      num_int_intervals_(1) {
  // default setup
  this->SetQuadPoints(30);
  this->SetNumIntegrationIntervals(10);
}

template < class DataType >
void PDF< DataType >::SetIntegrationInterval(DataType c, DataType d) {
  assert(c <= d);
  c_ = c;
  d_ = d;
}

template < class DataType >
void PDF< DataType >::SetNumIntegrationIntervals(int num_int_intervals) {
  num_int_intervals_ = num_int_intervals;
}

template < class DataType >
void PDF< DataType >::SetRestriction(DataType a, DataType b) {
  a_ = a;
  b_ = b;
  if (c_ > d_) {
    c_ = a;
    d_ = b;
  }
  // TODO: calculate new norm_const_?
  moments_.clear();
}

template < class DataType > bool PDF< DataType >::IsRestricted() const {
  return a_ < b_;
}

template < class DataType > void PDF< DataType >::SetQuadPoints(int order) {
  // assert( num > 0) ;
  gauss_quad_.set_quadrature_by_order("GaussLine", order);
  
  num_quad_points_ = gauss_quad_.size();
  quad_points_.resize(num_quad_points_);
  quad_weights_.resize(num_quad_points_);
  
  for (int i = 0; i < num_quad_points_; ++i) {
    quad_points_[i] = gauss_quad_.x(i);
    quad_weights_[i] = gauss_quad_.w(i);
  }
}

template < class DataType >
DataType PDF< DataType >::operator()(const DataType x) const {
  assert(norm_const_ > 0);
  if (IsRestricted()) {
    if (x >= a_ && x <= b_) {
      return evaluate_nonnormalized(x) / norm_const_;
    }
    return 0.;
  }
  return evaluate_nonnormalized(x) / norm_const_;
}

template < class DataType > void PDF< DataType >::Normalize() {
  norm_const_ = 0;
  for (int j = 0; j < num_int_intervals_; ++j) {
    DataType left = (DataType)j / (DataType)num_int_intervals_ * (d_ - c_) + c_;
    DataType right =
        ((DataType)j + 1.) / (DataType)num_int_intervals_ * (d_ - c_) + c_;

    DataType temp_int = 0;
    for (size_t i = 0; i < num_quad_points_; ++i) {
      DataType abs_quad_point =
          quad_points_[i] * right + (1. - quad_points_[i]) * left;
      temp_int += quad_weights_[i] * evaluate_nonnormalized(abs_quad_point);
    }
    temp_int *= right - left;
    norm_const_ += temp_int;
  }

  norm_const_ = std::abs(norm_const_);
  assert(norm_const_ > 0);
}

template < class DataType > DataType PDF< DataType >::GetMoment(int moment) {
  if (moments_.count(moment) > 0) {
    return moments_[moment] / norm_const_;
  }

  DataType new_moment = 0;
  for (int j = 0; j < num_int_intervals_; ++j) {
    DataType left = (DataType)j / (DataType)num_int_intervals_ * (d_ - c_) + c_;
    DataType right =
        ((DataType)j + 1.) / (DataType)num_int_intervals_ * (d_ - c_) + c_;

    DataType temp_int = 0;
    for (size_t i = 0; i < num_quad_points_; ++i) {
      DataType abs_quad_point =
          quad_points_[i] * right + (1. - quad_points_[i]) * left;
      temp_int += quad_weights_[i] * evaluate_nonnormalized(abs_quad_point) *
                  pow(abs_quad_point, (double)moment);
    }
    temp_int *= right - left;
    new_moment += temp_int;
  }
  moments_[moment] = new_moment;
  return new_moment / norm_const_;
}

/// template instantiation
template class PDF< double >;
template class PDF< float >;
} // namespace polynomialchaos
} // namespace hiflow
