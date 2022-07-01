// Copyright (C) 2011-2020 Vincent Heuveline
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

#ifndef HIFLOW_ADAPTIVITY_TIME_PATCH_INTERPOLATION
#define HIFLOW_ADAPTIVITY_TIME_PATCH_INTERPOLATION

/// \author Philipp Gerstner

#include <map>
#include <string>
#include <vector>
#include <boost/function.hpp>
#include <mpi.h>

#include "common/log.h"
#include "common/vector_algebra.h"

using namespace hiflow::mesh;
using namespace hiflow::la;

namespace hiflow {

// TODO generalize for arbitrary degree
/// \class TimePatchInterpolation patch_interpolation.h
/// \brief class for evaluating Time-FE functions with degree <= 1 on time
/// interval patch [t_{n-1}, t_n] \cup [t_n, t_{n+1}]
///

template < typename T, class DataType > 
class TimePatchInterpolation 
{
public:
  TimePatchInterpolation() {}

  ~TimePatchInterpolation() {}

  /// \brief Evaluate piecewise constant function
  /// @param[in] c relative time in active interval, c \in [0,1]
  /// @param[in] val_prev time dof value at time t_{n-1} (NOT USED)
  /// @param[in] val time dof value at time t_{n}
  /// @param[in] val_next time dof value at time t_{n+1}
  virtual T constant(const DataType c, 
                     const T &val_prev, 
                     const T &val,
                     const T &val_next) const;

  /// \brief Evaluate piecewise linear function
  /// @param[in] c relative time in active interval, c \in [0,1]
  /// @param[in] val_prev time dof value at time t_{n-1}
  /// @param[in] val time dof value at time t_{n}
  /// @param[in] val_next time dof value at time t_{n+1}
  virtual T linear(const DataType c, 
                   const T &val_prev, 
                   const T &val,
                   const T &val_next) const;

  /// \brief Evaluate temporal derivative of piecewise linear function
  /// @param[in] c relative time in active interval, c \in [0,1]
  /// @param[in] val_prev time dof value at time t_{n-1}
  /// @param[in] val time dof value at time t_{n}
  /// @param[in] val_next time dof value at time t_{n+1}
  virtual T dt_linear(const DataType c, 
                      const T &val_prev, 
                      const T &val,
                      const T &val_next) const;

  /// \brief Evaluate linear function on complete patch [t_{n-1}, t_n] \cup
  /// [t_n, t_{n+1}]
  /// @param[in] c relative time in active interval, c \in [0,1]
  /// @param[in] val_prev time dof value at time t_{n-1} (NOT USED)
  /// @param[in] val time dof value at time t_{n}
  /// @param[in] val_next time dof value at time t_{n+1}
  virtual T patch_linear(const DataType c, 
                         const T &val_prev, 
                         const T &val,
                         const T &val_next) const;

  /// \brief Evaluate temporal derivative of  linear function on complete patch
  /// [t_{n-1}, t_n] \cup [t_n, t_{n+1}]
  /// @param[in] c relative time in active interval, c \in [0,1]
  /// @param[in] val_prev time dof value at time t_{n-1}
  /// @param[in] val time dof value at time t_{n}
  /// @param[in] val_next time dof value at time t_{n+1}
  virtual T patch_dt_linear(const DataType c, 
                            const T &val_prev, 
                            const T &val,
                            const T &val_next) const;

  /// \brief Evaluate quadrative function on complete patch [t_{n-1}, t_n] \cup
  /// [t_n, t_{n+1}]
  /// @param[in] c relative time in active interval, c \in [0,1]
  /// @param[in] val_prev time dof value at time t_{n-1}
  /// @param[in] val time dof value at time t_{n}
  /// @param[in] val_next time dof value at time t_{n+1}
  virtual T patch_quadratic(const DataType c, 
                            const T &val_prev, 
                            const T &val,
                            const T &val_next) const;

  /// \brief Evaluate temporal derivative of quadrative function on complete
  /// patch [t_{n-1}, t_n] \cup [t_n, t_{n+1}]
  /// @param[in] c relative time in active interval, c \in [0,1]
  /// @param[in] val_prev time dof value at time t_{n-1}
  /// @param[in] val time dof value at time t_{n}
  /// @param[in] val_next time dof value at time t_{n+1}
  virtual T patch_dt_quadratic(const DataType c, 
                               const T &val_prev,
                               const T &val, 
                               const T &val_next) const;

  /// \brief Set time step size and active interval
  /// @param[in] dT_pc (t_{n} - t_{n-1})
  /// @param[in] dT_nc (t_{n+1} - t_{n})
  /// @param[in] rel_time 0: active interval is [t_{n-1}, t_n], 1: active
  /// interval is [t_n, t_{n+1}]
  virtual void set_time_steps(DataType dT_pc, 
                              DataType dT_cn, 
                              int rel_time);

  /// \brief compute coefficients for evaluating functions
  virtual void compute_weight_tau_coeff();

  /// \brief clear allocated data
  virtual void clear();

protected:
  /// \brief get absolute time in patch interval
  /// @param[in c relatvie time \in [0,1]
  /// @return absolute time
  DataType get_absolut_time(DataType c) const;

  DataType dT_pc_;
  DataType dT_cn_;
  int rel_time_;

  // coefficients for patchwise quadratic time interpolation for continuous,
  // piecewise linear functions
  DataType a_n2_;
  DataType a_n1_;
  DataType a_c2_;
  DataType a_c1_;
  DataType a_p2_;
  DataType a_p1_;
  DataType a_p0_;

  // coefficients for patchwise linear time interpolation for disconinuous,
  // piecewsie constant functions
  DataType b_n1_;
  DataType b_n0_;
  DataType b_c1_;
  DataType b_c0_;
};


////////////////////////////////////////////////////
/////// Implementation /////////////////////////////
////////////////////////////////////////////////////

template < typename T, class DataType >
T TimePatchInterpolation< T, DataType >::constant(const DataType c,
                                                  const T &val_prev,
                                                  const T &val,
                                                  const T &val_next) const 
{
  switch (this->rel_time_) 
  {
  case 0:
    return val;
    break;
  case 1:
    return val_next;
    break;
  default:
    LOG_ERROR("Invalid relative time!!!");
    exit(-1);
  }
}

template < typename T, class DataType >
T TimePatchInterpolation< T, DataType >::linear(const DataType c,
                                                const T &val_prev, 
                                                const T &val,
                                                const T &val_next) const 
{
  switch (this->rel_time_) 
  {
  case 0:
    return (c * val + (1. - c) * val_prev);
    break;
  case 1:
    return (c * val_next + (1. - c) * val);
    break;
  default:
    LOG_ERROR("Invalid relative time!!!");
    exit(-1);
  }
}

template < typename T, class DataType >
T TimePatchInterpolation< T, DataType >::dt_linear(const DataType c,
                                                   const T &val_prev,
                                                   const T &val,
                                                   const T &val_next) const 
{
  switch (this->rel_time_) 
  {
  case 0:
    return (val - val_prev) * (1. / this->dT_pc_);
    break;
  case 1:
    return (val_next - val) * (1. / this->dT_cn_);
    break;
  default:
    LOG_ERROR("Invalid relative time!!!");
    exit(-1);
  }
}

template < typename T, class DataType >
T TimePatchInterpolation< T, DataType >::patch_linear(const DataType c,
                                                      const T &val_prev,
                                                      const T &val,
                                                      const T &val_next) const 
{
  DataType t = this->get_absolut_time(c);

  return (this->b_c1_ * t + this->b_c0_) * val +
         (this->b_n1_ * t + this->b_n0_) * val_next;
}

template < typename T, class DataType >
T TimePatchInterpolation< T, DataType >::patch_dt_linear(const DataType c, 
                                                         const T &val_prev, 
                                                         const T &val,
                                                         const T &val_next) const 
{
  return (this->b_c1_ * val + this->b_n1_ * val_next);
}

template < typename T, class DataType >
T TimePatchInterpolation< T, DataType >::patch_quadratic(const DataType c, 
                                                         const T &val_prev, 
                                                         const T &val,
                                                         const T &val_next) const 
{
  DataType t = this->get_absolut_time(c);
  DataType tt = t * t;

  return (this->a_n2_ * tt + this->a_n1_ * t) * val_next +
         (this->a_c2_ * tt + this->a_c1_ * t) * val +
         (this->a_p2_ * tt + this->a_p1_ * t + this->a_p0_) * val_prev;
}

template < typename T, class DataType >
T TimePatchInterpolation< T, DataType >::patch_dt_quadratic(const DataType c, 
                                                            const T &val_prev, 
                                                            const T &val,
                                                            const T &val_next) const 
{
  DataType t = this->get_absolut_time(c);

  return (2. * this->a_n2_ * t + this->a_n1_) * val_next +
         (2. * this->a_c2_ * t + this->a_c1_) * val +
         (2. * this->a_p2_ * t + this->a_p1_) * val_prev;
}

template < typename T, class DataType >
void TimePatchInterpolation< T, DataType >::set_time_steps(DataType dT_pc,
                                                           DataType dT_cn,
                                                           int rel_time) 
{
  this->dT_pc_ = dT_pc;
  this->dT_cn_ = dT_cn;
  this->rel_time_ = rel_time;
}

template < typename T, class DataType >
void TimePatchInterpolation< T, DataType >::compute_weight_tau_coeff() 
{
  DataType t1 = this->dT_pc_;
  DataType t2 = this->dT_pc_ + this->dT_cn_;
  DataType alpha = t2 * t2 - t1 * t2;

  this->a_n2_ = 1. / alpha;
  this->a_n1_ = -t1 / alpha;
  this->a_c2_ = -t2 / (t1 * alpha);
  this->a_c1_ = t2 * t2 / (t1 * alpha);   // 1. / t1 + t2 / alpha;
  this->a_p2_ = (t2 - t1) / (t1 * alpha); // t2 / t1 - 1.;
  this->a_p1_ = -(t1 + t2) / (t1 * t2);   //-1. / t1 + t1 / alpha - t2 / alpha;
  this->a_p0_ = 1.;

  this->b_n1_ = 2. / t2;
  this->b_n0_ = -t1 / t2;
  this->b_c1_ = -2. / t2;
  this->b_c0_ = 1. + t1 / t2;
}

template < typename T, class DataType >
DataType TimePatchInterpolation< T, DataType >::get_absolut_time(DataType c) const 
{
  switch (this->rel_time_) {
  case 0:
    return this->dT_pc_ * c;
    break;
  case 1:
    return this->dT_pc_ + c * this->dT_cn_;
    break;
  default:
    LOG_ERROR("Invalid relative time!!!");
    exit(-1);
  }
}

template < typename T, class DataType >
void TimePatchInterpolation< T, DataType >::clear() 
{
  dT_pc_ = 0.;
  dT_cn_ = 0.;
  rel_time_ = 0;

  a_n2_ = 0.;
  a_n1_ = 0.;
  a_c2_ = 0.;
  a_c1_ = 0.;
  a_p2_ = 0.;
  a_p1_ = 0.;
  a_p0_ = 0.;

  b_n1_ = 0.;
  b_n0_ = 0.;
  b_c1_ = 0.;
  b_c0_ = 0.;
}

} // namespace hiflow
#endif
