// Copyright (C) 2011-2015 Vincent Heuveline
//
// HiFlow3 is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// HiFlow3 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with HiFlow3.  If not, see <http://www.gnu.org/licenses/>.

#ifndef HIFLOW_PDF_H_
#define HIFLOW_PDF_H_

/// \file pc_pdf.h
/// \brief Basic class for working with arbitrary probability density functions.
///
/// \author Kevin Schlecht, Jonathan Schwegler

#include "quadrature/quadrature.h"
#include <map>
#include <vector>

namespace hiflow {
namespace polynomialchaos {

template < class DataType > class PDF {
public:
  /// Constructor
  PDF();

  /// Destructor
  virtual ~PDF() {}

  /// Restricts the pdf to interval [a,b]
  void SetRestriction(DataType a, DataType b);

  /// Return if the pdf is restricted
  bool IsRestricted() const;

  ///// Returns begin and end of restriction interval for a restricted pdf
  // std::vector<DataType> GetRestriction();

  // Set integration interval for unrestricted integration
  void SetIntegrationInterval(DataType c, DataType d);

  void SetNumIntegrationIntervals(int num_int_intervals);

  /// Sets the number of quadrature points for calculate normalization constant
  void SetQuadPoints(int order);

  ///// Returns the number of quadrature points for calculate normalization
  /// constant
  // int GetNumQuadPoints();

  /// evaluate the pdf
  DataType operator()(DataType x) const;

  DataType GetMoment(int moment);

  /// Calculates Normalization constant for restricted distributions
  void Normalize();

  void GetIntegrationInterval(DataType *c, DataType *d) {
    *c = c_;
    *d = d_;
  }

protected:
  Quadrature< double > gauss_quad_;

  // returns value of initalized pdf at x
  virtual DataType evaluate_nonnormalized(DataType x) const = 0;

  /// Begin and end of restriction interval
  DataType a_, b_;

  /// Begin and end of integration interval
  DataType c_, d_;

  std::map< int, DataType > moments_;

  /// Normalization constant for restricted pdf
  DataType norm_const_;

  /// Number of quadrature points for calculate normalization constant
  /// (Default=30)
  int num_quad_points_;

  int num_int_intervals_;

  std::vector< DataType > quad_points_;
  std::vector< DataType > quad_weights_;
};

} // namespace polynomialchaos
} // namespace hiflow

#endif
