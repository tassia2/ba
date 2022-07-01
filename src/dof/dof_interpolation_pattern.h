// Copyright (C) 2011-2021 Vincent Heuveline
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

#ifndef _DOF_DOF_INTERPOLATION_PATTERN_H_
#define _DOF_DOF_INTERPOLATION_PATTERN_H_

#include <vector>

#include "common/macros.h"
#include "dof/dof_fem_types.h"
#include "dof/dof_interpolation.h"

namespace hiflow {
namespace doffem {

///
/// \class  DofInterpolationPattern dof_interpolation_pattern.h
/// \brief  Interpolation of DoFs for interface patterns due to hanging nodes
/// (h- or p-refinement). \author Michael Schick<br>Martin Baumann
///
/// This class is used to represent the interpolation and identification
/// between DoF on the pattern interface.
/// DoF identifications are stored in
/// DoFInterpolation::dof_identification_list(), where for each entry first is
/// the slave DoF and second the master DoF.
///
/// \see FEInterfacePattern
///
template <class DataType>
class DofInterpolationPattern {
public:
  DofInterpolationPattern() : num_slaves_(0) {}

  ~DofInterpolationPattern() {}

  typedef typename DofInterpolation<DataType>::iterator iterator;
  typedef typename DofInterpolation<DataType>::const_iterator const_iterator;

  void set_number_slaves(int i);

  /// insert interpolation or identification definition for master, weights
  /// should be non-zero
  void insert_interpolation_master(
      std::pair< int, std::vector< std::pair< int, DataType > > >);

  /// insert interpolation or identification definition for slave 's', weights
  /// should be non-zero
  void insert_interpolation_slave(
      int s, std::pair< int, std::vector< std::pair< int, DataType > > >);

  /// Get the interpolation information of master

  DofInterpolation<DataType> const &interpolation_master() const { return ic_master_; }

  /// Get the interpolation information of cell 'i'

  DofInterpolation<DataType> const &interpolation_slave(int i) const {
    return ic_slave_[i];
  }

  /// overloaded out stream operator
  template <class T>
  friend std::ostream &operator<<(std::ostream &, const DofInterpolationPattern<T> &);

private:
  DofInterpolation<DataType> ic_master_;

  std::vector< DofInterpolation<DataType> > ic_slave_;

  unsigned int num_slaves_;
};

} // namespace doffem
} // namespace hiflow
#endif
