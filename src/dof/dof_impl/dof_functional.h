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

#ifndef _DOF_DOF_FUNCTIONAL_H_
#define _DOF_DOF_FUNCTIONAL_H_

#include <map>
#include <vector>
#include <cassert>
#include <cstddef>
#include "dof/dof_fem_types.h"

namespace hiflow {
namespace doffem {

template <class DataType, int DIM> class RefCell;
template <class DataType, int DIM> class FunctionSpace;
template <class DataType, int DIM> class RefCellFunction;

/// Abstract base class for Dof functional
/// \author Philipp Gerstner

template < class DataType, int DIM > 
class DofFunctional 
{
public:

  /// Constructor
  DofFunctional()
  : type_(DofFunctionalType::NOT_SET)
  {
  }

  /// Destructor
  virtual ~DofFunctional()
  {}

  DofFunctionalType type() const 
  {
    return this->type_;
  }
  
  std::vector<int> get_subentities(size_t tdim) const
  {
    assert (tdim < this->attached_to_subentity_.size());
    return this->attached_to_subentity_[tdim];
  }
  
  /*
  virtual bool operator==(const DofFunctional<DataType, DIM> &dof_slave) const
  {
    return false;
  }
  */
  
protected:
  // Evaluate functions may only be called by a DofContainer
  /// evaluate dof functional for all basis functions defined in space, space has to live on the reference cell this->ref_cell
  virtual void evaluate (FunctionSpace<DataType, DIM> const * space,
                         std::vector<DataType>& dof_values ) const = 0;

  /// evaluate dof functional for function defined by Functor.
  /// this functions has to be defined on the reference cell this->ref_cell_
  virtual void evaluate (RefCellFunction<DataType, DIM> const * func, 
                         size_t offset, 
                         std::vector<DataType>& dof_values) const = 0;
  
  /// Type of Dof Functional
  DofFunctionalType type_;

  /// Pointer to underlying reference cell
  CRefCellSPtr<DataType, DIM> ref_cell_;

  /// attached_to_subentity[tdim] = list of all subentity numbers, to which this dof functional is attached
  std::vector< std::vector<int> > attached_to_subentity_;
};

} // namespace doffem
} // namespace hiflow
#endif
