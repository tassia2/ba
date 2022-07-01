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

#ifndef _DOF_DOF_CONTAINER_H_
#define _DOF_DOF_CONTAINER_H_

#include <map>
#include <vector>

#include "common/vector_algebra.h"
#include "dof/dof_fem_types.h"


namespace hiflow {

namespace mesh {
  class Entity;
  class MasterSlave;
  class CellType;
}

namespace doffem {

template <class DataType, int DIM> class RefCell;
template <class DataType, int DIM> class DofFunctional;
template <class DataType, int DIM> class CellTransformation;
template <class DataType, int DIM> class FunctionSpace;
template <class DataType, int DIM> class RefCellFunction;

/// Abstract base class for collectoin of Dof functionals that define a specfic nodal basis
/// \author Philipp Gerstner

template < class DataType, int DIM > 
class DofContainer
{
public:

  /// Constructor
  DofContainer(CRefCellSPtr<DataType, DIM> ref_cell)
  : initialized_(false),
  type_ (DofContainerType::NOT_SET),
  ref_cell_(ref_cell)
  {}

  /// Destructor
  virtual ~DofContainer();

  size_t tdim() const;
  
  /// Total number of dofs on the cell
  inline size_t nb_dof_on_cell() const
  {
    return this->dim_;
  }

  /// Number of subentities
  size_t nb_subentity(int tdim) const;

  /// Number of dofs on a subentity
  inline size_t nb_dof_on_subentity(int tdim, int index) const
  {
    assert(tdim >= 0 && tdim < ref_cell_->tdim());
    assert(index >= 0 && index < dof_on_subentity_[tdim].size());
    return dof_on_subentity_[tdim][index].size();
  }

  RefCellType ref_cell_type() const;

  inline CRefCellSPtr<DataType, DIM> ref_cell() const
  {
    return this->ref_cell_;
  }

  inline DofFunctional<DataType, DIM> const * get_dof (size_t i) const
  {
    assert (i < this->dofs_.size());
    return this->dofs_[i];
  }
  
  inline std::string name() const 
  {
    return this->name_;
  }
  
  inline DofContainerType type() const 
  {
    return this->type_;
  }
  
  /// Get information about the cDofIds on a specific subentity
  std::vector< cDofId > const &get_dof_on_subentity(int tdim, int index) const;

  /// add dof functional to container
  void push_back (DofFunctional<DataType, DIM> * dof);
  
  /// reset data structures
  virtual void clear();
  
  /// Operators needed to be able to create maps where DofContainer is
  /// used as key. \see FEInterfacePattern::operator < (const
  /// FEInterfacePattern& test) const Comparison by protected variable my_id_
  /// and fe_deg_ .
  virtual bool operator==(const DofContainer<DataType, DIM> &dof_slave) const;

  /// Operators needed to be able to create maps where DofContainer is
  /// used as key. \see FEInterfacePattern::operator < (const
  /// FEInterfacePattern& test) const Comparison by protected variable my_id_
  /// and fe_deg_ .
  virtual bool operator<(const DofContainer<DataType, DIM> &dof_slave) const;

  // TODO: avoid code duplication
  /// evaluate specified functional for all basis functions contained in space
  /// It is assumed that the functions in space are defined on the reference cell, which is the same as in DofContainer.
  /// Moreover, no FE transformation is applied to those functions before putting them into the DofFunctional.
  virtual void evaluate (FunctionSpace<DataType, DIM> const * space, 
                         const std::vector< cDofId > & dof_ids, 
                         std::vector< std::vector<DataType> >& dof_values ) const = 0;

  /// Evaluate a specified set of DofFunctionals for a given function that lives on the reference cell.
  /// Data structure of dof_values: dof_i <-> i-th entry of dof_ids, func_j <-> j-th function in func
  /// [dof_0 ( func_0 ), ... , dof_0 (func_m), dof_1 (func_0), ... , dof_1 (func_m), .....]
  virtual void evaluate (RefCellFunction<DataType, DIM> const * func, 
                         const std::vector< cDofId > & dof_ids, 
                         std::vector< std::vector<DataType> >& dof_values ) const = 0;
 
protected:
  /// initialize dof_on_subentity information
  virtual void init ();

  /// filter the DoF points for the cell's faces, edges or vertices
  virtual void init_dofs_on_subentities();

  std::vector< DofFunctional<DataType, DIM> * > dofs_;

  /// Storing an instance of the reference cell
  CRefCellSPtr<DataType, DIM> ref_cell_;
  
  /// DoF ID stored for specific subentities (point, line, face)
  std::vector< std::vector< std::vector< cDofId > > > dof_on_subentity_;
  
  /// number of dof functionals in container
  size_t dim_;
  
  /// topological dimension of underlying reference cell
  size_t tdim_;

  bool initialized_;

  DofContainerType type_;
  
  std::string name_;
};

} // namespace doffem
} // namespace hiflow
#endif
