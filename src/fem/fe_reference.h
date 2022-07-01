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

#ifndef __FEM_FINITE_ELEMENT_H_
#define __FEM_FINITE_ELEMENT_H_

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"
#include "common/log.h"
#include "dof/dof_fem_types.h"
#include "fem/function_space.h"
#include "fem/ansatz/ansatz_space.h"

namespace hiflow {

namespace la {
template <class DataType> class SeqDenseMatrix;
}

namespace doffem {

//template <class DataType, int DIM> class AnsatzSpace;
template <class DataType, int DIM> class DofContainer;
template <class DataType, int DIM> class FETransformation;
template <class DataType, int DIM> class RefCell;

///
/// \class RefElement finite_element.h
/// \brief Ancestor class of different Finite Elements. <br>
/// \details This class provides the functionality for evaluating FE basis functions on the
/// underlying reference cell. It has to be initialized with an AnsatzSpace object and a 
/// DoFCollection object. Then, a matrix is computed that transforms the basis provided by AnsatzSpace
/// to a nodal dof basis {\phi_i}_i that satisfies dof_j (\phi_i) = \delta_{i,j}. 
/// \author Philipp Gerstner
///

template < class DataType, int DIM > 
class RefElement final : virtual public FunctionSpace<DataType, DIM>
{
public:  
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  typedef std::function< void (const Coord &, std::vector<DataType> &) > BasisEvalFunction;

  /// Default Constructor
  RefElement();

  /// Default Destructor
  virtual ~RefElement();

  /// Initialize the Finite Element
  void init (const CAnsatzSpaceSPtr<DataType, DIM>& space, 
             const CDofContainerSPtr<DataType, DIM>& dofs,
             const CFETrafoSPtr<DataType, DIM>& fe_trafo,
             bool is_nodal_basis,
             FEType fe_type);

  /// Get information if this finite element was initialized 
  inline bool init_status() const;

  /// Get instance Id of the Finite Element
  inline int instance_id() const;

  /// Setting Id of the Finite Element (from FEInstance)
  inline void set_instance_id(int Id);

  inline CDofContainerSPtr<DataType, DIM> dof_container() const;

  inline CAnsatzSpaceSPtr<DataType, DIM> ansatz_space () const;

  inline CFETrafoSPtr<DataType, DIM> fe_trafo() const;

  /// Total number of dofs on the cell
  size_t nb_dof_on_cell() const;

  /// Number of subentities
  size_t nb_subentity(int tdim) const;

  /// Number of dofs on a subentity
  size_t nb_dof_on_subentity(int tdim, int index) const;

  /// Get information about the cDofIds on a specific subentity
  std::vector< cDofId > const &get_dof_on_subentity(int tdim, int index) const;
  
  /// Get ID of Finite Element, which is stored in enum RefElement
  inline FEType type() const;

  /// Operators needed to be able to create maps where RefElement is
  /// used as key. \see FEInterfacePattern::operator < (const
  /// FEInterfacePattern& test) const Comparison by protected variable my_id_
  /// and fe_deg_ .
  bool operator==(const RefElement &fe_slave) const;

  bool operator!=(const RefElement &fe_slave) const
  {
    return !(*this == fe_slave);
  }

  /// Operators needed to be able to create maps where RefElement is
  /// used as key. \see FEInterfacePattern::operator < (const
  /// FEInterfacePattern& test) const Comparison by protected variable my_id_
  /// and fe_deg_ .
  bool operator<(const RefElement &fe_slave) const;

  /// Index ordering from basis index i and component index var to position in weight vector
  inline size_t iv2ind(size_t i, size_t var) const
  {
    assert (this->space_ != nullptr);
    assert (var * this->dim_ + i == this->space_->iv2ind(i, var));
    return var * this->dim_ + i;
  }
   
  /// apply basis transformation
  void apply_transformation (const std::vector< DataType > &weight_in, std::vector< DataType > &weight_out) const;

  /// For given coordinates, get values and derivatives of all shapefunctions on
  /// reference cell.  <br>
  /// Use function iv2ind for indexing 
  /// @param[in] pt coordinate in reference cell
  /// @param[out] values of shape functions at pt
  void N(const Coord &pt, std::vector< DataType > &weight) const override;

  void N_x(const Coord &pt, std::vector<DataType> &weight) const override;
  void N_y(const Coord &pt, std::vector<DataType> &weight) const override;
  void N_z(const Coord &pt, std::vector<DataType> &weight) const override;

  void N_xx(const Coord &pt, std::vector<DataType> &weight) const override;
  void N_xy(const Coord &pt, std::vector<DataType> &weight) const override;
  void N_xz(const Coord &pt, std::vector<DataType> &weight) const override;
  void N_yy(const Coord &pt, std::vector<DataType> &weight) const override;
  void N_yz(const Coord &pt, std::vector<DataType> &weight) const override;
  void N_zz(const Coord &pt, std::vector<DataType> &weight) const override;
  
protected:
  void evaluate_and_transform(BasisEvalFunction fun, const Coord &pt, std::vector< DataType > &weight) const;

  virtual void compute_degree_hash () const;

  /// compute the transformation matrices for switching between basis defined by space object
  /// and nodal dof basis
  void compute_basis_transformation_matrix();

  /// Pointer to underlying finite dimensional ansatz space
  CAnsatzSpaceSPtr<DataType, DIM> space_;
  
  /// Pointers to dof functional defining the FE basis functions
  CDofContainerSPtr<DataType, DIM> dofs_; 

  /// Type of transformation of element from reference cell to physical cell
  CFETrafoSPtr<DataType, DIM> fe_trafo_;

  /// basis transformation matrix: space-basis to nodal dof-basis
  SharedPtr <la::SeqDenseMatrix<DataType> > V_inv_;

  /// basis transformation matrix: nodal dof-basis to space-basis 
  SharedPtr <la::SeqDenseMatrix<DataType> > V_;

  /// Id set by fe_instance
  int instance_id_;

  // type of finite element
  FEType type_;
  
  // set flag wether basis transformation is needed
  bool is_nodal_basis_;
  
  mutable std::vector<DataType> weight_psi_;
  
};


template < class DataType, int DIM >
inline FEType RefElement< DataType, DIM >::type() const 
{
  return type_;
}

template < class DataType, int DIM > 
inline int RefElement< DataType, DIM >::instance_id() const 
{
  return instance_id_;
}

template < class DataType, int DIM > 
inline void RefElement< DataType, DIM >::set_instance_id( int id ) 
{
  this->instance_id_ = id;
}

template < class DataType, int DIM > 
inline CDofContainerSPtr<DataType, DIM> RefElement< DataType, DIM >::dof_container() const
{
  return this->dofs_;
}

template < class DataType, int DIM > 
inline CAnsatzSpaceSPtr<DataType, DIM> RefElement< DataType, DIM >::ansatz_space () const
{
  return this->space_;
}

template < class DataType, int DIM >
inline CFETrafoSPtr<DataType, DIM> RefElement< DataType, DIM >::fe_trafo() const 
{
  return this->fe_trafo_;
}


} // namespace doffem
} // namespace hiflow
#endif
