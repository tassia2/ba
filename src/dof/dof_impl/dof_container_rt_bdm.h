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

#ifndef _DOF_DOF_CONTAINER_RT_BDM_H_
#define _DOF_DOF_CONTAINER_RT_BDM_H_

#include <map>
#include <vector>
#include "common/vector_algebra_descriptor.h"
#include "dof/dof_fem_types.h"
#include "dof/dof_impl/dof_container.h"
#include "quadrature/quadrature.h"


namespace hiflow {
namespace doffem {

template <class DataType, int DIM> class RefCell;
template <class DataType, int DIM> class DofFunctional;
template <class DataType, int DIM> class CellTransformation;
template <class DataType, int DIM> class AnsatzSpace;
template <class DataType, int DIM> class AnsatzSpaceSum;


/// Predefined collection of point evaluation dof functionals for Lagrange elements 
/// on given reference cell
/// \author Philipp Gerstner

template < class DataType, int DIM > 
class DofContainerRTBDM : public virtual DofContainer<DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using SCoord = typename StaticLA<DIM-1, DIM, DataType>::ColVectorType;

  DofContainerRTBDM(CRefCellSPtr<DataType, DIM> ref_cell)
  : DofContainer<DataType, DIM>(ref_cell), 
  nb_facets_(0)
  {
    this->ref_facet_ = nullptr;
    this->facet_test_space_ = nullptr;
    this->cell_test_space_ = nullptr;
    this->cell_test_space_1_ = nullptr;
    this->cell_test_space_2_ = nullptr;
  }

  virtual ~DofContainerRTBDM();

  // TODO: avoid code duplication
  void evaluate (FunctionSpace<DataType, DIM> const * space, 
                 const std::vector< cDofId > & dof_ids, 
                 std::vector< std::vector<DataType> >& dof_values ) const;

  void evaluate (RefCellFunction<DataType, DIM> const * func, 
                 const std::vector< cDofId > & dof_ids, 
                 std::vector< std::vector<DataType> >& dof_values ) const;
  
  /// initialize container for given reference cell and polynomial degree of ansatz space
  void init (size_t degree, DofContainerType type);

  void clear();

  int cell_quad_size() const 
  {
    return this->qc_x_.size();
  }
  
  int facet_quad_size() const 
  {
    return this->qf_x_.size();
  }
  
  const Quadrature<DataType>& cell_quadrature() const
  {
    return this->cell_quad_;
  }
  
  const Quadrature<DataType>& facet_quadrature() const
  {
    return this->facet_quad_;
  }
  
  const Coord& cell_quad_point(int q) const 
  {
    assert (q >= 0);
    assert (q < this->qc_x_.size());
    return this->qc_x_[q];
  }
  
  const SCoord& facet_quad_point(int q) const 
  {
    assert (q >= 0);
    assert (q < this->qf_x_.size());
    return this->qf_x_[q];
  }
  
private:
  // Test space for cell moments (sum of space_1 and space_2)
  AnsatzSpaceSPtr<DataType, DIM> cell_test_space_;
  AnsatzSpaceSPtr<DataType, DIM> cell_test_space_1_;
  AnsatzSpaceSPtr<DataType, DIM> cell_test_space_2_;

  // Test space for facet moments
  AnsatzSpaceSPtr<DataType, DIM-1> facet_test_space_;

  // Cell quadrature
  Quadrature<DataType> cell_quad_;

  // Facet quadrature
  Quadrature<DataType> facet_quad_;

  // quad points on cell
  std::vector< Coord > qc_x_;
  
  // quad points on facet
  std::vector< SCoord > qf_x_;

  // quad weights on cell
  std::vector< DataType > qc_w_;
  
  // quad weights on facet
  std::vector< DataType > qf_w_;

  // surface integration elements
  std::vector< std::vector< DataType> > ds_;
  
  // values of all cell test functions at all cell quad points
  std::vector< std::vector< DataType> > test_vals_c_;
  mutable std::vector< std::vector< DataType> > trial_vals_c_;

  // values of all cell test functions at all facet quad points on reference facet
  std::vector< std::vector< DataType> > test_vals_f_;
  // values of all cell test functions at all facet quad points on all facets
  mutable std::vector< std::vector< std::vector< DataType> > >trial_vals_f_;
  
  // polynomial degree
  size_t degree_;

  size_t nb_facets_;

  RefCellSPtr<DataType, DIM-1> ref_facet_;
};

} // namespace doffem
} // namespace hiflow
#endif
