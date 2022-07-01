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

#ifndef __FEM_FEINSTANCE_H_
#define __FEM_FEINSTANCE_H_


#include <vector>
#include "common/vector_algebra.h"
#include "dof/dof_fem_types.h"
#include "mesh/cell_type.h"
#include "mesh/periodicity_tools.h"

namespace hiflow {
namespace doffem {

template <class DataType, int DIM> class RefElement;
template <class DataType, int DIM> class RefCell;
template <class DataType, int DIM> class CellTransformation;
template <class DataType, int DIM> class FETransformation;
template <class DataType, int DIM> class AnsatzSpace;
template <class DataType, int DIM> class DofContainer;

///
/// \class FEInstance feinstance.h
/// \brief Holds instances of different FE ansatz functions
/// \author Philipp Gerstner
///

template < class DataType, int DIM > 
class FEInstance 
{
public:

  /// Default Constructor
  FEInstance()
  :lagrange_only_(true),
  nb_nonlin_trafos_(0)
  {}
  
  /// Default Destructor
  ~FEInstance()
  {
    this->clear();
  }

  /// add a new finite element which is defined by its type fe_type, the topological type of the underlying cell and a set of parameters
  /// @return fe_instande_id
  size_t add_fe ( FEType fe_type, mesh::CellType::Tag topo_cell_type, const std::vector< int > &param );

  CRefElementSPtr<DataType, DIM> get_fe (size_t fe_id) const ;
  
  /// create appropriate cell transformation object for Fintie element of given instance id and the align number of the underlying physical cell
  /// @return pointer to newly created cell_trafo object. This object still has to be initialized by passing the physical cell vertices
  //CellTrafoSPtr<DataType, DIM> create_cell_trafo (size_t fe_id, int align_number) const;
  CellTrafoSPtr<DataType, DIM> create_cell_trafo (size_t fe_id, 
                                                 std::vector<DataType> coord_vtx,
						                                     const mesh::Entity &cell,
                                                 const std::vector< mesh::MasterSlave >& period) const;                                                 
  void clear();
  
  inline size_t nb_fe_inst() const
  {
    return this->ref_elements_.size();
  }
  
  inline bool contain_only_lagrange_fe() const
  {
    return this->lagrange_only_;
  }

  FEConformity max_fe_conformity(size_t fe_id) const 
  {
    assert (fe_id < this->max_fe_conform_.size());
    return this->max_fe_conform_[fe_id];
  }
  
  int nb_nonlinear_trafos() const 
  {
    return this->nb_nonlin_trafos_;
  }
  
private:
  std::vector< CRefElementSPtr<DataType, DIM> > ref_elements_;
  std::vector< CAnsatzSpaceSPtr<DataType, DIM> > ansatz_spaces_;
  std::vector< CDofContainerSPtr<DataType, DIM> > dof_containers_;
  std::vector< CFETrafoSPtr<DataType, DIM> > fe_trafos_;
  std::vector< CRefCellSPtr<DataType, DIM> > ref_cells_;
  std::vector< FEConformity > max_fe_conform_;
    
  std::vector< FEType > added_fe_types_;
  std::vector< mesh::CellType::Tag > added_cell_types_;
  std::vector< std::vector<int> > added_params_;

  bool lagrange_only_;
  
  mutable int nb_nonlin_trafos_;

};

} // namespace doffem
} // namespace hiflow
#endif
