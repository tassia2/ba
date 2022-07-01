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

#ifndef _DOF_NUMBERING_STRATEGY_H_
#define _DOF_NUMBERING_STRATEGY_H_

#include <map>
#include <vector>

#include "common/vector_algebra_descriptor.h"
#include "dof/dof_fem_types.h"

namespace hiflow {

namespace mesh {
class Mesh;
class Entity;
}

namespace doffem {
template < class DataType > class DofInterpolation;
template < class DataType, int DIM > class FEManager;
template < class DataType, int DIM > class DofPartition;


/// \author Michael Schick<br>Martin Baumann<br>Simon Gawlok

template < class DataType, int DIM > 
class NumberingStrategy {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Constructor

  NumberingStrategy()
  : ordering_fe_cell_(true) 
  {}
  
  /// Destructor
  virtual ~NumberingStrategy()
  {};

  /// Initialization. Here, the critical member variables of DofPartitionLocal are
  /// being given to NumberingStrategy, such that the implementation class of
  /// the function void number() can calculate all neccessary information and
  /// store it in these variables
  void initialize(DofPartition< DataType, DIM > &dof);

  /// Kernel of numbering strategy. Here the user can specify in an inherited
  /// class his wishes for some numbering procedure, when dealing for example
  /// with varying finite element types
  /// @param[in] order Ordering strategy for DoFs.
  virtual void number_locally(DOF_ORDERING order,
                              bool ordering_fe_cell) = 0;

  /// Helper function which permutes data within the interpolation container and
  /// numer_ field
  void apply_permutation(const std::vector< gDofId > &permutation, const std::string & = "");

protected:
  /// Topological dimension
  int tdim_;
  /// Total number of variables
  size_t nb_fe_;

  /// The DofPartitionLocal class used throughout the numeration procedure
  DofPartition< DataType, DIM > *dof_;

  /// Const pointer to mesh
  const mesh::Mesh *mesh_;

  /// FEManager on the given mesh
  FEManager< DataType, DIM > const *fe_manager_;

  /// Holds DoF IDs, needed for cell2global
  std::vector< gDofId > *numer_cell_2_global_;
  std::vector< std::vector<DataType> > *cell_2_dof_factor_;

  /// Offset container for numer_cell_2_global_, needed for cell2global
  std::vector< std::vector< gDofId > > *numer_cell_2_global_offsets_;
  //std::vector< int > *numer_cell_2_global_offsets_per_cell_;

  /// Interpolation Container, which stores the interpolation weights
  DofInterpolation<DataType> *dof_interpolation_;

  /// Update local_nb_dofs_total_ and local_nb_dofs_for_fe_
  //void update_number_of_dofs(const std::string & = "");

  /// Total number of dofs for all variables
  lDofId local_nb_dofs_total_;

  /// Total number of dofs per variable
  std::vector< lDofId > local_nb_dofs_for_fe_;

  /// Get vertex points of mesh entity.
  void get_points(const mesh::Entity &entity, std::vector< Coord > &points);
  
  /// if true, numbering: 
  /// loop fe's
  ///   loop cell's
  
  /// if false, numbering:
  /// loop cell's
  ///   loop fe's
    
  bool ordering_fe_cell_;
};



} // namespace doffem
} // namespace hiflow

#endif
