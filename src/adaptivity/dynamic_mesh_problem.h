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

#ifndef HIFLOW_ADAPTIVITY_DYNAMIC_MESH_PROBLEM
#define HIFLOW_ADAPTIVITY_DYNAMIC_MESH_PROBLEM

/// \author Philipp Gerstner

#include <map>
#include <string>
#include <vector>
#include <boost/function.hpp>
#include <mpi.h>
#include <iterator>

#include "adaptivity/time_mesh.h"
#include "common/log.h"
#include "common/array_tools.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/la_descriptor.h"
#include "space/vector_space.h"

using namespace hiflow::mesh;
using namespace hiflow::la;

namespace hiflow {

///
/// \class  DynamicMeshProblem dynamic_mesh.h
/// \brief base class for problems involvoing a mesh that changes during a time
/// loop
///
///

template < class LAD, int DIM > 
class DynamicMeshProblem 
{
public:
  typedef typename LAD::DataType DataType;
  typedef typename LAD::VectorType VectorType;

  DynamicMeshProblem()
      : active_mesh_index_(-1) 
  {}

  ~DynamicMeshProblem() {}

  /// \brief routine which is called in init_fe_spaces() of DynamicMeshHandler
  /// to setup primal and dual fe space for a given mesh
  /// @param[out] space reference to space to be set up
  /// @param[out] coupling_vars vector containing information of coupled
  /// variables
  /// @param[in] mesh pointer to specific mesh
  /// @param[in] mode 1: primal space, -1: dual space
  virtual void 
  setup_space(VectorSpace< DataType, DIM > &space,
              std::vector< std::vector< bool > > &coupling_vars,
              MeshPtr mesh, 
              int mode) = 0;

  /// \brief routine which is called in udpate() of DynamicMeshHandler to setup
  /// required LA objects for specific fe space in case of primal problem time
  /// loop
  /// @param[in] space reference to primal space
  /// @param[in] space_dual referencde to dual space
  /// @param[in] coupling_vars information about coupled variables in primal
  /// space
  /// @param[in] coupling_vars_dual information about coupled variables in dual
  /// space
  /// @param[out] couplings LA couplings for primal LA objects
  /// @param[out] couplings_dual LA couplings for dual LA objects
  virtual void
  setup_LA_primal(VectorSpace< DataType, DIM > &space,
                  VectorSpace< DataType, DIM > &space_dual,
                  std::vector< std::vector< bool > > &coupling_vars,
                  std::vector< std::vector< bool > > &coupling_vars_dual,
                  BlockManagerSPtr block_manager,
                  BlockManagerSPtr block_manager_dual) = 0;

  /// \brief routine which is called in udpate() of DynamicMeshHandler to setup
  /// required LA objects for specific fe space in case of dual problem time
  /// loop
  /// @param[in] space reference to primal space
  /// @param[in] space_dual referencde to dual space
  /// @param[in] coupling_vars information about coupled variables in primal
  /// space
  /// @param[in] coupling_vars_dual information about coupled variables in dual
  /// space
  virtual void
  setup_LA_dual(VectorSpace< DataType, DIM > &space,
                VectorSpace< DataType, DIM > &space_dual,
                std::vector< std::vector< bool > > &coupling_vars,
                std::vector< std::vector< bool > > &coupling_vars_dual,
                BlockManagerSPtr block_manager,
                BlockManagerSPtr block_manager_dual) = 0;

  /// \brief routine which is called in udpate() of DynamicMeshHandler to setup
  /// required LA objects for specific fe space in case of error estimator
  /// problem time loop
  /// @param[in] space reference to primal space
  /// @param[in] space_dual referencde to dual space
  /// @param[in] coupling_vars information about coupled variables in primal
  /// space
  /// @param[in] coupling_vars_dual information about coupled variables in dual
  /// space
  virtual void
  setup_LA_est(VectorSpace< DataType, DIM > &space,
               VectorSpace< DataType, DIM > &space_dual,
               std::vector< std::vector< bool > > &coupling_vars,
               std::vector< std::vector< bool > > &coupling_vars_dual,
               BlockManagerSPtr block_manager,
               BlockManagerSPtr block_manager_dual) = 0;

  /// \brief routine which is called in update() of DynamicMeshHandler to set
  /// vector space pointer in application to active space
  /// @paam[in] space pointer to active space object
  /// @paam[in] mode 1: primal space, -1: dual space
  virtual void set_active_space(VectorSpaceSPtr< DataType, DIM > space, int mode) = 0;

  /// \brief routine which is called in update() of DynamicMeshHandler to set
  /// mesh pointer in application to active mesh
  /// @param[in] mesh pointer to active mesh
  virtual void set_active_mesh(MeshPtr mesh) = 0;

  /// \brief set index of active mesh
  /// @param[in] index active index

  void set_active_mesh_index(int index) 
  {
    this->active_mesh_index_ = index;
  }

  int active_mesh_index() 
  { 
    return this->active_mesh_index_; 
  }
  
  virtual void adapt_boundary(MeshPtr mesh)
  {
  };

  virtual bool need_higher_order_interpolation() const 
  { 
    return false; 
  }

  TimeMesh< DataType > const * time_mesh() 
  {
    return &(this->t_mesh_);
  }
  
  // for debugging purpose
  virtual void visualize_function(VectorType &sol, int time_step) 
  {
  }

  virtual void visualize_function(const VectorType &sol,
                                  VectorSpaceSPtr< DataType, DIM > space,
                                  std::string const &prefix,
                                  std::string const &path_vtu,
                                  std::string const &path_pvtu, int time_step,
                                  std::vector< std::string > &var_names) 
  {
  }
  
protected:
  int active_mesh_index_;

  TimeMesh< DataType > t_mesh_;

};

} // namespace hiflow

#endif
