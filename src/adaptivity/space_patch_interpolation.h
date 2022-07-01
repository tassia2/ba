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

#ifndef HIFLOW_ADAPTIVITY_SPACE_PATCH_INTERPOLATION
#define HIFLOW_ADAPTIVITY_SPACE_PATCH_INTERPOLATION

/// \author Philipp Gerstner

#include <map>
#include <string>
#include <vector>
#include <boost/function.hpp>
#include <mpi.h>

#include "common/array_tools.h"
#include "common/log.h"
#include "common/vector_algebra.h"
#include "fem/fe_reference.h"
#include "linear_algebra/la_descriptor.h"
#include "mesh/entity.h"
#include "mesh/mesh.h"
#include "mesh/mesh_pXest.h"
#include "mesh/types.h"
#include "space/vector_space.h"
#include "space/fe_interpolation_map.h"

using namespace hiflow::mesh;
using namespace hiflow::la;

namespace hiflow {
///
/// \class SpacePatchInterpolation patch_interpolation.h
/// \brief class for interpolating FE functions from one space to another one
/// with a coarser mesh and higher polynomial degree
///

template < class LAD, int DIM > 
class SpacePatchInterpolation 
{
  typedef typename LAD::DataType DataType;
  typedef typename LAD::VectorType VectorType;
  typedef Vec<DIM, DataType> Coord;

public:
  SpacePatchInterpolation();

  ~SpacePatchInterpolation() 
  { 
    this->clear(); 
  }

  /// \brief Initialize patch interpolation, this function involves the
  /// follwoing steps: <br> check whether input space is suitable for patching
  /// <br> copy mesh of input space and uniformly coarsen it <br> setup
  /// interpolating space <br> build dof interpoaltion map from interpolating
  /// space to input space
  /// @param[in] input_space space to be patched
  virtual void init(CVectorSpaceSPtr< DataType, DIM > input_space);

  /// \brief apply patch interpolation
  /// @param[in] input_vector vector to be interpoalted
  /// @param[out] vector interpolated vector
  virtual void interpolate(const VectorType &input_vector, VectorType &vector) const;

  /// \brief get patch interpolating vector space
  /// @return reference to space

  virtual VectorSpaceSPtr< DataType, DIM > get_patch_space() 
  { 
    return this->space_; 
  }

  /// \brief clear all data structs
  virtual void clear();

protected:
  /// \brief checks if input space is suitable for patch interpolation
  virtual bool check_space() const;

  /// \brief copy input mesh
  virtual void copy_mesh();

  /// \brief compute ghost cells of interpolating mesh
  virtual void compute_ghost();

  /// pointers to input space and input mesh
  CVectorSpaceSPtr< DataType, DIM > input_space_;
  MeshPtr input_mesh_;

  /// interpolating space
  VectorSpaceSPtr< DataType, DIM > space_;

  /// patched mesh
  MeshPtr mesh_;

  FeInterMapFullNodal<LAD, DIM> fe_mapping_;

  /// FE degrees of interpolating space
  std::vector< int > degrees_;

  /// CG / DG flags for interpolating space
  std::vector< bool > is_dg_;

  /// FE ansatz spaces
  std::vector< doffem::FEType > fe_ansatz_;
  
  /// number of variables in input space
  int nb_fe_;

  /// flag indicating whether patchinterpolation is initialized for given space
  bool initialized_;

  /// MPI rank
  int rank_;

  /// topological dimension of input space
  int tdim_;

  /// geometrical dimension of input space
  int gdim_;
};

////////////////////////////////////////////////////
/////// Implementation /////////////////////////////
////////////////////////////////////////////////////

template < class LAD, int DIM >
SpacePatchInterpolation< LAD, DIM >::SpacePatchInterpolation()
    : nb_fe_(0), initialized_(false), mesh_(nullptr),
      input_mesh_(nullptr), input_space_(nullptr), tdim_(0), gdim_(0), rank_(0) 
{
  this->degrees_.clear();
  this->is_dg_.clear();
  this->fe_ansatz_.clear();
}

template < class LAD, int DIM >
void SpacePatchInterpolation< LAD, DIM >::init(CVectorSpaceSPtr< DataType, DIM > input_space) 
{
  this->clear();
  this->input_mesh_ = input_space->meshPtr();
  this->tdim_ = input_mesh_->tdim();
  this->gdim_ = input_mesh_->gdim();
  this->input_space_ = input_space;
  this->nb_fe_ = input_space->nb_fe();

  assert(this->input_mesh_ != nullptr);

  const MPI_Comm &comm = input_space->get_mpi_comm();

  MPI_Comm_rank(comm, &this->rank_);

  // check if patch interpolation is possible for given space
  bool valid_space = this->check_space();
  if (!valid_space) 
  {
    LOG_DEBUG(0, " SpacePatchInterpolation: input space is not valid ! ");
    quit_program();
  }

  // get fe degrees of input space
  this->degrees_.resize(this->nb_fe_, 0);
  this->is_dg_.resize(this->nb_fe_, false);
  this->fe_ansatz_.resize(this->nb_fe_);
  
  for (int v = 0; v < this->nb_fe_; ++v) 
  {
    this->is_dg_[v] = input_space->fe_manager().is_dg(v);
    this->degrees_[v] = (input_space->fe_manager().get_fe(0, v)->max_deg()) * 2;
    this->fe_ansatz_[v] = input_space->fe_manager().fe_type(0, v);
  }

  // create coarse mesh
  LOG_DEBUG(1, " copy mesh ");
  this->copy_mesh();

  // coarsen mesh uniformly
  LOG_DEBUG(1, " coarsen mesh ");
  assert(this->input_mesh_->is_uniformly_coarsenable());
  std::vector< int > refs(this->mesh_->num_entities(this->tdim_), -1);
  this->mesh_ = this->mesh_->refine(refs);
  this->compute_ghost();

  // initialize interpolating space
  LOG_DEBUG(1, " init patch space ");
  this->space_.reset();
  this->space_ = VectorSpaceSPtr<DataType, DIM> (new VectorSpace<DataType, DIM> ());
  this->space_->Init(*this->mesh_, this->fe_ansatz_, this->is_dg_, this->degrees_);

  // create DOF mapping from interpolating space to input space
  LOG_DEBUG(0, " init fe mapping ");
  std::vector<size_t> in_fe_ind;
  std::vector<size_t> out_fe_ind;
  number_range<size_t>(0, 1, this->space_->nb_fe(), in_fe_ind);
  number_range<size_t>(0, 1, this->space_->nb_fe(), out_fe_ind);
     
  this->fe_mapping_.init(input_space.get(), this->space_.get(), true, in_fe_ind, out_fe_ind);

  this->initialized_ = true;
}

template < class LAD, int DIM >
bool SpacePatchInterpolation< LAD, DIM >::check_space() const 
{
  // check if mesh is uniformly coarsenable
  if (!this->input_mesh_->is_uniformly_coarsenable()) 
  {
    LOG_DEBUG(0, "Mesh is not unifromly coarsenable ");
    return false;
  }

  // check if fe functions have same degree on every cell
  for (int v = 0; v < this->nb_fe_; ++v) 
  {
    int ref_deg = this->input_space_->fe_manager().get_fe(0, v)->max_deg();
    for (EntityNumber jc = 1; jc < this->input_mesh_->num_entities(this->tdim_); ++jc) 
    {
      if (this->input_space_->fe_manager()
              .get_fe(jc, v)->max_deg() != ref_deg) 
      {
        LOG_DEBUG(0, "Non-uniform FE degree ");
        return false;
      }
    }
  }
  return true;
}

template < class LAD, int DIM >
void SpacePatchInterpolation< LAD, DIM >::interpolate(const VectorType &input_vector, 
                                                      VectorType &vector) const 
{
  if (!this->initialized_) 
  {
    std::cout << "SpacePatchInterpolation is not initialized ! " << std::endl;
    quit_program();
  }

  this->fe_mapping_.interpolate(input_vector, vector);
}

template < class LAD, int DIM >
void SpacePatchInterpolation< LAD, DIM >::copy_mesh() 
{
  if (this->input_mesh_->mesh_impl() == mesh::IMPL_P4EST) 
  {
    this->mesh_ = new MeshPXest(this->tdim_, this->gdim_);
  } 
  else 
  {
    std::cout << "SpacePatchInterpolation::copy_mesh() does only work for P4est mesh" << std::endl;
    quit_program();
  }
  this->mesh_->deep_copy_from(this->input_mesh_);
}

template < class LAD, int DIM >
void SpacePatchInterpolation< LAD, DIM >::compute_ghost() 
{
  SharedVertexTable shared_verts;
  const MPI_Comm &comm = this->input_space_->get_mpi_comm();
  this->mesh_ = compute_ghost_cells(*this->mesh_, 
                                    comm, 
                                    shared_verts, 
                                    this->mesh_->mesh_impl(),
                                    this->input_mesh_->get_ghost_layer_width());
}

template < class LAD, int DIM >
void SpacePatchInterpolation< LAD, DIM >::clear() 
{
  this->degrees_.clear();
  this->is_dg_.clear();
  this->fe_ansatz_.clear();
  this->tdim_ = 0;
  this->gdim_ = 0;
  this->input_mesh_ = nullptr;
  this->input_space_.reset();
  this->rank_ = 0;

  // clear space
  this->space_.reset();

  // TODO delete mesh?
  this->mesh_.reset();

  this->initialized_ = false;
}

} // namespace hiflow
#endif
