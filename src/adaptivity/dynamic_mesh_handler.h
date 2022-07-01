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

#ifndef HIFLOW_ADAPTIVITY_DYNAMIC_MESH_HANDLER
#define HIFLOW_ADAPTIVITY_DYNAMIC_MESH_HANDLER

/// \author Philipp Gerstner

#include <map>
#include <string>
#include <vector>
#include <boost/function.hpp>
#include <mpi.h>
#include <iterator>

#include "adaptivity/time_mesh.h"
#include "adaptivity/dynamic_mesh_problem.h"
#include "common/log.h"
#include "common/array_tools.h"
#include "common/ioassistant.h"
#include "common/timer.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/la_descriptor.h"
#include "mesh/mesh.h"
#include "mesh/mesh_db_view.h"
#include "mesh/mesh_pXest.h"
#include "mesh/mesh_tools.h"
#include "mesh/writer.h"
#include "space/vector_space.h"
#include "space/space_tools.h"

using namespace hiflow::mesh;
using namespace hiflow::la;

namespace hiflow {

///
/// \class DynamicMeshHandler dynamic_mesh.h
/// \brief Class that manages mesh transfer operations in a time stepping loop
///
///

enum MeshAdaptType
{
  MESH_ADAPT_NONE = 0,
  MESH_ADAPT_FIXED = 1,
  MESH_ADAPT_FRACTION = 2,
  MESH_ADAPT_ERROR = 3 
};

template < class LAD, int DIM, class FEInterpolator > 
class DynamicMeshHandler 
{
  typedef typename LAD::DataType DataType;
  typedef typename LAD::VectorType VectorType;
  typedef mesh::Id CellId;
  typedef mesh::EntityNumber CellIndex;
  

public:
  DynamicMeshHandler();

  ~DynamicMeshHandler() 
  { 
    this->clear(); 
  }
  

  /// \brief set type of mesh change adaption mode
  /// @param[in] type "None": single mesh, "Fixed": mesh changes according to
  /// initial mesh change times, \br "FixedFraction": change mesh at those
  /// (fixed number of ) times steps with largest difference in subsequent error
  /// indicators \br "FixedError": change mesh if difference of subsequent error
  /// indicators is above a certain threshold
  void init (const MPI_Comm &comm,
             DynamicMeshProblem< LAD, DIM > *problem,
             int adapt_ctr,
             MeshAdaptType adapt_type,
             bool read_initial_mesh_list,
             IMPL mesh_impl,
             int ghost_layer_width,
             const std::string& filename_mesh_change_in,
             const std::string& filename_mesh_change_out,
             std::vector< VectorType * > p_vectors,
             std::vector< VectorType * > d_p_vectors,
             std::vector< VectorType * > d_vectors);

  void clear();

  /// \brief set base mesh
  /// @param[in] mesh pointer to space mesh

  void set_initial_mesh(MeshPtr mesh) 
  {
    assert(mesh != 0);
    this->initial_mesh_ = mesh;
    this->tdim_ = mesh->tdim();
    this->gdim_ = mesh->gdim();

    this->fill_mesh_list();
  }
  
  ////////////////////////////////////////////////
  ///////////// Use in time loop /////////////////
  /// \brief update mesh, vector spaces and LA in case of mesh change
  /// @param[in] time_step  index of considered time step
  /// @param[in] mode 1: primal, -1: dual, 0: error estimation
  /// @param[in] initial_call indicates whether functions is called for the
  /// first time
  /// @return true if mesh was updated
  bool update(int time_step, int mode, bool initial_call);

  bool update(int time_step, std::vector<int>& markers); 
                                                            
  /// \brief refine mesh specified by index
  /// @param[in] mesh_index index of considered mesh
  /// @param[in] markers flags for mesh refinement
  void refine(int mesh_index, std::vector< int > &markers);

  /// \brief initialize fe spaces for all meshes in mesh_list
  void init_fe_spaces();

  /// \brief interpolate vector between different fe spaces
  /// @param[in] in_vec input vector for interpolation
  /// @param[in] in_index index of input space
  /// @param[in] out_vec output vector for interpolation
  /// @param[in] out_index index of output space
  void interpolate_vector(const VectorType &in_vec, 
                          int in_index,
                          VectorType &out_vec, 
                          int out_index);

  /// \brief load specific mesh
  /// @param[in] adapt_counter iteration index of outer adaption loop
  /// @param[in] num_mesh number of meshes to load from file
  /// @param[in] prefix path to saved mesh files
  void load_mesh_from_file(int adapt_counter, 
                           int num_mesh,
                           std::string &prefix);

  /// \brief save specific mesh
  /// @param[in] adapt_counter iteration index of outer adaption loop
  /// @param[in] prefix path to save mesh files
  void save_mesh_to_file(int adapt_counter, std::string &prefix);

  /// \brief visualize specific mesh
  /// @param[in] adapt_counter iteration index of outer adaption loop
  /// @param[in] prefix path to save visualization files
  void visualize_mesh(int adapt_counter, std::string &prefix);

  /// \brief add points when to change the mesh
  /// @param[in] additional changes time points to add for mesh change
  /// @param[in,out] indicator_mesh_indices map: indicator-time-step to mesh
  /// index
  void add_mesh_change_times(const std::vector< DataType > &additional_changes,
                             std::vector< int > &indicator_mesh_indices);

    /// \brief routine which selects point for changing the mesh according to
  /// spatial error indicators and passes to the dmh object
  /// @param[in] adapt_type type of mesh change adaption, see set_adapt_type()
  /// of DynamiMeshHandler
  /// @param[in] adapt_counter iteration index of outer adaption loop
  /// @param[in] min_steps_for_mesh minimum number of time steps between two
  /// successive mesh change points
  /// @param[in] max_mesh_number maximum number of mesh change points
  /// @param[in] fixed_fraction_rate number of newly created mesh changes in
  /// case of adapt_type = FixedFraction
  /// @param[in] fixed_error_tol threshold when to crate mesh change point in
  /// case of adapt_type = Fixed Error
  /// @param[in] space_indicator spatial error indicators
  /// @param[in,out] indicator_mesh_indices map indicaotr[t] to mesh index
  void adapt_mesh_change_list(MeshAdaptType adapt_type, 
                              int adapt_counter, 
                              int min_steps_for_mesh,
                              int max_mesh_number, 
                              int fixed_fraction_rate,
                              DataType fixed_error_tol,
                              const std::vector< std::vector< DataType > > &space_indicator,
                              std::vector< int > &indicator_mesh_indices);
    
  /// \brief write out list of mesh chang times
  /// @param[in] path path to mesh change list
  void write_mesh_change_list(const std::string &path) const;
  void write_mesh_change_list(int ac) const;

  /// \brief read in list of mesh chang times
  /// @param[in] path path to mesh change list
  void read_mesh_change_list(const std::string &path);
  void read_mesh_change_list(int ac);
  
  ////////////////////////////////////////////////////
  ////////////// get functions ///////////////////////
  /// \brief get mesh change points
  /// @return points

  std::vector< DataType > get_mesh_change_times() const {
    return this->mesh_change_times_.data();
  }

  /// \brief get mesh change points
  /// @return indices of points

  std::vector< int > get_mesh_change_steps() {
    this->update_mesh_change_steps();
    return this->mesh_change_steps_.data();
  }

  /// \brief get index of active mesh
  /// @return index

  inline int get_active_mesh_index() 
  { 
    return this->active_mesh_index_; 
  }

  /// \brief get pointer to vector space corresponding to specified mesh
  /// @param[in] time_step  index of considered time step
  /// @param[in] mode 1: primal -1: dual
  /// @return pointer to vector space
  VectorSpaceSPtr< DataType, DIM > get_space_by_step(int time_step, int mode) const;

  /// \brief get pointer to vector space corresponding to specified mesh
  /// @param[in] mesh_index index of considered mesh
  /// @param[in] mode 1: primal -1: dual
  /// @return pointer to vector space
  VectorSpaceSPtr< DataType, DIM > get_space_by_index(int mesh_index, int mode) const;

  /// \brief get pointer to vector space corresponding to active mesh
  /// @param[in] mode 1: primal -1: dual
  /// @return pointer to vector space

  VectorSpaceSPtr< DataType, DIM > get_active_space(int mode) const 
  {
    return this->get_space_by_index(this->active_mesh_index_, mode);
  }

  /// \brief get pointer to mesh for specific time step
  /// @param[in] time_step  index of considered time step
  /// @return mesh pointer
  MeshPtr get_mesh_by_step(int time_step) const;

  /// \brief get pointer to mesh for specific index
  /// @param[in] mesh_index index of considered mesh
  /// @return mesh pointer
  MeshPtr get_mesh_by_index(int mesh_index) const;

  inline std::vector< MeshPtr > get_mesh_list() const 
  {
    return this->mesh_list_;
  }

  /// \brief get pointer to active mesh
  /// @return mesh pointer

  inline MeshPtr get_active_mesh() const 
  { 
    return this->mesh_; 
  }

  /// \brief get LA couplings for active mesh
  /// @param[in] mode 1: primal -1: dual
  /// @return couplings

  BlockManagerSPtr get_active_block_manager(int mode) const {
    return this->get_block_manager_by_index(this->active_mesh_index_, mode);
  }

  /// \brief get LA couplings for specified mesh index
  /// @param[in] mesh_index index of considered mesh
  /// @param[in] mode 1: primal -1: dual
  /// @return couplings
  BlockManagerSPtr get_block_manager_by_index(int mesh_index, int mode) const;

  /// \brief get number of different meshes
  /// @return number

  inline int num_mesh() const 
  { 
    return this->mesh_list_.size(); 
  }

  /// \brief get number of mesh change points
  /// @return number

  inline int num_mesh_change() const 
  {
    return this->mesh_change_times_.size();
  }

  /// \brief get index of mesh corresponding to specified time point
  /// @param[in] time point in time
  /// @return mesh index
  int mesh_index(DataType time) const;

  /// \brief get index of mesh corresponding to specified time point index
  /// @param[in] time_step  index of considered time step
  /// @return mesh index
  int mesh_index(int time_step) const;

  /// \brief get index of first time point of specified mesh
  /// @param[in] mesh_index index of considered mesh
  /// @return time step
  int first_step_for_mesh(int mesh_index) const;

  /// \brief get index of last time point of specified mesh
  /// @param[in] mesh_index index of considered mesh
  /// @return time step
  int last_step_for_mesh(int mesh_index) const;

  /// \brief get number of cells in mesh specified by time step
  /// @param[in] time_step  index of considered time step
  /// @return number of cells
  int num_cells_by_step(int time_step) const;

  /// \brief get number of cells in mesh specified by index
  /// @param[in] mesh_index index of considered mesh
  /// @return number of cells
  int num_cells_by_index(int mesh_index) const;

  /// \brief check if mesh changes in given interval
  /// @param[in] low_step first time step
  /// @param[in] high_step last time step
  /// @return bool
  bool need_to_change_mesh_interval(int low_step, int high_step);
      

protected:
  ///////////////////////////////////////////////////
  //////////// setup ////////////////////////////////

  /// \brief
  /// @param[in] mode 1: primal loop, -1: dual loop, 0: error estimation
  /// @param[in] type type of vectors, 1: primal vector, -1: dual vectors
  /// @param[in] vectors pointers to upate vectors
  void set_update_vectors(int mode, 
                          int type,
                          std::vector< VectorType * > vectors);

  /// \brief fill mesh list by copying initial mesh
  void fill_mesh_list();

  /// \brief create fe interpolator objects
  /// @param[in] num_mesh number of meshes
  void create_fe_interpolator(int num_mesh);

  /// \brief update indices of mesh change time points
  void update_mesh_change_steps();

  /// \brief checks whether mesh changes
  /// @param[in] last_time last time point
  /// @param[in] cur_time current time point
  /// @return true if mesh changes between time points
  bool need_to_change_mesh(DataType last_time, DataType cur_time) const;

  /// \brief checks whether mesh changes
  /// @param[in] last_step index of last time point
  /// @param[in] cur_step index of current time point
  /// @return true if mesh changes between time steps
  bool need_to_change_mesh(int last_step, int cur_step) const;

  /// \brief checks whether mesh changes
  /// @param[in] time_step  index of considered time step
  /// @return true if mesh changes at time step
  bool need_to_change_mesh(int time_step) const;

  TimeMesh< DataType > const * time_mesh_;

  int tdim_;

  int gdim_;

  int active_mesh_index_;

  MeshPtr initial_mesh_;

  MeshPtr mesh_;

  ParCom* parcom_;

  int ghost_layer_width_;

  MeshAdaptType mesh_adapt_type_;
  IMPL mesh_impl_;
  
  std::string filename_mesh_change_in_;
  std::string filename_mesh_change_out_;
    
  VectorSpaceSPtr< DataType, DIM > space_primal_;
  VectorSpaceSPtr< DataType, DIM > space_dual_;

  std::vector< MeshPtr > mesh_list_;

  SortedArray< DataType > mesh_change_times_;

  SortedArray< int > mesh_change_steps_;

  std::vector< VectorSpaceSPtr< DataType, DIM > > space_list_primal_;

  std::vector< VectorSpaceSPtr< DataType, DIM > > space_list_dual_;

  std::vector< VectorSpaceSPtr< DataType, DIM > > space_list_tmp_;

  std::vector< std::vector< FEInterpolator * > > fe_interpolator_;

  std::vector< BlockManagerSPtr > block_manager_list_primal_;
  std::vector< BlockManagerSPtr > block_manager_list_dual_;

  std::vector< std::vector< std::vector< bool > > > coupling_vars_list_primal_;
  std::vector< std::vector< std::vector< bool > > > coupling_vars_list_dual_;

  std::vector< VectorType * > vec_primal_primal_;
  std::vector< VectorType * > vec_dual_primal_;
  std::vector< VectorType * > vec_dual_dual_;
  std::vector< VectorType * > vec_est_primal_;
  std::vector< VectorType * > vec_est_dual_;

  DynamicMeshProblem< LAD, DIM > *problem_;
  bool initialized_;
};

////////////////////////////////////////////////
//////// DynamicMeshHandler ////////////////////

template < class LAD, int DIM, class FEInterpolator > 
DynamicMeshHandler< LAD, DIM, FEInterpolator >::DynamicMeshHandler()
    : mesh_adapt_type_(MESH_ADAPT_NONE), 
      problem_(nullptr),
      mesh_(nullptr), 
      initial_mesh_(nullptr),
      initialized_(false),
      time_mesh_(nullptr), 
      space_primal_(nullptr), 
      space_dual_(nullptr),
      parcom_(nullptr), 
      tdim_(DIM), 
      gdim_(DIM),
      active_mesh_index_(-1),
      mesh_impl_(IMPL_NONE)
{
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::clear() 
{
  this->vec_primal_primal_.clear();
  this->vec_dual_primal_.clear();
  this->vec_dual_dual_.clear();
  this->vec_est_primal_.clear();
  this->vec_est_dual_.clear();

  this->mesh_change_times_.clear();
  this->mesh_change_steps_.clear();

  this->mesh_.reset(); // = nullptr;

  this->time_mesh_ = nullptr;
  this->initial_mesh_.reset(); // = nullptr;

  this->mesh_impl_ = IMPL_NONE;

  this->tdim_ = DIM;
  this->gdim_ = DIM;
  this->active_mesh_index_ = -1;
  this->mesh_adapt_type_ = MESH_ADAPT_NONE;
  this->space_primal_.reset();
  this->space_dual_.reset();

  for (int l = 0; l < this->mesh_list_.size(); ++l) {
    this->mesh_list_[l].reset();
  }
  this->mesh_list_.clear();

  for (int l = 0; l < this->space_list_primal_.size(); ++l) {
      this->space_list_primal_[l].reset();
  }
  this->space_list_primal_.clear();

  for (int l = 0; l < this->space_list_dual_.size(); ++l) {
      this->space_list_dual_[l].reset();
  }
  this->space_list_dual_.clear();

  for (int l = 0; l < this->fe_interpolator_.size(); ++l) {
    for (int k = 0; k < this->fe_interpolator_[l].size(); ++k) {
      if (this->fe_interpolator_[l][k] != nullptr) {
        delete this->fe_interpolator_[l][k];
        this->fe_interpolator_[l][k] = nullptr;
      }
    }
  }
  this->fe_interpolator_.clear();

  for (int l = 0; l < this->block_manager_list_primal_.size(); ++l) {
    this->block_manager_list_primal_[l].reset();
  }
  this->block_manager_list_primal_.clear();

  for (int l = 0; l < this->block_manager_list_dual_.size(); ++l) {
    this->block_manager_list_dual_[l].reset();
  }
  
  this->block_manager_list_dual_.clear();

  this->coupling_vars_list_primal_.clear();
  this->coupling_vars_list_dual_.clear();
  this->initialized_ = false;
  
  if (this->parcom_ != nullptr)
  {
    delete this->parcom_;
    this->parcom_ = nullptr;
  }
  this->initialized_ = false;
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::init(const MPI_Comm &comm,
                                                          DynamicMeshProblem< LAD, DIM > *problem,
                                                          int adapt_ctr,
                                                          MeshAdaptType adapt_type,
                                                          bool read_initial_mesh_list,
                                                          IMPL mesh_impl,
                                                          int ghost_layer_width,
                                                          const std::string& filename_mesh_change_in,
                                                          const std::string& filename_mesh_change_out,
                                                          std::vector< VectorType * > p_vectors,
                                                          std::vector< VectorType * > d_p_vectors,
                                                          std::vector< VectorType * > d_vectors)
{
  assert (problem != nullptr);
  assert (comm != MPI_COMM_NULL);
  
  this->clear();
  
  this->problem_ = problem;
  this->time_mesh_ = problem->time_mesh();
  this->filename_mesh_change_in_ = filename_mesh_change_in;
  this->filename_mesh_change_out_ = filename_mesh_change_out;
  this->mesh_impl_ = mesh_impl;
  this->ghost_layer_width_ = ghost_layer_width;
  this->parcom_ = new ParCom (comm);

  // set type mesh change process
  this->mesh_adapt_type_ = adapt_type;

  // set vectors to be updated
  this->set_update_vectors(1, 1, p_vectors);
  this->set_update_vectors(-1, 1, d_p_vectors);
  this->set_update_vectors(-1, -1, d_vectors);
  this->set_update_vectors(0, 1, d_p_vectors);
  this->set_update_vectors(0, -1, d_vectors);

  // set initial mesh change points 
  if (adapt_ctr < 0) 
  {
    //std::string start_type = base_params_["Adaptivity"]["DynamicMesh"]["StartChanges"].template get< std::string >();
    if (read_initial_mesh_list) 
    {
      this->read_mesh_change_list(-1);
    }
    this->write_mesh_change_list(0);
  } 
  else 
  {
    this->read_mesh_change_list(adapt_ctr);
  }
  this->initialized_ = true;
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::read_mesh_change_list(int adapt_counter) 
{
  std::string path;
  if (adapt_counter < 0) 
  {
    path = this->filename_mesh_change_in_;
  } 
  else 
  {
    path = this->filename_mesh_change_out_ + "." + std::to_string(adapt_counter) + ".txt";
  }
  this->read_mesh_change_list(path);
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::read_mesh_change_list(const std::string &path) 
{
  IOAssistant io_assi;
  
  std::vector< std::vector< DataType > > data;
  io_assi.read_array(path, " ", -1, data);
   
  int num_changes = data.size();
  this->mesh_change_times_.clear();
  
  for (int t = 0; t < num_changes; ++t) 
  {
    if (data[t][0] >= 0.) 
    {
      this->mesh_change_times_.insert(data[t][0]);
    }
  }
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::write_mesh_change_list(int adapt_counter) const
{
  std::string path = this->filename_mesh_change_out_ + "." + std::to_string(adapt_counter) + ".txt";
  this->write_mesh_change_list(path);
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::write_mesh_change_list(const std::string &path) const 
{
  if (this->parcom_->rank() == 0)
  {
    std::ofstream out;
    out.open(path.c_str());

    for (int l = 0; l < this->mesh_change_times_.data().size(); ++l) 
    {
      out << this->mesh_change_times_.data().at(l) << "\n";
    }
    out.close();
  }
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::set_update_vectors(int mode, 
                                                                        int type, 
                                                                        std::vector< VectorType * > vectors) 
{
  if (mode == 1) 
  {
    if (type == 1) 
    {
      this->vec_primal_primal_ = vectors;
    }
  } 
  else if (mode == -1) 
  {
    if (type == 1) 
    {
      this->vec_dual_primal_ = vectors;
    } 
    else if (type == -1) 
    {
      this->vec_dual_dual_ = vectors;
    }
  } 
  else if (mode == 0) 
  {
    if (type == 1) 
    {
      this->vec_est_primal_ = vectors;
    } 
    else if (type == -1) 
    {
      this->vec_est_dual_ = vectors;
    }
  }
}

template < class LAD, int DIM, class FEInterpolator >
int DynamicMeshHandler< LAD, DIM, FEInterpolator >::mesh_index(DataType time) const 
{
  if (time < 0.) 
  {
    return -1;
  }
  if (time > this->time_mesh_->end() + 1e-10) 
  {
    return -1;
  }

  int num_points = this->mesh_change_times_.size();
  int cur_point = 0;
  for (int t = 0; t < num_points; ++t) 
  {
    if (t == num_points - 1) 
    {
      if (this->mesh_change_times_[t] <= time) 
      {
        cur_point = t + 1;
        break;
      }
    } 
    else 
    {
      if (time < this->mesh_change_times_.data()[t + 1] &&
          time >= this->mesh_change_times_.data()[t]) 
      {
        cur_point = t + 1;
        break;
      }
    }
  }
  return cur_point;
}

template < class LAD, int DIM, class FEInterpolator >
int DynamicMeshHandler< LAD, DIM, FEInterpolator >::mesh_index(int time_step) const 
{
  if (time_step < 0) 
  {
    return -1;
  }
  if (time_step > this->time_mesh_->num_intervals()) 
  {
    return -1;
  }
  return this->mesh_index(this->time_mesh_->time(time_step));
}

template < class LAD, int DIM, class FEInterpolator >
int DynamicMeshHandler< LAD, DIM, FEInterpolator >::first_step_for_mesh(int mesh_index) const 
{
  assert(mesh_index >= 0);
  assert(mesh_index < this->num_mesh());

  if (mesh_index == 0) {
    return 0;
  }

  DataType mesh_start_time = this->mesh_change_times_[mesh_index - 1];

  for (int t = 0; t < this->time_mesh_->num_intervals(); ++t) {
    if (this->time_mesh_->time(t) >= mesh_start_time) {
      return t;
    }
  }
  return -1;
}

template < class LAD, int DIM, class FEInterpolator >
int DynamicMeshHandler< LAD, DIM, FEInterpolator >::last_step_for_mesh(int mesh_index) const 
{
  assert(mesh_index >= 0);
  assert(mesh_index < this->num_mesh());

  if (mesh_index == this->num_mesh() - 1) {
    return this->time_mesh_->num_intervals() - 1;
  }

  return this->first_step_for_mesh(mesh_index + 1) - 1;
}

template < class LAD, int DIM, class FEInterpolator >
int DynamicMeshHandler< LAD, DIM, FEInterpolator >::num_cells_by_step(int time_step) const 
{
  int mesh_index = this->mesh_index(time_step);
  return this->num_cells_by_index(mesh_index);
}

template < class LAD, int DIM, class FEInterpolator >
int DynamicMeshHandler< LAD, DIM, FEInterpolator >::num_cells_by_index(int mesh_index) const 
{
  if (mesh_index < 0) {
    return 0;
  }
  if (mesh_index >= this->num_mesh()) {
    return 0;
  }

  return this->mesh_list_[mesh_index]->num_entities(DIM);
}

template < class LAD, int DIM, class FEInterpolator >
VectorSpaceSPtr< typename LAD::DataType, DIM > 
DynamicMeshHandler< LAD, DIM, FEInterpolator >::get_space_by_step(int time_step, int mode) const 
{
  if (time_step < 0) 
  {
    return nullptr;
  }
  if (time_step > this->time_mesh_->num_intervals()) 
  {
    return nullptr;
  }

  int mesh_index = this->mesh_index(time_step);

  assert(mesh_index >= 0);
  assert(mesh_index < this->mesh_list_.size());

  if (mode == 1) 
  {
    return this->space_list_primal_[mesh_index];
  } 
  else if (mode == -1) 
  {
    return this->space_list_dual_[mesh_index];
  } 
  else if (mode == 2) 
  {
    return this->space_list_tmp_[mesh_index];
  }
  else
  {
    assert(false);
  }
  return nullptr;
}

template < class LAD, int DIM, class FEInterpolator >
VectorSpaceSPtr< typename LAD::DataType, DIM > 
DynamicMeshHandler< LAD, DIM, FEInterpolator >::get_space_by_index(int mesh_index, int mode) const 
{
  assert(mesh_index >= 0);
  assert(mesh_index < this->num_mesh());

  if (mode == 1) 
  {
    return this->space_list_primal_[mesh_index];
  } 
  else if (mode == -1) 
  {
    return this->space_list_dual_[mesh_index];
  } 
  else if (mode == 2) 
  {
    return this->space_list_tmp_[mesh_index];
  }
  else
  {
    assert(false);
  }
  return nullptr;
}

template < class LAD, int DIM, class FEInterpolator >
BlockManagerSPtr
DynamicMeshHandler< LAD, DIM, FEInterpolator >::get_block_manager_by_index(int mesh_index, int mode) const 
{
  assert(mesh_index >= 0);
  assert(mesh_index < this->num_mesh());

  if (mode == 1) 
  {
    return this->block_manager_list_primal_[mesh_index];
  } 
  else if (mode == -1) 
  {
    return this->block_manager_list_dual_[mesh_index];
  }
  else
  {
    assert(false);
  }
  return 0;
}

template < class LAD, int DIM, class FEInterpolator >
MeshPtr DynamicMeshHandler< LAD, DIM, FEInterpolator >::get_mesh_by_step( int time_step) const 
{
  if (time_step < 0) 
  {
    return nullptr;
  }
  if (time_step > this->time_mesh_->num_intervals()) 
  {
    return nullptr;
  }
  int mesh_index = this->mesh_index(time_step);

  return this->get_mesh_by_index(mesh_index);
}

template < class LAD, int DIM, class FEInterpolator >
MeshPtr DynamicMeshHandler< LAD, DIM, FEInterpolator >::get_mesh_by_index(int mesh_index) const 
{
  assert(mesh_index >= 0);
  assert(mesh_index < this->num_mesh());

  return this->mesh_list_[mesh_index];
}

template < class LAD, int DIM, class FEInterpolator >
bool DynamicMeshHandler< LAD, DIM, FEInterpolator >::need_to_change_mesh(DataType last_time, DataType cur_time) const 
{
  int cur_mesh_index = this->mesh_index(cur_time);
  int last_mesh_index = this->mesh_index(last_time);

  if (cur_mesh_index == last_mesh_index) 
  {
    return false;
  }
  return true;
}

template < class LAD, int DIM, class FEInterpolator >
bool DynamicMeshHandler< LAD, DIM, FEInterpolator >::need_to_change_mesh(int last_step, int cur_step) const 
{
  if (last_step < 0) 
  {
    return true;
  }

  DataType cur_time = this->time_mesh_->time(cur_step);
  DataType last_time = this->time_mesh_->time(last_step);

  return this->need_to_change_mesh(last_time, cur_time);
}

template < class LAD, int DIM, class FEInterpolator >
bool DynamicMeshHandler< LAD, DIM, FEInterpolator >::need_to_change_mesh(int step) const 
{
  int mesh_index = this->mesh_index(step);
  if (mesh_index != this->active_mesh_index_) {
    return true;
  }
  return false;
}

template < class LAD, int DIM, class FEInterpolator >
bool DynamicMeshHandler< LAD, DIM, FEInterpolator >::need_to_change_mesh_interval(int low_step, int high_step) 
{
  this->update_mesh_change_steps();
  for (int t = low_step; t < high_step; ++t) 
  {
    if (this->need_to_change_mesh(t, t + 1)) 
    {
      return true;
    }
  }
  return false;
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::update_mesh_change_steps() 
{
  this->mesh_change_steps_.clear();
  int old_mesh = -1;
  int new_mesh = -1;
  for (int t = 0; t < this->time_mesh_->num_intervals(); ++t) 
  {
    old_mesh = new_mesh;
    new_mesh = this->mesh_index(t);
    if (new_mesh >= 0) 
    {
      if (old_mesh != new_mesh) 
      {
        this->mesh_change_steps_.insert(t);
      }
    }
  }
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::fill_mesh_list() 
{
  int num_mesh = this->mesh_change_times_.size() + 1;
  this->mesh_list_.clear();
  this->mesh_list_.resize(num_mesh);
  for (int l = 0; l < num_mesh; ++l) 
  {
    if (this->mesh_impl_ == mesh::IMPL_P4EST) 
    {
      this->mesh_list_[l] = new MeshPXest(this->tdim_, this->gdim_);
    } 
    else if (this->mesh_impl_ == mesh::IMPL_DBVIEW) 
    {
      this->mesh_list_[l] = new MeshDbView(this->tdim_, this->gdim_);
    }

    this->mesh_list_[l]->deep_copy_from(this->initial_mesh_);
  }
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::init_fe_spaces() 
{
  int num_mesh = this->mesh_list_.size();
  LOG_INFO(1, "Init FE spaces for " << num_mesh << " different meshes");

  int num_space = this->space_list_primal_.size();
  int num_space_dual = this->space_list_dual_.size();

  this->space_list_tmp_.clear();
  for (int l = 0; l < num_space; ++l) {
      this->space_list_primal_[l].reset();
  }

  for (int l = 0; l < num_space_dual; ++l) {
    this->space_list_dual_[l].reset();
  }

  for (int l = 0; l < this->block_manager_list_primal_.size(); ++l) {
    this->block_manager_list_primal_[l].reset();
  }
  
  for (int l = 0; l < this->block_manager_list_dual_.size(); ++l) {
    this->block_manager_list_dual_[l].reset();
  }

  this->space_list_primal_.clear();
  this->space_list_dual_.clear();
  this->coupling_vars_list_primal_.clear();
  this->coupling_vars_list_dual_.clear();
  this->block_manager_list_primal_.clear();
  this->block_manager_list_dual_.clear();

  this->space_list_primal_.resize(num_mesh);
  this->space_list_dual_.resize(num_mesh);
  this->coupling_vars_list_primal_.resize(num_mesh);
  this->coupling_vars_list_dual_.resize(num_mesh);
  this->block_manager_list_primal_.resize(num_mesh);
  this->block_manager_list_dual_.resize(num_mesh);

  for (int l = 0; l < num_mesh; ++l) 
  {
    this->space_list_primal_[l] = VectorSpaceSPtr<DataType, DIM>(new VectorSpace< DataType, DIM >());
    this->space_list_dual_[l] = VectorSpaceSPtr<DataType, DIM>(new VectorSpace< DataType, DIM >());

    this->block_manager_list_primal_[l] = BlockManagerSPtr(new BlockManager());
    this->block_manager_list_dual_[l] = BlockManagerSPtr(new BlockManager());

    this->problem_->setup_space(*this->space_list_primal_[l],
                                this->coupling_vars_list_primal_[l],
                                this->mesh_list_[l], 1);
                                
    this->problem_->setup_space(*this->space_list_dual_[l],
                                this->coupling_vars_list_dual_[l],
                                this->mesh_list_[l], -1);
  }

  this->space_list_tmp_ = this->space_list_primal_;
  this->create_fe_interpolator(num_mesh);

  LOG_DEBUG(1, "Mesh change times "
                   << string_from_range(this->mesh_change_times_.data().begin(),
                                        this->mesh_change_times_.data().end()));
  
  for (int m = 0; m < this->time_mesh_->num_levels(); ++m) 
  {
    // LOG_DEBUG(1, "Time mesh " << m << " has points: " <<
    // string_from_range(this->time_mesh_->get_all_times(m).begin(),
    // this->time_mesh_->get_all_times(m).end()));
  }
}

template < class LAD, int DIM, class FEInterpolator >
bool DynamicMeshHandler< LAD, DIM, FEInterpolator >::update(int time_step, 
                                                            int mode,
                                                            bool initial_call) 
{
  if (!this->need_to_change_mesh(time_step) && !initial_call) 
  {
    LOG_INFO("DMH", "No Mesh change needed for time step " << time_step);
    return false;
  }

  // mode: 1 = primal, -1 = dual, 0 = estimation
  //int pp_mesh_index = this->mesh_index(time_step - 2);
  //int p_mesh_index = this->mesh_index(time_step - 1);
  //int n_mesh_index = this->mesh_index(time_step + 1);
  int c_mesh_index = this->mesh_index(time_step);

  int old_mesh_index = this->active_mesh_index_;
  int new_mesh_index = c_mesh_index;

  LOG_INFO("DMH", "Change from mesh " << this->active_mesh_index_ << " to mesh " << new_mesh_index);

  // 1. save previous and (in case of dual problem ) next solution
  std::vector< VectorType * > p_vectors;
  std::vector< VectorType * > d_vectors;

  if (!initial_call) 
  {
    if (mode == 1) 
    {
      p_vectors.resize(this->vec_primal_primal_.size());
      for (int l = 0; l < p_vectors.size(); ++l) 
      {
        p_vectors[l] = new VectorType;
        assert (this->vec_primal_primal_[l] != nullptr);
        p_vectors[l]->CloneFrom(*this->vec_primal_primal_[l]);
        p_vectors[l]->Update();
        interpolate_constrained_vector< DataType, DIM >(*this->get_active_space(1), *p_vectors[l]);
        p_vectors[l]->Update();
      }
    }
    if (mode == -1) 
    {
      p_vectors.resize(this->vec_dual_primal_.size());
      for (int l = 0; l < p_vectors.size(); ++l) 
      {
        p_vectors[l] = new VectorType;
        assert (this->vec_dual_primal_[l] != nullptr);
        p_vectors[l]->CloneFrom(*this->vec_dual_primal_[l]);
        p_vectors[l]->Update();
        interpolate_constrained_vector< DataType, DIM >(*this->get_active_space(1), *p_vectors[l]);
        p_vectors[l]->Update();
      }

      d_vectors.resize(this->vec_dual_dual_.size());
      for (int l = 0; l < d_vectors.size(); ++l) 
      {
        d_vectors[l] = new VectorType;
        assert (this->vec_dual_dual_[l] != nullptr);
        d_vectors[l]->CloneFrom(*this->vec_dual_dual_[l]);
        d_vectors[l]->Update();
        interpolate_constrained_vector< DataType, DIM >(*this->get_active_space(-1), *d_vectors[l]);
        d_vectors[l]->Update();
      }
    }
    if (mode == 0) 
    {
      p_vectors.resize(this->vec_est_primal_.size());
      for (int l = 0; l < p_vectors.size(); ++l) 
      {
        p_vectors[l] = new VectorType;
        assert (this->vec_est_primal_[l] != nullptr);
        p_vectors[l]->CloneFrom(*this->vec_est_primal_[l]);
        p_vectors[l]->Update();
        interpolate_constrained_vector< DataType, DIM >(*this->get_active_space(1), *p_vectors[l]);
        p_vectors[l]->Update();
      }

      d_vectors.resize(this->vec_est_dual_.size());
      for (int l = 0; l < d_vectors.size(); ++l) 
      {
        d_vectors[l] = new VectorType;
        assert (this->vec_est_dual_[l] != nullptr);
        d_vectors[l]->CloneFrom(*this->vec_est_dual_[l]);
        d_vectors[l]->Update();
        interpolate_constrained_vector< DataType, DIM >(*this->get_active_space(-1), *d_vectors[l]);
        d_vectors[l]->Update();
      }
    }
  }

  // 2. change pointers
  this->mesh_ = this->get_mesh_by_index(c_mesh_index);
  this->space_primal_ = this->get_space_by_index(c_mesh_index, 1);
  this->space_dual_ = this->get_space_by_index(c_mesh_index, -1);
  this->active_mesh_index_ = c_mesh_index;
  this->problem_->set_active_space(this->space_primal_, 1);
  this->problem_->set_active_space(this->space_dual_, -1);
  this->problem_->set_active_mesh(this->mesh_);
  this->problem_->set_active_mesh_index(c_mesh_index);

  int num_global_cells = this->space_primal_->meshPtr()
                             ->num_global_cells(this->space_primal_->get_mpi_comm());


  LOG_INFO("DMH", "  Active space has " << this->space_primal_->nb_dofs_global()
                                     << " total dofs and " << num_global_cells << " cells ");

  // 3. reinit LA objects
  if (mode == 1) 
  {
    this->problem_->setup_LA_primal(
        *this->space_primal_, *this->space_dual_,
        this->coupling_vars_list_primal_[c_mesh_index],
        this->coupling_vars_list_dual_[c_mesh_index],
        this->block_manager_list_primal_[c_mesh_index],
        this->block_manager_list_dual_[c_mesh_index]);
  } 
  else if (mode == -1) 
  {
    this->problem_->setup_LA_dual(
        *this->space_primal_, *this->space_dual_,
        this->coupling_vars_list_primal_[c_mesh_index],
        this->coupling_vars_list_dual_[c_mesh_index],
        this->block_manager_list_primal_[c_mesh_index],
        this->block_manager_list_dual_[c_mesh_index]);
  } 
  else if (mode == 0) 
  {
    this->problem_->setup_LA_est(*this->space_primal_, *this->space_dual_,
                                 this->coupling_vars_list_primal_[c_mesh_index],
                                 this->coupling_vars_list_dual_[c_mesh_index],
                                 this->block_manager_list_primal_[c_mesh_index],
                                 this->block_manager_list_dual_[c_mesh_index]);
  }

  if (initial_call) 
  {
    return true;
  }

  // 4. interpolate old vectors w.r.t. new space
  if (mode == 1) 
  {
    for (int l = 0; l < p_vectors.size(); ++l) 
    {
      LOG_DEBUG(2, "  Interpolate primal vector "
                << l << " from mesh " << old_mesh_index << " to mesh " << new_mesh_index);
      this->interpolate_vector(*p_vectors[l], old_mesh_index, *this->vec_primal_primal_[l], new_mesh_index);
    }
  } 
  else if (mode == -1) 
  {
    for (int l = 0; l < p_vectors.size(); ++l) 
    {
      LOG_DEBUG(2, "  Interpolate primal vector "
                << l << " from mesh " << old_mesh_index << " to mesh " << new_mesh_index);
      this->interpolate_vector(*p_vectors[l], old_mesh_index, *this->vec_dual_primal_[l], new_mesh_index);
    }
    for (int l = 0; l < d_vectors.size(); ++l) 
    {
      LOG_DEBUG(2, "  Interpolate dual vector " << l << " from mesh "
                << old_mesh_index << " to mesh " << new_mesh_index);
      this->interpolate_vector(*d_vectors[l], old_mesh_index, *this->vec_dual_dual_[l], new_mesh_index);
    }
  } 
  else if (mode == 0) 
  {
    for (int l = 0; l < p_vectors.size(); ++l) 
    {
      LOG_DEBUG(2, "  Interpolate primal vector "
                << l << " from mesh " << old_mesh_index << " to mesh " << new_mesh_index);
      this->interpolate_vector(*p_vectors[l], old_mesh_index,
                               *this->vec_est_primal_[l], new_mesh_index);
    }
    for (int l = 0; l < d_vectors.size(); ++l) 
    {
      LOG_DEBUG(2, "  Interpolate dual vector " << l << " from mesh "
                << old_mesh_index << " to mesh " << new_mesh_index);
      this->interpolate_vector(*d_vectors[l], old_mesh_index, *this->vec_est_dual_[l], new_mesh_index);
    }
    /*this->prepare_higher_order_space();*/
  }

  for (int l = 0; l < p_vectors.size(); ++l) 
  {
    delete p_vectors[l];
  }
  for (int l = 0; l < d_vectors.size(); ++l) 
  {
    delete d_vectors[l];
  }
  return true;
}

template < class LAD, int DIM, class FEInterpolator >
bool DynamicMeshHandler< LAD, DIM, FEInterpolator >::update(int time_step, 
                                                            std::vector<int>& markers) 
{  
  int loc_size = markers.size();
  int glob_size = 0;
  MPI_Allreduce(&loc_size, &glob_size, 1, MPI_INT, MPI_SUM, this->parcom_->comm());
  
  if (markers.size() == 0) 
  {
    assert (glob_size == 0);
    LOG_INFO("DMH", "No Mesh adaption needed for time step " << time_step);
    return false;
  }

  int loc_sum_refine_markers = 0;
  int loc_sum_markers = 0;
  for (int i=0; i<markers.size(); ++i)
  {
    loc_sum_refine_markers += std::max(0, markers[i]);
    loc_sum_markers += std::abs(markers[i]);
  }
  
  int glob_sum_refine_markers = 0;
  MPI_Allreduce(&loc_sum_refine_markers, &glob_sum_refine_markers, 1, MPI_INT, MPI_SUM, this->parcom_->comm());
  
  int glob_sum_markers = 0;
  MPI_Allreduce(&loc_sum_markers, &glob_sum_markers, 1, MPI_INT, MPI_SUM, this->parcom_->comm());

  if (glob_sum_markers == 0)
  {
    LOG_INFO("DMH", "No Mesh adaption needed for time step " << time_step);
    return false;
  }

  if (glob_sum_refine_markers == 0)
  {
    LOG_INFO("DMH", "No Mesh refinement needed for time step " << time_step);

    if (this->mesh_impl_ == mesh::IMPL_P4EST) 
    {
      MeshPtr cur_mesh = this->get_mesh_by_index(this->active_mesh_index_);

      boost::intrusive_ptr< MeshPXest > mesh_pXest =
          boost::static_pointer_cast< MeshPXest >(cur_mesh);

      LOG_INFO("DMH", "check for coarsening ");
      std::cout << "check mesh" << std::endl;
      bool change_mesh = mesh_pXest->check_for_coarsening(markers);

      int global_change = 0;
      int local_change = change_mesh;
      std::cout << "done" << std::endl;
      this->parcom_->sum(local_change, global_change);
      if (global_change == 0)
      {
        LOG_INFO("DMH", "No Mesh coarsening needed for time step " << time_step);
        return false;
      }
    }
  }

  LOG_INFO("DMH", "Mesh adaption needed for time step " << time_step);
  
  // this kind of adaption only works for a single mesh, i.e. no DWR run with predefined meshes
  assert (this->mesh_list_.size() == 1);  
  assert (this->active_mesh_index_ == 0);
  int c_mesh_index = this->active_mesh_index_;
  
  // 1. save previous solution
  PLOG_INFO(parcom_->rank(), "DMH", "save old vectors");
  std::vector< VectorType * > p_vectors;
  p_vectors.resize(this->vec_primal_primal_.size());
  for (int l = 0; l < p_vectors.size(); ++l) 
  {
    p_vectors[l] = new VectorType;
    assert (this->vec_primal_primal_[l] != nullptr);
    
    p_vectors[l]->CloneFrom(*this->vec_primal_primal_[l]);
    p_vectors[l]->Update();
    interpolate_constrained_vector< DataType, DIM >(*this->get_active_space(1), *p_vectors[l]);
    p_vectors[l]->Update();
  }

  // 2. create adapted mesh
  PLOG_INFO(parcom_->rank(), "DMH", "adapt mesh");
  MeshPtr old_mesh_with_ghost = this->get_mesh_by_index(c_mesh_index);
  assert(old_mesh_with_ghost != 0);

  bool finished_adaption = false;
  const int tdim = old_mesh_with_ghost->tdim();

  while (!finished_adaption)
  {
    // build map: cell_id -> refinement marker
    std::map<CellId, int> parent_id_2_marker;
    CellIndex num_cell = markers.size();
    int max_ref = -1;
    for (CellIndex c = 0; c!=num_cell; ++c)
    {
      CellId id = old_mesh_with_ghost->get_id(tdim, c);
      const int flag = markers[c];

      if (flag > max_ref)
      {
        max_ref = flag;
      }
      
      if (flag <= 0)
      {
        parent_id_2_marker[id] = 0;
      }
      else 
      {
        parent_id_2_marker[id] = flag - 1;
      }
    }

    bool my_finished = false;
    if (max_ref < 2)
    {
      my_finished = true;
    }
    this->parcom_->global_and(my_finished, finished_adaption);

    // refine mesh and compute ghost cells
    if (this->mesh_impl_ == mesh::IMPL_P4EST) 
    {
      boost::intrusive_ptr< MeshPXest > mesh_pXest =
          boost::static_pointer_cast< MeshPXest >(old_mesh_with_ghost);
          
      // TODO: lower connection mode possible?
      mesh_pXest->set_patch_mode(false);
      mesh_pXest->set_connection_mode(2);
    }

    SharedVertexTable shared_verts;
    MeshPtr new_mesh_without_ghost = old_mesh_with_ghost->refine(markers);
    this->problem_->adapt_boundary(new_mesh_without_ghost);

    LOG_INFO("DMH", "compute ghost cells");

    if (this->mesh_impl_ == IMPL_P4EST) 
    {
      this->mesh_list_[c_mesh_index] = compute_ghost_cells(*new_mesh_without_ghost, 
                                                          this->parcom_->comm(), 
                                                          shared_verts, 
                                                          mesh::IMPL_P4EST, this->ghost_layer_width_);
    } 
    else if (this->mesh_impl_ == IMPL_DBVIEW) 
    {
      assert (false);
      this->mesh_list_[c_mesh_index] = compute_ghost_cells(*new_mesh_without_ghost, 
                                                          this->parcom_->comm(), 
                                                          shared_verts, 
                                                          mesh::IMPL_DBVIEW, this->ghost_layer_width_);
    }

    if (finished_adaption)
    {
      break;
    }

    // modify adapt markers
    // -> remove coarsening and reduced refinement markers by one
    old_mesh_with_ghost = this->mesh_list_[c_mesh_index];
    num_cell = old_mesh_with_ghost->num_entities(tdim);
    markers.clear();
    markers.resize(num_cell, 0);

    for (CellIndex c = 0; c != num_cell; ++c)
    {
      CellId parent_id = old_mesh_with_ghost->get_parent_cell_id(c); 
      if (parent_id < 0)
      {
        continue;
      }
      auto it = parent_id_2_marker.find(parent_id);
      if (it != parent_id_2_marker.end())
      {
        markers[c] = it->second;
      }
    }
  }


  MeshPtr active_mesh = this->get_mesh_by_index(c_mesh_index);
  int num_global_cells = active_mesh->num_global_cells(this->parcom_->comm());

  LOG_DEBUG(0, "[" << this->parcom_->rank() << "] "
                   << "  Mesh " << c_mesh_index
                   << "  #local cells: " << active_mesh->num_local_cells()
                   << ", #ghost cells: " << active_mesh->num_ghost_cells()
                   << ", #sum: " << active_mesh->num_entities(DIM)
                   << ", #global: " << num_global_cells);
  
  // store old space objects     
          
  VectorSpaceSPtr<DataType, DIM> old_space_primal = this->space_list_primal_[0];
  VectorSpaceSPtr<DataType, DIM> old_space_dual = this->space_list_dual_[0];
  
  BlockManagerSPtr old_block_manager_primal = this->block_manager_list_primal_[0];
  BlockManagerSPtr old_block_manager_dual = this->block_manager_list_dual_[0];
  
  // 3. Setup FE space for new mesh
  LOG_INFO("DMH", "setup new space");  
  VectorSpaceSPtr<DataType, DIM> new_space_primal 
    = VectorSpaceSPtr<DataType, DIM> (new VectorSpace<DataType, DIM> ());
  VectorSpaceSPtr<DataType, DIM> new_space_dual 
    = VectorSpaceSPtr<DataType, DIM> (new VectorSpace<DataType, DIM> ());
    
  BlockManagerSPtr new_block_manager_primal = BlockManagerSPtr(new BlockManager());
  BlockManagerSPtr new_block_manager_dual = BlockManagerSPtr(new BlockManager());
  
  // Note: only primal objects are initialized
  this->problem_->setup_space(*new_space_primal,
                              this->coupling_vars_list_primal_[c_mesh_index],
                              this->mesh_list_[c_mesh_index], 
                              1);
  /* std::cout << "dmh " << parcom_->rank() << ": " << new_space_primal.get() 
            << " " << new_space_primal->meshPtr()->num_entities(DIM) 
            << " " << new_space_primal->fe_manager().nb_cell_trafos() << " "
            << std::endl;*/
  
  // 4. change pointers
  this->mesh_ = this->get_mesh_by_index(c_mesh_index);
  this->block_manager_list_primal_[c_mesh_index] = new_block_manager_primal;
  this->block_manager_list_dual_[c_mesh_index] = new_block_manager_dual;
  this->space_list_primal_[c_mesh_index] = new_space_primal;
  this->space_list_dual_[c_mesh_index] = new_space_dual;
  
  this->space_primal_ = this->get_space_by_index(c_mesh_index, 1);
  this->space_dual_ = this->get_space_by_index(c_mesh_index, -1);
  this->active_mesh_index_ = c_mesh_index;
  this->problem_->set_active_space(this->space_primal_, 1);
  this->problem_->set_active_space(this->space_dual_, -1);
  this->problem_->set_active_mesh(this->mesh_);
  this->problem_->set_active_mesh_index(c_mesh_index);

  LOG_INFO("DMH", "  Active space has " 
                  << this->space_primal_->nb_dofs_global()
                  << " total dofs and " << num_global_cells << " cells ");

  // 5. reinit LA objects
  this->problem_->setup_LA_primal(
        *this->space_primal_, *this->space_dual_,
         this->coupling_vars_list_primal_[c_mesh_index],
         this->coupling_vars_list_dual_[c_mesh_index],
         this->block_manager_list_primal_[c_mesh_index],
         this->block_manager_list_dual_[c_mesh_index]);

  // 6. interpolate old vectors w.r.t. new space
  // TODO: make more efficient by providing information of changed cells
  
  LOG_INFO("DMH update", "setup fe interpolators");
  LOG_INFO("DMH update", "#cells in old space " << old_space_primal->meshPtr()->num_entities(DIM));
  LOG_INFO("DMH update", "#fe's  in old space " << old_space_primal->fe_manager().fe_tank_size());
  
  LOG_INFO("DMH update", "#cells in new space " << new_space_primal->meshPtr()->num_entities(DIM));
  LOG_INFO("DMH update", "#fe's  in new space " << new_space_primal->fe_manager().fe_tank_size());
  
  assert (old_space_primal->meshPtr()->num_entities(DIM) == old_space_primal->fe_manager().fe_tank_size());
  assert (new_space_primal->meshPtr()->num_entities(DIM) == new_space_primal->fe_manager().fe_tank_size());
      
  Timer timer;
  timer.start();
  FEInterpolator fe_interpolator;
  fe_interpolator.init(old_space_primal.get(), new_space_primal.get(), true);

  this->parcom_->barrier();

  timer.stop();
  LOG_INFO("DMH update", "took " << timer.get_duration() << " sec");

  timer.reset();
  timer.start();
  for (int l = 0; l < p_vectors.size(); ++l) 
  {
    LOG_INFO("Interpolate", "primal vector " << l);
    fe_interpolator.interpolate(*p_vectors[l], *this->vec_primal_primal_[l]);
    LOG_INFO("Interpolate", "primal vector done ");
  }
  timer.stop();
  LOG_INFO("Interpolate", "took " << timer.get_duration() << " sec");

  for (int l = 0; l < p_vectors.size(); ++l) 
  {
    delete p_vectors[l];
  }

  return true;
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::create_fe_interpolator(int num_mesh) 
{
  this->fe_interpolator_.resize(num_mesh);
  for (int l = 0; l < num_mesh; ++l) 
  {
    this->fe_interpolator_[l].resize(num_mesh, nullptr);
    for (int k = 0; k < num_mesh; ++k) 
    {
      this->fe_interpolator_[l][k] = new FEInterpolator();
    }
  }
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::interpolate_vector(const VectorType &in_vec, 
                                                                        int in_index, 
                                                                        VectorType &out_vec,
                                                                        int out_index) 
{
  if (in_index == -1) 
  {
    return;
  }
  if (in_index == out_index) 
  {
    out_vec.CopyFrom(in_vec);
    return;
  }
  
  if (!this->fe_interpolator_.at(in_index).at(out_index)->is_initialized()) 
  {
    this->fe_interpolator_.at(in_index).at(out_index)->init(
        this->get_space_by_index(in_index, 1).get(),
        this->get_space_by_index(out_index, 1).get(),
        true);
  }

  this->fe_interpolator_.at(in_index).at(out_index)->interpolate(in_vec, out_vec);
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::refine(int mesh_index, std::vector< int > &markers) 
{
  MeshPtr active_mesh = this->get_mesh_by_index(mesh_index);
  assert(active_mesh != 0);
  if (this->mesh_impl_ == mesh::IMPL_P4EST) 
  {
    boost::intrusive_ptr< MeshPXest > mesh_pXest =
        boost::static_pointer_cast< MeshPXest >(active_mesh);
        
    if (this->problem_->need_higher_order_interpolation()) 
    {
      mesh_pXest->set_patch_mode(true);
    } 
    else 
    {
      mesh_pXest->set_patch_mode(false);
    }
    mesh_pXest->set_connection_mode(2);
  }

  SharedVertexTable shared_verts;

  active_mesh = active_mesh->refine(markers);
  this->problem_->adapt_boundary(active_mesh);

  if (this->mesh_impl_ == IMPL_P4EST) 
  {
    this->mesh_list_[mesh_index] = compute_ghost_cells(*active_mesh, 
                                                       this->parcom_->comm(), 
                                                       shared_verts, 
                                                       mesh::IMPL_P4EST, this->ghost_layer_width_);
  } 
  else if (this->mesh_impl_ == IMPL_DBVIEW) 
  {
    this->mesh_list_[mesh_index] = compute_ghost_cells(*active_mesh, 
                                                       this->parcom_->comm(), 
                                                       shared_verts, 
                                                       mesh::IMPL_DBVIEW, this->ghost_layer_width_);
  }
  active_mesh = this->get_mesh_by_index(mesh_index);
  int num_global_cells = active_mesh->num_global_cells(this->parcom_->comm());

  LOG_DEBUG(1, "[" << this->parcom_->rank() << "] "
                   << "  Mesh " << mesh_index
                   << "  #local cells: " << active_mesh->num_local_cells()
                   << ", #ghost cells: " << active_mesh->num_ghost_cells()
                   << ", #sum: " << active_mesh->num_entities(DIM)
                   << ", #global: " << num_global_cells);
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::load_mesh_from_file(int adapt_counter, 
                                                                         int num_mesh, 
                                                                         std::string &prefix) 
{
  this->mesh_list_.clear();
  this->mesh_list_.resize(num_mesh);

  for (int l = 0; l < num_mesh; ++l) 
  {
    std::stringstream pre;
    pre << prefix << "." << adapt_counter << "." << l << ".h5";
    std::string filename = pre.str();

    this->mesh_list_[l] = load_mesh(filename, this->parcom_->comm(), this->mesh_impl_);
  }
  this->problem_->set_active_mesh(this->mesh_list_[0]);
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::save_mesh_to_file(int adapt_counter, 
                                                                       std::string &prefix) 
{
  for (int l = 0; l < this->mesh_list_.size(); ++l) 
  {
    std::stringstream pre;
    pre << prefix << "." << adapt_counter << "." << l << ".h5";
    std::string filename = pre.str();

    save_mesh(filename, this->mesh_list_[l], this->parcom_->comm());
  }
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::visualize_mesh(int adapt_counter, 
                                                                    std::string &prefix) 
{
#if 0
  for (int m = 0; m < this->num_mesh(); ++m) 
  {
    // visualize adapt markers
    /*
    std::vector<double> attr_val(num_cells, 0.);
    for (int c=0; c<num_cells; ++c)
    {
    attr_val[c] = adapt_markers[c];
    }
    AttributePtr attr_est ( new DoubleAttribute ( attr_val ) );
    active_mesh->add_attribute ( "marker" , DIM, attr_est );

    std::stringstream a_input;
    std::stringstream a_pre;
    a_pre << this->root_ << "/mesh/adapt_mesh." << this->adapt_counter_ << "."
    << m;

    if(this->num_partitions_ > 1)
    a_input << ".pvtu";
    else
    a_input << ".vtu";

    std::string a_filename = a_pre.str() + a_input.str();

    PVtkWriter a_writer ( this->comm_ );
    std::ostringstream a_name;
    a_name << this->root_ << "/mesh/adapt_mesh." << this->adapt_counter_ << "."
    << m << ".pvtu"; std::string a_output_file = a_name.str ( );
    a_writer.add_all_attributes ( *active_mesh, true );
    a_writer.write ( a_output_file.c_str ( ), *active_mesh );
     */

    PVtkWriter writer(this->comm_);
    std::ostringstream name;
    name << prefix << "." << adapt_counter << "." << m << ".pvtu";
    std::string output_file = name.str();
    writer.add_all_attributes(*this->get_mesh_by_index(m), true);
    writer.write(output_file.c_str(), *this->get_mesh_by_index(m));
  }
#endif
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::add_mesh_change_times(const std::vector< DataType > &additional_changes,
                                                                           std::vector< int > &indicator_mesh_indices) 
{
  if (this->mesh_adapt_type_ == MESH_ADAPT_NONE || this->mesh_adapt_type_ == MESH_ADAPT_FIXED) 
  {
    this->space_list_tmp_ = this->space_list_primal_;
    return;
  }

  //int num_steps = this->time_mesh_->num_intervals();
  //int num_mesh = this->num_mesh();

  SortedArray< DataType > new_change_list;
  new_change_list.data() = this->mesh_change_times_.data();
  std::map< DataType, bool > is_new_change;
  for (int t = 0; t < new_change_list.size(); ++t) 
  {
    is_new_change.insert(std::pair< DataType, bool >(new_change_list.data().at(t), false));
  }

  // build new mesh change list
  for (int t = 0; t < additional_changes.size(); ++t) 
  {
    new_change_list.insert(additional_changes[t]);
    is_new_change.insert(std::pair< DataType, bool >(additional_changes[t], true));
  }

  LOG_DEBUG(2, "  Old mesh change list: "
                   << string_from_range(this->mesh_change_times_.begin(),
                                        this->mesh_change_times_.end()));

  // fill in copy of meshes
  std::vector< MeshPtr > new_mesh_list;

  std::vector< SortedArray< int > > old2new_mesh_indices;
  old2new_mesh_indices.resize(this->num_mesh());
  old2new_mesh_indices[0].insert(0);

  new_mesh_list.push_back(this->get_mesh_by_index(0));
  space_list_tmp_.clear();
  space_list_tmp_.push_back(this->get_space_by_index(0, 1));

  for (int t = 0; t < new_change_list.size(); ++t) 
  {
    DataType time = new_change_list[t];
    bool new_mesh = is_new_change[time];
    int mesh_index = this->mesh_index(time);
    old2new_mesh_indices[mesh_index].insert(t + 1);

    LOG_DEBUG(2, " mesh change time " << time << " is new mesh " << new_mesh
                                      << " gets mesh with index "
                                      << mesh_index);

    if (new_mesh) 
    {
      MeshPtr tmp_mesh;
      if (this->mesh_impl_ == mesh::IMPL_P4EST) 
      {
        tmp_mesh = new MeshPXest(DIM, DIM);
      } 
      else if (this->mesh_impl_ == mesh::IMPL_DBVIEW) 
      {
        tmp_mesh = new MeshDbView(DIM, DIM);
      }

      tmp_mesh->deep_copy_from(this->get_mesh_by_index(mesh_index));
      new_mesh_list.push_back(tmp_mesh);
    } 
    else 
    {
      new_mesh_list.push_back(this->get_mesh_by_index(mesh_index));
    }
    space_list_tmp_.push_back(this->get_space_by_index(mesh_index, 1));
  }

  this->mesh_list_ = new_mesh_list;
  this->mesh_change_times_.data() = new_change_list.data();

  std::vector< int > new_mesh_indices = indicator_mesh_indices;
  for (int t = 0; t < new_mesh_indices.size(); ++t) 
  {
    int old_index = indicator_mesh_indices[t];
    int pot_new_index = this->mesh_index(this->time_mesh_->time(t));
    int pos = -1;
    if (old2new_mesh_indices[old_index].find(pot_new_index, &pos)) 
    {
      new_mesh_indices[t] = pot_new_index;
    } 
    else 
    {
      new_mesh_indices[t] = old2new_mesh_indices[old_index][0];
    }
  }

  LOG_DEBUG(2, " old indicator 2 mesh index: " 
            << string_from_range(indicator_mesh_indices.begin(), indicator_mesh_indices.end()));

  indicator_mesh_indices = new_mesh_indices;

  LOG_DEBUG(2, " new indicator 2 mesh index: " 
            << string_from_range(indicator_mesh_indices.begin(), indicator_mesh_indices.end()));

  /* write mesh list */
}

template < class LAD, int DIM, class FEInterpolator >
void DynamicMeshHandler< LAD, DIM, FEInterpolator >::adapt_mesh_change_list(
    MeshAdaptType adapt_type, 
    int adapt_counter, 
    int min_steps_for_mesh,
    int max_mesh_number, 
    int rate, 
    DataType tol,
    const std::vector< std::vector< DataType > > &space_indicator,
    std::vector< int > &indicator_mesh_indices) 
{
  if (this->mesh_adapt_type_ == MESH_ADAPT_NONE || this->mesh_adapt_type_ == MESH_ADAPT_FIXED) 
  {
    return;
  }

  int num_steps = this->time_mesh_->num_intervals(adapt_counter);
  int num_mesh = this->num_mesh();

  // Compute differences of error estimators for each time step
  std::vector< double > global_diff(num_steps - 1, 0.);
  std::vector< double > local_diff(num_steps - 1, 0.);
  std::vector< double > local_norm(num_steps - 1, 0.);
  std::vector< double > global_norm(num_steps - 1, 0.);

  // Loop over all meshes
  for (int m = 0; m < num_mesh; ++m) 
  {
    int first_t = this->first_step_for_mesh(m);
    int last_t = this->last_step_for_mesh(m);

    assert (first_t >= 0);
    assert (last_t <= num_steps - 1);
    assert (last_t < space_indicator.size());
    
    int num_cells = this->num_cells_by_step(first_t);

    // Loop over all time step belongin to current mesh, i.e. we ignore
    // differences between time steps belonging to different meshes
    for (int t = first_t; t < last_t; ++t) 
    {
      assert (space_indicator[t].size() >= num_cells);               
      assert (space_indicator[t+1].size() >= num_cells);
      
      // Loop over all cells in current mesh
      for (int c = 0; c < num_cells; ++c) 
      {
        // get space indicator for current cell c and timestep t
        double old_val = space_indicator[t][c];
        double new_val = space_indicator[t + 1][c];

        local_diff[t] += (old_val - new_val) * (old_val - new_val);
        local_norm[t] += old_val * old_val;
      }
    }
  }

  // Allreduce
  this->parcom_->sum(local_diff, global_diff);
  this->parcom_->sum(local_norm, global_norm);

  std::vector< double > rel_diff(num_steps - 1, 0.);

  for (int l = 0; l < global_diff.size(); ++l) {
    global_diff[l] = std::sqrt(global_diff[l]);
    global_norm[l] = std::sqrt(global_norm[l]);

    rel_diff[l] = global_diff[l] / global_norm[l];
  }

  std::vector< int > change_steps;

  // sort differences in ascending order
  std::vector< int > sort_ind(global_diff.size(), 0);
  for (int i = 0; i < rel_diff.size(); ++i) {
    sort_ind[i] = i;
  }
  compute_sort_permutation_stable(rel_diff, sort_ind);

  SortedArray< int > mesh_change_steps;
  mesh_change_steps.data() = this->get_mesh_change_steps();

  LOG_DEBUG(1, "Old mesh change steps " << string_from_range(mesh_change_steps.begin(), mesh_change_steps.end()));

  // insert fixed number of mesh change points
  if (this->mesh_adapt_type_ == MESH_ADAPT_FRACTION) 
  {
    // find #rate largest differences
    int added_steps = 0;
    for (int t = sort_ind.size() - 1; t >= 0; --t) 
    {
      if (added_steps >= rate) {
        break;
      }
      int trial_step = sort_ind[t];

      // check if maximum numbre of mesh changes is reached
      if (change_steps.size() + num_mesh < max_mesh_number) 
      {
        // check if trial step is not first mesh
        if (trial_step > 0) 
        {
          int pos = -1;
          // check if trial step is already contained in list of mesh change
          // steps
          if (!mesh_change_steps.find(trial_step, &pos)) 
          {
            mesh_change_steps.insert(trial_step);
            mesh_change_steps.find(trial_step, &pos);
            int prev_step = -1;
            int next_step = -1;
            if (pos > 0) 
            {
              prev_step = mesh_change_steps.data().at(pos - 1);
            }
            if (pos < mesh_change_steps.size() - 1) 
            {
              next_step = mesh_change_steps.data().at(pos + 1);
            }

            // check if distance to previuous change is large enough
            if (prev_step >= 0 && (trial_step - prev_step) <= min_steps_for_mesh) 
            {
              mesh_change_steps.erase(pos);
              continue;
            }

            // check if distance to next change step is large enough
            if (next_step >= 0 && (next_step - trial_step) <= min_steps_for_mesh) 
            {
              mesh_change_steps.erase(pos);
              continue;
            }

            // accept trial step
            change_steps.push_back(trial_step);
            added_steps++;
          }
        }
      }
    }
  }
  // insert mesh change points according to given threshold
  else if (this->mesh_adapt_type_ == MESH_ADAPT_ERROR) 
  {
    // loop over differences is descending order
    for (int t = sort_ind.size() - 1; t >= 0; --t) 
    {
      int trial_step = sort_ind[t];

      // check if change of trial_step is large enough
      if (rel_diff[trial_step] >= tol) 
      {
        // check if maximum number of mesh change steps is reached
        if (change_steps.size() + num_mesh < max_mesh_number) 
        {
          // check if trial step is not first step
          if (trial_step > 0) 
          {
            int pos = -1;
            // check if trial step is already contained in list of change steps
            if (!mesh_change_steps.find(trial_step, &pos)) 
            {
              mesh_change_steps.insert(trial_step);
              mesh_change_steps.find(trial_step, &pos);
              int prev_step = -1;
              int next_step = -1;
              if (pos > 0) 
              {
                prev_step = mesh_change_steps.data().at(pos - 1);
              }
              if (pos < mesh_change_steps.size() - 1) 
              {
                next_step = mesh_change_steps.data().at(pos + 1);
              }

              // check if distance to previuous change is large enough
              if (prev_step >= 0 && (trial_step - prev_step) <= min_steps_for_mesh) 
              {
                mesh_change_steps.erase(pos);
                continue;
              }

              // check if distance to next change step is large enough
              if (next_step >= 0 && (next_step - trial_step) <= min_steps_for_mesh) 
              {
                mesh_change_steps.erase(pos);
                continue;
              }

              // accept trial step
              change_steps.push_back(trial_step);
            }
          }
        }
      }
    }
  }

  LOG_DEBUG(1, "New mesh change steps " 
            << string_from_range(mesh_change_steps.begin(), mesh_change_steps.end()));

  // adapt dynamic mesh handler
  std::vector< double > add_times;
  for (int t = 0; t < change_steps.size(); ++t) 
  {
    add_times.push_back(this->time_mesh_->time(change_steps[t], adapt_counter));
  }
  this->add_mesh_change_times(add_times, indicator_mesh_indices);
}

} // namespace hiflow

#endif
