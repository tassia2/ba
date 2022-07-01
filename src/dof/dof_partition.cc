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

#include "dof_partition.h"
#include "common/parcom.h"
#include "dof/dof_interpolation.h"
#include "dof/numbering_strategy.h"
#include "fem/fe_manager.h"
#include "fem/fe_reference.h"
#include "mesh/mesh.h"
#include "mesh/iterator.h"
#include "mesh/cell_type.h"

#include <set>

namespace hiflow {
namespace doffem {

template < class DataType, int DIM > 
DofPartition< DataType, DIM >::DofPartition() 
{
  this->mesh_ = nullptr;
  this->fe_manager_ = nullptr;
  this->number_strategy_ = nullptr;
  
  local_nb_dofs_total_ = -1;

  mesh_flag_ = false;
  fe_manager_flag_ = false;
  number_strategy_flag_ = false;
  
  this->applied_number_strategy_ = false;
  
  my_dof_offset_ = 0;

  // Default Value is COMM WORLD
  comm_ = MPI_COMM_WORLD;
  MPI_Comm_size(comm_, &nb_subdom_);
  nb_dofs_on_subdom_.resize(nb_subdom_, -1);
  MPI_Comm_rank(comm_, &my_subdom_);
  shared_subdomains_.resize(nb_subdom_, false);
  
  this->tdim_ = -1;
  this->nb_var_ = -1;
  
  this->called_renumber_ = false;
  this->ordering_fe_cell_ = true;
}

////////////////////////////////////////////////////////////////////////
////// set functions ///////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/// OK
///

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::set_mpi_comm(const MPI_Comm &comm) 
{
  comm_ = comm;
  MPI_Comm_size(comm, &nb_subdom_);

  nb_dofs_on_subdom_.resize(nb_subdom_, -1);
  MPI_Comm_rank(comm, &my_subdom_);
  shared_subdomains_.resize(nb_subdom_, false);
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::set_mesh(const mesh::Mesh *mesh) 
{
  mesh_ = mesh;
  mesh_flag_ = true;
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::set_fe_manager( FEManager< DataType, DIM > const *manager) 
{
  fe_manager_ = manager;
  tdim_ = fe_manager_->tdim();
  nb_fe_ = fe_manager_->nb_fe();
  nb_var_ = fe_manager_->nb_var();
  fe_manager_flag_ = true;
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::set_numbering_strategy (NumberingStrategy< DataType, DIM> * number_strategy)
{
  assert (number_strategy != nullptr);
  this->number_strategy_ = number_strategy;
  
  this->number_strategy_->initialize(*this);
  
  assert(this->check_mesh() == true);
  this->number_strategy_flag_ = true;
}

////////////////////////////////////////////////////////////////////////
////// nb of dofs, etc  ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


/// OK
template < class DataType, int DIM >
size_t DofPartition< DataType, DIM >::nb_dofs_on_subentity( size_t fe_ind , int cell_index, int tdim, int sub_index) const 
{
  return fe_manager_->get_fe(cell_index, fe_ind)->nb_dof_on_subentity(tdim, sub_index);
}

/// OK
template < class DataType, int DIM >
size_t DofPartition< DataType, DIM >::nb_dofs_on_subentity( int cell_index, int tdim, int sub_index) const 
{
  int result = 0;

  for (size_t fe_ind = 0; fe_ind < fe_manager_->nb_fe(); ++fe_ind) 
  {
    result += this->nb_dofs_on_subentity(fe_ind, cell_index, tdim, sub_index);
  }

  return result;
}

/// OK
template < class DataType, int DIM >
int DofPartition< DataType, DIM >::owner_of_dof(gDofId global_id) const 
{
  int result = 0;
  int bound = 0;

  for (size_t s = 0, e_s = nb_subdom_; s != e_s; ++s) {
    bound += nb_dofs_on_subdom_[s];
    if (global_id < bound) {
      result = static_cast< int >(s);
      break;
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////
////// mappings  ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::local2global(lDofId local_id, gDofId *global_id) const 
{
  assert(local_id >= 0 && local_id < nb_dofs_on_subdom_[my_subdom_]);
  *global_id = local_id + my_dof_offset_;
  //  *global_id = local2global_[local_id];
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::global2local(gDofId global_id, lDofId *local_id) const 
{
  assert(global_id >= my_dof_offset_);
  assert(global_id < my_dof_offset_ + nb_dofs_on_subdom_[my_subdom_]);
  *local_id = global_id - my_dof_offset_;
  //   std::map<gDofId,lDofId>::const_iterator it = global2local_.find(global_id);
  //   *local_id = (*it).second;
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::global2cells (gDofId gl, std::vector<int>& cells) const
{
  auto it = this->global2cells_.find(gl);
  if (it != this->global2cells_.end())
  {
    cells.assign(it->second.begin(), it->second.end());
  }
  else
  {
    cells.clear(); 
  }
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::global2cells (gDofId gl, 
                                                  std::vector<int>::iterator insert_pos,
                                                  std::vector<int>& cells, 
                                                  std::vector<int>::iterator& end_pos) const
{
  auto it = this->global2cells_.find(gl);
  if (it != this->global2cells_.end())
  {
    end_pos = insert_pos;
    cells.insert(insert_pos, it->second.begin(), it->second.end());
    
    int steps = std::distance(it->second.begin(), it->second.end());
    assert (steps >= 0);
    for (int c=0; c!=steps; ++c)
    {
      end_pos++;
    }
  }
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::fe2global (size_t fe_ind, std::vector<gDofId>& ids) const
{
  assert (fe_ind < this->nb_fe_);
  assert (this->fe2global_.size() == this->nb_fe_);
  ids.assign(this->fe2global_[fe_ind].begin(), this->fe2global_[fe_ind].end()); 
}
  

////////////////////////////////////////////////////////////////////////
////// get dof indices  ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/// OK
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::get_dofs_on_cell_local( size_t fe_ind, 
                                                            int cell_index, 
                                                            std::vector< lDofId > &ids) const 
{
  const size_t nb_dof = this->fe_manager_->get_fe(cell_index, fe_ind)->nb_dof_on_cell();
  ids.resize(nb_dof);

  // loop over DoFs
  for (size_t i = 0, e_i = nb_dof; i != e_i; ++i) 
  {
    ids[i] = this->cell2local(fe_ind, cell_index, i);
  }
}

/// OK
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::get_dofs_on_cell_local(int cell_index, std::vector< lDofId > &ids) const 
{
  const size_t nb_dof = this->nb_dofs_on_cell(cell_index);
  ids.resize(nb_dof);

  // loop over DoFs
  int offset = 0;
  for (size_t fe_ind=0; fe_ind < this->nb_fe(); ++fe_ind)
  {
    std::vector<lDofId> fe_dofs;
    this->get_dofs_on_cell_local(fe_ind, cell_index, fe_dofs);
    const size_t nb_fe_dofs = fe_dofs.size();
    
    for (size_t i=0; i<nb_fe_dofs; ++i)
    {
      ids[offset+i] = fe_dofs[i]; 
    }
    offset += nb_fe_dofs; 
  }
  assert (offset == nb_dof);
}

/// OK
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::get_dofs_on_cell( size_t fe_ind, 
                                                      int cell_index, 
                                                      std::vector< gDofId > &ids) const 
{
  const size_t nb_dof = this->nb_dofs_on_cell(fe_ind, cell_index);
  ids.resize(nb_dof);
  
  const auto it_b = this->numer_cell_2_global_.begin() + numer_cell_2_global_offsets_[fe_ind][cell_index];

  // loop over DoFs
  for (size_t i = 0; i != nb_dof; ++i) 
  {
    ids[i] = *(it_b+i);
  }

/* old
   const size_t nb_dof = this->fe_manager_->get_fe(cell_index, fe_ind)->nb_dof_on_cell();
   ids.resize(nb_dof);
 
   // loop over DoFs
   for (size_t i = 0, e_i = nb_dof; i != e_i; ++i) 
   {
     ids[i] = this->cell2global(fe_ind, cell_index, i);
   }
   */
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::get_dofs_on_cell(int cell_index, std::vector< gDofId > &ids) const 
{
/* new
  const size_t nb_dof = this->nb_dofs_on_cell(cell_index);
  ids.clear();
  ids.resize(nb_dof);

  const auto it_b = this->numer_cell_2_global_.begin() + numer_cell_2_global_offsets_per_cell_[cell_index];

  // loop over DoFs
  for (size_t i = 0; i != nb_dof; ++i) 
  {
    ids[i] = *(it_b+i);
  }
*/ 
  const size_t nb_dof = this->nb_dofs_on_cell(cell_index);
  ids.resize(nb_dof);
  int offset = 0;
  
  for (size_t fe_ind=0; fe_ind < this->nb_fe(); ++fe_ind)
  { 
    const auto it_b = this->numer_cell_2_global_.begin() + numer_cell_2_global_offsets_[fe_ind][cell_index];
    const size_t nb_dof_fe = this->nb_dofs_on_cell(fe_ind, cell_index);
    for (size_t i = 0; i != nb_dof_fe; ++i) 
    {
      ids[offset+i] = *(it_b+i);
    }
    offset += nb_dof_fe; 
  }
  assert (offset == nb_dof);
/*
  // loop over DoFs
  const size_t nb_dof = this->nb_dofs_on_cell(cell_index);
  ids.clear();
  ids.resize(nb_dof);
  
  int offset = 0;
  for (size_t fe_ind=0; fe_ind < this->nb_fe(); ++fe_ind)
  { 
    std::vector<DofID> fe_dofs;
    this->get_dofs_on_cell(fe_ind, cell_index, fe_dofs);
    const size_t nb_fe_dofs = fe_dofs.size();
    
    for (size_t i=0; i<nb_fe_dofs; ++i)
    {
      ids[offset+i] = fe_dofs[i]; 
    }
    
    offset += nb_fe_dofs; 
  }
  assert (offset == nb_dof);
  */
}

/// OK
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::get_dof_factors_on_cell(int cell_index, std::vector< DataType > &factors) const 
{
  assert (cell_index >= 0);
  assert (cell_index < this->cell_2_dof_factor_.size());
  factors = this->cell_2_dof_factor_[cell_index];
}

/// OK
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::get_dofs_on_subentity( size_t fe_ind, int cell_index, int tdim, int sindex,
                                                           std::vector< gDofId > &ids) const 
{
  // finite element type
  const RefElement< DataType, DIM > &ref_fe = *(fe_manager_->get_fe(cell_index, fe_ind));

  const size_t nb_dof = ref_fe.nb_dof_on_subentity(tdim, sindex);
  
  ids.resize(nb_dof);
  
  const std::vector< cDofId > sub_dof = ref_fe.get_dof_on_subentity(tdim, sindex);
  
  // loop over DoFs
  for (size_t i = 0, e_i = nb_dof; i != e_i; ++i) 
  {
    ids[i] = this->cell2global(fe_ind, cell_index, sub_dof[i]);
  }
}

////////////////////////////////////////////////////////////////////////
////// setup numbering  ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::number(DOF_ORDERING order, bool ordering_fe_cell)
{
  assert (mesh_flag_);
  assert (fe_manager_flag_);
  assert (number_strategy_flag_);
  
  //LOG_INFO("Number", order);
  
  this->ordering_fe_cell_ = ordering_fe_cell;
      
  // initialize dof factor mapping
  const int num_cells = this->mesh_->num_entities(this->tdim_);
  this->cell_2_dof_factor_.clear();
  this->cell_2_dof_factor_.resize(num_cells);
  
  // store nb dofs per cell
  this->nb_dofs_on_cell_for_fe_.clear();
  this->nb_dofs_on_cell_for_fe_.resize(this->nb_fe_);
  
  this->nb_dofs_on_cell_.clear();
  this->nb_dofs_on_cell_.resize(num_cells);
  
  for (size_t f=0; f<this->nb_fe_; ++f)
  {
    this->nb_dofs_on_cell_for_fe_[f].resize(num_cells, 0);
    
    for (int c=0; c<num_cells; ++c)
    {
      const int nb_dof = this->fe_manager_->get_fe(c, f)->nb_dof_on_cell();
      this->nb_dofs_on_cell_for_fe_[f][c] = nb_dof;
      this->nb_dofs_on_cell_[c] += nb_dof;
    }
  }
  
  // loop over cells in mesh
  for (int cell_index = 0; cell_index < num_cells; ++cell_index)
  {
    const size_t nb_dofs_on_cell = this->fe_manager_->nb_dof_on_cell(cell_index);
    this->cell_2_dof_factor_[cell_index].resize(nb_dofs_on_cell, 1.);
  }
  
  // order dofs on local subdomain -> includes identification of common dofs
  LOG_INFO("Number", "locally");
  LOG_INFO("#Dofs local", this->local_nb_dofs_total_);
  
  this->number_strategy_->number_locally(order, ordering_fe_cell);

  LOG_INFO("#Dofs local", this->local_nb_dofs_total_);
  LOG_INFO("Number", "update");
  this->update_number_of_dofs(this->local_nb_dofs_total_,
                              this->local_nb_dofs_for_fe_);

  LOG_INFO("#Dofs local", this->local_nb_dofs_total_);
  // number w.r.r. parallelization issues
  assert (this->applied_number_strategy_);
   
  this->numer_cell_2_local_ = this->numer_cell_2_global_;
  
  // Check if sequential case or truely parallel case is used
  if (nb_subdom_ == 1) 
  {
    nb_dofs_on_subdom_[0] = this->nb_dofs_local();
  } 
  else 
  {
    LOG_INFO("Number", "create ownerships");
    this->create_ownerships();
    
    LOG_INFO("Number", "renumber");
    this->renumber();
    
    if (!this->fe_manager_->contain_only_lagrange_fe())
    {
      LOG_INFO("Number", "correct dof factors");
      this->correct_dof_factors();
#ifndef NDEBUG
      LOG_INFO("Number", "check dof factors");
      this->check_dof_factors();
#endif
    }
    this->local_nb_dofs_total_ = nb_dofs_on_subdom_[my_subdom_];
  }
  //this->consecutive_numbering();
   
  // Calculate number of dofs on the global domain
  gl_nb_dofs_total_ = 0;
  for (size_t s = 0, e_s = nb_subdom_; s != e_s; ++s) {
    assert (nb_dofs_on_subdom_[s] > 0);
    gl_nb_dofs_total_ += nb_dofs_on_subdom_[s];
  }
  LOG_INFO("#Dofs on global domain", gl_nb_dofs_total_);
    
  // create dof -> cells mapping
  this->global2cells_.clear();
  this->fe2global_.clear();
  this->fe2global_.resize(this->nb_fe_);
  
  const int tdim = this->mesh_->tdim();
  for (mesh::EntityIterator cell = this->mesh_->begin(tdim); 
       cell != this->mesh_->end(tdim); ++cell) 
  {  
    std::vector< gDofId > gl_dofs_on_cell;
    this->get_dofs_on_cell(cell->index(), gl_dofs_on_cell);

    for (size_t i=0; i<gl_dofs_on_cell.size(); ++i)
    {
      const auto gl_i = gl_dofs_on_cell[i];
      auto it_i = this->global2cells_.find(gl_i);
      if (it_i == this->global2cells_.end())
      {
        std::set<int> tmp_cell;
        tmp_cell.insert(cell->index());
        this->global2cells_.insert(std::make_pair(gl_i, tmp_cell));
      }
      else
      {
        this->global2cells_[gl_i].insert(cell->index());
      }
    }
    for (size_t fe_ind=0; fe_ind<this->nb_fe_; ++fe_ind)
    {
      std::vector< gDofId > gl_dofs_on_cell_fe;
      this->get_dofs_on_cell(fe_ind, cell->index(), gl_dofs_on_cell_fe);
    
      for (size_t i=0; i<gl_dofs_on_cell_fe.size(); ++i)
      {
        this->fe2global_[fe_ind].insert(gl_dofs_on_cell_fe[i]);
      }
    }
  }
   
  if (nb_subdom_ == 1) 
  {
    this->global_nb_dofs_for_fe_ = this->local_nb_dofs_for_fe_;
  }
  else
  {
    this->global_nb_dofs_for_fe_.clear();
    this->global_nb_dofs_for_fe_.resize(this->nb_fe_,0);

    ParCom parcom(this->comm_);
    for (int f=0; f!=this->nb_fe_; ++f)
    {
      parcom.sum(this->local_nb_dofs_for_fe_[f], this->global_nb_dofs_for_fe_[f]);
    } 
  }

  // Some essential outputs
  /// OK
  if (my_subdom_ == 0) {
    int min_num_dof = 1e6;
    int max_num_dof = 0;
    for (size_t s = 0, e_s = nb_subdom_; s != e_s; ++s) {
      //                    LOG_INFO ( "#Dofs on subdomain " << s,
      //                    nb_dofs_on_subdom_[s] );
      if (nb_dofs_on_subdom_[s] > max_num_dof) {
        max_num_dof = nb_dofs_on_subdom_[s];
      }
      if (nb_dofs_on_subdom_[s] < min_num_dof) {
        min_num_dof = nb_dofs_on_subdom_[s];
      }
    }
    LOG_INFO("#Dofs on global domain", gl_nb_dofs_total_);
    LOG_INFO("Load balancing", static_cast< DataType >(max_num_dof) /
                                   static_cast< DataType >(min_num_dof));
  }
}

/// OK
template < class DataType, int DIM > 
void DofPartition< DataType, DIM >::create_ownerships() 
{
  ownership_.clear();
  ownership_.resize(this->nb_dofs_local(), -1); 

  assert (!this->called_renumber_);
  
  for (mesh::EntityIterator it = this->mesh_->begin(this->tdim_),
                            e_it = this->mesh_->end(this->tdim_); it != e_it; ++it) 
  {
    int subdomain_of_entity;
    it->get("_sub_domain_", &subdomain_of_entity);
  
#ifndef NDEBUG
    if (subdomain_of_entity != my_subdom_) 
    {
      LOG_DEBUG(3,"[" << my_subdom_ << "] remote cell with id " << it->id());
    }
    else
    {
      LOG_DEBUG(3, "[" << my_subdom_ << "] local cell with id " << it->id());
    }
#endif
    
    // dof ids (w.r.t. to local mesh) for given fe_ind on given cell
    std::vector< lDofId > local_dofs;
    this->get_dofs_on_cell_local(it->index(), local_dofs);

    for (size_t i = 0, e_i = local_dofs.size(); i != e_i; ++i) 
    {
      const auto ld_i = local_dofs[i];
      if (subdomain_of_entity < ownership_[ld_i] || ownership_[ld_i] == -1) 
      {
        ownership_[ld_i] = subdomain_of_entity;
      }
    }
  }
}


template < class DataType, int DIM >
void DofPartition< DataType, DIM >::correct_dof_factors()
{ 
  ParCom parcom(this->comm_);
  assert (parcom.rank() == this->my_subdom_);
  
  // prefix "request": refers to those dofs, that are owned by a neighboring process. I want to have the corresponding factors
  // prefix "send": refers to dofs owned by me, which are located on remote cells. The corresponding factors are sent to the respective cell owner
  // suffix "src": refers to process, that eventually sends the dof factors
  // suffix "dest": refers to process, that eventually receives the dof factors
  
  std::vector< std::map<mesh::EntityNumber, std::set<DofID> > > send_mdofids_src(this->nb_subdom_);
  std::vector< std::map<mesh::EntityNumber, std::set<DofID> > > request_mdofids_dest(this->nb_subdom_);
  std::vector< std::map<mesh::EntityNumber, mesh::EntityNumber > > remote_2_local_index(this->nb_subdom_);
  
  // Loop over cells with remote_index >= 0
  for (mesh::EntityIterator it = this->mesh_->begin(DIM), e_it = this->mesh_->end(DIM); it != e_it; ++it) 
  {
    mesh::EntityNumber cell_owner;
    mesh::EntityNumber remote_ind;
    this->mesh_->get_attribute_value("_remote_index_", DIM, it->index(), &remote_ind);
    this->mesh_->get_attribute_value("_sub_domain_", DIM, it->index(), &cell_owner);
    
    if (remote_ind >= 0)
    {
      remote_2_local_index[cell_owner][remote_ind] = it->index();
      
      // get dofs on cell
      std::vector<lDofId> local_dof_ids_on_cell;
      this->get_dofs_on_cell_local(it->index(), local_dof_ids_on_cell);
    
      const size_t nb_dof_on_cell = local_dof_ids_on_cell.size();
      
      // loop over dofs on cell
      for (size_t i=0; i<nb_dof_on_cell; ++i)
      {
        const int dof_owner = this->ownership_[local_dof_ids_on_cell[i]];
        if (dof_owner == this->my_subdom_)
        {
          // send my dof factors to cell owner
          send_mdofids_src[cell_owner][remote_ind].insert(i);
//        std::cout << this->my_subdom_ << " | " << cell_owner << " : send dof factor " << i << " on my cell " << it->index() << " to remote cell " 
//                             << remote_ind << " : value " << cell2factor(it->index(), i) << std::endl; 
        }
        else
        {
          // request dof factors from cell owner
          // note: dof_owner != cell_owner is possible. 
          // therefore, first owned dof factors are sent to cell owners,
          // then remotely owned factors are requested from cell owners
          request_mdofids_dest[cell_owner][remote_ind].insert(i);
        }
      }
    }
  }

  // flatten constructed maps to vectors, suitable for MPI operations
  std::vector< std::vector<mesh::EntityNumber> >request_cells_dest(this->nb_subdom_);
  std::vector< std::vector<DofID> >request_dofids_dest(this->nb_subdom_);
  std::vector< std::vector<int> >request_offsets_dest(this->nb_subdom_);
  std::vector< int >request_nb_cell_dest(this->nb_subdom_); 
  std::vector< int >request_nb_dof_dest(this->nb_subdom_); 
  
  std::vector< std::vector<mesh::EntityNumber> >send_cells_src(this->nb_subdom_);
  std::vector< std::vector<DofID> >send_dofids_src(this->nb_subdom_);
  std::vector< std::vector<int> >send_offsets_src(this->nb_subdom_);
  std::vector< int >send_nb_cell_src(this->nb_subdom_); 
  std::vector< int >send_nb_dof_src(this->nb_subdom_); 
  
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    request_nb_cell_dest[p] = request_mdofids_dest[p].size();
    send_nb_cell_src[p] = send_mdofids_src[p].size();
    
    int c = 0;
    int nb_dofs_p = 0;
    request_offsets_dest[p].push_back(0);
    for (auto cell_it = request_mdofids_dest[p].begin(), e_cell_it = request_mdofids_dest[p].end();
         cell_it != e_cell_it; ++cell_it) 
    {
      request_cells_dest[p].push_back( cell_it->first );
      for (auto dof_it=cell_it->second.begin(), e_dof_it = cell_it->second.end(); 
           dof_it != e_dof_it; ++dof_it)
      {
        request_dofids_dest[p].push_back(*dof_it);
        nb_dofs_p++;
      }
      request_offsets_dest[p].push_back(request_offsets_dest[p][c] + cell_it->second.size());  
      c++;
    }
    request_nb_dof_dest[p] = nb_dofs_p;
    assert(nb_dofs_p == request_offsets_dest[p][request_offsets_dest[p].size()-1]); 
  
    c = 0;
    nb_dofs_p = 0;
    send_offsets_src[p].push_back(0);
    for (auto cell_it = send_mdofids_src[p].begin(), e_cell_it = send_mdofids_src[p].end();
         cell_it != e_cell_it; ++cell_it) 
    {
      send_cells_src[p].push_back( cell_it->first );
      for (auto dof_it=cell_it->second.begin(), e_dof_it = cell_it->second.end(); 
           dof_it != e_dof_it; ++dof_it)
      {
        send_dofids_src[p].push_back(*dof_it);
        nb_dofs_p++;
      }
      send_offsets_src[p].push_back(send_offsets_src[p][c] + cell_it->second.size());  
      c++;
    }
    send_nb_dof_src[p] = nb_dofs_p;
    assert(nb_dofs_p == send_offsets_src[p][send_offsets_src[p].size()-1]); 
/*
    std::vector<int> tmp(request_vcells_from[p].size(),-1);
    for (int l=0; l<tmp.size(); ++l)
      tmp[l] = remote_2_local_index[p][request_vcells_from[p][l]];
      
    std::cout << this->my_subdom_ 
              << " : requested cells from " << p << " : " << string_from_range(request_vcells_from[p].begin(), request_vcells_from[p].end()) 
              << " corresponding to my cells : "  << string_from_range(tmp.begin(), tmp.end()) << std::endl; 
*/            
  }
  
  // exchange number of dofs and cells to be exchanged
  LOG_INFO("Exchange", "number dofs and cells");
  std::vector<int> request_nb_dof_src(this->nb_subdom_,0);
  std::vector<int> request_nb_cell_src(this->nb_subdom_,0);
  std::vector<int> send_nb_dof_dest(this->nb_subdom_,0);
  std::vector<int> send_nb_cell_dest(this->nb_subdom_,0);
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    parcom.send(request_nb_cell_dest[p], p, 0);
    parcom.send(request_nb_dof_dest[p], p, 1);
    parcom.send(send_nb_cell_src[p],    p, 10);
    parcom.send(send_nb_dof_src[p],     p, 11);
  }
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    int p1 = parcom.recv(request_nb_cell_src[p], p, 0);
    int p2 = parcom.recv(request_nb_dof_src[p],  p, 1);
    int p3 = parcom.recv(send_nb_cell_dest[p],   p, 10);
    int p4 = parcom.recv(send_nb_dof_dest[p],    p, 11);
    assert (p1 + p2 + p3 + p4 == 0);
  }

  parcom.barrier();
  std::vector< std::vector<MPI_Request> > send_reqs (6);
  std::vector< std::vector<MPI_Request> > recv_reqs (6);
  std::vector< std::vector<MPI_Status>  > stats (6);
    
  for (int l=0; l<6; ++l)
  {
    send_reqs[l].resize(this->nb_subdom_);
    recv_reqs[l].resize(this->nb_subdom_);
    stats[l].resize(this->nb_subdom_);
  }

  // --------------------------------------------
  // --------------------------------------------
  // dof owners send their factors to cell owners
  LOG_INFO("Exchange", "Dof owner to cell owner");
  std::vector< std::vector<mesh::EntityNumber> > send_cells_dest (this->nb_subdom_);
  std::vector< std::vector<DofID> > send_dofids_dest (this->nb_subdom_); 
  std::vector< std::vector<int> > send_offsets_dest (this->nb_subdom_);
  
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    //PLOG_INFO(parcom.rank(), "Isend", 12 << " : " << send_nb_cell_src[p] << " elements to " << p);
    parcom.Isend(send_cells_src[p],  send_nb_cell_src[p],  p, 12, send_reqs[0][p]);
    
    //PLOG_INFO(parcom.rank(), "Isend", 13 << " : " << send_nb_dof_src[p] << " elements to " << p);
    parcom.Isend(send_dofids_src[p], send_nb_dof_src[p],   p, 13, send_reqs[1][p]);
    
    //PLOG_INFO(parcom.rank(), "Isend", 14 << " : " << send_nb_cell_src[p]+1 << " elements to " << p);
    parcom.Isend(send_offsets_src[p],send_nb_cell_src[p]+1,p, 14, send_reqs[2][p]);  
  }
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }

    //PLOG_INFO(parcom.rank(), "Irecv", 12 << " : " << send_nb_cell_dest[p] << " elements from " << p);
    int p1 = parcom.Irecv(send_cells_dest[p],   send_nb_cell_dest[p],   p, 12, recv_reqs[0][p]);
    
    //PLOG_INFO(parcom.rank(), "Irecv", 13 << " : " << send_nb_dof_dest[p] << " elements from " << p);
    int p2 = parcom.Irecv(send_dofids_dest[p],  send_nb_dof_dest[p],    p, 13, recv_reqs[1][p]);
    
    //PLOG_INFO(parcom.rank(), "Irecv", 14 << " : " << send_nb_cell_dest[p]+1 << " elements from " << p);
    int p3 = parcom.Irecv(send_offsets_dest[p], send_nb_cell_dest[p]+1, p, 14, recv_reqs[2][p]);
    assert (p1 + p2 + p3 == 0);
  }

  // wait until communication is finished
  //std::cout << parcom.rank() << ":: wait" << std::endl;
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    for (int l=0; l<3; ++l)
    {
      //MPI_Status  status;
      //std::cout << parcom.rank() << ":: wait "<< l << " " << p << std::endl;
      int err = parcom.wait(recv_reqs[l][p]);
      //int err = MPI_Wait(&recv_reqs[l][p], &status);
      //PLOG_INFO(parcom.rank(), "wait", l << " " << p << " : " << err);
      //std::cout << parcom.rank() << ":: wait "<< l << " " << p << " : " << err << std::endl;
      assert (err == 0);
    }
  }

  parcom.barrier();
  
  // exchange dof factors
  std::vector< std::vector<DataType> > send_factors_src(this->nb_subdom_);
  std::vector< std::vector<DataType> > send_factors_dest(this->nb_subdom_);
  //PLOG_INFO(parcom.rank(), "exchange", "dof factors");
  LOG_INFO("exchange", "dof factors");
  
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    for (int c=0; c<send_nb_cell_src[p]; ++c)
    {
      mesh::EntityNumber remote_index = send_cells_src[p][c];
      mesh::EntityNumber cur_index = remote_2_local_index[p][remote_index];
      for (int i=send_offsets_src[p][c]; i<send_offsets_src[p][c+1]; ++i)
      {
        assert (this->cell2factor(cur_index, send_dofids_src[p][i]) != 0.);
        send_factors_src[p].push_back(this->cell2factor(cur_index, send_dofids_src[p][i]));
/*
        std::cout << "PREPARE SEND "<< this->my_subdom_ << " | " << p << " : send dof factor " << send_dofids_src[p][i] 
                << " on my cell " << cur_index << " remote " << remote_index << " : value " << this->cell2factor(cur_index, send_dofids_src[p][i]) << std::endl; 
*/                              
      } 
    }
    assert (send_factors_src[p].size() == send_nb_dof_src[p]);
    send_factors_dest[p].resize(send_nb_dof_dest[p], -99);
  }

  std::vector<MPI_Request> send_reqs_2 (this->nb_subdom_);
  std::vector<MPI_Request> recv_reqs_2 (this->nb_subdom_);
  std::vector<MPI_Status>  stats_2 (this->nb_subdom_);

  std::vector< size_t > nb_send_factors_dest (this->nb_subdom_);
  std::vector< size_t > nb_send_factors_src (this->nb_subdom_);
  
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    nb_send_factors_src[p] = send_factors_src[p].size();
    //PLOG_INFO(parcom.rank(), "send", 1001 << " : " << nb_send_factors_src[p] << " to " << p);
    parcom.send(nb_send_factors_src[p], p, 1001);
  }
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    
    int p1 = parcom.recv(nb_send_factors_dest[p], p, 1001);
    //PLOG_INFO(parcom.rank(), "recv", 1001 << " : " << nb_send_factors_dest[p] << " from " << p);
    assert (p1== 0);
  }
  
  parcom.barrier();
  //PLOG_INFO(parcom.rank(), "exchange", "done");
  
  // receive dof factors
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    //PLOG_INFO(parcom.rank(), "Isend", 15 << " : " << send_factors_src[p].size() << " elements to " << p);
    parcom.Isend(send_factors_src[p], send_factors_src[p].size(), p, 15, send_reqs_2[p]);
  }
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    //PLOG_INFO(parcom.rank(), "Irecv", 15 << " : " << nb_send_factors_dest[p] << " elements from " << p);
    int p1 = parcom.Irecv(send_factors_dest[p], nb_send_factors_dest[p], p, 15, recv_reqs_2[p]);
    assert (p1 == 0);
  }
  
  // wait until communication is finished
  LOG_INFO("wait", " for exchange of dof factors");
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    //PLOG_INFO(parcom.rank(), "wait", " for comm process " << p << "  " << recv_reqs_2[p]);
    int err = parcom.wait(recv_reqs_2[p]);
    //PLOG_INFO(parcom.rank(), "wait", " for comm process " << p << "  " << recv_reqs_2[p]<< " done " << err);
    assert (err == 0);
  }
  
  parcom.barrier();
  // insert received dof factors
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    // loop over local cells
    for (int c=0; c<send_cells_dest[p].size(); ++c)
    {
      const mesh::EntityNumber my_cell_index = send_cells_dest[p][c];
      
      // loop over remote dofs
      for (int i=send_offsets_dest[p][c]; i<send_offsets_dest[p][c+1]; ++i)
      {
        const DofID my_dof_index = send_dofids_dest[p][i];
        
        assert (my_cell_index < this->cell_2_dof_factor_.size());
        assert (my_dof_index < this->cell_2_dof_factor_[my_cell_index].size());
        assert (i < send_factors_dest[p].size());
/*
        std::cout << "SEND " << this->my_subdom_ << " | " << p << " : cell " << my_cell_index << " dof " 
                  << my_dof_index << " owner " << this->ownership_[this->cell2local(my_cell_index, my_dof_index)]
                  << " old factor " << this->cell_2_dof_factor_[my_cell_index][my_dof_index]
                  << " new factor " << send_factors_dest[p][i] << std::endl;
*/

// TODO: include this assert??
//        assert (this->my_subdom_ != this->ownership_[this->cell2local(my_cell_index, my_dof_index)] ||
//                this->cell_2_dof_factor_[my_cell_index][my_dof_index] == send_factors_dest[p][i]);
        this->cell_2_dof_factor_[my_cell_index][my_dof_index] = send_factors_dest[p][i];
      }
    }
  }
  
  // --------------------------------------------
  // --------------------------------------------
  // request factors from cell owners

  std::vector< std::vector<mesh::EntityNumber> > request_cells_src (this->nb_subdom_);
  std::vector< std::vector<DofID> > request_dofids_src (this->nb_subdom_); 
  std::vector< std::vector<int> > request_offsets_src (this->nb_subdom_);

  std::vector< size_t > nb_request_cells_src (this->nb_subdom_);
  std::vector< size_t > nb_request_dofids_src (this->nb_subdom_); 
  std::vector< size_t > nb_request_offsets_src (this->nb_subdom_);

  std::vector< size_t > nb_request_cells_dest (this->nb_subdom_);
  std::vector< size_t > nb_request_dofids_dest (this->nb_subdom_); 
  std::vector< size_t > nb_request_offsets_dest (this->nb_subdom_);
  
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    nb_request_cells_dest[p] = request_cells_dest[p].size();
    nb_request_dofids_dest[p] = request_dofids_dest[p].size();
    nb_request_offsets_dest[p] = request_offsets_dest[p].size();
    
    parcom.send(nb_request_cells_dest[p], p, 1002);
    parcom.send(nb_request_dofids_dest[p], p, 1003);
    parcom.send(nb_request_offsets_dest[p], p, 1004);
  }
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    
    int p1 = parcom.recv(nb_request_cells_src[p], p, 1002);
    int p2 = parcom.recv(nb_request_dofids_src[p], p, 1003);
    int p3 = parcom.recv(nb_request_offsets_src[p], p, 1004);
    assert (p1 + p2 + p3 == 0);
  }

  LOG_INFO("request", "dof factors from cell owners");

  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    assert (request_cells_dest[p].size()  == request_nb_cell_dest[p]);
    assert (request_dofids_dest[p].size() == request_nb_dof_dest[p]);
    assert (request_offsets_dest[p].size() == request_nb_cell_dest[p]+1);
    
    parcom.Isend(request_cells_dest[p], request_cells_dest[p].size(), p, 2, send_reqs[3][p]);
    parcom.Isend(request_dofids_dest[p], request_dofids_dest[p].size(), p, 3, send_reqs[4][p]);
    parcom.Isend(request_offsets_dest[p], request_offsets_dest[p].size(), p, 4, send_reqs[5][p]);
  }
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    
    int p1 = parcom.Irecv(request_cells_src[p], nb_request_cells_src[p], p, 2, recv_reqs[3][p]);
    int p2 = parcom.Irecv(request_dofids_src[p], nb_request_dofids_src[p], p, 3, recv_reqs[4][p]);
    int p3 = parcom.Irecv(request_offsets_src[p], nb_request_offsets_src[p], p, 4, recv_reqs[5][p]);
    
    assert (p1 + p2 + p3 == 0);
  }

  // wait until communication is finished
  //LOG_INFO("wait", "");
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    for (int l=3; l<6; ++l)
    {
      //LOG_INFO("wait", p << " " << l);
      int err = parcom.wait(recv_reqs[l][p]);
      //LOG_INFO("wait", l << " " << p << " : " << err);
      assert (err == 0);
    }
  }

  // answer to request
  std::vector< std::vector<DataType> > request_factors_src(this->nb_subdom_);
  std::vector< std::vector<DataType> > request_factors_dest (this->nb_subdom_);
  LOG_INFO("answer", "requests");
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    for (int c=0; c<request_cells_src[p].size(); ++c)
    {
      mesh::EntityNumber cur_index = request_cells_src[p][c];
      for (int i=request_offsets_src[p][c]; i<request_offsets_src[p][c+1]; ++i)
      {
        assert (this->cell2factor(cur_index, request_dofids_src[p][i]) != 0.);
        request_factors_src[p].push_back(this->cell2factor(cur_index, request_dofids_src[p][i]));
      } 
    }
    assert (request_factors_src[p].size() == request_nb_dof_src[p]);
    request_factors_dest[p].resize(request_nb_dof_dest[p], -99);
  }

  std::vector<MPI_Request> send_reqs_1 (this->nb_subdom_);
  std::vector<MPI_Request> recv_reqs_1 (this->nb_subdom_);
  std::vector<MPI_Status>  stats_1 (this->nb_subdom_);
  
  // receive requested dof factors
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
/*
    std::cout << this->my_subdom_ << " | " << p << " : " << string_from_range(requested_offsets.begin(), requested_offsets.end()) << std::endl;
    std::cout << this->my_subdom_ << " | " << p << " : " << send_factors.size() << " " << nb_dofs << std::endl;
    std::cout << this->my_subdom_ << " | " << p << " : factors " << string_from_range(send_factors.begin(), send_factors.end()) << std::endl;
    std::cout << "process " << this->my_subdom_ << " sends " << nb_dofs << " dof factors to " << p << std::endl;
    std::cout << "process " << this->my_subdom_ << " sends " << string_from_range(send_factors.begin(), send_factors.end()) << std::endl;
*/    
    parcom.Isend(request_factors_src[p], request_factors_src[p].size(), p, 5, send_reqs_1[p]);
  }

  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }

    int p1 = parcom.Irecv(request_factors_dest[p],request_factors_dest[p].size(), p, 5, recv_reqs_1[p]);
    assert (p1 == 0);
/*
    std::cout << "process " << this->my_subdom_ << " receives " << nb_recv_dofs << " dof factors from " << p << std::endl;
    std::cout << "process " << this->my_subdom_ << " received " << string_from_range(recv_factors.begin(), recv_factors.end()) << std::endl;
*/
  }

  // wait until communication is finished
  //LOG_INFO("wait", "");
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    int err = parcom.wait(recv_reqs_1[p]);
    assert (err == 0);
  }
  
  // correct dof factors
  //LOG_INFO("correct", "factors");
  for (int p=0; p<this->nb_subdom_; ++p)
  {
    if (p == this->my_subdom_)
    {
      continue;
    }
    // loop over remote cells
    for (int c=0; c<request_cells_dest[p].size(); ++c)
    {
      const mesh::EntityNumber remote_index = request_cells_dest[p][c];
      const mesh::EntityNumber my_cell_index = remote_2_local_index[p][remote_index];
    
      // loop over remote dofs
      for (int i=request_offsets_dest[p][c]; i<request_offsets_dest[p][c+1]; ++i)
      {
        const DofID my_dof_index = request_dofids_dest[p][i];
        
        assert (my_cell_index < this->cell_2_dof_factor_.size());
        assert (my_dof_index < this->cell_2_dof_factor_[my_cell_index].size());
        assert (i < request_factors_dest[p].size());
/*
        std::cout << "REQUEST " << this->my_subdom_ << " | " << p << " : cell " << my_cell_index << " dof " 
                  << my_dof_index << " owner " << this->ownership_[this->cell2local(my_cell_index, my_dof_index)]
                  << " old factor " << this->cell_2_dof_factor_[my_cell_index][my_dof_index]
                  << " new factor " << request_factors_dest[p][i] << std::endl;
*/
//        note (Philipp G): not sure, whether the following assert is necessary or not...
//        assert (this->my_subdom_ != this->ownership_[this->cell2local(my_cell_index, my_dof_index)] ||
//              this->cell_2_dof_factor_[my_cell_index][my_dof_index] == request_factors_dest[p][i]);
        this->cell_2_dof_factor_[my_cell_index][my_dof_index] = request_factors_dest[p][i];
      }
    }
  }
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::check_dof_factors()
{ 
  ParCom parcom(this->comm_);

  const int nb_sub = parcom.size();
  const int my_rank = parcom.rank();

  std::vector< std::vector<int> > remote_2_send (nb_sub);
  std::vector< std::vector< std::vector<DataType> > > factors_2_send (nb_sub);
  std::vector< std::vector< DataType > > own_factors(this->mesh_->num_entities(DIM));

  int nb_dof_factors = 0;
  
  // loop over cells
  for (mesh::EntityIterator it = this->mesh_->begin(DIM);
       it != this->mesh_->end(DIM); ++it) 
  {
    const int cell_index = it->index();
    int remote, domain;
    this->mesh_->get_attribute_value("_remote_index_", DIM, cell_index, &remote);
    this->mesh_->get_attribute_value("_sub_domain_", DIM, it->index(), &domain);

    std::vector<DataType> c_factors;
    this->get_dof_factors_on_cell(cell_index, c_factors);
    
    own_factors[cell_index] = c_factors;
    
    if (remote >= 0)
    {
      remote_2_send[domain].push_back(remote);
      factors_2_send[domain].push_back(c_factors);
    }
    nb_dof_factors = c_factors.size();
  }
  
  // exchange data
  std::vector<int> num_cells_2_recv(nb_sub, 0);
  
  for (int p=0; p<nb_sub; ++p)
  {
    if (p == my_rank)
    {
      continue;
    }
   
    int num_cells_2_send = remote_2_send[p].size();   
    parcom.send(num_cells_2_send, p, 0);
    parcom.send(remote_2_send[p], num_cells_2_send, p, 1);

    for (int i=0; i<num_cells_2_send; ++i)
    {
      parcom.send(factors_2_send[p][i], nb_dof_factors, p, 2+i);
    } 
  }

  
  std::vector< int > local_shared_ind;
  std::vector< std::vector<DataType> > local_shared_factors;
  for (int p=0; p<nb_sub; ++p)
  {
    if (p == my_rank)
    {
      continue;
    }
    
    int num_cells_from_p = 0;
    int err = parcom.recv(num_cells_from_p, p, 0);
    assert (err == 0);
    
    std::vector<int> remote_from_p(num_cells_from_p);
    err = parcom.recv(remote_from_p, num_cells_from_p, p, 1);
    assert (err == 0);
    
    std::vector< std::vector<DataType> > factors_from_p(num_cells_from_p);
    for (int i=0; i<num_cells_from_p; ++i)
    {
      factors_from_p[i].resize(nb_dof_factors, -99);
      err = parcom.recv(factors_from_p[i], nb_dof_factors, p, 2+i);
      assert (err == 0);
      
      local_shared_ind.push_back(remote_from_p[i]);
      local_shared_factors.push_back(factors_from_p[i]);
    }
  }
  
  parcom.barrier();

  for (int p=0; p<nb_sub; ++p)
  {
    if (p == my_rank)
    {
      for (int i=0; i<local_shared_ind.size(); ++i)
      {
        int index = local_shared_ind[i];
        for (int l=0; l<nb_dof_factors; ++l)
        {
          if (std::abs(own_factors[index][l]-local_shared_factors[i][l]) > 1e-8)
          {
            PLOG_INFO(parcom.rank(), "check", "p= " << p << " , l=" << l << " , index = " << index << " : " 
                      << own_factors[index][l] << " <> " << local_shared_factors[i][l]);
          }
          assert (std::abs(own_factors[index][l]-local_shared_factors[i][l]) < 1e-8);
        }
      }
    }
  }
}

template < class DataType, int DIM > 
void DofPartition< DataType, DIM >::renumber() 
{
  this->called_renumber_ = true;
  ParCom parcom(this->comm_);
  
  // Communicate number of dofs including ghost layer
  std::vector< int > ndofs_with_ghost( static_cast< size_t >(nb_subdom_));
  int ndofs_with_ghost_sent = this->nb_dofs_local();

  // Store information about number of dofs including ghost layer
  nb_dofs_incl_ghost_ = this->nb_dofs_local();

  parcom.allgather(ndofs_with_ghost_sent, ndofs_with_ghost);

  LOG_DEBUG(2, "[" << my_subdom_ << "]: Number of DOFs on each subdomain: "
                   << string_from_range(ndofs_with_ghost.begin(),
                                        ndofs_with_ghost.end()));
  LOG_DEBUG(3, "[" << my_subdom_ << "]: Ownership: "
                   << string_from_range(ownership_.begin(), ownership_.end()));

  // Calculate temporary dof offset
  int tmp_dof_offset = 0;
  for (size_t s = 0, e_s = static_cast< size_t >(my_subdom_); s != e_s;
       ++s) {
    tmp_dof_offset += ndofs_with_ghost[s];
  }

  // Fill first permutation to create a local consecutive numbering w.r.t.
  // tmp_dof_offset
  std::vector< int > permutation(ownership_.size());
  int dof_number = 0;

  for (size_t i = 0, e_i = ownership_.size(); i != e_i; ++i) {
    if (ownership_[i] == my_subdom_) {
      permutation[i] = dof_number + tmp_dof_offset;
      ++dof_number;
    }
  }

  LOG_DEBUG(2,
            "[" << my_subdom_
                << "]: Number of (interior) DOFs on subdomain: " << dof_number);

  LOG_DEBUG(3,
            "[" << my_subdom_ << "]: Permutation size " << permutation.size()
                << ", content: "
                << string_from_range(permutation.begin(), permutation.end()));

  int ghost_number = 0;
  for (size_t i = 0, e_i = ownership_.size(); i != e_i; ++i) {
    if (ownership_[i] != my_subdom_) {
      permutation[i] = dof_number + tmp_dof_offset + ghost_number;
      ++ghost_number;
    }
  }

  LOG_DEBUG(2,
            "[" << my_subdom_
                << "]: Number of (ghost) DOFs on subdomain: " << ghost_number);

  LOG_DEBUG(3,
            "[" << my_subdom_ << "]: Permutation size " << permutation.size()
                << ", content: "
                << string_from_range(permutation.begin(), permutation.end()));

  this->apply_permutation(permutation);

  LOG_DEBUG(2, " First permutation done ");

  // Calculate number of dofs which belong to my subdomain
  // Gather number of dofs of all subdomains on all processes
  parcom.allgather(dof_number, nb_dofs_on_subdom_);

  LOG_DEBUG(2,
            "[" << my_subdom_ << "]: ndofs_on_sd "
                << string_from_range(nb_dofs_on_subdom_.begin(), nb_dofs_on_subdom_.end()));

  // Setting up data structure for communication and management of dofs
  // which belong to ghost cells on current process.
  // The data structure maps subdomain indices to maps of (ghost) cell
  // indices. For each (ghost) cell, a map of size dof_nvar_ is hold which
  // itself contains the (global) dof numbers of this cell
  std::map< int, std::map< int, std::map< int, std::vector< int > > > >
      numer_ghost;

  // Create subdomain/ghost cell structure of numer_ghost
  for (mesh::EntityIterator it = this->mesh_->begin(this->tdim_),
                            e_it = this->mesh_->end(this->tdim_);
       it != e_it; ++it) {
    int subdomain_index;
    it->get("_sub_domain_", &subdomain_index);

    if (subdomain_index != my_subdom_) {
      // Access/create (on first access) map for remote subdomain
      numer_ghost[subdomain_index];
      int ghost_cell_index;
      it->get("_remote_index_", &ghost_cell_index);
      // Access/create (on first access) map entry for ghost cell
      numer_ghost[subdomain_index][ghost_cell_index];
    }
  }

  {
    // Number of ghost cells which I share with each other process
    std::vector< int > num_ghost_cells(
        static_cast< size_t >(this->nb_subdom_), 0);
    int total_num_ghost_cells = 0;
    for (int i = 0; i < this->nb_subdom_; ++i) {
      num_ghost_cells[i] = numer_ghost[i].size();

      LOG_DEBUG(2, "[" << my_subdom_ << "]: common ghost cells with process "
                       << i << ": " << num_ghost_cells[i]);

      total_num_ghost_cells += numer_ghost[i].size();
    }

    // Number of ghost cells which other processes share with me
    std::vector< int > num_ghost_cells_others(
        static_cast< size_t >(this->nb_subdom_), 0);

    // Exchange number of ghost cells
    MPI_Alltoall(vec2ptr(num_ghost_cells), 1, MPI_INT,
                 vec2ptr(num_ghost_cells_others), 1, MPI_INT, this->comm_);

#ifndef NDEBUG
    for (size_t i = 0; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      LOG_DEBUG(2, "[" << my_subdom_ << "]: process " << i
                       << " requests number of ghost cells: "
                       << num_ghost_cells_others[i]);
    }
#endif

    // Ghost cell indices which I need of others
    std::vector< int > ghost_indices;
    ghost_indices.reserve(static_cast< size_t >(total_num_ghost_cells));
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (typename std::map<
               int, std::map< int, std::vector< int > > >::const_iterator
               it = numer_ghost[i].begin(),
               e_it = numer_ghost[i].end();
           it != e_it; ++it) {
        ghost_indices.push_back(it->first);
      }
    }

    std::vector< int > offsets(
        static_cast< size_t >(this->nb_subdom_), 0);
    for (size_t i = 1; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      offsets[i] = offsets[i - 1] + num_ghost_cells[i - 1];
    }

    std::vector< int > offsets_others(
        static_cast< size_t >(this->nb_subdom_), 0);
    for (size_t i = 1; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      offsets_others[i] = offsets_others[i - 1] + num_ghost_cells_others[i - 1];
    }

    // Ghost cell indices which others need from me
    int total_num_ghost_cells_others = 0;
    for (size_t i = 0; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      total_num_ghost_cells_others += num_ghost_cells_others[i];
    }

    std::vector< int > ghost_indices_others(
        static_cast< size_t >(total_num_ghost_cells_others), 0);

    MPI_Alltoallv(vec2ptr(ghost_indices), vec2ptr(num_ghost_cells),
                  vec2ptr(offsets), MPI_INT, vec2ptr(ghost_indices_others),
                  vec2ptr(num_ghost_cells_others), vec2ptr(offsets_others),
                  MPI_INT, this->comm_);

    LOG_DEBUG(2, "Exchanged ghost cell indices");

    // Number of ghost dofs for all variables and all ghost cells which
    // I need from others
    std::vector< int > num_dofs_ghost(
        static_cast< size_t >(total_num_ghost_cells * this->nb_fe_), 0);

    // Number of ghost dofs for all variables and all ghost cell which
    // others need from me
    std::vector< int > num_dofs_ghost_others(
        static_cast< size_t >(total_num_ghost_cells_others * this->nb_fe_), 0);

    std::vector< int > offset_ghost_cells_others(
        static_cast< size_t >(this->nb_subdom_), 0);
    for (size_t i = 1; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      offset_ghost_cells_others[i] =
          offset_ghost_cells_others[i - 1] + num_ghost_cells_others[i - 1];
    }

    std::vector< int > offset_ghost_cells(
        static_cast< size_t >(this->nb_subdom_), 0);
    for (size_t i = 1; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      offset_ghost_cells[i] =
          offset_ghost_cells[i - 1] + num_ghost_cells[i - 1];
    }

    int num_ind_others = 0;
    for (size_t i = 0; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      for (size_t k = 0; k < static_cast< size_t >(num_ghost_cells_others[i]);
           ++k) {
        for (size_t l = 0; l < static_cast< size_t >(this->nb_fe_); ++l) {
          num_dofs_ghost_others[(offset_ghost_cells_others[i] + k) * this->nb_fe_ + l] =
              this->nb_dofs_on_cell( l, ghost_indices_others[offset_ghost_cells_others[i] + k]);
          num_ind_others += num_dofs_ghost_others[(offset_ghost_cells_others[i] + k) * this->nb_fe_ + l];
        }
      }
    }

    std::vector< int > num_dofs_ghost_others_vars(
        static_cast< size_t >(this->nb_subdom_), -1);
    for (size_t i = 0; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      num_dofs_ghost_others_vars[i] = num_ghost_cells_others[i] * this->nb_fe_;
    }

    for (size_t i = 1; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      offsets_others[i] =
          offsets_others[i - 1] + num_dofs_ghost_others_vars[i - 1];
    }

    std::vector< int > num_dofs_ghost_vars(
        static_cast< size_t >(this->nb_subdom_), -1);
    for (size_t i = 0; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      num_dofs_ghost_vars[i] = num_ghost_cells[i] * this->nb_fe_;
    }

    for (size_t i = 1; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      offsets[i] = offsets[i - 1] + num_dofs_ghost_vars[i - 1];
    }

    MPI_Alltoallv(
        vec2ptr(num_dofs_ghost_others), vec2ptr(num_dofs_ghost_others_vars),
        vec2ptr(offsets_others), MPI_INT, vec2ptr(num_dofs_ghost),
        vec2ptr(num_dofs_ghost_vars), vec2ptr(offsets), MPI_INT, this->comm_);

    LOG_DEBUG(2, "Exchanged number of DoF indices");

    int num_ind = 0;
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (int k = 0; k < num_ghost_cells[i]; ++k) {
        for (int l = 0; l < this->nb_fe_; ++l) {
          num_ind +=
              num_dofs_ghost[(offset_ghost_cells[i] + k) * this->nb_fe_ + l];
        }
      }
    }

    std::vector< int > recv_num_per_procs(
        static_cast< size_t >(this->nb_subdom_), 0);

    // Prepare final numer_ghost structure
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (int k = 0; k < num_ghost_cells[i]; ++k) {
        for (int l = 0; l < this->nb_fe_; ++l) {
          numer_ghost[i][ghost_indices[offset_ghost_cells[i] + k]][l].resize(
              num_dofs_ghost[(offset_ghost_cells[i] + k) * this->nb_fe_ + l]);
          recv_num_per_procs[i] +=
              num_dofs_ghost[(offset_ghost_cells[i] + k) * this->nb_fe_ + l];
        }
      }
    }

    LOG_DEBUG(2, "Prepared final numer_ghost structure");

    for (size_t i = 1; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      offsets[i] = offsets[i - 1] + recv_num_per_procs[i - 1];
    }

    // Indices which I receive of others
    std::vector< int > recv_indices(static_cast< size_t >(num_ind), 0);

    // Indices which I send to other
    std::vector< int > sent_indices;
    sent_indices.reserve(static_cast< size_t >(num_ind_others));

    std::vector< int > sent_num_per_procs(
        static_cast< size_t >(this->nb_subdom_), 0);

    // Prepare data to send
    for (size_t i = 0; i < this->nb_subdom_; ++i) {
      for (size_t k = 0; k < num_ghost_cells_others[i]; ++k) {
        for (size_t l = 0; l < this->nb_fe_; ++l) {
          std::vector< int > dof_indices;
          this->get_dofs_on_cell(
              l, ghost_indices_others[offset_ghost_cells_others[i] + k],
              dof_indices);
          for (int m = 0; m < dof_indices.size(); ++m) {
            sent_indices.push_back(dof_indices[m]);
          }
          sent_num_per_procs[i] += dof_indices.size();
        }
      }
    }
    assert(sent_indices.size() == num_ind_others);

    LOG_DEBUG(2, "Prepared DoF indices to be sent");

    for (size_t i = 1; i < static_cast< size_t >(this->nb_subdom_);
         ++i) {
      offsets_others[i] = offsets_others[i - 1] + sent_num_per_procs[i - 1];
    }

    MPI_Alltoallv(vec2ptr(sent_indices), vec2ptr(sent_num_per_procs),
                  vec2ptr(offsets_others), MPI_INT, vec2ptr(recv_indices),
                  vec2ptr(recv_num_per_procs), vec2ptr(offsets), MPI_INT,
                  this->comm_);

    LOG_DEBUG(2, "Exchanged DoF indices");

    // Unpack received data
    int ind = 0;
    for (size_t i = 0; i < this->nb_subdom_; ++i) {
      for (size_t k = 0; k < num_ghost_cells[i]; ++k) {
        for (size_t l = 0; l < this->nb_fe_; ++l) {
          for (int m = 0;
               m <
               num_dofs_ghost[(offset_ghost_cells[i] + k) * this->nb_fe_ + l];
               ++m) {
            numer_ghost[i][ghost_indices[offset_ghost_cells[i] + k]][l][m] =
                recv_indices[ind];
            ++ind;
          }
        }
      }
    }
  }

  // First exchange of temporary Dof Ids for ghost layer dofs

  int max_dof_id =
      *(std::max_element(this->numer_cell_2_global_.begin(), this->numer_cell_2_global_.end()));

  std::vector< int > tmp_permutation(static_cast< size_t >(max_dof_id + 1));
  for (size_t i = 0, e_i = tmp_permutation.size(); i != e_i; ++i) {
    tmp_permutation[i] = static_cast< int >(i);
  }

  for (mesh::EntityIterator it = this->mesh_->begin(this->tdim_),
                            e_it = this->mesh_->end(this->tdim_);
       it != e_it; ++it) {
    int subdomain_index;
    it->get("_sub_domain_", &subdomain_index);

    if (subdomain_index != my_subdom_) {
      int ghost_cell_index;
      it->get("_remote_index_", &ghost_cell_index);

      int hostile_tmp_dof_offset = 0;
      for (size_t s = 0, e_s = static_cast< size_t >(subdomain_index); s != e_s;
           ++s) {
        hostile_tmp_dof_offset += ndofs_with_ghost[s];
      }

      for (size_t fe_ind = 0, e_fe_ind = static_cast< size_t >(this->nb_fe_);
           fe_ind != e_fe_ind; ++fe_ind) {
        int size = this->fe_manager_->get_fe(it->index(), fe_ind)->nb_dof_on_cell();

        // Get temporary dof ids from other subdomain
        std::vector< DofID > ghost_layer_dofs(size);
        for (size_t i = 0, e_i = ghost_layer_dofs.size(); i != e_i; ++i) {
          ghost_layer_dofs[i] =
              numer_ghost[subdomain_index][ghost_cell_index][fe_ind][i];
        }

        // Get corresponding dof ids on ghost layer, which need to be updated
        std::vector< DofID > critical_ghost_layer_dofs;
        this->get_dofs_on_cell(fe_ind, it->index(), critical_ghost_layer_dofs);

        for (size_t i = 0, e_i = critical_ghost_layer_dofs.size(); i != e_i;
             ++i) {
          const int cgld_i = critical_ghost_layer_dofs[i];
          if (cgld_i >= tmp_dof_offset + nb_dofs_on_subdom_[my_subdom_] ||
              cgld_i < tmp_dof_offset) {
            const int gld_i = ghost_layer_dofs[i];
            if (gld_i >= hostile_tmp_dof_offset &&
                gld_i <
                    hostile_tmp_dof_offset + nb_dofs_on_subdom_[subdomain_index]) {
              assert(cgld_i >= 0 && cgld_i < tmp_permutation.size());
              tmp_permutation[cgld_i] = gld_i;
            }
          }
        }
      }
    }
  }

  this->apply_permutation(tmp_permutation);

  LOG_DEBUG(2, " Second permutation done ");

  // Update numer field for all subdomains
  numer_ghost.clear();

  // Create subdomain/ghost cell structure of numer_ghost
  for (mesh::EntityIterator it = this->mesh_->begin(this->tdim_),
                            e_it = this->mesh_->end(this->tdim_);
       it != e_it; ++it) {
    int subdomain_index;
    it->get("_sub_domain_", &subdomain_index);

    if (subdomain_index != my_subdom_) {
      // Access/create (on first access) map for remote subdomain
      numer_ghost[subdomain_index];
      int ghost_cell_index;
      it->get("_remote_index_", &ghost_cell_index);
      // Access/create (on first access) map entry for ghost cell
      numer_ghost[subdomain_index][ghost_cell_index];
    }
  }

  {
    // Number of ghost cells which I share with each other process
    std::vector< int > num_ghost_cells(this->nb_subdom_, 0);
    int total_num_ghost_cells = 0;
    for (int i = 0; i < this->nb_subdom_; ++i) {
      num_ghost_cells[i] = numer_ghost[i].size();

      LOG_DEBUG(2, "[" << my_subdom_ << "]: common ghost cells with process "
                       << i << ": " << num_ghost_cells[i]);

      total_num_ghost_cells += numer_ghost[i].size();
    }

    // Number of ghost cells which other processes share with me
    std::vector< int > num_ghost_cells_others(this->nb_subdom_, 0);

    // Exchange number of ghost cells
    MPI_Alltoall(vec2ptr(num_ghost_cells), 1, MPI_INT,
                 vec2ptr(num_ghost_cells_others), 1, MPI_INT, this->comm_);

#ifndef NDEBUG
    for (int i = 0; i < this->nb_subdom_; ++i) {
      LOG_DEBUG(2, "[" << my_subdom_ << "]: process " << i
                       << " requests number of ghost cells: "
                       << num_ghost_cells_others[i]);
    }
#endif

    // Ghost cell indices which I need of others
    std::vector< int > ghost_indices;
    ghost_indices.reserve(total_num_ghost_cells);
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (typename std::map<
               int, std::map< int, std::vector< int > > >::const_iterator
               it = numer_ghost[i].begin(),
               e_it = numer_ghost[i].end();
           it != e_it; ++it) {
        ghost_indices.push_back(it->first);
      }
    }

    std::vector< int > offsets(this->nb_subdom_, 0);
    for (int i = 1; i < this->nb_subdom_; ++i) {
      offsets[i] = offsets[i - 1] + num_ghost_cells[i - 1];
    }

    std::vector< int > offsets_others(this->nb_subdom_, 0);
    for (int i = 1; i < this->nb_subdom_; ++i) {
      offsets_others[i] = offsets_others[i - 1] + num_ghost_cells_others[i - 1];
    }

    // Ghost cell indices which others need from me
    int total_num_ghost_cells_others = 0;
    for (int i = 0; i < this->nb_subdom_; ++i) {
      total_num_ghost_cells_others += num_ghost_cells_others[i];
    }

    std::vector< int > ghost_indices_others(total_num_ghost_cells_others, 0);

    MPI_Alltoallv(vec2ptr(ghost_indices), vec2ptr(num_ghost_cells),
                  vec2ptr(offsets), MPI_INT, vec2ptr(ghost_indices_others),
                  vec2ptr(num_ghost_cells_others), vec2ptr(offsets_others),
                  MPI_INT, this->comm_);

    LOG_DEBUG(2, "Exchanged ghost cell indices");

    // Number of ghost dofs for all variables and all ghost cells which
    // I need from others
    std::vector< int > num_dofs_ghost(total_num_ghost_cells * this->nb_fe_, 0);

    // Number of ghost dofs for all variables and all ghost cell which
    // others need from me
    std::vector< int > num_dofs_ghost_others(
        total_num_ghost_cells_others * this->nb_fe_, 0);

    std::vector< int > offset_ghost_cells_others(this->nb_subdom_,
                                                 0);
    for (int i = 1; i < this->nb_subdom_; ++i) {
      offset_ghost_cells_others[i] =
          offset_ghost_cells_others[i - 1] + num_ghost_cells_others[i - 1];
    }

    std::vector< int > offset_ghost_cells(this->nb_subdom_, 0);
    for (int i = 1; i < this->nb_subdom_; ++i) {
      offset_ghost_cells[i] =
          offset_ghost_cells[i - 1] + num_ghost_cells[i - 1];
    }

    int num_ind_others = 0;
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (int k = 0; k < num_ghost_cells_others[i]; ++k) {
        for (int l = 0; l < this->nb_fe_; ++l) {
          num_dofs_ghost_others[(offset_ghost_cells_others[i] + k) *
                                    this->nb_fe_ +
                                l] =
              this->nb_dofs_on_cell(
                  l, ghost_indices_others[offset_ghost_cells_others[i] + k]);
          num_ind_others +=
              num_dofs_ghost_others[(offset_ghost_cells_others[i] + k) *
                                        this->nb_fe_ +
                                    l];
        }
      }
    }

    std::vector< int > num_dofs_ghost_others_vars(this->nb_subdom_,
                                                  -1);
    for (int i = 0; i < this->nb_subdom_; ++i) {
      num_dofs_ghost_others_vars[i] = num_ghost_cells_others[i] * this->nb_fe_;
    }

    for (int i = 1; i < this->nb_subdom_; ++i) {
      offsets_others[i] =
          offsets_others[i - 1] + num_dofs_ghost_others_vars[i - 1];
    }

    std::vector< int > num_dofs_ghost_vars(this->nb_subdom_, -1);
    for (int i = 0; i < this->nb_subdom_; ++i) {
      num_dofs_ghost_vars[i] = num_ghost_cells[i] * this->nb_fe_;
    }

    for (int i = 1; i < this->nb_subdom_; ++i) {
      offsets[i] = offsets[i - 1] + num_dofs_ghost_vars[i - 1];
    }

    MPI_Alltoallv(
        vec2ptr(num_dofs_ghost_others), vec2ptr(num_dofs_ghost_others_vars),
        vec2ptr(offsets_others), MPI_INT, vec2ptr(num_dofs_ghost),
        vec2ptr(num_dofs_ghost_vars), vec2ptr(offsets), MPI_INT, this->comm_);

    LOG_DEBUG(2, "Exchanged number of DoF indices");

    int num_ind = 0;
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (int k = 0; k < num_ghost_cells[i]; ++k) {
        for (int l = 0; l < this->nb_fe_; ++l) {
          num_ind +=
              num_dofs_ghost[(offset_ghost_cells[i] + k) * this->nb_fe_ + l];
        }
      }
    }

    std::vector< int > recv_num_per_procs(this->nb_subdom_, 0);

    // Prepare final numer_ghost structure
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (int k = 0; k < num_ghost_cells[i]; ++k) {
        for (int l = 0; l < this->nb_fe_; ++l) {
          numer_ghost[i][ghost_indices[offset_ghost_cells[i] + k]][l].resize(
              num_dofs_ghost[(offset_ghost_cells[i] + k) * this->nb_fe_ + l]);
          recv_num_per_procs[i] +=
              num_dofs_ghost[(offset_ghost_cells[i] + k) * this->nb_fe_ + l];
        }
      }
    }

    LOG_DEBUG(2, "Prepared final numer_ghost structure");

    for (int i = 1; i < this->nb_subdom_; ++i) {
      offsets[i] = offsets[i - 1] + recv_num_per_procs[i - 1];
    }

    // Indices which I receive of others
    std::vector< int > recv_indices(num_ind, 0);

    // Indices which I send to other
    std::vector< int > sent_indices;
    sent_indices.reserve(num_ind_others);

    std::vector< int > sent_num_per_procs(this->nb_subdom_, 0);

    // Prepare data to send
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (int k = 0; k < num_ghost_cells_others[i]; ++k) {
        for (int l = 0; l < this->nb_fe_; ++l) {
          std::vector< int > dof_indices;
          this->get_dofs_on_cell(
              l, ghost_indices_others[offset_ghost_cells_others[i] + k],
              dof_indices);
          for (int m = 0; m < dof_indices.size(); ++m) {
            sent_indices.push_back(dof_indices[m]);
          }
          sent_num_per_procs[i] += dof_indices.size();
        }
      }
    }
    assert(sent_indices.size() == num_ind_others);

    LOG_DEBUG(2, "Prepared DoF indices to be sent");

    for (int i = 1; i < this->nb_subdom_; ++i) {
      offsets_others[i] = offsets_others[i - 1] + sent_num_per_procs[i - 1];
    }

    MPI_Alltoallv(vec2ptr(sent_indices), vec2ptr(sent_num_per_procs),
                  vec2ptr(offsets_others), MPI_INT, vec2ptr(recv_indices),
                  vec2ptr(recv_num_per_procs), vec2ptr(offsets), MPI_INT,
                  this->comm_);

    LOG_DEBUG(2, "Exchanged DoF indices");

    // Unpack received data
    int ind = 0;
    for (int i = 0; i < this->nb_subdom_; ++i) {
      for (int k = 0; k < num_ghost_cells[i]; ++k) {
        for (int l = 0; l < this->nb_fe_; ++l) {
          for (int m = 0;
               m <
               num_dofs_ghost[(offset_ghost_cells[i] + k) * this->nb_fe_ + l];
               ++m) {
            numer_ghost[i][ghost_indices[offset_ghost_cells[i] + k]][l][m] =
                recv_indices[ind];
            ++ind;
          }
        }
      }
    }
  }

  // Fix temporary Dof Ids on ghost layer to correct dof ids (this step might
  // not always be necessary but is essential to ensure correctness in special
  // cases. See documentation file for more information

  max_dof_id = *(std::max_element(this->numer_cell_2_global_.begin(), this->numer_cell_2_global_.end()));

  std::vector< int > update_permutation(max_dof_id + 1);
  for (size_t i = 0, e_i = update_permutation.size(); i != e_i; ++i) {
    update_permutation[i] = i;
  }

  for (mesh::EntityIterator it = this->mesh_->begin(this->tdim_),
                            e_it = this->mesh_->end(this->tdim_);
       it != e_it; ++it) {
    int subdomain_index;
    it->get("_sub_domain_", &subdomain_index);

    if (subdomain_index != my_subdom_) {
      int ghost_cell_index;
      it->get("_remote_index_", &ghost_cell_index);

      for (size_t fe_ind = 0, e_fe_ind = this->nb_fe_; fe_ind != e_fe_ind; ++fe_ind) {
        // Get dofs from other subdomain
        int size = this->fe_manager_->get_fe(it->index(), fe_ind)->nb_dof_on_cell();

        std::vector< DofID > ghost_layer_dofs(size);
        for (size_t i = 0, e_i = ghost_layer_dofs.size(); i != e_i; ++i) {
          ghost_layer_dofs[i] =
              numer_ghost[subdomain_index][ghost_cell_index][fe_ind][i];
        }

        // Get dofs on ghost layer from view of my subdomain
        std::vector< DofID > critical_ghost_layer_dofs;
        this->get_dofs_on_cell(fe_ind, it->index(), critical_ghost_layer_dofs);

        for (size_t i = 0, e_i = critical_ghost_layer_dofs.size(); i != e_i;
             ++i) {
          const int cgld_i = critical_ghost_layer_dofs[i];
          if (cgld_i >= tmp_dof_offset + nb_dofs_on_subdom_[my_subdom_] ||
              cgld_i < tmp_dof_offset) {
            assert(cgld_i >= 0 && cgld_i < update_permutation.size());
            update_permutation[cgld_i] = ghost_layer_dofs[i];
          }
        }
      }
    }
  }

  this->apply_permutation(update_permutation);

  LOG_DEBUG(2, " Third permutation done ");

  // Finaly calculate real dof_offset and correct numer_ field w.r.t. new offset

  my_dof_offset_ = 0;
  for (size_t s = 0, e_s = my_subdom_; s != e_s; ++s) {
    my_dof_offset_ += nb_dofs_on_subdom_[s];
  }

  std::vector< int > old_dof_offsets(nb_subdom_, 0);
  for (size_t s = 0, e_s = nb_subdom_; s != e_s; ++s) {
    for (size_t t = 0; t != s; ++t) {
      old_dof_offsets[s] += ndofs_with_ghost[t];
    }
  }

  std::vector< int > real_dof_offset(nb_subdom_, 0);
  for (size_t s = 0, e_s = nb_subdom_; s != e_s; ++s) {
    for (size_t t = 0; t != s; ++t) {
      real_dof_offset[s] += nb_dofs_on_subdom_[t];
    }
  }

  int size_of_ownerships = old_dof_offsets[nb_subdom_ - 1] +
                           ndofs_with_ghost[nb_subdom_ - 1];
  std::vector< int > ownerships(size_of_ownerships);

  for (size_t s = 0, e_s = nb_subdom_ - 1; s != e_s; ++s) {
    for (size_t i = old_dof_offsets[s]; i < old_dof_offsets[s + 1]; ++i) {
      ownerships[i] = s;
    }
  }

  for (size_t i = old_dof_offsets[nb_subdom_ - 1],
              e_i = old_dof_offsets[nb_subdom_ - 1] +
                    ndofs_with_ghost[nb_subdom_ - 1];
       i < e_i; ++i) {
    ownerships[i] = nb_subdom_ - 1;
  }

  max_dof_id = *(std::max_element(this->numer_cell_2_global_.begin(), this->numer_cell_2_global_.end()));

  std::vector< int > final_permutation(max_dof_id + 1, -1);
  for (size_t i = 0, e_i = this->numer_cell_2_global_.size(); i != e_i; ++i) {
    int owner = ownerships[this->numer_cell_2_global_[i]];

    if (owner != my_subdom_) {
      final_permutation[this->numer_cell_2_global_[i]] =
          this->numer_cell_2_global_[i] - old_dof_offsets[owner] + real_dof_offset[owner];
    } else {
      final_permutation[this->numer_cell_2_global_[i]] =
          this->numer_cell_2_global_[i] - old_dof_offsets[my_subdom_] + my_dof_offset_;
    }
  }

  this->apply_permutation(final_permutation);

  // Last check if Dof Ids are still greater than -1
  for (size_t i = 0, e_i = this->numer_cell_2_global_.size(); i != e_i; ++i) {
    assert(this->numer_cell_2_global_[i] >= 0);
  }

  // Calculate number of dofs for each variable
  std::vector< DofID > tmp;
  tmp.reserve(this->numer_cell_2_global_.size());
  
  for (size_t fe_ind = 0, e_fe_ind = this->nb_fe_; fe_ind != e_fe_ind; ++fe_ind) 
  {
    tmp.clear();
    tmp.reserve(this->numer_cell_2_global_.size());
    
    for (mesh::EntityIterator
         it = this->mesh_->begin(this->fe_manager_->tdim()),
         e_it = this->mesh_->end(this->fe_manager_->tdim());
         it != e_it; ++it) 
    {
      const size_t nb_dof = this->nb_dofs_on_cell(fe_ind, it->index());
      const auto it_b = this->numer_cell_2_global_.begin() + numer_cell_2_global_offsets_[fe_ind][it->index()];

      // loop over DoFs
      for (size_t i = 0; i != nb_dof; ++i) 
      {
        tmp.push_back(*(it_b+i));
      }
    }
    
/*
    int begin_offset = this->numer_cell_2_global_offsets_[fe_ind][0];
    int end_offset;

    if (fe_ind + 1 < this->nb_fe_) {
      end_offset = this->numer_cell_2_global_offsets_[fe_ind + 1][0];
    } else {
      end_offset = this->numer_cell_2_global_.size();
    }

    std::vector< DofID > tmp(end_offset - begin_offset);

    for (size_t i = begin_offset; i < end_offset; ++i) {
      tmp[i - begin_offset] = this->numer_cell_2_global_[i];
    }
*/
    std::sort(tmp.begin(), tmp.end());

    std::vector< DofID >::iterator it = std::unique(tmp.begin(), tmp.end());
    int tmp_size = it - tmp.begin();

    int hostile_dof = 0;
    for (size_t i = 0, e_i = tmp_size; i != e_i; ++i) {
      if (owner_of_dof(tmp[i]) != my_subdom_) {
        hostile_dof++;
      }
    }

    this->local_nb_dofs_for_fe_[fe_ind] -= hostile_dof;
  }

  // consecutive_numbering();
}

#if 0
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::consecutive_numbering() 
{
  // Fill vector local 2 global and global 2 local map
  local2global_.resize(nb_dofs_incl_ghost_);

  for (size_t fe_ind = 0, e_fe_ind = this->nb_fe_; fe_ind != e_fe_ind; ++fe_ind) 
  {
    int local_dof_cntr = 0;
    // First regular mesh without ghost layer
    for (mesh::EntityIterator it = this->mesh_->begin(this->tdim_),
                              e_it = this->mesh_->end(this->tdim_);
         it != e_it; ++it) {
      int subdomain_index;
      it->get("_sub_domain_", &subdomain_index);

      if (subdomain_index == my_subdom_) {
        std::vector< DofID > global_dofs;
        this->get_dofs_on_cell(fe_ind, it->index(), global_dofs);

        for (size_t i = 0, e_i = global_dofs.size(); i != e_i; ++i) {
          if (global2local_.find(global_dofs[i]) == global2local_.end()) {
            global2local_.insert(
                std::pair< DofID, DofID >(global_dofs[i], local_dof_cntr));
            ++local_dof_cntr;
          }
        }
      }
    }
    // Next: Ghost layer
    for (mesh::EntityIterator it = this->mesh_->begin(this->tdim_),
                              e_it = this->mesh_->end(this->tdim_);
         it != e_it; ++it) {
      int subdomain_index;
      it->get("_sub_domain_", &subdomain_index);

      if (subdomain_index != my_subdom_) {
        std::vector< DofID > global_dofs;
        this->get_dofs_on_cell(fe_ind, it->index(), global_dofs);

        for (size_t i = 0, e_i = global_dofs.size(); i != e_i; ++i) {
          if (global2local_.find(global_dofs[i]) == global2local_.end()) {
            global2local_.insert(
                std::pair< DofID, DofID >(global_dofs[i], local_dof_cntr));
            ++local_dof_cntr;
          }
        }
      }
    }
  }

  // Fill local2global
  std::map< DofID, DofID >::iterator it = global2local_.begin();
  while (it != global2local_.end()) 
  {
    local2global_[(*it).second] = (*it).first;
    ++it;
  }

  // Fill numer_cell_2_local_ field
  this->numer_cell_2_local_.resize(this->numer_cell_2_global_.size(), -1);
  int offset = 0;
  for (size_t fe_ind = 0, e_fe_ind = this->nb_fe_; fe_ind != e_fe_ind; ++fe_ind) 
  {
    for (mesh::EntityIterator it = this->mesh_->begin(this->tdim_),
                              e_it = this->mesh_->end(this->tdim_); it != e_it; ++it) 
    {
      std::vector< DofID > global_dofs;
      this->get_dofs_on_cell(fe_ind, it->index(), global_dofs);

      for (size_t i = 0, e_i = global_dofs.size(); i != e_i; ++i) 
      {
        DofID local_dof;
        global2local(global_dofs[i], &local_dof);
        numer_cell_2_local_[i + offset] = local_dof;
      }

      offset += global_dofs.size();
    }
  }
  
#ifndef NDEBUG
  int diff = 0;
  for (size_t l=0; l<numer_cell_2_local_.size(); ++l)
  {
    diff += std::abs(this->numer_cell_2_local_[l] -  this->numer_cell_2_global_[l]);
  }
  std::cout << " difference numer_sd <> numer_cell_2_subdom : " << diff << std::endl;
  assert (diff == 0);
#endif  
}
#endif

#if 0
template < class DataType, int DIM >
void DofPartition< DataType, DIM >::apply_permutation_local(const std::vector< DofID > &permutation) 
{
  for (size_t i = 0, e_i = numer_cell_2_local_.size(); i != e_i; ++i) 
  {
    numer_cell_2_local_[i] = permutation[numer_cell_2_local_[i]];
  }

  // Fix local2global and global2local
  std::vector< DofID > l2g_backup(local2global_);
  for (size_t i = 0, e_i = local2global_.size(); i != e_i; ++i) 
  {
    local2global_[i] = l2g_backup[permutation[local2global_[i]]];
  }

  global2local_.clear();
  for (size_t i = 0, e_i = local2global_.size(); i != e_i; ++i) 
  {
    global2local_.insert(std::pair< DofID, DofID >(local2global_[i], i));
  }
}
#endif

/// OK
template < class DataType, int DIM > 
bool DofPartition< DataType, DIM >::check_mesh() 
{
  bool ret = true;

  // loop over mesh cells
  for (mesh::EntityIterator it = mesh_->begin(tdim()), e_it = mesh_->end(tdim()); it != e_it; ++it) 
  {
    // get coordinates of current cell
    std::vector< mesh::Coordinate > coords;
    it->get_coordinates(coords);
    // check whether vertex coordinates are conform to numbering rule of cell
    // type
    if (!static_cast< bool >(
            it->cell_type().check_cell_geometry(coords, it->gdim()))) {
      std::cout << "ERROR: vertex numbering of cell " << it->index()
                << " is not conform to cell type." << std::endl;
      ret = false;
    }
  }
  return ret;
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::update_number_of_dofs(DofID& total_nb_dof,
                                                          std::vector<DofID>& nb_dof_for_fe,
                                                          const std::string &description) const
{
  // Calculate number of DoFs
  total_nb_dof = 0;
  for (size_t i = 0, e_i = numer_cell_2_global_.size(); i != e_i; ++i) 
  {
    if (numer_cell_2_global_[i] > total_nb_dof) 
    {
      total_nb_dof = numer_cell_2_global_[i];
    }
  }
  ++total_nb_dof;

  // Calculate number of Dofs for each variable
  nb_dof_for_fe.clear();
  nb_dof_for_fe.resize(nb_fe_, 0);
  std::vector< DofID > tmp;
  tmp.reserve(this->numer_cell_2_global_.size());
    
  for (size_t fe_ind = 0; fe_ind != nb_fe_; ++fe_ind) 
  {
    tmp.clear();
    tmp.reserve(this->numer_cell_2_global_.size());
    
    for (mesh::EntityIterator
         it = this->mesh_->begin(this->fe_manager_->tdim()),
         e_it = this->mesh_->end(this->fe_manager_->tdim());
         it != e_it; ++it) 
    {
      const size_t nb_dof = this->nb_dofs_on_cell(fe_ind, it->index());
      const auto it_b = this->numer_cell_2_global_.begin() + numer_cell_2_global_offsets_[fe_ind][it->index()];

      // loop over DoFs
      for (size_t i = 0; i != nb_dof; ++i) 
      {
        tmp.push_back(*(it_b+i));
      }
    }
    
/*
    int begin_offset = numer_cell_2_global_offsets_[fe_ind][0];
    int end_offset;

    if (fe_ind < nb_fe_ - 1) 
    {
      end_offset = numer_cell_2_global_offsets_[fe_ind + 1][0];
    } 
    else 
    {
      end_offset = numer_cell_2_global_.size();
    }

    std::vector< DofID > tmp(end_offset - begin_offset);

    for (size_t i = 0, e_i = end_offset - begin_offset; i < e_i; ++i) 
    {
      tmp[i] = numer_cell_2_global_[i + begin_offset];
    }
*/
    std::sort(tmp.begin(), tmp.end());

    std::vector< DofID >::iterator it = std::unique(tmp.begin(), tmp.end());
    int tmp_size = it - tmp.begin();

    nb_dof_for_fe[fe_ind] = tmp_size;
  }
#if 0
            int rank = -1;
            MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
            if ( ( number_of_dofs_total_old != local_nb_dofs_total_ ) &&
                 ( rank == 0 ) )
                std::cout << "#Dofs: " << local_nb_dofs_total_
                    << " " << description << std::endl;
#endif
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::apply_permutation( const std::vector< DofID > &permutation, 
                                                       const std::string &description) 
{
  // apply permutation to cell2global
  //
  // DoF IDs are used in numer_ only
  for (size_t i = 0, e_i = numer_cell_2_global_.size(); i != e_i; ++i) 
  {
    numer_cell_2_global_[i] = permutation[numer_cell_2_global_[i]];
  }
#if 0
CHANGE
permute numer_cell_2_factor
#endif

  // apply permutation to DofInterpolation
  dof_interpolation().apply_permutation(permutation);

  // calculate number of dofs, as this could have changed
  update_number_of_dofs(this->local_nb_dofs_total_,
                        this->local_nb_dofs_for_fe_, 
                        description);
}

template < class DataType, int DIM >
void DofPartition< DataType, DIM >::print_numer() const 
{ 
  for (size_t i = 0, e_i = numer_cell_2_global_.size(); i != e_i; ++i) 
  {
    std::cout << i << "\t ->    " << numer_cell_2_global_[i] << " : " /*<< numer_cell_2_factor_[i]*/ << std::endl;
  }
}

template < class DataType, int DIM >
void permute_constrained_dofs_to_end(DofPartition< DataType, DIM > &dof) 
{
  const int num_dofs = dof.nb_dofs();
  std::vector< int > permutation(num_dofs, -1);
  for (int i = 0; i != num_dofs; ++i) 
  {
    permutation[i] = i;
  }

  const DofInterpolation<DataType> &dof_interp = dof.dof_interpolation();

  // Compute permutation that places constrained dofs after
  // unconstrained dofs. This is accomplished by keeping two
  // references, one to the first constrained dof, and one to the
  // last unconstrained dof. These are updated iteratively, and the
  // positions in the permutation are swapped until
  int first_constrained = 0;
  int last_unconstrained = num_dofs - 1;

  while (true) 
  {
    // make first_constrained point to first constrained dof
    while (first_constrained < last_unconstrained &&
           dof_interp.count(first_constrained) == 0) {
      ++first_constrained;
    }

    // make last_unconstrained point to last unconstrained dof
    while (first_constrained < last_unconstrained &&
           dof_interp.count(last_unconstrained) == 1) {
      --last_unconstrained;
    }

    // if we are not done, swap the two positions
    if (first_constrained < last_unconstrained) {
      std::swap(permutation[first_constrained],
                permutation[last_unconstrained]);

      // update pointers here
      ++first_constrained;
      --last_unconstrained;
    } else {
      // we are done, break out of loop
      break;
    }
  }

  // Apply the permutation
  dof.apply_permutation(permutation);
}

template class DofPartition< double, 3 >;
template class DofPartition< double, 2 >;
template class DofPartition< double, 1 >;

template class DofPartition< float, 3 >;
template class DofPartition< float, 2 >;
template class DofPartition< float, 1 >;

} // namespace doffem
} // namespace hiflow
