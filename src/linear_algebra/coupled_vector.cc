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

/// @author Chandramowli Subramanian, Nico Trost, Dimitar Lukarski, Martin
/// Wlotzka, Simon Gawlok

#include <cassert>
#include <cmath>
#include <cstdlib>

#include "config.h"

#include "common/pointers.h"
#include "linear_algebra/coupled_vector.h"
#include "lmp/init_vec_mat.h"

#ifdef WITH_HDF5
#include "common/hdf5_tools.h"
#include "hdf5.h"
#endif

#include "common/log.h"
#include "tools/mpi_tools.h"

namespace hiflow {
namespace la {

template < class DataType > CoupledVector< DataType >::CoupledVector() {
  this->la_couplings_ = nullptr;
  this->comm_ = MPI_COMM_NULL;
  this->nb_procs_ = -1;
  this->my_rank_ = -1;
  this->interior_ = nullptr;
  this->ghost_ = nullptr;
  this->ownership_begin_ = -1;
  this->ownership_end_ = -1;
  this->checked_for_dof_partition_ = false;
  this->initialized_ = false;
  this->parcom_.reset();
  this->nb_sends_ = 0;
  this->nb_recvs_ = 0;
}

template < class DataType > CoupledVector< DataType >::~CoupledVector() {
  this->Clear();
  int is_finalized;
  this->parcom_.reset();

  MPI_Finalized(&is_finalized);
  if (is_finalized == 0) {
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
    }
  }
}

template < class DataType >
void CoupledVector< DataType >::Init(const MPI_Comm &comm,
                                     const LaCouplings &cp, 
									                   PLATFORM plat,
                                     IMPLEMENTATION impl,
                                     const SYSTEM &my_system) 
{
  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
  }

  assert(comm != MPI_COMM_NULL);
  // MPI communicator

  // determine nb. of processes
#ifndef NDEBUG
  int info =
#endif
    MPI_Comm_size(comm, &(this->nb_procs_));
  assert(info == MPI_SUCCESS);
  assert(this->nb_procs_ > 0);

  // retrieve my rank
#ifndef NDEBUG
  info =
#endif
    MPI_Comm_rank(comm, &(this->my_rank_));
  assert(info == MPI_SUCCESS);
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->nb_procs_);

  // info = MPI_Comm_split ( comm, 0, this->my_rank_, &( this->comm_ ) );
#ifndef NDEBUG
  info =
#endif
    MPI_Comm_dup(comm, &(this->comm_));
  assert(info == MPI_SUCCESS);

  this->parcom_.reset();
  this->parcom_ = std::shared_ptr<ParCom>(new ParCom(this->comm_));

  // couplings
  this->la_couplings_ = &cp;
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());

  this->Init_la_system(plat, impl, my_system);
  this->InitStructure();

  this->initialized_ = true;
}

template < class DataType >
void CoupledVector< DataType >::Init_la_system(PLATFORM plat,
                                               IMPLEMENTATION impl,
                                               const SYSTEM &my_system) 
{
  // first clear old data
  this->Clear();
  
  // init interior
  this->interior_ = init_vector< DataType >(0, "interior", plat, impl);
  assert(this->interior_ != nullptr);
  // init ghost
  this->ghost_ = init_vector< DataType >(0, "ghost", plat, impl);
  assert(this->ghost_ != nullptr);
  
  if (plat == OPENCL) 
  {
    // init interior
    this->interior_ = init_vector< DataType >(0, "interior", plat, impl, my_system);
    assert(this->interior_ != nullptr);
    // init ghost
    this->ghost_ = init_vector< DataType >(0, "ghost", plat, impl, my_system);
    assert(this->ghost_ != nullptr);
  } 
  else 
  {
    // init interior
    this->interior_ = init_vector< DataType >(0, "interior", plat, impl);
    assert(this->interior_ != nullptr);
    // init ghost
    this->ghost_ = init_vector< DataType >(0, "ghost", plat, impl);
    assert(this->ghost_ != nullptr);
  }
}

template < class DataType > 
void CoupledVector< DataType >::InitStructure() 
{
  assert(this->comm_ != MPI_COMM_NULL);
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->nb_procs_);
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());
  assert(this->interior_ != nullptr);
  assert(this->ghost_ != nullptr);
  assert (this->parcom_);

  // compute ownership range
  this->ComputeOwnershipRange();

  // init structure of interior
  this->interior_->Clear();
  this->interior_->Init(this->la_couplings_->nb_dofs(this->my_rank_),
                        "interior");

  // set border indices
  this->interior_->set_indexset(this->la_couplings_->border_indices(),
                                this->la_couplings_->size_border_indices());

  // init structure of ghost
  if (this->la_couplings_->size_ghost() > 0) {
    this->ghost_->Clear();
    this->ghost_->Init(this->la_couplings_->size_ghost(), "ghost");
  }

  // prepare for communication
  this->nb_sends_ = 0;
  this->nb_recvs_ = 0;

  for (int id = 0; id < this->nb_procs_; ++id) 
  {
    if (this->la_couplings_->border_offsets(id + 1) - 
        this->la_couplings_->border_offsets(id) > 0) 
    {
      this->nb_sends_++;
    }
    if (this->la_couplings_->ghost_offsets(id + 1) -
        this->la_couplings_->ghost_offsets(id) > 0) 
    {
      this->nb_recvs_++;
    }
  }
  this->mpi_req_.resize(this->nb_sends_ + this->nb_recvs_);
  this->mpi_stat_.resize(this->nb_sends_ + this->nb_recvs_);
  this->border_val_.resize(this->la_couplings_->size_border_indices());
  this->ghost_val_.resize(this->la_couplings_->size_ghost());
}

template < class DataType > void CoupledVector< DataType >::Zeros() {
  assert(this->interior_ != nullptr);
  assert(this->ghost_ != nullptr);

  this->interior_->Zeros();
  this->ghost_->Zeros();
}

template < class DataType >
void CoupledVector< DataType >::SetToValue(DataType val) {
  assert(this->interior_ != nullptr);
  assert(this->ghost_ != nullptr);

  for (int i = 0; i < this->size_local(); ++i) {
    this->interior_->SetValues(&i, 1, &val);
  }

  for (int i = 0; i < this->size_local_ghost(); ++i) {
    this->ghost_->SetValues(&i, 1, &val);
  }
}

template < class DataType >
void CoupledVector< DataType >::SetToValue(const int *indices, 
                                           const int size_indices,
                                           const DataType value) 
{
  assert(this->interior_ != nullptr);

  for (int i = 0; i < size_indices; ++i) 
  {
    if (this->ownership_begin_ <= indices[i] && indices[i] < this->ownership_end_)
    {
      this->interior_->SetValue(indices[i] - this->ownership_begin_, value);
    }
  }
}

template < class DataType >
void CoupledVector< DataType >::Add(int global_dof_id, DataType val) {
  assert(this->interior_ != nullptr);
  assert(this->ownership_begin_ <= global_dof_id);
  assert(global_dof_id < this->ownership_end_);

  interior_->add_value(global_dof_id - this->ownership_begin_, val);
}

template < class DataType >
void CoupledVector< DataType >::Add(const int *indices, 
                                    const int size_indices,
                                    const DataType *values) {
  assert(this->interior_ != nullptr);

  this->_shifted_indices.clear();
  this->_shifted_indices.reserve(size_indices);
  this->_insert_values.clear();
  this->_insert_values.reserve(size_indices);
  for (int i = 0; i < size_indices; ++i) 
  {
    //assert(this->ownership_begin_ <= indices[i]);
    //assert(indices[i] < this->ownership_end_);
    if (this->ownership_begin_ <= indices[i] && indices[i] < this->ownership_end_)
    {
      this->_shifted_indices.push_back(indices[i] - this->ownership_begin_);
      this->_insert_values.push_back(values[i]);
    }
  }
  interior_->add_values(vec2ptr(this->_shifted_indices), size_indices, vec2ptr(this->_insert_values));
}

template < class DataType >
DataType CoupledVector< DataType >::Dot(const Vector< DataType > &vec) const {

  const CoupledVector< DataType > *cv_vec;

  cv_vec = dynamic_cast< const CoupledVector< DataType > * >(&vec);

  if (cv_vec != 0) {
    return this->Dot(*cv_vec);
  } else {
    LOG_ERROR("Called CoupledVector::Dot with incompatible argument type.");
    quit_program()
    return 0.0;
  }
}

template < class DataType >
DataType
CoupledVector< DataType >::Dot(const CoupledVector< DataType > &vec) const {
  assert(this->comm_ != MPI_COMM_NULL);
  assert(this->interior_ != nullptr);
  assert(this->parcom_);
  
  // local dot product
  DataType dot_local = this->interior_->Dot(vec.interior());

  // now sum up
  DataType dot_global;
  int err = this->parcom_->sum(dot_local, dot_global);
  assert (err == 0);
  
  return dot_global;
}

template < class DataType >
void CoupledVector< DataType >::Axpy(const Vector< DataType > &vec,
                                     const DataType alpha) {
  const CoupledVector< DataType > *cv_vec;

  cv_vec = dynamic_cast< const CoupledVector< DataType > * >(&vec);

  if (cv_vec != 0) {
    this->Axpy(*cv_vec, alpha);
  } else {
    LOG_ERROR("Called CoupledVector::Axpy with incompatible argument type.");
    quit_program();
  }
}

template < class DataType >
void CoupledVector< DataType >::Axpy(const CoupledVector< DataType > &vec,
                                     const DataType alpha) {
  assert(this->interior_ != nullptr);

  this->interior_->Axpy(vec.interior(), alpha);
}

template < class DataType >
void CoupledVector< DataType >::ScaleAdd(const Vector< DataType > &vec,
    const DataType alpha) {
  const CoupledVector< DataType > *cv_vec;

  cv_vec = dynamic_cast< const CoupledVector< DataType > * >(&vec);

  if (cv_vec != 0) {
    this->ScaleAdd(*cv_vec, alpha);
  } else {
    LOG_ERROR(
        "Called CoupledVector::ScaleAdd with incompatible argument type.");
    quit_program();
  }
}

template < class DataType >
void CoupledVector< DataType >::ScaleAdd(const CoupledVector< DataType > &vec,
    const DataType alpha) {
  assert(this->interior_ != nullptr);

  this->interior_->ScaleAdd(alpha, vec.interior());
}

template < class DataType >
void CoupledVector< DataType >::Scale(const DataType alpha) {
  assert(this->interior_ != nullptr);

  this->interior_->Scale(alpha);
}

template < class DataType > DataType CoupledVector< DataType >::Norm1() const {
  DataType norm1_local = this->interior_->Norm1();
  assert(norm1_local >= 0.0);
  DataType norm1_global;
  
  int info = this->parcom_->sum(norm1_local, norm1_global);

  assert(info == MPI_SUCCESS);
  assert(norm1_global >= 0.0);
  return norm1_global;
}

template < class DataType >
DataType CoupledVector< DataType >::NormMax() const {
  DataType norm_max_local = this->interior_->NormMax();
  assert(norm_max_local >= 0.0);
  DataType norm_max_global;

  int info = this->parcom_->max(norm_max_local, norm_max_global);

  assert(info == MPI_SUCCESS);
  assert(norm_max_global >= 0.0);
  return norm_max_global;
}

template < class DataType > DataType CoupledVector< DataType >::Norm2() const {
  const DataType norm_sq = this->Dot(*this);
  return sqrt(norm_sq);
}

template < class DataType >
void CoupledVector< DataType >::GetValues(const int *indices,
    const int size_indices,
    DataType *values) const {
  assert(this->interior_ != nullptr);
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());

  if (size_indices > 0) {
    assert(indices != 0);
    assert(values != 0);

    // extract interior and ghost indices (and if possible pp inidices)
    //this->mut_indices_interior_.clear();
    this->mut_indices_interior_.resize(size_indices);
    int ind_interior = 0;

    //this->mut_indices_ghost_.clear();
    this->mut_indices_ghost_.resize(size_indices);
    int ind_ghost = 0;

    // transform indices according to local numbering of interior and ghost
    for (int i = 0; i < size_indices; i++) {
      // interior
      if (this->ownership_begin_ <= indices[i] &&
          indices[i] < this->ownership_end_) {
        this->mut_indices_interior_[ind_interior] = indices[i] - this->ownership_begin_;
        ind_interior++;
      }

      // ghost
      else if (this->la_couplings_->global2offdiag().find(indices[i]) !=
               this->la_couplings_->global2offdiag().end()) {
        this->mut_indices_ghost_[ind_ghost] =
          this->la_couplings_->Global2Offdiag(indices[i]);
        ind_ghost++;
      }
    }

    // extract values
    //this->mut_values_interior_.clear();
    this->mut_values_interior_.resize(ind_interior);
    if (ind_interior > 0) {
      this->interior_->GetValues(&this->mut_indices_interior_.front(), ind_interior,
                                 &this->mut_values_interior_.front());
    }

    //this->mut_values_ghost_.clear();
    this->mut_values_ghost_.resize(ind_ghost);

    if (ind_ghost > 0) {
      this->ghost_->GetValues(&this->mut_indices_ghost_.front(), ind_ghost,
                              &this->mut_values_ghost_.front());
    }
    // put values together
    for (int i = 0; i < size_indices; i++) {
      // interior
      if (this->ownership_begin_ <= indices[size_indices - i - 1] &&
          indices[size_indices - i - 1] < this->ownership_end_) {
        values[size_indices - i - 1] = this->mut_values_interior_[ind_interior - 1];
        --ind_interior;
      }

      // ghost
      else if (this->la_couplings_->global2offdiag().find(
                 indices[size_indices - i - 1]) !=
               this->la_couplings_->global2offdiag().end()) {
        values[size_indices - i - 1] = this->mut_values_ghost_[ind_ghost - 1];
        --ind_ghost;
      }
      // zeros
      else {
        values[size_indices - i - 1] = static_cast< DataType >(0.);
      }
    }

    assert(ind_interior == 0);
    assert(ind_ghost == 0);
  }
}

template < class DataType >
DataType CoupledVector< DataType >::GetValue(const int index) const {
// Note (Philipp G): Is this assert necessary?
//  assert(index >= this->ownership_begin_);
//  assert(index < this->ownership_end_);
  DataType result;
  this->GetValues(&index, 1, &result);
  return result;
}

template < class DataType >
void CoupledVector< DataType >::GetLocalValues(DataType *values) const {
  assert(this->interior_ != nullptr);

  this->interior_->GetBlockValues(0, this->size_local(), values);
}

template < class DataType >
void CoupledVector< DataType >::GetLocalValues(const int *indices,
                                               const int size_indices,
                                               DataType *values) const 
{
  assert(this->interior_ != nullptr);
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());

  if (size_indices > 0) {
    assert(indices != 0);
    assert(values != 0);

    // extract interior and ghost indices (and if possible pp inidices)
    std::vector< int > indices_interior(size_indices);
    int ind_interior = 0;

    // transform indices according to local numbering of interior and ghost
    for (int i = 0; i < size_indices; i++) {
      // interior
      if (this->ownership_begin_ <= indices[i] &&
          indices[i] < this->ownership_end_) {
        indices_interior[ind_interior] = indices[i] - this->ownership_begin_;
        ind_interior++;
      }

    }

    // extract values
    std::vector< DataType > values_interior(ind_interior);
    if (ind_interior > 0) {
      this->interior_->GetValues(&indices_interior.front(), ind_interior,
                                 &values_interior.front());
    }

    // put values together
    for (int i = 0; i < size_indices; i++) {
      // interior
      if (this->ownership_begin_ <= indices[size_indices - i - 1] &&
          indices[size_indices - i - 1] < this->ownership_end_) {
        values[size_indices - i - 1] = values_interior[ind_interior - 1];
        --ind_interior;
      }
    }

    assert(ind_interior == 0);
  }
}

template < class DataType >
void CoupledVector< DataType >::SetValue(const int index,
                                         const DataType value) 
{

  assert(index >= this->ownership_begin_);
  assert(index < this->ownership_end_);

  const int local_index = index - this->ownership_begin_;
  this->interior_->SetValues(&local_index, 1, &value);
}

template < class DataType >
void CoupledVector< DataType >::SetValues(const int *indices,
                                          const int size_indices,
                                          const DataType *values) 
{
  assert(this->interior_ != nullptr);

  this->_shifted_indices.clear();
  this->_shifted_indices.reserve(size_indices);
  this->_insert_values.clear();
  this->_insert_values.reserve(size_indices);
  for (int i = 0; i < size_indices; ++i) 
  {
    //assert(this->ownership_begin_ <= indices[i]);
    //assert(indices[i] < this->ownership_end_);
    if (this->ownership_begin_ <= indices[i] && indices[i] < this->ownership_end_)
    {
      this->_shifted_indices.push_back(indices[i] - this->ownership_begin_);
      this->_insert_values.push_back(values[i]);
    }
  }
  interior_->SetValues(vec2ptr(this->_shifted_indices), size_indices, vec2ptr(this->_insert_values));
}

template < class DataType >
void CoupledVector< DataType >::SetLocalValues(const DataType *values) {
  assert(this->interior_ != nullptr);

  this->interior_->SetBlockValues(0, this->size_local(), values);
}

template < class DataType >
void CoupledVector< DataType >::SetGhostValues(const DataType *values) {
  assert(this->ghost_ != nullptr);

  this->ghost_->SetBlockValues(0, this->size_local_ghost(), values);
}

template < class DataType >
void CoupledVector< DataType >::Gather(int recv_id, DataType *values) const {
  assert(this->interior_ != nullptr);
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());

  if (this->interior().get_platform() == CPU) {
    // dynamic cast to CPU vector in order to access buffer
    const CPU_lVector< DataType > *casted_vec =
      dynamic_cast< const CPU_lVector< DataType > * >(this->interior_);
    assert(casted_vec != nullptr);

    int comm_size = this->parcom_->size();

    int tag = 1;

    // receive
    if (this->my_rank_ == recv_id) {

      int recv_index_begin;
      int recv_index_end = 0;

      for (int id = 0; id < comm_size; ++id) {
        recv_index_begin = recv_index_end;
        recv_index_end += this->la_couplings_->nb_dofs(id);

        if (id != this->my_rank_) {        
          int info = this->parcom_->recv(&values[recv_index_begin], recv_index_end - recv_index_begin, id, tag);
          assert(info == MPI_SUCCESS);
        }

        else {
          for (int i = recv_index_begin; i < recv_index_end; i++) {
            values[i] = casted_vec->buffer[i - recv_index_begin];
          }
        }
      }
    }

    // send
    else {
      this->parcom_->send(casted_vec->buffer, this->size_local(), recv_id, tag);
    }

  }

  else {
    LOG_ERROR("CoupledVector::Gather: not supported on non-CPU platforms.");
    quit_program();
  }
}

template < class DataType >
Vector< DataType > *CoupledVector< DataType >::Clone() const {
  CoupledVector< DataType > *clone = new CoupledVector< DataType >();

  clone->CloneFrom(*this);

  return clone;
}

template < class DataType >
void CoupledVector< DataType >::CloneFrom(
  const CoupledVector< DataType > &vec) {
  if (this != &vec) {
    // clone vector and structure
    this->CloneFromWithoutContent(vec);

    // copy entries
    this->interior_->CopyFrom(vec.interior());
    this->ghost_->CopyFrom(vec.ghost());
  }
}

template < class DataType >
void CoupledVector< DataType >::CopyFrom(const CoupledVector< DataType > &vec) {
  if (this != &vec) {
    assert(this->nb_procs_ == vec.nb_procs());
    assert(this->my_rank_ == vec.my_rank());
    assert(this->ownership_begin_ == vec.ownership_begin());
    assert(this->ownership_end_ == vec.ownership_end());

    this->interior_->CopyFrom(vec.interior());
    this->ghost_->CopyFrom(vec.ghost());
  }
}

template < class DataType >
void CoupledVector< DataType >::CastInteriorFrom(
  const CoupledVector< double > &other) {
  assert(this->nb_procs_ == other.nb_procs());
  assert(this->my_rank_ == other.my_rank());
  assert(this->ownership_begin_ == other.ownership_begin());
  assert(this->ownership_end_ == other.ownership_end());

  this->interior_->CastFrom(other.interior());

  return;
}

template < class DataType >
void CoupledVector< DataType >::CastInteriorFrom(
  const CoupledVector< float > &other) {
  assert(this->nb_procs_ == other.nb_procs());
  assert(this->my_rank_ == other.my_rank());
  assert(this->ownership_begin_ == other.ownership_begin());
  assert(this->ownership_end_ == other.ownership_end());

  this->interior_->CastFrom(other.interior());

  return;
}

template < class DataType >
void CoupledVector< DataType >::CastInteriorTo(
  CoupledVector< double > &other) const {
  assert(this->nb_procs_ == other.nb_procs());
  assert(this->my_rank_ == other.my_rank());
  assert(this->ownership_begin_ == other.ownership_begin());
  assert(this->ownership_end_ == other.ownership_end());

  this->interior_->CastTo(other.interior());

  return;
}

template < class DataType >
void CoupledVector< DataType >::CastInteriorTo(
  CoupledVector< float > &other) const {
  assert(this->nb_procs_ == other.nb_procs());
  assert(this->my_rank_ == other.my_rank());
  assert(this->ownership_begin_ == other.ownership_begin());
  assert(this->ownership_end_ == other.ownership_end());

  this->interior_->CastTo(other.interior());

  return;
}

template < class DataType >
void CoupledVector< DataType >::CopyTo(CoupledVector< DataType > &vec) const {
  if (this != &vec) {
    assert(this->nb_procs_ == vec.nb_procs());
    assert(this->my_rank_ == vec.my_rank());
    assert(this->ownership_begin_ == vec.ownership_begin());
    assert(this->ownership_end_ == vec.ownership_end());

    this->interior_->CopyTo(vec.interior());
    this->ghost_->CopyTo(vec.ghost());
  }
}

template < class DataType >
void CoupledVector< DataType >::CopyInteriorFrom(
  const CoupledVector< DataType > &vec) {
  if (this != &vec) {
    assert(this->nb_procs_ == vec.nb_procs());
    assert(this->my_rank_ == vec.my_rank());
    assert(this->ownership_begin_ == vec.ownership_begin());
    assert(this->ownership_end_ == vec.ownership_end());

    this->interior_->CopyFrom(vec.interior());
  }
}

template < class DataType >
void CoupledVector< DataType >::CopyFromWithoutGhost(
  const CoupledVector< DataType > &vec) {

  this->CopyInteriorFrom(vec);
}

template < class DataType >
void CoupledVector< DataType >::CloneFromWithoutContent(
  const CoupledVector< DataType > &vec) {
  if (this != &vec) {
    assert(vec.is_initialized());
    this->Clear();
#ifndef NDEBUG
    int info = 0;
#endif
    if (this->comm_ != MPI_COMM_NULL) {
#ifndef NDEBUG
      info =
#endif
        MPI_Comm_free(&this->comm_);
      assert(info == MPI_SUCCESS);
    }
    this->nb_procs_ = vec.nb_procs();
    this->my_rank_ = vec.my_rank();

    // info = MPI_Comm_split ( vec.comm ( ), 0, this->my_rank_, &( this->comm_ )
    // );
#ifndef NDEBUG
    info =
#endif
      MPI_Comm_dup(vec.comm(), &(this->comm_));

    assert(info == MPI_SUCCESS);
    this->la_couplings_ = &(vec.la_couplings());
    assert(this->la_couplings_ != nullptr);
    assert(this->la_couplings_->initialized());
    this->ownership_begin_ = vec.ownership_begin();
    this->ownership_end_ = vec.ownership_end();
    this->nb_sends_ = vec.nb_sends();
    this->nb_recvs_ = vec.nb_recvs();
    this->mpi_req_ = vec.mpi_req();
    this->mpi_stat_ = vec.mpi_stat();
    this->border_val_ = vec.border_val();
    this->ghost_val_ = vec.ghost_val();

    // clone the vectors
    this->interior_ = vec.interior().CloneWithoutContent();
    this->ghost_ = vec.ghost().CloneWithoutContent();

    // copy the structure
    this->interior_->CopyStructureFrom(vec.interior());
    this->ghost_->CopyStructureFrom(vec.ghost());

    this->interior_->set_indexset(this->la_couplings_->border_indices(),
                                  this->la_couplings_->size_border_indices());

	this->parcom_.reset();
	this->parcom_ = std::shared_ptr<ParCom>(new ParCom(this->comm_));
    this->initialized_ = true;
  }
}

template < class DataType >
void CoupledVector< DataType >::CopyStructureFrom(
  const CoupledVector< DataType > &vec) {
  if (this != &vec) {

    // no Clear() !
    this->nb_procs_ = vec.nb_procs();
    this->my_rank_ = vec.my_rank();
#ifndef NDEBUG
    int info = 0;
#endif
    if (this->comm_ != MPI_COMM_NULL) {
#ifndef NDEBUG
      info =
#endif
        MPI_Comm_free(&this->comm_);
      assert(info == MPI_SUCCESS);
    }
    // info = MPI_Comm_split ( vec.comm ( ), 0, this->my_rank_, &( this->comm_ )
    // );
#ifndef NDEBUG
    info =
#endif
      MPI_Comm_dup(vec.comm(), &(this->comm_));

    assert(info == MPI_SUCCESS);
    this->la_couplings_ = &(vec.la_couplings());
    assert(this->la_couplings_ != nullptr);
    assert(this->la_couplings_->initialized());
    this->ownership_begin_ = vec.ownership_begin();
    this->ownership_end_ = vec.ownership_end();
    this->nb_sends_ = vec.nb_sends();
    this->nb_recvs_ = vec.nb_recvs();
    this->mpi_req_ = vec.mpi_req();
    this->mpi_stat_ = vec.mpi_stat();
    this->border_val_ = vec.border_val();
    this->ghost_val_ = vec.ghost_val();
    this->interior_->CopyStructureFrom(vec.interior());
    this->ghost_->CopyStructureFrom(vec.ghost());

    this->interior_->set_indexset(this->la_couplings_->border_indices(),
                                  this->la_couplings_->size_border_indices());
                      
    this->parcom_.reset();
    this->parcom_ = std::shared_ptr<ParCom>(new ParCom(this->comm_));         
  }
}

template < class DataType > void CoupledVector< DataType >::Clear() {
  // clear interior
  if (this->interior_ != nullptr) {
    delete this->interior_;
  }
  this->interior_ = nullptr;

  // clear ghost
  if (this->ghost_ != nullptr) {
    delete this->ghost_;
  }
  this->ghost_ = nullptr;

  this->mpi_req_.clear();
  this->mpi_stat_.clear();
  this->border_val_.clear();
  this->ghost_val_.clear();
  this->checked_for_dof_partition_ = false;
}

template < class DataType > void CoupledVector< DataType >::SendBorder() {
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());
  assert(this->interior_ != nullptr);
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->nb_procs_);

  this->interior_->GetIndexedValues(&(this->border_val_[0]));

  int tag = 1;
  int ctr = 0;

  for (int id = 0; id < this->nb_procs_; id++) {
    if (this->la_couplings_->border_offsets(id + 1) -
            this->la_couplings_->border_offsets(id) > 0) {
        
      this->parcom_->Isend(&(this->border_val_[0]) + this->la_couplings_->border_offsets(id),
                           this->la_couplings_->border_offsets(id + 1) - this->la_couplings_->border_offsets(id),
                           id, tag, this->mpi_req_[this->nb_recvs_ + ctr]);
      /*
      MPI_Isend(&(this->border_val_[0]) +
                        this->la_couplings_->border_offsets(id),
                    this->la_couplings_->border_offsets(id + 1) -
                        this->la_couplings_->border_offsets(id),
                    mpi_data_type< DataType >::get_type(), id, tag, this->comm_,
                    &(this->mpi_req_[this->nb_recvs_ + ctr]));
      */
      ctr++;
    }
  }
  //std::cout << ctr << " " << this->nb_sends_ << " " << this->nb_procs_ << std::endl;
  assert(ctr == this->nb_sends_);
}

template < class DataType > void CoupledVector< DataType >::ReceiveGhost() {
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->nb_procs_);

  int tag = 1;
  int ctr = 0;

  for (int id = 0; id < this->nb_procs_; ++id) {
    if (this->la_couplings_->ghost_offsets(id + 1) -
            this->la_couplings_->ghost_offsets(id) > 0) {
            
      int info = this->parcom_->Irecv(&(this->ghost_val_[0]) + this->la_couplings_->ghost_offsets(id), 
                                      this->la_couplings_->ghost_offsets(id + 1) - this->la_couplings_->ghost_offsets(id), 
                                      id, tag, this->mpi_req_[ctr]); 
      /*
      MPI_Irecv(&(this->ghost_val_[0]) + this->la_couplings_->ghost_offsets(id),
                  this->la_couplings_->ghost_offsets(id + 1) - this->la_couplings_->ghost_offsets(id),
                  mpi_data_type< DataType >::get_type(), 
                  id, 
                  tag, 
                  this->comm_,
                  &(this->mpi_req_[ctr]));*/
                    
      //assert(info == MPI_SUCCESS);
      ctr++;
    }
  }
  assert(ctr == this->nb_recvs_);
}

template < class DataType > void CoupledVector< DataType >::WaitForSend() {
#ifndef NDEBUG
  int info =
#endif
    MPI_Waitall(this->nb_sends_, &(this->mpi_req_[this->nb_recvs_]),
                &(this->mpi_stat_[this->nb_recvs_]));
  assert(info == MPI_SUCCESS);
}

template < class DataType > void CoupledVector< DataType >::WaitForRecv() {
#ifndef NDEBUG
  int info =
#endif
    MPI_Waitall(this->nb_recvs_, &(this->mpi_req_[0]), &(this->mpi_stat_[0]));
  assert(info == MPI_SUCCESS);

  this->SetGhostValues(&(this->ghost_val_[0]));
}

template < class DataType > void CoupledVector< DataType >::begin_update() {
  assert(this->ghost_ != nullptr);

  this->ReceiveGhost();
  this->SendBorder();
}

template < class DataType > void CoupledVector< DataType >::end_update() {
  assert(this->ghost_ != nullptr);

  this->WaitForRecv();
  this->WaitForSend();
}

template < class DataType > void CoupledVector< DataType >::UpdateCouplings() {

  this->UpdateGhost();
}

template < class DataType > void CoupledVector< DataType >::UpdateGhost() {
  assert(this->ghost_ != nullptr);

  this->ReceiveGhost();
  this->SendBorder();

  this->WaitForRecv();
  this->WaitForSend();
}

template < class DataType > int CoupledVector< DataType >::size_global() const {
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());
  return this->la_couplings_->nb_total_dofs();
}

template < class DataType >
void CoupledVector< DataType >::Print(std::ostream &out) const {
  this->interior_->print(out);
  this->ghost_->print(out);
}

template < class DataType >
void CoupledVector< DataType >::ComputeOwnershipRange() {
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());
  assert(this->my_rank_ >= 0);

  this->ownership_begin_ = this->la_couplings_->dof_offset(this->my_rank_);
  this->ownership_end_ =
    this->ownership_begin_ + this->la_couplings_->nb_dofs(this->my_rank_);
}

template < class DataType >
void CoupledVector< DataType >::WriteHDF5(const std::string &filename,
    const std::string &groupname,
    const std::string &datasetname) {
#ifdef WITH_HDF5
  assert(this->is_initialized());

  DataType *data;
  // Define Data in memory
  data = (DataType *)malloc(sizeof(DataType) * (this->size_local()));
  this->interior_->GetBlockValues(0, size_local(), data);

  H5FilePtr file_ptr(new H5File(filename, "w", this->comm_));
  H5GroupPtr group_ptr(new H5Group(file_ptr, groupname, "w"));
  H5DatasetPtr dataset_ptr(
    new H5Dataset(group_ptr, this->size_global(), datasetname, "w", data));
  dataset_ptr->write(this->size_local(), this->ownership_begin_, data);

  free(data);
#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

template < class DataType >
void CoupledVector< DataType >::ReadHDF5(const std::string &filename,
    const std::string &groupname,
    const std::string &datasetname) {
#ifdef WITH_HDF5
  assert(this->is_initialized());

  DataType *buffer;
  buffer = (DataType *)malloc(sizeof(DataType) * (this->size_local()));

  H5FilePtr file_ptr(new H5File(filename, "r", this->comm_));
  H5GroupPtr group_ptr(new H5Group(file_ptr, groupname, "r"));
  H5DatasetPtr dataset_ptr(
    new H5Dataset(group_ptr, this->size_global(), datasetname, "r", buffer));
  dataset_ptr->read(this->size_local(), this->ownership_begin_, buffer);

  std::vector< int > ind(this->size_local());
  for (int i = 0, total = size_local(); i < total; ++i) {
    ind[i] = i;
  }
  this->interior_->SetValues(vec2ptr(ind), this->size_local(), buffer);

  // Update
  this->UpdateGhost();

  free(buffer);
#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

template < class DataType >
void CoupledVector< DataType >::GetAllDofsAndValues(
  std::vector< int > &id, std::vector< DataType > &val) const {
  assert(this->la_couplings_ != nullptr);
  assert(this->la_couplings_->initialized());
  assert(this->interior_ != nullptr);
  assert(this->ghost_ != nullptr);

  // Temporary containers for interior and ghost values
  DataType *values = new DataType[this->size_local()];
  DataType *ghost_values = new DataType[this->size_local_ghost()];

  // Combine interior, ghost and pp_data values
  this->interior_->GetBlockValues(0, this->size_local(), values);
  this->ghost_->GetBlockValues(0, this->size_local_ghost(), ghost_values);

  int total_size = this->size_local() + this->size_local_ghost();

  id.resize(total_size);
  val.resize(total_size);
  // First: the DoFs and values from the interior
  for (int i = 0; i < this->size_local(); ++i) {
    id[i] = this->ownership_begin_ + i;
    val[i] = values[i];
  }
  delete[] values;
  // Second: the DoFs and values from ghost
  int tmp_offset = this->size_local();
  for (int i = 0; i < this->size_local_ghost(); ++i) {
    id[i + tmp_offset] = this->la_couplings_->Offdiag2Global(i);
    val[i + tmp_offset] = ghost_values[i];
  }
  delete[] ghost_values;
}

/// template instantiation
template class CoupledVector< double >;
template class CoupledVector< float >;

} // namespace la
} // namespace hiflow
