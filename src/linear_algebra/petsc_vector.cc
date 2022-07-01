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

/// @author Bernd Doser, HITS gGmbH
/// @date 2015-11-17

#include "linear_algebra/petsc_vector.h"
#include "common/log.h"
#include "common/pointers.h"
#include "linear_algebra/lmp/init_vec_mat.h"
#include "petsc.h"
#include "petsc_environment.h"
#include "petsc_vector_interface.h"
#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef WITH_HDF5
#include "common/hdf5_tools.h"
#include "hdf5.h"
#endif

namespace hiflow {
namespace la {

template < class DataType >
PETScVector< DataType >::PETScVector()
  : comm_(MPI_COMM_NULL), my_rank_(-1), nb_procs_(-1), ilower_(-1),
    iupper_(-1), cp_(NULL),
    ghost_(init_vector< DataType >(0, "ghost", CPU, NAIVE)), nb_sends_(-1),
    nb_recvs_(-1), mpi_req_(), mpi_stat_(), border_val_(), ghost_val_(),
    border_indices_(), initialized_(false),
    ptr_vec_wrapper_(new petsc::Vec_wrapper) {
  this->global_indices_ = NULL;
  PETScEnvironment::initialize();
}

template < class DataType >
PETScVector< DataType >::~PETScVector() {
  this->Clear();

  // clear ghost
  if (this->ghost_ != NULL) {
    this->ghost_->Clear();
    delete this->ghost_;
  }
  this->ghost_ = NULL;

  if (this->global_indices_ != NULL)
    delete[] this->global_indices_;
  this->global_indices_ = NULL;

  int is_finalized;
  MPI_Finalized(&is_finalized);
  if (!is_finalized) {
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
      assert(this->comm_ == MPI_COMM_NULL);
    }
  }
}

template < class DataType >
Vector< DataType > *PETScVector< DataType >::Clone() const {
  LOG_ERROR("Called PETScVector::Clone not yet implemented!!!");
  quit_program();
  return NULL;
}

template < class DataType >
void PETScVector< DataType >::Init(const MPI_Comm &comm) {
  if (this->comm_ != MPI_COMM_NULL)
    MPI_Comm_free(&this->comm_);
  assert(comm != MPI_COMM_NULL);

  // determine nb. of processes
  int info = MPI_Comm_size(comm, &nb_procs_);
  assert(info == MPI_SUCCESS);
  assert(nb_procs_ > 0);

  // retrieve my rank
  info = MPI_Comm_rank(comm, &my_rank_);
  assert(info == MPI_SUCCESS);
  assert(my_rank_ >= 0);
  assert(my_rank_ < nb_procs_);

  info = MPI_Comm_split(comm, 0, my_rank_, &(this->comm_));
  assert(info == MPI_SUCCESS);
}

template < class DataType >
void PETScVector< DataType >::Init(const MPI_Comm &comm, const LaCouplings &cp) {
  // clear possibly existing DataType
  if (initialized_)
    Clear();

  Init(comm);
  cp_ = &cp;

  // Compute indices range of this process
  ilower_ = cp.dof_offset(my_rank_);
  iupper_ = ilower_ + cp.nb_dofs(my_rank_) - 1;

  // Create PETSC Vector
  VecCreateMPI(comm_, cp_->nb_dofs(my_rank_), cp_->nb_total_dofs(),
               &ptr_vec_wrapper_->vec_);

  const int local_size = iupper_ - ilower_ + 1;

  this->global_indices_ = new int[local_size];

  const int N = local_size;

  assert(N > 0);

  /* Version that uses loop unrolling by an unroll-factor of 5*/
  // compute overhead to unroll factor
  const int M = N % 5;

  // if N is a multiple of 5
  if (M == 0) {
PRAGMA_LOOP_VEC
    for (int i = 0; i < N; i += 5) {
      this->global_indices_[i] = ilower_ + i;
      this->global_indices_[i + 1] = ilower_ + i + 1;
      this->global_indices_[i + 2] = ilower_ + i + 2;
      this->global_indices_[i + 3] = ilower_ + i + 3;
      this->global_indices_[i + 4] = ilower_ + i + 4;
    }
  } else {
    // result for overhead to unroll factor
PRAGMA_LOOP_VEC
    for (int i = 0; i < M; ++i) {
      this->global_indices_[i] = ilower_ + i;
    }

    // result for rest of vectors if length is greater than the unroll factor
    if (N > 5) {
PRAGMA_LOOP_VEC
      for (int i = M; i < N; i += 5) {
        this->global_indices_[i] = ilower_ + i;
        this->global_indices_[i + 1] = ilower_ + i + 1;
        this->global_indices_[i + 2] = ilower_ + i + 2;
        this->global_indices_[i + 3] = ilower_ + i + 3;
        this->global_indices_[i + 4] = ilower_ + i + 4;
      }
    }
  }

  // set border indices
  const size_t size_border = this->cp_->size_border_indices();
  border_indices_.resize(size_border);
PRAGMA_LOOP_VEC
  for (size_t i = 0; i < size_border; ++i) {
    this->border_indices_[i] = this->cp_->border_indices()[i] + ilower_;
  }

  // Initialize ghost part
  // init structure of ghost
  if (this->cp_->size_ghost() > 0) {
    this->ghost_->Clear();
    this->ghost_->Init(this->cp_->size_ghost(), "ghost");
  }

  // prepare for communication
  this->nb_sends_ = 0;
  this->nb_recvs_ = 0;
  for (int id = 0; id < this->nb_procs_; ++id) {
    if (this->cp_->border_offsets(id + 1) - this->cp_->border_offsets(id) > 0) {
      this->nb_sends_++;
    }
    if (this->cp_->ghost_offsets(id + 1) - this->cp_->ghost_offsets(id) > 0) {
      this->nb_recvs_++;
    }
  }
  this->mpi_req_.resize(this->nb_sends_ + this->nb_recvs_);
  this->mpi_stat_.resize(this->nb_sends_ + this->nb_recvs_);
  this->border_val_.resize(this->cp_->size_border_indices());
  this->ghost_val_.resize(this->cp_->size_ghost());

  // Set initialized_ flag
  initialized_ = true;
}

template < class DataType >
void PETScVector< DataType >::Init(const MPI_Comm &comm, const LaCouplings &cp,
                                   const BlockManager &block_manager) {
  this->Init(comm, cp);
}

template < class DataType >
void PETScVector< DataType >::Clear() {
  if (initialized_) {
    VecDestroy(&ptr_vec_wrapper_->vec_);

    // clear ghost
    if (this->ghost_ != NULL) {
      this->ghost_->Clear();
      delete this->ghost_;
    }
    this->ghost_ = NULL;

    if (this->global_indices_ != NULL)
      delete[] this->global_indices_;
    this->global_indices_ = NULL;

    this->cp_ = NULL;

    this->mpi_req_.clear();
    this->mpi_stat_.clear();
    this->border_val_.clear();
    this->ghost_val_.clear();
    this->border_indices_.clear();
    this->ghost_ = init_vector< DataType >(0, "ghost", CPU, NAIVE);
    assert(this->ghost_ != NULL);
    this->initialized_ = false;
  }
}

template < class DataType >
void PETScVector< DataType >::CloneFromWithoutContent(
  const PETScVector< DataType > &vec) {
  if (this != &vec) {
    assert(vec.is_initialized());

    this->Clear();
    int info = 0;
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
      assert(this->comm_ == MPI_COMM_NULL);
    }

    assert(vec.comm() != MPI_COMM_NULL);

    MPI_Comm_dup(vec.comm(), &this->comm_);
    assert(this->comm_ != MPI_COMM_NULL);
    // MPI communicator

    // determine nb. of processes
    info = MPI_Comm_size(this->comm_, &nb_procs_);
    assert(info == MPI_SUCCESS);
    assert(nb_procs_ > 0);

    // retrieve my rank
    info = MPI_Comm_rank(this->comm_, &my_rank_);
    assert(info == MPI_SUCCESS);
    assert(my_rank_ >= 0);
    assert(my_rank_ < nb_procs_);

    this->cp_ = &(vec.la_couplings());
    assert(this->cp_ != NULL);
    assert(this->cp_->initialized());

    // Compute indices range of this process
    ilower_ = vec.la_couplings().dof_offset(my_rank_);
    iupper_ = ilower_ + vec.la_couplings().nb_dofs(my_rank_) - 1;

    // Prepare PETSc MPI interface
    PETScEnvironment::initialize();

    // Create PETSC Vector
    VecCreateMPI(comm_, cp_->nb_dofs(my_rank_), cp_->nb_total_dofs(),
                 &ptr_vec_wrapper_->vec_);

    // Initialize exact structure of vector. To achieve this, we set every
    // element to zero.
    const int local_size = iupper_ - ilower_ + 1;

    this->global_indices_ = new int[local_size];

    const int N = local_size;

    assert(N > 0);

    /* Version that uses loop unrolling by an unroll-factor of 5*/
    // compute overhead to unroll factor
    const int M = N % 5;

    // if N is a multiple of 5
    if (M == 0) {
PRAGMA_LOOP_VEC
      for (int i = 0; i < N; i += 5) {
        this->global_indices_[i] = ilower_ + i;
        this->global_indices_[i + 1] = ilower_ + i + 1;
        this->global_indices_[i + 2] = ilower_ + i + 2;
        this->global_indices_[i + 3] = ilower_ + i + 3;
        this->global_indices_[i + 4] = ilower_ + i + 4;
      }
    } else {
      // result for overhead to unroll factor
PRAGMA_LOOP_VEC
      for (int i = 0; i < M; ++i) {
        this->global_indices_[i] = ilower_ + i;
      }

      // result for rest of vectors if length is greater than the unroll factor
      if (N > 5) {
PRAGMA_LOOP_VEC
        for (int i = M; i < N; i += 5) {
          this->global_indices_[i] = ilower_ + i;
          this->global_indices_[i + 1] = ilower_ + i + 1;
          this->global_indices_[i + 2] = ilower_ + i + 2;
          this->global_indices_[i + 3] = ilower_ + i + 3;
          this->global_indices_[i + 4] = ilower_ + i + 4;
        }
      }
    }

    // prepare for communication
    this->nb_sends_ = 0;
    this->nb_recvs_ = 0;
    for (int id = 0; id < this->nb_procs_; ++id) {
      if (this->cp_->border_offsets(id + 1) - this->cp_->border_offsets(id) >
          0) {
        this->nb_sends_++;
      }
      if (this->cp_->ghost_offsets(id + 1) - this->cp_->ghost_offsets(id) > 0) {
        this->nb_recvs_++;
      }
    }
    this->mpi_req_.resize(this->nb_sends_ + this->nb_recvs_);
    this->mpi_stat_.resize(this->nb_sends_ + this->nb_recvs_);
    this->border_val_.resize(this->cp_->size_border_indices());
    this->ghost_val_.resize(this->cp_->size_ghost());

    // set border indices
    border_indices_.resize(this->cp_->size_border_indices());
PRAGMA_LOOP_VEC
    for (size_t i = 0; i < border_indices_.size(); ++i) {
      border_indices_[i] = this->cp_->border_indices()[i] + ilower_;
    }

    this->ghost_ = vec.ghost().CloneWithoutContent();
    this->ghost_->CopyStructureFrom(vec.ghost());

    initialized_ = true;
  }
}

template < class DataType >
int PETScVector< DataType >::size_local() const {
  int size;
  CHKERRQ(VecGetLocalSize(ptr_vec_wrapper_->vec_, &size));
  return size;
}

template < class DataType >
int PETScVector< DataType >::size_global() const {
  int size;
  VecGetSize(ptr_vec_wrapper_->vec_, &size);
  return size;
}

template < class DataType >
void PETScVector< DataType >::Zeros() {
  VecSet(ptr_vec_wrapper_->vec_, 0);
}

template < class DataType >
DataType PETScVector< DataType >::GetValue(int index) const {
  assert(initialized_);
  DataType value;
  VecGetValues(ptr_vec_wrapper_->vec_, 1, &index, &value);
  return value;
}

template < class DataType >
void PETScVector< DataType >::GetLocalValues(DataType *values) const {

  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());

  const size_t local_length =
    static_cast< size_t >(this->iupper_ - this->ilower_ + 1);

  std::vector< int > indices_interior(local_length);
  int ind_interior = 0;

  for (int i = this->ilower_; i <= this->iupper_; ++i) {
    indices_interior[ind_interior] = i;
    ++ind_interior;
  }

  assert(ind_interior == local_length);

  // extract values
  std::vector< DataType > values_interior(indices_interior.size());
  if (!indices_interior.empty()) {
    VecGetValues(ptr_vec_wrapper_->vec_, indices_interior.size(),
                 vec2ptr(indices_interior), vec2ptr(values_interior));
    VecAssemblyBegin(ptr_vec_wrapper_->vec_);
    VecAssemblyEnd(ptr_vec_wrapper_->vec_);
  }

  for (int i = 0; i < local_length; ++i) {
    values[i] = values_interior[i];
  }

}

template < class DataType >
void PETScVector< DataType >::GetValues(const int *indices, int length,
                                        DataType *values) const {
  assert(initialized_);
  if (length <= 0)
    return;
  assert(indices != NULL);
  assert(values != NULL);

  // extract interior indices
  std::vector< int > indices_interior;
  indices_interior.reserve(length);

  // extract ghost indices
  std::vector< int > indices_ghost;
  indices_interior.reserve(length);

  // transform indices according to local numbering of interior and ghost
  for (int i = 0; i < length; i++) {
    // interior
    if (this->ilower_ <= indices[i] && indices[i] <= this->iupper_) {
      indices_interior.push_back(indices[i]);
    }

    // ghost
    else if (this->cp_->global2offdiag().find(indices[i]) !=
             this->cp_->global2offdiag().end()) {
      indices_ghost.push_back(this->cp_->Global2Offdiag(indices[i]));
    }
  }

  // extract values
  std::vector< DataType > values_interior(indices_interior.size());
  if (!indices_interior.empty()) {
    VecGetValues(ptr_vec_wrapper_->vec_, indices_interior.size(),
                 vec2ptr(indices_interior), vec2ptr(values_interior));
    VecAssemblyBegin(ptr_vec_wrapper_->vec_);
    VecAssemblyEnd(ptr_vec_wrapper_->vec_);
  }

  std::vector< DataType > values_ghost(indices_ghost.size());
  if (!indices_ghost.empty()) {
    this->ghost_->GetValues(vec2ptr(indices_ghost), indices_ghost.size(),
                            vec2ptr(values_ghost));
  }

  typename std::vector< DataType >::const_iterator iter_values_interior_cur =
    values_interior.begin();
  typename std::vector< DataType >::const_iterator iter_values_ghost_cur =
    values_ghost.begin();

  // put values together
  for (int i = 0; i < length; i++) {
    int cont;
    // interior
    if (this->ilower_ <= indices[i] and indices[i] <= this->iupper_) {
      values[i] = *iter_values_interior_cur++;
    }
    // ghost
    else if (this->cp_->global2offdiag().find(indices[length - i - 1]) !=
             this->cp_->global2offdiag().end()) {
      values[i] = *iter_values_ghost_cur++;
    }
    // zeros
    else {
      values[i] = 0;
    }
  }

  // cleanup
  indices_interior.clear();
  indices_ghost.clear();
  values_interior.clear();
  values_ghost.clear();
}

template < class DataType >
DataType PETScVector< DataType >::Norm2() const {
  DataType value;
  VecNorm(ptr_vec_wrapper_->vec_, NORM_2, &value);
  return value;
}

template < class DataType >
DataType PETScVector< DataType >::Norm1() const {
  DataType value;
  VecNorm(ptr_vec_wrapper_->vec_, NORM_1, &value);
  return value;
}

template < class DataType >
DataType PETScVector< DataType >::NormMax() const {
  DataType value;
  VecNorm(ptr_vec_wrapper_->vec_, NORM_INFINITY, &value);
  return value;
}

template < class DataType >
DataType PETScVector< DataType >::Dot(const Vector< DataType > &vec) const {
  const PETScVector< DataType > *hv =
    dynamic_cast< const PETScVector< DataType > * >(&vec);
  if (!hv) {
    LOG_ERROR("Called PETScVector::Dot with incompatible vector type.");
    quit_program();
  }
  return this->Dot(*hv);
}

template < class DataType >
DataType PETScVector< DataType >::Dot(const PETScVector< DataType > &vec)
const {
  DataType value;
  VecDot(ptr_vec_wrapper_->vec_, vec.ptr_vec_wrapper_->vec_, &value);
  return value;
}

template < class DataType >
void PETScVector< DataType >::Add(int index, DataType value) {
  VecSetValues(ptr_vec_wrapper_->vec_, 1, &index, &value, ADD_VALUES);
  VecAssemblyBegin(ptr_vec_wrapper_->vec_);
  VecAssemblyEnd(ptr_vec_wrapper_->vec_);
}

template < class DataType >
void PETScVector< DataType >::Add(const int *indices, int length,
                                  const DataType *values) {
  VecSetValues(ptr_vec_wrapper_->vec_, length, indices, values, ADD_VALUES);
  VecAssemblyBegin(ptr_vec_wrapper_->vec_);
  VecAssemblyEnd(ptr_vec_wrapper_->vec_);
}

template < class DataType >
void PETScVector< DataType >::SetValue(int index, DataType value) {
  VecSetValues(ptr_vec_wrapper_->vec_, 1, &index, &value, INSERT_VALUES);
  VecAssemblyBegin(ptr_vec_wrapper_->vec_);
  VecAssemblyEnd(ptr_vec_wrapper_->vec_);
}

template < class DataType >
void PETScVector< DataType >::SetValues(const int *indices, const int length,
                                        const DataType *values) {
  VecSetValues(ptr_vec_wrapper_->vec_, length, indices, values, INSERT_VALUES);
  VecAssemblyBegin(ptr_vec_wrapper_->vec_);
  VecAssemblyEnd(ptr_vec_wrapper_->vec_);
}

template < class DataType >
void PETScVector< DataType >::SetLocalValues(const DataType *values) {

  const size_t local_length =
    static_cast< size_t >(this->iupper_ - this->ilower_ + 1);

  std::vector< int > indices_interior(local_length);
  int ind_interior = 0;

  for (int i = this->ilower_; i <= this->iupper_; ++i) {
    indices_interior[ind_interior] = i;
    ++ind_interior;
  }

  VecSetValues(ptr_vec_wrapper_->vec_, local_length, indices_interior.data(), values, INSERT_VALUES);
  VecAssemblyBegin(ptr_vec_wrapper_->vec_);
  VecAssemblyEnd(ptr_vec_wrapper_->vec_);
}

template < class DataType >
void PETScVector< DataType >::Axpy(const Vector< DataType > &vecx,
                                   DataType alpha) {
  const PETScVector< DataType > *hv =
    dynamic_cast< const PETScVector< DataType > * >(&vecx);
  if (!hv) 
  {
    LOG_INFO("Warning", "Called PETScVector::Axpy with incompatible vector type. Using slow axpy version");
    scale_axpy(1., *this, alpha, vecx);  
  }
  else 
  {
    this->Axpy(*hv, alpha);
  }
}

template < class DataType >
void PETScVector< DataType >::Axpy(const PETScVector< DataType > &vecx,
                                   DataType alpha) {
  VecAXPY(ptr_vec_wrapper_->vec_, alpha, vecx.ptr_vec_wrapper_->vec_);
}

template < class DataType >
void PETScVector< DataType >::ScaleAdd(const Vector< DataType > &vecx,
                                       DataType alpha) {
  const PETScVector< DataType > *hv =
    dynamic_cast< const PETScVector< DataType > * >(&vecx);
  if (!hv) 
  {
    LOG_INFO("Warning", "Called PETScVector::ScaleAdd with incompatible vector type. Using slow axpy version");
    scale_axpy(alpha, *this, 1., vecx);  
  }
  else 
  {
    this->ScaleAdd(*hv, alpha);
  }
}

template < class DataType >
void PETScVector< DataType >::ScaleAdd(const PETScVector< DataType > &vecx,
                                       DataType alpha) {
  VecScale(ptr_vec_wrapper_->vec_, alpha);
  VecAXPY(ptr_vec_wrapper_->vec_, 1, vecx.ptr_vec_wrapper_->vec_);
}

template < class DataType >
void PETScVector< DataType >::Scale(DataType alpha) {
  VecScale(ptr_vec_wrapper_->vec_, alpha);
}

template < class DataType >
void PETScVector< DataType >::SendBorder() {
  assert(this->cp_->initialized());
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->nb_procs_);

  this->GetValues(vec2ptr(border_indices_), border_indices_.size(),
                  vec2ptr(border_val_));

  int tag = 1;
  int ctr = 0;

  for (int id = 0; id < this->nb_procs_; id++) {
    if (this->cp_->border_offsets(id + 1) - this->cp_->border_offsets(id) > 0) {
      int info = MPI_Isend(
                   &(this->border_val_[0]) + this->cp_->border_offsets(id),
                   this->cp_->border_offsets(id + 1) - this->cp_->border_offsets(id),
                   mpi_data_type< DataType >::get_type(), id, tag, this->comm_,
                   &(this->mpi_req_[this->nb_recvs_ + ctr]));
      assert(info == MPI_SUCCESS);
      ctr++;
    }
  }
  assert(ctr == this->nb_sends_);
}

template < class DataType >
void PETScVector< DataType >::ReceiveGhost() {
  assert(this->cp_->initialized());
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->nb_procs_);

  int tag = 1;
  int ctr = 0;

  for (int id = 0; id < this->nb_procs_; ++id) {
    if (this->cp_->ghost_offsets(id + 1) - this->cp_->ghost_offsets(id) > 0) {
      int info = MPI_Irecv(
                   &(this->ghost_val_[0]) + this->cp_->ghost_offsets(id),
                   this->cp_->ghost_offsets(id + 1) - this->cp_->ghost_offsets(id),
                   mpi_data_type< DataType >::get_type(), id, tag, this->comm_,
                   &(this->mpi_req_[ctr]));
      assert(info == MPI_SUCCESS);
      ctr++;
    }
  }
  assert(ctr == this->nb_recvs_);
}

template < class DataType >
void PETScVector< DataType >::WaitForSend() {
  int info = MPI_Waitall(this->nb_sends_, &(this->mpi_req_[this->nb_recvs_]),
                         &(this->mpi_stat_[this->nb_recvs_]));
  assert(info == MPI_SUCCESS);
}

template < class DataType >
void PETScVector< DataType >::WaitForRecv() {
  int info =
    MPI_Waitall(this->nb_recvs_, &(this->mpi_req_[0]), &(this->mpi_stat_[0]));
  assert(info == MPI_SUCCESS);

  this->SetGhostValues(&(this->ghost_val_[0]));
}

template < class DataType >
void PETScVector< DataType >::Update() {
  this->UpdateGhost();
}

template < class DataType >
void PETScVector< DataType >::UpdateGhost() {
  assert(this->ghost_ != NULL);

  this->ReceiveGhost();
  this->SendBorder();

  this->WaitForRecv();
  this->WaitForSend();
}

template < class DataType >
void PETScVector< DataType >::SetGhostValues(const DataType *values) {
  assert(this->ghost_ != NULL);

  this->ghost_->SetBlockValues(0, this->size_local_ghost(), values);
}

template < class DataType >
void PETScVector< DataType >::GetAllDofsAndValues(
  std::vector< int > &id, std::vector< DataType > &val) {
  assert(this->ghost_ != NULL);

  // Temporary containers for interior and ghost values
  DataType *values = new DataType[this->size_local()];
  DataType *ghost_values = new DataType[this->size_local_ghost()];

  int total_size = this->size_local() + this->size_local_ghost();
  id.resize(total_size);

  // First: the DoFs from the interior
  for (int i = 0; i < this->size_local(); ++i) {
    id[i] = ilower_ + i;
  }

  // Combine interior, ghost and pp_data values
  this->GetValues(vec2ptr(id), this->size_local(), values);
  this->ghost_->GetBlockValues(0, this->size_local_ghost(), ghost_values);

  val.resize(total_size);
  // First: values from the interior
  for (int i = 0; i < this->size_local(); ++i) {
    val[i] = values[i];
  }
  delete[] values;
  // Second: the DoFs and values from ghost
  int tmp_offset = this->size_local();
  for (int i = 0; i < this->size_local_ghost(); ++i) {
    id[i + tmp_offset] = this->cp_->Offdiag2Global(i);
    val[i + tmp_offset] = ghost_values[i];
  }
  delete[] ghost_values;
}

template < class DataType >
void PETScVector< DataType >::WriteHDF5(const std::string &filename,
                                        const std::string &groupname,
                                        const std::string &datasetname) {
  assert(this->comm_ != MPI_COMM_NULL);
  assert(this->is_initialized());
#ifdef WITH_HDF5
  // Define Data in memory
  const size_t local_size = this->size_local();
  std::vector< DataType > data(local_size, 0.);

  this->GetValues(this->global_indices_, local_size, vec2ptr(data));

  H5FilePtr file_ptr(new H5File(filename, "w", this->comm_));
  H5GroupPtr group_ptr(new H5Group(file_ptr, groupname, "w"));
  H5DatasetPtr dataset_ptr(new H5Dataset(group_ptr, this->size_global(),
                                         datasetname, "w", vec2ptr(data)));
  dataset_ptr->write(this->size_local(), this->ilower_, vec2ptr(data));

  data.clear();
#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

template < class DataType >
void PETScVector< DataType >::ReadHDF5(const std::string &filename,
                                       const std::string &groupname,
                                       const std::string &datasetname) {
  assert(this->comm_ != MPI_COMM_NULL);
  assert(this->is_initialized());
#ifdef WITH_HDF5
  DataType *buffer;
  buffer = new DataType[this->size_local()];

  H5FilePtr file_ptr(new H5File(filename, "r", this->comm_));
  H5GroupPtr group_ptr(new H5Group(file_ptr, groupname, "r"));
  H5DatasetPtr dataset_ptr(
    new H5Dataset(group_ptr, this->size_global(), datasetname, "r", buffer));
  dataset_ptr->read(this->size_local(), this->ilower_, buffer);

  const size_t local_size = this->size_local();

  this->SetValues(this->global_indices_, local_size, buffer);

  // Update
  this->UpdateGhost();

  delete[] buffer;
#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

// template instantiation
template class PETScVector< double >;

} // namespace la
} // namespace hiflow
