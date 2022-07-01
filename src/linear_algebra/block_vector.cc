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

/// @author Jonas Kratzke, Simon Gawlok

#include "linear_algebra/block_vector.h"
#include "common/pointers.h"

#ifdef WITH_HDF5
#include "common/hdf5_tools.h"
#include "hdf5.h"
#endif

namespace hiflow {
namespace la {

template < class LAD > BlockVector< LAD >::BlockVector() {
  this->initialized_ = false;
  this->comm_ = MPI_COMM_NULL;
  this->num_blocks_ = -1;
  this->block_manager_ = nullptr;
}

template < class LAD > BlockVector< LAD >::~BlockVector() {
  if (this->initialized_) {
    this->Clear();
  }

  int is_finalized;
  MPI_Finalized(&is_finalized);
  if (!is_finalized) {
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
      assert(this->comm_ == MPI_COMM_NULL);
    }
  }
}

template < class LAD > void BlockVector< LAD >::Clear() {
  for (int i = 0; i < this->vec_.size(); ++i) {
    if (this->vec_[i] != nullptr) {
      delete this->vec_[i];
    }
  }
  this->vec_.clear();

  this->global_indices_.clear();

  this->num_blocks_ = -1;

  this->active_blocks_.clear();

  this->initialized_ = false;
}

template < class LAD > BlockVector< LAD > *BlockVector< LAD >::Clone() const {
  LOG_ERROR("Called BlockVector::Clone not yet implemented!!!");
  quit_program();
  return nullptr;
}

template < class LAD >
void BlockVector< LAD >::CloneFromWithoutContent(
    const BlockVector< LAD > &vec) {
  if (this != &vec) {
    assert(vec.is_initialized());
    this->Clear();
#ifndef NDEBUG
    int info = 0;
#endif
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
      assert(this->comm_ == MPI_COMM_NULL);
    }

    assert(vec.comm() != MPI_COMM_NULL);

    MPI_Comm_dup(vec.comm(), &this->comm_);
    assert(this->comm_ != MPI_COMM_NULL);

    // Clone number of blocks
    assert(vec.num_blocks() > 0);
    this->num_blocks_ = vec.num_blocks();

    // Clone global indices
    this->global_indices_ = vec.global_indices();

    // Clone pointer to block_manager
    assert(vec.block_manager() != 0);
    this->block_manager_ = vec.block_manager();

    // Initialize own vectors with cloned LaCouplings
    this->vec_.resize(this->num_blocks_);
    for (int i = 0; i < this->num_blocks_; ++i) {
      this->vec_[i] = new BVector();
      this->vec_[i]->Init(this->comm_,
                          *(this->block_manager_->la_c_blocks()[i]));
    }

    // Set initialized status to true
    this->initialized_ = true;

    this->set_active_blocks(vec.get_active_blocks());
    
    // check dimension
    // assert ( this->global_indices_.size ( ) == this->size_local ( ) );
  }
}

template < class DataType >
void BlockVector< DataType >::Init(const MPI_Comm &comm, 
                                   const LaCouplings &cp,
                                   PLATFORM plat,
                                   IMPLEMENTATION impl,
                                   CBlockManagerSPtr block_manager) {

  // clear possibly existing DataType
  if (this->initialized_) {
    this->Clear();
  }

  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
    assert(this->comm_ == MPI_COMM_NULL);
  }

  assert(comm != MPI_COMM_NULL);

  MPI_Comm_dup(comm, &this->comm_);
  assert(this->comm_ != MPI_COMM_NULL);

  // Get rank of current process
  int my_rank;
  MPI_Comm_rank(comm_, &my_rank);

  assert (block_manager != 0);
  this->block_manager_ = block_manager;

  // Set number of blocks
  this->num_blocks_ = this->block_manager_->num_blocks();

  this->active_blocks_.resize(this->num_blocks_, true);

  // Compute indices range of this process
  int ilower = cp.global_offsets()[my_rank];
  int iupper = cp.global_offsets()[my_rank + 1] - 1;

  // Initialize exact structure of vector. To achieve this, we set every element
  // to zero.
  const int local_size = iupper - ilower + 1;

  this->global_indices_.resize(local_size);

  const int N = local_size;
  LOG_DEBUG(1, "Number of values in interior on process " << my_rank << ": "
                                                          << local_size);

  assert(N > 0);

  /* Version that uses loop unrolling by an unroll-factor of 5*/
  // compute overhead to unroll factor
  const int M = N % 5;

  // if N is a multiple of 5
  if (M == 0) {
PRAGMA_LOOP_VEC
    for (int i = 0; i < N; i += 5) {
      this->global_indices_[i] = ilower + i;
      this->global_indices_[i + 1] = ilower + i + 1;
      this->global_indices_[i + 2] = ilower + i + 2;
      this->global_indices_[i + 3] = ilower + i + 3;
      this->global_indices_[i + 4] = ilower + i + 4;
    }
  } else {
    // result for overhead to unroll factor
PRAGMA_LOOP_VEC
    for (int i = 0; i < M; ++i) {
      this->global_indices_[i] = ilower + i;
    }

    // result for rest of vectors if length is greater than the unroll factor
    if (N > 5) {
PRAGMA_LOOP_VEC
      for (int i = M; i < N; i += 5) {
        this->global_indices_[i] = ilower + i;
        this->global_indices_[i + 1] = ilower + i + 1;
        this->global_indices_[i + 2] = ilower + i + 2;
        this->global_indices_[i + 3] = ilower + i + 3;
        this->global_indices_[i + 4] = ilower + i + 4;
      }
    }
  }

  //*****************************************************************
  // Initialize block vectors
  //*****************************************************************

  this->vec_.resize(this->num_blocks_);
  for (size_t i = 0; i < this->num_blocks_; ++i) {
    this->vec_[i] = new BVector();
    this->vec_[i]->Init(this->comm_, *(this->block_manager_->la_c_blocks()[i]), plat, impl);
  }

  this->initialized_ = true;

  // check dimension
  assert(this->global_indices_.size() == this->size_local());
}

template < class LAD >
const typename LAD::VectorType &
BlockVector< LAD >::GetBlock(const size_t block_number) const {
  assert(block_number < this->num_blocks_);
  assert(this->initialized_);
  
  return *(this->vec_[block_number]);
}

template < class LAD >
typename LAD::VectorType &
BlockVector< LAD >::GetBlock(const size_t block_number) {
  assert(block_number < this->num_blocks_);
  assert(this->initialized_);
  
  return *(this->vec_[block_number]);
}

template < class LAD > int BlockVector< LAD >::size_local() const {
  assert(this->initialized_);

  int size = 0;
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      size += this->vec_[i]->size_local();
    }
  }
  return size;
}

template < class LAD > int BlockVector< LAD >::size_global() const {
  assert(this->initialized_);

  int size = 0;
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      size += this->vec_[i]->size_global();
    }
  }
  return size;
}

template < class LAD > void BlockVector< LAD >::Update() {
  assert(this->initialized_);
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      this->vec_[i]->Update();
    }
  }
}

template < class LAD > void BlockVector< LAD >::begin_update() {
  assert(this->initialized_);

  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      this->vec_[i]->begin_update();
    }
  }
}

template < class LAD > void BlockVector< LAD >::end_update() {
  assert(this->initialized_);

  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      this->vec_[i]->end_update();
    }
  }
}

template < class LAD > void BlockVector< LAD >::Zeros() {
  assert(this->initialized_);

  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      this->vec_[i]->Zeros();
    }
  }
}

template < class LAD >
typename LAD::DataType BlockVector< LAD >::GetValue(const int index) const {
  assert(this->initialized_);

  BDataType val;
  this->GetValues(&index, 1, &val);
  return val;
}

template < class LAD >
void BlockVector< LAD >::GetValues(const int *indices, const int size_indices,
                                   BDataType *values) const {
  assert(this->initialized_);

  for (int i = 0; i < size_indices; ++i) {
    // Map system index to block number and block index
    int block_num = -1;
    int block_index = -1;
    this->block_manager_->map_system2block(indices[i], block_num, block_index);

    this->vec_[block_num]->GetValues(&block_index, 1, &values[i]);
  }
}

template < class LAD >
void BlockVector< LAD >::GetLocalValues(BDataType *values) const {
  assert(this->initialized_);
  
  int counter = 0;
  for (int k=0; k < num_blocks_; ++k)
  {
    const int vec_local_size = this->vec_[k]->size_local();
    std::vector<BDataType> bval (vec_local_size);
    this->vec_[k]->GetLocalValues(&(bval.front()));
    
    for (int i = 0; i < vec_local_size; ++i)
    {  
      values[counter + i] = bval[i];
    }
    counter += vec_local_size;
  } 
}


template < class LAD >
void BlockVector< LAD >::GetAllDofsAndValues(
    std::vector< int > &id, std::vector< BDataType > &val) const {
  assert(this->initialized_);

  id.clear();
  val.clear();

  int num_dofs = 0;

  // Determine number of entries
  for (int b = 0; b < this->num_blocks_; ++b) {
    num_dofs += this->vec_[b]->size_local() + this->vec_[b]->ghost().get_size();
  }
  id.reserve(num_dofs);
  val.reserve(num_dofs);

  for (int b = 0; b < this->num_blocks_; ++b) {
    std::vector< int > block_id;
    std::vector< BDataType > block_val;

    //*****************************************************************
    // Get all Dofs and Values of the blocks
    //*****************************************************************
    this->vec_[b]->GetAllDofsAndValues(block_id, block_val);
    assert(block_id.size() == block_val.size());

    //*****************************************************************
    // Map Dofs from block to system numbering
    //*****************************************************************
    for (int i = 0; i < block_id.size(); ++i) {
      int system_index = -1;
      this->block_manager_->map_block2system(b, block_id[i], system_index);

      id.push_back(system_index);
      val.push_back(block_val[i]);
    }
  }
}

template < class LAD >
typename LAD::DataType BlockVector< LAD >::Norm2() const {
  assert(this->initialized_);

  return std::sqrt(this->Dot(*this));
}

template < class LAD >
typename LAD::DataType BlockVector< LAD >::Norm1() const {
  assert(this->initialized_);

  BDataType value = 0;
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      value += this->vec_[i]->Norm1();
    }
  }
  return value;
}

template < class LAD >
typename LAD::DataType BlockVector< LAD >::NormMax() const {
  assert(this->initialized_);

  BDataType value = 0;
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      value = std::max(value, this->vec_[i]->NormMax());
    }
  }
  return value;
}

template < class LAD >
typename LAD::DataType
BlockVector< LAD >::Dot(const Vector< BDataType > &vec) const {
  assert(this->initialized_);

  const BlockVector< LAD > *hv =
      dynamic_cast< const BlockVector< LAD > * >(&vec);

  if (hv != 0) {
    return this->Dot(*hv);
  } else {
    LOG_ERROR("Called BlockVector::Dot with incompatible vector type.");
    quit_program();
    return -1.;
  }
  return -1.;
}

template < class LAD >
typename LAD::DataType
BlockVector< LAD >::Dot(const BlockVector< LAD > &vec) const {
  assert(this->initialized_);
  assert(this->num_blocks() == vec.num_blocks());
  BDataType value = 0.;
  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      value += this->vec_[i]->Dot(vec.GetBlock(i));
    }
  }
  return value;
}

template < class LAD >
void BlockVector< LAD >::Add(int index, BDataType scalar) {
  assert(this->initialized_);
  this->Add(&index, 1, &scalar);
}

template < class LAD >
void BlockVector< LAD >::Add(const int *indices, int length,
                             const BDataType *values) {
  assert(this->initialized_);

  //*****************************************************************
  // Map indices to block numbers and block indices
  //*****************************************************************
  std::vector< std::vector< int > > indices_block(this->num_blocks_);
  std::vector< std::vector< BDataType > > values_block(this->num_blocks_);
  for (int i = 0; i < this->num_blocks_; ++i) {
    indices_block[i].reserve(length);
    values_block[i].reserve(length);
  }

  for (int i = 0; i < length; ++i) {
    int block_num = -1;
    int block_index = -1;
    this->block_manager_->map_system2block(indices[i], block_num, block_index);

    indices_block[block_num].push_back(block_index);
    values_block[block_num].push_back(values[i]);
  }

  //*****************************************************************
  // Add values to the block vectors
  //*****************************************************************
  for (int i = 0; i < this->num_blocks_; ++i) {
    this->vec_[i]->Add(vec2ptr(indices_block[i]),
                       static_cast< int >(indices_block[i].size()),
                       vec2ptr(values_block[i]));
  }
}

template < class LAD >
void BlockVector< LAD >::SetValue(const int index, const BDataType value) {
  assert(this->initialized_);

  this->SetValues(&index, 1, &value);
}

template < class LAD >
void BlockVector< LAD >::SetValues(const int *indices, const int size_indices,
                                   const BDataType *values) {
  assert(this->initialized_);

  //*****************************************************************
  // Map indices to block numbers and block indices
  //*****************************************************************
  std::vector< std::vector< int > > indices_block(this->num_blocks_);
  std::vector< std::vector< BDataType > > values_block(this->num_blocks_);
  for (int i = 0; i < this->num_blocks_; ++i) {
    indices_block[i].reserve(size_indices);
    values_block[i].reserve(size_indices);
  }

  for (int i = 0; i < size_indices; ++i) {
    int block_num = -1;
    int block_index = -1;
    this->block_manager_->map_system2block(indices[i], block_num, block_index);

    indices_block[block_num].push_back(block_index);
    values_block[block_num].push_back(values[i]);
  }

  //*****************************************************************
  // Set values to the block vectors
  //*****************************************************************
  for (int i = 0; i < this->num_blocks_; ++i) {
    this->vec_[i]->SetValues(vec2ptr(indices_block[i]),
                             static_cast< int >(indices_block[i].size()),
                             vec2ptr(values_block[i]));
  }
}

template < class LAD >
void BlockVector< LAD >::SetLocalValues(const BDataType *values) {
  assert(this->initialized_);
  
  int counter = 0;
  for (int k=0; k < num_blocks_; ++k)
  {
    const int vec_local_size = this->vec_[k]->size_local();
    std::vector<BDataType> bval (vec_local_size);
    for (int i = 0; i < vec_local_size; ++i)
    {  
      bval[i] = values[counter + i];
    }
    counter += vec_local_size;
  
    this->vec_[k]->SetLocalValues(&(bval.front()));
  } 
}

template < class LAD >
void BlockVector< LAD >::Axpy(const Vector< BDataType > &vecx,
                              const BDataType alpha) {
  assert(this->initialized_);

  const BlockVector< LAD > *hv =
      dynamic_cast< const BlockVector< LAD > * >(&vecx);

  if (hv != 0) {
    this->Axpy(*hv, alpha);
  } else {
    LOG_ERROR("Called BlockVector::Axpy with incompatible vector type.");
    quit_program();
  }
}

template < class LAD >
void BlockVector< LAD >::Axpy(const BlockVector< LAD > &vecx,
                              const BDataType alpha) {
  assert(this->initialized_);
  assert(this->num_blocks() == vecx.num_blocks());

  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      this->vec_[i]->Axpy(vecx.GetBlock(i), alpha);
    }
  }
}

template < class LAD >
void BlockVector< LAD >::ScaleAdd(const Vector< BDataType > &vecx,
                                  const BDataType alpha) {
  assert(this->initialized_);

  const BlockVector< LAD > *hv =
      dynamic_cast< const BlockVector< LAD > * >(&vecx);

  if (hv != 0) {
    this->ScaleAdd(*hv, alpha);
  } else {
    LOG_ERROR("Called BlockVector::Axpy with incompatible vector type.");
    quit_program();
  }
}

template < class LAD >
void BlockVector< LAD >::ScaleAdd(const BlockVector< LAD > &vecx,
                                  const BDataType alpha) {
  assert(this->initialized_);
  assert(this->num_blocks() == vecx.num_blocks());

  this->Scale(alpha);
  this->Axpy(vecx, static_cast< BDataType >(1.));
}

template < class LAD > void BlockVector< LAD >::Scale(const BDataType alpha) {
  assert(this->initialized_);

  for (int i = 0; i < this->num_blocks_; ++i) {
    if (this->active_blocks_[i]) {
      this->vec_[i]->Scale(alpha);
    }
  }
}

template < class LAD > void BlockVector< LAD >::print_statistics() const {}

template < class LAD >
void BlockVector< LAD >::WriteHDF5(const std::string &filename,
                                   const std::string &groupname,
                                   const std::string &datasetname) {
  assert(this->comm_ != MPI_COMM_NULL);
  assert(this->initialized_);
#ifdef WITH_HDF5
  // Define Data in memory
  const size_t local_size = this->size_local();
  std::vector< BDataType > data(local_size, 0.);

  this->GetValues(vec2ptr(this->global_indices_), local_size, vec2ptr(data));

  H5FilePtr file_ptr(new H5File(filename, "w", this->comm_));
  H5GroupPtr group_ptr(new H5Group(file_ptr, groupname, "w"));
  H5DatasetPtr dataset_ptr(new H5Dataset(group_ptr, this->size_global(),
                                         datasetname, "w", vec2ptr(data)));
  dataset_ptr->write(this->size_local(), this->global_indices_[0],
                     vec2ptr(data));

  data.clear();
#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

template < class LAD >
void BlockVector< LAD >::ReadHDF5(const std::string &filename,
                                  const std::string &groupname,
                                  const std::string &datasetname) {
  assert(this->comm_ != MPI_COMM_NULL);
  assert(this->initialized_);
#ifdef WITH_HDF5
  BDataType *buffer;
  buffer = new BDataType[this->size_local()];

  H5FilePtr file_ptr(new H5File(filename, "r", this->comm_));
  H5GroupPtr group_ptr(new H5Group(file_ptr, groupname, "r"));
  H5DatasetPtr dataset_ptr(
      new H5Dataset(group_ptr, this->size_global(), datasetname, "r", buffer));
  dataset_ptr->read(this->size_local(), this->global_indices_[0], buffer);

  const size_t local_size = this->size_local();

  this->SetValues(vec2ptr(this->global_indices_), local_size, buffer);

  // Update
  for (int i = 0; i < this->num_blocks_; ++i) {
    this->vec_[i]->Update();
  }

  delete[] buffer;
#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

// template instantiation
template class BlockVector< LADescriptorCoupledD >;
template class BlockVector< LADescriptorCoupledS >;
#ifdef WITH_HYPRE
template class BlockVector< LADescriptorHypreD >;
#endif
#if defined(WITH_PETSC)
template class BlockVector< LADescriptorPETScD >;
#endif

} // namespace la
} // namespace hiflow
