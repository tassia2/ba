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

/// @author Simon Gawlok

#include "linear_algebra/hypre_vector.h"
#include "common/log.h"
#include "common/pointers.h"
#include "lmp/init_vec_mat.h"
#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef WITH_HYPRE
extern "C" {
#include "_hypre_parcsr_mv.h"
}
#endif

#ifdef WITH_HDF5
#include "common/hdf5_tools.h"
#include "hdf5.h"
#endif

namespace hiflow {
namespace la {

template < class DataType > HypreVector< DataType >::HypreVector() {
  this->initialized_ = false;
  this->cp_ = nullptr;
  this->comm_ = MPI_COMM_NULL;
  this->nb_procs_ = -1;
  this->my_rank_ = -1;
  this->ghost_ = nullptr;
  this->ilower_ = -1;
  this->iupper_ = -1;
  this->interior_ = hiflow::la::init_vector< DataType >(0, "interior", CPU, NAIVE);
  this->ghost_ = hiflow::la::init_vector< DataType >(0, "ghost", CPU, NAIVE);
  assert(this->interior_ != nullptr);
  assert(this->ghost_ != nullptr);
}

template < class DataType > HypreVector< DataType >::~HypreVector() 
{
  this->Clear();

  // clear ghost
  if (this->interior_ != nullptr) 
  {
    this->interior_->Clear();
    delete this->interior_;
  }
  this->interior_ = nullptr;

  if (this->ghost_ != nullptr) 
  {
    this->ghost_->Clear();
    delete this->ghost_;
  }
  this->ghost_ = nullptr;

  int is_finalized;
  MPI_Finalized(&is_finalized);
  if (is_finalized == 0) {
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&this->comm_);
      assert(this->comm_ == MPI_COMM_NULL);
    }
  }
  this->global_indices_.clear();

  this->initialized_ = false;
}

template < class DataType >
Vector< DataType > *HypreVector< DataType >::Clone() const {
  LOG_ERROR("Called HypreVector::Clone not yet implemented!!!");
  quit_program();
  return nullptr;
}

template < class DataType >
void HypreVector< DataType >::Init(const MPI_Comm &comm,
                                   const LaCouplings &cp) {
  // clear possibly existing DataType
  if (initialized_) {
    this->Clear();
  }

  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
    assert(this->comm_ == MPI_COMM_NULL);
  }

  assert(comm != MPI_COMM_NULL);

  MPI_Comm_dup(comm, &this->comm_);
  assert(this->comm_ != MPI_COMM_NULL);
  // MPI communicator

  // determine nb. of processes
#ifndef NDEBUG
  int info =
#endif
    MPI_Comm_size(this->comm_, &nb_procs_);
  assert(info == MPI_SUCCESS);
  assert(nb_procs_ > 0);

  // retrieve my rank
#ifndef NDEBUG
  info =
#endif
    MPI_Comm_rank(this->comm_, &my_rank_);
  assert(info == MPI_SUCCESS);
  assert(my_rank_ >= 0);
  assert(my_rank_ < nb_procs_);
#ifdef WITH_HYPRE

  this->cp_ = &cp;
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());

  // Get rank of current process
  MPI_Comm_rank(comm_, &my_rank_);

  // Compute indices range of this process
  ilower_ = this->cp_->dof_offset(my_rank_);
  iupper_ = ilower_ + this->cp_->nb_dofs(my_rank_) - 1;

  // Create HYPRE Vector
  HYPRE_IJVectorCreate(comm_, ilower_, iupper_, &x_);

  // Use parallel csr format
  HYPRE_IJVectorSetObjectType(x_, HYPRE_PARCSR);

  HYPRE_IJVectorSetPrintLevel(x_, 100);

  // Tell HYPRE that no vector entries need to be communicated to other
  // processors
  HYPRE_IJVectorSetMaxOffProcElmts(x_, 0);

  // Initialize
  HYPRE_IJVectorInitialize(x_);

  // Initialize exact structure of vector. To achieve this, we set every element
  // to zero.
  const int local_size = iupper_ - ilower_ + 1;

  this->global_indices_.resize(local_size);

  std::vector< HYPRE_Complex > val(local_size, static_cast< HYPRE_Complex >(0));

  const int N = local_size;
  LOG_DEBUG(1, "Number of values in interior on process "
            << this->my_rank_ << ": " << local_size);

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

  HYPRE_Int nvals = local_size;

  std::vector< HYPRE_Int > indices_temp(local_size);
  for (size_t i = 0; i < indices_temp.size(); ++i) {
    indices_temp[i] = static_cast< HYPRE_Int >(this->global_indices_[i]);
  }
  HYPRE_IJVectorSetValues(x_, nvals, vec2ptr(indices_temp), vec2ptr(val));

  // Finalize initialization of vector
  HYPRE_IJVectorAssemble(x_);
  HYPRE_IJVectorGetObject(x_, (void **)&parcsr_x_);

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

  LOG_DEBUG(1, "Number of values in ghost on process "
            << this->my_rank_ << ": " << this->ghost_->get_size());

  this->interior_->Clear();
  this->interior_->Init(local_size, "interior");
    
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

  val.clear();
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType > void HypreVector< DataType >::Clear() {
#ifdef WITH_HYPRE
  if (initialized_) {
    HYPRE_IJVectorDestroy(x_);
    
    if (this->interior_ != nullptr) {
      this->interior_->Clear();
      delete this->interior_;
    }
    this->interior_ = nullptr;
    
    // clear ghost
    if (this->ghost_ != nullptr) {
      this->ghost_->Clear();
      delete this->ghost_;
    }
    this->ghost_ = nullptr;

    this->global_indices_.clear();

    mpi_req_.clear();
    mpi_stat_.clear();
    border_val_.clear();
    ghost_val_.clear();
    border_indices_.clear();
    ghost_ = init_vector< DataType >(0, "ghost", CPU, NAIVE);
    assert(this->ghost_ != nullptr);
    
    interior_ = init_vector< DataType >(0, "interior", CPU, NAIVE);
    assert(this->interior_ != nullptr);
  }
  initialized_ = false;
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType >
void HypreVector< DataType >::CloneFromWithoutContent(
  const HypreVector< DataType > &vec) {
#ifdef WITH_HYPRE
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
    // MPI communicator

    // determine nb. of processes
#ifndef NDEBUG
    info =
#endif
      MPI_Comm_size(this->comm_, &nb_procs_);
    assert(info == MPI_SUCCESS);
    assert(nb_procs_ > 0);

    // retrieve my rank
#ifndef NDEBUG
    info =
#endif
      MPI_Comm_rank(this->comm_, &my_rank_);
    assert(info == MPI_SUCCESS);
    assert(my_rank_ >= 0);
    assert(my_rank_ < nb_procs_);

    assert(vec.la_couplings() != nullptr);

    this->cp_ = vec.la_couplings();
    assert(this->cp_ != nullptr);
    assert(this->cp_->initialized());

    // Compute indices range of this process
    ilower_ = vec.la_couplings()->dof_offset(my_rank_);
    iupper_ = ilower_ + vec.la_couplings()->nb_dofs(my_rank_) - 1;

    // Create HYPRE Vector
    HYPRE_IJVectorCreate(comm_, ilower_, iupper_, &x_);

    // Use parallel csr format
    HYPRE_IJVectorSetObjectType(x_, HYPRE_PARCSR);

    // Tell HYPRE that no vector entries need to be communicated to other
    // processors
    HYPRE_IJVectorSetMaxOffProcElmts(x_, 0);

    // Initialize
    HYPRE_IJVectorInitialize(x_);

    // Initialize exact structure of vector. To achieve this, we set every
    // element to zero.
    const int local_size = iupper_ - ilower_ + 1;

    this->global_indices_.resize(local_size);

    std::vector< HYPRE_Complex > val(local_size,
                                     static_cast< HYPRE_Complex >(0));

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

    HYPRE_Int nvals = local_size;

    std::vector< HYPRE_Int > indices_temp(local_size);
    for (size_t i = 0; i < indices_temp.size(); ++i) {
      indices_temp[i] = static_cast< HYPRE_Int >(this->global_indices_[i]);
    }
    HYPRE_IJVectorSetValues(x_, nvals, vec2ptr(indices_temp), vec2ptr(val));

    // Finalize initialization of vector
    HYPRE_IJVectorAssemble(x_);
    HYPRE_IJVectorGetObject(x_, (void **)&parcsr_x_);

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

    this->interior_ = vec.interior().CloneWithoutContent();
    this->interior_->CopyStructureFrom(vec.interior());
    
    this->initialized_ = true;

    val.clear();
  }
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType >
void HypreVector< DataType >::print_statistics() const {
#ifdef WITH_HYPRE
  int my_rank, num_procs;
  MPI_Comm_rank(comm_, &my_rank);
  MPI_Comm_size(comm_, &num_procs);

  HYPRE_Int jlower, jupper;
  HYPRE_IJVectorGetLocalRange(x_, &jlower, &jupper);

  // print statistics
  for (int i = 0; i < num_procs; ++i) {
    MPI_Barrier(comm_);
    if (i == my_rank) {
      std::cout << "HypreVector on process " << my_rank << ":" << std::endl;
      // print size information
      std::cout << "\t jlower: " << jlower << std::endl;
      std::cout << "\t jupper: " << jupper << std::endl;
    }
    MPI_Barrier(comm_);
  }
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType > int HypreVector< DataType >::size_local() const {
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());
  return this->cp_->nb_dofs(my_rank_);
}

template < class DataType > int HypreVector< DataType >::size_global() const {
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());
  return this->cp_->nb_total_dofs();
}

template < class DataType > void HypreVector< DataType >::Zeros() {
#ifdef WITH_HYPRE
  HYPRE_ParVectorSetConstantValues(parcsr_x_, static_cast< DataType >(0));
  this->ghost_->Zeros();
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}
   
template < class DataType >
DataType HypreVector< DataType >::GetValue(const int index) const {
  DataType val;
  this->GetValues(&index, 1, &val);
  return val;
}

template < class DataType >
void HypreVector< DataType >::GetLocalValues(DataType *values) const {
#ifdef WITH_HYPRE
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());

  const size_t local_length = this->iupper_ - this->ilower_ + 1;

  std::vector< HYPRE_Int > indices_interior(local_length);
  HYPRE_Int ind_interior = 0;

  for (HYPRE_Int i = this->ilower_; i <= this->iupper_; i++) {
    indices_interior[ind_interior] = i;
    ++ind_interior;
  }

  assert(ind_interior == local_length);
  if (ind_interior > 0) {
    std::vector< HYPRE_Complex > val_temp(local_length);
    HYPRE_IJVectorGetValues(x_, ind_interior, vec2ptr(indices_interior),
                            vec2ptr(val_temp));
    for (size_t i = 0; i < val_temp.size(); ++i) {
      values[i] = static_cast< DataType >(val_temp[i]);
    }
  }
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType >
void HypreVector< DataType >::GetLocalValues(const int *indices,
    const int size_indices, DataType *values) const {

#ifdef WITH_HYPRE
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());

  if (size_indices > 0) {
    assert(indices != 0);
    assert(values != 0);

    if (size_indices == 1) {
      const size_t local_length = this->iupper_ - this->ilower_;
      // interior
      if ((unsigned)(indices[0] - this->ilower_) <= local_length) {
        HYPRE_Int ind_temp = static_cast< HYPRE_Int >(indices[0]);
        HYPRE_Complex val_temp;
        HYPRE_IJVectorGetValues(x_, 1, &ind_temp, &val_temp);
        values[0] = static_cast< DataType >(val_temp);
      }

    } else {

      // extract interior and ghost indices (and if possible pp indices)
      std::vector< HYPRE_Int > indices_interior(size_indices);
      HYPRE_Int ind_interior = 0;

      const size_t local_length = this->iupper_ - this->ilower_;

      // transform indices according to local numbering of interior and ghost
      for (int i = 0; i < size_indices; i++) {
        // interior
        if ((unsigned)(indices[i] - this->ilower_) <= local_length) {
          indices_interior[ind_interior] = indices[i];
          ++ind_interior;
        }
      }

      // extract values
      std::vector< HYPRE_Complex > values_interior(ind_interior);
      if (ind_interior > 0) {
        HYPRE_IJVectorGetValues(x_, ind_interior, vec2ptr(indices_interior),
                                vec2ptr(values_interior));
      }

      // put values together
      for (int i = 0; i < size_indices; i++) {
        const size_t current_index = size_indices - i - 1;
        const size_t index_current = indices[current_index];
        // interior
        if ((unsigned)(index_current - this->ilower_) <= local_length) {
          values[current_index] =
            static_cast< DataType >(values_interior[--ind_interior]);
        }
      }

      assert(ind_interior == 0);
      indices_interior.clear();
      values_interior.clear();
    }
  }
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  exit(-1);
#endif
}

template < class DataType >
void HypreVector< DataType >::GetValues(const int *indices,
                                        const int size_indices,
                                        DataType *values) const {
#ifdef WITH_HYPRE
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());

  if (size_indices > 0) {
    assert(indices != 0);
    assert(values != 0);

    if (size_indices == 1) {
      const size_t local_length = this->iupper_ - this->ilower_;
      // interior
      if ((unsigned)(indices[0] - this->ilower_) <= local_length) {
        HYPRE_Int ind_temp = static_cast< HYPRE_Int >(indices[0]);
        HYPRE_Complex val_temp;
        HYPRE_IJVectorGetValues(x_, 1, &ind_temp, &val_temp);
        values[0] = static_cast< DataType >(val_temp);
      }

      // ghost
      else if (this->cp_->global2offdiag().find(indices[0]) !=
               this->cp_->global2offdiag().end()) {
        int index_ghost = this->cp_->Global2Offdiag(indices[0]);
        this->ghost_->GetValues(&index_ghost, 1, values);
      }
    } else {

      // extract interior and ghost indices (and if possible pp indices)
      std::vector< HYPRE_Int > indices_interior(size_indices);
      HYPRE_Int ind_interior = 0;

      std::vector< int > indices_ghost(size_indices);
      int ind_ghost = 0;

      const size_t local_length = this->iupper_ - this->ilower_;

      // transform indices according to local numbering of interior and ghost
      for (int i = 0; i < size_indices; i++) {
        // interior
        if ((unsigned)(indices[i] - this->ilower_) <= local_length) {
          indices_interior[ind_interior] = indices[i];
          ++ind_interior;
        }

        // ghost
        else if (this->cp_->global2offdiag().find(indices[i]) !=
                 this->cp_->global2offdiag().end()) {
          indices_ghost[ind_ghost] = this->cp_->Global2Offdiag(indices[i]);
          ++ind_ghost;
        }
      }

      // extract values
      std::vector< HYPRE_Complex > values_interior(ind_interior);
      if (ind_interior > 0) {
        HYPRE_IJVectorGetValues(x_, ind_interior, vec2ptr(indices_interior),
                                vec2ptr(values_interior));
      }

      std::vector< DataType > values_ghost(ind_ghost);

      if (ind_ghost > 0) {
        this->ghost_->GetValues(&indices_ghost.front(), ind_ghost,
                                &values_ghost.front());
      }

      // put values together
      for (int i = 0; i < size_indices; i++) {
        const size_t current_index = size_indices - i - 1;
        const size_t index_current = indices[current_index];
        // interior
        if ((unsigned)(index_current - this->ilower_) <= local_length) {
          values[current_index] =
            static_cast< DataType >(values_interior[--ind_interior]);
        }

        // ghost
        else if (this->cp_->global2offdiag().find(index_current) !=
                 this->cp_->global2offdiag().end()) {
          values[current_index] = values_ghost[--ind_ghost];
        }
        // zeros
        else {
          values[current_index] = static_cast< DataType >(0.);
        }
      }

      assert(ind_interior == 0);
      assert(ind_ghost == 0);
      indices_interior.clear();
      indices_ghost.clear();
      values_interior.clear();
      values_ghost.clear();
    }
  }
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType > 
void HypreVector< DataType >::update_interior() const 
{
  assert (this->interior_ != nullptr);
  assert (this->interior_->get_size() == this->size_local());
#ifdef WITH_HYPRE
  this->GetLocalValues(this->interior_->GetBuffer());
#endif
}

template < class DataType > 
DataType HypreVector< DataType >::Norm2() const {
  return std::sqrt(this->Dot(*this));
}

template < class DataType > 
DataType HypreVector< DataType >::Norm1() const {
  LOG_ERROR("Called HypreVector::Norm1 not yet implemented!!!");
  quit_program();
  return -1.;
}

template < class DataType > 
DataType HypreVector< DataType >::NormMax() const {
  LOG_ERROR("Called HypreVector::NormMax not yet implemented!!!");
  quit_program();
  return -1.;
}

template < class DataType >
DataType HypreVector< DataType >::Dot(const Vector< DataType > &vec) const {
  const HypreVector< DataType > *hv =
    dynamic_cast< const HypreVector< DataType > * >(&vec);

  if (hv != 0) {
    return this->Dot(*hv);
  }

  LOG_ERROR("Called HypreVector::Dot with incompatible vector type.");
  quit_program();
  return -1.;
}

template < class DataType >
DataType HypreVector< DataType >::Dot(const HypreVector< DataType > &vec) const {
#ifdef WITH_HYPRE
  HYPRE_Complex result = 0.;
  HYPRE_ParVectorInnerProd(parcsr_x_, *(vec.GetParVector()), &result);
  return static_cast< DataType >(result);
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
  return -1.;
#endif
}

template < class DataType >
void HypreVector< DataType >::Add(const int index, const DataType scalar) {
  this->Add(&index, 1, &scalar);
}

template < class DataType >
void HypreVector< DataType >::Add(const int *indices, const int length,
                                  const DataType *values) {
#ifdef WITH_HYPRE
  std::vector< HYPRE_Int > indices_interior;
  std::vector< HYPRE_Complex > values_interior;
  indices_interior.reserve(length);
  values_interior.reserve(length);
  for (int i = 0; i < length; ++i) {
    if (indices[i] >= this->ilower_ && indices[i] <= this->iupper_) {
      indices_interior.push_back(static_cast< HYPRE_Int >(indices[i]));
      values_interior.push_back(static_cast< HYPRE_Complex >(values[i]));
    }
  }
  HYPRE_IJVectorAddToValues(x_, static_cast< int >(indices_interior.size()),
                            vec2ptr(indices_interior),
                            vec2ptr(values_interior));
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType >
void HypreVector< DataType >::SetValue(const int index, const DataType value) {
  this->SetValues(&index, 1, &value);
}

template < class DataType >
void HypreVector< DataType >::SetValues(const int *indices,
                                        const int size_indices,
                                        const DataType *values) {
#ifdef WITH_HYPRE
  std::vector< HYPRE_Int > indices_interior;
  std::vector< HYPRE_Complex > values_interior;
  indices_interior.reserve(size_indices);
  values_interior.reserve(size_indices);
  for (int i = 0; i < size_indices; ++i) {
    if (indices[i] >= this->ilower_ && indices[i] <= this->iupper_) {
      indices_interior.push_back(static_cast< HYPRE_Int >(indices[i]));
      values_interior.push_back(static_cast< HYPRE_Complex >(values[i]));
    }
  }
  HYPRE_IJVectorSetValues(x_, static_cast< int >(indices_interior.size()),
                          vec2ptr(indices_interior), vec2ptr(values_interior));
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType >
void HypreVector< DataType >::SetLocalValues(const DataType *values) {
#ifdef WITH_HYPRE
  const size_t local_length =
    static_cast< size_t >(this->iupper_ - this->ilower_ + 1);

  std::vector< HYPRE_Int > indices_interior(local_length);
  std::vector< HYPRE_Complex > values_interior(local_length);
  for (size_t i = 0; i < local_length; ++i) {
    indices_interior[i] = static_cast< HYPRE_Int >(i) + this->ilower_;
    values_interior[i] = static_cast< HYPRE_Complex >(values[i]);
  }

  HYPRE_IJVectorSetValues(x_, static_cast< int >(indices_interior.size()),
                          vec2ptr(indices_interior), vec2ptr(values_interior));
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType >
void HypreVector< DataType >::SetToValue(const DataType value) {
#ifdef WITH_HYPRE
  const size_t local_length =
      static_cast< size_t >(this->iupper_ - this->ilower_ + 1);

  std::vector< HYPRE_Int > indices_interior(local_length);
  std::vector< HYPRE_Complex > values_interior(local_length);
  for (size_t i = 0; i < local_length; ++i) {
    indices_interior[i] = static_cast< HYPRE_Int >(i) + this->ilower_;
    values_interior[i] = static_cast< HYPRE_Complex >(value);
  }

  HYPRE_IJVectorSetValues(x_, static_cast< int >(indices_interior.size()),
                          vec2ptr(indices_interior), vec2ptr(values_interior));
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType >
void HypreVector< DataType >::Axpy(const Vector< DataType > &vecx,
                                   const DataType alpha) {
  const HypreVector< DataType > *hv =
    dynamic_cast< const HypreVector< DataType > * >(&vecx);

  if (hv != 0) {
    this->Axpy(*hv, alpha);
  } 
  else 
  {
    LOG_INFO("Warning", "Called HypreVector::Axpy with incompatible vector type. Use slow axpy version");
    scale_axpy(1., *this, alpha, vecx); 
  }
}

template < class DataType >
void HypreVector< DataType >::Axpy(const HypreVector< DataType > &vecx,
                                   const DataType alpha) {
#ifdef WITH_HYPRE
  HYPRE_ParVectorAxpy(alpha, *(vecx.GetParVector()), parcsr_x_);
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType >
void HypreVector< DataType >::ScaleAdd(const Vector< DataType > &vecx,
                                       const DataType alpha) {
  this->Scale(alpha);
  this->Axpy(vecx, static_cast< DataType >(1.));
}

template < class DataType >
void HypreVector< DataType >::Scale(const DataType alpha) {
#ifdef WITH_HYPRE
  HYPRE_ParVectorScale(alpha, parcsr_x_);
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class DataType > 
void HypreVector< DataType >::SendBorder() {
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->nb_procs_);
  assert(this->comm_ != MPI_COMM_NULL);

  this->GetValues(vec2ptr(border_indices_), border_indices_.size(),
                  vec2ptr(border_val_));

  int tag = 1;
  int ctr = 0;

  for (int id = 0; id < this->nb_procs_; id++) {
    if (this->cp_->border_offsets(id + 1) - this->cp_->border_offsets(id) > 0) {
#ifndef NDEBUG
      int info =
#endif
        MPI_Isend(&(this->border_val_[0]) + this->cp_->border_offsets(id),
                  this->cp_->border_offsets(id + 1) -
                  this->cp_->border_offsets(id),
                  mpi_data_type< DataType >::get_type(), id, tag, this->comm_,
                  &(this->mpi_req_[this->nb_recvs_ + ctr]));
      assert(info == MPI_SUCCESS);
      ctr++;
    }
  }
  assert(ctr == this->nb_sends_);
}

template < class DataType > 
void HypreVector< DataType >::ReceiveGhost() {
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());
  assert(this->my_rank_ >= 0);
  assert(this->my_rank_ < this->nb_procs_);
  assert(this->comm_ != MPI_COMM_NULL);

  int tag = 1;
  int ctr = 0;

  for (int id = 0; id < this->nb_procs_; ++id) {
    if (this->cp_->ghost_offsets(id + 1) - this->cp_->ghost_offsets(id) > 0) {
#ifndef NDEBUG
      int info =
#endif
        MPI_Irecv(&(this->ghost_val_[0]) + this->cp_->ghost_offsets(id),
                  this->cp_->ghost_offsets(id + 1) -
                  this->cp_->ghost_offsets(id),
                  mpi_data_type< DataType >::get_type(), id, tag, this->comm_,
                  &(this->mpi_req_[ctr]));
      assert(info == MPI_SUCCESS);
      ctr++;
    }
  }
  assert(ctr == this->nb_recvs_);
}

template < class DataType > 
void HypreVector< DataType >::WaitForSend() {
#ifndef NDEBUG
  int info =
#endif
    MPI_Waitall(this->nb_sends_, &(this->mpi_req_[this->nb_recvs_]),
                &(this->mpi_stat_[this->nb_recvs_]));
  assert(info == MPI_SUCCESS);
}

template < class DataType > void HypreVector< DataType >::WaitForRecv() {
#ifndef NDEBUG
  int info =
#endif
    MPI_Waitall(this->nb_recvs_, &(this->mpi_req_[0]), &(this->mpi_stat_[0]));
  assert(info == MPI_SUCCESS);

  this->SetGhostValues(&(this->ghost_val_[0]));
}

template < class DataType > void HypreVector< DataType >::Update() {
  this->UpdateGhost();
}

template < class DataType > void HypreVector< DataType >::UpdateGhost() {
  assert(this->ghost_ != nullptr);

  this->ReceiveGhost();
  this->SendBorder();

  this->WaitForRecv();
  this->WaitForSend();
}

template < class DataType >
void HypreVector< DataType >::SetGhostValues(const DataType *values) {
  assert(this->ghost_ != nullptr);

  this->ghost_->SetBlockValues(0, this->size_local_ghost(), values);
}

template < class DataType >
void HypreVector< DataType >::GetAllDofsAndValues(
  std::vector< int > &id, std::vector< DataType > &val) const {
  assert(this->cp_ != nullptr);
  assert(this->cp_->initialized());
  assert(this->ghost_ != nullptr);

  // Temporary containers for interior and ghost values
  DataType *values = new DataType[this->size_local()];
  DataType *ghost_values = new DataType[this->size_local_ghost()];

  int total_size = this->size_local() + this->size_local_ghost();
  id.resize(total_size);

  // First: the DoFs from the interior
PRAGMA_LOOP_VEC
  for (int i = 0; i < this->size_local(); ++i) {
    id[i] = ilower_ + i;
  }

  // Combine interior, ghost and pp_data values
  this->GetValues(vec2ptr(id), this->size_local(), values);
  this->ghost_->GetBlockValues(0, this->size_local_ghost(), ghost_values);

  val.resize(total_size);
  // First: values from the interior
PRAGMA_LOOP_VEC
  for (size_t i = 0; i < static_cast< size_t >(this->size_local()); ++i) {
    val[i] = values[i];
  }

  delete[] values;

  // Second: the DoFs and values from ghost
  int tmp_offset = this->size_local();
PRAGMA_LOOP_VEC
  for (int i = 0; i < this->size_local_ghost(); ++i) {
    id[i + tmp_offset] = this->cp_->Offdiag2Global(i);
    val[i + tmp_offset] = ghost_values[i];
  }

  delete[] ghost_values;
}

#ifdef WITH_HYPRE

template < class DataType >
HYPRE_ParVector *HypreVector< DataType >::GetParVector() {
  return &parcsr_x_;
}

template < class DataType >
const HYPRE_ParVector *HypreVector< DataType >::GetParVector() const {
  return &parcsr_x_;
}
#endif

template < class DataType >
void HypreVector< DataType >::WriteHDF5(const std::string &filename,
                                        const std::string &groupname,
                                        const std::string &datasetname) {
  assert(this->comm_ != MPI_COMM_NULL);
  assert(this->is_initialized());
#ifdef WITH_HDF5
  // Define Data in memory
  const int local_size = this->size_local();
  std::vector< DataType > data(local_size, 0.);

  this->GetValues(vec2ptr(this->global_indices_), local_size, vec2ptr(data));

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
void HypreVector< DataType >::ReadHDF5(const std::string &filename,
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

  this->SetValues(vec2ptr(this->global_indices_), local_size, buffer);

  // Update
  this->Update();

  delete[] buffer;
#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

// template instantiation
template class HypreVector< double >;

} // namespace la
} // namespace hiflow
