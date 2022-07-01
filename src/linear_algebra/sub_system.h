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

/// \author Simon Gawlok, Philipp Gerstner

#ifndef HIFLOW_LINEARALGEBRA_SUB_SYSTEM_H_
#define HIFLOW_LINEARALGEBRA_SUB_SYSTEM_H_

#include <iomanip>
#include <map>
#include <mpi.h>
#include <vector>

#include "common/log.h"
#include "common/sorted_array.h"
#include "common/timer.h"
#include "config.h"
#include "linear_algebra/la_descriptor.h"
#include "mesh/iterator.h"
#include "mesh/types.h"
#include "space/vector_space.h"

namespace hiflow {
namespace la {

/// \brief Schur complement solver interface

template < int DIM, class LAD, class SUBLAD = LAD > 
class SubSystem 
{
public:
  typedef typename LAD::MatrixType MatrixType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;
  
  typedef typename SUBLAD::MatrixType SubMatrixType;
  typedef typename SUBLAD::VectorType SubVectorType;

  /// standard constructor
  SubSystem();

  /// destructor
  virtual ~SubSystem();

  /// Init the Schur complement solver. Generates internally the needed
  /// sparsity structures for the submatrices and the mappings of the system
  /// DoFs to the new block-wise enumeration and vice versa.
  /// \param space Finite element space
  /// \param block_one_variables Variable numbers that belong to the first block
  /// \param block_two_variables Variable numbers that belong to the second
  /// block \param precond_two_type Type of Schur complement preconditioning
  /// \param dpx_precond_type Type of matrix in case of DPXPrecond
  virtual void Init(const hiflow::VectorSpace< DataType, DIM > &space,
                    const std::vector< int > &block_one_variables,
                    const std::vector< int > &block_two_variables);


  void InitMatrixA(SubMatrixType * A) const;
  void InitMatrixB(SubMatrixType * B) const;
  void InitMatrixC(SubMatrixType * C) const;
  void InitMatrixD(SubMatrixType * D) const;

  void InitVectorOne(SubVectorType * vec) const;
  void InitVectorTwo(SubVectorType * vec) const;

  void ExtractA(const MatrixType &op, SubMatrixType* A) const;
  void ExtractB(const MatrixType &op, SubMatrixType* B) const;
  void ExtractC(const MatrixType &op, SubMatrixType* C) const;
  void ExtractD(const MatrixType &op, SubMatrixType* D) const;
  
  void ExtractVectorOne(const VectorType &b, SubVectorType* b1) const;
  void ExtractVectorTwo(const VectorType &b, SubVectorType* b2) const;

  void InsertVectorOne(const SubVectorType &b1, VectorType* b) const;
  void InsertVectorTwo(const SubVectorType &b2, VectorType* b) const;
  
  void AddVectorOne(const SubVectorType &b1, VectorType* b, DataType scale) const;
  void AddVectorTwo(const SubVectorType &b2, VectorType* b, DataType scale) const;

  /// Clears allocated data.
  virtual void Clear();

protected:
  /// MPI communicator
  MPI_Comm comm_;
  /// Rank of current process
  int my_rank_;
  /// Global number of processes
  int nb_procs_;

  /// Number of variables in the first block
  int num_var_one_;
  /// Number of variables in the second block
  int num_var_two_;

  /// Couplings of submatrix \f$A\f$ in system numbering
  std::map< int, SortedArray< int > > couplings_A_;
  /// Couplings of submatrix \f$B\f$ in system numbering
  std::map< int, SortedArray< int > > couplings_B_;
  /// Couplings of submatrix \f$C\f$ in system numbering
  std::map< int, SortedArray< int > > couplings_C_;
  /// Couplings of submatrix \f$D\f$ in system numbering
  std::map< int, SortedArray< int > > couplings_D_;

  /// Sparsity structure of submatrix \f$A\f$ in block numbering
  std::vector< std::vector< int > > sparsity_A_;
  /// Sparsity structure of submatrix \f$B\f$ in block numbering
  std::vector< std::vector< int > > sparsity_B_;
  /// Sparsity structure of submatrix \f$C\f$ in block numbering
  std::vector< std::vector< int > > sparsity_C_;
  /// Sparsity structure of submatrix \f$D\f$ in block numbering
  std::vector< std::vector< int > > sparsity_D_;

  /// Offsets of the processes in the block numbering of the first block
  std::vector< int > offsets_block_one_;
  /// Offsets of the processes in the block numbering of the second block
  std::vector< int > offsets_block_two_;
  /// Mapping block->system numbering of first block
  std::vector< int > mapb2s_one_;
  /// Mapping block->system numbering of second block
  std::vector< int > mapb2s_two_;
  /// Mapping system->block numbering of first block
  std::map< int, int > maps2b_one_;
  /// Mapping system->block numbering of second block
  std::map< int, int > maps2b_two_;

  /// LaCouplings for first block
  LaCouplings la_c_one_;
  /// LaCouplings for second block
  LaCouplings la_c_two_;

  /// Global indices of DoFs in first block
  std::vector< int > indexset_one_;
  /// Global indices of DoFs in second block
  std::vector< int > indexset_two_;

  /// Vector for temporary values in first block
  mutable std::vector< DataType > val_temp_one_;
  /// Vector for temporary values in second block
  mutable std::vector< DataType > val_temp_two_;
  
  bool initialized_;
};

template < int DIM, class LAD, class SUBLAD >
SubSystem< DIM, LAD, SUBLAD >::SubSystem()
{
  this->comm_ = MPI_COMM_NULL;
  this->nb_procs_ = -1;
  this->my_rank_ = -1;
  this->initialized_ = false;
}

template < int DIM, class LAD, class SUBLAD >
SubSystem< DIM, LAD, SUBLAD >::~SubSystem() 
{
  this->Clear();
}


template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::Clear() 
{
  if (this->comm_ != MPI_COMM_NULL) 
  {
    MPI_Comm_free(&this->comm_);
    assert(this->comm_ == MPI_COMM_NULL);
  }

  this->my_rank_ = -1;
  this->nb_procs_ = -1;
  this->couplings_A_.clear();
  this->couplings_B_.clear();
  this->couplings_C_.clear();
  this->couplings_D_.clear();
  this->sparsity_A_.clear();
  this->sparsity_B_.clear();
  this->sparsity_C_.clear();
  this->sparsity_D_.clear();
  this->offsets_block_one_.clear();
  this->offsets_block_two_.clear();
  this->mapb2s_one_.clear();
  this->mapb2s_two_.clear();
  this->maps2b_one_.clear();
  this->maps2b_two_.clear();
  this->la_c_one_.Clear();
  this->la_c_two_.Clear();
  this->indexset_one_.clear();
  this->indexset_two_.clear();
  this->val_temp_one_.clear();
  this->val_temp_two_.clear();
  this->initialized_ = false;
}


template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::Init(const hiflow::VectorSpace< DataType, DIM > &space,
                                    const std::vector< int > &block_one_variables,
                                    const std::vector< int > &block_two_variables) 
{
  // clear member variables
  this->Clear();

  //*****************************************************************
  // Set number of variables in the blocks
  //*****************************************************************
  num_var_one_ = block_one_variables.size();
  num_var_two_ = block_two_variables.size();

  //*****************************************************************
  // Create own duplicate of MPI communicator of space
  //*****************************************************************
  if (this->comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&this->comm_);
  }

  assert(space.get_mpi_comm() != MPI_COMM_NULL);
  // MPI communicator

  // determine nb. of processes
  int info = MPI_Comm_size(space.get_mpi_comm(), &nb_procs_);
  assert(info == MPI_SUCCESS);
  assert(nb_procs_ > 0);

  // retrieve my rank
  info = MPI_Comm_rank(space.get_mpi_comm(), &my_rank_);
  assert(info == MPI_SUCCESS);
  assert(my_rank_ >= 0);
  assert(my_rank_ < nb_procs_);

  info = MPI_Comm_split(space.get_mpi_comm(), 0, my_rank_, &(this->comm_));
  assert(info == MPI_SUCCESS);

  //*****************************************************************
  // Create couplings/sparsity structure in system numbering
  //*****************************************************************
  couplings_A_.clear();
  couplings_B_.clear();
  couplings_C_.clear();
  couplings_D_.clear();

  typename VectorSpace< DataType, DIM >::MeshEntityIterator mesh_it =
      space.mesh().begin(space.tdim());
  typename VectorSpace< DataType, DIM >::MeshEntityIterator e_mesh_it =
      space.mesh().end(space.tdim());

  while (mesh_it != e_mesh_it) 
  {

    std::vector< doffem::gDofId > dof_ind;

    std::vector< std::vector< doffem::gDofId > > dof_ind_block_one;
    dof_ind_block_one.resize(block_one_variables.size());
    std::vector< std::vector< doffem::gDofId > > dof_ind_block_two;
    dof_ind_block_two.resize(block_two_variables.size());

    for (int k = 0, k_e = block_one_variables.size(); k != k_e; ++k) 
    {
      const int var = block_one_variables[k];
      // get dof indices for variable
      space.get_dof_indices(var, mesh_it->index(), dof_ind);
      dof_ind_block_one[k] = dof_ind;

      // clear dof_ind vector
      dof_ind.clear();
    }

    for (int k = 0, k_e = block_two_variables.size(); k != k_e; ++k) 
    {
      const int var = block_two_variables[k];
      // get dof indices for variable
      space.get_dof_indices(var, mesh_it->index(), dof_ind);
      dof_ind_block_two[k] = dof_ind;

      // clear dof_ind vector
      dof_ind.clear();
    }

    // sub matrix A, sub matrix B
    for (int i = 0, i_e = dof_ind_block_one.size(); i != i_e; ++i) 
    {

      // detect couplings
      for (int ii = 0, ii_e = dof_ind_block_one[i].size(); ii != ii_e; ++ii) {
        // dof indice of test variable
        const doffem::gDofId di_i = dof_ind_block_one[i][ii];

        // if my row
        if (space.dof().is_dof_on_subdom(di_i)) {
          // matrix A
          SortedArray< doffem::gDofId > *temp = &(couplings_A_[di_i]);

          for (size_t j = 0, j_e = dof_ind_block_one.size(); j != j_e; ++j) {
            for (size_t jj = 0, jj_e = dof_ind_block_one[j].size(); jj != jj_e;
                 ++jj) {
              temp->find_insert(dof_ind_block_one[j][jj]);
            }
          }

          // matrix B
          temp = &(couplings_B_[di_i]);

          for (size_t j = 0, j_e = dof_ind_block_two.size(); j != j_e; ++j) {
            for (size_t jj = 0, jj_e = dof_ind_block_two[j].size(); jj != jj_e;
                 ++jj) {
              temp->find_insert(dof_ind_block_two[j][jj]);
            }
          }
        } // if
      }   // for (int ii = 0 ...)
    }     // for (int i = 0 ...)

    // sub matrix C, sub matrix D
    for (int i = 0, i_e = dof_ind_block_two.size(); i != i_e; ++i) {

      // detect couplings
      for (int ii = 0, ii_e = dof_ind_block_two[i].size(); ii != ii_e; ++ii) {
        // dof indice of test variable
        const doffem::gDofId di_i = dof_ind_block_two[i][ii];

        // if my row
        if (space.dof().is_dof_on_subdom(di_i)) {
          // matrix C
          SortedArray< doffem::gDofId > *temp = &(couplings_C_[di_i]);

          for (size_t j = 0, j_e = dof_ind_block_one.size(); j != j_e; ++j) {
            for (size_t jj = 0, jj_e = dof_ind_block_one[j].size(); jj != jj_e;
                 ++jj) {
              temp->find_insert(dof_ind_block_one[j][jj]);
            }
          }

          // matrix D
          temp = &(couplings_D_[di_i]);

          for (size_t j = 0, j_e = dof_ind_block_two.size(); j != j_e; ++j) {
            for (size_t jj = 0, jj_e = dof_ind_block_two[j].size(); jj != jj_e;
                 ++jj) {
              temp->find_insert(dof_ind_block_two[j][jj]);
            }
          }
        } // if
      }   // for (int ii = 0 ...)
    }     // for (int i = 0 ...)
    // next cell
    ++mesh_it;
  } // while (mesh_it != ...)

  //*****************************************************************
  // Compute offsets for variables in block one and variables in
  // block two
  //*****************************************************************

  offsets_block_one_.clear();
  offsets_block_two_.clear();
  offsets_block_one_.resize(this->nb_procs_ + 1, 0);
  offsets_block_two_.resize(this->nb_procs_ + 1, 0);

  std::vector< int > local_dof_numbers_one;
  local_dof_numbers_one.reserve(this->nb_procs_);
  std::vector< int > local_dof_numbers_two;
  local_dof_numbers_two.reserve(this->nb_procs_);

  {
    std::vector< int > num_local_dofs(2, -1);
    num_local_dofs[0] = couplings_A_.size();
    num_local_dofs[1] = couplings_C_.size();

    std::vector< int > local_dof_numbers_temp(2 * this->nb_procs_, -1);

    MPI_Allgather(vec2ptr(num_local_dofs), 2, MPI_INT,
                  vec2ptr(local_dof_numbers_temp), 2, MPI_INT, this->comm_);

    // Unpack received data
    for (int i = 0; i < local_dof_numbers_temp.size(); i = i + 2) {
      local_dof_numbers_one.push_back(local_dof_numbers_temp[i]);
      local_dof_numbers_two.push_back(local_dof_numbers_temp[i + 1]);
    }
  }

#ifndef NDEBUG
  for (int i = 0; i < this->nb_procs_; ++i) {
    assert(local_dof_numbers_one[i] > -1);
    assert(local_dof_numbers_two[i] > -1);
  }
#endif

  for (int i = 1; i < this->nb_procs_ + 1; ++i) {
    offsets_block_one_[i] =
        offsets_block_one_[i - 1] + local_dof_numbers_one[i - 1];
    offsets_block_two_[i] =
        offsets_block_two_[i - 1] + local_dof_numbers_two[i - 1];
  }

  //*****************************************************************
  // Compute mappings: system->block and block->system
  //*****************************************************************
  mapb2s_one_.clear();
  mapb2s_two_.clear();

  maps2b_one_.clear();
  maps2b_two_.clear();
  int index = 0;
  mapb2s_one_.resize(local_dof_numbers_one[this->my_rank_], 0);
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_A_.begin(),
           e_it = couplings_A_.end();
       it != e_it; ++it) {
    mapb2s_one_[index] = it->first;
    maps2b_one_[it->first] = index + offsets_block_one_[this->my_rank_];
    ++index;
  }
  index = 0;
  mapb2s_two_.resize(local_dof_numbers_two[this->my_rank_], 0);
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_C_.begin(),
           e_it = couplings_C_.end();
       it != e_it; ++it) {
    mapb2s_two_[index] = it->first;
    maps2b_two_[it->first] = index + offsets_block_two_[this->my_rank_];
    ++index;
  }

  local_dof_numbers_one.clear();
  local_dof_numbers_two.clear();

  //*****************************************************************
  // Compute mappings: system->block of ghost dofs
  //*****************************************************************

  // determine which dofs are ghost and who is owner

  std::vector< SortedArray< int > > ghost_dofs_one;
  std::vector< SortedArray< int > > ghost_dofs_two;

  ghost_dofs_one.resize(this->nb_procs_);
  ghost_dofs_two.resize(this->nb_procs_);

  // sub matrix A
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_A_.begin(),
           e_it = couplings_A_.end();
       it != e_it; ++it) {
    for (int i = 0, e_i = it->second.size(); i != e_i; ++i) {
      if (!space.dof().is_dof_on_subdom(it->second[i])) {
        ghost_dofs_one[space.dof().owner_of_dof(it->second[i])].find_insert(
            it->second[i]);
      }
    }
  }
  // sub matrix C
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_C_.begin(),
           e_it = couplings_C_.end();
       it != e_it; ++it) {
    for (int i = 0, e_i = it->second.size(); i != e_i; ++i) {
      if (!space.dof().is_dof_on_subdom(it->second[i])) {
        ghost_dofs_one[space.dof().owner_of_dof(it->second[i])].find_insert(
            it->second[i]);
      }
    }
  }
  // sub matrix B
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_B_.begin(),
           e_it = couplings_B_.end();
       it != e_it; ++it) {
    for (int i = 0, e_i = it->second.size(); i != e_i; ++i) {
      if (!space.dof().is_dof_on_subdom(it->second[i])) {
        ghost_dofs_two[space.dof().owner_of_dof(it->second[i])].find_insert(
            it->second[i]);
      }
    }
  }
  // sub matrix D
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    for (int i = 0, e_i = it->second.size(); i != e_i; ++i) {
      if (!space.dof().is_dof_on_subdom(it->second[i])) {
        ghost_dofs_two[space.dof().owner_of_dof(it->second[i])].find_insert(
            it->second[i]);
      }
    }
  }

  // Each process exchanges with each process number of shared dofs and
  // determines block-dof-ids of ghost dofs
  std::vector< std::vector< int > > ghost_dofs_block_one;
  std::vector< std::vector< int > > ghost_dofs_block_two;

  ghost_dofs_block_one.resize(this->nb_procs_);
  ghost_dofs_block_two.resize(this->nb_procs_);

  {
    // BLOCK ONE FIRST

    // numbers of DoFs which I request of other processes
    std::vector< int > num_dofs_requested_block_one(this->nb_procs_, 0);
    int total_number_dofs_requested = 0;
    for (int i = 0; i < this->nb_procs_; ++i) {
      num_dofs_requested_block_one[i] = ghost_dofs_one[i].size();
      total_number_dofs_requested += ghost_dofs_one[i].size();
    }

    // numbers of DoFs which other processes request from me
    std::vector< int > num_dofs_requested_block_one_others(this->nb_procs_, 0);

    // Exchange requested numbers of DoFs
    MPI_Alltoall(vec2ptr(num_dofs_requested_block_one), 1, MPI_INT,
                 vec2ptr(num_dofs_requested_block_one_others), 1, MPI_INT,
                 this->comm_);

    // DoF numbers which I need from others
    std::vector< int > dofs_requested_block_one;
    dofs_requested_block_one.reserve(total_number_dofs_requested);

    std::vector< int > offsets_dofs_requested_block_one(this->nb_procs_ + 1, 0);
    for (int i = 0; i < this->nb_procs_; ++i) {
      dofs_requested_block_one.insert(dofs_requested_block_one.end(),
                                      ghost_dofs_one[i].begin(),
                                      ghost_dofs_one[i].end());
      offsets_dofs_requested_block_one[i + 1] =
          offsets_dofs_requested_block_one[i] + ghost_dofs_one[i].size();
    }

    // DoF numbers which others request from me
    std::vector< int > dofs_requested_block_one_others;
    int total_number_dofs_requested_others = 0;

    std::vector< int > offsets_dofs_requested_block_one_others(
        this->nb_procs_ + 1, 0);
    for (int i = 0; i < this->nb_procs_; ++i) {
      offsets_dofs_requested_block_one_others[i + 1] =
          offsets_dofs_requested_block_one_others[i] +
          num_dofs_requested_block_one_others[i];
      total_number_dofs_requested_others +=
          num_dofs_requested_block_one_others[i];
    }
    dofs_requested_block_one_others.resize(total_number_dofs_requested_others,0);

    // Exchange requested DoF IDs
    MPI_Alltoallv(vec2ptr(dofs_requested_block_one),
                  vec2ptr(num_dofs_requested_block_one),
                  vec2ptr(offsets_dofs_requested_block_one), MPI_INT,
                  vec2ptr(dofs_requested_block_one_others),
                  vec2ptr(num_dofs_requested_block_one_others),
                  vec2ptr(offsets_dofs_requested_block_one_others), MPI_INT,
                  this->comm_);

    // Prepare DoF IDs which others need from me
    for (int k = 0; k < total_number_dofs_requested_others; ++k) {
      assert(maps2b_one_.find(dofs_requested_block_one_others[k]) !=
             maps2b_one_.end());
      dofs_requested_block_one_others[k] =
          maps2b_one_[dofs_requested_block_one_others[k]];
    }

    // Exchange mapped DoF IDs
    MPI_Alltoallv(vec2ptr(dofs_requested_block_one_others),
                  vec2ptr(num_dofs_requested_block_one_others),
                  vec2ptr(offsets_dofs_requested_block_one_others), MPI_INT,
                  vec2ptr(dofs_requested_block_one),
                  vec2ptr(num_dofs_requested_block_one),
                  vec2ptr(offsets_dofs_requested_block_one), MPI_INT,
                  this->comm_);

    // Unpack received data
    for (int i = 0; i < this->nb_procs_; ++i) {
      ghost_dofs_block_one[i].resize(num_dofs_requested_block_one[i], -1);
      for (int j = 0; j < num_dofs_requested_block_one[i]; ++j) {
        ghost_dofs_block_one[i][j] =
            dofs_requested_block_one[offsets_dofs_requested_block_one[i] + j];
      }
    }
  }

  {
    // BLOCK TWO SECOND

    // numbers of DoFs which I request of other processes
    std::vector< int > num_dofs_requested_block_two(this->nb_procs_, 0);
    int total_number_dofs_requested = 0;
    for (int i = 0; i < this->nb_procs_; ++i) {
      num_dofs_requested_block_two[i] = ghost_dofs_two[i].size();
      total_number_dofs_requested += ghost_dofs_two[i].size();
    }

    // numbers of DoFs which other processes request from me
    std::vector< int > num_dofs_requested_block_two_others(this->nb_procs_, 0);

    // Exchange requested numbers of DoFs
    MPI_Alltoall(vec2ptr(num_dofs_requested_block_two), 1, MPI_INT,
                 vec2ptr(num_dofs_requested_block_two_others), 1, MPI_INT,
                 this->comm_);

    // DoF numbers which I need from others
    std::vector< int > dofs_requested_block_two;
    dofs_requested_block_two.reserve(total_number_dofs_requested);

    std::vector< int > offsets_dofs_requested_block_two(this->nb_procs_ + 1, 0);
    for (int i = 0; i < this->nb_procs_; ++i) {
      dofs_requested_block_two.insert(dofs_requested_block_two.end(),
                                      ghost_dofs_two[i].begin(),
                                      ghost_dofs_two[i].end());
      offsets_dofs_requested_block_two[i + 1] =
          offsets_dofs_requested_block_two[i] + ghost_dofs_two[i].size();
    }

    // DoF numbers which others request from me
    std::vector< int > dofs_requested_block_two_others;
    int total_number_dofs_requested_others = 0;

    std::vector< int > offsets_dofs_requested_block_two_others(
        this->nb_procs_ + 1, 0);
    for (int i = 0; i < this->nb_procs_; ++i) 
    {
      offsets_dofs_requested_block_two_others[i + 1] =
          offsets_dofs_requested_block_two_others[i] +
          num_dofs_requested_block_two_others[i];
      total_number_dofs_requested_others +=
          num_dofs_requested_block_two_others[i];
    }
    dofs_requested_block_two_others.resize(total_number_dofs_requested_others,0);

    // Exchange requested DoF IDs
    MPI_Alltoallv(vec2ptr(dofs_requested_block_two),
                  vec2ptr(num_dofs_requested_block_two),
                  vec2ptr(offsets_dofs_requested_block_two), MPI_INT,
                  vec2ptr(dofs_requested_block_two_others),
                  vec2ptr(num_dofs_requested_block_two_others),
                  vec2ptr(offsets_dofs_requested_block_two_others), MPI_INT,
                  this->comm_);

    // Prepare DoF IDs which others need from me
    for (int k = 0; k < total_number_dofs_requested_others; ++k) {
      assert(maps2b_two_.find(dofs_requested_block_two_others[k]) !=
             maps2b_two_.end());
      dofs_requested_block_two_others[k] =
          maps2b_two_[dofs_requested_block_two_others[k]];
    }

    // Exchange mapped DoF IDs
    MPI_Alltoallv(vec2ptr(dofs_requested_block_two_others),
                  vec2ptr(num_dofs_requested_block_two_others),
                  vec2ptr(offsets_dofs_requested_block_two_others), MPI_INT,
                  vec2ptr(dofs_requested_block_two),
                  vec2ptr(num_dofs_requested_block_two),
                  vec2ptr(offsets_dofs_requested_block_two), MPI_INT,
                  this->comm_);

    // Unpack received data
    for (int i = 0; i < this->nb_procs_; ++i) {
      ghost_dofs_block_two[i].resize(num_dofs_requested_block_two[i], -1);
      for (int j = 0; j < num_dofs_requested_block_two[i]; ++j) {
        ghost_dofs_block_two[i][j] =
            dofs_requested_block_two[offsets_dofs_requested_block_two[i] + j];
      }
    }
  }

  // build maps system->block
  std::map< int, int > maps2b_one_offdiag;
  std::map< int, int > maps2b_two_offdiag;

  for (int p = 0; p < this->nb_procs_; ++p) {
    for (int i = 0, e_i = ghost_dofs_one[p].data().size(); i != e_i; ++i) {
      maps2b_one_offdiag[ghost_dofs_one[p].data()[i]] =
          ghost_dofs_block_one[p][i];
    }

    for (int i = 0, e_i = ghost_dofs_two[p].data().size(); i != e_i; ++i) {
      maps2b_two_offdiag[ghost_dofs_two[p].data()[i]] =
          ghost_dofs_block_two[p][i];
    }
  }

  //*****************************************************************
  // Translate couplings from system to block numbering
  //*****************************************************************
  sparsity_A_.clear();
  sparsity_B_.clear();
  sparsity_C_.clear();
  sparsity_D_.clear();

  sparsity_A_.resize(couplings_A_.size());
  sparsity_B_.resize(couplings_B_.size());
  sparsity_C_.resize(couplings_C_.size());
  sparsity_D_.resize(couplings_D_.size());

  // sub matrix A
  index = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_A_.begin(),
           e_it = couplings_A_.end();
       it != e_it; ++it) {
    sparsity_A_[index].resize(it->second.size());
    for (int i = 0; i < it->second.size(); ++i) {
      if (maps2b_one_.find(it->second[i]) != maps2b_one_.end()) {
        sparsity_A_[index][i] = maps2b_one_[it->second[i]];
      } else {
        assert(maps2b_one_offdiag.find(it->second[i]) !=
               maps2b_one_offdiag.end());
        sparsity_A_[index][i] = maps2b_one_offdiag[it->second[i]];
      }
    }
    ++index;
  }
  // sub matrix B
  index = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_B_.begin(),
           e_it = couplings_B_.end();
       it != e_it; ++it) {
    sparsity_B_[index].resize(it->second.size());
    for (int i = 0; i < it->second.size(); ++i) {
      if (maps2b_two_.find(it->second[i]) != maps2b_two_.end()) {
        sparsity_B_[index][i] = maps2b_two_[it->second[i]];
      } else {
        assert(maps2b_two_offdiag.find(it->second[i]) !=
               maps2b_two_offdiag.end());
        sparsity_B_[index][i] = maps2b_two_offdiag[it->second[i]];
      }
    }
    ++index;
  }
  // sub matrix C
  index = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_C_.begin(),
           e_it = couplings_C_.end();
       it != e_it; ++it) {
    sparsity_C_[index].resize(it->second.size());
    for (int i = 0; i < it->second.size(); ++i) {
      if (maps2b_one_.find(it->second[i]) != maps2b_one_.end()) {
        sparsity_C_[index][i] = maps2b_one_[it->second[i]];
      } else {
        assert(maps2b_one_offdiag.find(it->second[i]) !=
               maps2b_one_offdiag.end());
        sparsity_C_[index][i] = maps2b_one_offdiag[it->second[i]];
      }
    }
    ++index;
  }
  // sub matrix D
  index = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    sparsity_D_[index].resize(it->second.size());
    for (int i = 0; i < it->second.size(); ++i) {
      if (maps2b_two_.find(it->second[i]) != maps2b_two_.end()) {
        sparsity_D_[index][i] = maps2b_two_[it->second[i]];
      } else {
        assert(maps2b_two_offdiag.find(it->second[i]) !=
               maps2b_two_offdiag.end());
        sparsity_D_[index][i] = maps2b_two_offdiag[it->second[i]];
      }
    }
    ++index;
  }

  //*****************************************************************
  // Setup LaCouplings for different blocks
  //*****************************************************************

  std::vector< int > offdiag_offsets_block_one(this->nb_procs_ + 1, 0);
  std::vector< int > offdiag_cols_block_one;

  for (size_t i = 0; i < ghost_dofs_block_one.size(); ++i) {
    offdiag_cols_block_one.insert(offdiag_cols_block_one.end(),
                                  ghost_dofs_block_one[i].begin(),
                                  ghost_dofs_block_one[i].end());
    offdiag_offsets_block_one[i + 1] =
        offdiag_offsets_block_one[i] + ghost_dofs_block_one[i].size();
  }

  std::vector< int > offdiag_offsets_block_two(this->nb_procs_ + 1, 0);
  std::vector< int > offdiag_cols_block_two;

  for (size_t i = 0; i < ghost_dofs_block_two.size(); ++i) {
    offdiag_cols_block_two.insert(offdiag_cols_block_two.end(),
                                  ghost_dofs_block_two[i].begin(),
                                  ghost_dofs_block_two[i].end());
    offdiag_offsets_block_two[i + 1] =
        offdiag_offsets_block_two[i] + ghost_dofs_block_two[i].size();
  }

  la_c_one_.Init(this->comm_);
  la_c_one_.InitializeCouplings(offsets_block_one_, offdiag_cols_block_one,
                                offdiag_offsets_block_one);
  la_c_two_.Init(this->comm_);
  la_c_two_.InitializeCouplings(offsets_block_two_, offdiag_cols_block_two,
                                offdiag_offsets_block_two);

  //*****************************************************************
  // Create block index sets
  //*****************************************************************
  this->indexset_one_.resize(this->mapb2s_one_.size());
  for (size_t i = 0, e_i = this->indexset_one_.size(); i != e_i; ++i) {
    this->indexset_one_[i] = i + offsets_block_one_[this->my_rank_];
  }

  this->indexset_two_.resize(this->mapb2s_two_.size());
  for (size_t i = 0, e_i = this->indexset_two_.size(); i != e_i; ++i) {
    this->indexset_two_[i] = i + offsets_block_two_[this->my_rank_];
  }

  this->val_temp_one_.resize(this->mapb2s_one_.size());
  this->val_temp_two_.resize(this->mapb2s_two_.size());
  
  this->initialized_ = true;
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::InitMatrixA(SubMatrixType * A) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  
  A->Init(this->comm_, la_c_one_, la_c_one_);

  std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;
  for (int i = 0; i < sparsity_A_.size(); ++i) {
    for (int j = 0; j < sparsity_A_[i].size(); ++j) {
      if ((sparsity_A_[i][j] >= offsets_block_one_[this->my_rank_]) &&
          (sparsity_A_[i][j] < offsets_block_one_[this->my_rank_ + 1])) {
        rows_diag.push_back(offsets_block_one_[this->my_rank_] + i);
        cols_diag.push_back(sparsity_A_[i][j]);
      } else {
        rows_offdiag.push_back(offsets_block_one_[this->my_rank_] + i);
        cols_offdiag.push_back(sparsity_A_[i][j]);
      }
    }
  }

  A->InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                   vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                   cols_offdiag.size());
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::InitMatrixB(SubMatrixType * B) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  assert (this->num_var_two_ > 0);
  
  B->Init(this->comm_, la_c_one_, la_c_two_);
 
  std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;
  for (int i = 0; i < sparsity_B_.size(); ++i) {
    for (int j = 0; j < sparsity_B_[i].size(); ++j) {
      if ((sparsity_B_[i][j] >= offsets_block_two_[this->my_rank_]) &&
          (sparsity_B_[i][j] < offsets_block_two_[this->my_rank_ + 1])) {
        rows_diag.push_back(offsets_block_one_[this->my_rank_] + i);
        cols_diag.push_back(sparsity_B_[i][j]);
      } else {
        rows_offdiag.push_back(offsets_block_one_[this->my_rank_] + i);
        cols_offdiag.push_back(sparsity_B_[i][j]);
      }
    }
  }

  B->InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                   vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                   cols_offdiag.size());
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::InitMatrixC(SubMatrixType * C) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  assert (this->num_var_two_ > 0);
  
  C->Init(this->comm_, la_c_two_, la_c_one_);

  std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;
  for (int i = 0; i < sparsity_C_.size(); ++i) {
    for (int j = 0; j < sparsity_C_[i].size(); ++j) {
      if ((sparsity_C_[i][j] >= offsets_block_one_[this->my_rank_]) &&
          (sparsity_C_[i][j] < offsets_block_one_[this->my_rank_ + 1])) {
        rows_diag.push_back(offsets_block_two_[this->my_rank_] + i);
        cols_diag.push_back(sparsity_C_[i][j]);
      } else {
        rows_offdiag.push_back(offsets_block_two_[this->my_rank_] + i);
        cols_offdiag.push_back(sparsity_C_[i][j]);
      }
    }
  }

  C->InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                   vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                   cols_offdiag.size());
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::InitMatrixD(SubMatrixType * D) const
{
  assert (this->initialized_);
  assert (this->num_var_two_ > 0);
  
  D->Init(this->comm_, la_c_two_, la_c_two_);

  std::vector< int > rows_diag, cols_diag, rows_offdiag, cols_offdiag;
  for (int i = 0; i < sparsity_D_.size(); ++i) {
    for (int j = 0; j < sparsity_D_[i].size(); ++j) {
      if ((sparsity_D_[i][j] >= offsets_block_two_[this->my_rank_]) &&
          (sparsity_D_[i][j] < offsets_block_two_[this->my_rank_ + 1])) {
        rows_diag.push_back(offsets_block_two_[this->my_rank_] + i);
        cols_diag.push_back(sparsity_D_[i][j]);
      } else {
        rows_offdiag.push_back(offsets_block_two_[this->my_rank_] + i);
        cols_offdiag.push_back(sparsity_D_[i][j]);
      }
    }
  }

  D->InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), cols_diag.size(),
                   vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                   cols_offdiag.size());
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::InitVectorOne(SubVectorType * vec) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);

  vec->Init(this->comm_, la_c_one_);
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::InitVectorTwo(SubVectorType * vec) const
{
  assert (this->initialized_);
  assert (this->num_var_two_ > 0);
  
  vec->Init(this->comm_, la_c_two_);
}

//*****************************************************************
// Operator setup means copying entries from system matrix op to
// submatrices A, B, C and D
//*****************************************************************
  
template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::ExtractA(const MatrixType &op, SubMatrixType* A) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_A_.begin(),
           e_it = couplings_A_.end();
       it != e_it; ++it) 
  {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< DataType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix A
    int block_index_system =
        row_index_block + offsets_block_one_[this->my_rank_];
    A->SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_A_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::ExtractB(const MatrixType &op, SubMatrixType* B) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  assert (this->num_var_two_ > 0);
  
  // sub matrix B
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_B_.begin(),
           e_it = couplings_B_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< DataType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix B
    int block_index_system =
        row_index_block + offsets_block_one_[this->my_rank_];
    B->SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_B_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::ExtractC(const MatrixType &op, SubMatrixType* C) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  assert (this->num_var_two_ > 0);
  
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_C_.begin(),
           e_it = couplings_C_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< DataType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix C
    int block_index_system =
        row_index_block + offsets_block_two_[this->my_rank_];
    C->SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_C_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::ExtractD(const MatrixType &op, SubMatrixType* D) const
{
  assert (this->initialized_);
  assert (this->num_var_two_ > 0);
  
  int row_index_block = 0;
  for (typename std::map< int, SortedArray< int > >::const_iterator
           it = couplings_D_.begin(),
           e_it = couplings_D_.end();
       it != e_it; ++it) {
    // Get values from system matrix op
    int row_index_system = it->first;
    std::vector< DataType > vals(it->second.size(), 0.);
    op.GetValues(&row_index_system, 1, vec2ptr(it->second.data()),
                 it->second.size(), vec2ptr(vals));

    // Set values in block matrix D
    int block_index_system =
        row_index_block + offsets_block_two_[this->my_rank_];
    D->SetValues(&block_index_system, 1,
                 vec2ptr(this->sparsity_D_[row_index_block]), it->second.size(),
                 vec2ptr(vals));
    ++row_index_block;
  }
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::ExtractVectorOne(const VectorType &b, SubVectorType* b1) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  
  b.GetValues(vec2ptr(this->mapb2s_one_), this->mapb2s_one_.size(),
              vec2ptr(this->val_temp_one_));
  b1->SetValues(vec2ptr(this->indexset_one_), this->indexset_one_.size(),
               vec2ptr(this->val_temp_one_));
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::ExtractVectorTwo(const VectorType &b, SubVectorType* b2) const
{
  assert (this->initialized_);
  assert (this->num_var_two_ > 0);
  
  b.GetValues(vec2ptr(this->mapb2s_two_), this->mapb2s_two_.size(),
              vec2ptr(this->val_temp_two_));
  b2->SetValues(vec2ptr(this->indexset_two_), this->indexset_two_.size(),
               vec2ptr(this->val_temp_two_));

}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::InsertVectorOne(const SubVectorType &b1, VectorType* b) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  
  b1.GetValues(vec2ptr(this->indexset_one_), this->indexset_one_.size(),
               vec2ptr(this->val_temp_one_));
               
  b->SetValues(vec2ptr(this->mapb2s_one_), this->mapb2s_one_.size(),
        vec2ptr(this->val_temp_one_));
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::InsertVectorTwo(const SubVectorType &b2, VectorType* b) const
{
  assert (this->initialized_);
  assert (this->num_var_two_ > 0);
  
  b2.GetValues(vec2ptr(this->indexset_two_), this->indexset_two_.size(),
               vec2ptr(this->val_temp_two_));
                        
  b->SetValues(vec2ptr(this->mapb2s_two_), this->mapb2s_two_.size(),
              vec2ptr(this->val_temp_two_));
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::AddVectorOne(const SubVectorType &b1, VectorType* b, DataType scale) const
{
  assert (this->initialized_);
  assert (this->num_var_one_ > 0);
  
  b1.GetValues(vec2ptr(this->indexset_one_), this->indexset_one_.size(),
               vec2ptr(this->val_temp_one_));
               
  for (int l=0; l<this->val_temp_one_.size(); ++l)
  {
    this->val_temp_one_[l] *= scale;
  }
                 
  b->Add(vec2ptr(this->mapb2s_one_), this->mapb2s_one_.size(),
        vec2ptr(this->val_temp_one_));
}

template < int DIM, class LAD, class SUBLAD >
void SubSystem< DIM, LAD, SUBLAD >::AddVectorTwo(const SubVectorType &b2, VectorType* b, DataType scale) const
{
  assert (this->initialized_);
  assert (this->num_var_two_ > 0);
  
  b2.GetValues(vec2ptr(this->indexset_two_), this->indexset_two_.size(),
               vec2ptr(this->val_temp_two_));
               
  for (int l=0; l<this->val_temp_two_.size(); ++l)
  {
    this->val_temp_two_[l] *= scale;
  }
                 
  b->Add(vec2ptr(this->mapb2s_two_), this->mapb2s_two_.size(),
        vec2ptr(this->val_temp_two_));
}

} // namespace la
} // namespace hiflow

#endif 
