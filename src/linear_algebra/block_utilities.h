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

#ifndef HIFLOW_LINEARALGEBRA_BLOCK_UTILITIES_H_
#define HIFLOW_LINEARALGEBRA_BLOCK_UTILITIES_H_

#include <cassert>
#include <cmath>
#include <vector>
// Check if C++11 or newer is supported
#if __cplusplus > 199711L
#include <unordered_map>
#define UMAP_NAMESPACE std
#else
#include <boost/unordered_map.hpp>
#define UMAP_NAMESPACE boost::unordered
#endif
#include "common/log.h"
#include "common/pointers.h"
#include "common/sorted_array.h"
#include "config.h"
#include "linear_algebra/la_couplings.h"
#include <mpi.h>
#include <utility>

namespace hiflow {
namespace la {
/// @author Simon Gawlok

class BlockManager;
using BlockManagerSPtr = SharedPtr< BlockManager >;
using CBlockManagerSPtr = SharedPtr< const BlockManager >;
  
class BlockManager {
public:
  /// Standard constructor

  BlockManager() {
    this->num_blocks_ = -1;
    this->la_c_blocks_.clear();
    this->mapb2s_.clear();
    this->maps2b_.clear();
    this->initialized_ = false;
  }

  /// Standard destructor

  ~BlockManager() {
    this->num_blocks_ = 0;
    this->la_c_blocks_.clear();
    this->mapb2s_.clear();
    this->maps2b_.clear();
  }

  /// Initialize BlockManager

  void Init(const MPI_Comm &comm, const LaCouplings &couplings,
            const std::vector< std::vector< int > > &block_dofs) {

    // TODO: Add validity checks of input data

    // Get number of processes as well as my rank
    int nb_procs = -1;
    int my_rank = -1;

    MPI_Comm_size(comm, &nb_procs);
    assert(nb_procs > 0);

    MPI_Comm_rank(comm, &my_rank);
    assert(my_rank >= 0);
    assert(my_rank < nb_procs);

    // Number of requested blocks
    this->num_blocks_ = block_dofs.size();
    assert(num_blocks_ > 0);

#ifndef NDEBUG
    LOG_DEBUG(2, "Number of blocks on process " << my_rank << ": "
                                                << this->num_blocks_);
    for (size_t i = 0; i < this->num_blocks_; ++i) {
      LOG_DEBUG(2, "\tNumber of DoFs in block " << i << " on process "
                                                << my_rank << ": "
                                                << block_dofs[i].size());
      LOG_DEBUG(2,
                "\tDoFs in block " << i << " on process [" << my_rank << "]:");
      for (size_t j = 0; j < block_dofs[i].size(); ++j) {
        LOG_DEBUG(2, "\t\t[" << my_rank << "] " << block_dofs[i][j]);
      }
    }
#endif

    // Total number of DoFs in blocks
    int num_dofs = 0;
    for (size_t i = 0; i < this->num_blocks_; ++i) {
      num_dofs += block_dofs[i].size();
    }

    //*****************************************************************
    // Clear possibly old data
    //*****************************************************************
    for (size_t i = 0; i < this->la_c_blocks_.size(); ++i) {
      delete this->la_c_blocks_[i];
    }
    this->la_c_blocks_.clear();

    this->mapb2s_.clear();
    this->maps2b_.clear();

    //*****************************************************************
    // Step 1: Sort block_dofs and distinguish between diagonal and
    // offdiagonal (i.e., ghost) dofs
    //*****************************************************************
    std::vector< SortedArray< int > > block_dofs_diag_sorted(this->num_blocks_);
    std::vector< SortedArray< int > > block_dofs_offdiag_sorted(
        this->num_blocks_);

    for (size_t i = 0; i < this->num_blocks_; ++i) {
      block_dofs_diag_sorted[i].reserve(block_dofs[i].size());
      block_dofs_offdiag_sorted[i].reserve(block_dofs[i].size());
      for (size_t j = 0; j < block_dofs[i].size(); ++j) {

        if ((block_dofs[i][j] >= couplings.global_offsets()[my_rank]) &&
            (block_dofs[i][j] < couplings.global_offsets()[my_rank + 1])) {
          block_dofs_diag_sorted[i].find_insert(block_dofs[i][j]);
        } else {
          block_dofs_offdiag_sorted[i].find_insert(block_dofs[i][j]);
        }
      }
    }

    //*****************************************************************
    // Step 2: Compute global offsets of the individual blocks by means
    // of the the formerly created block_dofs_diag_sorted data structure.
    // These offsets are needed for the initialization of the LaCouplings
    // of the individual blocks
    //*****************************************************************

    std::vector< std::vector< int > > global_offsets(
        this->num_blocks_, std::vector< int >(nb_procs + 1, -1));

    {
      // Pack number of local DoFs for all blocks in one data structure
      // in order to do it all in just one communication
      std::vector< int > num_local_dofs(this->num_blocks_, -1);

      for (size_t i = 0; i < this->num_blocks_; ++i) {
        num_local_dofs[i] = block_dofs_diag_sorted[i].size();
        assert(num_local_dofs[i] > -1);
      }

      std::vector< int > local_dof_numbers_temp(this->num_blocks_ * nb_procs,
                                                -1);

      MPI_Allgather(vec2ptr(num_local_dofs), this->num_blocks_, MPI_INT,
                    vec2ptr(local_dof_numbers_temp), this->num_blocks_, MPI_INT,
                    comm);

      // Unpack received data
      for (size_t i = 0; i < this->num_blocks_; ++i) {
        global_offsets[i][0] = 0;
        assert(global_offsets[i][0] > -1);
        for (int j = 0; j < nb_procs; ++j) {
          assert(j * this->num_blocks_ + i < this->num_blocks_ * nb_procs);
          global_offsets[i][j + 1] =
              global_offsets[i][j] +
              local_dof_numbers_temp[j * this->num_blocks_ + i];
          assert(global_offsets[i][j + 1] > -1);
        }
      }
    }

    //*****************************************************************
    // Step 3: Compute mappings: system->block and block->system for
    // diagonal part
    //*****************************************************************

    this->mapb2s_.resize(this->num_blocks_);
    for (size_t i = 0; i < this->num_blocks_; ++i) {
      this->mapb2s_[i].reserve(block_dofs_diag_sorted[i].size() +
                               block_dofs_offdiag_sorted[i].size());
    }

    {
      for (size_t i = 0; i < this->num_blocks_; ++i) {
        int index = 0;
        for (size_t k = 0; k < block_dofs_diag_sorted[i].size(); ++k) {
          this->maps2b_[block_dofs_diag_sorted[i][k]] =
              std::make_pair(i, index + global_offsets[i][my_rank]);
          this->mapb2s_[i][index + global_offsets[i][my_rank]] =
              block_dofs_diag_sorted[i][k];
          ++index;
        }
      }
    }

    //*****************************************************************
    // Step 4: Compute mappings: system->block and block->system for
    // ghost part and initialize LaCouplings of individual blocks
    //*****************************************************************

    this->la_c_blocks_.resize(this->num_blocks_);

    for (size_t i = 0; i < this->num_blocks_; ++i) {
      // Data structures for initializing LaCouplings for the individual
      // blocks
      std::vector< int > ghost_dofs;
      std::vector< int > ghost_offsets(nb_procs + 1, -1);

      // 1. Determine owners of ghost dofs
      std::vector< std::vector< int > > ghost_dofs_by_owner(nb_procs);
      for (int j = 0; j < nb_procs; ++j) {
        ghost_dofs_by_owner[j].reserve(block_dofs_offdiag_sorted[i].size());
      }

      for (size_t k = 0; k < block_dofs_offdiag_sorted[i].size(); ++k) {
        for (int j = 0; j < nb_procs; ++j) {
          if ((block_dofs_offdiag_sorted[i][k] >=
               couplings.global_offsets()[j]) &&
              (block_dofs_offdiag_sorted[i][k] <
               couplings.global_offsets()[j + 1])) {
            ghost_dofs_by_owner[j].push_back(block_dofs_offdiag_sorted[i][k]);
          }
        }
      }

      // 2. Fill ghost_offsets
      ghost_offsets[0] = 0;
      assert(ghost_offsets[0] > -1);
      for (int j = 0; j < nb_procs; ++j) {
        ghost_offsets[j + 1] = ghost_offsets[j] + ghost_dofs_by_owner[j].size();
        assert(ghost_offsets[j + 1] > -1);
      }

      // numbers of DoFs which I request of other processes
      std::vector< int > num_dofs_requested(nb_procs, 0);
      int total_number_dofs_requested = 0;
      for (int j = 0; j < nb_procs; ++j) {
        num_dofs_requested[j] = ghost_dofs_by_owner[j].size();
        total_number_dofs_requested += ghost_dofs_by_owner[j].size();
      }
      assert(total_number_dofs_requested == ghost_offsets.back());

      // numbers of DoFs which other processes request from me
      std::vector< int > num_dofs_requested_others(nb_procs, 0);

      // Exchange requested numbers of DoFs
      MPI_Alltoall(vec2ptr(num_dofs_requested), 1, MPI_INT,
                   vec2ptr(num_dofs_requested_others), 1, MPI_INT, comm);

      // DoF numbers which I need from others
      std::vector< int > dofs_requested;
      dofs_requested.reserve(total_number_dofs_requested);

      for (int j = 0; j < nb_procs; ++j) {
        dofs_requested.insert(dofs_requested.end(),
                              ghost_dofs_by_owner[j].begin(),
                              ghost_dofs_by_owner[j].end());
      }
      assert(dofs_requested.size() == total_number_dofs_requested);

      // DoF numbers which others request from me
      std::vector< int > dofs_requested_others;
      int total_number_dofs_requested_others = 0;

      std::vector< int > offsets_dofs_requested_others(nb_procs + 1, 0);
      for (int j = 0; j < nb_procs; ++j) {
        offsets_dofs_requested_others[j + 1] =
            offsets_dofs_requested_others[j] + num_dofs_requested_others[j];
        total_number_dofs_requested_others += num_dofs_requested_others[j];
      }
      dofs_requested_others.resize(total_number_dofs_requested_others, 0);

      // Exchange requested DoF IDs
      MPI_Alltoallv(vec2ptr(dofs_requested), vec2ptr(num_dofs_requested),
                    vec2ptr(ghost_offsets), MPI_INT,
                    vec2ptr(dofs_requested_others),
                    vec2ptr(num_dofs_requested_others),
                    vec2ptr(offsets_dofs_requested_others), MPI_INT, comm);

      // Prepare DoF IDs which others need from me
      for (int k = 0; k < total_number_dofs_requested_others; ++k) {
        assert(this->maps2b_.find(dofs_requested_others[k]) !=
               this->maps2b_.end());
        UMAP_NAMESPACE::unordered_map<
            int, std::pair< int, int > >::const_iterator entity =
            this->maps2b_.find(dofs_requested_others[k]);

        assert(entity->second.first == i);
        dofs_requested_others[k] = entity->second.second;
      }

      std::vector< int > dofs_requested_system(dofs_requested);

      // Exchange mapped DoF IDs
      MPI_Alltoallv(vec2ptr(dofs_requested_others),
                    vec2ptr(num_dofs_requested_others),
                    vec2ptr(offsets_dofs_requested_others), MPI_INT,
                    vec2ptr(dofs_requested), vec2ptr(num_dofs_requested),
                    vec2ptr(ghost_offsets), MPI_INT, comm);

      // Unpack received data
      ghost_dofs.reserve(total_number_dofs_requested);
      for (int j = 0; j < nb_procs; ++j) {
        for (int k = 0; k < num_dofs_requested[j]; ++k) {
          ghost_dofs.push_back(dofs_requested[ghost_offsets[j] + k]);
        }
      }

      // Initialize LaCouplings for current block
      this->la_c_blocks_[i] = new LaCouplings();
      assert(this->la_c_blocks_[i] != nullptr);
      this->la_c_blocks_[i]->Init(comm);
      this->la_c_blocks_[i]->InitializeCouplings(global_offsets[i], ghost_dofs,
                                                 ghost_offsets);

      // Add ghost dofs to mappings system-> block and block->system
      for (size_t j = 0; j < dofs_requested_system.size(); ++j) {
        this->maps2b_[dofs_requested_system[j]] =
            std::make_pair(i, dofs_requested[j]);
        this->mapb2s_[i][dofs_requested[j]] = dofs_requested_system[j];
      }
    }
    this->initialized_ = true;
  }

  /// Map global system index to block number and respective block index

  inline void map_system2block(const int system_index, int &block_num,
                               int &block_index) const {
    assert(system_index >= 0);
    UMAP_NAMESPACE::unordered_map< int, std::pair< int, int > >::const_iterator
        block = this->maps2b_.find(system_index);
    assert(block != this->maps2b_.end());
    assert(this->initialized_);
    block_num = block->second.first;
    block_index = block->second.second;
    assert(block_num >= 0);
    assert(block_num < this->num_blocks_);
    assert(block_index >= 0);
  }

  /// Map block index to system index

  inline void map_block2system(const int block_num, const int block_index,
                               int &system_index) const {

    assert(block_num >= 0);
    assert(block_num < this->num_blocks_);
    assert(block_index >= 0);
    assert(this->initialized_);
    UMAP_NAMESPACE::unordered_map< int, int >::const_iterator system_index_map =
        this->mapb2s_[block_num].find(block_index);
    assert(system_index_map != this->mapb2s_[block_num].end());
    system_index = system_index_map->second;
    assert(system_index >= 0);
  }

  const std::vector< LaCouplings * > &la_c_blocks() const {
    return this->la_c_blocks_;
  }

  size_t num_blocks() const { return this->num_blocks_; }

  BlockManager &operator=(const BlockManager &bm) {

    if (this == &bm) {
      return *this;
    }
    if (!bm.initialized_) {
      return *this;
    }
    
    this->num_blocks_ = bm.num_blocks();

    this->mapb2s_ = bm.mapb2s_;

    this->maps2b_ = bm.maps2b_;

    this->la_c_blocks_.clear();
    this->la_c_blocks_.resize(bm.la_c_blocks_.size());
    for (size_t l = 0; l < bm.la_c_blocks_.size(); ++l) {
        this->la_c_blocks_[l] = new LaCouplings(*(bm.la_c_blocks_[l]));
    }
    return *this;
  }

protected:
  size_t num_blocks_;
  std::vector< LaCouplings * > la_c_blocks_;
  std::vector< UMAP_NAMESPACE::unordered_map< int, int > > mapb2s_;
  UMAP_NAMESPACE::unordered_map< int, std::pair< int, int > > maps2b_;
  bool initialized_;
};
} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARALGEBRA_BLOCK_UTILITIES_H_
