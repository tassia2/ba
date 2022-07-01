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

/// @author Martin Wlotzka

#ifndef HIFLOW_LINEARALGEBRA_LA_COUPLINGS_H_
#define HIFLOW_LINEARALGEBRA_LA_COUPLINGS_H_

#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#include "mpi.h"

namespace hiflow {

namespace doffem {
template < class DataType, int DIM > class DofPartitionGlobal;
}

namespace la {

struct SparsityStructure {
  std::vector< int > diagonal_rows;
  std::vector< int > diagonal_cols;
  std::vector< int > off_diagonal_rows;
  std::vector< int > off_diagonal_cols;
  std::vector< int > col_off_diagonal_rows;
  std::vector< int > col_off_diagonal_cols;
#if 0
  std::vector< int > colrow_off_diagonal_rows;
  std::vector< int > colrow_off_diagonal_cols;
#endif
};

/// @brief Contains information for CoupledMatrix/CoupledVector, independent of
/// other HiFlow objects.
///
/// Contains border, ghost data and information about matrix structure.

class LaCouplings {
public:
  /// Standard constructor, sets initialization status to @c false. @c
  /// LaCouplings must be initialized with @c LaCouplings::InitializeCouplings
  /// before use.
  LaCouplings();
  /// Destructor
  virtual ~LaCouplings();

  /// Sets MPI communicator
  /// @param comm MPI communicator
  void Init(const MPI_Comm &comm);

  /// Initializes LaCouplings, i.e. computes needed ghost and border indices.
  /// The global order of variables must be sliced by owner. Sets the
  /// initialization status to
  /// @c true if succesful.
  /// @param global_offsets Offsets specifying the owner process rank of the
  /// variables
  /// @param ghost_dofs Global (DoF) indices of the ghost entries
  /// (must be sliced by owner; may contain duplicates)
  /// @param ghost_offsets Offsets specifying the owner process rank of the
  /// ghost variables
  void InitializeCouplings(const std::vector< int > &global_offsets,
                           const std::vector< int > &ghost_dofs,
                           const std::vector< int > &ghost_offsets);

  /// Clears border + ghost information, the global-offdiag mappings
  /// and sets initialization status to @c false.
  void Clear();

  /// @return Global 2 offidagonal mapping

  const std::map< int, int > &global2offdiag() const {
    return this->global2offdiag_;
  }

  /// @return Border indices vector

  const std::vector< int > &border_indices_vec() const {
    return this->border_indices_;
  }

  /// @return Border offsets vector

  const std::vector< int > &border_offsets_vec() const {
    return this->border_offsets_;
  }

  /// @return Ghost offsets vector

  const std::vector< int > &ghost_offsets_vec() const {
    return this->ghost_offsets_;
  }

  /// @return Global offsets vector

  const std::vector< int > &global_offsets() const {
    return this->global_offsets_;
  }

  /// @return Offdiagonal 2 global mapping

  const std::map< int, int > &offdiag2global() const {
    return this->offdiag2global_;
  }

  /// Returns global dof offset for a given process
  /// @param rk Process rank
  /// @return Dof offset for process with rank @em rk

  int dof_offset(const int rk) const {
    assert(rk >= 0);
    assert(rk < this->global_offsets_.size());
    return this->global_offsets_[rk];
  }

  /// @return Number of dofs owned by process with rank rk

  int nb_dofs(const int rk) const {
    assert(rk >= 0);
    assert(rk + 1 < this->global_offsets_.size());
    return this->global_offsets_[rk + 1] - this->global_offsets_[rk];
  }

  /// @return Total number of dofs

  int nb_total_dofs() const { return this->global_offsets_.back(); }

  /// @return Border indices

  const int *border_indices() const { return &(this->border_indices_[0]); }

  /// @return Border index at @em ind

  int border_indices(int ind) const {
    assert(ind >= 0);
    assert(ind < this->border_indices_.size());
    return this->border_indices_[ind];
  }

  /// @return Size of border indices

  int size_border_indices() const { return this->border_indices_.size(); }

  /// @return Border offsets

  const int *border_offsets() const { return &(this->border_offsets_[0]); }

  /// @return Border offset at @em ind

  int border_offsets(int ind) const {
    assert(ind >= 0);
    assert(ind < this->border_offsets_.size());
    return this->border_offsets_[ind];
  }

  /// @return Ghost offsets

  const int *ghost_offsets() const { return &(this->ghost_offsets_[0]); }

  /// @return Ghost offset at @em ind

  int ghost_offsets(int ind) const {
    assert(ind >= 0);
    assert(ind < this->ghost_offsets_.size());
    return this->ghost_offsets_[ind];
  }

  /// @return Size of ghost vector

  int size_ghost() const { return this->ghost_offsets_.back(); }

  /// @return Initialization status

  bool initialized() const { return this->initialized_; }

  /// @return MPI communicator

  const MPI_Comm &comm() const { return this->comm_; }

  /// @return Offdiagonal column index
  inline int Global2Offdiag(int global_col) const;
  /// @return Global column index
  inline int Offdiag2Global(int offdiag_col) const;

  LaCouplings &operator=(const LaCouplings &cp) {
    if (this == &cp) {
      return *this;
    }

    // Clone MPI communicator
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&(this->comm_));
    }

    MPI_Comm_dup(cp.comm(), &(this->comm_));
    assert(this->comm_ != MPI_COMM_NULL);

    // Get my rank
    MPI_Comm_rank(this->comm_, &(this->my_rank_));

    // Copy map global2offdiag
    this->global2offdiag_ = cp.global2offdiag();

    // Copy border_indices
    this->border_indices_ = cp.border_indices_vec();

    // Copy border offsets
    this->border_offsets_ = cp.border_offsets_vec();

    // Copy ghost offsets
    this->ghost_offsets_ = cp.ghost_offsets_vec();

    // Copy global offsets
    this->global_offsets_ = cp.global_offsets();

    // Copy offdiag2global
    this->offdiag2global_ = cp.offdiag2global();

    // Set initialization flag
    this->initialized_ = cp.initialized();

    return *this;
  }

  // no implementation of copy constructor or assignement operator

  LaCouplings(const LaCouplings &cp) {
    // Clone MPI communicator
    /*
    if (this->comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&(this->comm_));
    }
    */
    
    MPI_Comm_dup(cp.comm(), &(this->comm_));
    assert(this->comm_ != MPI_COMM_NULL);

    // Get my rank
    MPI_Comm_rank(this->comm_, &(this->my_rank_));

    // Copy map global2offdiag
    this->global2offdiag_ = cp.global2offdiag();

    // Copy border_indices
    this->border_indices_ = cp.border_indices_vec();

    // Copy border offsets
    this->border_offsets_ = cp.border_offsets_vec();

    // Copy ghost offsets
    this->ghost_offsets_ = cp.ghost_offsets_vec();

    // Copy global offsets
    this->global_offsets_ = cp.global_offsets();

    // Copy offdiag2global
    this->offdiag2global_ = cp.offdiag2global();

    // Set initialization flag
    this->initialized_ = cp.initialized();
  }

private:
  friend std::ostream &operator<<(std::ostream &os,
                                  LaCouplings const &laCouplings);

protected:
  /// Compresses offdiagonal block in order to have size_ghost_ columns,
  /// i.e. delete zero columns and shift local column indices, so that
  /// offdiagonal can be represented in one block. This mapping is stored in @em
  /// global2offdiag
  /// @param ghost_dofs global column indices to be shifted
  void CompressOffdiagonal(const std::vector< int > &ghost_dofs);

  // Map from 'global index' to 'offdiagonal column index' for offdiagonal block
  // since we compress it. This map sorts by global index.
  std::map< int, int > global2offdiag_;
  std::map< int, int > offdiag2global_;

  std::vector< int > global_offsets_;

  // local indices of border sliced by process ids
  std::vector< int > border_indices_;
  std::vector< int > border_offsets_;

  // offset in ghost to determine process
  std::vector< int > ghost_offsets_;

  bool initialized_;

  MPI_Comm comm_;
  int my_rank_;
};

inline int LaCouplings::Global2Offdiag(int global_col) const {
  assert(global2offdiag_.find(global_col) != global2offdiag_.end());
  return global2offdiag_.find(global_col)->second;
}

inline int LaCouplings::Offdiag2Global(int offdiag_col) const {
  assert(offdiag2global_.find(offdiag_col) != offdiag2global_.end());
  return offdiag2global_.find(offdiag_col)->second;
}

 //std::ostream &operator<<(std::ostream &os, LaCouplings const &laCouplings);

} // namespace la
} // namespace hiflow

#endif
