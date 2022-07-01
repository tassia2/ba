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

#ifndef HIFLOW_LINEARALGEBRA_BLOCK_VECTOR_H_
#define HIFLOW_LINEARALGEBRA_BLOCK_VECTOR_H_

//#    include <cstdlib>
//#    include <iostream>
#include "config.h"
#include "linear_algebra/block_utilities.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/vector.h"
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <vector>
//#    include "common/log.h"
//#    include "tools/mpi_tools.h"
namespace hiflow {
namespace la {

/// @author Jonas Kratzke, Simon Gawlok

/// @brief Block vector

template < class LAD >
class BlockVector : public Vector< typename LAD::DataType > {
public:
  typedef typename LAD::VectorType BVector;
  typedef typename LAD::DataType BDataType;

  /// Standard constructor
  BlockVector();
  /// Destructor
  ~BlockVector();

  /// Clear all allocated data
  void Clear();

  virtual BlockVector< LAD > *Clone() const;

  void CloneFromWithoutContent(const BlockVector< LAD > &vec);

  void CloneFrom(const BlockVector< LAD > &vec) {
    this->CloneFromWithoutContent(vec);
    this->CopyFrom(vec);
  }

  void CopyFrom(const BlockVector< LAD > &vec) {
    assert(this->initialized_);
    assert(this->num_blocks() == vec.num_blocks());

    if (this != &vec) {
      for (int i = 0; i < this->num_blocks_; ++i) {
        this->vec_[i]->CopyFrom(vec.GetBlock(i));
      }
    }
  }

  const MPI_Comm &comm() const { return this->comm_; }

  const std::vector< bool > &get_active_blocks() const {
    return this->active_blocks_;
  }

  void set_all_blocks_active()
  {
    this->active_blocks_.clear();
    this->active_blocks_.resize(this->num_blocks_, true);
  }
  
  void set_active_blocks(const std::vector< bool > &active_blocks) {
    assert (active_blocks.size() == this->num_blocks_);
    assert (this->initialized_);
    this->active_blocks_ = active_blocks;
  }

  bool block_is_active (int block_nr) const {
    assert (block_nr >= 0);
    assert (block_nr < this->active_blocks_.size());
    return this->active_blocks_[block_nr];
  }
  
  /// Initialize vector
  /// @param[in] comm MPI communicator to be used by vector
  /// @param[in] cp LaCouplings describing the global DoF distribution
  /// in global system numbering
  /// @param[in] block_dofs Vector of vectors. Size of vector determines
  /// number of blocks. The individual vectors describe the DoFs in global
  /// system numbering (including ghost DoFs!!!) to be contained in the
  /// respective blocks
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, cp, CPU, NAIVE, block_manager);
  }
  
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            PLATFORM plat,
            IMPLEMENTATION impl,
            CBlockManagerSPtr block_manager);
              
  /// Const access to blocks
  const BVector &GetBlock(size_t block_number) const;

  /// Non-Const access to blocks
  BVector &GetBlock(size_t block_number);

  /// Local size of Vector
  int size_local() const;
  /// Global size of Vector
  int size_global() const;

  int size_local_ghost() const {
    NOT_YET_IMPLEMENTED;
    return -1;
  }
  
  int my_rank() const {
    int my_rank;
    MPI_Comm_rank(this->comm_, &my_rank);
    return my_rank;
  }

  /// Number of blocks in Vector

  size_t num_blocks() const {
    assert(this->initialized_);
    return this->num_blocks_;
  }

  /// Update operator, i.e. exchange values for distributed vectors
  void Update();
  /// Initiate update
  void begin_update();
  /// Finalize update
  void end_update();

  /// Set Vector to zero
  void Zeros();
  /// Get value at a known index
  BDataType GetValue(int index) const;
  /// Get values at given indices
  void GetValues(const int *indices, int size_indices, BDataType *values) const;
  
  /// Retrieves the whole vector (interior).
  /// @param values Array of values, must fit size of interior
  void GetLocalValues(BDataType *values) const;
  
  /// @return All global Dofs and values that are in interior and ghost.
  /// They are NOT sorted.
  void GetAllDofsAndValues(std::vector< int > &id,
                           std::vector< BDataType > &val) const;

  /// Sets every element to given value.
  /// @param val some value

  void SetToValue(BDataType val) {
    for (int i = 0; i < this->num_blocks_; ++i) {
      this->vec_[i]->SetToValue(val);
    }
  }

  /// Euclidean length of vector
  BDataType Norm2() const;
  /// l1 norm of vector
  BDataType Norm1() const;
  /// maximum absolute value of entries
  BDataType NormMax() const;

  /// Scalar product
  BDataType Dot(const Vector< BDataType > &vec) const;
  /// Scalar product
  BDataType Dot(const BlockVector< LAD > &vec) const;

  /// Add value to a given index
  void Add(int index, BDataType scalar);
  /// Add values to given indices
  void Add(const int *indices, int length, const BDataType *values);
  /// Set given global index to value
  void SetValue(int index, BDataType value);
  /// Set given global indices to given values
  void SetValues(const int *indices, int size_indices, const BDataType *values);
  /// Set given global indices to given values
  void SetLocalValues(const BDataType *values);
  
  /// this <- this + alpha * vecx
  void Axpy(const Vector< BDataType > &vecx, BDataType alpha);
  /// this <- this + alpha * vecx
  void Axpy(const BlockVector< LAD > &vecx, BDataType alpha);

  /// this <- alpha * this + vecx
  void ScaleAdd(const Vector< BDataType > &vecx, BDataType alpha);
  /// this <- alpha * this + vecx
  void ScaleAdd(const BlockVector< LAD > &vecx, BDataType alpha);

  /// this <- alpha * this
  void Scale(BDataType alpha);

  /// Print statistical data
  void print_statistics() const;

  const std::vector< int > &global_indices() const {
    return this->global_indices_;
  }

  /// Write vector content to HDF5 file
  void WriteHDF5(const std::string &filename, const std::string &groupname,
                 const std::string &datasetname);

  /// Read vector content from HDF5 file
  void ReadHDF5(const std::string &filename, const std::string &groupname,
                const std::string &datasetname);

  CBlockManagerSPtr block_manager() const { return this->block_manager_; }

  bool is_initialized() const {
    for (int i = 0; i < this->num_blocks_; ++i) {
      if (!this->vec_[i]->is_initialized()) {
        return false;
      }
    }
    return this->initialized_;
  }

private:
  /// MPI communicator
  MPI_Comm comm_;

  bool initialized_;

  size_t num_blocks_;

  /// Vector describing the active blocks, i.e., inactive blocks are
  /// ignored in all vector-operations
  std::vector< bool > active_blocks_;

  /// Vector of local indices in global numbering
  std::vector< int > global_indices_;

  CBlockManagerSPtr block_manager_;

  // Vectors of individual blocks
  std::vector< BVector * > vec_;
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARALGEBRA_BLOCK_VECTOR_H_
