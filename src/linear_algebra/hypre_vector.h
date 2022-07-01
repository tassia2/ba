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

#ifndef HIFLOW_LINEARALGEBRA_HYPRE_VECTOR_H_
#define HIFLOW_LINEARALGEBRA_HYPRE_VECTOR_H_

#include "common/log.h"
#include "config.h"
#include "linear_algebra/block_utilities.h"
#include "linear_algebra/coupled_vector.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/vector.h"
#include "linear_algebra/lmp/la_global.h"
#include "tools/mpi_tools.h"
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#ifdef WITH_HYPRE
extern "C" {
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_utilities.h"
}
#endif

namespace hiflow {
namespace la {

/// @author Simon Gawlok

/// @brief Wrapper to HYPRE vector

template < class DataType > class HypreVector : public Vector< DataType > {
public:
  /// Standard constructor
  HypreVector();
  /// Destructor
  ~HypreVector();

  virtual Vector< DataType > *Clone() const;

  /// Initialize matrix
  /// @param[in] comm MPI communicator to be used by vector
  //void Init(const MPI_Comm &comm);

  /// Initialize matrix
  /// @param[in] comm MPI communicator to be used by Vector
  /// @param[in] cp Linear algebra couplings describing ownerships
  void Init(const MPI_Comm &comm, const LaCouplings &cp);

  /// Additional init interface which is compatible to both block and non-block
  /// matrix
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, cp);
    this->initialized_ = true;
  }

  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            PLATFORM plat,
            IMPLEMENTATION impl,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, cp);
  }

  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            PLATFORM plat,
            IMPLEMENTATION impl)
  {
    this->Init(comm, cp);
  }
    
  /// Clear all allocated data
  void Clear();

  /// Clone vector structure without content
  void CloneFromWithoutContent(const HypreVector< DataType > &vec);

  /// Copy vector content to this Vector

  void CopyFrom(const HypreVector< DataType > &vec) {
#ifdef WITH_HYPRE
    if (this != &vec) {
      const size_t local_size = this->size_local();
      std::vector< DataType > data(local_size, 0.);

      vec.GetValues(vec2ptr(this->global_indices_), local_size, vec2ptr(data));
      this->SetValues(vec2ptr(this->global_indices_), local_size,
                      vec2ptr(data));

      data.clear();

      this->Update();
    }
#else
    LOG_ERROR("HiFlow was not compiled with HYPRE support!");
    quit_program();
#endif
  }

  void CopyFromWithoutGhost(const HypreVector< DataType > &vec) {
#ifdef WITH_HYPRE
    if (this != &vec) {
      const size_t local_size = this->size_local();
      std::vector< DataType > data(local_size, 0.);

      vec.GetValues(vec2ptr(this->global_indices_), local_size, vec2ptr(data));
      this->SetValues(vec2ptr(this->global_indices_), local_size,
                      vec2ptr(data));

      data.clear();
    }
#else
    LOG_ERROR("HiFlow was not compiled with HYPRE support!");
    quit_program();
#endif
  }

  void CloneFrom(const HypreVector< DataType > &vec) {
    this->CloneFromWithoutContent(vec);
    this->CopyFrom(vec);
  }

  /// Get laCouplings

  const LaCouplings *la_couplings() const { return this->cp_; }

  /// Get MPI communicator

  const MPI_Comm &comm() const { return comm_; }

  int my_rank() const { return my_rank_; }

  /// @return Number of processes

  int nb_procs() const { return this->nb_procs_; }

  /// Print statistical data
  void print_statistics() const;

  /// Local size of Vector
  int size_local() const;
  /// Global size of Vector
  int size_global() const;

  /// Update operator, i.e. exchange values for distributed vectors
  void Update();
  /// Initiate update

  void begin_update() {}
  /// Finalize update

  void end_update() {}

  /// Send border values
  void SendBorder();
  /// Receive ghost values
  void ReceiveGhost();
  /// Wait for finishing of send operations
  void WaitForSend();
  /// Wait for finishing of receive operations
  void WaitForRecv();
  /// Update ghost values
  void UpdateGhost();
  /// Set ghost values
  void SetGhostValues(const DataType *values);
  /// Get all dofs and values (interior, ghost and pp data)
  void GetAllDofsAndValues(std::vector< int > &id,
                           std::vector< DataType > &val) const;

  /// @return Local size of ghost

  int size_local_ghost() const {
    assert(this->ghost_ != nullptr);
    return this->ghost_->get_size();
  }

  /// Set Vector to zero
  void Zeros();
  /// Get value at a known index
  DataType GetValue(const int index) const;
  /// Get values at given indices
  void GetValues(const int *indices, const int size_indices,
                 DataType *values) const;

  /// Retrieves the whole vector (interior).
  /// @param values Array of values, must fit size of interior
  void GetLocalValues(DataType *values) const;

  void GetLocalValues(const int *indices, const int size_indices,
                 DataType *values) const;

  /// Euclidean length of vector
  DataType Norm2() const;
  /// l1 norm of vector
  DataType Norm1() const;
  /// maximum absolute value of entries
  DataType NormMax() const;

  /// Scalar product
  DataType Dot(const Vector< DataType > &vec) const;
  /// Scalar product
  DataType Dot(const HypreVector< DataType > &vec) const;

  /// Add value to a given index
  void Add(int index, DataType scalar);
  /// Add values to given indices
  void Add(const int *indices, int length, const DataType *values);

  /// Sets every element to given value.
  /// @param val some value

  void SetToValue(DataType val);

  /// Set given global index to value
  void SetValue(const int index, const DataType value);
  /// Set given global indices to given values
  void SetValues(const int *indices, const int size_indices,
                 const DataType *values);

  /// Sets all the values in the vector (interior).
  /// @param values Array of values, must fit size of interior
  void SetLocalValues(const DataType *values);

  /// this <- this + alpha * vecx
  void Axpy(const Vector< DataType > &vecx, const DataType alpha);
  /// this <- this + alpha * vecx
  void Axpy(const HypreVector< DataType > &vecx, const DataType alpha);

  /// this <- alpha * this + vecx
  void ScaleAdd(const Vector< DataType > &vecx, const DataType alpha);

  /// this <- alpha * this
  void Scale(const DataType alpha);

#ifdef WITH_HYPRE
  /// Get pointer to HYPRE_ParVector objects
  HYPRE_ParVector *GetParVector();

  /// Get pointer to HYPRE_ParVector objects
  const HYPRE_ParVector *GetParVector() const;
#endif

  // auxiliary structure, needed in FE interpolation
  const lVector< DataType > &interior() const 
  { 
    this->update_interior();
    return *(this->interior_); 
  }

  lVector< DataType > &interior() 
  { 
    this->update_interior();
    return *(this->interior_); 
  }
  
  // need to call this after interior() was modified from outside
  void store_interior()
  {
    assert (this->interior_->get_size() == this->size_local());
    this->SetLocalValues(&(this->interior_->GetBuffer()[0]));
  }
  
  const lVector< DataType > &ghost() const 
  { 
    return *(this->ghost_); 
  }

  lVector< DataType > &ghost() 
  { 
    return *(this->ghost_); 
  }

  /// Write vector content to HDF5 file
  void WriteHDF5(const std::string &filename, const std::string &groupname,
                 const std::string &datasetname);

  /// Read vector content from HDF5 file
  void ReadHDF5(const std::string &filename, const std::string &groupname,
                const std::string &datasetname);

  /// Return global index of the first local entry

  int ownership_begin() const { return static_cast< int >(this->ilower_); }

  /// Retrun global index of the last local entry

  int ownership_end() const { return static_cast< int >(this->iupper_); }

  bool is_initialized() const { return this->initialized_; }
 
  IMPLEMENTATION get_impl() const {
    return HYPRE;
  }
  
private:
  void update_interior() const;
  
  /// MPI communicator
  MPI_Comm comm_;
  /// Global number of first component owned by this process
  HYPRE_Int ilower_;
  /// Global number of last component owned by this process
  HYPRE_Int iupper_;
  /// Vector of local indices in global numbering
  std::vector< int > global_indices_;

  /// Linear algebra couplings describing global dof distribution
  const LaCouplings *cp_;

  lVector< DataType > *ghost_;
  mutable lVector< DataType > *interior_;

  /// Number of processes.
  int nb_procs_;
  /// Rank of this process.
  int my_rank_;
  /// Number of send operations for border values
  int nb_sends_;
  /// Number of receive operations for ghost values
  int nb_recvs_;
  /// MPI requests
  std::vector< MPI_Request > mpi_req_;
  /// MPI status
  std::vector< MPI_Status > mpi_stat_;
  /// border values to be send
  std::vector< DataType > border_val_;
  /// ghost values to be received
  std::vector< DataType > ghost_val_;
  std::vector< int > border_indices_;

  /// Flag if vector is already initialized
  bool initialized_;

#ifdef WITH_HYPRE
  HYPRE_IJVector x_;
  HYPRE_ParVector parcsr_x_;
#endif
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARALGEBRA_HYPRE_VECTOR_H_
