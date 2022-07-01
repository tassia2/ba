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
/// Wlotzka

#ifndef HIFLOW_LINEARALGEBRA_COUPLED_VECTOR_H_
#define HIFLOW_LINEARALGEBRA_COUPLED_VECTOR_H_

#include "common/parcom.h"
#include "common/property_tree.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/lmp/cuda/lvector_gpu.h"
#include "linear_algebra/lmp/la_global.h"
#include "linear_algebra/lmp/init_vec_mat.h"
#include "linear_algebra/lmp/platform_management.h"
#include "linear_algebra/vector.h"
#include <iostream>
#include <memory>
#include "mpi.h"

namespace hiflow {
namespace la {

template < class DataType > class lVector;
class BlockManager;
using CBlockManagerSPtr = std::shared_ptr< const BlockManager >;

/// @brief Distributed vector.
///
/// This vector stores also couplings. After matrix-vector multiplication
/// these couplings have to be updated.

template < class DataType > class CoupledVector : public Vector< DataType > {
public:
  /// Standard constructor
  CoupledVector();

  /// Destructor
  virtual ~CoupledVector();

  /// Inits empty vector.
  /// @param comm MPI communicator
  /// @param cp LaCouplings (see la_couplings.h)
  /// @param plat System platform (see lmp/la_global.h)
  /// @param impl Implementation (see lmp/la_global.h)
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp, 
            PLATFORM plat,
            IMPLEMENTATION impl, 
            const SYSTEM &my_system);

  /// Additional init interface which is compatible to both block and non-block
  /// matrix
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp, 
            PLATFORM plat,
            IMPLEMENTATION impl)
  {
    SYSTEM dummy;
    assert (plat != OPENCL);
    this->Init(comm, cp, plat, impl, dummy);
  }
            
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            PLATFORM plat,
            IMPLEMENTATION impl,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, cp, plat, impl);
  }
  
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            CBlockManagerSPtr block_manager)
  {
    this->Init(comm, cp);
  }

  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp)
  {
    this->Init(comm, cp, CPU, NAIVE);
  }
  
  void Init(const MPI_Comm &comm, 
            const LaCouplings &cp,
            IMPLEMENTATION impl)
  {
    this->Init(comm, cp, CPU, impl);
  }
  
  /// Sets every element to zero.
  void Zeros();

  // Functions inherited from Vector class
  DataType Dot(const Vector< DataType > &vec) const;
  void Axpy(const Vector< DataType > &vec, const DataType alpha);
  void ScaleAdd(const Vector< DataType > &vec, const DataType alpha);

  // NOTE: values are added only to diagonal part, i.e. ghost indices are filtered out
  void Add(const int *indices, const int size_indices, const DataType *values);

  /// Update operator, i.e. exchange values for distributed vectors

  void Update() {
    this->begin_update();
    this->end_update();
  }
  void begin_update();
  void end_update();

  // Specializations of inherited member functions of Vector to CoupledVector
  DataType Dot(const CoupledVector< DataType > &vec) const;
  void Axpy(const CoupledVector< DataType > &vec, const DataType alpha);
  void ScaleAdd(const CoupledVector< DataType > &vec, const DataType alpha);

  /// Adds a value. Component must be owned by process.
  /// @param global_dof_id Global dof index
  /// @param val Value to add
  void Add(int global_dof_id, DataType val);

  /// Scales this vector.
  /// @param alpha Scale factor
  void Scale(const DataType alpha);

  /// Returns the one-norm of this vector.
  /// @return One-norm
  DataType Norm1() const;

  /// Returns the two-norm of this vector.
  /// @return Two-norm
  DataType Norm2() const;

  /// Returns the maximum-norm of this vector.
  /// @return Maximum-norm
  DataType NormMax() const;

  /// Retrieve certain entries of the vector.
  /// The user has to provide the allocated array @em values.
  /// @param indices Global indices of the entries to retrieve
  ///        These indices have to be owned by the process invoking this
  ///        function or have to be included as couplings, i.e. in the ghost
  ///        vector. Otherwise, zero is returned (so that assembling with
  ///        linearization on a cell where no dof is owned works)
  /// @param size_indices Size of the array @em indices
  /// @param values Array of retrieved values
  virtual void GetValues(const int *indices, const int size_indices,
                         DataType *values) const;

  /// Retrieves the whole vector (interior).
  /// @param values Array of values, must fit size of interior
  void GetLocalValues(DataType *values) const;

  void GetLocalValues(const int *indices, const int size_indices,
                      DataType *values) const;

  /// Get value at a known index
  ///@param index Global index of the entry to retrieve. Index has to be owned
  /// by
  ///             the process invoking this function or has to be included as
  ///             couplings. Otherwise, zero is returned.
  virtual DataType GetValue(const int index) const;

  /// @return All Dofs and values that are in interior, ghost and pp_data_.
  /// They are NOT sorted.
  void GetAllDofsAndValues(std::vector< int > &id,
                           std::vector< DataType > &val) const;

  /// Sets every element to given value.
  /// @param val some value
  virtual void SetToValue(DataType val);
  virtual void SetToValue(const int *indices, const int size_indices, DataType val);

  /// Sets value at global index.
  /// @param index Global index.
  /// @param value Value to be set.
  virtual void SetValue(const int index, const DataType value);

  /// Sets values at given global indices.
  /// @param indices Global indices of the entries to be set.
  ///        These indices have to be owned by the process invoking this
  ///        function.
  /// @param size_indices Size of the array @em indices
  /// @param values Array of values to be set
  virtual void SetValues(const int *indices, const int size_indices,
                         const DataType *values);

  /// Sets all the values in the vector (interior).
  /// @param values Array of values, must fit size of interior
  void SetLocalValues(const DataType *values);

  /// Sets all ghost values. Used inside UpdateCouplings.
  /// @param values Values to set of size @em size_ghost
  void SetGhostValues(const DataType *values);

  /// Gathers the vector (only CPU vectors).
  /// @param recv_id Process id which gathers
  /// @param values Array of values to be stored (needs to be allocated: only on
  /// @em recv_id)
  void Gather(int recv_id, DataType *values) const;

  virtual Vector< DataType > *Clone() const;
  /// Clones the whole vector (everything).
  /// @param vec Vector to be cloned
  void CloneFrom(const CoupledVector< DataType > &vec);

  /// Copies interior and ghost (no structure, no platform).
  /// Vector already needs to be initialized.
  /// @param vec Vector to copy
  void CopyFrom(const CoupledVector< DataType > &vec);

  /// Copies only the interior (no ghost, no structure, no platform).
  void CopyInteriorFrom(const CoupledVector< DataType > &vec);

  void CopyFromWithoutGhost(const CoupledVector< DataType > &vec);

  /// Cast data of interior from another CoupledVector.
  void CastInteriorFrom(const CoupledVector< double > &vec);
  void CastInteriorFrom(const CoupledVector< float > &vec);

  /// Cast data to interior of another CoupledVector.
  void CastInteriorTo(CoupledVector< double > &vec) const;
  void CastInteriorTo(CoupledVector< float > &vec) const;

  void CopyTo(CoupledVector< DataType > &vec) const;

  /// Copies only the structure.
  void CopyStructureFrom(const CoupledVector< DataType > &vec);

  /// Copies everything but the values.
  /// @param vec Vector whose strucutre is to be copied
  void CloneFromWithoutContent(const CoupledVector< DataType > &vec);

  /// Clears allocated local vectors, i.e. interior and ghost.
  void Clear();

  /// Sends border values.
  void SendBorder();

  /// Receives ghost values.
  void ReceiveGhost();

  /// Waits for the completion of the send operations of the border values.
  void WaitForSend();

  /// Waits for the completion of the receive operations of the ghost values
  /// and stores them.
  void WaitForRecv();

  /// Updates couplings, i.e. exchange border and ghost values and
  /// values needed for post processing (if possible) between processors.
  void UpdateCouplings();

  /// Updates Ghost, i.e. exchange border and ghost values between processors.
  void UpdateGhost();

  /// @return Global size of the vector
  virtual int size_global() const;

  /// Prints information to stream \em out.
  /// @param out Stream for output
  void Print(std::ostream &out = std::cout) const;

  void WriteHDF5(const std::string &filename, const std::string &groupname,
                 const std::string &datasetname);

  void ReadHDF5(const std::string &filename, const std::string &groupname,
                const std::string &datasetname);

  // inline functions
  /// @return Local size of the vector, i.e. size of the interior.

  virtual int size_local() const {
    assert(this->interior_ != nullptr);
    return this->interior_->get_size();
  }

  /// @return Local size of ghost

  inline int size_local_ghost() const {
    assert(this->ghost_ != nullptr);
    return this->ghost_->get_size();
  }

  /// @param begin Global index of the first local entry
  /// @param end One more than the global index of the last local entry

  void GetOwnershipRange(int *begin, int *end) const {
    assert(this->ownership_begin_ > -1);
    assert(this->ownership_end_ > -1);

    *begin = this->ownership_begin_;
    *end = this->ownership_end_;
  }

  /// @return Global index of the first local entry

  inline int ownership_begin() const {
    return this->ownership_begin_;
  }

  /// @return One more than the global index of the last local entry

  inline int ownership_end() const {
    return this->ownership_end_;
  }

  /// @return LaCouplings

  inline const LaCouplings &la_couplings() const {
    return *(this->la_couplings_);
  }

  void store_interior()
  {
  }
  
  inline const lVector< DataType > &interior() const {
    return *(this->interior_);
  }

  inline lVector< DataType > &interior() {
    return *(this->interior_);
  }

  /// @return Local vector holding the couplings

  inline const lVector< DataType > &ghost() const {
    return *(this->ghost_);
  }

  inline lVector< DataType > &ghost() {
    return *(this->ghost_);
  }

  /// @return MPI communicator

  inline const MPI_Comm &comm() const {
    return this->comm_;
  }

  /// @return Number of processes

  inline int nb_procs() const {
    return this->nb_procs_;
  }

  /// @return Rank of this process

  inline int my_rank() const {
    return this->my_rank_;
  }

  /// @return Number of send operations for border values

  inline int nb_sends() const {
    return this->nb_sends_;
  }

  /// @return Number of receive operations for ghost values

  inline int nb_recvs() const {
    return this->nb_recvs_;
  }

  /// @return std::vector of MPI requests for communication

  inline std::vector< MPI_Request > mpi_req() const {
    return this->mpi_req_;
  }

  /// @return std::vector of MPI status for communication

  inline std::vector< MPI_Status > mpi_stat() const {
    return this->mpi_stat_;
  }

  /// @return std::vector of border values to be sent

  inline std::vector< DataType > border_val() const {
    return this->border_val_;
  }

  /// @return std::vector for ghost values to be received

  inline std::vector< DataType > ghost_val() const {
    return this->ghost_val_;
  }

  inline bool is_initialized() const {
    return this->initialized_;
  }

  inline IMPLEMENTATION get_impl() const {
    return this->impl_;
  }
  
  bool ContainsNaN() const 
  {
    bool diag = this->interior_->ContainsNaN();
    if (diag) 
    {
      return true;
    }
    if (ghost_ != nullptr)
    {
      return this->ghost_->ContainsNaN();
    }
    return false;
  }
private:
  // no implementation of copy constructor or assignement operator
  // CoupledVector ( const CoupledVector<DataType>& );
  // CoupledVector<DataType>& operator= ( const CoupledVector<DataType>& );

  // creates interior and ghost local vector object
  void Init_la_system(PLATFORM plat, IMPLEMENTATION impl, const SYSTEM &my_system);
  
  /// Initializes the structure of the vector.
  /// LaCouplings need to be initialized.
  void InitStructure();

  /// Computes ownership range via LaCouplings.
  void ComputeOwnershipRange();
  /// Global index of the first local entry.
  int ownership_begin_;
  /// One more than the global index of the last local entry.
  int ownership_end_;

  const LaCouplings *la_couplings_;
  lVector< DataType > *interior_; // entries of own domain
  lVector< DataType > *ghost_;

  /// MPI communicator.
  MPI_Comm comm_;
  /// Number of processes.
  int nb_procs_;
  /// Rank of this process.
  int my_rank_;
  /// Tool for MPI communication
  std::shared_ptr<ParCom> parcom_;
  
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

  /// Is true if we already checked for a dof partition in la_couplings
  bool checked_for_dof_partition_;

  bool initialized_;
  
  IMPLEMENTATION impl_;

  mutable std::vector< int > mut_indices_interior_;
  mutable std::vector< int > mut_indices_ghost_;
  mutable std::vector< DataType > mut_values_interior_;
  mutable std::vector< DataType > mut_values_ghost_;
  mutable std::vector< int > _shifted_indices;
  mutable std::vector< DataType > _insert_values;

};

template < class DataType > 
class CoupledVectorCreator 
{
public:
  CoupledVector< DataType > *params(const MPI_Comm &comm,
                                    const LaCouplings &cp,
                                    const PropertyTree &c) 
  {
    CoupledVector< DataType > *newCoupledVector = new CoupledVector< DataType >();
    read_LA_parameters(c, la_sys_.Platform, la_impl_, la_matrix_format_);
    
    if (la_sys_.Platform != CPU && la_sys_.Platform != GPU)  
    {
      LOG_ERROR("CoupledVectorCreator::params: No format of this name "
                  "registered(platform).");
      return nullptr;
    } 

    if (   la_impl_ != NAIVE && la_impl_ != BLAS && la_impl_ != MKL 
        && la_impl_ != OPENMP &&  la_impl_ != SCALAR && la_impl_ != SCALAR_TEX)
    {
      LOG_ERROR("CoupledVectorCreator::params: No format of this name " "registered(implementation).");
      return nullptr;
    }
    
    init_platform(la_sys_);

    newCoupledVector->Init(comm, cp, la_sys_.Platform, la_impl_, la_sys_);
    return newCoupledVector;
  }
     
private:
  SYSTEM la_sys_;
  IMPLEMENTATION la_impl_;
  MATRIX_FORMAT la_matrix_format_;
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARALGEBRA_COUPLED_VECTOR_H_


