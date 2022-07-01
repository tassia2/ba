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

#ifndef HIFLOW_LINEARALGEBRA_PCE_VECTOR_
#define HIFLOW_LINEARALGEBRA_PCE_VECTOR_

#include "config.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/vector.h"
#include "polynomial_chaos/pc_tensor.h"

using namespace hiflow::polynomialchaos;

namespace hiflow {
namespace la {

template < class LAD >
class PCEVector : public Vector< typename LAD::DataType > {
public:

  typedef typename LAD::VectorType PVector;
  typedef typename LAD::MatrixType PMatrix;
  typedef typename LAD::DataType PDataType;

  // constructor
  PCEVector();

  // destructor
  ~PCEVector();

  // initialize
  void Init(const PCTensor& pctensor,
            const PVector& mean_vector);
  void Init(const PCTensor& pctensor, const MPI_Comm& comm,
            const LaCouplings& cp);

  // access member of mode_vector_
  PVector& Mode(const int mode);
  const PVector& GetMode(const int mode) const;

  // return number of modes
  int nb_mode() const;

  // return total level
  int total_level() const;

  // return pctensor
  PCTensor GetPCTensor() const;

  // Update
  void Update();
  void Update(const int mode);
  /// Initiate update
  void begin_update();
  /// Finalize update
  void end_update();

  // Clear
  void Clear();

  // Clone
  virtual PCEVector< LAD > *Clone() const;

  // Zeros
  void Zeros();

  // Axpy
  void Axpy(const Vector< PDataType >& vec, const PDataType alpha);
  void Axpy(const PCEVector< LAD >& vec, const PDataType alpha);
  void Axpy(const PCEVector< LAD >& vec, const PDataType alpha, const int l);
  void Axpy(const int mode, const PCEVector< LAD >& vec, const PDataType alpha);
  void Axpy(const int mode, const PVector& vec, const PDataType alpha);

  // Dot
  PDataType Dot(const Vector< PDataType >& vec) const;
  PDataType Dot(const PCEVector< LAD >& vec) const;
  PDataType Dot(const int mode, const PVector& vec) const;

  // Add
  void Add(int index, PDataType scalar);
  /// Add values to given indices
  void Add(const int *indices, int length, const PDataType *values);

  // ScaleAdd
  void ScaleAdd(const Vector< PDataType > &vecx, PDataType alpha);
  void ScaleAdd(const PCEVector< LAD >& vec, const PDataType alpha);
  void ScaleAdd(const int mode, const PVector& vec,
                const PDataType alpha);

  // Scale
  void Scale(const PDataType alpha);

  // Set Value
  void SetValue(int index, PDataType value);
  void SetValues(const int *indices, int size_indices, const PDataType *values);
  void SetToValue(PDataType val);

  // size
  int size_local() const;
  int size_local(const int mode) const;

  int size_global() const;
  int size_global(const int mode) const;

  int size_local_ghost() const;
  int size_local_ghost(const int mode) const;

  // Norm1, Norm2, NormMax
  PDataType Norm1() const;
  PDataType Norm2() const;
  PDataType NormMax() const;

  // CloneFrom
  void CloneFrom(const PCEVector< LAD >& vec);
  void CloneFrom(const PCEVector< LAD >& vec, const int l);
  void CloneFrom(const int mode, const PVector& vec);

  // CopyFrom
  void CopyFrom(const PCEVector< LAD >& vec);
  void CopyFrom(const PCEVector< LAD >& vec, const int l);
  void CopyFrom(const int mode, const PVector& vec);

  // CopyFromWithoutGhost
  void CopyFromWithoutGhost(const PCEVector< LAD >& vec);
  void CopyFromWithoutGhost(const PCEVector< LAD >& vec, const int l);
  void CopyFromWithoutGhost(const int mode, const PVector& vec);

  // CloneFromWithoutContent
  void CloneFromWithoutContent(const PCEVector< LAD >& vec);
  void CloneFromWithoutContent(const PCEVector< LAD >& vec, const int l);
  void CloneFromWithoutContent(const int mode, const PVector& vec);

  // Write and Read vector to/frome HDF5 file
  void WriteHDF5(const std::string filename, const std::string groupname,
                 const std::string datasetname);
  void ReadHDF5(const std::string filename, const std::string groupname,
                const std::string datasetname);

  /// @return Rank of this process

  int my_rank() const {
    return this->my_rank_;
  }

  /// @return Number of send operations for border values

  int nb_procs() const {
    return this->nb_procs_;
  }

  bool is_initialized() const {
    for (int i = 0; i < this->mode_vector_.size(); ++i) {
      if (!this->mode_vector_[i].is_initialized()) {
        return false;
      }
    }
    return this->initialized_;
  }

  /// @return All Dofs and values that are in interior, ghost and pp_data_.
  /// They are NOT sorted.
  void GetAllDofsAndValues(std::vector< int >& id,
                           std::vector< PDataType >& val) const
  {
    NOT_YET_IMPLEMENTED;
  }

  PDataType GetValue(const int index) const
  {
    PDataType value;
    this->GetValues(&index, 1, &value);
    return value;
  }

  void GetValues(const int* indices, const int size_indices,
                 PDataType* values) const
  {
    NOT_YET_IMPLEMENTED;
  }

  void GetLocalValues(PDataType *values) const
  {
    NOT_YET_IMPLEMENTED;
  }

  void SetLocalValues(const PDataType *values)
  {
    NOT_YET_IMPLEMENTED;
  }

private:
  // set of mode vectors
  std::vector< PVector > mode_vector_;

  // pc tensor
  PCTensor pctensor_;

  // number of modes
  int nmode_;

  // total level
  int totlevel_;

  // intialized flag
  bool initialized_;

  /// Number of processes.
  int nb_procs_;
  /// Rank of this process.
  int my_rank_;
};

} // namespace la
} // namespace hiflow

#endif
