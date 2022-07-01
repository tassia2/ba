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

#include "linear_algebra/pce_vector.h"

namespace hiflow {
namespace la {

// constructor

template < class LAD >
PCEVector< LAD >::PCEVector() {
  this->nmode_ = -1;
  this->totlevel_ = -1;
  this->initialized_ = false;
}

// deconstructor

template < class LAD >
PCEVector< LAD >::~PCEVector() {

  this->nmode_ = -1;
  this->totlevel_ = -1;
  this->initialized_ = false;
  this->Clear();
}

// initialize

template < class LAD >
void PCEVector< LAD >::Init(const PCTensor& pctensor,
                            const PVector& mean_vector) {

  // assign pctensor_
  this->pctensor_ = pctensor;

  // calculate nb of modes
  this->nmode_ = pctensor.Size();

  // calculate total level
  this->totlevel_ = pctensor_.GetLevel();

  // allocate nb of modes
  this->mode_vector_.resize(this->nmode_);

  // initialize with mean_vector
  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->mode_vector_[mode].CloneFromWithoutContent(mean_vector);
  }

  this->initialized_ = true;

  this->my_rank_ = mean_vector.my_rank();
  this->nb_procs_ = mean_vector.nb_procs();
}

template < class LAD >
void PCEVector< LAD >::Init(const PCTensor& pctensor, const MPI_Comm& comm,
                            const LaCouplings& cp) {

  // assign pctensor_
  this->pctensor_ = pctensor;

  // calculate nb of modes
  this->nmode_ = pctensor_.Size();

  // calculate total level
  this->totlevel_ = pctensor.GetLevel();

  // allocate nb of modes
  this->mode_vector_.resize(this->nmode_);

  // tmp vec
  PVector tmp;
  tmp.Init(comm, cp);

  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->mode_vector_[mode].CloneFromWithoutContent(tmp);
  }

  this->initialized_ = true;

  MPI_Comm_rank(comm, &this->my_rank_);
  MPI_Comm_size(comm, &this->nb_procs_);
}

// access the member of mode_vector_

template < class LAD >
typename LAD::VectorType&
PCEVector< LAD >::Mode(const int mode) {
  assert(this->initialized_ == true);
  return this->mode_vector_[mode];
}

template < class LAD >
const typename LAD::VectorType&
PCEVector< LAD >::GetMode(const int mode) const {
  assert(this->initialized_ == true);
  return this->mode_vector_[mode];
}

// return number of modes

template < class LAD >
int PCEVector< LAD >::nb_mode() const {
  assert(this->initialized_ == true);
  return this->nmode_;
}

// return total level

template < class LAD >
int PCEVector< LAD >::total_level() const {
  assert(this->initialized_ == true);
  return this->totlevel_;
}

// return pctensor

template < class LAD >
PCTensor PCEVector< LAD >::GetPCTensor() const {
  assert(this->initialized_ == true);
  return this->pctensor_;
}

// Update

template < class LAD >
void PCEVector< LAD >::Update() {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  for (int mode = 0; mode < this->nmode_; ++mode) {
    this->mode_vector_[mode].Update();
  }
}

template < class LAD >
void PCEVector< LAD >::Update(const int mode) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  assert(mode >= 0);

  this->mode_vector_[mode].Update();
}

template < class LAD >
void PCEVector< LAD >::begin_update() {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);

  for (int mode = 0; mode < this->nmode_; ++mode) {
    this->mode_vector_[mode].begin_update();
  }
}

template < class LAD >
void PCEVector< LAD >::end_update() {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);

  for (int mode = 0; mode < this->nmode_; ++mode) {
    this->mode_vector_[mode].end_update();
  }
}

// Clear
template < class LAD >
void PCEVector< LAD >::Clear() {
  if (this->initialized_ == true) {
    for (int mode = 0; mode != this->nmode_; ++mode) {
      this->mode_vector_[mode].Clear();
    }
  }
}

// Clone
template < class LAD >
PCEVector< LAD > *PCEVector< LAD >::Clone() const {
  LOG_ERROR("Called BlockVector::Clone not yet implemented!!!");
  exit(-1);
  return nullptr;
}


// Zeros
template < class LAD >
void PCEVector< LAD >::Zeros() {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->mode_vector_[mode].Zeros();
  }
}

// Axpy
template< class LAD >
void PCEVector< LAD >::Axpy(const Vector< PDataType >& vec,
                            const PDataType alpha) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  const PCEVector< LAD > *hv =
    dynamic_cast< const PCEVector< LAD > * >(&vec);

  if (hv != 0) {
    this->Axpy(*hv, alpha);
  } else {
    LOG_ERROR("Called BlockVector::Axpy with incompatible vector type.");
    exit(-1);
  }
}

template < class LAD >
void PCEVector< LAD >::Axpy(const PCEVector< LAD >& vec,
                            const PDataType alpha) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);

  for (int mode = 0; mode != this->nmode_; ++mode) {
    Axpy(mode, vec.GetMode(mode), alpha);
  }
}

template < class LAD >
void PCEVector< LAD >::Axpy(const PCEVector< LAD >& vec,
                            const PDataType alpha, const int l) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  assert(l >= 0);
  assert(this->totlevel_ >= 0);

  for (int mode = 0; mode != this->pctensor_.Size(l); ++mode) {
    Axpy(mode, vec.GetMode(mode), alpha);
  }
}

template < class LAD >
void PCEVector< LAD >::Axpy(const int mode,
                            const PCEVector< LAD >& vec,
                            const PDataType alpha) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  assert(this->totlevel_ >= 0);
  assert(mode >= 0);
  this->mode_vector_[mode].Axpy(vec.GetMode(mode), alpha);
}

template < class LAD >
void PCEVector< LAD >::Axpy(const int mode,
                            const PVector& vec,
                            const PDataType alpha) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  assert(this->totlevel_ >= 0);
  assert(mode >= 0);
  this->mode_vector_[mode].Axpy(vec, alpha);
}

// Dot
template < class LAD >
typename LAD::DataType
PCEVector< LAD >::Dot(const Vector< PDataType > &vec) const {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  const PCEVector< LAD > *hv =
    dynamic_cast< const PCEVector< LAD > * >(&vec);

  if (hv != 0) {
    return this->Dot(*hv);
  } else {
    LOG_ERROR("Called BlockVector::Dot with incompatible vector type.");
    exit(-1);
    return -1.;
  }
  return -1.;
}

template < class LAD >
typename LAD::DataType
PCEVector< LAD >::Dot(const PCEVector< LAD >& vec) const {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);

  PDataType dot = 0.0;
  for (int mode = 0; mode != this->nmode_; ++mode) {
    dot += this->Dot(mode, vec.GetMode(mode));
  }

  return dot;
}

template < class LAD >
typename LAD::DataType PCEVector< LAD >::Dot(const int mode,
    const PVector& vec) const {
  assert(this->initialized_);
  assert(this->nmode_ > 0);
  assert(mode >= 0);

  return this->mode_vector_[mode].Dot(vec);
}

// Add
template < class LAD >
void PCEVector< LAD >::Add(const int *indices, int length,
                           const PDataType *values) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  for (int mode = 0; mode < this->nmode_; ++mode) {
    this->mode_vector_[mode].Add(indices, length, values);
  }
}

template < class LAD >
void PCEVector< LAD >::Add(int index, PDataType scalar) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  this->Add(&index, 1, &scalar);
}

// ScaleAdd
template < class LAD >
void PCEVector< LAD >::ScaleAdd(const Vector< PDataType >& vec,
                                const PDataType alpha) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  const PCEVector< LAD > *hv =
    dynamic_cast< const PCEVector< LAD > * >(&vec);

  if (hv != 0) {
    this->ScaleAdd(*hv, alpha);
  } else {
    LOG_ERROR("Called BlockVector::Axpy with incompatible vector type.");
    exit(-1);
  }

}

template < class LAD >
void PCEVector< LAD >::ScaleAdd(const PCEVector< LAD >& vec,
                                const PDataType alpha) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->ScaleAdd(mode, vec.GetMode(mode), alpha);
  }
}

template < class LAD >
void PCEVector< LAD >::ScaleAdd(const int mode,
                                const PVector& vec,
                                const PDataType alpha) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);
  assert(mode >= 0);

  this->mode_vector_[mode].ScaleAdd(vec, alpha);
}

// Scale
template < class LAD >
void PCEVector< LAD >::Scale(const PDataType alpha) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->mode_vector_[mode].Scale(alpha);
  }
}

// Set Value
template < class LAD >
void PCEVector< LAD >::SetValues(const int *indices, const int size_indices,
                                 const PDataType *values) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  for (int mode = 0; mode < this->nmode_; ++mode) {
    this->mode_vector_[mode].SetValues(indices, size_indices, values);
  }
}

template < class LAD >
void PCEVector< LAD >::SetValue(const int index, const PDataType value) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  this->SetValues(&index, 1, &value);
}

template < class LAD >
void PCEVector< LAD >::SetToValue(PDataType value) {
  assert(this->initialized_);
  assert(this->nmode_ > 0);

  for (int mode = 0; mode < this->nmode_; ++mode) {
    this->mode_vector_[mode].SetToValue(value);
  }

}

// size
template < class LAD >
int PCEVector< LAD >::size_local() const {
  assert(this->initialized_ == true);
  return (this->nmode_ * this->size_local(0));
}

template < class LAD >
int PCEVector< LAD >::size_local(const int mode) const {
  assert(this->initialized_ == true);
  return this->mode_vector_[mode].size_local();
}

template < class LAD >
int PCEVector< LAD >::size_global() const {
  assert(this->initialized_ == true);
  return (this->nmode_ * this->size_global(0));
}

template < class LAD >
int PCEVector< LAD >::size_global(const int mode) const {
  assert(this->initialized_ == true);
  return this->mode_vector_[mode].size_global();
}

template < class LAD >
int PCEVector< LAD >::size_local_ghost() const {
  assert(this->initialized_ == true);
  return (this->nmode_ * this->size_local_ghost(0));
}

template < class LAD >
int PCEVector< LAD >::size_local_ghost(const int mode) const {
  assert(this->initialized_ == true);
  return this->mode_vector_[mode].size_local_ghost();
}

// Norm1
template < class LAD >
typename LAD::DataType PCEVector< LAD >::Norm1() const {
  assert(this->initialized_);

  PDataType val = 0.0;
  for (int mode = 0; mode < this->nmode_; ++mode) {
    val += this->mode_vector_[mode].Norm1();
  }

  return val;
}

// Norm2
template < class LAD >
typename LAD::DataType PCEVector< LAD >::Norm2() const {
  assert(this->initialized_ == true);
  return sqrt(this->Dot(*this));
}

// NormMax
template < class LAD >
typename LAD::DataType PCEVector< LAD >::NormMax() const {
  assert(this->initialized_);

  PDataType val = 0.0;
  for (int mode = 0; mode < this->nmode_; ++mode) {
    val = std::max(val, this->mode_vector_[mode].NormMax());
  }

  return val;
}


// CloneFrom
template < class LAD >
void PCEVector< LAD >::CloneFrom(const PCEVector< LAD >& vec) {
  if (this->initialized_ == false) {
    this->Init(vec.GetPCTensor(), vec.GetMode(0));
  }

  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->CloneFrom(mode, vec.GetMode(mode));
  }
}

template < class LAD >
void PCEVector< LAD >::CloneFrom(const PCEVector< LAD >& vec,
                                 const int l) {
  if (this->initialized_ == false) {
    PCTensor pctensor = vec.GetPCTensor();
    pctensor.SetLevel(l);

    // restrict to own size is smaller than input
    assert(pctensor.Size() < vec.nb_mode());
    this->Init(pctensor, vec.GetMode(0));
  }

  for (int mode = 0; mode != this->pctensor_.Size(l); ++mode) {
    this->CloneFrom(mode, vec.GetMode(mode));
  }
}

template < class LAD >
void PCEVector< LAD >::CloneFrom(const int mode,
                                 const PVector& vec) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  assert(mode >= 0);
  this->mode_vector_[mode].CloneFrom(vec);
}

// CopyFrom
template < class LAD >
void PCEVector< LAD >::CopyFrom(const PCEVector< LAD >& vec) {
  if (this->initialized_ == false) {
    this->Init(vec.GetPCTensor(), vec.GetMode(0));
  }

  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->CopyFrom(mode, vec.GetMode(mode));
  }
}

template < class LAD >
void PCEVector< LAD >::CopyFrom(const PCEVector< LAD >& vec,
                                const int l) {
  if (this->initialized_ == false) {
    PCTensor pctensor = vec.GetPCTensor();
    pctensor.SetLevel(l);

    // restrict to own size is smaller than input
    assert(pctensor.Size() < vec.nb_mode());
    this->Init(vec.GetPCTensor(), vec.GetMode(0));
  }

  for (int mode = 0; mode != this->pctensor_.Size(l); ++mode) {
    this->CopyFrom(mode, vec.GetMode(mode));
  }
}

template < class LAD >
void PCEVector< LAD >::CopyFrom(const int mode,
                                const PVector& vec) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  assert(mode >= 0);
  this->mode_vector_[mode].CopyFrom(vec);
}

// CopyFromWithoutGhost
template < class LAD >
void PCEVector< LAD >::CopyFromWithoutGhost(
  const PCEVector< LAD >& vec) {
  if (this->initialized_ == false) {
    this->Init(vec.GetPCTensor(), vec.GetMode(0));
  }

  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->CopyFromWithoutGhost(mode, vec.GetMode(mode));
  }
}

template < class LAD >
void PCEVector< LAD >::CopyFromWithoutGhost(
  const PCEVector< LAD >& vec, const int l) {
  if (this->initialized_ == false) {
    PCTensor pctensor = vec.GetPCTensor();
    pctensor.SetLevel(l);

    // restrict to own size is smaller than input
    assert(pctensor.Size() < vec.nb_mode());
    this->Init(vec.GetPCTensor(), vec.GetMode(0));
  }

  for (int mode = 0; mode != this->pctensor_.Size(l); ++mode) {
    this->CopyFromWithoutGhost(mode, vec.GetMode(mode));
  }
}

template < class LAD >
void PCEVector< LAD >::CopyFromWithoutGhost(
  const int mode, const PVector& vec) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  assert(mode >= 0);
  this->mode_vector_[mode].CopyFromWithoutGhost(vec);
}

// CloneFromWithoutContent
template < class LAD >
void PCEVector< LAD >::CloneFromWithoutContent(
  const PCEVector< LAD >& vec) {
  if (this->initialized_ == false) {
    this->Init(vec.GetPCTensor(), vec.GetMode(0));
  }

  for (int mode = 0; mode != this->nmode_; ++mode) {
    this->CloneFromWithoutContent(mode, vec.GetMode(mode));
  }
}

template < class LAD >
void PCEVector< LAD >::CloneFromWithoutContent(
  const PCEVector< LAD >& vec, const int l) {
  if (this->initialized_ == false) {
    PCTensor pctensor = vec.GetPCTensor();
    pctensor.SetLevel(l);

    // restrict to own size is smaller than input
    assert(pctensor.Size() < vec.nb_mode());

    this->Init(vec.GetPCTensor(), vec.GetMode(0));
  }

  for (int mode = 0; mode != this->pctensor_.Size(l); ++mode) {
    this->CloneFromWithoutContent(mode, vec.GetMode(mode));
  }
}

template < class LAD >
void PCEVector< LAD >::CloneFromWithoutContent(
  const int mode, const PVector& vec) {
  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);
  assert(mode >= 0);

  this->mode_vector_[mode].CloneFromWithoutContent(vec);
}

// Write and Read vector content to/from HDF5 file
template < class LAD >
void PCEVector< LAD >::WriteHDF5(const std::string filename,
                                 const std::string groupname,
                                 const std::string datasetname) {

  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);

#ifdef WITH_HDF5

  for (int mode = 0; mode != this->nmode_; ++mode) {
    std::ostringstream groupname_mode;
    groupname_mode << groupname << "_mode" << mode;

    std::ostringstream datasetname_mode;
    datasetname_mode << datasetname << ".Mode(" << mode << ")";

    this->mode_vector_[mode].WriteHDF5(filename, groupname_mode.str(),
                                       datasetname_mode.str());
  }

#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

template < class LAD >
void PCEVector< LAD >::ReadHDF5(const std::string filename,
                                const std::string groupname,
                                const std::string datasetname) {

  assert(this->initialized_ == true);
  assert(this->nmode_ > 0);

#ifdef WITH_HDF5
  for (int mode = 0; mode != this->nmode_; ++mode) {
    std::ostringstream groupname_mode;
    groupname_mode << groupname << "_mode" << mode;

    std::ostringstream datasetname_mode;
    datasetname_mode << datasetname << ".Mode(" << mode << ")";

    this->mode_vector_[mode].ReadHDF5(filename, groupname_mode.str(),
                                      datasetname_mode.str());
  }

#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

// template instantiation
template class PCEVector< LADescriptorCoupledD >;
template class PCEVector< LADescriptorCoupledS >;

#ifdef WITH_HYPRE
template class PCEVector< LADescriptorHypreD >;
#endif

//#ifdef WITH_PETSC
//template class PCEVector< LADescriptorPETScD >;
//#endif

} // namespace la
} // namespace hiflow
