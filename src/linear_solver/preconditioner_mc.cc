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

/// \author Dimitar Lukarski, Chandramowli Subramanian

#include <cassert>
#include <iostream>

#include "common/log.h"
#include "dof/dof_partition.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/lmp/lpreconditioner_mc.h"
#include "linear_solver/preconditioner_mc.h"

#include "linear_algebra/lmp/lpreconditioner_mc_gs.h"
#include "linear_algebra/lmp/lpreconditioner_mc_ilup.h"
#include "linear_algebra/lmp/lpreconditioner_mc_sgs.h"

namespace hiflow {
namespace la {

template < class LAD, int DIM >
PreconditionerMultiColoring< LAD, DIM >::PreconditionerMultiColoring()
    : PreconditionerBlockJacobi< LAD >() {
  this->lprecond_mc_ = nullptr;
}

template < class LAD, int DIM >
PreconditionerMultiColoring< LAD, DIM >::~PreconditionerMultiColoring() {
  this->Clear();
  this->lprecond_mc_ = nullptr;
}

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::Init_SymmetricGaussSeidel() {
  // clear old precond
  bool reuse = this->GetReuse();
  this->Clear();
  this->SetReuse(reuse);

  // build new precond
  lPreconditioner_MultiColoring_SymmetricGaussSeidel< DataType > *lp =
      new lPreconditioner_MultiColoring_SymmetricGaussSeidel< DataType >();
  //  lp->Init();
  this->lprecond_mc_ = lp;
}

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::Init_GaussSeidel() {
  // clear old precond
  bool reuse = this->GetReuse();
  this->Clear();
  this->SetReuse(reuse);

  // build new precond
  lPreconditioner_MultiColoring_GaussSeidel< DataType > *lp =
      new lPreconditioner_MultiColoring_GaussSeidel< DataType >();
  //  lp->Init();
  this->lprecond_mc_ = lp;
}

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::Init_ILU(const int fillins,
                                                  const int power) {
  // clear old precond
  bool reuse = this->GetReuse();
  this->Clear();
  this->SetReuse(reuse);

  // build new precond
  lPreconditioner_MultiColoring_ILUp< DataType > *lp =
      new lPreconditioner_MultiColoring_ILUp< DataType >();
  lp->Init(fillins, // fill ins
           power,   // matrix power
           true,    // MC
           true,    // drop-off
           false);  // LS
  this->lprecond_mc_ = lp;
}

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::Init_ILU(const int fillins) {
  // clear old precond
  bool reuse = this->GetReuse();
  this->Clear();
  this->SetReuse(reuse);

  // build new precond
  lPreconditioner_MultiColoring_ILUp< DataType > *lp =
      new lPreconditioner_MultiColoring_ILUp< DataType >();
  lp->Init(fillins, fillins + 1, true, true, false);
  this->lprecond_mc_ = lp;
}

// vectors needed for platform/implementation

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::Preprocess(
    const OperatorType &op, const VectorType &x,
    doffem::DofPartition< typename LAD::DataType, DIM > *dof) {
  assert(this->lprecond_mc_ != nullptr);

  // compute permutation
  this->lprecond_mc_->SetupOperator(op.diagonal());
  this->lprecond_mc_->SetupVector(&(x.interior()));
  this->lprecond_mc_->Analyse();

  int *ind, *index = this->lprecond_mc_->get_permut();

  ind = (int *)malloc(op.diagonal().get_num_row() * sizeof(int));
  assert(ind != nullptr);

  // permute the index
  // from index dst -> index src
  for (int i = 0; i < op.diagonal().get_num_row(); ++i) {
    ind[index[i]] = i;
  }

  // permute dof
  //  this->PermuteDofNumbering(this->lprecond_mc_->get_permut(), dof);
  this->PermuteDofNumbering(ind, dof);

  free(ind);
}

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::SetupOperator(OperatorType &op) {
  this->op_ = &op;
  this->lprecond_mc_->reInitOperator(op.diagonal());

  this->SetModifiedOperator(true);
  this->lprecond_mc_->SetModifiedOperator(true);
}

template < class LAD, int DIM >
LinearSolverState
PreconditionerMultiColoring< LAD, DIM >::SolveImpl(const VectorType &b,
                                              VectorType *x) {
  assert(this->lprecond_mc_ != nullptr);

  this->lprecond_mc_->ApplylPreconditioner(b.interior(), &(x->interior()));

  return kSolverSuccess;
}

template < class LAD, int DIM > void PreconditionerMultiColoring< LAD, DIM >::Clear() {
  if (this->lprecond_mc_ != nullptr) {
    delete this->lprecond_mc_;
  }
  this->lprecond_mc_ = nullptr;
  Preconditioner< LAD >::Clear();
}

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::BuildImpl(VectorType const *b,
                                                   VectorType *x) {
  assert(this->lprecond_mc_ != nullptr);
  assert(this->op_ != nullptr);

  this->lprecond_mc_->Build_LU();

  this->lprecond_mc_->SetModifiedOperator(false);
  this->lprecond_mc_->SetState(true);
}

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::PermuteDofNumbering(
    const doffem::lDofId *perm,
    doffem::DofPartition< typename LAD::DataType, DIM > *dof) {

  MPI_Comm comm = dof->get_mpi_comm();
  int my_rank = dof->my_subdom();
  int size_perm = dof-> nb_dofs_on_subdom(my_rank);

  // retrieve number of processes
  int number_proc;
  int info = MPI_Comm_size(comm, &number_proc);
  assert(info == MPI_SUCCESS);

  // local to global
  doffem::gDofId *global_perm = new doffem::gDofId[dof->nb_dofs_global()];
  for (int i = 0; i < size_perm; ++i) {
    dof->local2global(perm[i], &(global_perm[dof->my_dof_offset() + i]));
  }

  // number of data to be sent to each process i
  int *sendcounts = new int[number_proc];
  for (int i = 0; i < number_proc; ++i) {
    sendcounts[i] = size_perm;
  }
  sendcounts[my_rank] = 0;

  // offset for array to be sent
  int *sdispls = new int[number_proc];
  for (int i = 0; i < number_proc; ++i) {
    sdispls[i] = dof->my_dof_offset();
  }

  // number of data to be received from each process i
  int *recvcounts = new int[number_proc];
  for (int i = 0; i < number_proc; ++i) {
    recvcounts[i] = dof-> nb_dofs_on_subdom(i);
  }
  recvcounts[my_rank] = 0;
  // offset for array where is received
  int *rdispls = new int[number_proc];
  rdispls[0] = 0;
  for (int i = 1; i < number_proc; ++i) {
    rdispls[i] = rdispls[i - 1] + dof-> nb_dofs_on_subdom(i - 1);
  }

  // all to all communication, since border values could be changed
  info = MPI_Alltoallv(global_perm, sendcounts, sdispls, MPI_INT, global_perm,
                       recvcounts, rdispls, MPI_INT, comm);
  assert(info == MPI_SUCCESS);

  // now permute dof numbering
  // DOF: provide interface for int*
  //  dof->apply_permutation(global_perm);
  std::vector< doffem::gDofId > global_perm_vector(dof->nb_dofs_global());
  for (int j = 0; j < global_perm_vector.size(); ++j)
    global_perm_vector[j] = global_perm[j];

  dof->apply_permutation(global_perm_vector);

  // clear allocated memory
  delete[] global_perm;
  delete[] sendcounts;
  delete[] sdispls;
  delete[] recvcounts;
  delete[] rdispls;
}

template < class LAD, int DIM >
void PreconditionerMultiColoring< LAD, DIM >::Print(std::ostream &out) const {
  this->lprecond_mc_->print(out);
}

/// template instantiation
template class PreconditionerMultiColoring< LADescriptorCoupledD, 2>;
template class PreconditionerMultiColoring< LADescriptorCoupledD, 3>;

} // namespace la
} // namespace hiflow
