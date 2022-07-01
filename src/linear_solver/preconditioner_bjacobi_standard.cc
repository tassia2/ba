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

/// @author Dimitar Lukarski, Chandramowli Subramanian

#include "linear_solver/preconditioner_bjacobi_standard.h"
#include "linear_algebra/lmp/lpreconditioner.h"
#include "linear_solver/preconditioner_bjacobi.h"

#include <cassert>

#include "common/log.h"
#include "linear_algebra/la_descriptor.h"
#
namespace hiflow {
namespace la {

template < class LAD >
PreconditionerBlockJacobiStand< LAD >::PreconditionerBlockJacobiStand()
    : PreconditionerBlockJacobi< LAD >() {
  this->localPrecond_ = nullptr;
}

template < class LAD >
PreconditionerBlockJacobiStand< LAD >::~PreconditionerBlockJacobiStand() {
  this->Clear();
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Init(const std::string& local_solver_type, 
                                                 const PropertyTree &params) 
{
  PropertyTree cur_param = params[local_solver_type];
  
  if (local_solver_type == "HiflowILU")
  {
    this->Init_ILUp(cur_param["Bandwidth"].get<int>());
  }
  else if (local_solver_type == "SOR")
  {
    this->Init_SOR(cur_param["Omega"].get<DataType>());
  }
  else if (local_solver_type == "SSOR")
  {
    this->Init_SSOR(cur_param["Omega"].get<DataType>());
  }
  else if (local_solver_type == "Jacobi")
  {
    this->Init_Jacobi();
  }
  else if (local_solver_type == "FSAI")
  {
    this->Init_FSAI(cur_param["NumIter"].get<int>(),
                       cur_param["RelRes"].get<DataType>(),
                       cur_param["AbsRes"].get<DataType>(),
                       cur_param["Power"].get<int>() );
  }
  else
  {
    assert (false);
  }
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Init_Jacobi() {
  this->Clear();
  lPreconditioner_Jacobi< typename LAD::DataType > *lp =
      new lPreconditioner_Jacobi< typename LAD::DataType >;
  lp->Init();
  this->localPrecond_ = lp;
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Init_GaussSeidel() {
  this->Clear();
  lPreconditioner_GaussSeidel< typename LAD::DataType > *lp =
      new lPreconditioner_GaussSeidel< typename LAD::DataType >;
  lp->Init();
  this->localPrecond_ = lp;
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Init_SymmetricGaussSeidel() {
  this->Clear();
  lPreconditioner_SymmetricGaussSeidel< typename LAD::DataType > *lp =
      new lPreconditioner_SymmetricGaussSeidel< typename LAD::DataType >;
  lp->Init();
  this->localPrecond_ = lp;
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Init_SOR(
    const typename LAD::DataType omega) {
  this->Clear();
  lPreconditioner_SOR< typename LAD::DataType > *lp =
      new lPreconditioner_SOR< typename LAD::DataType >;
  lp->Init(omega);
  this->localPrecond_ = lp;
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Init_SSOR(
    const typename LAD::DataType omega) {
  this->Clear();
  lPreconditioner_SSOR< typename LAD::DataType > *lp =
      new lPreconditioner_SSOR< typename LAD::DataType >;
  lp->Init(omega);
  this->localPrecond_ = lp;
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Init_ILUp(const int ilu_p) {
  this->Clear();
  lPreconditioner_ILUp< typename LAD::DataType > *lp =
      new lPreconditioner_ILUp< typename LAD::DataType >;
  lp->Init(ilu_p);
  this->localPrecond_ = lp;
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Init_FSAI(const int solver_max_iter, 
                                                      const typename LAD::DataType solver_rel_eps,
                                                      const typename LAD::DataType solver_abs_eps, 
                                                      const int matrix_power)
{
  this->Clear();
  lPreconditioner_FSAI< typename LAD::DataType > *lp =
      new lPreconditioner_FSAI< typename LAD::DataType >;

  lp->Init(solver_max_iter, solver_rel_eps, solver_abs_eps, matrix_power); 
  this->localPrecond_ = lp;
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::SetupOperator(OperatorType &op) {
  assert (this->localPrecond_ != nullptr);
  
  this->op_ = &op;
  this->localPrecond_->SetupOperator(op.diagonal());

  this->SetModifiedOperator(true);
  this->localPrecond_->SetModifiedOperator(true);
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::BuildImpl(VectorType const *b,
                                                      VectorType *x) {
  assert(this->localPrecond_ != nullptr);
  assert(this->op_ != nullptr);
  
  if ((b!= nullptr) && (x!= nullptr))
  {
    this->localPrecond_->Build(&(b->interior()),&(x->interior()));
  }
  else
  {
    this->localPrecond_->Build(nullptr, nullptr);
  }
  
  this->localPrecond_->SetModifiedOperator(false);
  this->localPrecond_->SetState(true);
}

template < class LAD > void PreconditionerBlockJacobiStand< LAD >::Clear() {
  if (this->localPrecond_ != nullptr) 
  {
    this->localPrecond_->Clear();
    delete this->localPrecond_;
  }
  this->localPrecond_ = nullptr;
  Preconditioner< LAD >::Clear();
}

template < class LAD >
LinearSolverState
PreconditionerBlockJacobiStand< LAD >::SolveImpl(const VectorType &b,
                                                 VectorType *x) {
  assert(this->op_ != nullptr);
  assert(this->localPrecond_ != nullptr);

  this->localPrecond_->ApplylPreconditioner(b.interior(), &(x->interior()));

  return kSolverSuccess;
}

template < class LAD >
void PreconditionerBlockJacobiStand< LAD >::Print(std::ostream &out) const {
  this->localPrecond_->print(out);
}

/// template instantiation
template class PreconditionerBlockJacobiStand< LADescriptorCoupledD >;
template class PreconditionerBlockJacobiStand< LADescriptorCoupledS >;

} // namespace la
} // namespace hiflow
