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

/// @author Philipp Gerstner

#ifndef HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_MATFREE_H_
#define HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_MATFREE_H_

#include "linear_algebra/lmp/lpreconditioner.h"
#include "linear_solver/preconditioner_bjacobi.h"
#include "linear_solver/preconditioner.h"
#include "common/log.h"
#include "linear_algebra/la_descriptor.h"

#include <cassert>

namespace hiflow {
namespace la {

/// @brief Standard Block Jacobi preconditioners which do not need an explicit matrix

template < class LAD >
class PreconditionerBlockJacobiMatFree : public PreconditionerBlockJacobi< LAD > 
{
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  PreconditionerBlockJacobiMatFree();
  virtual ~PreconditionerBlockJacobiMatFree();

  /// Setup the local operator for the local preconditioner
  virtual void SetupOperator(OperatorType &op);

  /// Sets up paramaters
  virtual void Init_Jacobi(const VectorType& diag);
                                                                         
  /// Erase possibly allocated data.
  virtual void Clear();

  virtual void Print(std::ostream &out = std::cout) const;

protected:
  /// Applies the preconditioner on the diagonal block.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if preconditioning succeeded
  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x);

  /// Build the preconditioner
  virtual void BuildImpl(VectorType const *b, VectorType *x);

  lPreconditioner< typename LAD::DataType > *localPrecond_;
};

template < class LAD >
PreconditionerBlockJacobiMatFree< LAD >::PreconditionerBlockJacobiMatFree()
    : PreconditionerBlockJacobi< LAD >() {
  this->localPrecond_ = nullptr;
}

template < class LAD >
PreconditionerBlockJacobiMatFree< LAD >::~PreconditionerBlockJacobiMatFree() 
{
  this->Clear();
}

template < class LAD >
void PreconditionerBlockJacobiMatFree< LAD >::Init_Jacobi(const VectorType& diag) 
{
  this->Clear();
  lPreconditioner_Jacobi< typename LAD::DataType > *lp =
      new lPreconditioner_Jacobi< typename LAD::DataType >;
  lp->Init(&(diag.interior()));
  this->localPrecond_ = lp;
}

template < class LAD >
void PreconditionerBlockJacobiMatFree< LAD >::SetupOperator(OperatorType &op) 
{
  assert (this->localPrecond_ != nullptr);
  
  this->op_ = &op;

  this->SetModifiedOperator(true);
  this->localPrecond_->SetModifiedOperator(true);
}

template < class LAD >
void PreconditionerBlockJacobiMatFree< LAD >::BuildImpl(VectorType const *b,
                                                      VectorType *x) 
{
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

template < class LAD > void PreconditionerBlockJacobiMatFree< LAD >::Clear() 
{
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
PreconditionerBlockJacobiMatFree< LAD >::SolveImpl(const VectorType &b,
                                                   VectorType *x) 
{
  assert(this->op_ != nullptr);
  assert(this->localPrecond_ != nullptr);

  this->localPrecond_->ApplylPreconditioner(b.interior(), &(x->interior()));

  return kSolverSuccess;
}

template < class LAD >
void PreconditionerBlockJacobiMatFree< LAD >::Print(std::ostream &out) const {
  this->localPrecond_->print(out);
}

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_STANDARD_H_
