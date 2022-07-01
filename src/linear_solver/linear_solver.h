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

/// @author Chandramowli Subramanian, Philipp Gerstner

#ifndef HIFLOW_LINEARSOLVER_LINEAR_SOLVER_H_
#define HIFLOW_LINEARSOLVER_LINEAR_SOLVER_H_

#include <cstdlib>

#include "common/iterate_control.h"
#include "common/log.h"
#include "common/timer.h"
#include "linear_solver/preconditioner.h"
#include "nonlinear/nonlinear_problem.h"
#include <mpi.h>

namespace hiflow {
namespace la {

/// @brief Base class for all linear solvers in HiFlow.
///
/// A linear solver is in particular a preconditioner.

template < class LAD, class PreLAD = LAD >
class LinearSolver : virtual public Preconditioner< LAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  typedef typename PreLAD::MatrixType PrecondOperatorType;

  LinearSolver()
    : precond_(nullptr), pre_op_(nullptr), non_lin_op_(nullptr),
      filter_solution_(false), res_(0.0), res_rel_(0.), res_init_(0.),
      reltol_(1e-8), abstol_(1e-15), divtol_(1e6) {
    this->SetMethod("NoPreconditioning");
    this->state_ = false;
    this->print_level_ = 0;
    this->modified_op_ = false;
    this->reuse_ = true;
    this->op_ = nullptr;
    this->use_solver_op_ = false;
    this->is_critical_hypre_solver_ = false;
    this->print_level_ = 0;
    this->name_ = "LinearSolver";
    this->num_precond_ = 0;
    this->iter_precond_ = 0;
    this->time_precond_ = 0.;
    this->iter_ = 0;
    this->time_build_ = 0.;
    this->time_solve_ = 0.;
    this->num_solve_ = 0;
    this->num_build_ = 0;
    this->acc_iter_ = 0;
    this->maxits_ = 1000;
  }

  virtual ~LinearSolver() {}

  virtual void InitControl(int maxits);

  virtual void InitControl(int maxits, DataType abstol, DataType reltol,
                           DataType divtol);

  /// Set relative tolerance. Function is needed in nonlinear solver when
  /// a forcing strategy is applied

  virtual void SetRelativeTolerance(DataType reltol) {
    this->reltol_ = reltol;
  }

  /// Sets up parameters.

  virtual void InitParameter() {}

  /// Sets up parameters.

  virtual void InitParameter(std::string const &type) {}

  /// Sets up a preconditioner for the linear solver.
  /// @param precond preconditioner

  virtual void SetupPreconditioner(Preconditioner< PreLAD > &precond) {
    this->precond_ = &precond;
    this->SetMethod("RightPreconditioning");
  }

  /// Sets up the operator, e.g. the system matrix.
  /// If implemented, it must be invoked before
  /// @c ApplyPreconditioner is called.
  /// For instance it could compute an ILU decomposition of a GlobalMatrix.
  /// Sets up the operator for the linear solver.
  /// @param op linear operator

  virtual void SetupOperator(OperatorType &op) {
    this->op_ = &op;
    this->SetModifiedOperator(true);

    this->pre_op_ = dynamic_cast< PrecondOperatorType * >(&op);
    if (this->precond_ != nullptr) {
      if (this->precond_->GetUseSolverOperator()) {
        if (this->pre_op_ != 0) {
          this->precond_->SetupOperator(*this->pre_op_);
          this->precond_->SetUseSolverOperator(true);
        } else {
          LOG_INFO("LinearSolver::SetupOperator",
                   " Operator of linear solver could not be casted to operator "
                   "type of underlying preconditioner");
        }
      }
    }
  }

  /// Left, Right, or no preconditining
  /// \param[in] method "NoPreconditioning" or "Preconditioning" or
  /// "LeftPreconditioning" or "RightPreconditioning"

  virtual void SetMethod(const std::string &method) {
    this->precond_method_ = method;
  }

  /// @return Type of preconditioning

  virtual const std::string &Method() const {
    return this->precond_method_;
  }

  inline LinearSolverState ApplyPreconditioner(const VectorType &b,
      VectorType *x);

  /// Clear allocated data

  virtual void Clear() {
    this->res_ = 0.;
    this->iter_ = 0;
    this->op_ = nullptr;
    this->precond_ = nullptr;
    this->non_lin_op_ = nullptr;
    this->filter_solution_ = false;
    this->print_level_ = 0;
    this->modified_op_ = false;
    this->state_ = false;
    this->reuse_ = true;
    this->use_solver_op_ = false;
    this->is_critical_hypre_solver_ = false;
    this->res_ = 0.;
    this->res_rel_ = 0.;
    this->res_init_ = 0.;
    this->maxits_ = 1000;
    this->abstol_ = 1e-15;
    this->reltol_ = 1e-8;
    this->divtol_ = 1e6;
    this->name_ = "LinearSolver";

    this->num_precond_ = 0;
    this->iter_precond_ = 0;
    this->time_precond_ = 0.;

    this->time_build_ = 0.;
    this->time_solve_ = 0.;
    this->num_solve_ = 0;
    this->num_build_ = 0;
    this->acc_iter_ = 0;
    this->FreeBasis();
  }

  virtual void FreeBasis()
  {}
  
  /// Set NonlinearProblem operator which provides ApplyFilter function
  /// \param[in] nlp Pointer to nonlinear Problem

  virtual void SetupNonLinProblem(NonlinearProblem< LAD > *nlp) {
    this->non_lin_op_ = nlp;
    this->filter_solution_ = true;
  }

  /// @return pointer to preconditioner

  virtual Preconditioner< PreLAD > *GetPreconditioner() {
    return this->precond_;
  }

  /// @return iterate control

  virtual IterateControl &control() {
    return this->control_;
  }

  /// @return Absolute residual

  virtual DataType res() const {
    return this->res_;
  }

  /// @return Relative residual

  virtual DataType res_rel() const {
    return this->res_rel_;
  }

  /// @return Initial residual

  virtual DataType res_init() const {
    return this->res_init_;
  }

  virtual void GetStatisticsPrecond(int &num_precond, int &iter_precond,
                                    double &time_precond, bool erase = false) {
    num_precond = this->num_precond_;
    iter_precond = this->iter_precond_;
    time_precond = this->time_precond_;

    if (erase) {
      this->num_precond_ = 0;
      this->iter_precond_ = 0;
      this->time_precond_ = 0.;
    }
  }

protected:
  /// specific solve implementation
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if solver succeeded
  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x) = 0;

  /// specific build implementation
  virtual void BuildImpl(VectorType const *b, VectorType *x) {
    if (this->precond_ != nullptr) {
      if (!this->precond_->GetState() || !this->precond_->GetReuse()) {
        this->BuildPreconditioner(b, x);
      }
    }
  }

  /// Build the preconditioner, such that it is ready to be used inside of a
  /// solver. The Build function of the underlying preconditioner is always
  /// called, even if precond_setup == true

  virtual void BuildPreconditioner(VectorType const *b, VectorType *x) {
    assert(this->op_ != nullptr);
    assert(this->precond_ != nullptr);

    if (this->precond_->GetOperator() == nullptr) {
      if (this->pre_op_ != 0) {
        this->precond_->SetupOperator(*this->pre_op_);
        this->precond_->SetUseSolverOperator(true);
      } else {
        if (this->print_level_ > 0) {
          LOG_INFO(
            this->name_,
            "LinearSolver::SetupOperator: Operator of linear solver could "
            "not be casted to operator type of underlying preconditioner");
        }
      }
    }

    this->precond_->Build(b, x);
  }

  /// Pointer to preconditioner
  Preconditioner< PreLAD > *precond_;

  PrecondOperatorType *pre_op_;

  /// Pointer to nonlinear problem in order to apply solution filtering
  NonlinearProblem< LAD > *non_lin_op_;

  /// Flag whether solution should be filtered
  bool filter_solution_;

  /// Convergence control
  IterateControl control_;

  /// (Absolut) Residual norm
  DataType res_;

  /// (Relative) Residual norm
  DataType res_rel_;

  /// (Initial) Residual norm
  DataType res_init_;

  /// Relative convergence tolerance
  DataType reltol_;

  /// Absolute convergence tolerance
  DataType abstol_;

  /// Absolute divergence tolerance
  DataType divtol_;

  /// Left, right or no preconditioning
  std::string precond_method_;

  int num_precond_;
  int iter_precond_;
  double time_precond_;
};

/// Sets up linear control.
/// @param maxits maximum number of iteration steps
/// @param abstol absolute tolerance of residual to converge
/// @param reltol relative tolerance of residual to converge
/// @param divtol relative tolerance of residual to diverge

template < class LAD, class PreLAD >
void LinearSolver< LAD, PreLAD >::InitControl(int maxits, DataType abstol,
    DataType reltol,
    DataType divtol) {
  this->control_.Init(maxits, abstol, reltol, divtol);
  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, "Maximum iterations : " << maxits);
    LOG_INFO(this->name_, "Absolute tolerance [convergence]: " << abstol);
    LOG_INFO(this->name_, "Relative tolerance [convergence]: " << reltol);
    LOG_INFO(this->name_, "Relative tolerance [divergence]: " << divtol);
  }

  this->maxits_ = maxits;
  this->abstol_ = abstol;
  this->reltol_ = reltol;
  this->divtol_ = divtol;
}

template < class LAD, class PreLAD >
void LinearSolver< LAD, PreLAD >::InitControl(int maxits) {
  this->control_.Init(maxits, this->abstol_, this->reltol_, this->divtol_);
  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, "Maximum iterations : " << maxits);
  }

  this->maxits_ = maxits;
}

/// Applying the underlying preconditioner
/// @param b right hand side vector
/// @param x solution vector

template < class LAD, class PreLAD >
inline LinearSolverState
LinearSolver< LAD, PreLAD >::ApplyPreconditioner(const VectorType &b,
    VectorType *x) {

  assert(this->precond_ != nullptr);
  // assert ( this->precond_->GetState ( ) );

  Timer timer;
  timer.start();
  LinearSolverState state = this->precond_->Solve(b, x);
  timer.stop();

  this->time_precond_ += timer.get_duration();
  this->num_precond_++;
  this->iter_precond_ += this->precond_->iter();
  return state;
}

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_LINEAR_SOLVER_H_
