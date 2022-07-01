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

#ifndef HIFLOW_LINEARSOLVER_PRECONDITIONER_H_
#define HIFLOW_LINEARSOLVER_PRECONDITIONER_H_

#include "common/log.h"
#include "common/property_tree.h"
#include "common/timer.h"
#include <cassert>
#include <iostream>

namespace hiflow {
namespace la {

/// Enumerator @em LinearSolverState as return value for the preconditioners
/// and linear solvers.

enum LinearSolverState { kSolverSuccess = 0, kSolverExceeded, kSolverError };

/// @brief Base class for all preconditioners and linear solvers in HiFlow.

template < class LAD > class Preconditioner {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  /// Constructor

  Preconditioner()
      : modified_op_(false), state_(false), reuse_(true), use_solver_op_(false),
        print_level_(0), op_(nullptr), info_(nullptr), iter_(0),
        is_critical_hypre_solver_(false), num_build_(0), num_solve_(0),
        acc_iter_(0), time_build_(0.), time_solve_(0.),
        name_("Preconditioner") {}

  /// Destructor

  virtual ~Preconditioner() {}

  /// Sets up the operator, e.g. the system matrix.
  /// If implemented, it must be invoked before
  /// @c ApplyPreconditioner is called.
  /// For instance it could compute an ILU decomposition of a GlobalMatrix.
  /// Sets up the operator for the linear solver.
  /// @param op linear operator

  virtual void SetupOperator(OperatorType &op) {
    this->op_ = &op;
    this->SetModifiedOperator(true);
    this->SetUseSolverOperator(false);
  }

  virtual void SetupVector(const VectorType &vec) {}

  /// Sets up paramaters.

  virtual void InitParameter() {}

  virtual void InitControl(int maxits)
  {
    this->maxits_ = maxits;
  }
  
  /// Build the preconditioner such that is ready to be used inside of the
  /// solver
  virtual void Build() { this->Build(nullptr, nullptr); }

  virtual void Build(VectorType const *b, VectorType *x) {
    if (this->print_level_ > 2) {
      LOG_INFO(this->name_, "Build");
    }

    // assert ( this->op_ != nullptr );
    Timer timer;
    timer.start();

    this->BuildImpl(b, x);

    this->SetState(true);
    this->SetModifiedOperator(false);
    timer.stop();
    this->time_build_ += timer.get_duration();
    this->num_build_++;
  }

  /// Applies the preconditioner which is possibly an inexact linear solver.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if preconditioning succeeded
  virtual LinearSolverState Solve(const VectorType &b, VectorType *x) {
    assert(x->is_initialized());
    assert(b.is_initialized());

    if (!this->GetState()) {
      this->Build(&b, x);
    }

    if (this->print_level_ > 2) {
      LOG_INFO(this->name_, "Solve");
    }

    Timer timer;
    timer.start();
    LinearSolverState state = this->SolveImpl(b, x);
    timer.stop();

    this->num_solve_++;
    this->time_solve_ += timer.get_duration();
    this->acc_iter_ += this->iter();

    return state;
  }

  /// Erase possibly allocated data.

  virtual void Clear() {
    this->op_ = nullptr;
    this->modified_op_ = false;
    this->state_ = false;
    this->reuse_ = true;
    this->print_level_ = 0;
    this->use_solver_op_ = false;
    this->is_critical_hypre_solver_ = false;
    this->iter_ = 0;
    this->num_build_ = 0;
    this->num_solve_ = 0;
    this->time_build_ = 0.;
    this->time_solve_ = 0.;
    this->acc_iter_ = 0;
    this->name_ = "Preconditioner";
  }

  /// Destroy solver object, if external libraries like Hypre are used

  virtual void DestroySolver() {}

  /// Set level of information in log file
  /// 0: no output, 1: initial+final residual + iteration count,
  /// 2: CPU time, 3: additional information and parameters (if available)
  /// \param[in] print_level Level

  virtual void SetPrintLevel(int print_level) {
    this->print_level_ = print_level;
  }

  /// Print possibly info

  virtual void Print(std::ostream &out = std::cout) const {}

  /// Set state of the preconditioner, i.e. whether it is ready to use or not.
  /// In case of reuse_ == false, this function always sets the state to false
  /// @param[in] bool state

  /// Set solver information object

  virtual void SetInfo(PropertyTree *info) { info_ = info; }

  virtual void SetState(bool state) { this->state_ = state; }

  /// Get State of the preconditioner
  /// @return state

  virtual bool GetState() { return this->state_; }

  /// Set flag whether preconditioner should be resued by further calls to the
  /// outer solving routine. Usually, this option should always be set to true.
  /// However, there might be MPI communicator problems when reusing too many
  /// BoomerAMG preconditioners at the same time .
  /// @param[in] bool flag

  virtual void SetReuse(bool flag) { this->reuse_ = flag; }

  /// Get reuse flag
  /// @return  flag

  virtual bool GetReuse() { return this->reuse_; }

  /// Set flag whether solver operator should be used
  /// @param[in] bool flag

  virtual void SetUseSolverOperator(bool flag) { this->use_solver_op_ = flag; }

  /// Return flag if solver operator should be used
  /// @return flag

  virtual bool GetUseSolverOperator() { return this->use_solver_op_; }
  /// Set status of operator
  /// @param[in] bool flag

  virtual void SetModifiedOperator(bool flag) {
    this->modified_op_ = flag;
    if (flag) {
      this->SetState(false);
    }
  }

  /// Get status of operator
  /// @param[in] bool flag

  virtual bool GetModifiedOperator() { return this->modified_op_; }

  /// Return pointer to operator
  /// @return pointer

  virtual OperatorType *GetOperator() { return this->op_; }

  /// Set critical flag
  /// @param[in] flag

  virtual void SetCritical(bool flag) {
    this->is_critical_hypre_solver_ = flag;
  }

  /// Return critical flag
  /// @return flag

  virtual bool IsCritical() { return this->is_critical_hypre_solver_; }

  /// @return Number of iterations for last solve.

  virtual int iter() const { return this->iter_; }

  virtual void SetName(const std::string &name) { this->name_ = name; }

  virtual void GetStatistics(int &acc_iter, int &num_build, int &num_solve,
                             DataType &time_build, DataType &time_solve,
                             bool erase = false) {
    num_build = this->num_build_;
    num_solve = this->num_solve_;

    time_build = this->time_build_;
    time_solve = this->time_solve_;

    acc_iter = this->acc_iter_;

    if (erase) {
      this->num_build_ = 0;
      this->time_build_ = 0.;
      this->num_solve_ = 0;
      this->time_solve_ = 0.;
      this->acc_iter_ = 0;
    }
  }

  DataType ComputeResidual(const VectorType &b, VectorType *x) 
  {
    assert (this->op_ != nullptr);
    assert (x != nullptr);
    
    VectorType tmp;
    tmp.CloneFromWithoutContent (b);
    tmp.Zeros();
    
    this->op_->VectorMult(*x, &tmp);
    tmp.ScaleAdd(b, static_cast< DataType >(-1.));
    tmp.Update();
    
    DataType norm = tmp.Norm2();
    return norm;
  }
  
protected:
  /// specific solve implementation
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if solver succeeded
  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x) = 0;

  /// specific build implementation
  virtual void BuildImpl(VectorType const *b, VectorType *x) = 0;

  /// Pointer to operator
  OperatorType *op_;

  /// Flag if operator has changed
  bool modified_op_;

  /// Flag if preconditioner is set up, i.e. ready to use inside of the solver.
  /// This flag is set to false, if either the operator or some parameters have
  /// changed
  bool state_;

  /// Flag if preconditioner should be reused by several calls to outer solve
  /// routine. Only valid, if no changes for the oprator have been made. Default
  /// is true
  bool reuse_;

  /// Flag if the solver's operator is used
  bool use_solver_op_;

  /// Print Level
  int print_level_;

  /// Solver information
  PropertyTree *info_;

  /// Number of iterations
  int iter_;

  /// Maximum number of iterations
  int maxits_;
  
  /// Indicating whether preconditioner gives rise to MPI Problems, if it is
  /// reuse. E.g. Hypre CG
  bool is_critical_hypre_solver_;

  int num_build_;
  int num_solve_;
  int acc_iter_;

  DataType time_build_;
  DataType time_solve_;

  /// Helper for info log
  std::string name_;
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_PRECONDITIONER_H_
