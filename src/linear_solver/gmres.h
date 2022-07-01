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

/// @author Hendryk Bockelmann, Chandramowli Subramanian

#ifndef HIFLOW_LINEARSOLVER_GMRES_H_
#define HIFLOW_LINEARSOLVER_GMRES_H_

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/timer.h"
#include "linear_algebra/pce_matrix.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "linear_solver/linear_solver.h"
#include "linear_solver/linear_solver_creator.h"

namespace hiflow {
namespace la {

/// @brief GMRES solver
///
/// GMRES solver for linear systems Ax=b with left, right or no preconditioning.
/// (not flexible!)

template < class LAD, class PreLAD = LAD >
class GMRES : public LinearSolver< LAD, PreLAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  GMRES();
  virtual ~GMRES();

  virtual void InitParameter(int size_basis, std::string method);

  virtual int size_basis() const { return this->size_basis_; }

  virtual void Clear() {
    LinearSolver< LAD, PreLAD >::Clear();
    this->name_ = "GMRES";
  }

  /// Set flag whether or not Krylov basis should be reused

  virtual void SetReuseBasis(bool flag) { this->reuse_basis_ = flag; }
  /// Sets the relative tolerance.
  /// Needed by Inexact Newton Methods
  /// @param reltol relative tolerance of residual to converge

  virtual void SetRelativeTolerance(double reltol) {
    int maxits = this->control_.maxits();
    double atol = this->control_.absolute_tol();
    double dtol = this->control_.divergence_tol();
    this->control_.Init(maxits, atol, reltol, dtol);
  }

  /// Initialize pointers for Krylov Basis
  virtual void InitBasis(const VectorType &ref_vec, int iteration);

  /// Allocate Krylov basis vectors
  virtual void AllocateBasis(int basis_size);

  /// Set all basis vectors to zero
  virtual void SetBasisToZero();

  /// Deallocate Krylov basis vectors
  virtual void FreeBasis();

protected:
  /// Applies Givens rotation.
  /// @param cs cos(phi)
  /// @param sn sin(phi)
  /// @param dx first coordinate
  /// @param dy second coordinate

  inline virtual void ApplyPlaneRotation(const DataType &cs, const DataType &sn,
                                         DataType *dx, DataType *dy) const {
    const DataType temp = cs * (*dx) + sn * (*dy);
    *dy = -sn * (*dx) + cs * (*dy);
    *dx = temp;
  }

  /// Generates Givens rotation.
  /// @param dx first coordinate
  /// @param dy second coordinate
  /// @param cs cos(phi)
  /// @param sn sin(phi)

  inline virtual void GeneratePlaneRotation(const DataType &dx,
                                            const DataType &dy, DataType *cs,
                                            DataType *sn) const {
    const DataType beta = std::sqrt(dx * dx + dy * dy);
    *cs = dx / beta;
    *sn = dy / beta;
  }

  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x);

  virtual LinearSolverState SolveNoPrecond(const VectorType &b, VectorType *x);
  virtual LinearSolverState SolveLeft(const VectorType &b, VectorType *x);
  virtual LinearSolverState SolveRight(const VectorType &b, VectorType *x);

  virtual void UpdateSolution(VectorType **V,
                              const hiflow::la::SeqDenseMatrix< DataType > &H,
                              const std::vector< DataType > &g, int k,
                              VectorType *x) const;

  /// basis of subspace
  std::vector< VectorType * > V_;
  VectorType w_;
  VectorType z_;

  /// max size of the Krylov subspace basis
  int size_basis_;
  int num_init_basis_;
  bool aux_vec_init_;
  bool reuse_basis_;
};

/// standard constructor

template < class LAD, class PreLAD >
GMRES< LAD, PreLAD >::GMRES() : LinearSolver< LAD, PreLAD >() {
  this->size_basis_ = 100;
  this->num_init_basis_ = 0;
  this->aux_vec_init_ = false;
  this->reuse_basis_ = false;
  this->V_.clear();
  this->name_ = "GMRES";
  if (this->print_level_ > 2) {
    LOG_INFO("Linear solver", "GMRES");
  }
}

/// destructor

template < class LAD, class PreLAD > GMRES< LAD, PreLAD >::~GMRES() {
  this->op_ = nullptr;
  this->precond_ = nullptr;

  this->Clear();
  this->FreeBasis();
}

/// init parameter
/// @param size_basis dimension of Krylov subspace
/// @param method either "RightPreconditioning", "LeftPreconditioning" or
///               "NoPreconditioning"

template < class LAD, class PreLAD >
void GMRES< LAD, PreLAD >::InitParameter(int size_basis, std::string method) 
{
  // choose size_basis_
  this->size_basis_ = size_basis;

  // chose method_
  this->precond_method_ = method;
  assert((this->Method() == "RightPreconditioning") ||
         (this->Method() == "LeftPreconditioning") ||
         (this->Method() == "NoPreconditioning"));
  if (this->print_level_ > 2) 
  {
    LOG_INFO(this->name_, "Preconditioning  = " << this->Method());
    LOG_INFO(this->name_, "Max Krylov basis = " << this->size_basis_);
  }
}

/// Solves the linear system.
/// @param b right hand side vector
/// @param x start and solution vector

template < class LAD, class PreLAD >
LinearSolverState GMRES< LAD, PreLAD >::SolveImpl(const VectorType &b,
                                                  VectorType *x) {
  assert(x->is_initialized());
  assert(b.is_initialized());

  if (this->size_basis_ <= 0) {
    LOG_ERROR("GMRES::Solve: Improper Krylov basis size " << this->size_basis_);
    LOG_ERROR("Returning solver error...");
    return kSolverError;
  }

  Timer timer;
  if (this->info_ != nullptr) {
    timer.reset();
    timer.start();
  }

  LinearSolverState state = kSolverError;
  if (this->Method() == "RightPreconditioning" && this->precond_ != nullptr) {
    state = this->SolveRight(b, x);
  } else if (this->Method() == "LeftPreconditioning" && this->precond_ != nullptr) {
    state = this->SolveLeft(b, x);
  } else {
    state = this->SolveNoPrecond(b, x);
  }

  if (this->info_ != nullptr) {
    timer.stop();
    this->info_->add("iter", this->iter());
    this->info_->add("time", timer.get_duration());
  }

  return state;
}

/// Solve without preconditioning.

template < class LAD, class PreLAD >
LinearSolverState GMRES< LAD, PreLAD >::SolveNoPrecond(const VectorType &b,
                                                       VectorType *x) {
  //assert(this->Method() == "NoPreconditioning");

  assert(this->op_ != nullptr);
  assert(this->size_basis() > 0);

  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, "solve without preconditioning");
  }

  IterateControl::State conv = IterateControl::kIterate;

  // compute really used basis size as minimum of maximum iterations and
  // given basis size
  const int basis_size_actual =
      std::min(this->size_basis(), this->control().maxits());

  // Hessenberg matrix
  hiflow::la::SeqDenseMatrix< DataType > H;
  H.Resize(basis_size_actual, basis_size_actual + 1);

  // Allocate array of pointer for Krylov subspace basis
  this->AllocateBasis(basis_size_actual);
  this->SetBasisToZero();

  // Init Basis vectors and auxilary vectors
  this->InitBasis(b, 0);

  std::vector< DataType > g(basis_size_actual +
                            1); // rhs of least squares problem
  std::vector< DataType > cs(basis_size_actual + 1); // Givens rotations
  std::vector< DataType > sn(basis_size_actual + 1); // Givens rotations

  int iter = 0;
  
  // compute residual V[0] = b - Ax
  this->V_[0]->CloneFromWithoutContent(b);
  this->op_->VectorMult(*x, this->V_[0]);
  this->V_[0]->ScaleAdd(b, static_cast< DataType >(-1.));
  if (this->filter_solution_) {
    V_[0]->Update();
    assert (this->non_lin_op_ != nullptr);
    this->non_lin_op_->ApplyFilter(*this->V_[0]);
  }

  this->res_init_ = this->res_ = this->V_[0]->Norm2();
  this->res_rel_ = 1.;
  conv = this->control().Check(iter, this->res());

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "initial res norm   = " << this->res_);
  }
  
  // main loop
  while (conv == IterateControl::kIterate) {
    g.assign(g.size(), static_cast< DataType >(0.)); // g = 0
    H.Zeros();

    assert(this->res_ != static_cast< DataType >(0.));
    this->V_[0]->Scale(static_cast< DataType >(1.) /
                       this->res_); // norm residual
    g[0] = this->res_;

    for (int j = 0; j != basis_size_actual; ++j) {
      ++iter;

      this->op_->VectorMult(*this->V_[j], &this->w_); // w = Av_j
      if (this->filter_solution_) {
        this->w_.Update();
        this->non_lin_op_->ApplyFilter(this->w_);
      }
      // -- start building Hessenberg matrix H --
      // vectors in V are ONB of Krylov subspace K_i(A,V[0])
      for (int i = 0; i <= j; ++i) {
        H(j, i) = this->w_.Dot(*this->V_[i]);
        this->w_.Axpy(*this->V_[i], static_cast< DataType >(-1.) * H(j, i));
      }

      H(j, j + 1) = this->w_.Norm2();
      assert(H(j, j + 1) != static_cast< DataType >(0.));

      this->w_.Scale(static_cast< DataType >(1.) / H(j, j + 1));
      this->V_[j + 1]->CloneFrom(this->w_);
      // -- end building Hessenberg matrix H --

      // apply old Givens rotation on old H entries
      for (int k = 0; k != j; ++k) {
        this->ApplyPlaneRotation(cs[k], sn[k], &H(j, k), &H(j, k + 1));
      }

      // determine new Givens rotation for actual iteration i
      this->GeneratePlaneRotation(H(j, j), H(j, j + 1), &cs[j], &sn[j]);

      // apply Givens rotation on new H element
      this->ApplyPlaneRotation(cs[j], sn[j], &H(j, j), &H(j, j + 1));

      // update g for next dimension -> g[j+1] is norm of actual residual
      this->ApplyPlaneRotation(cs[j], sn[j], &g[j], &g[j + 1]);

      this->res_ = std::abs(g[j + 1]);
      this->res_rel_ = this->res_ / this->res_init_;
      if (this->print_level_ > 2) {
        LOG_INFO(this->name_, " iteration " << std::setw(3) << iter
                                            << ": residual: " << this->res_);
      }
      conv = this->control().Check(iter, this->res_);

      if (conv != IterateControl::kIterate) {
        this->UpdateSolution(&this->V_[0], H, g, j, x);
        if (this->filter_solution_) {
          x->Update();
          assert (this->non_lin_op_ != nullptr);
          this->non_lin_op_->ApplyFilter(*x);
        }
        break;
      }
    } // for (int j = 0; j < this->size_basis(); ++j)

    // setup for restart
    if (conv == IterateControl::kIterate) {
      // -> update solution
      this->UpdateSolution(&this->V_[0], H, g, basis_size_actual - 1,
                           x); // x = x + Vy
      if (this->filter_solution_) {
        x->Update();
        this->non_lin_op_->ApplyFilter(*x);
      }
      // -> compute residual Ax-b
      this->op_->VectorMult(*x, this->V_[0]);
      this->V_[0]->ScaleAdd(b, static_cast< DataType >(-1.));
      if (this->filter_solution_) {
        this->V_[0]->Update();
        assert (this->non_lin_op_ != nullptr);
        this->non_lin_op_->ApplyFilter(*this->V_[0]);
      }
    }
  } // while (conv == IterateControl::kIterate)

  this->iter_ = iter;

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "final iterations   = " << this->iter_);
    LOG_INFO(this->name_, "final abs res norm = " << this->res_);
    LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_);
  }

  // deallocate Krylov subspace basis V
  if (!this->reuse_basis_) {
    this->FreeBasis();
  }

  if (conv == IterateControl::kFailureDivergenceTol ||
      conv == IterateControl::kFailureMaxitsExceeded) {
    return kSolverExceeded;
  }
  return kSolverSuccess;
}

/// Solve with left preconditioning.

template < class LAD, class PreLAD >
LinearSolverState GMRES< LAD, PreLAD >::SolveLeft(const VectorType &b,
                                                  VectorType *x) {
  assert(this->Method() == "LeftPreconditioning");

  assert(this->op_ != nullptr);
  assert(this->precond_ != nullptr);
  assert(this->size_basis() > 0);

  LOG_ERROR("GMRES::SolveLeft: Not implemented yet.");
  LOG_ERROR("Returning solver error...\n");
  return kSolverError;
}

template < class LAD, class PreLAD >
void GMRES< LAD, PreLAD >::AllocateBasis(int basis_size) {
  int diff = basis_size + 1 - this->V_.size();

  for (int i = 0; i < diff; ++i) {
    this->V_.push_back(new VectorType);
  }
}

template < class LAD, class PreLAD >
void GMRES< LAD, PreLAD >::InitBasis(const typename LAD::VectorType &ref_vec,
                                     int iteration) {
  assert(ref_vec.is_initialized());
  if (iteration == 0) {
    if (!this->aux_vec_init_) {
      this->w_.CloneFromWithoutContent(ref_vec);
      this->z_.CloneFromWithoutContent(ref_vec);
      this->aux_vec_init_ = true;
    }
  }
  assert(this->V_.size() >= iteration + 1);
  int num_init = this->num_init_basis_;
  for (int i = num_init; i <= iteration; ++i) {
    this->V_[i]->CloneFromWithoutContent(ref_vec);
    this->num_init_basis_ = i + 1;
  }
}

template < class LAD, class PreLAD >
void GMRES< LAD, PreLAD >::SetBasisToZero() {
  if (this->aux_vec_init_) {
    this->w_.Zeros();
    this->z_.Zeros();
  }
  for (int i = 0; i < this->num_init_basis_; ++i) {
    this->V_[i]->Zeros();
  }
}

template < class LAD, class PreLAD > void GMRES< LAD, PreLAD >::FreeBasis() {
  this->w_.Clear();
  this->z_.Clear();
  for (size_t i = 0; i < this->V_.size(); ++i) {
    if (this->V_[i] != nullptr)
    {
      this->V_[i]->Clear();
      delete this->V_[i];
    }
  }
  this->V_.clear();
  this->num_init_basis_ = 0;
  this->aux_vec_init_ = false;
}

/// Solve with right preconditioning.

template < class LAD, class PreLAD >
LinearSolverState GMRES< LAD, PreLAD >::SolveRight(const VectorType &b,
                                                   VectorType *x) {
  assert(this->Method() == "RightPreconditioning");

  assert(this->op_ != nullptr);
  assert(this->precond_ != nullptr);
  assert(this->size_basis() > 0);

  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, "solve with right preconditioning");
  }

  IterateControl::State conv = IterateControl::kIterate;

  // compute really used basis size as minimum of maximum iterations and
  // given basis size
  const int basis_size_actual =
      std::min(this->size_basis(), this->control().maxits());

  // Hessenberg matrix
  hiflow::la::SeqDenseMatrix< DataType > H;
  H.Resize(basis_size_actual, basis_size_actual + 1);

  // Allocate array of pointer for Krylov subspace basis
  this->AllocateBasis(basis_size_actual);
  this->SetBasisToZero();

  // Init Basis vectors and auxilary vectors
  this->InitBasis(b, 0);

  std::vector< DataType > g(basis_size_actual +
                            1); // rhs of least squares problem
  std::vector< DataType > cs(basis_size_actual + 1); // Givens rotations
  std::vector< DataType > sn(basis_size_actual + 1); // Givens rotations

  int iter = 0;

  // compute residual V[0] = b - Ax
  // V[0]->CloneFromWithoutContent ( b );
  this->op_->VectorMult(*x, this->V_[0]);
  this->V_[0]->ScaleAdd(b, static_cast< DataType >(-1.));

  if (this->filter_solution_) {
    this->V_[0]->Update();
    assert (this->non_lin_op_ != nullptr);
    this->non_lin_op_->ApplyFilter(*this->V_[0]);
  }

  this->res_init_ = this->res_ = this->V_[0]->Norm2();
  this->res_rel_ = 1.;
  conv = this->control().Check(iter, this->res());

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "initial res norm   = " << this->res_);
  }

  // main loop
  while (conv == IterateControl::kIterate) {
    g.assign(g.size(), static_cast< DataType >(0.)); // g = 0
    H.Zeros();

    assert(this->res_ != static_cast< DataType >(0.));
    this->V_[0]->Scale(static_cast< DataType >(1.) /
                       this->res_); // norm residual
    g[0] = this->res_;

    for (int j = 0; j != basis_size_actual; ++j) {
      ++iter;
      this->z_.Zeros();

      // pass on information object
      if (this->info_ != nullptr) {
        std::stringstream pre_str;
        pre_str << "precond"
                << std::setw(1 + std::log10(this->control().maxits()))
                << std::setfill('0') << iter;
        this->info_->add(pre_str.str());
        this->precond_->SetInfo(this->info_->get_child(pre_str.str()));
      }
      this->ApplyPreconditioner(*this->V_[j], &this->z_); // z ~= M^-1 v_j
      
      
      
      if (this->filter_solution_) {
        this->z_.Update();
        this->non_lin_op_->ApplyFilter(this->z_);
      }
      this->op_->VectorMult(this->z_, &this->w_); // w = A z
      if (this->filter_solution_) {
        this->w_.Update();
        assert (this->non_lin_op_ != nullptr);
        this->non_lin_op_->ApplyFilter(this->w_);
      }
      
      // -- start building Hessenberg matrix H --
      // vectors in V are ONB of Krylov subspace K_i(A,V[0])
      for (int i = 0; i <= j; ++i) {
        H(j, i) = this->w_.Dot(*this->V_[i]);
        this->w_.Axpy(*this->V_[i], static_cast< DataType >(-1.) * H(j, i));
      }
  ////////////////////////////////////////
      /*std::vector<int> ids;
      std::vector<DataType> values;
      w_.GetAllDofsAndValues(ids, values);    
      for (int i = 0; i < values.size(); ++i) {
        std::cout << values[i] << std::endl;
      }*/
//////////////////////////////////////////
      H(j, j + 1) = this->w_.Norm2();
      assert(H(j, j + 1) != static_cast< DataType >(0.));

      this->w_.Scale(static_cast< DataType >(1.) / H(j, j + 1));

      // Init new basis vector
      this->InitBasis(b, j + 1);
      this->V_[j + 1]->CopyFrom(this->w_);

      // -- end building Hessenberg matrix H --

      // apply old Givens rotation on old H entries
      for (int k = 0; k != j; ++k)
        this->ApplyPlaneRotation(cs[k], sn[k], &H(j, k), &H(j, k + 1));

      // determine new Givens rotation for actual iteration i
      this->GeneratePlaneRotation(H(j, j), H(j, j + 1), &cs[j], &sn[j]);

      // apply Givens rotation on new H element
      this->ApplyPlaneRotation(cs[j], sn[j], &H(j, j), &H(j, j + 1));

      // update g for next dimension -> g[j+1] is norm of actual residual
      this->ApplyPlaneRotation(cs[j], sn[j], &g[j], &g[j + 1]);

      this->res_ = std::abs(g[j + 1]);
      this->res_rel_ = this->res_ / this->res_init_;
      if (this->print_level_ > 2) {
        LOG_INFO(this->name_, "iteration " << std::setw(3) << iter
                                            << ": residual: " << this->res_);
      }
      conv = this->control().Check(iter, this->res_);

      if (conv != IterateControl::kIterate) {
        this->z_.Zeros();
        this->UpdateSolution(&this->V_[0], H, g, j, &this->z_);
        this->w_.Zeros();

        // pass on information object
        if (this->info_ != nullptr) {
          std::stringstream pre_str;
          pre_str << "precondend"
                  << std::setw(1 + std::log10(this->control().maxits()))
                  << std::setfill('0') << iter;
          this->info_->add(pre_str.str());
          this->precond_->SetInfo(this->info_->get_child(pre_str.str()));
        }
        this->ApplyPreconditioner(this->z_, &this->w_);

        if (this->filter_solution_) {
          this->w_.Update();
          this->non_lin_op_->ApplyFilter(this->w_);
        }
        x->Axpy(this->w_, static_cast< DataType >(1.));
        if (this->filter_solution_) {
          x->Update();
          assert (this->non_lin_op_ != nullptr);
          this->non_lin_op_->ApplyFilter(*x);
        }
        break;
      }
    } // for (int j = 0; j < this->size_basis(); ++j)

    // setup for restart
    if (conv == IterateControl::kIterate) {
      // -> update solution
      this->z_.Zeros();
      this->UpdateSolution(&this->V_[0], H, g, basis_size_actual - 1,
                           &this->z_);
      this->w_.Zeros();

      // pass on information object
      if (this->info_ != nullptr) {
        std::stringstream pre_str;
        pre_str << "precondrestart"
                << std::setw(1 + std::log10(this->control().maxits()))
                << std::setfill('0') << iter;
        this->info_->add(pre_str.str());
        this->precond_->SetInfo(this->info_->get_child(pre_str.str()));
      }
      this->ApplyPreconditioner(this->z_, &this->w_);

      if (this->filter_solution_) {
        this->w_.Update();
        assert (this->non_lin_op_ != nullptr);
        this->non_lin_op_->ApplyFilter(this->w_);
      }
      x->Axpy(this->w_, static_cast< DataType >(1.));
      if (this->filter_solution_) {
        x->Update();
        assert (this->non_lin_op_ != nullptr);
        this->non_lin_op_->ApplyFilter(*x);
      }
      // -> compute residual Ax-b
      this->op_->VectorMult(*x, this->V_[0]);
      this->V_[0]->ScaleAdd(b, static_cast< DataType >(-1.));
      if (this->filter_solution_) {
        this->V_[0]->Update();
        assert (this->non_lin_op_ != nullptr);
        this->non_lin_op_->ApplyFilter(*this->V_[0]);
      }
    }
  } // while (conv == IterateControl::kIterate)
  this->iter_ = iter;

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "final iterations   = " << this->iter_);
    LOG_INFO(this->name_, "final abs res norm = " << this->res_);
    LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_);
  }

  if (!this->reuse_basis_) {
    this->FreeBasis();
  }

  if (conv == IterateControl::kFailureDivergenceTol ||
      conv == IterateControl::kFailureMaxitsExceeded) {
    return kSolverExceeded;
  }
  return kSolverSuccess;
}

/// Updates solution: x = x + Vy with y solution of least squares problem.
/// @param V Krylov subspace basis
/// @param H Hessenberg matrix
/// @param g rhs of least squares problem
/// @param k iteration step
/// @param x solution vector

template < class LAD, class PreLAD >
void GMRES< LAD, PreLAD >::UpdateSolution(
    VectorType **V, const hiflow::la::SeqDenseMatrix< DataType > &H,
    const std::vector< DataType > &g, int k, VectorType *x) const {
  std::vector< DataType > y(g);

  // back substitution
  for (int i = k + 1; (i--) != 0;) {
    assert(H(i, i) != static_cast< DataType >(0.));
    y[i] /= H(i, i);

    const DataType temp = y[i];
    for (int j = 0; j < i; ++j) {
      y[j] -= H(i, j) * temp;
    }
  }

  // compute solution
  for (int j = 0; j <= k; ++j) {
    x->Axpy(*V[j], y[j]);
  }
}

/// @brief GMRES creator class
/// @author Tobias Hahn

template < class LAD > class GMREScreator : public LinearSolverCreator< LAD > {
public:
  LinearSolver< LAD > *params(const PropertyTree &c) {
    GMRES< LAD > *newGMRES = new GMRES< LAD >();
    if (c.contains("Method") && c.contains("SizeBasis")) {
      newGMRES->InitParameter(
          c["SizeBasis"].template get< int >(),
          c["Method"].template get< std::string >().c_str());
    }
    if (c.contains("MaxIterations") && c.contains("AbsTolerance") &&
        c.contains("RelTolerance") && c.contains("DivTolerance")) {
      newGMRES->InitControl(c["MaxIterations"].template get< int >(),
                            c["AbsTolerance"].template get< double >(),
                            c["RelTolerance"].template get< double >(),
                            c["DivTolerance"].template get< double >());
    }
    return newGMRES;
  }
};

template < class LAD, class PreLAD >
void setup_GMRES_solver(GMRES< LAD, PreLAD > &gmres_solver,
                        const PropertyTree &params,
                        NonlinearProblem< LAD > *nonlin) {
  const int max_it = params["MaxIt"].get< int >(1000);
  const int max_size = params["KrylovSize"].get< int >(100);
  const double abs_tol = params["AbsTol"].get< double >(1e-12);
  const double rel_tol = params["RelTol"].get< double >(1e-10);
  const bool use_press_filter = params["UsePressureFilter"].get< bool >(false);

  gmres_solver.InitControl(max_it, abs_tol, rel_tol, 1e6);
  if (params["UsePrecond"].get< bool >(true)) {
    gmres_solver.InitParameter(max_size, "RightPreconditioning");
  } else {
    gmres_solver.InitParameter(max_size, "NoPreconditioning");
  }
  gmres_solver.SetPrintLevel(params["PrintLevel"].get< int >(0));
  gmres_solver.SetReuse(params["Reuse"].get< bool >(true));
  gmres_solver.SetReuseBasis(params["ReuseBasis"].get< bool >(true));

  gmres_solver.SetName(params["Name"].get< std::string >("GMRES"));

  if (use_press_filter && nonlin != nullptr) {
    gmres_solver.SetupNonLinProblem(nonlin);
  }
}

template < class LAD >
void setup_GMRES_solver(GMRES< LAD > &gmres_solver, const PropertyTree &params,
                        NonlinearProblem< LAD > *nonlin) {
  setup_GMRES_solver< LAD, LAD >(gmres_solver, params, nonlin);
}

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_GMRES_H_
