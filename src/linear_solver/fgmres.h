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

#ifndef HIFLOW_LINEARSOLVER_FGMRES_H_
#define HIFLOW_LINEARSOLVER_FGMRES_H_

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <string>
#include <vector>

#include "common/log.h"
#include "linear_algebra/pce_matrix.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "linear_solver/gmres.h"
#include "linear_solver/linear_solver.h"

namespace hiflow {
namespace la {

template < class DataType > class SeqDenseMatrix;

/// @brief Flexible GMRES solver
///
/// Flexible GMRES solver for linear systems Ax=b.

template < class LAD, class PreLAD = LAD >
class FGMRES : public GMRES< LAD, PreLAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  FGMRES();
  virtual ~FGMRES();

  virtual void Clear() {
    LinearSolver< LAD, PreLAD >::Clear();
    this->name_ = "FGMRES";
  }

  /// Initialize pointers for Krylov Basis
  virtual void InitBasis(const VectorType &ref_vec, int iteration);

  /// Allocate Krylov basis vectors
  virtual void AllocateBasis(int basis_size);

  /// Set all basis vectors to zero
  virtual void SetBasisToZero();

  /// Deallocate Krylov basis vectors
  virtual void FreeBasis();

private:
  LinearSolverState SolveLeft(const VectorType &b, VectorType *x);
  LinearSolverState SolveRight(const VectorType &b, VectorType *x);

  std::vector< VectorType * > Z_;
};

/// standard constructor

template < class LAD, class PreLAD >
FGMRES< LAD, PreLAD >::FGMRES() : GMRES< LAD, PreLAD >() {
  this->size_basis_ = 0;
  this->name_ = "FGMRES";
  this->SetMethod("RightPreconditioning");
  if (this->print_level_ > 2) {
    LOG_INFO("Linear solver", "FGMRES");
  }
}

/// destructor

template < class LAD, class PreLAD > FGMRES< LAD, PreLAD >::~FGMRES() {
  this->op_ = nullptr;
  this->precond_ = nullptr;

  this->Clear();
}

/// Solve with left preconditioning.

template < class LAD, class PreLAD >
LinearSolverState FGMRES< LAD, PreLAD >::SolveLeft(const VectorType &b,
                                                   VectorType *x) {
  assert(this->Method() == "LeftPreconditioning");

  assert(this->op_ != nullptr);
  assert(this->precond_ != nullptr);
  assert(this->size_basis() > 0);

  LOG_ERROR("FGMRES::SolveLeft: Not implemented yet.");
  LOG_ERROR("Returning solver error...");
  return kSolverError;
}

/// Solve with right preconditioning.

template < class LAD, class PreLAD >
LinearSolverState FGMRES< LAD, PreLAD >::SolveRight(const VectorType &b,
                                                    VectorType *x) {
  assert(this->Method() == "RightPreconditioning");

  assert(this->op_ != nullptr);
  assert(this->precond_ != nullptr);
  assert(this->size_basis() > 0);

  if (this->print_level_ > 2) 
  {
    LOG_INFO(this->name_, "solve with right preconditioning");
  }

  IterateControl::State conv = IterateControl::kIterate;

  // compute really used basis size as minimum of maximum iterations and
  // given basis size
  const size_t basis_size_actual =
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
  // this->V_[0]->CloneFromWithoutContent ( b );
  this->op_->VectorMult(*x, this->V_[0]);
  this->V_[0]->ScaleAdd(b, static_cast< DataType >(-1.));

  if (this->filter_solution_) 
  {
    assert (this->non_lin_op_ != nullptr);
    this->V_[0]->Update();
    this->non_lin_op_->ApplyFilter(*this->V_[0]);
    if (this->print_level_ > 2) 
    {
      LOG_INFO(this->name_, "filter V ");
    }
  }

  this->res_init_ = this->res_ = this->V_[0]->Norm2();
  this->res_rel_ = 1.;
  conv = this->control().Check(iter, this->res_);

  if (this->print_level_ > 1) 
  {
    LOG_INFO(this->name_, "initial res norm   = " << this->res_);
  }

  // main loop
  while (conv == IterateControl::kIterate) 
  {
    g.assign(g.size(), static_cast< DataType >(0.)); // g = 0
    H.Zeros();

    assert(this->res_ != static_cast< DataType >(0.));
    this->V_[0]->Scale(static_cast< DataType >(1.) /
                       this->res_); // norm residual
    g[0] = this->res_;

    for (size_t j = 0; j != basis_size_actual; ++j) 
    {
      ++iter;

      // pass on information object
      if (this->info_ != nullptr) {
        std::stringstream pre_str;
        pre_str << "precond"
                << std::setw(1 + std::log10(this->control().maxits()))
                << std::setfill('0') << iter;
        this->info_->add(pre_str.str());
        this->precond_->SetInfo(this->info_->get_child(pre_str.str()));
      }

      // apply preconditioner: this->Z_[j] ~= M^-1 v_j
      this->ApplyPreconditioner(*this->V_[j], this->Z_[j]);
      
      // apply filter
      if (this->filter_solution_) 
      {
        assert (this->non_lin_op_ != nullptr);
        this->Z_[j]->Update();
        this->non_lin_op_->ApplyFilter(*this->Z_[j]);
        if (this->print_level_ > 2) 
        {
          LOG_INFO(this->name_, " filter Z ");
        }
      }
      
      // w = A this->Z_[j]
      this->op_->VectorMult(*this->Z_[j], &this->w_); 
      if (this->filter_solution_) 
      {
        assert (this->non_lin_op_ != nullptr);
        this->w_.Update();
        this->non_lin_op_->ApplyFilter(this->w_);
        if (this->print_level_ > 2) 
        {
          LOG_INFO(this->name_, " filter W ");
        }
      }
      
      // -- start building Hessenberg matrix H --
      // vectors in V are ONB of Krylov subspace K_i(A,this->V_[0])
      for (size_t i = 0; i <= j; ++i) 
      {
        H(j, i) = this->w_.Dot(*this->V_[i]);
        this->w_.Axpy(*this->V_[i], static_cast< DataType >(-1.) * H(j, i));
      }

      H(j, j + 1) = this->w_.Norm2();
      assert(H(j, j + 1) != static_cast< DataType >(0.));

      this->w_.Scale(static_cast< DataType >(1.) / H(j, j + 1));

      // initialize next basis vector
      this->InitBasis(b, j + 1);
      this->V_[j + 1]->CopyFrom(this->w_);

      // -- end building Hessenberg matrix H --

      // apply old Givens rotation on old H entries
      for (size_t k = 0; k < j; ++k) {
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

      if (conv != IterateControl::kIterate) 
      {
        this->UpdateSolution(&this->Z_[0], H, g, j, x); // x = x + Zy
        if (this->filter_solution_) 
        {
          assert (this->non_lin_op_ != nullptr);
          x->Update();
          this->non_lin_op_->ApplyFilter(*x);
        }
        break;
      }
    } // for (int j = 0; j < this->size_basis(); ++j)

    // setup for restart
    if (conv == IterateControl::kIterate) 
    {
      this->UpdateSolution(&this->Z_[0], H, g, basis_size_actual - 1,
                           x); // x = x + Zy
      if (this->filter_solution_) 
      {
        assert (this->non_lin_op_ != nullptr);
        x->Update();
        this->non_lin_op_->ApplyFilter(*x);
      }

      this->op_->VectorMult(*x, this->V_[0]);
      this->V_[0]->Scale(static_cast< DataType >(-1.));
      this->V_[0]->Axpy(b, static_cast< DataType >(1.));
      if (this->filter_solution_) 
      {
        assert (this->non_lin_op_ != nullptr);
        this->V_[0]->Update();
        this->non_lin_op_->ApplyFilter(*this->V_[0]);
      }
      this->w_.Zeros();
    }
  } // while (conv == IterateControl::kIterate

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

template < class LAD, class PreLAD >
void FGMRES< LAD, PreLAD >::AllocateBasis(int basis_size) {
  assert(this->V_.size() == this->Z_.size());
  int diff = basis_size + 1 - this->V_.size();

  for (int i = 0; i < diff; ++i) {
    this->V_.push_back(new VectorType);
    this->Z_.push_back(new VectorType);
  }
}

template < class LAD, class PreLAD >
void FGMRES< LAD, PreLAD >::InitBasis(const typename LAD::VectorType &ref_vec,
                                      int iteration) {
  assert(ref_vec.is_initialized());
  if (iteration == 0) {
    if (!this->aux_vec_init_) {
      this->w_.CloneFromWithoutContent(ref_vec);
      this->aux_vec_init_ = true;
    }
  }

  assert(this->V_.size() == this->Z_.size());
  assert(this->V_.size() >= iteration + 1);

  int num_init = this->num_init_basis_;
  for (int i = num_init; i <= iteration; ++i) {
    this->V_[i]->CloneFromWithoutContent(ref_vec);
    this->Z_[i]->CloneFromWithoutContent(ref_vec);
    this->num_init_basis_ = i + 1;
  }
}

template < class LAD, class PreLAD >
void FGMRES< LAD, PreLAD >::SetBasisToZero() {
  if (this->aux_vec_init_) {
    this->w_.Zeros();
  }
  for (int i = 0; i < this->num_init_basis_; ++i) {
    this->V_[i]->Zeros();
    this->Z_[i]->Zeros();
  }
}

template < class LAD, class PreLAD > void FGMRES< LAD, PreLAD >::FreeBasis() {
  this->w_.Clear();
  assert(this->V_.size() == this->Z_.size());

  for (size_t i = 0; i < this->V_.size(); ++i) {
    this->V_[i]->Clear();
    delete this->V_[i];

    this->Z_[i]->Clear();
    delete this->Z_[i];
  }
  this->V_.clear();
  this->Z_.clear();
  this->num_init_basis_ = 0;
  this->aux_vec_init_ = false;
}

template < class LAD, class PreLAD >
void setup_FGMRES_solver(FGMRES< LAD, PreLAD > &fgmres_solver,
                         const PropertyTree &params,
                         NonlinearProblem< LAD > *nonlin) 
{
  const int max_it = params["MaxIt"].get< int >(1000);
  const int max_size = params["KrylovSize"].get< int >(100);
  const double abs_tol = params["AbsTol"].get< double >(1e-12);
  const double rel_tol = params["RelTol"].get< double >(1e-10);
  const bool use_press_filter = params["UsePressureFilter"].get< bool >(false);

  fgmres_solver.InitControl(max_it, abs_tol, rel_tol, 1e6);
  if (params["UsePrecond"].get< bool >(true)) 
  {
    fgmres_solver.InitParameter(max_size, "RightPreconditioning");
  } 
  else 
  {
    fgmres_solver.InitParameter(max_size, "NoPreconditioning");
  }
  fgmres_solver.SetPrintLevel(params["PrintLevel"].get< int >(0));
  fgmres_solver.SetReuse(params["Reuse"].get< bool >(true));
  fgmres_solver.SetReuseBasis(params["ReuseBasis"].get< bool >(true));

  if (use_press_filter && nonlin != nullptr) 
  {
    fgmres_solver.SetupNonLinProblem(nonlin);
  }
  fgmres_solver.SetName(params["Name"].get< std::string >("FGMRES"));
}

template < class LAD >
void setup_FGMRES_solver(FGMRES< LAD > &fgmres_solver,
                         const PropertyTree &params,
                         NonlinearProblem< LAD > *nonlin) {
  setup_FGMRES_solver< LAD, LAD >(fgmres_solver, params, nonlin);
}

/// @brief GMRES creator class
/// @author Tobias Hahn

template < class LAD > class FGMREScreator : public LinearSolverCreator< LAD > {
public:
  LinearSolver< LAD > *params(const PropertyTree &c) {
    FGMRES< LAD > *newFGMRES = new FGMRES< LAD >();
    if (c.contains("Method") && c.contains("SizeBasis")) {
      newFGMRES->InitParameter(
          c["SizeBasis"].template get< int >(),
          c["Method"].template get< std::string >().c_str());
    }
    if (c.contains("MaxIterations") && c.contains("AbsTolerance") &&
        c.contains("RelTolerance") && c.contains("DivTolerance")) {
      newFGMRES->InitControl(c["MaxIterations"].template get< int >(),
                            c["AbsTolerance"].template get< double >(),
                            c["RelTolerance"].template get< double >(),
                            c["DivTolerance"].template get< double >());
    }
    return newFGMRES;
  }
};

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_FGMRES_H_
