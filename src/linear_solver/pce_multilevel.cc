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

#include <nonlinear/nonlinear_problem.h>
#include <time.h>
#include <iomanip>

#include "linear_algebra/pce_matrix.h"
#include "pce_multilevel.h"

namespace hiflow {
namespace la {

//typedef LADescriptorPolynomialChaosExpansionD LAD;

// constructor
template<class LAD>
PCE_MultiLevel<LAD>::PCE_MultiLevel() {

  this->smoother_solver_ = NULL;
  this->smoother_precond_ = NULL;
}

// destructor
template<class LAD>
PCE_MultiLevel<LAD>::~PCE_MultiLevel() {
  this->Clear();
}

// Clear
template<class LAD>
void PCE_MultiLevel<LAD>::Clear() {

  this->smoother_solver_ = NULL;
  this->smoother_precond_ = NULL;
}

// Initialize
template<class LAD>
void PCE_MultiLevel<LAD>::Init(const PCTensor& pctensor) {
  this->pctensor_ = pctensor;
}

// set operator
template<class LAD>
//void PCE_MultiLevel<LAD, LAD_M>::SetupOperator(const OperatorType& op) {
void PCE_MultiLevel<LAD>::SetupOperator(PCE_OperatorType& op) {
  this->op_ = &op;
}

// set smoothing paramters
template<class LAD>
void PCE_MultiLevel<LAD>::SetPrePostSmoothingNumber(const int nu1,
    const int nu2) {
  assert(nu1 > 0);
  assert(nu2 > 0);

  this->nu1_ = nu1;
  this->nu2_ = nu2;
}

template<class LAD>
void PCE_MultiLevel<LAD>::SetMu(const int mu) {
  assert(mu > 0 && mu <= 2);

  this->mu_ = mu;
}

// set smoother solver
template<class LAD>
void PCE_MultiLevel<LAD>::SetSmootherSolver(LinearSolver<LAD>&
    smoother_solver) {

  this->smoother_solver_ = &smoother_solver;
  //this->smoother_solver_->SetupPreconditioner(*this->smoother_precond_);
}

// set smoother preconditioner
template<class LAD>
void PCE_MultiLevel<LAD>::SetSmootherPrecond(LinearSolver<LAD>&
    smoother_precond) {

  this->smoother_precond_ = &smoother_precond;
}

// build
template<class LAD>
void PCE_MultiLevel<LAD>::Build() {

  assert(this->op_ != NULL);
  if(this->print_level_ > 2) {
    LOG_INFO("Build Solver", 1);
  }

  this->SetState(true);
  this->SetModifiedOperator(false);
}

// set precondtioner
//template<class LAD>
//void PCE_MultiLevel<LAD>::SetupPreconditioner(Preconditioner<LAD>& precond) {
//    this->precond_ = &precond;
//}

// SolveLevelProblem
template<class LAD>
void PCE_MultiLevel<LAD>::SolveLevelProblem(const PCE_VectorType& b,
    PCE_VectorType* x, const int l) {

  LOG_INFO("MultiLevel", "solving level = " << l);

  for(int mode = 0; mode != this->pctensor_.Size(l); ++mode) {

    LOG_INFO("MultiLevel", "solving level = " << l << " mode = " << mode);

    this->smoother_solver_->Solve(b.GetMode(mode), &(x->Mode(mode)));
  }

  x->Update();
}

// SolveMeanProblem
template<class LAD>
void PCE_MultiLevel<LAD>::SolveMeanProblem(const PCE_VectorType& b,
    PCE_VectorType* x) {

  LOG_INFO("MultiLevel Mean Solver", "start");

  this->smoother_solver_->InitControl(3, 1.0e-10, 1.0e-1, 1.0e6);
  SolveLevelProblem(b, x, 0);
  this->smoother_solver_->InitControl(1, 1.0e-10, 1.0e-1, 1.0e6);

  LOG_INFO("MultiLevel Mean Solver", "finish");
}

// Solve
template<class LAD>
LinearSolverState PCE_MultiLevel<LAD>::SolveImpl(const PCE_VectorType& b,
    PCE_VectorType* x) {

  if(this->mu_ == 1) {
    LOG_INFO("MultiLevel Solver", "V-cylce ----------");
  } else if(this->mu_ == 2) {
    LOG_INFO("MultiLevel Solver", "W-cycle ----------");
  } else {
    exit(-1);
  }

  this->iter_ = 0;
  IterateControl::State conv = IterateControl::kIterate;

  PCE_VectorType res;
  res.CloneFrom(*x);
  res.Zeros();

  this->op_->VectorMult(*x, &res);
  res.ScaleAdd(b, -1.0);

  this->res_ = res.Norm2();
  conv = this->control().Check(this->iter_, this->res());

  LOG_INFO("MultiLevel", "starts with residual " << this->res_);

  while(conv == IterateControl::kIterate) {
    ++this->iter_;

    ML(b, x, this->pctensor_.GetLevel());

    res.Zeros();

    this->op_->VectorMult(*x, &res);
    res.ScaleAdd(b, -1.0);

    this->res_ = res.Norm2();
    conv = this->control().Check(this->iter_, this->res());

    if(conv != IterateControl::kIterate) {
      break;
    }

    LOG_INFO("MultiLevel residual",
             this->res_ << " at iteration : " << this->iter_);

  }

  LOG_INFO("MultiLevel", "ended after " << this->iter_ <<
           " iterations, with residual : " << this->res_);

  if(conv == IterateControl::kFailureDivergenceTol ||
      conv == IterateControl::kFailureMaxitsExceeded) {
    return kSolverExceeded;
  } else {
    return kSolverSuccess;
  }

}

// ApplyPreconditioner
template<class LAD>
LinearSolverState PCE_MultiLevel<LAD>::ApplyPreconditioner(
  const PCE_VectorType& b, PCE_VectorType* x) {
  assert(this->mu_ > 0 && this->mu_ <= 2);
  assert(this->nu1_ > 0);
  assert(this->nu2_ > 0);

  if(this->mu_ == 1) {
    LOG_INFO("MultiLevel Preconditioner", "V-cylce");
  } else if(this->mu_ == 2) {
    LOG_INFO("MultiLevel Preconditioner", "W-cycle");
  } else {
    exit(-1);
  }

  ML(b, x, this->pctensor_.GetLevel());

  LOG_INFO("MultiLevel Preconditioner", "preconditioner completed");

  return la::kSolverSuccess;

}

// ML
template<class LAD>
void PCE_MultiLevel<LAD>::ML(const PCE_VectorType& b, PCE_VectorType* x,
                             const int l) {
  assert(this->mu_ > 0 && this->mu_ <= 2);
  assert(this->nu1_ > 0);
  assert(this->nu2_ > 0);

  if(l == 0) {
    SolveMeanProblem(b, x);
  } else {

    // 1. apply pre-smoothing nu1 times on fine grid
    LOG_INFO("MultiLevel", "pre-smoothing");
    Smoothing(b, x, this->nu1_, l);

    // 2. compute residual on fine grid
    PCE_VectorType res_fine;
    res_fine.CloneFrom(*x);
    res_fine.Zeros();

    this->op_->VectorMult(*x, &res_fine, l);
    res_fine.ScaleAdd(b, -1.0);

    // 3. restric vector to coarse grid
    x->Update();

    PCE_VectorType x_coarse, c_coarse, res_coarse;

    int tmp_l = l;
    if(l == 3) {
      tmp_l = tmp_l - 1;
    }

    // create coarse vector
    x_coarse.CloneFrom(*x, tmp_l-1);
    res_coarse.CloneFrom(res_fine, tmp_l-1);

    c_coarse.CloneFrom(*x, tmp_l-1);
    c_coarse.Zeros();

    // 4. correction on coarse grid
    for(int i = 0; i != this->mu_; ++i) {
      ML(res_coarse, &c_coarse, tmp_l-1);
    }

    // 5. apply correction
    c_coarse.Update();
    x_coarse.Axpy(c_coarse, 1.0);

    tmp_l = l;
    if(l == 3) {
      tmp_l = tmp_l - 1;
    }

    // 6. prolongation
    x->CopyFrom(x_coarse, tmp_l-1);

    // 7. apply post-smoothing nu2 times on fine grid
    LOG_INFO("Multilevel", "post-smoothing");
    Smoothing(b, x, this->nu2_, l);

  }

}

// smoothing
template<class LAD>
void PCE_MultiLevel<LAD>::Smoothing(const PCE_VectorType& b, PCE_VectorType* x,
                                    const int nu, const int l) {
  assert(nu > 0);
  assert(l >= 0);
  assert(b.nb_mode() == x->nb_mode());
  assert(b.nb_mode() == this->pctensor_.Size(l));

  LOG_INFO("Multilevel", "Smoothing starts with level = " << l);

  PCE_VectorType res_fine, cor;
  res_fine.CloneFromWithoutContent(*x);
  res_fine.Zeros();
  cor.CloneFrom(res_fine);

  this->op_->VectorMult(*x, &res_fine, l);
  res_fine.ScaleAdd(b, -1.0);

  for(int i = 0; i != nu; ++i) {

    // set cor to zeros
    cor.Zeros();

    // get correction for each mode on level l
    SolveLevelProblem(res_fine, &cor, l);

    // set back to x
    x->Axpy(cor, 1.0);

    if(i != nu-1) {
      res_fine.Zeros();
      this->op_->VectorMult(*x, &res_fine);
      res_fine.ScaleAdd(b, -1.0);
    }
  }

}

// ExtractModeVector
template<class LAD>
void PCE_MultiLevel<LAD>::ExtractLevelVector(const PCE_VectorType& x_fine,
    PCE_VectorType* x_coarse, const int l) {
  assert(x_fine.nb_mode() > 0);
  assert(x_coarse->nb_mode() > 0);
  assert(x_fine.nb_mode() >= x_coarse->nb_mode());

  for(int mode = 0; mode != this->pctensor_.Size(l); ++mode) {
    x_coarse->CloneFrom(mode, x_fine.GetMode(mode));
  }
}

// AssignLevelVectorToGlobal
template<class LAD>
void PCE_MultiLevel<LAD>::AssignLevelVectorToGlobal(
  const PCE_VectorType& x_coarse, PCE_VectorType* x_fine, const int l) {
  assert(x_fine->nb_mode() > 0);
  assert(x_coarse.nb_mode() > 0);
  assert(x_fine->nb_mode() >= x_coarse.nb_mode());

  for(int mode = 0; mode != this->pctensor_.Size(l); ++mode) {
    x_fine->CloneFrom(mode, x_coarse.GetMode(mode));
  }
}

// template instantiation
template class PCE_MultiLevel< LADescriptorCoupledD >;
template class PCE_MultiLevel< LADescriptorCoupledS >;

#ifdef WITH_HYPRE
template class PCE_MultiLevel< LADescriptorHypreD >;
#endif


}
}
