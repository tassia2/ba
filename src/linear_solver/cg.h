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

/// @author Chandramowli Subramanian

#ifndef HIFLOW_LINEARSOLVER_CG_H_
#define HIFLOW_LINEARSOLVER_CG_H_

#include "linear_solver/linear_solver.h"
#include "linear_solver/linear_solver_creator.h"
#include <string>
#include <cassert>
#include <cmath>

#include "common/log.h"
#include "linear_algebra/block_matrix.h"
#include "linear_algebra/la_descriptor.h"
#include <iomanip>

namespace hiflow {
namespace la {

/// @brief CG solver
///
/// CG solver for symmetric linear systems Ax=b.

template < class LAD > class CG : public LinearSolver< LAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  CG() {
    this->SetMethod("NoPreconditioning");
    if (this->print_level_ > 2) {
      LOG_INFO("Linear solver", "CG");
      LOG_INFO("Preconditioning", this->Method());
    }
    this->name_ = "CG";
  }
  virtual ~CG() {}

  void InitParameter(std::string const &method) {
  // chose method_
  this->precond_method_ = method;
  assert((this->Method() == "NoPreconditioning") ||
         (this->Method() == "Preconditioning") ||
	 (this->Method() == "RightPreconditioning") ||
	 (this->Method() == "LeftPreconditioning"));
}

protected:
  LinearSolverState SolveImpl(const VectorType &b, VectorType *x) {
  assert(x->is_initialized());
  assert(b.is_initialized());

  LinearSolverState state;

  if (this->Method() == "NoPreconditioning") {
    state = this->SolveNoPrecond(b, x);
  } else if (this->Method() == "Preconditioning") {
    state = this->SolvePrecond(b, x);
  } else if (this->Method() == "RightPreconditioning") {
    state = this->SolvePrecond(b, x);
  } else if (this->Method() == "LeftPreconditioning") {
    state = this->SolvePrecond(b, x);
  } else {
    state = kSolverError;
  }

  return state;
}

private:
  LinearSolverState SolveNoPrecond(const VectorType &b, VectorType *x) {
  assert(this->Method() == "NoPreconditioning");
  assert(this->op_ != nullptr);

  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, "solve without preconditioning");
  }

  IterateControl::State conv = IterateControl::kIterate;

  // needed vectors
  VectorType r, p, Ap;
  r.CloneFromWithoutContent(b);
  p.CloneFromWithoutContent(b);
  Ap.CloneFromWithoutContent(b);

  // needed values
  DataType alpha, beta;

  // initialization step
  this->iter_ = 0;

  this->op_->VectorMult(*x, &r);

  r.ScaleAdd(b, static_cast< DataType >(-1.));

  DataType ressquared = r.Dot(r);
  this->res_ = std::sqrt(ressquared);
  this->res_init_ = this->res_;
  this->res_rel_ = 1.;
  conv = this->control().Check(this->iter_, this->res_);

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "initial res norm   =  " << this->res_);
  }

  p.CopyFrom(r);
  beta = ressquared;

  // main loop
  while (conv == IterateControl::kIterate) {
    ++(this->iter_);
    this->op_->VectorMult(p, &Ap);
    alpha = beta / (Ap.Dot(p));
    x->Axpy(p, alpha);
    r.Axpy(Ap, -alpha);

    ressquared = r.Dot(r);
    this->res_ = sqrt(ressquared);
    this->res_rel_ = this->res_ / this->res_init_;
    if (this->print_level_ > 2) {
      LOG_INFO(this->name_,
               "residual (iteration " << this->iter_ << "): " << this->res_);
    }

    conv = this->control().Check(this->iter_, this->res_);
    if (conv != IterateControl::kIterate) {
      break;
    }

    beta = ressquared / beta;
    p.ScaleAdd(r, beta);
    beta = ressquared;
  }

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "final iterations   = " << this->iter_);
    LOG_INFO(this->name_, "final abs res norm = " << this->res_);
    LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_)
  } 

  if (conv == IterateControl::kFailureDivergenceTol ||
      conv == IterateControl::kFailureMaxitsExceeded) {
    return kSolverExceeded;
  }
  return kSolverSuccess;
}

  LinearSolverState SolvePrecond(const VectorType &b, VectorType *x) {
  assert(this->Method() != "NoPreconditioning");
  assert(this->op_ != nullptr);
  assert(this->precond_ != nullptr);

  if (this->print_level_ > 2) {
    LOG_INFO(this->name_, "solve with right preconditioning");
  }

  IterateControl::State conv = IterateControl::kIterate;

  // needed vectors
  VectorType r, p, z, Ap;
  r.CloneFromWithoutContent(b);
  p.CloneFromWithoutContent(b);
  z.CloneFromWithoutContent(b);
  Ap.CloneFromWithoutContent(b);

  // needed values
  DataType alpha, beta, gamma, ressquared;

  // initialization step
  this->iter_ = 0;

  this->op_->VectorMult(*x, &r);
  r.ScaleAdd(b, static_cast< DataType >(-1.));
  ressquared = r.Dot(r);
  std::cout << "ressquared " << ressquared << std::endl;
  this->res_init_ = this->res_ = sqrt(ressquared);
  this->res_rel_ = 1.;
  conv = this->control().Check(this->iter_, this->res_);
  z.Zeros();
  this->ApplyPreconditioner(r, &z);
  p.CopyFrom(z);

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "initial res norm   = " << this->res_);
  }

  beta = r.Dot(z);

  // main loop
  while (conv == IterateControl::kIterate) {
    ++(this->iter_);
    this->op_->VectorMult(p, &Ap);
    alpha = beta / (Ap.Dot(p));
    x->Axpy(p, alpha);
    r.Axpy(Ap, -alpha);

    ressquared = r.Dot(r);
    this->res_ = sqrt(ressquared);
    this->res_rel_ = this->res_ / this->res_init_;
    if (this->print_level_ > 2) {
      LOG_INFO(this->name_,
               "residual (iteration " << this->iter_ << "): " << this->res_);
    }

    conv = this->control().Check(this->iter_, this->res_);
    if (conv != IterateControl::kIterate) {
      break;
    }

    z.Zeros();
    this->ApplyPreconditioner(r, &z);
    gamma = r.Dot(z);
    beta = gamma / beta;
    p.ScaleAdd(z, beta);
    beta = gamma;
  }

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "final iterations   = " << this->iter_);
    LOG_INFO(this->name_, "final abs res norm = " << this->res_);
    LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_);
  } 

  if (conv == IterateControl::kFailureDivergenceTol ||
      conv == IterateControl::kFailureMaxitsExceeded) {
    return kSolverExceeded;
  }
  return kSolverSuccess;
}
};

/// @brief CG creator class
/// @author Tobias Hahn

template < class LAD > class CGcreator : public LinearSolverCreator< LAD > {
public:
  LinearSolver< LAD > *params(const PropertyTree &c) {
    CG< LAD > *newCG = new CG< LAD >();
    if (c.contains("Method")) {
      newCG->InitParameter(c["Method"].template get< std::string >().c_str());
    }
    if (c.contains("MaxIterations") && c.contains("AbsTolerance") &&
        c.contains("RelTolerance") && c.contains("DivTolerance")) {
      newCG->InitControl(c["MaxIterations"].template get< int >(),
                         c["AbsTolerance"].template get< double >(),
                         c["RelTolerance"].template get< double >(),
                         c["DivTolerance"].template get< double >());
    }
    return newCG;
  }
};

template < class LAD >
void setup_CG_solver(CG< LAD > &cg_solver, const PropertyTree &params,
                     NonlinearProblem< LAD > *nonlin) {
  const int max_it = params["MaxIt"].get< int >(1000);
  const double abs_tol = params["AbsTol"].get< double >(1e-12);
  const double rel_tol = params["RelTol"].get< double >(1e-10);
  const bool use_press_filter = params["UsePressureFilter"].get< bool >(false);

  cg_solver.InitControl(max_it, abs_tol, rel_tol, 1e6);
  if (params["UsePrecond"].get< bool >(true)) {
    cg_solver.InitParameter("Preconditioning");
  } else {
    cg_solver.InitParameter("NoPreconditioning");
  }
  cg_solver.SetPrintLevel(params["PrintLevel"].get< int >(0));
  cg_solver.SetReuse(params["Reuse"].get< bool >(true));
  ;

  if (use_press_filter) {
    cg_solver.SetupNonLinProblem(nonlin);
  }
  
  cg_solver.SetName(params["Name"].get< std::string >("CG"));
}

} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_CG_H_
