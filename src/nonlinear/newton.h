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

#ifndef HIFLOW_NONLINEAR_NEWTON_H_
#define HIFLOW_NONLINEAR_NEWTON_H_

#include "common/log.h"
#include "common/macros.h"
#include "common/property_tree.h"
#include "common/timer.h"
#include "linear_algebra/la_descriptor.h"
#include "nonlinear/nonlinear_solver.h"
#include "nonlinear/nonlinear_solver_creator.h"
#include "nonlinear/damping_strategy.h"
#include "nonlinear/forcing_strategy.h"
#include "nonlinear/nonlinear_problem.h"
#include "space/vector_space.h"
#include "space/fe_evaluation.h"
#include "space/space_tools.h"
#include "visualization/cell_visualization.h"
#include "visualization/vtk_writer.h"

#include <cassert>
#include <iomanip>
#include <vector>

namespace hiflow {

template < class LAD, int DIM > class DampingStrategy;
template < class LAD > class ForcingStrategy;

/// @brief Newton nonlinear solver
/// @author Tobias Hahn, Michael Schick
///
/// Newton solver for generic nonlinear problems.
/// Requires allocated space for jacobian and residual.
/// May use forcing, damping and custom initial solution, but does not
/// by default.

template < class LAD, int DIM > class Newton : public NonlinearSolver< LAD, DIM > {
public:
  typedef typename LAD::MatrixType MatrixType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  enum NonlinearSolverParameter {
    NewtonInitialSolutionOwn,
    NewtonInitialSolution0,
    NewtonDampingStrategyNone,
    NewtonDampingStrategyOwn,
    NewtonForcingStrategyConstant,
    NewtonForcingStrategyOwn
  };

  Newton();
  Newton(VectorType *residual, MatrixType *matrix);
  ~Newton();

  void InitParameter(VectorType *residual, MatrixType *matrix);
  NonlinearSolverState InitParameter(NonlinearSolverParameter param);
  void SetDampingStrategy(DampingStrategy< LAD, DIM > &dampstrat);
  void SetForcingStrategy(ForcingStrategy< LAD > &forcingstrat);
  bool Forcing() const;

  void GetForcingTerm(DataType &forcing) const;
  void SetForcingTerm(DataType forcing);

  void ActivateNonConstMode() { non_const_mode_ = true; }

  bool GetNonConstMode() { return non_const_mode_; }

  NonlinearSolverState Solve(const VectorType &y, VectorType *x,
                             VectorSpace< DataType, DIM > const *space = nullptr);
  NonlinearSolverState Solve(VectorType *x,
                             VectorSpace< DataType, DIM > const *space = nullptr);

  void ComputeJacobian(const VectorType &x, MatrixType *jacobian);
  void ComputeJacobianNonConst(VectorType &x, MatrixType *jacobian);
  la::LinearSolverState SolveJacobian(const MatrixType &jacobian,
                                      const VectorType &residual,
                                      VectorType *correction);

  NonlinearSolverState UpdateSolution(const VectorType &cor, VectorType *sol,
                                      VectorSpace< DataType, DIM > const *space);
  NonlinearSolverState UpdateSolution(const VectorType &cor,
                                      const VectorType &rhs, VectorType *sol,
                                      VectorSpace< DataType, DIM > const *space);

  void ComputeResidual(const VectorType &sol, VectorType *res);
  void ComputeResidual(const VectorType &sol, const VectorType &rhs,
                       VectorType *res);
  void ComputeResidualNonConst(VectorType &sol, const VectorType &rhs,
                               VectorType *res);

  /// \brief Set flag whether computed solution should be returned of caller of
  /// Solve(), in case of failure of Newton's method. <br> This might be
  /// necessary when working with a fixed number of iterations.
  void SetForceReturnOfSol(bool flag) { this->force_return_sol_ = flag; }

  void SetStatisticsFilename(std::string filename) {
    this->filename_statistics_ = filename;
  }

  void SetVisuFilename(std::string filename) {
    this->filename_visu_ = filename;
  }
  
  void WriteStatistics(DataType asm_time, DataType solve_time,
                       DataType update_time, DataType lin_solver_rhs_norm);

  void GetStatisticsTiming(DataType &asm_time, DataType &solve_time,
                           DataType &update_time, bool erase = false) {
    asm_time = this->asm_time_;
    solve_time = this->solve_time_;
    update_time = this->update_time_;

    if (erase) {
      this->asm_time_ = 0.;
      this->solve_time_ = 0.;
      this->update_time_ = 0.;
    }
  }

  void visualize_solution (VectorType &sol, 
                           const VectorSpace< DataType, DIM > *space,
                           std::string const &prefix, 
                           int iter) const;
protected:
  VectorType *res_;
  MatrixType *jac_;
  DampingStrategy< LAD, DIM > *DampStratObject_;
  ForcingStrategy< LAD > *ForcingStratObject_;
  std::vector< DataType > resids_;

  NonlinearSolverParameter InitialSolution_;
  NonlinearSolverParameter DampingStrategy_;
  NonlinearSolverParameter ForcingStrategy_;

  bool non_const_mode_;
  std::string filename_statistics_;
  std::string filename_visu_;
  
  bool force_return_sol_;

  DataType asm_time_;
  DataType solve_time_;
  DataType update_time_;
};

/// @brief Newton creator class
/// @author Tobias Hahn

template < class LAD, int DIM >
class Newtoncreator : public NonlinearSolverCreator< LAD, DIM > {
public:
  typedef typename LAD::MatrixType MatrixType;
  typedef typename LAD::VectorType VectorType;

  NonlinearSolver< LAD, DIM > *params(VectorType *residual, MatrixType *matrix,
                                 const PropertyTree &c) {
    Newton< LAD, DIM > *newNewton = new Newton< LAD, DIM >(residual, matrix);
    if (c.contains("MaxIterations") && c.contains("AbsTolerance") &&
        c.contains("RelTolerance") && c.contains("DivTolerance")) {
      newNewton->InitControl(c["MaxIterations"].template get< int >(),
                             c["AbsTolerance"].template get< double >(),
                             c["RelTolerance"].template get< double >(),
                             c["DivTolerance"].template get< double >());
    }
    return newNewton;
  }

  NonlinearSolver< LAD, DIM > *params(const PropertyTree &c) {
    return new Newton< LAD, DIM >();
  }
};

template < class LAD, int DIM >
NonlinearSolverState
Newton< LAD, DIM >::InitParameter(NonlinearSolverParameter param) 
{
  if (param == NewtonDampingStrategyNone || param == NewtonDampingStrategyOwn) 
  {
    this->DampingStrategy_ = param;
  } 
  else if (param == NewtonForcingStrategyConstant ||
           param == NewtonForcingStrategyOwn) 
  {
    this->ForcingStrategy_ = param;
  } 
  else if (param == NewtonInitialSolution0 ||
           param == NewtonInitialSolutionOwn) 
  {
    this->InitialSolution_ = param;
  } 
  else 
  {
    return kNonlinearSolverInitError;
  }

  return kNonlinearSolverSuccess;
}

template < class LAD, int DIM >
void Newton< LAD, DIM >::InitParameter(VectorType *residual, 
                                       MatrixType *matrix) 
{
  res_ = residual;
  jac_ = matrix;
  assert(res_ != nullptr);
  assert(jac_ != nullptr);
}

/// Sets up the damping strategy
/// @param dampstrat DampingStrategy

template < class LAD, int DIM >
void Newton< LAD, DIM >::SetDampingStrategy(DampingStrategy< LAD, DIM > &dampstrat) 
{
  this->DampStratObject_ = &dampstrat;
  this->DampingStrategy_ = NewtonDampingStrategyOwn;
}

/// Sets up the forcing strategy
/// @param forcingstrat ForcingStrategy

template < class LAD, int DIM >
void Newton< LAD, DIM >::SetForcingStrategy(ForcingStrategy< LAD > &forcingstrat) 
{
  this->ForcingStratObject_ = &forcingstrat;
  this->ForcingStrategy_ = NewtonForcingStrategyOwn;
}

/// Get the current forcing term
/// returns zero if no forcing is activated

template < class LAD, int DIM >
void Newton< LAD, DIM >::GetForcingTerm(DataType &forcing) const 
{
  if (this->ForcingStrategy_ == NewtonForcingStrategyOwn) 
  {
    forcing = this->ForcingStratObject_->GetCurrentForcingTerm();
  } 
  else 
  {
    forcing = 0.;
  }
}

/// Set a forcing term, necessary in combination with damping
/// only if forcing is activated
/// @param forcing DataType

template < class LAD, int DIM > 
void Newton< LAD, DIM >::SetForcingTerm(DataType forcing) 
{
  if (this->ForcingStrategy_ == NewtonForcingStrategyOwn) 
  {
    this->ForcingStratObject_->SetForcingTerm(forcing);
  }
}

/// Provides information is forcing is used
/// necessary for damping

template < class LAD, int DIM > 
bool Newton< LAD, DIM >::Forcing() const 
{
  return (this->ForcingStrategy_ == NewtonForcingStrategyOwn);
}

/// Updates solution vector using correction and possible damping/
/// forcing strategies that use in turn a right-hand side
/// @param cor correction vector
/// @param rhs right-hand side vector
/// @param sol solution vector

template < class LAD, int DIM >
NonlinearSolverState Newton< LAD, DIM >::UpdateSolution(const VectorType &cor, 
                                                        const VectorType &rhs,
                                                        VectorType *sol,
                                                        VectorSpace< DataType, DIM > const *space) 
{
  if (this->DampingStrategy_ == NewtonDampingStrategyNone) 
  {
    assert(sol->size_local() == cor.size_local());
    assert(sol->size_global() == cor.size_global());
    sol->Axpy(cor, static_cast< DataType >(-1.));
    sol->Update();

    if (this->print_level_ >= 2)
    {
      DataType cor_norm = cor.Norm2();
      LOG_INFO("Correction Norm", cor_norm);
    }
    // space provided -> interpolate possible hanging dofs
    if (space != nullptr) 
    {
      if (this->print_level_ >= 2) 
      {
        LOG_INFO("Interpolate", " hanging dofs before filter ");
      }
      interpolate_constrained_vector< DataType, DIM >(*space, *sol);
      sol->Update();
    }

    this->op_->ApplyFilter(*sol);
    sol->Update();
    if (space != nullptr) 
    {
      if (this->print_level_ >= 2) 
      {
        LOG_INFO("Interpolate", " hanging dofs after filter  ");
      }
      interpolate_constrained_vector< DataType, DIM >(*space, *sol);
      sol->Update();
    }

    if (non_const_mode_) 
    {
      this->ComputeResidualNonConst(*sol, rhs, this->res_);
    } 
    else 
    {
      this->ComputeResidual(*sol, rhs, this->res_);
    }

    return kNonlinearSolverSuccess;
  } 
  else if (this->DampingStrategy_ == NewtonDampingStrategyOwn) 
  {
    assert(DampStratObject_ != nullptr);
    DampingState state = DampStratObject_->Update(cor, rhs, this->res_, sol, this, space);
    this->residual_ = DampStratObject_->GetResidual();
    this->resids_.push_back(this->residual_);

    if (state != 0) 
    {
      return kNonlinearSolverError;
    }
  } 
  else 
  {
    return kNonlinearSolverError;
  }
  return kNonlinearSolverSuccess;
}

/// Updates solution vector using correction and possible damping/
/// forcing strategies
/// @param cor correction vector
/// @param sol solution vector

template < class LAD, int DIM >
NonlinearSolverState
Newton< LAD, DIM >::UpdateSolution(const VectorType &cor, 
                                   VectorType *sol,
                                   VectorSpace< DataType, DIM > const *space) 
{
  VectorType *rhs = new VectorType();
  rhs->Clear();
  NonlinearSolverState state = this->UpdateSolution(cor, *rhs, sol, space);
  delete rhs;
  return state;
}

/// Returns jacobian matrix J of nonlinear problem F at x
/// @param x point of evaluation
/// @param jacobian at x

template < class LAD, int DIM >
void Newton< LAD, DIM >::ComputeJacobian(const VectorType &x, 
                                         MatrixType *jacobian) 
{
  this->op_->EvalGrad(x, jacobian);
}

/// Returns jacobian matrix J of nonlinear problem F at x
/// @param x point of evaluation
/// @param jacobian at x

template < class LAD, int DIM >
void Newton< LAD, DIM >::ComputeJacobianNonConst(VectorType &x,
                                                 MatrixType *jacobian) 
{
  this->op_->EvalGradNonConst(x, jacobian);
}

/// Solves linear problem J*c=r
/// If Forcing is activated, then the system
/// is solved in an inexact way with
/// relative tolerance determined by forcing terms
/// @param jacobian jacobian matrix J
/// @param residual residual vector r
/// @param correction correction vector c

template < class LAD, int DIM >
la::LinearSolverState Newton< LAD, DIM >::SolveJacobian(const MatrixType &jacobian,
                                                        const VectorType &residual,
                                                        VectorType *correction) 
{
  assert(correction != nullptr);
  assert(this->linsolve_ != nullptr);

  // Reset correction if needed
  if ((residual.size_local() != correction->size_local()) ||
      (residual.size_global() != correction->size_global())) 
  {
    correction->CloneFromWithoutContent(residual);
  }

  // start vector
  correction->Zeros();
  //      correction->Update( );

  if (this->ForcingStrategy_ == NewtonForcingStrategyOwn) 
  {
    assert(this->ForcingStratObject_ != nullptr);
    if (this->print_level_ >= 3) 
    {
      LOG_INFO("Forcing term", this->ForcingStratObject_->GetCurrentForcingTerm());
    }
    this->linsolve_->SetRelativeTolerance(this->ForcingStratObject_->GetCurrentForcingTerm());
  }

  // pass on information object
  if (this->info_ != nullptr) 
  {
    std::stringstream lin_str;
    lin_str << "linsolve" << std::setw(1 + std::log10(this->control().maxits()))
            << std::setfill('0') << this->iter_;
    this->info_->add(lin_str.str());
    this->linsolve_->SetInfo(this->info_->get_child(lin_str.str()));
  }

  // solve
  la::LinearSolverState state = this->linsolve_->Solve(residual, correction);

  correction->Update();

  return state; // solve jacobian
}

/// Computes residual vector F(sol)-rhs for non-linear problem F with
/// right-hand side rhs
/// @param sol solution vector
/// @param rhs right-hand side vector
/// @param res residual vector

template < class LAD, int DIM >
void Newton< LAD, DIM >::ComputeResidual(const VectorType &sol,
                                         const VectorType &rhs, 
                                         VectorType *res) 
{
  assert(res != nullptr);
  // Reset residual if needed
  if ((res->size_local() != sol.size_local()) ||
      (res->size_global() != sol.size_global())) 
  {
    res->CloneFromWithoutContent(sol);
    res->Zeros();
  }

  // Compute residual
  this->op_->EvalFunc(sol, res);
  if ((rhs.size_local() == res->size_local()) &&
      (rhs.size_global() == res->size_global())) 
  {
    res->Axpy(rhs, static_cast< DataType >(-1.));
  }

  // res->Update( );

  // Compute new residual norm
  this->residual_ = res->Norm2();

  // Store it in vector
  this->resids_.push_back(this->residual_);
}

/// Computes residual vector F(sol)-rhs for non-linear problem F with
/// right-hand side rhs
/// @param sol solution vector
/// @param rhs right-hand side vector
/// @param res residual vector

template < class LAD, int DIM >
void Newton< LAD, DIM >::ComputeResidualNonConst(VectorType &sol,
                                                 const VectorType &rhs,
                                                 VectorType *res) 
{
  assert(res != nullptr);
  // Reset residual if needed
  if ((res->size_local() != sol.size_local()) ||
      (res->size_global() != sol.size_global())) 
  {
    res->CloneFromWithoutContent(sol);
    res->Zeros();
  }

  // Compute residual
  this->op_->EvalFuncNonConst(sol, res);
  if ((rhs.size_local() == res->size_local()) &&
      (rhs.size_global() == res->size_global())) 
  {
    res->Axpy(rhs, static_cast< DataType >(-1.));
  }

  // res->Update( );

  // Compute new residual norm
  this->residual_ = res->Norm2();

  // Store it in vector
  this->resids_.push_back(this->residual_);
}

/// Computes residual vector F(sol) for non-linear problem F
/// @param sol solution vector
/// @param res residual vector

template < class LAD, int DIM >
void Newton< LAD, DIM >::ComputeResidual(const VectorType &sol, 
                                         VectorType *res) 
{
  VectorType *rhs = new VectorType();
  rhs->Clear();
  this->ComputeResidual(sol, *rhs, res);
  delete rhs;
}

/// Solves F(x)=y
/// @param y right hand side vectorNewtonDampingStrategyArmijo
/// @param x solution vector
/// @return status if solver succeeded

template < class LAD, int DIM >
NonlinearSolverState
Newton< LAD, DIM >::Solve(const VectorType &rhs, 
                          VectorType *x,
                          VectorSpace< DataType, DIM > const *space)
{
  assert(this->res_ != nullptr);
  assert(this->jac_ != nullptr);
  assert(this->op_ != nullptr);
  assert(this->linsolve_ != nullptr);
  assert(x != nullptr);

  Timer timer;
  if (this->info_ != nullptr) 
  {
    timer.reset();
    timer.start();
  }

  // Init
  la::LinearSolverState LinSolState = la::kSolverSuccess;
  IterateControl::State conv = IterateControl::kIterate;

  this->res_->Clear();
  this->res_->CloneFromWithoutContent(rhs);

  VectorType *cor = new VectorType();
  cor->Clear();
  cor->CloneFromWithoutContent(rhs);

  VectorType *sol = new VectorType();
  sol->Clear();
  this->x_ = sol;
  if (InitialSolution_ == NewtonInitialSolutionOwn) 
  {
    sol->CloneFrom(*x);
  } 
  else if (InitialSolution_ == NewtonInitialSolution0) 
  {
    sol->CloneFromWithoutContent(rhs);
    sol->Zeros();
  } 
  else 
  {
    return kNonlinearSolverInitError;
  }

  // Step 0
  this->iter_ = 0;
  this->op_->Reinit();

  sol->Update();

  if (space != nullptr) 
  {
    if (this->print_level_ >= 3) 
    {
      LOG_INFO("Interpolate", " hanging dofs");
    }
    interpolate_constrained_vector< DataType, DIM >(*space, *sol);
    sol->Update();
  }

  if (non_const_mode_) 
  {
    this->ComputeResidualNonConst(*sol, rhs, this->res_);
  } 
  else 
  {
    this->ComputeResidual(*sol, rhs, this->res_);
  }

  DataType lin_solver_rhs_norm = this->GetResidual();

  conv = this->control().Check(this->iter(), lin_solver_rhs_norm);
  if (this->ForcingStrategy_ == NewtonForcingStrategyOwn) 
  {
    assert(this->ForcingStratObject_ != nullptr);
    this->ForcingStratObject_->SetResidualNorm(this->GetResidual());
  }

  if (this->print_level_ >= 1) 
  {
    LOG_INFO("Starts with abs res ", this->GetResidual());
  }
  if (this->ForcingStrategy_ == NewtonForcingStrategyOwn) 
  {
    if (this->print_level_ >= 1) 
    {
      LOG_INFO("forcing term", this->ForcingStratObject_->GetCurrentForcingTerm());
    }
  }

  while (conv == IterateControl::kIterate) 
  {
    // NextStep
    this->iter_++;
    Timer timer;
    timer.start();

    if (non_const_mode_) 
    {
      this->ComputeJacobianNonConst(*sol, this->jac_);
    } 
    else 
    {
      this->ComputeJacobian(*sol, this->jac_);
    }
    timer.stop();
    DataType asm_time = timer.get_duration();
    timer.reset();

    lin_solver_rhs_norm = this->GetResidual();

    timer.start();
    LinSolState = this->SolveJacobian(*this->jac_, *this->res_, cor);

    timer.stop();
    DataType solve_time = timer.get_duration();
    timer.reset();
    timer.start();

    if (LinSolState == la::kSolverError) 
    {
      break;
    }
    this->UpdateSolution(*cor, rhs, sol, space);
    this->op_->VisualizeIterate(*sol, this->iter_);
    
    timer.stop();
    DataType update_time = timer.get_duration();

    conv = this->control().Check(this->iter(), this->GetResidual());

    int lin_iter = this->linsolve_->iter();
    DataType lin_res = this->linsolve_->res();

    if (this->print_level_ >= 1) 
    {
      LOG_INFO("================", "");
      LOG_INFO("Newton iteration", this->iter_);
      LOG_INFO("Newton residual", this->GetResidual());
    }
    if (this->print_level_ >= 2) 
    {
      LOG_INFO("LinSolver iter", lin_iter);
      LOG_INFO("LinSolver absres", lin_res);
      LOG_INFO("LinSolver relres", lin_res / lin_solver_rhs_norm);
    }

    if (this->ForcingStrategy_ == NewtonForcingStrategyOwn) 
    {
      assert(this->ForcingStratObject_ != nullptr);
      this->ForcingStratObject_->ComputeForcingTerm(this->GetResidual(), this->linsolve_->res());
      if (this->print_level_ >= 3) 
      {
        LOG_INFO("New forcing term ", this->ForcingStratObject_->GetCurrentForcingTerm());
      }
    }

    if (sol->my_rank() == 0) 
    {
      this->WriteStatistics(asm_time, solve_time, update_time, lin_solver_rhs_norm);
    }
    this->asm_time_ += asm_time;
    this->solve_time_ += solve_time;
    this->update_time_ += update_time;
  }
  delete cor;

  if (this->print_level_ >= 1) 
  {
    LOG_INFO("finished with abs res", this->GetResidual());
    LOG_INFO("finished with rel res", this->GetResidual() / this->resids_[0]);
  }
  if (this->info_ != nullptr) 
  {
    timer.stop();
    this->info_->add("iter", this->iter());
    this->info_->add("time", timer.get_duration());
  }

  if (LinSolState == la::kSolverError) 
  {
    if (this->force_return_sol_) 
    {
      x->CopyFrom(*sol);
      x->Update();
    }
    delete sol;
    return kNonlinearSolverError;
  } 
  else if (conv == IterateControl::kFailureDivergenceTol ||
           conv == IterateControl::kFailureMaxitsExceeded) 
  {
    if (this->force_return_sol_) 
    {
      x->CopyFrom(*sol);
      x->Update();
    }
    delete sol;
    return kNonlinearSolverExceeded;
  } 
  else 
  {
    x->CopyFrom(*sol);
    x->Update();
    delete sol;
    return kNonlinearSolverSuccess;
  }
}

/// Solves F(x)=0
/// @param x solution vector
/// @return status if solver succeeded

template < class LAD, int DIM >
NonlinearSolverState Newton< LAD, DIM >::Solve(VectorType *x, 
                                               VectorSpace< DataType, DIM > const *space) 
{
  VectorType *rhs = new VectorType();
  rhs->Clear();
  rhs->CloneFromWithoutContent(*x);
  rhs->Zeros();
  NonlinearSolverState state = this->Solve(*rhs, x, space);
  delete rhs;
  return state;
}

template < class LAD, int DIM >
void Newton< LAD, DIM >::WriteStatistics(DataType asm_time, DataType solve_time,
                                         DataType update_time,
                                         DataType lin_solver_rhs_norm) 
{
  if (this->filename_statistics_.empty()) 
  {
    return;
  }
  
  DataType lin_iter = this->linsolve_->iter();
  DataType lin_res = this->linsolve_->res();
  DataType newton_iter = this->iter_;
  int resids_size = this->resids_.size();
  DataType newton_res = this->resids_[resids_size - 1];

  std::string path = this->filename_statistics_;
  std::ofstream out;
  out.open(path.c_str(), std::ios::out | std::ios::app);
  out.precision(6);
  out << std::scientific;
  out << newton_iter << " " << newton_res << " " << lin_iter << " " << lin_res
      << " " << lin_res / lin_solver_rhs_norm << " " << asm_time << " "
      << solve_time << " " << update_time << " "
      << "\n";
  out.close();
}

/// Plain constructor that does not initialize the solver.

template < class LAD, int DIM > 
Newton< LAD, DIM >::Newton() 
{
  this->InitialSolution_ = NewtonInitialSolution0;
  this->DampingStrategy_ = NewtonDampingStrategyNone;
  this->ForcingStrategy_ = NewtonForcingStrategyConstant;
  this->non_const_mode_ = false;

  this->asm_time_ = 0.;
  this->solve_time_ = 0.;
  this->update_time_ = 0.;
  this->filename_statistics_ = "";
}

/// Standard constructor requiring pointers to user reserved space
/// for residual vector and jacobian. Sets no forcing and no damping and
/// uses initial solution zero.

template < class LAD, int DIM >
Newton< LAD, DIM >::Newton(VectorType *residual, MatrixType *matrix)
    : res_(residual), jac_(matrix) 
{
  assert(res_ != nullptr);
  assert(jac_ != nullptr);

  this->InitialSolution_ = NewtonInitialSolution0;
  this->DampingStrategy_ = NewtonDampingStrategyNone;
  this->ForcingStrategy_ = NewtonForcingStrategyConstant;
  this->non_const_mode_ = false;

  this->asm_time_ = 0.;
  this->solve_time_ = 0.;
  this->update_time_ = 0.;
  this->filename_statistics_ = "";
  this->force_return_sol_ = false;
}

/// Standard destructor

template < class LAD, int DIM > 
Newton< LAD, DIM >::~Newton() 
{
  this->res_ = nullptr;
  this->jac_ = nullptr;
  this->linsolve_ = nullptr;
  this->op_ = nullptr;
  this->DampStratObject_ = nullptr;
  this->ForcingStratObject_ = nullptr;
  this->info_ = nullptr;
  this->filename_statistics_ = "";
  this->force_return_sol_ = false;
}

} // namespace hiflow

#endif // HIFLOW_NONLINEAR_NEWTON_H_
