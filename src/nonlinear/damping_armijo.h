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

#ifndef HIFLOW_NONLINEAR_DAMPING_ARMIJO_H_
#define HIFLOW_NONLINEAR_DAMPING_ARMIJO_H_

#include "nonlinear/damping_strategy.h"
#include "linear_algebra/la_descriptor.h"
#include "nonlinear/newton.h"
#include <string>
#include <vector>

namespace hiflow {

/// Enumerator @em ArmijoParam User changable

enum ArmijoParam {
  ArmijoInitial,
  ArmijoMinimal,
  ArmijoMaxLoop,
  ArmijoDecrease,
  ArmijoSuffDec
};

/// @brief Base class for damping strategies
/// @author Tobias Hahn
///
/// Solves for x in F(x)=y with nonlinear F

template < class LAD, int DIM > 
class ArmijoDamping : public DampingStrategy< LAD, DIM > {
public:
  typedef typename LAD::MatrixType MatrixType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  ArmijoDamping();
  ArmijoDamping(DataType init, DataType mini, DataType dec, DataType suffdec,
                int maxloop);
  ~ArmijoDamping();

  DampingState Init(ArmijoParam param);
  DampingState Init(ArmijoParam param, int data);
  DampingState Init(ArmijoParam param, DataType data);

  DampingState Update(const VectorType &cor, const VectorType &rhs,
                      VectorType *res, VectorType *sol, void *myNewton,
                      VectorSpace< DataType, DIM > const *space = nullptr);

private:
  std::string name_;
  DataType Initial_;
  DataType Minimal_;
  DataType Decrease_;
  DataType SuffDec_;
  int MaxLoop_;
  std::vector< DataType > *ForcingTerm_;
};

template < class LAD, int DIM >
DampingState ArmijoDamping< LAD, DIM >::Init(ArmijoParam param, int data) {
  if (param == ArmijoMaxLoop) {
    this->MaxLoop_ = data;
  } else {
    return kDampingError;
  }
  return kDampingSuccess;
}

template < class LAD, int DIM >
DampingState ArmijoDamping< LAD, DIM >::Init(ArmijoParam param, DataType data) {
  switch (param) {
  case ArmijoInitial:
    this->Initial_ = data;
    break;

  case ArmijoMinimal:
    this->Minimal_ = data;
    break;

  case ArmijoDecrease:
    this->Decrease_ = data;
    break;

  case ArmijoSuffDec:
    this->SuffDec_ = data;
    break;

  default:
    return kDampingError;
  }

  return kDampingSuccess;
}

template < class LAD, int DIM >
DampingState
ArmijoDamping< LAD, DIM >::Update(const VectorType &cor, const VectorType &rhs,
                             VectorType *res, VectorType *sol, void *myNewton,
                             VectorSpace< DataType, DIM > const *space) {
  Newton< LAD, DIM > *NewtonSolver = (Newton< LAD, DIM > *)myNewton;
  // trial step

  DataType res_start = res->Norm2();
  if (this->print_level_ > 0) {
    LOG_INFO("Initial residual", res_start);
  }

  DataType res_cur = res_start;
  this->residual_ = res_start;
  int iter = 0;
  DataType theta = this->Initial_;
  DataType t = this->SuffDec_;
  DataType eta = 0.;
  if (NewtonSolver->Forcing())
    NewtonSolver->GetForcingTerm(eta);

  // New solution and residual

  VectorType sol_backup;
  sol_backup.CloneFrom(*sol);

  sol->Axpy(cor, -theta);
  sol->Update();

  if (space != nullptr) {
    if (this->print_level_ > 0) {
      LOG_INFO("Interpolate hanging dofs              ", 1);
    }
    interpolate_constrained_vector< DataType, DIM >(*space, *sol);
    sol->Update();
  }

  NewtonSolver->GetOperator()->ApplyFilter(*sol);
  sol->Update();
  if (space != nullptr) {
    if (this->print_level_ > 0) {
      LOG_INFO("Interpolate hanging dofs              ", 1);
    }
    interpolate_constrained_vector< DataType, DIM >(*space, *sol);
    sol->Update();
  }

  if (NewtonSolver->GetNonConstMode()) {
    NewtonSolver->ComputeResidualNonConst(*sol, rhs, res);
  } else {
    NewtonSolver->ComputeResidual(*sol, rhs, res);
  }
  res_cur = res->Norm2();

  if (this->print_level_ > 0) {
    LOG_INFO("Residual norm (trial step)", res_cur);
  }
  this->residual_ = res_cur;

  // -> Iterate if needed

  const DataType bound = (1. - t * (1. - eta)) * res_start;
  if (this->print_level_ > 0) {
    LOG_INFO("Armijo damping acceptance bound", bound);
  }

  bool check_lower = true;
  while ( (res_cur > bound)
		  && (iter <= this->MaxLoop_)
		  && (theta > this->Minimal_)
		  && check_lower) {

    // store solution and residual from the last step

    VectorType sol_tmp, res_vec_tmp;
    sol_tmp.CloneFrom(*sol);
    res_vec_tmp.CloneFrom(*res);

    // restore old solution

	sol->CloneFrom(sol_backup);

    // change damping since actual residual is rejected

    theta *= this->Decrease_;

    // New solution and residual

    sol->Axpy(cor, -theta);
    sol->Update();

    if (space != nullptr) {
      if (this->print_level_ > 0) {
        LOG_INFO("Interpolate hanging dofs              ", 1);
      }
      interpolate_constrained_vector< DataType, DIM >(*space, *sol);
      sol->Update();
    }

    NewtonSolver->GetOperator()->ApplyFilter(*sol);
    sol->Update();

    if (space != nullptr) {
      if (this->print_level_ > 0) {
        LOG_INFO("Interpolate hanging dofs              ", 1);
      }
      interpolate_constrained_vector< DataType, DIM >(*space, *sol);
      sol->Update();
    }

    if (NewtonSolver->GetNonConstMode()) {
      NewtonSolver->ComputeResidualNonConst(*sol, rhs, res);
    } else {
      NewtonSolver->ComputeResidual(*sol, rhs, res);
    }
    DataType res_tmp = res_cur;
    res_cur = res->Norm2();

    if (res_cur > res_tmp - bound && res_tmp < res_start) {

      // restore solution and residual from last step

      sol->CopyFrom ( sol_tmp );
      res->CopyFrom ( res_vec_tmp );
      theta /= this->Decrease_;
      check_lower = false;
    } else {
      if (this->print_level_ > 0) {
        LOG_INFO("Residual norm (damped)", res_cur);
      }
      this->residual_ = res_cur;
      ++iter;
      if (NewtonSolver->Forcing()) {
        eta = 1. - theta * ( 1 - eta );
        NewtonSolver->SetForcingTerm(eta);
      }
    }
  }
  if (this->print_level_ > 0) {
    LOG_INFO("Damping factor", theta);
  }
  return kDampingSuccess;
}

template < class LAD, int DIM > 
ArmijoDamping< LAD, DIM >::ArmijoDamping() {
  Initial_ = 1.;
  Minimal_ = 1.e-4;
  Decrease_ = 0.5;
  SuffDec_ = 1.e-4;
  MaxLoop_ = 10;
  name_ = "Armijo";
}

template < class LAD, int DIM >
ArmijoDamping< LAD, DIM >::ArmijoDamping(DataType init, DataType mini, DataType dec,
                                    DataType suffdec, int maxloop)
    : Initial_(init), Minimal_(mini), Decrease_(dec), SuffDec_(suffdec),
      MaxLoop_(maxloop) {
  name_ = "Armijo";
}

template < class LAD, int DIM > 
ArmijoDamping< LAD, DIM >::~ArmijoDamping() {
  ForcingTerm_ = nullptr;
}




} // namespace hiflow

#endif // HIFLOW_NONLINEAR_DAMPING_ARMIJO_H_
