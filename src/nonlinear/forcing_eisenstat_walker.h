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

#ifndef HIFLOW_NONLINEAR_EISENSTAT_WALKER_H_
#define HIFLOW_NONLINEAR_EISENSTAT_WALKER_H_

#include "nonlinear/forcing_strategy.h"
#include "linear_algebra/la_descriptor.h"
#include <string>
#include <vector>
#include <cmath>

namespace hiflow {

/// @brief Eisenstat and Walker computation of forcing terms
/// @author Michael Schick

template < class LAD > class EWForcing : public ForcingStrategy< LAD > {
public:
  typedef typename LAD::MatrixType MatrixType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  EWForcing();
  EWForcing(int choice);
  EWForcing(double initial, double max, int choice);
  EWForcing(double initial, double max, double min, int choice);
  ~EWForcing()
  {}

  // constructor
  EWForcing(double initial, double max, int choice, double gamma, double alpha);
  EWForcing(double initial, double max, double min, int choice, double gamma,
            double alpha);

  void ComputeForcingTerm(DataType new_residual, DataType lin_solve_accuracy);

private:
  void ComputeForcingChoice1(DataType new_residual,
                             DataType lin_solve_accuracy);
  void ComputeForcingChoice2(DataType new_residual);

  using ForcingStrategy< LAD >::name_;
  using ForcingStrategy< LAD >::forcing_terms_;
  using ForcingStrategy< LAD >::residuals_;

  DataType Initial_;
  DataType Maximal_;
  DataType Minimal_;
  int Choice_;

  DataType alpha_, gamma_;
};

template < class LAD >
void EWForcing< LAD >::ComputeForcingTerm(DataType new_residual,
                                          DataType lin_solve_accuracy) {
  if (Choice_ == 1) {
    ComputeForcingChoice1(new_residual, lin_solve_accuracy);
  } else if (Choice_ == 2) {
    ComputeForcingChoice2(new_residual);
  } else {
    ComputeForcingChoice1(new_residual, lin_solve_accuracy);
  }
}

template < class LAD >
void EWForcing< LAD >::ComputeForcingChoice1(DataType new_residual,
                                             DataType lin_solve_accuracy) {
  DataType old_residual_norm = residuals_.back();

  DataType eta =
      std::fabs(new_residual - lin_solve_accuracy) / old_residual_norm;

  if (eta > Maximal_) {
    eta = Maximal_;
  }

  if (eta < Minimal_) {
    eta = Minimal_;
  }

  forcing_terms_.push_back(eta);
  residuals_.push_back(new_residual);
}

template < class LAD >
void EWForcing< LAD >::ComputeForcingChoice2(DataType new_residual) {
  DataType old_residual = residuals_.back();

  DataType eta = gamma_ * pow((new_residual / old_residual), alpha_);

  if (eta > Maximal_) {
    eta = Maximal_;
  }

  if (eta < Minimal_) {
    eta = Minimal_;
  }

  forcing_terms_.push_back(eta);
  residuals_.push_back(new_residual);
}

template < class LAD > EWForcing< LAD >::EWForcing() {
  Initial_ = 0.5;
  Maximal_ = 0.9;
  Minimal_ = 0.;
  Choice_ = 1;
  name_ = "EisenstatWalker";
  gamma_ = 1.0;
  alpha_ = 0.5 * (1. + sqrt(5.0));
  forcing_terms_.push_back(Initial_);
}

template < class LAD > EWForcing< LAD >::EWForcing(int choice) {
  Initial_ = 0.5;
  Maximal_ = 0.9;
  Minimal_ = 0.;
  Choice_ = choice;
  name_ = "EisenstatWalker";
  forcing_terms_.push_back(Initial_);
  gamma_ = 1.0;
  alpha_ = 0.5 * (1. + sqrt(5.0));
}

template < class LAD >
EWForcing< LAD >::EWForcing(double initial, double max, int choice)
    : Initial_(initial), Maximal_(max), Minimal_(0.), Choice_(choice) {
  name_ = "EisenstatWalker";
  forcing_terms_.push_back(Initial_);
}

template < class LAD >
EWForcing< LAD >::EWForcing(double initial, double max, double min, int choice)
    : Initial_(initial), Maximal_(max), Minimal_(min), Choice_(choice) {
  name_ = "EisenstatWalker";
  forcing_terms_.push_back(Initial_);
}

template < class LAD >
EWForcing< LAD >::EWForcing(double initial, double max, int choice,
                            double gamma, double alpha)
    : Initial_(initial), Maximal_(max), Minimal_(0.), Choice_(choice),
      alpha_(alpha), gamma_(gamma) {
  name_ = "EisenstatWalker";
  forcing_terms_.push_back(Initial_);
}

template < class LAD >
EWForcing< LAD >::EWForcing(double initial, double max, double min, int choice,
                            double gamma, double alpha)
    : Initial_(initial), Maximal_(max), Minimal_(min), Choice_(choice),
      alpha_(alpha), gamma_(gamma) {
  name_ = "EisenstatWalker";
  forcing_terms_.push_back(Initial_);
}

} // namespace hiflow

#endif
