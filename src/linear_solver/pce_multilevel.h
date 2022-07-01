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

#ifndef HIFLOW_LINEARSOLVER_PCE_MULTILEVEL_H_
#define HIFLOW_LINEARSOLVER_PCE_MULTILEVEL_H_

#include <mpi.h>
#include <vector>
#include <map>

#include "config.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "linear_solver/linear_solver.h"
#include "space/vector_space.h"
#include "common/log.h"
#include "common/sorted_array.h"
#include "nonlinear/nonlinear_problem.h"
#include "common/timer.h"

namespace hiflow {
namespace la {

template<class LAD>
class PCE_MultiLevel : public LinearSolver< LADescriptorPCE< LAD > > {

public :
  typedef typename polynomialchaos::PCTensor PCTensor;

  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  typedef LADescriptorPCE< LAD > LAD_PCE;
  typedef typename LAD_PCE::MatrixType PCE_OperatorType;
  typedef typename LAD_PCE::VectorType PCE_VectorType;
  typedef typename LAD_PCE::DataType PCE_DataType;

  // constructor
  PCE_MultiLevel();

  // desctructor
  virtual ~PCE_MultiLevel();

  // Clear
  void Clear();

  // initialization
  void Init(const PCTensor& pctensor);

  // set relative tolerance
  void SetRelativeTolerance(double reltol) {
    int maxits = this->control_.maxits();
    double atol = this->control_.absolute_tol();
    double dtol = this->control_.divergence_tol();
    this->control_.Init(maxits, atol, reltol, dtol);
  }

  // set operator
  //void SetupOperator(const OperatorType& op);
  void SetupOperator(PCE_OperatorType& op);

  // ML solve algorithm
  void ML(const PCE_VectorType& b, PCE_VectorType* x, const int l);

  // set pre and post smooting number
  void SetPrePostSmoothingNumber(const int nu1,const int nu2);

  // set mu for V or W cylce
  void SetMu(const int mu);

  // set smoother solver
  void SetSmootherSolver(LinearSolver<LAD>& smoother_solver);

  // set smoother preconditioner
  void SetSmootherPrecond(LinearSolver<LAD>& smoother_precond);

  // solve routine
  LinearSolverState SolveImpl(const PCE_VectorType& b, PCE_VectorType* x);

  // apply preconditioner
  LinearSolverState ApplyPreconditioner(const PCE_VectorType& b,
                                        PCE_VectorType* x);

  // setup preconditioner, potential operation for preconditioning the system matrix
  // currently nothing is implemented yet
  //void SetupPreconditioner(Preconditioner<LAD>& precond);


  // build
  void Build();

protected :

  // smoothing process
  void Smoothing(const PCE_VectorType& b, PCE_VectorType* x, const int nu,
                 const int l);

  // extract mode vector
  void ExtractLevelVector(const PCE_VectorType& x_fine, PCE_VectorType* x_coarse,
                          const int l);

  // assign mode vector to global vector
  void AssignLevelVectorToGlobal(const PCE_VectorType& x_coarse,
                                 PCE_VectorType* x_fine,
                                 const int l);

  // solve mode problem
  void SolveLevelProblem(const PCE_VectorType& b, PCE_VectorType* x, const int l);

  // solve mean problem
  void SolveMeanProblem(const PCE_VectorType& b, PCE_VectorType* x);

  // pctensor
  PCTensor pctensor_;

  // mean solver
  LinearSolver<LAD>* smoother_solver_;

  // precondtioner of mean solver
  LinearSolver<LAD>* smoother_precond_;

  // operator
  const PCE_OperatorType* op_;

  // smoothing parameters
  int nu1_, nu2_, mu_;

  // preconditioner for the system matrix
  Preconditioner<LAD> *precond_;

};

}
}

#endif
