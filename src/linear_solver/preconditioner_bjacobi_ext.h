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

/// @author Philipp Gerstner

#ifndef HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_EXT_H_
#define HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_EXT_H_

#include "linear_algebra/lmp/lpreconditioner_ext.h"
#include "linear_solver/preconditioner_bjacobi.h"
#include "linear_solver/preconditioner.h"

namespace hiflow {
namespace la {

/// @brief Block Jacobi preconditioners with external factorization on each block 

template < class LAD >
class PreconditionerBlockJacobiExt : public PreconditionerBlockJacobi< LAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

  PreconditionerBlockJacobiExt();
  virtual ~PreconditionerBlockJacobiExt();

  // Note: if SetupOperator is called for the first time, then the local preconditioner performs a
  // symbolic factorization. 
  // Afterwards, new calls to SetupOperator let the symbolic Factorization unchanged, i.e. only numerical
  // factorization is performed in localPrecond::Build()
  // Use ClearSymbolicFactor() to force an update of the symbolic factorization
  void UpdateSymbolicFactor()
  {
    this->force_sym_factor_update_ = true;
  }
  
  /// Setup the local operator for the local preconditioner
  void SetupOperator(OperatorType &op);

  /// Sets up paramaters 
  
  // use Intel MKL LU factorization
  void Init_LU_mkl()
  {
    this->Init_LU_mkl(true, false, false);
  }
  
  void Init_LU_mkl (bool iterative_refinement, 
                    bool symmetric,
                    bool positive);
  
  void Init_ILU_mkl()
  {
    this->Init_ILU_mkl(true, 1e-16, 1e-10, 0);
  }
   
  void Init_ILU_mkl (bool diag_regularize,
                     double diag_rel_threshold,
                     double diag_rel_eps,
                     int maxfill);
                    
  void Init_ILU_pp(int prepro_type, 
                   int precond_no, 
                   int max_levels,
                   double mem_factor, 
                   double threshold, 
                   double min_pivot);
                                                 
  void Init_ILU_pp ()
  {   
    this->Init_ILU_pp(1, 1010, 20, 0.6, 3.5, 0.005);
  }
  
  void Init_ILU_pp (const PropertyTree &params)
  {
    const int prepro_type = params["PreprocessingType"].get< int >();
    const int precond_no = params["PreconditionerNumber"].get< int >();
    const int max_levels = params["MaxMultilevels"].get< int >();
    const double mem_factor = params["MemFactor"].get< double >();
    const double threshold = params["PivotThreshold"].get< double >();
    const double min_pivot = params["MinPivot"].get< double >();
  
    this->Init_ILU_pp(prepro_type, precond_no, max_levels, mem_factor, threshold, min_pivot);
  }
  
  void Init_LU_umfpack()
  {
    this->Init_LU_umfpack(true, false, 0.1, 0.1, true, false);
  }
  
  void Init_LU_umfpack(bool iterative_refinement,
                       bool symmetric,
                       double pivot_tol,
                       double sym_pivot_tol,
                       bool scale_sum,
                       bool scale_max);
  
  
  void Init (const std::string& local_solver_type,  
             const PropertyTree &params) 
  {
    PropertyTree cur_param = params[local_solver_type];
  
    if (local_solver_type == "ILUPP")
    {
      this->Init_ILU_pp(
              cur_param["PreprocessingType"].get< int >(),
              cur_param["PreconditionerNumber"].get< int >(),
              cur_param["MaxMultilevels"].get< int >(),
              cur_param["MemFactor"].get< DataType >(),
              cur_param["PivotThreshold"].get< DataType >(),
              cur_param["MinPivot"].get< DataType >());
    }
    else if (local_solver_type == "MklILU")
    {
      this->Init_ILU_mkl(cur_param["RegularizeDiag"].get< bool >(),
                          cur_param["DiagThreshold"].get< DataType >(),
                          cur_param["DiagEps"].get< DataType >(),
                          cur_param["Bandwidth"].get< int >());
    }
    else if (local_solver_type == "MklLU")
    {
      this->Init_LU_mkl(cur_param["IterativeRefinement"].get< bool >(), false, false);
    }
    else if (local_solver_type == "UmfpackLU")
    {
      this->Init_LU_umfpack(cur_param["IterativeRefinement"].get< bool >(),
                             false, 
                             cur_param["PivotTol"].get< DataType >(), 
                             0.1, 
                             cur_param["ScaleSum"].get< DataType >(), 
                             cur_param["ScaleMax"].get< DataType >()); 
    }
    else
    {
      assert(false);
    }
  }
  
  /// Erase possibly allocated data.
  void Clear();

  void Print(std::ostream &out = std::cout) const;

protected:
  /// Applies the preconditioner on the diagonal block.
  /// @param b right hand side vector
  /// @param x solution vector
  /// @return status if preconditioning succeeded
  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x);

  /// Build the preconditioner
  virtual void BuildImpl(VectorType const *b, VectorType *x);

  lPreconditionerExt< DataType, int > *localPrecond_;
  
  bool force_sym_factor_update_;
  
  int* ia_;
  int* ja_;
  DataType* val_;
};

template < class LAD >
PreconditionerBlockJacobiExt< LAD >::PreconditionerBlockJacobiExt()
    : PreconditionerBlockJacobi< LAD >() {
  this->localPrecond_ = nullptr;
  this->op_ = nullptr;
  this->force_sym_factor_update_ = true;
}

template < class LAD >
PreconditionerBlockJacobiExt< LAD >::~PreconditionerBlockJacobiExt() {
  this->Clear();
}

template < class LAD > void PreconditionerBlockJacobiExt< LAD >::Clear() {
  if (this->localPrecond_ != nullptr) 
  {
    this->localPrecond_->Clear();
    delete this->localPrecond_;
  }
  this->localPrecond_ = nullptr;
  this->force_sym_factor_update_ = true;
  Preconditioner< LAD >::Clear();
  this->op_ = nullptr;
}

template < class LAD >
void PreconditionerBlockJacobiExt< LAD >::SetupOperator(OperatorType &op) 
{
  // get CSR structure of operator
  assert (this->localPrecond_ != nullptr);
  
  const int nrow = op.num_rows_local();
  const int nnz = op.nnz_local_diag();
   
  op.GetDiagonalCSR(this->ia_, this->ja_, this->val_);

  // pass sparsity structure to local precond
  if (this->op_ == nullptr || this->force_sym_factor_update_)
  {                                
    this->localPrecond_->SetupOperatorStructure(this->ia_,
                                                this->ja_,
                                                nrow, nnz, true);
  }
  
  // pass values to local precond
  this->localPrecond_->SetupOperatorValues(this->val_);
  
  this->op_ = &op;
  this->force_sym_factor_update_ = false;
  
  this->SetModifiedOperator(true);
}

template < class LAD >
void PreconditionerBlockJacobiExt< LAD >::Init_LU_mkl(bool iterative_refinement, 
                                                      bool symmetric,
                                                      bool positive)
{
  this->Clear();
#ifdef WITH_MKL
  lPreconditionerExt_LUmkl< DataType > *lp =
      new lPreconditionerExt_LUmkl< DataType >;
  lp->Init(iterative_refinement, symmetric, positive);
  
  this->localPrecond_ = lp;
#else
  LOG_ERROR("Need Intel MKL for this routine");
  quit_program();
#endif
}

template < class LAD >
void PreconditionerBlockJacobiExt< LAD >::Init_ILU_mkl(bool diag_regularize,
                                                       double diag_rel_threshold,
                                                       double diag_rel_eps,
                                                       int maxfill)
{
  this->Clear();
#ifdef WITH_MKL
  lPreconditionerExt_ILUmkl< DataType > *lp =
      new lPreconditionerExt_ILUmkl< DataType >;
  lp->Init(diag_regularize, diag_rel_threshold, diag_rel_eps, maxfill);
  
  this->localPrecond_ = lp;
#else
  LOG_ERROR("Need Intel MKL for this routine");
  quit_program();
#endif
}

template < class LAD >
void PreconditionerBlockJacobiExt< LAD >::Init_LU_umfpack(bool iterative_refinement,
                                                          bool symmetric,
                                                          double pivot_tol,
                                                          double sym_pivot_tol,
                                                          bool scale_sum,
                                                          bool scale_max)
{
  this->Clear();
#ifdef WITH_UMFPACK
  lPreconditionerExt_LUumfpack< DataType > *lp =
      new lPreconditionerExt_LUumfpack< DataType >;
  
  lp->Init(iterative_refinement,
           symmetric,
           pivot_tol,
           sym_pivot_tol,
           scale_sum,
           scale_max);
  
  this->localPrecond_ = lp;
#else
  LOG_ERROR("Need Umfpack for this routine");
  quit_program();
#endif
}

template < class LAD >
void PreconditionerBlockJacobiExt< LAD >::Init_ILU_pp(int prepro_type, 
                                                      int precond_no, 
                                                      int max_levels,
                                                      double mem_factor, 
                                                      double threshold, 
                                                      double min_pivot)
{
  this->Clear();
#ifdef WITH_ILUPP
  lPreconditionerExt_ILUpp< DataType > *lp =
      new lPreconditionerExt_ILUpp< DataType >;
  
  lp->Init(prepro_type, 
           precond_no, 
           max_levels,
           mem_factor, 
           threshold, 
           min_pivot);
  
  this->localPrecond_ = lp;
#else
  LOG_ERROR("Need ILU++ for this routine");
  quit_program();
#endif
}

template < class LAD >
void PreconditionerBlockJacobiExt< LAD >::BuildImpl(VectorType const *b,
                                                    VectorType *x) {
  assert(this->localPrecond_ != nullptr);
  assert(this->op_ != nullptr);

  this->localPrecond_->Build();
  this->SetState(true);
  this->SetModifiedOperator(false);
}

template < class LAD >
LinearSolverState
PreconditionerBlockJacobiExt< LAD >::SolveImpl(const VectorType &b,
                                               VectorType *x) 
{
  assert(this->op_ != nullptr);
  assert(this->localPrecond_ != nullptr);
  assert (x != nullptr);
  
  const DataType* b_buffer = b.interior().GetBuffer();
  DataType* x_buffer = x->interior().GetBuffer();
  
  this->localPrecond_->ApplylPreconditioner(b_buffer, x_buffer);

  x->store_interior();
  
  return kSolverSuccess;
}

template < class LAD >
void PreconditionerBlockJacobiExt< LAD >::Print(std::ostream &out) const {
  this->localPrecond_->print(out);
}


} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_PRECONDITIONER_B_JACOBI_EXT_H_
