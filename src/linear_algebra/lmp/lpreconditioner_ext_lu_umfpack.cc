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

#include "lpreconditioner_ext.h"

#include "common/macros.h"
#include <assert.h>

#define nUSE_CSC

namespace hiflow {
namespace la {

// --------------------------------------------------
// Class lPreconditioner_LUmkl
template < typename ValueType >
void lPreconditionerExt_LUumfpack< ValueType >::Clear() 
{ 

#ifdef WITH_UMFPACK
  if (this->sym_factor_done_)
  {
    umfpack_di_free_symbolic(&symbolic_);
  }
  if (this->num_factor_done_)
  {
    umfpack_di_free_numeric(&numeric_);
  }
#endif

  this->clear_csc_matrix();
  
  this->symbolic_ = nullptr;
  this->numeric_ = nullptr;
  
  this->Control_.clear();
  this->Info_.clear();
  
  lPreconditionerExt< ValueType >::Clear();
}

template < typename ValueType >
void lPreconditionerExt_LUumfpack< ValueType >::clear_csc_matrix()
{
  if (this->csc_matrix_.row != nullptr)
  {
    delete[] this->csc_matrix_.row;
  }
  if (this->csc_matrix_.col != nullptr)
  {
    delete[] this->csc_matrix_.col;
  }
  if (this->csc_matrix_.val != nullptr)
  {
    delete[] this->csc_matrix_.val;
  }

  this->csc_matrix_.row = nullptr;
  this->csc_matrix_.col = nullptr;
  this->csc_matrix_.val = nullptr;  
}

template < typename ValueType >
void lPreconditionerExt_LUumfpack< ValueType >::init_csc_matrix()
{
  CSR_lMatrixTypeConst<ValueType> csr_data;
  csr_data.row = this->ai_;
  csr_data.col = this->aj_;
  csr_data.val = this->av_;
  
  this->clear_csc_matrix();
  
  csr_2_csc<ValueType, int>(csr_data, this->nrow_, this->nrow_, this->nnz_, true, this->csc_matrix_);
  
}

template < typename ValueType >
void lPreconditionerExt_LUumfpack< ValueType >::update_csc_values()
{
  CSR_lMatrixTypeConst<ValueType> csr_data;
  csr_data.row = this->ai_;
  csr_data.col = this->aj_;
  csr_data.val = this->av_;
  
  csrval_2_cscval<ValueType, int>(csr_data, this->nrow_, this->nrow_, this->nnz_, this->csc_matrix_);
}

template < typename ValueType >
void lPreconditionerExt_LUumfpack< ValueType >::Init(bool iterative_refinement,
                                                     bool symmetric,
                                                     double pivot_tol,
                                                     double sym_pivot_tol,
                                                     bool scale_sum,
                                                     bool scale_max)
{
#ifdef WITH_UMFPACK

  this->Control_.resize(UMFPACK_CONTROL, 0.);
  this->Info_.resize(UMFPACK_INFO, 0.);
  assert (!(scale_sum && scale_max));
  
  umfpack_di_defaults(&(this->Control_[0]));
  
  if (iterative_refinement)
  {
    this->Control_[UMFPACK_IRSTEP] = 2;
  }
  else
  {
    this->Control_[UMFPACK_IRSTEP] = 1;
  }
  
  if (symmetric)
  {
    this->Control_[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
    if (sym_pivot_tol > 0.)
    {
      this->Control_[UMFPACK_SYM_PIVOT_TOLERANCE] = sym_pivot_tol;
    }
    else
    {
      this->Control_[UMFPACK_SYM_PIVOT_TOLERANCE] = 0.1;
    }
  }
  else
  {
    this->Control_[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO;
  }

  if (pivot_tol > 0.)
  {
    this->Control_[UMFPACK_PIVOT_TOLERANCE] = pivot_tol;
  }
  else
  {
    this->Control_[UMFPACK_PIVOT_TOLERANCE] = 0.1;
  }
  
  if (scale_sum)
  {
    this->Control_[UMFPACK_SCALE] = UMFPACK_SCALE_SUM;
  }
  if (scale_max)
  {
    this->Control_[UMFPACK_SCALE] = UMFPACK_SCALE_MAX;
  }
  
#else
  LOG_ERROR("need to compile with Umfpack support");
  quit_program();
#endif

  this->init_done_ = true;    
}

template < typename ValueType >
void lPreconditionerExt_LUumfpack< ValueType >::FactorizeSymbolic() 
{
#ifdef WITH_UMFPACK
  if (this->GetState() == 0)
  {
    LOG_ERROR("Need to perform Init() before symbolic factorization");
    quit_program();
  }
  
  assert (this->nnz_ > 0);
  assert (this->nrow_ > 0);
  assert (this->ai_ != nullptr);
  assert (this->aj_ != nullptr);
  assert (this->av_ != nullptr);

#ifdef USE_CSC  
  this->init_csc_matrix();

  int status = umfpack_di_symbolic(this->nrow_, this->nrow_, 
                                   this->csc_matrix_.col, 
                                   this->csc_matrix_.row, 
                                   this->csc_matrix_.val, 
                                   &(this->symbolic_), 
                                   &(this->Control_[0]), 
                                   &(this->Info_[0]));
#else
  int status = umfpack_di_symbolic(this->nrow_, this->nrow_, 
                                   this->ai_, 
                                   this->aj_, 
                                   this->av_, 
                                   &(this->symbolic_), 
                                   &(this->Control_[0]), 
                                   &(this->Info_[0]));

#endif

  //umfpack_di_report_status(&(this->Control_[0]), 
  //                         &(this->Info_[0]));

  assert(status == UMFPACK_OK);
  assert(this->symbolic_ != nullptr);
  
  this->sym_factor_done_ = true;
  
#else
  LOG_ERROR("need to compile with Umfpack support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_LUumfpack< ValueType >::FactorizeNumeric() 
{
#ifdef WITH_UMFPACK
  // Analyzing is only done once, or when explicetly called from outside
  if (this->GetState() < 3)
  {
    this->FactorizeSymbolic(); 
  }

  assert (this->av_ != nullptr);
  
#ifdef USE_CSC
  this->update_csc_values();
  int status = umfpack_di_numeric(this->csc_matrix_.col, 
                                         this->csc_matrix_.row, 
                                         this->csc_matrix_.val, 
                                         this->symbolic_, 
                                         &(this->numeric_),
                                         &(this->Control_[0]), 
                                         &(this->Info_[0]));
#else
  int status = umfpack_di_numeric(this->ai_, 
                                         this->aj_, 
                                         this->av_, 
                                         this->symbolic_, 
                                         &(this->numeric_),
                                         &(this->Control_[0]), 
                                         &(this->Info_[0]));
#endif

  //umfpack_di_report_status(&(this->Control_[0]), 
  //                         &(this->Info_[0]));

  assert(status == UMFPACK_OK);
  assert(this->numeric_ != nullptr);

  this->num_factor_done_ = true;

#else
  LOG_ERROR("need to compile with Umfpack support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_LUumfpack< ValueType >::ApplylPreconditioner(const ValueType * input,
                                                                     ValueType *output) 
{
  assert (output != nullptr);
  assert (input != nullptr);
  if (this->GetState() < 7) 
  {
    this->Build();
  }
#ifdef WITH_UMFPACK

#ifdef USE_CSC
  int status = umfpack_di_solve(UMFPACK_A, 
                                this->csc_matrix_.col, 
                                this->csc_matrix_.row, 
                                this->csc_matrix_.val, 
                                output, 
                                input, 
                                this->numeric_,
                                &(this->Control_[0]), 
                                &(this->Info_[0]));
#else
  int status = umfpack_di_solve(UMFPACK_At, 
                                this->ai_, 
                                this->aj_, 
                                this->av_, 
                                output, 
                                input, 
                                this->numeric_,
                                &(this->Control_[0]), 
                                &(this->Info_[0]));
#endif
  umfpack_di_report_status(&(this->Control_[0]), status);
  //umfpack_di_report_control(&(this->Control_[0]));
  umfpack_di_report_info(&(this->Control_[0]), &(this->Info_[0]));

  assert (status == UMFPACK_OK);

#else
  LOG_ERROR("need to compile with Umfpack support");
  quit_program();
#endif
}

template class lPreconditionerExt_LUumfpack< double >;

} // namespace la
} // namespace hiflow

