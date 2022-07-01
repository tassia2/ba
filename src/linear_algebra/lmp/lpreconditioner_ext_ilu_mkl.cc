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

namespace hiflow {
namespace la {

template < typename ValueType >
void lPreconditionerExt_ILUmkl< ValueType >::Clear() 
{ 
#ifdef WITH_MKL
  if (this->num_factor_done_)
  {
    mkl_sparse_destroy(this->B_);
  }
  
  this->mkl_col_.clear();
  this->b_val_.clear();
  this->b_row_.clear();
  this->b_col_.clear();
  this->ipar_.clear();
  this->dpar_.clear();
  
#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif  

  lPreconditionerExt< ValueType >::Clear();
}

template < typename ValueType >
void lPreconditionerExt_ILUmkl< ValueType >::init_mkl_ilu(bool diag_regularize,
                                                          double diag_threshold,
                                                          double diag_eps,
                                                          int maxfill) 
{
#ifdef WITH_MKL
  assert (maxfill >= 0);
  this->maxfill_ = static_cast<MKL_INT>(maxfill);
  this->ipar_.clear();
  this->ipar_.resize(128,0);
  this->dpar_.clear();
  this->dpar_.resize(128,0.);
  
  // TODO: use 1-based indexing here??
  if (diag_regularize)
  {
    this->ipar_[30] = 1;
  }
  
  if (maxfill == 0)
  {    
    // ILU(0)
    this->tol_ = 0.;
    if (diag_threshold >= 0.)
    {
      this->dpar_[30] = diag_threshold;
    }
    else
    {
      this->dpar_[30] = 1e-16;
    }
  
    if (diag_eps > 0.)
    {
      this->dpar_[31] = diag_eps;
    }
    else
    {
      this->dpar_[31] = 1e-10;
    }
  }
  else
  {
    // ILU(p)
    this->tol_ = diag_threshold;
    if (diag_eps > 0.)
    {
      this->dpar_[30] = diag_eps;
    }
    else
    {
      this->dpar_[30] = 1e-10;
    }
  }
  
  this->init_done_ = true;
#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_ILUmkl< ValueType >::FactorizeSymbolic() 
{
  if (this->GetState() == 0)
  {
    LOG_ERROR("Need to perform Init() before symbolic factorization");
    quit_program();
  }
  assert (this->nnz_ > 0);
  assert (this->nrow_ > 0);
  assert (this->ai_ != nullptr);
  assert (this->aj_ != nullptr);
  
#ifndef NDEBUG
  CSR_lMatrixTypeConst<ValueType, MKL_INT> input;
  input.row = this->ai_;
  input.col = this->aj_;

  assert (csr_check_ascending_order(input, this->nrow_, this->nnz_));
#endif

  // MKL uses 1-based indexing for ILU factorizations
  if (!this->zero_index_)
  {
    this->colPtr_ = this->aj_;
    this->rowPtr_ = this->ai_;
  }
  else
  {
    this->mkl_col_.clear();
    this->mkl_col_.resize(this->nnz_,0);
    this->mkl_row_.clear();
    this->mkl_row_.resize(this->nrow_+1,0);
    
    for (MKL_INT i = 0; i != this->nnz_; ++i)
    {
      this->mkl_col_[i] = this->aj_[i]+1;
    }

    for (MKL_INT i = 0; i != this->nrow_+1; ++i)
    {
      this->mkl_row_[i] = this->ai_[i]+1;
    }
    
    this->colPtr_ = &(this->mkl_col_[0]);
    this->rowPtr_ = &(this->mkl_row_[0]);
  }
  
  this->sym_factor_done_ = true;
}

template < typename ValueType >
void lPreconditionerExt_ILUmkl< ValueType >::FactorizeNumeric() 
{ 
  if (this->GetState() < 3)
  {
    this->FactorizeSymbolic(); 
  }

  assert (this->av_ != nullptr);
  assert (this->rowPtr_ != nullptr);
  assert (this->colPtr_ != nullptr);
  assert (this->nrow_ > 0);
  
  
  this->b_val_.clear();
  this->b_row_.clear();
  this->b_col_.clear();
  
#ifdef WITH_MKL
  MKL_INT ierr = 0;
  sparse_status_t state =  SPARSE_STATUS_SUCCESS;
  
  if (this->maxfill_ == 0)
  {
    // compute ILU(0) factorization
    this->b_val_.resize(this->nnz_,0.);
  
    dcsrilu0 (&(this->nrow_), 
              this->av_,
              this->rowPtr_,
              this->colPtr_,
              &(this->b_val_[0]),
              &(this->ipar_[0]),
              &(this->dpar_[0]),
              &ierr);

    if (ierr != 0)
    {
      LOG_ERROR("MKL ILU(" << this->maxfill_ << ") factorization failed with error code " << ierr); 
    }
    
    // create matrix handle corresponding to B = L * U ~ A^-1
    state = mkl_sparse_d_create_csr(&this->B_, 
                                    SPARSE_INDEX_BASE_ONE, 
                                    this->nrow_, 
                                    this->nrow_, 
                                    this->rowPtr_, 
                                    this->rowPtr_+1, 
                                    this->colPtr_, 
                                    &(this->b_val_[0]));
  }
  else
  {
    // compute ILU(p) factorization
    // see INTEL MKL developper reference
    const MKL_INT nnz = (2*this->maxfill_+1)*this->nrow_ - this->maxfill_*(this->maxfill_+1) + 1;
    
    this->b_val_.resize(this->nnz_, 0.);
    this->b_row_.resize(this->nrow_+1,0);
    this->b_col_.resize(this->nnz_, 0.);
    
    dcsrilut (&(this->nrow_), 
              this->av_,
              this->rowPtr_,
              this->colPtr_,
              &(this->b_val_[0]),
              &(this->b_row_[0]),
              &(this->b_col_[0]),
              &(this->tol_),
              &(this->maxfill_),
              &(this->ipar_[0]),
              &(this->dpar_[0]),
              &ierr);

    if (ierr != 0)
    {
      LOG_ERROR("MKL ILU(" << this->maxfill_ << ") factorization failed with error code " << ierr); 
    }
                
    // create matrix handle corresponding to B = L * U ~ A^-1
    state = mkl_sparse_d_create_csr(&this->B_, 
                                    SPARSE_INDEX_BASE_ONE, 
                                    this->nrow_, this->nrow_, 
                                    &(this->b_row_[0]), 
                                    &(this->b_row_[0])+1, 
                                    &(this->b_col_[0]), 
                                    &(this->b_val_[0]));
  }
                                                  
  if (state != SPARSE_STATUS_SUCCESS)
  {
    LOG_ERROR("Creation of sparse matrix handle failed with error code " << state);
    quit_program();
  }
  
  this->L_descr_.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  this->L_descr_.mode = SPARSE_FILL_MODE_LOWER;
  this->L_descr_.diag = SPARSE_DIAG_NON_UNIT;

  this->U_descr_.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  this->U_descr_.mode = SPARSE_FILL_MODE_UPPER;
  this->U_descr_.diag = SPARSE_DIAG_NON_UNIT;
    
  this->num_factor_done_ = true;

#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_ILUmkl< ValueType >::ApplylPreconditioner(const ValueType * input,
                                                                  ValueType *output) 
{
  assert (output != nullptr);
  assert (input != nullptr);
  
  if (this->GetState()<7) 
  {
    this->Build();
  }
#ifdef WITH_MKL
  // y = Bx = L * Ux
  // solve Lz = y
  // solve Ux = z
  this->z_.clear();
  this->z_.resize(this->nrow_,0.);
  
  sparse_status_t state = mkl_sparse_d_trsv (SPARSE_OPERATION_NON_TRANSPOSE,
                                             1., 
                                             this->B_,
                                             this->L_descr_,
                                             input,
                                             &(this->z_[0]));
  if (state != SPARSE_STATUS_SUCCESS)
  {
    LOG_ERROR("Error code" << state);
    quit_program();
  }
  
  state = mkl_sparse_d_trsv (SPARSE_OPERATION_NON_TRANSPOSE,
                             1., 
                             this->B_,
                             this->U_descr_,
                             &(this->z_[0]),
                             output);

  if (state != SPARSE_STATUS_SUCCESS)
  {
    LOG_ERROR("Error code" << state);
    quit_program();
  }
  
#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif
}

template class lPreconditionerExt_ILUmkl< double >;
} // namespace la
} // namespace hiflow
