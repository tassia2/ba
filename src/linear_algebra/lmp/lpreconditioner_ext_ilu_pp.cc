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
#include <algorithm> 

namespace hiflow {
namespace la {

// --------------------------------------------------
// Class lPreconditioner_LUmkl
template < typename ValueType >
void lPreconditionerExt_ILUpp< ValueType >::Clear() 
{ 
  this->vec_ai_.clear();
  this->vec_aj_.clear();
  this->vec_av_.clear();
  
  lPreconditionerExt< ValueType >::Clear();
}

template < typename ValueType >
void lPreconditionerExt_ILUpp< ValueType >::Init(int prepro_type, 
                                                     int precond_no, 
                                                     int max_levels,
                                                     double mem_factor, 
                                                     double threshold, 
                                                     double min_pivot)
{
#ifdef WITH_ILUPP
  assert(prepro_type >= 0);
  assert(precond_no >= 0);
  assert(max_levels >= 1);
  assert(mem_factor > 0.);
  assert(threshold > 0.);
  assert(min_pivot >= 0.);

  // set preprocessing_type
  switch (prepro_type) {
  case 0:
    this->ilupp_preproc_.set_normalize();
    break;
  case 1:
    this->ilupp_preproc_.set_PQ();
    break;
  case 2:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING();
    break;
  case 3:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG();
    break;
  case 4:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
    break;
  case 5:
    this->ilupp_preproc_
        .set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG_DD_MOV_COR_IM();
    break;
  case 6:
    this->ilupp_preproc_.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING();
    break;
  case 7:
    this->ilupp_preproc_
        .set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG();
    break;
  case 8:
    this->ilupp_preproc_
        .set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
    break;
  case 9:
    this->ilupp_preproc_
        .set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG_DD_MOV_COR_IM();
    break;
  case 10:
    this->ilupp_preproc_.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING();
    break;
  case 11:
    this->ilupp_preproc_
        .set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
    break;
  case 12:
    this->ilupp_preproc_.set_SPARSE_FIRST();
    break;
  case 13:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_PQ();
    break;
  case 14:
    this->ilupp_preproc_.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_PQ();
    break;
  case 15:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER();
    break;
  case 16:
    this->ilupp_preproc_.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER();
    break;
  case 17:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM();
    break;
  case 18:
    this->ilupp_preproc_
        .set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM();
    break;
  case 19:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR();
    break;
  case 20:
    this->ilupp_preproc_.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR();
    break;
  case 21:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM();
    break;
  case 22:
    this->ilupp_preproc_
        .set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM();
    break;
  case 23:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR();
    break;
  case 24:
    this->ilupp_preproc_.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR();
    break;
  case 25:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM();
    break;
  case 26:
    this->ilupp_preproc_
        .set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM();
    break;
  case 27:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ();
    break;
  case 28:
    this->ilupp_preproc_.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ();
    break;
  case 29:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER();
    break;
  case 30:
    this->ilupp_preproc_
        .set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER();
    break;
  case 31:
    this->ilupp_preproc_
        .set_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM();
    break;
  case 32:
    this->ilupp_preproc_
        .set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM();
    break;
  case 33:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER();
    break;
  case 34:
    this->ilupp_preproc_
        .set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER();
    break;
  case 35:
    this->ilupp_preproc_.set_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM();
    break;
  case 36:
    this->ilupp_preproc_
        .set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM();
    break;

  default:
    this->ilupp_preproc_.set_normalize();
    break;
  }

  // set precond_parameters for ILU++
  this->ilupp_param_.init(this->ilupp_preproc_, precond_no, "some comment"); // ACHTUNG: setzt default-parameters

  // set_threshold(tau) sets dropping threshold to 10^{-tau}
  this->ilupp_param_.set_threshold(threshold);

  // set minimal pivot
  this->ilupp_param_.set_MEM_FACTOR(mem_factor);
  this->ilupp_param_.set_MIN_PIVOT(min_pivot);
  this->ilupp_param_.set_MAX_LEVELS(max_levels);
  
#else
  LOG_ERROR("need to compile with ILUPP support");
  quit_program();
#endif

  this->init_done_ = true;    
}

template < typename ValueType >
void lPreconditionerExt_ILUpp< ValueType >::FactorizeSymbolic() 
{
  LOG_INFO("ILUPP", "factorize symbolic");
  
#ifdef WITH_ILUPP
  if (this->GetState() == 0)
  {
    LOG_ERROR("Need to perform Init() before symbolic factorization");
    quit_program();
  }
  
  assert (this->nnz_ > 0);
  assert (this->nrow_ > 0);
  assert (this->ai_ != nullptr);
  assert (this->aj_ != nullptr);

  this->vec_ai_.clear();
  this->vec_aj_.clear();                               
  this->vec_ai_.reserve(this->nrow_+1);
  this->vec_aj_.reserve(this->nnz_);
  
  this->vec_ai_.assign(this->ai_, this->ai_+this->nrow_+1);
  this->vec_aj_.assign(this->aj_, this->aj_+this->nnz_);
                            
  this->sym_factor_done_ = true;
  
#else
  LOG_ERROR("need to compile with Umfpack support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_ILUpp< ValueType >::FactorizeNumeric() 
{
#ifdef WITH_ILUPP
  // Analyzing is only done once, or when explicetly called from outside
  if (this->GetState() < 3)
  {
    this->FactorizeSymbolic(); 
  }

  //std::cout << this->nnz_ << " " << this->nrow_ << std::endl;

  assert (this->av_ != nullptr);
  this->vec_av_.clear();                               
  this->vec_av_.reserve(this->nnz_);
  this->vec_av_.assign(this->av_, this->av_+this->nnz_);
  
  this->ilupp_precond_.setup(this->vec_av_, this->vec_aj_, this->vec_ai_, iluplusplus::ROW, this->ilupp_param_);
  
  this->num_factor_done_ = true;

#else
  LOG_ERROR("need to compile with Umfpack support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_ILUpp< ValueType >::ApplylPreconditioner(const ValueType * input,
                                                                 ValueType *output) 
{
  assert (output != nullptr);
  assert (input != nullptr);
  if (this->GetState() < 7) 
  {
    this->Build();
  }
#ifdef WITH_ILUPP

  std::copy(input, input+this->nrow_, output);

  this->ilupp_precond_.apply_preconditioner(output, this->nrow_);

#else
  LOG_ERROR("need to compile with Umfpack support");
  quit_program();
#endif
}

template class lPreconditionerExt_ILUpp< double >;

} // namespace la
} // namespace hiflow

