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

// --------------------------------------------------
// Class lPreconditioner_LUmkl
template <>
void lPreconditionerExt_LUmkl< float >::add_precision_option(MKL_INT& opt) const
{
#ifdef WITH_MKL
  opt += MKL_DSS_SINGLE_PRECISION; 
#endif
}

template <>
void lPreconditionerExt_LUmkl< double >::add_precision_option(MKL_INT& opt) const
{
}

template < typename ValueType >
void lPreconditionerExt_LUmkl< ValueType >::Clear() 
{ 
#ifdef WITH_MKL
  this->mkl_row_.clear();
  this->mkl_col_.clear();

  if (this->init_done_)
  {
    MKL_INT opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR;
    MKL_INT state = dss_delete (this->dss_handle_, opt);

    if (state != MKL_DSS_SUCCESS)
    {
      LOG_ERROR("dss_statistics  failed with error code " << state);
      quit_program();
    }
  }
#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif  

  this->symmetric_ = false;
  this->positive_ = false;
  this->solved_ = false;
  lPreconditionerExt< ValueType >::Clear();
}

template < typename ValueType >
void lPreconditionerExt_LUmkl< ValueType >::Init(bool iterative_refinement, 
                                                 bool symmetric,
                                                 bool positive) 
{
  this->Clear();
#ifdef WITH_MKL
  MKL_INT opt = MKL_DSS_MSG_LVL_WARNING
              + MKL_DSS_TERM_LVL_ERROR
              + MKL_DSS_ZERO_BASED_INDEXING;
              
  if (!iterative_refinement)
  {
    opt += MKL_DSS_REFINEMENT_OFF;
  }
  else
  {
    opt += MKL_DSS_REFINEMENT_ON;
  }
  
  this->add_precision_option (opt);

  MKL_INT state = dss_create(this->dss_handle_, opt); 

  if (state != MKL_DSS_SUCCESS)
  {
    LOG_ERROR("dss_create failed with error code " << state);
    quit_program();
  }
  
  this->init_done_ = true;
  this->symmetric_ = symmetric;
  this->positive_ = positive;
#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_LUmkl< ValueType >::FactorizeSymbolic() 
{
#ifdef WITH_MKL
  //LOG_INFO("local LU MKL", "analyze structure");
  if (this->GetState() == 0)
  {
    LOG_ERROR("Need to perform Init() before symbolic factorization");
    quit_program();
  }
  
  MKL_INT opt = MKL_DSS_MSG_LVL_WARNING
              + MKL_DSS_TERM_LVL_ERROR;
              
  if (this->symmetric_)
  {
    opt += MKL_DSS_SYMMETRIC;
  }
  else
  {
    opt += MKL_DSS_NON_SYMMETRIC;
  }
  
  assert (this->nnz_ > 0);
  assert (this->nrow_ > 0);
  assert (this->ai_ != nullptr);
  assert (this->aj_ != nullptr);
  
  // pass sparsity pattern to dss handle
  //LOG_INFO("local LU MKL", "dss define structure");
  MKL_INT state = dss_define_structure (this->dss_handle_, 
                                        opt,
                                        this->ai_,
                                        this->nrow_,
                                        this->nrow_,
                                        this->aj_,
                                        this->nnz_);
                                        
  //LOG_INFO("local LU MKL", "dss define structure, state = " << state);
  if (state != MKL_DSS_SUCCESS)
  {
    LOG_ERROR("dss_define_structure  failed with error code " << state);
    quit_program();
  }
  
  // perform reordering for redcued fill-in
  MKL_INT opt_reorder = MKL_DSS_AUTO_ORDER;
  MKL_INT* perm = nullptr;
  
  //LOG_INFO("local LU MKL", "dss reorder");
  state = dss_reorder(this->dss_handle_, opt_reorder, perm);

  //LOG_INFO("local LU MKL", "dss reorder, state = " << state);
  if (state != MKL_DSS_SUCCESS)
  {
    LOG_ERROR("dss_reorder  failed with error code " << state);
    quit_program();
  }
  
  this->sym_factor_done_ = true;
  
  //LOG_INFO("local LU MKL", "analyze structure done");
#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_LUmkl< ValueType >::FactorizeNumeric() 
{
#ifdef WITH_MKL
  // Analyzing is only done once, or when explicetly called from outside
  if (this->GetState() < 3)
  {
    this->FactorizeSymbolic(); 
  }

  assert (this->av_ != nullptr);
  
  MKL_INT opt;
  if (this->positive_)
  {
    opt = MKL_DSS_POSITIVE_DEFINITE;
  }
  else
  {
    opt = MKL_DSS_INDEFINITE;
  }
  
  MKL_INT state = dss_factor_real (this->dss_handle_,opt, this->av_);
    
  if (state != MKL_DSS_SUCCESS)
  {
    LOG_ERROR("dss_factor  failed with error code " << state);
    quit_program();
  }

  this->num_factor_done_ = true;

#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_LUmkl< ValueType >::ApplylPreconditioner(const ValueType * input,
                                                                 ValueType *output) 
{
  assert (output != nullptr);
  assert (input != nullptr);
  if (this->GetState() < 7) 
  {
    this->Build();
  }
#ifdef WITH_MKL
  MKL_INT opt = 0;
  MKL_INT nRhs = 1;

  MKL_INT state = dss_solve_real(this->dss_handle_, 
                                 opt, 
                                 input,
                                 nRhs, 
                                 output);

  if (state != MKL_DSS_SUCCESS)
  {
    LOG_ERROR("dss_solve_real  failed with error code " << state);
    quit_program();
  }
  
  this->solved_ = true;
#else
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif
}

template < typename ValueType >
void lPreconditionerExt_LUmkl< ValueType >::GetStatistics(std::vector<std::string>& param_types, 
                                                          std::vector<double>& param_vals ) 
{
  param_types.clear();
  param_vals.clear();

  if (this->GetState() < 3)
  {
    return;
  }
  
#ifdef WITH_MKL
  MKL_INT opt = 0;
  std::string args;
  
  if (this->solved_)
  {
    args = "ReorderTime,Peakmem,Factormem,FactorTime,Determinant,Inertia,Flops,Solvemem,SolveTime";
    param_vals.resize(11,0.);
    param_types.reserve(11);
    param_types.push_back("ReorderTime");
    param_types.push_back("PeakMem");
    param_types.push_back("FactorMem");
    param_types.push_back("FactorTime");
    param_types.push_back("Determinant");
    param_types.push_back("NumPosEigVal");
    param_types.push_back("NumNegEigVal");
    param_types.push_back("NumZeroEigVal");
    param_types.push_back("Flops");
    param_types.push_back("SolveMem");
    param_types.push_back("SolveTime");
  }
  else if (this->GetState() == 7)
  {
    args = "ReorderTime,Peakmem,Factormem,FactorTime,Determinant,Inertia,Flops,Solvemem";
    param_vals.resize(10,0.);
    param_types.reserve(10);
    param_types.push_back("ReorderTime");
    param_types.push_back("PeakMem");
    param_types.push_back("FactorMem");
    param_types.push_back("FactorTime");
    param_types.push_back("Determinant");
    param_types.push_back("NumPosEigVal");
    param_types.push_back("NumNegEigVal");
    param_types.push_back("NumZeroEigVal");
    param_types.push_back("Flops");
    param_types.push_back("SolveMem");
  }
  else if (this->GetState() == 3)
  {
    args = "ReorderTime,Peakmem,Factormem";
    param_vals.resize(3,0.);
    param_types.reserve(3);
    param_types.push_back("ReorderTime");
    param_types.push_back("PeakMem");
    param_types.push_back("FactorMem");
  }

  _CHARACTER_STR_t const * statArr = args.c_str();
  
  MKL_INT state = dss_statistics(this->dss_handle_, opt, statArr, &(param_vals[0]));

  if (state != MKL_DSS_SUCCESS)
  {
    LOG_ERROR("dss_statistics  failed with error code " << state);
    quit_program();
  }
#else  
  LOG_ERROR("need to compile with intel MKL support");
  quit_program();
#endif
}

template class lPreconditionerExt_LUmkl< double >;
template class lPreconditionerExt_LUmkl< float >;

} // namespace la
} // namespace hiflow


/*
 * 
 * if (!(std::is_same<int, MKL_INT>::value))
  {
    // Need to cast rowptr and column indices since MKL_INT = long int or long long int
    this->mkl_row_.resize(nRow+1,0);
    this->mkl_col_.resize(nNz,0);
    
    for (MKL_INT i = 0; i != nRow+1; ++i)
    {
      this->mkl_row_[i] = static_cast<MKL_INT>(cpu_csr_matrix->matrix_row(i));
    } 
    for (MKL_INT i = 0; i != nNz; ++i)
    {
      this->mkl_col_[i] = static_cast<MKL_INT>(cpu_csr_matrix->matrix_col(i));
    }
    
    rowPtr = &(this->mkl_row_[0]);
    colPtr = &(this->mkl_col_[0]);
  }
  else
  {
    rowPtr = cpu_csr_matrix->matrix_row();
    colPtr = cpu_csr_matrix->matrix_col();
  }
  * */
