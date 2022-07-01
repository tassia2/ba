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

#ifndef __LPRECONDITIONER_EXT_H
#define __LPRECONDITIONER_EXT_H

#include <iostream>
#include <stdlib.h>
#include <cassert>

#include "common/log.h"
#include "common/macros.h"
#include "common/property_tree.h"
#include "lmatrix_formats.h"

#ifdef WITH_MKL
#include "mkl.h"
#include "mkl_dss.h" 
#include "mkl_spblas.h"
#else
#define MKL_INT int  // to avoid compiling errors
#endif

#ifdef WITH_UMFPACK
#include "umfpack.h"
#endif

#ifdef WITH_ILUPP
#include "iluplusplus_interface.h"
#endif

namespace hiflow {
namespace la {

/// @brief Provides the base class to the local preconditioners based on external libraries
/// @author Philipp Gerstner

template < typename ValueType, typename IndexType = int > 
class lPreconditionerExt {
public:
  lPreconditionerExt()
  {
    this->csc_matrix_.row = nullptr;
    this->csc_matrix_.col = nullptr;
    this->csc_matrix_.val = nullptr;
    this->init_done_ = false;
    this->Clear();
  }

  virtual ~lPreconditionerExt()
  {
    this->Clear();
  }

  /// Clear the preconditioner
  virtual void Clear(void)
  {
    this->precond_name_ = "lPrecondExt";
    this->init_done_ = false;
    this->sym_factor_done_ = false;
    this->num_factor_done_ = false;
    this->av_ = nullptr;
    this->ai_ = nullptr;
    this->aj_ = nullptr;
    this->nnz_ = 0;
    this->nrow_ = 0;
    this->zero_index_ = true;
  }

  /// Setup the matrix operator for the preconditioner
  /// @param val, rowPtr, colPtr -> CSR format
  virtual void SetupOperatorStructure(IndexType* rowPtr,
                                      IndexType* colPtr,
                                      IndexType nrow,
                                      IndexType nnz,
                                      bool zero_indexing)
  {
    LOG_INFO("PreExt", "SetupOpStruct");
    assert (rowPtr != nullptr);
    assert (colPtr != nullptr);
    
    this->nnz_ = nnz;
    this->nrow_ = nrow;
    this->zero_index_ = zero_indexing;
    this->ai_ = rowPtr;
    this->aj_ = colPtr;
    this->SetModifiedOperator(true, true);
  }
  
  virtual void SetupOperatorValues(const ValueType* val)
  {
    LOG_INFO("PreExt", "SetupOpValues");
    assert (val != nullptr);
    this->av_ = val;
    this->SetModifiedOperator(true, false);
  }

  /// Build (internally) the preconditioning matrix
  virtual void Build()
  {
    switch (this->GetState())
    {
      case 0:
        LOG_ERROR("Build is called without previous initialization");
        quit_program();
        break;
      case 1:
        this->FactorizeSymbolic();
        this->FactorizeNumeric();
        break;
      case 3:
        this->FactorizeNumeric();
        break;
      case 7:
        break;
      default:
        LOG_ERROR("Inconsistent preconditioner state");
        quit_program();
    }
  }

  virtual void FactorizeSymbolic()
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
  
    this->sym_factor_done_ = true;
  }
  
  virtual void FactorizeNumeric() = 0;
  
  /// Apply the preconditioners (Solve Mz=r)
  /// @param - input vector r
  /// @return - output vector z
  virtual void ApplylPreconditioner(const ValueType * input,
                                    ValueType * output ) = 0;
                       

  /// Get State of the preconditioner
  int GetState() 
  { 
    return this->init_done_ + 2*this->sym_factor_done_ + 4*this->num_factor_done_; 
  }

  /// Set status of operator

  void SetModifiedOperator(bool val_changed, bool struct_changed) 
  {
    if (struct_changed)
    {
      this->sym_factor_done_ = false;
      this->num_factor_done_ = false;
    }
    if (val_changed)
    {
      this->num_factor_done_ = false;
    }
  }

  /// Print the type of preconditioner
  virtual void print(std::ostream &out = std::cout) const
  {
    LOG_INFO("lPreconditioner", this->precond_name_);  
  }

protected:
  /// pointer to the operator matrix
  const ValueType* av_;
  IndexType* ai_;
  IndexType* aj_;

  IndexType nnz_;
  IndexType nrow_;
  
  std::string precond_name_;

  /// Flag if operator has changed
  bool init_done_;
  bool sym_factor_done_;
  bool num_factor_done_;
  bool zero_index_;
  
  CSC_lMatrixType<ValueType> csc_matrix_;
};

/// @brief Sparse LU factorization by Intel MKL
/// @author Philipp Gerstner

template < typename ValueType >
class lPreconditionerExt_LUmkl : public hiflow::la::lPreconditionerExt< ValueType, MKL_INT > 
{
  
public:
  lPreconditionerExt_LUmkl()
  :lPreconditionerExt<ValueType, MKL_INT>() 
  {
    this->precond_name_ = "lPrecond_LU_mkl";
    this->symmetric_ = false;
    this->positive_ = false;
    this->solved_ = false;
  }

  virtual ~lPreconditionerExt_LUmkl()
  {
    this->Clear();
  }

  void Init()
  {
    this->Init(true, false, false);
  }
  
  // initialize MKL structs
  void Init(bool iterative_refinement, 
            bool symmetric,
            bool positive);
             
  // do reordering to reduce fill-in. Only depends on matrix sparsity structure, but not on its values
  // in case the operator changed: new Factorization is forced in Build(), but not new call to AnalyzeStructure()
  void FactorizeSymbolic();
  
  // numerical factorization
  void FactorizeNumeric();
  
  void Clear();

  // Forward-Backward solve
  void
  ApplylPreconditioner(const ValueType * input, ValueType * output );

  // retrieve some statistics
  void GetStatistics(std::vector<std::string>& param_types, 
                     std::vector<double>& param_values );
                                                       
protected:
  void add_precision_option(MKL_INT& opt) const; 

  bool symmetric_;
  bool positive_;
  bool solved_;
  
  std::vector<MKL_INT> mkl_row_;
  std::vector<MKL_INT> mkl_col_;

#ifdef WITH_MKL  
  _MKL_DSS_HANDLE_t dss_handle_;
#endif

};

template < typename ValueType >
class lPreconditionerExt_ILUmkl : public hiflow::la::lPreconditionerExt< ValueType, MKL_INT > {
public:
  lPreconditionerExt_ILUmkl()
    : lPreconditionerExt< ValueType >() 
  {
    this->precond_name_ = "lPrecond_ILU0_mkl";
  }

  virtual ~lPreconditionerExt_ILUmkl()
  {
    this->Clear();
  }

  void Init()
  {
    this->init_mkl_ilu(true, 1e-16, 1e-10, 0);
  }

  // initialize ILU(0) factorization, i.e. B = L*U has same sparsity structure as A
  // diag_regularize: if true, zero diagonal values are set to diag_eps (default: 1e-10)
  // diag_threshold : threshold, below which a diagonal entry is considered to be zero (default: 1e-16)
  void Init (bool diag_regularize,
             double diag_threshold,
             double diag_eps)
  {
    this->init_mkl_ilu(diag_regularize, diag_threshold, diag_eps, 0);
  }

  // initialize ILU(p) factorization, i.e. B = L*U with fill-in compared to A
  // diag_regularize    : if true, zero diagonal values are set to diag_rel_eps*norm(row)
  // diag_rel_threshold : if |diagonal entry| <= tol * norm(row), then diag entry is considered to be zero (default: 1e-16)
  // maxfill            : Maximum fill-in, which is half of the preconditioner bandwidth. 
  //                      The number of non-zero elements in the rows of the preconditioner cannot exceed (2*maxfil+1)
  void Init (bool diag_regularize,
             double diag_rel_threshold,
             double diag_rel_eps,
             int maxfill)
  {
    this->init_mkl_ilu(diag_regularize, diag_rel_threshold, diag_rel_eps, maxfill);
  }

  void FactorizeSymbolic();
  
  void FactorizeNumeric();
  
  void Clear();

  // Forward-Backward solve
  void  ApplylPreconditioner(const ValueType * input,
                             ValueType *output);
                                                       
protected:
  void init_mkl_ilu(bool diag_regularize,
                    double diag_threshold,
                    double diag_eps,
                    int maxfill); 
                                                
  std::vector<ValueType> z_;
  
  // sparsity structure of operator, casted into MKL Format (potentially different INT type and 1-based indexing)
  std::vector<MKL_INT> mkl_col_;
  std::vector<MKL_INT> mkl_row_;
  MKL_INT * colPtr_;
  MKL_INT * rowPtr_;
  
  // control parameters
  std::vector<MKL_INT> ipar_;
  std::vector<double> dpar_;
  MKL_INT maxfill_;
  double tol_;
  
  // B = L * U ~ A 
  std::vector<double> b_val_;
  std::vector<MKL_INT> b_row_;
  std::vector<MKL_INT> b_col_;

#ifdef WITH_MKL  
  sparse_matrix_t B_;
  matrix_descr L_descr_;
  matrix_descr U_descr_;
#endif

};

template < typename ValueType >
class lPreconditionerExt_LUumfpack : public hiflow::la::lPreconditionerExt< ValueType, int > 
{
public:
  lPreconditionerExt_LUumfpack()
    : lPreconditionerExt< ValueType >() 
  {
    this->precond_name_ = "lPrecond_LU_umfpack";
  }

  virtual ~lPreconditionerExt_LUumfpack()
  {
    this->Clear();
  }

  /// @param(in): symmetric
  /// @param(in): pivot_tolerance in (0,1]: large values: more stable but also more dense. small values: sparse but less stable
  /// @param(in): sym_pivot_tolerance in (0,1]: like pivot_tol, determines when diagonal entry is used as pivot element
  /// @param(in): scale_sum: scale each row by division with row sum
  /// @param(in): scale_max: scale each row by division with max row entry
  void Init(bool iterative_refinement,
            bool symmetric,
            double pivot_tol,
            double sym_pivot_tol,
            bool scale_sum,
            bool scale_max);
                                                 
  void Init ()
  {
    this->Init(true, false, 0.1, 0.1, true, false);
  }
  
  void FactorizeSymbolic();
  
  void FactorizeNumeric();
  
  void Clear();

  // Forward-Backward solve
  void  ApplylPreconditioner(const ValueType * input,
                             ValueType *output);
                                                       
protected:
  void init_csc_matrix();
  void clear_csc_matrix();
  void update_csc_values();

  std::vector<ValueType> Control_;
  std::vector<ValueType> Info_;

  void *symbolic_, *numeric_;

};

template < typename ValueType >
class lPreconditionerExt_ILUpp : public hiflow::la::lPreconditionerExt< ValueType, int > 
{
public:
  lPreconditionerExt_ILUpp()
    : lPreconditionerExt< ValueType >() 
  {
    this->precond_name_ = "lPrecond_ILUpp";
  }

  virtual ~lPreconditionerExt_ILUpp()
  {
    this->Clear();
  }

  /// Inits parameters for ILU++ preconditioner.
  /// \param prepro_type type of preprocessing
  /// \param precond_no number of preconditioner
  /// \param max_levels maximum number of multilevels
  /// \param mem_factor see ILU++ manual
  /// \param threshold see ILU++ manual
  /// \param min_pivot see ILU++ manual
  void Init(int prepro_type, 
            int precond_no, 
            int max_levels,
            double mem_factor, 
            double threshold, 
            double min_pivot);
                                                 
  void Init ()
  {   
    this->Init(1, 1010, 20, 0.6, 3.5, 0.005);
  }
  
  void Init (const PropertyTree &params)
  {
    const int prepro_type = params["PreprocessingType"].get< int >();
    const int precond_no = params["PreconditionerNumber"].get< int >();
    const int max_levels = params["MaxMultilevels"].get< int >();
    const double mem_factor = params["MemFactor"].get< double >();
    const double threshold = params["PivotThreshold"].get< double >();
    const double min_pivot = params["MinPivot"].get< double >();
  
    this->Init(prepro_type, precond_no, max_levels, mem_factor, threshold, min_pivot);
  }
  
  void FactorizeSymbolic();
  
  void FactorizeNumeric();
  
  void Clear();

  // Forward-Backward solve
  void  ApplylPreconditioner(const ValueType * input,
                             ValueType *output);
                                                       
protected:
  std::vector<int> vec_ai_;
  std::vector<int> vec_aj_;
  std::vector<ValueType> vec_av_;

#ifdef WITH_ILUPP
  iluplusplus::iluplusplus_precond_parameter ilupp_param_;
  iluplusplus::multilevel_preconditioner ilupp_precond_;
  iluplusplus::preprocessing_sequence ilupp_preproc_;
#endif


};
} // namespace la
} // namespace hiflow

#endif
