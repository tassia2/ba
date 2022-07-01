// Copyright (C) 2011-2017 Vincent Heuveline
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

#ifndef __FEM_FLUX_LIMITER_H_
#define __FEM_FLUX_LIMITER_H

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <cmath>

#include "common/macros.h"
#include "common/vector_algebra.h"
#include "common/log.h"
#include "dof/dof_fem_types.h"
#include "space/vector_space.h"

#define nDBG_FULL_ANTIDIFF
#define nDBG_NO_ANTIDIFF

namespace hiflow {

namespace la {
template <class DataType> class SeqDenseMatrix;
}

namespace doffem {
                         
template <class LAD, int DIM>
class FluxLimiter
{
  typedef typename LAD::DataType DataType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::MatrixType MatrixType;
  
public:
  FluxLimiter()
  {}
    
  ~FluxLimiter()
  {}
    
  void clear();
  
  void init(const VectorSpace<DataType, DIM>* space,
            const MatrixType& lumped_mass,
            const VectorType* vec,
            const std::vector<int>& dirichlet_dofs);
  
  void compute_residual_flux (const MatrixType& mass_matrix,
                              const MatrixType& diff_matrix,
                              const VectorType& mass_vec,
                              const VectorType& diff_vec,
                              const DataType delta_t,
                              std::vector<DataType>& flux_diag,
                              std::vector<DataType>& flux_offdiag);

  void compute_residual_flux (const MatrixType& mass_matrix,
                              const MatrixType& diff_matrix,
                              const VectorType& mass_vec_1,
                              const VectorType& mass_vec_2,
                              const VectorType& diff_vec_1,
                              const VectorType& diff_vec_2,
                              const DataType delta_t,
                              const DataType theta,
                              std::vector<DataType>& flux_diag,
                              std::vector<DataType>& flux_offdiag);
                              
  void prelimit_flux_std (const VectorType& sol,
                          const std::vector<DataType>& flux_diag,
                          const std::vector<DataType>& flux_offdiag,
                          std::vector<DataType>& prelim_flux_diag,
                          std::vector<DataType>& prelim_flux_offdiag);
                                               
  void compute_limiter_zalesak (const VectorType& sol,
                                const bool upwind_biased,
                                const DataType delta_t,
                                const std::vector<DataType>& flux_diag,
                                const std::vector<DataType>& flux_offdiag,
                                const std::vector<bool>& edge_orient_diag,
                                const std::vector<bool>& edge_orient_offdiag,
                                std::vector<DataType>& alpha_diag,
                                std::vector<DataType>& alpha_offdiag);
                                                     
  void apply_limiter (VectorType& sol,
                      const std::vector<DataType>& prelim_flux_diag,
                      const std::vector<DataType>& prelim_flux_offdiag,
                      const std::vector<DataType>& alpha_diag,
                      const std::vector<DataType>& alpha_offdiag);
                      
protected:
  void update_bounds (int i, DataType flux, DataType diff_u);

  void update_limiter (const int k, 
                       const bool i_is_upwind,
                       const DataType flux,
                       const DataType Rp_i, const DataType Rp_j,
                       const DataType Rm_i, const DataType Rm_j,
                       std::vector<DataType>& alpha );
                                               
  std::vector< DataType > P_p_;
  std::vector< DataType > P_m_;
  std::vector< DataType > Q_p_;
  std::vector< DataType > Q_m_;
  std::vector< DataType > R_p_;
  std::vector< DataType > R_m_;

  std::vector< DataType > dirichlet_one_;
  std::vector< int > dirichlet_dofs_;
  
  std::vector< DataType > lmass_diag_vals_;
  std::vector< DataType > rplus_vals_;
  std::vector< DataType > rminus_vals_;
  std::vector< int > row_inds_;

  std::vector<int> jT_M_;
  
  std::vector<DataType> limited_vals_;
  
  VectorType R_plus_;
  VectorType R_minus_;
  
  const VectorSpace<DataType, DIM> * space_;
  const MatrixType* lmass_;
  const VectorType* vec_;
  int nrows_local_;
  int row_begin_;
  bool do_prelimiting_;
  
  const CPU_CSR_lMatrix< DataType > *lmass_diag_;
  const CPU_CSR_lMatrix< DataType > *lmass_offdiag_;
  
  bool upwind_biased_;
  bool initialized_;
  bool parallel_;
};
                          
template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::init(const VectorSpace<DataType, DIM>* space,
                                 const MatrixType& lumped_mass,
                                 const VectorType* vec,
                                 const std::vector<int>& dirichlet_dofs)
{
  assert (vec != nullptr);
  assert (space != nullptr);
  
  this->clear();
  
  this->space_ = space;
  this->vec_ = vec;
    
  this->R_plus_.CloneFromWithoutContent(*vec);
  this->R_minus_.CloneFromWithoutContent(*vec);
  
  this->parallel_ = (lumped_mass.comm_size() > 1);
    
  this->nrows_local_ = vec_->size_local();
  this->row_begin_ = vec_->ownership_begin();
  
  this->P_p_.resize(nrows_local_, 0.);
  this->P_m_.resize(nrows_local_, 0.);
  this->Q_p_.resize(nrows_local_, 0.);
  this->Q_m_.resize(nrows_local_, 0.);
  this->R_p_.resize(nrows_local_, 0.);
  this->R_m_.resize(nrows_local_, 0.);
  
  this->dirichlet_dofs_ = dirichlet_dofs;
  this->dirichlet_one_.resize(dirichlet_dofs.size(), 1.0);
  
  this->lmass_diag_ = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(lumped_mass.diagonalPtr());
  this->lmass_offdiag_ = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(lumped_mass.offdiagonalPtr());
  
  assert(this->lmass_diag_  != nullptr);
  assert(this->lmass_offdiag_ != nullptr);
  
  this->row_inds_.resize(nrows_local_, row_begin_);
  for (int i = 0; i < nrows_local_; ++i) 
  {
    this->row_inds_[i] += i;
  }
  
  this->lmass_diag_vals_.resize(nrows_local_, 0.);
  
  lmass_diag_->extract_diagelements(0, nrows_local_, &(lmass_diag_vals_[0]));
    
  this->limited_vals_.clear();
  this->limited_vals_.resize(this->nrows_local_, 0.);
  
  const int nnz_diag = this->lmass_diag_->get_nnz();
  
  this->jT_M_.clear();
  this->jT_M_.resize(nnz_diag, -1);
  int* ptr = &(jT_M_[0]);
  la::get_transposed_ptr<DataType, int> (this->lmass_diag_->matrix,
                                         nrows_local_, nrows_local_, nnz_diag,
                                         false,
                                         ptr);
                                           
  this->initialized_ = true;
}

template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::clear()
{
  this->space_ = nullptr;
  this->lmass_ = nullptr;
  this->vec_ = nullptr;
  this->lmass_diag_ = nullptr;
  this->lmass_offdiag_ = nullptr;
  
  this->nrows_local_ = -1;
  this->row_begin_ = -1;
  
  this->P_p_.clear();
  this->P_m_.clear();
  this->Q_p_.clear();
  this->Q_m_.clear();
  this->R_p_.clear();
  this->R_m_.clear();
  this->dirichlet_dofs_.clear();
  this->dirichlet_one_.clear();
  
  this->row_inds_.clear();
  this->lmass_diag_vals_.clear();
  this->parallel_ = false;
  this->initialized_ = false;
  
}

template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::update_bounds (int i, DataType flux, DataType diff_u)
{     
  this->P_p_[i] += std::max(0., flux);
  this->P_m_[i] += std::min(0., flux);
        
  if (diff_u > this->Q_p_[i])
  {
    this->Q_p_[i] = diff_u;
  }
  if (diff_u < this->Q_m_[i])
  {
    this->Q_m_[i] = diff_u;
  }
}
                    
template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::update_limiter (const int k,
                                            bool i_is_upwind,
                                            const DataType flux,
                                            const DataType Rp_i, const DataType Rp_j,
                                            const DataType Rm_i, const DataType Rm_j,
                                            std::vector<DataType>& alpha )
{
  assert (k >= 0);
  assert (k < alpha.size());
  
  if (!this->upwind_biased_)
  {
    if (flux > 0.)
    {
      alpha[k] = std::min(Rp_i, Rm_j);
    }
    else 
    {
      alpha[k] = std::min(Rm_i, Rp_j);
    }
  }
  else
  {
    if (flux > 0.)
    {
      if (i_is_upwind)
      {
        alpha[k] = Rp_i;
      }
      else
      {
        alpha[k] = Rp_j;
      }
    }
    else //if (flux < 0.)
    {
      if (i_is_upwind)
      {
        alpha[k] = Rm_i;
      }
      else
      {
        alpha[k] = Rm_j;
      }
    }
  }
  // To test recovering Galerkin FEM, uncomment this line
  // alpha_diag[k] = 1.0;
  // For pure artificial diffusion (low-order scheme), uncomment this line
  // alpha_diag[k] = 0.0;
}

template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::compute_residual_flux (const MatrixType& mass_matrix,
                                                   const MatrixType& diff_matrix,
                                                   const VectorType& mass_vec,
                                                   const VectorType& diff_vec,
                                                   const DataType delta_t,
                                                   std::vector<DataType>& flux_diag,
                                                   std::vector<DataType>& flux_offdiag)
{ 
  assert (this->initialized_);
  
  const la::lVector< DataType > & mass_local = mass_vec.interior();
  const la::lVector< DataType > & mass_ghost = mass_vec.ghost();
  const la::lVector< DataType > & diff_local = diff_vec.interior();
  const la::lVector< DataType > & diff_ghost = diff_vec.ghost();
  
  const CPU_CSR_lMatrix< DataType > * mass_diag 
    = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(mass_matrix.diagonalPtr());
  const CPU_CSR_lMatrix< DataType > * mass_offdiag 
    = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(mass_matrix.offdiagonalPtr());
  
  const CPU_CSR_lMatrix< DataType > * diff_diag 
    = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(diff_matrix.diagonalPtr());
  const CPU_CSR_lMatrix< DataType > * diff_offdiag 
    = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(diff_matrix.offdiagonalPtr());

  assert (mass_diag != nullptr);
  assert (mass_offdiag != nullptr);
  assert (diff_diag != nullptr);
  assert (diff_offdiag != nullptr);
  
  const int nnz_diag = mass_diag->get_nnz();
  const int nnz_offdiag = mass_offdiag->get_nnz();
  
  flux_diag.clear();
  flux_offdiag.clear();
  flux_diag.resize(nnz_diag, 0);
  flux_offdiag.resize(nnz_offdiag, 0);
    
  // loop over i = local row
  for (int i = 0, e_i = mass_diag->get_num_row(); i != e_i; ++i) 
  { 
    DataType v_i = 0.;
    DataType u_tilde_i = 0.;
    mass_local.GetValues(&i, 1, &v_i);
    diff_local.GetValues(&i, 1, &u_tilde_i);
    
    // loop over j = local col
    for (int k_j = mass_diag->matrix_row(i), 
         e_j = mass_diag->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      const int j = mass_diag->matrix_col(k_j);
      
      DataType v_j = 0.;
      DataType u_tilde_j = 0.;
      mass_local.GetValues(&j, 1, &v_j);
      diff_local.GetValues(&j, 1, &u_tilde_j);
      
      flux_diag[k_j] = delta_t * (  mass_diag->matrix_val(k_j) * (v_i - v_j) 
                                  - diff_diag->matrix_val(k_j) * (u_tilde_i - u_tilde_j));
    }

    if (!parallel_)
    {
      continue;
    }
    // loop over j = ghost col
    for (int k_j = mass_offdiag->matrix_row(i), 
         e_j = mass_offdiag->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      const int j = mass_offdiag->matrix_col(k_j);
      
      DataType v_j = 0.;
      DataType u_tilde_j = 0.;
      mass_ghost.GetValues(&j, 1, &v_j);
      diff_ghost.GetValues(&j, 1, &u_tilde_j);
      
      flux_offdiag[k_j] = delta_t * (  mass_offdiag->matrix_val(k_j) * (v_i - v_j) 
                                      - diff_offdiag->matrix_val(k_j) * (u_tilde_i - u_tilde_j));
    }  
  }
}

template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::compute_residual_flux (const MatrixType& mass_matrix,
                                                   const MatrixType& diff_matrix,
                                                   const VectorType& mass_vec_1,
                                                   const VectorType& mass_vec_2,
                                                   const VectorType& diff_vec_1,
                                                   const VectorType& diff_vec_2,
                                                   const DataType delta_t,
                                                   const DataType theta,
                                                   std::vector<DataType>& flux_diag,
                                                   std::vector<DataType>& flux_offdiag)
{ 
  assert (this->initialized_);
  
  const la::lVector< DataType > & mass1_local = mass_vec_1.interior();
  const la::lVector< DataType > & mass1_ghost = mass_vec_1.ghost();
  const la::lVector< DataType > & mass2_local = mass_vec_2.interior();
  const la::lVector< DataType > & mass2_ghost = mass_vec_2.ghost();
  const la::lVector< DataType > & diff1_local = diff_vec_1.interior();
  const la::lVector< DataType > & diff1_ghost = diff_vec_1.ghost();
  const la::lVector< DataType > & diff2_local = diff_vec_2.interior();
  const la::lVector< DataType > & diff2_ghost = diff_vec_2.ghost();
  
  const CPU_CSR_lMatrix< DataType > * mass_diag 
    = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(mass_matrix.diagonalPtr());
  const CPU_CSR_lMatrix< DataType > * mass_offdiag 
    = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(mass_matrix.offdiagonalPtr());
  
  const CPU_CSR_lMatrix< DataType > * diff_diag 
    = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(diff_matrix.diagonalPtr());
  const CPU_CSR_lMatrix< DataType > * diff_offdiag 
    = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(diff_matrix.offdiagonalPtr());

  assert (mass_diag != nullptr);
  assert (mass_offdiag != nullptr);
  assert (diff_diag != nullptr);
  assert (diff_offdiag != nullptr);
  
  const int nnz_diag = mass_diag->get_nnz();
  const int nnz_offdiag = mass_offdiag->get_nnz();
  
  flux_diag.clear();
  flux_offdiag.clear();
  flux_diag.resize(nnz_diag, 0.);
  flux_offdiag.resize(nnz_offdiag, 0.);
    
  // loop over i = local row
  for (int i = 0, e_i = mass_diag->get_num_row(); i != e_i; ++i) 
  { 
    DataType v1_i = 0.;
    DataType v2_i = 0.;
    DataType u1_i = 0.;
    DataType u2_i = 0.;
    mass1_local.GetValues(&i, 1, &v1_i);
    mass2_local.GetValues(&i, 1, &v2_i);
    diff1_local.GetValues(&i, 1, &u1_i);
    diff2_local.GetValues(&i, 1, &u2_i);
    
    // loop over j = local col
    for (int q = mass_diag->matrix_row(i), 
         e_j = mass_diag->matrix_row(i+1);
         q != e_j; ++q) 
    {
      const int j = mass_diag->matrix_col(q);
      const int qT_j = this->jT_M_[q];
      assert (qT_j >= 0);
      assert (mass_diag->matrix_col(qT_j) == i);
          
      if (j > i)
      {
        DataType v1_j = 0.;
        DataType v2_j = 0.;
        DataType u1_j = 0.;
        DataType u2_j = 0.;
        mass1_local.GetValues(&j, 1, &v1_j);
        mass2_local.GetValues(&j, 1, &v2_j);
        diff1_local.GetValues(&j, 1, &u1_j);
        diff2_local.GetValues(&j, 1, &u2_j);
      
        const DataType f_ij = mass_diag->matrix_val(q) * ((v1_i - v1_j) - (v2_i - v2_j)) /  delta_t
                            + diff_diag->matrix_val(q) * (theta*(u1_i - u1_j) + (1.-theta)*(u2_i - u2_j));
        
        flux_diag[q] = f_ij;
        
        flux_diag[qT_j] = -f_ij; 
      }
    }

    if (!parallel_)
    {
      continue;
    }
    // loop over j = ghost col
    for (int q = mass_offdiag->matrix_row(i), 
         e_j = mass_offdiag->matrix_row(i+1);
         q != e_j; ++q) 
    {
      const int j = mass_offdiag->matrix_col(q);
      
      DataType v1_j = 0.;
      DataType v2_j = 0.;
      DataType u1_j = 0.;
      DataType u2_j = 0.;
      mass1_ghost.GetValues(&j, 1, &v1_j);
      mass2_ghost.GetValues(&j, 1, &v2_j);
      diff1_ghost.GetValues(&j, 1, &u1_j);
      diff2_ghost.GetValues(&j, 1, &u2_j);
      
      flux_offdiag[q] = mass_offdiag->matrix_val(q) * ((v1_i - v1_j) - (v2_i - v2_j)) /  delta_t
                        + diff_offdiag->matrix_val(q) * (theta*(u1_i - u1_j) + (1.-theta)*(u2_i - u2_j));
    }  
  }
}

template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::prelimit_flux_std (const VectorType& sol,
                                               const std::vector<DataType>& flux_diag,
                                               const std::vector<DataType>& flux_offdiag,
                                               std::vector<DataType>& prelim_flux_diag,
                                               std::vector<DataType>& prelim_flux_offdiag)
{  
  assert (this->initialized_);
  
  const la::lVector< DataType > & u_local = sol.interior();
  const la::lVector< DataType > & u_ghost = sol.ghost();
  
  prelim_flux_diag = flux_diag;
  prelim_flux_offdiag = flux_offdiag;

  // compute P_(plus/minus) and Q_(plus / minus)
  // loop over i = local row
  for (int i = 0; i < nrows_local_; ++i) 
  { 
    DataType u_tilde_i = 0.;
    u_local.GetValues(&i, 1, &u_tilde_i);
    
    // loop over j = local col
    for (int k_j = lmass_diag_->matrix_row(i), 
         e_j = lmass_diag_->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      const int j = lmass_diag_->matrix_col(k_j);
      DataType u_tilde_j = 0.;
      u_local.GetValues(&j, 1, &u_tilde_j);
            
      if (i != j)
      {
        // TODO: checks sign
        const DataType diff_u_tilde = u_tilde_j - u_tilde_i;
        if (flux_diag[k_j] * diff_u_tilde > 0.)
        {
          prelim_flux_diag[k_j] = 0.;
        }
      }
    }
    
    if (!parallel_)
    {
      continue;
    }
    // loop over j = ghost col
    for (int k_j = lmass_offdiag_->matrix_row(i), 
         e_j = lmass_offdiag_->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      const int j = lmass_offdiag_->matrix_col(k_j);
      DataType u_tilde_j = 0.;
      u_ghost.GetValues(&j, 1, &u_tilde_j);
      
      // TODO: check sign
      const DataType diff_u_tilde = u_tilde_j - u_tilde_i;
      if (flux_offdiag[k_j] * diff_u_tilde > 0.)
      {
        prelim_flux_offdiag[k_j] = 0.;
      }
    }  
  }
}

template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::compute_limiter_zalesak (const VectorType& sol,
                                                     const bool upwind_biased,
                                                     const DataType delta_t,
                                                     const std::vector<DataType>& flux_diag,
                                                     const std::vector<DataType>& flux_offdiag,
                                                     const std::vector<bool>& edge_orient_diag,
                                                     const std::vector<bool>& edge_orient_offdiag,
                                                     std::vector<DataType>& alpha_diag,
                                                     std::vector<DataType>& alpha_offdiag)
{  
  assert (this->initialized_);
  
  this->upwind_biased_ = upwind_biased;
  
  this->R_plus_.Zeros();
  this->R_minus_.Zeros();
  
  this->P_p_.clear();
  this->P_m_.clear();
  this->Q_p_.clear();
  this->Q_m_.clear();
  this->R_p_.clear();
  this->R_m_.clear();
  
  this->P_p_.resize(nrows_local_, 0.);
  this->P_m_.resize(nrows_local_, 0.);
  this->Q_p_.resize(nrows_local_, 0.);
  this->Q_m_.resize(nrows_local_, 0.);
  this->R_p_.resize(nrows_local_, 0.);
  this->R_m_.resize(nrows_local_, 0.);
  
  const la::lVector< DataType > & u_local = sol.interior();
  const la::lVector< DataType > & u_ghost = sol.ghost();
  
  assert (lmass_diag_ != nullptr);
  assert (lmass_offdiag_ != nullptr);
  assert (u_local.get_size() == lmass_diag_->get_num_row());
  assert (u_ghost.get_size() == lmass_offdiag_->get_num_col());
  assert (flux_diag.size() == lmass_diag_->get_nnz());
  assert (flux_offdiag.size() == lmass_offdiag_->get_nnz());
  assert (edge_orient_diag.size() == flux_diag.size());
  assert (edge_orient_offdiag.size() == flux_offdiag.size());
  
  // compute P_(plus/minus) and Q_(plus / minus)
  // loop over i = local row
  for (int i = 0; i < nrows_local_; ++i) 
  { 
    DataType u_tilde_i = 0.;
    u_local.GetValues(&i, 1, &u_tilde_i);
    
    // loop over j = local col
    for (int k_j = lmass_diag_->matrix_row(i), 
         e_j = lmass_diag_->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      const int j = lmass_diag_->matrix_col(k_j);
      DataType u_tilde_j = 0.;
      u_local.GetValues(&j, 1, &u_tilde_j);
            
      if (i != j)
      {
        // TODO: check sign
        const DataType diff_u_tilde = u_tilde_j - u_tilde_i;
       
        this->update_bounds(i, flux_diag[k_j], diff_u_tilde);
      }
    }
    if (!parallel_)
    {
      continue;
    }
    // loop over j = ghost col
    for (int k_j = lmass_offdiag_->matrix_row(i), 
         e_j = lmass_offdiag_->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      const int j = lmass_offdiag_->matrix_col(k_j);
      DataType u_tilde_j = 0.;
      u_ghost.GetValues(&j, 1, &u_tilde_j);
      
      // TODO: check sign
      const DataType diff_u_tilde = u_tilde_j - u_tilde_i;
       
      this->update_bounds(i, flux_offdiag[k_j], diff_u_tilde);
    }  
  }
  
  // scale by lumped mass matrix 
  this->rplus_vals_.clear();
  this->rplus_vals_.resize(nrows_local_, 1.);
  this->rminus_vals_.clear();
  this->rminus_vals_.resize(nrows_local_, 1.);
  
  for (int i = 0; i < nrows_local_; ++i) 
  {
    const DataType m_i = this->lmass_diag_vals_[i];
    
    assert (this->P_p_[i] >= 0.);
    assert (this->Q_p_[i] >= 0.);
    assert (this->P_m_[i] <= 0.);
    assert (this->Q_m_[i] <= 0.);
    assert (m_i > 0.);
    assert (delta_t > 0.);
    
    if (delta_t * this->P_p_[i] > 0.)
    {
      const DataType frac_rp = m_i * this->Q_p_[i] / (delta_t * this->P_p_[i]);
      if (frac_rp < 1.) 
      {
        this->rplus_vals_[i] = frac_rp;
      }
    }
    if (delta_t * this->P_m_[i] < 0.)
    {
      const DataType frac_rm = m_i * this->Q_m_[i] / (delta_t * this->P_m_[i]);
      if (frac_rm < 1.) 
      {
        this->rminus_vals_[i] = frac_rm;
      }
    }
    //std::cout << this->rplus_vals_[i] << std::endl;
  }
  
  // build vectors R_plus and R_minus
  R_plus_.SetValues(&(this->row_inds_[0]), nrows_local_, &(rplus_vals_[0]));
  R_minus_.SetValues(&(this->row_inds_[0]), nrows_local_, &(rminus_vals_[0]));
  
  if (!this->dirichlet_dofs_.empty()) 
  {
    R_plus_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_one_));
    R_minus_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_one_));
  }

  R_plus_.Update();
  R_minus_.Update();

  // compute weights alpha = limiter
  alpha_diag.clear();
  alpha_offdiag.clear();
  
#ifdef DBG_FULL_ANTIDIFF
  alpha_diag.resize(flux_diag.size(), 1.);
  alpha_offdiag.resize(flux_offdiag.size(), 1.);
  return;
#else
#ifdef DBG_NO_ANTIDIFF
  alpha_diag.resize(flux_diag.size(), 0.);
  alpha_offdiag.resize(flux_offdiag.size(), 0.);
  return;
#endif
#endif

  alpha_diag.resize(flux_diag.size(), 0.);
  alpha_offdiag.resize(flux_offdiag.size(), 0.);
  
  const la::lVector< DataType > & Rp_local = R_plus_.interior();
  const la::lVector< DataType > & Rp_ghost = R_plus_.ghost();
  const la::lVector< DataType > & Rm_local = R_minus_.interior();
  const la::lVector< DataType > & Rm_ghost = R_minus_.ghost();
    
  for (int i = 0; i < nrows_local_; ++i) 
  { 
    DataType Rp_i = 0.;
    DataType Rm_i = 0.;
    Rp_local.GetValues(&i, 1, &Rp_i);
    Rm_local.GetValues(&i, 1, &Rm_i);
    
    // loop over j = local col
    for (int k_j = lmass_diag_->matrix_row(i), 
         e_j = lmass_diag_->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      const int j = lmass_diag_->matrix_col(k_j);

      DataType Rp_j = 0.;
      DataType Rm_j = 0.;
      Rp_local.GetValues(&j, 1, &Rp_j);
      Rm_local.GetValues(&j, 1, &Rm_j);
    
      const bool i_is_upwind = edge_orient_diag[k_j];
      this->update_limiter(k_j, i_is_upwind, flux_diag[k_j],  Rp_i, Rp_j, Rm_i, Rm_j, alpha_diag); 
    }
    if (!parallel_)
    {
      continue;
    }
    
    // loop over j = ghost col
    for (int k_j = lmass_offdiag_->matrix_row(i), 
         e_j = lmass_offdiag_->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      const int j = lmass_offdiag_->matrix_col(k_j);

      DataType Rp_j = 0.;
      DataType Rm_j = 0.;
      Rp_ghost.GetValues(&j, 1, &Rp_j);
      Rm_ghost.GetValues(&j, 1, &Rm_j);
    
      const bool i_is_upwind = edge_orient_offdiag[k_j];
      this->update_limiter(k_j, i_is_upwind, flux_offdiag[k_j], Rp_i, Rp_j, Rm_i, Rm_j, alpha_offdiag); 
    }
  }
}

template <class LAD, int DIM>
void FluxLimiter<LAD, DIM>::apply_limiter (VectorType& anti_diff,
                                           const std::vector<DataType>& flux_diag,
                                           const std::vector<DataType>& flux_offdiag,
                                           const std::vector<DataType>& alpha_diag,
                                           const std::vector<DataType>& alpha_offdiag)
{ 
  anti_diff.Zeros();
  
  for (int i = 0; i < this->nrows_local_; ++i) 
  {   
    DataType val = 0.0;
    
    // loop over diagonal part
    for (int k_j = lmass_diag_->matrix_row(i), 
         e_j = lmass_diag_->matrix_row(i+1);
         k_j != e_j; ++k_j) 
    {
      val += flux_diag[k_j] * alpha_diag[k_j];
    }
    
    // loop over offdiagonal part
    if (parallel_)
    {
      for (int k_j = lmass_offdiag_->matrix_row(i), 
           e_j = lmass_offdiag_->matrix_row(i+1);
           k_j != e_j; ++k_j) 
      {
        val += flux_offdiag[k_j] * alpha_offdiag[k_j];
      }
    }
    
    this->limited_vals_[i] = val;
  }
  
  
  anti_diff.SetValues(&(this->row_inds_[0]), this->nrows_local_, &(this->limited_vals_[0]));
  anti_diff.Update();
}

} // namespace doffem
} // namespace hiflow
#endif
