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

#ifndef __FEM_AFC_TOOLS_H_
#define __FEM_AFC_TOOLS_H_

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <cmath>

#include "common/macros.h"
#include "common/vector_algebra.h"
#include "common/log.h"
#include "dof/dof_fem_types.h"
#include "../../linear_algebra/lmp/lmatrix_csr_cpu.h"

namespace hiflow {

namespace la {
template <class DataType> class SeqDenseMatrix;
}

namespace doffem {

template <class LAD>
void create_lumped_matrix(const typename LAD::MatrixType& input,
                          typename LAD::MatrixType& output)
{ 
  std::vector<typename LAD::DataType> local_row_sums(input.num_rows_local(), 0.);
  input.GetLocalRowSums(local_row_sums);
  
  output.CloneFromWithoutContent(input);
  output.Zeros();
  output.SetDiagValues(local_row_sums);
  output.Update();
}
                          
template <class LAD>
class FluxMatrixCreator
{
  typedef typename LAD::DataType DataType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::MatrixType MatrixType;
  
public:
  FluxMatrixCreator()
  {
    this->clear();
  }
   
   ~FluxMatrixCreator()
  {}
   
  void clear()
  {
    this->nrows_local_ = 0;
    this->ncols_ghost_ = 0;
    this->nnz_diag_ = 0;
    this->nnz_offdiag_ = 0;
    
    this->jT_K_.clear();
    this->jT_oK_.clear();
    
    this->initialized_ = false;
  }
   
  void init(const MatrixType& K)
  {
    this->clear();
    const bool parallel = (K.comm_size() > 1);
    
    const CPU_CSR_lMatrix< DataType > * K_diag 
      = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(K.diagonalPtr());
  
    const CPU_CSR_lMatrix< DataType > * K_offdiag 
      = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(K.offdiagonalPtr());
  
    const CPU_CSR_lMatrix< DataType > * K_coloffdiag 
      = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(K.col_offdiagonalPtr());

    assert (K_diag != nullptr);
    assert (K_offdiag != nullptr);
    assert (K_coloffdiag != nullptr);
    
    this->nrows_local_ = K_diag->get_num_row();
    this->ncols_ghost_ = K_offdiag->get_num_col();
    this->nnz_diag_ = K_diag->get_nnz();
    this->nnz_offdiag_ = K_offdiag->get_nnz();
  
    // transposed entry pointer for diagonal
    this->jT_K_.resize(this->nnz_diag_, -1);
    int* ptr = &(jT_K_[0]);
    la::get_transposed_ptr<DataType, int> (K_diag->matrix,
                                           nrows_local_, nrows_local_, nnz_diag_,
                                           false,
                                           ptr);
                    
    // transposed entry pointer for offdiagonal
    this->jT_oK_.resize(this->nnz_offdiag_, -1);
    
    const int nrows_offdiag = K_offdiag->get_num_row();
    for (int i=0; i!= nrows_offdiag; ++i)
    {
      // i = row
      const int i_first_nz = K_offdiag->matrix_row(i);
      const int i_last_nz = K_offdiag->matrix_row(i+1);
    
      // loop over cols of row i
      for (int c=i_first_nz; c!=i_last_nz; ++c)
      {
        // j = col 
        const int j = K_offdiag->matrix_col(c);

        // upper triangle
        // search index of transposed entry
        const int j_first_nz = K_coloffdiag->matrix_row(j);
        const int j_last_nz = K_coloffdiag->matrix_row(j+1);
        
        // loop over cols of row j
        for (int k=j_first_nz; k!=j_last_nz; ++k)
        {
          const int kj = K_coloffdiag->matrix_col(k);
          if (kj == i)
          {
            this->jT_oK_[c] = k;
            break;
          }
        }
      }
    }
    
    this->initialized_ = true;
  }

  void compute_L_and_D (const MatrixType& K,
                        MatrixType& L,
                        MatrixType& D,
                        bool reverse_sign,
                        std::vector<bool>& edge_orient_diag,
                        std::vector<bool>& edge_orient_offdiag)
  {
    if (!this->initialized_)
    {
      this->init(K);
    }
    const CPU_CSR_lMatrix< DataType > * K_diag 
      = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(K.diagonalPtr());
  
    const CPU_CSR_lMatrix< DataType > * K_offdiag 
      = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(K.offdiagonalPtr());
  
    const CPU_CSR_lMatrix< DataType > * K_coloffdiag 
      = dynamic_cast< const CPU_CSR_lMatrix< DataType > * >(K.col_offdiagonalPtr());

    assert (K_diag != nullptr);
    assert (K_offdiag != nullptr);
    assert (K_coloffdiag != nullptr);
    
    CPU_CSR_lMatrix< DataType > * L_diag 
      = dynamic_cast< CPU_CSR_lMatrix< DataType > * >(L.diagonalPtr());
    CPU_CSR_lMatrix< DataType > * L_offdiag 
      = dynamic_cast< CPU_CSR_lMatrix< DataType > * >(L.offdiagonalPtr());

    assert (L_diag != nullptr);
    assert (L_offdiag != nullptr);
    
    CPU_CSR_lMatrix< DataType > * D_diag 
      = dynamic_cast< CPU_CSR_lMatrix< DataType > * >(D.diagonalPtr());
    CPU_CSR_lMatrix< DataType > * D_offdiag 
      = dynamic_cast< CPU_CSR_lMatrix< DataType > * >(D.offdiagonalPtr());
    
    assert (D_diag != nullptr);
    assert (D_offdiag != nullptr);
    
    DataType* D_val_diag = D_diag->matrix_val();
    DataType* D_val_offdiag = D_offdiag->matrix_val();
    DataType* L_val_diag = L_diag->matrix_val();
    DataType* L_val_offdiag = L_offdiag->matrix_val();
  
    // compute diffusion matrix D_ij = max(-k_ij, 0 , -k_ji)
    // and low order matrix L = K + D
     
    edge_orient_diag.clear();
    edge_orient_offdiag.clear();
    edge_orient_diag.resize(this->nnz_diag_, true);
    edge_orient_offdiag.resize(this->nnz_offdiag_, true);
    
    // diagonal part
    const int* i_K = K_diag->matrix_row();
    const int* j_K = K_diag->matrix_col();
    const int* jD_K = K_diag->diag_ptr();
    const DataType* v_K = K_diag->matrix_val();
  
    DataType sign = 1.;
    if (reverse_sign)
    {
      sign = -1;
    }
    
    // L = K
    for (int q = 0; q!= nnz_diag_; ++q)
    {
      D_val_diag[q] = 0.;
      L_val_diag[q] = v_K[q];
    }
   
    // calculate D, L += D
    for (int i = 0; i != nrows_local_; ++i)
    {
      const int qD_i = jD_K[i];
      assert (i == j_K[qD_i]);
    
      for (int q = i_K[i], eq = i_K[i+1]; q != eq; ++q)
      {
        // column 
        const int j  = j_K[q];
        const int qD_j = jD_K[j];
        assert (qD_j >= 0);
        assert (j == j_K[qD_j]);
        
        // process upper triangular matrix only
        if (j > i)
        {
          const int qT = jT_K_[q];          
          //std::cout << i << ", " << j << " : " << qT << std::endl;
          assert (qT >= 0);
          assert (j_K[qT] == i);
                    
          const DataType val_K  = sign * v_K[q];   // K_ij
          const DataType val_KT = sign * v_K[qT];   // K_ji
        
          // TODO: korrekt?
          if (val_KT > val_K)
          {
            edge_orient_diag[q] = false;
          }
          else
          {
            edge_orient_diag[qT] = false; 
          }
          
          DataType d_ij = 0.;
          if (-val_K > d_ij)
          {
            d_ij = -val_K;
          }
          if (-val_KT > d_ij)
          {
            d_ij = -val_KT;
          }
          assert (d_ij >= 0.);
          
          //if (val_K < 0. || val_KT < 0.)
          //{
          //  const DataType d_ij = sign * std::max(-val_K, -val_KT); 
          D_val_diag[q] = d_ij;
          D_val_diag[qT] = d_ij;
            
          D_val_diag[qD_i] -= d_ij;
          D_val_diag[qD_j] -= d_ij;
            
          L_val_diag[q]    += d_ij; // L_ij += d_ij
          L_val_diag[qT]   += d_ij; // L_ji += d_ij
          L_val_diag[qD_i] -= d_ij; // L_ii -= d_ij 
          L_val_diag[qD_j] -= d_ij; // L_jj -= d_ij
            
          assert ( L_val_diag[q] >= 0.);
          assert ( L_val_diag[qT] >= 0.);
          //}
        }
      }
    }
    
    // offdiagonal part
    const int* i_oK = K_offdiag->matrix_row();
    const int* j_oK = K_offdiag->matrix_col();
    const DataType* v_oK = K_offdiag->matrix_val();
    const DataType* v_coK = K_coloffdiag->matrix_val();
    const int nrows_offdiag = K_offdiag->get_num_row();
    
    // L = K
    for (int q = 0; q!= nnz_offdiag_; ++q)
    {
      D_val_offdiag[q] = 0.;
      L_val_offdiag[q] = v_oK[q];
    }
     
    // calculate D, L += D
    for (int i = 0; i != nrows_offdiag; ++i)
    {
      const int qD_i = jD_K[i];

      for (int q = i_oK[i], eq = i_oK[i+1]; q != eq; ++q)
      {
        // column 
        const int j  = j_oK[q];
        
        const int qT = jT_oK_[q];
        assert (qT >= 0);
        const DataType val_K  = sign * v_oK[q];   // K_ij
        const DataType val_KT = sign * v_coK[qT];  // K_ji

        // TODO: korrekt?        
        if (val_KT > val_K)
        {
          edge_orient_offdiag[q] = false;
        }
        
        DataType d_ij = 0.;
        if (-val_K > d_ij)
        {
          d_ij = -val_K;
        }
        if (-val_KT > d_ij)
        {
          d_ij = -val_KT;
        }
        assert (d_ij >= 0.);
            
        //if (val_K < 0. || val_KT < 0.)
        //{
        //  const DataType d_ij = sign * std::max(-val_K, -val_KT); 
            
        D_val_offdiag[q] = d_ij;
            
        D_val_diag[qD_i] -= d_ij;
        
        L_val_offdiag[q] += d_ij; // L_ij += d_ij
          
        L_val_diag[qD_i] -= d_ij; // L_ii -= d_ij 
        //}
      }
    }
  
    L.Update();
    D.Update();
#ifndef NDEBUG
    std::vector<DataType> D_row_sums;
    D.GetLocalRowSums(D_row_sums);
    
    for (int i=0; i != D_row_sums.size(); ++i)
    {
      assert (std::abs(D_row_sums[i]) < 1e-10); 
    }
#endif
  }
  
protected:
  std::vector<int> jT_K_;
  std::vector<int> jT_oK_;
  
  int nrows_local_;
  int ncols_ghost_;
  int nnz_diag_;
  int nnz_offdiag_;
  
  bool initialized_;

};



} // namespace doffem
} // namespace hiflow
#endif
