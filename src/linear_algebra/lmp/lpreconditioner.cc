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

/// @author Dimitar Lukarski

#include "lpreconditioner.h"
#include "lmatrix.h"
#include "lmatrix_csr_cpu.h"
#include "lmp_log.h"
#include "lmatrix_csr_cpu.h"
#include "lmatrix_dense_cpu.h"
#include "lvector_cpu.h"

#include "solvers/cg.h"
#include "common/macros.h"
#include <assert.h>

namespace hiflow {
namespace la {

// Class lPreconditioner

template < typename ValueType >
lPreconditioner< ValueType >::lPreconditioner() {
  this->Operator_ = nullptr;
  this->precond_name_ = "";
  this->reuse_ = true;
  this->modified_op_ = false;
  this->state_ = false;
}

template < typename ValueType >
lPreconditioner< ValueType >::~lPreconditioner() {}

template < typename ValueType >
void lPreconditioner< ValueType >::SetupOperator(
    const hiflow::la::lMatrix< ValueType > &op) {
  this->Operator_ = &op;
  this->SetModifiedOperator(true);
}

template < typename ValueType > void lPreconditioner< ValueType >::Build(const hiflow::la::lVector< ValueType > *in, 
                                                                         const hiflow::la::lVector< ValueType > *out) 
{
  this->SetState(true);
  this->SetModifiedOperator(false);
}

template < typename ValueType >
void lPreconditioner< ValueType >::print(std::ostream &out) const {

  LOG_INFO("lPreconditioner", this->precond_name_);
}

template < typename ValueType >
void lPreconditioner< ValueType >::PermuteBack(
    hiflow::la::lMatrix< ValueType > *op) const {
  // do nothing by default
}

template < typename ValueType >
void lPreconditioner< ValueType >::PermuteBack(
    hiflow::la::lVector< ValueType > *vec) const {
  // do nothing by default
}

template < typename ValueType > void lPreconditioner< ValueType >::Clear() {
  this->Operator_ = NULL;
  this->precond_name_ = "";
  this->reuse_ = false;
  this->modified_op_ = false;
  this->state_ = false;
}

// Class lPreconditioner_Jacobi

template < typename ValueType >
lPreconditioner_Jacobi< ValueType >::lPreconditioner_Jacobi()
    : lPreconditioner< ValueType >() {
  this->precond_name_ = "Jacobi";
}

template < typename ValueType >
lPreconditioner_Jacobi< ValueType >::~lPreconditioner_Jacobi() {}

template < typename ValueType >
void lPreconditioner_Jacobi< ValueType >::Init() {
}

template < typename ValueType >
void lPreconditioner_Jacobi< ValueType >::Init(const hiflow::la::lVector< ValueType > *diag) {
  assert (diag != nullptr);
  this->diag_ = diag;
}

template < typename ValueType >
void lPreconditioner_Jacobi< ValueType >::Clear() 
{
  LOG_INFO("Jacobi","Clear");

  if (this->inv_D_ != nullptr)
  {
    this->inv_D_->Clear();
    delete this->inv_D_;
    this->inv_D_ = nullptr;
  }
  this->diag_ = nullptr;
  
  lPreconditioner< ValueType >::Clear();
}

template < typename ValueType >
void lPreconditioner_Jacobi< ValueType >::Build(const hiflow::la::lVector< ValueType > *in, 
                                                const hiflow::la::lVector< ValueType > *out)  
{
  assert (in != nullptr);
  this->inv_D_ = in->CloneWithoutContent();
  
  if (this->diag_ != nullptr)
  {
    const auto size = this->diag_->get_size();
   
    assert (size == in->get_size());
    assert (size == out->get_size());

    for (int i=0; i != size; ++i)
    {
      assert (this->diag_->GetValue(i) != 0.);
      this->inv_D_->SetValue(i, 1. / this->diag_->GetValue(i));
      //std::cout << " " << this->diag_->GetValue(i);
    }
  }
  else 
  {
    assert(this->Operator_ != NULL);
    assert(this->Operator_->get_num_row() == this->Operator_->get_num_col());

    auto mat_cpu = dynamic_cast<CPU_CSR_lMatrix< ValueType > const *> (this->Operator_);
    if (mat_cpu != nullptr)
    {
      mat_cpu->extract_invdiagelements(0, mat_cpu->get_num_row(), this->inv_D_);
    }
    else 
    {
      auto mat_cpu_new = new CPUsimple_CSR_lMatrix< ValueType >;

      // Permute the Operator
      mat_cpu_new->CloneFrom(*this->Operator_);

      mat_cpu_new->extract_invdiagelements(0, mat_cpu_new->get_num_row(), this->inv_D_);

      mat_cpu_new->Clear();

      delete mat_cpu_new;
    }
  }

  /*
  for (int i=0; i!=this->inv_D_->get_size(); ++i )
  {
    std::cout << " " << this->inv_D_->GetValue(i);
  }
  */

  assert (!this->inv_D_->ContainsNaN());

  this->SetState(true);
  this->SetModifiedOperator(false);
}

template < typename ValueType >
void lPreconditioner_Jacobi< ValueType >::ApplylPreconditioner(
    const hiflow::la::lVector< ValueType > &input,
    hiflow::la::lVector< ValueType > *output) {
  if (!this->GetState()) {
    this->Build(&input, output);
  }

  assert (input.get_size() == output->get_size());
  assert (output->get_size() == this->inv_D_->get_size());
  
  // CPU sequential
  //  this->Operator_->Pjacobi(input, output);

  // Parallel
  output->CopyFrom(input);
  output->ElementWiseMult(*this->inv_D_);
}

// Class lPreconditioner_GaussSeidel

template < typename ValueType >
lPreconditioner_GaussSeidel< ValueType >::lPreconditioner_GaussSeidel()
    : lPreconditioner< ValueType >() {
  this->precond_name_ = "Gauss-Seidel";
}

template < typename ValueType >
lPreconditioner_GaussSeidel< ValueType >::~lPreconditioner_GaussSeidel() {}

template < typename ValueType >
void lPreconditioner_GaussSeidel< ValueType >::Init() {
  // do nothing - it's a matrix free preconditioner
}

template < typename ValueType >
void lPreconditioner_GaussSeidel< ValueType >::ApplylPreconditioner(
    const hiflow::la::lVector< ValueType > &input,
    hiflow::la::lVector< ValueType > *output) {
  assert(this->Operator_ != NULL);
  if (!this->GetState()) {
    this->Build(&input, output);
  }

  this->Operator_->Pgauss_seidel(input, output);
}

// Class lPreconditioner_SymmetricGaussSeidel

template < typename ValueType >
lPreconditioner_SymmetricGaussSeidel<
    ValueType >::lPreconditioner_SymmetricGaussSeidel()
    : lPreconditioner< ValueType >() {
  this->precond_name_ = "Symmetric Gauss-Seidel";
}

template < typename ValueType >
lPreconditioner_SymmetricGaussSeidel<
    ValueType >::~lPreconditioner_SymmetricGaussSeidel() {}

template < typename ValueType >
void lPreconditioner_SymmetricGaussSeidel< ValueType >::Init() {
  // do nothing - it's a matrix free preconditioner
}

template < typename ValueType >
void lPreconditioner_SymmetricGaussSeidel< ValueType >::ApplylPreconditioner(
    const hiflow::la::lVector< ValueType > &input,
    hiflow::la::lVector< ValueType > *output) {
  if (!this->GetState()) {
    this->Build(&input, output);
  }

  this->Operator_->Psgauss_seidel(input, output);
}

// Class lPreconditioner_BlocksSymmetricGaussSeidel

template < typename ValueType >
lPreconditioner_BlocksSymmetricGaussSeidel<
    ValueType >::lPreconditioner_BlocksSymmetricGaussSeidel()
    : lPreconditioner< ValueType >() {
  this->precond_name_ = "Block-wise Symmetric Gauss-Seidel";
  this->num_blocks_ = 1;
}

template < typename ValueType >
lPreconditioner_BlocksSymmetricGaussSeidel<
    ValueType >::~lPreconditioner_BlocksSymmetricGaussSeidel() {}

template < typename ValueType >
void lPreconditioner_BlocksSymmetricGaussSeidel< ValueType >::Init(
    const int num_blocks) {
  this->num_blocks_ = num_blocks;
}

template < typename ValueType >
void lPreconditioner_BlocksSymmetricGaussSeidel< ValueType >::Init(void) {
  this->Init(1);
}

template < typename ValueType >
void lPreconditioner_BlocksSymmetricGaussSeidel< ValueType >::
    ApplylPreconditioner(const hiflow::la::lVector< ValueType > &input,
                         hiflow::la::lVector< ValueType > *output) {
  if (!this->GetState()) {
    this->Build(&input, output);
  }

  this->Operator_->BlocksPsgauss_seidel(input, output, this->num_blocks_);
}

// Class lPreconditioner_SOR

template < typename ValueType >
lPreconditioner_SOR< ValueType >::lPreconditioner_SOR()
    : lPreconditioner< ValueType >() {
  this->precond_name_ = "SOR";
  this->relax_parameter_ = 1.0;
}

template < typename ValueType >
lPreconditioner_SOR< ValueType >::~lPreconditioner_SOR() {}

template < typename ValueType >
void lPreconditioner_SOR< ValueType >::Init(const ValueType omega) {
  this->relax_parameter_ = omega;
}

template < typename ValueType >
void lPreconditioner_SOR< ValueType >::Init(void) {
  this->Init(1.0);
}

template < typename ValueType >
void lPreconditioner_SOR< ValueType >::ApplylPreconditioner(
    const hiflow::la::lVector< ValueType > &input,
    hiflow::la::lVector< ValueType > *output) {
  if (!this->GetState()) {
    this->Build(&input, output);
  }

  this->Operator_->Psor(this->relax_parameter_, input, output);
}

// Class lPreconditioner_SSOR

template < typename ValueType >
lPreconditioner_SSOR< ValueType >::lPreconditioner_SSOR()
    : lPreconditioner< ValueType >() {
  this->precond_name_ = "Symmetric SOR";
  this->relax_parameter_ = 1.0;
}

template < typename ValueType >
lPreconditioner_SSOR< ValueType >::~lPreconditioner_SSOR() {}

template < typename ValueType >
void lPreconditioner_SSOR< ValueType >::Init(const ValueType omega) {
  this->relax_parameter_ = omega;
}

template < typename ValueType >
void lPreconditioner_SSOR< ValueType >::Init(void) {
  this->Init(1.0);
}

template < typename ValueType >
void lPreconditioner_SSOR< ValueType >::ApplylPreconditioner(
    const hiflow::la::lVector< ValueType > &input,
    hiflow::la::lVector< ValueType > *output) {
  if (!this->GetState()) {
    this->Build(&input, output);
  }

  this->Operator_->Pssor(this->relax_parameter_, input, output);
}

// Class lPreconditioner_ILUp

template < typename ValueType >
lPreconditioner_ILUp< ValueType >::lPreconditioner_ILUp()
    : lPreconditioner< ValueType >() {
  this->precond_name_ = "ILU(p)";
  this->ilu_p_ = 0;
  this->LU_ = NULL;
}

template < typename ValueType >
lPreconditioner_ILUp< ValueType >::~lPreconditioner_ILUp() {}

template < typename ValueType >
void lPreconditioner_ILUp< ValueType >::Init(int ilu_p) {
  this->ilu_p_ = ilu_p;
  assert(this->ilu_p_ >= 0);
}

template < typename ValueType >
void lPreconditioner_ILUp< ValueType >::Init(void) {
  this->Init(0);
}

template < typename ValueType >
void lPreconditioner_ILUp< ValueType >::Clear() {
  this->LU_->Clear();
  lPreconditioner< ValueType >::Clear();
}

template < typename ValueType >
void lPreconditioner_ILUp< ValueType >::Build(const hiflow::la::lVector< ValueType > *in, 
                                              const hiflow::la::lVector< ValueType > *out)   
{
  // make LU_ the same as Operator
  this->LU_ = this->Operator_->CloneWithoutContent();

  // make LU cpu matrix
  lMatrix< ValueType > *LU_cpu = new CPUsimple_CSR_lMatrix< ValueType >;

  LU_cpu->CloneFrom(*this->Operator_);

  // factorize cpu matrix
  if (this->ilu_p_ == 0) {
    LU_cpu->ilu0();
  } else {
    LU_cpu->ilup(this->ilu_p_);
  }

  this->LU_->CloneFrom(*LU_cpu);

  delete LU_cpu;
}

template < typename ValueType >
void lPreconditioner_ILUp< ValueType >::ApplylPreconditioner(
    const hiflow::la::lVector< ValueType > &input,
    hiflow::la::lVector< ValueType > *output) {
  if (!this->GetState()) {
    this->Build(&input, output);
  }

  this->LU_->ilu_solve(input, output);
}

// Class lPreconditioner_FSAI

template < typename ValueType >
lPreconditioner_FSAI<ValueType >::lPreconditioner_FSAI() {
  this->precond_name_ = "Approximate Inverse - FSAI";

  // Parameters
  this->solver_max_iter_ = 1000;
  this->solver_rel_eps_ = 1e-10;
  this->solver_abs_eps_ = 1e-10;
  this->matrix_power_ = 1;

  this->mult_tmp_ = nullptr;
  this->ext_matrix_pat_ = false;

  this->AI_L_ = nullptr;
  this->AI_Lt_ = nullptr;
}

template < typename ValueType >
lPreconditioner_FSAI<
    ValueType >::~lPreconditioner_FSAI() {
  if (this->mult_tmp_ != nullptr)
  {
    delete this->mult_tmp_;
  }
  if (this->AI_L_ != nullptr)
  {
    delete this->AI_L_;
  }
  if (this->AI_Lt_ != nullptr)
  {
    delete this->AI_Lt_;
  }

  lPreconditioner< ValueType >::Clear();
}

template < typename ValueType >
void lPreconditioner_FSAI< ValueType >::Init(
    const int solver_max_iter, const ValueType solver_rel_eps,
    const ValueType solver_abs_eps, const int matrix_power) 
{
  assert(solver_max_iter > 0);
  this->solver_max_iter_ = solver_max_iter;

  assert(solver_rel_eps > 0);
  this->solver_rel_eps_ = solver_rel_eps;

  assert(solver_abs_eps > 0);
  this->solver_abs_eps_ = solver_abs_eps;

  assert(matrix_power >= 1);
  this->matrix_power_ = matrix_power;

  char data_info[255];
  this->precond_name_ = "Approximate Inverse - FSAI";
  sprintf(data_info, "(%d)", matrix_power);
  this->precond_name_.append(data_info);
}

template < typename ValueType >
void lPreconditioner_FSAI< ValueType >::set_ext_matrix_pattern(const hiflow::la::lMatrix< ValueType > &mat) {

  if (this->matrix_pat_ != nullptr)
  {
    delete this->matrix_pat_;
  }
  
  this->matrix_pat_ = new CPUsimple_CSR_lMatrix< ValueType >;
  this->matrix_pat_->CloneFrom(mat);

  for (int i = 0; i < this->matrix_pat_->get_nnz(); ++i)
    this->matrix_pat_->matrix.val[i] = 1.0;

  this->ext_matrix_pat_ = true;
}

template < typename ValueType >
void lPreconditioner_FSAI< ValueType >::Build(const hiflow::la::lVector< ValueType > *in, 
                                              const hiflow::la::lVector< ValueType > *out)   
{
  assert (in != nullptr);
  if (this->mult_tmp_ != nullptr)
  {
    delete this->mult_tmp_;
  }
  
  this->mult_tmp_ = in->CloneWithoutContent();
  mult_tmp_->Zeros();
  
  // For using the FSAI method, we need to solve the system: A G_j = e_j (for
  // (i,j) in P) P is a prescribed triangular pattern of the preconditioner. A
  // is a dense matrix. The values of A correspond to the values of the
  // coefficient matrix and the structure arise from the pattern P. G_j is the
  // j-th column of this pattern (only the nonzeros) e_j is a vector with only
  // entry corresponding to the diagonal is nonzero.

  int dense_size; // the size of the small dense matrix
  int dense_col;  // column index of the dense matrix
  int start_col;  // starting column index
  int pat_col;    // the column index of the pattern
  int insert;     // continuous index for inserting the sol_ vector into the
                  // preconditioner
  int max_iter;   // maximum number of iterations

  bool Stop = false; // stop criterion for loops
  ValueType buffer;  // temporary buffer for values

  CPU_CSR_lMatrix< ValueType > *mat =
      new CPUsimple_CSR_lMatrix< ValueType >; // the orginal coefficient matrix
  CPU_CSR_lMatrix< ValueType > *pat =
      new CPUsimple_CSR_lMatrix< ValueType >; // the prescribed pattern of the
                                              // preconditioner

  mat->Init(this->Operator_->get_nnz(), this->Operator_->get_num_row(),
            this->Operator_->get_num_col(), "Matrix");

  mat->CloneFrom(*this->Operator_);

  // Build or load the pattern of the preconditioner
  if (this->ext_matrix_pat_) {

    pat->CloneFrom(*this->matrix_pat_);

    delete this->matrix_pat_;
    this->matrix_pat_ = NULL;

  } else {

    // Ensure symmetric pattern if Dirichlet BC are prescribed

    for (int i = 0; i < mat->get_num_row(); ++i)
      for (int row = mat->matrix.row[i]; row < mat->matrix.row[i + 1]; ++row)
        if (mat->matrix.val[row] == 0.0) {

          for (int col = mat->matrix.row[mat->matrix.col[row]];
               col < mat->matrix.row[mat->matrix.col[row] + 1]; ++col)
            if (i == mat->matrix.col[col]) {
              mat->matrix.val[col] = 0.0;
            }
        }

    mat->compress_me();

    if (mat->issymmetric() == false) {
      LOG_ERROR("lprecond FSAI - input matrix is not symmetric!!!");
      quit_program();
    }

    lMatrix< ValueType > *matrix_cpu_power =
        new CPUsimple_CSR_lMatrix< ValueType >;

    matrix_cpu_power = mat->MatrixSupSPower(this->matrix_power_);
    pat->CloneFrom(*matrix_cpu_power);

    delete matrix_cpu_power;
  }

  //  pat->CloneFrom(*this->Operator_);

  pat->delete_lower_triangular();

  for (int i = 0; i < mat->get_num_row(); ++i) {

    // The size of the dense matrix is equal to the nonzero elements in G_j
    dense_size = 0;
    dense_size = pat->matrix.row[i + 1] - pat->matrix.row[i];

    // If only the diagonal entry of G_j is nonzero
    if (dense_size == 1) {
      for (int row = mat->matrix.row[i]; row < mat->matrix.row[i + 1]; ++row) {
        if (mat->matrix.col[row] == i) {
          pat->matrix.val[pat->matrix.row[i]] = 1.0 / mat->matrix.val[row];
          break;
        }
      }
    }

    if (dense_size > 1) {
      lVector< ValueType > *sol_, *rhs_;
      lMatrix< ValueType > *matrix_;
      // matrix_ is the dense matrix (size: dense_size x dense_size )
      // rhs_ is the right hand side and only the entry corresponding to the
      // diagonal is nonzero (e_j) sol_ is the solution vector and the entries
      // of sol_ correspond to the nonzero elements of the precondtioner (G_j)
      matrix_ = new CPUsimple_DENSE_lMatrix< ValueType >(
          dense_size * dense_size, dense_size, dense_size, "dense Matrix");
      sol_ = new CPUsimple_lVector< ValueType >(dense_size, "sol");
      rhs_ = new CPUsimple_lVector< ValueType >(dense_size, "rhs");

      // Set the initial guess to zero
      matrix_->Zeros();
      sol_->Zeros();
      rhs_->Zeros();

      // Set the diagonal to 1
      rhs_->add_value(dense_size - 1, 1.0);

      // Build the symmetric dense matrix by going through all indices
      // (dense_row, dense_col) and filling them with the corresponding values.
      for (int dense_row = 0; dense_row < dense_size; ++dense_row) {

        start_col = pat->matrix.col[pat->matrix.row[i] + dense_row];
        dense_col = 0;

        for (int mat_col = mat->matrix.row[start_col];
             mat_col < mat->matrix.row[start_col + 1]; ++mat_col) {

          pat_col = pat->matrix.col[pat->matrix.row[i] + dense_col];

          if (mat->matrix.col[mat_col] < pat_col)
            continue;

          while (mat->matrix.col[mat_col] > pat_col) {
            dense_col += 1;
            pat_col = pat->matrix.col[pat->matrix.row[i] + dense_col];
            if (dense_col >= dense_size) {
              Stop = true;
              break;
            }
          }

          if (Stop == true) {
            Stop = false;
            break;
          }

          if (mat->matrix.col[mat_col] == pat_col) {

            matrix_->add_value(dense_row, dense_col, mat->matrix.val[mat_col]);
            dense_col += 1;

            if (dense_col >= dense_size) {
              break;
            }
          }
        }
      }
      max_iter = this->solver_max_iter_;

      // Solve the linear equation system and insert the entries of sol_ into
      // the precondtioner
      cg(sol_, rhs_, matrix_, this->solver_rel_eps_, this->solver_abs_eps_,
         max_iter, -1);

      insert = 0;

      for (int j = pat->matrix.row[i]; j < pat->matrix.row[i + 1]; ++j) {

        if (pat->matrix.col[j] > i) {
          LOG_ERROR("lpreconditioner_ai FSAI - insert below the diagonal");
          quit_program();
          break;
        }
        sol_->GetValues(&insert, 1, &buffer);
        insert += 1;
        if (insert > dense_size) {
          LOG_ERROR(
              "lpreconditioner_ai FSAI - too many elements to be inserted");
          quit_program();
        }
        pat->matrix.val[j] = buffer;
      }

      delete sol_;
      delete rhs_;
      delete matrix_;
    }
    if (dense_size < 1) {
      LOG_ERROR("lpreconditioner_ai FSAI - zero column");
      quit_program();
    }
  }

  // Diagonal scaling of the preconditioner, in order to satisfy G A G^T = I (
  // for (i,j) in the prescribed pattern)
  for (int i = 0; i < pat->get_num_row(); ++i) {
    buffer = sqrt(1.0 / pat->matrix.val[pat->matrix.row[i + 1] - 1]);
    for (int j = pat->matrix.row[i]; j < pat->matrix.row[i + 1]; ++j) {
      pat->matrix.val[j] = pat->matrix.val[j] * buffer;
    }
  }

  if (this->AI_L_ != nullptr)
  {
    delete this->AI_L_;
  }
  
  this->AI_L_ = this->Operator_->CloneWithoutContent();
  this->AI_L_->CloneFrom(*pat);
  //  pat->WriteFile("FSAI_L.mtx");

  pat->transpose_me();

  if (this->AI_Lt_ != nullptr)
  {
    delete this->AI_Lt_;
  }
  
  this->AI_Lt_ = this->Operator_->CloneWithoutContent();
  this->AI_Lt_->CloneFrom(*pat);
  //  pat->WriteFile("FSAI_Lt.mtx");

  delete mat;
  delete pat;

  //  this->AI_L_->print();
  //  this->AI_Lt_->print();

  char data_info[255];
  sprintf(data_info, " / NNZ(L+L^T)=%d",
          this->AI_L_->get_nnz() + this->AI_L_->get_nnz());
  this->precond_name_.append(data_info);
  this->SetState(true);
  this->SetModifiedOperator(false);
}

template < typename ValueType >
void lPreconditioner_FSAI< ValueType >::ApplylPreconditioner(
    const hiflow::la::lVector< ValueType > &input,
    hiflow::la::lVector< ValueType > *output) {
  assert(this->AI_L_ != NULL);
  assert(this->AI_Lt_ != NULL);
  assert(this->mult_tmp_ != NULL);
  if (!this->GetState()) {
    this->Build(&input, output);
  }
  
  this->AI_L_->VectorMult(input, this->mult_tmp_);
  this->AI_Lt_->VectorMult(*this->mult_tmp_, output);
}

template class lPreconditioner< double >;
template class lPreconditioner< float >;

template class lPreconditioner_Jacobi< double >;
template class lPreconditioner_Jacobi< float >;

template class lPreconditioner_GaussSeidel< double >;
template class lPreconditioner_GaussSeidel< float >;

template class lPreconditioner_SymmetricGaussSeidel< double >;
template class lPreconditioner_SymmetricGaussSeidel< float >;

template class lPreconditioner_BlocksSymmetricGaussSeidel< double >;
template class lPreconditioner_BlocksSymmetricGaussSeidel< float >;

template class lPreconditioner_SOR< double >;
template class lPreconditioner_SOR< float >;

template class lPreconditioner_SSOR< double >;
template class lPreconditioner_SSOR< float >;

template class lPreconditioner_ILUp< double >;
template class lPreconditioner_ILUp< float >;

template class lPreconditioner_FSAI< double >;
template class lPreconditioner_FSAI< float >;

} // namespace la
} // namespace hiflow
