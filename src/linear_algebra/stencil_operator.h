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

#ifndef HIFLOW_LINEAR_ALGEBRA_STENCIL_OPERATOR_H
#define HIFLOW_LINEAR_ALGEBRA_STENCIL_OPERATOR_H

#include "assembly/assembly_types.h"
#include "common/log.h"
#include "common/vector_algebra.h"
#include "linear_algebra/linear_operator.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "linear_algebra/index_handler.h" 
#include "linear_algebra/cell_matrix_handler.h"
#include "space/vector_space.h"
#include "linear_algebra/lmp/lvector.h"
#include "dof/dof_fem_types.h"
#include <assert.h>
#include <cstddef>

namespace hiflow {
namespace la {

template <class DataType, int M, int N > class CPUsimpleStencil;

/// \brief Abstract base class for matrix-free stencil operator
/// LAD: Type of vector 
/// N: dimension of stencil

template < class LAD, int DIM, int N,
           class Stencil = CPUsimpleStencil<typename LAD::DataType, N, N>,
           class IndexFunctor = IndexFunctionUniformFE<LAD, DIM>,
           class LocalVector = CPU_lVector<typename LAD::DataType> > 
class StencilOperator : public virtual LinearOperator<typename LAD::DataType> 
{
public:
  using DataType = typename LAD::DataType;
  using GlobalVector = typename LAD::VectorType;
  using LocalMatrix = SeqDenseMatrix< DataType >;
  using VectorSpacePtr = CVectorSpaceSPtr<DataType, DIM>;
  using IndexPtr = IndexFunctor* ;
  using LMHandlerPtr = CellMatrixHandler<DataType, DIM>*;

  StencilOperator()
  {}

  virtual ~StencilOperator() {}

  void Init (const VectorSpace<DataType, DIM>& space,
             IndexPtr index,
             LMHandlerPtr handler);

  void VectorMult(Vector< DataType > &in,
                  Vector< DataType > *out) const override
  {
    GlobalVector *casted_in = dynamic_cast< GlobalVector * >(&in);
    GlobalVector *casted_out = dynamic_cast< GlobalVector * >(out);
    
    assert(casted_in != nullptr);
    assert(casted_out != nullptr);

    this->VectorMult(*casted_in, casted_out);
  }

  void VectorMultAdd(DataType alpha, Vector< DataType > &in,
                     DataType beta, Vector< DataType > *out) const override 
  {
    GlobalVector *casted_in = dynamic_cast< GlobalVector * >(&in);
    GlobalVector *casted_out = dynamic_cast< GlobalVector * >(out);
    
    assert(casted_in != nullptr);
    assert(casted_out != nullptr);

    this->VectorMultAdd(alpha, *casted_in, beta, casted_out);
  }

  inline void VectorMult(GlobalVector &in,
                         GlobalVector *out) const 
  {
    this->VectorMultAdd(1., in, 0., out);
  }
                          
  /// out = beta * out + alpha * this * in
  void VectorMultAdd(DataType alpha, GlobalVector &in,
                     DataType beta,  GlobalVector *out) const;

  bool IsInitialized() const 
  { 
    return this->is_initialized_; 
  }

private:

  bool is_initialized_ = false;

  ConstMeshPtr mesh_ = nullptr;
  IndexPtr index_ = nullptr;
  LMHandlerPtr lm_handler_ = nullptr;

  std::vector< Stencil > stencils_;
  std::vector< Stencil > cross_stencils_;
  int num_stencils_ = 0;
  int num_cross_stencils_ = 0;
  int tdim_ = -1;
  int num_cells_ = 0;
};

template < class LAD, int DIM, int N, class Stencil, class IndexFunctor, class LocalVector>
void StencilOperator <LAD, DIM, N, Stencil, IndexFunctor, LocalVector>::Init (const VectorSpace<DataType, DIM>& space,
                                                                              IndexPtr index,
                                                                              LMHandlerPtr handler)
{
  assert (index != nullptr);
  assert (handler != nullptr);

  this->index_ = index;
  this->lm_handler_ = handler;
  assert (space.nb_dof_on_cell(0) == N);

  this->mesh_ = space.meshPtr();
  assert (this->mesh_ != 0);

  this->tdim_ = this->mesh_->tdim();
  this->num_cells_ = this->mesh_->num_entities(tdim_);

  this->num_stencils_ = lm_handler_->num_matrices();
  assert (this->num_stencils_ > 0);

  this->stencils_.clear();
  this->stencils_.resize(num_stencils_);

  for (int s=0; s!=num_stencils_; ++s)
  {
    this->stencils_[s].init(lm_handler_->get_matrix(s));
  }

  this->num_cross_stencils_ = lm_handler_->num_cross_matrices();

  this->cross_stencils_.clear();
  this->cross_stencils_.resize(num_cross_stencils_);

  for (int s=0; s!=num_cross_stencils_; ++s)
  {
    this->cross_stencils_[s].init(lm_handler_->get_cross_matrix(s));
  }

  this->is_initialized_ = true;
}

template < class LAD, int DIM, int N, class Stencil, class IndexFunctor, class LocalVector>
void StencilOperator <LAD, DIM, N, Stencil, IndexFunctor, LocalVector>::VectorMultAdd(DataType alpha, GlobalVector &in,
                                                                                      DataType beta,  GlobalVector *out) const
{
  assert (this->IsInitialized());
  assert (out != nullptr);

  auto cell_begin = this->mesh_->begin(tdim_);
  auto cell_end = this->mesh_->end(tdim_);

  // cast vector here to avoid virtual function call for GetValue(), etc.. below
  auto in_interior  = static_cast< LocalVector* > (&(in.interior()));
  auto in_ghost     = static_cast< LocalVector* > (&(in.ghost())); 
  auto out_interior = static_cast< LocalVector* > (&(out->interior()));

  assert (in_interior != 0);
  assert (in_ghost != 0);
  assert (out_interior != 0);
  
  // call asynchronous receive for my ghost dofs
  in.ReceiveGhost();

  // call async send of my border values
  in.SendBorder();

  out_interior->Scale(beta);

  lDofId dof_ind_[N];
  int is_loc_[N];
  const auto algn = Stencil::get_alignment();

  alignas(algn) DataType rhs_vals[N];
  alignas(algn) DataType lhs_vals[N];
  alignas(algn) DataType tmp_vals[N];

  assert ((long long int) rhs_vals % algn == 0);
  assert ((long long int) lhs_vals % algn == 0);

  const auto begin_local = this->index_->first_local_cell();
  const auto end_local = this->index_->last_local_cell();
  const auto begin_ghost = this->index_->first_ghost_cell();
  const auto end_ghost = this->index_->last_ghost_cell();
  const bool consider_cross_matrices = (this->lm_handler_->num_cross_matrices() != 0);

  // this->lm_handler_->print();
  // TODO: take care of dof_factors, hanging nodes
  // TODO: insert check and NOT_YET_IMPLEmENTED for these cases

  for (auto cell_it = begin_local; cell_it != end_local; ++cell_it)
  {
    const auto cell_index = *cell_it;

    // ------------------------------------------------------------------------------------
    // Step 1: take care of cell-based local matrices, with RHS coming from interior vector

    // get global indices
    // Important: this routine is responsible for computing the correct interior / ghost lVector index   
    // extract values from rhs and lhs vector
    for (int j=0; j!=N; ++j)
    {
      const lDofId dof = this->index_->get_dof_index (cell_index, j);
      rhs_vals[j] = in_interior->GetValue(dof);
      lhs_vals[j] = 0.;
    }

    const auto begin_lm = this->lm_handler_->begin_lmindex(cell_index);
    const auto end_lm = this->lm_handler_->end_lmindex(cell_index);
    auto factor_it = this->lm_handler_->begin_factor(cell_index);

    // call Stencils for Matrix-Vector mult
    for (auto lm_it = begin_lm; lm_it != end_lm; ++lm_it)
    {
      assert (*lm_it >= 0);
      assert (*lm_it < this->stencils_.size());

      /*
      if (*factor_it == 0.)
      {
        continue;
      }*/

      this->stencils_[*lm_it].VectorMult(rhs_vals, tmp_vals);

      // PERF-TODO: put into stencil.vectormultadd
      const DataType cell_factor = alpha * (*factor_it); 
      for (int j=0; j!=N; ++j)
      {
        lhs_vals[j] += cell_factor * tmp_vals[j];
      }
      factor_it++;
    }

    // ------------------------------------------------------------------------------------
    // Step 2: take care of interface-based local matrices, with RHS coming from interior vector
    if (consider_cross_matrices)
    {
      const auto begin_cross_lm = this->lm_handler_->begin_cross_lmindex(cell_index);
      const auto end_cross_lm = this->lm_handler_->end_cross_lmindex(cell_index);
      auto cross_factor_it = this->lm_handler_->begin_cross_factor(cell_index);
      auto cross_cell_index = this->lm_handler_->begin_cross_index(cell_index);
 
      // loop over stencils
      for (auto lm_it = begin_cross_lm; lm_it != end_cross_lm; ++lm_it)
      {      
        assert (*lm_it >= 0);
        assert (*lm_it < this->cross_stencils_.size());

        /*
        if (*cross_factor_it == 0.)
        {
          continue;
        }*/

        const int in_cell_index = *cross_cell_index;
        for (int j=0; j!=N; ++j)
        {
          const lDofId dof = this->index_->get_dof_index (in_cell_index, j);
          rhs_vals[j] = in_interior->GetValue(dof);
        }

        this->cross_stencils_[*lm_it].VectorMult(rhs_vals, tmp_vals);

        const DataType cell_factor = alpha * (*cross_factor_it); 
        for (int j=0; j!=N; ++j)
        {
          lhs_vals[j] += cell_factor * tmp_vals[j];
        }
        cross_factor_it++;
        cross_cell_index++;
      }
    }

    // -------------------------------------------------------------------
    // insert lhs vectors into out vector
    for (int j=0; j!=N; ++j)
    {
      const lDofId dof = this->index_->get_dof_index (cell_index, j);
      out_interior->add_value(dof, lhs_vals[j]);
    }
  }

  // wait until I received all of my ghost values
  in.WaitForRecv();

  // loop through ghost cells
  for (auto cell_it = begin_ghost; cell_it != end_ghost; ++cell_it)
  {
    const auto cell_index = *cell_it;

    // ------------------------------------------------------------------------------------
    // Step 3: take care of cell-based local matrices, with RHS coming from interior and ghost vector

    // Important: this routine is responsible for computing the correct interior / ghost lVector index   
    // extract values from rhs and lhs vector
    for (int j=0; j!=N; ++j)
    {
      const lDofId dof = this->index_->get_dof_index (cell_index, j);
      if (this->index_->dof_is_local (cell_index, j))
      {
        rhs_vals[j] = in_interior->GetValue(dof);
      }
      else 
      {
        rhs_vals[j] = in_ghost->GetValue(dof);
      }
      lhs_vals[j] = 0.;
    }

    const auto begin_lm = this->lm_handler_->begin_lmindex(cell_index);
    const auto end_lm = this->lm_handler_->end_lmindex(cell_index);
    auto factor_it = this->lm_handler_->begin_factor(cell_index);

    // call Stencils for Matrix-Vector mult
    for (auto lm_it = begin_lm; lm_it != end_lm; ++lm_it)
    {
      assert (*lm_it >= 0);
      assert (*lm_it < this->stencils_.size());

      /*
      if (*factor_it == 0.)
      {
        continue;
      }*/

      this->stencils_[*lm_it].VectorMult(rhs_vals, tmp_vals);

      const DataType cell_factor = alpha * (*factor_it); 
      for (int j=0; j!=N; ++j)
      {
        lhs_vals[j] += cell_factor * tmp_vals[j];
      }
      factor_it++;
    }

    // ------------------------------------------------------------------------------------
    // Step 4: take care of interface-based local matrices, with RHS coming from interior and ghost vector
    if (consider_cross_matrices)
    {
      const auto begin_cross_lm = this->lm_handler_->begin_cross_lmindex(cell_index);
      const auto end_cross_lm = this->lm_handler_->end_cross_lmindex(cell_index);
      auto cross_factor_it = this->lm_handler_->begin_cross_factor(cell_index);
      auto cross_cell_index = this->lm_handler_->begin_cross_index(cell_index);

      // loop over stencils
      for (auto lm_it = begin_cross_lm; lm_it != end_cross_lm; ++lm_it)
      {      
        assert (*lm_it >= 0);
        assert (*lm_it < this->cross_stencils_.size());

        /*
        if (*cross_factor_it == 0.)
        {
          continue;
        }*/

        const int in_cell_index = *cross_cell_index;
        for (int j=0; j!=N; ++j)
        {
          const lDofId dof = this->index_->get_dof_index (in_cell_index, j);
          if (this->index_->dof_is_local (in_cell_index, j))
          {
            rhs_vals[j] = in_interior->GetValue(dof);
          }
          else 
          {
            rhs_vals[j] = in_ghost->GetValue(dof);
          }
        }

        this->cross_stencils_[*lm_it].VectorMult(rhs_vals, tmp_vals);

        const DataType cell_factor = alpha * (*cross_factor_it); 
        for (int j=0; j!=N; ++j)
        {
          lhs_vals[j] += cell_factor * tmp_vals[j];
        }
        cross_factor_it++;
        cross_cell_index++;
      }
    }

    // ----------------------------------------------------------------
    // insert lhs vectors into out vector
    for (int j=0; j!=N; ++j)
    {
      if (this->index_->dof_is_local (cell_index, j) != 0)
      {
        const lDofId dof = this->index_->get_dof_index (cell_index, j);
        out_interior->add_value(dof, lhs_vals[j]);
      }
    }
  }

  // take care of fixed dofs 
  const auto begin_fixed = this->index_->first_local_fixed_dof();
  const auto end_fixed = this->index_->last_local_fixed_dof();

  for (auto dof_it = begin_fixed; dof_it != end_fixed; ++dof_it)
  {
    const auto rhs_val = in_interior->GetValue(*dof_it);
    out_interior->SetValue(*dof_it, rhs_val);
  }

  // wait until I sent all my border values
  in.WaitForSend();
}

}
}

#endif