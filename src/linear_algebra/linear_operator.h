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

#ifndef HIFLOW_LINEAR_ALGEBRA_LINEAR_OPERATOR_H
#define HIFLOW_LINEAR_ALGEBRA_LINEAR_OPERATOR_H

#include "common/log.h"
#include "linear_algebra/vector.h"
#include "space/vector_space.h"
#include <assert.h>
#include <cstddef>

namespace hiflow {
namespace la {

/// \brief Abstract base class for distributed linear operator implementations.

template < class DataType > class LinearOperator {
public:
  /// Standard Constructor

  LinearOperator(){};

  /// Destructor

  virtual ~LinearOperator() {}
                         
  virtual void VectorMult(Vector< DataType > &in,
                          Vector< DataType > *out) const = 0;

  virtual void VectorMultOffdiag(Vector< DataType > &in,
                                 Vector< DataType > *out)
  {
    NOT_YET_IMPLEMENTED;
  }
                          
  /// out = beta * out + alpha * this * in

  virtual void VectorMultAdd(DataType alpha, Vector< DataType > &in,
                             DataType beta, Vector< DataType > *out) const {
  
    assert(out != nullptr);
    Vector< DataType > *tmp = out->Clone();
    this->VectorMult(in, tmp);
    out->Scale(beta);
    out->Axpy(*tmp, alpha);
  }

  virtual bool IsInitialized() const { return true; }
};

template < class DataType >
class ZeroOperator : public LinearOperator< DataType > {
public:
  /// Standard Constructor

  ZeroOperator() { this->is_initialized_ = true; };
  /// Destructor

  virtual ~ZeroOperator() {}

  /// out = this * in

  virtual void VectorMult(Vector< DataType > &in,
                          Vector< DataType > *out) const {
    assert(out != nullptr);
    assert(this->IsInitialized());
    out->Zeros();
  }
};

/// \brief class for getting a chained operator out of two individual ones

template < class DataType >
class ChainLinearOperator : virtual public LinearOperator< DataType > {
public:
  /// Standard Constructor

  ChainLinearOperator(){};
  /// Destructor

  virtual ~ChainLinearOperator() {}

  virtual void SetOperatorA(const LinearOperator< DataType > &op) {
    this->op_A_ = &op;
  }

  virtual void SetOperatorB(const LinearOperator< DataType > &op) {
    this->op_B_ = &op;
  }

  /// out = B[ A [n] ]

  virtual void VectorMult(Vector< DataType > &in,
                          Vector< DataType > *out) const {
    assert(op_A_ != nullptr);
    assert(op_B_ != nullptr);
    assert(out != nullptr);
    assert(this->IsInitialized());

    Vector< DataType > *tmp = in.CloneWithoutContent();

    // tmp = A[in]
    this->op_A_->VectorMult(in, tmp);

    // out = b[tmp]
    this->op_B_->VectorMult(*tmp, out);
  }

  virtual bool IsInitialized() const {
    if (this->op_A_ != nullptr) {
      if (this->op_A_->IsInitialized()) {
        if (this->op_B_ != nullptr) {
          if (this->op_B_->IsInitialized()) {
            return true;
          } else {
            LOG_DEBUG(0, "op_B not initialized");
          }
        } else {
          LOG_DEBUG(0, "op_B = nullptr");
        }
      } else {
        LOG_DEBUG(0, "op_A not initialized");
      }
    } else {
      LOG_DEBUG(0, "op_A = nullptr");
    }
    return false;
  }

protected:
  LinearOperator< DataType > const *op_A_;
  LinearOperator< DataType > const *op_B_;
};

/// \brief class for getting a linear combination of two linear operators

template < typename MatrixType_A, typename MatrixType_B, typename DataType >
class SumLinearOperator : virtual public LinearOperator< DataType > 
{
public:

  /// Standard Constructor

  SumLinearOperator(){};
  /// Destructor

  virtual ~SumLinearOperator() {}

  void Init (const MatrixType_A & op_A, DataType scale_A,
             const MatrixType_B & op_B, DataType scale_B)
  {
    this->op_A_ = &op_A;
    this->op_B_ = &op_B;
    this->scale_A_ = scale_A;
    this->scale_B_ = scale_B;
  }

  void set_unit_rows (const int *row_indices, int num_rows)
  {
    fixed_dofs_.clear();
    fixed_dofs_.resize(num_rows,0);
    for (int i=0; i!=num_rows; ++i)
    {
      fixed_dofs_[i] = row_indices[i];
    }
  }

  /// out = beta * out + alpha * (scale_A  * A[in] + scale_B * B[in])
  void VectorMultAdd(DataType alpha, Vector< DataType > &in,
                     DataType beta,  Vector< DataType > *out) const override
  {
    assert(op_A_ != nullptr);
    assert(op_B_ != nullptr);
    assert(out != nullptr);
    assert(this->IsInitialized());

    //std::cout << alpha << " " << beta << " " << scale_A_ << " " << scale_B_ << " " << fixed_dofs_.size() << std::endl;
    //std::cout << op_A_ << std::endl;
    
    // out = beta * out
    // out->Scale(beta);

    // out += alpha * scale_A * A[in]
    this->op_A_->VectorMultAdd(alpha * scale_A_, in, beta, out);
    
    // out += alpha * scale_B * B[in]
    if (scale_B_ != 0.)
    {
      this->op_B_->VectorMultAdd(alpha * scale_B_, in, 1., out);
    }

    if (fixed_dofs_.size() > 0)
    {
      for (auto dof : fixed_dofs_)
      {
        const auto rhs_val = in.GetValue(dof);
        out->SetValue(dof, rhs_val);
      }
    }
    out->Update();  // necessary?
  }

  inline void VectorMult(Vector< DataType > &in,
                         Vector< DataType > *out) const override 
  {
    this->VectorMultAdd(1., in, 0., out);
  }

  virtual bool IsInitialized() const {
    if (this->op_A_ != nullptr) {
      if (this->op_A_->IsInitialized()) {
        if (this->op_B_ != nullptr) {
          if (this->op_B_->IsInitialized()) {
            return true;
          } else {
            LOG_DEBUG(0, "op_B not initialized");
          }
        } else {
          LOG_DEBUG(0, "op_B = nullptr");
        }
      } else {
        LOG_DEBUG(0, "op_A not initialized");
      }
    } else {
      LOG_DEBUG(0, "op_A = nullptr");
    }
    return false;
  }

protected:
  MatrixType_A const *op_A_ = nullptr;
  MatrixType_B const *op_B_ = nullptr;

  DataType scale_A_ = 1.;
  DataType scale_B_ = 1.;

  std::vector<int> fixed_dofs_;
};

} // namespace la
} // namespace hiflow

#endif
