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

#ifndef _DOF_DOF_FUNCTIONAL_CELL_MOMENT_H_
#define _DOF_DOF_FUNCTIONAL_CELL_MOMENT_H_

#include <map>
#include <vector>

#include "common/vector_algebra_descriptor.h"
#include "dof/dof_impl/dof_functional.h"
#include "fem/function_space.h"

namespace hiflow {
namespace doffem {

template <class DataType, int DIM> class DofContainerRTBDM;

/// Implementation of dof given by quadrature 
/// l_j(phi) = sum_q { w(q) * dot(phi(q), psi_j(q) }  
/// \author Philipp Gerstner

template < class DataType, int DIM > 
class DofCellMoment : public DofFunctional <DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Constructor
  DofCellMoment()
  : test_ind_(0)
  {
    this->type_ = DofFunctionalType::CELL_MOMENT;
  }

  /// Destructor
  virtual ~DofCellMoment()
  {}
  
  inline size_t test_ind () const
  {
    return this->test_ind_;
  }
/*
  bool operator==(const DofFunctional<DataType, DIM> &dof_slave) const
  {
    // compare type
    if (dof_slave.type() != DofFunctionalType::CELL_MOMENT)
    {
      return false;
    }
    DofCellMoment<DataType, DIM> const * dof_slave_cell 
      = dynamic_cast<DofCellMoment<DataType, DIM> const *> (&dof_slave);
    
    assert(dof_slave_cell != nullptr);
    
    // compare test index
    if (dof_slave_cell->test_ind_ != this->test_ind_)
    {
      return false;
    }
    
    // compare weights
    if (dof_slave_cell->q_weights_->size() != this->q_weights_->size())
    {
      return false;
    }
    for (size_t l=0; l<this->q_weights_->size(); ++l)
    {
      if (dof_slave_cell->q_weights_->at(l) != this->q_weights_->at(l))
      {
        return false;
      }
    }
    
    // compare test values
    if (dof_slave_cell->test_vals_->size() != this->test_vals_->size())
    {
      return false;
    }
    for (size_t q=0; q<this->test_vals_->size(); ++q)
    {
      if (dof_slave_cell->test_vals_->at(q).size() != this->test_vals_->at(q).size())
      {
        return false;
      }
      for (size_t i=0; i<this->test_vals_->at(q).size(); ++i)
      {
        if (dof_slave_cell->test_vals_->at(q)[i] != this->test_vals_->at(q)[i])
        {
          return false;
        }
      }
    }
    return true;
  }
  */
private:
  void init (FunctionSpace<DataType, DIM> const * test_space,
             std::vector< std::vector<DataType> > const * test_vals,
             std::vector<DataType> const * q_weights,
             size_t test_ind,
             CRefCellSPtr<DataType, DIM> ref_cell)
  {
    assert (test_space != nullptr);
    assert (test_vals != nullptr);
    assert (q_weights != nullptr);
    assert (ref_cell != nullptr);
    assert (test_ind < test_space->dim());
    assert (test_vals->size() == q_weights->size()); 
    assert (test_space->tdim() == ref_cell->tdim());
    
    //std::cout << "DofCell init " << test_vals->at(0).size() << " " << test_space->dim() << " " << test_space->nb_comp() << std::endl;
    assert (test_vals->at(0).size() == test_space->weight_size());
    
    this->test_space_ = test_space;
    this->test_vals_ = test_vals;

    this->q_weights_ = q_weights;
    this->test_ind_ = test_ind;
    this->ref_cell_ = ref_cell;
  
    this->attached_to_subentity_.clear();
    this->attached_to_subentity_.resize(ref_cell->tdim()+1);
    this->attached_to_subentity_[ref_cell->tdim()].push_back(0);
  }

  void set_trial_values (std::vector< std::vector<DataType> > const * trial_vals)
  {
    assert (trial_vals != nullptr);
    assert (trial_vals->size() == this->q_weights_->size()); 
    this->trial_vals_ = trial_vals;
  }
  
  void evaluate (FunctionSpace<DataType, DIM> const * trial_space, 
                 std::vector<DataType>& dof_values ) const
  {
    assert (this->q_weights_ != nullptr);
    assert (this->test_vals_ != nullptr);
    assert (this->trial_vals_ != nullptr);
    assert (this->test_space_ != nullptr);

    assert (this->test_space_->nb_comp() == trial_space->nb_comp());
    assert (this->trial_vals_->at(0).size() == trial_space->weight_size());
    assert (dof_values.size() == trial_space->dim());

    const size_t nb_comp = this->test_space_->nb_comp();
    const size_t dim = trial_space->dim();
    const size_t num_q = this->q_weights_->size();

    // loop over quad points
    for (size_t q = 0; q < num_q; ++q)
    {
      // loop over trial basis functions
      for (size_t i = 0; i < dim; ++i)
      {
        DataType tmp = 0.;
        // loop over components -> dot product
        for (size_t v=0; v<nb_comp; ++v)
        {
          tmp += this->trial_vals_->at(q)[ trial_space->iv2ind(i,v) ] 
               * this->test_vals_->at(q)[ this->test_space_->iv2ind(this->test_ind_,v) ];
        }
        dof_values[i] += this->q_weights_->at(q) * tmp;
      }
    }
/*
    for (size_t i = 0; i < dim; ++i)
    {
      std::cout << " trial index " << i << ", test index " << this->test_ind_  << " DOF VAL " << dof_values[i] << std::endl;
    }
    * */
  }

  void evaluate (RefCellFunction<DataType, DIM> const * func, 
                 size_t offset, 
                 std::vector<DataType> &dof_values) const
  {
    assert (this->q_weights_ != nullptr);
    assert (this->test_vals_ != nullptr);
    assert (this->trial_vals_ != nullptr);
    assert (this->test_space_ != nullptr);
    assert (func != nullptr);
    
    const size_t num_q = this->q_weights_->size();
    const size_t nb_comp = func->nb_comp();
    const size_t nb_func = func->nb_func();
    assert (nb_comp == DIM);
    assert (offset + nb_func <= dof_values.size());
    assert (this->test_space_->nb_comp() == nb_comp);
    
    // clear dof values
    for (size_t i = offset; i < offset + nb_func; ++i)
    {
      dof_values[i] = 0.;
    }
    
    // loop over quad points
    for (size_t q = 0; q < num_q; ++q)
    {
      // loop over trial basis functions
      for (size_t i = 0; i < nb_func; ++i)
      {
        DataType tmp = 0.;
      
        // loop over components -> dot product
        for (size_t v=0; v<nb_comp; ++v)
        {
          tmp += this->trial_vals_->at(q)[func->iv2ind(i,v)/*v * nb_func + i*/] 
              * this->test_vals_->at(q)[ this->test_space_->iv2ind(this->test_ind_,v) ];
        }
        dof_values[offset + i] += this->q_weights_->at(q) * tmp;
      }
    }
  }
  
  friend class DofContainerRTBDM<DataType, DIM>;
  
  size_t test_ind_;
  std::vector< std::vector<DataType> > const * test_vals_;
  std::vector< std::vector<DataType> > const * trial_vals_;
  std::vector< DataType > const * q_weights_;
  FunctionSpace<DataType, DIM> const * test_space_;
};

} // namespace doffem
} // namespace hiflow
#endif
