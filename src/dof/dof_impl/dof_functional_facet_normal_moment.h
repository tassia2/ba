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

#ifndef _DOF_DOF_FUNCTIONAL_FACET_NORMAL_MOMENT_H_
#define _DOF_DOF_FUNCTIONAL_FACET_NORMAL_MOMENT_H_

#include <map>
#include <vector>

#include "common/vector_algebra_descriptor.h"
#include "dof/dof_impl/dof_functional.h"
#include "fem/function_space.h"

namespace hiflow {
namespace doffem {

template <class DataType, int DIM> class DofContainerRTBDM;

/// Implementation of dof given by quadrature 
/// l_j(phi) = sum_q { w(q) * dot(phi(q), normal(q)) * psi_j(q) * ds}  
/// \author Philipp Gerstner

template < class DataType, int DIM > 
class DofFacetNormalMoment : public DofFunctional <DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Constructor
  DofFacetNormalMoment()
  : test_ind_(0),
    facet_nr_(0)
  {
    this->type_ = DofFunctionalType::FACET_MOMENT;
  }

  /// Destructor
  virtual ~DofFacetNormalMoment()
  {}
  
  inline size_t test_ind () const
  {
    return this->test_ind_;
  }

  inline size_t facet_nr () const
  {
    return this->facet_nr_;
  }

  // Note: if this function returns false, then this != dof_slave.
  // However, if true is returned, this does imply this == dof_slave,
  // since FacetDofs
  /*
  bool operator==(const DofFunctional<DataType, DIM> &dof_slave) const
  {
    // compare type
    if (dof_slave.type() != DofFunctionalType::FACET_MOMENT)
    {
      return false;
    }
    DofCellMoment<DataType, DIM> const * dof_slave_cell 
      = dynamic_cast<DofFacetNormalMoment<DataType, DIM> const *> (&dof_slave);
    
    assert(dof_slave_cell != nullptr);
    
    // compare test index
    if (dof_slave_cell->test_ind_ != this->test_ind_)
    {
      return false;
    }
    return true;
  }
  */
private:
  void set_trial_values (std::vector< std::vector<DataType> > const * trial_vals)
  {
    assert (trial_vals != nullptr);
    assert (trial_vals->size() == this->q_weights_->size()); 
    this->trial_vals_ = trial_vals;
  }
  
  void init (FunctionSpace<DataType, DIM-1> const * test_space,
             std::vector< std::vector<DataType> > const * test_vals,
             std::vector<DataType> const * q_weights,
             std::vector<DataType> const * ds,
             size_t test_ind,
             CRefCellSPtr<DataType, DIM> ref_cell,
             size_t facet_nr,
             Coord facet_vec)
  {
    assert (DIM > 1);
    assert (test_space != nullptr);
    assert (test_vals != nullptr);
    assert (q_weights != nullptr);
    assert (ref_cell != nullptr);
    assert (ds != nullptr);
    assert (test_ind < test_space->dim());
    assert (q_weights->size() > 0);
    assert (test_vals->size() == q_weights->size());
    assert (ds->size() == q_weights->size());  
    assert (test_space->tdim() == ref_cell->tdim() - 1);
    assert (test_space->nb_comp() == 1);

    this->test_space_ = test_space;
    this->test_vals_ = test_vals;
    this->q_weights_ = q_weights;
    this->test_ind_ = test_ind;
    this->ref_cell_ = ref_cell;
    this->facet_nr_ = facet_nr;
    this->facet_vec_ = facet_vec;
    this->ds_ = ds;

    this->attached_to_subentity_.clear();
    this->attached_to_subentity_.resize(ref_cell->tdim()+1);
    
    this->attached_to_subentity_[ref_cell->tdim()-1].push_back(facet_nr);
  }
  
  void evaluate (FunctionSpace<DataType, DIM> const * trial_space, 
                 std::vector<DataType>& dof_values ) const
  {
    assert (this->q_weights_ != nullptr);
    assert (this->test_vals_ != nullptr);
    assert (this->trial_vals_ != nullptr);
    assert (this->test_space_ != nullptr);

    assert (this->test_space_->nb_comp() == 1);
    assert (trial_space->nb_comp() == DIM);

    assert (dof_values.size() == trial_space->dim());

    const size_t nb_comp = trial_space->nb_comp();
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
              * this->facet_vec_[v];
        }
        dof_values[i] += this->q_weights_->at(q) 
                      * tmp 
                      * this->test_vals_->at(q)[ this->test_space_->iv2ind(this->test_ind_, 0) ] 
                      * this->ds_->at(q);
      }
    }
/*    for (size_t i = 0; i < dim; ++i)
    {
      std::cout << " facet nr " << this->facet_nr_ << " trial index " << i << ", test index " << this->test_ind_  << " DOF VAL " << dof_values[i] << std::endl;
    }
*/
  }

  void evaluate (RefCellFunction<DataType, DIM> const * func, 
                 size_t offset, 
                 std::vector<DataType> &dof_values) const
  {
    assert (this->q_weights_ != nullptr);
    assert (this->test_vals_ != nullptr);
    assert (this->trial_vals_ != nullptr);
    assert (this->test_space_ != nullptr);
    assert (this->test_space_->nb_comp() == 1);
    assert (func != nullptr);
    
    const size_t num_q = this->q_weights_->size();
    const size_t nb_comp = func->nb_comp();
    const size_t nb_func = func->nb_func();
    assert (nb_comp == DIM);
    assert (offset + nb_func <= dof_values.size());

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
              * this->facet_vec_[v];
        }
        dof_values[offset + i] += this->q_weights_->at(q) 
                                * tmp 
                                * this->test_vals_->at(q)[ this->test_space_->iv2ind(this->test_ind_, 0) ] 
                                * this->ds_->at(q);
      }
    }
  }
  
  friend class DofContainerRTBDM<DataType, DIM>;
  
  Coord facet_vec_;

  size_t test_ind_;
  size_t facet_nr_;

  std::vector< std::vector<DataType> > const * test_vals_;
  std::vector< std::vector<DataType> > const * trial_vals_;
  std::vector< DataType > const * q_weights_;
  std::vector< DataType > const * ds_;
  FunctionSpace<DataType, DIM-1> const * test_space_;

};
/*
template <class DataType, int DIM>
bool equal_dof (const DofFacetNormalMoment<DataType, DIM> &dof_a,
                const DofFacetNormalMoment<DataType, DIM> &dof_b,
                const CellTransformation<DataType, DIM> &c_trafo_a,
                const CellTransformation<DataType, DIM> &c_trafo_b) 
{
  if (dof_a.test_ind() != dof_b.test_ind())
  {
    return false;
  }

  CRefCellSPtr<DataType, DIM> ref_cell_a = c_trafo_a->get_ref_cell();
  CRefCellSPtr<DataType, DIM> ref_cell_b = c_trafo_b->get_ref_cell();
  
  Vec< DIM, DataType > ref_n_a;
  Vec< DIM, DataType > ref_n_b;
  
  ref_cell_a->compute_facet_normal (dof_a.facet_nr(), ref_n_a);
  ref_cell_b->compute_facet_normal (dof_b.facet_nr(), ref_n_b);
  
  
  Coord ref_pt_a = dof_a.get_point();
  Coord ref_pt_b = dof_b.get_point();
    
  Coord pt_a;
  Coord pt_b;
    
  c_trafo_a.transform ( ref_pt_a, pt_a );
  c_trafo_b.transform ( ref_pt_b, pt_b );
    
  if (pt_a == pt_b)
  {
    return true;
  }
  return false;
}
* */

} // namespace doffem
} // namespace hiflow
#endif
