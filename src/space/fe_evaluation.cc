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

#include "space/fe_evaluation.h"

#include "common/permutation.h"
#include "fem/fe_manager.h"
#include "fem/fe_reference.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "linear_algebra/vector.h"
#include "mesh/entity.h"
#include "mesh/mesh.h"
#include "mesh/geometric_search.h"
#include "mesh/geometric_tools.h"
#include "space/vector_space.h"
#include "space/element.h"
#include <boost/function.hpp>
#include <set>

namespace hiflow {

template < class DataType >
void sort_dofs (const la::Vector<DataType> &coeff,  
                std::vector< hiflow::doffem::gDofId >& id,
                std::vector< DataType >& val)
{
  // get data
  std::vector< hiflow::doffem::gDofId > id_tmp;
  std::vector< DataType > val_tmp;

  coeff.GetAllDofsAndValues(id_tmp, val_tmp);

  assert(!id_tmp.empty());
  assert(!val_tmp.empty());
  assert(id_tmp.size() == val_tmp.size());

  // calculate permutation
  std::vector< int > permutation;
  compute_sorting_permutation(id_tmp, permutation);

  // permute
  permute_vector(permutation, id_tmp, id);
  permute_vector(permutation, val_tmp, val);
}

template void sort_dofs <double> (const la::Vector<double> &, std::vector< hiflow::doffem::gDofId >&, std::vector< double >&);
template void sort_dofs <float> (const la::Vector<float> &, std::vector< hiflow::doffem::gDofId >&, std::vector< float >&);

template <class DataType, int DIM>
void extract_dof_values ( const VectorSpace<DataType, DIM>& space, 
                          const mesh::Entity& cell, 
                          const Element<DataType, DIM>& elem,
                          const la::Vector<DataType>& fun,
                          const std::vector< doffem::gDofId >& sorted_id,
                          const std::vector< DataType >& sorted_val,
                          size_t fe_ind, 
                          std::vector<DataType> &dof_values)
{
  typedef hiflow::doffem::gDofId gDofId;
  std::vector< doffem::gDofId > global_dof_ids;
  space.get_dof_indices(fe_ind, cell.index(), global_dof_ids);

  const int num_dofs = global_dof_ids.size();

  dof_values.clear();
  dof_values.resize(num_dofs, 1.e25);

  size_t num_procs = space.nb_subdom(); 

  if (num_procs == 1) 
  {
    // in sequential world, gDofIds are already sorted
    fun.GetValues(&(global_dof_ids[0]), global_dof_ids.size(), &(dof_values[0]));
  } 
  else 
  {
    // in parallel world, gDofIds are not sorted
    // -> id and val fields need to be sorted and accesses are related to
    //    a seek through the data

    std::vector< gDofId >::const_iterator it;
    for (int i = 0; i < num_dofs; ++i) 
    {
      it = std::lower_bound(sorted_id.begin(), sorted_id.end(), global_dof_ids[i]);
      const int index = it - sorted_id.begin();
      dof_values[i] = sorted_val[index];
    }
    // slow version
    // fun_.GetValues(&global_dof_ids[0], num_dofs, &dof_values[0]);
  }
  
  // TODO: is this consistent with previous sorting of dofs?
  std::vector< DataType > dof_factors;
  space.dof().get_dof_factors_on_cell(cell.index(), dof_factors);
  
  const size_t start_dof = elem.dof_offset(fe_ind);
  const size_t end_dof = start_dof + elem.nb_dof(fe_ind);
  for (size_t i=start_dof; i != end_dof; ++i)
  {
    dof_values[i-start_dof] *= dof_factors[i];
  }
}

template void extract_dof_values <float, 1> ( const VectorSpace<float, 1>& , 
                                              const mesh::Entity& ,
                                              const Element<float, 1>& ,
                                              const la::Vector<float>&,
                                              const std::vector< hiflow::doffem::gDofId >&,
                                              const std::vector< float >&, 
                                              size_t fe_ind, 
                                              std::vector<float> &);
template void extract_dof_values <float, 2> ( const VectorSpace<float, 2>& , 
                                              const mesh::Entity& ,
                                              const Element<float, 2>& ,
                                              const la::Vector<float>&,
                                              const std::vector< hiflow::doffem::gDofId >&,
                                              const std::vector< float >&, 
                                              size_t fe_ind, 
                                              std::vector<float> &);
template void extract_dof_values <float, 3> ( const VectorSpace<float, 3>& , 
                                              const mesh::Entity& ,
                                              const Element<float, 3>& ,
                                              const la::Vector<float>&,
                                              const std::vector< hiflow::doffem::gDofId >&,
                                              const std::vector< float >&, 
                                              size_t fe_ind, 
                                              std::vector<float> &);
template void extract_dof_values <double, 1> ( const VectorSpace<double, 1>& , 
                                              const mesh::Entity& , 
                                              const Element<double, 1>& ,
                                              const la::Vector<double>&,
                                              const std::vector< hiflow::doffem::gDofId >&,
                                              const std::vector< double >&, 
                                              size_t fe_ind, 
                                              std::vector<double> &);
template void extract_dof_values <double, 2> ( const VectorSpace<double, 2>& , 
                                              const mesh::Entity& , 
                                              const Element<double, 2>& ,
                                              const la::Vector<double>&,
                                              const std::vector< hiflow::doffem::gDofId >&,
                                              const std::vector< double >&, 
                                              size_t fe_ind, 
                                              std::vector<double> &);
template void extract_dof_values <double, 3> ( const VectorSpace<double, 3>& , 
                                              const mesh::Entity& , 
                                              const Element<double, 3>& ,
                                              const la::Vector<double>&,
                                              const std::vector< hiflow::doffem::gDofId >&,
                                              const std::vector< double >&, 
                                              size_t fe_ind, 
                                              std::vector<double> &);

///////////////////////////////////////////////////////////////////
/////////////// FeEvalCell ////////////////////////////////////
///////////////////////////////////////////////////////////////////
template < class DataType, int DIM >
FeEvalCell<DataType, DIM>::FeEvalCell(const VectorSpace< DataType, DIM > &space, 
                                      std::vector< hiflow::la::Vector<DataType> const *> coeffs)
: space_(space), 
  _last_cell_index(-1),
  flow_trafo_(nullptr)
{
  this->coeffs_ = coeffs;
  this->vars_.clear();
  this->vars_.resize(space.nb_var(), 0);
  for (int l=0; l<this->vars_.size(); ++l)
  {
    this->vars_[l] = l;
  }
  this->setup();
  
}

template < class DataType, int DIM >
FeEvalCell<DataType, DIM>::FeEvalCell(const VectorSpace< DataType, DIM > &space, 
                                      const la::Vector<DataType> &coeff)
: space_(space), 
  _last_cell_index(-1),
  flow_trafo_(nullptr)
{
  this->coeffs_.push_back(&coeff);
  this->vars_.clear();
  this->vars_.resize(space.nb_var(), 0);
  for (int l=0; l<this->vars_.size(); ++l)
  {
    this->vars_[l] = l;
  }
  this->setup();
  
}

template < class DataType, int DIM >
FeEvalCell<DataType, DIM>::FeEvalCell(const VectorSpace< DataType, DIM > &space, 
                                      const la::Vector<DataType> &coeff, 
                                      size_t fe_ind )
: space_(space), 
  _last_cell_index(-1),
  flow_trafo_(nullptr)
{
  assert (fe_ind < space.nb_fe());
  this->coeffs_.push_back(&coeff);
  this->vars_ = space.fe_2_var(fe_ind);
  this->setup();
}

template < class DataType, int DIM >
FeEvalCell<DataType, DIM>::FeEvalCell(const VectorSpace< DataType, DIM > &space, 
                                      const la::Vector<DataType> &coeff, 
                                      const std::vector<size_t> &fe_ind )
: space_(space), 
  _last_cell_index(-1),
  flow_trafo_(nullptr)
{
  this->coeffs_.push_back(&coeff);
  for (size_t l=0; l<fe_ind.size(); ++l)
  {
    assert (fe_ind[l] < space.nb_fe());
    std::vector<size_t> vars = space.fe_2_var(fe_ind[l]);
    
    for (size_t d=0; d<vars.size(); ++d)
    {
      this->vars_.push_back(vars[d]);
    }
  }
  this->setup();
}

template < class DataType, int DIM >
FeEvalCell<DataType, DIM>::FeEvalCell(const VectorSpace< DataType, DIM > &space, 
                                      const la::Vector<DataType> &coeff, 
                                      const std::vector<size_t>& vars,
                                      FlowTrafo<DataType, DIM> const * map )
: space_(space), 
  vars_(vars),
  _last_cell_index(-1),
  flow_trafo_(map),
  use_flow_trafo_(true)
{
  assert (flow_trafo_ != nullptr);
  this->coeffs_.push_back(&coeff);
  this->setup();
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::setup()
{
  this->gdim_ = this->space_.mesh().gdim();
  const size_t nb_fe = this->space_.nb_fe();
  const size_t nb_var = this->space_.nb_var();
  
  // get fe indices to be evaluated
  this->fe_inds_.clear();
  this->fe_ind_2_comp_.clear();
  this->fe_ind_2_comp_.resize(nb_fe);
  
  std::sort(this->vars_.begin(), this->vars_.end()); 
  auto vlast = std::unique(this->vars_.begin(), this->vars_.end()); 
  this->vars_.erase(vlast, this->vars_.end()); 
  
  this->nb_comp_ = this->vars_.size();
  this->var_order_.clear();
  this->var_order_.resize(nb_var, -1);
  
  for (size_t l=0; l<this->vars_.size(); ++l)
  {
    const size_t v = this->vars_[l];
    assert (v < this->space_.nb_var());
    
    const size_t f = this->space_.var_2_fe(v);
    const size_t c = this->space_.var_2_comp(v);
    
    this->fe_inds_.push_back(f);
    this->fe_ind_2_comp_[f].push_back(c);
    this->var_order_[v] = l;
  }
  
  std::sort(this->fe_inds_.begin(), this->fe_inds_.end()); 
  auto last = std::unique(this->fe_inds_.begin(), this->fe_inds_.end()); 
  this->fe_inds_.erase(last, this->fe_inds_.end()); 
  
  for (size_t f=0; f<nb_fe; ++f)
  {
    if (this->fe_ind_2_comp_[f].size() > 0)
    {
      std::sort(this->fe_ind_2_comp_[f].begin(), this->fe_ind_2_comp_[f].end()); 
      auto flast = std::unique(this->fe_ind_2_comp_[f].begin(), this->fe_ind_2_comp_[f].end()); 
      this->fe_ind_2_comp_[f].erase(flast, this->fe_ind_2_comp_[f].end()); 
    }
  }
  
  const size_t num_coeff = this->coeffs_.size();
  this->_dof_values.clear();
  this->_dof_values.resize(num_coeff);
  for (size_t l=0; l<num_coeff; ++l)
  {
    this->_dof_values[l].resize(nb_fe);
  }
}
  
template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::update_dof_values(const mesh::Entity& cell) const 
{
  int cur_cell_index = cell.index();
  if (cur_cell_index == this->_last_cell_index)
  {
    return;
  }
  
  // TODO: need sort_dofs and extract_dofs as defined above?
  const size_t num_coeff = this->coeffs_.size();
  
  for (size_t i=0; i<num_coeff; ++i)
  {
    for (size_t l=0; l<this->fe_inds_.size(); ++l)
    {
      const size_t f = this->fe_inds_[l];
      //this->space_.set_print(this->print_);
      this->space_.extract_dof_values (f, cur_cell_index, *(this->coeffs_[i]), this->_dof_values[i][f]);
    }
  }
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::clear_return_values(std::vector<DataType>& vals) const 
{
  if (vals.size() != this->weight_size())
  {
    vals.clear();
    vals.resize(this->weight_size(), 0.);
  }
  else
  {
    for (size_t i=0; i<vals.size(); ++i)
    {
      vals[i] = 0.;
    }
  }
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::clear_return_values(std::vector<vec >& vals) const 
{
  if (vals.size() != this->weight_size())
  {
    vals.clear();
    vals.resize(this->weight_size());
  }
  else
  {
    for (size_t i=0; i<vals.size(); ++i)
    {
      vals[i] = Coord();
    }
  }
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::evaluate (const mesh::Entity& cell, 
                                          const Coord& pt, 
                                          std::vector<DataType>& vals) const
{
  assert (this->coeffs_.size() == 1);
  
  auto cell_trafo = this->space_.fe_manager().get_cell_transformation(cell.index());
    
  assert (cell_trafo != nullptr);
  
  Coord ref_pt;

  if (!cell_trafo->inverse(pt, ref_pt))
  {
    this->clear_return_values(vals);
  }
  else
  {
    this->r_evaluate (cell, ref_pt, vals);
  }
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::r_evaluate (const mesh::Entity& cell, 
                                            const Coord& ref_pt, 
                                            std::vector<DataType>& vals) const
{
  assert (this->coeffs_.size() == 1);
  std::vector< std::vector<DataType>* > tmp_vals;
  tmp_vals.push_back(&vals);
  this->r_evaluate(cell, ref_pt, tmp_vals);
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::r_evaluate_grad (const mesh::Entity& cell, 
                                                 const Coord& ref_pt, 
                                                 std::vector<vec >& vals) const
{
  assert (this->coeffs_.size() == 1);
  std::vector< std::vector<vec >* > tmp_vals;
  tmp_vals.push_back(&vals);
  this->r_evaluate_grad(cell, ref_pt, tmp_vals);
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::r_evaluate_weight_and_grad (const mesh::Entity& cell, 
                                                            const Coord& ref_pt, 
                                                            std::vector<DataType>& vals,
                                                            std::vector<vec >& grads) const
{
  assert (this->coeffs_.size() == 1);
  
  std::vector< std::vector<DataType>* > tmp_vals;
  tmp_vals.push_back(&vals);
  
  std::vector< std::vector<vec >* > tmp_grads;
  tmp_grads.push_back(&grads);
  
  this->r_evaluate_weight_and_grad(cell, ref_pt, tmp_vals, tmp_grads);
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::evaluate_grad (const mesh::Entity& cell, 
                                               const Coord& pt, 
                                               std::vector< vec >& vals) const
{
  assert (this->coeffs_.size() == 1);

  auto cell_trafo = this->space_.fe_manager().get_cell_transformation(cell.index());
        
  assert (cell_trafo != nullptr);
  Coord ref_pt;
  if (!cell_trafo->inverse(pt, ref_pt))
  {
    this->clear_return_values(vals);
    return;
  }
  else
  {
    this->r_evaluate_grad (cell, ref_pt, vals);
    return;
  }
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::r_evaluate (const mesh::Entity& cell, 
                                            const Coord& ref_pt, 
                                            std::vector< std::vector<DataType>* > vals) const
{
  const size_t num_coeff = this->coeffs_.size();
  const size_t num_eval = vals.size();
  assert (num_eval <= num_coeff);

  for (size_t i=0; i<num_eval; ++i)
  {
    assert (vals[i] != nullptr);
    this->clear_return_values(*(vals[i]));
  }
  this->update_dof_values(cell);
     
  const auto cell_index = cell.index();
  if (cell_index != this->_last_cell_index)
  {
    this->_element.init(this->space_, cell_index);
  }
  this->_last_cell_index = cell_index;

  for (size_t l=0; l<this->fe_inds_.size(); ++l)
  {
    const size_t f = this->fe_inds_[l];
    const auto ref_fe = this->_element.get_fe(f);
    const size_t dim = ref_fe->dim();
    const size_t fe_nb_comp = ref_fe->nb_comp();
    const size_t eval_nb_comp = this->fe_ind_2_comp_[f].size();
    
    const size_t weight_size = ref_fe->weight_size();
    _weights.clear();
    _weights.resize(weight_size, 0.);
    
    // evaluate mapped FE basis functions in given point
    this->_element.N_fe(ref_pt, f, _weights);
    
    for (size_t j=0; j<num_eval; ++j)
    {
      // Global DoF Ids on the given mesh cell
      const size_t num_dofs = this->_dof_values[j][f].size();
      assert (num_dofs == dim);
  
      for (size_t k=0; k<eval_nb_comp; ++k)
      {
        const size_t c = this->fe_ind_2_comp_[f][k];
        const size_t v = this->fe_comp_2_var(f, c);
        const size_t out_index = this->ivar2ind(0,v);
        
        // Summation over weights multiplied by dof_values
        // TODO_VECTORIZE
        for (size_t i_loc = 0; i_loc < num_dofs; ++i_loc) 
        {      
          vals[j]->operator[](out_index) += this->_dof_values[j][f][i_loc] * _weights[ref_fe->iv2ind(i_loc, c)];
        }
      }
    }
  }
  if (use_flow_trafo_)
  {
    assert (this->flow_trafo_ != nullptr);
    const std::vector<size_t> trafo_vars = this->flow_trafo_->get_trafo_vars();
    const size_t nb_trafo_vars = trafo_vars.size();
    std::vector<DataType> trafo_input (nb_trafo_vars);
    std::vector<DataType> trafo_output (nb_trafo_vars);
    
    Coord phys_pt;
    
    doffem::CCellTrafoSPtr<DataType, DIM> cell_trafo 
    = this->space_.fe_manager().get_cell_transformation(cell.index());
        
    cell_trafo->transform(ref_pt, phys_pt);
    
    for (size_t j=0; j<num_eval; ++j)
    {
      for (size_t l=0; l<nb_trafo_vars; ++l)
      { 
        trafo_input[l] = vals[j]->at(this->ivar2ind(0,trafo_vars[l]));
      }
        
      this->flow_trafo_->operator()(phys_pt, trafo_input, trafo_output);
    
      for (size_t l=0; l<nb_trafo_vars; ++l)
      { 
        vals[j]->at(this->ivar2ind(0,trafo_vars[l])) = trafo_output[l];
      }
    }
  }
}
    
template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::r_evaluate_grad (const mesh::Entity& cell, 
                                                 const Coord& ref_pt, 
                                                 std::vector< std::vector< vec >* > vals) const
{
  const size_t num_coeff = this->coeffs_.size();
  const size_t num_eval = vals.size();
  assert (num_eval <= num_coeff);

  for (size_t i=0; i<num_eval; ++i)
  {
    assert (vals[i] != nullptr);
    this->clear_return_values(*(vals[i]));
  }
  this->update_dof_values(cell);

  const auto cell_index = cell.index();
  if (cell_index != this->_last_cell_index)
  {
    this->_element.init(this->space_, cell_index);
  }
  this->_last_cell_index = cell_index;

  for (size_t l=0; l<this->fe_inds_.size(); ++l)
  {
    const size_t f = this->fe_inds_[l];
    const auto ref_fe = this->_element.get_fe(f);
    const size_t dim = ref_fe->dim();
    const size_t fe_nb_comp = ref_fe->nb_comp();
    const size_t eval_nb_comp = this->fe_ind_2_comp_[f].size();
    
    const size_t weight_size = ref_fe->weight_size();
  
    // evaluate mapped FE basis functions in given point
    _weights_grad.clear();
    _weights_grad.resize(weight_size);
    this->_element.grad_N_fe(ref_pt, f, _weights_grad);
    
    for (size_t j=0; j<num_eval; ++j)
    {
      // Global DoF Ids on the given mesh cell
      const size_t num_dofs = this->_dof_values[j][f].size();
      assert (num_dofs == dim);
  
      for (size_t k=0; k<eval_nb_comp; ++k)
      {
        const size_t c = this->fe_ind_2_comp_[f][k];
        const size_t v = this->fe_comp_2_var(f, c);
        const size_t out_index = this->ivar2ind(0,v);
        
        // Summation over weights multiplied by dof_values
        // TODO_VECTORIZE
        for (size_t i_loc = 0; i_loc < num_dofs; ++i_loc) 
        {      
          vals[j]->operator[](out_index) += this->_dof_values[j][f][i_loc] * _weights_grad[ref_fe->iv2ind(i_loc, c)];
        }
      }
    }
  }
}

template < class DataType, int DIM >
void FeEvalCell<DataType, DIM>::r_evaluate_weight_and_grad (const mesh::Entity& cell, 
                                                            const Coord& ref_pt, 
                                                            std::vector< std::vector<DataType>* > vals,
                                                            std::vector< std::vector< vec >* > gradients) const
{
  const size_t num_coeff = this->coeffs_.size();
  const size_t num_eval_fe = vals.size();
  const size_t num_eval_grad = gradients.size();
  
  assert (num_eval_fe <= num_coeff);
  assert (num_eval_grad <= num_coeff);

  assert (num_eval_fe >= num_eval_grad);
  
  for (size_t i=0; i<num_eval_fe; ++i)
  {
    assert (vals[i] != nullptr);
    this->clear_return_values(*(vals[i]));
  }
  
  for (size_t i=0; i<num_eval_grad; ++i)
  {
    assert (gradients[i] != nullptr);
    this->clear_return_values(*(gradients[i]));
  }
  
  this->update_dof_values(cell);

  const auto cell_index = cell.index();
  if (cell_index != this->_last_cell_index)
  {
    this->_element.init(this->space_, cell_index);
  }
  this->_last_cell_index = cell_index;

  for (size_t l=0; l<this->fe_inds_.size(); ++l)
  {
    const size_t f = this->fe_inds_[l];
    const auto ref_fe = this->_element.get_fe(f);
    const size_t dim = ref_fe->dim();
    const size_t fe_nb_comp = ref_fe->nb_comp();
    const size_t eval_nb_comp = this->fe_ind_2_comp_[f].size();
    
    const size_t weight_size = ref_fe->weight_size();
  
    // evaluate mapped FE basis functions in given point
    _weights.clear();
    _weights.resize(weight_size, 0.);
    _weights_grad.clear();
    _weights_grad.resize(weight_size);
    
    this->_element.N_and_grad_N_fe(ref_pt, f, _weights, _weights_grad);
    
    for (size_t j=0; j<num_eval_fe; ++j)
    {
      // Global DoF Ids on the given mesh cell
      const size_t num_dofs = this->_dof_values[j][f].size();
      assert (num_dofs == dim);
  
      for (size_t k=0; k<eval_nb_comp; ++k)
      {
        const size_t c = this->fe_ind_2_comp_[f][k];
        const size_t v = this->fe_comp_2_var(f, c);
        const size_t out_index = this->ivar2ind(0,v);
        
        // Summation over weights multiplied by dof_values
        // TODO_VECTORIZE
        for (size_t i_loc = 0; i_loc < num_dofs; ++i_loc) 
        {
          vals[j]->operator[](out_index) += this->_dof_values[j][f][i_loc] * _weights[ref_fe->iv2ind(i_loc, c)];      
        }
        
        if (j < num_eval_grad)
        {
          for (size_t i_loc = 0; i_loc < num_dofs; ++i_loc) 
          {
            gradients[j]->operator[](out_index) += this->_dof_values[j][f][i_loc] * _weights_grad[ref_fe->iv2ind(i_loc, c)];
          }
        }
      }
    }
  }
}

template class FeEvalCell <float, 1>;
template class FeEvalCell <float, 2>;
template class FeEvalCell <float, 3>;
template class FeEvalCell <double, 1>;
template class FeEvalCell <double, 2>;
template class FeEvalCell <double, 3>;

///////////////////////////////////////////////////////////////////
/////////////// FeEvalLocal ///////////////////////////////////////
///////////////////////////////////////////////////////////////////


template < class DataType, int DIM >
FeEvalLocal<DataType, DIM>::FeEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                        const la::Vector<DataType> &coeff)
: space_(space), search_(nullptr)
{
  this->fe_eval_cell_ = new FeEvalCell<DataType, DIM>(space, coeff);
  this->setup();
}

template < class DataType, int DIM >
FeEvalLocal<DataType, DIM>::FeEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                        const la::Vector<DataType> &coeff, 
                                        size_t fe_ind )
: space_(space), search_(nullptr)
{
  this->fe_eval_cell_ = new FeEvalCell<DataType, DIM>(space, coeff, fe_ind);
  this->setup();
}

template < class DataType, int DIM >
FeEvalLocal<DataType, DIM>::FeEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                        const la::Vector<DataType> &coeff, 
                                        const std::vector<size_t> &fe_ind )
: space_(space), search_(nullptr)
{
  this->fe_eval_cell_ = new FeEvalCell<DataType, DIM>(space, coeff, fe_ind);
  this->setup();
}

template < class DataType, int DIM >
FeEvalLocal<DataType, DIM>::FeEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                        const la::Vector<DataType> &coeff, 
                                        const std::vector<size_t>& vars,
                                        FlowTrafo<DataType, DIM> const * map )
: space_(space), search_(nullptr)
{
  this->fe_eval_cell_ = new FeEvalCell<DataType, DIM>(space, coeff, vars, map);
  this->setup();
}

template < class DataType, int DIM >
FeEvalLocal<DataType, DIM>::~FeEvalLocal()
{
  if (this->fe_eval_cell_ != nullptr)
  {
    delete this->fe_eval_cell_;
  }
  if (this->search_ != nullptr)
  {
    delete this->search_;
  }
}

template < class DataType, int DIM >
void FeEvalLocal< DataType, DIM >::set_trial_cells(const std::vector< int > &trial_cells) const
{
  this->vec_trial_cells_ = &trial_cells;
}

template < class DataType, int DIM >
void FeEvalLocal< DataType, DIM >::set_trial_cells(const std::set< int > &trial_cells) const
{
  this->set_trial_cells_ = &trial_cells;
}

template < class DataType, int DIM >
void FeEvalLocal<DataType, DIM>::setup()
{
  mesh::MeshPtr meshptr = this->space_.meshPtr();
  assert(meshptr != nullptr);

  mesh::GDim gdim = meshptr->gdim();

  if (this->search_ != nullptr)
  {
    delete this->search_;
  }
  
  if (meshptr->is_rectangular()) 
  {
    this->search_ = new mesh::RecGridGeometricSearch<DataType, DIM>(meshptr);
  }   
  else 
  {
    this->search_ = new mesh::GridGeometricSearch<DataType, DIM>(meshptr);
  }
  
  this->vec_trial_cells_ = nullptr;
  this->set_trial_cells_ = nullptr;
}

template < class DataType, int DIM >
void FeEvalLocal<DataType, DIM>::search_points(const std::vector<Coord>& pts, 
                                               std::vector< std::vector<int> >& cell_indices, 
                                               std::vector< std::vector<Coord> >& ref_pts) const
{
  assert (this->search_ != nullptr);
  const size_t nb_pts = pts.size();
  
  if (cell_indices.size() != nb_pts)
  {
    cell_indices.resize(nb_pts);
  }
  if (ref_pts.size() != nb_pts)
  {
    ref_pts.resize(nb_pts);
  }

  for (size_t i=0; i<nb_pts; ++i)
  {
    cell_indices[i].clear();
    ref_pts[i].clear();
  
    if (this->vec_trial_cells_ != nullptr)
    {
      if (!this->vec_trial_cells_->empty()) 
      {
        this->search_->find_cell(pts[i], *this->vec_trial_cells_, cell_indices[i] , ref_pts[i]);
      }
    }
    else if (this->set_trial_cells_ != nullptr)
    {
      if (!this->set_trial_cells_->empty()) 
      {
        this->search_->find_cell(pts[i], *this->set_trial_cells_, cell_indices[i] , ref_pts[i]);
      }
    }
    else 
    {
      this->search_->find_cell(pts[i], cell_indices[i], ref_pts[i]);
    }
#ifndef NDEBUG
    bool success = this->check_ref_coords(pts[i], cell_indices[i], ref_pts[i]);
    assert (success);
#endif
  }
}

template < class DataType, int DIM >
bool FeEvalLocal<DataType, DIM>::evaluate ( const Coord& pt, 
                                            DataType& value ) const 
{
  assert (this->weight_size() == 1);
  std::vector< std::vector< DataType> > tmp_val;
  std::vector< Coord > tmp_pt (1, pt);
  std::vector<bool> found = this->evaluate_impl (tmp_pt, tmp_val);
  assert (tmp_val.size() == 1);
  value = tmp_val[0][0];
  return found[0];
}

template < class DataType, int DIM >
bool FeEvalLocal<DataType, DIM>::evaluate ( const Coord& pt, 
                                            std::vector< DataType >& vals ) const 
{
  std::vector< std::vector< DataType> > tmp_val;
  std::vector< Coord > tmp_pt (1, pt);
  std::vector<bool> found = this->evaluate_impl (tmp_pt, tmp_val);
  assert (tmp_val.size() == 1);
  vals = tmp_val[0];
  return found[0];
}

template < class DataType, int DIM >
std::vector<bool> FeEvalLocal<DataType, DIM>::evaluate ( const std::vector<Coord>& pts, 
                                                         std::vector<std::vector<DataType> >& vals ) const 
{
  return this->evaluate_impl(pts, vals); 
}

template < class DataType, int DIM >
std::vector<bool> FeEvalLocal<DataType, DIM>::evaluate_impl ( const std::vector<Coord>& pts, 
                                                              std::vector<std::vector<DataType> >& vals ) const 
{
  assert (this->search_ != nullptr);
  std::vector<bool> success (pts.size(), true);
  
  if (vals.size() != pts.size())
  {
    vals.resize(pts.size());
  }
  
  mesh::MeshPtr meshptr = this->space_.meshPtr();
  
  const size_t w_size = this->weight_size();
  
  // get cell indices and reference coords for given physical points
  std::vector< std::vector<int> > cell_indices; 
  std::vector< std::vector<Coord> > ref_pts;
  this->search_points(pts, cell_indices, ref_pts);
  
  assert (ref_pts.size() == cell_indices.size());
  std::vector<DataType> tmp_vals (w_size, 0.);

  for (size_t p=0; p<pts.size(); ++p)
  {
    int value_count = ref_pts[p].size();
    vals[p].clear();
    vals[p].resize(w_size, 0.);
    assert (ref_pts[p].size() == cell_indices[p].size());
    
    /*
    if (print_)
    {
      std::cout << "FeEvalLocal: phys point " << pts[p] << " multipl. " << value_count << std::endl;
    }
    */
    
    if (value_count > 0) 
    {
      // point was found in local cells
      for (size_t i = 0; i < value_count; ++i) 
      {
        mesh::Entity cell = meshptr->get_entity(meshptr->tdim(), cell_indices[p][i]);
        tmp_vals.clear();
        tmp_vals.resize(w_size, 0.);

        this->fe_eval_cell_->set_print(true); // lenny tassia change, set to true print
        this->fe_eval_cell_->r_evaluate (cell, ref_pts[p][i], tmp_vals);
        
        for (size_t l=0; l<w_size; ++l)
        {
          vals[p][l] += tmp_vals[l];
        }
      } 
      for (size_t l=0; l<w_size; ++l)
      {
        vals[p][l] *= 1. / static_cast< DataType >(value_count);
      }
      
      //std::cout << "     ->  " << string_from_range(vals[p].begin(), vals[p].end()) << std::endl;
    }
    else
    {
      success[p] = false;
    }
  }
  return success;
}


template < class DataType, int DIM >
bool FeEvalLocal<DataType, DIM>::check_ref_coords(const Coord& pt, 
                                                  const std::vector<int> & cell_indices, 
                                                  const std::vector<Coord> & ref_pts) const
{
  const DataType eps = 1e-8;
  assert (cell_indices.size() == ref_pts.size());
  
  for (size_t i=0; i<cell_indices.size(); ++i)
  {
    doffem::CCellTrafoSPtr<DataType, DIM> trafo =
       this->space_.get_cell_transformation(cell_indices[i]);
  
    Coord mpt;
    trafo->transform(ref_pts[i], mpt);
    
    DataType diff = norm(mpt-pt);
    if (diff > eps)
    {
      // std::cout << trafo->name() << std::endl;
      //std::cout << pt << " <=> " << ref_pts[i] << " <=> " << mpt << std::endl;
      //trafo->print_vertex_coords();
      return false;
    }
  }
  return true;
}

template class FeEvalLocal <float, 1>;
template class FeEvalLocal <float, 2>;
template class FeEvalLocal <float, 3>;
template class FeEvalLocal <double, 1>;
template class FeEvalLocal <double, 2>;
template class FeEvalLocal <double, 3>;

///////////////////////////////////////////////////////////////////
/////////////// FeEvalGlobal //////////////////////////////////////
///////////////////////////////////////////////////////////////////


template < class DataType, int DIM >
FeEvalGlobal<DataType, DIM>::FeEvalGlobal(const VectorSpace< DataType, DIM > &space, 
                                          const la::Vector<DataType> &coeff)
: FeEvalLocal<DataType, DIM> (space, coeff)
{
  this->parcom_ = new ParCom(this->space_.get_mpi_comm());
}

template < class DataType, int DIM >
FeEvalGlobal<DataType, DIM>::FeEvalGlobal(const VectorSpace< DataType, DIM > &space, 
                                          const la::Vector<DataType> &coeff, 
                                          size_t fe_ind )
: FeEvalLocal<DataType, DIM> (space, coeff, fe_ind)
{
  this->parcom_ = new ParCom(this->space_.get_mpi_comm());
}

template < class DataType, int DIM >
FeEvalGlobal<DataType, DIM>::FeEvalGlobal(const VectorSpace< DataType, DIM > &space, 
                                          const la::Vector<DataType> &coeff, 
                                          const std::vector<size_t> &fe_ind )
: FeEvalLocal<DataType, DIM> (space, coeff, fe_ind)
{
  this->parcom_ = new ParCom(this->space_.get_mpi_comm());
}

template < class DataType, int DIM >
FeEvalGlobal<DataType, DIM>::FeEvalGlobal(const VectorSpace< DataType, DIM > &space, 
                                          const la::Vector<DataType> &coeff, 
                                          const std::vector<size_t>& vars,
                                          FlowTrafo<DataType, DIM> const * map )
: FeEvalLocal<DataType, DIM> (space, coeff, vars, map)
{
  this->parcom_ = new ParCom(this->space_.get_mpi_comm());
}

template < class DataType, int DIM >
FeEvalGlobal<DataType, DIM>::~FeEvalGlobal()
{
  if (this->parcom_ != nullptr)
  {
    delete this->parcom_;
  }
}

template < class DataType, int DIM >
std::vector<bool> FeEvalGlobal<DataType, DIM>::evaluate_impl (const std::vector<Coord>& pts, 
                                                              std::vector<std::vector<DataType> >& vals ) const 
{
  const size_t num_pt = pts.size();
  
  std::vector<bool> success (num_pt, true);
  
  if (vals.size() != num_pt)
  {
    vals.resize(num_pt);
  }

  std::vector<std::vector<DataType> > local_vals;
  std::vector<bool> local_found = FeEvalLocal<DataType, DIM>::evaluate_impl( pts, local_vals );

  assert (local_vals.size() == local_found.size());
  assert (local_found.size() == num_pt);
   
  this->parcom_->sum(local_vals, vals);
   
  assert (local_vals.size() == vals.size());
  
  std::vector<DataType> local_denom (num_pt, 0.);
  std::vector<DataType> denom (num_pt, 0.);
  
  for (size_t d=0; d<num_pt; ++d)
  {
    local_denom[d] = static_cast<DataType>(local_found[d]);
  } 
  
  this->parcom_->sum(local_denom, denom);  

  for (size_t d=0; d<num_pt; ++d)
  {
    if (denom[d] > 0)
    {
      const size_t size_d = vals[d].size();
      for (size_t i=0; i<size_d; ++i)
      {
        vals[d][i] /= denom[d];
      }
    }
    else
    {
      success[d] = false;
    }
  }
  return success;
}

template class FeEvalGlobal <float, 1>;
template class FeEvalGlobal <float, 2>;
template class FeEvalGlobal <float, 3>;
template class FeEvalGlobal <double, 1>;
template class FeEvalGlobal <double, 2>;
template class FeEvalGlobal <double, 3>;

} // namespace hiflow
