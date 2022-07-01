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

#include "space/element.h"

#include "dof/dof_partition.h"
#include "dof/dof_impl/dof_container.h"
#include "dof/dof_impl/dof_functional.h"
#include "dof/dof_impl/dof_functional_point.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/fe_reference.h"
#include "fem/fe_manager.h"
#include "fem/fe_transformation.h"
#include "space/vector_space.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"

//#include <boost/bind/bind.hpp> 

namespace hiflow {

//using namespace boost::placeholders;

template <class DataType, int DIM>
Element<DataType, DIM>::Element(const VectorSpace< DataType, DIM > &space, int cell_index)
  : space_(&space),
    cell_(space.mesh().get_entity(space.tdim(), cell_index)) 
{
  this->init(space, cell_index);
}

template <class DataType, int DIM>
void Element<DataType, DIM>::init (const VectorSpace< DataType, DIM > &space, int cell_index)
{
  space_ = &space;
  cell_ = space.mesh().get_entity(space.tdim(), cell_index);
  this->cell_index_ = cell_index;
  this->dim_ = 0;
  this->nb_comp_ = 0;
  this->weight_size_ = 0; 
  this->mapping_eval_ = false;
  
  const size_t nb_fe = this->space_->fe_manager().nb_fe();
  this->weight_offsets_.resize(nb_fe, 0);
  this->dim_offsets_.resize(nb_fe, 0);

  size_t weight_offset = 0;
  size_t dim_offset = 0;
  for (size_t i=0; i<nb_fe; ++i)
  { 
    auto cur_fe = this->space_->fe_manager().get_fe(this->cell_index_, i);
      
    this->weight_offsets_[i] = weight_offset;
    this->dim_offsets_[i] = dim_offset;
    this->nb_comp_ += cur_fe->nb_comp();

    weight_offset += cur_fe->weight_size();
    dim_offset += cur_fe->dim();
  }
  this->weight_size_ = weight_offset;
  this->dim_ = dim_offset;
  
}

 
template<class DataType, int DIM>
bool Element<DataType, DIM>::is_boundary() const 
{
  const mesh::TDim facet_dim = this->space_->mesh().tdim() - 1;
  for (mesh::IncidentEntityIterator it = this->cell_.begin_incident(facet_dim);
       it != this->cell_.end_incident(facet_dim); ++it) 
  {
    bool is_boundary = this->space_->mesh().is_boundary_facet(it->index());
    if (is_boundary) 
    {
      return true;
    }
  }
  return false;
}


template<class DataType, int DIM>
bool Element<DataType, DIM>::is_local() const 
{
  const mesh::TDim tdim = this->space_->mesh().tdim();
  if (this->space_->mesh().has_attribute("_sub_domain_", tdim)) {
    int cell_sub_subdomain;
    this->cell_.get("_sub_domain_", &cell_sub_subdomain);
    return cell_sub_subdomain == this->space_->dof().my_subdom();
  }
  // assume it is true if we have no other information
  return true;
}

template<class DataType, int DIM>
std::vector< int > Element<DataType, DIM>::boundary_facet_numbers() const 
{
  const mesh::TDim facet_dim = this->space_->mesh().tdim() - 1;
  std::vector< int > facet_numbers(0);
  if (is_boundary()) {
    facet_numbers.reserve(6);
    int i = 0;
    for (mesh::IncidentEntityIterator it = this->cell_.begin_incident(facet_dim);
         it != this->cell_.end_incident(facet_dim); ++it) 
    {
      if (this->space_->mesh().is_boundary_facet(it->index())) 
      {
        facet_numbers.push_back(i);
      }
      ++i;
    }
  }
  return facet_numbers;
}

template<class DataType, int DIM>
size_t Element<DataType, DIM>::iv2ind (size_t i, size_t var) const
{
  const size_t fe_ind = this->var_2_fe(var);

#ifndef NDEBUG
  assert (this->var_2_comp(var) < this->space_->fe_manager().get_fe(this->cell_index_, fe_ind)->nb_comp());
  assert (i < this->space_->fe_manager().get_fe(this->cell_index_, fe_ind)->dim());
#endif

  return this->weight_offsets_[fe_ind] 
         + this->space_->fe_manager().get_fe(this->cell_index_, fe_ind)->iv2ind(i, this->var_2_comp(var));
}

template<class DataType, int DIM>
void Element<DataType, DIM>::N (const Coord &ref_pt, std::vector< DataType > &weights) const
{ 
  const size_t nb_fe = this->nb_fe();
  size_t offset = 0;

  // loop over fe types
  for (size_t i=0; i<nb_fe; ++i)
  {
    auto ref_fe = this->get_fe(i);
    const size_t cur_weight_size = ref_fe->weight_size();
    
    this->cur_weights_.clear();
    this->cur_weights_.resize(cur_weight_size, 0.);
    this->N_fe (ref_pt, i, cur_weights_);

    assert (weights.size() > offset + cur_weight_size);
    for (size_t l=0; l<cur_weight_size; ++l)
    {
      weights[offset+l] = cur_weights_[l];
    }
    offset += cur_weight_size;
  }
}

template<class DataType, int DIM>
void Element<DataType, DIM>::grad_N (const Coord &ref_pt, std::vector< vec > &gradients) const
{ 
  const size_t nb_fe = this->nb_fe();
  size_t offset = 0;

  // loop over fe types
  for (size_t i=0; i<nb_fe; ++i)
  {
    auto ref_fe = this->get_fe(i);
    const size_t cur_weight_size = ref_fe->weight_size();
    
    this->cur_grads_.clear();
    this->cur_grads_.resize (cur_weight_size);
    this->grad_N_fe (ref_pt, i, cur_grads_);

    assert (gradients.size() > offset + cur_weight_size);
    for (size_t l=0; l<cur_weight_size; ++l)
    {
      gradients[offset+l] = cur_grads_[l];
    }
    offset += cur_weight_size;
  }
}

template<class DataType, int DIM>
void Element<DataType, DIM>::hessian_N (const Coord &ref_pt, std::vector< mat > &hessians) const
{ 
  const size_t nb_fe = this->nb_fe();
  size_t offset = 0;

  // loop over fe types
  for (size_t i=0; i<nb_fe; ++i)
  {
    auto ref_fe = this->get_fe(i);
    const size_t cur_weight_size = ref_fe->weight_size();
    
    this->cur_hessians_.clear();
    this->cur_hessians_.resize (cur_weight_size);
    this->hessian_N_fe (ref_pt, i, cur_hessians_);

    assert (hessians.size() > offset + cur_weight_size);
    for (size_t l=0; l<cur_weight_size; ++l)
    {
      hessians[offset+l] = cur_hessians_[l];
    }
    offset += cur_weight_size;
  }
}

template<class DataType, int DIM>
void Element<DataType, DIM>::N_var (const Coord &ref_pt, size_t var, std::vector< DataType > &weights) const
{ 
  assert (var < this->nb_var());
  const size_t fe_ind = this->var_2_fe(var);
  const size_t comp = this->var_2_comp(var);

  auto ref_fe = this->get_fe(fe_ind);
  const size_t fe_weight_size = ref_fe->weight_size();
  const size_t dim = ref_fe->dim();

  this->cur_weights_.clear();
  this->cur_weights_.resize (fe_weight_size, 0.);
  this->N_fe (ref_pt, fe_ind, this->cur_weights_);

  assert (weights.size() == dim);
  for (size_t i=0; i<dim; ++i)
  {
    weights[i] = this->cur_weights_[ref_fe->iv2ind(i,comp)];
  }
}

template<class DataType, int DIM>
void Element<DataType, DIM>::grad_N_var (const Coord &ref_pt, size_t var, std::vector< vec > &gradients) const
{ 
  assert (var < this->nb_var());
  const size_t fe_ind = this->var_2_fe(var);
  const size_t comp = this->var_2_comp(var);

  auto ref_fe = this->get_fe(fe_ind);
  const size_t fe_weight_size = ref_fe->weight_size();
  const size_t dim = ref_fe->dim();

  this->cur_grads_.clear();
  this->cur_grads_.resize(fe_weight_size);
  this->grad_N_fe (ref_pt, fe_ind, this->cur_grads_);

  assert (gradients.size() == dim);
  for (size_t i=0; i<dim; ++i)
  {
    gradients[i] = this->cur_grads_[ref_fe->iv2ind(i,comp)];
  }
}

template<class DataType, int DIM>
void Element<DataType, DIM>::hessian_N_var (const Coord &ref_pt, size_t var, std::vector< mat > &hessians) const
{ 
  assert (var < this->nb_var());
  const size_t fe_ind = this->var_2_fe(var);
  const size_t comp = this->var_2_comp(var);

  auto ref_fe = this->get_fe(fe_ind);
  const size_t fe_weight_size = ref_fe->weight_size();
  const size_t dim = ref_fe->dim();

  this->cur_hessians_.clear();
  this->cur_hessians_.resize(fe_weight_size);
  this->hessian_N_fe (ref_pt, fe_ind, this->cur_hessians_);

  assert (hessians.size() == dim);
  for (size_t i=0; i<dim; ++i)
  {
    hessians[i] = this->cur_hessians_[ref_fe->iv2ind(i,comp)];
  }
}

template<class DataType, int DIM>
void Element<DataType, DIM>::N_fe (const Coord &ref_pt, 
                                   size_t fe_ind, 
                                   std::vector< DataType > &weight) const
{
  doffem::CCellTrafoSPtr<DataType, DIM> cell_trafo = this->get_cell_transformation();
  assert (cell_trafo->contains_reference_point(ref_pt));
  
  // evaluate shape functions on reference cell
  auto ref_fe = this->get_fe(fe_ind);
  const size_t dim = ref_fe->dim();
  const size_t nb_comp = ref_fe->nb_comp();

  assert (weight.size() == ref_fe->weight_size());

  this->shape_vals_.clear();
  this->shape_vals_.resize (ref_fe->weight_size(), 0.);
  ref_fe->N(ref_pt, this->shape_vals_);

  // map shape function values to element living on physical cell
  //IndexFunction ind_fun = boost::bind ( &doffem::RefElement<DataType, DIM>::iv2ind, ref_fe, _1, _2);
  //auto ind_fun = [ref_fe] (size_t _i, size_t _var) { return ref_fe->iv2ind(_i, _var); };

  ref_fe->fe_trafo()->map_shape_function_values (*cell_trafo, ref_pt,
                                                 0, dim, nb_comp, *ref_fe,
                                                 this->shape_vals_, weight);
}

template<class DataType, int DIM>
void Element<DataType, DIM>::grad_N_fe (const Coord &ref_pt, 
                                        size_t fe_ind, 
                                        std::vector< vec > &gradients) const
{
  doffem::CCellTrafoSPtr<DataType, DIM> cell_trafo = this->get_cell_transformation();
  assert (cell_trafo->contains_reference_point(ref_pt));

  // evaluate shape functions values and derivatives on reference cell
  auto ref_fe = this->get_fe(fe_ind);
  assert (gradients.size() == ref_fe->weight_size());
  
  const size_t dim = ref_fe->dim();
  const size_t nb_comp = ref_fe->nb_comp();

  this->shape_vals_.clear();
  this->shape_vals_.resize (ref_fe->weight_size(), 0.);
  
  this->shape_grads_.clear();
  this->shape_grads_.resize (ref_fe->weight_size());

  ref_fe->N(ref_pt, shape_vals_);
  ref_fe->grad_N(ref_pt, shape_grads_);
  
  // evaluate mapped values
  this->mapped_vals_.clear();
  this->mapped_vals_.resize (ref_fe->weight_size());
  this->N_fe (ref_pt, fe_ind, mapped_vals_);
  
  // map shape function values to element living on physical cell
  // IndexFunction ind_fun = boost::bind ( &doffem::RefElement<DataType, DIM>::iv2ind, ref_fe, _1, _2);
  //auto ind_fun = [ref_fe] (size_t _i, size_t _var) { return ref_fe->iv2ind(_i, _var); };

  ref_fe->fe_trafo()->map_shape_function_gradients (*cell_trafo, ref_pt,
                                                    0, dim, nb_comp, *ref_fe,
                                                    shape_vals_, shape_grads_,
                                                    mapped_vals_, gradients);
}

template<class DataType, int DIM>
void Element<DataType, DIM>::N_and_grad_N_fe (const Coord &ref_pt, 
                                              size_t fe_ind, 
                                              std::vector< DataType > &weight,
                                              std::vector< vec > &gradients) const
{
  doffem::CCellTrafoSPtr<DataType, DIM> cell_trafo = this->get_cell_transformation();
  assert (cell_trafo->contains_reference_point(ref_pt));
  
  // evaluate shape functions on reference cell
  auto ref_fe = this->get_fe(fe_ind);
  const size_t dim = ref_fe->dim();
  const size_t nb_comp = ref_fe->nb_comp();

  assert (weight.size() == ref_fe->weight_size());
  assert (gradients.size() == ref_fe->weight_size());

  this->shape_vals_.clear();
  this->shape_vals_.resize (ref_fe->weight_size(), 0.);
  
  this->shape_grads_.clear();
  this->shape_grads_.resize (ref_fe->weight_size());
  
  ref_fe->N(ref_pt, shape_vals_);
  ref_fe->grad_N(ref_pt, shape_grads_);

  // map shape function values to element living on physical cell
  // IndexFunction ind_fun = boost::bind ( &doffem::RefElement<DataType, DIM>::iv2ind, ref_fe, _1, _2);
  //auto ind_fun = [ref_fe] (size_t _i, size_t _var) { return ref_fe->iv2ind(_i, _var); };

  ref_fe->fe_trafo()->map_shape_function_values (*cell_trafo, ref_pt,
                                                 0, dim, nb_comp, *ref_fe,
                                                 shape_vals_, weight);
                                                 

  // map shape function values to element living on physical cell
  ref_fe->fe_trafo()->map_shape_function_gradients (*cell_trafo, ref_pt,
                                                    0, dim, nb_comp, *ref_fe,
                                                    shape_vals_, shape_grads_,
                                                    weight, gradients);
}

template<class DataType, int DIM>
void Element<DataType, DIM>::hessian_N_fe (const Coord &ref_pt, 
                                           size_t fe_ind, 
                                           std::vector< mat > &hessians) const
{
  doffem::CCellTrafoSPtr<DataType, DIM> cell_trafo = this->get_cell_transformation();
  assert (cell_trafo->contains_reference_point(ref_pt));

  // evaluate shape functions values and derivatives on reference cell
  auto ref_fe = this->get_fe(fe_ind);
  const size_t dim = ref_fe->dim();
  const size_t nb_comp = ref_fe->nb_comp();

  assert (hessians.size() == ref_fe->weight_size());
  
  this->shape_grads_.clear();
  this->shape_hessians_.clear();
  this->shape_grads_.resize (ref_fe->weight_size());
  this->shape_hessians_.resize (ref_fe->weight_size());

  ref_fe->hessian_N(ref_pt, shape_hessians_);
  ref_fe->grad_N(ref_pt, shape_grads_);

  // evaluate mapped gradients
  this->mapped_grads_.clear();
  this->mapped_grads_.resize (ref_fe->weight_size());
  this->grad_N_fe (ref_pt, fe_ind, mapped_grads_);

  // map shape function values to element living on physical cell
  //IndexFunction ind_fun = boost::bind ( &doffem::RefElement<DataType, DIM>::iv2ind, ref_fe, _1, _2);
  //auto ind_fun = [ref_fe] (size_t _i, size_t _var) { return ref_fe->iv2ind(_i, _var); };

  ref_fe->fe_trafo()->map_shape_function_hessians (*cell_trafo, ref_pt,
                                                   0, dim, nb_comp, *ref_fe,
                                                   shape_grads_, shape_hessians_,   
                                                   mapped_grads_, hessians);
}

template < class DataType, int DIM >
void Element<DataType, DIM>::print_lagrange_dof_coords (size_t fe_ind) const
{
  // get reference element
  auto ref_fe = this->get_fe(fe_ind);

  // get dof container
  auto dof_container = ref_fe->dof_container();
  
  // get cell transformation
  auto cell_trafo = this->get_cell_transformation();
  
  // loop through dof functionals
  const int num_dofs = dof_container->nb_dof_on_cell();
  
  for (int i=0; i!=num_dofs; ++i)
  {
    // get dof functional
    auto dof_func_i = dof_container->get_dof(i);
    
    // cast to point evaluation
    doffem::DofPointEvaluation<DataType, DIM> const * dof_func_point 
      = dynamic_cast<doffem::DofPointEvaluation<DataType, DIM> const *>(dof_func_i);
    
    if (dof_func_point == nullptr)
    {
      continue;
    }
   
    // get point
    Coord dof_point = dof_func_point->get_point();
    
    // map point to physical space
    Coord phys_dof_point = cell_trafo->transform(dof_point);
    
    int gl_dof_id = this->space_->dof().cell2global(fe_ind, this->cell_index_, i);
    
    std::vector<DataType> print_coord(DIM);
    for (int d=0; d!=DIM; ++d)
    {
      print_coord[d] = phys_dof_point[d];
    }
    std::cout << "Rank " << this->space_->rank() << ": gl dof id " << gl_dof_id 
              << " -> " << string_from_range(print_coord.begin(), print_coord.end()) << std::endl;
  }
}

template class Element<float, 3 >;
template class Element<float, 2 >;
template class Element<float, 1 >;

template class Element<double, 3 >;
template class Element<double, 2 >;
template class Element<double, 1 >;

} // namespace hiflow
