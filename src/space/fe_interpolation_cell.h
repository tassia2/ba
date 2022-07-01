// Copyright (C) 2011-2020 Vincent Heuveline
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

#ifndef HIFLOW_SPACE_FE_INTERPOLATION_CELL
#define HIFLOW_SPACE_FE_INTERPOLATION_CELL

/// \author Philipp Gerstner

#include <vector>

#include "dof/dof_impl/dof_container.h"
#include "fem/fe_reference.h"
#include "fem/fe_mapping.h"
#include "space/vector_space.h"

namespace hiflow {

namespace mesh {
class Entity;
}

namespace la {

}

/// \brief Class for computing the cell-wise nodal interpolation w.r.t. a given Fe space 
/// The template argument type "Functor" may contain multiple functions and should provied the following routines: </br>
///
/// size_t nb_func()      : returns number of functions contained in functor
/// size_t nb_comp()      : returns number of components of functions (all functons are assumed to have same number of components)
/// size_t iv2ind(i, c)   : indexing of weights. i <-> index of function, c <-> index of component 
/// size_t weight_size()  : size of weights vector (return argument of evaluate)
/// void evaluate(const Entity& cell, 
///               const Vec<DIM, DataType>& pt, 
///               std::vector<DataType>& weights): returns function values on given entity at given physical coordinates

template < class DataType, int DIM, class Functor > 
class FeInterCellNodal 
{
  
public:

  FeInterCellNodal(const VectorSpace< DataType, DIM > &space);

  ~FeInterCellNodal()
  {
  }
   
  void set_function (Functor const * func) const;
  
  void compute_fe_coeff (mesh::Entity const * cell, 
                         size_t fe_ind, 
                         std::vector< std::vector<DataType> >& coeff) const;
protected:
  mutable Functor const * func_;
  const VectorSpace< DataType, DIM>& space_;

  size_t num_cells_;
  mutable size_t nb_func_;
  mutable size_t nb_comp_;
  
  mutable std::vector< doffem::cDofId > loc_dofs_on_cell_;
};

////////////////////////////////////////////////////
///////////// FeInterNodal /////////////////////////
////////////////////////////////////////////////////

template < class DataType, int DIM, class Functor >
FeInterCellNodal<DataType, DIM, Functor>::FeInterCellNodal(const VectorSpace< DataType, DIM > &space)
: space_(space), func_(nullptr)
{
  this->num_cells_ = this->space_.meshPtr()->num_entities(DIM);
}

template < class DataType, int DIM, class Functor >
void FeInterCellNodal<DataType, DIM, Functor>::set_function(Functor const * func) const
{
  assert (func != nullptr);
  this->func_ = func;
  this->nb_func_ = this->func_->nb_func();
  this->nb_comp_ = this->func_->nb_comp();
}

template < class DataType, int DIM, class Functor >
void FeInterCellNodal<DataType, DIM, Functor>::compute_fe_coeff (mesh::Entity const * cell, 
                                                                 size_t fe_ind, 
                                                                 std::vector< std::vector<DataType> >& coeff) const
{ 
  assert (cell != nullptr);
  const int cell_index = cell->index();
  
  assert (this->func_ != nullptr);
  assert (cell_index >= 0);
  assert (cell_index < this->num_cells_);
  assert (fe_ind < this->space_.nb_fe());
  
  // get reference element for fe index fe_ind
  auto ref_fe = this->space_.fe_manager().get_fe(cell_index, fe_ind);
  
  assert (this->nb_comp_ == ref_fe->nb_comp());
  
  // Get corresponding dof container
  auto dofs = ref_fe->dof_container();
  auto fe_trafo = ref_fe->fe_trafo();
  
  // create object that maps the user-defined function to 
  // a function defined on the reference cell. This object is needed for evaluating
  // the dof functionals
  doffem::MappingPhys2Ref < DataType, DIM, Functor> * ref_cell_eval
      = new doffem::MappingPhys2Ref < DataType, DIM, Functor> (this->func_, 
                                                               cell, 
                                                               fe_trafo,
                                                               this->space_.get_cell_transformation(cell_index));
                                                               
  // get the local dof ids on current cell
  const size_t nb_dofs_on_cell = dofs->nb_dof_on_cell();
  if (this->loc_dofs_on_cell_.size() != nb_dofs_on_cell)
  {
    this->loc_dofs_on_cell_.resize(nb_dofs_on_cell);
    for (size_t l=0; l<nb_dofs_on_cell; ++l)
    {
      this->loc_dofs_on_cell_[l] = l;
    }
  }
  
  // evaluate dof functionals attached of current cell for func as input
  assert (nb_dofs_on_cell > 0);
  coeff.resize(nb_dofs_on_cell);
  
  dofs->evaluate (ref_cell_eval, this->loc_dofs_on_cell_, coeff);
  assert (coeff.size() == nb_dofs_on_cell);
  assert (coeff[0].size() == ref_cell_eval->nb_func());
  delete ref_cell_eval;
}


} // namespace hiflow
#endif
