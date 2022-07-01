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

#include "dof/dof_impl/dof_container.h"
#include "dof/dof_impl/dof_functional.h"
#include "fem/ansatz/ansatz_space.h"
#include "fem/reference_cell.h"
#include "mesh/entity.h"

namespace hiflow {
namespace doffem {

template < class DataType, int DIM >
DofContainer< DataType, DIM >::~DofContainer() 
{
  this->clear();
}

template < class DataType, int DIM >
size_t DofContainer< DataType, DIM >::tdim() const 
{
  return this->ref_cell()->tdim();
}

template < class DataType, int DIM >
size_t DofContainer< DataType, DIM >::nb_subentity(int tdim) const 
{
  assert(tdim >= 0 && tdim < ref_cell_->tdim());
  return this->ref_cell_->num_entities(tdim);
}

template < class DataType, int DIM > 
RefCellType DofContainer< DataType, DIM >::ref_cell_type() const 
{
  assert(this->ref_cell_);
  return this->ref_cell_->type();
}

template < class DataType, int DIM >
std::vector< cDofId > const & DofContainer< DataType, DIM >::get_dof_on_subentity(int tdim, int index) const 
{
  assert(tdim >= 0 && tdim <= ref_cell_->tdim());
  assert(index >= 0 && index < dof_on_subentity_[tdim].size());
  return dof_on_subentity_[tdim][index];
}

template < class DataType, int DIM >
void DofContainer< DataType, DIM >::push_back (DofFunctional<DataType, DIM> * dof)
{
  assert (dof != nullptr);
  this->dofs_.push_back(dof);
  this->initialized_ = false;
}

template < class DataType, int DIM >
void DofContainer< DataType, DIM >::init()
{
  this->dim_ = this->dofs_.size();
  this->init_dofs_on_subentities();  
  this->initialized_ = true;
} 

template < class DataType, int DIM >
void DofContainer< DataType, DIM >::clear()
{
  this->dim_ = 0;
  this->dof_on_subentity_.clear();
  this->initialized_ = false;  
  
  for (size_t d = 0; d < this->dofs_.size(); ++d)
  {
    if (this->dofs_[d] != nullptr)
    {
      delete this->dofs_[d];
    }
  }
  this->dofs_.clear();
}

template < class DataType, int DIM >
void DofContainer< DataType, DIM >::init_dofs_on_subentities()
{
  assert(this->ref_cell_);
  const size_t tdim = this->tdim();

  this->dof_on_subentity_.clear();
  this->dof_on_subentity_.resize(tdim+1);
  
  // loop over all subentity dimensions
  for (size_t d = 0; d <= tdim; ++d)
  {
    this->dof_on_subentity_[d].resize(this->ref_cell_->num_entities(d));
  
    // loop over all dofs
    for (size_t i=0; i<this->dofs_.size(); ++i)
    {
      // loop over all attached subentities of current dof
      std::vector<int> dof_sub_ent = this->dofs_[i]->get_subentities(d);
         
      // loop over subentities
      for (size_t l=0; l<dof_sub_ent.size(); ++l)
      {
        assert (dof_sub_ent[l] < this->dof_on_subentity_[d].size());
        this->dof_on_subentity_[d][dof_sub_ent[l]].push_back(i);
      }
    }
  }
}

template < class DataType, int DIM >
bool DofContainer< DataType, DIM >::operator==(const DofContainer< DataType, DIM > &dof_slave) const 
{
  if (this->type_ == DofContainerType::NOT_SET || dof_slave.type_ == DofContainerType::NOT_SET) 
  {
    assert(0);
  }
  return this->type_ == dof_slave.type_ 
      && this->dim_ == dof_slave.dim_
      && this->ref_cell_->type() == dof_slave.ref_cell_->type()
      && this->dof_on_subentity_ == dof_slave.dof_on_subentity_;
  // TODO: further distinction necessary?
}

template < class DataType, int DIM >
bool DofContainer< DataType, DIM >::operator<(const DofContainer< DataType, DIM > &dof_slave) const 
{
  if (this->type_ == DofContainerType::NOT_SET || dof_slave.type_ == DofContainerType::NOT_SET) {
    assert(0);
  }

  if (this->type_ < dof_slave.type_) 
  {
    return true;
  } 
  else if (this->type_ == dof_slave.type_) 
  {
    if (this->ref_cell_->type() < dof_slave.ref_cell_->type())
    {
      return true;
    }
    else if (this->ref_cell_->type() == dof_slave.ref_cell_->type())
    {
      if (this->dim_ < dof_slave.dim_)
      {
        return true;
      }
      else if (this->dim_ == dof_slave.dim_)
      {
        for (size_t tdim=0; tdim<this->dof_on_subentity_.size(); ++tdim)
        {
          assert (this->dof_on_subentity_[tdim].size() == dof_slave.dof_on_subentity_[tdim].size());
          for (size_t ent=0; ent<this->dof_on_subentity_[tdim].size(); ++ent)
          {
            if (this->dof_on_subentity_[tdim][ent].size() < dof_slave.dof_on_subentity_[tdim][ent].size())
            {
              return true;
            }
            else if (this->dof_on_subentity_[tdim][ent].size() == dof_slave.dof_on_subentity_[tdim][ent].size())
            {
              for (size_t i=0; i<this->dof_on_subentity_[tdim][ent].size(); ++i)
              {
                if (this->dof_on_subentity_[tdim][ent][i] < dof_slave.dof_on_subentity_[tdim][ent][i])
                {
                  return true;
                }
              }
            }
          } 
        }
      }
    }
  }
  return false;
}

template class DofContainer< float, 3 >;
template class DofContainer< float, 2 >;
template class DofContainer< float, 1 >;

template class DofContainer< double, 3 >;
template class DofContainer< double, 2 >;
template class DofContainer< double, 1 >;

} // namespace doffem
} // namespace hiflow
