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

#include "dof/numbering_strategy.h"
#include "dof/dof_interpolation.h"
#include "dof/dof_interpolation_pattern.h"
#include "dof/dof_partition.h"
#include "dof/fe_interface_pattern.h"
#include "mesh/entity.h"

namespace hiflow {
namespace doffem {

template < class DataType, int DIM >
void NumberingStrategy< DataType, DIM >::initialize(DofPartition< DataType, DIM > &dof) 
{
  dof_ = &dof;
  mesh_ = &(dof_->get_mesh());
  fe_manager_ = &(dof_->get_fe_manager());

  numer_cell_2_global_ = dof.numer_cell_2_global();
  cell_2_dof_factor_ = dof.cell_2_dof_factor();
  numer_cell_2_global_offsets_ = dof.numer_cell_2_global_offsets();
  //numer_cell_2_global_offsets_per_cell_ = dof.numer_cell_2_global_offsets_per_cell();
  dof_interpolation_ = &(dof.dof_interpolation());

  tdim_ = dof.tdim();
  nb_fe_ = dof.nb_fe();
}

/// \param description is an optional parameter that should describe what
///                    the permutation represents

template < class DataType, int DIM >
void NumberingStrategy< DataType, DIM >::apply_permutation(const std::vector< gDofId > &permutation, 
                                                           const std::string &description) 
{
  // apply permutation to cell2global

  // DoF IDs are used in numer_cell_2_global_ only
//  std::vector<DataType> old_factors (*this->numer_cell_2_factor_);
  for (size_t i = 0, e_i = numer_cell_2_global_->size(); i != e_i; ++i) 
  {
    (*numer_cell_2_global_)[i] = permutation[(*numer_cell_2_global_)[i]];
  }
  
  // apply permutation to DofInterpolation

  dof_interpolation_->apply_permutation(permutation);

  // calculate number of dofs, as this could have changed

  this->dof_->update_number_of_dofs(this->local_nb_dofs_total_, 
                                    this->local_nb_dofs_for_fe_,
                                    description);
}

/// \param description is an optional parameter that should describe what
///                    the permutation represents

/*
template < class DataType, int DIM >
void NumberingStrategy< DataType, DIM >::update_number_of_dofs(const std::string &description) 
{
  // Calculate number of DoFs

  local_nb_dofs_total_ = 0;
  for (size_t i = 0, e_i = numer_cell_2_global_->size(); i != e_i; ++i) 
  {
    if ((*numer_cell_2_global_)[i] > local_nb_dofs_total_) 
    {
      local_nb_dofs_total_ = (*numer_cell_2_global_)[i];
    }
  }
  ++local_nb_dofs_total_;

  // Calculate number of Dofs for each variable
  local_nb_dofs_for_fe_.clear();
  local_nb_dofs_for_fe_.resize(nb_fe_, 0);
  for (size_t fe_ind = 0; fe_ind != nb_fe_; ++fe_ind) 
  {
    int begin_offset = (*numer_cell_2_global_offsets_)[fe_ind][0];
    int end_offset;

    if (fe_ind < nb_fe_ - 1) 
    {
      end_offset = (*numer_cell_2_global_offsets_)[fe_ind + 1][0];
    } 
    else 
    {
      end_offset = numer_cell_2_global_->size();
    }

    for (size_t i = begin_offset; i < end_offset; ++i) 
    {
      if ((*numer_cell_2_global_)[i] > local_nb_dofs_for_fe_[fe_ind]) 
      {
        local_nb_dofs_for_fe_[fe_ind] = (*numer_cell_2_global_)[i];
      }
    }

    for (size_t j = 0; j != fe_ind; ++j) 
    {
      local_nb_dofs_for_fe_[fe_ind] -= local_nb_dofs_for_fe_[j];
    }

    ++local_nb_dofs_for_fe_[fe_ind];
  }
}
*/

template < class DataType, int DIM >
void NumberingStrategy< DataType, DIM >::get_points(const mesh::Entity &entity,
                                                    std::vector< Coord > &points) 
{
  points.reserve(points.size() + entity.num_vertices());
  for (size_t p = 0; p != entity.num_vertices(); ++p) 
  {
    std::vector< DataType > coords;
    entity.get_coordinates(coords);
    points.push_back( Coord(coords) );
  }
}

template class NumberingStrategy<float, 3>;
template class NumberingStrategy<float, 2>;
template class NumberingStrategy<float, 1>;
template class NumberingStrategy<double, 3>;
template class NumberingStrategy<double, 2>;
template class NumberingStrategy<double, 1>;

} // namespace doffem
} // namespace hiflow
