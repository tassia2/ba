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

#include "space/space_tools.h"
#include "space/vector_space.h"
#include "dof/dof_partition.h"
#include "dof/numbering_strategy.h"
#include "dof/numbering_lagrange.h"
#include "dof/dof_impl/dof_container.h"
#include "dof/dof_impl/dof_functional_point.h"
#include "fem/fe_reference.h"
#include "fem/fe_mapping.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "linear_algebra/vector.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"
#include "mesh/mesh.h"
#include "mesh/types.h"

namespace hiflow {

template < class DataType, int DIM >
void interpolate_constrained_std_vector(const VectorSpace< DataType, DIM > &space,
                                        la::Vector< DataType > &vector) 
{
  // TODO: necessary to take sub-domain into account here?
  const doffem::DofInterpolation<DataType> &interp = space.dof().dof_interpolation();

  const size_t num_constrained_dofs = interp.size();
  
  LOG_INFO("# constrained dofs", num_constrained_dofs);
  
  // return early if there are no constrained dofs
  if (num_constrained_dofs == 0) {
    return;
  }

  std::vector< int > constrained_dofs;
  constrained_dofs.reserve(num_constrained_dofs);
  std::vector< DataType > constrained_values;
  constrained_values.reserve(num_constrained_dofs);

  std::vector< int > dependencies;
  std::vector< DataType > coefficients;
  std::vector< DataType > dependency_values;

  for (typename doffem::DofInterpolation<DataType>::const_iterator it = interp.begin(),
                                                end = interp.end(); it != end; ++it) {
    if (space.dof().is_dof_on_subdom(it->first)) {

      const size_t num_dependencies = it->second.size();
      LOG_DEBUG(1, " num_dep " << num_dependencies);
      // probably should not happen, but we check to avoid later problems
      if (num_dependencies > 0) {

        dependencies.resize(num_dependencies, -1);
        coefficients.resize(it->second.size(), 0.);
        dependency_values.resize(num_dependencies, 0.);

        for (size_t i = 0; i != num_dependencies; ++i) {
          dependencies[i] = it->second[i].first;  // id of dependencies
          coefficients[i] = it->second[i].second; // coefficient of dependencies
        }

        LOG_DEBUG(1,
                  string_from_range(dependencies.begin(), dependencies.end()));
        LOG_DEBUG(1,
                  string_from_range(coefficients.begin(), coefficients.end()));

        // get values of dependency dofs from vector
        vector.GetValues(&dependencies.front(), num_dependencies,
                         &dependency_values.front());

        // compute dot product of dependency_values and coefficients
        DataType val = 0.;
        for (size_t i = 0; i != num_dependencies; ++i) {
          val += coefficients[i] * dependency_values[i];
        }

        // store information
        constrained_dofs.push_back(it->first);
        constrained_values.push_back(val);
      } else {
        LOG_INFO("Constrained DoF without dependencies found", true);
        quit_program();
      }
    }
  }

  // write constrained dofs to vector
  vector.SetValues(&constrained_dofs.front(), constrained_dofs.size(),
                   &constrained_values.front());
}

template void interpolate_constrained_std_vector<double, 3> (const VectorSpace< double, 3 > &, la::Vector< double > &);
template void interpolate_constrained_std_vector<double, 2> (const VectorSpace< double, 2 > &, la::Vector< double > &);
template void interpolate_constrained_std_vector<double, 1> (const VectorSpace< double, 1 > &, la::Vector< double > &);
template void interpolate_constrained_std_vector<float, 3> (const VectorSpace< float, 3 > &, la::Vector< float > &);
template void interpolate_constrained_std_vector<float, 2> (const VectorSpace< float, 2 > &, la::Vector< float > &);
template void interpolate_constrained_std_vector<float, 1> (const VectorSpace< float, 1 > &, la::Vector< float > &);

template < class DataType, int DIM >
void interpolate_constrained_vector( const VectorSpace< DataType, DIM > &space, la::Vector< DataType > &vector)
{
  interpolate_constrained_std_vector<DataType, DIM>(space, vector);
}

template void interpolate_constrained_vector<double, 3> (const VectorSpace< double, 3 > &, la::Vector< double > &);
template void interpolate_constrained_vector<double, 2> (const VectorSpace< double, 2 > &, la::Vector< double > &);
template void interpolate_constrained_vector<double, 1> (const VectorSpace< double, 1 > &, la::Vector< double > &);
template void interpolate_constrained_vector<float, 3> (const VectorSpace< float, 3 > &, la::Vector< float > &);
template void interpolate_constrained_vector<float, 2> (const VectorSpace< float, 2 > &, la::Vector< float > &);
template void interpolate_constrained_vector<float, 1> (const VectorSpace< float, 1 > &, la::Vector< float > &);

template < class DataType, int DIM >
void get_lagrange_dof_coordinates ( const VectorSpace< DataType, DIM > &space, 
                                    int fe_ind,
                                    std::vector< Vec<DIM, DataType> >& dof_coords,
                                    std::vector< doffem::gDofId >& dof_global_id,
                                    std::vector< bool >& dof_is_local)
{
  mesh::ConstMeshPtr mesh = space.meshPtr();
  const int tdim = mesh->tdim();
  const int num_dof = space.nb_dofs_local() + space.nb_dofs_ghost();
  
  dof_coords.clear();
  dof_coords.resize(num_dof);
  dof_is_local.clear();
  dof_is_local.resize(num_dof);
  dof_global_id.clear();
  dof_global_id.resize(num_dof);

  auto cell_begin = mesh->begin(tdim);
  auto cell_end = mesh->end(tdim);

  auto fe_manager = space.fe_manager();
  auto dof_partition = space.dof();
  std::vector< doffem::lDofId > local_dof_ids;
  std::vector< doffem::gDofId > global_dof_ids;
  std::vector<int> dofs_visited(num_dof,0);

  for (auto cell_it = cell_begin; cell_it != cell_end; ++cell_it)
  {
    const auto cell_index = cell_it->index();
    auto cell_trafo_ptr = space.get_cell_transformation(cell_index);
    auto ref_fe = fe_manager.get_fe(cell_index, fe_ind);
    auto dof_container = ref_fe->dof_container();
    const auto nb_dofs_on_cell = space.nb_dof_on_cell(fe_ind, cell_index);

    local_dof_ids.clear();
    global_dof_ids.clear();
    space.get_dof_indices_local(fe_ind, cell_index, local_dof_ids);
    space.get_dof_indices(fe_ind, cell_index, global_dof_ids);

    assert (local_dof_ids.size() == nb_dofs_on_cell);
    assert (global_dof_ids.size() == nb_dofs_on_cell);
    assert (dof_container->nb_dof_on_cell() == nb_dofs_on_cell);

    for (int i=0; i!=nb_dofs_on_cell; ++i)
    {
      auto dof_functional = dof_container->get_dof(i);
      auto dof_pt_eval = dynamic_cast< const doffem::DofPointEvaluation<DataType, DIM>* >(dof_functional); 

      assert (dof_pt_eval != nullptr);

      const auto local_dof = local_dof_ids[i];
      const auto global_dof = global_dof_ids[i];
      
      assert (local_dof >= 0);
      assert (local_dof < dof_coords.size());

      auto dof_pt_ref = dof_pt_eval->get_point();
      Vec<DIM, DataType> dof_pt_phy;
      cell_trafo_ptr->transform(dof_pt_ref, dof_pt_phy);

      //std::cout << local_dof << " " << global_dof << ": " << dof_pt_ref << " -> " << dof_pt_phy << std::endl;
      
      dof_coords[local_dof] = dof_pt_phy;
      dof_is_local[local_dof] = dof_partition.is_dof_on_subdom(global_dof);
      dof_global_id[local_dof] = global_dof;
      dofs_visited[local_dof] = 1;
    }
  }
  int num_dofs_visited = std::accumulate(dofs_visited.begin(),dofs_visited.end(), 0 );

  PLOG_INFO (space.rank(), "#dofs", "local = " << space.nb_dofs_local() 
                               << ", ghost = " << space.nb_dofs_ghost());
  PLOG_INFO (space.rank(), "#dofs total", num_dof);
  PLOG_INFO (space.rank(), "#dofs visited", num_dofs_visited);
  assert (num_dof == num_dofs_visited);
}

template void get_lagrange_dof_coordinates<float, 1> ( const VectorSpace< float, 1 > &, int,
                                                       std::vector< Vec<1, float> >&,
                                                       std::vector< doffem::gDofId >&,
                                                       std::vector< bool >&);

template void get_lagrange_dof_coordinates<float, 2> ( const VectorSpace< float, 2 > &, int,
                                                       std::vector< Vec<2, float> >&,
                                                       std::vector< doffem::gDofId >&,
                                                       std::vector< bool >&);

template void get_lagrange_dof_coordinates<float, 3> ( const VectorSpace< float, 3 > &, int,
                                                       std::vector< Vec<3, float> >&,
                                                       std::vector< doffem::gDofId >&,
                                                       std::vector< bool >&);

template void get_lagrange_dof_coordinates<double, 1> ( const VectorSpace< double, 1 > &, int,
                                                        std::vector< Vec<1, double> >&,
                                                        std::vector< doffem::gDofId >&,
                                                        std::vector< bool >&);

template void get_lagrange_dof_coordinates<double, 2> ( const VectorSpace< double, 2 > &, int,
                                                        std::vector< Vec<2, double> >&,
                                                        std::vector< doffem::gDofId >&,
                                                        std::vector< bool >&);

template void get_lagrange_dof_coordinates<double, 3> ( const VectorSpace< double, 3 > &, int,
                                                        std::vector< Vec<3, double> >&,
                                                        std::vector< doffem::gDofId >&,
                                                        std::vector< bool >&);

} // namespace hiflow
