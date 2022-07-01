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

/// \author Philipp Gerstner

#ifndef HIFLOW_CUTFEM_INTERFACE_H_
#define HIFLOW_CUTFEM_INTERFACE_H_

#include "common/bbox.h"
#include "common/log.h"
#include "common/vector_algebra_descriptor.h"
#include "common/array_tools.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "mesh/cell_type.h"
#include "mesh/entity.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "space/vector_space.h"

#include <map>
#include <cmath>
#include <vector>
#include <numeric>
#include <set>

namespace hiflow {

namespace doffem {
                        
const int CELL_ID_INTERFACE = 0;

template <class DataType, int DIM, class DomainEvaluator>
void determine_cell_domain (const VectorSpace<DataType,DIM>& space,
                            const DomainEvaluator& domain_eval,
                            const DataType distinction_threshold,
                            const int cell_id_domain_lower,
                            const int cell_id_domain_higher,
                            std::vector< int >& cell_domain);

template <class DataType, int DIM, class DomainEvaluator>
void determine_cell_domain (const VectorSpace<DataType,DIM>& space,
                            const DomainEvaluator& domain_eval,
                            const DataType distinction_threshold,
                            std::vector< int >& cell_domain)
{
  determine_cell_domain (space, domain_eval, distinction_threshold, 1, -1, cell_domain);
}
                          
template <class DataType, int DIM>
void determine_interface_neighborhood (mesh::ConstMeshPtr mesh,
                                       const MPI_Comm& comm,
                                       const std::vector< int >& cell_domain,
                                       std::vector<int>& num_recv_cells,
                                       std::vector<int>& num_send_cells,
                                       std::vector<std::vector<int> >& recv_cells,
                                       std::vector<std::vector<int> >& send_cells,
                                       std::vector<int>& ext_cell_domain);
                                       
template <class DataType, int DIM>
void determine_interface_neighborhood (mesh::ConstMeshPtr mesh,
                                       const MPI_Comm& comm,
                                       const std::vector< int >& cell_domain,
                                       const DataType delta,
                                       std::vector<int>& num_recv_cells,
                                       std::vector<int>& num_send_cells,
                                       std::vector<int>& ext_cell_domain);

template <class DataType, int DIM>
void determine_interface_neighborhood (mesh::ConstMeshPtr mesh,
                                       const Grid<DataType, DIM>& grid, 
                                       const std::vector<Vec<DIM, DataType> >& midpoint_coords,
                                       const std::vector< int >& midpoint_2_grid_map,
                                       const std::vector< SortedArray< int > >& grid_2_midpoint_map,
                                       const MPI_Comm& comm,
                                       const std::vector< int >& cell_domain,
                                       const DataType delta,
                                       std::vector<int>& num_recv_cells,
                                       std::vector<int>& num_send_cells,
                                       std::vector<int>& ext_cell_domain);
                             
///// Implementation of templated functions //////////
//////////////////////////////////////////////////////

template <class DataType, int DIM, class DomainEvaluator>
void determine_cell_domain (const VectorSpace<DataType,DIM>& space,
                            const DomainEvaluator& domain_eval,
                            const DataType distinction_threshold,
                            const int cell_id_domain_lower,
                            const int cell_id_domain_higher,
                            std::vector< int >& cell_domain)
{
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  assert (cell_id_domain_lower != CELL_ID_INTERFACE);
  assert (cell_id_domain_higher != CELL_ID_INTERFACE);
  assert (cell_id_domain_lower * cell_id_domain_higher < 0);

  mesh::ConstMeshPtr mesh = space.meshPtr();
  const int tdim = mesh->tdim();
  const int num_cells = mesh->num_entities(tdim);
  
  cell_domain.clear();
  cell_domain.resize(num_cells, -99);

  std::vector< Coord > ref_pts;
  
  for (int c=0; c<num_cells; ++c)
  {
    int num_dom1_vertices = 0;
    int num_dom2_vertices = 0;
    
    doffem::CCellTrafoSPtr<DataType, DIM> c_trafo = space.get_cell_transformation(c);
    mesh::Entity cur_cell = mesh->get_entity(tdim, c);

    const int num_pts = c_trafo->num_vertices();

    for (int v = 0; v!=num_pts; ++v)
    {
      Coord xp = c_trafo->get_coordinate(v);
    
      DataType chi = domain_eval.evaluate(cur_cell, xp);
    
      if (chi < distinction_threshold)
      {
        num_dom1_vertices++; 
      }
      else
      {
        num_dom2_vertices++;
      }
    }
    
    if (num_dom1_vertices == num_pts)
    {
      cell_domain[c] = cell_id_domain_lower;
    }
    else if (num_dom1_vertices == 0)
    {
      cell_domain[c] = cell_id_domain_higher;
    }
    else
    {
      cell_domain[c] = CELL_ID_INTERFACE;
    }
  }
}


} // namespace mesh
} // namespace hiflow

#endif
