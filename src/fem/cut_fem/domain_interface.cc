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

#include "fem/cut_fem/domain_interface.h"

#include "common/bbox.h"
#include "common/grid.h"
#include "common/array_tools.h"
#include "common/parcom.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"
#include "mesh/mesh_tools.h"
#include "mesh/refinement.h"
#include "mesh/types.h"

namespace hiflow {
namespace doffem {

template <class DataType, int DIM>
void determine_interface_neighborhood (mesh::ConstMeshPtr mesh,
                                       const MPI_Comm& comm,
                                       const std::vector< int >& cell_domain,
                                       std::vector<int>& num_recv_cells,
                                       std::vector<int>& num_send_cells,
                                       std::vector<std::vector<int> >& recv_cells,
                                       std::vector<std::vector<int> >& send_cells,
                                       std::vector<int>& ext_cell_domain)
{
  ext_cell_domain = cell_domain;
  
  const int tdim = mesh->tdim();
  const int num_cells = mesh->num_entities(tdim);

  assert (num_cells == cell_domain.size());
  ParCom parcom (comm, 0);
  const int num_proc = parcom.size();
   
  // loop through interface cells
  for (int c=0; c<num_cells; ++c)
  {
    const int cell_type = cell_domain[c];
    
    // interface
    if (cell_type == CELL_ID_INTERFACE)
    {
      // loop through all neighboring cells
      mesh::Entity cur_cell = mesh->get_entity(tdim, c);
      mesh::IncidentEntityIterator end_it  = mesh->end_incident(cur_cell, tdim);
      for (mesh::IncidentEntityIterator cell_it = mesh->begin_incident(cur_cell, tdim); 
           cell_it != end_it; ++cell_it)
      {
        const int nc = cell_it->index();
        ext_cell_domain[nc] = 2 * cell_domain[nc];
      }
    }
  }
  
  // exchange data with neighboring subdomains
  if (num_proc == 1)
  {
    return;
  }
  
  if (num_recv_cells.size() == 0 || num_send_cells.size() == 0 || recv_cells.size() == 0 || send_cells.size() == 0)
  {
    prepare_cell_exchange_requests(mesh, 
                                     parcom,
                                     num_recv_cells,
                                     num_send_cells,
                                     recv_cells,
                                     send_cells);
  }
  
  std::vector< std::vector< int > > recv_cell_data;
  exchange_cell_data<int>(parcom, 
                          num_recv_cells,
                          send_cells,
                          ext_cell_domain,
                          recv_cell_data);
  
  for (int p=0; p != num_proc; ++p)
  {
    const size_t num_cells_p = recv_cell_data[p].size();
    assert (recv_cells[p].size() == num_cells_p);
    
    if (num_cells_p == 0)
    {
      continue;
    }
    for (size_t ci = 0; ci != num_cells_p; ++ci)
    {
      const int c = recv_cells[p][ci];
      ext_cell_domain[c] = recv_cell_data[p][ci];
    }
  }
}

template <class DataType, int DIM>
void determine_interface_neighborhood (mesh::ConstMeshPtr mesh,
                                       const MPI_Comm& comm,
                                       const std::vector< int >& cell_domain,
                                       const DataType delta,
                                       std::vector<int>& num_recv_cells,
                                       std::vector<int>& num_send_cells,
                                       std::vector<int>& ext_cell_domain)
{
  ext_cell_domain = cell_domain;
  
  const int tdim = mesh->tdim();
  const int num_cells = mesh->num_entities(tdim);

  assert (num_cells == cell_domain.size());
  ParCom parcom (comm, 0);
  const int num_proc = parcom.size();
  
  // prepare exchange data with neighboring subdomains
  if (num_proc > 1)
  {
    if (num_recv_cells.size() == 0 || num_send_cells.size() == 0)
    {
      prepare_broadcast_cell_data(mesh, parcom, num_recv_cells, num_send_cells);
    }
  }
  
  size_t data_size_per_cell = DIM;
  
  // determine cell centers
  std::vector< DataType > my_centers(num_cells * DIM, 0.);
  std::vector<DataType> coord_c(DIM, 0.);

  for (int c=0; c<num_cells; ++c)
  {
    mesh::Entity cur_cell = mesh->get_entity(tdim, c);
    cur_cell.get_midpoint(coord_c);
    
    for (int d=0; d!=DIM; ++d)
    {
      my_centers[c*DIM+d] = coord_c[d];
    }
  }
   
  // exchange cell data
  std::vector< std::vector< DataType > >recv_cell_data_center;
  std::vector< std::vector< int > > recv_cell_data_domain;
  
  if (num_proc > 1)
  {
    broadcast_cell_data<int>(parcom, 
                             1,
                             num_recv_cells,
                             num_send_cells,
                             cell_domain,
                             recv_cell_data_domain);

    broadcast_cell_data<DataType>(parcom, 
                                  DIM,
                                  num_recv_cells,
                                  num_send_cells,
                                  my_centers,
                                  recv_cell_data_center);
  }
  else
  {
    recv_cell_data_domain.resize(num_proc);
    recv_cell_data_center.resize(num_proc);
  }
  
  // determine which own cells are in a delta-neighborhood of the interface
  const DataType delta2 = delta * delta;
  for (int c = 0; c!= num_cells; ++c)
  {
    if (cell_domain[c] == CELL_ID_INTERFACE)
    {
      // skip interface cells
      continue;
    }
    
    bool next_cell = false;
    
    // loop over own interface cells
    for (int ci = 0; ci!= num_cells; ++ci)
    {
      if (cell_domain[ci] != CELL_ID_INTERFACE)
      {
        // skip non-interface cells
        continue;
      }
      DataType dist = 0.;
      for (int d = 0; d!= DIM; ++d)
      {
        DataType a = my_centers[c*DIM+d];
        DataType b = my_centers[ci*DIM+d];
        dist += (a - b) * (a - b);
      }
      
      if (dist < delta2)
      {
        ext_cell_domain[c] *= 2;
        next_cell = true;
        break;
      }
    }
    if (next_cell)
    {
      continue;
    }
    
    if (num_proc == 1)
    {
      continue;
    }
    
    // loop over remote interface cells
    for (int p=0; p != num_proc; ++p)
    {
      const size_t num_cells_p = recv_cell_data_domain[p].size();
    
      if (num_cells_p == 0)
      {
        continue;
      }
      for (size_t cr = 0; cr != num_cells_p; ++cr)
      {
        // skip non-interface cells
        if (recv_cell_data_domain[p][cr] != CELL_ID_INTERFACE)
        {
          continue;
        }
        
        DataType dist = 0.;
        for (int d = 0; d!= DIM; ++d)
        {
          DataType a = my_centers[c*DIM+d];
          DataType b = recv_cell_data_center[p][cr*DIM+d];
          dist += (a - b) * (a - b);
        }
      
        if (dist < delta2)
        {
          ext_cell_domain[c] *= 2;
          next_cell = true;
          break;
        }
      }
      if (next_cell)
      {
        break;
      }
    }
  }
}
                 
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
                                       std::vector<int>& ext_cell_domain)
{
  ext_cell_domain = cell_domain;
  
  const int tdim = mesh->tdim();
  const int num_cells = mesh->num_entities(tdim);

  assert (num_cells == cell_domain.size());
  ParCom parcom (comm, 0);
  const int num_proc = parcom.size();
  
  // prepare exchange data with neighboring subdomains
  if (num_proc > 1)
  {
    if (num_recv_cells.size() == 0 || num_send_cells.size() == 0)
    {
      prepare_broadcast_cell_data(mesh, parcom, num_recv_cells, num_send_cells);
    }
  }
  
  size_t data_size_per_cell = DIM;
  
  // determine cell centers
  std::vector<DataType> my_centers(num_cells * DIM, 0.);
  std::vector<DataType> coord_c(DIM, 0.);

  for (int c=0; c<num_cells; ++c)
  {    
    for (int d=0; d!=DIM; ++d)
    {
      my_centers[c*DIM+d] = midpoint_coords[c][d];
    }
  }
   
  // exchange cell data
  std::vector< std::vector< DataType > >recv_cell_data_center;
  std::vector< std::vector< int > > recv_cell_data_domain;
  
  if (num_proc > 1)
  {
    broadcast_cell_data<int>(parcom, 
                             1,
                             num_recv_cells,
                             num_send_cells,
                             cell_domain,
                             recv_cell_data_domain);

    broadcast_cell_data<DataType>(parcom, 
                                  DIM,
                                  num_recv_cells,
                                  num_send_cells,
                                  my_centers,
                                  recv_cell_data_center);
  }
  else
  {
    recv_cell_data_domain.resize(num_proc);
    recv_cell_data_center.resize(num_proc);
  }
  
  // determine which own cells are in a delta-neighborhood of the interface
  const DataType delta2 = delta * delta;

  BBox<DataType, DIM> iface_bbox;
  std::vector<DataType> extents(2*DIM,0.);
  std::vector<int> inter_cells;

  // loop over own interface cells
  std::vector<int> local_iface_cells;
  local_iface_cells.reserve(static_cast<size_t>(0.1 * num_cells));

  for (int ci = 0; ci!= num_cells; ++ci)
  {
    if (cell_domain[ci] != CELL_ID_INTERFACE)
    {
      // skip non-interface cells
      continue;
    }
    local_iface_cells.push_back(ci);

    // center of interface cell
    Vec<DIM, DataType> iface_center = midpoint_coords[ci];

    // find points in delta-neighborhood of iface_center
    for (int d=0; d!=DIM; ++d)
    {
      extents[2*d] = iface_center[d] - delta;
      extents[2*d+1] = iface_center[d] + delta;
    }
    iface_bbox.reset(extents);

    inter_cells.clear();
    grid.intersect(iface_bbox, inter_cells);

    for (auto c : inter_cells)
    {
      for (auto k : grid_2_midpoint_map[c])
      {
        if (distance(midpoint_coords[k], iface_center) < delta)
        {
          ext_cell_domain[k] *= 2;
          ext_cell_domain[k] = std::min(ext_cell_domain[k], 2);
          ext_cell_domain[k] = std::max(ext_cell_domain[k], -2);
        }
      }
    }
  }

  // loop over remote interface cells 
  if (num_proc > 1)
  {  
    // loop over remote interface cells
    for (int p=0; p != num_proc; ++p)
    {
      const size_t num_cells_p = recv_cell_data_domain[p].size();
    
      if (num_cells_p == 0)
      {
        continue;
      }
      for (size_t cr = 0; cr != num_cells_p; ++cr)
      {
        // skip non-interface cells
        if (recv_cell_data_domain[p][cr] != CELL_ID_INTERFACE)
        {
          continue;
        }

        // center of interface cell
        Vec<DIM, DataType> iface_center;
        for (int d=0; d!=DIM; ++d)
        {
          iface_center.set(d, recv_cell_data_center[p][cr*DIM+d]);
        }

        // TODO: make this more efficient by including information 
        // on the distance between a cell and the local subdomain boundary
        
        
        // find local cell in delta-neighborhood of remote iface_center
        Vec<DIM, DataType> anchor;
        bool found_anchor = false;

        // heuristic: search iface cells first
        for (auto c : local_iface_cells)
        {
          if (distance(midpoint_coords[c], iface_center) < delta)
          {
            anchor = midpoint_coords[c];
            found_anchor = true;
            break;
          }
        }
        if (!found_anchor)
        {
          // search remaining cells
          for (int c = 0; c!= num_cells; ++c)
          {
            if (distance(midpoint_coords[c], iface_center) < delta)
            {
              anchor = midpoint_coords[c];
              found_anchor = true;
              break;
            }
          }
        }

        if (!found_anchor)
        {
          // remote iface cell is too far away
          continue;
        }

        // check cells in 2*delta neigborhood of anchor 
        for (int d=0; d!=DIM; ++d)
        {
          extents[2*d] = anchor[d] - 2*delta;
          extents[2*d+1] = anchor[d] + 2*delta;
        }
        iface_bbox.reset(extents);

        inter_cells.clear();
        grid.intersect(iface_bbox, inter_cells);

        for (auto c : inter_cells)
        {
          for (auto k : grid_2_midpoint_map[c])
          {
            if (distance(midpoint_coords[k], iface_center) < delta)
            {
              ext_cell_domain[k] *= 2;
              ext_cell_domain[k] = std::min(ext_cell_domain[k], 2);
              ext_cell_domain[k] = std::max(ext_cell_domain[k], -2);
            }
          }
        }
      }
    }
  }

}


template void determine_interface_neighborhood<double,1> (mesh::ConstMeshPtr,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&, 
                                                          std::vector<int>&, std::vector<int>&, 
                                                          std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, 
                                                          std::vector<int>&);
template void determine_interface_neighborhood<double,2> (mesh::ConstMeshPtr,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&, 
                                                          std::vector<int>&, std::vector<int>&, 
                                                          std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, 
                                                          std::vector<int>&);
template void determine_interface_neighborhood<double,3> (mesh::ConstMeshPtr,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&, 
                                                          std::vector<int>&, std::vector<int>&, 
                                                          std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, 
                                                          std::vector<int>&);
template void determine_interface_neighborhood<float,1>  (mesh::ConstMeshPtr,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&, 
                                                          std::vector<int>&, std::vector<int>&, 
                                                          std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, 
                                                          std::vector<int>&);                                                        
template void determine_interface_neighborhood<float,2>  (mesh::ConstMeshPtr,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&, 
                                                          std::vector<int>&, std::vector<int>&, 
                                                          std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, 
                                                          std::vector<int>&);
template void determine_interface_neighborhood<float,3>  (mesh::ConstMeshPtr,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&, 
                                                          std::vector<int>&, std::vector<int>&, 
                                                          std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, 
                                                          std::vector<int>&);

template void determine_interface_neighborhood<float, 1> (mesh::ConstMeshPtr,
                                                          const Grid<float, 1>&,
                                                          const std::vector<Vec<1, float> >&,
                                                          const std::vector< int >&,
                                                          const std::vector< SortedArray< int > >&,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&,
                                                          const float,
                                                          std::vector<int>& ,
                                                          std::vector<int>& ,
                                                          std::vector<int>& );

template void determine_interface_neighborhood<float, 2> (mesh::ConstMeshPtr,
                                                          const Grid<float, 2>&,
                                                          const std::vector<Vec<2, float> >&,
                                                          const std::vector< int >&,
                                                          const std::vector< SortedArray< int > >&,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&,
                                                          const float,
                                                          std::vector<int>& ,
                                                          std::vector<int>& ,
                                                          std::vector<int>& );

template void determine_interface_neighborhood<float, 3> (mesh::ConstMeshPtr,
                                                          const Grid<float, 3>&,
                                                          const std::vector<Vec<3, float> >&,
                                                          const std::vector< int >&,
                                                          const std::vector< SortedArray< int > >&,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&,
                                                          const float,
                                                          std::vector<int>& ,
                                                          std::vector<int>& ,
                                                          std::vector<int>& );

template void determine_interface_neighborhood<double, 1> (mesh::ConstMeshPtr,
                                                          const Grid<double, 1>&,
                                                          const std::vector<Vec<1, double> >&,
                                                          const std::vector< int >&,
                                                          const std::vector< SortedArray< int > >&,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&,
                                                          const double,
                                                          std::vector<int>& ,
                                                          std::vector<int>& ,
                                                          std::vector<int>& );

template void determine_interface_neighborhood<double, 2> (mesh::ConstMeshPtr,
                                                          const Grid<double, 2>&,
                                                          const std::vector<Vec<2, double> >&,
                                                          const std::vector< int >&,
                                                          const std::vector< SortedArray< int > >&,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&,
                                                          const double,
                                                          std::vector<int>& ,
                                                          std::vector<int>& ,
                                                          std::vector<int>& );

template void determine_interface_neighborhood<double, 3> (mesh::ConstMeshPtr,
                                                          const Grid<double, 3>&,
                                                          const std::vector<Vec<3, double> >&,
                                                          const std::vector< int >&,
                                                          const std::vector< SortedArray< int > >&,
                                                          const MPI_Comm&,
                                                          const std::vector< int >&,
                                                          const double,
                                                          std::vector<int>& ,
                                                          std::vector<int>& ,
                                                          std::vector<int>& );

template void determine_interface_neighborhood<double, 1> (mesh::ConstMeshPtr , const MPI_Comm&, const std::vector< int >&,
                                                           const double, std::vector<int>&, std::vector<int>&, std::vector<int>&);
template void determine_interface_neighborhood<double, 2> (mesh::ConstMeshPtr , const MPI_Comm&, const std::vector< int >&,
                                                           const double,std::vector<int>&, std::vector<int>&, std::vector<int>&);
template void determine_interface_neighborhood<double, 3> (mesh::ConstMeshPtr , const MPI_Comm&, const std::vector< int >&,
                                                           const double,std::vector<int>&, std::vector<int>&, std::vector<int>&);
template void determine_interface_neighborhood<float, 1> (mesh::ConstMeshPtr , const MPI_Comm&, const std::vector< int >&,
                                                          const float, std::vector<int>&, std::vector<int>&, std::vector<int>&);
template void determine_interface_neighborhood<float, 2> (mesh::ConstMeshPtr , const MPI_Comm&, const std::vector< int >&,
                                                          const float, std::vector<int>&, std::vector<int>&, std::vector<int>&);
template void determine_interface_neighborhood<float, 3> (mesh::ConstMeshPtr , const MPI_Comm&, const std::vector< int >&,
                                                          const float, std::vector<int>&, std::vector<int>&, std::vector<int>&);
                                                           
                                       
} // namespace mesh
} // namespace hiflow
