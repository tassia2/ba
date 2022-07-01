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

#include "geometric_search.h"
#include "fem/cell_trafo/linear_hexahedron_transformation.h"
#include "fem/cell_trafo/linear_quad_transformation.h"
#include "mesh/geometric_tools.h"
#include "mesh/iterator.h"
#include "mesh/mesh.h"
#include "mesh/periodicity_tools.h"

namespace hiflow {
namespace mesh {

template <class DataType, int DIM>
GeometricSearch<DataType, DIM>::GeometricSearch(ConstMeshPtr mesh)
: mesh_(mesh) 
{
  boundary_mesh_ = mesh_->extract_boundary_mesh();
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

template <class DataType, int DIM>
GridGeometricSearch<DataType, DIM>::GridGeometricSearch(ConstMeshPtr mesh)
    : GeometricSearch<DataType, DIM> (mesh) 
{
  GDim gdim = this->mesh_->gdim();
  TDim tdim = this->mesh_->tdim();
  assert(gdim == 2 || gdim == 3);
  assert (gdim == DIM);

  // Initialise container to find cells efficiently
  // 1. Compute bounding box of the mesh
  BBox< DataType, DIM > bbox(gdim);

  create_bbox_for_mesh(mesh, bbox);
  
  for (size_t d = 0; d<DIM; ++d)
  {
    this->bbox_max_.set(d, bbox.max(d));
    this->bbox_min_.set(d, bbox.min(d));
  }

  // 2. Determine a suitable number of intervals for the underlying grid
  std::vector< DataType > mean_edge_length = compute_mean_edge_length<DataType, DIM>(mesh);
  for (int d = 0; d < gdim; ++d) {
    if (mean_edge_length[d] <= 10. * GEOM_TOL) {
      mean_edge_length[d] = bbox.max(d) - bbox.min(d);
    }
  }

  this->num_intervals_.clear();
  this->num_intervals_.resize(gdim, 0);
  
  for (int d = 0; d < gdim; ++d) 
  {
    this->num_intervals_[d] = static_cast< int >(
        std::max(static_cast<DataType>(1), (bbox.max(d) - bbox.min(d)) / mean_edge_length[d]));
  }

  LOG_DEBUG(1, "Number of grid intervals per direction: " << string_from_range(
                   this->num_intervals_.begin(), this->num_intervals_.end()));

  // 3. Construct grid
  grid_.reset(new Grid< DataType, DIM >(this->num_intervals_, bbox));
  assert(grid_->get_num_points() > 0);
  assert(grid_->get_num_cells() > 0);
  for (int d = 0; d < gdim; ++d) {
    assert(grid_->delta(d) > 0);
  }

  // 4. Initialisation of the cell and boundary facet index map
  compute_cell_index_map();
  compute_facet_index_map();
}

template <class DataType, int DIM>
GridGeometricSearch<DataType, DIM>::GridGeometricSearch(ConstMeshPtr mesh,
                                                        const BBox<DataType, DIM>& bbox,
                                                        const std::vector< int >& num_intervals,
                                                        bool need_bdy_stuff)
{
  this->mesh_ = mesh;
  if (need_bdy_stuff)
  {
    this->boundary_mesh_ = this->mesh_->extract_boundary_mesh();
  }
  else
  {
    this->boundary_mesh_ = nullptr;
  }

  GDim gdim = this->mesh_->gdim();
  TDim tdim = this->mesh_->tdim();
  assert (gdim == DIM);

  // Initialise container to find cells efficiently
  for (size_t d = 0; d<DIM; ++d)
  {
    this->bbox_max_.set(d, bbox.max(d));
    this->bbox_min_.set(d, bbox.min(d));
  }

  this->num_intervals_ = num_intervals;
  
  LOG_DEBUG(1, "Number of grid intervals per direction: " << string_from_range(
                   this->num_intervals_.begin(), this->num_intervals_.end()));

  // 3. Construct grid
  grid_.reset(new Grid< DataType, DIM >(this->num_intervals_, bbox));
  assert(grid_->get_num_points() > 0);
  assert(grid_->get_num_cells() > 0);
  for (int d = 0; d < gdim; ++d) {
    assert(grid_->delta(d) > 0);
  }

  // 4. Initialisation of the cell and boundary facet index map
  compute_cell_index_map();

  if (this->boundary_mesh_ != nullptr)
  {
    compute_facet_index_map();
  }
}

template <class DataType, int DIM>
bool GridGeometricSearch<DataType, DIM>::point_inside_bbox( const Coord &point ) const 
{
  for (size_t d=0; d<DIM; ++d)
  {
    if (point[d] < this->bbox_min_[d])
    {
      return false;
    }
    if (point[d] > this->bbox_max_[d])
    {
      return false;
    }
  }
  return true;
}


template <class DataType, int DIM>
void GridGeometricSearch<DataType, DIM>::find_cell( const Coord &point, 
                                                    std::vector< int > &cells,
                                                    std::vector< Coord > &ref_points) const 
{
  assert(DIM == this->mesh_->gdim());
  TDim tdim = this->mesh_->tdim();

  // check if point lies inside bounding box of mesh.
  // If not, no need to continue further
  if (!this->point_inside_bbox(point))
  {
    LOG_DEBUG(1, "Point " << point
                          << " not contained in mesh bounding box ");
    return;
  }
  
  // get the grid cell containing the point
  const int grid_index = grid_->cell_with_point(point);

  // check, whether the grid cell is part of the grid and if so,
  // check, whether this grid cell has any points
  if (grid_index == -1 || cell_index_map_[grid_index].empty()) 
  {
    LOG_DEBUG(1, "Point " << point
                          << " not found in overlay grid ");
    return;
  }
  
  cells.clear();
  cells.reserve(8);
  ref_points.clear();
  ref_points.reserve(8);
  
  // only look for the point in the mesh cells, that were assigned to the grid
  // cell
  //std::cout << " num grid cells " << cell_index_map_[grid_index].size() << std::endl;
  for (std::list< int >::const_iterator
           ind_it = cell_index_map_[grid_index].begin(),
           e_ind_it = cell_index_map_[grid_index].end();
       ind_it != e_ind_it; ++ind_it) 
  {
    
    Coord ref_coord;
    //std::vector<Coordinate> double_coord  = this->mesh_->get_coordinates(tdim, *ind_it);
    //std::vector<DataType> tmp_coord (double_coord.begin(), double_coord.end());
    Entity cell_it(this->mesh_, DIM, *ind_it);
    std::vector<DataType> tmp_coord;
    cell_it.get_coordinates(tmp_coord);
    
    if (this->point_is_inside_cell(point, tmp_coord, ref_coord)) 
    {
      cells.push_back(*ind_it);
      ref_points.push_back(ref_coord);
    } 
    else 
    {
      LOG_DEBUG(2, " DID NOT FIND point in cell " << *ind_it);
    }
  }
  return;
}

template <class DataType, int DIM>
void GridGeometricSearch<DataType, DIM>::find_cell(const Coord &point, const std::vector< int > &trial_cells, 
                                                   std::vector< int > &cells,
                                                   std::vector< Coord > &ref_points) const 
{
  assert(DIM == this->mesh_->gdim());
  TDim tdim = this->mesh_->tdim();

  cells.clear();
  cells.reserve(8);
  ref_points.clear();
  ref_points.reserve(8);

  //std::cout << " num trial cells " << trial_cells.size() << std::endl;
  // check wether point lies in one of the trial cells
  for (std::vector< int >::const_iterator ind_it = trial_cells.begin(),
                                          e_ind_it = trial_cells.end();
       ind_it != e_ind_it; ++ind_it) 
  {
    Coord ref_coord;
    //std::vector<Coordinate> double_coord  = this->mesh_->get_coordinates(tdim, *ind_it);
    //std::vector<DataType> tmp_coord (double_coord.begin(), double_coord.end());
    Entity cell_it(this->mesh_, DIM, *ind_it);
    std::vector<DataType> tmp_coord;
    cell_it.get_coordinates(tmp_coord);
    
    if (this->point_is_inside_cell(point, tmp_coord, ref_coord)) 
    {
      cells.push_back(*ind_it);
      ref_points.push_back(ref_coord);
    } 
    else 
    {
      LOG_DEBUG(2, " DID NOT FIND point in cell " << *ind_it);
    }
  }

  // if point is not contained in trial cells, search whole grid
  if (cells.size() == 0) 
  {
    //std::cout << " search whole grid, vec " << std::endl;
    find_cell(point, cells, ref_points);
  }
  return;
}

template <class DataType, int DIM>
void GridGeometricSearch<DataType, DIM>::find_cell(const Coord &point, const std::set< int > &trial_cells, 
                                                   std::vector< int > &cells,
                                                   std::vector< Coord > &ref_points) const 
{
  assert(DIM == this->mesh_->gdim());
  TDim tdim = this->mesh_->tdim();

  cells.clear();
  cells.reserve(8);
  ref_points.clear();
  ref_points.reserve(8);

  //std::cout << " num trial cells " << trial_cells.size() << std::endl;
  // check wether point lies in one of the trial cells
  for (std::set< int >::const_iterator ind_it = trial_cells.begin(),
                                       e_ind_it = trial_cells.end();
       ind_it != e_ind_it; ++ind_it) 
  {
    Coord ref_coord;
    //std::vector<Coordinate> double_coord  = this->mesh_->get_coordinates(tdim, *ind_it);
    //std::vector<DataType> tmp_coord (double_coord.begin(), double_coord.end());
    Entity cell_it(this->mesh_, DIM, *ind_it);
    std::vector<DataType> tmp_coord;
    cell_it.get_coordinates(tmp_coord);
    
    if (this->point_is_inside_cell(point, tmp_coord, ref_coord)) 
    {
      cells.push_back(*ind_it);
      ref_points.push_back(ref_coord);
    } 
    else 
    {
      LOG_DEBUG(2, " DID NOT FIND point in cell " << *ind_it);
    }
  }

  // if point is not contained in trial cells, search whole grid
  if (cells.size() == 0) 
  {
    //std::cout << " search whole grid, set " << std::endl;
    find_cell(point, cells, ref_points);
  }
  return;
}

template <class DataType, int DIM>
bool GridGeometricSearch<DataType, DIM>::point_inside_mesh(const Coord &point) const 
{
  assert(DIM == this->mesh_->gdim());
  bool in_bbox = this->point_inside_bbox(point);
  if (!in_bbox)
  {
    return false;
  }
  std::vector< int > cells;
  std::vector< Coord > ref_points;
  this->find_cell(point, cells, ref_points);
  return (!cells.empty());
}

template <class DataType, int DIM>
typename GridGeometricSearch<DataType, DIM>::Coord 
GridGeometricSearch<DataType, DIM>::find_closest_point(const Coord &point,
                                                                          int &facet_index,
                                                                          int material_id,
                                                                          bool &success) const 
{
  // TODO: This function only works for 3d with tetrahedrons and in 2d.

  // Finding the closest point by searching through the facet that are
  // near the given point. Those facets are found by searching through
  // the closest grid cells and than using the facet_index_map to get
  // the facet cells that are "close" to the grid cell.
  GDim gdim = this->mesh_->gdim();
  TDim tdim = this->mesh_->tdim();

  assert(DIM == gdim);
  assert (this->boundary_mesh_ != nullptr);

  // Allocate return variables.
  Coord closest_point;
  facet_index = -1;

  // Compute closest point on the grid by a projection on the grid.
  BBox< DataType, DIM > grid_box = grid_->get_bbox();
  Coord nearest_grid_point(point);
  for (GDim i = 0; i < DIM; i++) {
    if (point[i] < grid_box.min(i)) {
      nearest_grid_point.set(i, grid_box.min(i));
    } else if (point[i] > grid_box.max(i)) {
      nearest_grid_point.set(i, grid_box.max(i));
    } else {
      nearest_grid_point.set(i, point[i]);
    }
  }

  // Compute mean grid spacing.
  DataType mean_delta = 0;
  for (GDim d = 0; d < gdim; d++) {
    mean_delta += grid_->delta(d);
  }
  mean_delta /= (DataType)gdim;

  // Compute the initial search radius.
  DataType initial_radius = norm (nearest_grid_point - point) + mean_delta;

  // The maximal search radius is given by the distance
  // of the point to the outer vertices of the grid.
  DataType max_search_radius = 0;
  std::vector< Coord > grid_box_vertices = grid_box.get_vertices();
  int num_vert = grid_box_vertices.size();
  assert(num_vert == (int)std::pow((double)2, (double)gdim));
  for (GDim n = 0; n < num_vert; n++) {
    Coord box_vertex = grid_box_vertices[n];
    max_search_radius = 
        std::max(max_search_radius, norm(point - box_vertex));
  }
  max_search_radius += mean_delta;

  // Initialize the distance variable with infinity.
  DataType dist = std::numeric_limits< DataType >::max();

  // Scanned_facets saves all mesh facets that were already scanned
  // and prevents from searching a facet twice.
  SortedArray< int > scanned_cells(0);

  // Search for the closest boundary facet in successively
  // increasing the search radius if not any facet
  // could be found so far.
  for (DataType search_radius = initial_radius;
       search_radius < max_search_radius; search_radius += mean_delta) {
    // The search area is represented by a sphere.
    BSphere< DataType, DIM > search_sphere(point, search_radius);
    // Get the indices of the grid cells that intersect the sphere.
    std::vector< int > grid_cell_indices;
    grid_->intersect(search_sphere, grid_cell_indices);
    // Iterate the grid cell selection.
    for (std::vector< int >::iterator grid_it = grid_cell_indices.begin();
         grid_it != grid_cell_indices.end(); ++grid_it) {
      // Iterate the boundary facets assigned to the grid cell.
      for (std::list< int >::const_iterator
               facet_it = facet_index_map_[*grid_it].begin(),
               e_facet_it = facet_index_map_[*grid_it].end();
           facet_it != e_facet_it; ++facet_it) {
        if (!scanned_cells.find_insert(*facet_it)) {
          if ((material_id != -1) &&
              (this->boundary_mesh_->get_entity(tdim - 1, *facet_it)
                   .get_material_number() != material_id))
            continue;
          // Calculate the distance between the point and the current facet.
          // At this, retrieve the closest point on the facet.
          Coord temp_point;
          std::vector< DataType > facet_vertices;

          this->boundary_mesh_->get_entity(tdim - 1, *facet_it)
              .get_coordinates(facet_vertices);
          DataType temp_dist =
              distance_point_facet<DataType, DIM>(point, facet_vertices, temp_point);
          // we have a closer point if the distance is smaller
          if (temp_dist < dist) {
            dist = temp_dist;
            closest_point = temp_point;
            facet_index = *facet_it;
          }
        }
      }
    }
    // If the determined point does not lie within the search sphere,
    // there could be other boundary facets being closer to the point.
    if (dist < search_radius) {
      success = true;
      return closest_point;
    }
  }

  // If the grid does not contain any boundary facets, an empty vector is
  // returned.
  success = false;
  return closest_point;
}

template <class DataType, int DIM>
typename GridGeometricSearch<DataType, DIM>::Coord
GridGeometricSearch<DataType, DIM>::find_closest_point_parallel( const Coord &point, int &facet_index,
                                                                                    const MPI_Comm &comm, int material_id) const 
{
  GDim gdim = this->mesh_->gdim();
  assert(DIM == gdim);
  int rank, num_partitions;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_partitions);

  facet_index = -1;
  int facet_index_local = -1;
  // has to be double because MPI_DOUBLE is used
  bool success= false;
  Coord closest_point = find_closest_point(point, facet_index_local, material_id, success);
  
  // has to be double because MPI_DOUBLE is used
  DataType local_distance = std::numeric_limits< DataType >::max();
  
  if (success) {
    local_distance = norm(point - closest_point);
  } else {
    closest_point.Zeros();
  }

  std::vector< MPI_Request > req(num_partitions);
  std::vector< MPI_Status > status(num_partitions);
  // has to be double because MPI_DOUBLE is used
  std::vector< DataType > all_distances(num_partitions);
  // send all local_distances to the process with rank 0
  int rk_with_closest_point = -1;
  if (rank == 0) {
    all_distances[0] = local_distance;
    for (int part = 1; part < num_partitions; part++) {
      MPI_Irecv(&(all_distances[part]), 1, MPI_DOUBLE, part, 0, comm,
                &req[part]);
    }
  } else {
    MPI_Send(&local_distance, 1, MPI_DOUBLE, 0, 0, comm);
  }

  if (rank == 0) {
    DataType global_distance = local_distance;
    rk_with_closest_point = rank;
    for (int part = 1; part < num_partitions; part++) {
      MPI_Wait(&req[part], &status[part]);
      if (all_distances[part] < global_distance) {
        global_distance = all_distances[part];
        rk_with_closest_point = part;
      }
    }
  }
  // at this point rank 0 is the only process who knows which
  // process has the closest point
  // rank 0 Bcasts his knowledge
  MPI_Bcast(&rk_with_closest_point, 1, MPI_INT, 0, comm);
  assert(rk_with_closest_point != -1);
  if (rank == rk_with_closest_point) {
    // assert(facet_index_local != -1);
    facet_index = facet_index_local;
  } else {
    assert(facet_index == -1);
  }
  // now since every point knows who is the owner of the closest
  // point, they send/wait for the data
  assert(DIM == gdim);

  std::vector< DataType > mpi_closest_point(DIM,0.);
  for (int i=0; i!=DIM; ++i)
  {
    mpi_closest_point[i] = closest_point[i];
  }
  MPI_Bcast(&mpi_closest_point[0], gdim, MPI_DOUBLE, rk_with_closest_point, comm);
  // since the facet_id is a local index, only the owner of the point
  // has to know it. The other processes will return an index of -1
  // to indicate that they are not the owner of the point.

  Coord return_point(mpi_closest_point);
  return return_point;
}

template <class DataType, int DIM>
std::vector< typename GridGeometricSearch<DataType, DIM>::Coord > 
GridGeometricSearch<DataType, DIM>::intersect_boundary( const Coord &point_a,
                                                        const Coord &point_b) const 
{
  GDim gdim = this->mesh_->gdim();
  TDim tdim = this->mesh_->tdim();

  assert(DIM == gdim);

  std::vector< Coord > intersections;
  assert (this->boundary_mesh_ != nullptr);

  // compute bounding box of a and b
  BBox< DataType, DIM > line_bbox(gdim);
  line_bbox.add_point(point_a);
  line_bbox.add_point(point_b);
  // small enlargement of the box for the case, that the line is parallel to a
  // coordinate axis
  line_bbox.uniform_extension(GEOM_TOL);

  // get grid cells
  std::vector< int > grid_cell_indices;
  grid_->intersect(line_bbox, grid_cell_indices);

  // a container to make sure to check a facet only once
  SortedArray< int > scanned_cells;

  // iterate grid cell selection
  for (std::vector< int >::iterator grid_it = grid_cell_indices.begin();
       grid_it != grid_cell_indices.end(); ++grid_it) {
    // iterate boundary facets
    for (std::list< int >::const_iterator
             facet_it = facet_index_map_[*grid_it].begin(),
             e_facet_it = facet_index_map_[*grid_it].end();
         facet_it != e_facet_it; ++facet_it) {
      if (!scanned_cells.find_insert(*facet_it)) {
        // compute intersection
        const std::vector< DataType > vertices ( this->boundary_mesh_->get_coordinates(tdim - 1, *facet_it).begin(),
                                                 this->boundary_mesh_->get_coordinates(tdim - 1, *facet_it).end());
          
        bool success = false;
        const Coord current_intersection =
            intersect_facet<DataType, DIM>(point_a, point_b, vertices, success);
        if (success) {
          intersections.push_back( current_intersection );
        }
      }
    }
  }
  return intersections;
}

template <class DataType, int DIM>
bool GridGeometricSearch<DataType, DIM>::crossed_boundary( const Coord &point_a,
                                                           const Coord &point_b) const 
{
  GDim gdim = this->mesh_->gdim();
  TDim tdim = this->mesh_->tdim();
  assert(DIM == gdim);

  bool crossed = false;
  assert (this->boundary_mesh_ != nullptr);

  // compute bounding box of a and b
  BBox< DataType, DIM > line_bbox(gdim);
  line_bbox.add_point(point_a);
  line_bbox.add_point(point_b);
  // small enlargement of the box for the case, that the line is parallel to a
  // coordinate axis
  line_bbox.uniform_extension(GEOM_TOL);

  // get grid cells
  std::vector< int > grid_cell_indices;
  grid_->intersect(line_bbox, grid_cell_indices);

  // a container to make sure to check a facet only once
  SortedArray< int > scanned_cells;

  // iterate grid cell selection
  for (std::vector< int >::iterator grid_it = grid_cell_indices.begin();
       grid_it != grid_cell_indices.end(); ++grid_it) {
    // iterate boundary facets
    for (std::list< int >::const_iterator
             facet_it = facet_index_map_[*grid_it].begin(),
             e_facet_it = facet_index_map_[*grid_it].end();
         facet_it != e_facet_it; ++facet_it) {
      if (!scanned_cells.find_insert(*facet_it)) {
        // compute intersection
        const std::vector< DataType > vertices (this->boundary_mesh_->get_coordinates(tdim - 1, *facet_it).begin(), 
                                                this->boundary_mesh_->get_coordinates(tdim - 1, *facet_it).end());
        if (crossed_facet<DataType, DIM>(point_a, point_b, vertices)) {
          return true;
        }
      }
    }
  }

  return crossed;
}

template <class DataType, int DIM>
int GridGeometricSearch<DataType, DIM>::max_mesh_cells() const 
{
  int max_mesh_cells = 0;
  for (std::vector< std::list< int > >::const_iterator
           ind_it = cell_index_map_.begin(),
           ind_it_end = cell_index_map_.end();
       ind_it != ind_it_end; ++ind_it) {
    max_mesh_cells = std::max((int)ind_it->size(), max_mesh_cells);
  }
  return max_mesh_cells;
}

template <class DataType, int DIM>
DataType GridGeometricSearch<DataType, DIM>::mean_mesh_cells() const 
{
  DataType mean_mesh_cells = 0;
  for (std::vector< std::list< int > >::const_iterator
           ind_it = cell_index_map_.begin(),
           ind_it_end = cell_index_map_.end();
       ind_it != ind_it_end; ++ind_it) {
    mean_mesh_cells += ind_it->size();
  }
  assert(cell_index_map_.size() > 0);
  return mean_mesh_cells / (DataType)cell_index_map_.size();
}

template <class DataType, int DIM>
void GridGeometricSearch<DataType, DIM>::compute_cell_index_map() 
{
  compute_mesh_grid_map<DataType, DIM>(this->mesh_, *this->grid_, false, cell_index_map_, inverse_cell_index_map_);
}

template <class DataType, int DIM>
void GridGeometricSearch<DataType, DIM>::compute_facet_index_map() 
{
  facet_index_map_.clear();
  facet_index_map_.resize(grid_->get_num_cells());
  TDim tdim = this->mesh_->tdim();
  GDim gdim = this->mesh_->gdim();
  assert (this->boundary_mesh_ != nullptr);

  std::vector< MasterSlave > period = this->mesh_->get_period();

  // iterate boundary facets
  for (mesh::EntityIterator it = this->boundary_mesh_->begin(tdim - 1),
                            end_it = this->boundary_mesh_->end(tdim - 1);
       it != end_it; ++it) {
    /*
                    // skip ghost facets
                    if ( this->mesh_->has_attribute ( "_remote_index_", tdim ) )
                    {
                        int this->mesh_facet_index;
                        boundary_this->mesh_->get_attribute_value (
       "_this->mesh_facet_index_", tdim - 1, it->index ( ), &this->mesh_facet_index ); int
       remote_index; const Entity this->mesh_facet = this->mesh_->get_entity ( tdim - 1,
       this->mesh_facet_index ); assert ( this->mesh_facet.num_incident_entities ( tdim ) ==
       1 ); const IncidentEntityIterator cell = this->mesh_facet.begin_incident ( tdim
       ); this->mesh_->get_attribute_value ( "_remote_index_", tdim, cell->index ( ),
       &remote_index ); if ( remote_index != -1 ) continue;
                    }
     */
    // only consider physical boundaries with a material number != -1
    int material_number =
        this->boundary_mesh_->get_material_number(tdim - 1, it->index());
    if (material_number == -1)
      continue;

    // create a bounding box of the current boundary facet
    BBox< DataType, DIM > facet_bbox(gdim);
    std::vector< DataType > coords;
    it->get_coordinates(coords);

    // check if mesh has a periodic boundary
    if (period.size() == 0) {
      // no periodic boundary
      facet_bbox.Aadd_points(coords);
    } else {
      // periodic boundary present: need to unperiodify coords
      std::vector< DataType > unperiodic_coords_on_cell =
          unperiodify(coords, gdim, period);
      facet_bbox.Aadd_points(unperiodic_coords_on_cell);
    }

    // small enlargement of the box for the case, that the box lies on the grid
    // boundary
    facet_bbox.uniform_extension(GEOM_TOL);
    // get list of grid cell indices that intersect the bounding box of the
    // current boundary facet
    std::vector< int > grid_cell_indices;
    grid_->intersect(facet_bbox, grid_cell_indices);
    // assign the cell indices to the current grid cell
    for (std::vector< int >::iterator ind_it = grid_cell_indices.begin();
         ind_it != grid_cell_indices.end(); ++ind_it) {
      facet_index_map_[*ind_it].push_back(it->index());
    }
  }
}

template <class DataType, int DIM>
bool GridGeometricSearch<DataType, DIM>::point_is_inside_cell( const Coord &point, 
                                                               const std::vector< DataType > &ent_coords,
                                                               Coord &ref_coord) const 
{
  GDim gdim = this->mesh_->gdim();
  assert(DIM == gdim);

  std::vector< MasterSlave > period = this->mesh_->get_period();

  if (period.size() == 0) 
  {
    return point_inside_cell<DataType, DIM>(point, ent_coords, ref_coord);
  } 
  else 
  {
    std::vector< DataType > coords = unperiodify(ent_coords, gdim, period);
    bool found = point_inside_cell<DataType, DIM>(point, coords, ref_coord);
#ifndef NDEBUG
    if (!found) 
    {
      LOG_DEBUG(3, "Point " << point
                            << " not found in entity consiting of vertices");
      int num_vert = ent_coords.size() / gdim;
      for (int v = 0; v < num_vert; ++v) 
      {
        LOG_DEBUG(3,
                  "( " << string_from_range(coords.begin() + v * gdim,
                                            coords.begin() + (v + 1) * gdim));
      }
    }
#endif
    return found;
  }
}

//////////////////////////////////////////////////////////
///// RecGridGeometricSearch<DataType, DIM> implementation //////////////
//////////////////////////////////////////////////////////
template <class DataType, int DIM>
RecGridGeometricSearch<DataType, DIM>::RecGridGeometricSearch( ConstMeshPtr mesh)
    : GridGeometricSearch<DataType, DIM>(mesh) 
{
}

template <class DataType, int DIM>
bool RecGridGeometricSearch<DataType, DIM>::point_is_inside_cell( const Coord &point, 
                                                                  const std::vector< DataType > &ent_coords,
                                                                  Coord &ref_coord) const 
{
  GDim gdim = this->mesh_->gdim();
  assert(DIM == gdim);

  std::vector< MasterSlave > period = this->mesh_->get_period();
  std::vector< DataType > coords;
  if (period.size() == 0) 
  {
    coords = ent_coords;
  } 
  else 
  {
    coords = unperiodify(ent_coords, gdim, period);
  }

  if (gdim == 2) 
  {
    doffem::CRefCellSPtr<DataType, DIM> ref_cell 
        = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellQuadStd<DataType, DIM> );
    hiflow::doffem::LinearQuadTransformation< DataType, DIM > ht(ref_cell);
    ht.reinit(coords);
    return (ht.contains_physical_point(point, ref_coord));
  } 
  else if (gdim == 3) 
  {
    doffem::CRefCellSPtr<DataType, DIM> ref_cell 
        = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellHexStd<DataType, DIM> );
    hiflow::doffem::LinearHexahedronTransformation< DataType, DIM > ht(ref_cell);
    ht.reinit(coords);
    return (ht.contains_physical_point(point, ref_coord));
  } 
  else 
  {
    std::cout << "wrong geometrical dimension" << std::endl;
    quit_program();
  }
}

template class GridGeometricSearch<float, 3>;
template class GridGeometricSearch<float, 2>;
template class GridGeometricSearch<float, 1>;
template class GridGeometricSearch<double, 3>;
template class GridGeometricSearch<double, 2>;
template class GridGeometricSearch<double, 1>;

template class RecGridGeometricSearch<float, 3>;
template class RecGridGeometricSearch<float, 2>;
template class RecGridGeometricSearch<float, 1>;
template class RecGridGeometricSearch<double, 3>;
template class RecGridGeometricSearch<double, 2>;
template class RecGridGeometricSearch<double, 1>;

} // namespace mesh
} // namespace hiflow
