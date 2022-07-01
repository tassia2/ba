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

/// \author Jonathan Schwegler, Jonas Kratzke, Jonas Wildberger, Jona Ackerschott, Philipp Gerstner

#ifndef HIFLOW_GEOMETRIC_TOOLS_H_
#define HIFLOW_GEOMETRIC_TOOLS_H_


#include "common/log.h"
#include "common/vector_algebra_descriptor.h"
#include "common/sorted_array.h"
#include "mesh/types.h"

#include <mpi.h>
#include <map>
#include <cmath>
#include <vector>
#include <numeric>
#include <set>
#include <list>

namespace {
// Tolerance for geometrical comparisons.
const double GEOM_TOL = 1.e-14;
} // namespace

namespace hiflow {

template <class DataType, int DIM> class Grid;
template <class DataType, int DIM> class BBox;
template <class DataType, int DIM> class VectorSpace;

namespace doffem {
template <class DataType, int DIM> class CellTransformation;
}

namespace mesh {
template <class DataType, int DIM> class BoundaryDomainDescriptor;
class MasterSlave;
class Entity;
class CellType;

/// \brief normal of a hyperplane given by directional vectors
template <class DataType, int DIM>
Vec<DIM, DataType> normal(const std::vector< Vec<DIM, DataType> > &directions);

template <class DataType, int DIM>
Vec<DIM, DataType> normal_of_facet(const std::vector< Vec<DIM, DataType> >& facet_pts);

template <class DataType, int DIM>
Vec<DIM, DataType> normal_of_facet(const Entity& facet);

template <class DataType, int DIM>
void compute_facet_projection_matrix (const std::vector< Vec<DIM, DataType> >& facet_pts,
                                      Mat<DIM, DIM-1, DataType>& mat);
                                      
/// \brief Distance between point and hyperplane with orientation
/// It should hold: ||normal|| = 1
template <class DataType, int DIM>
DataType distance_point_hyperplane(const Vec<DIM, DataType> &point,
                                   const Vec<DIM, DataType> &origin,
                                   const Vec<DIM, DataType> &normal);

/// \brief Distance between point and line (without orientation)
template <class DataType, int DIM>
DataType distance_point_line(const Vec<DIM, DataType> &point,
                             const Vec<DIM, DataType> &origin,
                             const Vec<DIM, DataType> &direction);

/// \brief returns the foot of a point on a hyperplane
template <class DataType, int DIM>
Vec<DIM, DataType> foot_point_hyperplane(const Vec<DIM, DataType> &point,
                                         const Vec<DIM, DataType> &origin,
                                         const Vec<DIM, DataType> &normal);

/// \brief returns the foot of a point on a line
template <class DataType, int DIM>
Vec<DIM, DataType> foot_point_line(const Vec<DIM, DataType> &point,
                                   const Vec<DIM, DataType> &origin,
                                   const Vec<DIM, DataType> &direction);

/// \brief returns the area of a triangle
template<class DataType, int DIM>
DataType triangle_area(const std::vector< DataType > &vertices);

/// \brief returns the area of a facet
/// the area can be a convex simplex with sorted vertices
/// TODO: For quadrilaterals, the vertices currently have to lie in one plane.
template<class DataType, int DIM>
DataType facet_area(const std::vector< DataType > &vertices,
                    const GDim gdim);

/// \brief checks if a point lies in a hyperplane
/// with an precision of eps
template <class DataType, int DIM>
bool in_plane(const Vec<DIM, DataType> &point,
              const Vec<DIM, DataType> &origin,
              const Vec<DIM, DataType> &normal, const DataType eps) ;

/// \brief checks if the connection of two points crosses a plane
template <class DataType, int DIM>
bool crossed_plane(const Vec<DIM, DataType> &point_a,
                   const Vec<DIM, DataType> &point_b,
                   const Vec<DIM, DataType> &origin,
                   const Vec<DIM, DataType> &normal);

/// \brief checks if a facet is crossed by a line from a to b
/// The input facet is given by the coordinates of its vertices
template<class DataType, int DIM>
bool crossed_facet(const Vec<DIM, DataType> &point_a,
                   const Vec<DIM, DataType> &point_b,
                   const std::vector< DataType > &vertices);

/// \brief Computes the intersection of a line with a facet.
/// The input facet is given by the coordinates of its vertices
// TODO: implementation for hexahedron facets
template<class DataType, int DIM>
Vec<DIM, DataType> intersect_facet(const Vec<DIM, DataType> &point_a,
                                   const Vec<DIM, DataType> &point_b,
                                   const std::vector< DataType > &vertices,
                                   bool &success);

/// \brief Calculates the distance from a point to a facet where the
/// facet is of dimension DIM - 1. It also returns the closest point.
/// The facet is given by it's vertices and the ordering of them is
/// crucial for quadrilateral facets.
/// TODO: For quadrilaterals, the vertices currently have to lie in one plane.
template <class DataType, int DIM>
DataType distance_point_facet(const Vec<DIM, DataType> &point,
                              const std::vector< DataType > &vertices,
                              Vec<DIM, DataType> &closest_point);

/// \brief Determines, whether a point lies in a cell spanned by vertices.
template<class DataType, int DIM>
bool point_inside_cell(const Vec<DIM, DataType> &point, 
                       const std::vector< DataType > &vertices,
                       Vec<DIM, DataType> &ref_point);
                       
/// \brief Determines, whether a point lies in an entity of topologic dimension
/// tdim. The entity is given by the coordinates of its vertices.
/// TODO: For quadrilaterals, the vertices currently have to lie in one plane.
template<class DataType, int DIM>
bool point_inside_entity(const Vec<DIM, DataType> &point,
                         const TDim tdim,
                         const std::vector< DataType > &vertices);

/// \brief Determines, whether a set of point lies in one and the same
/// hyperplane and if true returns the normal of that hyperplane.
template <class DataType, int DIM>
bool vertices_inside_one_hyperplane(const std::vector<DataType> &vertices,
                                    const TDim tdim, const DataType eps);

/// \brief Returns the point that is closest to p but still on the
/// domain given by the BoundaryDomainDescriptor
template <class DataType, int DIM>
Vec<DIM, DataType> project_point(const BoundaryDomainDescriptor<DataType, DIM> &bdd,
                                 const Vec<DIM, DataType> &p,
                                 const MaterialNumber mat);

template<class DataType, int DIM>
bool map_ref_coord_to_other_cell ( const Vec<DIM, DataType> & my_ref_coord,
                                   Vec<DIM, DataType> & other_ref_coord, 
                                   doffem::CellTransformation<DataType, DIM> const * my_trans,
                                   doffem::CellTransformation<DataType, DIM> const * other_trans,
                                   std::vector< mesh::MasterSlave > const &period );

template < class DataType, int DIM > 
void find_subentities_containing_point (const Vec<DIM, DataType>& pt, 
                                        const mesh::CellType *ref_cell,
                                        const std::vector< Vec<DIM, DataType> >& coord_vertices, 
                                        std::vector< std::vector<int> > &dof_on_subentity);

template < class DataType, int DIM >
bool is_point_on_subentity(const Vec<DIM, DataType>& point, const std::vector< Vec<DIM, DataType> > &points);

template < class DataType, int DIM >
void create_bbox_for_mesh (ConstMeshPtr meshptr, BBox<DataType, DIM>& bbox);

template <class DataType, int DIM>
void compute_mesh_grid_map(ConstMeshPtr meshptr, 
                           Grid<DataType, DIM>& grid, 
                           bool skip_ghosts,
                           std::vector< std::list< int > >& grid_2_mesh_map,  
                           std::vector< std::list< int > >& mesh_2_grid_map);

template < class DataType, int DIM >
std::vector< DataType > compute_mean_edge_length (ConstMeshPtr meshptr);

template < class DataType, int DIM >
void find_adjacent_cells (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh, std::map<EntityNumber,
    std::set<EntityNumber>>& cell_map);

// Only works if in_mesh and out_mesh use the same Database
template < class DataType, int DIM >
void find_adjacent_cells_related(ConstMeshPtr in_mesh, ConstMeshPtr out_mesh,
                                 std::map<EntityNumber, std::set<EntityNumber>>& cell_map);

template < class DataType, int DIM >
void find_contained_cells (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh, std::map< int, std::set<int> >& cell_map);

template < class DataType, int DIM >
bool cells_intersect (const std::vector < DataType>& in_vertex_coords,
                      const std::vector < DataType>& out_vertex_coords);

template < class DataType, int DIM >
bool cells_intersect (const std::vector < DataType>& in_vertex_coords,
                      const std::vector < DataType>& out_vertex_coords,
                      const std::vector<Vec<DIM, DataType> >& in_points,
                      const std::vector<Vec<DIM, DataType> >& out_points);

template <class DataType>
bool is_aligned_rectangular_quad (const std::vector < DataType>& in_vertex_coords) ;

template <class DataType>
bool is_aligned_rectangular_cuboid (const std::vector < DataType>& in_vertex_coords) ;

template <class DataType>
bool is_parallelogram(const std::vector<DataType>& in_vertex_coords);

template <class DataType>
bool is_parallelepiped(const std::vector<DataType>& vertex_coords);

template <class DataType >
void parametrize_object (const std::vector<Vec<3, DataType>>& in_points,
                         std::vector <Vec<3, DataType>> &dir_vectors, 
                         std::vector < Vec<3, DataType> >&sup_vectors);
                         
template <class DataType, int DIM>
DataType compute_entity_diameter (const Entity& ent);
                          
template < class DataType, int DIM >
void create_bbox_for_entity (const Entity& entity, BBox<DataType, DIM>& bbox);

template <class DataType, int DIM>
void get_boundary_cells_and_normals(ConstMeshPtr mesh, 
                                    const MPI_Comm& comm,
                                    const int mode,
                                    DataType delta, 
                                    SortedArray< int >& bdy_cells,
                                    std::vector< int >& is_bdy_cell,
                                    std::vector< int >& bdy_material_cell,
                                    std::vector< int >& ext_bdy_material_cell,
                                    std::vector< Vec<DIM, DataType> >& cell_normals);
                                    
} // namespace mesh
} // namespace hiflow

#endif
