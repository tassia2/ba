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

/// \author Jonas Kratzke, Jonathan Schwegler

#ifndef HIFLOW_GEOMETRIC_SEARCH_H_
#define HIFLOW_GEOMETRIC_SEARCH_H_

#include "common/log.h"
#include "common/bbox.h"
#include "common/bsphere.h"
#include "common/grid.h"
#include "common/macros.h"
#include "common/sorted_array.h"
#include "common/vector_algebra_descriptor.h"

#include "mesh/types.h"

#include <list>
#include "mpi.h"
#include <set>

/*
 * Here, the geometric tolerance from geometric_tools.h
 * is used:
  const double GEOM_TOL = 1.e-14;
}
 */

namespace hiflow {
namespace mesh {
class Mesh;

template <class DataType, int DIM> class GeometricSearch;
template <class DataType, int DIM> 
  using GeometricSearchPtr = std::shared_ptr< GeometricSearch<DataType, DIM> >;
  
/// \brief Abstract base class for geometric searches.

template<class DataType, int DIM>
class GeometricSearch {
public:
  using Coord = Vec<DIM, DataType>;

  GeometricSearch(ConstMeshPtr mesh);
                                                        
  virtual ~GeometricSearch() {}

  /// \brief Finds the cell in which the given coordinates
  /// lie. If the point is on an interface, it returns the list
  /// of cells, if no cell was found, the returned vector is
  /// empty.
  virtual void find_cell(const Coord &point, 
                         std::vector< int > &cells,
                         std::vector< Coord > &ref_points) const = 0;

  /// \brief Finds the cell in which the given coordinates
  /// lie. If the point is on an interface, it returns the list
  /// of cells, if no cell was found, the returned vector is
  /// empty. The user can provide a set of trial cells, which are
  /// searched before the complete grid is searched.
  /// Caution: if the point is contained in the trial cells, then the complete
  /// grid is not searched anymore. Therefore, only elements of trial_cells can
  /// be returned in this case. In particular, cells that are not element of
  /// trial_cells but contain the given point, are NOT returned.
  virtual void find_cell(const Coord &point,
                         const std::vector< int > &trial_cells, 
                         std::vector< int > &cells,
                         std::vector< Coord > &ref_points) const = 0;
                         
  // TODO: maybe use a template here to avoid redundant code
  virtual void find_cell(const Coord &point,
                         const std::set< int > &trial_cells, 
                         std::vector< int > &cells,
                         std::vector< Coord > &ref_points) const = 0;
                         
  /// \brief Returns true, if the given coordinate is inside the mesh geometry.
  virtual bool
  point_inside_mesh(const Coord &point) const = 0;

  /// \brief Finds the closest point on the boundary to a given
  /// coordinate. If material_id != -1 you get the
  /// closest point on the boundary facets with the specified material number.
  /// Returns the index of the facet, the closest point lies on.
  virtual Coord
  find_closest_point(const Coord &point, int &facet_index, int material_id , bool &success ) const = 0;

  /// \brief Finds the closest point on the boundary to a given
  /// coordinate - parallel version. If material_id != -1 you get the
  /// closest point on the boundary facets with the specified material number.
  /// Returns the index of the facet, the closest point lies on.
  virtual Coord
  find_closest_point_parallel(const Coord &point,
                              int &facet_index, const MPI_Comm &comm,
                              int material_id = -1) const = 0;

  /// \brief Compute every intersection of the straight segment
  /// between the two given points with the boundary.
  virtual std::vector< Coord >
  intersect_boundary(const Coord &point_a,
                     const Coord &point_b) const = 0;

  /// \brief Returns true, if the given straight segment between
  /// the two given points crossed the boundary of the mesh.
  virtual bool
  crossed_boundary(const Coord &point_a,
                   const Coord &point_b) const = 0;

  MeshPtr get_boundary_mesh() { return boundary_mesh_; }

  virtual bool point_inside_bbox (const Coord& pt) const = 0;

protected:
  GeometricSearch()
  {
    this->boundary_mesh_ = nullptr;
    this->mesh_ = nullptr;
  }
  
  ConstMeshPtr mesh_;
  MeshPtr boundary_mesh_;
};

/// \brief  Geometric searches based on a regular grid.
///
/// \detail This class overlays the mesh with a regular grid
/// and uses the simple topology of the grid for fast
/// point searches. At initialization each grid cell is
/// assigned with the intersecting mesh cells and boundary
/// facets. When a point is to be found or to be mapped at
/// shortest distance to the boundary, only the small subset of
/// assigned mesh cells have to be searched.

template<class DataType, int DIM>
class GridGeometricSearch : public GeometricSearch<DataType, DIM> {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  GridGeometricSearch(ConstMeshPtr mesh);

  GridGeometricSearch(ConstMeshPtr mesh, 
                      const BBox<DataType, DIM>& bbox,
                      const std::vector< int >& num_intervals,
                      bool need_bdy_stuff);
                      
  virtual void find_cell(const Coord &point, 
                         std::vector< int > &cells, 
                         std::vector< Coord > &ref_points) const;

  virtual void
  find_cell(const Coord &point,
            const std::vector< int > &trial_cells, 
            std::vector< int > &cells,
            std::vector< Coord > &ref_points) const;

  virtual void
  find_cell(const Coord &point,
            const std::set< int > &trial_cells, 
            std::vector< int > &cells,
            std::vector< Coord > &ref_points) const;
            
  bool point_inside_mesh(const Coord &point) const;

  /// \brief Finds the closest point on the boundary of the mesh.
  /// Your mesh cells must have a material number != -1 to work.
  /// Be aware that the standard material number if you did not
  /// choose one is always -1. If material_id != -1 you get the
  /// closest point on the boundary facets with the specified material number.
  Coord find_closest_point(const Coord &point, int &facet_index,
                           int material_id, bool &success ) const;

  /// \brief Finds the closest point on the global mesh. Returns the
  /// closest point on all processes but the facet_index only on the owner
  /// of the point since the facet_index is local. All others will return
  /// -1 as facet_index to indicate that they don't own the point. For further
  /// details see find_closest_point. If material_id != -1 you get the
  /// closest point on the boundary facets with the specified material number.
  Coord find_closest_point_parallel(const Coord &point,
                                    int &facet_index, const MPI_Comm &comm,
                                    int material_id = -1) const;

  std::vector< Coord > intersect_boundary(const Coord &point_a,
                                          const Coord &point_b) const;

  bool crossed_boundary(const Coord &point_a,
                        const Coord &point_b) const;

  /// \brief computes the maximal number of mesh cells that belong
  /// to one grid cell
  int max_mesh_cells() const;

  /// \brief computes the average number of mesh cells that belong
  /// to one grid cell
  DataType mean_mesh_cells() const;

  virtual bool
  point_is_inside_cell(const Coord &point, 
                       const std::vector< DataType > &ent_coords,
                       Coord &ref_coord) const;

  bool point_inside_bbox (const Coord& pt) const;

  std::vector<int> get_num_intervals() const 
  {
    return this->num_intervals_;  
  }
  
  const std::vector< std::list< int > >& get_cell_map() const
  {
    return this->cell_index_map_; 
  }
  
  const std::vector< std::list< int > >& get_inverse_cell_map() const
  {
    return this->inverse_cell_index_map_; 
  }
  
protected:
  /// \brief Determines, in which grid cells the mesh cells can be found,
  /// respectively
  void compute_cell_index_map();

  /// \brief Determines, in which grid cells the boundary facets can be found,
  /// respectively
  // only physical boundaries with a material number != -1 are collected here
  void compute_facet_index_map();
  
  std::vector< int > num_intervals_;
  
  typename ScopedPtr< Grid< DataType, DIM > >::Type grid_;
  std::vector< std::list< int > > cell_index_map_;
  std::vector< std::list< int > > inverse_cell_index_map_;
  std::vector< std::list< int > > facet_index_map_;
  
  Coord bbox_max_;
  Coord bbox_min_;
};

template<class DataType, int DIM>
class RecGridGeometricSearch : public GridGeometricSearch<DataType, DIM> {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  RecGridGeometricSearch(ConstMeshPtr mesh);

  bool point_is_inside_cell(const Coord &point, 
                            const std::vector< DataType > &ent_coords,
                            Coord &ref_coord) const;

protected:
};



} // namespace mesh
} // namespace hiflow

#endif /* _GEOMETRIC_SEARCH_H_ */
