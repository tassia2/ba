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

#include "common/bbox.h"
#include "common/array_tools.h"
#include "common/sorted_array.h"
#include "common/parcom.h"
#include "common/grid.h"
#include "mesh/geometric_tools.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/cell_trafo/linear_line_transformation.h"
#include "fem/cell_trafo/linear_triangle_transformation.h"
#include "fem/cell_trafo/linear_tetrahedron_transformation.h"
#include "fem/cell_trafo/bilinear_quad_transformation.h"
#include "fem/cell_trafo/linear_pyramid_transformation.h"
#include "fem/cell_trafo/trilinear_hexahedron_transformation.h"
#include "fem/cell_trafo/linear_hexahedron_transformation.h"
#include "fem/cell_trafo/linear_quad_transformation.h"
#include "fem/cut_fem/domain_interface.h"
#include "mesh/boundary_domain_descriptor.h"
#include "mesh/periodicity_tools.h"
#include "mesh/entity.h"
#include "mesh/geometric_search.h"
#include "mesh/iterator.h"
#include "mesh/mesh_tools.h"
#include "mesh/mesh_db_view.h"
#include "mesh/periodicity_tools.h"
#include "mesh/refinement.h"
#include "mesh/types.h"
#include "mesh/cell_type.h"
#include "space/vector_space.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/foreach.hpp>

namespace hiflow {
namespace mesh {

template <class DataType, int DIM> 
Vec<DIM, DataType> intersect_facet(const Vec<DIM, DataType> &point_a,
                                   const Vec<DIM, DataType> &point_b,
                                   const std::vector< DataType > &vertices,
                                   bool &success) {
  assert(!vertices.empty());
//  assert(!(vertices.size() % point_a.size()));

  Vec<DIM, DataType> intersection;
  const GDim gdim = DIM;
  assert((gdim == 2) || (gdim == 3));

  if (static_cast< int >(vertices.size()) != gdim * gdim) {
    // TODO: implementation for hexahedron facets
    NOT_YET_IMPLEMENTED;
    success = false;
    return intersection;
  }

  // implemented for a facet beeing a line in 2D and a triangle in 3D

  /*
   * 2D line:    3D triangle:    connection from point_a to point_b:
   *   B        C                       F = point_a
   *   *        *                      *
   *   |        |\                    /
   *   |        | \ E                / g
   *   |        |  \                *
   *   *        *---*              G = point_b
   *   A        A    B
   *
   *   E: x = A + x1(B-A) [+ x2(C-A)]  with xi >= 0 and sum xi <= 1
   *   g: x = F + x3(G-F)              with 0 <= x3 <= 1
   *   equating yields GLS:
   *   x1(B-A) [+ x2(C-A)] + x3(F-G) = F-A
   */

  // compute intersection
  Mat<DIM, DIM, DataType>MAT;
  Vec<DIM, DataType> VEC;
  for (int d = 0; d < DIM; ++d) {
    for (int dir = 0; dir < DIM - 1; ++dir) {
     MAT.set(d,dir, vertices[(dir + 1) * DIM + d] - vertices[d]);
    }
   MAT.set(d,DIM - 1, point_a[d] - point_b[d]);
    VEC.set(d, point_a[d] - vertices[d]);
  }

  // If the system is not solvable, line and facet are parallel
  Mat<DIM, DIM, DataType> MATinv;
  const bool solved = (std::abs(det(MAT)) > 1.e-14);
  
  inv(MAT, MATinv);
  Vec<DIM, DataType> sol;
  MATinv.VectorMult(VEC, sol);

  //const bool solved = gauss(MAT, VEC);

  if (!solved) {
    // check parallelism
    std::vector<Vec<DIM, DataType> > directions (gdim - 1);
    for (int d = 0; d < gdim; ++d) {
      for (int dir = 0; dir < gdim - 1; ++dir) {
        directions[dir].set(d,
            vertices[(dir + 1) * gdim + d] - vertices[d]);
      }
    }
    Vec<DIM, DataType> facet_normal = normal<DataType, DIM>(directions);
    Vec<DIM, DataType> dir_a_b(point_b);
    dir_a_b.Axpy(point_a, -1);
    if (std::abs(dot(dir_a_b, facet_normal)) < GEOM_TOL) {
      // VECtors are parallel
      // TODO: check intersection in this case
      NOT_YET_IMPLEMENTED;
    }
    success = false;
    return intersection;
  }

  // the facet is intersected if
  // 0 <= x3 <= 1
  // xi >= 0 and sum xi <= 1
  if (sol[gdim - 1] < -GEOM_TOL || sol[gdim - 1] > 1 + GEOM_TOL) {
    success = false;
    return intersection;
  }
  DataType sum = 0;
  for (int d = 0; d < gdim - 1; ++d) {
    if (sol[d] < -GEOM_TOL) {
      success = false;
      return intersection;
    }
    sum += sol[d];
  }
  if (sum > 1 + GEOM_TOL) {
    success = false;
    return intersection;
  }

  // fill intersection coordinate soltor
  for (int d = 0; d < gdim; ++d) {
    intersection.set(d, point_a[d] + sol[gdim - 1] * (point_b[d] - point_a[d]));
  }

  success = true;
  return intersection;
}

template <class DataType, int DIM> 
bool crossed_facet(const Vec<DIM, DataType> &point_a,
                   const Vec<DIM, DataType> &point_b,
                   const std::vector< DataType > &vertices) {

  assert(!vertices.empty());
  assert(!(vertices.size() % point_a.size()));

  bool success;
  intersect_facet<DataType, DIM>(point_a, point_b, vertices, success);
  return success;
}

template <class DataType, int DIM>
DataType compute_entity_diameter (const Entity& ent)
{
  const IncidentEntityIterator first_vertex = ent.begin_incident(0);
  const IncidentEntityIterator last_vertex = ent.end_incident(0);
  std::vector< DataType > v1_coord (DIM, 0.);
  std::vector< DataType > v2_coord (DIM, 0.);
  
  DataType h = 0.;
  int id1 = 0;
  for (IncidentEntityIterator v1_it = first_vertex; 
       v1_it != last_vertex; ++v1_it)
  {
    v1_it->get_coordinates(v1_coord);
    assert (v1_coord.size() == DIM);
    
    Vec<DIM, DataType> v_1 (&(v1_coord[0]));
    
    int id2 = 0;
    for (IncidentEntityIterator v2_it = first_vertex; 
         v2_it != last_vertex; ++v2_it)
    {
      if (id2 <= id1)
      {
        id2++;
        continue;
      }
      v2_it->get_coordinates(v2_coord);
      assert (v2_coord.size() == DIM);
      
      Vec<DIM, DataType> v_2  (&(v2_coord[0]));
      const DataType dist = distance(v_1, v_2);
      
      if (dist > h)
      {
        h = dist;
      }
      id2++;
    }
    id1++;
  }
  return h;
}

template <class DataType, int DIM>
bool point_inside_entity(const Vec<DIM, DataType> &point, 
                         const int tdim,
                         const std::vector< DataType > &vertices) {
  // implemented for lines,
  //                 triangles,
  //                 quadrilaterals in one plane,
  //                 tetrahedrons and
  //                 hexahedrons
  // in the dimensions 2D and 3D
  assert(tdim == 1 || tdim == 2 || tdim == 3);
  assert(!vertices.empty());
  assert(!(vertices.size() % DIM));

  GDim gdim = DIM;
  assert(gdim == 2 || gdim == 3);
  bool inside = false;

  switch (tdim) {
    // lines, rectangles and triangles can be handled via distance / volume
    // computation: if vol(ABCD) == vol(PAB) + vol(PBC) + vol(PCD) + vol(PDA)
    // the point lies in the entity
  case 1: {
    assert(gdim >= 1);
    assert(static_cast< int >(vertices.size()) == gdim * 2);
    Vec<DIM, DataType> p_a(vertices, 0);
    Vec<DIM, DataType> p_b(vertices, gdim);


    DataType point_to_a = norm(point - p_a);
    DataType point_to_b = norm(point - p_b);
    DataType a_to_b = norm(p_a - p_b);
    return (point_to_a + point_to_b < a_to_b + GEOM_TOL);
    break;
  }
  case 2: {
    assert(gdim >= 2);

    DataType area_sum = 0;
    const int num_vertices = vertices.size() / gdim;
    // Attention: Here we assume that all vertices lie in one plane!
    assert((gdim < 3 || num_vertices < 4 ||
           vertices_inside_one_hyperplane<DataType, DIM>(vertices, gdim, GEOM_TOL)));
    for (int i = 0; i < num_vertices; ++i) {
      std::vector< DataType > tri_verts(DIM);
      if (point[0] == vertices[i * gdim] &&
          point[1] == vertices[i * gdim + 1]) {
        return true;
      }
      for (int d=0; d<DIM; ++d)
      {
        tri_verts[d] = point[d];
      }

      tri_verts.insert(tri_verts.end(), vertices.begin() + i * gdim,
                       vertices.begin() + (i + 1) * gdim);
      tri_verts.insert(tri_verts.end(),
                       vertices.begin() + ((i + 1) % num_vertices) * gdim,
                       vertices.begin() + ((i + 1) % num_vertices + 1) * gdim);
      area_sum += triangle_area<DataType, DIM>(tri_verts);
    }
    DataType entitity_area = 0;
    for (int i = 1; i < num_vertices - 1; ++i) {
      std::vector< DataType > tri_verts(vertices.begin(),
                                        vertices.begin() + gdim);
      tri_verts.insert(tri_verts.end(), vertices.begin() + i * gdim,
                       vertices.begin() + (i + 2) * gdim);
      entitity_area += triangle_area<DataType, DIM>(tri_verts);
    }
    return (area_sum < entitity_area + GEOM_TOL);
    break;
  }
  case 3: {
    assert(gdim == 3);
    if (static_cast< int >(vertices.size()) == (gdim * (gdim + 1))) {
      // Tetrahedron
      // Parametric equation to check, where the point is:
      // P = A + x0(B-A) [ + x1(C-A) ( + x2(D-A) ) ]
      // This algorithm could also handle lines in 1D and triangles in 2D
      Mat<DIM, DIM, DataType >MAT;
      Vec<DIM, DataType > VEC;
      for (int i = 0; i < gdim; ++i) {
        for (int j = 0; j < gdim; ++j) {
         MAT.set(i,j, vertices[(j + 1) * gdim + i] - vertices[i]);
        }
        VEC.set(i, point[i] - vertices[i]);
      }
      // solve this linear system of equations
      Mat<DIM, DIM, DataType> MATinv;
      const bool solved = (std::abs(det(MAT)) > 1.e-14);
  
      inv(MAT, MATinv);
      Vec<DIM, DataType> sol;
      MATinv.VectorMult(VEC, sol);
      VEC = sol;

      //const bool solved = gauss(MAT, VEC);
      // if the system is not solvable, the cell is degenerated
      assert(solved);

      // the point lies in the cell, if
      // xi >= 0
      // sum xi <= 1
      DataType sum = 0;
      for (int d = 0; d < gdim; ++d) {
        if (VEC[d] < -GEOM_TOL) {
          return false;
        }
        sum += VEC[d];
      }
      return (sum < 1 + GEOM_TOL);
    } else if (static_cast< int >(vertices.size()) == (gdim * 8)) {
      // TODO: implementation of a more performant solutions for hexahedrons
      // For an arbitrary hex, the four vertices of one facet may not lie in one
      // plane.
      doffem::CRefCellSPtr<DataType, DIM> ref_cell 
        = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellHexStd<DataType, DIM> );
        
      hiflow::doffem::TriLinearHexahedronTransformation< DataType, DIM > ht(ref_cell);
      ht.reinit(vertices);
      Vec<DIM, DataType> ref_pt;
      return (ht.contains_physical_point(point, ref_pt));
    } else {
      // Pyramid
      doffem::CRefCellSPtr<DataType, DIM> ref_cell 
        = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellPyrStd<DataType, DIM> );
        
      hiflow::doffem::LinearPyramidTransformation< DataType, DIM > ht(ref_cell);
      ht.reinit(vertices);
      Vec<DIM, DataType> ref_pt;
      return (ht.contains_physical_point(point, ref_pt));
    }
    break;
  }
  default:
    assert(0);
    return false;
    break;
  }
  // A return should have been called before.
  assert(0);
  return inside;
}

template <class DataType, int DIM>
bool point_inside_cell(const Vec<DIM, DataType> &point, 
                       const std::vector< DataType > &vertices,
                       Vec<DIM, DataType> &ref_point) 
{
  // implemented for lines,
  //                 triangles,
  //                 quadrilaterals,
  //                 tetrahedrons,
  //                 hexahedrons,
  //                 pyramids
  // in the dimensions 2D and 3D
  int tdim = DIM;
  GDim gdim = DIM;

  assert(!vertices.empty());
  assert(!(vertices.size() % DIM));
  const int num_vertices = static_cast< int >(vertices.size() / gdim);

  //assert(gdim == 2 || gdim == 3);
  bool inside = false;
  
  switch (tdim) 
  {
    case 1: 
    {
      // line
      assert(gdim == 1);
      assert(num_vertices == 2);

      doffem::CRefCellSPtr<DataType, DIM> ref_cell 
        = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellLineStd<DataType, DIM> );
      hiflow::doffem::LinearLineTransformation< DataType, DIM > ht(ref_cell);
      ht.reinit(vertices);
      bool found = (ht.contains_physical_point(point, ref_point));
      return found;  
      break;
    }
    case 2: 
    {
      assert(gdim == 2);
      assert(num_vertices == 3 || num_vertices == 4);

      if (num_vertices == 3)
      {
        // triangle
        doffem::CRefCellSPtr<DataType, DIM> ref_cell 
          = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellTriStd<DataType, DIM> );
        
        hiflow::doffem::LinearTriangleTransformation< DataType, DIM > ht(ref_cell);
        ht.reinit(vertices);
        bool found = (ht.contains_physical_point(point, ref_point));
        return found;  
      }
      else 
      {
        if (is_parallelogram(vertices))
        {
          // parallelogram
          doffem::CRefCellSPtr<DataType, DIM> ref_cell 
            = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellQuadStd<DataType, DIM> );
        
          hiflow::doffem::LinearQuadTransformation< DataType, DIM > ht(ref_cell);
          ht.reinit(vertices);
          bool found = (ht.contains_physical_point(point, ref_point));
          return found;  
        }
        else 
        {
          // general quadrilateral
          doffem::CRefCellSPtr<DataType, DIM> ref_cell 
            = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellQuadStd<DataType, DIM> );
        
          hiflow::doffem::BiLinearQuadTransformation< DataType, DIM > ht(ref_cell);
          ht.reinit(vertices);
          bool found = (ht.contains_physical_point(point, ref_point));
          return found;  
        }
      }
      break;
    }
    case 3: 
    {
      assert(gdim == 3);
      assert(num_vertices == 4 || num_vertices == 5 || num_vertices == 8);
        
      if (num_vertices == 4)
      {
        // Tetrahedron
        doffem::CRefCellSPtr<DataType, DIM> ref_cell 
          = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellTetStd<DataType, DIM> );
        
        hiflow::doffem::LinearTetrahedronTransformation< DataType, DIM > ht(ref_cell);
        ht.reinit(vertices);
        bool found = (ht.contains_physical_point(point, ref_point));
        return found;  
      } 
      else if (num_vertices == 8)
      {
        // Hexahedron
        if (is_parallelepiped (vertices))
        {
          doffem::CRefCellSPtr<DataType, DIM> ref_cell 
            = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellHexStd<DataType, DIM> );
        
          hiflow::doffem::LinearHexahedronTransformation< DataType, DIM > ht(ref_cell);
          ht.reinit(vertices);
          bool found = (ht.contains_physical_point(point, ref_point));
          return found;  
        }
        else
        {
          doffem::CRefCellSPtr<DataType, DIM> ref_cell 
            = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellHexStd<DataType, DIM> );
        
          hiflow::doffem::TriLinearHexahedronTransformation< DataType, DIM > ht(ref_cell);
          ht.reinit(vertices);
          bool found = (ht.contains_physical_point(point, ref_point));
          return found;  
        }
        return false;
      } 
      else 
      {
        // Pyramid
        doffem::CRefCellSPtr<DataType, DIM> ref_cell 
        = doffem::CRefCellSPtr<DataType, DIM>(new doffem::RefCellPyrStd<DataType, DIM> );
        
        hiflow::doffem::LinearPyramidTransformation< DataType, DIM > ht(ref_cell);
        ht.reinit(vertices);
        bool found = (ht.contains_physical_point(point, ref_point));
        return found;  
      }
      break;
    }
    default:
      assert(0);
      return false;
      break;
  }
  // A return should have been called before.
  assert(0);
  return inside;
}

template <class DataType, int DIM>
bool vertices_inside_one_hyperplane(const std::vector<DataType> &vertices,
    const TDim tdim, const DataType eps)
{
  assert(!vertices.empty());
  assert((vertices.size() % DIM) == 0);

  int num_points = vertices.size() / DIM;
  if (num_points <= tdim)
    return true;

  std::vector< Vec<DIM, DataType> > directions(tdim - 1);
  for (int i = 0; i < directions.size(); ++i) {
    for (GDim d = 0; d < DIM; ++d) {
      directions[i].set(d, vertices[(i + 1) * DIM + d] - vertices[d]);
    }
  }

  Vec<DIM, DataType> origin(vertices, 0);
  Vec<DIM, DataType> plane_normal = normal<DataType, DIM>(directions);
  for (int n = DIM; n < num_points; ++n) {
    Vec<DIM, DataType> test_point(vertices, n * DIM);
    if (!in_plane<DataType, DIM>(test_point, origin, plane_normal, eps)) {
      return false;
    }
  }

  // at this point, it is proved, that all points lie in one plane.
  return true;
}

template <class DataType, int DIM>
DataType triangle_area(const std::vector< DataType > &vertices) {
  assert(!vertices.empty());
  assert(!(vertices.size() % 3));
  const GDim gdim = vertices.size() / 3;
  assert(gdim > 1);

  // 0.5 * |A-B|*dist(C,AB)
  Vec<DIM, DataType> p_a(vertices, 0);
  Vec<DIM, DataType> p_b(vertices, DIM);
  Vec<DIM, DataType> p_c(vertices, 2*DIM);
  Vec<DIM, DataType> dir_a_b(p_a);
  DataType dir_a_b_norm = 0;
  for (int i = 0; i < gdim; ++i) {
    dir_a_b.add(i, -p_b[i]);
    dir_a_b_norm += dir_a_b[i] * dir_a_b[i];
  }
  dir_a_b_norm = std::sqrt(dir_a_b_norm);

  return 0.5 * dir_a_b_norm * distance_point_line<DataType, DIM>(p_c, p_a, dir_a_b);
}

template <class DataType, int DIM>
DataType facet_area(const std::vector< DataType > &vertices,
                    const GDim gdim) {
  assert(!vertices.empty());
  assert(gdim == 2 || gdim == 3);
  assert(!(vertices.size() % gdim));
  Coordinate area = 0;
  switch (gdim) {
  case (2): {
    // in 2D: Facet = Line
    assert(vertices.size() == 4);
    Vec<DIM, DataType > vert_a(vertices, 0);
    Vec<DIM, DataType > vert_b(vertices, gdim);
    area = norm(vert_a - vert_b);
  } break;
  case (3): {
    // The vertices have to lie in one plane.
    const int num_vertices = vertices.size() / gdim;
    assert( ((num_vertices < 4) || vertices_inside_one_hyperplane<DataType, DIM>(vertices, gdim, GEOM_TOL) ));
    for (int i = 1; i < num_vertices - 1; ++i) {
      std::vector< DataType > tri_verts(vertices.begin(),
                                        vertices.begin() + gdim);
      tri_verts.insert(tri_verts.end(), vertices.begin() + i * gdim,
                       vertices.begin() + (i + 2) * gdim);
      area += triangle_area<DataType, DIM>(tri_verts);
    }
  } break;
  default:
    NOT_YET_IMPLEMENTED;
    break;
  }
  return area;
}

template <class DataType, int DIM>
bool in_plane(const Vec<DIM, DataType> &point,
              const Vec<DIM, DataType> &origin,
              const Vec<DIM, DataType> &normal, const DataType eps)
{
  const DataType distance = distance_point_hyperplane<DataType, DIM>(point, origin, normal);
  return (std::abs(distance) < eps);
}

template <class DataType, int DIM>
bool crossed_plane(const Vec<DIM, DataType> &point_a,
                   const Vec<DIM, DataType> &point_b,
                   const Vec<DIM, DataType> &origin,
                   const Vec<DIM, DataType> &normal) {
  
  const DataType distance_a = distance_point_hyperplane<DataType, DIM>(point_a, origin, normal);
  const DataType distance_b = distance_point_hyperplane<DataType, DIM>(point_b, origin, normal);
  return (distance_a * distance_b < 0);
}

template <class DataType, int DIM>
DataType distance_point_hyperplane(const Vec<DIM, DataType> &point,
                                   const Vec<DIM, DataType> &origin,
                                   const Vec<DIM, DataType> &normal) {
  return dot(point, normal) - dot(origin, normal);
}

template <class DataType, int DIM>
DataType distance_point_line(const Vec<DIM, DataType> &point,
                             const Vec<DIM, DataType> &origin,
                             const Vec<DIM, DataType> &direction) {

  Vec<DIM, DataType> foot = foot_point_line<DataType, DIM>(point, origin, direction);
  return norm(point - foot);
}
template <class DataType, int DIM>
static Vec<DIM, DataType> distance_line_line(
    const Vec<DIM, DataType>& aorig, const Vec<DIM, DataType>& borig,
    const Vec<DIM, DataType>& adir, const Vec<DIM, DataType>& bdir,
    DataType& aparam, DataType& bparam)
{
  DataType det = dot(adir, bdir) * dot(adir, bdir) - dot(adir, adir) * dot(bdir, bdir);
  aparam = (dot(bdir, bdir) * dot(aorig - borig, adir)
      - dot(adir, bdir) * dot(aorig - borig, bdir)) / det;
  bparam = (dot(adir, bdir) * dot(aorig - borig, adir)
      - dot(adir, adir) * dot(aorig - borig, bdir)) / det;
  Vec<DIM, DataType> pa = aorig + aparam * adir;
  Vec<DIM, DataType> pb = borig + bparam * bdir;
  return pb - pa;
}

template <class DataType, int DIM>
Vec<DIM, DataType> foot_point_hyperplane(const Vec<DIM, DataType> &point,
                      const Vec<DIM, DataType> &origin,
                      const Vec<DIM, DataType> &normal) {
  // orthogonal projection
  // f = p - n * (p-o) dot n / (n dot n)
  Vec<DIM, DataType> tmp(point);
  tmp.Axpy(origin, -1);
  DataType factor = dot(tmp, normal) / dot(normal, normal);
  tmp = point;
  tmp.Axpy(normal, -factor);
  return tmp; 
}

template <class DataType, int DIM>
Vec<DIM, DataType> foot_point_line(const Vec<DIM, DataType> &point,
                const Vec<DIM, DataType> &origin,
                const Vec<DIM, DataType> &direction) {

  // foot of point on line
  // foot = o + d * ((p-o) dot d) / (d dot d)
  Vec<DIM, DataType> tmp(point);
  tmp.Axpy(origin, -1);
  DataType factor = dot(tmp, direction) / dot(direction, direction);

  tmp = origin;
  tmp.Axpy(direction, factor);
  return tmp; 
}

template <class DataType, int DIM>
Vec<DIM, DataType> normal(const std::vector< Vec<DIM, DataType> > &directions)
{
  assert(!directions.empty());

  Vec<DIM, DataType> result;
  if constexpr (DIM == 2) {
    assert(directions.size() == 1);

    DataType dir_norm = norm(directions[0]);
    assert(dir_norm > 0);
    result.set(0,  directions[0][1] / dir_norm);
    result.set(1, -directions[0][0] / dir_norm);
  } 
  else if constexpr (DIM == 3) {
    assert(directions.size() == 2);
    result = cross(directions[0], directions[1]);
    DataType res_norm = norm(result);
    assert(res_norm > 0);
    result *= (1. / res_norm);
  } else {
    // directions has the wrong size
    assert(0);
  }
  return result;
}

template <class DataType, int DIM>
Vec<DIM, DataType> normal_of_facet(const std::vector< Vec<DIM, DataType> >& facet_pts)
{
  Vec<DIM, DataType> result;
  if constexpr (DIM == 2) 
  {
    assert(facet_pts.size() == 2);
    Vec<DIM, DataType> tangent = facet_pts[1]-facet_pts[0];
    
    DataType dir_norm = norm(tangent);
    assert(dir_norm > 0);
    result.set(0,  tangent[1] / dir_norm);
    result.set(1, -tangent[0] / dir_norm);
  } 
  else if constexpr (DIM == 3) 
  {
    assert(facet_pts.size() >= 3);
    Vec<DIM, DataType> tangent1 = facet_pts[1]-facet_pts[0];
    Vec<DIM, DataType> tangent2 = facet_pts[2]-facet_pts[0];
    
    result = cross(tangent1, tangent2);
    DataType res_norm = norm(result);
    assert(res_norm > 0);
    result *= (1. / res_norm);
  } else {
    // directions has the wrong size
    assert(0);
  }
  return result;
}

template <class DataType, int DIM>
Vec<DIM, DataType> normal_of_facet(const Entity& facet)
{
  std::vector< Vec<DIM, DataType> > facet_pts;
  facet.get_coordinates<DIM, DataType>(facet_pts);
  return normal_of_facet<DataType, DIM>(facet_pts);
}

template <class DataType, int DIM>
void compute_facet_projection_matrix (const std::vector< Vec<DIM, DataType> >& facet_pts,
                                      Mat<DIM, DIM-1, DataType>& MAT)
{
  MAT.Zeros();
  if constexpr (DIM == 2)
  {
    assert (facet_pts.size() == 2);
    MAT.set(0,0, facet_pts[1][0] - facet_pts[0][0]);
    MAT.set(1,0, facet_pts[1][1] - facet_pts[0][1]);
  }
  else if constexpr (DIM == 3)
  {
    assert (facet_pts.size() == 3);
    MAT.set(0,0, facet_pts[1][0] - facet_pts[0][0]);
    MAT.set(1,0, facet_pts[1][1] - facet_pts[0][1]);
    MAT.set(2,0, facet_pts[1][2] - facet_pts[0][2]);
    MAT.set(0,1, facet_pts[2][0] - facet_pts[0][0]);
    MAT.set(1,1, facet_pts[2][1] - facet_pts[0][1]);
    MAT.set(2,1, facet_pts[2][2] - facet_pts[0][2]);
  }
  else
  {
    assert(false);
  }
}
                                      
template <class DataType, int DIM>
DataType distance_point_facet(const Vec<DIM, DataType> &point,
                              const std::vector< DataType > &vertices,
                              Vec<DIM, DataType> &closest_point)
{
  assert(!vertices.empty());
  assert(!(vertices.size() % point.size()));

  // Working for 3d and 2d
  GDim gdim = DIM;
  closest_point.Zeros();

  int num_verts = vertices.size() / gdim;

  switch (gdim) {
  case 1: {
    NOT_YET_IMPLEMENTED;
  } break;
  case 2: {
    // 2D: Distance to a finite line.
    // Check first if the closest point is the orthogonal projection on the
    // facet.
    assert(num_verts == 2);
    Vec<DIM, DataType> origin(vertices, 0);
    Vec<DIM, DataType> direction;
    for (int i = 0; i < gdim; i++) {
      direction.set(i, vertices[i + gdim] - vertices[i]);
    }

    Vec<DIM, DataType> foot_point = foot_point_line<DataType, DIM> (point, origin, direction);
    
    if (point_inside_entity<DataType, DIM>(foot_point, 1, vertices)) {
      closest_point = foot_point;
      return norm(closest_point - point);
    }
  } break;
  case 3: {
    // 3D: first we do a orthogonal projection onto the plane spanned
    // by the facet. The projected point is called foot_point.
    assert(num_verts == 3 || num_verts == 4);
    Vec<DIM, DataType> origin(vertices, 0);

    std::vector<Vec<DIM, DataType> >directions (2);
    for (int i = 0; i < 2; ++i) 
    {
      for (int d=0; d<DIM; ++d) 
      {
        directions[i].set(d, vertices[(i+1)*DIM+d] - vertices[d]);
      }
    }

    Vec<DIM, DataType> facet_normal = normal<DataType, DIM> (directions);
    // TODO: The vertices of the facet currently have to lie in one plane.
    assert((num_verts < 4 || vertices_inside_one_hyperplane<DataType, DIM>(vertices, gdim, GEOM_TOL)));
    Vec<DIM, DataType> foot_point =
        foot_point_hyperplane<DataType, DIM> (point, origin, facet_normal);
    // if the foot_point is inside the entity we are done and return
    // the foot_point as closest_point
    if (point_inside_entity<DataType, DIM>(foot_point, 2, vertices)) {
      closest_point = foot_point;
      return norm(closest_point - point);
    }

    // else we project our point onto the subentities (lines)
    DataType dist = std::numeric_limits< DataType >::max();
    bool found = false;
    for (int j = 0; j < num_verts; j++) {
      std::vector< DataType > line(2 * DIM);
      Vec<DIM, DataType> direction;
      for (int i = 0; i < DIM; i++) {
        line[i] = vertices[i + j * gdim];
        origin.set(i, line[i]);
        line[i + gdim] = vertices[i + ((j + 1) % num_verts) * DIM];
        direction.set(i, line[i + gdim] - line[i]);
      }

      foot_point = foot_point_line<DataType, DIM>(point, origin, direction);
      // if the projected point is inside the entity we are done!
      if (point_inside_entity<DataType, DIM>(foot_point, 1, line)) {
        DataType tmp_dist = norm (foot_point - point);
        if (tmp_dist < dist) {
          dist = tmp_dist;
          closest_point = foot_point;
          found = true;
        }
      }
    }
    if (found) {
      return dist;
    }
  } break;
  }
  // If the closest point is not an orthogonal projection of the point
  // on the facet (or its edges in 3D), one of the vertices is the
  // closest point
  DataType dist = std::numeric_limits< DataType >::max();
  for (int n = 0; n < num_verts; n++) {
    Vec<DIM, DataType> current_vertex (vertices, n * gdim);

    DataType temp_dist = norm (current_vertex - point);
    if (temp_dist < dist) {
      dist = temp_dist;
      closest_point = current_vertex;
    }
  }
  return dist;
}

template <class DataType, int DIM>
Vec<DIM, DataType> project_point(const BoundaryDomainDescriptor<DataType, DIM> &bdd,
                                 const Vec<DIM, DataType> &p,
                                 const MaterialNumber mat)
{
  LOG_DEBUG(2, "Starting vector: " << p);
  // starting vector is p
  Vec<DIM, DataType> xk = p;
  int num_it = 0;

  // step_length is used as a stopping criteria. Initially chosen 1.
  // to not fullfill the criteria
  DataType step_length = 1.;
  // if steplength < TOL the algorithm will stop and it is assumed that
  // a solution has been found.
  const DataType TOL = 1e-8;
  // maximal number of iteration
  const int max_it = 1e3;

  // leave early if initial point is already a zero of g.
  {
    const DataType init_g_xk = bdd.eval_func(xk, mat);
    const DataType ABS_TOL = 1e-8;
    if (std::abs(init_g_xk) < ABS_TOL) {
      LOG_DEBUG(2, "Left early because point already on boundary");
      return xk;
    }
  }

  // Following does solve the problem:
  //  min_{x \in R^3} 1/2|| p - x ||^2
  //  s.t.  g(x) = 0
  // Where g is given through the BoundaryDomainDescriptor.
  // The algorithm is based on an SQP approach:
  //
  // x_0 starting point
  // for k=0 to ...
  //   min_{d_k \in R^3} 1/2|| p -(x_k + d_k)||^2
  //   s.t.  g(x_k) + d_k*grad(g)(x_k) = 0
  //
  //   x_{k+1} = x_k + d_k
  // endfor
  //
  // The solution of the minimizing problem is done via the Lagrange
  // function: L(d, l) = 1/2|| p -(x_k + d_k)||^2 + l*(g(x_k)+d*grad(g)(x_k))
  // Need to solve grad(L) = 0.
  // This can be done on an algebraic level to get an "exact" solution.

  while ((step_length > TOL) && (num_it < max_it)) {
    const DataType g_xk = bdd.eval_func(xk, mat);
    const Vec<DIM, DataType> grad_xk = bdd.eval_grad(xk, mat);
    Vec<DIM, DataType> pmxk = p - xk; 

    const DataType grad_g_dot_pmxk = dot(grad_xk, pmxk);

    // lambda_k = (grad(g)(x_k)*(p - x_k) + g(x_k))/||grad(g)(x_k)||^2
    const DataType lambdak = (grad_g_dot_pmxk + g_xk) / dot(grad_xk, grad_xk);

    // d_k = p - x_k - lambda_k*grad(x_k)
    Vec<DIM, DataType> dk(pmxk);
    dk.Axpy(grad_xk, -lambdak);

    // damping?
    DataType damping_factor = 1.;
    // x_{k+1} = x_k + d_k
    xk.Axpy(dk, damping_factor);

    step_length = sqrt(norm(dk));
    ++num_it;

    // Some high level debug information.
    LOG_DEBUG(99, "lambdak: " << lambdak);
//    LOG_DEBUG(99, "dk: " << string_from_range(dk.begin(), dk.end()));
//    LOG_DEBUG(99, "xk(updated): " << string_from_range(xk.begin(), xk.end()));
    LOG_DEBUG(99, "g(xk): " << g_xk);
//    LOG_DEBUG(99, "grad_g(xk): " << string_from_range(grad_xk.begin(),
//                                                      grad_xk.end()));
    LOG_DEBUG(99, "steplength: " << step_length);
  }
  if (num_it == max_it)
  {
    LOG_DEBUG(2, "Stopped Iteration because max Iteration was reached");
  }
//  LOG_DEBUG(2, "Final vector: " << string_from_range(xk.begin(), xk.end()));
  LOG_DEBUG(2, "Final defect: " << bdd.eval_func(xk));
  LOG_DEBUG(2, "Number of Iterations needed: " << num_it);

  return xk;
}

template < class DataType, int DIM >
bool is_point_on_subentity(const Vec<DIM, DataType>& point, const std::vector< Vec<DIM, DataType> > &points) 
{
  int verbatim_mode = 0; // 0 -> no output
  // 1 -> some output

  if (verbatim_mode > 0) {
    std::cout << "Is point (";
    for (size_t k = 0; k < point.size(); ++k) {
      std::cout << point[k] << " ";
    }
    std::cout << ") on subentity?\n";
  }

  assert(!points.empty());

  DataType eps = 1.0e-6;
  DataType delta = 1.0e-6;
  if (typeid(DataType) == typeid(double)) {
    delta = 1.0e-13;
  }

  DataType dist = 1.0; // value to be calculated

  if (points.size() == 1) 
  {
    // single point
    dist = norm(point - points[0]);
  } 
  else if (points.size() == 2) 
  {
    // line

    // distance between point P and even (defined by point A and B)

    Vec<DIM, DataType> AP (point);
    AP -= points[0];

    DataType norm_AP = norm(AP);

    Vec<DIM,DataType> AB (points[1]);
    AB -= points[0];
    
    DataType norm_AB = norm(AB);

    DataType AP_times_AB = dot(AP, AB);

    DataType tmp = norm_AP * norm_AP - (AP_times_AB * AP_times_AB) / (norm_AB * norm_AB);

    tmp = std::abs(tmp);
    assert(tmp >= 0.0 || tmp <= eps);
    dist = std::sqrt(tmp);

    if (verbatim_mode > 0) {
      std::cout << "dist to line = " << dist << "\n";
    }

    if (dist < delta) {

      // further test: point between the two defining points of the line?

      // Compute alpha such that P = A + alpha * AB . Then
      // alpha == 0 => P = A
      // alpha == 1 => P = B
      // 0 < alpha < 1 => P lies between A and B
      // otherwise P lies outside A and B .
      const DataType alpha = dot(AB, AP) / dot(AB, AB);

      if (alpha < -eps || alpha > 1 + eps) {
        return false;
      }
    }

  } else {
    // face

    assert(DIM == 3);

    // 1. calculate normed normal vector of face (p0, p1, p2)

    Vec<DIM,DataType> v(points[1]); // v := points.at(1) - points.at(0)
    v -= points[0];
    
    Vec<DIM,DataType> w(points[2]); // w := points.at(2) - points.at(0)
    w -= points[0];

    Vec<DIM,DataType> normal; // cross product
    normal.set(0, (v[1] * w[2] - v[2] * w[1]));
    normal.set(1, (v[2] * w[0] - v[0] * w[2]));
    normal.set(2, (v[0] * w[1] - v[1] * w[0]));

    // normalize normal vector
    DataType norm_normal = norm(normal);
    
    normal.set(0, normal[0] / norm_normal);
    normal.set(1, normal[1] / norm_normal);
    normal.set(2, normal[2] / norm_normal);

    // 2. calculate parameter d

    DataType d = dot(points[0], normal);

    // 3. calculate value for considered point

    DataType d_point = dot(point, normal);

    // 4. calculate distance point to face

    dist = d - d_point;
    dist = std::abs(dist);

    // 5. in case of more than 3 points

    for (size_t i = 3, e_i = points.size(); i < e_i; ++i) {
      DataType d_temp = dot(points[i], normal);
      assert(std::abs(d - d_temp) < eps);
    }

    // 6. for refined cells

    // until now it has been checked that the DoF point is on the plane
    // defined by the given points, now it is further checked whether
    // the DoF point lies within the convex envelope

    /// Idea of point on face test:
    /// Calculate the normal vectors by sucessive cross products
    /// \f \vec{n}_i(p_{i}-p_{i-1})\times (p-p_{i-1})\quad i=1\cdots n_points \f
    /// and check whether these normal vectors are parallel or zero.

    /// TODO (staffan): an alternative, but similar algorithm that one might try
    /// out would be as follows:
    ///
    /// For each edge compute edge vector e_i = p_i - p_{i-1} and
    /// point vector v_i = p - p_{i-1} (like above). Then compute a
    /// vector u_i = cross(e_i, v_i) and the normal to a plane
    /// containing e_i: n_i = cross(e_i, u_i) . Finally check the sign
    /// of dot(v_i, e_i) to find out what side of the plane that p
    /// lies on. This should remain the same for all edges.
    ///
    /// A special case is when p lies on the edge in question. As in
    /// the current algorithm, this would have to be tested for
    /// separately. Perhaps both algorithms actually work the same?

    if (dist < eps) {
      // normal vector
      Vec< DIM, DataType > vec_normal;
      bool first_time = true;

      for (size_t i = 1, e_i = points.size(); i < e_i; ++i) {
        // set vectors

        Vec< DIM, DataType > vec_edge(points[i]); // p_i - p_{i-1}
        vec_edge -= points[i-1];

        Vec< DIM, DataType > vec_to_point(point); // point - p_{i-1}
        vec_to_point -= points[i-1];

        // cross product
        Vec< DIM, DataType > vec_normal_temp;
        vec_normal_temp = cross(vec_edge, vec_to_point);
        DataType norm = std::abs(hiflow::norm(vec_normal_temp));
        if (verbatim_mode > 0) {
          std::cout << "vec_edge = " << vec_edge << ", "
                    << "\nvec_to_point = " << vec_to_point
                    << ",\nvec_normal_temp = " << vec_normal_temp << "\n";
          std::cout << "norm = " << norm << "\n";
        }
        if (norm > delta) {
          vec_normal_temp =
              vec_normal_temp * (1. / hiflow::norm(vec_normal_temp));
        }
        if (norm > delta) {
          if (first_time) {
            vec_normal = vec_normal_temp;
            first_time = false;
          } else {
            DataType diff = 0;
            for (size_t d = 0; d < DIM; ++d) {
              diff += std::abs(vec_normal[d] - vec_normal_temp[d]);
            }

            if (diff > delta) {
              dist = 1.0; // not within the convex envelope
            }
          }
        }
      }
    } else {
      if (verbatim_mode > 0) {
        std::cout << "dist > eps: dist = " << dist << ", eps = " << eps << "\n";
      }
    }
  }
  return dist < eps;
}

/// find all subentities on which a given point lies
template < class DataType, int DIM > 
void find_subentities_containing_point (const Vec<DIM, DataType>& pt, 
                                        const mesh::CellType *ref_cell,
                                        const std::vector< Vec<DIM, DataType> >& coord_vertices, 
                                        std::vector< std::vector<int> > &dof_on_subentity)
{
  const int verbatim_mode = 0; 
  
  dof_on_subentity.clear();
  dof_on_subentity.resize(ref_cell->tdim()+1);

  // trival case tdim (=cell)
  dof_on_subentity[ref_cell->tdim()].push_back(0);

  // get point coordinates for this entity, including refined entities
  const int gdim = DIM;
  std::vector< double > cv; // coord_vertices_ in other numeration
  for (size_t p = 0, e_p = coord_vertices.size(); p != e_p; ++p) 
  {
    for (size_t c = 0, e_c = gdim; c != e_c; ++c)
    {
      cv.push_back( static_cast<double>(coord_vertices[p][c]) );
    }
  }

  std::vector< double > coord_vertices_refined;
  mesh::compute_refined_vertices(*ref_cell, gdim, cv, coord_vertices_refined);

  if (verbatim_mode > 0) 
  {
    for (size_t i = 0, e_i = coord_vertices_refined.size(); i != e_i; ++i) 
    {
      std::cout << "\t" << coord_vertices_refined[i] << std::endl;
    }
  }

  // insert filtered DoFs
  /// \todo also tdim = ref_cell_->tdim() should be available
  /// \todo coord_ could also be stored by coord_on_subentity_

  for (size_t tdim = 0, e_tdim = ref_cell->tdim(); tdim != e_tdim; ++tdim) 
  {
    // for output purposes
    std::string entity_type;
    if (tdim == 0) 
    {
      entity_type = "point";
    }
    if (tdim == 1) 
    {
      entity_type = "line";
    }
    if (tdim == 2) 
    {
      entity_type = "face";
    }
    if (tdim == 3) 
    {
      entity_type = "volume";
    }

    for (size_t idx_entity = 0, e_idx_entity = ref_cell->num_entities(tdim);
         idx_entity != e_idx_entity; ++idx_entity) 
    {
      if (verbatim_mode > 0) 
      {
        std::cout << "DoF points on " << entity_type << " " << idx_entity << ":" << std::endl;
      }

      // get point coordinates for this entity
      std::vector< Vec<DIM, DataType> > points;
      std::vector< int > vertex = ref_cell->local_vertices_of_entity(tdim, idx_entity);

      // points can't be handled using local_vertices_of_entity()
      if (tdim == 0) 
      {
        vertex.clear();
        vertex.push_back(idx_entity);
      }

      // fill list of points that define the entity
      for (size_t point = 0, e_point = vertex.size(); point != e_point; ++point) 
      {
        Vec<DIM, DataType> temp;
        for (int d = 0; d < gdim; ++d) 
        {
          temp.set(d, coord_vertices_refined[vertex[point] * gdim + d]);
        }
        points.push_back(temp);
      }

      if (verbatim_mode > 0) 
      {
        std::cout << "defining points: " << std::endl;
        for (size_t p = 0; p < points.size(); ++p) 
        {
          for (size_t d = 0; d < points[0].size(); ++d) 
          {
            std::cout << "\t" << points[p][d];
          }
          std::cout << "\t --- ";
        }
        std::cout << std::endl;
      }

      // filter points that are on subentity
      if (verbatim_mode > 0) 
      {
        std::cout << "Filtering points on subentity (dim = " << tdim << ", idx = " << idx_entity << ")\n";
      }

      Vec<DIM, DataType> dof_coord = pt;

      // filter DoF
      if (static_cast< bool >(mesh::is_point_on_subentity<DataType, DIM> (dof_coord, points))) 
      {
          //coord_on_subentity_[tdim][idx_entity].push_back(dof_coord);
          dof_on_subentity[tdim].push_back(idx_entity);

          if (verbatim_mode > 0) 
          {
            std::cout << "-> ";
            for (size_t d = 0; d < dof_coord.size(); ++d) 
            {
              std::cout << dof_coord[d] << "\t";
            }
            std::cout << std::endl;
          }
        } // if (is_point_on_subentity(dof_coord, points) == true)
      else 
      {
          if (verbatim_mode > 0) 
          {
            std::cout << " dof point is not on subentity.\n";
          }
      }
      if (verbatim_mode > 0) 
      {
        std::cout << "\n\n";
      }
    }   // for (int idx_entity=0; idx_entity<ref_cell_->num_entities(tdim); ++idx_entity)
  } // for (int tdim=0; tdim<ref_cell_->tdim(); ++tdim)
  if (verbatim_mode > 0) 
  {
    std::cout << "\n\n";
  }

  // print summary
  if (verbatim_mode > 0) 
  {
    std::cout << "========" << std::endl;
    for (int tdim = 0; tdim < ref_cell->tdim()+1; ++tdim) 
    {
        std::cout << dof_on_subentity[tdim].size() << " subentities:" << std::endl;
        std::cout << string_from_range(
                         dof_on_subentity[tdim].begin(),
                         dof_on_subentity[tdim].end())
                  << "\n\n";
      for (size_t l = 0; l < dof_on_subentity[tdim].size(); ++l) 
      {
        size_t idx_entity = dof_on_subentity[tdim][l];
        std::cout << "TDim " << tdim << ", " << idx_entity << " ";
        if (idx_entity >= ref_cell->num_regular_entities(tdim)) 
        {
          std::cout << "(refined entity) ";
        } 
        else 
        {
          std::cout << "(regular entity) ";
        }
      }
      std::cout << "========" << std::endl;
    }
    std::cout << std::endl;
  }
}

#if 0
template<class DataType, int DIM>
bool map_ref_coord_to_other_cell ( const Vec<DIM, DataType> & my_ref_coord,
                                   Vec<DIM, DataType> & other_ref_coord,
                                   doffem::CellTransformation<DataType, DIM> const * my_trans,
                                   doffem::CellTransformation<DataType, DIM> const * other_trans,
                                   std::vector< mesh::MasterSlave > const &period )
{
  const DataType COEFFICIENT_EPS = 1.e3 * std::numeric_limits< DataType >::epsilon();

  const int gdim = DIM;
  
  // Compute coordinates of my cell's DoFs (physical cell).
  Vec<DIM, DataType> phys_coord;
  my_trans->transform(my_ref_coord, phys_coord);

  // Map coordinates back to other reference cell 
  bool found = other_trans->inverse(phys_coord, other_ref_coord);
  if (found) 
  {
    found = other_trans->contains_reference_point(other_ref_coord);
  }
/*
  LOG_DEBUG(2, " ======================================== ");
  LOG_DEBUG(2, "My Cell = " << my_cell.index() << ", other Cell = " << other_cell.index());
  LOG_DEBUG(2, "my reference coords = " << string_from_range(my_ref_coord.begin(), my_ref_coord.end()));
  LOG_DEBUG(2, "physical coords = " << string_from_range(phys_coord.begin(), phys_coord.end()));
  LOG_DEBUG(2, "other reference coords = " << string_from_range(other_ref_coord.begin(), other_ref_coord.end()));
  LOG_DEBUG(2, "Coord trafo successful? : " << found );
*/
  // exception handling when other reference point could not be computed
  if (!found) 
  {
    LOG_DEBUG(2, " - - - - - - - - - - - - - - - - - - - - ");

    // There should be no problems in computing the above inverse if A and B do coincide
    //Sassert(my_cell.index() != other_cell.index());

    // try inverse map again, now take into account periodic boundaries
    // 1. check whether point lies on master or slave  boundary
    std::vector< std::vector< mesh::PeriodicBoundaryType > > per_type =
          mesh::get_periodicity_type<DataType>(phys_coord, DIM, period);

    assert(per_type.size() == 1);
    assert(per_type[0].size() == period.size());

    // TODO_ avoid conversion
    std::vector<DataType> phys_coord_tmp (DIM, 0.);
    for (int d=0; d<DIM; ++d)
    {
      phys_coord_tmp[d] = phys_coord[d];
    }

    // 2. loop over all periods and swap master and slave boundaries until
    // inverse is found
    for (int k = 0; k < period.size(); ++k) 
    {
      if (per_type[0][k] == mesh::NO_PERIODIC_BDY) 
      {
        // point does not lie on periodic boundary
        continue;
      }
      if (per_type[0][k] == mesh::MASTER_PERIODIC_BDY) 
      {
        // dof lies on master periodic boundary -> compute corresponding
        // coordinates on slave boundary

        std::vector<DataType> phys_coord_slave_tmp = mesh::periodify_master_to_slave<DataType>(phys_coord_tmp, gdim, period, k);
        Vec<DIM, DataType> phys_coord_slave ( phys_coord_slave_tmp );

        LOG_DEBUG(2, "Point mapped to slave boundary of period " << k << " : "
                     << string_from_range(phys_coord_slave_tmp.begin(), phys_coord_slave_tmp.end()));

        found = other_trans->inverse(phys_coord_slave, other_ref_coord);
        if (found) 
        {
          found = other_trans->contains_reference_point(other_ref_coord);
        }
      }
      if (per_type[0][k] == mesh::SLAVE_PERIODIC_BDY) 
      {
        // dof lies on slave periodic boundary -> compute corresponding
        // coordinates on master boundary

        std::vector<DataType> phys_coord_master_tmp = mesh::periodify_slave_to_master<DataType>(phys_coord_tmp, gdim, period, k);
        Vec<DIM, DataType> phys_coord_master (phys_coord_master_tmp);

        LOG_DEBUG(2, "Point mapped to master boundary of period " << k << " : "
                     << string_from_range(phys_coord_master_tmp.begin(), phys_coord_master_tmp.end()));

        found = other_trans->inverse(phys_coord_master, other_ref_coord);
        if (found) 
        {
          found = other_trans->contains_reference_point(other_ref_coord);
        }
      }
      if (found) 
      {
        LOG_DEBUG(2, "Found cell after periodifying w.r.t. period " << k);
        LOG_DEBUG(2, "other reference coords = " << other_ref_coord);
        break;
      }
    }
  }
#ifndef NDEBUG
  if (!found) 
  {
    LOG_DEBUG(2, "other cell: did not find inverse to physical point " << phys_coord);
    LOG_DEBUG(2, " my cell ");
    std::vector< Vec<DIM, DataType> > my_coord_vtx = my_trans->get_coordinates();
    for (int v=0; v<my_coord_vtx.size(); ++v)
    {
      LOG_DEBUG(2, "vertex " << v << " : " << my_coord_vtx[v][0] << ", " << my_coord_vtx[v][1] << ", " << my_coord_vtx[v][2]);
                 
    }
    LOG_DEBUG ( 2," other cell " );
    std::vector< Vec<DIM, DataType> > other_coord_vtx = other_trans->get_coordinates();
    for (int v=0; v<other_coord_vtx.size(); ++v)
    {
      LOG_DEBUG(2, "vertex " << v << " : " << other_coord_vtx[v][0] << ", " << other_coord_vtx[v][1] << ", " << other_coord_vtx[v][2]);
    }
  }
#endif
  return found;
}
#endif

template < class DataType, int DIM >
void create_bbox_for_entity (const Entity& entity, BBox<DataType, DIM>& bbox)
{
  bbox.reset(DIM);

  std::vector<double> tmp_coord;
  entity.get_coordinates(tmp_coord);
  int num_vert = entity.num_incident_entities(0);
  assert (tmp_coord.size() == DIM * num_vert);
    
  for (int vert = 0; vert != num_vert; ++vert) 
  {
    Vec<DIM, DataType> pt;
    for (size_t d=0; d<DIM; ++d)
    {
      pt.set(d, static_cast<DataType>(tmp_coord[vert*DIM+d]));
    }
    bbox.add_point(pt);
  }
}

template < class DataType, int DIM >
void create_bbox_for_mesh (ConstMeshPtr meshptr, BBox<DataType, DIM>& bbox)
{
  bbox.reset(DIM);
  int tdim = meshptr->tdim();
  
  // check if mesh has a periodic boundary
  std::vector< MasterSlave > period = meshptr->get_period();

  if (period.size() == 0) 
  {
    // no periodic boundary -> simply loop over all vertices in mesh
    for (int vert = 0; vert < meshptr->num_entities(0); ++vert) 
    {
      std::vector<double> tmp_coord = meshptr->get_coordinates(0, vert);
      assert (tmp_coord.size() == DIM);
      Vec<DIM, DataType> pt;
      for (size_t d=0; d<DIM; ++d)
      {
        pt.set(d, static_cast<DataType>(tmp_coord[d]));
      }
      bbox.add_point(pt);
    }
  } 
  else 
  {
    // periodic boundary is present -> need to take into account slave boundary
    // points which are not present as vertices in the mesh loop over all cells
    for (mesh::EntityIterator it = meshptr->begin(tdim),
                              end_it = meshptr->end(tdim);
                              it != end_it; ++it) 
    {
      // get coordinates of current cell and unperiodify them
      std::vector< DataType > periodic_coords_on_cell;
      it->get_coordinates(periodic_coords_on_cell);

      std::vector< DataType > unperiodic_coords_on_cell =
          unperiodify(periodic_coords_on_cell, DIM, period);
      
      int num_vert = unperiodic_coords_on_cell.size() / DIM;
      std::vector<Vec<DIM, DataType> > coords_on_cell (num_vert);
      for (int i=0; i<num_vert; ++i)
      { 
        for (int d=0; d<DIM; ++d)
        {
          coords_on_cell[i].set(d, unperiodic_coords_on_cell[i*DIM +d]);
        } 
      }
      bbox.add_points(coords_on_cell);
    }
  }
  bbox.uniform_extension(10 * GEOM_TOL);
}

template < class DataType, int DIM >
std::vector< DataType > compute_mean_edge_length (ConstMeshPtr meshptr)
{
  const int num_edges = meshptr->num_entities(1);
  const int sqrt_num_edges = std::sqrt(num_edges);
  std::vector< DataType > mean_edge_length(DIM, 0.);
  int count = 0;
  for (int index = 0; index < num_edges; index += sqrt_num_edges, ++count) 
  {
    std::vector< double > coords = meshptr->get_coordinates(1, index);
    assert(static_cast< int >(coords.size()) == 2 * DIM);
    for (int d = 0; d < DIM; ++d) 
    {
      mean_edge_length[d] += std::abs(static_cast<DataType>(coords[d] - coords[d + DIM]));
    }
  }
  for (int d = 0; d < DIM; ++d) 
  {
    mean_edge_length[d] /= static_cast< DataType >(count);
  }
  return mean_edge_length;
}

template <class DataType, int DIM>
void compute_mesh_grid_map(ConstMeshPtr meshptr, 
                           Grid<DataType, DIM>& grid,
                           bool skip_ghosts, 
                           std::vector< std::list< int > >& grid_2_mesh_map,  // cell_index_map
                           std::vector< std::list< int > >& mesh_2_grid_map)  // inverse_cell_index_map 
{
  const TDim tdim = meshptr->tdim();
  const GDim gdim = meshptr->gdim();
  
  grid_2_mesh_map.clear();
  mesh_2_grid_map.clear();
  grid_2_mesh_map.resize(grid.get_num_cells());
  mesh_2_grid_map.resize(meshptr->num_entities(tdim));

  std::vector< MasterSlave > period = meshptr->get_period();
  std::vector< DataType > coords;
  std::vector< int > grid_cell_indices;
  BBox< DataType, DIM > cell_bbox(gdim);

  // iterate mesh cells
  for (mesh::EntityIterator it = meshptr->begin(tdim), end_it = meshptr->end(tdim);
       it != end_it; ++it) 
  {
    // skip ghost cells
    if (skip_ghosts)
    {
      if ( meshptr->has_attribute ( "_remote_index_", tdim ) )
      {
        int remote_index;
        meshptr->get_attribute_value ( "_remote_index_", tdim, it->index ( ), &remote_index ); 
        if ( remote_index != -1 ) 
        {
          continue;
        }
      }
    }

    // create a bounding box of the current mesh cell
    cell_bbox.reset(gdim);
    coords.clear();
    it->get_coordinates(coords);

    // check if mesh has a periodic boundary
    if (period.size() == 0) 
    {
      // no periodic boundary
      cell_bbox.Aadd_points(coords);
    } 
    else 
    {
      // periodic boundary present: need to unperiodify coords
      std::vector< DataType > unperiodic_coords_on_cell = unperiodify(coords, gdim, period);
      cell_bbox.Aadd_points(unperiodic_coords_on_cell);
    }

    // get list of grid cell indices that intersect the bounding box of the
    // current mesh cell
    grid_cell_indices.clear();
    grid.intersect(cell_bbox, grid_cell_indices);

    // assign the cell indices to the current grid cell
    for (auto ind_it = grid_cell_indices.begin(), 
         e_it = grid_cell_indices.end(); 
         ind_it != e_it; ++ind_it) 
    {
      grid_2_mesh_map[*ind_it].push_back(it->index());
      mesh_2_grid_map[it->index()].push_back(*ind_it);
    }
  }
}

template < class DataType, int DIM >
void find_adjacent_cells (ConstMeshPtr in_mesh, 
                          ConstMeshPtr out_mesh, 
                          std::map<EntityNumber, std::set<EntityNumber>>& cell_map)
{
  assert (in_mesh != nullptr);
  assert (out_mesh != nullptr);
  
  cell_map.clear();
  
  const int tdim = in_mesh->tdim();
  
  // construct common grid search for both meshes
  BBox<DataType, DIM> in_bbox(DIM);
  BBox<DataType, DIM> out_bbox(DIM);
  
  create_bbox_for_mesh (in_mesh, in_bbox);
  create_bbox_for_mesh (out_mesh, out_bbox);
  
  std::vector<DataType> extends(2*DIM, 0.);
  for (size_t d=0; d<DIM; ++d)
  {
    extends[2*d] = std::min(in_bbox.min(d), out_bbox.min(d));
    extends[2*d+1] = std::max(in_bbox.max(d), out_bbox.max(d));
  }
  BBox<DataType, DIM> bbox(extends);
    
  std::vector< DataType > in_mean_edge = compute_mean_edge_length<DataType, DIM> (in_mesh);
  std::vector< DataType > out_mean_edge = compute_mean_edge_length<DataType, DIM> (out_mesh);
  std::vector< DataType > mean_edge (DIM, 0.);
  
  for (int d = 0; d < DIM; ++d) 
  {
    if (in_mean_edge[d] <= 10. * GEOM_TOL) 
    {
      in_mean_edge[d] = in_bbox.max(d) - in_bbox.min(d);
    }
    if (out_mean_edge[d] <= 10. * GEOM_TOL) 
    {
      out_mean_edge[d] = out_bbox.max(d) - out_bbox.min(d);
    }
    mean_edge[d] = std::min(in_mean_edge[d], out_mean_edge[d]);
  }

  std::vector<int> num_intervals (DIM, 0);
  for (int d = 0; d < DIM; ++d) 
  {
    DataType num = std::max(static_cast<DataType>(1.), (bbox.max(d) - bbox.min(d)) / mean_edge[d]);
    num_intervals[d] = static_cast< int >(num);
  }
  
  // create rectangular grid that covers both meshes
  GridGeometricSearch<DataType,DIM> in_search  (in_mesh, bbox, num_intervals, false);
  GridGeometricSearch<DataType,DIM> out_search (out_mesh, bbox, num_intervals, false);

  // map: grid index to cell index
  const std::vector< std::list< int > >& in_grid_2_cells = in_search.get_cell_map();
  const std::vector< std::list< int > >& out_grid_2_cells = out_search.get_cell_map();

  // map: cell index to grid index
  const std::vector< std::list< int > >& in_cell_2_grids = in_search.get_inverse_cell_map();
  const std::vector< std::list< int > >& out_cell_2_grids = out_search.get_inverse_cell_map();
  
  std::map< int, std::set<EntityNumber> > checked_cells;
  std::vector<double> in_vertex_coords;
  std::vector<double> out_vertex_coords;
  std::vector<Vec<DIM, double> > in_points, out_points;

  // loop over all cells of in_mesh
  for (EntityIterator in_cell = in_mesh->begin(tdim), e_cell = in_mesh->end(tdim); in_cell != e_cell; ++in_cell) 
  { 
    const EntityNumber in_cell_index = in_cell->index();
    std::set<int> adj_cells;
    cell_map[in_cell_index] = adj_cells;
    
    std::set<int> cur_checked_cells;
    checked_cells[in_cell_index] = cur_checked_cells;
    
    in_vertex_coords.clear();   
    in_mesh->get_coordinates(DIM, in_cell_index, in_vertex_coords);
  
    in_points.clear();
    interlaced_coord_to_points<double, double, DIM>(in_vertex_coords, in_points);
  
    // get possible neighbors belonging to out_mesh by using rectangular grid
    // loop over all grid cells that contain current in_cell
    for (auto grid_it = in_cell_2_grids[in_cell_index].begin(), 
         e_in = in_cell_2_grids[in_cell_index].end();
         grid_it != e_in; ++grid_it) 
    {
      // loop over all out_cells, that are contained in current grid_cell
      for (auto out_it = out_grid_2_cells[*grid_it].begin(), 
           e_out = out_grid_2_cells[*grid_it].end();
           out_it != e_out; ++out_it)
      {
        const int out_index = *out_it;
        if (checked_cells[in_cell_index].find(out_index) != checked_cells[in_cell_index].end())
        {
          continue; 
        }
        
        // check for intersection of in_cell and out_cell
        out_vertex_coords.clear();   
        out_mesh->get_coordinates(tdim, out_index, out_vertex_coords);
        out_points.clear();
        interlaced_coord_to_points<double, double, DIM>(out_vertex_coords, out_points);

        bool intersection = cells_intersect<double, DIM>(in_vertex_coords, out_vertex_coords, in_points, out_points);
          
        if (intersection)
        {
          cell_map[in_cell_index].insert(out_index);
        }
        checked_cells[in_cell_index].insert(out_index);
      }
    }
  }

  // reduce map in case of in_cell == out_cell 

  // check if in_mesh and out_mesh are related, 
  // i.e. if they have the same origin, 
  // i.e. if they shared the same mesh database
  boost::intrusive_ptr< const MeshDbView > in_mesh_dbview =
    boost::static_pointer_cast< const MeshDbView >(in_mesh);

  boost::intrusive_ptr< const MeshDbView > out_mesh_dbview =
    boost::static_pointer_cast< const MeshDbView >(out_mesh);

  if (in_mesh_dbview == 0 || out_mesh_dbview == 0)
  {
    return;
  }

  if (in_mesh_dbview->get_db() != out_mesh_dbview->get_db())
  {
    return;
  }
    
  for (EntityIterator in_cell = in_mesh->begin(tdim), e_cell = in_mesh->end(tdim); in_cell != e_cell; ++in_cell) 
  { 
    const EntityNumber in_cell_index = in_cell->index();
    const Id in_cell_id = in_cell->id();

    bool exact_cell_match = false;
    EntityNumber out_cell_index = -1;
    for (auto out_it = cell_map[in_cell_index].begin(), e_out = cell_map[in_cell_index].end();
         out_it != e_out; ++out_it)
    {
      out_cell_index = *out_it;
      const Id out_cell_id = out_mesh->get_id(tdim, out_cell_index);
      if (out_cell_id == in_cell_id)
      {
        exact_cell_match = true;
        break;
      }
    }

    if (exact_cell_match)
    {
      assert (out_cell_index >= 0);
      cell_map[in_cell_index].clear();
      cell_map[in_cell_index].insert(out_cell_index);
    }
  }
}

static void find_descendant_shared_neighbors(const Entity& cell, const Entity& descendant,
    std::set<EntityNumber>& neighbors)
{
  neighbors.clear();
  neighbors.insert(cell.index());

  const int tdim = cell.tdim();
  for (IncidentEntityIterator d = descendant.begin_incident(tdim);
      d != descendant.end_incident(tdim); ++d) {
    if (d->is_descendant(cell))
      continue;

    for (IncidentEntityIterator c = cell.begin_incident(tdim);
        c != cell.end_incident(tdim); ++c) {
      if (d->is_descendant(*c))
        neighbors.insert(c->index());
    }
  }
}


template < class DataType, int DIM >
void find_adjacent_cells_related(ConstMeshPtr in_mesh, 
                                 ConstMeshPtr out_mesh,
                                 std::map<EntityNumber, std::set<EntityNumber>>& cell_map)
{
  assert(in_mesh != nullptr && out_mesh != nullptr);
  
  cell_map.clear();

  const int tdim = in_mesh->tdim();
  for (EntityIterator in_cell = in_mesh->begin(tdim), in_end_cell = in_mesh->end(tdim);
      in_cell != in_end_cell; ++in_cell) 
  {
    std::set<EntityNumber> adjcells;

    for (EntityIterator out_cell = out_mesh->begin(tdim), out_end_cell = out_mesh->end(tdim);
        out_cell != out_end_cell; ++out_cell) 
    {
      // if in_cell, out_cell are related we can find out_mesh cells incident to in_cell, out_cell
      if (in_cell->id() == out_cell->id())
      {
        adjcells.insert(out_cell->index());
      }
      else if (in_cell->is_descendant(*out_cell)) 
      {
        std::set<EntityNumber> neighbors;
        find_descendant_shared_neighbors(*out_cell, *in_cell, neighbors);
        adjcells.insert(neighbors.begin(), neighbors.end());
        break;
      } 
      else if (out_cell->is_descendant(*in_cell)) 
      {
        for (IncidentEntityIterator c = out_cell->begin_incident(tdim);
            c != out_cell->end_incident(tdim); ++c) 
        {
          adjcells.insert(c->index());
        }
        adjcells.insert(out_cell->index());
      }
    }
    cell_map[in_cell->index()] = adjcells;
  }
}

template <class DataType, int DIM>
static Vec<DIM, DataType> simplex_facet_normal(const std::vector<Vec<DIM, DataType>> &points,
    size_t ifacetstart)
{
  size_t npoints = points.size();
  if (npoints == 2)
      return points[ifacetstart] - points[(ifacetstart + 1) % 2];

  if (npoints == DIM + 1) {
    assert(DIM == 2 || DIM == 3);

    std::vector<Vec<DIM, DataType>> dirs(npoints - 2);
    for (size_t i = 0; i < dirs.size(); ++i) {
      dirs[i] = points[(ifacetstart + i + 1) % npoints] - points[ifacetstart];
    }
    Vec<DIM, DataType> n = normal<DataType, DIM>(dirs);
    Vec<DIM, DataType> v = points[ifacetstart] - points[(ifacetstart - 1 + npoints) % npoints];
    if (dot(v, n) < 0.0)
      n = -1.0 * n;
    return n;
  } else if (npoints == DIM) {
    assert(DIM == 3);

    std::vector<Vec<DIM, DataType>> dirs(npoints - 1);
    for (size_t i = 0; i < dirs.size(); ++i) {
      dirs[i] = points[(ifacetstart + i + 1) % npoints] - points[ifacetstart];
    }

    Vec<DIM, DataType> n = normal<DataType, DIM>(dirs);
    return cross(dirs[0], n);
  }
  assert(0);
  Vec<DIM, DataType> n;
  return n;
}
template <class DataType, int DIM>
static Vec<DIM, DataType> nearest_simplex_line(
    std::vector<Vec<DIM, DataType>> &simplexpoints,
    const Vec<DIM, DataType> &point)
{
  assert(DIM > 0 && DIM <= 3);

  Vec<DIM, DataType> v = simplexpoints[1] - simplexpoints[0];
  if (dot(point - simplexpoints[0], v) < -GEOM_TOL) {
    simplexpoints.erase(simplexpoints.begin() + 1);
    return point - simplexpoints[0];
  } else if (dot(point - simplexpoints[1], -1.0 * v) < -GEOM_TOL) {
    simplexpoints.erase(simplexpoints.begin() + 0);
    return point - simplexpoints[1];
  } else {
    Vec<DIM, DataType> fp = foot_point_line<DataType, DIM>(point, simplexpoints[0], v);
    return fp;
  }
}
template <class DataType, int DIM>
static Vec<DIM, DataType> nearest_simplex_triangle(
    std::vector<Vec<DIM, DataType>> &simplexpoints,
    const Vec<DIM, DataType> &point)
{
  assert(DIM == 2 || DIM == 3);
  size_t npoints = simplexpoints.size();

  for (size_t i = 0; i < npoints; ++i) {
    Vec<DIM, DataType> n = simplex_facet_normal<DataType, DIM>(simplexpoints, i);
    if (dot(point - simplexpoints[i], n) > GEOM_TOL) {
      simplexpoints.erase(simplexpoints.begin() + (i - 1 + npoints) % npoints);
      return nearest_simplex_line<DataType, DIM>(simplexpoints, point);
    }
  }

  if constexpr (DIM == 2) {
    return point;
  } else if constexpr (DIM == 3) {
    Vec<DIM, DataType> n = normal<DataType, DIM>({simplexpoints[1] - simplexpoints[0],
      simplexpoints[2] - simplexpoints[0]});
    Vec<DIM, DataType> fp = foot_point_hyperplane<DataType, DIM>(point, simplexpoints[0], n);
    return fp;
  }
  assert(0);
  Vec<DIM, DataType> p;
  return p;
}
template <class DataType, int DIM>
static Vec<DIM, DataType> nearest_simplex_tetra(
    std::vector<Vec<DIM, DataType>> &simplexpoints,
    const Vec<DIM, DataType> &point)
{
  assert(DIM == 3);
  size_t npoints = simplexpoints.size();

  std::vector<size_t> facetinds;
  for (size_t i = 0; i < npoints; ++i) {
    Vec<DIM, DataType> n = simplex_facet_normal<DataType, DIM>(simplexpoints, i);
    if (dot(point - simplexpoints[i], n) > GEOM_TOL) {
      simplexpoints.erase(simplexpoints.begin() + (i - 1 + npoints) % npoints);
      return nearest_simplex_triangle<DataType, DIM>(simplexpoints, point);
    }
  }
  return point;
}
template <class DataType, int DIM>
static void nearest_simplex(std::vector<Vec<DIM, DataType>>& simplexpoints,
    const Vec<DIM, DataType>& point, Vec<DIM, DataType>& n, bool &containspoint)
{
  assert(simplexpoints.size() <= DIM + 1);

  Vec<DIM, DataType> nearestpoint;
  switch (simplexpoints.size()) {
  case 1:
    nearestpoint = simplexpoints[0];
    break;
  case 2:
    nearestpoint = nearest_simplex_line<DataType, DIM>(simplexpoints, point);
    break;
  case 3:
    nearestpoint = nearest_simplex_triangle<DataType, DIM>(simplexpoints, point);
    break;
  case 4:
    nearestpoint = nearest_simplex_tetra<DataType, DIM>(simplexpoints, point);
    break;
  default:
    assert(0);
  }
  n = point - nearestpoint;
  containspoint = norm(n) < GEOM_TOL;
}

template <class DataType, int DIM>
static Vec<DIM, DataType> support(const std::vector<Vec<DIM, DataType>>& points,
                                  const Vec<DIM, DataType> dir)
{
  int m = -1;
  DataType maxprod = -std::numeric_limits<DataType>::max();
  for (int i = 0; i < points.size(); ++i) 
  {
    DataType prod = dot(points[i], dir);
    if (prod > maxprod) 
    {
      maxprod = prod;
      m = i;
    }
  }
  assert ( m >= 0 );
  return points[m];
}

template <class DataType, int DIM>
bool cells_intersect (const std::vector<DataType>& in_vertex_coords, 
                      const std::vector<DataType>& out_vertex_coords)
{
  std::vector<Vec<DIM, DataType> > in_points;
  std::vector<Vec<DIM, DataType> > out_points;

  interlaced_coord_to_points<DataType, DataType, DIM>(in_vertex_coords, in_points);
  interlaced_coord_to_points<DataType, DataType, DIM>(out_vertex_coords, out_points);

  return cells_intersect<DataType, DIM>(in_vertex_coords, out_vertex_coords, in_points, out_points);
}
                      


template <class DataType, int DIM>
bool cells_intersect (const std::vector<DataType>& in_vertex_coords, 
                      const std::vector<DataType>& out_vertex_coords,
                      const std::vector<Vec<DIM, DataType> >& in_points,
                      const std::vector<Vec<DIM, DataType> >& out_points)
{ 
  const size_t maxgjkiterations = 1000;

  // this function only works for convex cells (maybe add an assertion)
  assert(in_vertex_coords.size() % DIM == 0 && out_vertex_coords.size() % DIM == 0);

  // no need for expensive tests when the bounding boxes do not overlap
  BBox<DataType, DIM> in_box(in_vertex_coords, in_vertex_coords.size() / DIM);
  BBox<DataType, DIM> out_box(out_vertex_coords, out_vertex_coords.size() / DIM);
  if (!in_box.intersects(out_box, GEOM_TOL))
    return false;

  // we are finished in one dimension, for aligned quads and cuboids
  if (DIM == 1 || (DIM == 2 && is_aligned_rectangular_quad(in_vertex_coords)
        && is_aligned_rectangular_quad(out_vertex_coords))
      || (DIM == 3 && is_aligned_rectangular_cuboid(in_vertex_coords)
        && is_aligned_rectangular_cuboid(out_vertex_coords)))
    return true;

  // Gilbert-Johnson-Keerthi distance algorithm
  // see https://en.wikipedia.org/wiki/Gilbert-Johnson-Keerthi_distance_algorithm
  std::vector<Vec<DIM, DataType>> simplexpoints;
  Vec<DIM, DataType> dir;
  dir.set(0, 1.0);

  Vec<DIM, DataType> p = support<DataType, DIM>(in_points, dir)
    - support<DataType, DIM>(out_points, -1.0 * dir);
  simplexpoints.push_back(p);
  dir = -1.0 * p;

  bool containsorigin;
  Vec<DIM, DataType> origin;
  int i = 0;
  while (i < maxgjkiterations) {
    nearest_simplex<DataType, DIM>(simplexpoints, origin, dir, containsorigin);
    if (containsorigin)
      return true;

    p = support<DataType, DIM>(in_points, dir) - support<DataType, DIM>(out_points, -1.0 * dir);
    if (dot(p, dir) < -GEOM_TOL)
      return false;
    simplexpoints.push_back(p);

    ++i;
  }
  assert(0);
  return false;
}

template <class DataType>
bool is_aligned_rectangular_cuboid (const std::vector < DataType>& vertex_coords) 
{
  //        7 ----------------- 6
  //       /|                /|
  //      / |               / |
  //     /  |z             /  |
  //    4 ----------------- 5 |
  //    |   |             |   |
  //    |   |       y     |   |
  //    |   3 ------------|-- 2
  //    |  /              |  /
  //    | /x              | /
  //    |/                |/
  //    0 ----------------- 1
  
  bool is_aligned_rectangular_cuboid;
  
  if (vertex_coords.size() != 24)
    return false;

  std::vector <Vec<3, DataType>> points;
  interlaced_coord_to_points<DataType, DataType, 3> (vertex_coords, points);
  
  std::vector < Vec<3, DataType > > x_edges(4);
  std::vector < Vec<3, DataType > > y_edges(4);
  std::vector < Vec<3, DataType > > z_edges(4);
  
  x_edges[0] = (points[0] - points[3]);
  x_edges[1] = (points[1] - points[2]);
  x_edges[2] = (points[4] - points[7]);
  x_edges[3] = (points[5] - points[6]);
  
  y_edges[0] = (points[1] - points[0]);
  y_edges[1] = (points[2] - points[3]);
  y_edges[2] = (points[5] - points[4]);
  y_edges[3] = (points[7] - points[6]);
  
  z_edges[0] = (points[4] - points[0]);
  z_edges[1] = (points[5] - points[1]);
  z_edges[2] = (points[7] - points[3]);
  z_edges[3] = (points[6] - points[2]);
  
  //x- and y-edges could be swapped if cuboid is rotated by 90 degrees, so check orientation first
  if (std::abs(x_edges[0][1]) < GEOM_TOL) { //if true, these edges can only be aligned to x-axis
    for (int i = 0; i < 4; i++) {   //you could check fewer edges whether they are aligned to their respective axis
      if (std::abs(x_edges[i][1]) > GEOM_TOL || std::abs(x_edges[i][2]) > GEOM_TOL ) return false;
      if (std::abs(y_edges[i][0]) > GEOM_TOL || std::abs(y_edges[i][2]) > GEOM_TOL ) return false;
      if (std::abs(z_edges[i][0]) > GEOM_TOL || std::abs(z_edges[i][1]) > GEOM_TOL ) return false;
    }
    return true;
  }
  else if (std::abs(x_edges[0][0]) < GEOM_TOL) { //if true, these edges can only be aligned to y-axis
    for (int i = 0; i < 4; i++) {   
      if (std::abs(x_edges[i][0]) > GEOM_TOL || std::abs(x_edges[i][2]) > GEOM_TOL ) return false;
      if (std::abs(y_edges[i][1]) > GEOM_TOL || std::abs(y_edges[i][2]) > GEOM_TOL ) return false;
      if (std::abs(z_edges[i][0]) > GEOM_TOL || std::abs(z_edges[i][1]) > GEOM_TOL ) return false;
    }
    return true;
  }
  else return false; //aligned to neither of the axes -> not aligned at all
}

template <class DataType>
bool is_aligned_rectangular_quad(const std::vector < DataType>& vertex_coords) 
{
  //                y    
  //       3 ----------------- 2
  //      /                  /
  //     / x                /
  //    /                  /
  //   0 -----------------1
  
  if (vertex_coords.size()!= 8)
  {
    return false;
  }
  std::vector <Vec<2, DataType>> points;
  interlaced_coord_to_points<DataType, DataType, 2> (vertex_coords, points);
  
  std::vector < Vec<2, DataType > > a_edges(2);
  std::vector < Vec<2, DataType > > b_edges(2);

  
  a_edges[0] = (points[0] - points[3]);
  a_edges[1] = (points[1] - points[2]);
  b_edges[0] = (points[1] - points[0]);
  b_edges[1] = (points[2] - points[3]);

  return (   (std::abs(a_edges[0][0]) + std::abs(a_edges[1][0]) < 2.0 * GEOM_TOL)
          && (std::abs(b_edges[0][1]) + std::abs(b_edges[1][1]) < 2.0 * GEOM_TOL) )
      || (   (std::abs(a_edges[0][1]) + std::abs(a_edges[1][1]) < 2.0 * GEOM_TOL)
          && (std::abs(b_edges[0][0]) + std::abs(b_edges[1][0]) < 2.0 * GEOM_TOL) );
}

template <class DataType>
bool is_parallelogram(const std::vector<DataType>& vertex_coords)
{
  //                y
  //       3 ----------------- 2
  //      /                  /
  //     / x                /
  //    /                  /
  //   0 -----------------1

  if (vertex_coords.size() != 8)
    return false;

  std::vector<Vec<2, DataType>> points;
  interlaced_coord_to_points<DataType, DataType, 2>(vertex_coords, points);

  Vec<2, DataType> a1 = points[1] - points[0];
  Vec<2, DataType> a2 = points[2] - points[3];

  Vec<2, DataType> b1 = points[2] - points[1];
  Vec<2, DataType> b2 = points[3] - points[0];

  if (!is_parallel(a1, a2))
  {
    return false;
  }
  return is_parallel(b1, b2);
}

template <class DataType>
bool is_parallelepiped(const std::vector<DataType>& vertex_coords)
{
  //        7 ----------------- 6
  //       /|                /|
  //      / |               / |
  //     /  |z             /  |
  //    4 ----------------- 5 |
  //    |   |             |   |
  //    |   |       y     |   |
  //    |   3 ------------|---- 2
  //    |  /              |  /
  //    | /x              | /
  //    |/                |/
  //    0 ----------------- 1

  if (vertex_coords.size() != 24)
    return false;

  std::vector<Vec<3, DataType>> points;
  interlaced_coord_to_points<DataType, DataType, 3>(vertex_coords, points);

  Vec<3, DataType> edges[3][4];
  edges[0][0] = points[2] - points[1];
  edges[0][1] = points[3] - points[0];
  edges[0][2] = points[4] - points[7];
  edges[0][3] = points[5] - points[6];

  edges[1][0] = points[1] - points[0];
  edges[1][1] = points[3] - points[2];
  edges[1][2] = points[5] - points[4];
  edges[1][3] = points[7] - points[6];

  edges[2][0] = points[4] - points[0];
  edges[2][1] = points[5] - points[1];
  edges[2][2] = points[6] - points[2];
  edges[2][3] = points[7] - points[3];

  for (int i = 0; i < 3; ++i) {
    for (int j = 1; j < 4; ++j) {
      if (!is_parallel(edges[i][j], edges[i][j - 1])) {
        return false;
      }
    }
  }

  return true;
}

template <class DataType >
void parametrize_object (const std::vector<Vec<3, DataType>>& in_points,
                         std::vector < Vec<3, DataType>> &dir_vectors, 
                         std::vector < Vec<3, DataType> >&sup_vectors)
{
  assert((in_points.size() == 4 ) || (in_points.size() == 5) || (in_points.size() == 8 ) );
  //check if first polyhedron is a tetrahedron, pyramid or hexahedron
  
  if (in_points.size() == 4) {    //tetrahedron
    
    //check if all edges intersect the out polyhedron and additionally if the lines connecting the midpoints of the edges with the vertices intersect the polhedron -> 18 lines
    
    
    for (int i = 0; i < 6; i++) {
      sup_vectors.push_back(in_points[0]);
    }
    //direction vectors connecting vertex 0 to the other relevant points
    dir_vectors.push_back(in_points[1]- in_points[0]);
    dir_vectors.push_back(in_points[2]- in_points[0]);
    dir_vectors.push_back(in_points[3]- in_points[0]);
    dir_vectors.push_back(in_points[1] + 0.5*(in_points[3]- in_points[1]) - in_points[0]);
    dir_vectors.push_back(in_points[1] + 0.5*(in_points[2]- in_points[1]) - in_points[0]);
    dir_vectors.push_back(in_points[3] + 0.5*(in_points[2]- in_points[3]) - in_points[0]);
    
    for (int i = 0; i< 5; ++i ) {
      sup_vectors.push_back(in_points[1]);
    }
    dir_vectors.push_back(in_points[2]- in_points[1]);
    dir_vectors.push_back(in_points[3]- in_points[1]);
    dir_vectors.push_back(in_points[0] + 0.5*(in_points[3]- in_points[0]) - in_points[1]);
    dir_vectors.push_back(in_points[0] + 0.5*(in_points[2]- in_points[0]) - in_points[1]);
    dir_vectors.push_back(in_points[3] + 0.5*(in_points[2]- in_points[3]) - in_points[1]);
    
    for (int i = 0; i< 4; ++i ) {
      sup_vectors.push_back(in_points[2]);
    }
    dir_vectors.push_back(in_points[3]- in_points[2]);
    dir_vectors.push_back(in_points[0] + 0.5*(in_points[1]- in_points[0]) - in_points[2]);
    dir_vectors.push_back(in_points[0] + 0.5*(in_points[3]- in_points[0]) - in_points[2]);
    dir_vectors.push_back(in_points[3] + 0.5*(in_points[1]- in_points[3]) - in_points[2]);
    
    for (int i = 0; i< 3; ++i ) {
      sup_vectors.push_back(in_points[3]);
    }

    dir_vectors.push_back(in_points[0] + 0.5*(in_points[1]- in_points[0]) - in_points[3]);
    dir_vectors.push_back(in_points[0] + 0.5*(in_points[2]- in_points[0]) - in_points[3]);
    dir_vectors.push_back(in_points[1] + 0.5*(in_points[2]- in_points[1]) - in_points[3]);
    
    
  }
  
  if (in_points.size() == 5) {    //pyramid
    
    for (int i = 0; i < 4; ++i) {
      sup_vectors.push_back(in_points[0]);
    }
    dir_vectors.push_back(in_points[1]- in_points[0]);
    dir_vectors.push_back(in_points[2]- in_points[0]);
    dir_vectors.push_back(in_points[3]- in_points[0]);
    dir_vectors.push_back(in_points[4]- in_points[0]);
    
    for (int i = 0; i < 3; ++i) {
      sup_vectors.push_back(in_points[1]);
    }
    dir_vectors.push_back(in_points[2]- in_points[1]);
    dir_vectors.push_back(in_points[3]- in_points[1]);
    dir_vectors.push_back(in_points[4]- in_points[1]);
    
    for (int i = 0; i < 2; ++i) {
      sup_vectors.push_back(in_points[2]);
    }

    dir_vectors.push_back(in_points[3]- in_points[2]);
    dir_vectors.push_back(in_points[4]- in_points[2]);
    
    sup_vectors.push_back(in_points[3]);
    dir_vectors.push_back(in_points[4] - in_points[3]);
    
    for (int i = 0; i < 4; ++i) {
      sup_vectors.push_back(in_points[4]);
    }
    dir_vectors.push_back(in_points[0] + 0.5 * (in_points[1]- in_points[0]) - in_points[4]);
    dir_vectors.push_back(in_points[1] + 0.5 * (in_points[2]- in_points[1])- in_points[4]);
    dir_vectors.push_back(in_points[2] + 0.5 * (in_points[3]- in_points[2])- in_points[4]);
    dir_vectors.push_back(in_points[3] + 0.5 * (in_points[0]- in_points[3])- in_points[4]);
    
    //lines connecting midpoints of the ground edges
    
    sup_vectors.push_back(in_points[0] + 0.5 * (in_points[1]- in_points[0]));
    sup_vectors.push_back(in_points[1] + 0.5 * (in_points[2]- in_points[1]));
    
    dir_vectors.push_back(in_points[2] + 0.5 * (in_points[3]- in_points[2]) - (in_points[0] + 0.5 * (in_points[1]- in_points[0] ) ) );
    dir_vectors.push_back(in_points[3] + 0.5 * (in_points[0]- in_points[3])- (in_points[1] + 0.5 * (in_points[2]- in_points[1]) )  );
    
    
    
    
  }
  
  if (in_points.size() == 8) {    //hexahedron
    
    for (int i = 0; i < 7; ++i) {
      sup_vectors.push_back(in_points[0]);
    }
    dir_vectors.push_back(in_points[1]- in_points[0]);
    dir_vectors.push_back(in_points[2]- in_points[0]);
    dir_vectors.push_back(in_points[3]- in_points[0]);
    dir_vectors.push_back(in_points[4]- in_points[0]);
    dir_vectors.push_back(in_points[5]- in_points[0]);
    dir_vectors.push_back(in_points[6]- in_points[0]);
    dir_vectors.push_back(in_points[7]- in_points[0]);
    
    for (int i = 0; i < 6; ++i) {
      sup_vectors.push_back(in_points[1]);
    }
    dir_vectors.push_back(in_points[2]- in_points[1]);
    dir_vectors.push_back(in_points[3]- in_points[1]);
    dir_vectors.push_back(in_points[4]- in_points[1]);
    dir_vectors.push_back(in_points[5]- in_points[1]);
    dir_vectors.push_back(in_points[6]- in_points[1]);
    dir_vectors.push_back(in_points[7]- in_points[1]);
    
    for (int i = 0; i < 5; ++i) {
      sup_vectors.push_back(in_points[2]);
    }

    dir_vectors.push_back(in_points[3]- in_points[2]);
    dir_vectors.push_back(in_points[4]- in_points[2]);
    dir_vectors.push_back(in_points[5]- in_points[2]);
    dir_vectors.push_back(in_points[6]- in_points[2]);
    dir_vectors.push_back(in_points[7]- in_points[2]);
    
    for (int i = 0; i < 4; ++i) {
      sup_vectors.push_back(in_points[3]);
    }

    dir_vectors.push_back(in_points[4]- in_points[3]);
    dir_vectors.push_back(in_points[5]- in_points[3]);
    dir_vectors.push_back(in_points[6]- in_points[3]);
    dir_vectors.push_back(in_points[7]- in_points[3]);
    
    for (int i = 0; i < 3; ++i) {
      sup_vectors.push_back(in_points[4]);
    }

    dir_vectors.push_back(in_points[5]- in_points[4]);
    dir_vectors.push_back(in_points[6]- in_points[4]);
    dir_vectors.push_back(in_points[7]- in_points[4]);
    
    for (int i = 0; i < 2; ++i) {
      sup_vectors.push_back(in_points[5]);
    }

    dir_vectors.push_back(in_points[6]- in_points[5]);
    dir_vectors.push_back(in_points[7]- in_points[5]);
    
    sup_vectors.push_back(in_points[6]);
    dir_vectors.push_back(in_points[7] - in_points[6]);
  }
}              
                     
template <class DataType, int DIM>
void get_boundary_cells_and_normals(ConstMeshPtr mesh, 
                                    const MPI_Comm& comm,
                                    const int mode,
                                    DataType delta, 
                                    SortedArray< int >& bdy_cells,
                                    std::vector< int >& is_bdy_cell,
                                    std::vector< int >& bdy_material_cell,
                                    std::vector< int >& ext_bdy_material_cell,
                                    std::vector< Vec<DIM, DataType> >& cell_normals)
{
  cell_normals.clear();
  bdy_cells.clear();
  is_bdy_cell.clear();
  bdy_material_cell.clear();

  const int tdim = mesh->tdim();
  const int num_cells = mesh->num_entities(tdim);
  
  cell_normals.resize(num_cells);
  is_bdy_cell.resize(num_cells);
  bdy_cells.reserve(num_cells);
  bdy_material_cell.resize(num_cells, -1);
  
  std::vector< std::vector< Vec<DIM, DataType> > > tmp_cell_normals(num_cells);
  
  MeshPtr bdy_mesh = mesh->extract_boundary_mesh();
  
  // Loop over all faces which belong to the boundary
  // -> collect normals
  for (EntityIterator it_boundary = bdy_mesh->begin(tdim - 1);
       it_boundary != bdy_mesh->end(tdim - 1); ++it_boundary) 
  {
    // get id of boundary face
    const Id boundary_id = it_boundary->id();

    // check if the boundary face exists and get the location
    // where the entity number should be stored
    int face_number;
    const bool check = mesh->find_entity(tdim - 1, boundary_id, &face_number);
    assert(check);

    // Get the face to be able to access to the data associated with the face
    Entity face = mesh->get_entity(tdim - 1, face_number);
    const int mat_num = face.get_material_number();
    if (mat_num < 0)
    {
      continue;
    }
    
    Vec<DIM, DataType> f_normal = normal_of_facet<DataType, DIM>(face);
    
    IncidentEntityIterator cell_inc = face.begin_incident(tdim);
    is_bdy_cell[cell_inc->index()] = 2;
    bdy_material_cell[cell_inc->index()] = mat_num;
    tmp_cell_normals[cell_inc->index()].push_back(f_normal);
    bdy_cells.find_insert(cell_inc->index());
    
    if (mode > 0)
    {
      IncidentEntityIterator vertex_begin = face.begin_incident(0);
      IncidentEntityIterator vertex_end = face.end_incident(0);
    
      for (auto vertex_it = vertex_begin; vertex_it != vertex_end; ++vertex_it)
      {
        IncidentEntityIterator cell_begin = vertex_it->begin_incident(tdim);
        IncidentEntityIterator cell_end = vertex_it->end_incident(tdim);
    
        for (auto cell_it = cell_begin; cell_it != cell_end; ++cell_it)
        {
          int cell_index = cell_it->index();
          tmp_cell_normals[cell_index].push_back(f_normal);
          bdy_cells.find_insert(cell_index);
          is_bdy_cell[cell_index] = std::max(is_bdy_cell[cell_index], 1);
          bdy_material_cell[cell_index] = std::max(bdy_material_cell[cell_index], mat_num);
        }
      }
    }
  }
  
  // average normals
  for (int c=0; c!=num_cells; ++c)
  {
    Vec<DIM, DataType> normal;
    for (int l=0; l!=tmp_cell_normals[c].size(); ++l)
    {
      normal += tmp_cell_normals[c][l]; 
    }
    if (tmp_cell_normals[c].size() > 0)
    {
      DataType nb_normals = static_cast<DataType>(tmp_cell_normals[c].size());
      normal *= (1. / nb_normals);
    }
    cell_normals[c] = normal;
  }

  // compute neighborhood of boundary of width delta
  std::vector<int> num_recv_cells;
  std::vector<int> num_send_cells;
   
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
  std::vector< std::vector< int > > recv_cell_mat_number;
  
  if (num_proc > 1)
  {
    broadcast_cell_data<int>(parcom, 
                             1,
                             num_recv_cells,
                             num_send_cells,
                             bdy_material_cell,
                             recv_cell_mat_number);

    broadcast_cell_data<DataType>(parcom, 
                                  DIM,
                                  num_recv_cells,
                                  num_send_cells,
                                  my_centers,
                                  recv_cell_data_center);
  }
  else
  {
    recv_cell_mat_number.resize(num_proc);
    recv_cell_data_center.resize(num_proc);
  }
  
  // determine which own cells are in a delta-neighborhood of the interface
  ext_bdy_material_cell.clear();
  ext_bdy_material_cell.resize(num_cells, -1);

  for (int c = 0; c!= num_cells; ++c)
  {
    // no further action needed for boundary cells
    if (is_bdy_cell[c] > 0)
    {
      ext_bdy_material_cell[c] = bdy_material_cell[c];
      continue;
    }
    
    bool next_cell = false;
    
    // loop over own boundary cells
    for (int ci = 0; ci!= num_cells; ++ci)
    {
      if (is_bdy_cell[ci] == 0)
      {
        // skip non-bdy cells
        continue;
      }
      DataType dist = 0.;
      for (int d = 0; d!= DIM; ++d)
      {
        DataType a = my_centers[c*DIM+d];
        DataType b = my_centers[ci*DIM+d];
        dist += (a - b) * (a - b);
      }
      dist = std::sqrt(dist);
      
      if (dist < delta)
      {
        ext_bdy_material_cell[c] = bdy_material_cell[ci];
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
    
    // loop over remote boundary cells
    for (int p=0; p != num_proc; ++p)
    {
      const size_t num_cells_p = recv_cell_mat_number[p].size();
    
      if (num_cells_p == 0)
      {
        continue;
      }
      for (size_t cr = 0; cr != num_cells_p; ++cr)
      {
        // skip non-bdy cells
        if (recv_cell_mat_number[p][cr] < 0)
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
        dist = std::sqrt(dist);
      
        if (dist < delta)
        {
          ext_bdy_material_cell[c] = recv_cell_mat_number[p][cr];
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



                     
                            
template Vec<3, float> normal<float, 3>(const std::vector< Vec<3, float> > &);
template Vec<2, float> normal<float, 2>(const std::vector< Vec<2, float> > &);
template Vec<1, float> normal<float, 1>(const std::vector< Vec<1, float> > &);
template Vec<3, double> normal<double, 3>(const std::vector< Vec<3, double> > &);
template Vec<2, double> normal<double, 2>(const std::vector< Vec<2, double> > &);
template Vec<1, double> normal<double, 1>(const std::vector< Vec<1, double> > &);

template Vec<3, float> normal_of_facet<float, 3>(const std::vector< Vec<3, float> > &);
template Vec<2, float> normal_of_facet<float, 2>(const std::vector< Vec<2, float> > &);
template Vec<1, float> normal_of_facet<float, 1>(const std::vector< Vec<1, float> > &);
template Vec<3, double> normal_of_facet<double, 3>(const std::vector< Vec<3, double> > &);
template Vec<2, double> normal_of_facet<double, 2>(const std::vector< Vec<2, double> > &);
template Vec<1, double> normal_of_facet<double, 1>(const std::vector< Vec<1, double> > &);

template Vec<3, float> normal_of_facet<float, 3>(const Entity&);
template Vec<2, float> normal_of_facet<float, 2>(const Entity&);
template Vec<1, float> normal_of_facet<float, 1>(const Entity&);
template Vec<3, double> normal_of_facet<double, 3>(const Entity&);
template Vec<2, double> normal_of_facet<double, 2>(const Entity&);
template Vec<1, double> normal_of_facet<double, 1>(const Entity&);

template void compute_facet_projection_matrix<float, 2> (const std::vector< Vec<2, float> >&, Mat<2, 1, float>&);
template void compute_facet_projection_matrix<float, 3> (const std::vector< Vec<3, float> >&, Mat<3, 2, float>&);
template void compute_facet_projection_matrix<double, 2> (const std::vector< Vec<2, double> >&, Mat<2, 1, double>&);
template void compute_facet_projection_matrix<double, 3> (const std::vector< Vec<3, double> >&, Mat<3, 2, double>&);                                      
                                      
template float distance_point_hyperplane<float, 3>(const Vec<3, float> &, const Vec<3, float> &, const Vec<3, float> &);
template float distance_point_hyperplane<float, 2>(const Vec<2, float> &point, const Vec<2, float> &, const Vec<2, float> &);
template float distance_point_hyperplane<float, 1>(const Vec<1, float> &, const Vec<1, float> &, const Vec<1, float> &);
template double distance_point_hyperplane<double, 3>(const Vec<3, double> &, const Vec<3, double> &, const Vec<3, double> &);
template double distance_point_hyperplane<double, 2>(const Vec<2, double> &, const Vec<2, double> &, const Vec<2, double> &);
template double distance_point_hyperplane<double, 1>(const Vec<1, double> &, const Vec<1, double> &, const Vec<1, double> &);

template float distance_point_line<float, 3>(const Vec<3, float> &, const Vec<3, float> &, const Vec<3, float> &);
template float distance_point_line<float, 2>(const Vec<2, float> &, const Vec<2, float> &, const Vec<2, float> &);
template float distance_point_line<float, 1>(const Vec<1, float> &, const Vec<1, float> &, const Vec<1, float> &);
template double distance_point_line<double, 3>(const Vec<3, double> &, const Vec<3, double> &, const Vec<3, double> &);
template double distance_point_line<double, 2>(const Vec<2, double> &, const Vec<2, double> &, const Vec<2, double> &);
template double distance_point_line<double, 1>(const Vec<1, double> &, const Vec<1, double> &, const Vec<1, double> &);

template Vec<3, float> foot_point_hyperplane<float, 3>(const Vec<3, float> &, const Vec<3, float> &, const Vec<3, float> &);
template Vec<2, float> foot_point_hyperplane<float, 2>(const Vec<2, float> &, const Vec<2, float> &, const Vec<2, float> &);
template Vec<1, float> foot_point_hyperplane<float, 1>(const Vec<1, float> &, const Vec<1, float> &, const Vec<1, float> &);
template Vec<3, double> foot_point_hyperplane<double, 3>(const Vec<3, double> &, const Vec<3, double> &, const Vec<3, double> &);
template Vec<2, double> foot_point_hyperplane<double, 2>(const Vec<2, double> &, const Vec<2, double> &, const Vec<2, double> &);
template Vec<1, double> foot_point_hyperplane<double, 1>(const Vec<1, double> &, const Vec<1, double> &, const Vec<1, double> &);

template Vec<3, float> foot_point_line<float, 3>(const Vec<3, float> &, const Vec<3, float> &, const Vec<3, float> &);
template Vec<2, float> foot_point_line<float, 2>(const Vec<2, float> &, const Vec<2, float> &, const Vec<2, float> &);
template Vec<1, float> foot_point_line<float, 1>(const Vec<1, float> &, const Vec<1, float> &, const Vec<1, float> &);
template Vec<3, double> foot_point_line<double, 3>(const Vec<3, double> &, const Vec<3, double> &, const Vec<3, double> &);
template Vec<2, double> foot_point_line<double, 2>(const Vec<2, double> &, const Vec<2, double> &, const Vec<2, double> &);
template Vec<1, double> foot_point_line<double, 1>(const Vec<1, double> &, const Vec<1, double> &, const Vec<1, double> &);

template float triangle_area<float, 3>(const std::vector< float > &);
template float triangle_area<float, 2>(const std::vector< float > &);
template float triangle_area<float, 1>(const std::vector< float > &);
template double triangle_area<double, 3>(const std::vector< double > &);
template double triangle_area<double, 2>(const std::vector< double > &);
template double triangle_area<double, 1>(const std::vector< double > &);

template float facet_area<float, 3>(const std::vector< float > &, const GDim );
template float facet_area<float, 2>(const std::vector< float > &, const GDim );
template float facet_area<float, 1>(const std::vector< float > &, const GDim );
template double facet_area<double, 3>(const std::vector< double > &, const GDim );
template double facet_area<double, 2>(const std::vector< double > &, const GDim );
template double facet_area<double, 1>(const std::vector< double > &, const GDim );

template bool in_plane<float, 3>(const Vec<3, float> &, const Vec<3, float> &, const Vec<3, float> &, const float ) ;
template bool in_plane<float, 2>(const Vec<2, float> &, const Vec<2, float> &, const Vec<2, float> &, const float ) ;
template bool in_plane<float, 1>(const Vec<1, float> &, const Vec<1, float> &, const Vec<1, float> &, const float ) ;
template bool in_plane<double, 3>(const Vec<3, double> &, const Vec<3, double> &, const Vec<3, double> &, const double ) ;
template bool in_plane<double, 2>(const Vec<2, double> &, const Vec<2, double> &, const Vec<2, double> &, const double ) ;
template bool in_plane<double, 1>(const Vec<1, double> &, const Vec<1, double> &, const Vec<1, double> &, const double ) ;

template bool crossed_plane<float, 3>(const Vec<3, float> &, const Vec<3, float> &, const Vec<3, float> &, const Vec<3, float> &);
template bool crossed_plane<float, 2>(const Vec<2, float> &, const Vec<2, float> &, const Vec<2, float> &, const Vec<2, float> &);
template bool crossed_plane<float, 1>(const Vec<1, float> &, const Vec<1, float> &, const Vec<1, float> &, const Vec<1, float> &);
template bool crossed_plane<double, 3>(const Vec<3, double> &, const Vec<3, double> &, const Vec<3, double> &, const Vec<3, double> &);
template bool crossed_plane<double, 2>(const Vec<2, double> &, const Vec<2, double> &, const Vec<2, double> &, const Vec<2, double> &);
template bool crossed_plane<double, 1>(const Vec<1, double> &, const Vec<1, double> &, const Vec<1, double> &, const Vec<1, double> &);

template bool crossed_facet<float, 3>(const Vec<3, float> &, const Vec<3, float> &, const std::vector< float > &);
template bool crossed_facet<float, 2>(const Vec<2, float> &, const Vec<2, float> &, const std::vector< float > &);
template bool crossed_facet<float, 1>(const Vec<1, float> &, const Vec<1, float> &, const std::vector< float > &);
template bool crossed_facet<double, 3>(const Vec<3, double> &, const Vec<3, double> &, const std::vector< double > &);
template bool crossed_facet<double, 2>(const Vec<2, double> &, const Vec<2, double> &, const std::vector< double > &);
template bool crossed_facet<double, 1>(const Vec<1, double> &, const Vec<1, double> &, const std::vector< double > &);

template Vec<3, float> intersect_facet<float, 3>(const Vec<3, float> &, const Vec<3, float> &, const std::vector< float > &, bool &);
template Vec<2, float> intersect_facet<float, 2>(const Vec<2, float> &, const Vec<2, float> &, const std::vector< float > &, bool &);
template Vec<1, float> intersect_facet<float, 1>(const Vec<1, float> &, const Vec<1, float> &, const std::vector< float > &, bool &);
template Vec<3, double> intersect_facet<double, 3>(const Vec<3, double> &, const Vec<3, double> &, const std::vector< double > &, bool &);
template Vec<2, double> intersect_facet<double, 2>(const Vec<2, double> &, const Vec<2, double> &, const std::vector< double > &, bool &);
template Vec<1, double> intersect_facet<double, 1>(const Vec<1, double> &, const Vec<1, double> &, const std::vector< double > &, bool &);

template float distance_point_facet<float, 3>(const Vec<3, float> &, const std::vector< float > &, Vec<3, float> &);
template float distance_point_facet<float, 2>(const Vec<2, float> &, const std::vector< float > &, Vec<2, float> &);
template float distance_point_facet<float, 1>(const Vec<1, float> &, const std::vector< float > &, Vec<1, float> &);
template double distance_point_facet<double, 3>(const Vec<3, double> &, const std::vector< double > &, Vec<3, double> &);
template double distance_point_facet<double, 2>(const Vec<2, double> &, const std::vector< double > &, Vec<2, double> &);
template double distance_point_facet<double, 1>(const Vec<1, double> &, const std::vector< double > &, Vec<1, double> &);

template double compute_entity_diameter<double, 3> (const Entity&);
template double compute_entity_diameter<double, 2> (const Entity&);
template double compute_entity_diameter<double, 1> (const Entity&);

template float compute_entity_diameter<float, 3> (const Entity&);
template float compute_entity_diameter<float, 2> (const Entity&);
template float compute_entity_diameter<float, 1> (const Entity&);

template bool point_inside_entity<float, 3>(const Vec<3, float> &, const TDim , const std::vector< float > &);
template bool point_inside_entity<float, 2>(const Vec<2, float> &, const TDim , const std::vector< float > &);
template bool point_inside_entity<float, 1>(const Vec<1, float> &, const TDim , const std::vector< float > &);
template bool point_inside_entity<double, 3>(const Vec<3, double> &, const TDim , const std::vector< double > &);
template bool point_inside_entity<double, 2>(const Vec<2, double> &, const TDim , const std::vector< double > &);
template bool point_inside_entity<double, 1>(const Vec<1, double> &, const TDim , const std::vector< double > &);

template bool point_inside_cell<float, 3>(const Vec<3, float> &, const std::vector< float > &, Vec<3, float> &);
template bool point_inside_cell<float, 2>(const Vec<2, float> &, const std::vector< float > &, Vec<2, float> &);
template bool point_inside_cell<float, 1>(const Vec<1, float> &, const std::vector< float > &, Vec<1, float> &);
template bool point_inside_cell<double, 3>(const Vec<3, double> &, const std::vector< double > &, Vec<3, double> &);
template bool point_inside_cell<double, 2>(const Vec<2, double> &, const std::vector< double > &, Vec<2, double> &);
template bool point_inside_cell<double, 1>(const Vec<1, double> &, const std::vector< double > &, Vec<1, double> &);

template bool vertices_inside_one_hyperplane<float, 3>(const std::vector< float > &,
    const TDim, const float);
template bool vertices_inside_one_hyperplane<float, 2>(const std::vector< float > &,
    const TDim, const float);
template bool vertices_inside_one_hyperplane<float, 1>(const std::vector< float > &,
    const TDim, const float);
template bool vertices_inside_one_hyperplane<double, 3>(const std::vector< double > &,
    const TDim, const double);
template bool vertices_inside_one_hyperplane<double, 2>(const std::vector< double > &,
    const TDim, const double );
template bool vertices_inside_one_hyperplane<double, 1>(const std::vector< double > &,
    const TDim, const double );

template Vec<3, float> project_point<float, 3>(const BoundaryDomainDescriptor<float, 3> &, const Vec<3, float> &, const MaterialNumber );
template Vec<2, float> project_point<float, 2>(const BoundaryDomainDescriptor<float, 2> &, const Vec<2, float> &, const MaterialNumber );
template Vec<1, float> project_point<float, 1>(const BoundaryDomainDescriptor<float, 1> &, const Vec<1, float> &, const MaterialNumber );
template Vec<3, double> project_point<double, 3>(const BoundaryDomainDescriptor<double, 3> &, const Vec<3, double> &, const MaterialNumber );
template Vec<2, double> project_point<double, 2>(const BoundaryDomainDescriptor<double, 2> &, const Vec<2, double> &, const MaterialNumber );
template Vec<1, double> project_point<double, 1>(const BoundaryDomainDescriptor<double, 1> &, const Vec<1, double> &, const MaterialNumber );

#if 0
template bool map_ref_coord_to_other_cell<float, 3> ( const Vec<3, float> & ,
                                                      Vec<3, float> & ,
                                                      doffem::CellTransformation<float, 3> const * ,
                                                      doffem::CellTransformation<float, 3> const * ,
                                                      std::vector< mesh::MasterSlave > const & );
template bool map_ref_coord_to_other_cell<float, 2> ( const Vec<2, float> &,
                                                      Vec<2, float> &,
                                                      doffem::CellTransformation<float, 2> const *,
                                                      doffem::CellTransformation<float, 2> const *,
                                                      std::vector< mesh::MasterSlave > const & );
template bool map_ref_coord_to_other_cell<float, 1> ( const Vec<1, float> &,
                                                      Vec<1, float> &,
                                                      doffem::CellTransformation<float, 1> const *,
                                                      doffem::CellTransformation<float, 1> const *,
                                                      std::vector< mesh::MasterSlave > const & );
template bool map_ref_coord_to_other_cell<double, 3> ( const Vec<3, double> &,
                                                      Vec<3, double> &,
                                                      doffem::CellTransformation<double, 3> const *,
                                                      doffem::CellTransformation<double, 3> const *,
                                                      std::vector< mesh::MasterSlave > const & );
template bool map_ref_coord_to_other_cell<double, 2> ( const Vec<2, double> &,
                                                      Vec<2, double> &,
                                                      doffem::CellTransformation<double, 2> const *,
                                                      doffem::CellTransformation<double, 2> const *,
                                                      std::vector< mesh::MasterSlave > const & );
template bool map_ref_coord_to_other_cell<double, 1> ( const Vec<1, double> &,
                                                      Vec<1, double> &,
                                                      doffem::CellTransformation<double, 1> const *,
                                                      doffem::CellTransformation<double, 1> const *,
                                                      std::vector< mesh::MasterSlave > const & );
#endif
template void find_subentities_containing_point<float, 3> (const Vec<3, float> &, 
                                                           const mesh::CellType *,
                                                           const std::vector< Vec<3, float> >& , 
                                                           std::vector< std::vector<int> > &);
template void find_subentities_containing_point<float, 2> (const Vec<2, float>&, 
                                                           const mesh::CellType *,
                                                           const std::vector< Vec<2, float> >& , 
                                                           std::vector< std::vector<int> > &);
template void find_subentities_containing_point<float, 1> (const Vec<1, float> &, 
                                                           const mesh::CellType *,
                                                           const std::vector< Vec<1, float> >& , 
                                                           std::vector< std::vector<int> > &);
template void find_subentities_containing_point<double, 3> (const Vec<3, double> &, 
                                                            const mesh::CellType *,
                                                            const std::vector< Vec<3, double> >& , 
                                                            std::vector< std::vector<int> > &);
template void find_subentities_containing_point<double, 2> (const Vec<2, double> &, 
                                                            const mesh::CellType *,
                                                            const std::vector< Vec<2, double> >& , 
                                                            std::vector< std::vector<int> > &);
template void find_subentities_containing_point<double, 1> (const Vec<1, double>&, 
                                                            const mesh::CellType *,
                                                            const std::vector< Vec<1, double> >& , 
                                                            std::vector< std::vector<int> > &);


template bool is_point_on_subentity<float, 3> (const Vec<3, float>& , const std::vector< Vec<3, float> > &);
template bool is_point_on_subentity<float, 2> (const Vec<2, float>& , const std::vector< Vec<2, float> > &);
template bool is_point_on_subentity<float, 1> (const Vec<1, float>& , const std::vector< Vec<1, float> > &);
template bool is_point_on_subentity<double, 3> (const Vec<3, double>& , const std::vector< Vec<3, double> > &);
template bool is_point_on_subentity<double, 2> (const Vec<2, double>& , const std::vector< Vec<2, double> > &);
template bool is_point_on_subentity<double, 1> (const Vec<1, double>& , const std::vector< Vec<1, double> > &);

template void create_bbox_for_mesh<float, 3>  (ConstMeshPtr meshptr, BBox<float, 3>& bbox);
template void create_bbox_for_mesh<float, 2>  (ConstMeshPtr meshptr, BBox<float, 2>& bbox);
template void create_bbox_for_mesh<float, 1>  (ConstMeshPtr meshptr, BBox<float, 1>& bbox);
template void create_bbox_for_mesh<double, 3> (ConstMeshPtr meshptr, BBox<double, 3>& bbox);
template void create_bbox_for_mesh<double, 2> (ConstMeshPtr meshptr, BBox<double, 2>& bbox);
template void create_bbox_for_mesh<double, 1> (ConstMeshPtr meshptr, BBox<double, 1>& bbox);

template void create_bbox_for_entity<float, 3> (const Entity&, BBox<float, 3>& bbox);
template void create_bbox_for_entity<float, 2> (const Entity&, BBox<float, 2>& bbox);
template void create_bbox_for_entity<float, 1> (const Entity&, BBox<float, 1>& bbox);
template void create_bbox_for_entity<double, 3> (const Entity&, BBox<double, 3>& bbox);
template void create_bbox_for_entity<double, 2> (const Entity&, BBox<double, 2>& bbox);
template void create_bbox_for_entity<double, 1> (const Entity&, BBox<double, 1>& bbox);

template std::vector< float >  compute_mean_edge_length<float, 3>  (ConstMeshPtr meshptr);
template std::vector< float >  compute_mean_edge_length<float, 2>  (ConstMeshPtr meshptr);
template std::vector< float >  compute_mean_edge_length<float, 1>  (ConstMeshPtr meshptr);
template std::vector< double > compute_mean_edge_length<double, 3> (ConstMeshPtr meshptr);
template std::vector< double > compute_mean_edge_length<double, 2> (ConstMeshPtr meshptr);
template std::vector< double > compute_mean_edge_length<double, 1> (ConstMeshPtr meshptr);

template void compute_mesh_grid_map<float, 1> (ConstMeshPtr, Grid<float, 1>&, bool, std::vector< std::list< int > >&, std::vector< std::list< int > >&);
template void compute_mesh_grid_map<float, 2> (ConstMeshPtr, Grid<float, 2>&, bool,std::vector< std::list< int > >&, std::vector< std::list< int > >&);
template void compute_mesh_grid_map<float, 3> (ConstMeshPtr, Grid<float, 3>&, bool,std::vector< std::list< int > >&, std::vector< std::list< int > >&);
template void compute_mesh_grid_map<double, 1> (ConstMeshPtr, Grid<double, 1>&, bool, std::vector< std::list< int > >&, std::vector< std::list< int > >&);
template void compute_mesh_grid_map<double, 2> (ConstMeshPtr, Grid<double, 2>&, bool, std::vector< std::list< int > >&, std::vector< std::list< int > >&);
template void compute_mesh_grid_map<double, 3> (ConstMeshPtr, Grid<double, 3>&, bool, std::vector< std::list< int > >&, std::vector< std::list< int > >&);

template void find_adjacent_cells <float, 3> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh, std::map< EntityNumber, std::set<EntityNumber> >& adjacent_map);
template void find_adjacent_cells <float, 2> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh, std::map< EntityNumber, std::set<EntityNumber> >& adjacent_map);
template void find_adjacent_cells <float, 1> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh, std::map< EntityNumber, std::set<EntityNumber> >& adjacent_map);
template void find_adjacent_cells <double, 3> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh, std::map< EntityNumber, std::set<EntityNumber> >& adjacent_map);
template void find_adjacent_cells <double, 2> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh, std::map< EntityNumber, std::set<EntityNumber> >& adjacent_map);
template void find_adjacent_cells <double, 1> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh, std::map< EntityNumber, std::set<EntityNumber> >& adjacent_map);

template void find_adjacent_cells_related <float, 3> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh,
    std::map<EntityNumber, std::set<EntityNumber>>& cell_map);
template void find_adjacent_cells_related <float, 2> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh,
    std::map<EntityNumber, std::set<EntityNumber>>& cell_map);
template void find_adjacent_cells_related <float, 1> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh,
    std::map<EntityNumber, std::set<EntityNumber>>& cell_map);
template void find_adjacent_cells_related <double, 3> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh,
    std::map<EntityNumber, std::set<EntityNumber>>& cell_map);
template void find_adjacent_cells_related <double, 2> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh,
    std::map<EntityNumber, std::set<EntityNumber>>& cell_map);
template void find_adjacent_cells_related <double, 1> (ConstMeshPtr in_mesh, ConstMeshPtr out_mesh,
    std::map<EntityNumber, std::set<EntityNumber>>& cell_map);

template bool cells_intersect<double, 1>(const std::vector < double>&,const  std::vector < double>&  );
template bool cells_intersect<double, 2>(const std::vector < double>&,const  std::vector < double>&  );
template bool cells_intersect<double, 3>(const std::vector < double>&,const  std::vector < double>&  );
template bool cells_intersect<float, 1> (const std::vector < float >&,const  std::vector < float >&  );
template bool cells_intersect<float, 2> (const std::vector < float >&,const  std::vector < float >&  );
template bool cells_intersect<float, 3> (const std::vector < float >&,const  std::vector < float >&  );

template bool cells_intersect<double, 1>(const std::vector < double>&,const  std::vector < double>& ,
                                         const std::vector<Vec<1, double> >&, const std::vector<Vec<1, double> >&);
template bool cells_intersect<double, 2>(const std::vector < double>&,const  std::vector < double>& ,
                                         const std::vector<Vec<2, double> >&, const std::vector<Vec<2, double> >&);
template bool cells_intersect<double, 3>(const std::vector < double>&,const  std::vector < double>& ,
                                         const std::vector<Vec<3, double> >&, const std::vector<Vec<3, double> >&);
template bool cells_intersect<float, 1> (const std::vector < float >&,const  std::vector < float >& ,
                                         const std::vector<Vec<1, float > >&, const std::vector<Vec<1, float > >&);
template bool cells_intersect<float, 2> (const std::vector < float >&,const  std::vector < float >& ,
                                         const std::vector<Vec<2, float > >&, const std::vector<Vec<2, float > >&);
template bool cells_intersect<float, 3> (const std::vector < float >&,const  std::vector < float >& ,
                                         const std::vector<Vec<3, float > >&, const std::vector<Vec<3, float > >&);

template bool is_aligned_rectangular_cuboid<float>  (const std::vector < float>& in_vertex_coords);
template bool is_aligned_rectangular_cuboid<double> (const std::vector < double>& in_vertex_coords);

template bool is_aligned_rectangular_quad<float>  (const std::vector < float>& in_vertex_coords);
template bool is_aligned_rectangular_quad<double> (const std::vector < double>& in_vertex_coords);

template bool is_parallelogram<float> (const std::vector<float>& in_vertex_coords);
template bool is_parallelogram<double>(const std::vector<double>& in_vertex_coords);

template bool is_parallelepiped<float> (const std::vector<float>& in_vertex_coords);
template bool is_parallelepiped<double>(const std::vector<double>& in_vertex_coords);

template void parametrize_object<double>(const std::vector<Vec<3, double>>& in_points, std::vector <Vec<3, double >> &dir_vectors, std::vector < Vec<3,double> >&sup_vectors);
template void parametrize_object<float> (const std::vector<Vec<3, float>>& in_points, std::vector <Vec<3, float >> &dir_vectors, std::vector < Vec<3,float> >&sup_vectors);

template void get_boundary_cells_and_normals<float, 3> (ConstMeshPtr, const MPI_Comm&, const int, float, SortedArray< int >&,                                 
                                                        std::vector< int >& , std::vector< int >& , std::vector< int >& ,
                                                        std::vector< Vec<3, float> >&);
template void get_boundary_cells_and_normals<float, 2> (ConstMeshPtr, const MPI_Comm&, const int, float, SortedArray< int >&,                                 
                                                        std::vector< int >& , std::vector< int >& , std::vector< int >& ,
                                                        std::vector< Vec<2, float> >&);                                                        
template void get_boundary_cells_and_normals<float, 1> (ConstMeshPtr, const MPI_Comm&, const int, float, SortedArray< int >&,                                 
                                                        std::vector< int >& , std::vector< int >& , std::vector< int >& ,
                                                        std::vector< Vec<1, float> >&);
template void get_boundary_cells_and_normals<double, 3> (ConstMeshPtr, const MPI_Comm&, const int, double, SortedArray< int >&,                                 
                                                        std::vector< int >& , std::vector< int >& , std::vector< int >& ,
                                                        std::vector< Vec<3, double> >&);
template void get_boundary_cells_and_normals<double, 2> (ConstMeshPtr, const MPI_Comm&, const int, double, SortedArray< int >&,                                 
                                                         std::vector< int >& , std::vector< int >& , std::vector< int >& ,
                                                         std::vector< Vec<2, double> >&);
template void get_boundary_cells_and_normals<double, 1> (ConstMeshPtr, const MPI_Comm&, const int, double, SortedArray< int >&,                                 
                                                         std::vector< int >& , std::vector< int >& , std::vector< int >& ,
                                                         std::vector< Vec<1, double> >&);
} // namespace mesh
} // namespace hiflow
