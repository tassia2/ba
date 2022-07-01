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

#include "fem/cut_fem/cell_cut.h"

#include "common/bbox.h"
#include "common/array_tools.h"
#include "common/parcom.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "mesh/entity.h"
#include "mesh/geometric_tools.h"
#include "mesh/iterator.h"
#include "mesh/mesh_tools.h"
#include "mesh/refinement.h"
#include "mesh/types.h"

#include <limits>

namespace hiflow {
namespace doffem {

template <class DataType, int DIM>
CutType CellCutter<DataType, DIM>::check_cut(const std::vector<mesh::Id>& cutpoint_ids, 
                                             const std::vector< VertexCoord >& cutpoint_coords, 
                                             const mesh::TDim tdim,
                                             bool &decompose_non_simplex)
{
  assert (cutpoint_ids.size() == cutpoint_coords.size());
  assert (cutpoint_coords.size() > 0);
  bool edgecut = true;
  for (size_t i = 0; i < cutpoint_ids.size(); ++i) 
  {
    if (cutpoint_ids[i] < 0) 
    {
      edgecut = false;
      break;
    }
  }
  if (edgecut && cutpoint_ids.size() == 1) 
  {
    return CutType::Point;
  } 
  else if  (edgecut && cutpoint_ids.size() >= DIM) 
  {
    return CutType::Interface;
  }
  else if (edgecut)
  {
    return CutType::Edge;
  } 

  std::vector<DataType> tmp_cutpoint_coords;
  points_to_interlaced_coord<DataType, DataType, DIM>(cutpoint_coords, tmp_cutpoint_coords);
  if (!mesh::vertices_inside_one_hyperplane<DataType, DIM>(tmp_cutpoint_coords, tdim, GEOM_TOL)
      || cutpoint_coords.size() > 4)
  {
    decompose_non_simplex = true;
  }
  return CutType::Proper;
}

static void assemble_partition_vertices(const mesh::Entity &entity,
                                        const std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>[2]> &faces,
                                        const std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>> &cutpoint_ids,
                                        std::set<mesh::Id> subpolys[2])
{
  for (int k = 0; k < 2; ++k) 
  {
    const mesh::TDim tdim = entity.tdim();
    for (mesh::IncidentEntityIterator it = entity.begin_incident(tdim - 1);
        it != entity.end_incident(tdim - 1); ++it) 
    {
      mesh::EntityNumber e = it->index();
      subpolys[k].insert(faces.at(e)[k].begin(), faces.at(e)[k].end());
    }

    mesh::EntityNumber e = entity.index();
    subpolys[k].insert(cutpoint_ids.at(e).begin(), cutpoint_ids.at(e).end());
  }
}

template <class DataType, int DIM>
void CellCutter<DataType, DIM>::compute_entity_partitions(const mesh::Entity &entity,
                                                          const std::unordered_map<mesh::Id, std::vector<int> >& domains,
                                                          const std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>> cutpoint_ids[DIM],
                                                          std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>[2]> partitions[DIM],
                                                          std::unordered_map<mesh::Id, SubCellPointType>& point_types)
{
  const mesh::TDim tdim = entity.tdim() - 1;
  mesh::EntityNumber e = entity.index();

  if (tdim == DIM-1)
  {
    // cell level -> clear point types
    point_types.clear();
  }

  if (tdim == 0) 
  {
    mesh::VertexIdIterator v = entity.begin_vertex_ids();
    mesh::Id idxs[2] = { *(v++), *v };
    std::vector<mesh::Id> doms[2] = { domains.at(idxs[0]), domains.at(idxs[1]) };

    std::vector<mesh::Id> cutidxs = cutpoint_ids[tdim].at(entity.index());

    // determine type of vertices in current cell
    for (int k=0; k!=2; ++k)
    {
      if (doms[k].size() == 2) 
      {
        assert ( (point_types.find(idxs[k]) == point_types.end()) ||  (point_types[idxs[k]] == SubCellPointType::CutPoint) );
        point_types[idxs[k]] = SubCellPointType::CutPoint;
      }
      else 
      {
        assert ( (point_types.find(idxs[k]) == point_types.end()) ||  (point_types[idxs[k]] == SubCellPointType::NoCutPoint) );
        point_types[idxs[k]] = SubCellPointType::NoCutPoint;
      }
    }
    for (auto cut_id : cutidxs)
    {
      assert ( (point_types.find(cut_id) == point_types.end()) ||  (point_types[cut_id] == SubCellPointType::CutPoint) );
      point_types[cut_id] = SubCellPointType::CutPoint;
    }

    if (doms[0].size() == 2 && doms[1].size() == 2) 
    {
      // idxs_0 and idxs_1 are vertex cut points
      partitions[tdim][e][doms[0][0]] = {idxs[0], idxs[1]};
      partitions[tdim][e][doms[0][1]] = {idxs[0], idxs[1]};
    } 
    else if (doms[0].size() == 2) 
    {
      // only idxs_0 is vertex cut points
      assert (doms[1].size() == 1);

      partitions[tdim][e][doms[1][0]] = {idxs[0], idxs[1]};
    } 
    else if (doms[1].size() == 2) 
    {
      // only idxs_1 is vertex cut points
      assert (doms[0].size() == 1);

      partitions[tdim][e][doms[0][0]] = {idxs[0], idxs[1]};
    } 
    else if (doms[0] == doms[1]) 
    {
      // idxs_0 and idxs_1 are no cutpoints and inside same domain
      assert (doms[0].size() == 1);
      assert (doms[1].size() == 1);

      partitions[tdim][e][doms[0][0]] = {idxs[0], idxs[1]};
    } 
    else 
    {
      // idxs_0 and idxs_1 are no cutpoints, but inside different domains
      assert (doms[0].size() == 1);
      assert (doms[1].size() == 1);

      partitions[tdim][e][doms[0][0]] = {idxs[0], cutidxs[0]};
      partitions[tdim][e][doms[1][0]] = {cutidxs[0], idxs[1]};
    }
    return;
  }

  for (mesh::IncidentEntityIterator it = entity.begin_incident(tdim);
      it != entity.end_incident(tdim); ++it) 
  {
    mesh::Entity face = *it;
    compute_entity_partitions(face, domains, cutpoint_ids, partitions, point_types);
  }

  std::set<mesh::Id> subpolys[2];
  assemble_partition_vertices(entity, partitions[tdim - 1], cutpoint_ids[tdim], subpolys);
  partitions[tdim][e][0] = std::vector<int>(subpolys[0].begin(), subpolys[0].end());
  partitions[tdim][e][1] = std::vector<int>(subpolys[1].begin(), subpolys[1].end());
}

template <class DataType, int DIM>
void CellCutter<DataType, DIM>::assemble_sub_polytopes(std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>[2]> partitions[DIM],
                                                       std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>> cutpoint_ids[DIM],
                                                       std::vector<std::vector<mesh::Id>> (&subpolys)[2][DIM])
{
  for (int d = 0; d < DIM; ++d) 
  {
    for (int k = 0; k < 2; ++k) 
    {
      for (auto it = partitions[d].begin(); it != partitions[d].end(); ++it) 
      {
        if (partitions[d][it->first][k].size() == 0)
          continue;
        
        subpolys[k][d].push_back(partitions[d][it->first][k]);
      }
    }
  }

  for (int d = 0; d < DIM - 1; ++d) 
  {
    for (auto it = cutpoint_ids[d + 1].begin(); it != cutpoint_ids[d + 1].end(); ++it) 
    {
      if (cutpoint_ids[d + 1][it->first].size() == 0)
        continue;

      subpolys[0][d].push_back(cutpoint_ids[d + 1][it->first]);
      subpolys[1][d].push_back(cutpoint_ids[d + 1][it->first]);
    }
  }
}

template <int DIM>
static bool polytope_is_triangle(const std::vector<std::vector<mesh::Id>> poly[DIM])
{
  assert(DIM >= 2);
  if (poly[1].size() != 1 || poly[1][0].size() != 3)
    return false;

  if (poly[0].size() != 3)
    return false;

  return true;
}

template <int DIM>
static bool polytope_is_quadriliteral(const std::vector<std::vector<mesh::Id>> poly[DIM])
{
  assert(DIM >= 2);
  if (poly[1].size() != 1 || poly[1][0].size() != 4)
    return false;

  if (poly[0].size() != 4)
    return false;

  return true;
}

template <int DIM>
static bool polytope_is_tetrahedron(const std::vector<std::vector<mesh::Id>> poly[DIM])
{
  assert(DIM == 3);

  if (poly[2].size() != 1 || poly[2][0].size() != 4)
    return false;

  if (poly[1].size() != 4)
    return false;

  for (size_t i = 0; i < poly[1].size(); ++i) {
    if (poly[1][i].size() != 3)
      return false;
  }

  if (poly[0].size() != 6)
    return false;

  return true;
}

template <int DIM>
static bool polytope_is_pyramid(const std::vector<std::vector<mesh::Id>> poly[DIM])
{
  assert(DIM == 3);

  if (poly[2].size() != 1 || poly[2][0].size() != 5)
    return false;

  if (poly[1].size() != 5)
    return false;

  bool found_quad = false;
  for (size_t i = 0; i < poly[1].size(); ++i) {
    if (!found_quad && poly[1][i].size() == 4) {
      found_quad = true;
      continue;
    } else {
      return false;
    }

    if (poly[1][i].size() != 3)
      return false;
  }

  if (poly[0].size() != 8)
    return false;

  return true;
}

template <int DIM>
static bool polytope_is_hexahedron(const std::vector<std::vector<mesh::Id>> poly[DIM])
{
  assert(DIM == 3);

  if (poly[2].size() != 1 || poly[2][0].size() != 8)
    return false;

  if (poly[1].size() != 6)
    return false;

  for (size_t i = 0; i < poly[1].size(); ++i) {
    if (poly[1][i].size() != 4)
      return false;
  }

  if (poly[0].size() != 12)
    return false;

  return true;
}

template <class DataType, int DIM>
void CellCutter<DataType, DIM>::try_determine_tag(const mesh::TDim tdim,
                                                  const std::vector<std::vector<mesh::Id>> poly[DIM],
                                                  mesh::CellType::Tag &tag)
{
  if (tdim == 2 && polytope_is_triangle<DIM>(poly)) 
  {
    tag = mesh::CellType::TRIANGLE;
  } 
  else if (tdim == 2 && polytope_is_quadriliteral<DIM>(poly)) 
  {
    tag = mesh::CellType::QUADRILATERAL;
  } 
  else if (tdim == 3 && polytope_is_tetrahedron<DIM>(poly)) 
  {
    tag = mesh::CellType::TETRAHEDRON;
  } 
  else if (tdim == 3 && polytope_is_hexahedron<DIM>(poly)) 
  {
    tag = mesh::CellType::HEXAHEDRON;
  } 
  else if (tdim == 3 && polytope_is_pyramid<DIM>(poly)) 
  {
    tag = mesh::CellType::PYRAMID;
  } 
  else 
  {
    tag = mesh::CellType::NOT_SET;
  }
}

static void edges_to_quad(const std::vector<std::vector<mesh::Id>>& edges,
                          std::vector<mesh::Id> &quad)
{
  quad.resize(4);

  quad[0] = edges[0][0];
  size_t ilastedge = 0;
  for (size_t i = 1; i < quad.size(); ++i) 
  {
    mesh::Id v = std::numeric_limits<mesh::Id>::min();

    for (size_t j = 0; j < edges.size(); ++j) 
    {
      if (j == ilastedge)
        continue;

      if (edges[j][0] == quad[i - 1]) 
      {
        v = edges[j][1];
        ilastedge = j;
        break;
      } 
      else if (edges[j][1] == quad[i - 1]) 
      {
        v = edges[j][0];
        ilastedge = j;
        break;
      }
    }
    assert(v != std::numeric_limits<mesh::Id>::min());

    quad[i] = v;
  }
}

template <int DIM>
static void polytope_to_triangle(const std::vector<std::vector<mesh::Id>> poly[DIM],
                                 std::vector<mesh::Id> &triangle)
{
  triangle.resize(3);
  std::copy(poly[1][0].begin(), poly[1][0].end(), triangle.begin());
}

template <int DIM>
static void polytope_to_quadriliteral(const std::vector<std::vector<mesh::Id>> poly[DIM],
                                      std::vector<mesh::Id> &quad)
{
  edges_to_quad(poly[0], quad);
}

template <int DIM>
static void polytope_to_tetrahedron(
    const std::vector<std::vector<mesh::Id>> poly[DIM],
    std::vector<mesh::Id> &tetra)
{
  tetra.resize(4);
  std::copy(poly[2][0].begin(), poly[2][0].end(), tetra.begin());
}

template <int DIM>
static void polytope_to_pyramid(const std::vector<std::vector<mesh::Id>> poly[DIM],
                                std::vector<mesh::Id> &pyramid)
{
  std::vector<mesh::Id> base;
  for (size_t i = 0; i < poly[1].size(); ++i) 
  {
    if (poly[1][i].size() == 4) 
    {
      base = poly[1][i];
      break;
    }
  }
  assert(base.size() != 0);

  mesh::Id vtip;
  for (size_t i = 0; i < poly[2][0].size(); ++i) 
  {
    if (std::find(base.begin(), base.end(), poly[2][0][i]) == base.end()) 
    {
      vtip = poly[2][0][i];
      break;
    }
  }
  pyramid.push_back(vtip);

  std::vector<std::vector<mesh::Id>> baseedges;
  for (size_t i = 0; i < poly[0].size(); ++i) 
  {
    if (std::find(poly[0][i].begin(), poly[0][i].end(), vtip) == poly[0][i].end()) 
    {
      baseedges.push_back(poly[0][i]);
    }
  }

  std::vector<mesh::Id> basequad;
  edges_to_quad(baseedges, basequad);

  pyramid.insert(pyramid.end(), basequad.begin(), basequad.end());
}

template <int DIM>
static void polytope_to_hexahedron(const std::vector<std::vector<mesh::Id>> poly[DIM],
                                   std::vector<mesh::Id> &hexa)
{
  std::vector<mesh::Id> quads[2];
  quads[0] = poly[1][0];
  for (size_t i = 0; i < poly[1].size(); ++i) 
  {
    bool intersect = false;
    for (size_t j = 0; j < poly[1][i].size(); ++j) 
    {
      if (std::find(quads[0].begin(), quads[0].end(), poly[1][i][j]) != quads[0].end()) 
      {
        intersect = true;
        break;
      }
    }
    if (!intersect) {
      quads[1] = poly[1][i];
      break;
    }
  }

  std::vector<std::vector<mesh::Id>> edges[2];
  for (size_t i = 0; i < poly[0].size(); ++i) 
  {
    int n[2];
    for (size_t j = 0; j < poly[0][i].size(); ++j) 
    {
      if (std::find(quads[0].begin(), quads[0].end(), poly[0][i][j]) != quads[0].end()) 
      {
        n[0] += 1;
      } 
      else if (std::find(quads[1].begin(), quads[1].end(), poly[1][i][j]) != quads[1].end()) 
      {
        n[1] += 1;
      }
    }
    if (n[0] == 2) 
    {
      edges[0].push_back(poly[0][i]);
    } 
    else if (n[1] == 2) 
    {
      edges[1].push_back(poly[0][i]);
    }
  }

  for (size_t i = 0; i < 2; ++i) 
  {
    std::vector<mesh::Id> quad;
    edges_to_quad(edges[i], quad);
    hexa.insert(hexa.end(), quad.begin(), quad.end());
  }
}

template <class DataType, int DIM>
void CellCutter<DataType, DIM>::polytope_to_cell(const mesh::CellType::Tag tag,
                                                 const std::vector<std::vector<mesh::Id>> poly[DIM],
                                                 std::vector<mesh::Id> &cell)
{
  switch (tag) 
  {
  case mesh::CellType::TRIANGLE:
    polytope_to_triangle<DIM>(poly, cell);
    break;
  case mesh::CellType::QUADRILATERAL:
    polytope_to_quadriliteral<DIM>(poly, cell);
    break;
  case mesh::CellType::TETRAHEDRON:
    polytope_to_tetrahedron<DIM>(poly, cell);
    break;
  case mesh::CellType::PYRAMID:
    polytope_to_pyramid<DIM>(poly, cell);
    break;
  case mesh::CellType::HEXAHEDRON:
    polytope_to_hexahedron<DIM>(poly, cell);
    break;
  default:
    assert(0);
  }
}

template <class DataType, int DIM>
void CellCutter<DataType, DIM>::compute_sub_cells(const mesh::TDim tdim,
                                                  const mesh::CellType::Tag prelim_tags[2],
                                                  const bool decompose_non_simplex,
                                                  std::vector<std::vector<mesh::Id>> &subcell_point_ids,
                                                  std::vector<mesh::CellType::Tag> &subcell_tags,
                                                  std::vector<DomainSide> &subcell_domains)
{
  subcell_point_ids.clear();
  subcell_tags.clear();
  subcell_domains.clear();

  for (int k = 0; k < 2; ++k)
  {
    if (subpolytopes_[k][tdim - 1].size() == 0)
      continue;

    bool simplex = prelim_tags[k] == mesh::CellType::TRIANGLE
      || prelim_tags[k] == mesh::CellType::TETRAHEDRON;

    if (prelim_tags[k] == mesh::CellType::NOT_SET || (!simplex && decompose_non_simplex))
    {
      mesh::Id anchor = subpolytopes_[k][tdim - 1][0][0];
      std::vector<std::vector<mesh::Id>> simplices;
      triangulate_convex_polytope<DIM>(tdim, subpolytopes_[k][tdim - 1][0], subpolytopes_[k], anchor, simplices);

      subcell_point_ids.insert(subcell_point_ids.end(), simplices.begin(), simplices.end());

      mesh::CellType::Tag tag = tdim == 3 ? mesh::CellType::TETRAHEDRON : mesh::CellType::TRIANGLE;
      std::vector<mesh::CellType::Tag> simplicetags(simplices.size(), tag);
      subcell_tags.insert(subcell_tags.end(), simplicetags.begin(), simplicetags.end());

      std::vector<DomainSide> simplicedoms(simplices.size(), k == 0 ? DomainSide::LOW_XI : DomainSide::HIGH_XI);
      subcell_domains.insert(subcell_domains.end(), simplicedoms.begin(), simplicedoms.end());
    }
    else
    {
      std::vector<mesh::Id> subcell;
      subcell.reserve(8);
      polytope_to_cell(prelim_tags[k], subpolytopes_[k], subcell);

      subcell_point_ids.push_back(subcell);
      subcell_tags.push_back(prelim_tags[k]);
      subcell_domains.push_back(k == 0 ? DomainSide::LOW_XI : DomainSide::HIGH_XI);
    }
  }
}

template <class DataType, int DIM>
void CellCutter<DataType, DIM>::get_cutfaces(const std::vector<std::vector<mesh::Id>> &subcell_point_ids,
                                             const std::vector<DomainSide> &subcell_domains,
                                             const std::unordered_map<mesh::Id, SubCellPointType>& point_types,
                                             std::vector<std::vector<mesh::Id>> &cutpoint_ids,
                                             std::vector<std::vector<VertexCoord>> &cutpoint_coords)
{
  cutpoint_ids.clear();
  std::vector<mesh::Id> cur_cutpoint_ids;
  for (size_t i = 0; i < subcell_point_ids.size(); ++i) 
  {
    if (subcell_domains[i] != DomainSide::LOW_XI)
    {
      continue;
    }

    cur_cutpoint_ids.clear();
    for (size_t j = 0, e_j = subcell_point_ids[i].size(); j != e_j; ++j) 
    {
      const auto point_id = subcell_point_ids[i][j];
      assert (point_types.find(point_id) != point_types.end());
      if (point_types.at(point_id) == SubCellPointType::CutPoint)
      {
        cur_cutpoint_ids.push_back(point_id);
      }
      //if (subcell_point_ids[i][j] < 0)
      //  cur_cutpoint_ids.push_back(subcell_point_ids[i][j]);
    
    }

    if (cur_cutpoint_ids.size() >= DIM)
    {
      cutpoint_ids.push_back(cur_cutpoint_ids);
    }
  }

  cutpoint_coords.resize(cutpoint_ids.size());
  for (size_t i = 0, e_i = cutpoint_ids.size(); i != e_i; ++i) 
  {
    cutpoint_coords[i].resize(cutpoint_ids[i].size());
    for (size_t j = 0, e_j = cutpoint_ids[i].size(); j != e_j; ++j) 
    {
      cutpoint_coords[i][j] = cutpoint_coords_[cutpoint_ids[i][j]];
    }
  }
}

template <class DataType, int DIM>
void CellCutter<DataType, DIM>::get_subcell_coordinates(const mesh::Entity &cell,
                                                        const std::unordered_map<mesh::Id, VertexCoord> &cutpoint_coords,
                                                        const std::vector<std::vector<mesh::Id>> &subcell_point_ids,
                                                        std::vector<std::vector<VertexCoord>> &subcell_point_coords)
{
  subcell_point_coords.resize(subcell_point_ids.size());
  std::vector<VertexCoord> coords;
  std::vector<mesh::Coordinate> c;
  std::vector<VertexCoord> points;

  for (size_t i = 0; i < subcell_point_ids.size(); ++i) 
  {
    coords.clear();
    coords.resize(subcell_point_ids[i].size());

    for (size_t j = 0; j < subcell_point_ids[i].size(); ++j) 
    {
      mesh::Id id = subcell_point_ids[i][j];
      if (id < 0) 
      {
        coords[j] = cutpoint_coords.at(id);
      } 
      else 
      {
        mesh::VertexIdIterator it = std::find(cell.begin_vertex_ids(), cell.end_vertex_ids(), id);
        int vertnum = it - cell.begin_vertex_ids();

        c.clear();
        cell.get_coordinates(c, vertnum);

        points.clear();
        interlaced_coord_to_points<mesh::Coordinate, DataType, DIM>(c, points);
        coords[j] = points[0];
      }
    }

    subcell_point_coords[i] = coords;
  }
}

template <class DataType, int DIM>
CellCutter<DataType, DIM>::CellCutter() {}

template class CellCutter <double, 2>;
template class CellCutter <double, 3>;
template class CellCutter <float, 2>;
template class CellCutter <float, 3>;

} // namespace doffem
} // namespace hiflow
