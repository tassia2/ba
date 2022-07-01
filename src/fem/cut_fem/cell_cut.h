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

#ifndef HIFLOW_CUTFEM_CELLCUT_H_
#define HIFLOW_CUTFEM_CELLCUT_H_

#include "common/log.h"
#include "common/vector_algebra_descriptor.h"
#include "common/array_tools.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/cut_fem/triangulation.h"
#include "mesh/cell_type.h"
#include "mesh/entity.h"
#include "mesh/geometric_tools.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "space/vector_space.h"

#include <map>
#include <cmath>
#include <vector>
#include <numeric>
#include <set>
#include <unordered_set>
#include <unordered_map>

#define nCUT_DBG_OUT

#ifdef CUT_DBG_OUT
const static int CELLCUT_DBG = 0;
#else
const static int CELLCUT_DBG = 3;
#endif

namespace hiflow {
namespace doffem {
                     
enum class DomainSide 
{
  NOT_SET = 0,
  LOW_XI = 1,
  HIGH_XI = 2
};

enum class CutType 
{
  Pathologic = -1,
  Proper = 0,
  Interface = 1,
  Edge = 2,
  Point = 3,
  None = 4 
};

enum class SubCellPointType {
  CutPoint = 0,
  NoCutPoint = 1
};

template <class DataType, int DIM>
class CellCutter
{
public: 
  typedef mesh::EntityNumber VertexIndex;
  using VertexCoord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  CellCutter();
  
  virtual ~CellCutter() {}

  // TODO: subcell_domains <-> int 
  template <class DomainEvaluator>
  CutType cut_cell(const mesh::Entity &cell, 
                   const DomainEvaluator *domain_eval,
                   const DataType domain_treshold, 
                   const DataType search_tol,
                   const bool decompose_non_simplex, 
                   std::vector<std::vector<mesh::Id>> &cutpoint_ids,
                   std::vector<std::vector<VertexCoord>> &cutpoint_coords, 
                   std::vector<DomainSide> &subcell_domains,
                   std::vector<mesh::CellType::Tag> &subcell_tags,
                   std::vector<std::vector<mesh::Id>> &subcell_point_ids,
                   std::vector<std::vector<VertexCoord>> &subcell_point_coords,
                   std::unordered_map<mesh::Id, SubCellPointType>& vertex_types);

private:

  template <class DomainEvaluator>
  void compute_cut(const mesh::Entity &edge, 
                   const mesh::Entity &cell,
                   const DomainEvaluator *domeval, 
                   const DataType domtresh, 
                   const DataType searchtol,
                   mesh::Id &lastcutid, 
                   std::vector<mesh::Id> &cutids, 
                   std::vector<VertexCoord> &cutpoint,
                   std::vector< std::vector<int> >& doms);

  template <class DomainEvaluator>
  void compute_entity_cuts(const mesh::Entity &entity, 
                           const mesh::Entity &cell,
                           const DomainEvaluator *domeval, 
                           const DataType domtresh, 
                           const DataType searchtol,
                           int &lastcutid, 
                           std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>> cutpoint_ids[DIM],
                           std::unordered_map<mesh::Id, VertexCoord> &cutpoint_coords,
                           std::unordered_map<mesh::Id, std::vector<int>> &domains);

  CutType check_cut(const std::vector<mesh::Id>& cutface,
                    const std::vector<VertexCoord>& cutpts, 
                    const mesh::TDim tdim,
                    bool &decompose_non_simplex);

// TODO: check for &
  void compute_entity_partitions(const mesh::Entity &entity,
                                 const std::unordered_map<mesh::Id, std::vector<int> >& domains,
                                 const std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>> cutpoint_ids[DIM],
                                 std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>[2]> partitions[DIM],
                                 std::unordered_map<mesh::Id, SubCellPointType>& point_types);

  void assemble_sub_polytopes( std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>[2]> partitions[DIM],
                               std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>> cutpoint_ids[DIM],
                               std::vector<std::vector<mesh::Id>> (&subpolys)[2][DIM]);

  void try_determine_tag(const mesh::TDim tdim,
                         const std::vector<std::vector<mesh::Id>> poly[DIM], 
                         mesh::CellType::Tag &tag);

  void polytope_to_cell(const mesh::CellType::Tag tag,
                        const std::vector<std::vector<mesh::Id>> poly[DIM],
                        std::vector<mesh::Id> &cell);

  void compute_sub_cells(const mesh::TDim tdim,
                         const mesh::CellType::Tag tags[2],
                         const bool decompose_non_simplex,
                         std::vector<std::vector<mesh::Id>> &subcell_point_ids,
                         std::vector<mesh::CellType::Tag> &subcell_tags,
                         std::vector<DomainSide> &subcell_domains);

  void get_cutfaces(const std::vector<std::vector<mesh::Id>> &subcell_point_ids,
                    const std::vector<DomainSide> &subcell_domains,
                    const std::unordered_map<mesh::Id, SubCellPointType>& point_types,
                    std::vector<std::vector<mesh::Id>> &cutpoint_ids,
                    std::vector<std::vector<VertexCoord>> &cutpts);

  void get_subcell_coordinates(const mesh::Entity &cell,
                               const std::unordered_map<mesh::Id, VertexCoord> &cutpoint_coords,
                               const std::vector<std::vector<mesh::Id>> &subcell,
                               std::vector<std::vector<VertexCoord>> &subcell_point_coords);

  std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>> cutpoint_ids_[DIM];
  std::unordered_map<mesh::Id, VertexCoord > cutpoint_coords_;
  std::unordered_map<mesh::Id, std::vector<int> > domains_;
  std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>[2]> partitions_[DIM];
  std::vector<std::vector<mesh::Id>> subpolytopes_[2][DIM];

};

template <class DataType, int DIM>
template <class DomainEvaluator>
void CellCutter<DataType, DIM>::compute_cut(const mesh::Entity &edge, 
                                            const mesh::Entity &cell,
                                            const DomainEvaluator *domeval, 
                                            const DataType domtresh, 
                                            const DataType searchtol,
                                            mesh::Id &lastcutid, 
                                            std::vector<mesh::Id> &cutpoint_ids, 
                                            std::vector<VertexCoord> &cutpoint_coords,
                                            std::vector< std::vector<int> > & doms)
{
  // output:
  // lastcutid: last set/used cut id in this function
  // cutids: id's of existing (positive index) or new (negative index taken from lastcutid) cut
  //    vertices
  // cutpoint: The coordinates if a new cut vertex is required
  // doms: List of domains the vertices occupy; only two for cut vertices

  std::vector<DataType> coords;
  edge.get_coordinates(coords);

  VertexCoord p[2];
  for (int i = 0; i < 2; ++i) 
  {
    std::vector<DataType> v(coords.begin() + i * DIM, coords.begin() + (i + 1) * DIM);
    p[i] = VertexCoord(v);
  }

  DataType xi[2];
  xi[0] = domeval->evaluate(cell, p[0]);
  xi[1] = domeval->evaluate(cell, p[1]);

  // determine domains
  assert(doms.size() == 2);
  for (int i = 0; i < 2; ++i) 
  {
    if (xi[i] <= domtresh)
      doms[i].push_back(0);
    if (xi[i] > domtresh)
      doms[i].push_back(1);
  }

  // set ids of cut vertices
  assert(doms[0].size() >= 1 && doms[0].size() <= 2
      && doms[1].size() >= 1 && doms[1].size() <= 2);
  
  bool vertexcut = false;
  for (int k = 0; k < 2; ++k) 
  {
    if (doms[k].size() == 2) 
    { // vertex cut at k
      cutpoint_ids.push_back(edge.vertex_id(k));
      
      std::vector<mesh::Coordinate> c;
      edge.get_coordinates(c, k);

      VertexCoord p;
      for (size_t i = 0; i < c.size(); ++i) 
      {
        p.set(i, static_cast<DataType>(c[i]));
      }
      cutpoint_coords.push_back(p);
      vertexcut = true;
    }
  }
  if (vertexcut)
    return;

  if (doms[0][0] == doms[1][0]) // uncut edge
    return;

  // remaining case: edge with simple cut
  cutpoint_ids.push_back(--lastcutid);

  // compute coordinates of cut
  int dom1 = doms[0][0];
  int dom2 = doms[1][0];

  VertexCoord pcut = 0.5 * (p[0] + p[1]);
  const DataType edgelen = distance(p[0], p[1]);
  DataType xic = domeval->evaluate(cell, pcut);
  while (distance(p[0], p[1]) > searchtol * edgelen) 
  {
    int domc = xic > domtresh;
    if (domc == dom1) {
      p[0] = pcut;
    } else {
      p[1] = pcut;
    }
    pcut = 0.5 * (p[0] + p[1]);

    xic = domeval->evaluate(cell, pcut);
  }
  cutpoint_coords.push_back(pcut);
}

template <class DataType, int DIM>
template <class DomainEvaluator>
void CellCutter<DataType, DIM>::compute_entity_cuts(const mesh::Entity &entity,
                                                    const mesh::Entity &cell, 
                                                    const DomainEvaluator *domeval,
                                                    const DataType domtresh, 
                                                    const DataType searchtol, 
                                                    int &lastcutid,
                                                    std::unordered_map<mesh::EntityNumber, std::vector<mesh::Id>> cutpoint_ids[DIM],
                                                    std::unordered_map<mesh::Id, VertexCoord> &cutpoint_coords,
                                                    std::unordered_map<mesh::Id, std::vector<int>> &domains)
{
  const mesh::TDim tdim = entity.tdim() - 1;
  if (tdim == 0) 
  {
    // entity is an edge
    if (cutpoint_ids[tdim][entity.index()].size() > 0)
      return;

    std::vector<mesh::Id> cutids;
    std::vector<VertexCoord> tmp_cutpoint_coords;
    std::vector< std::vector<int> > doms(2);
    compute_cut<DomainEvaluator>(entity, cell, 
                                 domeval, domtresh, searchtol, 
                                 lastcutid, cutids, tmp_cutpoint_coords, doms);

    cutpoint_ids[tdim][entity.index()] = cutids;

    for (size_t i = 0; i < cutids.size(); ++i) {
      cutpoint_coords[cutids[i]] = tmp_cutpoint_coords[i];
    }

    mesh::VertexIdIterator v = entity.begin_vertex_ids();
    int idxs[2] = { *(v++), *v };
    domains[idxs[0]] = doms[0];
    domains[idxs[1]] = doms[1];
    return;
  }

  // loop over lower dimensional subentities and cut them
  std::set<mesh::Id> facet;
  for (mesh::IncidentEntityIterator it = entity.begin_incident(tdim);
      it != entity.end_incident(tdim); ++it) 
  {
    mesh::Entity face = *it;
    mesh::EntityNumber e = face.index();

    compute_entity_cuts<DomainEvaluator>(face, cell,
                                         domeval, domtresh, searchtol, 
                                         lastcutid, cutpoint_ids, cutpoint_coords, domains);

    facet.insert(cutpoint_ids[tdim - 1][e].begin(), cutpoint_ids[tdim - 1][e].end());
  }

  cutpoint_ids[tdim][entity.index()] = std::vector<mesh::Id>(facet.begin(), facet.end());
}


// cut_type:  0 :  proper cut, 2 subcells with positive volume
//            1 :  cut plane coincides with cell facet
//            2 :  single cut point or cut edge (in 3D)
//           -1 :  pathological case, perhaps multiple cuts per cell
template <class DataType, int DIM>
template <class DomainEvaluator>
CutType CellCutter<DataType, DIM>::cut_cell(const mesh::Entity &cell, 
                                            const DomainEvaluator *domain_eval,
                                            const DataType domain_treshold, 
                                            const DataType search_tol,
                                            bool decompose_non_simplex, 
                                            std::vector<std::vector<mesh::Id>> &cutpoint_ids,
                                            std::vector<std::vector<VertexCoord>> &cutpoint_coords, 
                                            std::vector<DomainSide> &subcell_domains,
                                            std::vector<mesh::CellType::Tag> &subcell_tags,
                                            std::vector<std::vector<mesh::Id>> &subcell_point_ids,
                                            std::vector<std::vector<VertexCoord>> &subcell_point_coords,
                                            std::unordered_map<mesh::Id, SubCellPointType>& vertex_types)
{
  assert(DIM == 2 || DIM == 3);

  for (int i = 0; i < DIM; ++i) 
  {
    cutpoint_ids_[i].clear();
  }
  cutpoint_coords.clear();
  domains_.clear();

  for (int i = 0; i < DIM; ++i) 
  {
    partitions_[i].clear();
  }
  for (int k = 0; k < 2; ++k) 
  {
    for (int i = 0; i < DIM; ++i) 
    {
      subpolytopes_[k][i].clear();
    }
  }

  const mesh::TDim tdim = cell.tdim();

  int lastcutid = 0;
  compute_entity_cuts<DomainEvaluator>(cell, cell, 
                                       domain_eval, domain_treshold, search_tol, 
                                       lastcutid, cutpoint_ids_, cutpoint_coords_, domains_);
  assert(cutpoint_ids_[cell.tdim() - 1].size() == 1);

  std::vector<mesh::Id> cutids = cutpoint_ids_[cell.tdim() - 1][cell.index()];
  std::vector<VertexCoord> tmp_cutpoint_coords(cutids.size());
  for (size_t i = 0; i < cutids.size(); ++i)
  {
    tmp_cutpoint_coords[i] = cutpoint_coords_[cutids[i]];
  }

  CutType cuttype = check_cut(cutids, tmp_cutpoint_coords, tdim, decompose_non_simplex);
  if (cuttype != CutType::Proper)
  {
    return cuttype;
  }

  vertex_types.clear();
  compute_entity_partitions(cell, domains_, cutpoint_ids_, partitions_, vertex_types);

  assemble_sub_polytopes(partitions_, cutpoint_ids_, subpolytopes_);

  mesh::CellType::Tag tags[2];
  try_determine_tag(tdim, subpolytopes_[0], tags[0]);
  try_determine_tag(tdim, subpolytopes_[1], tags[1]);

  compute_sub_cells(tdim, tags, decompose_non_simplex, subcell_point_ids, subcell_tags, subcell_domains);

  std::vector<std::vector<VertexCoord>> cur_subcell_coords;
  get_subcell_coordinates(cell, cutpoint_coords_, subcell_point_ids, cur_subcell_coords);
  subcell_point_coords.clear();
  subcell_point_coords.insert(subcell_point_coords.end(), cur_subcell_coords.begin(), cur_subcell_coords.end());

  get_cutfaces(subcell_point_ids, subcell_domains, vertex_types, cutpoint_ids, cutpoint_coords);

  assert (subcell_point_ids.size() > 0);
  return CutType::Proper;
}

} // namespace doffem
} // namespace hiflow

#endif
