#include <algorithm>
#include <cassert>
#include <iostream>
#include <stddef.h>
#include <vector>

#include "fem/cut_fem/triangulation.h"

namespace hiflow {
namespace doffem {

static void polytope_get_subfaces(const std::vector<mesh::Id> &face,
                                  const std::vector<std::vector<int>> &faces,
                                  std::vector<std::vector<int>> &subfaces)
{
  for (size_t i = 0; i < faces.size(); ++i) 
  {
    bool subface = true;
    for (size_t j = 0; j < faces[i].size(); ++j) 
    {
      if (std::find(face.begin(), face.end(), faces[i][j]) == face.end()) {
        subface = false;
        break;
      }
    }
    if (subface)
      subfaces.push_back(faces[i]);
  }
}

template <int DIM>
void triangulate_convex_polytope(const mesh::TDim tdim,
                                 const std::vector<mesh::Id>& entity,
                                 const std::vector<std::vector<mesh::Id>> faces[DIM],
                                 const mesh::Id anchor, 
                                 std::vector<std::vector<mesh::Id>> &simplices)
{
  if (tdim == 1) {
    simplices.push_back(entity);
    return;
  }

  const mesh::TDim subdim = tdim - 1;
  std::vector<std::vector<mesh::Id>> subsimplices;
  std::vector<std::vector<mesh::Id>> subfaces;
  polytope_get_subfaces(entity, faces[subdim - 1], subfaces);
  for (size_t i = 0; i < subfaces.size(); ++i) {
    std::vector<mesh::Id> face = subfaces[i];

    int a = anchor;
    if (std::find(face.begin(), face.end(), anchor) == face.end())
      a = face[0];

    triangulate_convex_polytope<DIM>(tdim - 1, face, faces, a, subsimplices);
  }

  for (size_t i = 0; i < subsimplices.size(); ++i) 
  {
    if (std::find(subsimplices[i].begin(), subsimplices[i].end(), anchor) != subsimplices[i].end())
      continue;

    std::vector<mesh::Id> simplex;
    simplex.push_back(anchor);
    simplex.insert(simplex.end(), subsimplices[i].begin(), subsimplices[i].end());
    simplices.push_back(simplex);
  }
}

template void triangulate_convex_polytope<2>(const mesh::TDim tdim,
                                             const std::vector<mesh::Id>& entity,
                                             const std::vector<std::vector<mesh::Id>> faces[2],
                                             const mesh::Id anchor, 
                                             std::vector<std::vector<mesh::Id>> &simplices);
template void triangulate_convex_polytope<3>(const mesh::TDim tdim,
                                             const std::vector<mesh::Id>& entity,
                                             const std::vector<std::vector<mesh::Id>> faces[3],
                                             const mesh::Id anchor, 
                                             std::vector<std::vector<mesh::Id>> &simplices);

} // namespace doffem
} // namespace hiflow
