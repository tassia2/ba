#ifndef HIFLOW_CUTFEM_TRIANGULATION_H_
#define HIFLOW_CUTFEM_TRIANGULATION_H_

#include <vector>

#include "mesh/types.h"

namespace hiflow {
namespace doffem {

template <int DIM>
void triangulate_convex_polytope(const mesh::TDim tdim,
                                 const std::vector<mesh::Id>& entity,
                                 const std::vector<std::vector<mesh::Id>> faces[DIM],
                                 const mesh::Id anchor, 
                                 std::vector<std::vector<mesh::Id>> &simplices);

} // namespace doffem
} // namespace hiflow

#endif // HIFLOW_CUTFEM_TRIANGULATION_H
