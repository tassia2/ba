// Copyright (C) 2011-2017 Vincent Heuveline
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

/// \author Staffan Ronnas, Simon Gawlok

#ifndef _STANDARD_ASSEMBLY_H_
#define _STANDARD_ASSEMBLY_H_

#include <vector>
#include "assembly/assembly_utils.h"
#include "assembly/assembly_routines.h"
#include "assembly/assembly_routines_deprecated.h"
#include "assembly/global_assembler_deprecated.h"
#include "assembly/generic_assembly_algorithm.h"
#include "common/pointers.h"
#include "common/array_tools.h"
#include "mesh/attributes.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "space/element.h"
#include "space/vector_space.h"

namespace hiflow {

//////////////// Implementation of StandardGlobalAssembler ////////////////
template < class DataType, int DIM >
class StandardGlobalAssembler : public GlobalAssembler< DataType, DIM > 
{
  typedef GlobalAssembler< DataType, DIM > GlobalAsm;
  typedef VectorSpace< DataType, DIM > VecSpace;

  virtual void assemble_scalar_impl(const VecSpace &space,
                                    typename GlobalAsm::ScalarAssemblyFunction local_asm,
                                    std::vector< DataType > &vec,
                                    typename GlobalAsm::QuadratureSelectionFunction q_select) const;

  virtual void assemble_multiple_scalar_impl(const VecSpace &space,
                                             typename GlobalAsm::MultipleScalarAssemblyFunction local_asm,
                                             int num_scalars, 
                                             std::vector< std::vector< DataType > > &vec,
                                             typename GlobalAsm::QuadratureSelectionFunction q_select) const;

  virtual void assemble_vector_impl(const VecSpace &space,
                                    typename GlobalAsm::VectorAssemblyFunction local_asm,
                                    typename GlobalAsm::GlobalVector &vec,
                                    typename GlobalAsm::QuadratureSelectionFunction q_select) const;

  virtual void assemble_matrix_impl(const VecSpace &space,
                                    typename GlobalAsm::MatrixAssemblyFunction local_asm,
                                    typename GlobalAsm::GlobalMatrix &mat,
                                    typename GlobalAsm::QuadratureSelectionFunction q_select) const;

  virtual void assemble_scalar_boundary_impl(const VecSpace &space,
                                             typename GlobalAsm::BoundaryScalarAssemblyFunction local_asm,
                                             std::vector< DataType > &vec,
                                             typename GlobalAsm::FacetQuadratureSelectionFunction fq_select) const;

  virtual void assemble_multiple_scalar_boundary_impl(const VecSpace &space,
                                                      typename GlobalAsm::BoundaryMultipleScalarAssemblyFunction local_asm,
                                                      int num_scalars, 
                                                      std::vector< std::vector< DataType > > &vec,
                                                      typename GlobalAsm::FacetQuadratureSelectionFunction fq_select) const;
                                                     
  virtual void assemble_vector_boundary_impl(const VecSpace &space,
                                             typename GlobalAsm::BoundaryVectorAssemblyFunction local_asm,
                                             typename GlobalAsm::GlobalVector &vec,
                                             typename GlobalAsm::FacetQuadratureSelectionFunction fq_select) const;

  virtual void assemble_matrix_boundary_impl(const VecSpace &space,
                                             typename GlobalAsm::BoundaryMatrixAssemblyFunction local_asm,
                                             typename GlobalAsm::GlobalMatrix &mat,
                                             typename GlobalAsm::FacetQuadratureSelectionFunction fq_select) const;
};

template < class DataType, int DIM >
void StandardGlobalAssembler< DataType, DIM >::assemble_scalar_impl(const VecSpace &space,
                                                                    typename GlobalAsm::ScalarAssemblyFunction local_asm,
                                                                    std::vector< DataType > &vec,
                                                                    typename GlobalAsm::QuadratureSelectionFunction q_select) const 
{
  StandardScalarAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, q_select);
}

template < class DataType, int DIM >
void StandardGlobalAssembler< DataType, DIM >::assemble_multiple_scalar_impl(const VecSpace &space,
                                                                             typename GlobalAsm::MultipleScalarAssemblyFunction local_asm,
                                                                             const int num_scalars, std::vector< std::vector< DataType > > &vec,
                                                                             typename GlobalAsm::QuadratureSelectionFunction q_select) const 
{
  StandardMultipleScalarAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, vec, num_scalars);
  assembly.assemble(local_asm, q_select);
}

template < class DataType, int DIM >
void StandardGlobalAssembler< DataType, DIM >::assemble_vector_impl(const VecSpace &space,
                                                                    typename GlobalAsm::VectorAssemblyFunction local_asm,
                                                                    typename GlobalAsm::GlobalVector &vec,
                                                                    typename GlobalAsm::QuadratureSelectionFunction q_select) const 
{
  StandardVectorAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, q_select);
}

template < class DataType, int DIM >
void StandardGlobalAssembler< DataType, DIM >::assemble_matrix_impl(const VecSpace &space,
                                                                    typename GlobalAsm::MatrixAssemblyFunction local_asm,
                                                                    typename GlobalAsm::GlobalMatrix &mat,
                                                                    typename GlobalAsm::QuadratureSelectionFunction q_select) const 
{
  StandardMatrixAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, mat);
  assembly.assemble(local_asm, q_select);
}

template < class DataType, int DIM >
void StandardGlobalAssembler< DataType, DIM >::assemble_scalar_boundary_impl(const VecSpace &space,
                                                                             typename GlobalAsm::BoundaryScalarAssemblyFunction local_asm,
                                                                             std::vector< DataType > &vec,
                                                                             typename GlobalAsm::FacetQuadratureSelectionFunction fq_select) const 
{
  // TODO: how should vec be defined -> all facets or only boundary facets?
  // If the latter, then what ordering should be used?
  StandardBoundaryScalarAssembly< BoundaryAssemblyAlgorithm, DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, fq_select);
}

template < class DataType, int DIM >
void StandardGlobalAssembler< DataType, DIM >::assemble_multiple_scalar_boundary_impl(const VecSpace &space,
                                                                                      typename GlobalAsm::BoundaryMultipleScalarAssemblyFunction local_asm,
                                                                                      const int num_scalars,
                                                                                      std::vector< std::vector<DataType> > &vec,
                                                                                      typename GlobalAsm::FacetQuadratureSelectionFunction fq_select) const 
{
  // TODO: how should vec be defined -> all facets or only boundary facets?
  // If the latter, then what ordering should be used?
  StandardBoundaryMultipleScalarAssembly< BoundaryAssemblyAlgorithm, DataType, DIM > assembly(space, vec, num_scalars);
  assembly.assemble(local_asm, fq_select);
}

template < class DataType, int DIM >
void StandardGlobalAssembler< DataType, DIM >::assemble_vector_boundary_impl(const VecSpace &space,
                                                                             typename GlobalAsm::BoundaryVectorAssemblyFunction local_asm,
                                                                             typename GlobalAsm::GlobalVector &vec,
                                                                             typename GlobalAsm::FacetQuadratureSelectionFunction fq_select) const 
{
  StandardVectorAssembly< BoundaryAssemblyAlgorithm, DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, fq_select);
}

template < class DataType, int DIM >
void StandardGlobalAssembler< DataType, DIM >::assemble_matrix_boundary_impl(const VecSpace &space,
                                                                             typename GlobalAsm::BoundaryMatrixAssemblyFunction local_asm,
                                                                             typename GlobalAsm::GlobalMatrix &mat,
                                                                             typename GlobalAsm::FacetQuadratureSelectionFunction fq_select) const 
{
  StandardMatrixAssembly< BoundaryAssemblyAlgorithm, DataType, DIM > assembly(space, mat);
  assembly.assemble(local_asm, fq_select);
}

} // namespace hiflow
#endif
