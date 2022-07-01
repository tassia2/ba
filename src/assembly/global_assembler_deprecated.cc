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

#include "assembly/global_assembler_deprecated.h"
#include "assembly/quadrature_selection.h"
#include "space/element.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/matrix.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "linear_algebra/vector.h"
#include "quadrature/quadrature.h"
#include "dof/dof_interpolation.h"
#include "mesh/mesh.h"
#include "space/vector_space.h"


namespace hiflow {

template < class DataType, int DIM >
GlobalAssembler< DataType, DIM >::GlobalAssembler()
    : q_select_(default_select_), fq_select_(default_facet_select_),
      should_reset_assembly_target_(true) {}

template < class DataType, int DIM > 
GlobalAssembler< DataType, DIM >::~GlobalAssembler() {}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::integrate_scalar(const VecSpace &space, 
                                                        ScalarAssemblyFunction local_asm,
                                                        DataType &integral) const 
{
  const size_t num_cells = space.mesh().num_entities(space.mesh().tdim());
  std::vector< DataType > cell_values(num_cells, 0.);

  assemble_scalar_impl(space, local_asm, cell_values, q_select_);
  integral = std::accumulate(cell_values.begin(), cell_values.end(), 0.);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::integrate_multiple_scalar(const VecSpace &space,
                                                                 MultipleScalarAssemblyFunction local_asm, 
                                                                 const int num_scalars,
                                                                 std::vector< DataType > &integral) const 
{
  const size_t num_cells = space.mesh().num_entities(space.mesh().tdim());
  std::vector< std::vector< DataType > > cell_values(num_cells);
  for (size_t l = 0; l < num_cells; ++l) {
    cell_values[l].resize(num_scalars, 0.);
  }

  assemble_multiple_scalar_impl(space, local_asm, num_scalars, cell_values, q_select_);
  integral.clear();
  integral.resize(num_scalars, 0.);

  for (size_t i = 0; i < num_cells; ++i) 
  {
    for (size_t l = 0; l < num_scalars; ++l) 
    {
      integral[l] += cell_values[i][l];
    }
  }
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::assemble_scalar(const VecSpace &space, 
                                                       ScalarAssemblyFunction local_asm,
                                                       std::vector< DataType > &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.clear();
    const size_t num_cells = space.mesh().num_entities(space.mesh().tdim());
    vec.resize(num_cells, 0.);
  }

  assemble_scalar_impl(space, local_asm, vec, q_select_);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::assemble_multiple_scalar(const VecSpace &space,
                                                                MultipleScalarAssemblyFunction local_asm, 
                                                                const int num_scalars,
                                                                std::vector< std::vector< DataType > > &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.clear();
    const size_t num_cells = space.mesh().num_entities(space.mesh().tdim());
    vec.resize(num_cells);
    for (size_t l = 0; l < num_cells; ++l) 
    {
      vec[l].resize(num_scalars, 0.);
    }
  }

  assemble_multiple_scalar_impl(space, local_asm, num_scalars, vec, q_select_);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::assemble_vector(const VecSpace &space, 
                                                       VectorAssemblyFunction local_asm,
                                                       GlobalVector &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.Zeros();
  }

  assemble_vector_impl(space, local_asm, vec, q_select_);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::assemble_matrix(const VecSpace &space, 
                                                       MatrixAssemblyFunction local_asm,
                                                       GlobalMatrix &matrix) const 
{
  if (should_reset_assembly_target_) 
  {
    matrix.Zeros();
  }

  assemble_matrix_impl(space, local_asm, matrix, q_select_);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::integrate_scalar_boundary(const VecSpace &space,
                                                                 BoundaryScalarAssemblyFunction local_asm, 
                                                                 DataType &integral) const 
{
  // TODO: actually the number of boundary facets is needed
  const size_t num_facets = space.mesh().num_entities(space.mesh().tdim() - 1);
  std::vector< DataType > cell_values(num_facets, 0.);

  assemble_scalar_boundary_impl(space, local_asm, cell_values, fq_select_);
  integral = std::accumulate(cell_values.begin(), cell_values.end(), 0.);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::integrate_multiple_scalar_boundary(const VecSpace &space,
                                                                          BoundaryMultipleScalarAssemblyFunction local_asm, 
                                                                          const int num_scalars,
                                                                          std::vector< DataType > &integral) const 
{
  const size_t num_facets = space.mesh().num_entities(space.mesh().tdim() - 1);
  std::vector< std::vector<DataType> > cell_values(num_facets);
  
  for (size_t l = 0; l < num_facets; ++l) 
  {
    cell_values[l].resize(num_scalars, 0.);
  }

  assemble_multiple_scalar_boundary_impl(space, local_asm, num_scalars, cell_values, fq_select_);
  
  integral.clear();
  integral.resize(num_scalars, 0.);

  for (size_t i = 0; i < num_facets; ++i) 
  {
    for (size_t l = 0; l < num_scalars; ++l) 
    {
      integral[l] += cell_values[i][l];
    }
  }
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::maximize_scalar_boundary(const VecSpace &space,
                                                                BoundaryScalarAssemblyFunction local_asm, 
                                                                DataType &maximum) const 
{
  // TODO: actually the number of boundary facets is needed
  const size_t num_facets = space.mesh().num_entities(space.mesh().tdim() - 1);
  std::vector< DataType > cell_values(num_facets, 0.);

  assemble_scalar_boundary_impl(space, local_asm, cell_values, fq_select_);
  maximum = *std::max_element(cell_values.begin(), cell_values.end());
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::assemble_scalar_boundary(const VecSpace &space,
                                                                BoundaryScalarAssemblyFunction local_asm,
                                                                std::vector< DataType > &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.clear();
    const size_t num_cells = space.mesh().num_entities(space.mesh().tdim() - 1);
    vec.resize(num_cells, 0.);
  }

  assemble_scalar_boundary_impl(space, local_asm, vec, fq_select_);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::assemble_multiple_scalar_boundary(const VecSpace &space,
                                                                         BoundaryMultipleScalarAssemblyFunction local_asm,
                                                                         const int num_scalars,
                                                                         std::vector< std::vector< DataType > > &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.clear();
    const size_t num_facets = space.mesh().num_entities(space.mesh().tdim() - 1);
    vec.resize(num_facets);
    for (size_t l = 0; l < num_facets; ++l) 
    {
      vec[l].resize(num_scalars, 0.);
    }
  }

  assemble_multiple_scalar_boundary_impl(space, local_asm, num_scalars, vec, fq_select_);
}


template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::assemble_vector_boundary(const VecSpace &space,
                                                                BoundaryVectorAssemblyFunction local_asm, 
                                                                GlobalVector &vec) const 
{
  if (should_reset_assembly_target_) {
    vec.Zeros();
  }

  assemble_vector_boundary_impl(space, local_asm, vec, fq_select_);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::assemble_matrix_boundary(const VecSpace &space,
                                                                BoundaryMatrixAssemblyFunction local_asm, 
                                                                GlobalMatrix &matrix) const 
{
  if (should_reset_assembly_target_) {
    matrix.Zeros();
  }

  assemble_matrix_boundary_impl(space, local_asm, matrix, fq_select_);
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::set_quadrature_selection_function(QuadratureSelectionFunction q_select) {
  if (q_select != 0) {
    q_select_ = q_select;
  } else {
    q_select_ = default_select_;
  }
}

template < class DataType, int DIM >
void GlobalAssembler< DataType, DIM >::should_reset_assembly_target(bool should_reset) 
{
  should_reset_assembly_target_ = should_reset;
}

template <>
const GlobalAssembler< double, 3 >::QuadratureSelectionFunction
    GlobalAssembler< double, 3 >::default_select_ =
        DefaultQuadratureSelection< double, 3 >();
template <>
const GlobalAssembler< double, 3 >::FacetQuadratureSelectionFunction
    GlobalAssembler< double, 3 >::default_facet_select_ =
        DefaultFacetQuadratureSelection< double, 3 >();
template <>
const GlobalAssembler< double, 2 >::QuadratureSelectionFunction
    GlobalAssembler< double, 2 >::default_select_ =
        DefaultQuadratureSelection< double, 2 >();
template <>
const GlobalAssembler< double, 2 >::FacetQuadratureSelectionFunction
    GlobalAssembler< double, 2 >::default_facet_select_ =
        DefaultFacetQuadratureSelection< double, 2 >();
template <>
const GlobalAssembler< double, 1 >::QuadratureSelectionFunction
    GlobalAssembler< double, 1 >::default_select_ =
        DefaultQuadratureSelection< double, 1 >();
template <>
const GlobalAssembler< double, 1 >::FacetQuadratureSelectionFunction
    GlobalAssembler< double, 1 >::default_facet_select_ =
        DefaultFacetQuadratureSelection< double, 1 >();

template <>
const GlobalAssembler< float, 3 >::QuadratureSelectionFunction
    GlobalAssembler< float, 3 >::default_select_ =
        DefaultQuadratureSelection< float, 3 >();
template <>
const GlobalAssembler< float, 3 >::FacetQuadratureSelectionFunction
    GlobalAssembler< float, 3 >::default_facet_select_ =
        DefaultFacetQuadratureSelection< float, 3 >();
template <>
const GlobalAssembler< float, 2 >::QuadratureSelectionFunction
    GlobalAssembler< float, 2 >::default_select_ =
        DefaultQuadratureSelection< float, 2 >();
template <>
const GlobalAssembler< float, 2 >::FacetQuadratureSelectionFunction
    GlobalAssembler< float, 2 >::default_facet_select_ =
        DefaultFacetQuadratureSelection< float, 2 >();
template <>
const GlobalAssembler< float, 1 >::QuadratureSelectionFunction
    GlobalAssembler< float, 1 >::default_select_ =
        DefaultQuadratureSelection< float, 1 >();
template <>
const GlobalAssembler< float, 1 >::FacetQuadratureSelectionFunction
    GlobalAssembler< float, 1 >::default_facet_select_ =
        DefaultFacetQuadratureSelection< float, 1 >();

template class GlobalAssembler <double, 3>;
template class GlobalAssembler <double, 2>;
template class GlobalAssembler <double, 1>;

template class GlobalAssembler <float, 3>;
template class GlobalAssembler <float, 2>;
template class GlobalAssembler <float, 1>;

} // namespace hiflow
