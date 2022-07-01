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

#ifndef HIFLOW_SPACE_TOOLS_H_
#define HIFLOW_SPACE_TOOLS_H_

#include <vector>
#include "mpi.h"
#include "common/vector_algebra_descriptor.h"
#include "dof/dof_fem_types.h"
#include "linear_algebra/vector.h"
#include "space/space_types.h"

namespace hiflow {

/// @brief HiFlow vector space tools
/// @author Philipp Gerstner
///

template <class DataType, int DIM> class VectorSpace;

// TODO: this function contains some functionality which might be put into vectorspace for performance reasons
template < class DataType, int DIM >
void interpolate_constrained_std_vector(const VectorSpace< DataType, DIM > &space,
                                        la::Vector< DataType > &vector);

template < class DataType, int DIM >
void interpolate_constrained_vector( const VectorSpace< DataType, DIM > &space, 
                                     la::Vector< DataType > &vector);

/// \brief get coordinates of DOFs for Finite Element fe_ind in VectorSpace space,
/// if this element is of Lagrange type 
template < class DataType, int DIM >
void get_lagrange_dof_coordinates ( const VectorSpace< DataType, DIM > &space, 
                                    int fe_ind,
                                    std::vector< Vec<DIM, DataType> >& dof_coords,
                                    std::vector< doffem::gDofId >& dof_global_id,
                                    std::vector< bool >& dof_is_local);

} // namespace hiflow

#endif // HIFLOW_SPACE_TOOLS_H_

