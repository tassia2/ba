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

#include "linear_algebra/la_descriptor.h"
#include "mesh/entity.h"
#include "space/dirichlet_boundary_conditions.h"
#include "space/fe_evaluation.h"
#include "space/fe_interpolation_map.h"
#include "space/fe_interpolation_global.h"

//#include "space/periodic_boundary_conditions.h"
//#include "space/periodic_boundary_conditions_cartesian.h"


namespace hiflow {

/*
template class PeriodicBoundaryConditions<float, 3 >;
template class PeriodicBoundaryConditionsCartesian<float, 3>;
* */

template class FeInterNodal<double, 3, FeEvalCell<double, 3> >;
template class FeInterMapFullNodal<la::LADescriptorCoupledD, 3>;
template class FeInterMapRedNodal<la::LADescriptorCoupledD, 3>;
template void find_dofs_on_face< float, 3 >(mesh::Id, const VectorSpace< float, 3 > &, size_t, std::vector< doffem::DofID > &, int&);
template void find_dofs_on_face< float, 3 >(const mesh::EntityIterator &, const VectorSpace< float, 3 > &, size_t, std::vector< doffem::DofID > &);
                        
} // namespace hiflow
