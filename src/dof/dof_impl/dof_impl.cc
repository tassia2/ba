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

#include "dof/dof_impl/dof_functional.h"
#include "dof/dof_impl/dof_functional_point.h"
#include "dof/dof_impl/dof_functional_facet_normal_moment.h"
#include "dof/dof_impl/dof_functional_cell_moment.h"

namespace hiflow {
namespace doffem {

template class DofFunctional< float, 3 >;
template class DofPointEvaluation< float, 3 >;
template class DofFacetNormalMoment< float, 3 >;
template class DofCellMoment< float, 3 >;

} // namespace doffem
} // namespace hiflow
