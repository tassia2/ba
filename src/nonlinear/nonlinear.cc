// Copyright (C) 2011-2020 Vincent Heuveline
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

#include "nonlinear/newton.h"
#include "nonlinear/damping_armijo.h"
#include "nonlinear/forcing_eisenstat_walker.h"

/// @author Tobias Hahn, Michael Schick

using namespace hiflow::la;

namespace hiflow {

/// template instantiation
template class Newton< LADescriptorCoupledD, 3 >;
template class ArmijoDamping< la::LADescriptorCoupledD, 3 >;
template class EWForcing< la::LADescriptorCoupledD >;

} // namespace hiflow
