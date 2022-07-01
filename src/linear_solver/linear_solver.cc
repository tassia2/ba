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

#include "linear_solver/cg.h"
#include "linear_solver/fgmres.h"
#include "linear_solver/gmres.h"
#include "linear_solver/bicgstab.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_solver/preconditioner_bjacobi_ext.h"
#include "linear_solver/richardson.h"

namespace hiflow {
namespace la {

template class CG< LADescriptorCoupledD >;
template class FGMRES< LADescriptorCoupledD >;
template class GMRES< LADescriptorCoupledD >;
template class BiCGSTAB< LADescriptorCoupledD >;
template class PreconditionerBlockJacobiExt< LADescriptorCoupledD >;
template class Richardson< LADescriptorCoupledD >;
             
}
} // namespace hiflow
