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

#include "fem/reference_cell.h"
#include "fem/function_space.h"

namespace hiflow {
namespace doffem {

template class RefCellFunction< float, 2>;
template class FunctionSpace< float, 2>;
template class RefCellLineStd< float, 1 >;
template class RefCellQuadStd< float, 2 >;
template class RefCellTriStd< float, 2 >;
template class RefCellTetStd< float, 3 >;
template class RefCellHexStd< float, 3 >;
template class RefCellPyrStd< float, 3 >;

} // namespace doffem
} // namespace hiflow
