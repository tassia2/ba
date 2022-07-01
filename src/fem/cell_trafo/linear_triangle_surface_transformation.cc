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

#include "fem/cell_trafo/linear_triangle_surface_transformation.h"

/// \author Philipp Gerstner

namespace hiflow {
namespace doffem {
    
template <>
bool LinearTriangleSurfaceTransformation< double, 2, 3 >::inverse_impl(const Coord<3,double>& co_phy, Coord<2,double> &co_ref) const 
{
  return this->inverse_2Dto3D(co_phy, co_ref);
}
template <>
bool LinearTriangleSurfaceTransformation< float, 2, 3 >::inverse_impl(const Coord<3,float>& co_phy, Coord<2,float> &co_ref) const 
{
  return this->inverse_2Dto3D(co_phy, co_ref);
}

} // namespace doffem
} // namespace hiflow
