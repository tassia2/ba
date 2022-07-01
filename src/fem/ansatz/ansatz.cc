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

#include "fem/ansatz/ansatz_space.h"
#include "fem/ansatz/ansatz_sum.h"
#include "fem/ansatz/ansatz_transformed.h"
#include "fem/ansatz/ansatz_p_line_lagrange.h"
#include "fem/ansatz/ansatz_p_tri_lagrange.h"
#include "fem/ansatz/ansatz_pyr_lagrange.h"
#include "fem/ansatz/ansatz_p_tet_lagrange.h"
#include "fem/ansatz/ansatz_q_quad_lagrange.h"
#include "fem/ansatz/ansatz_q_hex_lagrange.h"
#include "fem/ansatz/ansatz_skew_aug_p_tri_mono.h"
#include "fem/ansatz/ansatz_aug_p_tri_mono.h"

namespace hiflow {
namespace doffem {

template class AnsatzSpace< float, 3 >;
template class AnsatzSpaceSum <double, 2>;
template class AnsatzSpaceTransformed <double, 1>;
template class AnsatzSpaceTransformed <double, 2>;
template class AnsatzSpaceTransformed <double, 3>;

template class PLineLag< double,1 >;
template class PTriLag< double, 2 >;
template class PTetLag< double,3 >;
template class PyrLag< double,3 >;
template class QQuadLag< double,2 >;
template class QHexLag< double,3 >;

template class SkewAugPTriMono< double, 2>;
template class AugPTriMono< double, 2>;

} // namespace doffem
} // namespace hiflow
