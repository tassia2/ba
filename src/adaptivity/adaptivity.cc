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

#include "adaptivity/time_mesh.h"
#include "adaptivity/dynamic_mesh_handler.h"
#include "adaptivity/dynamic_mesh_problem.h"
#include "adaptivity/space_patch_interpolation.h"
#include "adaptivity/time_patch_interpolation.h"
#include "adaptivity/refinement_strategies.h"
#include "linear_algebra/block_matrix.h"
#include "space/fe_interpolation_map.h"

namespace hiflow {

template class TimeMesh< double >;

template class SpacePatchInterpolation< la::LADescriptorCoupledD, 2>;

template class DynamicMeshHandler< la::LADescriptorCoupledD, 2, FeInterMapFullNodal<la::LADescriptorCoupledD, 2> >;
//template class DynamicMeshHandler< la::LADescriptorBlock< LADescriptorCoupledD >, 3, FeInterMapFullNodal<la::LADescriptorBlock< LADescriptorCoupledD >, 3> >;

template class DynamicMeshProblem< la::LADescriptorCoupledD, 2 >;
template class DynamicMeshProblem< la::LADescriptorBlock< LADescriptorCoupledD >, 2 >;

//template class SpacePatchInterpolation< LADescriptorCoupledD, mesh::IMPL_P4EST, 3 >;
template class TimePatchInterpolation< double, double >;

template void local_fixed_fraction_strategy< double >(
    double refine_frac, double coarsen_frac, int threshold, int coarsen_marker,
    const std::vector< double > &indicators, std::vector< int > &adapt_markers);

template void global_fixed_fraction_strategy< double >(
    MPI_Comm comm, double refine_frac, double coarsen_frac, int threshold,
    int num_global_cells, int coarsen_marker,
    const std::vector< double > &indicators,
    const std::vector< bool > &is_local, std::vector< int > &adapt_markers);

template void
doerfler_marking< double >(MPI_Comm comm, double theta, int num_global_cells,
                           const std::vector< double > &indicators,
                           const std::vector< bool > &is_local,
                           std::vector< int > &adapt_markers);

template void fixed_error_strategy< double >(
    double tol, int num_global_cells, double conv_order, int threshold,
    int coarsen_marker, const std::vector< double > &indicators,
    std::vector< int > &adapt_markers);

template void sum_residuals_and_weights< double >(
    int est_var, EstimatorMode est_mode, int num_eq, int num_steps,
    int num_mesh, bool csi, const std::vector< std::vector< double > > &hK,
    const std::vector< int > &indicator_mesh_indices,
    const std::vector< std::vector< std::vector< double > > >
        &est_cell_residual,
    const std::vector< std::vector< std::vector< double > > >
        &est_cell_timejump,
    const std::vector< std::vector< std::vector< double > > > &est_interface,
    std::vector< std::vector< double > > &element_residual,
    std::vector< std::vector< double > > &element_weight);

template void reduce_estimators< double >(
    const MPI_Comm &comm, const std::string &reduction_type, int num_steps,
    int num_mesh, const std::vector< double > &time_step_size,
    const std::vector< int > &num_cells_per_mesh,
    const std::vector< int > &indicator_mesh_indices,
    const std::vector< std::vector< double > > &time_indicator,
    const std::vector< std::vector< double > > &space_indicator,
    const std::vector< mesh::MeshPtr > &mesh_list,
    std::vector< std::vector< double > > &reduced_space_indicator,
    std::vector< double > &reduced_time_indicator);

//#endif
} // namespace hiflow
