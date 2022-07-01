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

#ifndef HIFLOW_ADAPTIVITY_REFINEMENT_STRATEGIES
#define HIFLOW_ADAPTIVITY_REFINEMENT_STRATEGIES

/// \author Philipp Gerstner

#include "common/log.h"
#include "common/array_tools.h"
#include "mesh/mesh.h"
#include "mesh/types.h"
#include <map>
#include <mpi.h>
#include <string>
#include <vector>
#include <cmath>
#include <utility>

enum class EstimatorMode {
 DWR = 1,
 STD_RESIDUAL = 2
};

namespace hiflow {
///
/// \brief set of functions for building mesh adaption flags out of local error
/// indicators
///

/// \brief refine / coarsen a fixed fraction of cells on local processor
/// @param[in] refine_frac refine the refince_frac% cells with largest
/// indicators
/// @param[in] coarsen_frac coarsen the rcoarsen_frac% cells with smallest
/// indicators
/// @param[in] threshold start coarsening when more than #threshold cells are
/// present in mesh
/// @param[in] coarsen_marker coarsen family l of cells k if \sum_{k \in
/// Family(l)} coarsen_marker(k) <= #Family(l)
/// @param[in] indicators error indicators
/// @param[out] adapt_markers resulting adaption markers to be passed to refine
/// routine of mesh
template < class DataType >
void local_fixed_fraction_strategy(DataType refine_frac, DataType coarsen_frac,
                                   int threshold, int coarsen_marker,
                                   const std::vector< DataType > &indicators,
                                   std::vector< int > &adapt_markers);

/// \brief refine / coarsen a fixed fraction of cells (globally)
/// @param[in] comm underlying MPI communicator
/// @param[in] refine_frac refine the refince_frac% cells with largest
/// indicators
/// @param[in] coarsen_frac coarsen the rcoarsen_frac% cells with smallest
/// indicators
/// @param[in] threshold start coarsening when more than #threshold cells are
/// present in mesh
/// @param[in] coarsen_marker coarsen family l of cells k if \sum_{k \in
/// Family(l)} coarsen_marker(k) <= #Family(l)
/// @param[in] num_global_cells number of cells in global mesh
/// @param[in] indicators error indicators
/// @param[in] is_local flag for each cell whether it is local or not
/// @param[out] adapt_markers resulting adaption markers to be passed to refine
/// routine of mesh
template < class DataType >
void global_fixed_fraction_strategy(MPI_Comm comm, DataType refine_frac,
                                    DataType coarsen_frac, int threshold,
                                    int num_global_cells, int coarsen_marker,
                                    const std::vector< DataType > &indicators,
                                    const std::vector< bool > &is_local,
                                    std::vector< int > &adapt_markers);

/// \brief refine according to Dörfler marking: refine set of cells M such that
/// sum_{c in M} ind[c] >= theta * sum_{c} ind[c]
/// @param[in] comm underlying MPI communicator
/// @param[in] theta parameter
/// @param[in] num_global_cells number of cells in global mesh
/// @param[in] indicators error indicators
/// @param[in] is_local flag for each cell whether it is local or not
/// @param[out] adapt_markers resulting adaption markers to be passed to refine
/// routine of mesh
template < class DataType >
void doerfler_marking(MPI_Comm comm, DataType theta, int num_global_cells,
                      const std::vector< DataType > &indicators,
                      const std::vector< bool > &is_local,
                      std::vector< int > &adapt_markers);

/// \brief adapt mesh to obtain a desired accuracy: refine cell k if
/// indicator(k) > tol / num_global_cells \br coarsen cell k if 2^conv_order *
/// indicator(k) < tol / num_global_cells
/// @param[in] tol desired accuracy
/// @param[in] num_global_cells number of global cells in mesh
/// @param[in] conv_order assumed convergence order of applied numerical scheme
/// @param[in] threshold start coarsening when more than #threshold cells are
/// present in mesh
/// @param[in] coarsen_marker coarsen family l of cells k if \sum_{k \in
/// Family(l)} coarsen_marker(k) <= #Family(l)
/// @param[in] indicators error indicators
/// @param[out] adapt_markers resulting adaption markers to be passed to refine
/// routine of mesh
template < class DataType >
void fixed_error_strategy(DataType tol, int num_global_cells,
                          DataType conv_order, int threshold,
                          int coarsen_marker,
                          const std::vector< DataType > &indicators,
                          std::vector< int > &adapt_markers);

/// \brief sum up residual contributions of different equations and different
/// types of residuals, e.g. cell and facet residuals
/// @param[in] est_var 0: spatial error indicators, 1: temporal error indicators
/// @param[in] est_mode DWR or STD_RESIDUAL
/// @param[in] num_eq number of equations of underlying PDE
/// @param[in] num_steps number of involved time steps
/// @param[in] num_mesh number of involved meshes
/// @param[in] csi true: apply Cauchy-Schwarz inequality when combining
/// residuals and weights
/// @param[in] hK cell diameters for each cell in each mesh
/// @param[in] indicator_mesh_indices mapping time step to mesh index
/// @param[in] est_cell_residual spatial and temporal cell residuals and weights
/// for each cell and each time step
/// @param[in] est_cell_timejump spatial and temporal cell time jumps and
/// weights for each cell and each time step
/// @param[in] est_interface spatial and temporal facet jumps and weights for
/// each cell and each time step
/// @param[out] element_residual final residual for each space-time slap
/// @param[out] element_weight final weight for each space-time slab
template < class DataType >
void sum_residuals_and_weights(
    int est_var, EstimatorMode est_mode, int num_eq, int num_steps,
    int num_mesh, bool csi, const std::vector< std::vector< DataType > > &hK,
    const std::vector< int > &indicator_mesh_indices,
    const std::vector< std::vector< std::vector< DataType > > > &est_cell_residual,
    const std::vector< std::vector< std::vector< DataType > > > &est_cell_timejump,
    const std::vector< std::vector< std::vector< DataType > > > &est_interface,
    std::vector< std::vector< DataType > > &element_residual,
    std::vector< std::vector< DataType > > &element_weight);

template < class DataType >
void get_equation_contributions( int est_var, EstimatorMode est_mode, int num_eq, int num_steps,
                                 int num_mesh, bool csi, const std::vector< std::vector< DataType > > &hK,
                                 const std::vector< int > &indicator_mesh_indices,
                                 const std::vector< std::vector< std::vector< DataType > > > &est_cell_residual,
                                 const std::vector< std::vector< std::vector< DataType > > > &est_cell_timejump,
                                 const std::vector< std::vector< std::vector< DataType > > > &est_interface,
                                 std::vector< DataType > &primal_cell_residual_contrib, 
                                 std::vector< DataType > &primal_cell_timejump_contrib, 
                                 std::vector< DataType > &primal_interface_contrib,
                                 std::vector< DataType > &dual_cell_residual_contrib, 
                                 std::vector< DataType > &dual_cell_timejump_contrib, 
                                 std::vector< DataType > &dual_interface_contrib);
                                 
/// \brief combine residuals and weights to obtain spatial and temporal error
/// indicators for each space-time slab
/// @param[in] num_steps number of involved time steps
/// @param[in] num_mesh number of involved meshes
/// @param[in] num_cells_per_mesh number of cells for each mesh
/// @param[in] indicator_mesh_indices mapping time step to mesh index
/// @param[in] mesh_list array of pointers to involved meshes
/// @param[in] element_residual_h spatial residual for each space-time slab
/// @param[in] element_residual_tau temporal residual for each space-time slab
/// @param[in] element_weight_h spatial weight for each space-time slab
/// @param[in] element_weight_tau temporal weight for each space-time slab
/// @param[out] time_indicator temporal error indicator for each cell and each
/// timestep
/// @param[out] space_indicator spatial error indicator for each cell and each
/// timestep
/// @param[out] local_time_estimator sum of time_indicators over all time steps
/// and all local cells
/// @param[out] local_space_estimator sum of space_indicators over all time
/// steps and all local cells
template < class DataType >
void compute_space_time_estimators(
    int num_steps, int num_mesh, const std::vector< int > &num_cells_per_mesh,
    const std::vector< int > &indicator_mesh_indices,
    const std::vector< mesh::MeshPtr > &mesh_list,
    const std::vector< std::vector< DataType > > &element_residual_h,
    const std::vector< std::vector< DataType > > &element_residual_tau,
    const std::vector< std::vector< DataType > > &element_weight_h,
    const std::vector< std::vector< DataType > > &element_weight_tau,
    std::vector< std::vector< DataType > > &time_indicator,
    std::vector< std::vector< DataType > > &space_indicator,
    DataType local_time_estimator, DataType local_space_estimator);

/// \brief reduce space error indicators for several time steps to single
/// indicator. reduce time error indicators for several cells to single time
/// indicator
/// @param[in] comm underlying MPI group
/// @param[in] reduction type currently only SUM implemented
/// @param[in] num_steps number of involved time steps
/// @param[in] num_mesh number of involved meshes
/// @param[in] time_step_size time step size for each time step
/// @param[in] num_cells_per_mesh number of cells for each mesh
/// @param[in] indicator_mesh_indices mapping time step to mesh index
/// @param[in] time_indicator temporal error indicator for each cell and each
/// timestep
/// @param[in] space_indicator spatial error indicator for each cell and each
/// timestep
/// @param[in] mesh_list array of pointers to involved meshes
/// @param[out] reduced_space_indicator single spatial error indicator for each
/// cell on each mesh
/// @param[out] reduced_time_indicator single temporal error indicator for each
/// time step
template < class DataType >
void reduce_estimators(
    const MPI_Comm &comm, const std::string &reduction_type, int num_steps,
    int num_mesh, const std::vector< DataType > &time_step_size,
    const std::vector< int > &num_cells_per_mesh,
    const std::vector< int > &indicator_mesh_indices,
    const std::vector< std::vector< DataType > > &time_indicator,
    const std::vector< std::vector< DataType > > &space_indicator,
    const std::vector< mesh::MeshPtr > &mesh_list,
    std::vector< std::vector< DataType > > &reduced_space_indicator,
    std::vector< DataType > &reduced_time_indicator);

template <class DataType>
void write_space_indicators_to_hdf5 (const MPI_Comm& comm, const int my_offset, const int global_size,
                                     const std::string &filename, const std::string &groupname, const std::string &datasetname,
                                     const std::vector< std::vector< DataType > > &indicators);
                                
template <class DataType>
void write_spacetime_indicators_to_hdf5 (const MPI_Comm& comm, const size_t first_step_to_write, const size_t num_steps_to_write,
                                         const std::string &filename, const std::string &groupname, const std::string &dataset_prefix,
                                         const std::vector< std::vector< std::vector< DataType > > > &indicators);
                                         
template <class DataType>
void read_space_indicators_from_hdf5 (const MPI_Comm& comm, const int my_offset, const int global_size,
                                      const std::string &filename, const std::string &groupname, const std::string &datasetname,
                                      std::vector< std::vector< DataType > > &indicators);
                                      
template <class DataType>
void read_spacetime_indicators_from_hdf5 (const MPI_Comm& comm, const size_t first_step_to_read, const size_t num_steps_to_read,
                                          const std::string &filename, const std::string &groupname, const std::string &dataset_prefix,
                                           std::vector< std::vector< std::vector< DataType > > > &indicators);
                                           
/////////////////////////////////////////////////////////////////////
/////////////// Implementation //////////////////////////////////////
/////////////////////////////////////////////////////////////////////

template < class DataType >
void local_fixed_fraction_strategy(DataType refine_frac, DataType coarsen_frac,
                                   int threshold, int coarsen_marker,
                                   const std::vector< DataType > &indicators,
                                   std::vector< int > &adapt_markers) {
  adapt_markers.resize(indicators.size(), 0);
  std::vector< int > sort_ind(indicators.size(), 0);

  for (int i = 0; i < indicators.size(); ++i) {
    sort_ind[i] = i;
  }
  compute_sort_permutation_stable(indicators, sort_ind);

  // 1. Mark cells for refinement
  int first_cell = std::floor((1. - refine_frac) * sort_ind.size());
  int num_global_cells = indicators.size();

  for (int l = first_cell; l < sort_ind.size(); ++l) {
    int cell_index = sort_ind[l];
    adapt_markers[cell_index] = 1;
  }

  // 2.Mark cells for coarsening
  if (num_global_cells >= threshold) {
    int last_cell = std::ceil(coarsen_frac * sort_ind.size());
    for (int l = 0; l < last_cell; ++l) {
      int cell_index = sort_ind[l];
      adapt_markers[cell_index] = coarsen_marker;
    }
  }
}

template < class DataType >
void global_fixed_fraction_strategy(MPI_Comm comm, DataType refine_frac,
                                    DataType coarsen_frac, int threshold,
                                    int num_global_cells, int coarsen_marker,
                                    const std::vector< DataType > &indicators,
                                    const std::vector< bool > &is_local,
                                    std::vector< int > &adapt_markers) {
  int rank;
  int num_proc;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_proc);

  LOG_DEBUG(1, "Number of global cells " << num_global_cells);

  adapt_markers.resize(indicators.size(), 0);
  std::vector< int > sort_ind(indicators.size(), 0);

  for (int i = 0; i < indicators.size(); ++i) {
    sort_ind[i] = i;
  }
  compute_sort_permutation_stable(indicators, sort_ind);

  // 1. Mark potential cells for refinement
  std::vector< int > my_refine_cand;
  std::vector< int > my_coarsen_cand;
  std::vector< DataType > my_refine_cand_ind;
  std::vector< DataType > my_coarsen_cand_ind;

  my_refine_cand.reserve(indicators.size());
  my_coarsen_cand.reserve(indicators.size());
  my_refine_cand_ind.reserve(indicators.size());
  my_coarsen_cand_ind.reserve(indicators.size());

  int k = sort_ind.size() - 1;
  while (my_refine_cand.size() < refine_frac * num_global_cells && k >= 0) {
    int cell_index = sort_ind[k];
    if (is_local[cell_index]) {
      my_refine_cand.push_back(cell_index);
      my_refine_cand_ind.push_back(indicators[cell_index]);
    }
    k--;
  }

  LOG_DEBUG(1, "Process " << rank << " marked " << my_refine_cand.size()
                          << " cells for potential refinement");

  // 2.Mark potential cells for coarsening
  if (num_global_cells >= threshold) {
    int l = 0;
    while (my_coarsen_cand.size() < coarsen_frac * num_global_cells &&
           l < sort_ind.size()) {
      int cell_index = sort_ind[l];
      if (is_local[cell_index]) {
        my_coarsen_cand.push_back(cell_index);
        my_coarsen_cand_ind.push_back(indicators[cell_index]);
      }
      l++;
    }
  }

  LOG_DEBUG(1, "Process " << rank << " marked " << my_coarsen_cand.size()
                          << " cells for potential coarsening");

  std::vector< int > my_refine;
  std::vector< int > my_coarsen;

  if (rank != 0) {
    // 3. send marked cells to master
    int num_refine = my_refine_cand.size();
    int num_coarsen = my_coarsen_cand.size();
    int num[2];
    num[0] = num_refine;
    num[1] = num_coarsen;

    assert(my_refine_cand_ind.size() == num_refine);
    assert(my_coarsen_cand_ind.size() == num_coarsen);

    MPI_Send(&num, 2, MPI_INT, 0, 0, comm);
    MPI_Send(&my_refine_cand[0], num_refine, MPI_INT, 0, 1, comm);
    MPI_Send(&my_coarsen_cand[0], num_coarsen, MPI_INT, 0, 2, comm);
    MPI_Send(&my_refine_cand_ind[0], num_refine, MPI_DOUBLE, 0, 3, comm);
    MPI_Send(&my_coarsen_cand_ind[0], num_coarsen, MPI_DOUBLE, 0, 4, comm);

    // 6. recv definite cells to be refined and coarsened

    MPI_Status recv_stat;
    int num_cells[2];
    MPI_Recv(&num_cells, 2, MPI_INT, 0, 0, comm, &recv_stat);

    my_refine.resize(num_cells[0], -1);
    my_coarsen.resize(num_cells[1], -1);

    MPI_Recv(&my_refine[0], num_cells[0], MPI_INT, 0, 1, comm, &recv_stat);
    MPI_Recv(&my_coarsen[0], num_cells[1], MPI_INT, 0, 2, comm, &recv_stat);

    LOG_DEBUG(2, "Process " << rank << " : recv done ");
  } else {
    // 3. master receives marked cells
    std::vector< std::vector< int > > recv_num(num_proc);
    for (int p = 0; p < num_proc; ++p) {
      recv_num[p].resize(2, 0);
    }

    std::vector< int > refine_cand;
    std::vector< int > coarsen_cand;

    std::vector< DataType > refine_cand_ind;
    std::vector< DataType > coarsen_cand_ind;

    refine_cand.reserve(2 * num_proc * my_refine_cand.size());
    coarsen_cand.reserve(2 * num_proc * my_coarsen_cand.size());

    refine_cand_ind.reserve(2 * num_proc * my_refine_cand.size());
    coarsen_cand_ind.reserve(2 * num_proc * my_coarsen_cand.size());

    refine_cand.insert(refine_cand.end(), my_refine_cand.begin(),
                       my_refine_cand.end());
    coarsen_cand.insert(coarsen_cand.end(), my_coarsen_cand.begin(),
                        my_coarsen_cand.end());

    refine_cand_ind.insert(refine_cand_ind.end(), my_refine_cand_ind.begin(),
                           my_refine_cand_ind.end());
    coarsen_cand_ind.insert(coarsen_cand_ind.end(), my_coarsen_cand_ind.begin(),
                            my_coarsen_cand_ind.end());

    MPI_Status recv_stat;

    std::vector< int > refine_procs(my_refine_cand.size(), 0);
    std::vector< int > coarsen_procs(my_coarsen_cand.size(), 0);

    for (int p = 1; p < num_proc; ++p) {
      MPI_Recv(&recv_num[p][0], 2, MPI_INT, p, 0, comm, &recv_stat);
      LOG_DEBUG(2, "Process " << rank << " : got from process " << p
                              << ": num_refine= " << recv_num[p][0]
                              << ", num_coarsen= " << recv_num[p][1]);

      std::vector< int > tmp_refine_cand(recv_num[p][0], -1);
      std::vector< int > tmp_coarsen_cand(recv_num[p][1], -1);

      std::vector< DataType > tmp_refine_cand_ind(recv_num[p][0], -1.);
      std::vector< DataType > tmp_coarsen_cand_ind(recv_num[p][1], -1.);

      MPI_Recv(&tmp_refine_cand[0], recv_num[p][0], MPI_INT, p, 1, comm,
               &recv_stat);
      MPI_Recv(&tmp_coarsen_cand[0], recv_num[p][1], MPI_INT, p, 2, comm,
               &recv_stat);
      MPI_Recv(&tmp_refine_cand_ind[0], recv_num[p][0], MPI_DOUBLE, p, 3, comm,
               &recv_stat);
      MPI_Recv(&tmp_coarsen_cand_ind[0], recv_num[p][1], MPI_DOUBLE, p, 4, comm,
               &recv_stat);

      refine_cand.insert(refine_cand.end(), tmp_refine_cand.begin(),
                         tmp_refine_cand.end());
      coarsen_cand.insert(coarsen_cand.end(), tmp_coarsen_cand.begin(),
                          tmp_coarsen_cand.end());

      refine_cand_ind.insert(refine_cand_ind.end(), tmp_refine_cand_ind.begin(),
                             tmp_refine_cand_ind.end());
      coarsen_cand_ind.insert(coarsen_cand_ind.end(),
                              tmp_coarsen_cand_ind.begin(),
                              tmp_coarsen_cand_ind.end());

      for (int l = 0; l < recv_num[p][0]; ++l) {
        refine_procs.push_back(p);
      }
      for (int l = 0; l < recv_num[p][1]; ++l) {
        coarsen_procs.push_back(p);
      }

      LOG_DEBUG(2, "Process " << rank << " : recv done for process " << p);
    }

    // 4. Master process sorts all refine and coarsen candidates
    std::vector< int > sort_ind_refine(refine_cand.size(), 0);
    for (int i = 0; i < refine_cand.size(); ++i) {
      sort_ind_refine[i] = i;
    }
    //            std::cout << string_from_range(refine_cand_ind.begin(),
    //            refine_cand_ind.end());

    compute_sort_permutation_stable(refine_cand_ind, sort_ind_refine);

    std::vector< int > sort_ind_coarsen(coarsen_cand.size(), 0);
    for (int i = 0; i < coarsen_cand.size(); ++i) {
      sort_ind_coarsen[i] = i;
    }
    compute_sort_permutation_stable(coarsen_cand_ind, sort_ind_coarsen);

    // 5. master selects cells to refine and coarsen
    std::vector< std::vector< int > > refine_per_process(num_proc);
    std::vector< std::vector< int > > coarsen_per_process(num_proc);

    k = sort_ind_refine.size() - 1;
    int counter = 0;
    while (counter < refine_frac * num_global_cells && k >= 0) {
      int cand_index = sort_ind_refine[k];
      int cell_index = refine_cand[cand_index];
      int proc_index = refine_procs[cand_index];
      //                std::cout << k << ": " << cand_index << ": " << " cell "
      //                << cell_index << " proc: " << proc_index << std::endl;

      refine_per_process[proc_index].push_back(cell_index);

      k--;
      counter++;
    }

    counter = 0;
    k = 0;
    while (counter < coarsen_frac * num_global_cells &&
           k < sort_ind_coarsen.size()) {
      int cand_index = sort_ind_coarsen[k];
      int cell_index = coarsen_cand[cand_index];
      int proc_index = coarsen_procs[cand_index];

      coarsen_per_process[proc_index].push_back(cell_index);
      k++;
      counter++;
    }

    my_refine = refine_per_process[0];
    my_coarsen = coarsen_per_process[0];

    // 6. send refine and coarsen info to all processors
    for (int p = 1; p < num_proc; ++p) {
      int num_cells[2];
      num_cells[0] = refine_per_process[p].size();
      num_cells[1] = coarsen_per_process[p].size();

      LOG_DEBUG(2, "Process " << rank << " : send to process " << p
                              << " num_def_refine = " << num_cells[0]
                              << ", num_def_coarsen = " << num_cells[1]);

      MPI_Send(&num_cells, 2, MPI_INT, p, 0, comm);

      MPI_Send(&refine_per_process[p][0], num_cells[0], MPI_INT, p, 1, comm);
      MPI_Send(&coarsen_per_process[p][0], num_cells[1], MPI_INT, p, 2, comm);

      LOG_DEBUG(2, "Process " << rank << " : send done for process " << p);
    }
  }

  // 7. setup adapt_markers
  assert(my_refine.size() <= my_refine_cand.size());
  assert(my_coarsen.size() <= my_coarsen_cand.size());

  LOG_DEBUG(1, "Process " << rank << " marked " << my_refine.size()
                          << " cells for definite refinement");
  LOG_DEBUG(1, "Process " << rank << " marked " << my_coarsen.size()
                          << " cells for definite coarsening");

  for (int l = 0; l < my_refine.size(); ++l) {
    assert(my_refine[l] >= 0);
    assert(my_refine[l] < indicators.size());

    LOG_DEBUG(2, "Process " << rank << " marks cell " << my_refine[l]
                            << " for definite refinement");
    adapt_markers[my_refine[l]] = 1;
  }

  for (int l = 0; l < my_coarsen.size(); ++l) {
    assert(my_coarsen[l] >= 0);
    assert(my_coarsen[l] < indicators.size());

    adapt_markers[my_coarsen[l]] = coarsen_marker;
  }
}

template < class DataType >
void doerfler_marking(MPI_Comm comm, DataType theta, int num_global_cells,
                      const std::vector< DataType > &indicators,
                      const std::vector< bool > &is_local,
                      std::vector< int > &adapt_markers) {
  int rank;
  int num_proc;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_proc);
  int num_ind = indicators.size();

  LOG_DEBUG(1, "Number of global cells " << num_global_cells);

  adapt_markers.resize(indicators.size(), 0);

  std::vector< int > my_refine;
  std::vector< int > my_refine_cand;
  std::vector< DataType > my_refine_cand_ind;

  my_refine_cand.reserve(num_ind);
  my_refine_cand_ind.reserve(num_ind);

  // 1. extract local indicators
  for (int l = 0; l < num_ind; ++l) {
    if (is_local[l]) {
      my_refine_cand.push_back(l);
      my_refine_cand_ind.push_back(indicators[l]);
    }
  }

  if (rank != 0) {
    // 2. send error indicators to master
    int num_cand = my_refine_cand.size();

    MPI_Send(&num_cand, 1, MPI_INT, 0, 0, comm);
    MPI_Send(&my_refine_cand[0], num_cand, MPI_INT, 0, 1, comm);
    MPI_Send(&my_refine_cand_ind[0], num_cand, MPI_DOUBLE, 0, 2, comm);

    // 6. receive cells to be refined
    MPI_Status recv_stat;
    int num_cells;
    MPI_Recv(&num_cells, 1, MPI_INT, 0, 0, comm, &recv_stat);

    my_refine.resize(num_cells, -1);
    MPI_Recv(&my_refine[0], num_cells, MPI_INT, 0, 1, comm, &recv_stat);

    LOG_DEBUG(2, "Process " << rank << " : recv done ");
  } else {
    // 2. receive error indicators from slave procs
    std::vector< int > recv_num(num_proc);

    std::vector< int > refine_cand;
    std::vector< DataType > refine_cand_ind;
    std::vector< int > refine_procs(my_refine_cand.size(), 0);

    refine_cand.reserve(num_global_cells);
    refine_cand_ind.reserve(num_global_cells);

    refine_cand.insert(refine_cand.end(), my_refine_cand.begin(),
                       my_refine_cand.end());
    refine_cand_ind.insert(refine_cand_ind.end(), my_refine_cand_ind.begin(),
                           my_refine_cand_ind.end());

    MPI_Status recv_stat;

    for (int p = 1; p < num_proc; ++p) {
      MPI_Recv(&recv_num[p], 1, MPI_INT, p, 0, comm, &recv_stat);
      LOG_DEBUG(2, "Process " << rank << " : got from process " << p
                              << ": num_refine= " << recv_num[p]);

      std::vector< int > tmp_refine_cand(recv_num[p], -1);
      std::vector< DataType > tmp_refine_cand_ind(recv_num[p], -1.);

      MPI_Recv(&tmp_refine_cand[0], recv_num[p], MPI_INT, p, 1, comm,
               &recv_stat);
      MPI_Recv(&tmp_refine_cand_ind[0], recv_num[p], MPI_DOUBLE, p, 2, comm,
               &recv_stat);

      refine_cand.insert(refine_cand.end(), tmp_refine_cand.begin(),
                         tmp_refine_cand.end());
      refine_cand_ind.insert(refine_cand_ind.end(), tmp_refine_cand_ind.begin(),
                             tmp_refine_cand_ind.end());

      for (int l = 0; l < recv_num[p]; ++l) {
        refine_procs.push_back(p);
      }
      LOG_DEBUG(2, "Process " << rank << " : recv done for process " << p);
    }

    // 4. Master process sorts all refine candidates
    std::vector< int > sort_ind_refine(refine_cand.size(), 0);
    for (int i = 0; i < refine_cand.size(); ++i) {
      sort_ind_refine[i] = i;
    }

    compute_sort_permutation_stable(refine_cand_ind, sort_ind_refine);

    // 5. master process selects cells to be refined: minimal amount of cells M
    // such that sum_{c in M} ind[c] >= theta * sum_{c} ind[c]
    DataType marked_sum = 0.;
    DataType total_sum = 0.;
    std::vector< std::vector< int > > refine_per_process(num_proc);

    for (int l = 0; l < refine_cand_ind.size(); ++l) {
      total_sum += refine_cand_ind[l];
    }

    LOG_DEBUG(1, "Total error sum " << total_sum);

    int k = sort_ind_refine.size() - 1;
    while (marked_sum < theta * total_sum && k >= 0) {
      int cand_index = sort_ind_refine[k];
      int cell_index = refine_cand[cand_index];
      int proc_index = refine_procs[cand_index];
      //              std::cout << k << ": " << cand_index << ": " << " cell "
      //              << cell_index << " proc: " << proc_index << std::endl;

      refine_per_process[proc_index].push_back(cell_index);
      marked_sum += refine_cand_ind[cand_index];

      LOG_DEBUG(1, "Marked sum = " << marked_sum
                                   << " after adding cell from proc "
                                   << proc_index);
      k--;
    }

    LOG_DEBUG(1, "Marked error sum " << marked_sum);
    my_refine = refine_per_process[0];

    // 6. send refine and coarsen info to all processors
    for (int p = 1; p < num_proc; ++p) {
      int num_cells = refine_per_process[p].size();

      LOG_DEBUG(2, "Process " << rank << " : send to process " << p
                              << " num_def_refine = " << num_cells);

      MPI_Send(&num_cells, 1, MPI_INT, p, 0, comm);
      MPI_Send(&refine_per_process[p][0], num_cells, MPI_INT, p, 1, comm);

      LOG_DEBUG(2, "Process " << rank << " : send done for process " << p);
    }
  }

  // 7. setup adapt_markers
  assert(my_refine.size() <= my_refine_cand.size());

  LOG_DEBUG(1, "Process " << rank << " marked " << my_refine.size()
                          << " cells for definite refinement");

  for (int l = 0; l < my_refine.size(); ++l) {
    assert(my_refine[l] >= 0);
    assert(my_refine[l] < indicators.size());

    LOG_DEBUG(2, "Process " << rank << " marks cell " << my_refine[l]
                            << " for definite refinement");
    adapt_markers[my_refine[l]] = 1;
  }
}

template < class DataType >
void fixed_error_strategy(DataType tol, int num_global_cells,
                          DataType conv_order, int threshold,
                          int coarsen_marker,
                          const std::vector< DataType > &indicators,
                          std::vector< int > &adapt_markers) {
  adapt_markers.resize(indicators.size(), 0);
  DataType av_max_error = tol / num_global_cells;
  int num_cells = indicators.size();

  for (int c = 0; c < num_cells; ++c) {
    // mark cells for refinement
    if (indicators[c] > av_max_error) {
      adapt_markers[c] = 1;
    }

    // mark cells for coarsening
    if (num_global_cells >= threshold) {
      if (indicators[c] * std::pow(2.0, conv_order) < av_max_error) {
        adapt_markers[c] = coarsen_marker;
      }
    }
  }
}

// TODO: checken ob CSI Fall korrekt
template < class DataType >
void sum_residuals_and_weights(
    int est_var, EstimatorMode est_mode, int num_eq, int num_steps,
    int num_mesh, bool csi, const std::vector< std::vector< DataType > > &hK,
    const std::vector< int > &indicator_mesh_indices,
    const std::vector< std::vector< std::vector< DataType > > >
        &est_cell_residual,
    const std::vector< std::vector< std::vector< DataType > > >
        &est_cell_timejump,
    const std::vector< std::vector< std::vector< DataType > > > &est_interface,
    std::vector< std::vector< DataType > > &element_residual,
    std::vector< std::vector< DataType > > &element_weight) {
  // Data structure of input vectors (est_cell_residual, est_cell_timejump,
  // est_interface) [time_step][entity_index] { {primal_eq_1_res_h,
  // primal_eq_1_weight_h, primal_eq_1_res_tau, primal_eq_1_weight_tau;
  //                              primal_eq_n_res_h, primal_eq_n_weight_h,
  //                              primal_eq_n_res_tau, primal_eq_n_weight_tau}
  //                                  ........
  //                             {dual_eq_1_res_h, dual_eq_1_weight_h,
  //                             dual_eq_1_res_tau, dual_eq_1_weight_tau;
  //                              dual_eq_n_res_h, dual_eq_n_weight_h,
  //                              dual_eq_n_res_tau, dual_eq_n_weight_tau}}

  // est_type = 0: space error estimation
  // est_type = 1: time error estimation
  int off = 0;
  if (est_var == 1) {
    off = 2;
  }

  // allocate arrays
  element_residual.clear();
  element_weight.clear();

  element_residual.resize(num_steps);
  element_weight.resize(num_steps);

  for (int t = 0; t < num_steps; ++t) {
    int num_cells = est_cell_residual[t].size();
    element_residual[t].resize(num_cells, 0.);
    element_weight[t].resize(num_cells, 0.);
  }

  assert(est_cell_residual.size() == num_steps);
  assert(est_cell_timejump.size() == num_steps);
  assert(est_interface.size() == num_steps);

  // combine indicators
  // loop over all time steps
  for (int t = 0; t < num_steps; ++t) {
    int num_cells = est_cell_residual[t].size();
    int mesh_index = indicator_mesh_indices[t];

    assert(est_cell_residual[t].size() == num_cells);
    assert(est_cell_timejump[t].size() == num_cells);
    assert(est_interface[t].size() == num_cells);

    // loop over all cells
    for (int c = 0; c < num_cells; ++c) {
      const DataType h = hK[mesh_index][c];
      const DataType inv_h = 1. / h;

      assert(est_cell_residual[t][c].size() == 8 * num_eq);
      assert(est_cell_timejump[t][c].size() == 8 * num_eq);
      assert(est_interface[t][c].size() == 8 * num_eq);

      // loop over all equations
      for (int e = 0; e < num_eq; ++e) {
        // DWR error estimator
        if (est_mode == EstimatorMode::DWR) {
          if (csi) {
            // element residual =
            //    primal cell residual ||res||_K^{2}
            // +  dual cell residual ||res||_K^{2}
            // +  1/h * primal interface residual ||res||_dK^{2}
            // +  1/h * dual interface residual   ||res||_dK^{2}
            // +  primal cell timejump ||res||_K^{2}
            // +  dual cell timejump ||res||_K^{2}

            element_residual[t][c] +=
                est_cell_residual[t][c][4 * e + off] +
                est_cell_residual[t][c][4 * (e + num_eq) + off] +
                inv_h * est_interface[t][c][4 * e + off] +
                inv_h * est_interface[t][c][4 * (e + num_eq) + off] +
                est_cell_timejump[t][c][4 * e + off] +
                est_cell_timejump[t][c][4 * (e + num_eq) + off];

            // element weight =
            //    primal cell residual ||weight||_K^{2}
            // +  dual cell residual ||weight||_K^{2}
            // +  h * primal interface residual ||weight||_dK^{2}
            // +  h * dual interface residual   ||weight||_dK^{2}
            // +  primal cell timejump |weight||_K^{2}
            // +  dual cell timejump ||weight||_K^{2}

            element_weight[t][c] +=
                est_cell_residual[t][c][4 * e + off + 1] +
                est_cell_residual[t][c][4 * (e + num_eq) + off + 1] +
                h * est_interface[t][c][4 * e + off + 1] +
                h * est_interface[t][c][4 * (e + num_eq) + off + 1] +
                est_cell_timejump[t][c][4 * e + off + 1] +
                est_cell_timejump[t][c][4 * (e + num_eq) + off + 1];
          } else {
            // element residual =
            //    primal cell residual (res_K, weight)_K
            // +  dual cell residual (res_K, weight)_K
            // +  primal interface residual (res_K, weight)_K
            // +  dual interface residual (res_K, weight)_K
            // +  primal cell timejump (res_K, weight)_K
            // +  dual cell timejump (res_K, weight)_K

            element_residual[t][c] +=
                est_cell_residual[t][c][4 * e + off] +
                est_cell_residual[t][c][4 * (e + num_eq) + off] +
                est_interface[t][c][4 * e + off] +
                est_interface[t][c][4 * (e + num_eq) + off] +
                est_cell_timejump[t][c][4 * e + off] +
                est_cell_timejump[t][c][4 * (e + num_eq) + off];
          }
        }
        // Standard residual estimator
        else if (est_mode == EstimatorMode::STD_RESIDUAL) {
          // element residual =
          //    h²_K primal cell residual ||res||_K^{2}
          // +  h²_K dual cell residual ||res||_K^{2}
          // +  h_E * primal interface residual ||res||_dK^{2}
          // +  h_E * dual interface residual   ||res||_dK^{2}
          // +  h_K * primal cell timejump ||res||_K^{2}
          // +  h_K * dual cell timejump ||res||_K^{2}
          // (where h_K and h_E are contained in residual array)

          element_residual[t][c] +=
              est_cell_residual[t][c][4 * e + off] +
              est_cell_residual[t][c][4 * (e + num_eq) + off] +
              est_interface[t][c][4 * e + off] +
              est_interface[t][c][4 * (e + num_eq) + off] +
              est_cell_timejump[t][c][4 * e + off] +
              est_cell_timejump[t][c][4 * (e + num_eq) + off];
        }
      }
      if (est_mode == EstimatorMode::DWR) {
        if (csi) {
          element_residual[t][c] = std::sqrt(element_residual[t][c]);
          element_weight[t][c] = std::sqrt(element_weight[t][c]);
        } else {
          element_residual[t][c] = std::abs(element_residual[t][c]);
          element_weight[t][c] = 1.;
        }
      } else if (est_mode == EstimatorMode::STD_RESIDUAL) {
        element_residual[t][c] = std::abs(element_residual[t][c]);
        element_weight[t][c] = 1.;
      }
    }
  }
}

template < class DataType >
void reduce_estimators(
    const MPI_Comm &comm, const std::string &reduction_type, int num_steps,
    int num_mesh, const std::vector< DataType > &time_step_size,
    const std::vector< int > &num_cells_per_mesh,
    const std::vector< int > &indicator_mesh_indices,
    const std::vector< std::vector< DataType > > &time_indicator,
    const std::vector< std::vector< DataType > > &space_indicator,
    const std::vector< mesh::MeshPtr > &mesh_list,
    std::vector< std::vector< DataType > > &reduced_space_indicator,
    std::vector< DataType > &reduced_time_indicator) {
  assert(num_cells_per_mesh.size() == num_mesh);
  assert(time_step_size.size() >= num_steps + 1);
  assert(indicator_mesh_indices.size() >= num_steps);
  assert(mesh_list.size() == num_mesh);

  // reset indicators and estimators
  reduced_space_indicator.resize(num_mesh);
  for (int m = 0; m < num_mesh; ++m) {
    int num_cells = num_cells_per_mesh[m];
    reduced_space_indicator[m].resize(num_cells, 0.);
  }

  reduced_time_indicator.resize(num_steps, 0.);
  std::vector< DataType > local_reduced_time_indicator(num_steps, 0.);

  // compute indicators for cells and time steps
  // ---------------------------------------
  if (reduction_type == "SUM") {
    // loop over all time steps
    for (int t = 0; t < num_steps; ++t) {
      //DataType delta_t = time_step_size[t + 1];
      int mesh_index = indicator_mesh_indices[t];

      assert(mesh_index < num_mesh);
      int num_cells = num_cells_per_mesh[mesh_index];

      // loop over all cells
      for (int c = 0; c < num_cells; ++c) {
        if (mesh_list[mesh_index]->cell_is_local(c)) {
          reduced_space_indicator[mesh_index][c] +=
              space_indicator[t][c] /*/ delta_t*/;
          local_reduced_time_indicator[t] += time_indicator[t][c];
        }
      }
    }
    // compute global reduced time indicator
    MPI_Allreduce(&local_reduced_time_indicator[0], &reduced_time_indicator[0],
                  num_steps, MPI_DOUBLE, MPI_SUM, comm);
  } else if (reduction_type == "MAX") {
    assert(false);
    /*
    // time indicator
    for (int t=0; t<num_steps; ++t)
    {
        DataType max_est_cell = 0.;
        int mesh_index = mesh_indices[t];
        int num_cells  = this->get_num_cells(mesh_index, 0);

        for (int c=0; c<num_cells; ++c)
        {
            if (this->get_mesh(mesh_index, 0)->cell_is_local(c))
            {
                if (this->time_indicator_[t][c] >= max_est_cell)
                {
                    max_est_cell = this->time_indicator_[t][c];
                }
            }
        }
        local_reduced_time_indicator[t] = max_est_cell;
    }

    // compute global reduced time indicator
    MPI_Allreduce ( &local_reduced_time_indicator[0],
    &this->reduced_time_indicator_[0], num_steps, MPI_DOUBLE, MPI_MAX, comm_ );

    // TODO: evtl buggy
    // space indicator
    for (int m=0; m<num_mesh; ++m)
    {
        int first_t   = this->get_first_step_for_mesh(m);
        int last_t    = this->get_last_step_for_mesh(m);
        int num_cells = this->get_num_cells(first_t);

        std::cout << "mesh " << m << " first time interval " << first_t << "
    last interval " << last_t << " num_cells " << num_cells << std::endl; for
    (int c=0; c<num_cells; ++c)
        {
            if (this->get_mesh(first_t)->cell_is_local(c))
            {
                DataType max_est_interval = 0.;
                for (int t=first_t; t<=last_t; ++t)
                {
                    assert (m == mesh_indices[t]);
                    DataType delta_t = this->get_delta_t(t+1);
                    if (this->space_indicator_[t][c] / delta_t >=
    max_est_interval)
                    {
                        max_est_interval = this->space_indicator_[t][c] /
    delta_t;
                    }
                }
            }
            this->reduced_space_indicator_[m][c] = max_est_interval;
        }
    }
     * */
  }
}

template < class DataType >
void compute_space_time_estimators(
    int num_steps, int num_mesh, const std::vector< int > &num_cells_per_mesh,
    const std::vector< int > &indicator_mesh_indices,
    const std::vector< mesh::MeshPtr > &mesh_list,
    const std::vector< std::vector< DataType > > &element_residual_h,
    const std::vector< std::vector< DataType > > &element_residual_tau,
    const std::vector< std::vector< DataType > > &element_weight_h,
    const std::vector< std::vector< DataType > > &element_weight_tau,
    std::vector< std::vector< DataType > > &time_indicator,
    std::vector< std::vector< DataType > > &space_indicator,
    DataType local_time_estimator, DataType local_space_estimator) {
  assert(num_cells_per_mesh.size() == num_mesh);
  assert(indicator_mesh_indices.size() >= num_steps);
  assert(mesh_list.size() == num_mesh);

  // reset indicators and estimators
  time_indicator.clear();
  time_indicator.resize(num_steps);
  space_indicator.clear();
  space_indicator.resize(num_steps);

  for (int t = 0; t < num_steps; ++t) {
    int num_cells = num_cells_per_mesh[indicator_mesh_indices[t]];
    time_indicator[t].resize(num_cells, 0.);
    space_indicator[t].resize(num_cells, 0.);
  }

  local_time_estimator = 0.;
  local_space_estimator = 0.;

  const DataType scaling = 0.5;

  // compute indicators for each space-time slab -----------------------------
  // loop over all time steps
  for (int t = 0; t < num_steps; ++t) {
    int num_cells = num_cells_per_mesh[indicator_mesh_indices[t]];

    // loop over all cells
    for (int c = 0; c < num_cells; ++c) {
      // cell residual contribution
      space_indicator[t][c] =
          scaling * element_residual_h[t][c] * element_weight_h[t][c];
      time_indicator[t][c] =
          scaling * element_residual_tau[t][c] * element_weight_tau[t][c];
    }
  }

  // compute total error estimation
  for (int t = 0; t < num_steps; ++t) {
    int mesh_index = indicator_mesh_indices[t];
    int num_cells = num_cells_per_mesh[mesh_index];
    for (int c = 0; c < num_cells; ++c) {
      if (mesh_list[mesh_index]->cell_is_local(c)) {
        local_time_estimator += time_indicator[t][c];
        local_space_estimator += space_indicator[t][c];
      } else {
        // assert (this->space_indicator_[t][c] == 0.);
        // assert (this->time_indicator_[t][c] == 0.);
      }
    }
  }
}

template < class DataType >
void get_equation_contributions( int est_var, EstimatorMode est_mode, int num_eq, int num_steps,
                                 int num_mesh, bool csi, const std::vector< std::vector< DataType > > &hK,
                                 const std::vector< int > &indicator_mesh_indices,
                                 const std::vector< std::vector< std::vector< DataType > > > &est_cell_residual,
                                 const std::vector< std::vector< std::vector< DataType > > > &est_cell_timejump,
                                 const std::vector< std::vector< std::vector< DataType > > > &est_interface,
                                 std::vector< DataType > &primal_cell_residual_contrib, 
                                 std::vector< DataType > &primal_cell_timejump_contrib, 
                                 std::vector< DataType > &primal_interface_contrib,
                                 std::vector< DataType > &dual_cell_residual_contrib, 
                                 std::vector< DataType > &dual_cell_timejump_contrib, 
                                 std::vector< DataType > &dual_interface_contrib)
{
  int off = 0;
  if (est_var == 1) 
  {
    off = 2;
  }

  /*
  bool multiply_by_weight = false;
  if (est_mode == EstimatorMode::DWR && csi)
  {
    NOT_YET_IMPLEMENTED;
    multiply_by_weight = true;
  }
  */
  
  primal_cell_residual_contrib.clear();
  primal_cell_timejump_contrib.clear();
  primal_interface_contrib.clear();
  primal_cell_residual_contrib.resize(num_eq, 0.);
  primal_cell_timejump_contrib.resize(num_eq, 0.);
  primal_interface_contrib.resize(num_eq, 0.);

  dual_cell_residual_contrib.clear();
  dual_cell_timejump_contrib.clear();
  dual_interface_contrib.clear();
  dual_cell_residual_contrib.resize(num_eq, 0.);
  dual_cell_timejump_contrib.resize(num_eq, 0.);
  dual_interface_contrib.resize(num_eq, 0.);
  
  // combine indicators
  // loop over all time steps
  for (int t = 0; t < num_steps; ++t) 
  {
    int num_cells = est_cell_residual[t].size();
    int mesh_index = indicator_mesh_indices[t];

    assert(est_cell_residual[t].size() == num_cells);
    assert(est_cell_timejump[t].size() == num_cells);
    assert(est_interface[t].size() == num_cells);

    // loop over all cells
    for (int c = 0; c < num_cells; ++c) 
    {
      //const DataType h = hK[mesh_index][c];
      //const DataType inv_h = 1. / h;

      assert(est_cell_residual[t][c].size() == 8 * num_eq);
      assert(est_cell_timejump[t][c].size() == 8 * num_eq);
      assert(est_interface[t][c].size() == 8 * num_eq);

      // loop over all equations
      for (int e = 0; e < num_eq; ++e) 
      {
        if (est_mode == EstimatorMode::DWR) 
        {
          if (!csi) 
          {
            primal_cell_residual_contrib[e] += std::abs(est_cell_residual[t][c][4 * e + off]);
            dual_cell_residual_contrib[e] += std::abs(est_cell_residual[t][c][4 * (e + num_eq) + off]);
            primal_interface_contrib[e] += std::abs(est_interface[t][c][4 * e + off]);
            dual_interface_contrib[e] += std::abs(est_interface[t][c][4 * (e + num_eq) + off]);
            primal_cell_timejump_contrib[e] += std::abs(est_cell_timejump[t][c][4 * e + off]);
            dual_cell_timejump_contrib[e] += std::abs(est_cell_timejump[t][c][4 * (e + num_eq) + off]);
          }
          else
          {
            NOT_YET_IMPLEMENTED;
          }
        }
        else if (est_mode == EstimatorMode::STD_RESIDUAL)
        {
          primal_cell_residual_contrib[e] += std::abs(est_cell_residual[t][c][4 * e + off]);
          dual_cell_residual_contrib[e] += std::abs(est_cell_residual[t][c][4 * (e + num_eq) + off]);
          primal_interface_contrib[e] += std::abs(est_interface[t][c][4 * e + off]);
          dual_interface_contrib[e] += std::abs(est_interface[t][c][4 * (e + num_eq) + off]);
          primal_cell_timejump_contrib[e] += std::abs(est_cell_timejump[t][c][4 * e + off]);
          dual_cell_timejump_contrib[e] += std::abs(est_cell_timejump[t][c][4 * (e + num_eq) + off]);
        }
      }
    }
  }
}

template <class T>
void determine_sizes_and_offsets (const MPI_Comm& comm, const std::vector< T > &local_values,
                                  int &local_size, int &global_size, int &local_offset)
{
  assert (comm != MPI_COMM_NULL);
  int my_rank = -1;
  int num_procs = -1;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &num_procs);
  
  assert (my_rank >= 0);
  assert (num_procs >= 1);
  
  // determine local data size
  int my_num_cells = local_values.size();
  local_size = 0;
  for (size_t l=0; l != my_num_cells; ++l)
  {
    local_size += local_values[l].size();
  }
  
  std::vector< int > all_sizes (num_procs, -1);
  MPI_Allgather(&local_size, 1, MPI_INT, &all_sizes[0], 1, MPI_INT, comm);

  local_offset = 0;
  for (size_t l = 0; l < my_rank; ++l)
  {
    local_offset += all_sizes[l];
  }
  global_size = local_offset;
  for (size_t l = my_rank; l < num_procs; ++l)
  {
    global_size += all_sizes[l];
  }
}

template <class DataType>
void write_space_indicators_to_hdf5 (const MPI_Comm& comm, const int my_offset, const int global_size,
                                     const std::string &filename, const std::string &groupname, const std::string &datasetname,
                                     const std::vector< std::vector< DataType > > &indicators)
{
#ifdef WITH_HDF5
  assert (comm != MPI_COMM_NULL);

  // open hdf5 file
  H5FilePtr file_ptr(new H5File(filename, "w", comm));
  H5GroupPtr group_ptr(new H5Group(file_ptr, groupname, "w"));
  H5DatasetPtr dataset_ptr(new H5Dataset(group_ptr, global_size, datasetname, "w", &indicators[0][0]));
  
  const size_t num_cells = indicators.size();
  int offset = my_offset;
  for (size_t c = 0; c < num_cells; ++c)
  {
    const int cell_size = indicators[c].size();
    dataset_ptr->write(cell_size, offset, &indicators[c][0]);
    offset += cell_size;
  }

#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

template <class DataType>
void read_space_indicators_from_hdf5 (const MPI_Comm& comm, const int my_offset, const int global_size,
                                      const std::string &filename, const std::string &groupname, const std::string &datasetname,
                                      std::vector< std::vector< DataType > > &indicators)
{
#ifdef WITH_HDF5
  assert (comm != MPI_COMM_NULL);
  assert (indicators.size() > 0);
  assert (indicators[0].size() > 0);

  H5FilePtr file_ptr(new H5File(filename, "r", comm));
  H5GroupPtr group_ptr(new H5Group(file_ptr, groupname, "r"));
  H5DatasetPtr dataset_ptr(new H5Dataset(group_ptr, global_size, datasetname, "r", &indicators[0][0]));
  
  const size_t num_cells = indicators.size();
  int offset = my_offset;
  for (size_t c = 0; c < num_cells; ++c)
  {
    const int cell_size = indicators[c].size();
    dataset_ptr->read(cell_size, offset, &indicators[c][0]);
    offset += cell_size;
  }
#else
  throw "HiFlow was not compiled with HDF5 support!\n";
#endif
}

template <class DataType>
void write_spacetime_indicators_to_hdf5 (const MPI_Comm& comm, const size_t first_step_to_write, const size_t num_steps_to_write,
                                         const std::string &filename, const std::string &groupname, const std::string &dataset_prefix,
                                         const std::vector< std::vector< std::vector< DataType > > > &indicators)
{
  const int num_steps = indicators.size();
  const int last_step_to_write = first_step_to_write + num_steps_to_write - 1;
  
  assert (last_step_to_write < num_steps);
  
  // loop over al time steps
  for (int t = first_step_to_write; t <= last_step_to_write; ++t)
  {
    int local_size, global_size, local_offset;
    determine_sizes_and_offsets< std::vector<DataType> >  (comm, indicators[t], local_size, global_size, local_offset);
    
    std::string datasetname = dataset_prefix + "_" + std::to_string(t);
    write_space_indicators_to_hdf5<DataType> (comm, local_offset, global_size, filename,  groupname,  datasetname,  indicators[t]);
  }
}

template <class DataType>
void read_spacetime_indicators_from_hdf5 (const MPI_Comm& comm, const size_t first_step_to_read, const size_t num_steps_to_read,
                                          const std::string &filename, const std::string &groupname, const std::string &dataset_prefix,
                                          std::vector< std::vector< std::vector< DataType > > > &indicators)
{
  const int num_steps = indicators.size();
  const int last_step_to_read = first_step_to_read + num_steps_to_read - 1;
  
  assert (last_step_to_read < num_steps);
  
  // loop over al time steps
  for (int t = first_step_to_read; t <= last_step_to_read; ++t)
  {
    int local_size, global_size, local_offset;
    determine_sizes_and_offsets< std::vector<DataType> >  (comm, indicators[t], local_size, global_size, local_offset);
    
    std::string datasetname = dataset_prefix + "_" + std::to_string(t);
    read_space_indicators_from_hdf5<DataType> (comm, local_offset, global_size, filename,  groupname,  datasetname, indicators[t]);
  }
}

} // namespace hiflow
#endif
