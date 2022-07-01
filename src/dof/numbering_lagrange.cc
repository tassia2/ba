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

#include "dof/dof_fem_types.h"
#include "dof/numbering_lagrange.h"
#include "dof/dof_partition.h"
#include "dof/numbering_strategy.h"
#include "dof/fe_interface_pattern.h"
#include "dof/dof_interpolation_pattern.h"
#include "dof/dof_impl/dof_container.h"
#include "dof/dof_impl/dof_container_lagrange.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/fe_manager.h"
#include "fem/fe_mapping.h"
#include "fem/fe_reference.h"
#include "fem/fe_transformation.h"
#include "mesh/cell_type.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"
#include "mesh/types.h"

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/bandwidth.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/king_ordering.hpp>
#include <boost/graph/properties.hpp>

namespace hiflow {
namespace doffem {

template < class DataType, int DIM >
void log_debug_fe_pattern_info(const mesh::Entity &entity,
                               CRefElementSPtr< DataType, DIM > fe_type, int gdim) {
  LOG_DEBUG(2, "Cell index = " << entity.index()
                               << ", fe degree = " << fe_type->max_deg());
  std::vector< mesh::Coordinate > coord;
  entity.get_coordinates(coord);
  for (int v = 0; v < entity.num_vertices(); ++v) {
    LOG_DEBUG(2, "Vertex " << v << " = "
                           << string_from_pointer_range(
                                  &coord[gdim * v], &coord[gdim * (v + 1)]));
  }
}

template < class DataType, int DIM >
void log_compute_interpolation_start(
    const FEInterfacePattern< DataType, DIM >  &pattern,
    const mesh::Interface &interface, const mesh::Mesh *mesh) {
  LOG_DEBUG(1, "\nCOLLECTING INFORMATION FOR THE INTERPOLATION\n"
                   << "============================================\n"
                   << "FEInterfacePattern< DataType, DIM >  = \n"
                  /* << pattern << "\n"*/);
// TODO_COORD

  const mesh::Entity &master =
      mesh->get_entity(mesh->tdim(), interface.master_index());

  LOG_DEBUG(2, "Master Info");
  log_debug_fe_pattern_info(master, pattern.fe_type_master(), mesh->gdim());

  auto &fe_type_slaves = pattern.fe_type_slaves();

  for (int s = 0; s < pattern.num_slaves(); ++s) {
    const mesh::Entity &slave =
        mesh->get_entity(mesh->tdim(), interface.slave_index(s));

    LOG_DEBUG(2, "Slave " << s << " Info");
    log_debug_fe_pattern_info(slave, fe_type_slaves[s], mesh->gdim());
  }
}

template < class DataType, int DIM >
void log_interface_matrix(
    int level, const typename NumberingLagrange< DataType, DIM >::InterfaceMatrix
                   &interface_matrix) {
  std::stringstream sstr;
  sstr << std::setprecision(4);
  
  //std::cout << " interface matrix " << std::endl;
  for (typename NumberingLagrange< DataType, DIM >::InterfaceMatrix::const_iterator
           it = interface_matrix.begin(),
           end = interface_matrix.end();
       it != end; ++it) {
    sstr << "\t" << it->first << "\t->";

    for (int i = 0; i < it->second.size(); ++i) {
      sstr << "\t" << it->second[i];
    }

    const double sum =
        std::accumulate(it->second.begin(), it->second.end(), 0.);

    sstr << "\tRow sum = " << sum << "\n";
  }
  LOG_DEBUG(level, "\n" << sstr.str());
}

template < class DataType, int DIM >
void extract_interface_matrix( const typename NumberingLagrange< DataType, DIM >::InterpolationWeights &interpolation_weights,
                               const std::vector< DofID > &interface_dofs,
                               typename NumberingLagrange< DataType, DIM >::InterfaceMatrix &interface_matrix) 
{
  // Copy rows corresponding to interface dofs.
  for (size_t i = 0, end = interface_dofs.size(); i != end; ++i) 
  {
    const DofID dof = interface_dofs[i];
    interface_matrix.insert(std::make_pair(dof, interpolation_weights[dof]));
  }

#if 0
  // Filter small entries (not needed if interpolation_weights are already filtered?)
  for ( InterpolationMap::iterator it = interface_matrix.begin ( ), end = interface_matrix.end ( ); it != end; ++it )
  {
    for ( size_t i = 0, e_i = it->second.size ( ); i != e_i; ++i )
    {
      if ( std::abs ( it->second[i] ) < COEFFICIENT_EPS )
      {
        it->second[i] = 0.0;
      }
    }
  }
#endif
}

template < class DataType, int DIM >
void multiply_interface_matrices(
    const typename NumberingLagrange< DataType, DIM >::InterfaceMatrix &A,
    const typename NumberingLagrange< DataType, DIM >::InterfaceMatrix &B,
    typename NumberingLagrange< DataType, DIM >::InterfaceMatrix &C) 
{
  // It is assumed that the sizes of the matrices are as follows: A
  // = M x N; B = P x Q; where M is the number of interface dofs of
  // the constrained element, N is the number of cell dofs of the
  // constraining element, P is the number of interface dofs of the
  // constraining element, and Q is the number of cell dofs of the
  // constrained element. This means that the keys of B all lie in
  // the range 0..N-1, and the keys of A lie in the range 0..Q-1 .

  // Copy A into C, since they will have the same row indices.
  C = A;

  // Determine num columns of C = num columns of B (same for all rows).
  const int num_cols = B.begin()->second.size();

  // Resize all columns of C.
  for (typename NumberingLagrange< DataType, DIM >::InterfaceMatrix::iterator
           it = C.begin(),
           end = C.end();
       it != end; ++it) {
    it->second.clear();
    it->second.resize(num_cols, 0.);
  }

  // Matrix multiplication: triple loop. In index notation: 
  // C_ij = \sum{k}{A_ik * B_kj} . Here, i = rowC->first, k = rowB->first
  // and j loops through columns of B.
  // TODO: not clear if this is an efficient matrix-matrix multiplication.
  for (typename NumberingLagrange< DataType, DIM >::InterfaceMatrix::iterator
           rowC = C.begin(),
           endRowC = C.end();
       rowC != endRowC; ++rowC) {
    for (typename NumberingLagrange< DataType, DIM >::InterfaceMatrix::const_iterator
             rowB = B.begin(),
             endRowB = B.end();
         rowB != endRowB; ++rowB) {
      const DataType A_coef = A.find(rowC->first)->second[rowB->first];

      for (size_t j = 0; j != num_cols; ++j) 
      {
        rowC->second[j] += A_coef * rowB->second[j];
      }
    }
  }

#if 0 // old implementation
                std::map<int, std::vector<DataType> > map_m2m = map_m2v;
                for ( std::map<int, std::vector<DataType> >::iterator it = map_m2m.begin ( ); it != map_m2m.end ( ); ++it )
                {
                    it->second.resize ( master_ansatz.nb_dof_on_cell ( ) );
                    for ( int i = 0; i < it->second.size ( ); ++i )
                        it->second[i] = 0.;

                    for ( int vindex = 0; vindex < map_m2v[it->first].size ( ); ++vindex )
                    {
                        DataType weight = map_m2v[it->first][vindex];
                        if ( std::abs ( weight ) > COEFFICIENT_EPS )
                        {
                            for ( int mindex = 0; mindex < map_v2m[vindex].size ( ); ++mindex )
                            {
                                it->second[mindex] += weight * map_v2m[vindex][mindex];
                            }
                        }
                    }
                }
#endif
}

template < class DataType, int DIM >
void find_constrained_dofs( const typename NumberingLagrange< DataType, DIM >::InterfaceMatrix &matrix,
                            DataType tol, 
                            std::vector< int > &constrained) 
{
  constrained.clear();
  constrained.reserve(matrix.size());

  for (typename NumberingLagrange< DataType, DIM >::InterfaceMatrix::const_iterator it = matrix.begin(), 
       e_it = matrix.end(); it != e_it; ++it) 
  {
    for (size_t i = 0, end = it->second.size(); i != end; ++i) 
    {
      const DataType coef = it->second[i];

      // A dof is unconstrained if the corresponding row
      // contains exactly one entry == 1. Since the row sum is
      // guaranteed to be 1., we only check if each
      // coefficient is either 0 or 1 here.
      if ((std::abs(coef) > tol) && (std::abs(coef - 1.) > tol)) 
      {
        constrained.push_back(it->first);
        break;
      }
    }
  }
}

template < class DataType, int DIM >
void correct_number_unconstrained_master_dofs( typename NumberingLagrange< DataType, DIM >::InterfaceMatrix &matrix,
                                               DataType tol, 
                                               int target_num_unconstrained) 
{
  std::vector< int > constrained_dofs;
  find_constrained_dofs< DataType, DIM >(matrix, tol, constrained_dofs);

  const int num_unconstrained = matrix.size() - constrained_dofs.size();

  LOG_DEBUG(2, "Current number unconstrained dofs = "
                   << num_unconstrained
                   << "\nNeeded number unconstrained dofs = "
                   << target_num_unconstrained);

  assert(num_unconstrained <= target_num_unconstrained);

  // Sort the constrained dofs so that they can be eliminated in order of their
  // ID:s.
  std::sort(constrained_dofs.begin(), constrained_dofs.end());
  const int num_dofs_to_relax = target_num_unconstrained - num_unconstrained;

  for (std::vector< int >::const_iterator it = constrained_dofs.begin(),
       end = constrained_dofs.begin() + num_dofs_to_relax; it != end; ++it) 
  {
    typename NumberingLagrange< DataType, DIM >::InterfaceMatrix::iterator row = matrix.find(*it);
    assert(row != matrix.end());

    // Replace dependencies with identity row.
    row->second.assign(row->second.size(), 0.);
    row->second[row->first] = 1.;
  }
}

template < class DataType, int DIM >
void add_master_dof_interpolation_pattern(const typename NumberingLagrange< DataType, DIM >::InterfaceMatrix &matrix,
                                          DataType tol, DofInterpolationPattern<DataType> &pattern) 
{
  // Only add interpolations for constrained dofs.
  std::vector< int > constrained_dofs;
  find_constrained_dofs< DataType, DIM >(matrix, tol, constrained_dofs);

  std::vector< std::pair< int, DataType > > dependencies;

  for (std::vector< int >::const_iterator it = constrained_dofs.begin(),
       end = constrained_dofs.end(); it != end; ++it) 
  {
    dependencies.clear();

    typename NumberingLagrange< DataType, DIM >::InterfaceMatrix::const_iterator row_it = matrix.find(*it);
    assert(row_it != matrix.end());

    const int row = row_it->first;
    const DataType row_coef = row_it->second[row];

    for (size_t col = 0, end_col = row_it->second.size(); col != end_col; ++col) 
    {
      const DataType col_coef = row_it->second[col];

      // Add dependencies from columns with non-zero
      // weights, skipping the column corresponding to the constrained dof
      // itself.
      if (col != row && (std::abs(col_coef) > tol)) 
      {
        // Normalize weight with 1 - row_dof_coef, since
        // the weight for the left-hand-side constrained
        // dof might not be zero.
        // TODO: should it not always be zero, so that weight is always one?
        const double weight = static_cast< DataType >(col_coef / (1. - row_coef));
        dependencies.push_back(std::make_pair(col, weight));
      }
    }
    pattern.insert_interpolation_master(std::make_pair(row, dependencies));
  }
}

template < class DataType, int DIM >
void add_slave_interpolation_pattern(
    int slave,
    const typename NumberingLagrange< DataType, DIM >::InterfaceMatrix &matrix,
    DataType tol, DofInterpolationPattern<DataType> &pattern) 
{
  // On slave, all dofs are constrained, so no need to extract
  // them specifically.

  std::vector< std::pair< int, DataType > > dependencies;

  for (typename NumberingLagrange< DataType, DIM >::InterfaceMatrix::const_iterator
           row_it = matrix.begin(),
           end_it = matrix.end();
       row_it != end_it; ++row_it) 
  {
    dependencies.clear();

    const int row = row_it->first;

    for (size_t col = 0, end_col = row_it->second.size(); col != end_col; ++col) 
    {
      const double col_coef = static_cast< double >(row_it->second[col]);
      //std::cout << col_coef << " ";
      
      // TODO: it is assumed here that the entry for which
      // col == row will be zero, and so automatically
      // skipped. No normalization is necessary here.
      if (std::abs(col_coef) > tol) 
      {
        dependencies.push_back(std::make_pair(col, col_coef));
      }
    }
    // std::cout << std::endl;
    // if dependencies = {(j,1.)}, then (row, j) is added to the identification list of pattern.slave[slave_id] 
    // later on, in indentify_common_dofs(), global dof identification is based on the pattern.slave[s].identification_list 
    // Note (Philipp G.): Whta about the case of BDM, where dof are integrals over facet of phi*n with outer normal n
    // Here, neighboring cells lead to -n and n, i.e. should also be the case {(j,-1)} be considered as identification??
    pattern.insert_interpolation_slave(slave, std::make_pair(row, dependencies));
  }
}

template < class DataType, int DIM >
void NumberingLagrange< DataType, DIM >::number_locally(DOF_ORDERING order, bool ordering_fe_cell) 
{
  this->ordering_fe_cell_ = ordering_fe_cell;
  
  const int verbatim_mode = 0; // 0 -> no output
  // 1 -> some output
  // initial numbering, some DoFs may have multiple DoF IDs
  initial_numbering();

  // find DoF IDs that lie on same coordinate and should be identified
  identify_common_dofs();

  // sort the dof_identification_list
  this->dof_interpolation_->sort();

  // Determine equivalence classes and modify numer_cell_2_global_ s.t.
  // each numer_cell_2_global_ entry maps to one representer of the equivalence class
  //
  // The following algorithm is based on the description given in the book
  // "Numerical Recipes in C", Second edition,
  // W. Press, S. Teukolsky, W. Vetterling, B. P. Flannery, Cambridge University
  // Press, Pages 345 / 346

  size_t cntr = 0;

#if 0
CHANGE
(*this->numer_cell_2_factor_)
#endif

  while (cntr < this->dof_interpolation_->dof_identification_list().size()) 
  {
    int cntr_val = this->dof_interpolation_->dof_identification_list()[cntr].first;

    while ((*this->numer_cell_2_global_)[cntr_val] != cntr_val) 
    {
      cntr_val = (*this->numer_cell_2_global_)[cntr_val];
    }
    int cntr_val2 = this->dof_interpolation_->dof_identification_list()[cntr].second;

    while ((*this->numer_cell_2_global_)[cntr_val2] != cntr_val2) 
    {
      cntr_val2 = (*this->numer_cell_2_global_)[cntr_val2];
    }
    if (cntr_val != cntr_val2) 
    {
      (*this->numer_cell_2_global_)[cntr_val2] = cntr_val;
    }
    ++cntr;
  }

  const size_t e_cntr = this->numer_cell_2_global_->size();
  for (cntr = 0; cntr != e_cntr; ++cntr) 
  {
    while ((*this->numer_cell_2_global_)[cntr] != (*this->numer_cell_2_global_)[(*this->numer_cell_2_global_)[cntr]]) 
    {
      (*this->numer_cell_2_global_)[cntr] = (*this->numer_cell_2_global_)[(*this->numer_cell_2_global_)[cntr]];
    }
  }

  // Need equivalence reduction also in dof constraints
  this->dof_interpolation_->apply_permutation(*(this->numer_cell_2_global_));

  // 2. Consecutive numbering

  std::vector< int > numer_copy(*(this->numer_cell_2_global_));
  std::sort(numer_copy.begin(), numer_copy.end());

  std::vector< int >::iterator it =
      std::unique(numer_copy.begin(), numer_copy.end());
  size_t numer_copy_size = it - numer_copy.begin();

  std::vector< int > permutation(this->numer_cell_2_global_->size(), -5);
  for (size_t i = 0; i != numer_copy_size; ++i) {
    permutation[numer_copy[i]] = i;
  }

  LOG_DEBUG(2, "Numer size = " << this->numer_cell_2_global_->size());
  LOG_DEBUG(2, "Numer = " << string_from_range(this->numer_cell_2_global_->begin(),
                                               this->numer_cell_2_global_->end()));
  LOG_DEBUG(2, "Numer copy size = " << numer_copy.size());
  LOG_DEBUG(2, "Numer copy = " << string_from_range(numer_copy.begin(),
                                                    numer_copy.end()));
  LOG_DEBUG(2, "Permutation size = " << permutation.size());
  LOG_DEBUG(2, "Permutation = " << string_from_range(permutation.begin(),
                                                     permutation.end()));

  // applay permutation on numer_cell_2_global_
  this->apply_permutation(permutation);

  if (verbatim_mode > 0) {
    std::cout << std::endl;
    std::cout << "INTERPOLATION INFORMATION" << std::endl;
    std::cout << "=========================" << std::endl;
    this->dof_interpolation_->backup(std::cout);
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "NUMER-FIELD" << std::endl;
    std::cout << "===========" << std::endl;
    for (int i = 0; i < this->numer_cell_2_global_->size(); ++i) {
      //                    std::cout << i << "\t ->    " << this->numer_cell_2_global_->at (
      //                    i ) << std::endl;
      std::cout << std::endl;
    }

    if (this->dof_interpolation_->size() > 1) {
      std::cout << "# DoFs:              " << this->local_nb_dofs_total_
                << std::endl;
      std::cout << "# interpolated DoFs: " << this->dof_interpolation_->size()
                << std::endl;
      std::cout << "# real DoFs:         "
                << this->local_nb_dofs_total_ -
                       this->dof_interpolation_->size()
                << std::endl;
    } else {
      std::cout << "# DoFs: " << this->local_nb_dofs_total_ << std::endl;
    }
  }

  // 3. Apply optimization to ordering according to order
  if (order != DOF_ORDERING::HIFLOW_CLASSIC) {
    

    // Some Typedefs for clearer notation
    using namespace boost;
    using namespace std;
    typedef adjacency_list< vecS, vecS, undirectedS,
                            property< vertex_color_t, default_color_type,
                                      property< vertex_degree_t, int > > >
        Graph;
    typedef graph_traits< Graph >::vertex_descriptor Vertex;
    typedef graph_traits< Graph >::vertices_size_type size_type;

    // Create Boost graph of local dof couplings
    Graph G(this->local_nb_dofs_total_);
    {
      // Create graph topology
      std::vector< int > dof_ind_test, dof_ind_trial;
      std::vector< SortedArray< int > > raw_struct(this->local_nb_dofs_total_);

      // loop over every cell (including ghost cells)
      mesh::EntityIterator mesh_it = this->mesh_->begin(this->mesh_->gdim());
      mesh::EntityIterator e_mesh_it = this->mesh_->end(this->mesh_->gdim());
      while (mesh_it != e_mesh_it) 
      {
        // loop over test variables
        for (int test_var = 0, tv_e = this->nb_fe_; test_var != tv_e; ++test_var) 
        {
          // get dof indices for test variable
          this->dof_->get_dofs_on_cell(test_var, (*mesh_it).index(), dof_ind_test);

          // loop over trial variables
          for (int trial_var = 0, vt_e = this->nb_fe_; trial_var != vt_e;
               ++trial_var) 
          {

            // get dof indices for trial variable
            this->dof_->get_dofs_on_cell(trial_var, (*mesh_it).index(), dof_ind_trial);

            // detect couplings
            for (size_t i = 0, i_e = dof_ind_test.size(); i != i_e; ++i) 
            {
              for (size_t j = 0, j_e = dof_ind_trial.size(); j != j_e; ++j) 
              {
                raw_struct[dof_ind_test[i]].find_insert(dof_ind_trial[j]);
              } // for (int j=0;...
            }   // for (int i=0;...
          }
        }
        // next cell
        ++mesh_it;
      } // while (mesh_it != ...

      for (size_t k = 0, k_e = raw_struct.size(); k != k_e; ++k) 
      {
        for (size_t l = 0, l_e = raw_struct[k].size(); l != l_e; ++l) 
        {
          add_edge(k, raw_struct[k][l], G);
        }
      }
    }

    graph_traits< Graph >::vertex_iterator ui, ui_end;

    property_map< Graph, vertex_degree_t >::type deg = get(vertex_degree, G);
    for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui) 
    {
      deg[*ui] = degree(*ui, G);
    }

    property_map< Graph, vertex_index_t >::type index_map =
        get(vertex_index, G);

    LOG_INFO("Original bandwidth", bandwidth(G));

    std::vector< Vertex > inv_perm(num_vertices(G));
    std::vector< int > perm(num_vertices(G));

    switch (order) {
    case DOF_ORDERING::HIFLOW_CLASSIC: {
      LOG_INFO("DoF reordering strategy", "classic");
      break;
    }

    case DOF_ORDERING::CUTHILL_MCKEE: {
      LOG_INFO("DoF reordering strategy", "Cuthill-McKee");
      // reverse cuthill_mckee_ordering
      cuthill_mckee_ordering(G, inv_perm.rbegin(), get(vertex_color, G),
                             make_degree_map(G));
      break;
    }

    case DOF_ORDERING::KING: {
      LOG_INFO("DoF reordering strategy", "King");
      // king_ordering
      king_ordering(G, inv_perm.rbegin());
      break;
    }

    default: { break; }
    }

    for (size_type c = 0; c != inv_perm.size(); ++c) {
      perm[index_map[inv_perm[c]]] = static_cast< int >(c);
    }
    LOG_INFO(
        "New Bandwidth",
        bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0])));

    this->apply_permutation(perm);
  }

  this->dof_->set_applied_number_strategy(true);

  if (verbatim_mode > 0) {
    std::cout << std::endl;
    std::cout << "INTERPOLATION INFORMATION" << std::endl;
    std::cout << "=========================" << std::endl;
    this->dof_interpolation_->backup(std::cout);
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "NUMER-FIELD" << std::endl;
    std::cout << "===========" << std::endl;
    for (int i = 0; i < this->numer_cell_2_global_->size(); ++i) {
      //                    std::cout << i << "\t ->    " << this->numer_cell_2_global_->at (
      //                    i ) << std::endl;
    }
    std::cout << std::endl;

    if (this->dof_interpolation_->size() > 1) {
      std::cout << "# DoFs:              " << this->local_nb_dofs_total_
                << std::endl;
      std::cout << "# interpolated DoFs: " << this->dof_interpolation_->size()
                << std::endl;
      std::cout << "# real DoFs:         "
                << this->local_nb_dofs_total_ -
                       this->dof_interpolation_->size()
                << std::endl;
    } else {
      std::cout << "# DoFs: " << this->local_nb_dofs_total_ << std::endl;
    }
  }
}

/// create initial DoF numbering ignoring that some DoFs coincide

template < class DataType, int DIM >
void NumberingLagrange< DataType, DIM >::initial_numbering() 
{
  this->numer_cell_2_global_offsets_->clear();
  //this->numer_cell_2_global_offsets_per_cell_->clear();
  //this->numer_cell_2_global_offsets_per_cell_->reserve(this->mesh_->num_entities(this->fe_manager_->tdim()));
    
  int numer_size = 0; 
  this->numer_cell_2_global_offsets_->resize(this->nb_fe_);
  for (int fe_ind = 0; fe_ind < this->nb_fe_; ++fe_ind) 
  {
    this->numer_cell_2_global_offsets_->at(fe_ind).resize(this->mesh_->num_entities(this->fe_manager_->tdim()),0);
  }

  if (this->ordering_fe_cell_)
  {
    // loop over vars
    for (int fe_ind = 0; fe_ind < this->nb_fe_; ++fe_ind) 
    {
      // loop over cells
      int ctr = 0;
      for (mesh::EntityIterator
           it = this->mesh_->begin(this->fe_manager_->tdim()),
           e_it = this->mesh_->end(this->fe_manager_->tdim());
           it != e_it; ++it) 
      {
        const int nb_cell_dofs = this->fe_manager_->nb_dof_on_cell(it->index(), fe_ind);
        (*this->numer_cell_2_global_offsets_)[fe_ind][ctr] = numer_size;
        numer_size += nb_cell_dofs;
        ctr++;
      }
    }
  }
  else
  {
    // loop over cells
    int ctr = 0;
    for (mesh::EntityIterator
         it = this->mesh_->begin(this->fe_manager_->tdim()),
         e_it = this->mesh_->end(this->fe_manager_->tdim());
         it != e_it; ++it) 
    {
      // loop over vars
      for (int fe_ind = 0; fe_ind < this->nb_fe_; ++fe_ind) 
      {
        const int nb_cell_dofs = this->fe_manager_->nb_dof_on_cell(it->index(), fe_ind);
        (*this->numer_cell_2_global_offsets_)[fe_ind][ctr] = numer_size;
        numer_size += nb_cell_dofs;
      }
      ctr++;
    }
  }

/*  
  numer_size = 0;
  for (mesh::EntityIterator
             it = this->mesh_->begin(this->fe_manager_->tdim()),
             e_it = this->mesh_->end(this->fe_manager_->tdim());
             it != e_it; ++it) 
  {
    const int nb_cell_dofs = this->fe_manager_->nb_dof_on_cell(it->index());
    (*this->numer_cell_2_global_offsets_per_cell_).push_back(numer_size);
    numer_size += nb_cell_dofs;
  }
*/

// CHANGE ??

  // initialize numer_cell_2_global_
  this->numer_cell_2_global_->clear();
//  this->numer_cell_2_factor_.clear();
  
  this->numer_cell_2_global_->resize(numer_size);
//  this->numer_cell_2_factor_->resize(numer_size, 1.);

  for (size_t i = 0; i != numer_size; ++i) 
  {
    (*this->numer_cell_2_global_)[i] = i;
  }
}

/// common dofs are identified and unified
/// in this function, dof coordinates are not used explicitely

template < class DataType, int DIM >
void NumberingLagrange< DataType, DIM >::identify_common_dofs() {

  // prepare interface list

  mesh::InterfaceList interface_list = mesh::InterfaceList::create(this->mesh_);
  // clear old interpolation and identification information

  this->dof_interpolation_->clear_entries();
  int ctr = 0;
  
  // loop over interfaces
  for (mesh::InterfaceList::const_iterator interface = interface_list.begin(), e_interface = interface_list.end();
       interface != e_interface; ++interface) 
  {
    // loop over vars
    for (size_t fe_ind = 0; fe_ind != this->nb_fe_; ++fe_ind) 
    {
      // only identify continuous vars

      if (!this->fe_manager_->is_dg(fe_ind)) 
      {
        if (ctr == 0)
        {
            LOG_INFO("identify dofs of FE", fe_ind);
        }
        // calculate interface pattern
      
        // FE ansatz of master cell
        auto master_type = this->fe_manager_->get_fe(interface->master_index(), fe_ind);

        // FE ansatz of slave cells
        std::vector< CRefElementSPtr< DataType, DIM > > slave_types;
        slave_types.reserve(interface->num_slaves());
        for (size_t i = 0, e_i = interface->num_slaves(); i != e_i; ++i) 
        {
          slave_types.push_back(this->fe_manager_->get_fe(interface->slave_index(i), fe_ind));
        }

        // definition of the interface pattern

        mesh::InterfacePattern pre_pattern = mesh::compute_interface_pattern(this->mesh_, *interface);
        FEInterfacePattern< DataType, DIM > pattern(pre_pattern, master_type, slave_types);

        // only treat interfaces between at least two cells (-> no boundaries)
        if (pre_pattern.num_slaves() > 0) 
        {
          // add new interpolation definition for new interface patterns if
          // needed
          typename std::map< FEInterfacePattern< DataType, DIM > ,
                             DofInterpolationPattern<DataType> >::const_iterator interpolation_it;

          interpolation_it = interface_patterns_interpolation_.find(pattern); 

          if (interpolation_it == interface_patterns_interpolation_.end()) 
          {
            // pattern not found in interface_patterns_interpolation_ -> compute new dofinterpolation pattern
            DofInterpolationPattern<DataType> interpolation;
            compute_interpolation(pattern, *interface, interpolation);
            
            interface_patterns_interpolation_.insert( std::make_pair(pattern, interpolation));
            interpolation_it = interface_patterns_interpolation_.find(pattern);
          }

          DofInterpolationPattern<DataType> const &interpolation = interpolation_it->second;

          // ------------------------------------------------------------------------------
          // apply interpolation rules to DoFs
          // ------------------------------------------------------------------------------

          // ------ get DoF IDs of master and slave cells --------------------------------

          std::vector< DofID > dof_master;
          this->dof_->get_dofs_on_cell(fe_ind, interface->master_index(), dof_master);

          std::vector< std::vector< DofID > > dof_slaves;
          dof_slaves.resize(pattern.num_slaves());
          std::vector< int > dof_offset_on_slave_wrt_fe(pattern.num_slaves(), 0);
          
          for (size_t s = 0, e_s = pattern.num_slaves(); s != e_s; ++s)
          {
            this->dof_->get_dofs_on_cell(fe_ind, interface->slave_index(s), dof_slaves[s]);
          
            for (size_t fe_slave = 0; fe_slave < fe_ind; ++fe_slave)
            {
              dof_offset_on_slave_wrt_fe[s] += this->dof_->nb_dofs_on_cell(fe_slave, interface->slave_index(s));
            }
          }

          // ------ identification of DoFs (Slave <-> Master) -----------------------------
          for (size_t s = 0, e_s = pattern.num_slaves(); s != e_s; ++s) 
          {
            std::vector< std::pair< DofID, DofID > > const 
                &identification_list = interpolation.interpolation_slave(s).dof_identification_list();
            std::vector< std::pair< DofID, DataType > > const 
                &identification_factors = interpolation.interpolation_slave(s).dof_identification_factors();

            assert (identification_list.size() == identification_factors.size());
            
            const int cell_index_slave = interface->slave_index(s); 
            //const int cell_index_master = interface->master_index();
                        
            for (size_t i = 0, e_i = identification_list.size(); i != e_i; ++i) 
            {
              const DofID cell_dof_id_slave = identification_list[i].first;
              const DofID cell_dof_id_master = identification_list[i].second;
                           
              const DofID dof_id_slave = dof_slaves[s][cell_dof_id_slave];
              const DofID dof_id_master = dof_master[cell_dof_id_master];
              const DataType dof_factor = identification_factors[i].second;
              
              // dof_slave = dof_factor * dof_master
              this->dof_interpolation_->insert_dof_identification(dof_id_slave, dof_id_master, dof_factor);
              if (DEBUG_LEVEL > 0) 
              {
                std::cout << "master dof " << dof_id_master << " <=> slave dof " << dof_id_slave << " : dof_factor " << dof_factor << std::endl;
              }
              // save dof_factor for each dof on each slave cell  
              assert (cell_index_slave < this->cell_2_dof_factor_->size());
              assert (dof_offset_on_slave_wrt_fe[s]+cell_dof_id_slave 
                      < (*this->cell_2_dof_factor_)[cell_index_slave].size());
                      
              //assert (cell_dof_id_master < (*this->cell_2_dof_factor_)[cell_index_master].size());
              //assert (cell_index_master < this->cell_2_dof_factor_->size());
              
              // TODO: BUG?
              (*this->cell_2_dof_factor_)[cell_index_slave][dof_offset_on_slave_wrt_fe[s]+cell_dof_id_slave] = dof_factor;
              //OLD: (*this->cell_2_dof_factor_)[cell_index_master][cell_dof_id_master] = 1.;
            }
          }
          // result: dof_interpolation gets pairs of master/slave dof ids that should be identified
          
          // ------- interpolation of DoFs (Master <-> Master) ----------------------------
          
          // get master interpolation list
          DofInterpolation<DataType> const &int_master = interpolation.interpolation_master();
          
          // loop over interpolated Dofs
          for (typename DofInterpolation<DataType>::const_iterator it = int_master.begin(), e_it = int_master.end(); it != e_it; ++it) 
          {
            // get DoF ID of interpolated DoF : map local to global id
            DofID dof_id_master = dof_master[it->first];

            // get DoF IDs of interpolating DoFs
            std::vector< std::pair< DofID, DataType > > sum = it->second;
            
            // replace interpolating dof-id by dof-id from dof_master 
            // Why? -> because dof-ids in int_master are local, but we need global ones in dof_interpolation_ 
            for (size_t md = 0, e_md = sum.size(); md != e_md; ++md)
            {
              sum[md].first = dof_master[sum[md].first];
            }

            // add interpolation definition to DofInterpolation
            bool status;
            status = this->dof_interpolation_->push_back(make_pair(dof_id_master, sum));
            // TODO (Staffan): Is the following assert correct? (see below)

            assert(status);
          }
          // TODO: refactor master-master and slave-master translation into common function

          // ------ interpolation of DoFs (Slave <-> Master) ------------------------------

          // loop over slaves
          for (size_t s = 0, e_s = pattern.num_slaves(); s != e_s; ++s) 
          {
            // loop over interpolated dofs
            DofInterpolation<DataType> const &int_slave = interpolation.interpolation_slave(s);
            for (typename DofInterpolation<DataType>::const_iterator it = int_slave.begin(), e_it = int_slave.end(); it != e_it; ++it) 
            {
              // get DoF ID of interpolated slave DoF
              DofID dof_id_slave = dof_slaves[s][it->first];

              // get DoF IDs of interpolating master DoFs
              std::vector< std::pair< DofID, DataType > > sum = it->second;
              for (size_t md = 0, e_md = sum.size(); md != e_md; ++md)
                sum[md].first = dof_master[sum[md].first];

              // add interpolation definition to DofInterpolation
              this->dof_interpolation_->push_back(make_pair(dof_id_slave, sum));
              // Note (staffan): This assert is not correct, at least
              // in 3d, since constraints for dofs on hanging edges
              // will be created at least twice.

              // assert ( status );
            }
          } // for (int s=0; s<pattern.num_slaves(); ++s)
        }   // if (pre_pattern.num_slaves() >= 1)
      }     // if (fe_manager_->get_ca(fe_ind) == true)
    }       // for (int fe_ind=0; fe_ind<fe_manager_->get_nb_fe_ind(); ++fe_ind)
    ctr++;
  }         // for (interface = interface_list.begin();...)
}

template < class DataType, int DIM >
void NumberingLagrange< DataType, DIM >::print_interface_patterns() const 
{
  typename std::map< FEInterfacePattern< DataType, DIM > ,
                     DofInterpolationPattern<DataType> >::const_iterator it;

  std::cout << "Interface-Modes:" << std::endl;
  for (it = interface_patterns_interpolation_.begin();
       it != interface_patterns_interpolation_.end(); ++it) 
  {
    std::cout << it->first << std::endl;
    std::cout << it->second << std::endl;
  }
}

/// \brief computes the interpolation description for given interface mode
/// \details the interpolation description is given by ...
/// \param[in] pattern the FEInterfacePattern< DataType, DIM >  that describes the neighbouring
///                    cells at the considered interface
/// \return the interface mode interpolation
/// \see DofInterpolationPattern<DataType>

template < class DataType, int DIM >
void NumberingLagrange< DataType, DIM >::compute_interpolation(
    const FEInterfacePattern< DataType, DIM >  &pattern,
    const mesh::Interface &interface,
    DofInterpolationPattern<DataType> &interpolation) 
{
  // Log general information for this interpolation.
  log_compute_interpolation_start< DataType >(pattern, interface, this->mesh_);

  const mesh::TDim tdim = this->mesh_->tdim();

  // TODO: check reasonable tolerances
  const DataType COEFFICIENT_EPS = 1.e3 * std::numeric_limits< DataType >::epsilon();

  // Prepare output argument.
  interpolation.set_number_slaves(pattern.num_slaves());

  // Find degree and ansatz of interface.
  int which_slave = -1;
  int iface_master_facet_nr = interface.master_facet_number();
  int pattern_master_facet_nr = pattern.master_facet_number();
  assert(iface_master_facet_nr == pattern_master_facet_nr);

  int virtual_facet_nr = -1;

  const int interface_deg = pattern.get_interface_degree(&which_slave);
  CRefElementSPtr< DataType, DIM > virtual_ansatz(nullptr);
  if (which_slave == -1) 
  {
    // master has smaller or equal degree than slave(s)
    virtual_ansatz = pattern.fe_type_master();
    virtual_facet_nr = iface_master_facet_nr;
    //std::cout << "virtual ansatz = master ansatz, virtual facet nr = " << virtual_facet_nr << std::endl; 
  } 
  else  
  {
    // slave has smallest degree
    virtual_ansatz = pattern.fe_type_slaves()[which_slave];
    virtual_facet_nr = interface.slave_facet_number(which_slave);
    //std::cout << "virtual ansatz = slave ansatz " << which_slave << ", virtual facet nr = " << virtual_facet_nr << std::endl; 
  }

  //LOG_DEBUG(2, "Interface degree = " << interface_deg);
  if (which_slave == -1) 
  {
    //LOG_DEBUG(2, "Dominating cell: master");
  } 
  else 
  {
    //LOG_DEBUG(2, "Dominating cell: slave " << which_slave);
  }
  //LOG_DEBUG(2, "Virtual FE Ansatz = " << virtual_ansatz->name());

  // ***************************************************************************
  // Master <-> Master interpolation
  // ***************************************************************************

  ////////////////
  // 1. Compute master -> master interpolation matrix, as product of
  // master -> virtual and virtual -> master interface matrices.
  // if same fe space on both cells -> identity matrix

  const mesh::Entity &master_cell = this->mesh_->get_entity(tdim, interface.master_index());
  auto master_ansatz = pattern.fe_type_master();

  CCellTrafoSPtr< DataType, DIM > trafo_master = this->fe_manager_->get_cell_transformation(master_cell.index());
  int ref_master_facet_nr = trafo_master->phys2ref_facet_nr(pattern_master_facet_nr);

  // Master to virtual mapping.
  // weight_m2v(i,j) = master_dof_i ( virtual_phi_j )
  // i = 1, ..., num_master_dof
  // j = 1, ..., dim_virtual_ansatz 

  // if both ansatz spaces coincide and no hanging nodes in the mesh are present, there holds
  // weight_m2v(i,j) = 1 if virtual_phi_j is modal ansatz basis of master_dof i, 0 otherwise
  // compute_weights_general is restricted to this case and works for all FE (TODO)
  // compute_weights_lagrange only works for Lagrange elements, but can also handle the case of hanging nodes and 
  // different polynomial degrees of the ansatz space
  
  InterpolationWeights weights_m2v;
  
  //LOG_DEBUG(2, "master - virtual ");
  //std::cout << "master - virtual " << std::endl;
  compute_weights(master_cell, virtual_ansatz, 
                  master_cell, master_ansatz,
                  this->mesh_->get_period(), weights_m2v);

#ifndef NDEBUG
  for (int l = 0; l < weights_m2v.size(); ++l) 
  {
    //LOG_DEBUG(0, string_from_range(weights_m2v[l].begin(), weights_m2v[l].end()));
    //std::cout << string_from_range(weights_m2v[l].begin(), weights_m2v[l].end()) << std::endl;
  }
#endif

  // Get master DoFs lying on the interface and which are shared with neighboring cells
  const std::vector< DofID > &master_dofs_on_interface = master_ansatz->get_dof_on_subentity(tdim - 1, ref_master_facet_nr);
  //const std::vector< DofID > &master_dofs_on_interface = master_ansatz.get_dof_on_subentity(tdim - 1, pattern.master_facet_number());
  //std::cout << "master dofs on interface " << string_from_range(master_dofs_on_interface.begin(), master_dofs_on_interface.end()) << std::endl; 
  
  // TODO: implement this routine to allow the case that not allm interface dofs are identified with their respective counterpart
  //const std::vector< DofID > &master_dofs_on_interface = master_ansatz.get_shared_dof_on_subentity(tdim - 1, pattern.master_facet_number());

  //LOG_DEBUG(2, "Master dofs on interface = "
    //               << string_from_range(master_dofs_on_interface.begin(),
      //                                  master_dofs_on_interface.end()));

  // m2v(i,j) = master_dof_i ( virtual_phi_j )
  // i \in shared interface master dofs
  // j = 1, ..., dim_virtual_ansatz 
  InterfaceMatrix m2v;
  extract_interface_matrix< DataType, DIM >(weights_m2v, master_dofs_on_interface, m2v);

  //LOG_DEBUG(2, "m2v =");
  //std::cout << "m2v" << std::endl;
  //log_interface_matrix< DataType, DIM >(0, m2v);

  // Virtual to master mapping.
  // weights_v2m(i,j) = virtual_dof_i ( master_phi_j )
  // i = 1, ..., num_virtual_dof
  // j = 1, ..., dim_master_ansatz 
  InterpolationWeights weights_v2m;
  
  //LOG_DEBUG(2, "virtual - master ");
  //LOG_DEBUG(2, "master facet nr " << ref_master_facet_nr << " " << pattern_master_facet_nr);
  compute_weights(master_cell, master_ansatz,
                  master_cell, virtual_ansatz,
                  this->mesh_->get_period(), weights_v2m);

  // Get virtual DoFs lying on the interface and which are shared with neighboring cells
  const std::vector< DofID > &virtual_dofs_on_interface = virtual_ansatz->get_dof_on_subentity(tdim - 1, ref_master_facet_nr);
  //const std::vector< DofID > &virtual_dofs_on_interface = virtual_ansatz->get_dof_on_subentity(tdim - 1, pattern.master_facet_number());
//  const std::vector< DofID > &virtual_dofs_on_interface = virtual_ansatz->get_shared_dof_on_subentity(tdim - 1, pattern.master_facet_number());


  //LOG_DEBUG(2, "Virtual dofs on interface = "
    //               << string_from_range(virtual_dofs_on_interface.begin(),
      //                                  virtual_dofs_on_interface.end()));
  if (DEBUG_LEVEL >= 0) 
  {
    //std::cout << "Virtual dofs on interface " << string_from_range(virtual_dofs_on_interface.begin(),
      //                                           virtual_dofs_on_interface.end()) << std::endl;
  }
  
  // v2m(i,j) = virtual_dof_i ( master_phi_j )
  // i \in shared interface virtual dofs
  // j = 1, ..., dim_master_ansatz 
  InterfaceMatrix v2m;
  extract_interface_matrix< DataType, DIM >(weights_v2m, virtual_dofs_on_interface, v2m);

  //LOG_DEBUG(0, "v2m =");
  //std::cout << "v2m" << std::endl;
  //log_interface_matrix< DataType, DIM >(0, v2m);

  // Multiply m2v and v2m to get m2m (master -> master interpolation).
  InterfaceMatrix m2m;
  multiply_interface_matrices< DataType, DIM >(m2v, v2m, m2m);

  //LOG_DEBUG(0, "m2m =");
  //std::cout << "m2m" << std::endl;
  //log_interface_matrix< DataType, DIM >(0, m2m);

  // RESULT:
  // m2m is a n_M x n_M matrix, with n_M denoting the number of dofs on the master cell
  // if i is a master interface dof, then
  //    m2m_(ij) = sum_(k is virtual interface dof) master_dof_i (virtual_phi_k) * virtual_dof_k (master_phi_j)  
  // otherwise, 
  //    m2m_(ij) = 0
  
  ////////////////
  // 2. Correct the number of unconstrained "real" DoFs on interface.
  const int needed_num_unconstrained_dofs = virtual_ansatz->nb_dof_on_subentity(tdim - 1, ref_master_facet_nr);
  //const int needed_num_unconstrained_dofs = virtual_ansatz->nb_shared_dof_on_subentity(tdim - 1, pattern.master_facet_number());
  //const int needed_num_unconstrained_dofs = virtual_ansatz->nb_dof_on_subentity(tdim - 1, pattern.master_facet_number());

  //std::cout << "num constrained dofs " << needed_num_unconstrained_dofs << std::endl;
  correct_number_unconstrained_master_dofs< DataType, DIM >(m2m, COEFFICIENT_EPS, needed_num_unconstrained_dofs);

  //LOG_DEBUG(0, "corrected m2m = ");
  //std::cout << "corrected m2m " << std::endl;
  //log_interface_matrix< DataType, DIM >(0, m2m);

  ////////////////
  // 3. Insert master interpolation into output argument.
  add_master_dof_interpolation_pattern< DataType, DIM >(m2m, COEFFICIENT_EPS, interpolation);

  // ***************************************************************************
  // Master -> Slave interpolation
  // ***************************************************************************

  for (size_t s = 0, e_s = pattern.num_slaves(); s != e_s; ++s) 
  {
    ////////////////
    // 1. Compute slave -> master interpolation matrix as product
    // of slave -> virtual and virtual -> master interface matrices.
    
    const mesh::Entity &slave_cell = this->mesh_->get_entity(tdim, interface.slave_index(s));
    CCellTrafoSPtr< DataType, DIM > trafo_slave = this->fe_manager_->get_cell_transformation(slave_cell.index());
    auto slave_ansatz = pattern.fe_type_slaves()[s];
    int iface_slave_facet_nr = interface.slave_facet_number(s);
    int pattern_slave_facet_nr = pattern.slave_facet_number(s);
    assert(iface_slave_facet_nr == pattern_slave_facet_nr);
    int ref_slave_facet_nr = trafo_slave->phys2ref_facet_nr(pattern_slave_facet_nr);

    //LOG_DEBUG(2, "slave " << s << " facet nr " << ref_slave_facet_nr << " " << pattern_slave_facet_nr);
    
    // Slave to virtual mapping.
    std::vector< std::vector< DataType > > weights_s2v;
    //LOG_DEBUG(2, "master - slave ");
    //std::cout << "master - slave " << s << std::endl;
    //std::cout << "weights s2v" << std::endl;
    compute_weights (master_cell, virtual_ansatz,
                     slave_cell, slave_ansatz,
                     this->mesh_->get_period(), weights_s2v);

    // Get slave DoFs lying on the interface.
    //LOG_DEBUG(2, "Slave " << s << " facet nr " << pattern.slave_facet_number(s));
    std::vector< int > slave_dofs_on_interface = slave_ansatz->get_dof_on_subentity(tdim - 1, ref_slave_facet_nr);
    //std::vector< int > slave_dofs_on_interface = slave_ansatz.get_dof_on_subentity(tdim - 1, pattern.slave_facet_number(s));
    //std::vector< int > slave_dofs_on_interface = slave_ansatz.get_shared_dof_on_subentity(tdim - 1, pattern.slave_facet_number(s));

    //LOG_DEBUG(2, "Slave " << s << " dofs on interface = "
      //                    << string_from_range(slave_dofs_on_interface.begin(),
        //                                       slave_dofs_on_interface.end()));
    if (DEBUG_LEVEL >= 0) 
    {
      //std::cout << "Slave " << s << " dofs on interface = "
        //                  << string_from_range(slave_dofs_on_interface.begin(),
          //                                      slave_dofs_on_interface.end()) << std::endl;;
    }
    
#ifndef NDEBUG
    for (int l = 0; l < weights_s2v.size(); ++l) 
    {
      //LOG_DEBUG(0, string_from_range(weights_s2v[l].begin(), weights_s2v[l].end()));
      //std::cout << string_from_range(weights_s2v[l].begin(), weights_s2v[l].end()) << std::endl;
    }
#endif
    InterfaceMatrix s2v;
    extract_interface_matrix< DataType, DIM >(weights_s2v, slave_dofs_on_interface, s2v);

    //LOG_DEBUG(0, "Slave " << s << " s2v =")
    //std::cout << "s2v" << std::endl;
    //log_interface_matrix< DataType, DIM >(0, s2v);

    // Multiply s2v and v2m to get s2m (slave -> master interpolation.
    InterfaceMatrix s2m;
    multiply_interface_matrices< DataType, DIM >(s2v, v2m, s2m);
    //std::cout << "v2m" << std::endl;
    //log_interface_matrix< DataType, DIM >(0, v2m);
    //std::cout << "s2m" <<  std::endl;
    //log_interface_matrix< DataType, DIM >(0, s2m);

    // RESULT:
    // s2m is a n_S x n_M matrix, with n_S denoting the number of dofs on the slave cell
    // if i is a slave interface dof, then
    //    s2m_(ij) = sum_(k is virtual interface dof) slave_dof_i (virtual_phi_k) * virtual_dof_k (master_phi_j)  
    // otherwise, 
    //    s2m_(ij) = 0
  
    // in the standard setting , i.e. no hanging nodes and p refinement, V = M and number_slaves = 1
    // thus:
    //    s2m_(ij) = slave_dof_i (master_phi_j) if i is slave interface dof, j is master interface dof  
    //    s2m_(ij) = 0, otherwise
  
    // if s2m(i,:) = e_j, where e_j is the j-th unit vector, then slave dof i and master dof j 
    // (i.e. their corresponding global ids) will be identifed in identify_common_dofs()
    
    //LOG_DEBUG(0, "Slave " << s << " s2m =");

    //log_interface_matrix< DataType, DIM >(2, s2m);

    ////////////////
    // 2. Insert slave interpolation into output argument
    add_slave_interpolation_pattern< DataType, DIM >(s, s2m, COEFFICIENT_EPS, interpolation);
  }

  //LOG_DEBUG(2, "Interpolation pattern computed: \n" << interpolation);
}

/// for all DoFs of (cellB, ansatzB) the weights of DoFs of (cellA, ansatzA)
/// are calculated -> assume same cell type, same ansatz type and same degree on both cells and no hanging nodes
template < class DataType, int DIM >
void NumberingLagrange< DataType, DIM >::compute_weights(
    mesh::Entity const &cellA, CRefElementSPtr< DataType, DIM >& ansatzA,
    mesh::Entity const &cellB, CRefElementSPtr< DataType, DIM >& ansatzB,
    std::vector< mesh::MasterSlave > const &period,
    InterpolationWeights &weights) const 
{
  const DataType COEFFICIENT_EPS = 1.e3 * std::numeric_limits< DataType >::epsilon();
  
  int fe_indA = this->fe_manager_->get_fe_index(ansatzA, cellA.index());
  int fe_indB = this->fe_manager_->get_fe_index(ansatzB, cellB.index());

  assert (fe_indA >= 0);
  assert (fe_indB >= 0);
  
  //std::cout << "cell index A " << cellA.index() << " , cell index B " << cellB.index() << " fe_indA " << fe_indA << " , fe_indB " << fe_indB << std::endl;
  CCellTrafoSPtr< DataType, DIM > transA = this->fe_manager_->get_cell_transformation(cellA.index());
  CCellTrafoSPtr< DataType, DIM > transB = this->fe_manager_->get_cell_transformation(cellB.index());

  /*std::vector< Vec< DIM, DataType > > coordsA = transA->get_coordinates();
  std::vector< Vec< DIM, DataType > > coordsB = transB->get_coordinates();
  size_t nb_coords = coordsA.size();
  for(int i = 0; i < nb_coords; ++i){
    std::cout << "coordA: " << coordsA[i];
  }
  std::cout << std::endl;
  for(int i = 0; i < nb_coords; ++i){
    std::cout << "coordB: " << coordsB[i];
  }
  std::cout << std::endl;*/
  //transA->print_vertex_coords();
  //transB->print_vertex_coords();

  /*Vec<DIM, DataType> coord_phys, coord_ref;
  coord_phys[0] = 0.5;
  coord_phys[1] = 0.5;
  coord_phys[2] = 11.;
  transA->inverse(coord_phys, coord_ref);
  std::cout << "Ref coords: " << coord_ref << std::endl;
  coord_phys[0] = 0.5;
  coord_phys[1] = 0.5;
  coord_phys[2] = 11.;
  transB->inverse(coord_phys, coord_ref);
  std::cout << "Ref coords: " << coord_ref << std::endl;*/

  const int row_length = ansatzA->nb_dof_on_cell();

  const int col_length = ansatzB->nb_dof_on_cell();
  weights.clear();
  weights.resize(col_length);

  // dof_B (phi_A) = dof_ref ( fe_trafo_B ( (fe_trafo_A)^{-1} (phi_ref) ) ) 

  // object for evaluating mapped shape functions of ansatz A at arbitrary physical coordinates
  // => phi_A = fe_trafo_A)^{-1} (phi_ref) : Omega -> R^d
  auto fe_trafoA = ansatzA->fe_trafo();
  MappingRef2Phys<DataType, DIM, RefElement< DataType, DIM > > * evalA 
    = new MappingRef2Phys<DataType, DIM, RefElement< DataType, DIM > > ( ansatzA.get(), fe_trafoA , transA);
                        
  // object for transforming mapped shape functions onto reference cell of B
  // => fe_trafo_B (phi_A|_B) : RefCell -> R^d 
  auto fe_trafoB = ansatzB->fe_trafo();
  MappingPhys2Ref <DataType,DIM,MappingRef2Phys<DataType,DIM,RefElement< DataType, DIM > > > * evalA_on_B
    = new MappingPhys2Ref <DataType, DIM, MappingRef2Phys<DataType, DIM, RefElement< DataType, DIM > > > (evalA, &cellB, fe_trafoB, transB);

  std::vector<DofID> all_dofs(col_length);
  for (size_t l=0; l<col_length; ++l)
  {
    all_dofs[l] = l;
  }
  
  // TODO: put weights directly into evaluate 
  // evaluate dofs of B for mapped shape functions of A
  std::vector< std::vector<DataType> > dof_B_of_phi_A;
  ansatzB->dof_container()-> evaluate (evalA_on_B, all_dofs,  dof_B_of_phi_A);

  //std::cout << cellA.index()<< " " << cellB.index() << std::endl;
  // put into weights
  // loop over dofs of element B
  for (size_t i = 0, e_i = col_length; i != e_i; ++i) 
  { 
    weights[i].resize(row_length, 0.);
    for (size_t j = 0, e_j = row_length; j!= e_j; ++j) 
    {
      weights[i][j] = dof_B_of_phi_A[i][j];
      
      DataType &w = weights[i][j];

      if (std::abs(w) < COEFFICIENT_EPS) 
      {
        w = 0.;
      } 
      else if (std::abs(w - 1.) < COEFFICIENT_EPS) 
      {
        w = 1.;
      }
      else if (std::abs(w + 1.) < COEFFICIENT_EPS) 
      {
        w = -1.;
      }
      //std::cout << " " << w ;
    }
    //std::cout << std::endl;
  }
  //std::cout << "========" << std::endl;
  //std::cout << std::endl << std::endl;
  delete evalA_on_B;
  delete evalA;
}

template class NumberingLagrange<float, 3>;
template class NumberingLagrange<float, 2>;
template class NumberingLagrange<float, 1>;

template class NumberingLagrange<double, 3>;
template class NumberingLagrange<double, 2>;
template class NumberingLagrange<double, 1>;

} // namespace doffem
} // namespace hiflow
