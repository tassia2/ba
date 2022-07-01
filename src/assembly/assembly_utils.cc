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

/// \author Staffan Ronnas, Simon Gawlok, Philipp Gerstner

#include "assembly/assembly_utils.h"
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <boost/function.hpp>

#include "common/csv_writer.h"
#include "common/sorted_array.h"
#include "common/array_tools.h"
#include "dof/dof_interpolation.h"
#include "dof/dof_partition.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "linear_algebra/la_descriptor.h"
#include "linear_algebra/la_couplings.h"
#include "linear_algebra/matrix.h"
#include "linear_algebra/seq_dense_matrix.h"
#include "linear_algebra/vector.h"
#include "mesh/mesh.h"
#include "mesh/iterator.h"
#include "mesh/interface.h"
#include "quadrature/quadrature.h"
#include "space/vector_space.h"
#include "space/element.h"
#include "quadrature/quadrature.h"

namespace hiflow {

template <class DataType, int DIM, class Container>
void create_sparsity_struct(const VectorSpace<DataType, DIM> &space, 
                            la::SparsityStructure &sparsity,
                            std::vector< Container >& raw_diag,
                            std::vector< Container >& raw_offdiag)
{
  const size_t ndof_total = raw_diag.size();
  
  // post-process to SparsityStructure
  // compute nnz for diagonal and offdiagonal blocks
  size_t nnz_diag = 0, nnz_offdiag = 0;
  for (size_t k = 0; k < ndof_total; ++k) 
  {
    nnz_diag += raw_diag[k].size();
    nnz_offdiag += raw_offdiag[k].size();
  }
  
  sparsity.diagonal_rows.resize(nnz_diag);
  sparsity.diagonal_cols.resize(nnz_diag);
  sparsity.off_diagonal_rows.resize(nnz_offdiag);
  sparsity.off_diagonal_cols.resize(nnz_offdiag);
#if 0
  sparsity.col_off_diagonal_rows.resize(nnz_offdiag);
  sparsity.col_off_diagonal_cols.resize(nnz_offdiag);
#endif
  doffem::gDofId global_dof_i;

  // copy diagonal sparsity structure
  size_t k = 0;
  for (doffem::lDofId r = 0; r < ndof_total; ++r) {
    space.dof().local2global(r, &global_dof_i);

    for (auto it = raw_diag[r].begin(), end = raw_diag[r].end();
         it != end; ++it) 
    {
      sparsity.diagonal_rows[k] = global_dof_i;
      sparsity.diagonal_cols[k] = *it;
      ++k;
    }
  }
  assert(k == nnz_diag);

  // copy off-diagonal sparsity structure
  k = 0;
  for (doffem::lDofId r = 0; r < ndof_total; ++r) 
  {
    space.dof().local2global(r, &global_dof_i);
    for (auto it = raw_offdiag[r].begin(), end = raw_offdiag[r].end();
         it != end; ++it) 
    {
      sparsity.off_diagonal_rows[k] = global_dof_i;
      sparsity.off_diagonal_cols[k] = *it;
      ++k;
    }
  }
  assert(k == nnz_offdiag);
  
  // copy columnwise - off-diagonal sparsity structure
#if 0
  k = 0;
  for (int r = 0; r < ndof_total; ++r) {
    space.dof().local2global(r, &global_dof_i);

    for (SortedArray< int >::const_iterator it = raw_offdiag[r].begin(),
                                            end = raw_offdiag[r].end();
         it != end; ++it) {
      
      sparsity.col_off_diagonal_rows[k] = *it;
      sparsity.col_off_diagonal_cols[k] = global_dof_i;
      ++k;
    }
  }
  assert(k == nnz_offdiag);
#endif
}

template<class DataType>
void add_interpolating_dofs (const doffem::DofInterpolation<DataType> &interp, 
                             std::vector<doffem::gDofId>& dof_list)
{
  const auto nb_dofs = dof_list.size();
  for (int i=0; i!=nb_dofs; ++i)
  {
    const doffem::gDofId gl_i = dof_list[i];
    auto it_i = interp.find(gl_i);

    if (it_i != interp.end()) 
    {
      // dof gl_i is constrained 
      for (auto c_it = it_i->second.begin(), c_end = it_i->second.end();
           c_it != c_end; ++c_it) 
      {
        dof_list.push_back(c_it->first);
      }
    }
  }
}

template <class DataType, int DIM>
void compute_sparsity_structure(const VectorSpace<DataType, DIM> &space, 
                                la::SparsityStructure &sparsity,
                                const std::vector< std::vector< bool > >& pre_coupling_vars,
                                const bool use_interface_integrals)
{
  int DBG_LVL = 1;

  std::vector< std::vector< bool > > coupling_vars = pre_coupling_vars;
  if (coupling_vars.empty()) 
  {
    coupling_vars.resize(space.nb_fe());
    for (size_t i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
    {
      coupling_vars[i].resize(space.nb_fe(), true);
    }
  }

  // Assert correct size of coupling_vars
  assert(coupling_vars.size() == space.nb_fe());
  for (size_t i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
  {
    assert(coupling_vars[i].size() == space.nb_fe());
  }

  // TODO: refactor function to avoid all the repetitions  
  typedef typename std::vector< std::pair< int, DataType> >::const_iterator ConstraintIterator;

  typedef std::vector< doffem::gDofId >::const_iterator DofIterator;
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive

  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  const doffem::DofInterpolation<DataType> &interpolation = space.dof().dof_interpolation();
  const auto num_total_dofs = space.dof().nb_dofs_on_subdom(space.dof().my_subdom());


  // NB: We assume that unconstrained dofs are numbered before
  // constrained dofs, in order to be able to use a vector here.

  //std::vector< SortedArray< int > > diagonal_couplings(num_total_dofs);
  //std::vector< SortedArray< int > > off_diagonal_couplings(num_total_dofs);

  //std::vector< std::vector< int > > diagonal_couplings(num_total_dofs);
  //std::vector< std::vector< int > > off_diagonal_couplings(num_total_dofs);
  
  std::vector< std::unordered_set< doffem::gDofId > > diagonal_couplings(num_total_dofs);
  std::vector< std::unordered_set< doffem::gDofId > > off_diagonal_couplings(num_total_dofs);
  
  
  for (doffem::lDofId i=0; i!=num_total_dofs; ++i)
  {
    diagonal_couplings[i].reserve(space.nb_dof_on_cell(0)*20);
    off_diagonal_couplings[i].reserve(space.nb_dof_on_cell(0)*20);
  }

  // step 1: consider first all variables, that are (partially) discontinuous across interfaces
  std::vector< doffem::gDofId > dof_list, slave_dofs;
  dof_list.reserve(space.nb_dof_on_cell(0) * 8);
  
  const int nb_fe = space.nb_fe();
  
  for (mesh::InterfaceList::const_iterator it = if_list.begin(),
        end_it = if_list.end(); it != end_it; ++it) 
  {
    const auto master_index = it->master_index();
    
    // loop over test variables
    for (size_t test_var = 0; test_var != nb_fe; ++test_var) 
    {
      // Get dof id:s on cell 
      dof_list.clear();
      space.get_dof_indices(test_var, master_index, dof_list);

      // loop over trial variables
      for (size_t trial_var = 0; trial_var != nb_fe; ++trial_var) 
      {
        // check if coupling exists
        if (coupling_vars[test_var][trial_var]) 
        {
          // dont consider continuous case if no interface integrals are needed
          if ((space.fe_conformity(trial_var) != doffem::FEConformity::H1) 
              || (space.fe_conformity(test_var) != doffem::FEConformity::H1)
              || use_interface_integrals)
          {
            // trial dofs from master cell
            if (trial_var != test_var)
            {
              space.get_dof_indices(trial_var, master_index, slave_dofs);
              dof_list.insert(dof_list.end(), slave_dofs.begin(), slave_dofs.end());
            }
          
            // from slave cell
            for (int s = 0, s_e = it->num_slaves(); s < s_e; ++s) 
            {
              space.get_dof_indices(trial_var, it->slave_index(s), slave_dofs);
              dof_list.insert(dof_list.end(), slave_dofs.begin(), slave_dofs.end());
            }
          }
        }
      }

      // get dof id's of interpolating dofs 
      add_interpolating_dofs<DataType>(interpolation, dof_list);
      
      sort_and_erase_duplicates<int>(dof_list);

      // All these dofs now potentially couple with one another.
      const size_t nd = dof_list.size();

      for (size_t i = 0; i != nd; ++i) 
      { // rows
        const auto dof_i = dof_list[i];
        if (!space.dof().is_dof_on_subdom(dof_i)) 
        {
          continue; // skip remote rows.
        }

        // get local row dof index
        doffem::lDofId local_dof_i;
        space.dof().global2local(dof_i, &local_dof_i);
        assert (local_dof_i >= 0);

        for (size_t j = 0; j != nd; ++j) 
        { // cols
          const auto dof_j = dof_list[j];
          // diagonal coupling (my col)
          if (space.dof().is_dof_on_subdom(dof_j)) 
          {
            // add if coupling is new
            
            //diagonal_couplings[local_dof_i].find_insert(dof_j);
            //diagonal_couplings[local_dof_i].push_back(dof_j);
            diagonal_couplings[local_dof_i].insert(dof_j);
          } 
          else 
          {
            //off_diagonal_couplings[local_dof_i].find_insert(dof_j);
            //off_diagonal_couplings[local_dof_i].push_back(dof_j);
            off_diagonal_couplings[local_dof_i].insert(dof_j);
          }
        }
      } // for (int i=0;...
    }
  }

  // step 2: consider all continuous variables, esspecially taking care of constrained dofs
  std::vector< doffem::gDofId > dofs_test, dofs_trial;
  dofs_test.reserve(space.nb_dof_on_cell(0) * 4);
  dofs_trial.reserve(space.nb_dof_on_cell(0) * 4);
  
  // Loop over all cells
  for (auto cell_it = mesh->begin(mesh->tdim()), 
       cell_end = mesh->end(mesh->tdim());
       cell_it != cell_end; ++cell_it) 
  {
    // loop over test variables
    for (size_t test_var = 0, e_test_var = space.nb_fe();
         test_var != e_test_var; ++test_var) 
    {
      // Get dof id:s on cell
      space.get_dof_indices(test_var, cell_it->index(), dofs_test);

      /*
      if (cell_it->index() == 778)
      {
        LOG_DEBUG(0, cell_it->index() << " : " << string_from_range(dofs_test.begin(), dofs_test.end()));
      }
      */

      // Loop over rows corresponding to local dofs.
      for (auto it_i = dofs_test.begin(), end_i = dofs_test.end();
           it_i != end_i; ++it_i) 
      {
        // search current dof it_i in DofInterpolation map
        auto dof_i = interpolation.find(*it_i);

        /*
        if (cell_it->index() == 778)
        {
          LOG_DEBUG(0, "dof " << *it_i << " is constrained ? " << (dof_i != interpolation.end()) 
                    << " is local " << space.dof().is_dof_on_subdom(*it_i) );
          if (dof_i != interpolation.end())
          {
            LOG_DEBUG(0, "dof " << *it_i << " interpolation i " << dof_i->first << " is local " << space.dof().is_dof_on_subdom(*it_i) );
            std::cout << "interpolating dofs : ";
            for (ConstraintIterator ci_it = dof_i->second.begin(),
                                    ci_end = dof_i->second.end();
                 ci_it != ci_end; ++ci_it) 
            {
              std::cout << " " << ci_it->first;
            }
            std::cout << std::endl;
          }
        }
        */
        if (dof_i != interpolation.end()) 
        {
          // Case A: dofs_test[*it_i] (row) constrained

          // Loop over all interpolating dofs of current dof it_i
          for (auto ci_it = dof_i->second.begin(),
                    ci_end = dof_i->second.end();
                    ci_it != ci_end; ++ci_it) 
          {
            assert (dof_i->first == *it_i);
             
            // skip rows that are not on our sub-domain
            if (space.dof().is_dof_on_subdom(ci_it->first)) 
            {
              // get row index to use for insertion
              doffem::lDofId local_row_dof = -1;
              space.dof().global2local(ci_it->first, &local_row_dof);
              assert (local_row_dof >= 0);
    
              // loop over trial variables
              for (size_t trial_var = 0, e_trial_var = space.nb_fe();
                   trial_var != e_trial_var; ++trial_var) 
              {
                // check if coupling exists
                if (coupling_vars[test_var][trial_var]) 
                {
                  // Get dof id:s on cell
                  space.get_dof_indices(trial_var, cell_it->index(), dofs_trial);

                  // Loop over columns corresponding to local dofs.
                  for (auto it_j = dofs_trial.begin(),
                            end_j = dofs_trial.end();
                       it_j != end_j; ++it_j) 
                  {
                    // search current dof it_j in DofInterpolation map
                    auto dof_j = interpolation.find(*it_j);

                    if (dof_j != interpolation.end()) 
                    {
                      // Case A1: dofs_trial[*it_j] (column) constrained

                      // Loop over all interpolating dofs of current dof it_j
                      for (auto cj_it = dof_j->second.begin(),
                                cj_end = dof_j->second.end();
                                cj_it != cj_end; ++cj_it) 
                      {
                        assert (dof_j->first == *it_j);
                        
                        // determine target for insertion
                        if (space.dof().is_dof_on_subdom(cj_it->first)) 
                        {
                          //diagonal_couplings[local_row_dof].find_insert(cj_it->first);
                          //diagonal_couplings[local_row_dof].push_back(cj_it->first);
                          diagonal_couplings[local_row_dof].insert(cj_it->first);
                        } 
                        else 
                        {
                          //off_diagonal_couplings[local_row_dof].find_insert(cj_it->first);
                          //off_diagonal_couplings[local_row_dof].push_back(cj_it->first);
                          off_diagonal_couplings[local_row_dof].insert(cj_it->first);
                        }

                        LOG_DEBUG(
                            2, "[" << space.dof().my_subdom()
                                   << "]   Constrained row = " << local_row_dof
                                   << ", constrained col = " << cj_it->first
                                   << ", diagonal ? "
                                   << space.dof().is_dof_on_subdom(cj_it->first));
                      }
                    } 
                    else 
                    {
                      // Case A2: dofs_trial[*it_j] (column) unconstrained
                      // determine target for insertion
                      if (space.dof().is_dof_on_subdom(*it_j)) 
                      {
                        //diagonal_couplings[local_row_dof].find_insert(*it_j);
                        //diagonal_couplings[local_row_dof].push_back(*it_j);
                        diagonal_couplings[local_row_dof].insert(*it_j);
                      } 
                      else 
                      {
                        //off_diagonal_couplings[local_row_dof].find_insert(*it_j);
                        //off_diagonal_couplings[local_row_dof].push_back(*it_j);
                        off_diagonal_couplings[local_row_dof].insert(*it_j);
                      }

                      LOG_DEBUG(2,
                                "[" << space.dof().my_subdom()
                                    << "]   Constrained row = " << local_row_dof
                                    << ", unconstrained col = " << *it_j
                                    << ", diagonal ? "
                                    << space.dof().is_dof_on_subdom(*it_j));
                    }
                  }
                }
              }
            }
          }
        } 
        else 
        {
          // Case B: dofs_test[*it_i] (row) unconstrained

          // skip rows that are not on our sub-domain
          if (space.dof().is_dof_on_subdom(*it_i)) 
          {
            doffem::lDofId local_row_dof = -1;
            space.dof().global2local(*it_i, &local_row_dof);
            assert (local_row_dof >= 0);
            
            // loop over trial variables
            for (size_t trial_var = 0, e_trial_var = space.nb_fe();
                 trial_var != e_trial_var; ++trial_var) 
            {
              // check if coupling exists
              if (coupling_vars[test_var][trial_var]) 
              {
                // Get dof id:s on cell
                space.get_dof_indices(trial_var, cell_it->index(), dofs_trial);

                // Loop over columns corresponding to local dofs.
                for (DofIterator it_j = dofs_trial.begin(),
                                 end_j = dofs_trial.end();
                     it_j != end_j; ++it_j) 
                {
                  // search current dof it_j in DofInterpolation map
                  auto dof_j = interpolation.find(*it_j);

                  if (dof_j != interpolation.end()) 
                  {
                    // Case B1: dofs_trial[*it_j] (column) constrained

                    // Loop over all interpolating dofs of current dof it_j
                    for (ConstraintIterator cj_it = dof_j->second.begin(),
                                            cj_end = dof_j->second.end();
                         cj_it != cj_end; ++cj_it) 
                    {
                      // determine target for insertion
                      // -> diagonal or off-diagonal
                      if (space.dof().is_dof_on_subdom(cj_it->first)) 
                      {
                        //diagonal_couplings[local_row_dof].find_insert(cj_it->first);
                        //diagonal_couplings[local_row_dof].push_back(cj_it->first);
                        diagonal_couplings[local_row_dof].insert(cj_it->first);
                      } 
                      else 
                      {
                        //off_diagonal_couplings[local_row_dof].find_insert(cj_it->first);
                        //off_diagonal_couplings[local_row_dof].push_back(cj_it->first);
                        off_diagonal_couplings[local_row_dof].insert(cj_it->first);
                      }

                      LOG_DEBUG(2,
                                "[" << space.dof().my_subdom()
                                    << "] Unconstrained row = " << local_row_dof
                                    << ", constrained col = " << cj_it->first
                                    << ", diagonal ? "
                                    << space.dof().is_dof_on_subdom(cj_it->first));
                    }
                  } 
                  else 
                  {
                    // Case B2: dofs_trial[*it_j] (column) unconstrained
                    // determine target for insertion
                    // -> diagonal or off-diagonal
                    if (space.dof().is_dof_on_subdom(*it_j)) 
                    {
                      //diagonal_couplings[local_row_dof].find_insert(*it_j);
                      //diagonal_couplings[local_row_dof].push_back(*it_j);
                      diagonal_couplings[local_row_dof].insert(*it_j);
                    } 
                    else 
                    {
                      //off_diagonal_couplings[local_row_dof].find_insert(*it_j);
                      //off_diagonal_couplings[local_row_dof].push_back(*it_j);
                      off_diagonal_couplings[local_row_dof].insert(*it_j);
                    }

                    LOG_DEBUG(2,
                              "[" << space.dof().my_subdom()
                                  << "] Unconstrained row = " << local_row_dof
                                  << ", unconstrained col = " << *it_j
                                  << ", diagonal ? "
                                  << space.dof().is_dof_on_subdom(*it_j));
                  }
                }
              }
            }
          }
        }
        // Add diagonal entry
        if (space.dof().is_dof_on_subdom(*it_i)) 
        {
          doffem::lDofId local_row_dof = -1;
          space.dof().global2local(*it_i, &local_row_dof);
          assert (local_row_dof >= 0);
          
          //diagonal_couplings[local_row_dof].find_insert(*it_i);
          //diagonal_couplings[local_row_dof].push_back(*it_i);
          diagonal_couplings[local_row_dof].insert(*it_i);

          LOG_DEBUG(2, "[" << space.dof().my_subdom()
                           << "]   Diagonal row = " << local_row_dof
                           << ", diagonal col = " << local_row_dof);
        }
      }
    }
  }

  // some statistics
  double average_diag_couplings = 0;
  double average_offdiag_couplings = 0;
   
  // post-process to SparsityStructure
  // compute nnz for diagonal and offdiagonal blocks

  for (size_t i=0; i<num_total_dofs; ++i)
  {
    // nothing to do
    
    //sort_and_erase_duplicates<int>(diagonal_couplings[i]);
    // sort_and_erase_duplicates<int>(off_diagonal_couplings[i]);

    // nothing to do 

    average_diag_couplings += diagonal_couplings[i].size();
    average_offdiag_couplings += off_diagonal_couplings[i].size();
  }
  
  average_diag_couplings /= static_cast<double>(num_total_dofs);
  average_offdiag_couplings /= static_cast<double>(num_total_dofs);
  
  LOG_INFO("# Diag couplings", average_diag_couplings);
  LOG_INFO("# Offdiag couplings", average_offdiag_couplings);
   
  create_sparsity_struct(space, sparsity, diagonal_couplings, off_diagonal_couplings);

  LOG_INFO("create sparsity", "done");
}

template < class DataType, int DIM >
void compute_std_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                    la::SparsityStructure &sparsity,
                                    const std::vector< std::vector< bool > >& pre_coupling_vars)
{
  std::vector< std::vector< bool > > coupling_vars = pre_coupling_vars;
    
  if (coupling_vars.empty()) 
  {
    coupling_vars.resize(space.nb_fe());
    for (size_t i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
    {
      coupling_vars[i].resize(space.nb_fe(), true);
    }
  }

  // Assert correct size of coupling_vars
  assert(coupling_vars.size() == space.nb_fe());
  for (size_t i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
  {
    assert(coupling_vars[i].size() == space.nb_fe());
  }

#ifdef OUTPUT_STRUCT
  // CSV writer for visual check of sparsity structure
  CSVWriter< int > writer("sparsity.csv");

  std::vector< std::string > names;
  names.push_back("i");
  names.push_back("j");
  names.push_back("val");

  writer.Init(names);

  std::vector< int > values(3, -1);
#endif

  // total number of own dofs
  doffem::lDofId ndof_total = space.dof().nb_dofs_on_subdom(space.dof().my_subdom());

  // a set of columns for every row
  std::vector< SortedArray< doffem::gDofId > > raw_struct_diag(ndof_total);
  std::vector< SortedArray< doffem::gDofId > > raw_struct_offdiag(ndof_total);

  std::vector< doffem::gDofId > dof_ind_test, dof_ind_trial;
  doffem::lDofId local_dof_i;

  // loop over every cell (including ghost cells)
  typename VectorSpace< DataType, DIM >::MeshEntityIterator mesh_it =
      space.mesh().begin(space.tdim());
  typename VectorSpace< DataType, DIM >::MeshEntityIterator e_mesh_it =
      space.mesh().end(space.tdim());
  while (mesh_it != e_mesh_it) {
    // loop over test variables
    for (int test_var = 0, tv_e = space.nb_fe(); test_var != tv_e;
         ++test_var) {
      // get dof indices for test variable
      space.get_dof_indices(test_var, mesh_it->index(), dof_ind_test);

      // loop over trial variables
      for (int trial_var = 0, vt_e = space.nb_fe(); trial_var != vt_e;
           ++trial_var) {

        // check whether test_var and trial_var couple
        if (coupling_vars[test_var][trial_var]) {

          // get dof indices for trial variable
          space.get_dof_indices(trial_var, mesh_it->index(), dof_ind_trial);

          // detect couplings
          for (size_t i = 0, i_e = dof_ind_test.size(); i != i_e; ++i) {
            const auto di_i = dof_ind_test[i];

            // if my row
            if (space.dof().is_dof_on_subdom(di_i)) {

              space.dof().global2local(di_i, &local_dof_i);

              for (size_t j = 0, j_e = dof_ind_trial.size(); j != j_e; ++j) {
                const auto di_j = dof_ind_trial[j];

                // diagonal coupling (my col)
                if (space.dof().is_dof_on_subdom(di_j)) {
                  raw_struct_diag[local_dof_i].find_insert(di_j);
                } else {
                  // nondiagonal coupling (not my col)
                  raw_struct_offdiag[local_dof_i].find_insert(di_j);
                }
              } // endif my row

            } // for (int j=0;...
          }   // for (int i=0;...
        }
      }
    }
    // next cell
    ++mesh_it;
  } // while (mesh_it != ...

#ifdef OUTPUT_STRUCT
  for (size_t k = 0, k_e = raw_struct_diag.size(); k != k_e; ++k) {
    values[0] = k;
    for (size_t l = 0, l_e = raw_struct_diag[k].size(); l != l_e; ++l) {
      values[1] = raw_struct_diag[k][l];
      values[2] = 1;
      writer.write(values);
    }
  }

  // compute nnz for nondiagonal block
  for (size_t k = 0, k_e = raw_struct_offdiag.size(); k != k_e; ++k) {
    values[0] = k;
    for (size_t l = 0, l_e = raw_struct_offdiag[k].size(); l != l_e; ++l) {
      values[1] = raw_struct_offdiag[k][l];
      values[2] = 1;
      writer.write(values);
    }
  }
#endif

  // some statistics
  double average_diag_couplings = 0;
  double average_offdiag_couplings = 0;
  
  for (size_t i=0; i<ndof_total; ++i)
  {
    average_diag_couplings += raw_struct_diag[i].size();
    average_offdiag_couplings += raw_struct_offdiag[i].size();
  }
  
  average_diag_couplings /= static_cast<double>(ndof_total);
  average_offdiag_couplings /= static_cast<double>(ndof_total);
  
  LOG_INFO("# Diag couplings", average_diag_couplings);
  LOG_INFO("# Offdiag couplings", average_offdiag_couplings);
  
  create_sparsity_struct(space, sparsity, raw_struct_diag, raw_struct_offdiag);

}
              
template <class DataType, int DIM>
void compute_dg_sparsity_structure(const VectorSpace<DataType, DIM> &space, 
                                   la::SparsityStructure &sparsity,
                                   const std::vector< std::vector< bool > >& pre_coupling_vars)
{
  std::vector< std::vector< bool > > coupling_vars = pre_coupling_vars;
  
  if (coupling_vars.empty()) 
  {
    coupling_vars.resize(space.nb_fe());
    for (size_t i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
    {
      coupling_vars[i].resize(space.nb_fe(), true);
    }
  }

  // Assert correct size of coupling_vars
  assert(coupling_vars.size() == space.nb_fe());
  for (size_t i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
  {
    assert(coupling_vars[i].size() == space.nb_fe());
  }
  
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  const auto ndof_total =
      space.dof().nb_dofs_on_subdom(space.dof().my_subdom());

  // a set of columns for every row
  std::vector< SortedArray< doffem::gDofId > > raw_diag(ndof_total);
  std::vector< SortedArray< doffem::gDofId > > raw_offdiag(ndof_total);

  std::vector< doffem::gDofId > dof_list, slave_dofs;

  for (mesh::InterfaceList::const_iterator it = if_list.begin(),
        end_it = if_list.end(); it != end_it; ++it) 
  {
    for (int test_var = 0, e_test_var = space.nb_fe();
         test_var < e_test_var; ++test_var) 
    {
      // get dof indices
      Element< DataType, DIM > master_elem(space, it->master_index());
      space.get_dof_indices(test_var, it->master_index(), dof_list);

      // loop over trial variables
      for (int trial_var = 0, e_trial_var = space.nb_fe();
           trial_var != e_trial_var; ++trial_var) 
      {
        // check whether test_var and trial_var couple
        if (coupling_vars[test_var][trial_var]) 
        {
          // from master cell
          if (trial_var != test_var) 
          {
            space.get_dof_indices(trial_var, it->master_index(), slave_dofs);
            dof_list.insert(dof_list.end(), slave_dofs.begin(), slave_dofs.end());
          }
          // from slave cell
          for (int s = 0, s_e = it->num_slaves(); s < s_e; ++s) 
          {
            Element< DataType, DIM > slave_elem(space, it->slave_index(s));
            space.get_dof_indices(trial_var, it->slave_index(s), slave_dofs);
            dof_list.insert(dof_list.end(), slave_dofs.begin(), slave_dofs.end());
          }
        }
      }
      // All these dofs now potentially couple with one another.
      const size_t nd = dof_list.size();

      for (size_t i = 0; i != nd; ++i) { // rows
        if (!space.dof().is_dof_on_subdom(dof_list[i])) {
          continue; // skip remote rows.
        }

        // get local row dof index
        doffem::lDofId local_dof_i;
        space.dof().global2local(dof_list[i], &local_dof_i);

        for (size_t j = 0; j != nd; ++j) { // cols
          // diagonal coupling (my col)
          if (space.dof().is_dof_on_subdom(dof_list[j])) {
            // add if coupling is new
            raw_diag[local_dof_i].find_insert(dof_list[j]);
          } else {
            raw_offdiag[local_dof_i].find_insert(dof_list[j]);
          }
        }
      } // for (int i=0;...
    }
  }
  
    // some statistics
  double average_diag_couplings = 0;
  double average_offdiag_couplings = 0;
  
  for (size_t i=0; i<ndof_total; ++i)
  {
    average_diag_couplings += raw_diag[i].size();
    average_offdiag_couplings += raw_offdiag[i].size();
  }
  
  average_diag_couplings /= static_cast<double>(ndof_total);
  average_offdiag_couplings /= static_cast<double>(ndof_total);
  
  LOG_INFO("# Diag couplings", average_diag_couplings);
  LOG_INFO("# Offdiag couplings", average_offdiag_couplings);
  
  create_sparsity_struct(space, sparsity, raw_diag, raw_offdiag);
}

                                                    
template < class DataType, int DIM >
void compute_hp_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                   la::SparsityStructure &sparsity,
                                   const std::vector< std::vector< bool > > &pre_coupling_vars) 
{
  std::vector< std::vector< bool > > coupling_vars = pre_coupling_vars;
  if (coupling_vars.empty()) 
  {
    coupling_vars.resize(space.nb_fe());
    for (size_t i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
    {
      coupling_vars[i].resize(space.nb_fe(), true);
    }
  }

  // Assert correct size of coupling_vars
  assert(coupling_vars.size() == space.nb_fe());
  for (size_t i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
  {
    assert(coupling_vars[i].size() == space.nb_fe());
  }
  
  typedef typename std::vector< std::pair< int, DataType> >::const_iterator ConstraintIterator;

  // TODO: refactor function to avoid all the repetitions

  typedef typename VectorSpace< DataType, DIM >::MeshEntityIterator CellIterator;
  typedef std::vector< doffem::gDofId >::const_iterator DofIterator;
  const mesh::Mesh &mesh = space.mesh();
  const mesh::TDim tdim = mesh.tdim();

  std::vector< doffem::gDofId > dofs_test, dofs_trial;

  const doffem::DofInterpolation<DataType> &interpolation = space.dof().dof_interpolation();
  const auto num_total_dofs =
      space.dof().nb_dofs_on_subdom(space.dof().my_subdom());
  doffem::lDofId local_row_dof;

  // NB: We assume that unconstrained dofs are numbered before
  // constrained dofs, in order to be able to use a vector here.
  std::vector< SortedArray< doffem::lDofId > > diagonal_couplings(num_total_dofs);
  std::vector< SortedArray< doffem::lDofId > > off_diagonal_couplings(num_total_dofs);

  // Loop over all cells
  for (CellIterator cell_it = mesh.begin(tdim), cell_end = mesh.end(tdim);
       cell_it != cell_end; ++cell_it) {

    // loop over test variables
    for (size_t test_var = 0, e_test_var = space.nb_fe();
         test_var != e_test_var; ++test_var) {
      // Get dof id:s on cell
      space.get_dof_indices(test_var, cell_it->index(), dofs_test);

      // Loop over rows corresponding to local dofs.
      for (DofIterator it_i = dofs_test.begin(), end_i = dofs_test.end();
           it_i != end_i; ++it_i) {

        // search current dof it_i in DofInterpolation map
        typename doffem::DofInterpolation<DataType>::const_iterator dof_i = interpolation.find(*it_i);

        if (dof_i != interpolation.end()) {
          // Case A: dofs_test[*it_i] (row) constrained

          // Loop over all interpolating dofs of current dof it_i
          for (ConstraintIterator ci_it = dof_i->second.begin(),
                                  ci_end = dof_i->second.end();
               ci_it != ci_end; ++ci_it) {

            // skip rows that are not on our sub-domain
            if (space.dof().is_dof_on_subdom(ci_it->first)) {

              // get row index to use for insertion
              space.dof().global2local(ci_it->first, &local_row_dof);

              // loop over trial variables
              for (size_t trial_var = 0, e_trial_var = space.nb_fe();
                   trial_var != e_trial_var; ++trial_var) {

                // check if coupling exists
                if (coupling_vars[test_var][trial_var]) {

                  // Get dof id:s on cell
                  space.get_dof_indices(trial_var, cell_it->index(), dofs_trial);

                  // Loop over columns corresponding to local dofs.
                  for (DofIterator it_j = dofs_trial.begin(),
                                   end_j = dofs_trial.end();
                       it_j != end_j; ++it_j) {

                    // search current dof it_j in DofInterpolation map
                    auto dof_j = interpolation.find(*it_j);

                    if (dof_j != interpolation.end()) {
                      // Case A1: dofs_trial[*it_j] (column) constrained

                      // Loop over all interpolating dofs of current dof it_j
                      for (ConstraintIterator cj_it = dof_j->second.begin(),
                                              cj_end = dof_j->second.end();
                           cj_it != cj_end; ++cj_it) {
                        // determine target for insertion
                        if (space.dof().is_dof_on_subdom(cj_it->first)) {
                          diagonal_couplings[local_row_dof].find_insert(
                              cj_it->first);
                        } else {
                          off_diagonal_couplings[local_row_dof].find_insert(
                              cj_it->first);
                        }

                        LOG_DEBUG(
                            2, "[" << space.dof().my_subdom()
                                   << "]   Constrained row = " << local_row_dof
                                   << ", constrained col = " << cj_it->first
                                   << ", diagonal ? "
                                   << space.dof().is_dof_on_subdom(cj_it->first));
                      }
                    } else {
                      // Case A2: dofs_trial[*it_j] (column) unconstrained
                      // determine target for insertion
                      if (space.dof().is_dof_on_subdom(*it_j)) {
                        diagonal_couplings[local_row_dof].find_insert(*it_j);
                      } else {
                        off_diagonal_couplings[local_row_dof].find_insert(
                            *it_j);
                      }

                      LOG_DEBUG(2,
                                "[" << space.dof().my_subdom()
                                    << "]   Constrained row = " << local_row_dof
                                    << ", unconstrained col = " << *it_j
                                    << ", diagonal ? "
                                    << space.dof().is_dof_on_subdom(*it_j));
                    }
                  }
                }
              }
            }
          }
        } else {
          // Case B: dofs_test[*it_i] (row) unconstrained

          // skip rows that are not on our sub-domain
          if (space.dof().is_dof_on_subdom(*it_i)) {

            space.dof().global2local(*it_i, &local_row_dof);

            // loop over trial variables
            for (size_t trial_var = 0, e_trial_var = space.nb_fe();
                 trial_var != e_trial_var; ++trial_var) {

              // check if coupling exists
              if (coupling_vars[test_var][trial_var]) {

                // Get dof id:s on cell
                space.get_dof_indices(trial_var, cell_it->index(), dofs_trial);

                // Loop over columns corresponding to local dofs.
                for (DofIterator it_j = dofs_trial.begin(),
                                 end_j = dofs_trial.end();
                     it_j != end_j; ++it_j) {

                  // search current dof it_j in DofInterpolation map
                  auto dof_j = interpolation.find(*it_j);

                  if (dof_j != interpolation.end()) {
                    // Case B1: dofs_trial[*it_j] (column) constrained

                    // Loop over all interpolating dofs of current dof it_j
                    for (ConstraintIterator cj_it = dof_j->second.begin(),
                                            cj_end = dof_j->second.end();
                         cj_it != cj_end; ++cj_it) {
                      // determine target for insertion
                      // -> diagonal or off-diagonal
                      if (space.dof().is_dof_on_subdom(cj_it->first)) {
                        diagonal_couplings[local_row_dof].find_insert(
                            cj_it->first);
                      } else {
                        off_diagonal_couplings[local_row_dof].find_insert(
                            cj_it->first);
                      }

                      LOG_DEBUG(2,
                                "[" << space.dof().my_subdom()
                                    << "] Unconstrained row = " << local_row_dof
                                    << ", constrained col = " << cj_it->first
                                    << ", diagonal ? "
                                    << space.dof().is_dof_on_subdom(cj_it->first));
                    }
                  } else {
                    // Case B2: dofs_trial[*it_j] (column) unconstrained
                    // determine target for insertion
                    // -> diagonal or off-diagonal
                    if (space.dof().is_dof_on_subdom(*it_j)) {
                      diagonal_couplings[local_row_dof].find_insert(*it_j);
                    } else {
                      off_diagonal_couplings[local_row_dof].find_insert(*it_j);
                    }

                    LOG_DEBUG(3,
                              "[" << space.dof().my_subdom()
                                  << "] Unconstrained row = " << local_row_dof
                                  << ", unconstrained col = " << *it_j
                                  << ", diagonal ? "
                                  << space.dof().is_dof_on_subdom(*it_j));
                  }
                }
              }
            }
          }
        }
        // Add diagonal entry
        if (space.dof().is_dof_on_subdom(*it_i)) {
          space.dof().global2local(*it_i, &local_row_dof);
          diagonal_couplings[local_row_dof].find_insert(*it_i);

          LOG_DEBUG(2, "[" << space.dof().my_subdom()
                           << "]   Diagonal row = " << local_row_dof
                           << ", diagonal col = " << local_row_dof);
        }
      }
    }
  }

  // some statistics
  double average_diag_couplings = 0;
  double average_offdiag_couplings = 0;
  
  for (size_t i=0; i<num_total_dofs; ++i)
  {
    average_diag_couplings += diagonal_couplings[i].size();
    average_offdiag_couplings += off_diagonal_couplings[i].size();
  }
  
  average_diag_couplings /= static_cast<double>(num_total_dofs);
  average_offdiag_couplings /= static_cast<double>(num_total_dofs);
  
  LOG_INFO("# Diag couplings", average_diag_couplings);
  LOG_INFO("# Offdiag couplings", average_offdiag_couplings);
  
  create_sparsity_struct(space, sparsity, diagonal_couplings, off_diagonal_couplings);
}

#if 0
// deprecated                                                         
template < class DataType, int DIM >
void InitStructure(const VectorSpace< DataType, DIM > &space,
                   std::vector< int > *rows_diag,
                   std::vector< int > *cols_diag,
                   std::vector< int > *rows_offdiag,
                   std::vector< int > *cols_offdiag,
                   std::vector< std::vector< bool > > &coupling_vars) {

  // Assert correct size of coupling_vars

  assert(coupling_vars.size() == space.nb_fe());
  for (int i = 0, i_e = space.nb_fe(); i != i_e; ++i) 
  {
    assert(coupling_vars[i].size() == space.nb_fe());
  }

#ifdef OUTPUT_STRUCT
  // CSV writer for visual check of sparsity structure
  CSVWriter< int > writer("sparsity.csv");

  std::vector< std::string > names;
  names.push_back("i");
  names.push_back("j");
  names.push_back("val");

  writer.Init(names);

  std::vector< int > values(3, -1);
#endif

  // total number of own dofs
  size_t ndof_total = space.dof().nb_dofs_on_subdom(space.dof().my_subdom());

  // a set of columns for every row
  std::vector< SortedArray< int > > raw_struct_diag(ndof_total);
  std::vector< SortedArray< int > > raw_struct_offdiag(ndof_total);

  std::vector< int > dof_ind_test, dof_ind_trial;
  int local_dof_i;

  // loop over every cell (including ghost cells)
  typename VectorSpace< DataType, DIM >::MeshEntityIterator mesh_it =
      space.mesh().begin(space.tdim());
  typename VectorSpace< DataType, DIM >::MeshEntityIterator e_mesh_it =
      space.mesh().end(space.tdim());
  while (mesh_it != e_mesh_it) {
    // loop over test variables
    for (int test_var = 0, tv_e = space.nb_fe(); test_var != tv_e;
         ++test_var) {
      // get dof indices for test variable
      space.get_dof_indices(test_var, mesh_it->index(), dof_ind_test);

      // loop over trial variables
      for (int trial_var = 0, vt_e = space.nb_fe(); trial_var != vt_e;
           ++trial_var) {

        // check whether test_var and trial_var couple
        if (coupling_vars[test_var][trial_var]) {

          // get dof indices for trial variable
          space.get_dof_indices(trial_var, mesh_it->index(), dof_ind_trial);

          // detect couplings
          for (size_t i = 0, i_e = dof_ind_test.size(); i != i_e; ++i) {
            const int di_i = dof_ind_test[i];

            // if my row
            if (space.dof().is_dof_on_subdom(di_i)) {

              space.dof().global2local(di_i, &local_dof_i);

              for (size_t j = 0, j_e = dof_ind_trial.size(); j != j_e; ++j) {
                const int di_j = dof_ind_trial[j];

                // diagonal coupling (my col)
                if (space.dof().is_dof_on_subdom(di_j)) {
                  raw_struct_diag[local_dof_i].find_insert(di_j);
                } else {
                  // nondiagonal coupling (not my col)
                  raw_struct_offdiag[local_dof_i].find_insert(di_j);
                }
              } // endif my row

            } // for (int j=0;...
          }   // for (int i=0;...
        }
      }
    }
    // next cell
    ++mesh_it;
  } // while (mesh_it != ...

#ifdef OUTPUT_STRUCT
  for (size_t k = 0, k_e = raw_struct_diag.size(); k != k_e; ++k) {
    values[0] = k;
    for (size_t l = 0, l_e = raw_struct_diag[k].size(); l != l_e; ++l) {
      values[1] = raw_struct_diag[k][l];
      values[2] = 1;
      writer.write(values);
    }
  }

  // compute nnz for nondiagonal block
  for (size_t k = 0, k_e = raw_struct_offdiag.size(); k != k_e; ++k) {
    values[0] = k;
    for (size_t l = 0, l_e = raw_struct_offdiag[k].size(); l != l_e; ++l) {
      values[1] = raw_struct_offdiag[k][l];
      values[2] = 1;
      writer.write(values);
    }
  }
#endif

  SparsityStructure tmp_sparsity;
  create_sparsity_struct(space, tmp_sparsity, raw_struct_diag, raw_struct_offdiag);
  
  // TODO: avoid copy
  (*rows_diag) = tmp_sparsity.diagonal_rows;
  (*cols_diag) = tmp_sparsity.diagonal_cols;
  (*rows_offdiag) = tmp_sparsity.offdiagonal_rows;
  (*cols_offdiag) = tmp_sparsity.offdiagonal_cols;
}

#endif

template < class DataType, int DIM >
void init_master_quadrature(const Element< DataType, DIM > &slave_elem,
                            const Element< DataType, DIM > &master_elem,
                            const Quadrature< DataType > &slave_quad,
                            Quadrature< DataType > &master_quad) 
{
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  
  // Exit early if both elements are the same.
  if (master_elem.get_cell().id() == slave_elem.get_cell().id()) {
    master_quad = slave_quad; // copy quadrature (might not be necessary)
    return;
  }

  doffem::CCellTrafoSPtr<DataType, DIM> Ts = slave_elem.get_cell_transformation();
  doffem::CCellTrafoSPtr<DataType, DIM> Tm = master_elem.get_cell_transformation();

  const size_t num_q = slave_quad.size();
  std::vector< Coord > pts(num_q);
  std::vector<DataType> wgt(num_q, 0.);

  const int dim = slave_elem.get_cell().tdim();

  Coord ref_pt, phys_pt;

  if (dim == 2) 
  {
    for (size_t q = 0; q != num_q; ++q) 
    {
      // reference point on slave cell
      ref_pt.set(0, slave_quad.x(q));
      ref_pt.set(1, slave_quad.y(q));

      // physical point
      phys_pt.set(0, Ts->x(ref_pt));
      phys_pt.set(1, Ts->y(ref_pt));

      // reference point on master cell
      bool found = Tm->inverse(phys_pt, pts[q]);
      assert (found);
      
      // weight
      wgt[q] = slave_quad.w(q);
    }
  } 
  else if (dim == 3) 
  {
    for (size_t q = 0; q != num_q; ++q) 
    {
      // reference point on slave cell
      ref_pt.set(0, slave_quad.x(q));
      ref_pt.set(1, slave_quad.y(q));
      ref_pt.set(2, slave_quad.z(q));

      // physical point
      phys_pt.set(0, Ts->x(ref_pt));
      phys_pt.set(1, Ts->y(ref_pt));
      phys_pt.set(2, Ts->z(ref_pt));
      // reference point on master cell
      bool found = Tm->inverse(phys_pt, pts[q]);
      assert (found);
      
      // weight
      wgt[q] = slave_quad.w(q);
    }
  }
  std::vector<DataType> xc(num_q, 0.), yc(num_q,0.), zc(num_q,0.);
  if (dim == 1)
  {
    for (size_t q=0; q<num_q; ++q)
    {
      xc[q] = pts[q][0];
    }
  } 
  else if (dim == 2)
  {    
    for (size_t q=0; q<num_q; ++q)
    {
      xc[q] = pts[q][0];
      yc[q] = pts[q][1];
    }
  }
  else if (dim == 3)
  {    
    for (size_t q=0; q<num_q; ++q)
    {
      xc[q] = pts[q][0];
      yc[q] = pts[q][1];
      zc[q] = pts[q][2];
    }
  }  
  master_quad.set_custom_quadrature(slave_quad.order(), master_elem.ref_cell()->tag(), xc, yc, zc, wgt);
}

template void compute_sparsity_structure <double, 3> (const VectorSpace< double, 3> &, 
                                                      la::SparsityStructure &,
                                                      const std::vector< std::vector< bool > > &,
                                                      const bool); 
template void compute_sparsity_structure <double, 2> (const VectorSpace< double, 2> &, 
                                                      la::SparsityStructure &,
                                                      const std::vector< std::vector< bool > > &,
                                                      const bool);  
template void compute_sparsity_structure <double, 1> (const VectorSpace< double, 1> &, 
                                                      la::SparsityStructure &,
                                                      const std::vector< std::vector< bool > > &,
                                                      const bool);  
template void compute_sparsity_structure <float, 3> (const VectorSpace< float, 3> &, 
                                                     la::SparsityStructure &,
                                                     const std::vector< std::vector< bool > > &,
                                                     const bool);  
template void compute_sparsity_structure <float, 2> (const VectorSpace< float, 2> &, 
                                                     la::SparsityStructure &,
                                                     const std::vector< std::vector< bool > > &,
                                                     const bool);  
template void compute_sparsity_structure <float, 1> (const VectorSpace< float, 1> &, 
                                                     la::SparsityStructure &,
                                                     const std::vector< std::vector< bool > > &,
                                                     const bool);  
                                                         
template void compute_std_sparsity_structure <double, 3> (const VectorSpace< double, 3> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_std_sparsity_structure <double, 2> (const VectorSpace< double, 2> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_std_sparsity_structure <double, 1> (const VectorSpace< double, 1> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_std_sparsity_structure <float, 3> (const VectorSpace< float, 3> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_std_sparsity_structure <float, 2> (const VectorSpace< float, 2> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_std_sparsity_structure <float, 1> (const VectorSpace< float, 1> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 

template void compute_dg_sparsity_structure <double, 3> (const VectorSpace< double, 3> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_dg_sparsity_structure <double, 2> (const VectorSpace< double, 2> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_dg_sparsity_structure <double, 1> (const VectorSpace< double, 1> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_dg_sparsity_structure <float, 3> (const VectorSpace< float, 3> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_dg_sparsity_structure <float, 2> (const VectorSpace< float, 2> &, 
                                                         la::SparsityStructure &,
                                                        const  std::vector< std::vector< bool > > &); 
template void compute_dg_sparsity_structure <float, 1> (const VectorSpace< float, 1> &, 
                                                         la::SparsityStructure &,
                                                        const  std::vector< std::vector< bool > > &);
                                                         
template void compute_hp_sparsity_structure <double, 3> (const VectorSpace< double, 3> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_hp_sparsity_structure <double, 2> (const VectorSpace< double, 2> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_hp_sparsity_structure <double, 1> (const VectorSpace< double, 1> &, 
                                                         la::SparsityStructure &,
                                                         const std::vector< std::vector< bool > > &); 
template void compute_hp_sparsity_structure <float, 3> (const VectorSpace< float, 3> &, 
                                                         la::SparsityStructure &,
                                                        const  std::vector< std::vector< bool > > &); 
template void compute_hp_sparsity_structure <float, 2> (const VectorSpace< float, 2> &, 
                                                         la::SparsityStructure &,
                                                        const  std::vector< std::vector< bool > > &); 
template void compute_hp_sparsity_structure <float, 1> (const VectorSpace< float, 1> &, 
                                                         la::SparsityStructure &,
                                                        const  std::vector< std::vector< bool > > &); 
#if 0
template void InitStructure <double, 3> (const VectorSpace< double, 3 > &,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< std::vector< bool > > &);
template void InitStructure <double, 2> (const VectorSpace< double, 2 > &,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< std::vector< bool > > &);
template void InitStructure <double, 1> (const VectorSpace< double, 1 > &,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< std::vector< bool > > &);
template void InitStructure <float, 3> (const VectorSpace< float, 3 > &,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< std::vector< bool > > &);
template void InitStructure <float, 2> (const VectorSpace< float, 2 > &,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< std::vector< bool > > &);
template void InitStructure <float, 1> (const VectorSpace< float, 1 > &,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< int > *,
                                         std::vector< std::vector< bool > > &);
#endif 
template void init_master_quadrature <double, 3>(const Element< double, 3 > &, const Element< double, 3 > &,
                                                 const Quadrature< double > &, Quadrature< double > &);
template void init_master_quadrature <double, 2>(const Element< double, 2 > &, const Element< double, 2 > &,
                                                 const Quadrature< double > &, Quadrature< double > &);
template void init_master_quadrature <double, 1>(const Element< double, 1 > &, const Element< double, 1 > &,
                                                 const Quadrature< double > &, Quadrature< double > &);
template void init_master_quadrature <float, 3>(const Element< float, 3 > &, const Element< float, 3 > &,
                                                 const Quadrature< float > &, Quadrature< float > &);
template void init_master_quadrature <float, 2>(const Element< float, 2 > &, const Element< float, 2 > &,
                                                 const Quadrature< float > &, Quadrature< float > &);
template void init_master_quadrature <float, 1>(const Element< float, 1 > &, const Element< float, 1 > &,
                                                 const Quadrature< float > &, Quadrature< float > &);

} // namespace hiflow
