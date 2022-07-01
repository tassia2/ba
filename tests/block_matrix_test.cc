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

/// @author Simon Gawlok

#define BOOST_TEST_MODULE block_matrix

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#include "hiflow.h"
#include "mpi.h"

using namespace hiflow::la;
using namespace hiflow;

/// Test writing and reading values to/from HDF5 file in parallel

typedef boost::mpl::list< LADescriptorCoupledD
#ifdef WITH_HYPRE
,
LADescriptorHypreD
#endif
#ifdef WITH_PETSC
,
LADescriptorPETScD
#endif
#ifdef WITH_COMPLEX_PETSC
,
LADescriptorPETScD
#endif
>
MyTypeListBlockMatrix;

struct mpi_set {
  mpi_set() {
    int argc = boost::unit_test::framework::master_test_suite().argc;
    char** argv = boost::unit_test::framework::master_test_suite().argv;

    MPI_Init(&argc, &argv);
  }

  ~mpi_set() {
    MPI_Finalize();
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mpi_set);


/// Test adding and getting values to/from HypreMatrix

BOOST_AUTO_TEST_CASE_TEMPLATE(add_get_value, T, MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // first row
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower);
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower + 1);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    rows_diag.push_back(i);
    cols_diag.push_back(i - 1);
    rows_diag.push_back(i);
    cols_diag.push_back(i);
    rows_diag.push_back(i);
    cols_diag.push_back(i + 1);
  }
  // last row
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper - 1);
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  // first row
  matrix.Add(local_ilower, local_ilower, 2.);
  matrix.Add(local_ilower, local_ilower + 1, -1.);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    matrix.Add(i, i - 1, -1.);
    matrix.Add(i, i, 2.);
    matrix.Add(i, i + 1, -1.);
  }
  // last row
  matrix.Add(local_iupper, local_iupper - 1, -1.);
  matrix.Add(local_iupper, local_iupper, 2.);

  // Get values in matrix
  std::vector< int > rows;
  std::vector< int > cols;
  std::vector< double > val_res(local_dim * local_dim, 0.);
  for (int i = local_ilower; i <= local_iupper; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }
  matrix.GetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val_res));

  // Check values in val_res
  // first row
  BOOST_REQUIRE_CLOSE(2., val_res[0], 1.0e-10);
  BOOST_REQUIRE_CLOSE(-1., val_res[1], 1.0e-10);
  for (int j = 2; j <= local_iupper - local_ilower; ++j) {
    BOOST_REQUIRE_CLOSE(0., val_res[j], 1.0e-10);
  }

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // first cols
    for (int j = 0; j <= i - 2; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
    // tridiagonal PARTICULAR
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i - 1], 1.0e-10);
    BOOST_REQUIRE_CLOSE(2., val_res[i * local_dim + i], 1.0e-10);
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i + 1], 1.0e-10);
    // last cols
    for (int j = i + 2; j <= local_iupper - local_ilower - 1; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
  }

  // last row
  for (int j = 0; j <= local_iupper - local_ilower - 2; ++j) {
    BOOST_REQUIRE_CLOSE(0.,
                        val_res[(local_iupper - local_ilower) * local_dim + j], 1.0e-10);
  }
  BOOST_REQUIRE_CLOSE(-1., val_res[(local_iupper - local_ilower) * local_dim +
                                                                 (local_iupper - local_ilower - 1)], 1.0e-10);
  BOOST_REQUIRE_CLOSE(2., val_res[(local_iupper - local_ilower) * local_dim +
                                                                (local_iupper - local_ilower)], 1.0e-10);
}

/// Test setting values in HypreMatrix

BOOST_AUTO_TEST_CASE_TEMPLATE(set_value, T, MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // first row
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower);
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower + 1);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    rows_diag.push_back(i);
    cols_diag.push_back(i - 1);
    rows_diag.push_back(i);
    cols_diag.push_back(i);
    rows_diag.push_back(i);
    cols_diag.push_back(i + 1);
  }
  // last row
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper - 1);
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);
  
  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  // first row
  matrix.SetValue(local_ilower, local_ilower, 2.);
  matrix.SetValue(local_ilower, local_ilower + 1, -1.);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    matrix.SetValue(i, i - 1, -1.);
    matrix.SetValue(i, i, 2.);
    matrix.SetValue(i, i + 1, -1.);
  }
  // last row
  matrix.SetValue(local_iupper, local_iupper - 1, -1.);
  matrix.SetValue(local_iupper, local_iupper, 2.);

  // Get values in matrix
  std::vector< int > rows;
  std::vector< int > cols;
  std::vector< double > val_res(local_dim * local_dim, 0.);
  for (int i = local_ilower; i <= local_iupper; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }

  matrix.GetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val_res));

  // Check values in val_res
  // first row
  BOOST_REQUIRE_CLOSE(2., val_res[0], 1.0e-10);
  BOOST_REQUIRE_CLOSE(-1., val_res[1], 1.0e-10);
  for (int j = 2; j <= local_iupper - local_ilower; ++j) {
    BOOST_REQUIRE_CLOSE(0., val_res[j], 1.0e-10);
  }

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // first cols
    for (int j = 0; j <= i - 2; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
    // tridiagonal PARTICULAR
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i - 1], 1.0e-10);
    BOOST_REQUIRE_CLOSE(2., val_res[i * local_dim + i], 1.0e-10);
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i + 1], 1.0e-10);
    // last cols
    for (int j = i + 2; j <= local_iupper - local_ilower - 1; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
  }

  // last row
  for (int j = 0; j <= local_iupper - local_ilower - 2; ++j) {
    BOOST_REQUIRE_CLOSE(0.,
                        val_res[(local_iupper - local_ilower) * local_dim + j], 1.0e-10);
  }
  BOOST_REQUIRE_CLOSE(-1., val_res[(local_iupper - local_ilower) * local_dim +
                                                                 (local_iupper - local_ilower - 1)], 1.0e-10);
  BOOST_REQUIRE_CLOSE(2., val_res[(local_iupper - local_ilower) * local_dim +
                                                                (local_iupper - local_ilower)], 1.0e-10);
}

/// Test adding values (vectorized) to HypreMatrix

BOOST_AUTO_TEST_CASE_TEMPLATE(add_values, T, MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // first row
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower);
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower + 1);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    rows_diag.push_back(i);
    cols_diag.push_back(i - 1);
    rows_diag.push_back(i);
    cols_diag.push_back(i);
    rows_diag.push_back(i);
    cols_diag.push_back(i + 1);
  }
  // last row
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper - 1);
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  std::vector< int > rows;
  std::vector< int > cols;
  std::vector< double > val(local_dim * local_dim, 0.);
  for (int i = local_ilower; i <= local_iupper; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }
  // first row
  val[0] = 2.;
  val[1] = -1.;

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // tridiagonal PARTICULAR
    val[i * local_dim + i - 1] = -1.;
    val[i * local_dim + i] = 2.;
    val[i * local_dim + i + 1] = -1.;
    ;
  }

  // last row
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower - 1)] = -1.;
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower)] = 2.;

  matrix.Add(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
             vec2ptr(val));

  // Get values in matrix
  std::vector< double > val_res(local_dim * local_dim, 0.);

  matrix.GetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val_res));

  // Check values in val_res
  // first row
  BOOST_REQUIRE_CLOSE(2., val_res[0], 1.0e-10);
  BOOST_REQUIRE_CLOSE(-1., val_res[1], 1.0e-10);
  for (int j = 2; j <= local_iupper - local_ilower; ++j) {
    BOOST_REQUIRE_CLOSE(0., val_res[j], 1.0e-10);
  }

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // first cols
    for (int j = 0; j <= i - 2; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
    // tridiagonal PARTICULAR
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i - 1], 1.0e-10);
    BOOST_REQUIRE_CLOSE(2., val_res[i * local_dim + i], 1.0e-10);
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i + 1], 1.0e-10);
    // last cols
    for (int j = i + 2; j <= local_iupper - local_ilower - 1; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
  }

  // last row
  for (int j = 0; j <= local_iupper - local_ilower - 2; ++j) {
    BOOST_REQUIRE_CLOSE(0.,
                        val_res[(local_iupper - local_ilower) * local_dim + j], 1.0e-10);
  }
  BOOST_REQUIRE_CLOSE(-1., val_res[(local_iupper - local_ilower) * local_dim +
                                                                 (local_iupper - local_ilower - 1)], 1.0e-10);
  BOOST_REQUIRE_CLOSE(2., val_res[(local_iupper - local_ilower) * local_dim +
                                                                (local_iupper - local_ilower)], 1.0e-10);
}

/// Test setting values (vectorized) to HypreMatrix

BOOST_AUTO_TEST_CASE_TEMPLATE(set_values, T, MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // first row
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower);
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower + 1);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    rows_diag.push_back(i);
    cols_diag.push_back(i - 1);
    rows_diag.push_back(i);
    cols_diag.push_back(i);
    rows_diag.push_back(i);
    cols_diag.push_back(i + 1);
  }
  // last row
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper - 1);
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  std::vector< int > rows;
  std::vector< int > cols;
  std::vector< double > val(local_dim * local_dim, 0.);
  for (int i = local_ilower; i <= local_iupper; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }
  // first row
  val[0] = 2.;
  val[1] = -1.;

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // tridiagonal PARTICULAR
    val[i * local_dim + i - 1] = -1.;
    val[i * local_dim + i] = 2.;
    val[i * local_dim + i + 1] = -1.;
    ;
  }

  // last row
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower - 1)] = -1.;
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower)] = 2.;

  matrix.SetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val));

  // Get values in matrix
  std::vector< double > val_res(local_dim * local_dim, 0.);

  matrix.GetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val_res));

  // Check values in val_res
  // first row
  BOOST_REQUIRE_CLOSE(2., val_res[0], 1.0e-10);
  BOOST_REQUIRE_CLOSE(-1., val_res[1], 1.0e-10);
  for (int j = 2; j <= local_iupper - local_ilower; ++j) {
    BOOST_REQUIRE_CLOSE(0., val_res[j], 1.0e-10);
  }

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // first cols
    for (int j = 0; j <= i - 2; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
    // tridiagonal PARTICULAR
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i - 1], 1.0e-10);
    BOOST_REQUIRE_CLOSE(2., val_res[i * local_dim + i], 1.0e-10);
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i + 1], 1.0e-10);
    // last cols
    for (int j = i + 2; j <= local_iupper - local_ilower - 1; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
  }

  // last row
  for (int j = 0; j <= local_iupper - local_ilower - 2; ++j) {
    BOOST_REQUIRE_CLOSE(0.,
                        val_res[(local_iupper - local_ilower) * local_dim + j], 1.0e-10);
  }
  BOOST_REQUIRE_CLOSE(-1., val_res[(local_iupper - local_ilower) * local_dim +
                                                                 (local_iupper - local_ilower - 1)], 1.0e-10);
  BOOST_REQUIRE_CLOSE(2., val_res[(local_iupper - local_ilower) * local_dim +
                                                                (local_iupper - local_ilower)], 1.0e-10);
}

/// Test scaling of HypreMatrix

BOOST_AUTO_TEST_CASE_TEMPLATE(scale, T, MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // first row
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower);
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower + 1);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    rows_diag.push_back(i);
    cols_diag.push_back(i - 1);
    rows_diag.push_back(i);
    cols_diag.push_back(i);
    rows_diag.push_back(i);
    cols_diag.push_back(i + 1);
  }
  // last row
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper - 1);
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  std::vector< int > rows;
  std::vector< int > cols;
  std::vector< double > val(local_dim * local_dim, 0.);
  for (int i = local_ilower; i <= local_iupper; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }
  // first row
  val[0] = 2.;
  val[1] = -1.;

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // tridiagonal PARTICULAR
    val[i * local_dim + i - 1] = -1.;
    val[i * local_dim + i] = 2.;
    val[i * local_dim + i + 1] = -1.;
    ;
  }

  // last row
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower - 1)] = -1.;
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower)] = 2.;

  matrix.SetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val));

  // Scale matrix
  const double factor = 0.5;
  matrix.Scale(factor);

  // Get values in matrix
  std::vector< double > val_res(local_dim * local_dim, 0.);

  matrix.GetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val_res));

  // Check values in val_res
  // first row
  BOOST_REQUIRE_CLOSE(2. * factor, val_res[0], 1.0e-10);
  BOOST_REQUIRE_CLOSE(-1. * factor, val_res[1], 1.0e-10);
  for (int j = 2; j <= local_iupper - local_ilower; ++j) {
    BOOST_REQUIRE_CLOSE(0. * factor, val_res[j], 1.0e-10);
  }

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // first cols
    for (int j = 0; j <= i - 2; ++j) {
      BOOST_REQUIRE_CLOSE(0. * factor, val_res[i * local_dim + j], 1.0e-10);
    }
    // tridiagonal PARTICULAR
    BOOST_REQUIRE_CLOSE(-1. * factor, val_res[i * local_dim + i - 1], 1.0e-10);
    BOOST_REQUIRE_CLOSE(2. * factor, val_res[i * local_dim + i], 1.0e-10);
    BOOST_REQUIRE_CLOSE(-1. * factor, val_res[i * local_dim + i + 1], 1.0e-10);
    // last cols
    for (int j = i + 2; j <= local_iupper - local_ilower - 1; ++j) {
      BOOST_REQUIRE_CLOSE(0. * factor, val_res[i * local_dim + j], 1.0e-10);
    }
  }

  // last row
  for (int j = 0; j <= local_iupper - local_ilower - 2; ++j) {
    BOOST_REQUIRE_CLOSE(0. * factor,
                        val_res[(local_iupper - local_ilower) * local_dim + j], 1.0e-10);
  }
  BOOST_REQUIRE_CLOSE(-1. * factor,
                      val_res[(local_iupper - local_ilower) * local_dim +
                                                            (local_iupper - local_ilower - 1)], 1.0e-10);
  BOOST_REQUIRE_CLOSE(2. * factor,
                      val_res[(local_iupper - local_ilower) * local_dim +
                                                            (local_iupper - local_ilower)], 1.0e-10);
}

/// Test zeroing a HypreMatrix

BOOST_AUTO_TEST_CASE_TEMPLATE(zeros, T, MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // first row
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower);
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower + 1);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    rows_diag.push_back(i);
    cols_diag.push_back(i - 1);
    rows_diag.push_back(i);
    cols_diag.push_back(i);
    rows_diag.push_back(i);
    cols_diag.push_back(i + 1);
  }
  // last row
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper - 1);
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  std::vector< int > rows;
  std::vector< int > cols;
  std::vector< double > val(local_dim * local_dim, 0.);
  for (int i = local_ilower; i <= local_iupper; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }
  // first row
  val[0] = 2.;
  val[1] = -1.;

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // tridiagonal PARTICULAR
    val[i * local_dim + i - 1] = -1.;
    val[i * local_dim + i] = 2.;
    val[i * local_dim + i + 1] = -1.;
    ;
  }

  // last row
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower - 1)] = -1.;
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower)] = 2.;

  matrix.SetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val));

  // Set matrix to zero
  matrix.Zeros();

  // Get values in matrix
  std::vector< double > val_res(local_dim * local_dim, 0.);

  matrix.GetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val_res));

  // Check values in val_res
  // first row
  BOOST_REQUIRE_CLOSE(0., val_res[0], 1.0e-10);
  BOOST_REQUIRE_CLOSE(0., val_res[1], 1.0e-10);
  for (int j = 2; j <= local_iupper - local_ilower; ++j) {
    BOOST_REQUIRE_CLOSE(0., val_res[j], 1.0e-10);
  }

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // first cols
    for (int j = 0; j <= i - 2; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
    // tridiagonal PARTICULAR
    BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + i - 1], 1.0e-10);
    BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + i], 1.0e-10);
    BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + i + 1], 1.0e-10);
    // last cols
    for (int j = i + 2; j <= local_iupper - local_ilower - 1; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
  }

  // last row
  for (int j = 0; j <= local_iupper - local_ilower - 2; ++j) {
    BOOST_REQUIRE_CLOSE(0.,
                        val_res[(local_iupper - local_ilower) * local_dim + j], 1.0e-10);
  }
  BOOST_REQUIRE_CLOSE(0., val_res[(local_iupper - local_ilower) * local_dim +
                                                                (local_iupper - local_ilower - 1)], 1.0e-10);
  BOOST_REQUIRE_CLOSE(0., val_res[(local_iupper - local_ilower) * local_dim +
                                                                (local_iupper - local_ilower)], 1.0e-10);
}

/// Test making some rows of matrix diagonal

BOOST_AUTO_TEST_CASE_TEMPLATE(diagonalize_rows, T, MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // first row
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower);
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower + 1);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    rows_diag.push_back(i);
    cols_diag.push_back(i - 1);
    rows_diag.push_back(i);
    cols_diag.push_back(i);
    rows_diag.push_back(i);
    cols_diag.push_back(i + 1);
  }
  // last row
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper - 1);
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper);

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  // first row
  matrix.Add(local_ilower, local_ilower, 2.);
  matrix.Add(local_ilower, local_ilower + 1, -1.);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    matrix.Add(i, i - 1, -1.);
    matrix.Add(i, i, 2.);
    matrix.Add(i, i + 1, -1.);
  }
  // last row
  matrix.Add(local_iupper, local_iupper - 1, -1.);
  matrix.Add(local_iupper, local_iupper, 2.);

  // Get values in matrix
  std::vector< int > rows;
  std::vector< int > cols;
  std::vector< double > val_res(local_dim * local_dim, 0.);
  for (int i = local_ilower; i <= local_iupper; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }

  // diagonalize first and last local row
  std::vector< int > diag_indices;
  diag_indices.push_back(local_ilower);
  diag_indices.push_back(local_iupper);

  double diag_val = 10.;
  matrix.diagonalize_rows(vec2ptr(diag_indices), diag_indices.size(), diag_val);

  matrix.GetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val_res));

  // Check values in val_res
  // first row
  BOOST_REQUIRE_CLOSE(diag_val, val_res[0], 1.0e-10);
  BOOST_REQUIRE_CLOSE(0., val_res[1], 1.0e-10);
  for (int j = 2; j <= local_iupper - local_ilower; ++j) {
    BOOST_REQUIRE_CLOSE(0., val_res[j], 1.0e-10);
  }

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // first cols
    for (int j = 0; j <= i - 2; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
    // tridiagonal PARTICULAR
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i - 1], 1.0e-10);
    BOOST_REQUIRE_CLOSE(2., val_res[i * local_dim + i], 1.0e-10);
    BOOST_REQUIRE_CLOSE(-1., val_res[i * local_dim + i + 1], 1.0e-10);
    // last cols
    for (int j = i + 2; j <= local_iupper - local_ilower - 1; ++j) {
      BOOST_REQUIRE_CLOSE(0., val_res[i * local_dim + j], 1.0e-10);
    }
  }

  // last row
  for (int j = 0; j <= local_iupper - local_ilower - 2; ++j) {
    BOOST_REQUIRE_CLOSE(0.,
                        val_res[(local_iupper - local_ilower) * local_dim + j], 1.0e-10);
  }
  BOOST_REQUIRE_CLOSE(0., val_res[(local_iupper - local_ilower) * local_dim +
                                                                (local_iupper - local_ilower - 1)], 1.0e-10);
  BOOST_REQUIRE_CLOSE(diag_val,
                      val_res[(local_iupper - local_ilower) * local_dim +
                                                            (local_iupper - local_ilower)], 1.0e-10);
}

/// Test Matrix-Vector Multiplication

BOOST_AUTO_TEST_CASE_TEMPLATE(vector_mult, T, MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // first row
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower);
  rows_diag.push_back(local_ilower);
  cols_diag.push_back(local_ilower + 1);

  // rows except first and last one
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    rows_diag.push_back(i);
    cols_diag.push_back(i - 1);
    rows_diag.push_back(i);
    cols_diag.push_back(i);
    rows_diag.push_back(i);
    cols_diag.push_back(i + 1);
  }
  // last row
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper - 1);
  rows_diag.push_back(local_iupper);
  cols_diag.push_back(local_iupper);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  std::vector< int > rows;
  std::vector< int > cols;
  std::vector< double > val(local_dim * local_dim, 0.);
  for (int i = local_ilower; i <= local_iupper; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }
  // first row
  val[0] = 2.;
  val[1] = -1.;

  // rows except first and last one
  for (int i = 1; i <= local_iupper - local_ilower - 1; ++i) {
    // tridiagonal PARTICULAR
    val[i * local_dim + i - 1] = -1.;
    val[i * local_dim + i] = 2.;
    val[i * local_dim + i + 1] = -1.;
    ;
  }

  // last row
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower - 1)] = -1.;
  val[(local_iupper - local_ilower) * local_dim +
                                    (local_iupper - local_ilower)] = 2.;

  matrix.SetValues(vec2ptr(rows), rows.size(), vec2ptr(cols), cols.size(),
                   vec2ptr(val));

  BlockVector< T > in, out;
  in.Init(comm, laCouplings, block_manager);
  out.Init(comm, laCouplings, block_manager);

  // Set different vector components
  std::vector< int > ind;
  std::vector< double > vals;
  for (int i = local_ilower; i <= local_iupper; ++i) {
    ind.push_back(i);
    vals.push_back(1.);
  }

  in.Add(vec2ptr(ind), ind.size(), vec2ptr(vals));

  matrix.VectorMult(in, &out);

  // Check for correct result
  BOOST_REQUIRE_CLOSE(1., out.GetValue(local_ilower), 1.0e-10);
  for (int i = local_ilower + 1; i <= local_iupper - 1; ++i) {
    BOOST_REQUIRE_CLOSE(0., out.GetValue(i), 1.0e-10);
  }
  BOOST_REQUIRE_CLOSE(1., out.GetValue(local_iupper), 1.0e-10);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vector_mult_with_offdiag, T,
                                 MyTypeListBlockMatrix) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate dimensions
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set up input and output vectors
  std::vector< int > global_offsets(num_procs + 1);
  for (int i = 0; i < num_procs + 1; ++i) {
    global_offsets[i] = i * local_dim;
  }

  // Generate offdiag offsets
  // The first rank don't have off-diagonals since all border vertices are
  // allocated by the lowest rank.
  std::vector< int > offdiag_offsets(num_procs + 1, 0);

  // Generate offdiag cols
  std::vector< int > offdiag_cols;

  if (rank > 0) {
    offdiag_cols.push_back(local_ilower - 1);
    for (int i = rank; i < num_procs + 1; ++i) {
      offdiag_offsets[i] += 1;
    }
  }

  if (rank < num_procs - 1) {
    offdiag_cols.push_back(local_iupper + 1);
    for (int i = rank + 2; i < num_procs + 1; ++i) {
      offdiag_offsets[i] += 1;
    }
  }

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Generate sparsity structure
  std::vector< int > rows_diag;
  std::vector< int > cols_diag;
  std::vector< int > rows_offdiag;
  std::vector< int > cols_offdiag;

  // rows
  for (int i = local_ilower; i <= local_iupper; ++i) {
    if (i > 0) {
      if (i == local_ilower) {
        rows_offdiag.push_back(i);
        cols_offdiag.push_back(i - 1);
      } else {
        rows_diag.push_back(i);
        cols_diag.push_back(i - 1);
      }
    }

    rows_diag.push_back(i);
    cols_diag.push_back(i);

    if (i < global_dim - 1) {
      if (i == local_iupper) {
        rows_offdiag.push_back(i);
        cols_offdiag.push_back(i + 1);
      } else {
        rows_diag.push_back(i);
        cols_diag.push_back(i + 1);
      }
    }
  }

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = local_ilower - 1; i <= local_iupper + 1; ++i) {
    if (i >= 0 && i < global_dim) {
      block_dofs[i % 2].push_back(i);
    }
  }

  // Create and initialize HypreMatrix
  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  BlockMatrix< T > matrix;
  matrix.Init(comm, block_manager);

  matrix.InitStructure(vec2ptr(rows_diag), vec2ptr(cols_diag), rows_diag.size(),
                       vec2ptr(rows_offdiag), vec2ptr(cols_offdiag),
                       rows_offdiag.size());

  // Add values to matrix
  // rows
  for (int i = local_ilower; i <= local_iupper; ++i) {
    if (i > 0) {
      matrix.Add(i, i - 1, -1.);
    }
    matrix.Add(i, i, 2.);
    if (i < global_dim - 1) {
      matrix.Add(i, i + 1, -1.);
    }
  }

  BlockVector< T > in, out;
  in.Init(comm, laCouplings, block_manager);
  out.Init(comm, laCouplings, block_manager);

  // Set different vector components
  std::vector< int > ind;
  std::vector< double > vals;
  for (int i = local_ilower; i <= local_iupper; ++i) {
    ind.push_back(i);
    vals.push_back(1.);
  }

  in.Add(vec2ptr(ind), ind.size(), vec2ptr(vals));
  out.Zeros();

  matrix.VectorMult(in, &out);

  // Check for correct result
  for (int i = local_ilower; i <= local_iupper; ++i) {
    if (i == 0 || i == global_dim - 1) {
      BOOST_REQUIRE_CLOSE(1., out.GetValue(i), 1.0e-10);
    } else {
      BOOST_REQUIRE_CLOSE(0., out.GetValue(i), 1.0e-10);
    }
  }
}
