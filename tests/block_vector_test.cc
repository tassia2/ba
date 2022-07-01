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

#define BOOST_TEST_MODULE block_vector

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#include "hiflow.h"
#include "mpi.h"

#include <typeinfo>

using namespace hiflow::la;
using namespace hiflow;

/// Test writing and reading values to/from HDF5 file in parallel

typedef boost::mpl::list< BlockVector< LADescriptorCoupledD >
#ifdef WITH_HYPRE
,
BlockVector< LADescriptorHypreD >
#endif
#ifdef WITH_PETSC
,
BlockVector< LADescriptorPETScD >
#endif
#ifdef WITH_COMPLEX_PETSC
,
BlockVector< LADescriptorPETScD >
#endif
>
MyTypeListBlockVector;

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

BOOST_AUTO_TEST_CASE_TEMPLATE(set_get_value, T, MyTypeListBlockVector) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.SetValue(i, static_cast< double >(i));
  }

  // Get vector components and check for correct values
  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(static_cast< double >(i), vec.GetValue(i), 1.0e-10);
  }
}

/// Test setting values (vectorized) in HpreVector

BOOST_AUTO_TEST_CASE_TEMPLATE(set_get_values, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);
  
  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  std::vector< int > ind;
  std::vector< double > val;
  for (int i = local_ilower; i <= local_iupper; ++i) {
    ind.push_back(i);
    val.push_back(static_cast< double >(i));
  }

  vec.SetValues(vec2ptr(ind), ind.size(), vec2ptr(val));

  // Get vector components and check for correct values
  std::vector< double > val_res(val.size());
  vec.GetValues(vec2ptr(ind), ind.size(), vec2ptr(val_res));

  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(static_cast< double >(i), val_res[i - local_ilower],
                        1.0e-10);
  }
}

/// Test adding value in HpreVector

BOOST_AUTO_TEST_CASE_TEMPLATE(add_value, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.Add(i, static_cast< double >(i));
  }

  // Get vector components and check for correct values
  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(static_cast< double >(i), vec.GetValue(i), 1.0e-10);
  }
}

/// Test adding values (vectorized) in HpreVector

BOOST_AUTO_TEST_CASE_TEMPLATE(add_values, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  std::vector< int > ind;
  std::vector< double > val;
  for (int i = local_ilower; i <= local_iupper; ++i) {
    ind.push_back(i);
    val.push_back(static_cast< double >(i));
  }

  vec.Add(vec2ptr(ind), ind.size(), vec2ptr(val));

  // Get vector components and check for correct values
  std::vector< double > val_res(val.size());
  vec.GetValues(vec2ptr(ind), ind.size(), vec2ptr(val_res));

  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(static_cast< double >(i), val_res[i - local_ilower],
                        1.0e-10);
  }
}

/// Test cloning HypreVector without content

BOOST_AUTO_TEST_CASE_TEMPLATE(clone_without_content, T,
                                 MyTypeListBlockVector) {
  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);
  
  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // Clone vector without content
  T vec2;
  vec2.CloneFromWithoutContent(vec);
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // Check for correct dimensions
  BOOST_CHECK_EQUAL(vec.size_local(), vec2.size_local());
  BOOST_CHECK_EQUAL(vec.size_global(), vec2.size_global());
}

/// Test cloning complete HypreVector

BOOST_AUTO_TEST_CASE_TEMPLATE(clone1, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.Add(i, static_cast< double >(i));
  }

  // Clone to different HypreVector
  T vec2;
  vec2.CloneFrom(vec);
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // Check for correct dimensions
  BOOST_CHECK_EQUAL(vec.size_local(), vec2.size_local());
  BOOST_CHECK_EQUAL(vec.size_global(), vec2.size_global());

  // Get vector components and check for correct values
  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(static_cast< double >(i), vec2.GetValue(i), 1.0e-10);
  }
}

// Test dot product

BOOST_AUTO_TEST_CASE_TEMPLATE(dot1, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.Add(i, static_cast< double >(i));
  }

  // Clone structure to different HypreVector
  T vec2;
  vec2.CloneFromWithoutContent(vec);
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // Check for correct dimensions
  BOOST_CHECK_EQUAL(vec.size_local(), vec2.size_local());
  BOOST_CHECK_EQUAL(vec.size_global(), vec2.size_global());

  // Set components of second vector
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec2.Add(i, static_cast< double >(i + 2));
  }

  // Compute dot product
  double dot_res = vec.Dot(vec2);

  // Maximum component index
  const double N = static_cast< double >(global_dim - 1);
  const double res_expected =
    (1. / 6.) * N * (N + 1.) * (2. * N + 1.) + N * (N + 1.);

  BOOST_REQUIRE_CLOSE(res_expected, dot_res, 1.0e-10);
}

// Test axpy

BOOST_AUTO_TEST_CASE_TEMPLATE(axpy, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.Add(i, static_cast< double >(i));
  }

  // Clone structure to different HypreVector
  T vec2;
  vec2.CloneFromWithoutContent(vec);
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // Check for correct dimensions
  BOOST_CHECK_EQUAL(vec.size_local(), vec2.size_local());
  BOOST_CHECK_EQUAL(vec.size_global(), vec2.size_global());

  // Set components of second vector
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec2.Add(i, static_cast< double >(i + 2));
  }

  const double factor = 4.0;

  vec.Axpy(vec2, factor);

  // Get vector components and check for correct values
  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(static_cast< double >(i + factor * (i + 2)),
                        vec.GetValue(i), 1.0e-10);
  }
}

/// Test scaling a vector

BOOST_AUTO_TEST_CASE_TEMPLATE(scale, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);
  
  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.Add(i, static_cast< double >(i));
  }

  // Scale vector
  const double factor = 0.5;
  vec.Scale(factor);

  // Get vector components and check for correct values
  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(static_cast< double >(i * factor), vec.GetValue(i),
                        1.0e-10);
  }
}

/// Test scaling vector and adding another vector

BOOST_AUTO_TEST_CASE_TEMPLATE(scale_add, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.Add(i, static_cast< double >(i));
  }

  // Clone structure to different HypreVector
  T vec2;
  vec2.CloneFromWithoutContent(vec);
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec2.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // Check for correct dimensions
  BOOST_CHECK_EQUAL(vec.size_local(), vec2.size_local());
  BOOST_CHECK_EQUAL(vec.size_global(), vec2.size_global());

  // Set components of second vector
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec2.Add(i, static_cast< double >(i + 2));
  }

  const double factor = 4.0;

  vec.ScaleAdd(vec2, factor);

  // Get vector components and check for correct values
  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(static_cast< double >(i * factor + (i + 2)),
                        vec.GetValue(i), 1.0e-10);
  }
}

/// Test norm2

BOOST_AUTO_TEST_CASE_TEMPLATE(norm2, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);
  
  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.Add(i, static_cast< double >(i));
  }

  // Compute dot product
  double norm_res = vec.Norm2();

  // Maximum component index
  const double N = static_cast< double >(global_dim - 1);
  const double res_expected =
    std::sqrt((1. / 6.) * N * (N + 1.) * (2. * N + 1.));

  BOOST_REQUIRE_CLOSE(res_expected, norm_res, 1.0e-10);
}

/// Test zeroing a HypreVector

BOOST_AUTO_TEST_CASE_TEMPLATE(zeros, T, MyTypeListBlockVector) {

  MPI_Comm comm(MPI_COMM_WORLD);

  int rank, num_procs;
  BOOST_CHECK(!(MPI_Comm_rank(comm, &rank)));
  BOOST_CHECK(!(MPI_Comm_size(comm, &num_procs)));

  // Setup couplings object
  LaCouplings laCouplings;
  laCouplings.Init(comm);

  // Generate global offsets
  int local_dim = 10;
  int global_dim = num_procs * local_dim;

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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(vec.GetBlock(i).ghost().get_size(), 0);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_local(), local_dim / 2);
    BOOST_CHECK_EQUAL(vec.GetBlock(i).size_global(), local_dim / 2 * num_procs);
  }

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  std::vector< int > ind;
  std::vector< double > val;
  for (int i = local_ilower; i <= local_iupper; ++i) {
    ind.push_back(i);
    val.push_back(static_cast< double >(i));
  }

  vec.Add(vec2ptr(ind), ind.size(), vec2ptr(val));

  // Set vector to zero
  vec.Zeros();

  // Get vector components and check for correct values
  std::vector< double > val_res(val.size());
  vec.GetValues(vec2ptr(ind), ind.size(), vec2ptr(val_res));

  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(0., val_res[i - local_ilower], 1.0e-10);
  }
}

/// Test GetAllDofsAndValues

BOOST_AUTO_TEST_CASE_TEMPLATE(get_all_dofs_and_values, T,
                                 MyTypeListBlockVector) {
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
    offdiag_cols.push_back(local_ilower - 2);
    offdiag_cols.push_back(local_ilower - 1);
    for (int i = rank; i < num_procs + 1; ++i) {
      offdiag_offsets[i] += 2;
    }
  }

  if (rank < num_procs - 1) {
    offdiag_cols.push_back(local_iupper + 1);
    offdiag_cols.push_back(local_iupper + 2);
    for (int i = rank + 2; i < num_procs + 1; ++i) {
      offdiag_offsets[i] += 2;
    }
  }

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = local_ilower - 2; i <= local_iupper + 2; ++i) {
    if (i >= 0 && i < global_dim) {
      block_dofs[i % 2].push_back(i);
    }
  }

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  T vec;
  vec.Init(comm, laCouplings, block_manager);

  // Set different vector components
  std::vector< int > ind;
  std::vector< double > vals;
  for (int i = local_ilower; i <= local_iupper; ++i) {
    ind.push_back(i);
    vals.push_back(static_cast< double >(i));
  }

  vec.Add(vec2ptr(ind), ind.size(), vec2ptr(vals));

  vec.Update();

  std::vector< int > dof_ids;
  std::vector< double > dof_vals;

  vec.GetAllDofsAndValues(dof_ids, dof_vals);

  // Check for correct result
  for (int i = 0; i < dof_ids.size(); ++i) {
    BOOST_REQUIRE_CLOSE(dof_vals[i], static_cast< double >(dof_ids[i]), 1.0e-10);
  }
}
