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

#define BOOST_TEST_MODULE petsc_vector

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include "hiflow.h"
#include "mpi.h"

using namespace hiflow::la;

/// Test fixture for PETScVectorTest

struct PETScVectorTest {
  PETScVectorTest() {

    //int argc = boost::unit_test::framework::master_test_suite().argc;
    //char** argv = boost::unit_test::framework::master_test_suite().argv;

    //MPI_Init(&argc, &argv);

    comm = MPI_COMM_WORLD;
    BOOST_CHECK(!(MPI_Comm_rank(comm, &mpi_rank)));
    BOOST_CHECK(!(MPI_Comm_size(comm, &mpi_num_procs)));

    // Setup couplings object
    laCouplings.Init(comm);

    // Generate global offsets
    local_dim = 2;
    global_dim = mpi_num_procs * local_dim;

    std::vector< int > global_offsets(mpi_num_procs + 1);
    global_offsets[0] = 0;
    for (int i = 0; i < mpi_num_procs; ++i)
      global_offsets[i + 1] = global_offsets[i] + local_dim;

    // Generate offdiag offsets
    // The first rank don't have off-diagonals since all border vertices are
    // allocated by the lowest rank.
    std::vector< int > offdiag_offsets(mpi_num_procs + 1);
    offdiag_offsets[0] = 0;
    for (int i = 0; i < mpi_num_procs; ++i)
      offdiag_offsets[i + 1] =
        offdiag_offsets[i] + (mpi_rank == i ? 0 : local_dim);

    // Generate offdiag cols
    int offdiag_cols_dim = (mpi_num_procs - 1) * local_dim;
    std::vector< int > offdiag_cols(offdiag_cols_dim);
    for (int i = 0; i < mpi_rank * local_dim; ++i)
      offdiag_cols[i] = i;
    for (int i = (mpi_rank + 1) * local_dim; i < mpi_num_procs * local_dim; ++i)
      offdiag_cols[i - local_dim] = i;

    // Initialize laCouplings
    laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                    offdiag_offsets);

  }

  ~PETScVectorTest() {
    laCouplings.Clear();
    //MPI_Finalize();
  }

  void set_values(PETScVector< double > &v) const {
    for (int i = 0; i != local_dim; ++i) {
      int global_index = 2 * mpi_rank + i;
      v.SetValue(global_index, global_index + 0.2);
    }
  }

  // leaving this out produces an error
  MPI_Comm comm;
  int mpi_rank;
  int mpi_num_procs;

  int local_dim;
  int global_dim;

  LaCouplings laCouplings;

};

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

BOOST_FIXTURE_TEST_CASE(default_constructor, PETScVectorTest) {
  PETScVector< double > v;
}

BOOST_FIXTURE_TEST_CASE(init, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
}

BOOST_FIXTURE_TEST_CASE(size_local, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  BOOST_CHECK_EQUAL(local_dim, v.size_local());
}

BOOST_FIXTURE_TEST_CASE(size_global, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  BOOST_CHECK_EQUAL(global_dim, v.size_global());
}

BOOST_FIXTURE_TEST_CASE(set_value, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  set_values(v);
}

BOOST_FIXTURE_TEST_CASE(zeros, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  for (int i = 0; i != global_dim; ++i)
    v.SetValue(i, i + 0.2);
  v.Zeros();
  BOOST_CHECK_EQUAL(0.0, v.Norm1());
}

BOOST_FIXTURE_TEST_CASE(get_value, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  set_values(v);
  for (int i = 0; i != local_dim; ++i) {
    int global_index = 2 * mpi_rank + i;
    BOOST_CHECK_EQUAL(global_index + 0.2, v.GetValue(global_index));
  }
}

BOOST_FIXTURE_TEST_CASE(add_value, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  set_values(v);
  for (int i = 0; i != local_dim; ++i) {
    int global_index = 2 * mpi_rank + i;
    v.Add(global_index, global_index + 0.2);
  }
  for (int i = 0; i != local_dim; ++i) {
    int global_index = 2 * mpi_rank + i;
    BOOST_CHECK_EQUAL(2 * (global_index + 0.2), v.GetValue(global_index));
  }
}

BOOST_FIXTURE_TEST_CASE(norm1, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  double value, norm = 0.0;
  for (int i = 0; i != global_dim; ++i) {
    value = i + 0.2;
    v.SetValue(i, value);
    norm += std::abs(value);
  }
  BOOST_CHECK_EQUAL(norm, v.Norm1());
}

BOOST_FIXTURE_TEST_CASE(norm2, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  double value, norm = 0.0;
  for (int i = 0; i != global_dim; ++i) {
    value = i + 0.2;
    v.SetValue(i, value);
    norm += value * value;
  }
  norm = std::sqrt(norm);
  BOOST_CHECK_EQUAL(norm, v.Norm2());
}

BOOST_FIXTURE_TEST_CASE(norMax, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  double value, norm = 0.0;
  for (int i = 0; i != global_dim; ++i) {
    value = i + 0.2;
    v.SetValue(i, value);
    norm = std::max(norm, std::abs(value));
  }
  BOOST_CHECK_EQUAL(norm, v.NormMax());
}

BOOST_FIXTURE_TEST_CASE(dot1, PETScVectorTest) {
  PETScVector< double > v1;
  v1.Init(comm, laCouplings);
  v1.SetValue(0, 7.2);
  v1.SetValue(1, 3.7);
  PETScVector< double > v2;
  v2.Init(comm, laCouplings);
  v2.SetValue(0, 2.7);
  v2.SetValue(1, 7.3);
  double result = 7.2 * 2.7 + 3.7 * 7.3;
  BOOST_CHECK_EQUAL(result, v1.Dot(v2));
  BOOST_CHECK_EQUAL(result, v2.Dot(v1));
}

BOOST_FIXTURE_TEST_CASE(scale, PETScVectorTest) {
  PETScVector< double > v;
  v.Init(comm, laCouplings);
  for (int i = 0; i != global_dim; ++i)
    v.SetValue(i, i + 0.2);
  v.Scale(2.0);
  BOOST_CHECK_EQUAL(((mpi_rank * local_dim + 0.2) * 2),
                    v.GetValue(mpi_rank * local_dim));
  BOOST_CHECK_EQUAL(((mpi_rank * local_dim + 1.2) * 2),
                    v.GetValue(mpi_rank * local_dim + 1));
}

