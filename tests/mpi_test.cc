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

#define BOOST_TEST_MODULE mpi

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <mpi.h>

BOOST_AUTO_TEST_CASE(mpi) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;

  MPI_Init(&argc, &argv);

  const MPI_Comm my_comm = MPI_COMM_WORLD;

  int rank, num_ranks;

  MPI_Comm_rank(my_comm, &rank);
  MPI_Comm_size(my_comm, &num_ranks);

  std::cout << "Rank = " << rank << ", num ranks = " << num_ranks << "\n";

  MPI_Finalize();

}
