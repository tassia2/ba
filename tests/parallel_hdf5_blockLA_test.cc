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

#define BOOST_TEST_MODULE parallel_hdf5_blockLA

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
MyTypeListBlockLA;


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


BOOST_AUTO_TEST_CASE_TEMPLATE(write_read_hdf5, T, MyTypeListBlockLA) {
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

  // Set block Dofs (only needed in case in BlockVector tests
  std::vector< std::vector< int > > block_dofs(2);
  for (int i = 0; i < local_dim / 2; ++i) {
    block_dofs[0].push_back(global_offsets[rank] + i);
  }
  for (int i = local_dim / 2; i < local_dim; ++i) {
    block_dofs[1].push_back(global_offsets[rank] + i);
  }

  // Initialize laCouplings
  laCouplings.InitializeCouplings(global_offsets, offdiag_cols,
                                  offdiag_offsets);

  BlockVector< T > vec;

  std::shared_ptr<BlockManager> block_manager = std::shared_ptr<BlockManager> (new BlockManager());
  block_manager->Init(comm, laCouplings, block_dofs);

  vec.Init(comm, laCouplings, block_manager);

  // compute first and last component of current process
  int local_ilower = rank * local_dim;
  int local_iupper = (rank + 1) * local_dim - 1;

  // Set different vector components
  for (int i = local_ilower; i <= local_iupper; ++i) {
    vec.SetValue(i, static_cast< double >(i));
  }

  // Get vector implementation type
  std::ostringstream vector_type;
  if (typeid(T) == typeid(LADescriptorCoupledD)) {
    vector_type << "CoupledVector";
  }
#ifdef WITH_HYPRE
  else if (typeid(T) == typeid(LADescriptorHypreD)) {
    vector_type << "HypreVector";
  }
#endif
#if defined(WITH_PETSC) || defined(WITH_COMPLEX_PETSC)
  else if (typeid(T) == typeid(LADescriptorPETScD)) {
    vector_type << "PETScVector";
  }
#endif

  std::stringstream filename;
  filename << num_procs << "_" << vector_type.str()
           << "_ParallelHDF5BlockLA_Test.h5";
  const std::string file_name = filename.str();
  std::ostringstream vec_name;
  vec_name << "_vec_";
  vec.WriteHDF5(file_name, "solution", vec_name.str());

  // Create second vector to read in written values
  BlockVector< T > vec_check;
  vec_check.Init(comm, laCouplings, block_manager);
  vec_check.ReadHDF5(file_name, "solution", vec_name.str());

  // Get vector components and check for correct values
  for (int i = local_ilower; i <= local_iupper; ++i) {
    BOOST_REQUIRE_CLOSE(vec_check.GetValue(i), vec.GetValue(i), 1.0e-10);
  }
}
