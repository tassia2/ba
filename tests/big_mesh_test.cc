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

/// \author Thomas Gengenbach

#define BOOST_TEST_MODULE big_mesh

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>


#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hiflow.h"

using namespace std;
using namespace hiflow;
using namespace hiflow::la;
using namespace hiflow::mesh;
using namespace hiflow::doffem;

const TDim tdim = 3;
const GDim gdim = 3;

static const char *datadir = MESH_DATADIR;

BOOST_AUTO_TEST_CASE(big_mesh) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;

  MPI_Init(&argc, &argv);
  std::ofstream debug_file("dbg_output.log");
  LogKeeper::get_log("debug").set_target(&debug_file);

  // read mesh
  MPI_Comm comm = MPI_COMM_WORLD;
  std::string filename =
    std::string(datadir) + std::string("unitcube_refinementlevel_5.vtu");
  MeshPtr mesh = read_mesh_from_file(filename, tdim, gdim, &comm);

  BOOST_CHECK_EQUAL(35937, mesh->num_entities(0));
  BOOST_CHECK_EQUAL(32768, mesh->num_entities(tdim));

  LogKeeper::get_log("debug").flush();
  MPI_Finalize();
}
