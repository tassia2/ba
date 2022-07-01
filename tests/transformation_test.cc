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

/// \author Michael Schick

#define BOOST_TEST_MODULE transformation

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <string>

#include "hiflow.h"
#include "test.h"

using namespace std;
using namespace hiflow;
using namespace hiflow::mesh;
using namespace hiflow::doffem;

const TDim tdim = 3;
const GDim gdim = 3;

static const char *datadir = MESH_DATADIR;

BOOST_AUTO_TEST_CASE(transformation) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;
  
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  
  if (rank > 0)
  {
    return;
  }
  
  typedef Vec<gdim, double> Coord;
  
  // mesh
  const string filename = string(datadir) + string("unit_cube.inp");

  MeshBuilder *mb(new MeshDbViewBuilder(tdim, gdim));
  ScopedPtr< Reader >::Type reader(new UcdReader(mb));
  MeshPtr mesh;
  reader->read(filename.c_str(), mesh);

  std::vector< int > fe_params(1, 1);  
  std::vector< FEType > fe_ansatz (1, FEType::LAGRANGE);
  std::vector< bool > is_cg (1, true);
  
  VectorSpace< double, gdim > space;
  space.Init(*mesh, fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC);

  for (EntityIterator it = mesh->begin(gdim); it != mesh->end(gdim); ++it) {
    std::vector< double > cv;
    it->get_coordinates(cv);

    CONSOLE_OUTPUT(rank, "==================================");
    CONSOLE_OUTPUT(rank, "Coordinates before Transformation:");
    CONSOLE_OUTPUT(rank, "==================================");
    if (gdim == 2) {
      for (int vtx = 0; vtx < it->num_vertices(); ++vtx)
        CONSOLE_OUTPUT(rank, "ID: " << vtx << "  Coord x: " << cv[vtx * gdim]
                  << "  Coord y: " << cv[vtx * gdim + 1]);
    } else {
      for (int vtx = 0; vtx < it->num_vertices(); ++vtx)
        CONSOLE_OUTPUT(rank, "ID: " << vtx << "  Coord x: " << cv[vtx * gdim]
                  << "  Coord y: " << cv[vtx * gdim + 1]
                  << "  Coord z: " << cv[vtx * gdim + 2]);
    }
    std::cout << "==================================" << std::endl;

    std::vector< double > cv_at(cv.size());
    double x, y, z;
    for (int vtx = 0; vtx < it->num_vertices(); ++vtx) {
      Coord tmp;
      Coord pt(cv, vtx * gdim);

      space.fe_manager().get_cell_transformation(it->index())->inverse(pt, tmp);
      if (gdim == 2) {
        cv_at[vtx * gdim] = tmp[0];
        cv_at[vtx * gdim + 1] = tmp[1];
      } else {
        cv_at[vtx * gdim] = tmp[0];
        cv_at[vtx * gdim + 1] = tmp[1];
        cv_at[vtx * gdim + 2] = tmp[2];
      }
    }

    CONSOLE_OUTPUT(rank, "==================================");
    CONSOLE_OUTPUT(rank, "Coordinates after Transformation:");
    CONSOLE_OUTPUT(rank, "==================================");
    if (gdim == 2) {
      for (int vtx = 0; vtx < it->num_vertices(); ++vtx)
        CONSOLE_OUTPUT(rank, "ID: " << vtx << "  Coord x: " << cv_at[vtx * gdim]
                  << "  Coord y: " << cv_at[vtx * gdim + 1]);
    } else {
      for (int vtx = 0; vtx < it->num_vertices(); ++vtx)
        CONSOLE_OUTPUT(rank, "ID: " << vtx << "  Coord x: " << cv_at[vtx * gdim]
                  << "  Coord y: " << cv_at[vtx * gdim + 1]
                  << "  Coord z: " << cv_at[vtx * gdim + 2]);
    }
    CONSOLE_OUTPUT(rank, "==================================");

    std::vector< double > cv_bk(cv.size());

    for (int vtx = 0; vtx < it->num_vertices(); ++vtx) {
      Coord tmp0(cv_at, vtx*gdim);
      Coord tmp;

      if (gdim == 2) {
        tmp.set(0, space.fe_manager().get_cell_transformation(it->index())->x(tmp0));
        tmp.set(1, space.fe_manager().get_cell_transformation(it->index())->y(tmp0));
        cv_bk[vtx * gdim] = tmp[0];
        cv_bk[vtx * gdim + 1] = tmp[1];
      } else {
        tmp.set(0, space.fe_manager().get_cell_transformation(it->index())->x(tmp0));
        tmp.set(1, space.fe_manager().get_cell_transformation(it->index())->y(tmp0));
        tmp.set(2, space.fe_manager().get_cell_transformation(it->index())->z(tmp0));
        cv_bk[vtx * gdim] = tmp[0];
        cv_bk[vtx * gdim + 1] = tmp[1];
        cv_bk[vtx * gdim + 2] = tmp[2];
      }
    }

    CONSOLE_OUTPUT(rank, "==================================");
    CONSOLE_OUTPUT(rank, "Coordinates Back Transformation:");
    CONSOLE_OUTPUT(rank, "==================================");
    if (gdim == 2) {
      for (int vtx = 0; vtx < it->num_vertices(); ++vtx)
        CONSOLE_OUTPUT(rank, "ID: " << vtx << "  Coord x: " << cv_bk[vtx * gdim]
                  << "  Coord y: " << cv_bk[vtx * gdim + 1]);
    } else {
      for (int vtx = 0; vtx < it->num_vertices(); ++vtx)
        CONSOLE_OUTPUT(rank, "ID: " << vtx << "  Coord x: " << cv_bk[vtx * gdim]
                  << "  Coord y: " << cv_bk[vtx * gdim + 1]
                  << "  Coord z: " << cv_bk[vtx * gdim + 2]);
    }
    CONSOLE_OUTPUT(rank, "==================================");
  }
  MPI_Finalize();
}
