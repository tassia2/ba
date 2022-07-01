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

/// \author Staffan Ronnas

#define BOOST_TEST_MODULE eval_fe_solution

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "hiflow.h"

using namespace std;
using namespace hiflow;
using namespace hiflow::mesh;

const TDim tdim = 3;
const GDim gdim = 3;

static const char *datadir = MESH_DATADIR;

// Refinement level of unit cube.
const int REF_LEVEL = 1;

// Polynomial element degree. This should be chosen so that TestFunction can be
// represented exactly (to machine precision).
const int FE_DEGREE = 2;

// Number of points in each direction for evaluation (including end
// points). This should not correspond to dof nodes for the given FE_DEGREE.
const int N = 4;

// Absolute tolerance for error in FE function evaluation.
const double TOL = 1.e-12;

struct TestFunction {

  double operator()(const Vec<gdim, double > &pt) const {
    return 2. * pt[0] + pt[1] - 3. * pt[2];
  }
};

BOOST_AUTO_TEST_CASE(evaluate_fe_solution, *utf::tolerance(TOL)) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;

  MPI_Init(&argc, &argv);
  {
    std::string filename = std::string(datadir) + std::string("unit_cube.inp");

    // Create a VectorSpace.
    MeshPtr mesh = read_mesh_from_file(filename, tdim, gdim, 0);

    for (int i = 0; i < REF_LEVEL; ++i) {
      mesh = mesh->refine();
    }

    VectorSpace< double, gdim > space;
    space.Init(FE_DEGREE, *mesh);

    const int ndofs = space.dof().ndofs_global();

    // Project TestFunction into discrete space. This function should be
    // represented "exactly".
    TestFunction f;
    std::vector< double > dof_values(ndofs, 0);

    for (EntityIterator it = mesh->begin(tdim), end = mesh->end(tdim);
         it != end; ++it) {
      std::vector< Vec<gdim, double > > coords;
      std::vector< int > dofs;

      space.dof().aaa_get_coord_on_cell(0, it->index(), coords);
      space.dof().aaa_get_dofs_on_cell(0, it->index(), dofs);

      for (int i = 0; i < static_cast< int >(dofs.size()); ++i) {
        dof_values.at(dofs.at(i)) = f(coords.at(i));
      }
    }

    // Evaluate the corresponding FE function on a grid inside each cell, to
    // verify that the functions
    // Element::evaluate_fe_solution() and VectorSpace::get_solution_value()
    // work correctly.

    // First, create a uniform grid on the reference cell.
    std::vector< Vec<gdim, double > > grid;
    Vec<gdim, double > pt;

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
          pt[0] = i / double(N - 1);
          pt[1] = j / double(N - 1);
          pt[2] = k / double(N - 1);
          grid.push_back(pt);
        }
      }
    }

    int k = 0;

    for (EntityIterator it = mesh->begin(tdim), end = mesh->end(tdim);
         it != end; ++it) {
<<<<<<< HEAD:tests/old/eval_fe_solution_test.cc
      Element< double, gdim > elem(space, it->index());
      const doffem::CellTransformation< double, gdim > *trans =
          elem.get_cell_transformation();
=======
      Element< double > elem(space, it->index());
      const doffem::CellTransformation< double > *trans =
        elem.get_cell_transformation();
>>>>>>> master:tests/eval_fe_solution_test.cc

      for (int i = 0; i < static_cast< int >(grid.size()); ++i) {
        // Transform grid point from reference cell to physical cell.
        pt[0] = trans->x(grid.at(i));
        pt[1] = trans->y(grid.at(i));
        pt[2] = trans->z(grid.at(i));

        // Compute exact value of solution.
        const double exact_val = f(pt);

        // Compute FE value of solution.
        const double fe_val = elem.evaluate_fe_solution(0, pt, dof_values);

        BOOST_TEST(fe_val - exact_val == 0.);
        ++k;
      }
    }

    std::clog << "Values correspond at all " << k << " points!\n";
  }

  MPI_Finalize();

}
