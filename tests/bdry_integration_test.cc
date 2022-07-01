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

/// \author Jonathan Schwegler

#define BOOST_TEST_MODULE bdry_integration

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
namespace tt = boost::test_tools;

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hiflow.h"

using namespace hiflow;
using namespace hiflow::la;
using namespace hiflow::mesh;
using namespace hiflow::doffem;

// parameters
const TDim tdim = 3;
const GDim gdim = 3;
const int DIM = 3;
const int fe_degree = 2;

static const char *datadir = MESH_DATADIR;

class Integrator : private AssemblyAssistant< tdim, double > {
public:
  Integrator() {}

  void operator()(const Element< double, gdim > &element, int facet_number,
                  const Quadrature< double > &quadrature,
                  std::vector< double > &value) {
    AssemblyAssistant< tdim, double >::initialize_for_facet(element, quadrature,
                                                            facet_number, false);

    const int num_q = num_quadrature_points();
    for (int q = 0; q < num_q; ++q) {
      // error
      const double one = 1.0;

      // multiply with weight and transformation factor
      value[facet_number] += w(q) * one * ds(q);
    }
  }
};

void compute_surface_area(const std::string filename, const int max_refinements,
                          const double correct_surface_area) {
  // read mesh
  MeshPtr mesh;
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    mesh = read_mesh_from_file(filename, tdim, gdim, 0);
    // refinements
    for (int refinements = 0; refinements < max_refinements; ++refinements) {
      mesh = mesh->refine();
    }
    LOG_DEBUG(1, "mesh->num_entities(0) == " << mesh->num_entities(0));
  }

  int uniform_ref_steps = 0;
  MeshPtr local_mesh = partition_and_distribute(mesh, 0, MPI_COMM_WORLD,
                       uniform_ref_steps, IMPL_DBVIEW);
  assert(local_mesh != 0);

  SharedVertexTable shared_verts;
  mesh = compute_ghost_cells(*local_mesh, MPI_COMM_WORLD, shared_verts);

  // create finite element space
  VectorSpace< double, DIM > space;
  std::vector< FEType > fe_ansatz (1);
  std::vector< bool > is_cg (1);
  std::vector < int >fe_degrees(1);

  is_cg[0] = 1;
  fe_ansatz[0] = FEType::LAGRANGE;
  fe_degrees[0] = fe_degree;
  
  space.Init(*mesh, fe_ansatz, is_cg, fe_degrees, hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC);
  

  // compute integral over the whole domain
  Integrator integrator;
  StandardGlobalAssembler< double, DIM > global_asm;

  double surface_area;
  global_asm.integrate_scalar_boundary(space, integrator, surface_area);

  // get local surface area from each process
  double global_surface_area;
  MPI_Reduce(&surface_area, &global_surface_area, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // test if result is correct
  if (rank == 0) {
    LOG_DEBUG(1,
              filename << " has the surface area == " << global_surface_area);
    std::cerr << "Surface area diff for = "
              << correct_surface_area - global_surface_area << "\n";
    BOOST_TEST(correct_surface_area - global_surface_area == 0.0,
               tt::tolerance(1e-6 * correct_surface_area));
  }
}

BOOST_AUTO_TEST_CASE(bdry_integration) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;

  MPI_Init(&argc, &argv);

  std::ofstream info_log("bdry_integration_test_info_output.log");
  LogKeeper::get_log("info").set_target(&info_log);
  std::ofstream debug_file("bdry_integration_test_output.log");
  LogKeeper::get_log("debug").set_target(&std::cout);

  // test geometries

  // TEST WITH SMALL TETRA GEOMETRY //////////////////////
  std::string filename = std::string(datadir) + std::string("one_tet.inp");
  int refinements = 4;
  double correct_surface_area = 1.5 + sqrt(2 * 1.5) / 2;
  //compute_surface_area(filename, refinements, correct_surface_area);

  // TEST WITH SMALL HEXA GEOMETRY ///////////////////////
  filename = std::string(datadir) + std::string("unit_cube.inp");
  refinements = 3;
  correct_surface_area = 6.0;
  compute_surface_area(filename, refinements, correct_surface_area);

  LogKeeper::get_log("debug").flush();
  MPI_Finalize();
}
