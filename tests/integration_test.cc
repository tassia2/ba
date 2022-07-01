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

#define BOOST_TEST_MODULE integration

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


static const char *datadir = MESH_DATADIR;

class Integrator : private AssemblyAssistant< tdim, double > {
public:
  Integrator() {}

  void operator()(const Element< double, gdim > &element,
                  const Quadrature< double > &quadrature, double &value) {
    AssemblyAssistant< tdim, double >::initialize_for_element(element,
                                                              quadrature, false);
    const int num_q = num_quadrature_points();

    for (int q = 0; q < num_q; ++q) {
      // error
      const double one = 1.0;

      // multiply with weight and transformation factor
      value += w(q) * one * std::abs(detJ(q));
    }
  }
};

void compute_volume(const std::string filename, const int max_refinements,
                    const double correct_volume) {
  // read mesh
  MeshPtr mesh;
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    mesh = read_mesh_from_file(filename, tdim, gdim, 0);
    // refinenements
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
  std::vector < int >fe_degree(1);

  is_cg[0] = 1;
  fe_ansatz[0] = FEType::LAGRANGE;
  fe_degree[0] = 2;
  
  space.Init(*mesh, fe_ansatz, is_cg, fe_degree, hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC);
  

  // compute integral over the whole domain
  Integrator integrator;
  StandardGlobalAssembler< double, DIM > global_asm;
  double volume = 0.0;
  global_asm.integrate_scalar(space, integrator, volume);

  // get local volume from each process
  double global_volume;
  MPI_Reduce(&volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // test if result is correct
  if (rank == 0) {
    LOG_DEBUG(1, filename << " has the volume == " << global_volume);
    std::cerr << "Vol diff = " << correct_volume - global_volume << "\n";
    BOOST_TEST(correct_volume - global_volume == 0.0, tt::tolerance(1e-6 * correct_volume));
  }
}


BOOST_AUTO_TEST_CASE(integration) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;

  MPI_Init(&argc, &argv);

  std::ofstream debug_file("integration_test_output.log");
  LogKeeper::get_log("debug").set_target(&std::cerr);

  // test geometries

  // TEST WITH SMALL TETRA GEOMETRY //////////////////////
  std::string filename =
      std::string(datadir) + std::string("two_tetras_3d.inp");
  int refinements = 4;
  double correct_volume = 0.3875;
  //compute_volume(filename, refinements, correct_volume);    //Tetra elements not yet implementes

  // TEST WITH SMALL HEXA GEOMETRY ///////////////////////
  filename = std::string(datadir) + std::string("unit_cube.inp");
  refinements = 3;
  correct_volume = 1.0;
  compute_volume(filename, refinements, correct_volume);

  // TEST WITH DFG BENCHMARK /////////////////////////////
  filename = std::string(datadir) + std::string("dfg_bench3d_cyl.vtu");
  refinements = 0;
  correct_volume = 0.417111; // according to paraview
  //compute_volume(filename, refinements, correct_volume);    //Functionality not yet implemented

  LogKeeper::get_log("debug").flush();
  MPI_Finalize();
}
