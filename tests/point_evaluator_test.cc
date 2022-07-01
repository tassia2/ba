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

#define BOOST_TEST_MODULE point_evaluator3D

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

static const char *PARAM_FILENAME = "point_eval_test.xml";
#ifndef MESH_DATADIR
#define MESH_DATADIR "./"
#endif
static const char *DATADIR = MESH_DATADIR;

// System includes.
#include "hiflow.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

// All names are imported for simplicity.
using namespace hiflow;
using namespace hiflow::doffem;
using namespace hiflow::la;
using namespace hiflow::mesh;

// Shorten some datatypes with typedefs.
typedef LADescriptorCoupledD LAD;
typedef LAD::DataType Scalar;
typedef LAD::VectorType Vector;
typedef LAD::MatrixType Matrix;

// Rank of the master process.
const int MASTER_RANK = 0;

// Dimension of the problem.
const int DIMENSION = 2;

typedef Vec<DIMENSION, double> Coord;

Vec<DIMENSION, double> exact_sol(Coord pt) {

  return pt;
}

class LocalMassMatrixAssembler
  : private AssemblyAssistant< DIMENSION, double > {
public:
  void operator()(const Element< double, DIMENSION > &element,
                  const Quadrature< double > &quadrature, LocalMatrix &lm) {
    AssemblyAssistant< DIMENSION, double >::initialize_for_element(element,
                                                                   quadrature,
                                                                   false);

    // compute local matrix
    const int num_q = num_quadrature_points();
    for (int q = 0; q < num_q; ++q) 
    {
      const double wq = w(q);
     
      for (int i = 0; i < this->num_dofs_total(); ++i) 
      {
        for (int j = 0; j < this->num_dofs_total(); ++j) 
        {
          for (int v = 0; v<DIMENSION; ++v)
          {
            lm(i, j) += wq * this->Phi(j, q, v) * this->Phi(i, q, v) * std::abs(detJ(q));
          }
        }
      }
    }
  }

  void operator()(const Element< double, DIMENSION > &element,
                  const Quadrature< double > &quadrature, LocalVector &lv) {
    AssemblyAssistant< DIMENSION, double >::initialize_for_element(element,
                                                                   quadrature,
                                                                   false);
    // compute local vector
    const int num_q = num_quadrature_points();
    for (int q = 0; q < num_q; ++q) 
    {
      Vec<DIMENSION, double> rhs = f(x(q));
      const double wq = w(q);
      for (int i = 0; i < this->num_dofs_total(); ++i) 
      {
        for (int v = 0; v<DIMENSION; ++v)
        {
          lv[i] += wq * rhs[v] * this->Phi(i, q, v) * std::abs(detJ(q));
        }
      }
    }
  }

  Vec<DIMENSION, double> f(Coord pt) {
    return exact_sol(pt);
  }
};

class PointEvalTest {
public:
  PointEvalTest(const std::string param_filename)
    : comm_(MPI_COMM_WORLD), rank_(-1), num_partitions_(-1),
      params_(DATADIR + param_filename, MASTER_RANK, MPI_COMM_WORLD), rhs_(0),
      sol_(0) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &num_partitions_);
  }

  // Main algorithm

  void run() {
    // Construct / read in the initial mesh.
    build_initial_mesh();
    // Initialize space and linear algebra.
    prepare_system();
    // Compute the stiffness matrix and right-hand side.
    assemble_system();
    // Solve the linear system.
    solve_system();
    // Compute the error to the exact solution.
    compute_error();
  }

  ~PointEvalTest() {
    delete sol_;
    delete rhs_;
  }

  void evaluate_by_basis(BasisEvalLocal<double, DIMENSION> *basis_eval, 
                         const Coord& pts, 
                         std::vector<double>& values); 

private:
  std::string mesh_filename_;
  void build_initial_mesh();
  // Setup space, linear algebra, and compute Dirichlet values.
  void prepare_system();
  // Compute the matrix and rhs.
  void assemble_system();
  // Compute solution x.
  void solve_system();
  // Compute errors compared to exact solution.
  void compute_error();
  // MPI communicator.
  MPI_Comm comm_;
  // Local process rank and number of processes.
  int rank_, num_partitions_;

  // Local mesh and mesh on master process.
  MeshPtr mesh_, master_mesh_;
  // Solution space.
  VectorSpace< double, DIMENSION > space_;
  // Parameter data read in from file.
  PropertyTree params_;
  // Vectors for solution and load vector.
  CoupledVector< Scalar > *rhs_, *sol_;
  // System matrix.
  CoupledMatrix< Scalar > matrix_;

  // Global assembler.
  StandardGlobalAssembler< double, DIMENSION > global_asm_;
};

BOOST_AUTO_TEST_CASE(point_evaluator3D, *utf::tolerance(1.0e-10)) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;

  MPI_Init(&argc, &argv);
  // set default parameter file
  std::string param_filename(PARAM_FILENAME);

  LogKeeper::get_log ( "debug" ).set_target ( &( std::cout ) );
  LogKeeper::get_log ( "error" ).set_target ( &( std::cout ) );
  LogKeeper::get_log ( "info" ).set_target ( &( std::cout ) );
  
  PointEvalTest tester(param_filename);
  tester.run();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

void PointEvalTest::build_initial_mesh() {
  // Read in the mesh on the master process. The mesh is chosen according to the
  // dimension of the problem.
  if (rank_ == MASTER_RANK) {
    std::string mesh_name;
    switch (DIMENSION) {
    case (2): {
      mesh_name = params_["Mesh"]["Filename2"].get< std::string >(
          "unit_square-2_tri.inp");
    } break;
    case (3): {
      mesh_name = params_["Mesh"]["Filename3"].get< std::string >(
          "two_tetras_3d.inp");
    } break;
    }

    std::string mesh_filename;
    mesh_filename = std::string(DATADIR) + mesh_name;

    master_mesh_ = read_mesh_from_file(mesh_filename, DIMENSION, DIMENSION, 0);

    // Refine the mesh until the initial refinement level is reached.
    const int initial_ref_lvl = params_["Mesh"]["RefLevel"].get< int >(2);
    for (int r = 0; r < initial_ref_lvl; ++r) {
      master_mesh_ = master_mesh_->refine();
    }
  }

  // Distribute mesh over all processes, and compute ghost cells
  int uniform_ref_steps;
  MeshPtr local_mesh = partition_and_distribute(master_mesh_, MASTER_RANK,
                       comm_, uniform_ref_steps);
  SharedVertexTable shared_verts;
  mesh_ = compute_ghost_cells(*local_mesh, comm_, shared_verts);
}

void PointEvalTest::prepare_system() {
  // Assign degrees to each element.
  const int fe_degree = params_["Mesh"]["FeDegree"].get< int >(1);
  std::vector< int > degrees(1, fe_degree);
  std::vector< bool > is_cg(1, true);
  std::vector< FEType > fe_ansatz (1, FEType::BDM);
  
  // Initialize the VectorSpace object.
  space_.Init(*mesh_, fe_ansatz, is_cg, degrees, DOF_ORDERING::CUTHILL_MCKEE);
  // Compute the matrix graph.
  SparsityStructure sparsity;
  compute_sparsity_structure(space_, sparsity);

  // Setup linear algebra objects.
  delete rhs_;
  delete sol_;

  matrix_.Init(comm_, space_.la_couplings());

  CoupledVectorFactory< Scalar > CoupVecFact;
  rhs_ = CoupVecFact
         .Get(params_["LinearAlgebra"]["NameVector"].get< std::string >(
                "CoupledVector"))
         ->params(comm_, space_.la_couplings(), params_["LinearAlgebra"]);
  sol_ = CoupVecFact
             .Get(
               params_["LinearAlgebra"]["NameVector"].get< std::string >("CoupledVector"))
             ->params(comm_, space_.la_couplings(), params_["LinearAlgebra"]);

  // Initialize structure of LA objects.
  matrix_.InitStructure(sparsity);

  // Zero all linear algebra objects.
  matrix_.Zeros();
  rhs_->Zeros();
  sol_->Zeros();
}

void PointEvalTest::assemble_system() {
  // Assemble matrix and right-hand-side vector.
  LocalMassMatrixAssembler local_asm;
  global_asm_.assemble_matrix(space_, local_asm, matrix_);
  global_asm_.assemble_vector(space_, local_asm, *rhs_);
}

void PointEvalTest::solve_system() {
  LinearSolver< LAD > *solver_;
  LinearSolverFactory< LAD > SolFact;
  solver_ =
    SolFact.Get(params_["LinearSolver"]["Name"].get< std::string >("CG"))
    ->params(params_["LinearSolver"]);
  solver_->SetupOperator(matrix_);
  solver_->Solve(*rhs_, sol_);

  delete solver_;
}

void PointEvalTest::evaluate_by_basis(BasisEvalLocal<double, DIMENSION> *basis_eval, 
                                      const Coord& pt, 
                                      std::vector<double>& values) 
{
  assert (basis_eval != nullptr);
  
  // get values of all basis functions at given point
  std::vector<double> basis_vals;
  basis_eval->evaluate (pt, basis_vals); 
  
  // get global basis ids
  std::vector<int> ids;
  basis_eval->get_basis_ids(ids);
  BOOST_TEST(DIMENSION * ids.size() == basis_vals.size());
  
  // get values from solution vector
  std::vector<double> coeff(ids.size(), 0.);
  sol_->GetValues (&ids[0], ids.size(), &coeff[0]);
  BOOST_TEST(ids.size() == coeff.size());
    
  // compute fe value
  values.clear();
  values.resize(DIMENSION, 0.);
  
  for (size_t i=0; i<ids.size(); ++i)
  {
    for (size_t v=0; v<DIMENSION; ++v)
    {
      values[v] += coeff[i] * basis_vals[basis_eval->iv2ind(i,v)];
    }
  }
}

void PointEvalTest::compute_error() {
  sol_->Update();

  srand(time(NULL));
  int num_points = params_["Test"]["NumPts"].get< int >(1000);
  bool with_comm = true;//params_["Test"]["WithCommunication"].get< bool >(true);

  if (this->num_partitions_ == 1)
  {
    with_comm = false;
  }
  
  // measure the initialization time of the PointEvaluator
  double timer_init = MPI_Wtime();

  // set up the PointEvaluator
  double timer_eval = MPI_Wtime();
  timer_init = timer_eval - timer_init;

  FeEvalGlobal <double, DIMENSION> gl_evaluator (space_, *sol_, 0);
  FeEvalLocal  <double, DIMENSION> lc_evaluator (space_, *sol_, 0);
  FeEvalBasisLocal<double, DIMENSION> bs_evaluator (space_, *sol_, 0);
  GridGeometricSearch<double, DIMENSION> geom_search(mesh_);
  
  double max_diff = 0;
  std::vector< Coord > pts;
  for (int i = 0; i < num_points; ++i) 
  {
    // get some "random" test points.
    Coord pt;
    for (int j = 0; j < DIMENSION; ++j) 
    {
      pt.set(j, ((double)rand() / (RAND_MAX)));
    }
    pts.push_back(pt);
  }
  timer_eval = MPI_Wtime();

  // use FeEvaluator (Local/Global) without trial cells
  for (int i = 0; i < num_points; ++i) 
  {
    std::vector<double> values;
    bool has_point;
    if (with_comm) 
    {
      has_point = gl_evaluator.evaluate(pts[i], values);
    } 
    else 
    {
      has_point = lc_evaluator.evaluate(pts[i], values);
    }

    if (has_point) 
    {
      double diff = 0.;
      for (int v=0; v<DIMENSION; ++v)
      {
        diff += std::abs(exact_sol(pts[i])[v] - values[v]);
      }
      if (max_diff < diff) 
      {
        max_diff = diff;
      }
    }
  }

  timer_eval = MPI_Wtime() - timer_eval;

  // get the time needed for just finding the points for comparison
  double timer_finder = MPI_Wtime();
  std::vector< std::vector< int > > trial_cells;
  for (int i = 0; i < num_points; ++i) 
  {
    std::vector< Coord > ref_points;
    std::vector< int > cells;
    geom_search.find_cell(pts[i], cells, ref_points);
    trial_cells.push_back(cells);
  }
  timer_finder = MPI_Wtime() - timer_finder;

  // use FeEvaluator (Local/Global) with trial cells
  double timer_eval2 = MPI_Wtime();
  double max_diff2 = 0.;
  for (int i = 0; i < num_points; ++i) 
  {
    std::vector<double> values;

    bool has_point;
    gl_evaluator.set_trial_cells(trial_cells[i]);
    lc_evaluator.set_trial_cells(trial_cells[i]);
    if (with_comm) 
    {
      has_point = gl_evaluator.evaluate(pts[i], values);
    } 
    else 
    {
      has_point = lc_evaluator.evaluate(pts[i], values);
    }

    if (has_point) 
    {
      double diff = 0.;
      for (int v=0; v<DIMENSION; ++v)
      {
        diff += std::abs(exact_sol(pts[i])[v] - values[v]);
      }
      if (max_diff2 < diff) 
      {
        max_diff2 = diff;
      }
    }
  }
  timer_eval2 = MPI_Wtime() - timer_eval2;
  

  // use BasisEvaluator 
  double timer_init3 = MPI_Wtime();
  timer_init3 = MPI_Wtime() - timer_init3;
  
  double timer_eval3 = MPI_Wtime();
  double max_diff3 = 0.;
  for (int i = 0; i < num_points; ++i) 
  {
    std::vector<double> values;
    bs_evaluator.evaluate(pts[i], values);

    double diff = 0.;
    for (int v=0; v<DIMENSION; ++v)
    {
      diff += std::abs(exact_sol(pts[i])[v] - values[v]);
    }
    if (max_diff3 < diff) 
    {
      max_diff3 = diff;
    }
  }
  timer_eval3 = MPI_Wtime() - timer_eval3;
  
  

  std::cout << "Time for initializing GridGeometricSearch: " << timer_init << std::endl;
  std::cout << "Time for evaluating " << num_points << " points: " << timer_eval << std::endl;
  std::cout << "Time for finding " << num_points << " points: " << timer_finder << std::endl;
  std::cout << "Time for evaluating " << num_points << " points with known cells: " << timer_eval2 << std::endl;
  std::cout << "Time for initializing BasisEvalLocal " << timer_init3 << std::endl;
  std::cout << "Time for evaluating " << num_points << " points with BasisEvalLocal: " << timer_eval3 << std::endl;

  std::cout << "Mean Cells per GridCell: " << geom_search.mean_mesh_cells() << std::endl;
  std::cout << "Max Cells per GridCell: " << geom_search.max_mesh_cells() << std::endl;

  std::cout << "Max Diff (~L_inf) on process " << rank_ << ":  " << max_diff << std::endl;
  std::cout << "Max Diff2 (~L_inf) on process " << rank_ << ":  " << max_diff2 << std::endl;
  std::cout << "Max Diff3 (~L_inf) on process " << rank_ << ":  " << max_diff3 << std::endl;

  BOOST_TEST(max_diff == 0.0);
}
