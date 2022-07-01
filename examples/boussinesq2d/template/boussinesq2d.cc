#include "boussinesq2d.h"

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <vector>
#include <filesystem>
#include <boost/filesystem.hpp>
#include<stdio.h>
#include<stdlib.h>
#include <chrono>

namespace {
static const char *DATADIR = "./";
static const int MASTER_RANK = 0;
static const char *PARAM_FILENAME = "boussinesq2d.xml";
static const char *OUT_PATH = ".";
static bool CONSOLE_OUTPUT_ACTIVE = true;
static const int CONSOLE_THRESHOLD_LEVEL = 3;
} // namespace
#define CONSOLE_OUTPUT(lvl, x)                                                 \
  {                                                                            \
    if (CONSOLE_OUTPUT_ACTIVE && lvl <= CONSOLE_THRESHOLD_LEVEL) {             \
      for (int i = 0; i < lvl; ++i) {                                          \
        std::cout << "  ";                                                     \
      }                                                                        \
      std::cout << x << "\n";                                                  \
    }                                                                          \
  }

struct TimingData {
  double time_elapsed;
};

class TimingScope {
public:
  TimingScope(const std::string &name) {
    if (report_) {
      report_->begin_section(name);
    }
  }

  TimingScope(int iteration) {
    if (report_) {
      std::stringstream sstr;
      sstr << "Iteration " << iteration;
      report_->begin_section(sstr.str());
      timer_.reset();
      timer_.start();
    }
  }

  ~TimingScope() {
    timer_.stop();
    if (report_) {
      TimingData *data = report_->end_section();
      data->time_elapsed = timer_.get_duration();
    }
  }

  static void set_report(HierarchicalReport< TimingData > *report) {
    report_ = report;
  }

private:
  static HierarchicalReport< TimingData > *report_;
  Timer timer_;
};

HierarchicalReport< TimingData > *TimingScope::report_ = 0;

class TimingReportOutputVisitor {
public:
  TimingReportOutputVisitor(std::ostream &os) : os_(os), level_(0) {}

  void enter(const std::string &name, TimingData *data) {
    if (name == "root") {
      os_ << "+++ Timing Report +++\n\n";
    } else {
      for (int l = 0; l < level_; ++l) {
        os_ << "  ";
      }
      os_ << name << " took " << data->time_elapsed << " s.\n";
      ++level_;
    }
  }

  void exit(const std::string &name, TimingData *data) {
    if (name == "root") {
      os_ << "\n+++ End Timing Report +++\n\n";
    } else {
      --level_;
    }
  }

private:
  std::ostream &os_;
  int level_;
};

class Boussinesq : public NonlinearProblem< LAD > {
public:
  Boussinesq(const std::string &param_filename,
             const std::string &path_out, int MASTER_RANK, double results_points_distance2)
      : comm_(MPI_COMM_WORLD),
        path_out_(path_out),
        params_(param_filename.c_str(), MASTER_RANK, MPI_COMM_WORLD),
        use_pressure_filter_(false), refinement_level_(0), is_done_(false)

  {}

  virtual ~Boussinesq() {}

  virtual void run() {
    simul_name_ = params_["OutputPrefix"].get< std::string >();

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &num_partitions_);

    // Turn off INFO log except on master proc.
    if (rank_ != MASTER_RANK) {
      INFO = false;
      CONSOLE_OUTPUT_ACTIVE = false;
    }

    std::ofstream info_log((simul_name_ + "_info_log").c_str());
    LogKeeper::get_log("info").set_target(&(std::cout));
    std::ofstream debug_log((simul_name_ + "_debug_log").c_str());
    LogKeeper::get_log("debug").set_target(&(std::cout));
    std::ofstream error_log((simul_name_ + "_error_log").c_str());
    LogKeeper::get_log("error").set_target(&(std::cout));

    CONSOLE_OUTPUT(
        0, "============================================================");
    CONSOLE_OUTPUT(
        0, "====    Boussinesq                                       ===");
    CONSOLE_OUTPUT(
        0, "====    built using HiFlow3.                             ===");
    CONSOLE_OUTPUT(
        0, "====                                                     ===");
    CONSOLE_OUTPUT(
        0, "==== Engineering Mathematics and Computing Lab (EMCL)    ===");
    CONSOLE_OUTPUT(
        0, "============================================================");
    CONSOLE_OUTPUT(0, "");

    // output parameters for debugging
    LOG_INFO("parameters", params_);

    // setup timing report
    TimingScope::set_report(&time_report_);

    {
      TimingScope tscope("Setup");

      setup_linear_algebra();

      read_mesh();

      // The simulation has two modes: stationary and
      // instationary. Which one is used, depends on the parameter
      // Instationary.SolveInstationary. In stationary mode, solve()
      // is only called once, whereas in instationary mode, it is
      // called several times via the time-stepping method implemented in
      // run_time_loop() .
      solve_instationary_ =
          params_["Instationary"]["SolveInstationary"].get< bool >();

      prepare();
    }

    if (solve_instationary_) {
      LOG_INFO("simulation", "Solving instationary problem");
      run_time_loop();
    }/* else {
      LOG_INFO("simulation", "Solving stationary problem");
      if (use_hiflow_newton_) {

        Timer timer;

        newton_.Solve(&sol_);
        CONSOLE_OUTPUT(1, "Newton ended with residual norm "
                              << newton_.GetResidual() << " after "
                              << newton_.iter() << " iterations.");

        timer.stop();

        CONSOLE_OUTPUT(0, "");
        CONSOLE_OUTPUT(1, "Measured time of interest "
                              << (int)timer.get_duration() / 60 << "m"
                              << (timer.get_duration() -
                                  ((int)timer.get_duration() / 60) * 60)
                              << "s");
        CONSOLE_OUTPUT(0, "");
        CONSOLE_OUTPUT(1, "Measured time in seconds " << timer.get_duration());

      } else {
        solve();
      }

      visualize();
      
      if (compute_bench_quantities_) {
        compute_dfg_benchmark_quantities();
        output_norms();
      }
      
      write_results();
    }*/

    CONSOLE_OUTPUT(0, "");

/*
    if (rank_ == MASTER_RANK) {
      // Output time report
      TimingReportOutputVisitor visitor(std::cout);
      time_report_.traverse_depth_first(visitor);

      // Output results table
      std::vector< std::string > column_names;
      column_names.push_back("Time");
      column_names.push_back("|u|_L2");
      column_names.push_back("|p|_L2");
      column_names.push_back("|u|_H1");
      column_names.push_back("|p|_H1");
      column_names.push_back("Fd");
      column_names.push_back("Cd");
      column_names.push_back("Fl");
      column_names.push_back("Cl");
      column_names.push_back("delta-P");
      //  results_table_.print(std::cout, column_names);

      //  std::ofstream file((simul_name_ + "_results.csv").c_str());
      //  results_table_.print_csv(file, column_names);
    }
*/
    LogKeeper::get_log("info").flush();
    LogKeeper::get_log("debug").flush();
    LogKeeper::get_log("info").set_target(0);
    LogKeeper::get_log("debug").set_target(0);

    CONSOLE_OUTPUT(
        0, "============================================================");
        
  }  

private:
  const MPI_Comm &communicator() { return comm_; }

  int rank() { return rank_; }

  int num_partitions() { return num_partitions_; }

  PLATFORM la_platform() const { return la_sys_.Platform; }

  IMPLEMENTATION la_implementation() const { return la_impl_; }

  MATRIX_FORMAT la_matrix_format() const { return la_matrix_format_; }

  // Read, refine and partition mesh.
  void read_mesh();

  // Set up datastructures and read in some parameters.
  void prepare();

  // Set up boundary conditions
  void prepare_bc();

  // Solve nonlinear problem
  void solve();

  // Computate instationary solution by time-stepping method.
  void run_time_loop();

  // Compute exact solution for one constructed example, see docu of function
  // at end of file (where implemented)
  void eval_ic();

  // Visualize the solution in a file. In stationary mode, the
  // filename contains 'stationary', in instationary mode, it contains the
  // current time-step ts_.
  void visualize();

  // Compute L2 or H1 norm of variable defined through vars on the
  // master process.
  double compute_norm(int norm_type, const std::vector< int > &vars);

  // Output various norms of the solution.
  void output_norms();

  // Write some results. -- only norms outdated
  void write_results();

  // Write results csv init 
  void init_results_csv();
  void write_results_csv();

  virtual void Reinit();

  // Helper functions for nonlinear solver
  void ApplyFilter(LAD::VectorType &u);
  virtual void EvalFunc(const LAD::VectorType &in, LAD::VectorType *out);
  void compute_residual(const LAD::VectorType &in,
                        LAD::VectorType *out); // updates res_ with the residual
  void compute_instationary_residual(
      const LAD::VectorType &in,
      LAD::VectorType *out); // residual computation in instationary mode

  virtual void EvalGrad(const LAD::VectorType &in, LAD::MatrixType *out);
  void compute_jacobian(
      const LAD::VectorType &in,
      LAD::MatrixType *out); // updates matrix_ with the jacobian matrix
  void compute_instationary_matrix(
      const LAD::VectorType &in,
      LAD::MatrixType *out); // jacobi matrix computation in instationary mode

  // Pressure filter: substracts the mean of the pressure from each
  // pressure dof in sol_ .
  void filter_pressure();

  // Linear algebra set up
  void setup_linear_algebra();

  // compute L2-Error and H1semi-Error
  void compute_errors();

  // compute difference between solution last and penultmate timestep
  void compute_difference();

  // MPI stuff
  MPI_Comm comm_;
  int rank_, num_partitions_;

  // Linear algebra stuff
  SYSTEM la_sys_;
  IMPLEMENTATION la_impl_;
  MATRIX_FORMAT la_matrix_format_;

  // Parameter data read in from file.
  PropertyTree params_;
  std::string path_out_;

  std::string simul_name_; // parameter 'OutputPrefix': prefix for output files

  // Time-stepping variables
  int ts_;
  double dt_;
  double alpha1_, alpha2_, alpha3_;

  // Flow model variables
  bool dimensionless_;
  double rho_, Pr_, Ra_, hT_, cT_;

  // Flag for pressure filter -- parameter 'UsePressureFilter'
  bool use_pressure_filter_;

  // Meshes
  MeshPtr mesh_;
  int refinement_level_;

  VectorSpace< double, DIM > space_;

  // linear algebra objects
  CMatrix matrix_;
  CVector sol_, prev_sol_, cor_, res_, pressure_correction_, initialCond_, error_;

  // linear solver parameters
  int lin_max_iter;
  double lin_abs_tol;
  double lin_rel_tol;
  double lin_div_tol;
  int basis_size;

  // linear solver
  GMRES< LAD > gmres_;

  // nonlinear solver parameters
  int nls_max_iter;
  double nls_abs_tol;
  double nls_rel_tol;
  double nls_div_tol;
  double eta_; // forcing term
  std::vector< double > residual_history_, forcing_term_history_;
  bool do_armijo_update_;
  std::string forcing_strategy_;
  bool use_forcing_strategy_;

  // damping strategy paramters
  double theta_initial;
  double theta_min;
  double armijo_dec;
  double suff_dec;
  int max_armijo_ite;

  // forcing strategy parameters
  double eta_initial;
  double eta_max;
  double gamma_EW2;
  double alpha_EW2;

  // nonlinear solver
  Newton< LAD, DIM > newton_;

  StandardGlobalAssembler< double, DIM > global_asm_;

  bool is_done_, solve_instationary_, convergence_test_;

  std::vector< int > dirichlet_dofs_;
  std::vector< double > dirichlet_values_;

#ifdef WITH_ILUPP
  PreconditionerIlupp< LAD > ilupp_;
  bool use_ilupp_;
#endif

  bool is_dfg_benchmark_;
  bool compute_bench_quantities_;
  bool use_hiflow_newton_;

  HierarchicalReport< TimingData > time_report_;
  Table results_table_;

  // CSV output
  CSVWriter< double > bench_quantity_writer_;
  std::vector< std::string > bench_names_;
  std::vector< double > bench_quantities_;  
  CSVWriter< double > output_csv_writer_;
  std::vector< std::string > output_names_;
  std::vector< double > output_quantities_;
  std::vector< Coord > results_points;
  double results_points_distance2 = 0.25;
};

// program entry point

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  std::string param_filename(PARAM_FILENAME);
  if (argc > 1) {
    param_filename = std::string(argv[1]);
  }

  std::string path_out(OUT_PATH);
  if (argc > 2) {
    path_out = std::string(argv[2]);
  }

  double results_points_distance2 = 0.3;
  if (argc > 3) {
    results_points_distance2 = strtod(argv[3], NULL);
  }

  std::cout << "distance : " << results_points_distance2 << std::endl;

  try {
    Boussinesq app(param_filename, path_out, 0, results_points_distance2);
    app.run();
  } catch (std::exception &e) {
    std::cerr << "\nProgram ended with uncaught exception.\n";
    std::cerr << e.what() << "\n";
    return -1;
  }
  MPI_Finalize();

  return 0;
}

void Boussinesq::read_mesh() {
  TimingScope tscope("read_mesh");

  MeshPtr master_mesh;

  refinement_level_ = 0;
  const int initial_ref_lvl = params_["Mesh"]["InitialRefLevel"].get< int >();

  const bool use_bdd =
      params_["UseBoundaryDomainDescriptor"].get< bool >(false);

#ifdef WITH_PARMETIS

  if (rank() == MASTER_RANK) {
#if DIM == 2
    const std::string mesh_name =
        params_["Mesh"]["Filename1"].get< std::string >();
#else
    const std::string mesh_name =
        params_["Mesh"]["Filename2"].get< std::string >();
#endif
    std::string mesh_filename = std::string(DATADIR) + mesh_name;

    master_mesh = read_mesh_from_file(mesh_filename, DIM, DIM, 0);

    CONSOLE_OUTPUT(1, "Read mesh with " << master_mesh->num_entities(DIM)
                                        << " cells.");

    while (refinement_level_ < initial_ref_lvl &&
           master_mesh->num_entities(DIM) < 8 * this->num_partitions_) {
      master_mesh = master_mesh->refine();
#if DIM == 3
      if (use_bdd)
        adapt_boundary_to_function(master_mesh, cyl);
#endif
      ++refinement_level_;
    }
    LOG_INFO("mesh", "Initial refinement level = " << refinement_level_);
    CONSOLE_OUTPUT(1, "Refined mesh (level "
                          << refinement_level_ << ") has "
                          << master_mesh->num_entities(DIM) << " cells.");
  }

  MPI_Bcast(&refinement_level_, 1, MPI_INT, MASTER_RANK, comm_);
  // Distribute mesh using METIS
  ParMetisGraphPartitioner metis;

  MeshPtr local_mesh;
  int uniform_ref_steps;
  if (num_partitions_ <= 1) {
    NaiveGraphPartitioner partitioner;
    std::cerr << "METIS not used, because number of partitions is "
              << num_partitions_ << ".\n";
    local_mesh = partition_and_distribute(master_mesh, MASTER_RANK, comm_,
                                          &partitioner, uniform_ref_steps);
    if (rank_ == MASTER_RANK) {
      LOG_INFO("Partitioner", "NAIVE");
    }
  } else {
    local_mesh = partition_and_distribute(master_mesh, MASTER_RANK, comm_,
                                          &metis, uniform_ref_steps);
    if (rank_ == MASTER_RANK) {
      LOG_INFO("Partitioner", "METIS");
    }
  }
  assert(local_mesh != 0);

  master_mesh.reset();

  if (rank_ == MASTER_RANK) {
    LOG_INFO("Mesh", "Partitioned and distributed");
  }

  while (refinement_level_ < initial_ref_lvl) {
    local_mesh = local_mesh->refine();
    ++refinement_level_;
  }

  assert(local_mesh != 0);

  if (rank_ == MASTER_RANK) {
    LOG_INFO("Mesh refinement level", refinement_level_);
  }

  // Compute ghost cells.
  SharedVertexTable shared_verts;

  MeshPtr repartitioned_mesh;
  ParMetisGraphPartitioner parmetis_partitioner;
  repartitioned_mesh =
      repartition_mesh(local_mesh, comm_, &parmetis_partitioner);
  if (rank_ == MASTER_RANK) {
    LOG_INFO("Repartitioning of mesh", "done");
  }
  mesh_ = compute_ghost_cells(*repartitioned_mesh, comm_, shared_verts);

    // Write out mesh of initial refinement level
    PVtkWriter writer(comm_);
    std::ostringstream name;
    name << this->path_out_ << "/boussinesq2d_mesh_" << refinement_level_ << ".pvtu";
    std::string output_file = name.str();
    writer.add_all_attributes(*mesh_, true);
    writer.write(output_file.c_str(), *mesh_);

  local_mesh.reset();

  if (rank_ == MASTER_RANK) {
    LOG_INFO("Ghost cell computation", "done");
  }

#else
  if (rank() == MASTER_RANK) {
#if DIM == 2
    const std::string mesh_name =
        params_["Mesh"]["Filename1"].get< std::string >();
#else
    const std::string mesh_name =
        params_["Mesh"]["Filename2"].get< std::string >();
#endif
    std::string mesh_filename = std::string(DATADIR) + mesh_name;

    master_mesh = read_mesh_from_file(mesh_filename, DIM, DIM, 0);

    CONSOLE_OUTPUT(1, "Read mesh with " << master_mesh->num_entities(DIM)
                                        << " cells.");

    for (int r = 0; r < initial_ref_lvl; ++r) {
      master_mesh = master_mesh->refine();
#if DIM == 3
      if (use_bdd)
        adapt_boundary_to_function(master_mesh, cyl);
#endif
      ++refinement_level_;
    }
    LOG_INFO("mesh", "Initial refinement level = " << refinement_level_);
    CONSOLE_OUTPUT(1, "Refined mesh (level "
                          << refinement_level_ << ") has "
                          << master_mesh->num_entities(DIM) << " cells.");
  }

  int num_ref_seq_steps;
  MeshPtr local_mesh = partition_and_distribute(master_mesh, MASTER_RANK, comm_,
                                                num_ref_seq_steps);
  assert(local_mesh != 0);
  SharedVertexTable shared_verts;
  mesh_ = compute_ghost_cells(*local_mesh, comm_, shared_verts);
#endif
}

void Boussinesq::prepare() {
  TimingScope tscope("prepare");

  // prepare timestep
  ts_ = 0;
  dt_ = params_["Instationary"]["Timestep"].get< double >();

  // set the alpha coefficients correctly for the
  // Crank-Nicolson method.
  alpha1_ = 0.5 * dt_;
  alpha2_ = dt_;
  alpha3_ = 0.5 * dt_;

#ifdef WITH_ILUPP
  // prepare preconditioner
  use_ilupp_ = params_["LinearSolver"]["Preconditioning"].get< bool >();
  if (use_ilupp_) {
    ilupp_.InitParameter(params_["ILUPP"]["PreprocessingType"].get< int >(),
                         params_["ILUPP"]["PreconditionerNumber"].get< int >(),
                         params_["ILUPP"]["MaxMultilevels"].get< int >(),
                         params_["ILUPP"]["MemFactor"].get< double >(),
                         params_["ILUPP"]["PivotThreshold"].get< double >(),
                         params_["ILUPP"]["MinPivot"].get< double >());
  }
#endif

  // prepare problem parameters
  dimensionless_ = params_["FlowModel"]["Dimensionless"].get< bool >();
  rho_ = params_["FlowModel"]["Density"].get< double >();
  hT_ = params_["FlowModel"]["HotTemperature"].get< double >();
  cT_ = params_["FlowModel"]["ColdTemperature"].get< double >();
  Pr_ = params_["FlowModel"]["PrandtlNumber"].get< double >();
  Ra_ = params_["FlowModel"]["RayleighNumber"].get< double >();
  
  // prepare space
  std::vector< int > degrees(DIM + 2);

  const int u_deg = params_["FiniteElements"]["VelocityDegree"].get< int >(); 
  const int p_deg = params_["FiniteElements"]["PressureDegree"].get< int >(); // WARUM hab ich das auskommentiert?
  const int t_deg = params_["FiniteElements"]["TemperatureDegree"].get< int >();
  
  for (int c = 0; c < DIM; ++c) {
    degrees.at(c) = u_deg;
  }

  degrees.at(DIM) = p_deg;
  degrees.at(DIM + 1) = t_deg;

  std::vector< bool > is_cg(DIM + 2, true);
  std::vector< FEType > fe_ansatz (DIM+2, FEType::LAGRANGE);

  space_.Init(*mesh_, fe_ansatz, is_cg, degrees, DOF_ORDERING::HIFLOW_CLASSIC); 

  CONSOLE_OUTPUT(1, "Total number of dofs = " << space_.dof().nb_dofs_global());

  for (int p = 0; p < num_partitions(); ++p) {
    CONSOLE_OUTPUT(2, "Num dofs on process " << p << " = "
                                             << space_.dof().nb_dofs_on_subdom(p));
  }

  // pressure filter
  use_pressure_filter_ = params_["UsePressureFilter"].get< bool >();

  // prepare global assembler
  QuadratureSelection q_sel(params_["QuadratureOrder"].get< int >());
  global_asm_.set_quadrature_selection_function(q_sel);


  // compute matrix graph

  std::vector< std::vector< bool > > coupling_vars;

  coupling_vars.resize(DIM + 2);
  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM + 2; ++j) {
      coupling_vars[i].push_back(true);
    }
  }
  for (int i = 0; i < DIM; ++i) {
    coupling_vars[DIM].push_back(true);
    coupling_vars[DIM+1].push_back(true);
  }
  coupling_vars[DIM].push_back(false);
  coupling_vars[DIM].push_back(false);
  coupling_vars[DIM+1].push_back(false);
  coupling_vars[DIM+1].push_back(true);


  
  SparsityStructure sparsity;
  compute_sparsity_structure(space_, sparsity, coupling_vars, false);

  matrix_.Init(communicator(), space_.la_couplings(), la_platform(), la_implementation(), la_matrix_format());
  matrix_.InitStructure(sparsity);
  matrix_.Zeros();

  sol_.Init(communicator(), space_.la_couplings(), la_platform(), la_implementation());
  prev_sol_.Init(communicator(), space_.la_couplings(), la_platform(), la_implementation());
  cor_.Init(communicator(), space_.la_couplings(), la_platform(), la_implementation());
  res_.Init(communicator(), space_.la_couplings(), la_platform(), la_implementation());

  sol_.Zeros();
#if EXACTSOL == 1
  // debugging example with constructed right hand side
  eval_ic();
#endif

  prev_sol_.Zeros();
  cor_.Zeros();
  res_.Zeros();

  // setup linear solver
  lin_max_iter = params_["LinearSolver"]["MaximumIterations"].get< int >();
  lin_abs_tol = params_["LinearSolver"]["AbsoluteTolerance"].get< double >();
  lin_rel_tol = params_["LinearSolver"]["RelativeTolerance"].get< double >();
  lin_div_tol = params_["LinearSolver"]["DivergenceLimit"].get< double >();
  basis_size = params_["LinearSolver"]["BasisSize"].get< int >();

#ifdef WITH_ILUPP
  if (use_ilupp_) {
    gmres_.SetupPreconditioner(ilupp_);
    gmres_.InitParameter(basis_size, "RightPreconditioning");
  } else {
    gmres_.InitParameter(basis_size, "NoPreconditioning");
  }
#else
  gmres_.InitParameter(basis_size, "NoPreconditioning");
#endif

  gmres_.InitControl(lin_max_iter, lin_abs_tol, lin_rel_tol, lin_div_tol);
  gmres_.SetupOperator(matrix_);

  // get nonlinear solver parameters from param file
  nls_max_iter = params_["NonlinearSolver"]["MaximumIterations"].get< int >();
  nls_abs_tol = params_["NonlinearSolver"]["AbsoluteTolerance"].get< double >();
  nls_rel_tol = params_["NonlinearSolver"]["RelativeTolerance"].get< double >();
  nls_div_tol = params_["NonlinearSolver"]["DivergenceLimit"].get< double >();
  do_armijo_update_ = params_["NonlinearSolver"]["ArmijoUpdate"].get< bool >();
  forcing_strategy_ =
      params_["NonlinearSolver"]["ForcingStrategy"].get< std::string >();
  use_forcing_strategy_ = (forcing_strategy_ != "None");
  eta_ = 1.e-4; // initial value of forcing term

  // get damping strategy parameters from param file
  theta_initial = params_["NonlinearSolver"]["ThetaInitial"].get< double >();
  theta_min = params_["NonlinearSolver"]["ThetaMinimal"].get< double >();
  armijo_dec = params_["NonlinearSolver"]["ArmijoDecrease"].get< double >();
  suff_dec = params_["NonlinearSolver"]["SufficientDecrease"].get< double >();
  max_armijo_ite =
      params_["NonlinearSolver"]["MaxArmijoIteration"].get< int >();

  // get forcing strategy parameters from param file
  eta_initial =
      params_["NonlinearSolver"]["InitialValueForcingTerm"].get< double >();
  eta_max = params_["NonlinearSolver"]["MaxValueForcingTerm"].get< double >();
  gamma_EW2 = params_["NonlinearSolver"]["GammaParameterEW2"].get< double >();
  alpha_EW2 = params_["NonlinearSolver"]["AlphaParameterEW2"].get< double >();

  // setup nonlinear solver
  newton_.InitParameter(&res_, &matrix_);
  newton_.InitParameter(Newton< LAD, DIM >::NewtonInitialSolutionOwn);
  newton_.InitControl(nls_max_iter, nls_abs_tol, nls_rel_tol, nls_div_tol);
  newton_.SetOperator(*this);
  newton_.SetLinearSolver(gmres_);

  // Damping strategy object
  if (do_armijo_update_) {
    ArmijoDamping< LAD, DIM > *Armijo_Damping = new ArmijoDamping< LAD, DIM >(
        theta_initial, theta_min, armijo_dec, suff_dec, max_armijo_ite);
    newton_.SetDampingStrategy(*Armijo_Damping);
  }

  // Forcing strategy object
  if (forcing_strategy_ == "EisenstatWalker1") {
    EWForcing< LAD > *EW_Forcing =
        new EWForcing< LAD >(eta_initial, eta_max, 1);
    newton_.SetForcingStrategy(*EW_Forcing);
  } else if (forcing_strategy_ == "EisenstatWalker2") {
    EWForcing< LAD > *EW_Forcing =
        new EWForcing< LAD >(eta_initial, eta_max, 2, gamma_EW2, alpha_EW2);
    newton_.SetForcingStrategy(*EW_Forcing);
  }

  // prepare dirichlet BC
  prepare_bc();
}

void Boussinesq::prepare_bc() {
  TimingScope tscope("prepare_bc");

  dirichlet_dofs_.clear();
  dirichlet_values_.clear();

  const int top_bdy = params_["Boundary"]["TopBdy"].get< int >();
  const int bottom_bdy = params_["Boundary"]["BottomBdy"].get< int >();
  const int left_bdy = params_["Boundary"]["LeftBdy"].get< int >();
  const int right_bdy = params_["Boundary"]["RightBdy"].get< int >();
/*
#if DIM == 3
#if EXACTSOL == 1
  // debugging example with constructed right hand side
  ExactSolChannelFlowBC3d bc[3] = {ExactSolChannelFlowBC3d(0),
                                   ExactSolChannelFlowBC3d(1),
                                   ExactSolChannelFlowBC3d(2)};
#else
  ChannelFlowBC3d bc[3] = {
      ChannelFlowBC3d(0, W_, H_, Um_, inflow_bdy, outflow_bdy),
      ChannelFlowBC3d(1, W_, H_, Um_, inflow_bdy, outflow_bdy),
      ChannelFlowBC3d(2, W_, H_, Um_, inflow_bdy, outflow_bdy)};
#endif
  for (int var = 0; var < DIM; ++var) {
    compute_dirichlet_dofs_and_values(bc[var], space_, var, dirichlet_dofs_,
                                      dirichlet_values_);
  }
#else
*/
  BoussinesqBC2d bc[3] = {
      BoussinesqBC2d(0, hT_, cT_, left_bdy, right_bdy, top_bdy, bottom_bdy),
      BoussinesqBC2d(1, hT_, cT_, left_bdy, right_bdy, top_bdy, bottom_bdy),
      BoussinesqBC2d(DIM+1, hT_, cT_, left_bdy, right_bdy, top_bdy, bottom_bdy),
      };


  compute_dirichlet_dofs_and_values(bc[0], space_, 0, dirichlet_dofs_, dirichlet_values_);
  compute_dirichlet_dofs_and_values(bc[1], space_, 1, dirichlet_dofs_, dirichlet_values_);
  compute_dirichlet_dofs_and_values(bc[2], space_, DIM+1, dirichlet_dofs_, dirichlet_values_);
  
/*
#endif
*/
}

void Boussinesq::run_time_loop() {
  TimingScope tscope("Timestepping loop");

#ifdef WITH_HDF5
  if (params_["Backup"]["Restore"].get< bool >()) {
    ts_ = params_["Backup"]["LastTimeStep"].get< int >();
    std::stringstream filename;
    filename << this->num_partitions_ << "_"
             << params_["Backup"]["Filename"].get< std::string >();
    const std::string backup_name = filename.str();
    std::ostringstream vec_name;
    vec_name << "sol_" << ts_;
    prev_sol_.ReadHDF5(backup_name, "backup", vec_name.str());
    prev_sol_.Update();
    CONSOLE_OUTPUT(1, "Restarting from backup in file "
                          << backup_name << " after timestep " << ts_);
  }
  else{
    this -> eval_ic();
  }
  #else
    this -> eval_ic();
#endif

  // apply BC to initial solution
  if (!dirichlet_dofs_.empty()) 
  {
    // correct solution with dirichlet BC
    sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
    prev_sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
  }

  sol_.Update();
  prev_sol_.Update();
  
  // Visualize initial solution.
  if (ts_ == 0) {
    if (rank() == MASTER_RANK) {
      results_table_.insert("Time", 0.);

      results_table_.insert("Fd", 0.);
      results_table_.insert("Cd", 0.);
      results_table_.insert("Fl", 0.);
      results_table_.insert("Cl", 0.);
      results_table_.insert("delta-P", 0.);
    }

    visualize();

    if (compute_bench_quantities_) {
      output_norms();
    }

#ifdef WITH_HDF5
    std::stringstream filename;
    filename << this->num_partitions_ << "_"
             << params_["Backup"]["Filename"].get< std::string >();
    const std::string backup_name = filename.str();
    std::ostringstream vec_name;
    vec_name << "sol_" << ts_;
    prev_sol_.WriteHDF5(backup_name, "backup", vec_name.str());
#endif
  }

  const double end_time = params_["Instationary"]["Endtime"].get< double >();
  LOG_INFO("timestep", "End time = " << end_time);
  LOG_INFO("timestep", "Step length = " << dt_);

  // Set up CSV output of benchmarking quantities
  bench_names_.push_back("Timestep");
  bench_names_.push_back("Iterations (Newton)");
  bench_names_.push_back("Time to compute time-step [s]");
  bench_names_.push_back("Time to compute residuals [s]");
  bench_names_.push_back(
      "Time to compute Jacobians and setup preconditioner [s]");
  bench_names_.push_back("(Total) Number of GMRES iterations per time-step");

  bench_quantities_.resize(bench_names_.size(), 0.);


  // Initialize CSV Writer for results.out
  //init_results_csv();

  // Initialize CSV Writer
  std::stringstream bench_file;
  bench_file << num_partitions_ << "_" << dt_ << "_"
             << "benchmarking_quantities.csv";

  if (this->rank_ == MASTER_RANK) {
    LOG_INFO("Benchmarking quantities file", bench_file.str());
  }
  bench_quantity_writer_.InitFilename(bench_file.str());

  if (params_["Backup"]["Restore"].get< bool >() && ts_ != 0) {
    std::vector< std::vector< double > > stored_bench;
    std::vector< std::vector< double > > preserve_bench;
    bench_quantity_writer_.read(stored_bench);

    for (int i = 0; i < static_cast< int >(stored_bench.size()); ++i) {
      if (stored_bench[i][0] <= ts_) {
        preserve_bench.push_back(stored_bench[i]);
      }
    }

    // keep only bench quantities of former timesteps
    if (rank_ == MASTER_RANK) {
      bench_quantity_writer_.Init(bench_names_);

      for (int i = 0; i < static_cast< int >(preserve_bench.size()); ++i) {
        bench_quantity_writer_.write(preserve_bench[i]);
      }
    }
  } else {
    if (this->rank_ == MASTER_RANK) {
      bench_quantity_writer_.Init(bench_names_);
    }
  }

  // Set first timestep to compute.

  CONSOLE_OUTPUT(1, "Starting time loop from t = " << ts_ * dt_ << " to "
                                                   << end_time
                                                   << " with timestep " << dt_);
  ++ts_;

  // Crank-Nicolson time-stepping method. At the beginning of each
  // time-step, the solution from the previous time-step is stored
  // in prev_sol_, which is used in InstationaryFlowAssembler. The
  // variable ts_ is used to keep track of the current
  // time-step. The solution is visualized at the end of each
  // time-step, in order to be able to animate it in Paraview.
  while (ts_ * dt_ <= end_time) {
    TimingScope tscope(ts_);
    CONSOLE_OUTPUT(1, "Solving timestep " << ts_ << " (for t = " << ts_ * dt_
                                          << ")");
    LOG_INFO("timestep", "Solving time step " << ts_);

    bench_quantities_[1] = bench_quantities_[2] = bench_quantities_[3] =
        bench_quantities_[4] = bench_quantities_[5] = 0.;

    // check benchmarking quantities
    bench_quantities_[0] = ts_;

    Timer time_step_timer;
    time_step_timer.start();

    if (!dirichlet_dofs_.empty()) 
    {
      // correct solution with dirichlet BC
      sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
    }
    sol_.Update();
  
    newton_.Solve(&sol_);
    CONSOLE_OUTPUT(1, "Newton ended with residual norm "
                            << newton_.GetResidual() << " after "
                            << newton_.iter() << " iterations.");
      bench_quantities_[1] = newton_.iter();
      bench_quantities_[5] += gmres_.iter();

    time_step_timer.stop();
    bench_quantities_[2] = time_step_timer.get_duration();

    if (this->rank_ == MASTER_RANK) {
      bench_quantity_writer_.write(bench_quantities_);
    }

    prev_sol_.CloneFrom(sol_);
    // this also clones the ghost DoFs, so we don't need to call
    // Update()

#ifdef WITH_HDF5
    std::stringstream filename;
    filename << this->num_partitions_ << "_"
             << params_["Backup"]["Filename"].get< std::string >();
    const std::string backup_name = filename.str();
    std::ostringstream vec_name;
    vec_name << "sol_" << ts_;
    prev_sol_.WriteHDF5(backup_name, "backup", vec_name.str());
#endif

    results_table_.insert("Time", ts_ * dt_);

    LOG_INFO("timestep", "Visualizing solution at time "
                             << ts_ * dt_ << " (time-step " << ts_ << ")");
    visualize();
    //write_results_csv();
    if (compute_bench_quantities_) {
      // Tassia Achtung
      // compute_dfg_benchmark_quantities();
      output_norms();
    }

    ++ts_;
  }

    // Write some results.
  //write_results();
    
}

void Boussinesq::visualize() {
  TimingScope tscope("Visualization");

  // Setup visualization object.
  int num_intervals = 1;
  CellVisualization< Scalar, DIM > visu(space_, num_intervals);

  std::stringstream input;

  input << simul_name_ << "_solution";

  if (solve_instationary_) {
    if (ts_ < 10)
      input << "000" << ts_;
    else if (ts_ < 100)
      input << "00" << ts_;
    else if (ts_ < 1000)
      input << "0" << ts_;
    else
      input << "" << ts_;
  } else {
    input << "_stationary";
  }

  // Generate filename.
  std::stringstream name;
  //name << this->path_out_ << "/solution" << refinement_level_;
  name << this->path_out_ << "/" << input.str();// << refinement_level_;

  std::cout << "                       outpath: " << name.str() << std::endl;

  std::vector< double > remote_index(mesh_->num_entities(mesh_->tdim()), 0);
  std::vector< double > sub_domain(mesh_->num_entities(mesh_->tdim()), 0);
  std::vector< double > material_number(mesh_->num_entities(mesh_->tdim()), 0);

  for (mesh::EntityIterator it = mesh_->begin(mesh_->tdim());
       it != mesh_->end(mesh_->tdim()); ++it) {
    int temp1, temp2;
    mesh_->get_attribute_value("_remote_index_", mesh_->tdim(), it->index(),
                               &temp1);
    mesh_->get_attribute_value("_sub_domain_", mesh_->tdim(), it->index(),
                               &temp2);
    material_number.at(it->index()) =
        mesh_->get_material_number(mesh_->tdim(), it->index());
    remote_index.at(it->index()) = temp1;
    sub_domain.at(it->index()) = temp2;
  }

  sol_.Update();

  visu.visualize(FeEvalCell< double, DIM >(space_, sol_, 0), "u");
  visu.visualize(FeEvalCell< double, DIM >(space_, sol_, 1), "v");
#if DIM == 2
  visu.visualize(FeEvalCell< double, DIM >(space_, sol_, 2), "p");
  visu.visualize(FeEvalCell< double, DIM >(space_, sol_, 3), "t");
#elif DIM == 3
  visu.visualize(FeEvalCell< double, DIM >(space_, sol_, 2), "w");
  visu.visualize(FeEvalCell< double, DIM >(space_, sol_, 3), "p");
  visu.visualize(FeEvalCell< double, DIM >(space_, sol_, 4), "t");
#endif

  visu.visualize_cell_data(material_number, "Material Id");
  visu.visualize_cell_data(remote_index, "_remote_index_");
  visu.visualize_cell_data(sub_domain, "_sub_domain_");

  VTKWriter< Scalar, DIM> vtk_writer (visu, this->comm_, MASTER_RANK);
  vtk_writer.write(name.str());
}

double Boussinesq::compute_norm(int norm_type,
                                      const std::vector< int > &vars) {
  double local_norm = -1.e30, global_norm = 0.;
  sol_.Update();
  switch (norm_type) {
  case 0: // L2-norm
  {
    L2NormIntegratorPp L2_int(sol_, vars);
    global_asm_.integrate_scalar(space_, L2_int, local_norm);
    break;
  }
  case 1: // H1-seminorm
  {
    H1semiNormIntegratorPp H1_int(sol_, vars);
    global_asm_.integrate_scalar(space_, H1_int, local_norm);
    break;
  }
  default:
    std::cerr << "unknown type of norm!\n";
    assert(false);
  };

  // NB: global value will only be returned on master proc -- others will return
  // 0.

  MPI_Reduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MASTER_RANK,
             comm_);

  return std::sqrt(global_norm);
}

void Boussinesq::output_norms() {
  TimingScope tscope("Norm computation");
  std::vector< int > vel_vars, p_var;
  vel_vars.push_back(0);
  vel_vars.push_back(1);
  vel_vars.push_back(2);
  p_var.push_back(3);

  const double L2_vel_norm = compute_norm(0, vel_vars);
  const double L2_p_norm = compute_norm(0, p_var);
  const double H1_vel_norm = compute_norm(1, vel_vars);
  const double H1_p_norm = compute_norm(1, p_var);

  if (rank_ == MASTER_RANK) {
    LOG_INFO("L2-norm of velocity", L2_vel_norm);
    LOG_INFO("L2-norm of pressure", L2_p_norm);
    LOG_INFO("H1-seminorm of velocity", H1_vel_norm);
    LOG_INFO("H1-seminorm of pressure", H1_p_norm);

    results_table_.insert("|u|_L2", L2_vel_norm);
    results_table_.insert("|p|_L2", L2_p_norm);
    results_table_.insert("|u|_H1", H1_vel_norm);
    results_table_.insert("|p|_H1", H1_p_norm);
  }
}

void Boussinesq::Reinit() {
  //  prev_sol_.CloneFrom(sol_);
}

void Boussinesq::EvalFunc(const LAD::VectorType &in,
                                LAD::VectorType *out) {
  Timer assembly_timer;
  assembly_timer.start();
  compute_residual(in, out);
  //    out->Scale(-1.0);
  assembly_timer.stop();

  bench_quantities_[3] += assembly_timer.get_duration();
}

void Boussinesq::compute_residual(const LAD::VectorType &in,
                                        LAD::VectorType *out) {
  TimingScope tscope("Compute Residual");

  // the evaluation of the residual needs updated ghost DoFs,
  // so make sure you call Update() on the input vector
  // before you call this function!

  if (solve_instationary_) {
    compute_instationary_residual(in, out);
  } 

  // correct BC -- set Dirichlet dofs to 0
  std::cout << "dir dofs size " << dirichlet_dofs_.size() << std::endl;
   
  if (!dirichlet_dofs_.empty()) 
  {
    std::vector< LAD::DataType > zeros(dirichlet_dofs_.size(), 0.);
    out->SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),vec2ptr(zeros));
  }
}


/*
  InstationaryFlowAssembler local_asm(nu_, rho_); --> Boussinesq2d(nu_, rho_, Pr_, Ra_)

  :
*/
void Boussinesq::compute_instationary_residual(const LAD::VectorType &in,
                                                     LAD::VectorType *out) {
  BoussinesqAssembler local_asm(rho_, Pr_, Ra_);
  // InstationaryFlowAssembler local_asm(nu_, rho_);
  local_asm.set_newton_solution(&in);
  // the computation of the instationary residual also needs updated
  // ghost DoFs of the previous solution, so make sure you call
  // Update() on prev_sol_ before calling this function
  local_asm.set_time_solution(&prev_sol_);
  local_asm.set_time_stepping_weights(alpha1_, alpha2_, alpha3_);

  global_asm_.assemble_vector(space_, local_asm, *out);
}

void Boussinesq::EvalGrad(const LAD::VectorType &in,
                                LAD::MatrixType *out) {
  bench_quantities_[5] += gmres_.iter();

  Timer assembly_timer;
  assembly_timer.start();

  compute_jacobian(in, out);

  assembly_timer.stop();
  bench_quantities_[4] += assembly_timer.get_duration();
}

void Boussinesq::compute_jacobian(const LAD::VectorType &in,
                                        LAD::MatrixType *out) {
  {
    TimingScope tscope("Compute Jacobian");

    // the computation of the Jacobian needs updated ghost DoFs,
    // so make sure you call Update() on the input vector
    // before calling this function!

    if (solve_instationary_) {
      compute_instationary_matrix(in, out);
    }
    /* // STATIONARY! 
    else {
      compute_stationary_matrix(in, out);
    }
    */

    // correct BC -- set Dirichlet rows to identity
    if (!dirichlet_dofs_.empty()) {
      out->diagonalize_rows(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), 1.);
    }
  }

#ifdef WITH_ILUPP
  {
    TimingScope tscope("ILU++ Factorization");
    if (use_ilupp_) {
      ilupp_.SetupOperator(*out);
    }
  }
#endif
}

void Boussinesq::compute_instationary_matrix(const LAD::VectorType &in,
                                                   LAD::MatrixType *out) {
  BoussinesqAssembler local_asm(rho_, Pr_, Ra_);

  local_asm.set_newton_solution(&in);
  // the computation of the Jacobian in the instationary case also
  // also needs updated ghost DoFs of the previous solution, so
  // make sure you call Update() on prev_sol_ before
  // calling this function
  local_asm.set_time_solution(&prev_sol_);
  local_asm.set_time_stepping_weights(alpha1_, alpha2_, alpha3_);

  global_asm_.assemble_matrix(space_, local_asm, *out);
}

//////////////// Pressure Filtering ////////////////

struct PressureIntegral : private AssemblyAssistant< DIM, Scalar > {

  PressureIntegral(const CVector &sol) : sol_(sol) {}

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, double &pressure) {
    AssemblyAssistant< DIM, double >::initialize_for_element(element, quadrature, false);
    evaluate_fe_function(sol_, DIM, p_);

    const int num_q = num_quadrature_points();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) {
      const double wq = w(q);
      const double dJ = std::abs(detJ(q));

      pressure += wq * p_[q] * dJ;
    }
  }

  const CVector &sol_;
  FunctionValues< double > p_;
};

struct VolumeIntegral : private AssemblyAssistant< DIM, double > {

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, double &vol) {
    AssemblyAssistant< DIM, double >::initialize_for_element(element, quadrature, false);
    const int num_q = num_quadrature_points();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) {
      const double wq = w(q);
      const double dJ = std::abs(detJ(q));
      vol += wq * dJ;
    }
  }
};

void Boussinesq::ApplyFilter(LAD::VectorType &u) {
  if (!use_pressure_filter_)
    return;
  double recv;
  u.Update();
  PressureIntegral int_p(u);
  double total_pressure;
  global_asm_.integrate_scalar(space_, int_p, total_pressure);

  MPI_Allreduce(&total_pressure, &recv, 1, MPI_DOUBLE, MPI_SUM, comm_);
  total_pressure = recv;

  double integrated_vol;
  VolumeIntegral vol_int;
  global_asm_.integrate_scalar(space_, vol_int, integrated_vol);

  MPI_Allreduce(&integrated_vol, &recv, 1, MPI_DOUBLE, MPI_SUM, comm_);
  integrated_vol = recv;

  const double average_pressure = total_pressure / integrated_vol;

  LOG_INFO("pressure_filter",
           "Average pressure before filter = " << average_pressure);

  pressure_correction_.CloneFromWithoutContent(u);
  pressure_correction_.Zeros();

  // set value for pressure dofs to average pressure
  std::vector< int > cell_p_dofs;
  std::vector< int > local_p_dofs;
  for (EntityIterator it = mesh_->begin(DIM), end = mesh_->end(DIM);
       it != end; ++it) {
    cell_p_dofs.clear();
    space_.get_dof_indices(DIM, it->index(), cell_p_dofs);
    for (int i = 0, sz = cell_p_dofs.size(); i < sz; ++i) {
      if (space_.dof().is_dof_on_subdom(cell_p_dofs[i])) {
        local_p_dofs.push_back(cell_p_dofs[i]);
      }
    }
  }

  std::sort(local_p_dofs.begin(), local_p_dofs.end());
  std::unique(local_p_dofs.begin(), local_p_dofs.end());

  // remove average pressure from solution
  std::vector< double > p_correction_values(local_p_dofs.size());
  std::fill(p_correction_values.begin(), p_correction_values.end(),
            average_pressure);

  pressure_correction_.SetValues(vec2ptr(local_p_dofs), local_p_dofs.size(),
                                 vec2ptr(p_correction_values));

  u.Axpy(pressure_correction_, -1.);

  u.Update();
  PressureIntegral int_p_check(u);
  global_asm_.integrate_scalar(space_, int_p_check, total_pressure);
  MPI_Allreduce(&total_pressure, &recv, 1, MPI_DOUBLE, MPI_SUM, comm_);
  total_pressure = recv;
  LOG_INFO("pressure_filter", "Average pressure after filter = "
                                  << total_pressure / integrated_vol);
}

class ForceIntegral : private AssemblyAssistant< DIM, double > {
public:
  enum FORCE_TYPE { DRAG = 0, LIFT = 1 };

  ForceIntegral(double nu, const CVector *sol,
                const std::vector< int > &bdy_dofs, FORCE_TYPE type)
      : nu_(nu), x_var_(type == DRAG ? 0 : 1), bdy_dofs_(bdy_dofs), sol_(sol) {
  } // rho assumed to be one

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, double &val) {
    AssemblyAssistant< DIM, double >::initialize_for_element(element, quadrature, false);

    std::vector< int > dofs;
    element.get_dof_indices(dofs);

    std::sort(dofs.begin(), dofs.end());

    std::vector< int > bdofs; // cylinder boundary dofs on the cell

    // find dofs on cell that also lie on cylinder boundary
    std::set_intersection(dofs.begin(), dofs.end(), bdy_dofs_.begin(),
                          bdy_dofs_.end(), std::back_inserter(bdofs));

    if (!bdofs.empty()) {
      dofs.clear();
      element.get_dof_indices(dofs);

      // We compute function values here only, since otherwise they will not be
      // needed.
      recompute_function_values();

      for (std::vector< int >::const_iterator d = bdofs.begin(),
                                              d_end = bdofs.end();
           d != d_end; ++d) {
        // Find local dof number for *d
        std::vector< int >::iterator i_it = std::find(
            dofs.begin() + dof_index(0, x_var_),
            dofs.begin() + dof_index(0, x_var_) + num_dofs(x_var_), *d);
        const int i = std::distance(dofs.begin() + dof_index(0, x_var_), i_it);
        assert(i >= 0);
        assert(i < num_dofs(x_var_));

        const int num_q = num_quadrature_points();

        for (int q = 0; q < num_q; ++q) {
          const double wq = w(q);
          const double dJ = std::abs(detJ(q));

          val -= wq *
                 (nu_ * dot(grad_phi(i, q, x_var_), grad_u_[x_var_][q]) -
                  p_[q] * grad_phi(i, q, x_var_)[x_var_]) *
                 dJ;

          // non-linear term involves all components of u_ and grad_u_.
          for (int v = 0; v < DIM; ++v) {
            val -= wq * (u_[v][q] * grad_u_[v][q][x_var_] * phi(i, q, x_var_)) *
                   dJ;
          }
        }
      }
    }
  }

private:
  void recompute_function_values() {
    for (int d = 0; d < DIM; ++d) {
      u_[d].clear();
      grad_u_[d].clear();

      evaluate_fe_function(*sol_, d, u_[d]);
      evaluate_fe_function_gradients(*sol_, d, grad_u_[d]);
    }
    p_.clear();
    evaluate_fe_function(*sol_, DIM, p_);
  }

  double nu_;
  int surface_material_num_, x_var_;
  FunctionValues< double > u_[DIM], p_;
  FunctionValues< Vec< DIM, double > > grad_u_[DIM];
  const std::vector< int > &bdy_dofs_;
  const CVector *sol_;
};

void Boussinesq::setup_linear_algebra() {
  TimingScope tscope("setup_linear_algebra");
  const std::string platform_str =
      params_["LinearAlgebra"]["Platform"].get< std::string >();
  if (platform_str == "CPU") {
    la_sys_.Platform = CPU;
  } else if (platform_str == "GPU") {
    la_sys_.Platform = GPU;
  } else {
    throw UnexpectedParameterValue("LinearAlgebra.Platform", platform_str);
  }
  init_platform(la_sys_);

  const std::string impl_str =
      params_["LinearAlgebra"]["Implementation"].get< std::string >();
  if (impl_str == "Naive") {
    la_impl_ = NAIVE;
  } else if (impl_str == "BLAS") {
    la_impl_ = BLAS;
  } else if (impl_str == "MKL") {
    la_impl_ = MKL;
  } else if (impl_str == "OPENMP") {
    la_impl_ = OPENMP;
  } else if (impl_str == "SCALAR") {
    la_impl_ = SCALAR;
  } else if (impl_str == "SCALAR_TEX") {
    la_impl_ = SCALAR_TEX;
  } else {
    throw UnexpectedParameterValue("LinearAlgebra.Implementation", impl_str);
  }

  const std::string matrix_str =
      params_["LinearAlgebra"]["MatrixFormat"].get< std::string >();
  if (matrix_str == "CSR") {
    la_matrix_format_ = CSR;
  } else if (matrix_str == "COO") {
    la_matrix_format_ = COO;
  } else {
    throw UnexpectedParameterValue("LinearAlgebra.MatrixFormat", impl_str);
  }
}

void Boussinesq::eval_ic() 
{
  for (size_t fe_ind = 0; fe_ind < DIM+2; ++fe_ind)
  {
    InitialBoussinesq initialCond(fe_ind, hT_, cT_);
    FeInterNodal<double, DIM, InitialBoussinesq > fe_inter (this->space_, &initialCond, fe_ind);
    fe_inter.interpolate (this->sol_);
  }

  prev_sol_.CloneFrom(this->sol_);
  
  sol_.Update();
  prev_sol_.Update();

}

void Boussinesq::write_results() {
  
  // compute pressure drop
  std::vector< Coord > p(2);
  // std::vector<std::vector<double>> p(2);

  p[0][0] == 0.15;
  p[0][1] == 0.2;
  p[1][0] == 0.25;
  p[1][1] == 0.2;

  // p[0][0] = 0.15;
  // p[0][1] = 0.2;
  // p[1][0] = 0.25;
  // p[1][1] = 0.2;

  FeEvalGlobal<Scalar, DIM> fe_eval(space_, sol_);
  
  std::vector< std::vector<Scalar> > fe_vals;
  
  fe_eval.evaluate (p, fe_vals); 
  
  std::cout << "      -- write out -- Pressure difference delta-p = " << fe_vals[0][DIM] - fe_vals[1][DIM] << std::endl;

  double* array;
  sol_.GetLocalValues(array);
  std::cout << "        example : " << std::to_string(array[0]) << std::endl;

  double results[3];
  results[0] = sol_.Norm1(); // Summennorm
  results[1] = sol_.Norm2(); // Euklidische Norm
  results[2] = sol_.NormMax(); // Maximumsnorm

  std::stringstream filename;
  filename << this->path_out_ << "/results.out";

  std::ofstream file (filename.str().c_str());
  for (int r = 0; r < 3; r++ ){
    file << results[r] << " ";
  }
  file.close();
  
}

void write_csv(std::string filename, std::vector< std::vector<Scalar> > fe_vals, std::vector< Coord > points){
    // Make a CSV file with one or more columns of integer values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<int>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    std::ofstream myFile(filename);
    
    myFile << "point_x,point_y,u,v,p,t, \n";
    
    // Send data to the stream
    for(int i = 0; i < fe_vals.size(); ++i)
    {
      // add point coordinates
      myFile << points[i][0] << "," << points[i][1] << ",";
      for(int j = 0; j < 4; ++j)
      {
          myFile << fe_vals[i][j] << ",";
      }
      myFile << "\n";
    }
    
    // Close the file
    myFile.close();
}

void Boussinesq::init_results_csv() {
    // points of interest in regular distance
    //std::filesystem:create_directory(path_out_ + std::string("/results_points"))
    //boost::filesystem::create_directories((path_out_ + std::string("/results_points")).c_str());
    double results_points_distance = 0.25;
    std::cout << "distance2 : " << results_points_distance << std::endl;
    double x = results_points_distance;
    double y = results_points_distance;
    while (x < 0.9999 ){
      while (y < 0.9999 ){
        double vp1[2] = {x,y};
        Coord p1(vp1);
        results_points.push_back(p1);
        std::stringstream filename;
        filename << this->path_out_ << "/results_points/" << x << "_" << y << ".csv";
        std::ofstream myFile(filename.str());
        myFile << "u,v,p,t, \n";
        myFile.flush();
        myFile.close();
        y += results_points_distance;
      }
      x += results_points_distance;
      y = results_points_distance;
    }
}

void Boussinesq::write_results_csv() {
  std::cout << "start writing" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  FeEvalGlobal<Scalar, DIM> fe_eval(space_, sol_);
  std::vector< std::vector<Scalar> > fe_vals;
  fe_eval.evaluate (results_points, fe_vals); // fe_vals is of size (points x 4) where the 4 values are [u,v ,p t]
  double results_points_distance = 0.25;
  double x = results_points_distance;
  double y = results_points_distance;
  int point = 0;
  while (x < 0.9999 ){
    while (y < 0.9999 ){
      std::stringstream filename;
      filename << this->path_out_ << "/results_points/" << x << "_" << y << ".csv";
      std::ofstream out_file(filename.str(),std::ios::app );
      // If file is really open...
      if (out_file.is_open()) {
        for (int i = 0; i < fe_vals[point].size(); ++i) {
          out_file << fe_vals[point][i];

          if (i == fe_vals[point].size() - 1) {
            out_file << "\n";
          } else {
            out_file << ", ";
          }
        }
        // Write output to file
        out_file.flush();
        out_file.close();
      }
      point += 1;
      y += results_points_distance;
    }
    x += results_points_distance;
    y = results_points_distance;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "end writing " << ms_double.count() << "sec"<< std::endl;
}