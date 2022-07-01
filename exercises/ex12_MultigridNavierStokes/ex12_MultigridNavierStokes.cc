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

#include "ex12_MultigridNavierStokes.h"

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <vector>

#ifdef WITH_GPERF
#include "profiler.h"
#endif

namespace {
static const char *DATADIR = MESHES_DATADIR;
static const int MASTER_RANK = 0;
static const char *PARAM_FILENAME = "ex12_MultigridNavierStokes.xml";

static bool CONSOLE_OUTPUT_ACTIVE = true;
static const int CONSOLE_THRESHOLD_LEVEL = 2;
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

//#if !(DIM == 3)
//#error "The channel benchmark only works in 3d!"
//#endif

struct TimingData 
{
  Scalar time_elapsed;
};

class TimingScope 
{
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

class TimingReportOutputVisitor 
{
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

class ChannelBenchmark : public NonlinearProblem< LAD > {

  
public:
  ChannelBenchmark(const std::string &param_filename)
      : comm_(MPI_COMM_WORLD),
        params_(param_filename.c_str(), MASTER_RANK, MPI_COMM_WORLD),
        use_pressure_filter_(false), refinement_level_(0), is_done_(false), linear_solver_(nullptr), gmg_coarse_(nullptr), precond_(nullptr)

  {}

  virtual ~ChannelBenchmark() {}

  virtual void run() 
  {
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
        0, "==== ChannelBenchmark                                    ===");
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
    //LOG_INFO("parameters", params_);

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
    } else {
      LOG_INFO("simulation", "Solving stationary problem");
      if (true) {

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

      }

      visualize();
      output_norms();
      if (compute_bench_quantities_) {
        compute_dfg_benchmark_quantities();
        
      }
    }

    CONSOLE_OUTPUT(0, "");

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

    LogKeeper::get_log("info").flush();
    LogKeeper::get_log("debug").flush();
    LogKeeper::get_log("info").set_target(0);
    LogKeeper::get_log("debug").set_target(0);

    CONSOLE_OUTPUT(
        0, "============================================================");
  }

  virtual void prepare_dirichlet_bc_solver(int level, 
                                           const VectorSpace< Scalar, DIM >* space, 
                                           std::vector<int>& dirichlet_dofs,
                                           std::vector<Scalar>& dirichlet_vals)
  {
    // set values to zero, since LinearSolver computes update deltaX
    this->prepare_bc(space, dirichlet_dofs, dirichlet_vals);
    const size_t ndof = dirichlet_vals.size();
    for (size_t i=0; i!= ndof; ++i)
    {
      dirichlet_vals[i] = 0.;
    }
  }
    
  virtual void prepare_operator(int level,
                                const VectorSpace< Scalar, DIM >* space, 
                                CMatrix* matrix) const;
  
  virtual void prepare_rhs(int level,
                           const VectorSpace< Scalar, DIM >* space, 
                           CVector* vector) const;
                              
  virtual void assemble_operator(int level,
                                 const VectorSpace< Scalar, DIM >* space, 
                                 const std::vector<int>& dirichlet_dofs,
                                 const std::vector<Scalar>& dirichlet_vals,
                                 CMatrix* matrix);
  
  virtual void assemble_rhs(int level,
                            const VectorSpace< Scalar, DIM >* space,
                            const std::vector<int>& dirichlet_dofs,
                            const std::vector<Scalar>& dirichlet_vals,
                            CVector* vector);
                            
  virtual void prepare_fixed_dofs_gmg(int gmg_level,
                                      const VectorSpace< Scalar, DIM >* space, 
                                      std::vector<int>& fixed_dofs,
                                      std::vector<Scalar>& fixed_vals) const;

private:
  virtual void prepare_linear_solver();
  virtual void prepare_outer_solver(LinearSolver<LAD>*& solver, const PropertyTree &params);
  
  virtual void prepare_gmg(const PropertyTree &gmg_param,
                           const PropertyTree &locsolver_param);



  virtual void update_preconditioner(const CVector &u, CMatrix *DF);
  
  const MPI_Comm &communicator() const { return comm_; }

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
  virtual void prepare_bc(const VectorSpace< Scalar, DIM >* space, 
                                    std::vector<int>& dirichlet_dofs,
                                    std::vector<Scalar>& dirichlet_vals) const;
  // Update step in nonlinear solver
  void update_solution();
  void update_solution_naive();
  void update_solution_armijo();

  // Compute forcing term for inexact Newton method.
  void choose_forcing_term(int iter);

  // Computate instationary solution by time-stepping method.
  void run_time_loop();

  // Visualize the solution in a file. In stationary mode, the
  // filename contains 'stationary', in instationary mode, it contains the
  // current time-step ts_.
  void visualize();

  // Compute L2 or H1 norm of variable defined through vars on the
  // master process.
  Scalar compute_norm(int norm_type, const std::vector< int > &vars);

  void compute_divergence (Scalar& l2_div);
  
  // Output various norms of the solution.
  void output_norms();

  virtual void Reinit();

  // Helper functions for nonlinear solver
  void ApplyFilter(LAD::VectorType &u);
  virtual void EvalFunc(const LAD::VectorType &in, LAD::VectorType *out);
  void compute_residual(const LAD::VectorType &in, LAD::VectorType *out); // updates res_ with the residual

  virtual void EvalGrad(const LAD::VectorType &in, LAD::MatrixType *out);
  void compute_jacobian(const LAD::VectorType &in, LAD::MatrixType *out); // updates matrix_ with the jacobian matrix

  // Pressure filter: substracts the mean of the pressure from each
  // pressure dof in sol_ .
  void filter_pressure();

  // Linear algebra set up
  void setup_linear_algebra();

  // compute L2-Error and H1semi-Error
  void compute_errors();

  // compute difference between solution last and penultmate timestep
  void compute_difference();

  void compute_dfg_benchmark_quantities();
  Scalar compute_drag_force();
  void compute_forces(Scalar &drag_force, Scalar &lift_force);
  void compute_force_coefficients(Scalar drag_force, Scalar lift_force,
                                  Scalar &drag_coef, Scalar &lift_coef) const;

  void find_cylinder_boundary_dofs(int cylinder_mat_num, size_t fe_ind,
                                   std::vector< int > &bdy_dofs);

  void get_linear_solver_statistics ( bool erase );

  // MPI stuff
  MPI_Comm comm_;
  int rank_, num_partitions_;

  // Linear algebra stuff
  SYSTEM la_sys_;
  IMPLEMENTATION la_impl_;
  MATRIX_FORMAT la_matrix_format_;

  // Parameter data read in from file.
  PropertyTree params_;

  std::string simul_name_; // parameter 'OutputPrefix': prefix for output files

  // Time-stepping variables
  int ts_;
  Scalar dt_;
  Scalar alpha0_, alpha1_, alpha2_, alpha3_;

  // Flow model variables
  Scalar Um_, H_, W_, rho_, nu_;
  Scalar U_mean_, diam_; // (DFG-specific) mean velocity and diameter of cylinder.

  int inflow_bdy_;
  int outflow_bdy_;
  
  // Flag for pressure filter -- parameter 'UsePressureFilter'
  bool use_pressure_filter_;

  // Meshes
  MeshPtr mesh_;
  int refinement_level_;

  VectorSpaceSPtr< Scalar, DIM > space_;

  // linear algebra objects
  CMatrix matrix_;
  CVector sol_, prev_sol_, cor_, res_, pressure_correction_, exact_sol_, error_;

  // linear solver parameters
  int lin_max_iter;
  Scalar lin_abs_tol;
  Scalar lin_rel_tol;
  Scalar lin_div_tol;
  int basis_size;

  // nonlinear solver parameters
  int nls_max_iter;
  Scalar nls_abs_tol;
  Scalar nls_rel_tol;
  Scalar nls_div_tol;
  Scalar eta_; // forcing term
  std::vector< Scalar > residual_history_, forcing_term_history_;
  bool do_armijo_update_;
  std::string forcing_strategy_;
  bool use_forcing_strategy_;

  // damping strategy paramters
  Scalar theta_initial;
  Scalar theta_min;
  Scalar armijo_dec;
  Scalar suff_dec;
  int max_armijo_ite;

  // forcing strategy parameters
  Scalar eta_initial;
  Scalar eta_max;
  Scalar gamma_EW2;
  Scalar alpha_EW2;

  // nonlinear solver
  Newton< LAD, DIM > newton_;

  AllGlobalAssembler< Scalar, DIM > global_asm_;

  bool is_done_, solve_instationary_, convergence_test_;

  std::vector< int > dirichlet_dofs_;
  std::vector< Scalar > dirichlet_values_;



  bool is_dfg_benchmark_;
  bool compute_bench_quantities_;
  bool use_hiflow_newton_;

  HierarchicalReport< TimingData > time_report_;
  Table results_table_;

  // CSV output
  CSVWriter< Scalar > bench_quantity_writer_;
  std::vector< std::string > bench_names_;
  std::vector< Scalar > bench_quantities_;
  
  int nb_lvl_;
  std::vector< VectorSpaceSPtr< Scalar, DIM > > all_spaces_;
  std::vector< MeshPtr > all_meshes_;
  std::vector< MeshPtr > all_meshes_no_ghost_;
  MeshPtr master_mesh_;
  
    // outer solver
  LinearSolver< LAD > *linear_solver_;
  Preconditioner< LAD > *precond_;
  
  // standard block jacobi precond
  PreconditionerBlockJacobiStand< LAD > bJacobiStd_;
  
  // external block jacobi precond
  PreconditionerBlockJacobiExt< LAD > bJacobiExt_;
  
  // richardson iteration
  Richardson< LAD > richardson_;
  
  // vanka
  PreconditionerVanka< LAD, DIM > vanka_;
  
  // geometric multigrid
  GMGStandard < ChannelBenchmark, LAD, DIM> gmg_;
  
  LinearSolver<LAD>* gmg_coarse_;
  //PreconditionerBlockJacobiStand< LAD > gmg_coarse_bJacobiStd_;
  //PreconditionerBlockJacobiExt< LAD > gmg_coarse_bJacobiExt_;
  //Richardson< LAD > gmg_coarse_richardson_;
  //PreconditionerVanka< LAD, DIM > gmg_coarse_vanka_;
  
  std::vector< Preconditioner< LAD >* > gmg_smoothers_;
  std::vector< VectorSpace< Scalar, DIM >* > gmg_spaces_;

  std::vector< CVector * > gmg_newton_;
  std::vector< CVector * > gmg_solP_;
  std::vector< CVector * > gmg_solP_prev_;
  
  bool use_gmg_;      
  int gmg_nb_lvl_;

  int solver_update_time_step_;
  int solver_update_newton_step_;
};

// program entry point

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);

#ifdef WITH_GPERF
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str_time(buffer);
  
    std::string prof_name = "ChannelBM_" + str_time + ".log";
    ProfilerStart(prof_name.c_str());
#endif

  std::string param_filename(PARAM_FILENAME);
  if (argc > 1) {
    param_filename = std::string(argv[1]);
  }

  try {
    ChannelBenchmark app(param_filename);
    app.run();
  } catch (std::exception &e) {
    std::cerr << "\nProgram ended with uncaught exception.\n";
    std::cerr << e.what() << "\n";
    return -1;
  }


#ifdef WITH_GPERF
    ProfilerStop();
#endif
  MPI_Finalize();

  return 0;
}

void ChannelBenchmark::read_mesh() 
{
  TimingScope tscope("read_mesh");

  std::string precond_type = params_["LinearSolver"]["PrimalSolver"]["PrecondType"].get<std::string>();
  
  if (precond_type == "GMG")
  {
    this->use_gmg_ = true;
  }
  else
  {
    this->use_gmg_ = false;
  }
  
  const bool use_bdd =
      params_["UseBoundaryDomainDescriptor"].get< bool >(false);

  Scalar radius = 0.05;
  Coord center;
#if DIM == 2
  center.set(0, 0.2);
#else
  center.set(0, 0.5);
#endif
  center.set(1, 0.2);
  CylinderDescriptor cyl(radius, center);
  
  LOG_INFO("Build","sequential initial mesh " );
  
  mesh::IMPL MESHIMPL = mesh::IMPL_DBVIEW;
  
  // read in mesh
#if DIM == 2
    const std::string mesh_name =
        params_["Mesh"]["Filename1"].get< std::string >();
#else
    const std::string mesh_name =
        params_["Mesh"]["Filename2"].get< std::string >();
#endif
    std::string mesh_filename = std::string(DATADIR) + mesh_name;
    

  // on master:  read mesh and refine sequentially
  if (rank() == MASTER_RANK) 
  {
    LOG_INFO("Read"," mesh from file " << mesh_filename );
    this->master_mesh_ = read_mesh_from_file(mesh_filename, DIM, DIM, 0, MESHIMPL);

    int seq_ref_lvl = params_["Mesh"]["SequentialRefLevel"].template get< int >(0);

    int current_nb_cells = master_mesh_->num_entities(DIM);
    int needed_cells = 8 * this->num_partitions_;
    
    int min_ref_lvl = 0;
    while (current_nb_cells < needed_cells)
    {
      min_ref_lvl++;
#if DIM == 3
      current_nb_cells *= 8;
#else
      current_nb_cells *= 4;
#endif
    }
    if (seq_ref_lvl < min_ref_lvl)
    {
      seq_ref_lvl = min_ref_lvl;
    }
    
    LOG_INFO("Refine","sequential mesh " << seq_ref_lvl << " times ");
    if (seq_ref_lvl > 0) 
    {
      this->master_mesh_ = this->master_mesh_->refine_uniform_seq(seq_ref_lvl);
#if DIM == 3
      if (use_bdd)
        adapt_boundary_to_function(this->master_mesh_, cyl);
#endif
    }
    this->refinement_level_ = seq_ref_lvl;
  }
  MPI_Bcast(&refinement_level_, 1, MPI_INT, MASTER_RANK, this->comm_);
  
  // partition mesh
  MeshPtr mesh_no_ghost = 0;
  int uniform_ref_steps = 0;
  if (MESHIMPL == mesh::IMPL_P4EST) 
  {
    mesh_no_ghost = partition_and_distribute(this->master_mesh_, MASTER_RANK, this->comm_, 
                                             uniform_ref_steps, 
                                             mesh::IMPL_P4EST);
  }
  else
  {
#ifdef USE_PARMETIS
    ParMetisGraphPartitioner partitioner;
    LOG_INFO("",">  Partition initial mesh with ParMetis " );
#else
# ifdef WITH_METIS
    MetisGraphPartitioner partitioner;
    LOG_INFO("",">  Partition initial mesh with Meetis " );
# else
    NaiveGraphPartitioner partitioner;
    LOG_INFO("",">  Partition initial mesh with naive partitioner " );
# endif
#endif
    const GraphPartitioner *p = &partitioner;
    mesh_no_ghost = partition_and_distribute(this->master_mesh_, MASTER_RANK, this->comm_,
                                             p, uniform_ref_steps, mesh::IMPL_DBVIEW);  
  }
  this->refinement_level_ += uniform_ref_steps;
  assert(mesh_no_ghost != 0);
  
  // refine mesh locally until desired refinement is achieved
  const int start_ref_lvl = this->refinement_level_;
  int end_ref_lvl = params_["Mesh"]["InitialRefLevel"].get< int >(5);
  
  LOG_INFO("","> Refinement level after partitioning:  " << refinement_level_ );
  LOG_INFO("","> Final uniform refinement level:       " << end_ref_lvl );
    
  this->nb_lvl_ = end_ref_lvl - start_ref_lvl + 1;
   
  if (use_gmg_)
  {
    SharedVertexTable shared_verts;
    
    this->all_meshes_.resize(this->nb_lvl_, 0);
    this->all_meshes_no_ghost_.resize(this->nb_lvl_, 0); 
  
    this->all_meshes_no_ghost_[0] = mesh_no_ghost;
    this->all_meshes_[0] = compute_ghost_cells(*(all_meshes_no_ghost_[0]), this->comm_, shared_verts, MESHIMPL, 1);    
    
    assert (all_meshes_[0] != 0);

    for (int i = 1; i < this->nb_lvl_; ++i) 
    {
      //std::cout << "level " << i << std::endl;
      SharedVertexTable shared_verts;
    
      all_meshes_no_ghost_[i] = all_meshes_no_ghost_[i-1]->refine();
      assert(all_meshes_no_ghost_[i] != 0);

#if DIM == 3
      if (use_bdd)
        adapt_boundary_to_function(all_meshes_no_ghost_[i]);
#endif

    
      all_meshes_[i] = compute_ghost_cells(*(all_meshes_no_ghost_[i]), this->comm_, shared_verts, MESHIMPL, 1);    
      assert (all_meshes_[i] != 0);
    
      this->refinement_level_++;
    }
     
    this->mesh_ = all_meshes_[this->nb_lvl_-1];
  }
  else
  {
    this->mesh_ = mesh_no_ghost;
  
    // uniform refinement of distributed mesh
    for (int i = 1; i < this->nb_lvl_; ++i) 
    {
      this->mesh_ = this->mesh_->refine();
#if DIM == 3
      if (use_bdd)
        adapt_boundary_to_function(this->mesh_, cyl);
#endif
      
      this->refinement_level_++;
    }
    SharedVertexTable shared_verts;
    this->mesh_ = compute_ghost_cells(*(this->mesh_), this->comm_, shared_verts, MESHIMPL, 1);
  }
  
  LOG_INFO("","> Refinement level after refinement:  " << refinement_level_ );
}

void ChannelBenchmark::prepare() 
{
  TimingScope tscope("prepare");

  // prepare timestep
  ts_ = 0;
  dt_ = params_["Instationary"]["Timestep"].get< Scalar >();

  solve_instationary_ = params_["Instationary"]["SolveInstationary"].get< bool >();
          
  // set the alpha coefficients correctly for the
  // Crank-Nicolson method.
  if (solve_instationary_)
  {
    alpha0_ = 1.;
    alpha1_ = 0.5 * dt_;
    alpha2_ = dt_;
    alpha3_ = 0.5 * dt_;
  }
  else
  {
    alpha0_ = 0.;
    alpha1_ = 1.;
    alpha2_ = 1.;
    alpha3_ = 0.;
  }
  
  // prepare problem parameters
  rho_ = params_["FlowModel"]["Density"].get< Scalar >();
  nu_ = params_["FlowModel"]["Viscosity"].get< Scalar >();

  Um_ = params_["FlowModel"]["InflowSpeed"].get< Scalar >();
  H_ = params_["FlowModel"]["InflowHeight"].get< Scalar >();
  W_ = params_["FlowModel"]["InflowWidth"].get< Scalar >();

  
  inflow_bdy_ = params_["Boundary"]["InflowMaterial"].get< int >();
  outflow_bdy_ = params_["Boundary"]["OutflowMaterial"].get< int >();
  
  // prepare space
  const int u_deg = params_["FiniteElements"]["VelocityDegree"].get< int >();
  const int p_deg = params_["FiniteElements"]["PressureDegree"].get< int >();
  
  
  std::vector< int > fe_params;
  for (int c = 0; c < DIM; ++c) {
    fe_params.push_back(u_deg);
  }
  fe_params.push_back(p_deg);
  
  std::vector< FEType > fe_ansatz (DIM+1, FEType::LAGRANGE);
  std::vector< bool > is_cg (DIM+1, true);
  
  // Initialize the VectorSpace object.
  bool use_gmg = false;
  std::string precond_type = params_["LinearSolver"]["PrimalSolver"]["PrecondType"].get<std::string>();
  
  if (precond_type == "GMG")
  {
    use_gmg = true;
  }
  
  
  this->all_spaces_.clear();
  this->space_ = VectorSpaceSPtr<Scalar, DIM>(new VectorSpace<Scalar, DIM>());
  
  std::string number_type = params_["LinearAlgebra"]["Numbering"].get<std::string>();
  bool order_fe_first  = params_["LinearAlgebra"]["OrderFEbeforeCell"].get<bool>();
  
  if (number_type == "CuthillMcKee")
  {
    space_->Init(*mesh_, fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::CUTHILL_MCKEE, order_fe_first);
  }
  else if (number_type == "King")
  {
    space_->Init(*mesh_, fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::KING, order_fe_first);
  }
  else
  {
    space_->Init(*mesh_, fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC, order_fe_first);
  }
      
  if (use_gmg)
  {
    this->all_spaces_.resize(this->nb_lvl_);
    for (int l=0; l<this->nb_lvl_-1; ++l)
    {
      this->all_spaces_[l] = VectorSpaceSPtr<Scalar, DIM>(new VectorSpace< Scalar, DIM >());
      
        if (number_type == "CuthillMcKee")
        {
          this->all_spaces_[l]->Init(*this->all_meshes_[l], fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::CUTHILL_MCKEE, order_fe_first);
        }
        else if (number_type == "King")
        {
          this->all_spaces_[l]->Init(*this->all_meshes_[l], fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::KING, order_fe_first);
        }
        else
        {
          this->all_spaces_[l]->Init(*this->all_meshes_[l], fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC, order_fe_first);
        }
    }
    this->all_spaces_[this->nb_lvl_-1] = this->space_;
  }
  
  
  CONSOLE_OUTPUT(1, "Total number of dofs = " << space_->dof().nb_dofs_global());

  for (int p = 0; p < num_partitions(); ++p) {
    CONSOLE_OUTPUT(2, "Num dofs on process " << p << " = "
                                             << space_->dof().nb_dofs_on_subdom(p));
  }

  // pressure filter
  use_pressure_filter_ = params_["UsePressureFilter"].get< bool >();

  // prepare global assembler
  QuadratureSelection q_sel(params_["QuadratureOrder"].get< int >());
  global_asm_.set_quadrature_selection_function(q_sel);

  // set DFG benchmark flag
  is_dfg_benchmark_ = params_["DFGbenchmark"].get< bool >();
  compute_bench_quantities_ = params_["BenchQuantities"].get< bool >();
  use_hiflow_newton_ =
      params_["NonlinearSolver"]["UseHiFlowNewton"].get< bool >();

  if (is_dfg_benchmark_) {
#if DIM == 2
    U_mean_ = 2. / 3. * Um_;
#elif DIM == 3
    U_mean_ = 4. / 9. * Um_;
#endif
    diam_ = 0.1;
    CONSOLE_OUTPUT(1, "Reynolds number = " << U_mean_ * diam_ / nu_);
  }

  // compute matrix graph

  std::vector< std::vector< bool > > coupling_vars;

  coupling_vars.resize(DIM + 1);
  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM + 1; ++j) {
      coupling_vars[i].push_back(true);
    }
  }
  for (int i = 0; i < DIM; ++i) {
    coupling_vars[DIM].push_back(true);
  }
  coupling_vars[DIM].push_back(false);

  SparsityStructure sparsity;

  compute_sparsity_structure(*space_, sparsity, coupling_vars, false);

  matrix_.Init(communicator(), space_->la_couplings(), la_platform(), la_implementation(),
               la_matrix_format());
  sol_.Init(communicator(), space_->la_couplings(), la_platform(), la_implementation());
  prev_sol_.Init(communicator(), space_->la_couplings(), la_platform(),
                 la_implementation());
  cor_.Init(communicator(), space_->la_couplings(), la_platform(), la_implementation());
  res_.Init(communicator(), space_->la_couplings(), la_platform(), la_implementation());

  matrix_.InitStructure(sparsity);
  matrix_.Zeros();

  sol_.Zeros();
  prev_sol_.Zeros();
  cor_.Zeros();
  res_.Zeros();

  // setup linear solver
  this->prepare_linear_solver();

  // get nonlinear solver parameters from param file
  nls_max_iter = params_["NonlinearSolver"]["MaximumIterations"].get< int >();
  nls_abs_tol = params_["NonlinearSolver"]["AbsoluteTolerance"].get< Scalar >();
  nls_rel_tol = params_["NonlinearSolver"]["RelativeTolerance"].get< Scalar >();
  nls_div_tol = params_["NonlinearSolver"]["DivergenceLimit"].get< Scalar >();
  do_armijo_update_ = params_["NonlinearSolver"]["ArmijoUpdate"].get< bool >();
  forcing_strategy_ =
      params_["NonlinearSolver"]["ForcingStrategy"].get< std::string >();
  use_forcing_strategy_ = (forcing_strategy_ != "None");
  eta_ = 1.e-4; // initial value of forcing term

  // get damping strategy parameters from param file
  theta_initial = params_["NonlinearSolver"]["ThetaInitial"].get< Scalar >();
  theta_min = params_["NonlinearSolver"]["ThetaMinimal"].get< Scalar >();
  armijo_dec = params_["NonlinearSolver"]["ArmijoDecrease"].get< Scalar >();
  suff_dec = params_["NonlinearSolver"]["SufficientDecrease"].get< Scalar >();
  max_armijo_ite =
      params_["NonlinearSolver"]["MaxArmijoIteration"].get< int >();

  // get forcing strategy parameters from param file
  eta_initial =
      params_["NonlinearSolver"]["InitialValueForcingTerm"].get< Scalar >();
  eta_max = params_["NonlinearSolver"]["MaxValueForcingTerm"].get< Scalar >();
  gamma_EW2 = params_["NonlinearSolver"]["GammaParameterEW2"].get< Scalar >();
  alpha_EW2 = params_["NonlinearSolver"]["AlphaParameterEW2"].get< Scalar >();

  // setup nonlinear solver
  newton_.InitParameter(&res_, &matrix_);
  newton_.InitParameter(Newton< LAD, DIM >::NewtonInitialSolutionOwn);
  newton_.InitControl(nls_max_iter, nls_abs_tol, nls_rel_tol, nls_div_tol);
  newton_.SetOperator(*this);
  newton_.SetLinearSolver(*this->linear_solver_);
  newton_.SetPrintLevel(2);
  
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
  prepare_bc(this->space_.get(), this->dirichlet_dofs_, this->dirichlet_values_);
  
  // apply BC to initial solution
  if (!dirichlet_dofs_.empty()) 
  {
    // correct solution with dirichlet BC
    sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
  }
}

void ChannelBenchmark::prepare_bc(const VectorSpace< Scalar, DIM >* space, 
                                    std::vector<int>& dirichlet_dofs,
                                    std::vector<Scalar>& dirichlet_values) const
{
  TimingScope tscope("prepare_bc");

  dirichlet_dofs.clear();
  dirichlet_values.clear();

# if DIM == 3
  ChannelFlowBC3d bc[3] = {
      ChannelFlowBC3d(0, W_, H_, Um_, inflow_bdy_, outflow_bdy_),
      ChannelFlowBC3d(1, W_, H_, Um_, inflow_bdy_, outflow_bdy_),
      ChannelFlowBC3d(2, W_, H_, Um_, inflow_bdy_, outflow_bdy_)};
# else // 2D
  ChannelFlowBC2d bc[2] = {
      ChannelFlowBC2d(0, H_, Um_, inflow_bdy_, outflow_bdy_),
      ChannelFlowBC2d(1, H_, Um_, inflow_bdy_, outflow_bdy_)};
#endif // 3D


  for (int var = 0; var < DIM; ++var) 
  {
    compute_dirichlet_dofs_and_values(bc[var], *space, var, dirichlet_dofs, dirichlet_values);
  }
}

void ChannelBenchmark::run_time_loop() 
{
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
#endif

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

    output_norms();

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

  const Scalar end_time = params_["Instationary"]["Endtime"].get< Scalar >();
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

  // Initialize CSV Writer
  std::stringstream bench_file;
  bench_file << num_partitions_ << "_" << dt_ << "_"
             << "benchmarking_quantities.csv";

  if (this->rank_ == MASTER_RANK) {
    LOG_INFO("Benchmarking quantities file", bench_file.str());
  }
  bench_quantity_writer_.InitFilename(bench_file.str());

  if (params_["Backup"]["Restore"].get< bool >() && ts_ != 0) {
    std::vector< std::vector< Scalar > > stored_bench;
    std::vector< std::vector< Scalar > > preserve_bench;
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

    newton_.SetPrintLevel(3);
    newton_.Solve(&sol_);
    CONSOLE_OUTPUT(1, "Newton ended with residual norm "
                            << newton_.GetResidual() << " after "
                            << newton_.iter() << " iterations.");
    bench_quantities_[1] = newton_.iter();
    bench_quantities_[5] += this->linear_solver_->iter();

    this->get_linear_solver_statistics(true);

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
    output_norms();

    if (compute_bench_quantities_) {
      compute_dfg_benchmark_quantities();
    }

    ++ts_;
  }
}

void ChannelBenchmark::visualize() 
{
  TimingScope tscope("Visualization");

  // prepare cell attributes to be visualized
  std::vector< Scalar > remote_index(mesh_->num_entities(mesh_->tdim()), 0);
  std::vector< Scalar > sub_domain(mesh_->num_entities(mesh_->tdim()), 0);
  std::vector< Scalar > material_number(mesh_->num_entities(mesh_->tdim()), 0);

  for (mesh::EntityIterator it = mesh_->begin(mesh_->tdim());
       it != mesh_->end(mesh_->tdim()); ++it) 
  {
    int temp1, temp2;
    mesh_->get_attribute_value("_remote_index_", mesh_->tdim(), it->index(), &temp1);
    mesh_->get_attribute_value("_sub_domain_", mesh_->tdim(), it->index(), &temp2);
    material_number.at(it->index()) = mesh_->get_material_number(mesh_->tdim(), it->index());
    remote_index.at(it->index()) = temp1;
    sub_domain.at(it->index()) = temp2;
  }
  
  // Setup visualization object.
  int num_intervals = 1;
  std::vector<size_t> visu_vars(DIM+1, 0);
  for (size_t d=0; d<DIM+1; ++d)
  {
    visu_vars[d] = d;
  }

  std::vector< std::string > names;
  for (size_t d=0; d<DIM; ++d)
  {
    names.push_back("u_" + std::to_string(d));
  }
  names.push_back("p");

  const int inflow_bdy = params_["Boundary"]["InflowMaterial"].get< int >();
  const int outflow_bdy = params_["Boundary"]["OutflowMaterial"].get< int >();

  CellVisualization< Scalar, DIM > visu(*space_, num_intervals);

  // evaluate functions to be visualized at visualization grid points
  sol_.Update();
  visu.visualize (FeEvalCell< Scalar, DIM >(*space_, sol_, visu_vars), names);
  
  // visualize cell data
  visu.visualize_cell_data(material_number, "Material Id");
  visu.visualize_cell_data(remote_index, "_remote_index_");
  visu.visualize_cell_data(sub_domain, "_sub_domain_");
  
  // generate filename
  std::stringstream input;
  input << simul_name_ << "_solution";

  if (solve_instationary_) 
  {
    if (ts_ < 10) input << "000" << ts_;
    else if (ts_ < 100) input << "00" << ts_;
    else if (ts_ < 1000) input << "0" << ts_;
    else input << "" << ts_;
  } 
  else 
  {
    input << "_stationary";
  }
  
  // writer object
  VTKWriter< Scalar, DIM> vtk_writer (visu, this->comm_, MASTER_RANK);
  vtk_writer.write(input.str());
}

Scalar ChannelBenchmark::compute_norm(int norm_type,
                                      const std::vector< int > &vars) 
{
  Scalar local_norm = -1.e30, global_norm = 0.;
  sol_.Update();
  switch (norm_type) {
  case 0: // L2-norm
  {
    L2NormIntegratorPp L2_int(sol_, vars);
    global_asm_.integrate_scalar(*space_, L2_int, local_norm);
    break;
  }
  case 1: // H1-seminorm
  {
    H1semiNormIntegratorPp H1_int(sol_, vars);
    global_asm_.integrate_scalar(*space_, H1_int, local_norm);
    break;
  }
  default:
    std::cerr << "unknown type of norm!\n";
    assert(false);
  };

  // NB: global value will only be returned on master proc -- others will return
  // 0.

  MPI_Reduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MASTER_RANK, comm_);

  return std::sqrt(global_norm);
}

void ChannelBenchmark::output_norms() 
{
  TimingScope tscope("Norm computation");
  std::vector< int > vel_vars, p_var;
  vel_vars.push_back(0);
  vel_vars.push_back(1);
#if DIM == 3
  vel_vars.push_back(2);
  p_var.push_back(3);
#else
  p_var.push_back(2);
#endif

  const Scalar L2_vel_norm = compute_norm(0, vel_vars);
  const Scalar L2_p_norm = compute_norm(0, p_var);
  const Scalar H1_vel_norm = compute_norm(1, vel_vars);
  const Scalar H1_p_norm = compute_norm(1, p_var);

  Scalar l2_div;
  compute_divergence(l2_div);
  
  if (rank_ == MASTER_RANK) 
  {
    LOG_INFO("L2-norm of velocity", L2_vel_norm);
    LOG_INFO("L2-norm of pressure", L2_p_norm);
    LOG_INFO("H1-seminorm of velocity", H1_vel_norm);
    LOG_INFO("H1-seminorm of pressure", H1_p_norm);
    LOG_INFO("L2 norm of div(u)", l2_div);

    results_table_.insert("|u|_L2", L2_vel_norm);
    results_table_.insert("|p|_L2", L2_p_norm);
    results_table_.insert("|u|_H1", H1_vel_norm);
    results_table_.insert("|p|_H1", H1_p_norm);
    results_table_.insert("|div(u)|_L2", l2_div);
  }
}

void ChannelBenchmark::Reinit() 
{
  //  prev_sol_.CloneFrom(sol_);
}

void ChannelBenchmark::EvalFunc(const LAD::VectorType &in,
                                LAD::VectorType *out) 
{
  Timer assembly_timer;
  assembly_timer.start();
  compute_residual(in, out);
  //    out->Scale(-1.0);
  assembly_timer.stop();

  bench_quantities_[3] += assembly_timer.get_duration();
}

void ChannelBenchmark::compute_residual(const LAD::VectorType &in,
                                        LAD::VectorType *out) 
{
  TimingScope tscope("Compute Residual");
  CONSOLE_OUTPUT(3, "Compute residual");

  // the evaluation of the residual needs updated ghost DoFs,
  // so make sure you call Update() on the input vector
  // before you call this function!

  InstationaryFlowAssembler local_asm(nu_, rho_);
  local_asm.set_newton_solution(&in);
  // the computation of the instationary residual also needs updated
  // ghost DoFs of the previous solution, so make sure you call
  // Update() on prev_sol_ before calling this function
  local_asm.set_time_solution(&prev_sol_);
  local_asm.set_time_stepping_weights(alpha0_, alpha1_, alpha2_, alpha3_);

  global_asm_.assemble_vector(*space_, local_asm, *out);

  // correct BC -- set Dirichlet dofs to 0
  if (!dirichlet_dofs_.empty()) 
  {
    std::vector< Scalar > zeros(dirichlet_dofs_.size(), 0.);
    out->SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(zeros));
  }
}

void ChannelBenchmark::EvalGrad(const LAD::VectorType &in,
                                LAD::MatrixType *out) 
{
  bench_quantities_[5] += this->linear_solver_->iter();

  Timer assembly_timer;
  assembly_timer.start();

  compute_jacobian(in, out);

  assembly_timer.stop();
  bench_quantities_[4] += assembly_timer.get_duration();
}

void ChannelBenchmark::compute_jacobian(const LAD::VectorType &in,
                                        LAD::MatrixType *out) 
{
  CONSOLE_OUTPUT(3, "Compute jacobian");

  {
    TimingScope tscope("Compute Jacobian");

    // the computation of the Jacobian needs updated ghost DoFs,
    // so make sure you call Update() on the input vector
    // before calling this function!
  InstationaryFlowAssembler local_asm(nu_, rho_);

  local_asm.set_newton_solution(&in);
  // the computation of the Jacobian in the instationary case also
  // also needs updated ghost DoFs of the previous solution, so
  // make sure you call Update() on prev_sol_ before
  // calling this function
  local_asm.set_time_solution(&prev_sol_);
  local_asm.set_time_stepping_weights(alpha0_, alpha1_, alpha2_, alpha3_);

  global_asm_.should_reset_assembly_target(true);
  global_asm_.assemble_matrix(*space_, local_asm, *out);

  // correct BC -- set Dirichlet rows to identity
  if (!dirichlet_dofs_.empty()) 
    {
      out->diagonalize_rows(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), 1.);
    }
  }

  this->update_preconditioner(in, out);
}

//////////////// Pressure Filtering ////////////////

struct PressureIntegral : private AssemblyAssistant< DIM, Scalar > 
{
  PressureIntegral(const CoupledVector< Scalar > &sol) : sol_(sol) {}

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, Scalar &pressure) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);
    evaluate_fe_function(sol_, DIM, p_);
    const int num_q = num_quadrature_points();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) {
      const Scalar wq = w(q);
      const Scalar dJ = std::abs(detJ(q));

      pressure += wq * p_[q] * dJ;
    }
  }

  const CoupledVector< Scalar > &sol_;
  FunctionValues< Scalar > p_;
};

struct VolumeIntegral : private AssemblyAssistant< DIM, Scalar > 
{
  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, Scalar &vol) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);
    const int num_q = num_quadrature_points();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) {
      const Scalar wq = w(q);
      const Scalar dJ = std::abs(detJ(q));
      vol += wq * dJ;
    }
  }
};

void ChannelBenchmark::ApplyFilter(LAD::VectorType &u) 
{
  if (!use_pressure_filter_)
  {
    return;
  }
  Scalar recv;
  u.Update();
  PressureIntegral int_p(u);
  Scalar total_pressure;
  global_asm_.integrate_scalar(*space_, int_p, total_pressure);

  MPI_Allreduce(&total_pressure, &recv, 1, MPI_DOUBLE, MPI_SUM, comm_);
  total_pressure = recv;

  Scalar integrated_vol;
  VolumeIntegral vol_int;
  global_asm_.integrate_scalar(*space_, vol_int, integrated_vol);

  MPI_Allreduce(&integrated_vol, &recv, 1, MPI_DOUBLE, MPI_SUM, comm_);
  integrated_vol = recv;

  const Scalar average_pressure = total_pressure / integrated_vol;

  LOG_INFO("pressure_filter",
           "Average pressure before filter = " << average_pressure);

  pressure_correction_.CloneFromWithoutContent(u);
  pressure_correction_.Zeros();

  // set value for pressure dofs to average pressure
  std::vector< int > cell_p_dofs;
  std::vector< int > local_p_dofs;
  for (EntityIterator it = mesh_->begin(DIM), end = mesh_->end(DIM); it != end; ++it) 
  {
    cell_p_dofs.clear();
    space_->get_dof_indices(space_->var_2_fe(DIM), it->index(), cell_p_dofs);
    for (int i = 0, sz = cell_p_dofs.size(); i < sz; ++i) 
    {
      if (space_->dof().is_dof_on_subdom(cell_p_dofs[i])) 
      {
        local_p_dofs.push_back(cell_p_dofs[i]);
      }
    }
  }

  std::sort(local_p_dofs.begin(), local_p_dofs.end());
  std::unique(local_p_dofs.begin(), local_p_dofs.end());

  // remove average pressure from solution
  std::vector< Scalar > p_correction_values(local_p_dofs.size());
  std::fill(p_correction_values.begin(), p_correction_values.end(), average_pressure);

  pressure_correction_.SetValues(vec2ptr(local_p_dofs), local_p_dofs.size(), vec2ptr(p_correction_values));

  u.Axpy(pressure_correction_, -1.);

  u.Update();
  PressureIntegral int_p_check(u);
  global_asm_.integrate_scalar(*space_, int_p_check, total_pressure);
  MPI_Allreduce(&total_pressure, &recv, 1, MPI_DOUBLE, MPI_SUM, comm_);
  total_pressure = recv;
  LOG_INFO("pressure_filter", "Average pressure after filter = "
                                  << total_pressure / integrated_vol);
}

class ForceIntegral : private AssemblyAssistant< DIM, Scalar > 
{
public:
  enum FORCE_TYPE { DRAG = 0, LIFT = 1 };

  ForceIntegral(Scalar nu, const CoupledVector< Scalar > *sol,
                const std::vector< int > &bdy_dofs, FORCE_TYPE type)
      : nu_(nu), x_var_(type == DRAG ? 0 : 1), bdy_dofs_(bdy_dofs), sol_(sol) 
      {
      } // rho assumed to be one

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, Scalar &val) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);

    std::vector< int > dofs;
    element.get_dof_indices(dofs);

    std::sort(dofs.begin(), dofs.end());

    std::vector< int > bdofs; // cylinder boundary dofs on the cell

    // find dofs on cell that also lie on cylinder boundary
    std::set_intersection(dofs.begin(), dofs.end(), bdy_dofs_.begin(),
                          bdy_dofs_.end(), std::back_inserter(bdofs));

    if (!bdofs.empty()) 
    {
      dofs.clear();
      element.get_dof_indices(dofs);

      // We compute function values here only, since otherwise they will not be
      // needed.
      recompute_function_values();

      for (std::vector< int >::const_iterator d = bdofs.begin(), d_end = bdofs.end(); d != d_end; ++d) 
      {
        // Find local dof number for *d
        std::vector< int >::iterator i_it = 
          std::find(dofs.begin() + dof_index(0, x_var_), dofs.begin() + dof_index(0, x_var_) + num_dofs(x_var_), *d);
        
        const int i = std::distance(dofs.begin() + dof_index(0, x_var_), i_it);
        assert(i >= 0);
        assert(i < num_dofs(x_var_));

        const int num_q = num_quadrature_points();

        for (int q = 0; q < num_q; ++q) {
          const Scalar wq = w(q);
          const Scalar dJ = std::abs(detJ(q));

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
  void recompute_function_values() 
  {
    for (int d = 0; d < DIM; ++d) 
    {
      u_[d].clear();
      grad_u_[d].clear();

      evaluate_fe_function(*sol_, d, u_[d]);
      evaluate_fe_function_gradients(*sol_, d, grad_u_[d]);
    }
    p_.clear();
    evaluate_fe_function(*sol_, DIM, p_);
  }

  Scalar nu_;
  int surface_material_num_, x_var_;
  FunctionValues< Scalar > u_[DIM], p_;
  FunctionValues< Vec< DIM, Scalar > > grad_u_[DIM];
  const std::vector< int > &bdy_dofs_;
  const CVector *sol_;
};

class DivergenceIntegral : private AssemblyAssistant< DIM, Scalar > 
{
public:

  DivergenceIntegral(const CoupledVector< Scalar > *sol)
      : sol_(sol) 
      {
      } // rho assumed to be one

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, Scalar &val) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);

    recompute_function_values();

    const int num_q = num_quadrature_points();

    val = 0.;
  
    for (int q = 0; q < num_q; ++q) 
    {
      const Scalar wq = w(q);
      const Scalar dJ = std::abs(detJ(q));

      Scalar div = 0.;
      for (size_t d=0; d<DIM; ++d)
      {
        div += grad_u_[d][q][d];
      }
    
      val += wq * div * div * dJ;
    }
  }

private:
  void recompute_function_values() 
  {
    for (int d = 0; d < DIM; ++d) 
    {
      grad_u_[d].clear();
      evaluate_fe_function_gradients(*sol_, d, grad_u_[d]);
    }
  }

  FunctionValues< Vec< DIM, Scalar > > grad_u_[DIM];
  const CoupledVector< Scalar > *sol_;
};

void ChannelBenchmark::compute_forces(Scalar &drag_force, Scalar &lift_force) 
{
  std::vector< int > dof_ids;
  std::vector< Scalar > dof_values; // dummy argument -- not used.

  const int cylinder_material = params_["Boundary"]["CylinderMaterial"].get< int >();
  Scalar local_force[2] = {0., 0.};

  find_cylinder_boundary_dofs(cylinder_material, 0, dof_ids);
  LOG_INFO("dfg_benchmark", "Total number of dof indices on cylinder bdy 0 = " << dof_ids.size());

  ForceIntegral int_F0(nu_, &sol_, dof_ids, ForceIntegral::DRAG);
  global_asm_.integrate_scalar(*space_, int_F0, local_force[0]);

  find_cylinder_boundary_dofs(cylinder_material, 1, dof_ids);
  LOG_INFO("dfg_benchmark", "Total number of dof indices on cylinder bdy 1 = " << dof_ids.size());

  ForceIntegral int_F1(nu_, &sol_, dof_ids, ForceIntegral::LIFT);
  global_asm_.integrate_scalar(*space_, int_F1, local_force[1]);

  LOG_INFO("dfg_benchmark",
           "On process " << rank() << ", Fd = " << local_force[0] << " and " << " Fl = " << local_force[1]);

  Scalar recv[2] = {0., 0.};
  MPI_Allreduce(&local_force[0], &recv[0], 2, MPI_DOUBLE, MPI_SUM, comm_);

  drag_force = recv[0];
  lift_force = recv[1];
}

void ChannelBenchmark::compute_divergence(Scalar &l2_div) 
{  
  Scalar local_div_squared = 0.;
  DivergenceIntegral int_div(&sol_);
  global_asm_.integrate_scalar(*space_, int_div, local_div_squared);

  Scalar global_div_squared = 0.;
  MPI_Allreduce(&local_div_squared, &global_div_squared, 1, MPI_DOUBLE, MPI_SUM, comm_);
  
  l2_div = std::sqrt(global_div_squared);
}

void ChannelBenchmark::compute_force_coefficients(Scalar drag_force,
                                                  Scalar lift_force,
                                                  Scalar &drag_coef,
                                                  Scalar &lift_coef) const 
{
#if DIM == 2
  drag_coef = 2. * drag_force / (U_mean_ * U_mean_ * diam_);
  lift_coef = 2. * lift_force / (U_mean_ * U_mean_ * diam_);
#elif DIM == 3
  drag_coef = 2. * drag_force / (U_mean_ * U_mean_ * diam_ * H_);
  lift_coef = 2. * lift_force / (U_mean_ * U_mean_ * diam_ * H_);
#endif
}

void ChannelBenchmark::compute_dfg_benchmark_quantities() {
  TimingScope tscope("Compute DFG Benchmark Quantities");

  // compute force coefficients
  // NB: rho is assumed to be one
  Scalar Fd, Fl, Cd, Cl;
  sol_.Update();
  compute_forces(Fd, Fl);
  compute_force_coefficients(Fd, Fl, Cd, Cl);

  if (rank() == MASTER_RANK) {
    LOG_INFO("dfg_benchmark", "Drag force = " << Fd);
    LOG_INFO("dfg_benchmark", "Drag coefficient = " << Cd);

    LOG_INFO("dfg_benchmark", "Lift force = " << Fl);
    LOG_INFO("dfg_benchmark", "Lift coefficient = " << Cl);

    results_table_.insert("Fd", Fd);
    results_table_.insert("Cd", Cd);
    results_table_.insert("Fl", Fl);
    results_table_.insert("Cl", Cl);
  }

  // compute pressure drop
  std::vector< Coord > p(2);

#if DIM == 2
  p[0].set(0, 0.15);
  p[0].set(1, 0.2 );
  p[1].set(0, 0.25);
  p[1].set(1, 0.2 );
#elif DIM == 3
  p[0].set(0, 0.448 );
  p[0].set(1, 0.2   );
  p[0].set(2, 0.205 );
  p[1].set(0, 0.5502);
  p[1].set(1, 0.2   );
  p[1].set(2, 0.205 );
#endif
  FeEvalGlobal<Scalar, DIM> fe_eval(*space_, sol_);
  
  std::vector< std::vector<Scalar> > fe_vals;
  
  fe_eval.evaluate (p, fe_vals); 
  
  if (rank_ == MASTER_RANK) {
    LOG_INFO("dfg_benchmark",
             "Pressure difference delta-p = " << fe_vals[0][DIM] - fe_vals[1][DIM]);
    results_table_.insert("delta-P", fe_vals[0][DIM] - fe_vals[1][DIM]);
  }
}

void ChannelBenchmark::find_cylinder_boundary_dofs(int cylinder_mat_num, size_t fe_ind, std::vector< int > &bdy_dofs) 
{
  bdy_dofs.clear();

  const Mesh &mesh = space_->mesh();
  const TDim tdim = mesh.tdim();

  MeshPtr boundary_mesh = mesh.extract_boundary_mesh();
  const bool is_sequential = (num_partitions() == 1);
  if (!is_sequential) 
  {
    assert(mesh.has_attribute("_sub_domain_", tdim));
  }

  std::vector< doffem::DofID > dofs_on_face;
  for (EntityIterator it_boundary = boundary_mesh->begin(tdim - 1);
       it_boundary != boundary_mesh->end(tdim - 1); ++it_boundary) 
  {
    // get id of boundary face
    const Id boundary_id = it_boundary->id();

    // check if the boundary face exists and get the location
    // where the entity number should be stored
    int face_number;
    const bool check = mesh.find_entity(tdim - 1, boundary_id, &face_number);
    assert(check);

    // Get the face to be able to access to the data associated with the face
    Entity face = mesh.get_entity(tdim - 1, face_number);
    if (face.get_material_number() != cylinder_mat_num) {
      continue;
    }

    IncidentEntityIterator cell = face.begin_incident(tdim);

    // loop over all faces of the cell to get the local face index for
    // identifying the dofs
    int local_face_number = 0;
    for (IncidentEntityIterator global_face = cell->begin_incident(tdim - 1);
         global_face != cell->end_incident(tdim - 1); ++global_face) 
    {
      // if the global face id equals the boundary id the local face index is
      // found
      if (global_face->id() == boundary_id) 
      {
        break;
      } 
      else {
        local_face_number++;
      }
    }

    dofs_on_face.clear();
    space_->dof().get_dofs_on_subentity(fe_ind, cell->index(), tdim - 1, local_face_number, dofs_on_face);
    bdy_dofs.insert(bdy_dofs.end(), dofs_on_face.begin(), dofs_on_face.end());
  }

  std::sort(bdy_dofs.begin(), bdy_dofs.end());
  std::vector< int >::iterator new_end = std::unique(bdy_dofs.begin(), bdy_dofs.end());
  bdy_dofs.resize(std::distance(bdy_dofs.begin(), new_end));
}

void ChannelBenchmark::setup_linear_algebra() 
{
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

void ChannelBenchmark::prepare_linear_solver() 
{   
  PropertyTree outer_param = params_["LinearSolver"]["PrimalSolver"];
  PropertyTree precond_param = params_["LinearSolver"]["Preconditioner"];
  PropertyTree gmg_param = params_["LinearSolver"]["Preconditioner"]["GMG"];
  PropertyTree locsolver_param = params_["LinearSolver"]["LocalSolver"];
  
  LinearSolver<LAD>* solver = nullptr;
  CMatrix* matrix = nullptr;
  
  this->solver_update_newton_step_ = outer_param["UpdateFreqNewton"].get<int>();
  this->solver_update_time_step_ = outer_param["UpdateFreqTime"].get<int>();
  
  std::string precond_type;
  
  LOG_INFO("Prepare", "primal linear solver " );
  this->prepare_outer_solver(this->linear_solver_, outer_param);
    
  solver = this->linear_solver_;
  precond_type = outer_param["PrecondType"].get<std::string>();
  matrix = &this->matrix_;
  
  if (precond_type == "BlockJacobiStd")
  {
    std::string locsolver_type = precond_param["BlockJacobi"]["LocalSolverType"].get<std::string>();
    this->bJacobiStd_.Init(locsolver_type, locsolver_param);
    this->precond_ = &(this->bJacobiStd_);
  }
  else if (precond_type == "BlockJacobiExt")
  {
    std::string locsolver_type = precond_param["BlockJacobi"]["LocalSolverType"].get<std::string>();
    this->bJacobiExt_.Init(locsolver_type, locsolver_param);
    this->precond_ = &(this->bJacobiExt_);
  }
  else if (precond_type == "GMG")
  {
    this->prepare_gmg(gmg_param, locsolver_param);
    this->precond_ = &(this->gmg_);
  }  
  else if (precond_type == "Vanka")
  {
    this->vanka_.InitParameter(
        *space_,
         locsolver_param["Vanka"]["Damping"].get< Scalar >(),
         locsolver_param["Vanka"]["Iterations"].get< int >(),
         locsolver_param["Vanka"]["BlockCells"].get< bool >());
    this->precond_ = &(this->vanka_);
  }
  else
  {
    assert (false);
  }
  
  assert (matrix != nullptr);
  assert (this->precond_ != nullptr);
  assert (solver != nullptr);
  
  this->precond_->SetupOperator(*matrix);
  solver->SetupOperator(*matrix);
  solver->SetupPreconditioner(*this->precond_);
}

void ChannelBenchmark::prepare_outer_solver(LinearSolver<LAD>*& solver, 
                                            const PropertyTree &params) 
{ 
  std::string main_solver_type = params["Type"].get<std::string> ();
  if (solver != nullptr)
  {
    delete solver;
    solver = nullptr;
  }
  if (main_solver_type == "FGMRES")
  {
    FGMRES<LAD>* tmp = new FGMRES<LAD>();
    setup_FGMRES_solver<LAD, LAD> (*tmp, params, this);
    solver = tmp;
    solver->SetName ( "OUTER_FGMRES" );
  }
  else if (main_solver_type == "GMRES")
  {
    GMRES<LAD>* tmp = new GMRES<LAD>();
    setup_GMRES_solver<LAD, LAD> (*tmp, params, this);
    solver = tmp;
    solver->SetName ( "OUTER_GMRES" );
  }
  else
  {
    assert (false);
  }
}
                                                            
void ChannelBenchmark::prepare_gmg(const PropertyTree &gmg_param,
                                   const PropertyTree &locsolver_param) 
{
  PropertyTree coarse_param = gmg_param["CoarseSolver"];
  PropertyTree smoother_param = gmg_param["KrylovSmoother"];
  
  std::string smoother_type = gmg_param["SmootherType"].get<std::string>();
  
  // spaces
  this->gmg_nb_lvl_ = gmg_param["NumLevel"].get<int>();
  assert (this->gmg_nb_lvl_ <= this->all_spaces_.size());
  
  this->gmg_spaces_.clear();
  this->gmg_spaces_.resize(this->gmg_nb_lvl_);
  for (int l=0; l<this->gmg_nb_lvl_; ++l)
  {
    assert (this->all_spaces_[this->nb_lvl_ - this->gmg_nb_lvl_ + l] != 0);
    this->gmg_spaces_[l] = this->all_spaces_[this->nb_lvl_ - this->gmg_nb_lvl_ + l].get();
  }
  
  // coarse solver
  if (this->gmg_coarse_ != nullptr)
  {
    if (this->gmg_coarse_->GetPreconditioner() != nullptr)
    {
      delete this->gmg_coarse_->GetPreconditioner();
    }
    delete this->gmg_coarse_;
    this->gmg_coarse_ = nullptr;
  }
  
  Preconditioner< LAD>* coarse_precond;

  prepare_krylov_solver(this->gmg_coarse_,
                        coarse_precond, 
                        coarse_param,
                        locsolver_param,
                        this->gmg_spaces_[0],
                        this); 

  // smoothers
  for (int l=0; l<this->gmg_smoothers_.size(); ++l)
  {
    if (this->gmg_smoothers_[l] != nullptr)
    {
      LinearSolver<LAD>* cast_sm = dynamic_cast<LinearSolver<LAD>* >(this->gmg_smoothers_[l]);
      if (cast_sm != 0)
      {
        if (cast_sm->GetPreconditioner() != nullptr)
        {
          delete cast_sm->GetPreconditioner();
        }
      }
      delete this->gmg_smoothers_[l];
      this->gmg_smoothers_[l] = nullptr;
    }
  }
  
  this->gmg_smoothers_.clear();
  this->gmg_smoothers_.resize(this->gmg_nb_lvl_-1, nullptr);
    
  for (int l=0; l<this->gmg_nb_lvl_-1; ++l)
  {
    if (smoother_type == "FSAI" || 
        smoother_type == "HiflowILU" || 
        smoother_type == "SOR" || 
        smoother_type == "SSOR" || 
        smoother_type == "Jacobi")
    {
      PreconditionerBlockJacobiStand< LAD >* sm = new PreconditionerBlockJacobiStand< LAD >();
      sm->Init(smoother_type, locsolver_param);
      this->gmg_smoothers_[l] = sm;
    }
    else if (smoother_type == "ILUPP" || 
             smoother_type == "MklILU" || 
             smoother_type == "MklLU" || 
             smoother_type == "UmfpackLU")
    {
      PreconditionerBlockJacobiExt< LAD >* sm = new PreconditionerBlockJacobiExt< LAD >();
      sm->Init(smoother_type, locsolver_param);
      this->gmg_smoothers_[l] = sm;
    }
    else if (smoother_type == "Vanka")
    {
      PreconditionerVanka< LAD, DIM >* sm = new PreconditionerVanka< LAD, DIM >();
      sm->InitParameter(
        *this->gmg_spaces_[l+1],
         locsolver_param["Vanka"]["Damping"].get< Scalar >(),
         locsolver_param["Vanka"]["Iterations"].get< int >(),
         locsolver_param["Vanka"]["BlockCells"].get< bool >());
      this->gmg_smoothers_[l] = sm;
    }
    else if (smoother_type == "Krylov")
    {
      LinearSolver<LAD>* sm;
      Preconditioner<LAD>* precond;
      prepare_krylov_solver(sm, precond, smoother_param, locsolver_param, this->gmg_spaces_[l+1], this);
      this->gmg_smoothers_[l] = sm;
    }
  }
  
  this->gmg_.set_application(this);
  this->gmg_.set_spaces(this->gmg_spaces_);
  this->gmg_.InitControl(gmg_param["Iterations"].get<int>(), 1e-20, 1e-10, 1e6);
  this->gmg_.InitParameter(gmg_param["CycleType"].get<std::string>(), 
                           gmg_param["NestedIteration"].get<bool>(), 
                           gmg_param["PreSmoothingSteps"].get<int>(), 
                           gmg_param["PostSmoothingSteps"].get<int>(), 
                           gmg_param["SmootherRelax"].get<Scalar>(), 
                           false, 
                           gmg_param["TransposedP"].get<bool>(),
                           false,
                           true);
  this->gmg_.SetPrintLevel(gmg_param["PrintLevel"].get<int>());
  this->gmg_.set_coarse_solver(this->gmg_coarse_);
  this->gmg_.set_smoothers(this->gmg_smoothers_);
  
  // vectors of previous solutions
  this->gmg_.setup_gmg_vectors(this->gmg_newton_);
  this->gmg_.setup_gmg_vectors(this->gmg_solP_);
  this->gmg_.setup_gmg_vectors(this->gmg_solP_prev_);
  
  delete this->gmg_newton_[this->gmg_nb_lvl_-1];
  delete this->gmg_solP_[this->gmg_nb_lvl_-1];
  delete this->gmg_solP_prev_[this->gmg_nb_lvl_-1];
  
  this->gmg_newton_[this->gmg_nb_lvl_-1] = &this->sol_;
  this->gmg_solP_[this->gmg_nb_lvl_-1] = &this->sol_;
  this->gmg_solP_prev_[this->gmg_nb_lvl_-1] = &this->prev_sol_;


}

void ChannelBenchmark::update_preconditioner(const CVector& u, CMatrix* DF)
{
  this->precond_->SetupOperator(*DF);
  this->precond_->Build(&this->res_, &this->sol_);
}


/// ***************************************************************************
/// GMG
/// ***************************************************************************
void ChannelBenchmark::prepare_operator(int level,
                                        const VectorSpace< Scalar, DIM >* space, 
                                        CMatrix* matrix) const
{
  assert (matrix != nullptr);
  assert (space != nullptr);

  std::vector< std::vector< bool > > coupling_vars;

  coupling_vars.resize(DIM + 1);
  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM + 1; ++j) {
      coupling_vars[i].push_back(true);
    }
  }
  for (int i = 0; i < DIM; ++i) {
    coupling_vars[DIM].push_back(true);
  }
  coupling_vars[DIM].push_back(false);
  
  SparsityStructure sparsity;
  compute_sparsity_structure(*space, sparsity, coupling_vars, false);
 
  IMPLEMENTATION impl = NAIVE;
  
  matrix->Init(comm_, space->la_couplings(), CPU, impl, CSR);
  
  // Initialize structure of LA objects.
  matrix->InitStructure(sparsity);

  // Zero all linear algebra objects.
  matrix->Zeros();
}

void ChannelBenchmark::prepare_rhs(int level, 
                                   const VectorSpace< Scalar, DIM >* space, 
                                   CVector* vector) const 
{
  assert (space != nullptr);
  assert (vector != nullptr);

  vector->Init(communicator(), space->la_couplings(), la_platform(), la_implementation());
  vector->Zeros();
}

void ChannelBenchmark::assemble_operator(int level, 
                                         const VectorSpace< Scalar, DIM >* space, 
                                         const std::vector<int>& dirichlet_dofs,
                                         const std::vector<Scalar>& dirichlet_vals,
                                         CMatrix* matrix)
{
  assert (space != nullptr);
  assert (matrix != nullptr);
  
  // TODO: set local assembler to preconditioning mode
  // TODO: dual assembler
  
  if (level < this->gmg_nb_lvl_-1)
  {
    this->gmg_.Restrict(level, this->gmg_newton_[level+1], this->gmg_newton_[level]);
    this->gmg_.Restrict(level, this->gmg_solP_prev_[level+1], this->gmg_solP_prev_[level]);
  }
  else
  {
    assert (level == this->gmg_nb_lvl_-1);
    this->gmg_newton_[level] = this->newton_.get_iterate();
    this->gmg_solP_prev_[level] = &this->prev_sol_;
  }
  
  InstationaryFlowAssembler local_asm(nu_, rho_);

  local_asm.set_newton_solution(this->gmg_newton_[level]);
  // the computation of the Jacobian in the instationary case also
  // also needs updated ghost DoFs of the previous solution, so
  // make sure you call Update() on prev_sol_ before
  // calling this function
  local_asm.set_time_solution(this->gmg_solP_prev_[level]);
  local_asm.set_time_stepping_weights(alpha0_, alpha1_, alpha2_, alpha3_);

  global_asm_.should_reset_assembly_target(true);
  global_asm_.assemble_matrix(*space, local_asm, *matrix);
  

  if (dirichlet_dofs.size() > 0)
  {
    matrix->diagonalize_rows(vec2ptr(dirichlet_dofs), dirichlet_dofs.size(), 1.0);
  }
}

void ChannelBenchmark::assemble_rhs(int level,
                                    const VectorSpace< Scalar, DIM >* space,
                                    const std::vector<int>& dirichlet_dofs,
                                    const std::vector<Scalar>& dirichlet_vals,
                                    CVector* vector) 
{
 
}

void ChannelBenchmark::prepare_fixed_dofs_gmg(int gmg_level,
                                              const VectorSpace< Scalar, DIM >* space, 
                                              std::vector<int>& fixed_dofs,
                                              std::vector<Scalar>& fixed_vals) const
{
  fixed_dofs.clear();
  fixed_vals.clear();

  prepare_bc(space, fixed_dofs, fixed_vals);
}

void ChannelBenchmark::get_linear_solver_statistics ( bool erase )
{
  LOG_INFO("get", "Solver statistics " );    

  // complete solver
  int num_build_full = 0;
  int num_solve_full = 0;
  int iter_full = 0;
  double time_build_full = 0.;
  double time_solve_full = 0.;

  this->linear_solver_->GetStatistics ( iter_full, num_build_full, num_solve_full, time_build_full, time_solve_full, erase );
        

  LOG_INFO("linear solver", "build: " << std::setw(3) << num_build_full
                      << ", t build: " << std::setw(9) <<  time_build_full
                      << ", # solve: " << std::setw(4) <<  num_solve_full
                      << ", t solve: " << std::setw(9) <<  time_solve_full
                      << ", iter: " << std::setw(5) << iter_full );


  if (this->use_gmg_)
  {
    bool erase = true;
    int gmg_iter = 0;
    int gmg_num_build = 0;
    int gmg_num_solve = 0;
    double gmg_time_build = 0.;
    double gmg_time_solve = 0.;
    double time_res = 0.;
    double time_pro = 0.;

    this->gmg_.GetStatistics(gmg_iter, gmg_num_build, gmg_num_solve,
                             gmg_time_build, gmg_time_solve, 
                             time_res, time_pro, erase);
                             
    int coarse_iter = 0;
    int coarse_num_build = 0;
    int coarse_num_solve = 0;
    double coarse_time_build = 0.;
    double coarse_time_solve = 0.;
    
    this->gmg_coarse_->GetStatistics(coarse_iter, coarse_num_build, coarse_num_solve,
                                coarse_time_build, coarse_time_solve, erase);
                             
    int sm_iter = 0;
    int sm_num_build = 0;
    int sm_num_solve = 0;
    double sm_time_build = 0.;
    double sm_time_solve = 0.;
    
    for (int i=0; i<this->gmg_nb_lvl_-1; ++i)
    {
      int i_iter = 0;
      int i_num_build = 0;
      int i_num_solve = 0;
      double i_time_build = 0.;
      double i_time_solve = 0.;
    
      this->gmg_smoothers_[i]->GetStatistics(i_iter, i_num_build, i_num_solve,
                                  i_time_build, i_time_solve, erase);
      sm_iter += i_iter;
      sm_num_build += i_num_build;
      sm_num_solve += i_num_solve;
      sm_time_build += i_time_build;
      sm_time_solve += i_time_solve;
    }
    
    LOG_INFO("GMG", "complete build time " << gmg_time_build);
    LOG_INFO("GMG", "coarse   build time " << coarse_time_build);
    LOG_INFO("GMG", "smoother build time " << sm_time_build);
    LOG_INFO("GMG","");
    LOG_INFO("GMG", "complete solve time " << gmg_time_solve);
    LOG_INFO("GMG", "coarse   solve time " << coarse_time_solve);
    LOG_INFO("GMG", "smoother solve time " << sm_time_solve);
    LOG_INFO("GMG","");
    LOG_INFO("GMG", "complete iter       " << gmg_iter);
    LOG_INFO("GMG", "coarse   iter       " << coarse_iter);
    LOG_INFO("GMG", "smoother iter       " << sm_iter);
    LOG_INFO("GMG","");
    LOG_INFO("GMG", "complete #build     " << gmg_num_build);
    LOG_INFO("GMG", "coarse   #build     " << coarse_num_build);
    LOG_INFO("GMG", "smoother #build     " << sm_num_build);
    LOG_INFO("GMG","");
    LOG_INFO("GMG", "complete #solve     " << gmg_num_solve);
    LOG_INFO("GMG", "coarse   #solve     " << coarse_num_solve);
    LOG_INFO("GMG", "smoother #solve     " << sm_num_solve);
  }
}
