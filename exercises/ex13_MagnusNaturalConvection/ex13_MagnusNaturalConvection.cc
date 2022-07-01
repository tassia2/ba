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

/// \author Philipp Gerstner

#include "ex13_MagnusNaturalConvection.h"

#define ILUPP

#ifdef WITH_GPERF
#include "profiler.h"
#endif

static const char *PARAM_FILENAME = "ex13_MagnusNaturalConvection.xml";
static const char *DATADIR = "./";

// *************************************************
// TODO
// put your name, student id and degree course  here 

static const std::string student_name = "your_name";
static const int student_id = 999999999;
static const std::string student_course = "your_course";

// *************************************************

// Main application class ///////////////////////////////////

class ExamCFD : public NonlinearProblem< LAD >
{
public:
  ExamCFD(const std::string &param_filename,
                  const std::string &path_mesh)
      : path_mesh(path_mesh), comm_(MPI_COMM_WORLD), rank_(-1),
        num_partitions_(-1),
        params_(param_filename, MASTER_RANK, MPI_COMM_WORLD), 
        is_done_(false), refinement_level_(0) 
  {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &num_partitions_);
    this->parcom_ = std::shared_ptr<ParCom>(new ParCom(this->comm_));
    
    // Setup Parallel Output / Logging
    if (rank_ == 0)
    {
      INFO = true;
    }
    else
    {
      INFO = false;
    }
  }

  // Main algorithm

  void run() {

#ifdef WITH_GPERF
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
  std::string str_time(buffer);
 
  //std::string prof_name = "PoissonProfile_" + str_time + ".log";
  std::string prof_name = "PoissonProfile.log";
  ProfilerStart(prof_name.c_str());
#endif

    // Construct / read in the initial mesh.
    build_initial_mesh();
    // Main adaptation loop.
    while (!is_done_) 
    {
      Timer timer;
      timer.start();
      
      // Initialize space and linear algebra.
      LOG_INFO ("do", "Prepare System");
        
      prepare_system();
     
      timer.stop();
      LOG_INFO("duration", timer.get_duration()); 
      timer.reset();
      timer.start();
      
      
      // Compute the stiffness matrix and right-hand side.
      LOG_INFO ("run", "Time Loop ");
     
      time_loop();
     
      timer.stop();
      LOG_INFO("duration", timer.get_duration()); 
      timer.reset();
      timer.start();

      // refine mesh
      LOG_INFO ("do", "Adapt Mesh");
      
      adapt();
      
      timer.stop();
      LOG_INFO("duration", timer.get_duration()); 
      timer.reset();
      timer.start();
    }

#ifdef WITH_GPERF
  ProfilerStop();
#endif
  }

  ~ExamCFD() {
  }

private:
  // Member functions

  // Read and distribute mesh.
  std::string path_mesh;
  void build_initial_mesh();
  
  // Setup space, linear algebra, and compute Dirichlet values.
  void prepare_system();

  // setup dirichlet BC
  void compute_bc(DataType time);
  
  // assembler routines for Jacobian and residual in Newton's method
  void eval_LHS_and_RHS(MatrixType *lhs, 
                        VectorType *rhs,
                        DataType dt, 
                        int bdf_type); 

  void solve_linear_system(VectorType &rhs, VectorType* sol); 
    
  // time stepping
  void time_loop();
  
  // Visualize the results.
  void visualize(int time_step);
  
  // Adapt the space (mesh and/or degree).
  void adapt();

  // apply zero average for pressure
  void ApplyFilter(VectorType &u);
  
  // post processing
  DataType compute_average_pressure(VectorType &u);
  DataType compute_volume_int();
  DataType compute_pressure_int(VectorType &u);
  DataType compute_L2_divergence(VectorType &u);
  DataType compute_dissipation_energy(VectorType &u);
  DataType compute_kinetic_energy(VectorType &u);

  void evaluate_solution(VectorType &u, std::vector<DataType>& cur_pp_vals);
  void compute_forces(VectorType &u,
                      DataType &force_x, 
                      DataType &force_y);


  // member variables

  // MPI communicator.
  MPI_Comm comm_;
  
  // Local process rank and number of processes.
  int rank_, num_partitions_;

  // Parameter data read in from file.
  PropertyTree params_;

  // Local mesh and mesh on master process.
  MeshPtr mesh_, mesh_without_ghost_, master_mesh_;

  // Solution space.
  VectorSpace< DataType, DIM > space_;

  // Vectors for solution and load vector.
  VectorType rhs_, res_, sol_, sol_prev_, sol_pprev_;

  // System matrix.
  MatrixType matrix_;

  // Global assembler.
  DGGlobalAssembler< DataType, DIM > global_asm_;
 
  // linear solver
  LinearSolver< LAD > * solver_;
  
  // preconditioner
  PreconditionerBlockJacobiExt< LAD > precond_ext_;
    
  // Flag for stopping adaptive loop.
  bool is_done_;
  
  // Current refinement level.
  int refinement_level_;

  // Dof id:s for Dirichlet boundary conditions.
  std::vector< int > dirichlet_dofs_;
  
  // Dof values for Dirichlet boundary conditions.
  std::vector< DataType > dirichlet_values_;
  
  // wrapper for frequently used MPI routines
  std::shared_ptr<ParCom> parcom_;
  
}; // end class ExamCFD

// Program entry point

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

#ifdef WITH_GPERF
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
  std::string str_time(buffer);
  
  std::string prof_name = "StokesProfile_" + str_time + ".log";
  ProfilerStart(prof_name.c_str());
#endif

  // set default parameter file
  std::string param_filename(PARAM_FILENAME);
  std::string path_mesh;
  // if set take parameter file specified on console
  if (argc > 1) {
    param_filename = std::string(argv[1]);
  }
  // if set take mesh following path specified on console
  if (argc > 2) {
    path_mesh = std::string(argv[2]);
  }
  try {
    // Create log files for INFO and DEBUG output
    //std::ofstream info_log("poisson_tutorial_info_log");
    //std::ofstream warning_log("poisson_tutorial_warning_log");
    //std::ofstream debug_log("poisson_tutorial_debug_log");
    //std::ofstream error_log("poisson_tutorial_error_log");
        
    LogKeeper::get_log("info").set_target(&(std::cout));
    LogKeeper::get_log("debug").set_target(&(std::cout));
    LogKeeper::get_log("error").set_target(&(std::cout));
    LogKeeper::get_log("warning").set_target(&(std::cout));

    // Create application object and run it
    ExamCFD app(param_filename, path_mesh);
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

//////////////// ExamCFD implementation //////////////

void ExamCFD::build_initial_mesh() 
{
  mesh::IMPL impl = mesh::IMPL_DBVIEW;

  // Read in the mesh on the master process. The mesh is chosen according to the
  // DIM of the problem.
  if (rank_ == MASTER_RANK) 
  {
    std::string mesh_name;
    mesh_name = params_["Mesh"]["Filename"].get< std::string >("unit_square.inp");

    std::string mesh_filename;
    if (path_mesh.empty()) 
    {
      mesh_filename = std::string(DATADIR) + mesh_name;
    } 
    else 
    {
      mesh_filename = path_mesh + mesh_name;
    }

    std::vector< MasterSlave > period(0, MasterSlave(0., 0., 0., 0));
    // read the mesh
    master_mesh_ = read_mesh_from_file(mesh_filename, DIM, DIM, 0, impl, period);

    // Refine the mesh until the initial refinement level is reached.
    const int initial_ref_lvl = params_["Mesh"]["InitialRefLevel"].get< int >(3);

    if (initial_ref_lvl > 0) 
    {
      master_mesh_ = master_mesh_->refine_uniform_seq(initial_ref_lvl);
    }
    refinement_level_ = initial_ref_lvl;
  }

  MPI_Bcast(&refinement_level_, 1, MPI_INT, MASTER_RANK, comm_);

  // partition the mesh and distribute the subdomains across all processes
  int uniform_ref_steps;
  mesh_without_ghost_ = partition_and_distribute(master_mesh_, MASTER_RANK, comm_, uniform_ref_steps, impl);

  assert(mesh_without_ghost_ != 0);
  
  // compute ghost cells
  SharedVertexTable shared_verts;
  mesh_ = compute_ghost_cells(*mesh_without_ghost_, comm_, shared_verts, impl);

}

// refine mesh
void ExamCFD::adapt() 
{
  if (rank_ == MASTER_RANK) 
  {
    const int final_ref_level = params_["Mesh"]["FinalRefLevel"].get< int >(6);
    if (refinement_level_ >= final_ref_level) 
    {
      is_done_ = true;
    } 
    else 
    {
      master_mesh_ = master_mesh_->refine();
      ++refinement_level_;
    }
  }
  
  // Broadcast information from master to slaves.
  MPI_Bcast(&refinement_level_, 1, MPI_INT, MASTER_RANK, comm_);
  MPI_Bcast(&is_done_, 1, MPI_CHAR, MASTER_RANK, comm_);

  if (!is_done_) 
  {
    // Distribute the new mesh.
    int uniform_ref_steps;
    MeshPtr local_mesh = partition_and_distribute(master_mesh_, MASTER_RANK, comm_, uniform_ref_steps);
    assert(local_mesh != 0);
    SharedVertexTable shared_verts;
    mesh_ = compute_ghost_cells(*local_mesh, comm_, shared_verts);
  }
}

// setup the FE space and linear algebra objects
void ExamCFD::prepare_system() 
{
  // Assign degrees to each element.
  const int nb_fe_var = 3;
  
  const int velocity_fe_degree = params_["FESpace"]["VelocityDegree"].get< int >(2);
  const int pressure_fe_degree = params_["FESpace"]["PressureDegree"].get< int >(1);
  const int temp_fe_degree = params_["FESpace"]["TemperatureDegree"].get< int >(1);
  
  std::vector< int > fe_params(nb_fe_var);
  fe_params[0] = velocity_fe_degree;
  fe_params[1] = pressure_fe_degree;
  fe_params[2] = temp_fe_degree;
  
  // assign types of FE 
  std::vector< FEType > fe_ansatz (nb_fe_var);
  fe_ansatz[0] = FEType::LAGRANGE_VECTOR;
  fe_ansatz[1] = FEType::LAGRANGE;
  fe_ansatz[2] = FEType::LAGRANGE;
  
  // both variables should be continuous
  std::vector< bool > is_cg (nb_fe_var, true);
  
  // Initialize the VectorSpace object.
  space_.Init(*mesh_, fe_ansatz, is_cg, fe_params, DOF_ORDERING::HIFLOW_CLASSIC);
  
  LOG_INFO("nb nonlin trafos", space_.fe_manager().nb_nonlinear_trafos());
  
  // Compute the matrix sparsity structure
  SparsityStructure sparsity;
  compute_sparsity_structure(space_, sparsity);

  // initialize matrix object
  matrix_.Init(comm_, space_.la_couplings());
  matrix_.InitStructure(sparsity);
  matrix_.Zeros();
  
  // initialize vector objects for solution coefficitons and 
  // right hand side
  res_.Init(comm_, space_.la_couplings());
  rhs_.Init(comm_, space_.la_couplings());
  sol_.Init(comm_, space_.la_couplings());
  sol_prev_.Init(comm_, space_.la_couplings());
  sol_pprev_.Init(comm_, space_.la_couplings());

  res_.Zeros();  
  rhs_.Zeros();
  sol_.Zeros();
  sol_pprev_.Zeros();

  // setup  linear solver
  LinearSolverFactory< LAD > SolFact;
  solver_ = SolFact.Get(params_["LinearSolver"]["Name"].get< std::string >("GMRES"))->params(params_["LinearSolver"]);
  solver_->SetupOperator(matrix_);

#ifdef ILUPP
  // use incomplete LU factorization as preconditioner
  precond_ext_.Init_ILU_pp();
  solver_->SetupPreconditioner(precond_ext_);
#else
  precond_ext_.Init_LU_mkl();
  solver_->SetupPreconditioner(precond_ext_);
#endif
}

// compute the Dirichlet BC at given time
void ExamCFD::compute_bc(DataType time)
{
  // Compute Dirichlet BC dofs and values using known exact solution.
  dirichlet_dofs_.clear();
  dirichlet_values_.clear();
                     
  VelocityDirichletBC bc_dirichlet_v(time,
                                   params_["BoundaryConditions"]["MaterialNumbers"]["Foil"].get< int >(),
                                   params_["BoundaryConditions"]["MaterialNumbers"]["Inflow"].get< int >(),
                                   params_["BoundaryConditions"]["MaterialNumbers"]["Outflow"].get< int >(),
                                   params_["BoundaryConditions"]["MaterialNumbers"]["SlipX"].get< int >(),
                                   params_["BoundaryConditions"]["InflowVelocityX"].get< DataType >(),
                                   params_["BoundaryConditions"]["InflowRamp"].get< DataType >(),
                                   params_["BoundaryConditions"]["RevolutionsPerSecond"].get< DataType >(),
                                   params_["BoundaryConditions"]["CenterX"].get< DataType >(),
                                   params_["BoundaryConditions"]["CenterY"].get< DataType >());
                     
  TemperatureDirichletBC bc_dirichlet_t(time,
                                      params_["BoundaryConditions"]["MaterialNumbers"]["Foil"].get< int >(),
                                      params_["BoundaryConditions"]["MaterialNumbers"]["Inflow"].get< int >(),
                                      params_["BoundaryConditions"]["MaterialNumbers"]["Outflow"].get< int >(),
                                      params_["BoundaryConditions"]["MaterialNumbers"]["SlipX"].get< int >(),
                                      params_["BoundaryConditions"]["InflowTemperature"].get< DataType >(),
                                      params_["BoundaryConditions"]["FoilTemperature"].get< DataType >());

  std::vector<int> dirichlet_dofs_t, dirichlet_dofs_v;
  std::vector<DataType> dirichlet_values_t, dirichlet_values_v;

  compute_dirichlet_dofs_and_values(bc_dirichlet_v, space_, 0, dirichlet_dofs_v, dirichlet_values_v);
  compute_dirichlet_dofs_and_values(bc_dirichlet_t, space_, 2, dirichlet_dofs_t, dirichlet_values_t);

  for (int i=0; i!= dirichlet_dofs_v.size(); ++i)
  {
    dirichlet_dofs_.push_back(dirichlet_dofs_v[i]);
    dirichlet_values_.push_back(dirichlet_values_v[i]);
  }
  for (int i=0; i!= dirichlet_dofs_t.size(); ++i)
  {
    dirichlet_dofs_.push_back(dirichlet_dofs_t[i]);
    dirichlet_values_.push_back(dirichlet_values_t[i]);
  }
}

// assemble the system matrix and rhs vector
// for a given time step size dt and bdf_type
void ExamCFD::eval_LHS_and_RHS(MatrixType *lhs, 
                               VectorType *rhs,
                               DataType dt, 
                               int bdf_type) 
{  
  // pass parameters to local assembler
  LocalFlowAssembler local_asm;
  local_asm.set_parameters(dt,
                           params_["Equation"]["nu"].get< DataType >(), 
                           params_["Equation"]["Cv"].get< DataType >(),
                           params_["Equation"]["Cp"].get< DataType >(),
                           params_["Equation"]["kappa"].get< DataType >(),
                           params_["Equation"]["gamma"].get< DataType >(), 
                           bdf_type, 
                           params_["Equation"]["WithConvection"].get< bool >(),
                           params_["Equation"]["WithStabilization"].get< bool >());

  // pass current Newton iterate to local assembler
  local_asm.set_previous_solution(&sol_prev_);
  local_asm.set_previous_previous_solution(&sol_pprev_);

  // call assemble routine
  this->global_asm_.should_reset_assembly_target(true);
  this->global_asm_.assemble_matrix(this->space_, local_asm, *lhs);
  this->global_asm_.assemble_vector(this->space_, local_asm, *rhs);

  // Correct Dirichlet dofs.
  if (!this->dirichlet_dofs_.empty()) 
  {
    std::vector< DataType > zeros(this->dirichlet_dofs_.size(), 0.);
    rhs->SetValues(vec2ptr(this->dirichlet_dofs_), this->dirichlet_dofs_.size(), vec2ptr(this->dirichlet_values_));
    lhs->diagonalize_rows(vec2ptr(this->dirichlet_dofs_), this->dirichlet_dofs_.size(), 1.0);
  }
  
  rhs->Update();

  // update matrix factorization, used in preconditioner
#ifdef ILUPP
  this->precond_ext_.SetupOperator(*lhs);
  this->precond_ext_.Build();
#else 
  this->precond_ext_.SetupOperator(*lhs);
  this->precond_ext_.Build();
#endif
}

// solve the linear system 
// matrix_ * sol = rhs
void ExamCFD::solve_linear_system(VectorType &rhs, VectorType* sol) 
{   
  sol->CloneFrom(sol_prev_);
  
  // apply BC to initial solution
  if (!dirichlet_dofs_.empty()) 
  {
    sol->SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
  }
  
  // solve single linear system
  DataType norm_rhs = rhs.Norm2();

  this->solver_->Solve(rhs, sol);
  sol->Update();
  DataType res = solver_->res();
  int iter = solver_->iter();

  LOG_INFO("Linear Solver", "absolute residual norm " << res);
  LOG_INFO("Linear Solver", "relative residual norm " << res / norm_rhs);
  LOG_INFO("Linear Solver", "iteratons              " << iter); 
            
  if (params_["Equation"]["PressureZeroAverage"].get< bool >())
  {
    this->ApplyFilter(*sol);
  }
}

// run the time loop
void ExamCFD::time_loop() 
{
  // setup csv file for collection post-processing data
  std::string log_name;
  {  
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,sizeof(buffer),"%d-%m %H:%M",timeinfo);
    std::string str_time(buffer);
    log_name = "Exam_" + student_name + "_" + str_time + ".csv";
  }
  CSVWriter<DataType> csv_writer (log_name);
  
  // read in time stepping parameters
  int step = 0;
  DataType dt = params_["TimeStepping"]["dt"].get< DataType >();
  int bdf_type = params_["TimeStepping"]["BDFType"].get< int >();
  DataType T = params_["TimeStepping"]["T"].get< DataType >();
  DataType time = 0.;

  // set Dirichlet BC to solution vectors 
  this->compute_bc(0.);
  if (!dirichlet_dofs_.empty()) 
  {
    sol_pprev_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
    sol_prev_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
    sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
  }

  // visiualize initial solution
  this->visualize(0);
  
  // run the time loop
  while (time < T)
  {
    LOG_INFO ("", " ================== ");
    LOG_INFO ("", "current time " << time << " / " << T);
    
    // *********************************************************************
    // TODO


    
    // **********************************************************************
    
    // visualize solution
    this->visualize(step);

    // compute post-processing values
    std::vector<DataType> cur_pp_vals(6, 0.);
    cur_pp_vals[0] = time;
    
    DataType l2_div = this->compute_L2_divergence(sol_);
    LOG_INFO ("Divergence", "L2(div)     = " << l2_div );

    DataType kinetic = this->compute_kinetic_energy(sol_);
    LOG_INFO ("Energy", "Kinetic     = " << kinetic );

    DataType dissipation = this->compute_dissipation_energy(sol_);
    LOG_INFO ("Energy", "Dissipation = " << dissipation );

    cur_pp_vals[1] = l2_div;
    cur_pp_vals[2] = kinetic;
    cur_pp_vals[3] = dissipation;
    
    DataType force_x;
    DataType force_y;
    this->compute_forces(sol_, force_x, force_y);
    cur_pp_vals[4] = force_x;
    cur_pp_vals[5] = force_y;
    
    LOG_INFO("Force on foil ", "F_x         = " << force_x);
    LOG_INFO("Force on foil ", "F_y         = " << force_y);

    // write into csv file
    if (rank_ == 0)
    {
      csv_writer.write(cur_pp_vals);
    }
  }
}

// compute the force acting on the foil
// u: vector containing the coefficients of the FE function
void ExamCFD::compute_forces(VectorType &u,
                             DataType &force_x, 
                             DataType &force_y) 
{
  DataType mu = params_["Equation"]["nu"].get< DataType >() * 1.; //(density equals 1)
  int foil_mat_num = params_["BoundaryConditions"]["MaterialNumbers"]["Foil"].get< int >();
 
  u.Update();

  // ******************************************
  // TODO 

  // Hint: look at compute_l2_divergence and use assembly routine 
  // integrate_scalar_boundary(...)
  // and use the local assembler ForceIntegral



  
  // ******************************************
}

// compute the kinetic energy of the FE function determined by vector u
typename LAD::DataType ExamCFD::compute_kinetic_energy(VectorType &u) 
{
  DataType recv = 0.;
  DataType local = 0.;
  u.Update();

  // ******************************************
  // TODO 
  
  // Hint: look at compute_l2_divergence and use assembly routine 
  // and use the local assembler PostProcessingIntegral 



  
  // ******************************************************
  return recv;
}

// compute the dissipation energy of the FE function determined by vector u
typename LAD::DataType ExamCFD::compute_dissipation_energy(VectorType &u) 
{
  DataType recv = 0.;
  DataType local = 0.;
  u.Update();

  // ******************************************
  // TODO 
  
  // Hint: look at compute_l2_divergence and use assembly routine 
  // and use the local assembler PostProcessingIntegral 


  
  // ******************************************************
  return recv;
}

// visualize FE function determined by vector sol_
void ExamCFD::visualize(int time_step) 
{     
  // Setup visualization object.
  int num_sub_intervals = 1;
  CellVisualization< DataType, DIM > visu(space_, num_sub_intervals);

  // collect cell-wise data from mesh object
  const int tdim = mesh_->tdim();
  const int num_cells = mesh_->num_entities(tdim);
  std::vector< DataType > remote_index(num_cells, 0);
  std::vector< DataType > sub_domain(num_cells, 0);
  std::vector< DataType > material_number(num_cells, 0);

  // loop through all cells in the mesh
  for (mesh::EntityIterator it = mesh_->begin(tdim); it != mesh_->end(tdim); ++it) 
  {
    int temp1, temp2;
    const int cell_index = it->index();
    if (DIM > 1) 
    {
      mesh_->get_attribute_value("_remote_index_", tdim, cell_index, &temp1);
      mesh_->get_attribute_value("_sub_domain_", tdim, cell_index, &temp2);
      remote_index.at(cell_index) = temp1;
      sub_domain.at(cell_index) = temp2;
    }
    material_number.at(cell_index) = mesh_->get_material_number(tdim, cell_index);
  }

  // visualize finite element function corresponding to 
  // coefficient vector sol_
  visu.visualize(sol_, 0, "v_x");
  visu.visualize(sol_, 1, "v_y");
  if (DIM > 2)
  {
    visu.visualize(sol_, DIM-1, "v_z");
  }
  visu.visualize(sol_, DIM, "p");
  visu.visualize(sol_, DIM+1, "T");
  
  // visualize some mesh data
  visu.visualize_cell_data(remote_index, "_remote_index_");
  visu.visualize_cell_data(sub_domain, "_sub_domain_");
  visu.visualize_cell_data(material_number, "Material Id");
  
  // write out data
  std::stringstream name;
  name << "exam_" << student_name << "_" << time_step;
  
  VTKWriter< DataType, DIM> vtk_writer (visu, this->comm_, MASTER_RANK);
  vtk_writer.write(name.str());    
}

// compute the L2 norm of the divergence of the FE function determined by vector u
typename LAD::DataType ExamCFD::compute_L2_divergence(VectorType &u) 
{
  DataType recv = 0.;
  DataType l2_div = 0.;
  u.Update();

  PostProcessingIntegral< DIM, LAD > int_div(u, 2, params_["Equation"]["nu"].get< DataType >());
  this->global_asm_.integrate_scalar(space_, int_div, l2_div);

  // collect results from all parallel processes
  this->parcom_->sum(l2_div, recv);
  recv = std::sqrt(recv);
  return recv;
}

// compute the average pressure of the FE function determined by vector u
typename LAD::DataType ExamCFD::compute_pressure_int(VectorType &u) 
{
  DataType recv = 0.;
  DataType total_pressure = 0.;
  u.Update();

  PressureIntegral< DIM, LAD > int_p(u);
  this->global_asm_.integrate_scalar(space_, int_p, total_pressure);

  // collect results from all parallel processes
  this->parcom_->sum(total_pressure, recv);
  return recv;
}

typename LAD::DataType ExamCFD::compute_volume_int() 
{
  DataType integrated_vol = 0.;
  DataType recv = 0.;
  VolumeIntegral< DIM, LAD > vol_int;
  this->global_asm_.integrate_scalar(space_, vol_int, integrated_vol);

  // collect results from all parallel processes
  this->parcom_->sum(integrated_vol, recv);
  return recv;
}

typename LAD::DataType ExamCFD::compute_average_pressure(VectorType &u) 
{
  DataType total_pressure = compute_pressure_int(u);
  DataType integrated_vol = compute_volume_int();
  const DataType average_pressure = total_pressure / integrated_vol;

  //LOG_INFO ( "Average pressure", average_pressure );
  return average_pressure;
}

void ExamCFD::ApplyFilter(VectorType &u) 
{
  if (!params_["Equation"]["PressureZeroAverage"].get< bool >())
  {
    return;
  }
  
  LOG_INFO ("Apply", "pressure filter" );
  
  DataType average_pressure = this->compute_average_pressure(u);
  LOG_INFO ( "pressure_filter", "Average pressure before filter = " << average_pressure );
  
  VectorType pressure_correction;
  pressure_correction.CloneFromWithoutContent(u);
  pressure_correction.Zeros();

  // set value for pressure dofs to average pressure
  std::vector< int > cell_p_dofs;
  std::vector< int > local_p_dofs;
  for (EntityIterator it = mesh_->begin(DIM), end = mesh_->end(DIM); it != end; ++it) 
  {
    cell_p_dofs.clear();
    this->space_.get_dof_indices(this->space_.var_2_fe(DIM), it->index(), cell_p_dofs);
    
    for (int i = 0, sz = cell_p_dofs.size(); i < sz; ++i) 
    {
      if (this->space_.dof().is_dof_on_subdom(cell_p_dofs[i])) 
      {
        local_p_dofs.push_back(cell_p_dofs[i]);
      }
    }
  }

  std::sort(local_p_dofs.begin(), local_p_dofs.end());
  std::unique(local_p_dofs.begin(), local_p_dofs.end());

  // remove average pressure from solution
  std::vector< DataType > p_correction_values(local_p_dofs.size());
  std::fill(p_correction_values.begin(), p_correction_values.end(), average_pressure);

  pressure_correction.SetValues(vec2ptr(local_p_dofs), local_p_dofs.size(), vec2ptr(p_correction_values));
  pressure_correction.Update();
  
  u.Axpy(pressure_correction, -1.);
  u.Update();
  
  average_pressure = this->compute_average_pressure(u);
  LOG_INFO ( "pressure_filter", "Average pressure after filter = " << average_pressure );
}
