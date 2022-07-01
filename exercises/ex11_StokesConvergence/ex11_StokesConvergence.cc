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

#include "ex11_StokesConvergence.h"

#define nNU_SERIES

#define USE_LUMKL
#define nUSE_ILUPP

#define nUSE_P4EST
#define USE_METIS

const int GHOST_LAYER_WIDTH = 2;

#ifdef WITH_GPERF
#include "profiler.h"
#endif

#ifndef WITH_MKL
#undef USE_MKL
#undef USE_LUMKL
#endif

#ifndef WITH_P4EST
#undef USE_P4EST
#endif

#ifndef WITH_METIS
#undef USE_METIS
#endif

#ifndef WITH_P4EST
#undef USE_P4EST
#endif

static const char *PARAM_FILENAME = "ex11_StokesConvergence.xml";
#ifndef MESHES_DATADIR
#define MESHES_DATADIR "./"
#endif
static const char *DATADIR = MESHES_DATADIR;

// Main application class ///////////////////////////////////

class ConvergenceStokes {
public:
  ConvergenceStokes(const std::string &param_filename,
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
    
    errors_.clear();
    errors_.resize(5);
    
    // Main adaptation loop.
    while (!is_done_) 
    {
      LOG_INFO ("", "==============================");
      LOG_INFO ("", "mesh refinement level " << refinement_level_);
      LOG_INFO ("", "==============================");
      Timer timer;
      timer.start();
      
      // Initialize space and linear algebra.
      LOG_INFO ("do", "Prepare System");
        
      prepare_system();
     
      timer.stop();
      LOG_INFO("duration", timer.get_duration()); 
      timer.reset();
      timer.start();
      
#ifdef NU_SERIES
      int num_nu_steps = params_["Equation"]["StepsNu"].get< int >();
      DataType nu = params_["Equation"]["InitialNu"].get< DataType >();
#else
      int num_nu_steps = 1;
      DataType nu = params_["Equation"]["nu"].get< DataType >();
#endif
      for (int n = 0; n!= num_nu_steps; ++n)
      {
#ifdef NU_SERIES
        errors_[0].push_back(nu);
#endif
        // Compute the stiffness matrix and right-hand side.
        LOG_INFO ("do", "Assemble System for nu = " << nu );
     
        assemble_system(nu);
     
        timer.stop();
        LOG_INFO("duration", timer.get_duration()); 
        timer.reset();
        timer.start();

        // Solve the linear system.
        LOG_INFO ("do", "Solve System ");
      
        solve_system();
      
        timer.stop();
        LOG_INFO("duration", timer.get_duration()); 
        timer.reset();
        timer.start();

        // compute the difference to the exact solution
        LOG_INFO ("do", "Compute error ");
      
        compute_error();
        compute_L2_divergence(sol_);
      
        // Visualize the solution and the errors.
        LOG_INFO ("do", "Visualize Solution ");
        
        visualize();
      
        timer.stop();
        LOG_INFO("duration", timer.get_duration()); 
        timer.reset();
        timer.start();
        
        nu *= 0.1;
      }
      // refine mesh
      LOG_INFO ("do", "Adapt Mesh");
      
      adapt();
      
      timer.stop();
      LOG_INFO("duration", timer.get_duration()); 
      timer.reset();
      timer.start();
    }

    write_2d_array("errors.csv", ",", false, 10, this->rank_, errors_);
                    
#ifdef WITH_GPERF
  ProfilerStop();
#endif
  }

  ~ConvergenceStokes() {
  }

private:
  // Member functions

  // Read and distribute mesh.
  std::string path_mesh;
  void build_initial_mesh();
  
  // Setup space, linear algebra, and compute Dirichlet values.
  void prepare_system();

  // Compute the matrix and rhs.
  void assemble_system(DataType nu);
  
  // Compute solution x.
  void solve_system();
  
  // Visualize the results.
  void visualize();
  
  // Adapt the space (mesh and/or degree).
  void adapt();

  void compute_error();
  
  void ApplyFilter(VectorType &u);
  DataType compute_average_pressure(VectorType &u);
  DataType compute_volume_int();
  DataType compute_pressure_int(VectorType &u);
  DataType compute_L2_divergence(VectorType &u);

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
  VectorType rhs_, sol_;

  // System matrix.
  MatrixType matrix_;

  // Global assembler.
  DGGlobalAssembler< DataType, DIM > global_asm_;

  // Flag for stopping adaptive loop.
  bool is_done_;
  
  // Current refinement level.
  int refinement_level_;

  // Dof id:s for Dirichlet boundary conditions.
  std::vector< int > dirichlet_dofs_;
  
  // Dof values for Dirichlet boundary conditions.
  std::vector< DataType > dirichlet_values_;
  
  std::shared_ptr<ParCom> parcom_;
  
  VelocityDirichletBC bc_dirichlet_;
  
  bool assemble_DG_;
  
  std::vector< std::vector< DataType > > errors_;
  
}; // end class ConvergenceStokes

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
    ConvergenceStokes app(param_filename, path_mesh);
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

//////////////// ConvergenceStokes implementation //////////////

void ConvergenceStokes::build_initial_mesh() 
{
  // Read in the mesh on the master process.
  if (rank_ == MASTER_RANK) 
  {
    std::string mesh_name = params_["Mesh"]["Filename" + std::to_string(DIM)].get< std::string >();
    std::string mesh_filename;
    if (path_mesh.empty()) 
    {
      mesh_filename = std::string(DATADIR) + mesh_name;
    } 
    else 
    {
      mesh_filename = path_mesh + mesh_name;
    }
#ifdef USE_P4EST
    master_mesh_ = read_mesh_from_file(mesh_filename, DIM, DIM, 0, IMPL_P4EST);
#else
    master_mesh_ = read_mesh_from_file(mesh_filename, DIM, DIM, 0, IMPL_DBVIEW);
#endif
    // Refine the mesh until the initial refinement level is reached.
    const int initial_ref_lvl = params_["Mesh"]["InitialRefLevel"].get< int >();
    this->master_mesh_ = this->master_mesh_->refine_uniform_seq(initial_ref_lvl);
    refinement_level_ = initial_ref_lvl;
  }

  MPI_Bcast(&refinement_level_, 1, MPI_INT, MASTER_RANK, comm_);

  MeshPtr local_mesh;
  int uniform_ref_steps;
    
#ifdef USE_P4EST
  local_mesh = partition_and_distribute(this->master_mesh_, 0, MPI_COMM_WORLD, uniform_ref_steps,IMPL_P4EST);
#else
# ifdef USE_PARMETIS
  ParMetisGraphPartitioner partitioner;
  LOG_INFO("",">  Partition initial mesh with ParMetis " );
# else
#   ifdef USE_METIS
  MetisGraphPartitioner partitioner;
  LOG_INFO("",">  Partition initial mesh with Metis " );
#   else
  NaiveGraphPartitioner partitioner;
  LOG_INFO("",">  Partition initial mesh with naive partitioner " );
# endif
#endif
  const GraphPartitioner *p = &partitioner;
  local_mesh = partition_and_distribute(this->master_mesh_, 0, MPI_COMM_WORLD, p, uniform_ref_steps, IMPL_DBVIEW);
#endif  
       
  assert(local_mesh != 0);
  SharedVertexTable shared_verts;
  
#ifdef USE_P4EST
  this->mesh_ = compute_ghost_cells(*local_mesh, MPI_COMM_WORLD, shared_verts, IMPL_P4EST, GHOST_LAYER_WIDTH);
#else
  this->mesh_ = compute_ghost_cells(*local_mesh, MPI_COMM_WORLD, shared_verts, IMPL_DBVIEW, GHOST_LAYER_WIDTH);
#endif
  // InterfaceList if_list = InterfaceList::create(master_mesh_);
  // std::cout << "If-list = \n" << if_list << "\n";
}

void ConvergenceStokes::prepare_system() 
{
  // Assign degrees to each element.
  const int nb_fe_var = 2;
  
  // prepare space
  std::string fe_type_u = params_["FESpace"]["Velocity"]["Type"].get< std::string >();
  std::string fe_type_p = params_["FESpace"]["Pressure"]["Type"].get< std::string >();
  
  const int u_deg = params_["FESpace"]["Velocity"]["Degree"].get< int >();
  const int p_deg = params_["FESpace"]["Pressure"]["Degree"].get< int >();
  
  std::vector< int > fe_params(2);  
  std::vector< FEType > fe_ansatz (2);
  std::vector< bool > is_cg (2);
  
  fe_params[0] = u_deg;
  fe_params[1] = p_deg;
  this->assemble_DG_ = false;
  
  if (fe_type_u == "BDM")
  {
    fe_ansatz[0] = FEType::BDM;  
    is_cg[0] = true;
    this->assemble_DG_ = true;
  }
  else if (fe_type_u == "RT")
  {
    fe_ansatz[0] = FEType::RT;  
    is_cg[0] = true;
    this->assemble_DG_ = true;
  }
  else if (fe_type_u == "CG")
  {
    fe_ansatz[0] = FEType::LAGRANGE_VECTOR;  
    is_cg[0] = true;
  }
  else if (fe_type_u == "DG")
  {
    fe_ansatz[0] = FEType::LAGRANGE_VECTOR;  
    is_cg[0] = false;
    this->assemble_DG_ = true;
  }
  
  if (fe_type_p == "CG")
  {
    fe_ansatz[1] = FEType::LAGRANGE;  
    is_cg[1] = true;
  }
  else if (fe_type_p == "DG")
  {
    fe_ansatz[1] = FEType::LAGRANGE;  
    is_cg[1] = false;
    this->assemble_DG_ = true;
  }
  
  this->assemble_DG_ = true;
  // Initialize the VectorSpace object.
  space_.Init(*mesh_, fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC);
  
  
  LOG_INFO("nb nonlin trafos", space_.fe_manager().nb_nonlinear_trafos());
  
  // Compute the matrix sparsity structure
  std::vector< std::vector< bool > > coupling_vars;
  coupling_vars.resize(2);
  coupling_vars[0].push_back(true);
  coupling_vars[0].push_back(true);
  coupling_vars[1].push_back(true);
  coupling_vars[1].push_back(true);
  
  SparsityStructure sparsity;
  compute_sparsity_structure(space_, sparsity, coupling_vars, true);

  // initialize matrix object
  matrix_.Init(comm_, space_.la_couplings());
  matrix_.InitStructure(sparsity);
  matrix_.Zeros();
  
  // initialize vector objects for solution coefficitons and 
  // right hand side
  sol_.Init(comm_, space_.la_couplings());
  rhs_.Init(comm_, space_.la_couplings()); 
  rhs_.Zeros();
  sol_.Zeros();

  // Compute Dirichlet BC dofs and values using known exact solution.
  dirichlet_dofs_.clear();
  dirichlet_values_.clear();

  const bool enforce_bc = params_["FESpace"]["Velocity"]["EnforceBC"].get< bool >();
  
  if (enforce_bc)
  {
    compute_dirichlet_dofs_and_values(bc_dirichlet_, space_, 0, dirichlet_dofs_, dirichlet_values_);
  }
}

void ConvergenceStokes::assemble_system(DataType nu) 
{ 
  DataType gamma_p    = params_["Equation"]["Continuity"]["Gamma"].get< DataType >();
  int pressure_stab_type  = params_["Equation"]["Continuity"]["PressureStabType"].get< int >();
  DataType sigma_u    = params_["Equation"]["Momentum"]["SIPPenalty"].get< DataType >();
  DataType sigma_p    = params_["Equation"]["Continuity"]["JumpPenalty"].get< DataType >();
  DataType gamma      = 0.;
  bool do_sip_u       = params_["Equation"]["Momentum"]["SIP"].get< bool >();
  bool do_press_flux  = params_["Equation"]["Momentum"]["PressFlux"].get< bool >();
  bool do_press_stab  = params_["Equation"]["Continuity"]["PressStab"].get< bool >();
                       
  // Assemble matrix and right-hand-side vector.
  // pressure stabilization type: 
  // 0: no stabilization
  // 1: h * M_p, with M_p denoting the pressure mass matrix
  // 2: h * A_p, with A_p denoting the pressure stiffness matrix
  // 3: PSPG
  DataType hmax = -1e6;
  
  LocalStokesAssembler local_asm;
  local_asm.set_parameters(nu, &(this->bc_dirichlet_) ,
                           pressure_stab_type, gamma_p,
                           sigma_u, sigma_p, gamma,
                           do_sip_u, do_press_stab, do_press_flux, &hmax);
                           
  global_asm_.should_reset_assembly_target(true);
  global_asm_.assemble_matrix(space_, local_asm, matrix_);
  
  if (this->assemble_DG_)
  {
    global_asm_.should_reset_assembly_target(false);
    global_asm_.assemble_interface_matrix(space_, local_asm, matrix_);
    global_asm_.should_reset_assembly_target(true);
  }
  
  global_asm_.should_reset_assembly_target(true);
  global_asm_.assemble_vector(space_, local_asm, rhs_);
  if (this->assemble_DG_)
  {
    global_asm_.should_reset_assembly_target(false);
    global_asm_.assemble_interface_vector(space_, local_asm, rhs_);
    global_asm_.should_reset_assembly_target(true);  
  }
  
  // Correct Dirichlet dofs.
  if (!dirichlet_dofs_.empty()) 
  {
    matrix_.diagonalize_rows(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), 1.0);
    rhs_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
    sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
  }
  
  rhs_.Update();
  sol_.Update();
  
#ifndef NU_SERIES
  errors_[0].push_back(hmax);
#endif
}

void ConvergenceStokes::compute_error() 
{ 
  sol_.Update();
  ExactSol exact_sol;
  
  // L2 velocity error 
  exact_sol.set_fe_ind(0);
  WkpErrorIntegrator l2_vel_asm (0, 2, sol_, &exact_sol); 
  
  DataType local_l2_vel_err = 0.;
  global_asm_.integrate_scalar(this->space_, l2_vel_asm, local_l2_vel_err);
  
  DataType global_l2_vel_err = 0.;
  MPI_Reduce(&local_l2_vel_err, &global_l2_vel_err, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
  global_l2_vel_err = std::pow( global_l2_vel_err, 0.5 );
  
  LOG_INFO("L2 velocity error", global_l2_vel_err);
  
  // H1 velocity error 
  exact_sol.set_fe_ind(0);
  WkpErrorIntegrator h1_vel_asm (1, 2, sol_, &exact_sol); 
  
  DataType local_h1_vel_err = 0.;
  global_asm_.integrate_scalar(this->space_, h1_vel_asm, local_h1_vel_err);
  
  DataType global_h1_vel_err = 0.;
  MPI_Reduce(&local_h1_vel_err, &global_h1_vel_err, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
  global_h1_vel_err = std::pow( global_h1_vel_err, 0.5 );
  
  LOG_INFO("H1 velocity error", global_h1_vel_err);
  
  // L2 pressure error 
  exact_sol.set_fe_ind(1);
  WkpErrorIntegrator l2_pre_asm (0, 2, sol_, &exact_sol); 
  
  DataType local_l2_pre_err = 0.;
  global_asm_.integrate_scalar(this->space_, l2_pre_asm, local_l2_pre_err);
  
  DataType global_l2_pre_err = 0.;
  MPI_Reduce(&local_l2_pre_err, &global_l2_pre_err, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
  global_l2_pre_err = std::pow( global_l2_pre_err, 0.5 );
  LOG_INFO("L2 pressure error", global_l2_pre_err);

  errors_[1].push_back(global_l2_vel_err);
  errors_[2].push_back(global_h1_vel_err);
  errors_[3].push_back(global_l2_pre_err);

}

void ConvergenceStokes::solve_system() 
{
  LinearSolver< LAD > *solver_;
  LinearSolverFactory< LAD > SolFact;
  solver_ = SolFact.Get(params_["LinearSolver"]["Name"].get< std::string >("CG"))->params(params_["LinearSolver"]);
  solver_->SetupOperator(matrix_);

#ifdef USE_ILUPP
  PreconditionerBlockJacobiExt< LAD > precond;
  precond.Init_ILU_pp();
  if (params_["LinearSolver"]["Method"].get< std::string >() == "RightPreconditioning")
  {
    precond.SetupOperator(matrix_);
    precond.Build();
    solver_->SetupPreconditioner(precond);
  }
  
#else
# ifdef USE_LUMKL
  PreconditionerBlockJacobiExt< LAD > precond;
  precond.Init_LU_mkl();
  if (params_["LinearSolver"]["Method"].get< std::string >() == "RightPreconditioning")
  {
    precond.SetupOperator(matrix_);
    precond.Build();
    solver_->SetupPreconditioner(precond);
  }
# endif
#endif

  
  rhs_.Update();
  solver_->Solve(rhs_, &sol_);
  sol_.Update();
  
  this->ApplyFilter(sol_);
  
  sol_.Update();
  
  // *************************************
  // TODO ex c): insert your solution here
  int num_iter = solver_->iter();
  DataType residual = solver_->res();
  LOG_INFO("GMRES iter", num_iter);
  LOG_INFO("GMRES residual", residual);

  // *************************************
  
  delete solver_;
}

void ConvergenceStokes::visualize() 
{
  sol_.Update();
  
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
  if (DIM == 3)
  {
    visu.visualize(sol_, DIM-1, "v_z");
  }
  visu.visualize(sol_, DIM, "p");
  
  // visualize some mesh data
  visu.visualize_cell_data(remote_index, "_remote_index_");
  visu.visualize_cell_data(sub_domain, "_sub_domain_");
  visu.visualize_cell_data(material_number, "Material Id");
  
  // write out data
  std::stringstream name;
  name << "ex5_solution" << refinement_level_;
  
  VTKWriter< DataType, DIM> vtk_writer (visu, this->comm_, MASTER_RANK);
  vtk_writer.write(name.str());    
}

void ConvergenceStokes::adapt() 
{
  const int final_ref_level = params_["Mesh"]["FinalRefLevel"].get< int >(6);
  if (refinement_level_ >= final_ref_level) 
  {
    is_done_ = true;
  } 
  else 
  {
    if (rank_ == MASTER_RANK)
    {
      this->master_mesh_ = this->master_mesh_->refine_uniform_seq(1);
    }
    ++refinement_level_;
  }
  
  MeshPtr local_mesh;
  int uniform_ref_steps;
    
#ifdef USE_P4EST
  local_mesh = partition_and_distribute(this->master_mesh_, 0, MPI_COMM_WORLD, uniform_ref_steps,IMPL_P4EST);
#else
# ifdef USE_PARMETIS
  ParMetisGraphPartitioner partitioner;
  LOG_INFO("",">  Partition initial mesh with ParMetis " );
# else
#   ifdef USE_METIS
  MetisGraphPartitioner partitioner;
  LOG_INFO("",">  Partition initial mesh with Metis " );
#   else
  NaiveGraphPartitioner partitioner;
  LOG_INFO("",">  Partition initial mesh with naive partitioner " );
# endif
#endif
  const GraphPartitioner *p = &partitioner;
  local_mesh = partition_and_distribute(this->master_mesh_, 0, MPI_COMM_WORLD, p, uniform_ref_steps, IMPL_DBVIEW);
#endif  
       
  assert(local_mesh != 0);
  SharedVertexTable shared_verts;
  
#ifdef USE_P4EST
  this->mesh_ = compute_ghost_cells(*local_mesh, MPI_COMM_WORLD, shared_verts, IMPL_P4EST, 2);
#else
  this->mesh_ = compute_ghost_cells(*local_mesh, MPI_COMM_WORLD, shared_verts);
#endif
}

typename LAD::DataType ConvergenceStokes::compute_pressure_int(VectorType &u) 
{
  DataType recv;
  DataType total_pressure;
  u.Update();

  PressureIntegral< DIM, LAD > int_p(u);
  this->global_asm_.integrate_scalar(space_, int_p, total_pressure);

  this->parcom_->sum(total_pressure, recv);
  return recv;
}

typename LAD::DataType ConvergenceStokes::compute_volume_int() 
{
  DataType integrated_vol;
  DataType recv;
  VolumeIntegral< DIM, LAD > vol_int;
  this->global_asm_.integrate_scalar(space_, vol_int, integrated_vol);

  this->parcom_->sum(integrated_vol, recv);
  
  LOG_INFO("|Omega|", recv);
  return recv;
}

typename LAD::DataType ConvergenceStokes::compute_average_pressure(VectorType &u) 
{
  DataType total_pressure = compute_pressure_int(u);
  DataType integrated_vol = compute_volume_int();
  const DataType average_pressure = total_pressure / integrated_vol;

  LOG_INFO ( "Average pressure", average_pressure );

  return average_pressure;
}

typename LAD::DataType ConvergenceStokes::compute_L2_divergence(VectorType &u) 
{
  DataType recv = 0.;
  DataType l2_div = 0.;
  u.Update();

  DivergenceIntegral< DIM, LAD > int_div(u);
  this->global_asm_.integrate_scalar(space_, int_div, l2_div);

  // collect results from all parallel processes
  this->parcom_->sum(l2_div, recv);
  recv = std::sqrt(recv);
  
  errors_[4].push_back(recv);
    
  LOG_INFO("L2 divergence", recv);
  return recv;
}

void ConvergenceStokes::ApplyFilter(VectorType &u) 
{
  if (!params_["UsePressureFilter"].get< bool >())
  {
    return;
  }
  
  LOG_INFO ("Apply", "pressure filter" );
  u.Update();
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
