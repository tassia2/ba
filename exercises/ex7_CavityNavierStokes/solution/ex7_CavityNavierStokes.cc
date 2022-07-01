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

#include "ex7_CavityNavierStokes.h"

#ifdef WITH_GPERF
#include "profiler.h"
#endif

static const char *PARAM_FILENAME = "ex7_CavityNavierStokes.xml";
#ifndef MESHES_DATADIR
#define MESHES_DATADIR "./"
#endif
static const char *DATADIR = MESHES_DATADIR;

// Main application class ///////////////////////////////////

class CavityNavierStokes : public NonlinearProblem< LAD >
{
public:
  CavityNavierStokes(const std::string &param_filename,
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
  std::string prof_name = "profiling.log";
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
      
      // Solve the nonlinear system.
      LOG_INFO ("", "==============================")
      LOG_INFO ("", "==============================")
      LOG_INFO ("do", "Solve Navier Stokes System ");
      this->equation_type_ = 1;
      solve_nonlinear_system(params_["Equation"]["InitialNu"].get< DataType >(), 
                             params_["Equation"]["Nu"].get< DataType >(), 
                             params_["Equation"]["NumberContinuationSteps"].get< int >());
      
      timer.stop();
      LOG_INFO("duration", timer.get_duration()); 
      timer.reset();
      timer.start();
      
      // Solve the linear system.
      LOG_INFO ("", "==============================")
      LOG_INFO ("", "==============================")
      LOG_INFO ("do", "Solve Stokes System ");
      this->equation_type_ = 0;
      solve_nonlinear_system(params_["Equation"]["InitialNu"].get< DataType >(), 
                             params_["Equation"]["Nu"].get< DataType >(), 
                             params_["Equation"]["NumberContinuationSteps"].get< int >());
      
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

  ~CavityNavierStokes() {
  }

private:
  // Member functions

  // Read and distribute mesh.
  std::string path_mesh;
  void build_initial_mesh();
  
  // Setup space, linear algebra, and compute Dirichlet values.
  void prepare_system();
 
   // assembler routines for Jacobian and residual in Newton's method
  void EvalGrad(const VectorType &in, MatrixType *out); 
  void EvalFunc(const VectorType &in, VectorType *out);
  
  // Compute solution x.
  void solve_nonlinear_system(DataType initial_nu, DataType final_nu, int num_steps); 
  
  // Visualize the results.
  void visualize(int continuation_step);
  
  // Adapt the space (mesh and/or degree).
  void adapt();

  void ApplyFilter(VectorType &u);
  DataType compute_average_pressure(VectorType &u);
  DataType compute_volume_int();
  DataType compute_pressure_int(VectorType &u);

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

  // nonlinear solver
  Newton< LAD, DIM > newton_;
  
  // linear solver
  LinearSolver< LAD > * solver_;
  
  // preconditioner
  PreconditionerBlockJacobiExt< LAD > precond_;
  
  // Flag for stopping adaptive loop.
  bool is_done_;
  
  // Current refinement level.
  int refinement_level_;

  // Dof id:s for Dirichlet boundary conditions.
  std::vector< int > dirichlet_dofs_;
  
  // Dof values for Dirichlet boundary conditions.
  std::vector< DataType > dirichlet_values_;
  
  std::shared_ptr<ParCom> parcom_;
  
  int equation_type_;
  DataType nu_;
  
}; // end class CavityNavierStokes

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
    CavityNavierStokes app(param_filename, path_mesh);
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

//////////////// CavityNavierStokes implementation //////////////

void CavityNavierStokes::build_initial_mesh() {
  mesh::IMPL impl = mesh::IMPL_DBVIEW;

  // Read in the mesh on the master process. The mesh is chosen according to the
  // DIM of the problem.
  if (rank_ == MASTER_RANK) 
  {
    std::string mesh_name;

    switch (DIM) 
    {
      case 2: 
      {
        mesh_name = params_["Mesh"]["Filename2"].get< std::string >("unit_square.inp");
        break;
      }
      case 3: 
      {
        mesh_name = params_["Mesh"]["Filename3"].get< std::string >("unit_cube.inp");
        break;
      }

      default:
        assert(0);
    }
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

void CavityNavierStokes::prepare_system() 
{
  // Assign degrees to each element.
  const int nb_fe_var = 2;
  
  const int velocity_fe_degree = params_["FESpace"]["VelocityDegree"].get< int >(1);
  const int pressure_fe_degree = params_["FESpace"]["PressureDegree"].get< int >(1);
  
  std::vector< int > fe_params(nb_fe_var);
  fe_params[0] = velocity_fe_degree;
  fe_params[1] = pressure_fe_degree;
  
  // assign types of FE 
  std::vector< FEType > fe_ansatz (nb_fe_var);
  fe_ansatz[0] = FEType::LAGRANGE_VECTOR;
  fe_ansatz[1] = FEType::LAGRANGE;
  
  // both variables should be continuous
  std::vector< bool > is_cg (nb_fe_var, true);
  
  // Initialize the VectorSpace object.
  space_.Init(*mesh_, fe_ansatz, is_cg, fe_params, hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC);
  
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
  sol_.Init(comm_, space_.la_couplings());
  rhs_.Init(comm_, space_.la_couplings());  
  rhs_.Zeros();
  sol_.Zeros();

  // Compute Dirichlet BC dofs and values using known exact solution.
  dirichlet_dofs_.clear();
  dirichlet_values_.clear();

  VelocityDirichletBC bc_dirichlet(params_["Equation"]["TopMaterialNumber"].get< int >(),
                                   params_["Equation"]["TopVelocityX"].get< DataType >(),
                                   params_["Equation"]["TopEPS"].get< DataType >());
  compute_dirichlet_dofs_and_values(bc_dirichlet, space_, 0, dirichlet_dofs_, dirichlet_values_);

  // setup  linear solver
  LinearSolverFactory< LAD > SolFact;
  solver_ = SolFact.Get(params_["LinearSolver"]["Name"].get< std::string >("GMRES"))->params(params_["LinearSolver"]);
  solver_->SetupOperator(matrix_);

  precond_.Init_ILU_pp();
  solver_->SetupPreconditioner(precond_);
  
  // get nonlinear solver parameters from param file
  int nls_max_iter = params_["NonlinearSolver"]["MaximumIterations"].get< int >();
  DataType nls_abs_tol = params_["NonlinearSolver"]["AbsoluteTolerance"].get< DataType >();
  DataType nls_rel_tol = params_["NonlinearSolver"]["RelativeTolerance"].get< DataType >();
  DataType nls_div_tol = params_["NonlinearSolver"]["DivergenceLimit"].get< DataType >();
  std::string forcing_strategy = params_["NonlinearSolver"]["ForcingStrategy"].get< std::string >();
  bool use_forcing_strategy = (forcing_strategy != "None");
  DataType eta = 1.e-4; // initial value of forcing term

  // get forcing strategy parameters from param file
  DataType eta_initial = params_["NonlinearSolver"]["InitialValueForcingTerm"].get< DataType >();
  DataType eta_max = params_["NonlinearSolver"]["MaxValueForcingTerm"].get< DataType >();
  DataType gamma_EW2 = params_["NonlinearSolver"]["GammaParameterEW2"].get< DataType >();
  DataType alpha_EW2 = params_["NonlinearSolver"]["AlphaParameterEW2"].get< DataType >();

  // setup nonlinear solver
  newton_.InitParameter(&rhs_, &matrix_);
  newton_.InitParameter(Newton< LAD, DIM >::NewtonInitialSolutionOwn);
  newton_.InitControl(nls_max_iter, nls_abs_tol, nls_rel_tol, nls_div_tol);
  newton_.SetOperator(*this);
  newton_.SetLinearSolver(*this->solver_);
  newton_.SetPrintLevel(1);
  
  // Forcing strategy object
  if (forcing_strategy == "EisenstatWalker1") 
  {
    EWForcing< LAD > *EW_Forcing = new EWForcing< LAD >(eta_initial, eta_max, 1);
    newton_.SetForcingStrategy(*EW_Forcing);
  } 
  else if (forcing_strategy == "EisenstatWalker2") 
  {
    EWForcing< LAD > *EW_Forcing = new EWForcing< LAD >(eta_initial, eta_max, 2, gamma_EW2, alpha_EW2);
    newton_.SetForcingStrategy(*EW_Forcing);
  }  
}

void CavityNavierStokes::solve_nonlinear_system(DataType initial_nu, DataType final_nu, int num_steps) 
{   
  sol_.Zeros();
  // apply BC to initial solution
  if (!dirichlet_dofs_.empty()) 
  {
    // correct solution with dirichlet BC
    sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), vec2ptr(dirichlet_values_));
  }
  
  // solve single nonlinear system
  if (num_steps == 1)
  {
    this->nu_ = params_["Equation"]["Nu"].get< DataType >();
  
    newton_.Solve(&sol_);
    LOG_INFO(1, "Newton ended with residual norm "
              << newton_.GetResidual() << " after "
              << newton_.iter() << " iterations.");
              
    this->visualize(0);
    return;
  }
  
  // solve series of nonlinear systems
  for (int i=0; i<num_steps; ++i)
  {
    //this->nu_ = initial_nu + static_cast<DataType>(i) / static_cast<DataType>(num_steps-1)  * (final_nu - initial_nu);
    this->nu_ = initial_nu * std::pow(final_nu / initial_nu, static_cast<DataType>(i) / static_cast<DataType>(num_steps-1));
    
    LOG_INFO(1, "------------------------------");
    LOG_INFO(1, "------------------------------");
    LOG_INFO(1, "start Newton method for i = " << i << ", nu = " << this->nu_);  

    newton_.Solve(&sol_);
    LOG_INFO(1, "Newton ended with residual norm " << newton_.GetResidual() << " after " << newton_.iter() << " iterations.");
  
    this->visualize(i);
  }
}

// Assemble jacobian Matrix for Newton method for solving F(u) = 0
// out = D_u F[in]
void CavityNavierStokes::EvalGrad(const VectorType &in,
                                MatrixType *out) 
{ 
  // update ghost values of input vector (related to parallelisation)
  //in.Update();
  
  // pass parameters to local assembler
  LocalFlowAssembler local_asm;
  local_asm.set_parameters(this->nu_, 
                           params_["Equation"]["Fz"].get< DataType >(),
                           this->equation_type_);
                           
  // pass current Newton iterate to local assembler
  local_asm.set_newton_solution(&in);
  
  // call assemble routine
  this->global_asm_.should_reset_assembly_target(true);
  this->global_asm_.assemble_matrix(this->space_, local_asm, *out);
  
  // Correct Dirichlet dofs.
  if (!this->dirichlet_dofs_.empty()) 
  {
    out->diagonalize_rows(vec2ptr(this->dirichlet_dofs_), this->dirichlet_dofs_.size(), 1.0);
  }
  
  // update matrix factorization, used in preconditioner
  this->precond_.SetupOperator(*out);
  this->precond_.Build();
}

// Assemble Residual for use in Newton's method for solving F(u) = 0
// out = F(in)
void CavityNavierStokes::EvalFunc(const VectorType &in,
                                VectorType *out) 
{
  // update ghost values of input vector (related to parallelisation)
  //in.Update();
  
  // pass parameters to local assembler
  LocalFlowAssembler local_asm;
  local_asm.set_parameters(this->nu_, 
                           params_["Equation"]["Fz"].get< DataType >(),
                           this->equation_type_);
                           
  // pass current Newton iterate to local assembler
  local_asm.set_newton_solution(&in);
  
  // call assemble routine for cell contributions
  this->global_asm_.should_reset_assembly_target(true); // true = set vector out to zero before assembly
  this->global_asm_.assemble_vector(this->space_, local_asm, *out);
  
  // Correct Dirichlet dofs -- set Dirichlet dofs to 0
  if (!this->dirichlet_dofs_.empty()) 
  {
    std::vector< DataType > zeros(this->dirichlet_dofs_.size(), 0.);
    out->SetValues(vec2ptr(this->dirichlet_dofs_), this->dirichlet_dofs_.size(), vec2ptr(zeros));
  }
}

void CavityNavierStokes::visualize(int continuation_step) 
{     
  std::string prefix;
  if (this->equation_type_ == 0)
  {
    prefix = "Stokes";
  }
  else 
  {
    prefix = "NavierStokes";
  }
  
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
  if (DIM >2)
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
  name << "ex7_" << prefix << "_" << refinement_level_ << "_" << continuation_step;
  
  VTKWriter< DataType, DIM> vtk_writer (visu, this->comm_, MASTER_RANK);
  vtk_writer.write(name.str());    
}

void CavityNavierStokes::adapt() {
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
    MeshPtr local_mesh = partition_and_distribute(master_mesh_, MASTER_RANK,
                                                    comm_, uniform_ref_steps);
    assert(local_mesh != 0);
    SharedVertexTable shared_verts;
    mesh_ = compute_ghost_cells(*local_mesh, comm_, shared_verts);
  }
}

typename LAD::DataType CavityNavierStokes::compute_pressure_int(VectorType &u) 
{
  DataType recv;
  DataType total_pressure;
  u.Update();

  PressureIntegral< DIM, LAD > int_p(u);
  this->global_asm_.integrate_scalar(space_, int_p, total_pressure);

  this->parcom_->sum(total_pressure, recv);
  return recv;
}

typename LAD::DataType CavityNavierStokes::compute_volume_int() 
{
  DataType integrated_vol;
  DataType recv;
  VolumeIntegral< DIM, LAD > vol_int;
  this->global_asm_.integrate_scalar(space_, vol_int, integrated_vol);

  this->parcom_->sum(integrated_vol, recv);
  return recv;
}

typename LAD::DataType CavityNavierStokes::compute_average_pressure(VectorType &u) 
{
  DataType total_pressure = compute_pressure_int(u);
  DataType integrated_vol = compute_volume_int();
  const DataType average_pressure = total_pressure / integrated_vol;

  LOG_INFO ( "Average pressure", average_pressure );

  return average_pressure;
}

void CavityNavierStokes::ApplyFilter(VectorType &u) 
{
  if (!params_["Equation"]["ApplyPressureFilter"].get< bool >())
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
