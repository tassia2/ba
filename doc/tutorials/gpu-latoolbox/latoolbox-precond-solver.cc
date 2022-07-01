void ConvDiff::set_up_preconditioner() {
  TimingScope tscope("Set up preconditioner");
  if (matrix_precond_ == JACOBI) {
    PreconditionerBlockJacobiStand< LAD > *precond;
    precond = new PreconditionerBlockJacobiStand< LAD >;

    precond->Init_Jacobi(*dev_sol_);

    precond_ = precond;
  } else {
    PreconditionerMultiColoring< LAD > *precond;
    precond = new PreconditionerMultiColoring< LAD >;
    if (matrix_precond_ == GAUSS_SEIDEL) {
      precond->Init_GaussSeidel();
    } else if (matrix_precond_ == SGAUSS_SEIDEL) {
      precond->Init_SymmetricGaussSeidel();
    } else if (matrix_precond_ == ILU) {
      precond->Init_ILU(0);
    } else if (matrix_precond_ == ILU2) {
      int param1 = param_["LinearAlgebra"]["ILU2Param1"].get< int >();
      int param2 = param_["LinearAlgebra"]["ILU2Param2"].get< int >();
      precond->Init_ILU(param1, param2);
    }
    precond->Preprocess(*dev_matrix_, *dev_sol_, &(space_.dof()));
    // sync
    MPI_Barrier(comm_);

    prepare_lin_alg_structures();

    prepare_bc();

    assemble_system();

    precond_ = precond;
  }
  precond_->SetupOperator(*dev_matrix_);
  precond_->Build();
  precond_->Print();

  linear_solver_->SetupPreconditioner(*precond_);
}

void ConvDiff::solve_system() {
  linear_solver_->SetupOperator(*dev_matrix_);
  // sync
  MPI_Barrier(comm_);
  {
    TimingScope tscope("Solve system.");

    linear_solver_->Solve(*dev_rhs_, dev_sol_);
  }
  // sync
  MPI_Barrier(comm_);

  sol_.CopyFrom(*dev_sol_);
}
