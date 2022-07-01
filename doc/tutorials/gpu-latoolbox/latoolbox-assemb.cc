void ConvDiff::assemble_system() {
  TimingScope tscope("Assemble system.");
  ConvDiffAssembler local_asm(beta_, nu_);
  StandardGlobalAssembler< double > global_asm;

  global_asm.assemble_matrix(space_, local_asm, matrix_);
  global_asm.assemble_vector(space_, local_asm, rhs_);

  // treatment of dirichlet boudary dofs
  if (!dirichlet_dofs_.empty()) {
    matrix_.diagonalize_rows(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                             static_cast< Scalar >(1.));
    rhs_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                   vec2ptr(dirichlet_values_));
    sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                   vec2ptr(dirichlet_values_));
  }

  rhs_.UpdateCouplings();
  sol_.UpdateCouplings();

  dev_matrix_->CopyStructureFrom(matrix_);
  dev_sol_->CopyStructureFrom(sol_);
  dev_rhs_->CopyStructureFrom(rhs_);

  dev_sol_->CopyFrom(sol_);
  dev_rhs_->CopyFrom(rhs_);
  dev_matrix_->CopyFrom(matrix_);
}
