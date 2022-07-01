CMatrix matrix_, *dev_matrix_;
CVector sol_, rhs_, *dev_sol_, *dev_rhs_;

void ConvDiff::initialize_platform() {
  // init platform for matrix
  init_platform_mat< CMatrix >(la_sys_.Platform, matrix_impl_,
                               la_matrix_format_, matrix_precond_, comm_,
                               couplings_, &matrix_, &dev_matrix_, la_sys_);
  // init platform for solution and right hand side vectors
  init_platform_vec< CVector >(la_sys_.Platform, vector_impl_, comm_,
                               couplings_, &sol_, &dev_sol_, la_sys_);
  init_platform_vec< CVector >(la_sys_.Platform, vector_impl_, comm_,
                               couplings_, &rhs_, &dev_rhs_, la_sys_);
  init_platform_ = false;
}

ConvDiff::~ConvDiff() {
  // Delete Platform Mat/Vec
  if (la_sys_.Platform != CPU) {
    // matrix
    delete dev_matrix_;

    // vector
    delete dev_sol_;
    delete dev_rhs_;
  }
}
