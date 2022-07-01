CPU_lMatrix< double > mat_cpu;
// Read matrix from a file
mat_cpu.ReadFile('matrix.mtx');

lVector< double > *x, *y;
// init a vector for a specific platform and implementation
x = init_vector< double >(size, "vec x", platform, impl);
// clone y as x
y = x->CloneWithoutContent();

lMatrix< double > *mat;
// init empty matrix on a specific platform
// (nnz,nrow,ncol,name,platform,implementation,format)
mat = init_matrix< double >(0, 0, 0, "A", platform, impl, CSR);
// Copy the sparse structure of the matrix
mat->CopyStructureFrom(mat_cpu);
// Copy only the values of the matrix
mat->CopyFrom(mat_cpu);

...

    // Usage of BLAS 1 routines
    y->CopyFrom(*x); // y = x
y->Axpy(*x, 2.3);    // y = y + 2.3*x
x->Scale(6.0);       // x = x * 6.0
// print the scalar product of x and y
cout << y->dot(*x);

// Usage of BLAS 2 routines
mat->VectorMult(*y, x); // x = mat*y

...

    delete x,
    y, mat;
