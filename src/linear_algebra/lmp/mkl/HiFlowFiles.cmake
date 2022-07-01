set(mkl_SOURCES
  CPUmkl_lvector.cc  
  CPUmkl_csr_lmatrix.cc
  CPUmkl_coo_lmatrix.cc
 )

set(mkl_PUBLIC_HEADERS 
  CPUmkl_lvector.h  
  CPUmkl_csr_lmatrix.h
  CPUmkl_coo_lmatrix.h
  CPUmkl_stencil.h
  CPUmkl_blas_routines.h
)
