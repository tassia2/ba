set(lmp_SOURCES
  lvector.cc
  lvector_cpu.cc
  mmio.cc
  lmatrix.cc
  CPUsimple_lvector.cc
  CPUsimple_csr_lmatrix.cc
  CPUsimple_coo_lmatrix.cc
  CPUsimple_dense_lmatrix.cc
  lmatrix_coo.cc
  lmatrix_csr.cc
  lmatrix_csr_cpu.cc
  lmatrix_coo_cpu.cc
  lmatrix_dense_cpu.cc
  lmatrix_dense.cc
  lpreconditioner.cc
  lpreconditioner_ext_lu_mkl.cc
  lpreconditioner_ext_lu_umfpack.cc
  lpreconditioner_ext_ilu_mkl.cc
  lpreconditioner_ext_ilu_pp.cc
  lpreconditioner_mc.cc  
  lpreconditioner_mc_sgs.cc
  lpreconditioner_mc_gs.cc
  lpreconditioner_mc_ilup.cc
  init_vec_mat.cc
  platform_management.cc
  lmp_mem.cc
  solvers/cg.cc)

# why? 
# g++ vs gcc
# warning: deprecated conversion from string constant to ‘char*
#set_source_files_properties(linear_algebra/lmp/mmio.cc PROPERTIES COMPILE_FLAGS "-Wno-write-strings")

set(lmp_PUBLIC_HEADERS 
  la_global.h
  lmatrix.h
  lmatrix_coo.h
  lmatrix_coo_cpu.h
  lmatrix_csr_cpu.h  
  lmatrix_csr.h
  lvector.h
  lvector_cpu.h
  cuda/lvector_gpu.h
  init_vec_mat.h
  lmatrix_formats.h
  lpreconditioner.h
  lpreconditioner_ext.h
  lpreconditioner_mc.h
  lpreconditioner_mc_gs.h
  lpreconditioner_mc_sgs.h
  lpreconditioner_mc_ilup.h
  platform_management.h
  lmp_log.h
  CPUsimple_stencil.h
)

# Recurse into CUDA directory
include(linear_algebra/lmp/cuda/HiFlowFiles.cmake)
foreach(i ${cuda_SOURCES})
  list(APPEND lmp_SOURCES "cuda/${i}")
endforeach()

foreach(i ${cuda_PUBLIC_HEADERS})
  list(APPEND lmp_PUBLIC_HEADERS "cuda/${i}")
endforeach()

# Recurse into OpenCL directory
include(linear_algebra/lmp/opencl/HiFlowFiles.cmake)
foreach(i ${opencl_SOURCES})
  list(APPEND lmp_SOURCES "opencl/${i}")
endforeach()

foreach(i ${opencl_PUBLIC_HEADERS})
  list(APPEND lmp_PUBLIC_HEADERS "opencl/${i}")
endforeach()

# Recurse into MKL directory
include(linear_algebra/lmp/mkl/HiFlowFiles.cmake)
foreach(i ${mkl_SOURCES})
  list(APPEND lmp_SOURCES "mkl/${i}")
endforeach()

foreach(i ${mkl_PUBLIC_HEADERS})
  list(APPEND lmp_PUBLIC_HEADERS "mkl/${i}")
endforeach()

# Recurse into SIMD directory
include(linear_algebra/lmp/simd/HiFlowFiles.cmake)
foreach(i ${simd_SOURCES})
  list(APPEND lmp_SOURCES "simd/${i}")
endforeach()

foreach(i ${simd_PUBLIC_HEADERS})
  list(APPEND lmp_PUBLIC_HEADERS "simd/${i}")
endforeach()

# For other sub-directories: include source code directly here.
# TODO shouldn't we check for options here, and only compile the necessary files?
list(APPEND lmp_SOURCES cblas/CPUcblas_lvector.cc)

list(APPEND lmp_SOURCES
  openmp/CPUopenmp_lvector.cc 
  openmp/CPUopenmp_csr_lmatrix.cc 
  openmp/CPUopenmp_coo_lmatrix.cc)
