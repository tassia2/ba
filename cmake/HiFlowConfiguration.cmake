# This file is used to create the file config.h containing the
# configuration of the library.  In the process, it creates the
# options for additional libraries, and checks for their availability.
#
# Any new dependencies on external libraries should be added here. The
# generated config.h file can then be used to check for features in
# the HiFlow code.

###### Required libraries   ##############################

# Message Passing Interface implementation
find_package(MPI REQUIRED)


# Boost library
find_package(Boost REQUIRED)

# TinyXML2 library
find_package(TinyXML2 REQUIRED)

###### Optional libraries   ##############################

# PETSc
set(WITH_PETSC 0)
set(WITH_COMPLEX_PETSC 0)
option(WANT_PETSC "Compile linear algebra module with PETSc interface")
if (WANT_PETSC)
  find_package(PETSC REQUIRED)
  if (PETSC_FOUND)
    set(WITH_PETSC 1)
    add_library(petsc STATIC IMPORTED)
    set_target_properties(petsc PROPERTIES IMPORTED_LOCATION "${PETSC_LIBRARIES}")
  endif (PETSC_FOUND)
  option(PETSC_COMPLEX "Use PETSc version which is compiled to use complex numbers. CAUTION: The Hiflow interface
     for complex PETSc is not implemented yet. Use complex PETSc only for eigenvalue stuff combined with non-PETSc Linear Algebra.")
  if (PETSC_COMPLEX)
    set(WITH_COMPLEX_PETSC 1)
    set(WITH_PETSC 0)
  endif (PETSC_COMPLEX)
endif (WANT_PETSC)

# SLEPc
set(WITH_SLEPC 0)
option(WANT_SLEPC "Compile linear algebra module with PETSc interface and add SLEPc eigenvalue solver")
if (WANT_SLEPC)
  find_package(SLEPC REQUIRED)
  if (SLEPC_FOUND)
    set(WITH_SLEPC 1)
    add_library(slepc STATIC IMPORTED)
    set_target_properties(slepc PROPERTIES IMPORTED_LOCATION ${SLEPC_LIBRARIES})
  endif (SLEPC_FOUND)
endif (WANT_SLEPC)

# p4est
set(WITH_P4EST 0)
option(WANT_P4EST "Compile mesh module with p4est interface")
if (WANT_P4EST)
  find_package(P4EST REQUIRED)
  if (P4EST_FOUND)
    set(WITH_P4EST 1)
    add_library(P4EST STATIC IMPORTED)
    set_target_properties(P4EST PROPERTIES IMPORTED_LOCATION ${P4EST_LIBRARY})
    add_library(P4EST_SC STATIC IMPORTED)
    set_target_properties(P4EST_SC PROPERTIES IMPORTED_LOCATION ${P4EST_SC_LIBRARY})
    add_library(P4EST_Z STATIC IMPORTED)
    set_target_properties(P4EST_Z PROPERTIES IMPORTED_LOCATION ${P4EST_Z_LIBRARY})
  endif (P4EST_FOUND)
endif (WANT_P4EST)

# HYPRE
set(WITH_HYPRE 0)
option(WANT_HYPRE "Compile linear algebra module with HYPRE interface.")
if (WANT_HYPRE)
  find_package(HYPRE REQUIRED)
  if (HYPRE_FOUND)
    set(WITH_HYPRE 1)
    add_library(HYPRE STATIC IMPORTED)
    set_target_properties(HYPRE PROPERTIES IMPORTED_LOCATION ${HYPRE_LIBRARIES})
  endif (HYPRE_FOUND)
endif (WANT_HYPRE)

# ParMETIS
set(WITH_PARMETIS 0)
option(WANT_PARMETIS "Compile mesh module with ParMETIS partitioning.")
if (WANT_PARMETIS)
  if (NOT WANT_METIS)
    message (FATAL_ERROR "ParMETIS requires METIS!")
  endif (NOT WANT_METIS)
  find_package(ParMETIS REQUIRED)
  if (ParMETIS_FOUND)
    set(WITH_PARMETIS 1)
    add_library(parmetis SHARED IMPORTED)
    set_target_properties(parmetis PROPERTIES IMPORTED_LOCATION ${ParMETIS_LIBRARY})
  endif (ParMETIS_FOUND)
endif (WANT_PARMETIS)

# METIS
set(WITH_METIS 0)
option(WANT_METIS "Compile mesh module with METIS partitioning.")
if (WANT_METIS)
  find_package(METIS REQUIRED)
  if (METIS_FOUND)
    set(WITH_METIS 1)
    add_library(metis SHARED IMPORTED)
    set_target_properties(metis PROPERTIES IMPORTED_LOCATION ${METIS_LIBRARY})
    set(METIS_VERSION "5" CACHE STRING "Version of METIS library used (4 or 5).")
  endif (METIS_FOUND)
endif (WANT_METIS)

# VCL
set(WITH_VCL 0)
option(WANT_VCL "Compile with VCL for SIMD." )
if (WANT_VCL)
  find_package(VCL REQUIRED)
  if (VCL_FOUND)
    set(WITH_VCL 1)
    message ("found vcl ${WITH_VCL}")
  endif (VCL_FOUND)
endif (WANT_VCL)

# Likwid
set(WITH_LIKWID 0)
option(WANT_LIKWID "Compile with likwid tools for performance measruing.")
if (WANT_LIKWID)
  find_package(LIKWID REQUIRED)
  if (LIKWID_FOUND)
    set(WITH_LIKWID 1)
    add_library(likwid SHARED IMPORTED)
    set_target_properties(likwid PROPERTIES IMPORTED_LOCATION ${LIKWID_LIBRARY})
  endif (LIKWID_FOUND)
endif (WANT_LIKWID)

# GPerfTools
set(WITH_GPERF 0)
option(WANT_GPERF "Compile with GPerfTools support.")
if (WANT_GPERF)
  find_package(GPERF REQUIRED)
  if (GPERF_FOUND)
    set(WITH_GPERF 1)
    add_library(profiler SHARED IMPORTED)
    set_target_properties(profiler PROPERTIES IMPORTED_LOCATION ${GPERF_LIBRARY})
  endif (GPERF_FOUND)
endif (WANT_GPERF)

# HDF5
set(WITH_HDF5 0)
option(WANT_HDF5 "Compile with HDF5 support.")
if (WANT_HDF5)
  set(HDF5_USE_STATIC_LIBRARIES 1)
  find_package(HDF5 REQUIRED)
  if (HDF5_FOUND)
    set(WITH_HDF5 1)
    add_library(hdf5 STATIC IMPORTED)
    set_target_properties(hdf5 PROPERTIES IMPORTED_LOCATION ${HDF5_HDF5_LIBRARY})
    set_target_properties(hdf5 PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "${HDF5_HDF5_HL_LIBRARY};${HDF5_M_LIBRARY};${HDF5_Z_LIBRARY}")   
  endif (HDF5_FOUND)
endif (WANT_HDF5)

# ILU++
set(WITH_ILUPP 0)
option(WANT_ILUPP "Compile linear algebra module with ILU++ preconditioner." ON)
if (WANT_ILUPP)
  find_package(ILU++ REQUIRED)
  if (ILUPP_FOUND)
    set(WITH_ILUPP 1)
    #add_library(ilupp STATIC IMPORTED)
    #set_target_properties(ilupp PROPERTIES IMPORTED_LOCATION ${ILUPP_LIBRARY})
  endif (ILUPP_FOUND)
endif (WANT_ILUPP)

# MUMPS (shared parallel version)
set(WITH_MUMPS 0)
option(WANT_MUMPS "Compile linear algebra module with MUMPS solver support.")
if (WANT_MUMPS)
  find_package(MUMPS REQUIRED)
  if (MUMPS_FOUND)

    # MUMPS consists of 4 libraries (dmumps, smumps, mumps_common and pord), and depends on ScaLAPACK and METIS.
    # ScaLAPACK in turn depends on BLACS, which in turn depends on BLAS.
    # For static linking, we want to locate the libraries and add internal "dummy" library targets that libhiflow can depend on.
    # Unfortunately, it is not possible to define dependencies between such dummy targets, so instead we have to link HiFlow
    # to all the dependencies in a flat order. The problem is further complicated due to the fact that the BLACS library depends on a
    # BLACS_Init library, which in turn depends on BLACS again. This means that we have to link BLACS two times, which can only be accomplished
    # with the "IMPORTED_LINK_INTERFACE_LIBRARIES" property.
    #
    # In the future, we should seriously reconsider how to solve this problem properly, and also add support for linking against, and producing, shared
    # libraries. Not only will the linking be easier, but we will typically save disk space and conserve RAM as well.

    enable_language (Fortran)
    find_package(MPI REQUIRED)
    set(WITH_MUMPS 1)
    add_library(mumps SHARED IMPORTED)
    set_target_properties(mumps PROPERTIES IMPORTED_LOCATION ${MUMPS_LIBRARY})

    add_library(pord SHARED IMPORTED)
    set_target_properties(pord PROPERTIES IMPORTED_LOCATION ${MUMPS_PORD_LIBRARY})

    # Extra dependencies for MUMPS library. Normally these should not
    # be needed, due to shared library mechanism, but linking fails
    # without them...
    set_target_properties(mumps PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "${SMUMPS_LIBRARY};${MUMPS_COMMON_LIBRARY};${MPI_Fortran_LIBRARIES};gfortran")

    find_package(ScaLAPACK REQUIRED)
    add_library(scalapack SHARED IMPORTED)
    set_target_properties(scalapack PROPERTIES IMPORTED_LOCATION ${ScaLAPACK_LIBRARY})

    find_package(BLACS REQUIRED)
    add_library(blacs SHARED IMPORTED)
    set_target_properties(blacs PROPERTIES IMPORTED_LOCATION ${BLACS_LIBRARY})

    # Here we do some magic to resolve the BLACS -> BLACS_Init -> BLACS dependency
    add_library(blacs_init SHARED IMPORTED)
    set_target_properties(blacs_init PROPERTIES IMPORTED_LOCATION ${BLACS_INIT_LIBRARY})
    set_target_properties(blacs_init PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "${BLACS_LIBRARY}")

    if (NOT WITH_METIS)
      find_package(METIS REQUIRED)
      add_library(metis SHARED IMPORTED)
      set_target_properties(metis PROPERTIES IMPORTED_LOCATION ${METIS_LIBRARY})
    endif()

    find_package(BLAS REQUIRED)
    add_library(blas SHARED IMPORTED)
    set_target_properties(blas PROPERTIES IMPORTED_LOCATION ${BLAS_LIBRARY})

    # The linking order is defined in src/CMakeLists.txt

  endif()
endif()

# UMFPACK
set(WITH_UMFPACK 0)
option(WANT_UMFPACK "Compile linear algebra module with UMFPACK solver support.")
if (WANT_UMFPACK)
  find_package(UMFPACK REQUIRED)
  if (UMFPACK_FOUND)
    set(WITH_UMFPACK 1)
    add_library(umfpack SHARED IMPORTED)
    set_target_properties(umfpack PROPERTIES IMPORTED_LOCATION ${UMFPACK_LIBRARY})
    set_target_properties(umfpack PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "${AMD_LIBRARY};blas")
  endif()
endif()

# MKL
set(WITH_MKL 0)
option(WANT_MKL "Compile linear algebra with MKL support.")
if (WANT_MKL)
  find_package(MKL REQUIRED)
  if (MKL_FOUND)
    set(WITH_MKL 1)
    add_library(mkl SHARED IMPORTED)

    # From MKL link advisor: http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
    # "main" library is mkl_intel_lp64.so, extra libraries to be included on link line are listed below.
    set_target_properties(mkl PROPERTIES IMPORTED_LOCATION ${MKL_LIBRARY})
    set_target_properties(mkl PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "mkl_intel_thread;mkl_core;iomp5;pthread")

    # workaround to find directory containing library dependencies
    get_filename_component(MKL_LIB_PATH ${MKL_LIBRARY} PATH)
    link_directories(${MKL_LIB_PATH})

  else ()
    message("MKL library not found.")
  endif ()
endif (WANT_MKL)


# CBLAS
set(WITH_CBLAS 0)
option(WANT_CBLAS "Compile linear algebra module with CBLAS support.")
if (WANT_CBLAS)
  find_package(CBLAS REQUIRED)
  if (CBLAS_FOUND)
    set(WITH_CBLAS 1)
    #add_library(cblas SHARED IMPORTED)
    #set_target_properties(cblas PROPERTIES IMPORTED_LOCATION ${CBLAS_LIBRARY})
  endif (CBLAS_FOUND)
endif (WANT_CBLAS)

set(WITH_CLAPACK 0)
option(WANT_CLAPACK "Compile linear algebra module with CLAPACK support.")
if (WANT_CLAPACK)
  # This imports the targets from the clapack CMake build.
  find_package(clapack REQUIRED)
  if(CLAPACK_FOUND)
    set(WITH_CLAPACK 1)
  endif()
endif (WANT_CLAPACK)

# CUDA
set(WITH_CUDA 0)
option(WANT_CUDA "Compile linear algebra with CUDA support.")
if (WANT_CUDA)
  find_package(CUDA REQUIRED)
  if(CUDA_FOUND)
    set(WITH_CUDA 1)

    # No propagation of host compiler flags to CUDA compiler via -Xcompiler
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)

    # Set compiler compiler to be used by nvcc
    set(CUDA_HOST_COMPILER g++)

    # Set flags for CUDA compiler
    #list(APPEND CUDA_NVCC_FLAGS "-arch=compute_20 -code=compute_20,sm_20 --compiler-options -Wall -I${PROJECT_SOURCE_DIR}/src/linear_algebra/lmp")
    list(APPEND CUDA_NVCC_FLAGS "-I${PROJECT_SOURCE_DIR}/src/linear_algebra/lmp")
    list(APPEND CUDA_NVCC_FLAGS_RELEASE "-O3")
    list(APPEND CUDA_NVCC_FLAGS_DEBUG "-O0 -g -G -DDEBUG_MODE")
    list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O3 -g -G -DDEBUG_MODE")
    list(APPEND CUDA_NVCC_FLAGS_PROFILING "-O3 -g -G -pg -DDEBUG_MODE")

    include_directories(${CUDA_INCLUDE_DIRS})

    #set(CUDA_NVCC_FLAGS -arch=compute_13 --host-compilation C++ -I${PROJECT_SOURCE_DIR}/src/linear_algebra/lmp)
    #set(CUDA_NVCC_FLAGS -arch=compute_13 -I${PROJECT_SOURCE_DIR}/src/linear_algebra/lmp)
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch sm_20 --compiler-options -Wall")
    #set(CUDA_NVCC_FLAGS " -O3 -arch sm_20 --compiler-options -Wall -I${PROJECT_SOURCE_DIR}/src/linear_algebra/lmp")

    #cuda_include_directories(/usr/local/cuda/sdk/C/common/inc)     # @@@ TOFIX
    #cuda_include_directories(/opt/nvidia/sdk_2.3b/C/common/inc)     # @@@ TOFIX
    #cuda_include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)     @@@ TOFIX

  endif(CUDA_FOUND)
endif(WANT_CUDA)

# OpenCL
set(WITH_OPENCL 0)
option(WANT_OPENCL "Compile linear algebra with OpenCL support.")
if (WANT_OPENCL)
  find_package(OpenCL REQUIRED)
  if (OPENCL_FOUND)
    set(WITH_OPENCL 1)
  endif(OPENCL_FOUND)
endif(WANT_OPENCL)


# OpenMP
set(WITH_OPENMP 0)
option(WANT_OPENMP "Compile with OpenMP support.")
if (WANT_OPENMP)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    set(WITH_OPENMP 1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# GAUSSQ
set(WITH_GAUSSQ 0)
option(WANT_GAUSSQ "Compile with GAUSSQ Fortran support.")
if (WANT_GAUSSQ)
  set(WITH_GAUSSQ 1)
endif()


####### Compiler options #############################
if (CMAKE_CXX_COMPILER MATCHES g[+][+].* OR CMAKE_CXX_COMPILER MATCHES c[+][+].*)
  set(GCC_COMPILER 1)
else()
  set(GCC_COMPILER 0)
endif()

if (CMAKE_CXX_COMPILER MATCHES icpc.*)
  set(INTEL_COMPILER 1)
else() 
  set(INTEL_COMPILER 0)
endif()

if (CMAKE_CXX_COMPILER MATCHES pgCC.*)
  set(PGI_COMPILER 1)
else ()
  set(PGI_COMPILER 0)
endif()

if (CMAKE_CXX_COMPILER MATCHES clang[+][+].*)
  set(CLANG_COMPILER 1)
else()
  set(CLANG_COMPILER 0)
endif()

###### Installation Options   ##############################
# Examples
option(BUILD_EXAMPLES "Build and install example code." OFF)
option(BUILD_EXERCISES "Build and install exercises ." ON)
option(BUILD_TESTS "Build tests." OFF)
option(RUN_TEDIOUS_TESTS "Build and run tests that take a long time." OFF)
option(BUILD_UTILS "Build and install utility programs and scripts." OFF)
option(USE_SIMD_VECTOR_ALGEBRA "use SIMD vector algebra objects." ON)
option(INFO_LOG_ACTIVE "Activate INFO log." ON)
option(EXPLICIT_TEMPLATE_INSTANTS "Compile test instances for header only template definitions (for developers)" OFF)
set(DEBUG_LEVEL 0 CACHE STRING "Level of Debug output")

if (USE_SIMD_VECTOR_ALGEBRA)
  set(WITH_SIMD_VECTOR_ALGEBRA 1)
else() 
  set(WITH_SIMD_VECTOR_ALGEBRA 0)
endif()


# Create file config.h #########################################
configure_file(${HIFLOW_SOURCE_DIR}/src/config.h.in ${HIFLOW_BINARY_DIR}/src/include/config.h)
