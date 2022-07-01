set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIFLOW_VENDOR_FLAGS} ${USED_WARNING_FLAGS} ")
message("${USED_WARNING_FLAGS}")
include_directories(${PROJECT_BINARY_DIR}/src/include)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})

if (WITH_ILUPP)
  include_directories(${ILUPP_INCLUDE_DIR})
  link_directories(${ILUPP_LIBRARY})
endif()

if (WITH_MUMPS)
  include_directories(${MUMPS_INCLUDE_DIR})
  link_directories(${MUMPS_LIBRARY})
  link_directories(${SMUMPS_LIBRARY})
  link_directories(${MUMPS_COMMON_LIBRARY})
  link_directories(${MUMPS_PORD_LIBRARY})
endif()

if (WITH_UMFPACK)
  include_directories(${UMFPACK_INCLUDE_DIR})
endif()

if(WITH_CBLAS)
  include_directories(${CBLAS_INCLUDE_DIR})
  link_directories(${CBLAS_LIBRARY})
endif()

if(WITH_CLAPACK)
  include_directories(${CBLAS_INCLUDE_DIR})
  include_directories(${CLAPACK_INCLUDE_DIR})
  link_directories(${CLAPACK_LIBRARIES})
  link_directories(${CBLAS_LIBRARY})
endif()

if (WITH_MKL)
  include_directories(${MKL_INCLUDE_DIR})
  link_directories(${MKL_LIBRARY})
endif()

# HDF5
if (WITH_HDF5)
  include_directories(${HDF5_INCLUDE_DIR})
  link_directories(${HDF5_LIBRARY_DIRS})
endif()

if (WITH_LIKWID)
  include_directories(${LIKWID_INCLUDE_DIR})
  link_directories(${LIKWID_LIBRARY})
  add_definitions(-DLIKWID_PERFMON)
endif()

if (WITH_GPERF)
  include_directories(${GPERF_INCLUDE_DIR})
  link_directories(${GPERF_LIBRARY})
endif()

if (WITH_CUDA)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
endif()

if (WITH_HYPRE)
  include_directories(${HYPRE_INCLUDE_DIR})
  link_directories(${HYPRE_LIBRARIES})
endif()

if(WITH_OPENCL)
  include_directories(${OPENCL_INCLUDE_DIR})
endif()

if(WITH_PETSC)
  include_directories(${PETSC_INCLUDE_DIR})
endif()

if(WITH_COMPLEX_PETSC)
  include_directories(${PETSC_INCLUDE_DIR})
endif()

if(WITH_SLEPC)
  include_directories(${SLEPC_INCLUDE_DIR})
endif()

if (WITH_VCL)
  include_directories(${VCL_INCLUDE_DIR})
endif()

if(WITH_P4EST)
  include_directories(${P4EST_INCLUDE_DIR})
  link_directories(${P4EST_SC_LIBRARIES})
  link_directories(${P4EST_Z_LIBRARIES})
  link_directories(${P4EST_LIBRARIES})
endif()

add_definitions(-DMESHES_DATADIR="${PROJECT_BINARY_DIR}/examples/data/")
add_definitions(-DHIFLOW_DIR="${PROJECT_SOURCE_DIR}")

set(SUBDIRS_EXAMPLES
  #met_flow
  #data
  boussinesq2d
  #assembly                #(OK?)
  #benchmarks              #TODO: Point Evaluator
  #benchmark_local_solver  #(OK?)
  #blood_flow              #(OK?)
  #convection_diffusion    #check visualisation in instationary_convdiff_tutorial, TODO: DG
  #diffusion_dual_mixed
  ###direct_inverse          #(compiler bugs)
  ###distributed_control_poisson #(compiler bugs)
  #elasticity              #(OK?)
  #instationary_flow       #(OK?)
  #laminar_flow
  #laplace                 #(OK)
  ###mesh
  #mixed_precision
  #natural_convection_mixed
  #newton                  #(OK?)
  #poisson                 #(OK)
  #poisson_boltzmann
  ###poisson_eigenvalue
  #poisson_periodic        #data file does not exist, change it
  #poisson_uncertainty     #(compiler bugs) #TODO: CalculateExpectiationValues/-Variance, Linking error??
  #poisson_adaptive        
  #poisson_precond         #(OK?)
  #poisson_dg
  ###poisson_lecture
  #porous_media            #(OK?)
  #poisson_chaospy
  #poisson_matrix
  #poisson_interface
  #fem_fct
)

# Recurse through sub-directories and gather information for libhiflow
foreach(dir ${SUBDIRS_EXAMPLES})

  include(${dir}/HiFlowFiles.cmake)

  make_directory(${PROJECT_BINARY_DIR}/examples/${dir})

  foreach(i ${${dir}_SOURCES})
    add_executable(${i} ${dir}/${i}.cc)
    set_target_properties(${i} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${dir})
    target_link_libraries(${i} hiflow ${TINYXML2_LIBRARIES})
    if (WITH_HDF5)
      target_link_libraries(${i} pthread hdf5 dl)
    endif()
    if (WITH_GPERF)
      target_link_libraries(${i} profiler)
    endif()
    if(WITH_PETSC)
      target_link_libraries(${i} ${PETSC_LIBRARIES})
    endif()
    if (WITH_GPERF)
      target_link_libraries(${i} ${GPERF_LIBRARY})
    endif()
    if (WITH_LIKWID)
    target_link_libraries(${i} likwid )
  endif()
  endforeach()

  foreach(i ${${dir}_ADDITIONAL_FILES})
    file(COPY ${dir}/${i}
      DESTINATION ${PROJECT_BINARY_DIR}/examples/${dir})
  endforeach()

  string(TOUPPER "${dir}" DIRECTORY)

endforeach()
