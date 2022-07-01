# ======= TESTS ========
#
# This file contains the programs that make up the test suite.  These
# should all verify that some aspect of the various modules function
# correctly.  If a test succeeds, it should return 0, and otherwise it
# should return -1. The include test.h includes some useful C macros
# for writing tests.
#
# The naming convention of tests is that they should exist in a source file with the suffix
# _test.cc . The name of the test is the source filename without this suffix.
#

#link_directories(${PROJECT_BINARY_DIR}/contrib/unittest++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIFLOW_VENDOR_FLAGS} ${USED_WARNING_FLAGS}")

# Common defines
add_definitions(-DMESH_DATADIR="${PROJECT_SOURCE_DIR}/tests/data/")
add_definitions(-DHIFLOW_DIR="${PROJECT_SOURCE_DIR}")

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/tools)
include_directories(${PROJECT_BINARY_DIR}/src/include)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})

if (WITH_ILUPP)
  include_directories(${ILUPP_INCLUDE_DIR})
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

if (WITH_MUMPS)
  include_directories(${MUMPS_INCLUDE_DIR})
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

if (WITH_HDF5)
  include_directories(${HDF5_INCLUDE_DIR})
endif()

if (WITH_HYPRE)
  include_directories(${HYPRE_INCLUDE_DIR})
  link_directories(${HYPRE_LIBRARIES})
endif()

if (WITH_PETSC)
  include_directories(${PETSC_INCLUDE_DIR})
endif()

if (WITH_SLEPC)
  include_directories(${SLEPC_INCLUDE_DIR})
endif()

if(WITH_COMPLEX_PETSC)
  include_directories(${PETSC_INCLUDE_DIR})
endif()

if(WITH_P4EST)
  include_directories(${P4EST_INCLUDE_DIR})
endif()

if(WITH_OPENCL)
  include_directories(${OPENCL_INCLUDE_DIR})
endif()

if (WITH_VCL)
  include_directories(${VCL_INCLUDE_DIR})
endif()

# This macros can be used to define a test. It takes as input the name
# of the test (without the _test.cc suffix), and the HiFlow module
# libraries that the test depends on in a space-delimited string. It
# creates a new executable target and adds it to the list of tests
# that are executed when "make test" is run.

macro(def_test TEST_NAME)
  if (${ARGC} GREATER 1)
    set(N ${ARGV1})
  elseif(${ARGC} EQUAL 1)
    set(N 1)
  endif()

  add_executable(${TEST_NAME}_test ${TEST_NAME}_test.cc)
  target_link_libraries(${TEST_NAME}_test hiflow ${TINYXML2_LIBRARIES})
  if (WITH_HDF5)
    target_link_libraries(${TEST_NAME}_test pthread hdf5 dl)
  endif()
  if(WITH_PETSC)
    target_link_libraries(${TEST_NAME}_test ${PETSC_LIBRARIES})
  endif()
  if (WITH_GPERF)
    target_link_libraries(${TEST_NAME}_test profiler)
  endif()
  if (WITH_GPERF)
    target_link_libraries(${TEST_NAME}_test ${GPERF_LIBRARY})
  endif()
  if (WITH_LIKWID)
    target_link_libraries(${TEST_NAME}_test likwid )
  endif()
  #add_test(${TEST_NAME} ${TEST_NAME}_test)
  add_Test(${TEST_NAME} ${MPIEXEC} -np ${N} ${PROJECT_BINARY_DIR}/tests/${TEST_NAME}_test)
endmacro(def_test)

# TODO: check which of these are real tests, in that they can fail...
# "false" tests are marked with NOTEST

# Linear Algebra test
def_test(coupled_matrix_dense 2)
def_test(coupled_matrix_mesh 2)
def_test(seq_dense_matrix)
def_test(block_matrix 2)
def_test(block_vector 2)
def_test(parallel_vector 2)
def_test(parallel_matrix 2)
def_test(lmp)
def_test(cg)
def_test(simd)

def_test(mpi 2)                  # NOTEST
def_test(vector_algebra)          # TODO: implement tests

# Dof tests
def_test(dof_identification)      # TODO: k>=3 for RT and BDM

##def_test(dof_extended_identification_2D)  # needs update with no "get_coords..." function
##def_test(dof_extended_identification_3D)
##def_test(parallel_adaptive_jump_term)      # NOTEST, needs update with "get_coords..." function

# FEM tests
def_test(fe_weight_ordering)    # TODO: Pyramid
def_test(fe_weight)             # TODO: Pyramid
def_test(fe_nodal)              # TODO: k>=3 for RT and BDM
def_test(fe_hdiv)               # TODO: k>=3 for RT and BDM
def_test(fe_bdm1)               # NOTEST
def_test(fe_transformation)     # TODO: k>=3 for RT and BDM
def_test(transformation)        # NOTEST (OK). ToDO: Zu richtigem Test ausbauen....
def_test(parallel_dof 2)       # TODO: k>=3 for RT and BDM !!!!!
def_test(point_evaluator 2) 
def_test(cell_transformation)
def_test(geometric_tools)


## Mesh tests
def_test(attributes)           
def_test(boundary_extraction)   
def_test(cell_type)            
def_test(incidence_iteration)   
def_test(material_number)       
def_test(mesh_interface)        # OK
def_test(parallel_vtk_reader)   # NOTEST
def_test(closest_point)         
def_test(reader)              # NOTEST
def_test(vertex_lookup)       # NOTEST
def_test(vtk_reader)          # NOTEST
def_test(writer)              # NOTEST
def_test(incident_vertex_iterator 4)

# Quadrature tests
def_test(quadrature_order)    
def_test(facet_integral)     

# Vector space tests
def_test(binary_io)           
def_test(tensor)              

# Assembly tests
def_test(hp_assembly)
def_test(facet_normal)
# HDF5 tests
if (WITH_HDF5)
  def_test(hdf5)
  def_test(parallel_hdf5 2)
  def_test(parallel_hdf5_blockLA 2) 
endif()

# CUDA
if(WITH_CUDA)
  def_test(cuda)
  target_link_libraries(cuda_test UnitTest++ ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
endif()

# PETSc
if(WITH_PETSC)
  def_test(petsc_matrix)
  def_test(petsc_vector)
  def_test(petsc_linear_solver)
endif()

if (RUN_TEDIOUS_TESTS)
  def_test(big_mesh)    
  def_test(integration)       # TODO: Other geometries than HEXA not yet implemented)
  def_test(bdry_integration)  # TODO: Other geometries than HEXA not yet implemented)
  def_test(refinement)        # FEATURE 90,
  def_test(tetra_refinement)  # FEATURE 91 leads to internal compiler error
  def_test(fe_interpol_conv 1)   # NOTEST TODO: k>=3 for RT and BDM 
  def_test(fe_projection_conv)    # NOTEST TODO: k>=3 for RT and BDM
  def_test(fe_interpol_eval 1)   # TODO: k>=3 for RT and BDM , 
endif()


# This macros can be used to define a test. It takes as input the name
# of the test (without the _test.cc suffix), and the HiFlow module
# libraries that the test depends on in a space-delimited string. It
# creates a new executable target and adds it to the list of tests
# that are executed when "make test" is run.


# TODO: check which of these are real tests, in that they can fail...
# "false" tests are marked with NOTEST

