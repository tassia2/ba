set(linear_solver_SOURCES
  linear_solver.cc
  preconditioner_mc.cc
  preconditioner_bjacobi_standard.cc
  preconditioner_vanka.cc
  regularized_matrix_solver.cc
  schur_complement_light.cc
  schur_complement.cc
  single_subblock_solver.cc
  pcd_preconditioner.cc
  mpir.cc
  jacobi.cc
  tri_block_solver.cc
  diag_block_solver.cc
  pce_multilevel.cc
)

set(linear_solver_PUBLIC_HEADERS
  bicgstab.h
  cg.h
  fgmres.h
  gmg.h
  gmres.h
  linear_solver.h
  linear_solver_setup.h
  preconditioner.h
  preconditioner_ilupp.h
  preconditioner_bjacobi.h
  preconditioner_bjacobi_standard.h
  preconditioner_bjacobi_matfree.h
  preconditioner_bjacobi_ext.h
  preconditioner_ilupp.h
  preconditioner_vanka.h
  preconditioner_mc.h
  linear_solver_creator.h
  linear_solver_factory.h
  preconditioner.h
  regularized_matrix_solver.h
  richardson.h
  pcd_preconditioner.h
  schur_complement_light.h
  schur_complement.h
  single_subblock_solver.h
  mpir.h
  jacobi.h
  tri_block_solver.h
  diag_block_solver.h
  pce_multilevel.h
)

if (WITH_HYPRE)
  list(APPEND linear_solver_SOURCES
      hypre_cg.cc
      hypre_bicgstab.cc
      hypre_boomer_amg.cc
      hypre_gmres.cc
      hypre_preconditioner_euclid.cc
      hypre_preconditioner_parasails.cc
  )
  list(APPEND linear_solver_PUBLIC_HEADERS
      hypre_linear_solver.h
      hypre_preconditioner.h
      hypre_cg.h
      hypre_bicgstab.h
      hypre_boomer_amg.h
      hypre_gmres.h
      hypre_preconditioner_euclid.h
      hypre_preconditioner_parasails.h
  )
endif()

if (WITH_PETSC)
  list(APPEND linear_solver_SOURCES
    petsc_general_ksp.cc
  )
  list(APPEND linear_solver_PUBLIC_HEADERS
    petsc_general_ksp.h
    petsc_linear_solver.h
    petsc_preconditioner.h
  )
endif()

# add files from specific directory
include(linear_solver/amg/HiFlowFiles.cmake)

foreach (i ${amg_PUBLIC_HEADERS})
  list(APPEND linear_solver_PUBLIC_HEADERS "amg/${i}")
endforeach()

# add files from specific directory
#include(linear_solver/gmg/HiFlowFiles.cmake)

#foreach (i ${gmg_SOURCES})
#  list(APPEND linear_solver_SOURCES "gmg/${i}")
#endforeach()

#foreach (i ${gmg_PUBLIC_HEADERS})
#  list(APPEND linear_solver_PUBLIC_HEADERS "gmg/${i}")
#endforeach()

include(linear_solver/gpu-based/HiFlowFiles.cmake)

foreach (i ${gpu-based_SOURCES})
  list(APPEND linear_solver_SOURCES "gpu-based/${i}")
endforeach()

foreach (i ${gpu-based_PUBLIC_HEADERS})
  list(APPEND linear_solver_PUBLIC_HEADERS "gpu-based/${i}")
endforeach()

if (${WITH_MUMPS})
  list(APPEND linear_solver_SOURCES mumps_solver.cc)
  list(APPEND linear_solver_PUBLIC_HEADERS mumps_solver.h mumps_structure.h)
endif()

#if (${WITH_ILUPP})
#  list(APPEND linear_solver_SOURCES preconditioner_ilupp.cc)
#  list(APPEND linear_solver_PUBLIC_HEADERS preconditioner_ilupp.h)
#endif()

#if (${WITH_UMFPACK})
#  list(APPEND linear_solver_SOURCES umfpack_solver.cc)
#  list(APPEND linear_solver_PUBLIC_HEADERS umfpack_solver.h)
#endif()
