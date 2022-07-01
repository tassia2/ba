set(dof_SOURCES
  fe_interface_pattern.cc 
  dof_interpolation.cc 
  dof_interpolation_pattern.cc
  dof_partition.cc
  numbering_strategy.cc
  numbering_lagrange.cc
)

set(dof_PUBLIC_HEADERS
  dof_fem_types.h
  dof_partition.h 
  dof_interpolation.h 
  dof_interpolation_pattern.h 
  fe_interface_pattern.h 
  numbering_strategy.h
  numbering_lagrange.h)


# add files from specific directory
include(dof/dof_impl/HiFlowFiles.cmake)

foreach (i ${dof_impl_SOURCES})
  list(APPEND dof_SOURCES "dof_impl/${i}")
endforeach()

foreach (i ${dof_impl_PUBLIC_HEADERS})
  list(APPEND dof_PUBLIC_HEADERS "dof_impl/${i}")
endforeach()
