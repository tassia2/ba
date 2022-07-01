set(fem_SOURCES
  fe_instance.cc
  fe_manager.cc
  fe_transformation.cc
  fe_reference.cc
)

# the following source files are only included for developping purpose
if(EXPLICIT_TEMPLATE_INSTANTS)
  list(APPEND fem_SOURCES   
    fem.cc
  )
endif()

set(fem_PUBLIC_HEADERS
  fe_instance.h
  fe_manager.h
  fe_mapping.h
  fe_reference.h
  fe_transformation.h 
  function_space.h 
  reference_cell.h)


# add files from specific directory
include(fem/cell_trafo/HiFlowFiles.cmake)

foreach (i ${cell_trafo_SOURCES})
  list(APPEND fem_SOURCES "cell_trafo/${i}")
endforeach()

foreach (i ${cell_trafo_PUBLIC_HEADERS})
  list(APPEND fem_PUBLIC_HEADERS "cell_trafo/${i}")
endforeach()

include(fem/ansatz/HiFlowFiles.cmake)

foreach (i ${ansatz_SOURCES})
  list(APPEND fem_SOURCES "ansatz/${i}")
endforeach()

foreach (i ${ansatz_PUBLIC_HEADERS})
  list(APPEND fem_PUBLIC_HEADERS "ansatz/${i}")
endforeach()

include(fem/flux_correction/HiFlowFiles.cmake)

foreach (i ${flux_correction_SOURCES})
  list(APPEND fem_SOURCES "flux_correction/${i}")
endforeach()

foreach (i ${flux_correction_PUBLIC_HEADERS})
  list(APPEND fem_PUBLIC_HEADERS "flux_correction/${i}")
endforeach()

include(fem/cut_fem/HiFlowFiles.cmake)

foreach (i ${cut_fem_SOURCES})
  list(APPEND fem_SOURCES "cut_fem/${i}")
endforeach()

foreach (i ${cut_fem_PUBLIC_HEADERS})
  list(APPEND fem_PUBLIC_HEADERS "cut_fem/${i}")
endforeach()
