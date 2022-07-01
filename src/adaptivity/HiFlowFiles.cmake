set(adaptivity_SOURCES
)

# the following source files are only included for developping purpose
if(EXPLICIT_TEMPLATE_INSTANTS)
  list(APPEND adaptivity_SOURCES   
    adaptivity.cc
)
endif()

set(adaptivity_PUBLIC_HEADERS
  refinement_strategies.h
  space_patch_interpolation.h
  time_patch_interpolation.h
  time_mesh.h
  dynamic_mesh_handler.h
  dynamic_mesh_problem.h
)

