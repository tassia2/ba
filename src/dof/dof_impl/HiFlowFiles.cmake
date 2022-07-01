set(dof_impl_SOURCES
  dof_container.cc
  dof_container_lagrange.cc
  dof_container_rt_bdm.cc
)

# the following source files are only included for developping purpose
if(EXPLICIT_TEMPLATE_INSTANTS)
  list(APPEND dof_impl_SOURCES   
    dof_impl.cc
)
endif()

set(dof_impl_PUBLIC_HEADERS
 dof_container.h
 dof_container_lagrange.h
 dof_container_rt_bdm.h
 dof_functional.h
 dof_functional_facet_normal_moment.h
 dof_functional_cell_moment.h
 dof_functional_point.h
 )
