set(cell_trafo_SOURCES
  linear_line_surface_transformation.cc
  linear_triangle_surface_transformation.cc
  bilinear_quad_surface_transformation.cc
)

# the following source files are only included for developping purpose
if(EXPLICIT_TEMPLATE_INSTANTS)
  list(APPEND cell_trafo_SOURCES   
    cell_trafo.cc
)
endif()

set(cell_trafo_PUBLIC_HEADERS
 cell_transformation.h
 cell_trafo_inverse.h
 linear_line_transformation.h
 linear_triangle_transformation.h
 linear_tetrahedron_transformation.h
 linear_pyramid_transformation.h
 bilinear_quad_transformation.h
 trilinear_hexahedron_transformation.h
 linear_quad_transformation.h
 linear_hexahedron_transformation.h
 surface_transformation.h
 linear_line_surface_transformation.h
 linear_triangle_surface_transformation.h
 bilinear_quad_surface_transformation.h
 )
