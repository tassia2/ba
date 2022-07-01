set(visualization_SOURCES
  cell_visualization.cc
  vtk_writer.cc
  xdmf_writer.cc
  )

# The following sources are needed for developing purpose only
if(EXPLICIT_TEMPLATE_INSTANTS)
  list(APPEND visualization_SOURCES   
  #  visualization.cc
  )
endif()

set(visualization_PUBLIC_HEADERS
  visualization_data.h 
  cell_visualization.h
  vtk_writer.h
  xdmf_writer.h
  )
