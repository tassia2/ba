include(boussinesq2d/template/HiFlowFiles.cmake)
make_directory(${PROJECT_BINARY_DIR}/examples/boussinesq2dtemplate)

foreach(i ${template_SOURCES})
    add_executable(${i} boussinesq2d/template/${i}.cc)
    set_target_properties(${i} PROPERTIES RUNTIME_OUTPUT_DIRECTORY boussinesq2d/template)
    target_link_libraries(${i} hiflow ${TINYXML2_LIBRARIES})
    if (WITH_HDF5)
      target_link_libraries(${i} pthread hdf5 dl)
    endif()
    if(WITH_PETSC)
      target_link_libraries(${i} ${PETSC_LIBRARIES})
    endif()
endforeach()

foreach(i ${template_ADDITIONAL_FILES})
  file(COPY boussinesq2d/template/${i}
    DESTINATION ${PROJECT_BINARY_DIR}/examples/boussinesq2d/template)
endforeach()

string(TOUPPER "template" DIRECTORY)

