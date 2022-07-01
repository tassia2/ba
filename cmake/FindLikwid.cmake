#  LIKWID_LIBRARY - library to link to

find_path (LIKWID_DIR include/likwid.h DOC "Likwid Install Directory")

IF(EXISTS ${LIKWID_DIR}/include/likwid.h)
  SET(LIKWID_FOUND YES)
  find_path(LIKWID_INCLUDE_DIR likwid.h PATH ${LIKWID_DIR})
  find_library(LIKWID_LIBRARY NAMES likwid HINTS ${LIKWID_DIR})
ELSE(EXISTS ${LIKWID_DIR}/likwid.h)
  SET(LIKWID_FOUND NO)
ENDIF(EXISTS ${LIWKID_DIR}/include/likwid.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Likwid DEFAULT_MSG LIKWID_LIBRARY LIKWID_INCLUDE_DIR)