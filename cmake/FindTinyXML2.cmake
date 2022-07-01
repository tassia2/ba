# - Try to find TinyXML2
#
find_path (TINYXML2_DIR include/tinyxml2.h DOC "TinyXML2 Install Directory")

IF(EXISTS ${TINYXML2_DIR}/include/tinyxml2.h)
  SET(TINYXML2_FOUND YES)
  find_path(TINYXML2_INCLUDE_DIR tinyxml2.h PATH ${TINYXML2_DIR})
  find_library(TINYXML2_LIBRARIES NAMES tinyxml2 HINTS ${TINYXML2_DIR})
ELSE(EXISTS ${TINYXML2_DIR}/tinyxml2.h)
  SET(TINYXML2_FOUND NO)
ENDIF(EXISTS ${TINYXML2_DIR}/include/tinyxml2.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TinyXML2 DEFAULT_MSG TINYXML2_LIBRARIES TINYXML2_INCLUDE_DIR)

