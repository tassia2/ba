# - Try to find ILU++
#

find_path (ILUPP_DIR lib/libiluplusplus.h DOC "ILUPP Install Directory")

IF(EXISTS ${ILUPP_DIR}/lib/iluplusplus.h)
  SET(ILUPP_FOUND YES)
  find_path(ILUPP_INCLUDE_DIR iluplusplus.h HINTS "${ILUPP_DIR}" PATH_SUFFIXES lib NO_DEFAULT_PATH)
  find_library(ILUPP_LIBRARY libiluplusplus-1.1.1.a PATHS ${ILUPP_DIR} NO_DEFAULT_PATH)
ELSE(EXISTS ${ILUPP_DIR}/lib/iluplusplus.h)
  SET(ILUPP_FOUND NO)
ENDIF(EXISTS ${ILUPP_DIR}/lib/iluplusplus.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ILU++ DEFAULT_MSG ILUPP_LIBRARY ILUPP_INCLUDE_DIR)
