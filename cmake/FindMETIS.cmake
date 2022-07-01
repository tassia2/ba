# - Try to find METIS
# Once done, this will define
#
#  METIS_FOUND - system has METIS
#  METIS_INCLUDE_DIRS - the METIS include directories
#  METIS_LIBRARY - library to link to

find_path (METIS_DIR include/metis.h DOC "Metis Install Directory")

IF(EXISTS ${METIS_DIR}/include/metis.h)
  SET(METIS_FOUND YES)
  find_path(METIS_INCLUDE_DIR metis.h PATH ${METIS_DIR})
  find_library(METIS_LIBRARY NAMES metis HINTS ${METIS_DIR})
ELSE(EXISTS ${METIS_DIR}/metis.h)
  SET(METIS_FOUND NO)
ENDIF(EXISTS ${METIS_DIR}/include/metis.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metis DEFAULT_MSG METIS_LIBRARY METIS_INCLUDE_DIR)



#include(LibFindMacros)

# Dependencies
# 

# Use pkg-config to get hints about paths
#libfind_pkg_check_modules(METIS_PKGCONF METIS)

# install dir
#find_path (METIS_DIR include/metis.h HINTS ENV METIS_DIR DOC "METIS Install Directory")

#IF(EXISTS ${METIS_DIR}/include/metis.h)
#  SET(METIS_FOUND YES)
#  find_path (METIS_INCLUDE_DIR metis.h HINTS "${METIS_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
#  find_library(METIS_LIBRARY NAMES metis HINTS "${METIS_DIR}" PATH_SUFFIXES lib OR /usr/lib/x86_64-linux-gnu/libmetis.so OR build/Linux-x86_64/libmetis/ NO_DEFAULT_PATH)
#ELSE(EXISTS ${METIS_DIR}/include/metis.h)
#  SET(METIS_FOUND NO)
#ENDIF(EXISTS ${METIS_DIR}/include/metis.h)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
#set(METIS_PROCESS_INCLUDES METIS_INCLUDE_DIR)
#set(METIS_PROCESS_LIBS METIS_LIBRARY)
#libfind_process(METIS)





