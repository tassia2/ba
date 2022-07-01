# - Try to find GPerftools
# Once done, this will define
#

include(LibFindMacros)

find_path (GPERF_DIR include/gperftools/profiler.h HINTS ENV GPERF_DIR DOC "gperf Directory")
IF(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)
  SET(GPERF_FOUND YES)
  find_path (GPERF_INCLUDE_DIR profiler.h HINTS "${GPERF_DIR}" PATH_SUFFIXES include/gperftools NO_DEFAULT_PATH)
  find_library (GPERF_LIBRARY NAMES libprofiler.so OR libprofiler.a PATHS ${GPERF_DIR}/lib)
ELSE(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)
  SET(GPERF_FOUND NO)
ENDIF(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(GPERF_PROCESS_INCLUDES GPERF_INCLUDE_DIR)
set(GPERF_PROCESS_LIBS GPERF_LIBRARY)
libfind_process(GPERF)




#find_path (GPERF_DIR include/gperftools/profiler.h HINTS ENV GPERF_DIR DOC "gperf Directory")

#IF(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)
#  SET(GPERF_FOUND YES)
#  SET(GPERF_INCLUDES ${GPERF_DIR})
#  find_path (GPERF_INCLUDE_DIR profiler.h HINTS "${GPERF_DIR}" PATH_SUFFIXES include/gperftools NO_DEFAULT_PATH)
#  list(APPEND GPERF_INCLUDES ${GPERF_INCLUDE_DIR})
#  find_path (GPERF_LIBRARY_DIR libprofiler.a HINTS "${GPERF_DIR}" PATH_SUFFIXES lib NO_DEFAULT_PATH)
#  list(APPEND GPERF_LIBRARIES ${GPERF_LIBRARY_DIR}/libprofiler.a)
#ELSE(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)
#  SET(GPERF_FOUND NO)
#ENDIF(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)

#include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(GPERF DEFAULT_MSG GPERF_LIBRARIES GPERF_INCLUDES)












#  METIS_FOUND - system has METIS
#  METIS_INCLUDE_DIRS - the METIS include directories
#  METIS_LIBRARY - library to link to

#include(LibFindMacros)

# Dependencies
# 

# Use pkg-config to get hints about paths
#libfind_pkg_check_modules(METIS_PKGCONF METIS)

# install dir
#find_path (GPERF_DIR include/gperftools/profiler.h HINTS ENV GPERF_DIR DOC "GPERF Install Directory")
#IF(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)
#  SET(GPERF_FOUND YES)
#  find_path (GPERF_INCLUDE_DIR profiler.h HINTS "${GPERF_DIR}" PATH_SUFFIXES include/gperftools NO_DEFAULT_PATH)
#  find_library(GPERF_LIBRARY NAMES libprofiler.so OR libprofiler.a PATHS ${GPERF_DIR}/lib)
#  find_library(GPERF_LIBRARY NAMES profiler PATHS ${GPERF_DIR}/lib)
#ELSE(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)
#  SET(GPERF_FOUND NO)
#ENDIF(EXISTS ${GPERF_DIR}/include/gperftools/profiler.h)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
#set(GPERF_PROCESS_INCLUDES GPERF_INCLUDE_DIR)
#set(GPERF_PROCESS_LIBS GPERF_LIBRARY)
#libfind_process(GPERF)

