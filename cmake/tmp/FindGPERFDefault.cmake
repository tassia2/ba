# - Try to find METIS
# Once done, this will define
#
#  METIS_FOUND - system has METIS
#  METIS_INCLUDE_DIRS - the METIS include directories
#  METIS_LIBRARY - library to link to

include(LibFindMacros)

# Dependencies
# 

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(GPERF_PKGCONF GPERF)

# Include dir
find_path(GPERF_INCLUDE_DIR
  NAMES profiler.h
  PATHS ${GPERF_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(GPERF_LIBRARY
  NAMES libprofiler.a
  PATHS ${GPERF_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(GPERF_PROCESS_INCLUDES GPERF_INCLUDE_DIR)
set(GPERF_PROCESS_LIBS GPERF_LIBRARY)
libfind_process(GPERF)
