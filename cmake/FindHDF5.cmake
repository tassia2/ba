# - Try to find HDF5
#

find_path(HDF5_DIR include/hdf5.h HINTS ENV HDF5_DIR DOC "HDF5 Install Directory")
find_path(HDF5_INCLUDE_DIR hdf5.h HINTS "${HDF5_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
find_library(HDF5_HDF5_LIBRARY NAMES libhdf5.so OR libhdf5.a PATHS ${HDF5_DIR}/lib)
find_library(HDF5_HDF5_HL_LIBRARY NAMES libhdf5_hl.so OR libhdf5_hl.a PATHS ${HDF5_DIR}/lib)
find_library(HDF5_Z_LIBRARY NAMES libz.so)
find_library(HDF5_M_LIBRARY NAMES libm.so)
  
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HDF5 DEFAULT_MSG HDF5_HDF5_LIBRARY HDF5_HDF5_HL_LIBRARY HDF5_M_LIBRARY HDF5_Z_LIBRARY HDF5_INCLUDE_DIR)
