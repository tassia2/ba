# - Try to find VCL

find_path (VCL_DIR vectorclass.h DOC "VLC Include Directory")

IF(EXISTS ${VCL_DIR}/vectorclass.h)
  SET(VCL_FOUND YES)
  find_path(VCL_INCLUDE_DIR vectorclass.h PATH ${VCL_DIR})
ELSE(EXISTS ${VCL_DIR}/vectorclass.h)
  SET(VCL_FOUND NO)
ENDIF(EXISTS ${VCL_DIR}/vectorclass.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vcl DEFAULT_MSG VCL_INCLUDE_DIR)