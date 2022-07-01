# - Try to find ILU++
#

set(ILUPP_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/contrib/ILU++_1.1.1/lib/)
set(ILUPP_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/contrib/ILU++_1.1.1/libiluplusplus-1.1.1.a)

SET(ILUPP_FOUND YES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ILU++ DEFAULT_MSG ILUPP_LIBRARY ILUPP_INCLUDE_DIR

)

