project(HIFLOW CXX)

cmake_minimum_required(VERSION 3.0.2)

# Prevent in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Please create a build directory and run cmake from there, passing the path to this source directory as the last argument.
Before building, delete the file CMakeCache.txt and the directory CMakeFiles that were created in the source directory.")
endif()

# guard against bad build-type strings
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message("Build type: ${CMAKE_BUILD_TYPE}")

# add cmake directory to module path to use our own modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include(Version)

# Verbose compiler printing
set(CMAKE_VERBOSE_MAKEFILE false)

# Give hints about where to search for includes and libraries.
set(CMAKE_INCLUDE_PATH "/opt/intel/mkl/*/include;/opt/intel/mkl/10.2.5.035/include;/usr/include;/usr/include/mkl")
set(CMAKE_LIBRARY_PATH "/opt/intel/mkl/*/lib/em64t;/opt/intel/mkl/10.2.5.035/lib/em64t;/usr/lib;/usr/lib/x86_64-linux-gnu")

# Configure required and optional libraries, and generate config.h file.
include(HiFlowConfiguration)

# Set internal variables
set(HIFLOW_LIB_DIR lib)
set(HIFLOW_BIN_DIR bin)
set(HIFLOW_INCLUDE_DIR include/hiflow3)
set(HIFLOW_EXAMPLES_DIR share/hiflow3/examples)

# DEBUG / RELEASE build flags
add_definitions(-DMPICH_IGNORE_CXX_SEEK)  # Some MPI implementations need this.

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# -----------------------
# custom compiler options
option(CUSTOM_USE_ASSERT "Activate assert() checks")
if (CUSTOM_USE_ASSERT)
    set(CXX_FLAGS_CUSTOM_ASSERT "")
else ()
    set(CXX_FLAGS_CUSTOM_ASSERT "-DNDEBUG")
endif()

option(CUSTOM_USE_DEBUGINFO "Compile with debug information added to the executables")
if (CUSTOM_USE_DEBUGINFO)
    set(CXX_FLAGS_CUSTOM_G "-g")
else ()
    set(CXX_FLAGS_CUSTOM_G "")
endif()

option(CUSTOM_USE_CPU_ARCH "Use instruction set of specific CPU architecture")
if (CUSTOM_USE_CPU_ARCH)
    set(CUSTOM_CPU_ARCH_TYPE "${CUSTOM_CPU_ARCH_TYPE}" CACHE STRING
    "Choose the type architecture, e.g. native (compiler uses all instructions available on the machine on which compilation is done), 
    haswell, broadwel, skylake, knl, ... " FORCE)
    set(CXX_FLAGS_CUSTOM_ARCH "-march=${CUSTOM_CPU_ARCH_TYPE}")
else() 
    set(CXX_FLAGS_CUSTOM_ARCH "")
endif()

option(CUSTOM_USE_FAST_MATH "Use fast math optimization option")
if (CUSTOM_USE_FAST_MATH)
    set(CXX_FLAGS_FAST_MATH "-ffast-math")
else() 
    set(CXX_FLAGS_FAST_MATH "")
endif()

set(CUSTOM_CPU_OPT_LEVEL "O0" CACHE STRING "level of compiler optimization")
if (CUSTOM_CPU_OPT_LEVEL STREQUAL "O0")
  set(CXX_FLAGS_CUSTOM_OPT "-O0")
elseif (CUSTOM_CPU_OPT_LEVEL STREQUAL "O1")
  set(CXX_FLAGS_CUSTOM_OPT "-O1")
elseif (CUSTOM_CPU_OPT_LEVEL STREQUAL "O2")
  set(CXX_FLAGS_CUSTOM_OPT "-O2")
elseif (CUSTOM_CPU_OPT_LEVEL STREQUAL "O3")
  set(CXX_FLAGS_CUSTOM_OPT "-O3")
endif()

if (CMAKE_CXX_COMPILER MATCHES icpc.*)
  set(CXX_FLAGS_CUSTOM_ICC "")
  option(CUSTOM_USE_INTEL_OPTIM "Use intel compiler optimization")
  if (CUSTOM_USE_INTEL_OPTIM)
    set(CXX_FLAGS_CUSTOM_OPT_ICC "-vec -simd -ipo " CACHE STRING "Intel Compiler Optimization flags" FORCE)
  else()
    set(CXX_FLAGS_CUSTOM_OPT_ICC "")
  endif()
else()
  set(CXX_FLAGS_CUSTOM_OPT_ICC "")
  set(CXX_FLAGS_CUSTOM_ICC "")
endif()

option(CUSTOM_USE_INTEL_ADVISOR "Use compiler options recommended for use with Intel Advisor")
if (CUSTOM_USE_INTEL_ADVISOR)
  set(CUSTOM_USE_ASSERT OFF)
  set(CUSTOM_USE_DEBUGINFO ON)
  set(CUSTOM_USE_INTEL_OPTIM OFF)

  set(CXX_FLAGS_CUSTOM_ASSERT "-DNDEBUG")
  set(CXX_FLAGS_CUSTOM_G "-g")
  if (CMAKE_CXX_COMPILER MATCHES icpc.*)
    set(CXX_FLAGS_CUSTOM_OPT_ICC "-vec -simd -no-ipo")
  else() 
    set(CXX_FLAGS_CUSTOM_OPT_ICC "")
  endif()

  if (CUSTOM_CPU_OPT_LEVEL STREQUAL "O0")
    set(CXX_FLAGS_CUSTOM_OPT "-O2")
  elseif (CUSTOM_CPU_OPT_LEVEL STREQUAL "O1")
    set(CXX_FLAGS_CUSTOM_OPT "-O2")
  endif()
endif()

set(CMAKE_CXX_FLAGS_CUSTOM "${CXX_FLAGS_CUSTOM_ASSERT} ${CXX_FLAGS_CUSTOM_G} ${CXX_FLAGS_CUSTOM_ARCH} ${CXX_FLAGS_FAST_MATH} ${CXX_FLAGS_CUSTOM_OPT} ${CXX_FLAGS_CUSTOM_ICC} ${CXX_FLAGS_CUSTOM_OPT_ICC}" CACHE STRING 
      "compiler flags for custom build" FORCE)

# end custom compiler options

option(USE_WARNINGS "Use compiler warnings")
set(WARNING_FLAGS "-Wredundant-decls -Wcast-align -Wmissing-declarations -Wmissing-include-dirs -Wswitch-enum -Wswitch-default -Wextra -Wall -Winvalid-pch -Wformat=2 -Wmissing-format-attribute -Wformat-nonliteral " CACHE STRING "compiler warnings" FORCE)   
mark_as_advanced(WARNING_FLAGS)

# -Werror
set(WARNING_SUPPRESSED_FLAGS "-Wno-reorder -Wno-sign-compare -Wno-unused-parameter" CACHE STRING "supressed compiler warnings" FORCE)
mark_as_advanced(WARNING_SUPPRESSED_FLAGS)

if (USE_WARNINGS)
  set(USED_WARNING_FLAGS "${WARNING_FLAGS} ${WARNING_SUPPRESSED_FLAGS}")
else ()
  set(USED_WARNING_FLAGS "")
endif()

message ("used warnings ${USED_WARNING_FLAGS}")

#-Wno-unknown-pragmas -Wno-sign-compare -Wno-cast-function-type -Wno-comment -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable 

set(CMAKE_CXX_FLAGS_COVERAGE "-coverage" CACHE STRING
    "Flags used by the C++ compiler during coverage builds."
    FORCE)
set(CMAKE_C_FLAGS_COVERAGE "-coverage" CACHE STRING
    "Flags used by the C compiler during coverage builds."
    FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "-coverage" CACHE STRING
    "Flags used for linking binaries during coverage builds."
    FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "-coverage" CACHE STRING
    "Flags used by the shared libraries linker during coverage builds."
    FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_EXE_LINKER_FLAGS_COVERAGE
    CMAKE_SHARED_LINKER_FLAGS_COVERAGE )

# Update the documentation string of CMAKE_BUILD_TYPE for GUIs
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage Custom."
    FORCE)

# VENDOR-specific compiler flags
if (CMAKE_CXX_COMPILER MATCHES g[+][+].* OR CMAKE_CXX_COMPILER MATCHES c[+][+].*)
  message("Setting vendor-specific compiler flags for GNU Compiler")
  set(HIFLOW_VENDOR_FLAGS "-pthread -ansi")
endif()

if (CMAKE_CXX_COMPILER MATCHES icpc.*)
  message("Setting vendor-specific compiler flags for Intel Compiler")
  # Use interprocedural optimizations -- requires xiar archiver.
  set(HIFLOW_VENDOR_FLAGS "-ipo-separate")
  set(CMAKE_AR "xiar")
endif()

if (CMAKE_CXX_COMPILER MATCHES pgCC.*)
  message("Setting vendor-specific compiler flags for PGI Compiler")
  set(HIFLOW_VENDOR_FLAGS "-noswitcherror -cppsuffix=.cu")
endif()

if (CMAKE_CXX_COMPILER MATCHES clang[+][+].*)
  message("Setting vendor-specific compiler flags for Clang Compiler")
  set(HIFLOW_VENDOR_FLAGS "")
endif()

# Option for c++ iso standard
set(HIFLOW_CPP_ISO_STANDARD "c++17" CACHE STRING "C++ ISO standard")
if (HIFLOW_CPP_ISO_STANDARD STREQUAL "c++17")
  message("Use c++17 iso standard")
  set(CMAKE_CXX_STANDARD 17)
elseif (HIFLOW_CPP_ISO_STANDARD STREQUAL "c++14")
  message("Use c++14 iso standard")
  set(CMAKE_CXX_STANDARD 14)
elseif (HIFLOW_CPP_ISO_STANDARD STREQUAL "c++11")
  message("Use c++11 iso standard")
  set(CMAKE_CXX_STANDARD 11)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Boost include directories
include_directories(${Boost_INCLUDE_DIR})

# TinyXML include directory
include_directories(${TINYXML2_INCLUDE_DIR})

# Process subdirectories
add_subdirectory(src)
add_subdirectory(doc)
add_subdirectory(contrib)

if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if (BUILD_UTILS)
  add_subdirectory(utils)
endif()

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if (BUILD_EXERCISES)
  add_subdirectory(exercises)
endif()

# Install export file
install(EXPORT HiFlow DESTINATION share/hiflow3)

# Set these variable _before_ including CPack (why is this necessary???)
set(CPACK_INSTALL_CMAKE_PROJECTS ${PROJECT_BINARY_DIR} hiflow ALL /)

# CPack configuration variables
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HiFlow3")
set(CPACK_PACKAGE_FILE_NAME "HiFlow3")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "3")
set(CPACK_PACKAGE_VERSION_PATCH "2")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/doc/INSTALLATION")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/COPYING.GPL")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/COPYING.LGPL")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/AUTHORS")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "HiFlow3_version ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}.${CMake_VERSION_PATCH}")

include(CPack)
