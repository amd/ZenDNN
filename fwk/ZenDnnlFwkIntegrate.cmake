# *******************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/
include_guard(GLOBAL)
include(ExternalProject)
include("${CMAKE_CURRENT_LIST_DIR}/ZenDnnlFwkMacros.cmake")

find_package(OpenMP REQUIRED QUIET)

# Paths: placeholders unless overridden before this file runs (e.g. ZenTorch stub sets CACHE).
zendnnl_add_option(NAME ZENDNNL_SOURCE_DIR
  VALUE "<zendnnl source dir>"
  TYPE PATH
  CACHE_STRING "zendnnl_source_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_BINARY_DIR
  VALUE "<zendnnl binary dir>"
  TYPE PATH
  CACHE_STRING "zendnnl_binary_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_INSTALL_PREFIX
  VALUE "<zendnnl install prefix>"
  TYPE PATH
  CACHE_STRING "zendnnl_install_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

foreach(_znl_path ZENDNNL_SOURCE_DIR ZENDNNL_BINARY_DIR ZENDNNL_INSTALL_PREFIX)
  if("${${_znl_path}}" MATCHES "^<")
    message(AUTHOR_WARNING "(ZENDNNL) ${_znl_path} is not set; set it before including this file.")
  endif()
endforeach()

## general zendnnl options
zendnnl_add_option(NAME ZENDNNL_FWK_BUILD
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl framework build"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_BUILD_TYPE
  VALUE "Release"
  TYPE STRING
  CACHE_STRING "zendnnl build type"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# Compiler flags propagation from parent build
zendnnl_add_option(NAME ZENDNNL_C_FLAGS
  VALUE ""
  TYPE STRING
  CACHE_STRING "C compiler flags from parent"
  EXCLUDE_FROM_COMMAND_LIST)

zendnnl_add_option(NAME ZENDNNL_CXX_FLAGS
  VALUE ""
  TYPE STRING
  CACHE_STRING "C++ compiler flags from parent"
  EXCLUDE_FROM_COMMAND_LIST)

# Propagate compiler flags through nested ExternalProject builds
if(ZENDNNL_C_FLAGS)
  list(APPEND ZNL_CMAKE_ARGS "-DCMAKE_C_FLAGS:STRING=${ZENDNNL_C_FLAGS}")
endif()
if(ZENDNNL_CXX_FLAGS)
  list(APPEND ZNL_CMAKE_ARGS "-DCMAKE_CXX_FLAGS:STRING=${ZENDNNL_CXX_FLAGS}")
endif()

# ExternalProject does not inherit the parent cache compilers. The parent
# already resolved CMAKE_C/CXX_COMPILER from $CC/$CXX and/or
# -DCMAKE_*_COMPILER; pass that resolved compiler. The child still inherits
# $CC/$CXX; command-line wins if both are set.
if(CMAKE_C_COMPILER)
  list(APPEND ZNL_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
endif()
if(CMAKE_CXX_COMPILER)
  list(APPEND ZNL_CMAKE_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
endif()

zendnnl_add_option(NAME ZENDNNL_MESSAGE_LOG_LEVEL
  VALUE "DEBUG"
  TYPE STRING
  CACHE_STRING "zendnnl message log level"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_VERBOSE_MAKEFILE
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl verbose makefile"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## zendnnl library outputs (must be set before ExternalProject; match zendnnl/CMakeLists.txt defaults)
zendnnl_add_option(NAME ZENDNNL_LIB_BUILD_ARCHIVE
  VALUE ON
  TYPE BOOL
  CACHE_STRING "build zendnnl archive library"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_LIB_BUILD_SHARED
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl shared library"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## components options
zendnnl_add_option(NAME ZENDNNL_BUILD_EXAMPLES
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl examples"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_BUILD_GTEST
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl gtests"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_BUILD_DOXYGEN
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl doxygen documentation"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_BUILD_BENCHDNN
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl benchdnn"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_CODE_COVERAGE
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl code coverage"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## dependencies
# aocl-dlp is a mandatory dependency.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_AOCLDLP
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl aocldlp dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_DEPENDS_ONEDNN
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl onednn dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_DEPENDS_LIBXSMM
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl libxsmm dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_DEPENDS_PARLOOPER
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl parlooper dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_DEPENDS_FBGEMM
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl fbgemm dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# Empty = do not inject (ZenDNN fetches/builds deps). Non-empty enables injection
# and must be a real filesystem path — never use angle-bracket placeholders here:
# they are non-empty so INJECTED=ON, and '<' breaks shell commands in symlink steps.
zendnnl_add_option(NAME ZENDNNL_AOCLDLP_INJECT_DIR
  VALUE ""
  TYPE PATH
  CACHE_STRING "zendnnl aocldlp injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_ONEDNN_INJECT_DIR
  VALUE ""
  TYPE PATH
  CACHE_STRING "zendnnl onednn injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_LIBXSMM_INJECT_DIR
  VALUE ""
  TYPE PATH
  CACHE_STRING "zendnnl libxsmm injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_PARLOOPER_INJECT_DIR
  VALUE ""
  TYPE PATH
  CACHE_STRING "zendnnl parlooper injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

zendnnl_add_option(NAME ZENDNNL_FBGEMM_INJECT_DIR
  VALUE ""
  TYPE PATH
  CACHE_STRING "zendnnl fbgemm injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

set(zendnnl_ROOT "${ZENDNNL_INSTALL_PREFIX}/zendnnl")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl QUIET)
if(zendnnl_FOUND)
  message(STATUS "(ZENDNNL) ZENDNNL FOUND AT ${zendnnl_ROOT}")
  message(STATUS "(ZENDNNL) if zendnnl options are changed from previous build,")
  message(STATUS "(ZENDNNL) they will not be reflected")
  message(STATUS "(ZENDNNL) If options are changed, please do a clean build.")
  if(TARGET zendnnl::zendnnl_archive)
    set_target_properties(zendnnl::zendnnl_archive
      PROPERTIES IMPORTED_GLOBAL ON)
  endif()
  if(TARGET zendnnl::zendnnl)
    set_target_properties(zendnnl::zendnnl
      PROPERTIES IMPORTED_GLOBAL ON)
  endif()
else()
  message(STATUS "(ZENDNNL) ZENDNNL NOT FOUND, will be built as an external project.")

  set(ZENDNNL_LIBRARY_INC_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl/include")
  set(ZENDNNL_LIBRARY_LIB_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl/lib")

  if(NOT EXISTS ${ZENDNNL_LIBRARY_INC_DIR})
    file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
  endif()

  if(ZENDNNL_LIB_BUILD_ARCHIVE)
    add_library(zendnnl_library STATIC IMPORTED GLOBAL)
    set_target_properties(zendnnl_library
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
      INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")

    # OpenMP::OpenMP_CXX lets the consumer pick the runtime. A bare -fopenmp
    # link option would make clang add its default libomp next to it.
    target_link_libraries(zendnnl_library
      INTERFACE OpenMP::OpenMP_CXX
      INTERFACE ${CMAKE_DL_LIBS})

    add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

    list(APPEND ZNL_BYPRODUCTS "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a")
  endif()

  if(ZENDNNL_LIB_BUILD_SHARED)
    add_library(zendnnl_shared_library SHARED IMPORTED GLOBAL)
    set_target_properties(zendnnl_shared_library
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}zendnnl${CMAKE_SHARED_LIBRARY_SUFFIX}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")

    target_link_libraries(zendnnl_shared_library
      INTERFACE OpenMP::OpenMP_CXX
      INTERFACE ${CMAKE_DL_LIBS})

    add_library(zendnnl::zendnnl ALIAS zendnnl_shared_library)

    list(APPEND ZNL_BYPRODUCTS "${ZENDNNL_LIBRARY_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}zendnnl${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()

  if(NOT ZENDNNL_LIB_BUILD_ARCHIVE AND NOT ZENDNNL_LIB_BUILD_SHARED)
    message(FATAL_ERROR "At least one of ZENDNNL_LIB_BUILD_ARCHIVE or ZENDNNL_LIB_BUILD_SHARED must be ON")
  endif()

  # declare all dependencies
  # Static archive import (zendnnl_library): use WHOLE_ARCHIVE for .a deps when folding into consumers (e.g. shared extensions).
  # Shared zendnnl import (zendnnl_shared_library): normal INTERFACE links only.

  zendnnl_add_dependency(NAME json
    PATH "${ZENDNNL_INSTALL_PREFIX}/deps/json"
    ALIAS "zendnnl::json"
    INCLUDE_ONLY)

  if(ZENDNNL_LIB_BUILD_ARCHIVE)
    target_link_libraries(zendnnl_library INTERFACE zendnnl::json)
  endif()
  if(ZENDNNL_LIB_BUILD_SHARED)
    target_link_libraries(zendnnl_shared_library INTERFACE zendnnl::json)
  endif()

  # AOCL-Utils is built with CMAKE_INSTALL_LIBDIR=lib, so the archives are
  # always under lib/ (including on RHEL and manylinux).
  zendnnl_add_dependency(NAME aoclutils
    PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
    ARCHIVE_FILE "libaoclutils.a"
    ALIAS "au::aoclutils")

  if(ZENDNNL_LIB_BUILD_ARCHIVE)
    target_link_libraries(zendnnl_library INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>")
  endif()
  if(ZENDNNL_LIB_BUILD_SHARED)
    target_link_libraries(zendnnl_shared_library INTERFACE au::aoclutils)
  endif()

  zendnnl_add_dependency(NAME aucpuid
    PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
    ARCHIVE_FILE "libau_cpuid.a"
    ALIAS "au::au_cpuid")

  if(ZENDNNL_LIB_BUILD_ARCHIVE)
    target_link_libraries(zendnnl_library INTERFACE au::au_cpuid)
  endif()
  if(ZENDNNL_LIB_BUILD_SHARED)
    target_link_libraries(zendnnl_shared_library INTERFACE au::au_cpuid)
  endif()

  if(ZENDNNL_DEPENDS_AOCLDLP)
    zendnnl_add_dependency(NAME aocldlp
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aocldlp"
      ARCHIVE_FILE "libaocl-dlp.a"
      ALIAS "aocldlp::aocl_dlp_static")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,aocldlp::aocl_dlp_static>")
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE aocldlp::aocl_dlp_static)
    endif()
  endif()

  if(ZENDNNL_DEPENDS_ONEDNN)
    if(DEFINED ENV{ZENDNNL_MANYLINUX_BUILD})
      zendnnl_add_dependency(NAME onednn
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
        LIB_SUFFIX lib64
        ARCHIVE_FILE "libdnnl.a"
        ALIAS "DNNL::dnnl")
    else()
      zendnnl_add_dependency(NAME onednn
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
        ARCHIVE_FILE "libdnnl.a"
        ALIAS "DNNL::dnnl")
    endif()

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,DNNL::dnnl>")
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE DNNL::dnnl)
    endif()
  endif()

  if(ZENDNNL_DEPENDS_LIBXSMM)
    zendnnl_add_dependency(NAME libxsmm
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/libxsmm"
      ARCHIVE_FILE "libxsmm.a"
      ALIAS "libxsmm::libxsmm_archive")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,libxsmm::libxsmm_archive>")
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE libxsmm::libxsmm_archive)
    endif()
  endif()

  if(ZENDNNL_DEPENDS_PARLOOPER)
    zendnnl_add_dependency(NAME parlooper
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/parlooper"
      ARCHIVE_FILE "libparlooper.a"
      ALIAS "parlooper::parlooper_archive")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,parlooper::parlooper_archive>")
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE parlooper::parlooper_archive)
    endif()
  endif()

  if(ZENDNNL_DEPENDS_FBGEMM)
    if(DEFINED ENV{ZENDNNL_MANYLINUX_BUILD})
      zendnnl_add_dependency(NAME asmjit
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/fbgemm"
        LIB_SUFFIX lib64
        ARCHIVE_FILE "libasmjit.a"
        ALIAS "fbgemm::asmjit")
      zendnnl_add_dependency(NAME cpuinfo
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/fbgemm"
        LIB_SUFFIX lib64
        ARCHIVE_FILE "libcpuinfo.a"
        ALIAS "fbgemm::cpuinfo")
      zendnnl_add_dependency(NAME fbgemm
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/fbgemm"
        LIB_SUFFIX lib64
        ARCHIVE_FILE "libfbgemm.a"
        ALIAS "fbgemm::fbgemm_archive")
    else()
      zendnnl_add_dependency(NAME asmjit
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/fbgemm"
        ARCHIVE_FILE "libasmjit.a"
        ALIAS "fbgemm::asmjit")
      zendnnl_add_dependency(NAME cpuinfo
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/fbgemm"
        ARCHIVE_FILE "libcpuinfo.a"
        ALIAS "fbgemm::cpuinfo")
      zendnnl_add_dependency(NAME fbgemm
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/fbgemm"
        ARCHIVE_FILE "libfbgemm.a"
        ALIAS "fbgemm::fbgemm_archive")
    endif()

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,fbgemm::fbgemm_archive>")
      target_link_libraries(zendnnl_library INTERFACE fbgemm::asmjit)
      target_link_libraries(zendnnl_library INTERFACE fbgemm::cpuinfo)
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE fbgemm::fbgemm_archive)
      target_link_libraries(zendnnl_shared_library INTERFACE fbgemm::asmjit)
      target_link_libraries(zendnnl_shared_library INTERFACE fbgemm::cpuinfo)
    endif()
  endif()

  message(STATUS "(ZENDNNL) ZNL_BYPRODUCTS=${ZNL_BYPRODUCTS}")
  message(STATUS "(ZENDNNL) ZNL_CMAKE_ARGS=${ZNL_CMAKE_ARGS}")

  ExternalProject_ADD(fwk_zendnnl
    SOURCE_DIR  "${ZENDNNL_SOURCE_DIR}"
    BINARY_DIR  "${ZENDNNL_BINARY_DIR}"
    CMAKE_ARGS  "${ZNL_CMAKE_ARGS}"
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${ZNL_BYPRODUCTS})

  list(APPEND ZENDNNL_CLEAN_FILES "${ZENDNNL_BINARY_DIR}")
  list(APPEND ZENDNNL_CLEAN_FILES "${ZENDNNL_INSTALL_PREFIX}")
  set_target_properties(fwk_zendnnl
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${ZENDNNL_CLEAN_FILES}")

  # Reminder only: fwk_zendnnl cannot have dependencies yet at this point.
  if(ZENDNNL_AOCLDLP_INJECT_DIR OR ZENDNNL_ONEDNN_INJECT_DIR OR ZENDNNL_LIBXSMM_INJECT_DIR
     OR ZENDNNL_PARLOOPER_INJECT_DIR OR ZENDNNL_FBGEMM_INJECT_DIR)
    message(AUTHOR_WARNING "(ZENDNNL) dependencies are injected; make fwk_zendnnl depend on the targets that build them.")
  endif()

  if(ZENDNNL_LIB_BUILD_ARCHIVE)
    add_dependencies(zendnnl_library fwk_zendnnl)
  endif()
  if(ZENDNNL_LIB_BUILD_SHARED)
    add_dependencies(zendnnl_shared_library fwk_zendnnl)
  endif()
  add_dependencies(zendnnl_json_deps fwk_zendnnl)
  add_dependencies(zendnnl_aoclutils_deps fwk_zendnnl)
  add_dependencies(zendnnl_aucpuid_deps fwk_zendnnl)

  if(ZENDNNL_DEPENDS_AOCLDLP)
    add_dependencies(zendnnl_aocldlp_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_ONEDNN)
    add_dependencies(zendnnl_onednn_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_LIBXSMM)
    add_dependencies(zendnnl_libxsmm_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_PARLOOPER)
    add_dependencies(zendnnl_parlooper_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_FBGEMM)
    add_dependencies(zendnnl_asmjit_deps fwk_zendnnl)
    add_dependencies(zendnnl_cpuinfo_deps fwk_zendnnl)
    add_dependencies(zendnnl_fbgemm_deps fwk_zendnnl)
  endif()

endif()
