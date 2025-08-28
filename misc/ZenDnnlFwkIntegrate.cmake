# *******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# set ZenDNNL source, build and install folders
set(ZENDNNL_SOURCE_DIR <zendnnl source dir>
  CACHE PATH "zendnnl source dir")
set(ZENDNNL_BUILD_DIR "${ZENDNNL_SOURCE_DIR}/build"
  CACHE PATH "zendnnl build dir")
set(ZENDNNL_INSTALL_DIR "${ZENDNNL_BUILD_DIR}/install"
  CACHE PATH "zendnnl install dir")

# blis path if framework builds it
set(ZENDNNL_AMDBLIS_FWK_DIR <framework amdblis dir>
  CACHE PATH "framework amd-blis dir")

# # try to find pre-built package
set(zendnnl_ROOT "${ZENDNNL_INSTALL_DIR}/zendnnl")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl QUIET)
if(zendnnl_FOUND)
  message(STATUS "zendnnl found at ${zendnnl_ROOT}")
else()
  message(STATUS "zendnnl not found... building as an external project")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_ZENTORCH_BUILD:BOOL=ON")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_DEPENDS_AMDBLIS:BOOL=ON")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_DEPENDS_ONEDNN:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS
    "-DZENDNNL_AMDBLIS_FWK_INSTALL_DIR:PATH=${ZENDNNL_AMDBLIS_FWK_DIR}")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_BUILD_EXAMPLES:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_BUILD_GTEST:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_BUILD_DOXYGEN:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_BUILD_BENCHDNN:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_CODE_COVERAGE:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  ExternalProject_ADD(fwk_zendnnl
    SOURCE_DIR  "${ZENDNNL_SOURCE_DIR}"
    BINARY_DIR  "${ZENDNNL_BUILD_DIR}"
    INSTALL_DIR "${ZENDNNL_INSTALL_DIR}"
    CMAKE_ARGS  "${ZNL_CMAKE_ARGS}"
    INSTALL_COMMAND cmake --build . --target all -j
    BUILD_BYPRODUCTS <INSTALL_DIR>/deps/aoclutils/lib/libaoclutils.a
                     <INSTALL_DIR>/deps/aoclutils/lib/libau_cpuid.a
                     <INSTALL_DIR>/zendnnl/lib/libzendnnl_archive.a )

  # list(APPEND ZENDNNL_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/ZenDNNL")
  # list(APPEND ZENDNNL_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/amdblis")
  # set_target_properties(fwk_zendnnl
  #   PROPERTIES
  #   ADDITIONAL_CLEAN_FILES "${ZENDNNL_CLEAN_FILES}")

  add_dependencies(fwk_zendnnl <add dependencies>)

  set(ZENDNNL_LIBRARY_INC_DIR "${ZENDNNL_INSTALL_DIR}/zendnnl/include")
  set(ZENDNNL_LIBRARY_LIB_DIR "${ZENDNNL_INSTALL_DIR}/zendnnl/lib")
  file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
  add_library(zendnnl_library STATIC IMPORTED GLOBAL)
  add_dependencies(zendnnl_library fwk_zendnnl)
  set_target_properties(zendnnl_library
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")

  add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

endif()
