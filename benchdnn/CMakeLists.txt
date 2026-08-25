# /********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.26)

# project information
project(ZENDNNL_BENCHDNN)
set(ZENDNNL_BENCHDNN_VERSION "1.0.0")

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# cmake module path
set(CMAKE_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

# display cmake variables
message(DEBUG "BENCHDNN CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
message(DEBUG "BENCHDNN CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# MSVC consumer baseline. benchdnn compiles ZenDNN's installed AVX-512 public
# headers, so it must use the same baseline the library was built with:
# NOMINMAX (so <windows.h>'s min/max macros do not clobber std::min/std::max and
# std::numeric_limits<>::max()), _USE_MATH_DEFINES (M_* constants), /arch:AVX512,
# /bigobj and /EHsc. These directory-scoped settings are inherited by the
# benchdnn sub-libraries added below. Linux/GCC path is left unchanged.
if(MSVC)
  set(CMAKE_CXX_FLAGS "/W3 ${CMAKE_CXX_FLAGS}")
  add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
  add_compile_options(/EHsc /arch:AVX512 /bigobj /wd4141)
endif()

# find common packages
# On MSVC, request the LLVM OpenMP runtime (/openmp:llvm, OpenMP 3.1) so benchdnn
# matches the zendnnl library and AOCL-DLP (built /openmp:llvm) and resolves the
# OpenMP 3.x APIs (e.g. omp_get_max_active_levels) used by the installed headers.
# vcomp (/openmp) only implements OpenMP 2.0 and lacks those symbols. Requires
# CMake >= 3.30; no effect elsewhere.
if(MSVC)
  if(CMAKE_VERSION VERSION_LESS "3.30")
    message(FATAL_ERROR
      "ZenDNN MSVC builds require CMake >= 3.30: OpenMP_RUNTIME_MSVC (which "
      "selects the LLVM OpenMP runtime /openmp:llvm) is only honored by CMake "
      ">= 3.30. On older CMake it is silently ignored and MSVC falls back to "
      "/openmp (vcomp), which breaks OpenMP 3.x linking and mixes two OpenMP "
      "runtimes with the zendnnl library and AOCL-DLP. Use CMake >= 3.30.")
  endif()
  set(OpenMP_RUNTIME_MSVC "llvm")
endif()
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Propagate the OpenMP usage requirements (notably the MSVC /openmp:llvm compile
# option) to the benchdnn sub-libraries added below: they compile the installed
# public headers that call omp_* APIs, so they need the matching OpenMP flag.
link_libraries(OpenMP::OpenMP_CXX)

# find zendnnl
cmake_path(GET CMAKE_INSTALL_PREFIX PARENT_PATH ZENDNNL_INSTALL_PREFIX)
set(ZENDNNL_INSTALL_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl")
set(zendnnl_ROOT "${ZENDNNL_INSTALL_DIR}")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl REQUIRED)
if(zendnnl_FOUND)
  message(STATUS "zendnnl found at ${zendnnl_ROOT}")
endif()

set(benchdnn_sources
benchdnn.cpp
${ZENDNNL_SOURCE_DIR}/examples/example_utils.cpp) # TODO: Replace example_utils.cpp with a dedicated common utils file

include_directories(${ZENDNNL_SOURCE_DIR}/examples) # TODO: Replace with common utils directory
include_directories(${ZENDNNL_SOURCE_DIR}/benchdnn)

add_subdirectory(matmul)
add_subdirectory(reorder)
add_subdirectory(embag)
add_subdirectory(normalization)
add_subdirectory(grp_matmul)
add_subdirectory(sdpa)
add_subdirectory(utils)

add_executable(benchdnn ${benchdnn_sources})

# Link to OpenMP and benchdnn modules
target_link_libraries(benchdnn
  OpenMP::OpenMP_CXX
  matmul_benchdnn
  reorder_benchdnn
  embag_benchdnn
  normalization_benchdnn
  grp_matmul_benchdnn
  sdpa_benchdnn
  benchdnn_utils)

# Link to whichever zendnnl library variant was built
if(TARGET zendnnl::zendnnl_archive)
  # Prefer static library if available
  target_link_libraries(benchdnn zendnnl::zendnnl_archive)
  message(STATUS "Benchdnn linking to static library (zendnnl_archive)")
elseif(TARGET zendnnl::zendnnl)
  # Fall back to shared library
  target_link_libraries(benchdnn zendnnl::zendnnl)
  message(STATUS "Benchdnn linking to shared library (zendnnl)")
  # The installed benchdnn binary (install/benchdnn/bin/benchdnn) must be able
  # to find libzendnnl.so (installed to install/zendnnl/lib) at runtime. Embed
  # an $ORIGIN-relative RUNPATH so it works without LD_LIBRARY_PATH.
  set_target_properties(benchdnn PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../zendnnl/lib")
else()
  message(FATAL_ERROR "No zendnnl library target found for benchdnn!")
endif()

if(ZENDNNL_CODE_COVERAGE)
  message(STATUS "code-coverage enabled for zendnnl benchdnn")
  target_compile_options(benchdnn PRIVATE --coverage -O0 -g)
  target_link_options(benchdnn PRIVATE --coverage)
endif()

if(ZENDNNL_BUILD_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "ASAN enabled for zendnnl benchdnn")
  set(SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer -O1)
  target_compile_options(benchdnn PRIVATE ${SANITIZER_FLAGS})
  target_link_options(benchdnn PRIVATE "-fsanitize=address")
endif()

install(TARGETS benchdnn)
