# /********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.25)

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

# find common packages
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

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
add_subdirectory(utils)

add_executable(benchdnn ${benchdnn_sources})

target_link_libraries(benchdnn
  OpenMP::OpenMP_CXX
  zendnnl::zendnnl_archive
  matmul_benchdnn
  reorder_benchdnn
  embag_benchdnn
  benchdnn_utils)

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
