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

cmake_minimum_required(VERSION 3.25)

# project information
project(ZENDNNL_EXAMPLES)
#set(ZENDNNL_EXAMPLES_VERSION "${PROJECT_VERSION}")
set(ZENDNNL_EXAMPLES_VERSION "1.0.0")

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# cmake verbose mode
set(CMAKE_VERBOSE_MAKEFILE ON)

# cmake module path
set(CMAKE_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

# display cmake variables
message(DEBUG "EXAMPLES CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
message(DEBUG "EXAMPLES CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

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
  message(STATUS "zendnnl forund at ${zendnnl_ROOT}")
endif()

set(examples_sources
  examples.cpp
  example_utils.cpp
  tensor_example.cpp
  sample_example.cpp
  matmul_example.cpp
  compare_op_example.cpp
  reorder_example.cpp)

set(example_headers
  example_utils.hpp
  tensor_example.hpp
  sample_example.hpp
  matmul_example.hpp
  compare_op_example.hpp
  reorder_example.hpp)

add_executable(examples ${examples_sources})
target_link_libraries(examples
  OpenMP::OpenMP_CXX
  zendnnl::zendnnl_archive)

#  amdblis::amdblis_archive
install(TARGETS examples)


