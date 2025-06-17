# *******************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
project(ZENDNNL_LIBRARY)
#set(ZENDNNL_LIBRARY_VERSION "${PROJECT_VERSION}")
set(ZENDNNL_LIBRARY_VERSION "1.0.0")

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# cmake verbose mode
set(CMAKE_VERBOSE_MAKEFILE ON)

# cmake variables
set(CMAKE_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

message(DEBUG "ZENDNNL_LIBRARY_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(DEBUG "ZENDNNL_LIBRARY_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# include zendnnl files
include(ZenDnnlMacros)
include(ZenDnnlDependenciesInfo)

# global cxx flags
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wall -Werror ${CMAKE_CXX_FLAGS}")

# find common packages
zendnnl_required_packages()

# find required dependencies
find_dependencies(${CMAKE_INSTALL_PREFIX})

# build library
add_subdirectory(src)

# build gtests
add_subdirectory(gtests)



