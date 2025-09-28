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
project(ZENDNNL_LIBRARY VERSION ${ZENDNNL_PROJECT_VERSION})

# library version
set(ZENDNNL_LIBRARY_VERSION ${PROJECT_VERSION})
set(ZENDNNL_LIBRARY_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(ZENDNNL_LIBRARY_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(ZENDNNL_LIBRARY_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# library components
set(ZENDNNL_LIB_BUILD_ARCHIVE ON CACHE BOOL "build zendnnl archive")
set(ZENDNNL_LIB_BUILD_SHARED OFF CACHE BOOL "build zendnnl shared lib")

# library dependencies
set(ZENDNNL_AMDBLIS_INJECTED OFF CACHE BOOL "amdblis injected")
set(ZENDNNL_AOCLDLP_INJECTED OFF CACHE BOOL "amdblis injected")
set(ZENDNNL_ONEDNN_INJECTED  OFF CACHE BOOL "onednn injected")
set(ZENDNNL_LIBXSMM_INJECTED  OFF CACHE BOOL "libxsmm injected")
set(ZENDNNL_PARLOOPER_INJECTED  OFF CACHE BOOL "parlooper injected")

# cmake variables
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "verbose makefile")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_MESSAGE_LOG_LEVEL "DEBUG")
set(CMAKE_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

# informative messages
message(STATUS
  "${ZENDNNL_MSG_PREFIX} ZENDNNL_LIBRARY_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "ZENDNNL_LIBRARY_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  message(STATUS "building zendnnl library archive")
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  message(STATUS "building zendnnl library shared lib")
endif()

# include zendnnl files
include(ZenDnnlMacros)
include(ZenDnnlDependenciesInfo)

# global cxx flags
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wall -Werror ${CMAKE_CXX_FLAGS}")

# find common packages
zendnnl_required_packages()

# find required dependencies
find_build_dependencies(${CMAKE_INSTALL_PREFIX})

# build library
add_subdirectory(src)

# build gtests
add_subdirectory(gtests)

# create version file
set(ZENDNNL_VERSION_FILE "${CMAKE_BINARY_DIR}/zendnnl_version.hpp")
create_version_header_file(${ZENDNNL_VERSION_FILE}
  ${ZENDNNL_LIBRARY_VERSION_MAJOR}
  ${ZENDNNL_LIBRARY_VERSION_MINOR}
  ${ZENDNNL_LIBRARY_VERSION_PATCH})

# install config file
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/config/zendnnl_user_config.json"
  DESTINATION zendnnl/config)

# install version file
install(FILES "${ZENDNNL_VERSION_FILE}"
  DESTINATION zendnnl/include)


