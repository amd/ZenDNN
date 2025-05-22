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
project(ZENDNNL VERSION 1.0
  DESCRIPTION "Accelerated Deep Learning Inference on AMD Zen Architecture"
  LANGUAGES CXX)

# log level
set(CMAKE_MESSAGE_LOG_LEVEL "DEBUG")

# specify build type
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
message(DEBUG "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# global cxx flags
if(ZENDNNL_USE_CXX11_ABI_FLAG)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
endif()

# installation
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

# cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

# cmake modules
include(ExternalProject)
include(CMakeUtils)
include(ConfigOptions)

# common packages
FindCommonPackages()

# find denepnedcies
if(ZENDNNL_DEPENDS_AMDBLIS)
  find_package(AMDBLIS)
  if(NOT AMDBLIS_FOUND)
    message(DEBUG "Setting up external_project build for AMDBLIS")
    include(ExternProjAMDBLIS)
  endif()
endif()

if(ZENDNNL_DEPENDS_ONEDNN)
  get_property(ONEDNN_INSTALL_DIR GLOBAL PROPERTY ONEDNNROOT)
  set(dnnl_ROOT "${ONEDNN_INSTALL_DIR}")
  message(DEBUG "Seraching for oneDNN in ${dnnl_ROOT}")

  find_file(DNNLCONFIG
    NAMES dnnlConfig.cmake dnnl-config.cmake
    PATHS ${dnnl_ROOT}
    PATH_SUFFIXES lib lib/cmake lib/cmake/dnnl)

  if("${DNNLCONFIG}" STREQUAL "DNNLCONFIG-NOTFOUND")
    message(DEBUG "Could not find prebuilt ONEDNN")
    message(DEBUG "Setting up external_project build for ONEDNN")
    include(ExternProjONEDNN)
  else()
    set(dnnl_DIR "${dnnl_ROOT}/lib/cmake/dnnl")
    find_package(dnnl REQUIRED)
  endif()
endif()

if(ZENDNNL_DEPENDS_AOCLUTILS)
  get_property(AU_INSTALL_DIR GLOBAL PROPERTY AOCLUTILSROOT)
  set(aocl-utils_ROOT "${AU_INSTALL_DIR}")
  message(DEBUG "Seraching for aocl-utils in ${aocl-utils_ROOT}")
  find_file(AUCONFIG
    NAMES aocl-utils-config.cmake
    PATHS ${aocl-utils_ROOT}
    PATH_SUFFIXES lib lib/cmake lib/CMake)

  if("${AUCONFIG}" STREQUAL "AUCONFIG-NOTFOUND")
    message(DEBUG "Could not find prebuilt AOCLUTILS")
    message(DEBUG "Setting up external_project build for AOCLUTILS")
    include(ExternProjAOCLUTILS)
  else()
    set(aocl-utils_DIR "${aocl-utils_ROOT}/lib/CMake")
    find_package(aocl-utils REQUIRED)
    include_directories(${aocl-utils_ROOT}/include)
  endif()
endif()

#if(ZENDNNL_DEPENDS_GTEST)
#  get_property(GTEST_INSTALL_DIR GLOBAL PROPERTY GTESTROOT)
#  set(GTest_ROOT "${GTEST_INSTALL_DIR}")
#  message(DEBUG "Seraching for GTEST in ${GTest_ROOT}")
#  find_file(GTESTCONFIG
#    NAMES GTestConfig.cmake
#    PATHS ${GTest_ROOT}
#    PATH_SUFFIXES lib lib/cmake lib/cmake/GTest)

#  if("${GTESTCONFIG}" STREQUAL "GTESTCONFIG-NOTFOUND")
#    message(DEBUG "Could not find prebuilt GTEST")
#    message(DEBUG "Setting up external_project build for GTEST")
#    include(ExternProjGTEST)
#  else()
#    set(GTest_DIR "${GTest_ROOT}/lib/cmake/GTest")
#    find_package(GTest REQUIRED)
    # include_directories(${GTEST_ROOT}/include)
    # include_directories(${GTEST_ROOT}/include/gtest)
    # include_directories(${GTEST_ROOT}/include/gmock)
   #  endif()
#endif()

if(ZENDNNL_DEPENDS_GTEST)
  message(DEBUG "Building gtest library")
  include(FetchGoogleTest)
endif()

# build sources
add_subdirectory(src)

# build examples
if(ZENDNNL_BUILD_EXAMPLES)
  message(DEBUG "Building examples")
  add_subdirectory(examples)
endif()

if(ZENDNNL_BUILD_DOXYGEN_DOCS)
  message(DEBUG "Building doxygen documentation")
  add_subdirectory(doxygen)
endif()

#build gtests
if(ZENDNNL_BUILD_GTESTS)
  message(DEBUG "Building gtests")
  add_subdirectory(gtests)
endif()

# export targets
# disabling export as of now
# install(EXPORT zendnnl_targets
#   FILE zendnnl_targets.cmake
#   NAMESPACE "zendnnl::"
#   DESTINATION lib/cmake/zendnnl)

# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#   ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
#   ${CMAKE_CURRENT_BINARY_DIR}/zendnnlConfig.cmake
#   INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zendnnl)

# write_basic_package_version_file(
#   "zendnnlConfigVersion.cmake"
#   VERSION ${zendnnl_VERSION}
#   COMPATIBILITY AnyNewerVersion)

# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/zendnnlConfig.cmake"
#   "${CMAKE_CURRENT_BINARY_DIR}/zendnnlConfigVersion.cmake"
#   DESTINATION lib/cmake/zendnnl)
