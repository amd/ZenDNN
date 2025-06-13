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
project(ZENDNNL VERSION 1.0.0
  DESCRIPTION "Accelerated Deep Learning Inference on AMD Zen Architecture"
  LANGUAGES CXX)

set(CMAKE_MESSAGE_LOG_LEVEL "DEBUG")

# display project information
message(DEBUG "PROJECT_NAME = ${PROJECT_NAME}")
message(DEBUG "PROJECT_VERSION = ${PROJECT_VERSION}")
message(DEBUG "ZENDNNL_SOURCE_DIR=${ZENDNNL_SOURCE_DIR}")
message(DEBUG "ZENDNNL_BINARY_DIR=${ZENDNNL_BINARY_DIR}")

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# cmake variables
set(ZENDNNL_INSTALL_PREFIX "${ZENDNNL_BINARY_DIR}/install")
set(ZENDNNL_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE Release)
message(DEBUG "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
set(CMAKE_INSTALL_PREFIX "${ZENDNNL_INSTALL_PREFIX}")
message(DEBUG "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ZENDNNL_SOURCE_DIR}/cmake)
message(DEBUG "CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

# cmake modules
include(ExternalProject)

# zendnnl modules
include(ZenDnnlMacros)
include(ZenDnnlOptions)

# build dependencies
include(BuildDependencies)

# build zendnnl library
include(ExternProjZENDNNL)
include(ExternProjZenDnnlExamples)

# build doxygen docs
add_subdirectory(docs/doxygen)

# !!!
# HACK to make zendnnl a sub-project using add_directory() !
# ZenDNNL packages the exported information of this dependency in its own
# package config file. However to use this information, ZenDNNL package
# need to be installed.
# when ZenDNNL is included as a sub-project using add_subdirectory it
# can not be installed before the super-project of which it is a sub-project
# is built. Thus super-project can not use its package information. This
# makes this hack of manually interfacing the libraries necessary.
# Since we do not know what kind of information and targets dependencies export,
# this kind of manual interface could be error-prone.
#
# UNCOMMENT the code below for manual interface.



add_library(zendnnl_zendnnl_archive INTERFACE)
add_dependencies(zendnnl_zendnnl_archive zendnnl_library)

target_link_libraries(zendnnl_zendnnl_archive
  INTERFACE ${ZENDNNL_LINK_LIBS})
target_include_directories(zendnnl_zendnnl_archive
  INTERFACE ${ZENDNNL_INCLUDE_DIRECTORIES})

add_library(zendnnl::zendnnl_archive ALIAS zendnnl_zendnnl_archive)

# !!!
