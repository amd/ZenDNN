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

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# zendnnl module path
set(ZENDNNL_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake"
  CACHE PATH "zendnnl module path" FORCE)
set(CMAKE_MODULE_PATH "${ZENDNNL_MODULE_PATH}")

# cmake modules
include(ExternalProject)
include(CMakeDependentOption)

# zendnnl modules
include(ZenDnnlMacros)
include(ZenDnnlOptions)
include(ZenDnnlDependenciesInfo)

# build system information
if(NOT DEFINED ZENDNNL_BUILD_SYS_NPROC)
  cmake_host_system_information(RESULT ZENDNNL_BUILD_SYS_NPROC QUERY NUMBER_OF_PHYSICAL_CORES)
endif()

# openmp support
find_package(OpenMP REQUIRED QUIET)

# build dependencies
include(BuildDependencies)

# build zendnnl library
include(ExternProjZENDNNL)
include(ExternProjZenDnnlExamples)
include(ExternProjZenDnnlBenchDNN)

# build doxygen docs
add_subdirectory(docs/doxygen)

