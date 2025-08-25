# *******************************************************************************
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
include_guard(GLOBAL)

# include dependencies options
include(ZenDnnlDependenciesOptions)

# check if top level project dir is set
if(NOT DEFINED ZENDNNL_SOURCE_DIR)
  message(FATAL_ERROR "ZENDNNL_SOURCE_DIR is undefined")
  return()
endif()

# dependencies directory
set(ZENDNNL_DEPS_DIR "${ZENDNNL_SOURCE_DIR}/dependencies")
message(DEBUG "ZENDNNL_DEPS_DIR=${ZENDNNL_DEPS_DIR}")

# amdblis repo information
set(AMDBLIS_ROOT_DIR "${ZENDNNL_DEPS_DIR}/amdblis"
  CACHE PATH "AMD BLIS root dir")
set(AMDBLIS_GIT_REPO "https://github.com/amd/blis.git")
# amdblis tag AOCL-Aug2025-b1
set(AMDBLIS_GIT_TAG "c96e7eb197c6860338ff76bb6631b60b3e3644de")
option(AMDBLIS_GIT_PROGRESS ON)

# aocl-utils repo information
set(AOCLUTILS_ROOT_DIR "${ZENDNNL_DEPS_DIR}/aoclutils"
  CACHE PATH "AOCL UTILS root dir")
set(AOCLUTILS_GIT_REPO "https://github.com/amd/aocl-utils")
set(AOCLUTILS_GIT_TAG "5.0")

# gtest repo information
set(GTEST_ROOT_DIR "${ZENDNNL_DEPS_DIR}/gtest"
  CACHE PATH "GTEST root dir")
set(GTEST_GIT_REPO "https://github.com/google/googletest.git")
set(GTEST_GIT_TAG "v1.17.0")
option(GTEST_GIT_PROGRESS ON)

# JSON repo information
set(JSON_ROOT_DIR "${ZENDNNL_DEPS_DIR}/json"
  CACHE PATH "json root dir")
set(JSON_GIT_REPO "https://github.com/nlohmann/json.git")
set(JSON_GIT_TAG "v3.12.0")
option(JSON_GIT_PROGRESS ON)

# oneDNN repo information
set(ONEDNN_ROOT_DIR "${ZENDNNL_DEPS_DIR}/onednn"
  CACHE PATH "ONEDNN root dir")
set(ONEDNN_GIT_REPO "https://github.com/oneapi-src/oneDNN.git")
set(ONEDNN_GIT_TAG "v3.8.1")
option(ONEDNN_GIT_PROGRESS ON)

