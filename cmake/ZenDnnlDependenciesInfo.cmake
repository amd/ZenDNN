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
#set(ZENDNNL_DEPS_DIR "${ZENDNNL_SOURCE_DIR}/dependencies")

# aocl-dlp repo information
set(AOCLDLP_ROOT_DIR "${ZENDNNL_DEPS_DIR}/aocldlp"
  CACHE PATH "AOCL DLP root dir")
set(AOCLDLP_GIT_REPO "https://github.com/amd/aocl-dlp.git")
# aocl-dlp tag AOCL-Weekly-101025
set(AOCLDLP_GIT_TAG "f837cc5efcd41f68a11de423af66639cd87fa80c")
option(AOCLDLP_GIT_PROGRESS ON)

# amdblis repo information
set(AMDBLIS_ROOT_DIR "${ZENDNNL_DEPS_DIR}/amdblis"
  CACHE PATH "AMD BLIS root dir")
set(AMDBLIS_GIT_REPO "https://github.com/amd/blis.git")
# amdblis tag AOCL-Sep2025-b1
set(AMDBLIS_GIT_TAG "fb2a682725bdb56cdacb31d4ab7430c42b2eb24b")
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
set(ONEDNN_GIT_TAG "v3.7.1")
option(ONEDNN_GIT_PROGRESS ON)

# libxsmm repo information
set(LIBXSMM_ROOT_DIR "${ZENDNNL_DEPS_DIR}/libxsmm"
  CACHE PATH "LIBXSMM root dir")
set(LIBXSMM_GIT_REPO "https://github.com/libxsmm/libxsmm.git")
set(LIBXSMM_GIT_TAG "eedaa03d49a1dffe6048711598bc5a4da5a86008")
option(LIBXSMM_GIT_PROGRESS ON)

# parlooper repo information
set(PARLOOPER_ROOT_DIR "${ZENDNNL_DEPS_DIR}/parlooper"
  CACHE PATH "PARLOOPER root dir")
set(PARLOOPER_GIT_REPO "https://github.com/libxsmm/parlooper.git")
set(PARLOOPER_GIT_TAG "630b6396369c2dab1fd96372c054cd1f34c35e7e")
option(PARLOOPER_GIT_PROGRESS ON)