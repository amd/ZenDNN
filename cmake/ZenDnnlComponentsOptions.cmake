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
include_guard(GLOBAL)

set(ZENDNNL_BUILD_DEPS      ON CACHE BOOL "Build depedencies")
set(ZENDNNL_BUILD_EXAMPLES  ON CACHE BOOL "Build examples")
set(ZENDNNL_BUILD_GTEST    OFF CACHE BOOL "Build gtest")
set(ZENDNNL_BUILD_DOXYGEN  OFF CACHE BOOL "Build doxygen docs")
set(ZENDNNL_CODE_COVERAGE  OFF CACHE BOOL "Enable code coverage instrumentation")
set(ZENDNNL_BUILD_BENCHDNN OFF CACHE BOOL "Build benchdnn")
set(ZENDNNL_BUILD_ASAN     OFF CACHE BOOL "Build With Address Sanitizer")

set(ZENDNNL_LIB_BUILD_ARCHIVE  ON CACHE BOOL "Build zendnnl archive library")
set(ZENDNNL_LIB_BUILD_SHARED  OFF CACHE BOOL "Build zendnl shared library")

# sanity check
if((NOT ZENDNNL_LIB_BUILD_ARCHIVE) AND (NOT ZENDNNL_LIB_BUILD_SHARED))
  message(FATAL_ERROR,
  "At least one of the zendnnl shared libary or zendnnl archive library need to be built.")
endif()

# informative message
if(ZENDNNL_LIB_BUILD_ARCHIVE)
  list(APPEND ZENDNNL_MSG_COMPONENTS "archive zendnnl lib")
endif()
if(ZENDNNL_LIB_BUILD_SHARED)
  list(APPEND ZENDNNL_MSG_COMPONENTS "shared zendnnl lib")
endif()
if(ZENDNNL_BUILD_GTEST)
  list(APPEND ZENDNNL_MSG_COMPONENTS "zendnnl gtests")
endif()
if(ZENDNNL_BUILD_DEPS)
  list(APPEND ZENDNNL_MSG_COMPONENTS "dependencies")
endif()
if(ZENDNNL_BUILD_EXAMPLES)
  list(APPEND ZENDNNL_MSG_COMPONENTS "examples")
endif()
if(ZENDNNL_BUILD_DOXYGEN)
  list(APPEND ZENDNNL_MSG_COMPONENTS "doxygen-documentation")
endif()
if(ZENDNNL_CODE_COVERAGE)
  list(APPEND ZENDNNL_MSG_COMPONENTS "code-coverage")
endif()
if(ZENDNNL_BUILD_BENCHDNN)
  list(APPEND ZENDNNL_MSG_COMPONENTS "benchdnn")
endif()

message(STATUS "${ZENDNNL_MSG_PREFIX}Components : ${ZENDNNL_MSG_COMPONENTS}")

