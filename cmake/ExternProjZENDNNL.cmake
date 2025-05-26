#  *******************************************************************************
#  * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  *******************************************************************************
include_guard(GLOBAL)

include(ZenDnnlOptions)

message(STATUS "building zendnnl library")

set(ZENDNNL_ROOT ${ZENDNNL_SOURCE_DIR}/zendnnl)

message(DEBUG "ZENDNNL_DEPS=${ZENDNNL_DEPS}")

list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_SOURCE_DIR=${ZENDNNL_SOURCE_DIR}")
list(APPEND ZL_CMAKE_ARGS "-DPROJECT_VERSION=${PROJECT_VERSION}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_MESSAGE_LOG_LEVEL}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

message(DEBUG "ZL_CMAKE_ARGS = ${ZL_CMAKE_ARGS}")

# cmake install prefix need to be same as projects install prefix, as all the
# paths will be computed relative to it.

ExternalProject_ADD(zendnnl_library
  DEPENDS "${ZENDNNL_DEPS}"
  SOURCE_DIR "${ZENDNNL_ROOT}"
  BINARY_DIR "${CMAKE_BINARY_DIR}/zendnnl"
  INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS "${ZL_CMAKE_ARGS}"
  BUILD_COMMAND cmake --build .
  INSTALL_COMMAND cmake --install .)

