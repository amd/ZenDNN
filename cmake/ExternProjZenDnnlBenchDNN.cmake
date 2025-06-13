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

if(ZENDNNL_BUILD_BENCHDNN)
  message(STATUS "building benchdnn...")

  set(ZENDNNL_BENCHDNN_ROOT ${ZENDNNL_SOURCE_DIR}/benchdnn)

  list(APPEND ZLE_CMAKE_ARGS "-DZENDNNL_SOURCE_DIR=${ZENDNNL_SOURCE_DIR}")
  list(APPEND ZLE_CMAKE_ARGS "-DPROJECT_VERSION=${PROJECT_VERSION}")
  list(APPEND ZLE_CMAKE_ARGS "-DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_MESSAGE_LOG_LEVEL}")
  list(APPEND ZLE_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
  list(APPEND ZLE_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "ZLE_CMAKE_ARGS = ${ZLE_CMAKE_ARGS}")

  ExternalProject_ADD(zendnnl-benchdnn
    DEPENDS    "zendnnl"
    SOURCE_DIR "${ZENDNNL_BENCHDNN_ROOT}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/benchdnn"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/benchdnn"
    CMAKE_ARGS "${ZLE_CMAKE_ARGS}"
    BUILD_COMMAND cmake --build .
    INSTALL_COMMAND cmake --install .)

  list(APPEND BENCHDNN_CLEAN_FILES "${CMAKE_BINARY_DIR}/benchdnn")
  list(APPEND BENCHDNN_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/benchdnn")

  set_target_properties(zendnnl-benchdnn
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${BENCHDNN_CLEAN_FILES}")

else()
  message(DEBUG "skipping benchdnn examples...")
endif()
