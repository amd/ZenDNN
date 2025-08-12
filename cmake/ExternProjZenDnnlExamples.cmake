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

if(ZENDNNL_BUILD_EXAMPLES)
  message(STATUS "building examples...")

  set(ZENDNNL_EXAMPLES_ROOT ${ZENDNNL_SOURCE_DIR}/examples)

  list(APPEND ZLE_CMAKE_ARGS "-DZENDNNL_SOURCE_DIR=${ZENDNNL_SOURCE_DIR}")
  list(APPEND ZLE_CMAKE_ARGS "-DPROJECT_VERSION=${PROJECT_VERSION}")
  list(APPEND ZLE_CMAKE_ARGS "-DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_MESSAGE_LOG_LEVEL}")
  list(APPEND ZLE_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
  list(APPEND ZLE_CMAKE_ARGS "-DZENDNNL_DEPENDS_ONEDNN=${ZENDNNL_DEPENDS_ONEDNN}")
  list(APPEND ZLE_CMAKE_ARGS "-DZENDNNL_CODE_COVERAGE=${ZENDNNL_CODE_COVERAGE}")
  list(APPEND ZLE_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "ZLE_CMAKE_ARGS = ${ZLE_CMAKE_ARGS}")

  ExternalProject_ADD(zendnnl-examples
    DEPENDS    "zendnnl"
    SOURCE_DIR "${ZENDNNL_EXAMPLES_ROOT}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/examples"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/examples"
    CMAKE_ARGS "${ZLE_CMAKE_ARGS}"
    BUILD_COMMAND cmake --build .
    INSTALL_COMMAND cmake --install .)

  list(APPEND EXAMPLES_CLEAN_FILES "${CMAKE_BINARY_DIR}/examples")
  list(APPEND EXAMPLES_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/examples")

  set_target_properties(zendnnl-examples
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${EXAMPLES_CLEAN_FILES}")

else()
  message(DEBUG "skipping building examples...")
endif()
