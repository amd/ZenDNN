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

  message(DEBUG "${ZENDNNL_MSG_PREFIX}Configuring ZENDNNL BENCHDNN...")

  set(ZENDNNL_BENCHDNN_ROOT ${ZENDNNL_SOURCE_DIR}/benchdnn)

  # project options
  list(APPEND BDN_CMAKE_ARGS  "-DZENDNNL_SOURCE_DIR=${ZENDNNL_SOURCE_DIR}")
  list(APPEND BDN_CMAKE_ARGS  "-DPROJECT_VERSION=${PROJECT_VERSION}")

  # cmake options
  list(APPEND BDN_CMAKE_ARGS  "-DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
  list(APPEND BDN_CMAKE_ARGS  "-DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_MESSAGE_LOG_LEVEL}")
  list(APPEND BDN_CMAKE_ARGS  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
  list(APPEND BDN_CMAKE_ARGS  "-DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}")

  # optional dependencies options
  list(APPEND BDN_CMAKE_ARGS  "-DZENDNNL_DEPENDS_ONEDNN=${ZENDNNL_DEPENDS_ONEDNN}")
  list(APPEND BDN_CMAKE_ARGS  "-DZENDNNL_DEPENDS_LIBXSMM=${ZENDNNL_DEPENDS_LIBXSMM}")
  list(APPEND BDN_CMAKE_ARGS  "-DZENDNNL_DEPENDS_PARLOOPER=${ZENDNNL_DEPENDS_PARLOOPER}")
  list(APPEND BDN_CMAKE_ARGS  "-DZENDNNL_DEPENDS_AOCLDLP=${ZENDNNL_DEPENDS_AOCLDLP}")
  list(APPEND BDN_CMAKE_ARGS  "-DZENDNNL_DEPENDS_AMDBLIS=${ZENDNNL_DEPENDS_AMDBLIS}")

  # other options
  list(APPEND BDN_CMAKE_ARGS  "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "${ZENDNNL_MSG_PREFIX}BENCHDNN_CMAKE_ARGS  = ${BDN_CMAKE_ARGS}")

  ExternalProject_ADD(zendnnl-benchdnn
    DEPENDS    "zendnnl"
    SOURCE_DIR "${ZENDNNL_BENCHDNN_ROOT}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/benchdnn"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/benchdnn"
    CMAKE_ARGS "${BDN_CMAKE_ARGS}"
    BUILD_COMMAND cmake --build .
    INSTALL_COMMAND cmake --install .)

  add_dependencies(zendnnl-benchdnn zendnnl)

  list(APPEND BENCHDNN_CLEAN_FILES "${CMAKE_BINARY_DIR}/benchdnn")
  list(APPEND BENCHDNN_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/benchdnn")

  set_target_properties(zendnnl-benchdnn
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${BENCHDNN_CLEAN_FILES}")

else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building ZENDNNL BENCHDNN will be skipped.")
endif()
