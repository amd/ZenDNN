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

if(ZENDNNL_DEPENDS_AMDBLIS)
  list(APPEND AMDBLIS_CMAKE_ARGS "-DBLIS_CONFIG_FAMILY=amdzen")
  list(APPEND AMDBLIS_CMAKE_ARGS "-DENABLE_ADDON=aocl_gemm")
  list(APPEND AMDBLIS_CMAKE_ARGS "-DENABLE_THREADING=openmp")
  list(APPEND AMDBLIS_CMAKE_ARGS "-DENABLE_CBLAS=ON")
  list(APPEND AMDBLIS_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "AMDBLIS_CMAKE_ARGS=${AMDBLIS_CMAKE_ARGS}")

  ExternalProject_ADD(zendnnl-deps-amdblis
    SOURCE_DIR "${AMDBLIS_ROOT_DIR}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/amdblis"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/amdblis"
    GIT_REPOSITORY ${AMDBLIS_GIT_REPO}
    GIT_TAG ${AMDBLIS_GIT_TAG}
    GIT_PROGRESS ${AMDBLIS_GIT_PROGRESS}
    CMAKE_ARGS ${AMDBLIS_CMAKE_ARGS}
    BUILD_COMMAND cmake --build . --config release --target install -j
    UPDATE_DISCONNECTED TRUE)

  list(APPEND AMDBLIS_CLEAN_FILES "${CMAKE_BINARY_DIR}/amdblis")
  list(APPEND AMDBLIS_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/amdblis")

  set_target_properties(zendnnl-deps-amdblis
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${AMDBLIS_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-amdblis")
else()
  message(DEBUG "skipping building amdblis.")
endif()



