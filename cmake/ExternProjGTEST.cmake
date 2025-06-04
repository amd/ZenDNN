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

if(ZENDNNL_BUILD_GTEST)
  list(APPEND GTEST_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "GTEST_CMAKE_ARGS=${GTEST_CMAKE_ARGS}")

  ExternalProject_ADD(zendnnl-deps-gtest
    SOURCE_DIR "${GTEST_ROOT_DIR}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/gtest"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/gtest"
    GIT_REPOSITORY ${GTEST_GIT_REPO}
    GIT_TAG ${GTEST_GIT_TAG}
    GIT_PROGRESS ${GTEST_GIT_PROGRESS}
    CMAKE_ARGS ${GTEST_CMAKE_ARGS}
    INSTALL_COMMAND cmake --build . --config release --target install -j
    UPDATE_DISCONNECTED TRUE)

  list(APPEND GTEST_CLEAN_FILES "${CMAKE_BINARY_DIR}/gtest")
  list(APPEND GTEST_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/gtest")

  set_target_properties(zendnnl-deps-gtest
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${GTEST_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-gtest")
else()
  message(DEBUG "skipping building gtests...")
endif()
