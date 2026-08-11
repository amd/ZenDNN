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
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Configurig GTEST...")

  # Option to include AI gtests
  option(ZENDNNL_BUILD_AI_GTESTS "Build AI-specific Google Tests" ON)

  if(ZENDNNL_BUILD_AI_GTESTS)
    message(STATUS "AI gtests will be included in the build")
  else()
    message(STATUS "AI gtests will be excluded from the build")
  endif()

  list(APPEND GTEST_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  # On Windows, googletest must be built with the same compiler, build type, and
  # C runtime as ZenDNN or linking the gtests binary fails with CRT /
  # _ITERATOR_DEBUG_LEVEL mismatches (LNK2038) and unresolved debug-CRT symbols.
  # Without CMAKE_BUILD_TYPE the sub-build produces a Debug gtest.lib (MDd) while
  # the rest of the tree is Release (MD); googletest also defaults to the static
  # CRT (/MT) on MSVC, so force the shared (DLL) CRT to match ZenDNN's -MD.
  if(WIN32)
    list(APPEND GTEST_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
    list(APPEND GTEST_CMAKE_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
    list(APPEND GTEST_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release")
    list(APPEND GTEST_CMAKE_ARGS "-Dgtest_force_shared_crt=ON")
  endif()

  message(DEBUG "${ZENDNNL_MSG_PREFIX}GTEST_CMAKE_ARGS=${GTEST_CMAKE_ARGS}")

  set(NPROC ${ZENDNNL_BUILD_SYS_NPROC})
  ExternalProject_ADD(zendnnl-gtest
    SOURCE_DIR "${GTEST_ROOT_DIR}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/gtest"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/gtest"
    GIT_REPOSITORY ${GTEST_GIT_REPO}
    GIT_TAG ${GTEST_GIT_TAG}
    GIT_PROGRESS ${GTEST_GIT_PROGRESS}
    CMAKE_ARGS ${GTEST_CMAKE_ARGS}
    BUILD_COMMAND cmake --build . --config Release --target all -- -j${NPROC}
    INSTALL_COMMAND cmake --build . --config Release --target install
    UPDATE_DISCONNECTED TRUE)

  list(APPEND GTEST_CLEAN_FILES "${CMAKE_BINARY_DIR}/gtest")
  list(APPEND GTEST_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/gtest")

  set_target_properties(zendnnl-gtest
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${GTEST_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-gtest")
else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building GTEST will be skipped.")
endif()
