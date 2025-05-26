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

if(ZENDNNL_DEPENDS_AOCLUTILS)
  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_EXAMPLES=ON")
  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_DOCS=OFF")
  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_TESTS=OFF")
  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_STATIC_LIBS=ON")
  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_SHARED_LIBS=ON")
  list(APPEND AU_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "AU_CMAKE_ARGS=${AU_CMAKE_ARGS}")

  ExternalProject_ADD(zendnnl_deps_aoclutils
    SOURCE_DIR "${AOCLUTILS_ROOT_DIR}"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/aoclutils"
    GIT_REPOSITORY ${AOCLUTILS_GIT_REPO}
    GIT_TAG ${AOCLUTILS_GIT_TAG}
    GIT_PROGRESS ${AOCLUTILS_GIT_PROGRESS}
    CMAKE_ARGS ${AU_CMAKE_ARGS}
    INSTALL_COMMAND cmake --build . --config release --target install -j
    UPDATE_DISCONNECTED TRUE)

  list(APPEND ZENDNNL_DEPS "zendnnl_deps_aoclutils")
else()
  message(DEBUG "skipping building aocl-utils.")
endif()


