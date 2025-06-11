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

if(ZENDNNL_DEPENDS_JSON)
  list(APPEND JSON_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "JSON_CMAKE_ARGS=${JSON_CMAKE_ARGS}")

  ExternalProject_ADD(zendnnl-deps-json
    SOURCE_DIR "${JSON_ROOT_DIR}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/json"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/json"
    GIT_REPOSITORY ${JSON_GIT_REPO}
    GIT_TAG ${JSON_GIT_TAG}
    GIT_PROGRESS ${JSON_GIT_PROGRESS}
    CMAKE_ARGS ${JSON_CMAKE_ARGS}
    INSTALL_COMMAND cmake --build . --config release --target install -j
    UPDATE_DISCONNECTED TRUE)

  list(APPEND JSON_CLEAN_FILES "${CMAKE_BINARY_DIR}/json")
  list(APPEND JSON_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/json")

  set_target_properties(zendnnl-deps-json
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${JSON_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-json")
else()
  message(DEBUG "skipping building json...")
endif()
