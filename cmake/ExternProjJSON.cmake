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

  if(ZENDNNL_LOCAL_JSON)

    message(DEBUG "Using local JSON from ${JSON_ROOT_DIR}")

    ExternalProject_ADD(zendnnl-deps-json
      SOURCE_DIR "${JSON_ROOT_DIR}"
      BINARY_DIR "${CMAKE_BINARY_DIR}/json"
      INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/json"
      CMAKE_ARGS ${JSON_CMAKE_ARGS}
      INSTALL_COMMAND cmake --build . --config release --target install -j)
  else()
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
  endif()

  list(APPEND JSON_CLEAN_FILES "${CMAKE_BINARY_DIR}/json")
  list(APPEND JSON_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/json")

  set_target_properties(zendnnl-deps-json
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${JSON_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-json")

  # !!!
  # HACK to make zendnnl a sub-project using add_directory() !
  # ZenDNNL packages the exported information of this dependency in its own
  # package config file. However to use this information, ZenDNNL package
  # need to be installed.
  # when ZenDNNL is included as a sub-project using add_subdirectory it
  # can not be installed before the super-project of which it is a sub-project
  # is built. Thus super-project can not use its package information. This
  # makes this hack of manually interfacing the libraries this dependency has
  # built necessary.
  # Since we do not know what kind of information and targets this package exports
  # this kind of manual interface could be error-prone.
  #
  # UNCOMMENT the code below for manual interface.

  set(ZENDNNL_JSON_INC_DIR "${CMAKE_INSTALL_PREFIX}/deps/json/include")

  file(MAKE_DIRECTORY ${ZENDNNL_JSON_INC_DIR})
  add_library(zendnnl_json_deps INTERFACE IMPORTED GLOBAL)
  add_dependencies(zendnnl_json_deps zendnnl-deps-json)
  set_target_properties(zendnnl_json_deps PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "\$<\$<NOT:\$<BOOL:ON>>:JSON_USE_GLOBAL_UDLS=0>;\$<\$<NOT:\$<BOOL:ON>>:JSON_USE_IMPLICIT_CONVERSIONS=0>;\$<\$<BOOL:OFF>:JSON_DISABLE_ENUM_SERIALIZATION=1>;\$<\$<BOOL:OFF>:JSON_DIAGNOSTICS=1>;\$<\$<BOOL:OFF>:JSON_DIAGNOSTIC_POSITIONS=1>;\$<\$<BOOL:OFF>:JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON=1>"
    INTERFACE_COMPILE_FEATURES "cxx_std_11"
    INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_JSON_INC_DIR};${ZENDNNL_JSON_INC_DIR}")

  add_library(nlohmann_json::nlohmann_json ALIAS zendnnl_json_deps)
  list(APPEND ZENDNNL_LINK_LIBS "nlohmann_json::nlohmann_json")

else()
  message(DEBUG "skipping building json...")
endif()
