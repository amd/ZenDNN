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
if (ZENDNNL_DEPENDS_ONEDNN)
  get_property(ONEDNN_INSTALL_DIR GLOBAL PROPERTY ONEDNNROOT)

  list(APPEND ONEDNN_CMAKE_ARGS "-DONEDNN_BUILD_GRAPH=OFF")
  list(APPEND ONEDNN_CMAKE_ARGS "-DONEDNN_BUILD_TESTS=OFF")
  list(APPEND ONEDNN_CMAKE_ARGS "-DONEDNN_BUILD_EXAMPLES=OFF")
  list(APPEND ONEDNN_CMAKE_ARGS "-DONEDNN_LIBRARY_TYPE=STATIC")
  list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release")
  list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "ONEDNN_CMAKE_ARGS=${ONEDNN_CMAKE_ARGS}")
  cmake_host_system_information(RESULT NPROC QUERY NUMBER_OF_PHYSICAL_CORES)

  if (ZENDNNL_LOCAL_ONEDNN)
    message(DEBUG "Using local OneDNN from ${ONEDNN_ROOT_DIR}")
    ExternalProject_ADD(zendnnl-deps-onednn
      SOURCE_DIR "${ONEDNN_ROOT_DIR}"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn"
      INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/onednn"
      CMAKE_ARGS ${ONEDNN_CMAKE_ARGS}
      BUILD_COMMAND cmake --build . --config release --target all -- -j${NPROC}
      INSTALL_COMMAND cmake --build . --config release --target install
      BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libdnnl.a)
  else()
    ExternalProject_ADD(zendnnl-deps-onednn
      SOURCE_DIR "${ONEDNN_ROOT_DIR}"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn"
      INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/onednn"
      GIT_REPOSITORY ${ONEDNN_GIT_REPO}
      GIT_TAG ${ONEDNN_GIT_TAG}
      GIT_PROGRESS ${ONEDNN_GIT_PROGRESS}
      CMAKE_ARGS ${ONEDNN_CMAKE_ARGS}
      BUILD_COMMAND cmake --build . --config release --target all -- -j${NPROC}
      INSTALL_COMMAND cmake --build . --config release --target install
      BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libdnnl.a
      UPDATE_DISCONNECTED TRUE)
  endif()

  list(APPEND ONEDNN_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/onednn")
  list(APPEND ONEDNN_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/onednn")

  set_target_properties(zendnnl-deps-onednn
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${ONEDNN_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-onednn")

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

  set(ONEDNN_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/deps/onednn/include)
  set(ONEDNN_LIB_DIR ${CMAKE_INSTALL_PREFIX}/deps/onednn/lib)

  file(MAKE_DIRECTORY ${ONEDNN_INCLUDE_DIR})
  add_library(DNNL_dnnl STATIC IMPORTED GLOBAL)
  add_dependencies(DNNL_dnnl zendnnl-deps-onednn)
  set_target_properties(DNNL_dnnl
    PROPERTIES IMPORTED_LOCATION "${ONEDNN_LIB_DIR}/libdnnl.a"
    INCLUDE_DIRECTORIES "${ONEDNN_INCLUDE_DIR}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_INCLUDE_DIR}")

  add_library(DNNL::dnnl ALIAS DNNL_dnnl)

  list(APPEND ZENDNNL_LINK_LIBS "DNNL::dnnl")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ONEDNN_INCLUDE_DIR})

  # !!!
else()
message(DEBUG "skipping building onednn.")
endif()