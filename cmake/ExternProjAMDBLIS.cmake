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
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/amdblis"
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/amdblis"
    GIT_REPOSITORY ${AMDBLIS_GIT_REPO}
    GIT_TAG ${AMDBLIS_GIT_TAG}
    GIT_PROGRESS ${AMDBLIS_GIT_PROGRESS}
    CMAKE_ARGS ${AMDBLIS_CMAKE_ARGS}
    BUILD_COMMAND cmake --build . --config release --target install -j
    BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libblis-mt.a
    UPDATE_DISCONNECTED TRUE)

  list(APPEND AMDBLIS_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/amdblis")
  list(APPEND AMDBLIS_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/amdblis")

  set_target_properties(zendnnl-deps-amdblis
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${AMDBLIS_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-amdblis")

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

  set(ZENDNNL_AMDBLIS_INC_DIR "${CMAKE_INSTALL_PREFIX}/deps/amdblis/include")
  set(ZENDNNL_AMDBLIS_LIB_DIR "${CMAKE_INSTALL_PREFIX}/deps/amdblis/lib")

  file(MAKE_DIRECTORY ${ZENDNNL_AMDBLIS_INC_DIR})
  add_library(zendnnl_amdblis_deps STATIC IMPORTED GLOBAL)
  add_dependencies(zendnnl_amdblis_deps zendnnl-deps-amdblis)
  set_target_properties(zendnnl_amdblis_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AMDBLIS_LIB_DIR}/libblis-mt.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_AMDBLIS_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AMDBLIS_INC_DIR}")

  list(APPEND ZENDNNL_LINK_LIBS "zendnnl_amdblis_deps")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_AMDBLIS_INC_DIR})

  # !!!
else()
  message(DEBUG "skipping building amdblis.")
endif()



