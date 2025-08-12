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

find_package(OpenMP REQUIRED)

message(STATUS "building zendnnl library")

set(ZENDNNL_ROOT ${ZENDNNL_SOURCE_DIR}/zendnnl)

message(DEBUG "ZENDNNL_DEPS=${ZENDNNL_DEPS}")

list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_SOURCE_DIR=${ZENDNNL_SOURCE_DIR}")
list(APPEND ZL_CMAKE_ARGS "-DPROJECT_VERSION=${PROJECT_VERSION}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_MESSAGE_LOG_LEVEL}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_BUILD_GTEST=${ZENDNNL_BUILD_GTEST}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_DEPENDS_ONEDNN=${ZENDNNL_DEPENDS_ONEDNN}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_CODE_COVERAGE=${ZENDNNL_CODE_COVERAGE}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

message(DEBUG "ZL_CMAKE_ARGS = ${ZL_CMAKE_ARGS}")
cmake_host_system_information(RESULT NPROC QUERY NUMBER_OF_PHYSICAL_CORES)

# cmake install prefix need to be same as projects install prefix, as all the
# paths will be computed relative to it.

ExternalProject_ADD(zendnnl
  DEPENDS "zendnnl-deps"
  SOURCE_DIR "${ZENDNNL_ROOT}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/zendnnl"
  INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS "${ZL_CMAKE_ARGS}"
  BUILD_COMMAND cmake --build . --target all -- -j${NPROC}
  INSTALL_COMMAND cmake --build .  --target install
  BUILD_BYPRODUCTS <INSTALL_DIR>/zendnnl/lib/libzendnnl_archive.a
  BUILD_ALWAYS TRUE
  CONFIGURE_HANDLED_BY_BUILD TRUE)

list(APPEND ZENDNNL_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/zendnnl")
list(APPEND ZENDNNL_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/zendnnl")
list(APPEND ZENDNNL_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/gtests")

set_target_properties(zendnnl
  PROPERTIES
  ADDITIONAL_CLEAN_FILES "${ZENDNNL_CLEAN_FILES}")

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

set(ZENDNNL_LIBRARY_INC_DIR "${CMAKE_INSTALL_PREFIX}/zendnnl/include")
set(ZENDNNL_LIBRARY_LIB_DIR "${CMAKE_INSTALL_PREFIX}/zendnnl/lib")
set(ZENDNNL_JSON_INC_DIR "${CMAKE_INSTALL_PREFIX}/deps/json/include")

file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
add_library(zendnnl_library STATIC IMPORTED GLOBAL)

add_dependencies(zendnnl_library
  zendnnl zendnnl-deps)

set_target_properties(zendnnl_library
  PROPERTIES
  IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
  INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
  INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR};${ZENDNNL_JSON_INC_DIR}")

if (ZENDNNL_DEPENDS_ONEDNN)
  target_link_libraries(zendnnl_library
    INTERFACE ${CMAKE_DL_LIBS}
    INTERFACE OpenMP::OpenMP_CXX
    INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>"
    INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,DNNL::dnnl>"
    INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,amdblis::amdblis_archive>")
else()
  target_link_libraries(zendnnl_library
    INTERFACE ${CMAKE_DL_LIBS}
    INTERFACE OpenMP::OpenMP_CXX
    INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>"
    INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,amdblis::amdblis_archive>")
endif()

target_link_options(zendnnl_library INTERFACE "-fopenmp")

add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

list(APPEND ZENDNNL_LINK_LIBS "zendnnl_library")
list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_LIBRARY_INC_DIR})

# !!!
