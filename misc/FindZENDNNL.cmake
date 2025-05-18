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

set(ZENDNNL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL"
  CACHE STRING "zendnnl root path")

set(ZENDNNL_DEPENDENCIES_ROOT "${ZENDNNL_ROOT}/dependencies"
  CACHE STRING "zendnnl dependencies root path")

set(ZENDNNL_INSTALL_PATH "${ZENDNNL_ROOT}/build/install")
set(ZENDNNL_LIB_PATH "${ZENDNNL_INSTALL_PATH}/lib")
set(ZENDNNL_INCLUDE_PATH "${ZENDNNL_INSTALL_PATH}/include")


find_library(ZENDNNL_ARCHIVE_LIB
  NAMES libzenai_archive.a
  PATHS ${ZENDNNL_LIB_PATH}
  REQUIRED)

find_library(ZENDNNL_LIB
  NAMES libzenai.so
  PATHS ${ZENDNNL_LIB_PATH}
  REQUIRED)

find_path(ZENDNNL_INCLUDE_DIR
  NAMES zendnnl.hpp
  PATHS ${ZENDNNL_INCLUDE_PATH}
  REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZENDNNL
  DEFAULT_MSG
  ZENDNNL_ARCHIVE_LIB
  ZENDNNL_LIB
  ZENDNNL_INCLUDE_DIR)

if(ZENDNNL_FOUND)
  # file(COPY ${ZENDNNL_LIB_PATH}/libzenai_archive.a
  #   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)

  # list(APPEND ZENDNNL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib/libzenai_archive.a)

  # mark_as_advanced(ZENDNNL_INCLUDE_DIR
  #   ZENDNNL_LIBRARIES
  #   zenai_archive)

  add_library(zenddnnl::zendnnl_archive STATIC IMPORTED GLOBAL)
  set_target_properties(zenai::zenai_archive
    PROPERTIES
    IMPORTED_LOCATION ${ZENDNNL_ARCHIVE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${ZENDNNL_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${ZENDNNL_INCLUDE_DIR})
  mark_as_advanced(zendnnl::zendnnl_archive)

  add_library(zendnnl::zendnnl SHARED IMPORTED GLOBAL)
  set_target_properties(zendnnl::zendnnl
    PROPERTIES
    IMPORTED_LOCATION ${ZENDNNL_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${ZENDNNL_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${ZENDNNL_INCLUDE_DIR})
  mark_as_advanced(zendnnl::zendnnl)
endif()
