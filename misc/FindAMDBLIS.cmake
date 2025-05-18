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

set(AMDBLIS_ROOT "${ZENDNNL_DEPENDENCIES_ROOT}/blis/")
set(AMDBLIS_INSTALL_PATH "${AMDBLIS_ROOT}/build/install")
set(AMDBLIS_LIB_PATH "${AMDBLIS_INSTALL_PATH}/lib")
set(AMDBLIS_INCLUDE_PATH "${AMDBLIS_INSTALL_PATH}/include")

find_library(AMDBLIS_ARCHIVE_LIB
  NAMES libblis-mt.a libblis.a
  PATHS ${AMDBLIS_LIB_PATH}
  PATH_SUFFIXES LP64 amdzen blis)

find_library(AMDBLIS_LIB
  NAMES libblis-mt.so libblis.so
  PATHS ${AMDBLIS_LIB_PATH}
  PATH_SUFFIXES LP64 amdzen blis)

find_path(AMDBLIS_INCLUDE_DIR
  NAMES blis.h blis.hh cblas.h cblas.hh
  PATHS ${AMDBLIS_INCLUDE_PATH}
  PATH_SUFFIXES LP64 amdzen blis)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMDBLIS
  DEFAULT_MSG
  AMDBLIS_LIB
  AMDBLIS_ARCHIVE_LIB
  AMDBLIS_INCLUDE_DIR)

if(AMDBLIS_FOUND)
    add_library(amdblis::amdblis SHARED IMPORTED GLOBAL)
    set_target_properties(amdblis::amdblis
      PROPERTIES
      IMPORTED_LOCATION ${AMDBLIS_LIB}
      INTERFACE_INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR}
      INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR})
    mark_as_advanced(amdblis::amdblis)

    add_library(amdblis::amdblis_archive STATIC IMPORTED GLOBAL)
    set_target_properties(amdblis::amdblis_archive
      PROPERTIES
      IMPORTED_LOCATION ${AMDBLIS_ARCHIVE_LIB}
      INTERFACE_INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR}
      INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR})
    mark_as_advanced(amdblis::amdblis_archive)
endif()
