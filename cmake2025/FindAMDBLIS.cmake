#  *******************************************************************************
#  * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

if(DEFINED ENV{ZENDNN_BLIS_PATH})
    set(AMDBLIS_ROOT "$ENV{ZENDNN_BLIS_PATH}"
        CACHE STRING "amd blis root path")
else()
    message(FATAL_ERROR "Environment variable ZENDNN_BLIS_PATH not set.")
    return()
endif()


set(AMDBLIS_LIB_ROOT "${AMDBLIS_ROOT}/lib")
find_library(AMDBLIS_LIB
  NAMES libblis-mt.so libblis.so
  PATHS ${AMDBLIS_LIB_ROOT}
  PATH_SUFFIXES LP64 amdzen
  )
message(STATUS "AMDBLIS_LIB, ${AMDBLIS_LIB}")

set(AMDBLIS_INCLUDE_ROOT "${AMDBLIS_ROOT}/include")
find_path(AMDBLIS_INCLUDE_DIR
  NAMES blis.h blis.hh cblas.h cblas.hh
  PATHS ${AMDBLIS_INCLUDE_ROOT}
  PATH_SUFFIXES LP64 amdzen
    NO_DEFAULT_PATH
  )
message(STATUS "AMDBLIS_INCLUDE_DIR, ${AMDBLIS_INCLUDE_DIR}")

if(AMDBLIS_LIB AND AMDBLIS_INCLUDE_DIR)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(AMDBLIS
    DEFAULT_MSG
    AMDBLIS_LIB
    AMDBLIS_INCLUDE_DIR)

  add_library(amdblis::amdblis SHARED IMPORTED GLOBAL)
  set_target_properties(amdblis::amdblis
      PROPERTIES
      IMPORTED_LOCATION ${AMDBLIS_LIB}
      INTERFACE_INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR}
      INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR})
  mark_as_advanced(amdblis::amdblis)
else()
    message(STATUS "FindAMDBLIS.cmake, package not found")
endif()
