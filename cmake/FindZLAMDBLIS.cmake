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

#message(DEBUG "Searching for AMDBLIS in ${AMDBLIS_INSTALL_DIR}")

set(AMDBLIS_LIB_ROOT "${AMDBLIS_INSTALL_DIR}/lib")
find_library(AMDBLIS_LIB
  NAMES libblis-mt.so libblis.so
  PATHS ${AMDBLIS_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES LP64 amdzen)

find_library(AMDBLIS_ARCHIVE_LIB
  NAMES libblis-mt.a libblis.a
  PATHS ${AMDBLIS_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES LP64 amdzen)

set(AMDBLIS_INCLUDE_ROOT "${AMDBLIS_INSTALL_DIR}/include")
find_path(AMDBLIS_INCLUDE_DIR
  NAMES blis.h blis.hh cblas.h cblas.hh
  PATHS ${AMDBLIS_INCLUDE_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES LP64 amdzen blis)

include(FindPackageHandleStandardArgs)
if(AMDBLIS_LIB-NOTFOUND)
  find_package_handle_standard_args(ZLAMDBLIS
    DEFAULT_MSG
    AMDBLIS_ARCHIVE_LIB
    AMDBLIS_INCLUDE_DIR)
else()
  find_package_handle_standard_args(ZLAMDBLIS
    DEFAULT_MSG
    AMDBLIS_LIB
    AMDBLIS_ARCHIVE_LIB
    AMDBLIS_INCLUDE_DIR)
endif()

if(ZLAMDBLIS_FOUND)
  if(AMDBLIS_LIB-NOTFOUND)
    message(STATUS "amdblis shared library not found.")
  else()
    add_library(amdblis::amdblis SHARED IMPORTED GLOBAL)
    set_target_properties(amdblis::amdblis
      PROPERTIES
      IMPORTED_LOCATION ${AMDBLIS_LIB}
      INTERFACE_INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR}
      INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR})
    mark_as_advanced(amdblis::amdblis)
  endif()
    add_library(amdblis::amdblis_archive STATIC IMPORTED GLOBAL)
    set_target_properties(amdblis::amdblis_archive
      PROPERTIES
      IMPORTED_LOCATION ${AMDBLIS_ARCHIVE_LIB}
      INTERFACE_INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR}
      INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR})
    mark_as_advanced(amdblis::amdblis_archive)
endif()


