# *******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

# Library and include roots
set(LIBXSMM_LIB_ROOT "${LIBXSMM_INSTALL_DIR}/lib")
set(LIBXSMM_INCLUDE_ROOT "${LIBXSMM_INSTALL_DIR}/include")

# Shared library
find_library(LIBXSMM_LIB
  NAMES libxsmm.so
  PATHS ${LIBXSMM_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES lib
)

# Static library
find_library(LIBXSMM_ARCHIVE_LIB
  NAMES libxsmm.a
  PATHS ${LIBXSMM_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES lib
)

# Includes
find_path(LIBXSMM_INCLUDE_DIR
  NAMES libxsmm.h
  PATHS ${LIBXSMM_INCLUDE_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES include
)

# Validate findings
include(FindPackageHandleStandardArgs)
if(LIBXSMM_LIB-NOTFOUND)
  find_package_handle_standard_args(LIBXSMM
    DEFAULT_MSG
    LIBXSMM_ARCHIVE_LIB
    LIBXSMM_INCLUDE_DIR
  )
else()
  find_package_handle_standard_args(LIBXSMM
    DEFAULT_MSG
    LIBXSMM_LIB
    LIBXSMM_ARCHIVE_LIB
    LIBXSMM_INCLUDE_DIR
  )
endif()

# Define imported targets
if(LIBXSMM_FOUND)
  if(LIBXSMM_LIB-NOTFOUND)
    message(STATUS "libxsmm shared library not found.")
  else()
    add_library(libxsmm::libxsmm SHARED IMPORTED GLOBAL)
    set_target_properties(libxsmm::libxsmm PROPERTIES
      IMPORTED_LOCATION             ${LIBXSMM_LIB}
      INTERFACE_INCLUDE_DIRECTORIES ${LIBXSMM_INCLUDE_DIR}
      INCLUDE_DIRECTORIES           ${LIBXSMM_INCLUDE_DIR}
    )
    mark_as_advanced(libxsmm::libxsmm)
  endif()
    add_library(libxsmm::libxsmm_archive STATIC IMPORTED GLOBAL)
    set_target_properties(libxsmm::libxsmm_archive PROPERTIES
      IMPORTED_LOCATION             ${LIBXSMM_ARCHIVE_LIB}
      INTERFACE_INCLUDE_DIRECTORIES ${LIBXSMM_INCLUDE_DIR}
      INCLUDE_DIRECTORIES           ${LIBXSMM_INCLUDE_DIR}
    )
    mark_as_advanced(libxsmm::libxsmm_archive)
endif()
