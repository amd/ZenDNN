#  *******************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Library and include roots
set(FBGEMM_LIB_ROOT "${FBGEMM_INSTALL_DIR}/lib")
set(FBGEMM_INCLUDE_ROOT "${FBGEMM_INSTALL_DIR}/include")

# Shared library
find_library(FBGEMM_LIB
  NAMES libfbgemm.so
  PATHS ${FBGEMM_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES lib
)

# Static library
find_library(FBGEMM_ARCHIVE_LIB
  NAMES libfbgemm.a
  PATHS ${FBGEMM_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES lib
)

# Includes
find_path(FBGEMM_INCLUDE_DIR
  NAMES fbgemm/Fbgemm.h
  PATHS ${FBGEMM_INCLUDE_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES include
)

# Find FBGEMM dependencies: cpuinfo and asmjit
find_library(CPUINFO_LIB
  NAMES libcpuinfo.a
  PATHS ${FBGEMM_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES lib
)

find_library(ASMJIT_LIB
  NAMES libasmjit.a
  PATHS ${FBGEMM_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES lib
)

# Validate findings
include(FindPackageHandleStandardArgs)
if(NOT FBGEMM_LIB)
  find_package_handle_standard_args(FBGEMM
    DEFAULT_MSG
    FBGEMM_ARCHIVE_LIB
    FBGEMM_INCLUDE_DIR
    CPUINFO_LIB
    ASMJIT_LIB
  )
else()
  find_package_handle_standard_args(FBGEMM
    DEFAULT_MSG
    FBGEMM_LIB
    FBGEMM_ARCHIVE_LIB
    FBGEMM_INCLUDE_DIR
    CPUINFO_LIB
    ASMJIT_LIB
  )
endif()

# Define imported targets
if(FBGEMM_FOUND)
  # Create imported target for cpuinfo
  add_library(fbgemm::cpuinfo STATIC IMPORTED GLOBAL)
  set_target_properties(fbgemm::cpuinfo PROPERTIES
    IMPORTED_LOCATION ${CPUINFO_LIB}
  )

  # Create imported target for asmjit (required)
  add_library(fbgemm::asmjit STATIC IMPORTED GLOBAL)
  set_target_properties(fbgemm::asmjit PROPERTIES
    IMPORTED_LOCATION ${ASMJIT_LIB}
  )

  # Build list of dependencies
  set(FBGEMM_DEPENDENCIES fbgemm::cpuinfo fbgemm::asmjit)

  if(NOT FBGEMM_LIB)
    message(STATUS "fbgemm shared library not found.")
  else()
    add_library(fbgemm::fbgemm SHARED IMPORTED GLOBAL)
    set_target_properties(fbgemm::fbgemm PROPERTIES
      IMPORTED_LOCATION             ${FBGEMM_LIB}
      INTERFACE_INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIR}
      INCLUDE_DIRECTORIES           ${FBGEMM_INCLUDE_DIR}
      INTERFACE_LINK_LIBRARIES      "${FBGEMM_DEPENDENCIES}"
    )
    mark_as_advanced(fbgemm::fbgemm)
  endif()

  add_library(fbgemm::fbgemm_archive STATIC IMPORTED GLOBAL)
  set_target_properties(fbgemm::fbgemm_archive PROPERTIES
    IMPORTED_LOCATION             ${FBGEMM_ARCHIVE_LIB}  
    INTERFACE_INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIR}
    INCLUDE_DIRECTORIES           ${FBGEMM_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES      "${FBGEMM_DEPENDENCIES}"
  )
  mark_as_advanced(fbgemm::fbgemm_archive)
endif()