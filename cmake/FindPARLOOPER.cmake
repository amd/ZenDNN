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
set(PARLOOPER_LIB_ROOT "${PARLOOPER_INSTALL_DIR}/lib")
set(PARLOOPER_INCLUDE_ROOT "${PARLOOPER_INSTALL_DIR}/include")

# Static library
find_library(PARLOOPER_ARCHIVE_LIB
  NAMES libparlooper.a
  PATHS ${PARLOOPER_LIB_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES lib
)

# Includes
find_path(PARLOOPER_INCLUDE_DIR
  NAMES threaded_loops.h
  PATHS ${PARLOOPER_INCLUDE_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES include
)

# Validate findings
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PARLOOPER
  DEFAULT_MSG
  PARLOOPER_ARCHIVE_LIB
  PARLOOPER_INCLUDE_DIR
)

# Define imported target
if (PARLOOPER_FOUND)
  add_library(parlooper::parlooper_archive STATIC IMPORTED GLOBAL)
  set_target_properties(parlooper::parlooper_archive PROPERTIES
    IMPORTED_LOCATION             ${PARLOOPER_ARCHIVE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${PARLOOPER_INCLUDE_DIR}
  )
  mark_as_advanced(parlooper::parlooper_archive)
endif()
