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

set(DNNL_LIB_ROOT "${DNNL_INSTALL_DIR}/lib")
set(DNNL_LIB_ROOT64 "${DNNL_INSTALL_DIR}/lib64")
find_library(DNNL_LIB
  NAMES dnnl.so
  PATHS ${DNNL_LIB_ROOT} ${DNNL_LIB_ROOT64}
  NO_DEFAULT_PATH)

find_library(DNNL_ARCHIVE_LIB
  NAMES libdnnl.a dnnl.a
  PATHS ${DNNL_LIB_ROOT} ${DNNL_LIB_ROOT64}
  NO_DEFAULT_PATH)

set(DNNL_INCLUDE_ROOT "${DNNL_INSTALL_DIR}/include")
find_path(DNNL_INCLUDE_DIR
  NAMES dnnl.h dnnl.hpp
  PATHS ${DNNL_INCLUDE_ROOT}
  NO_DEFAULT_PATH
  PATH_SUFFIXES oneapi oneapi/dnnl)

include(FindPackageHandleStandardArgs)
if(DNNL_LIB-NOTFOUND)
  find_package_handle_standard_args(ZLONEDNN
    DEFAULT_MSG
    DNNL_ARCHIVE_LIB
    DNNL_INCLUDE_DIR)
else()
  # For zentf build, DNNL_LIB is found but its failed to link.
  find_package_handle_standard_args(ZLONEDNN
    DEFAULT_MSG
    # DNNL_LIB
    DNNL_ARCHIVE_LIB
    DNNL_INCLUDE_DIR)
endif()

if(ZLONEDNN_FOUND)
  if(DNNL_LIB-NOTFOUND)
    message(STATUS "onednn shared library not found.")
  # else()
  #   add_library(DNNL::dnnl SHARED IMPORTED GLOBAL)
  #   set_target_properties(DNNL::dnnl
  #     PROPERTIES
  #     IMPORTED_LOCATION ${DNNL_LIB}
  #     INTERFACE_INCLUDE_DIRECTORIES ${DNNL_INCLUDE_DIR}
  #     INCLUDE_DIRECTORIES ${DNNL_INCLUDE_DIR})
  #   mark_as_advanced(DNNL::dnnl)
  endif()
  add_library(DNNL::dnnl STATIC IMPORTED GLOBAL)
  set_target_properties(DNNL::dnnl
    PROPERTIES
    IMPORTED_LOCATION ${DNNL_ARCHIVE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${DNNL_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${DNNL_INCLUDE_DIR})
  mark_as_advanced(DNNL::dnnl)
endif()

