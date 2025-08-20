#  *******************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

set(AOCLDLP_LIB_ROOT "${AOCLDLP_INSTALL_DIR}/lib")
find_library(AOCLDLP_LIB
  NAMES libaocl-dlp.so
  PATHS ${AOCLDLP_LIB_ROOT}
  NO_DEFAULT_PATH)

find_library(AOCLDLP_ARCHIVE_LIB
  NAMES libaocl-dlp.a
  PATHS ${AOCLDLP_LIB_ROOT}
  NO_DEFAULT_PATH)

set(AOCLDLP_INCLUDE_ROOT "${AOCLDLP_INSTALL_DIR}/include")
find_path(AOCLDLP_INCLUDE_DIR
  NAMES aocl_dlp.h
  PATHS ${AOCLDLP_INCLUDE_ROOT}
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AOCLDLP
  DEFAULT_MSG
  AOCLDLP_LIB
  AOCLDLP_ARCHIVE_LIB
  AOCLDLP_INCLUDE_DIR)

if(AOCLDLP_FOUND)
  add_library(aocldlp::aocl_dlp SHARED IMPORTED GLOBAL)
  set_target_properties(aocldlp::aocl_dlp PROPERTIES
    IMPORTED_LOCATION ${AOCLDLP_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${AOCLDLP_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${AOCLDLP_INCLUDE_DIR})
  mark_as_advanced(aocldlp::aocl_dlp)

  add_library(aocldlp::aocl_dlp_static STATIC IMPORTED GLOBAL)
  set_target_properties(aocldlp::aocl_dlp_static PROPERTIES
    IMPORTED_LOCATION ${AOCLDLP_ARCHIVE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${AOCLDLP_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${AOCLDLP_INCLUDE_DIR})
  mark_as_advanced(aocldlp::aocl_dlp_static)
endif()