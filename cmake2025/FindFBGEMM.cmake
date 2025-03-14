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

message(STATUS "FindFBGEMM.cmake, started!")

if(DEFINED ENV{FBGEMM_INSTALL_PATH})
  set(FBGEMM_ROOT "$ENV{FBGEMM_INSTALL_PATH}"
    CACHE STRING "FBGEMM root path")
else()
  message(STATUS "Environment variable FBGEMM_INSTALL_PATH not set.")
  return()
endif()


set(FBGEMM_LIB_ROOT "${FBGEMM_ROOT}/lib")
# Initialize an empty list to store found libraries
set(FBGEMM_LIBRARIES)

find_library(FBGEMM_LIB_FBGEMM fbgemm PATHS ${FBGEMM_LIB_ROOT})
if(NOT FBGEMM_LIB_FBGEMM)
    message(STATUS "FindFBGEMM.cmake, Library fbgemm not found!")
    return()
else()
    message(STATUS "FindFBGEMM.cmake, Library fbgemm found, ${FBGEMM_LIB_FBGEMM}")
endif()

find_library(FBGEMM_LIB_ASMJIT asmjit PATHS ${FBGEMM_LIB_ROOT})
if(NOT FBGEMM_LIB_ASMJIT)
    message(FATAL_ERROR "Library asmjit not found!")
    return()
endif()

find_library(FBGEMM_LIB_CPUINFO cpuinfo PATHS ${FBGEMM_LIB_ROOT})
if(NOT FBGEMM_LIB_CPUINFO)
    message(STATUS "Library cpuinfo not found!")
    return()
endif()

find_library(FBGEMM_LIB_CLOG clog PATHS ${FBGEMM_LIB_ROOT})
if(NOT FBGEMM_LIB_CLOG)
    message(STATUS "Library clog not found!")
    return()
endif()

set(FBGEMM_LIBRARIES ${FBGEMM_LIB_FBGEMM} ${FBGEMM_LIB_ASMJIT} ${FBGEMM_LIB_CPUINFO} ${FBGEMM_LIB_CLOG})

set(FBGEMM_INCLUDE_ROOT "${FBGEMM_ROOT}/include")
set(FBGEMM_INCLUDE_DIRS "${FBGEMM_INCLUDE_ROOT}")

#check 1 header file in each directory
find_path(FBGEMM_INCLUDE_FILE_FBGEMMEMBEDDING_H
  NAMES FbgemmEmbedding.h
  PATHS ${FBGEMM_INCLUDE_ROOT}/fbgemm
  REQUIRED)

find_path(FBGEMM_INCLUDE_FILE_ASMJIT_H
  NAMES asmjit.h
  PATHS ${FBGEMM_INCLUDE_ROOT}/asmjit
  REQUIRED)

find_path(FBGEMM_INCLUDE_FILE_CPUINFO_H
  NAMES cpuinfo.h
  PATHS ${FBGEMM_INCLUDE_ROOT}
  REQUIRED)

#check fbgemm, asmjit and cpuinfo libraries

add_library(fbgemm::fbgemm STATIC IMPORTED GLOBAL)
set_target_properties(fbgemm::fbgemm
   PROPERTIES
   IMPORTED_LOCATION ${FBGEMM_LIB_FBGEMM}
   INTERFACE_INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIRS}
   INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIRS})
mark_as_advanced(fbgemm::fbgemm)

add_library(fbgemm::asmjit STATIC IMPORTED GLOBAL)
set_target_properties(fbgemm::asmjit
   PROPERTIES
   IMPORTED_LOCATION ${FBGEMM_LIB_ASMJIT}
   INTERFACE_INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIRS}
   INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIRS})
mark_as_advanced(fbgemm::asmjit)

add_library(fbgemm::cpuinfo STATIC IMPORTED GLOBAL)
set_target_properties(fbgemm::cpuinfo
   PROPERTIES
   IMPORTED_LOCATION ${FBGEMM_LIB_CPUINFO}
   INTERFACE_INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIRS}
   INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIRS})
mark_as_advanced(fbgemm::cpuinfo)

add_library(fbgemm::clog STATIC IMPORTED GLOBAL)
set_target_properties(fbgemm::clog
   PROPERTIES
   IMPORTED_LOCATION ${FBGEMM_LIB_CLOG}
   INTERFACE_INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIRS}
   INCLUDE_DIRECTORIES ${FBGEMM_INCLUDE_DIRS})
mark_as_advanced(fbgemm::clog)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FBGEMM
  DEFAULT_MSG
  FBGEMM_LIBRARIES
  FBGEMM_INCLUDE_DIRS)

message(STATUS "FindFBGEMM.cmake, completed!")
