# *******************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#
# common packages and dependencies
#
function (FindCommonPackages)
  message(DEBUG "inside SubModuleUpdate function...")

  # find git
  find_package(Git)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "git not found.")
  else()
    message(DEBUG "git found.")
  endif()

  # find openmp
  find_package(OpenMP)
  if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP is not found")
  else()
    message(DEBUG "OpenMP found")
  endif()

  # fins pthreads
  find_package(Threads)
  if(NOT Threads_FOUND)
    message(FATAL_ERROR "Threads is not found")
  else()
    message(DEBUG "Threads found")
  endif()

endfunction()

#
# submodule update
#
function(SubmoduleUpdate)
  message(DEBUG "inside SubModuleUpdate function...")

  # Update submodules as needed
  if(GIT_SUBMODULE_UPDATE)
    message(STATUS "git submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}.")
    endif()
  endif()
endfunction()

#
# find dependencies
#
function(FindDependencies)
  message(DEBUG "inside FindDependencies function...")

  # find amd blis
  find_package(AMDBLIS)
  if(NOT AMDBLIS_FOUND)
    message(FATAL_ERROR "amd blis not detected")
  endif()

  # find oneDNN
  get_property(ONEDNN_ROOT GLOBAL PROPERTY ONEDNNROOT)
  if (NOT DEFINED ONEDNN_ROOT)
    message(FATAL_ERROR "property ONEDNNROOT not set. don't know where to find OneDNN.")
  else()
    message(DEBUG "searching onednn in ${ONEDNN_ROOT}")
  endif()

  set(dnnl_ROOT "${ONEDNN_ROOT}/build/install"
    CACHE STRING "oneDNN root path")

  find_package(dnnl)
  if(NOT dnnl_FOUND)
    message(FATAL_ERROR "dnnl not detected")
  else()
    message(STATUS "dnnl detected at ${dnnl_ROOT}")
  endif()
endfunction()
