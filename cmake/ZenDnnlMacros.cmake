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
include_guard(GLOBAL)

# required packages
macro(zendnnl_required_packages)
  message(DEBUG "finding required packages...")
  # find openmp
  find_package(OpenMP REQUIRED)
  # pthreads
  find_package(Threads REQUIRED)
endmacro()

# dependencies
macro(find_dependencies  _install_prefix)
  # find amdblis
  if(ZENDNNL_DEPENDS_AMDBLIS)
    set(AMDBLIS_INSTALL_DIR "${_install_prefix}/deps/amdblis")
    find_package(ZLAMDBLIS REQUIRED)
  endif()

  # find aocl dlp
  if(ZENDNNL_DEPENDS_AOCLDLP)
    set(AOCLDLP_INSTALL_DIR "${_install_prefix}/deps/aocldlp")
    find_package(AOCLDLP REQUIRED)
  endif()

  if(ZENDNNL_DEPENDS_ONEDNN)
    set(dnnl_INSTALL_DIR "${_install_prefix}/deps/onednn")
    set(dnnl_ROOT "${dnnl_INSTALL_DIR}")
    set(dnnl_DIR "${dnnl_ROOT}/lib/cmake/dnnl")
    find_package(dnnl REQUIRED)
    if (dnnl_FOUND)
      message(STATUS "Found ONEDNN at ${dnnl_ROOT}")
      include_directories(${dnnl_ROOT}/include)
    else()
      message(FATAL_ERROR "ONEDNN not found at ${dnnl_INSTALL_DIR}")
    endif()
  endif()

  # find aocl utils
  if(ZENDNNL_DEPENDS_AOCLUTILS)
    set(AOCLUTILS_INSTALL_DIR "${_install_prefix}/deps/aoclutils")
    set(aocl-utils_ROOT "${AOCLUTILS_INSTALL_DIR}")
    set(aocl-utils_DIR "${aocl-utils_ROOT}/lib/CMake")
    find_package(aocl-utils REQUIRED)
    if(aocl-utils_FOUND)
      message(STATUS "Found AOCL UTILS at ${aocl-utils_ROOT}")
      include_directories(${aocl-utils_ROOT}/include)
    endif()
  endif()

  # find json
  if(ZENDNNL_DEPENDS_JSON)
    set(JSON_INSTALL_DIR "${_install_prefix}/deps/json")
    set(nlohmann_json_ROOT "${JSON_INSTALL_DIR}")
    set(nlohmann_json_DIR "${json_ROOT}/share/cmake/nlohmann_json")
    find_package(nlohmann_json REQUIRED)
    if(nlohmann_json_FOUND)
      message(STATUS "Found JSON at ${nlohmann_json_ROOT}")
      include_directories(${nlohmann_json_ROOT}/include)
    endif()
  endif()

endmacro()

