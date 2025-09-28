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
  find_package(OpenMP REQUIRED GLOBAL)
  # pthreads
  find_package(Threads REQUIRED GLOBAL)
endmacro()

# dependencies
macro(find_build_dependencies  _install_prefix)
  # find aocl utils
  if(ZENDNNL_DEPENDS_AOCLUTILS)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AOCL-UTILS presencce...")
    set(AOCLUTILS_INSTALL_DIR "${_install_prefix}/deps/aoclutils")
    set(aocl-utils_ROOT "${AOCLUTILS_INSTALL_DIR}")
    #set(aocl-utils_DIR "${aocl-utils_ROOT}/lib/CMake")
    find_package(aocl-utils REQUIRED GLOBAL CONFIG
      PATH_SUFFIXES "lib" "lib/CMake" "lib64" "lib64/CMake")
    if(aocl-utils_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found AOCL-UTILS at ${aocl-utils_ROOT}")
      if(TARGET au::aoclutils)
        target_include_directories(au::aoclutils
          INTERFACE ${aocl-utils_ROOT}/include)
      endif()
      include_directories(${aocl-utils_ROOT}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}AOCL-UTILS dependency not found.")
    endif()
  endif()

  # find json
  if(ZENDNNL_DEPENDS_JSON)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking JSON presencce...")
    set(JSON_INSTALL_DIR "${_install_prefix}/deps/json")
    set(nlohmann_json_ROOT "${JSON_INSTALL_DIR}")
    set(nlohmann_json_DIR "${json_ROOT}/share/cmake/nlohmann_json")
    find_package(nlohmann_json REQUIRED GLOBAL)
    if(nlohmann_json_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found JSON at ${nlohmann_json_ROOT}")
      include_directories(${nlohmann_json_ROOT}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}AOCL-UTILS dependency not found.")
    endif()
  endif()

  # find amdblis
  if(ZENDNNL_DEPENDS_AMDBLIS)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AMD-BLIS presencce...")
    set(AMDBLIS_INSTALL_DIR "${_install_prefix}/deps/amdblis")
    find_package(ZLAMDBLIS REQUIRED GLOBAL)
  endif()

  # find aocl dlp
  if(ZENDNNL_DEPENDS_AOCLDLP)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AOCL-DLP presencce...")
    set(AOCLDLP_INSTALL_DIR "${_install_prefix}/deps/aocldlp")
    find_package(AOCLDLP REQUIRED GLOBAL)
  endif()

  # find onednn
  if(ZENDNNL_DEPENDS_ONEDNN)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking ONEDNN presencce...")
    set(dnnl_INSTALL_DIR "${_install_prefix}/deps/onednn")
    set(dnnl_ROOT "${dnnl_INSTALL_DIR}")
    set(dnnl_DIR "${dnnl_ROOT}/lib/cmake/dnnl")
    find_package(dnnl REQUIRED GLOBAL)
    if (dnnl_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found ONEDNN at ${dnnl_ROOT}")
      if(TARGET DNNL::dnnl)
        target_include_directories(DNNL::dnnl
          INTERFACE ${dnnl_ROOT}/include)
      endif()
      include_directories(${dnnl_ROOT}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}ONEDNN not found at ${dnnl_INSTALL_DIR}")
    endif()
  endif()

  # find libxsmm
  if(ZENDNNL_DEPENDS_LIBXSMM)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking LIBXSMM presencce...")
    set(LIBXSMM_INSTALL_DIR "${_install_prefix}/deps/libxsmm")
    find_package(LIBXSMM REQUIRED GLOBAL)
  endif()

  # find parlooper
  if(ZENDNNL_DEPENDS_PARLOOPER)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking PARLOOPER presencce...")
    set(PARLOOPER_INSTALL_DIR "${_install_prefix}/deps/parlooper")
    find_package(PARLOOPER REQUIRED GLOBAL)
  endif()

endmacro()

# dependencies
macro(find_install_dependencies  _install_prefix)
  # find aocl utils
  if(ZENDNNL_DEPENDS_AOCLUTILS)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AOCL-UTILS presencce...")
    set(AOCLUTILS_INSTALL_DIR "${_install_prefix}/deps/aoclutils")
    set(aocl-utils_ROOT "${AOCLUTILS_INSTALL_DIR}")
    #set(aocl-utils_DIR "${aocl-utils_ROOT}/lib/CMake")
    find_package(aocl-utils REQUIRED GLOBAL CONFIG
      PATH_SUFFIXES "lib" "lib/CMake" "lib64" "lib64/CMake")
    if(aocl-utils_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found AOCL-UTILS at ${aocl-utils_ROOT}")
      if(TARGET au::aoclutils)
        target_include_directories(au::aoclutils
          INTERFACE ${aocl-utils_ROOT}/include)
      endif()
      include_directories(${aocl-utils_ROOT}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}AOCL-UTILS dependency not found.")
    endif()
  endif()

  # find json
  if(ZENDNNL_DEPENDS_JSON)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking JSON presencce...")
    set(JSON_INSTALL_DIR "${_install_prefix}/deps/json")
    set(nlohmann_json_ROOT "${JSON_INSTALL_DIR}")
    set(nlohmann_json_DIR "${json_ROOT}/share/cmake/nlohmann_json")
    find_package(nlohmann_json REQUIRED GLOBAL)
    if(nlohmann_json_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found JSON at ${nlohmann_json_ROOT}")
      include_directories(${nlohmann_json_ROOT}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}AOCL-UTILS dependency not found.")
    endif()
  endif()

  # find amdblis
  if(ZENDNNL_DEPENDS_AMDBLIS)
    if(NOT ZENDNNL_AMDBLIS_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AMD-BLIS presencce...")
      set(AMDBLIS_INSTALL_DIR "${_install_prefix}/deps/amdblis")
      find_package(ZLAMDBLIS REQUIRED GLOBAL)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}AMD-BLIS seems injected, will not check its presence...")
    endif()
  endif()

  # find aocl dlp
  if(ZENDNNL_DEPENDS_AOCLDLP)
    if(NOT ZENDNNL_AOCLDLP_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AOCL-DLP presencce...")
      set(AOCLDLP_INSTALL_DIR "${_install_prefix}/deps/aocldlp")
      find_package(AOCLDLP REQUIRED GLOBAL)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}AOCL-DLP seems injected, will not check its presence...")
    endif()
  endif()

  # find onednn
  if(ZENDNNL_DEPENDS_ONEDNN)
    if(NOT ZENDNNL_ONEDNN_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking ONEDNN presencce...")
      set(dnnl_INSTALL_DIR "${_install_prefix}/deps/onednn")
      set(dnnl_ROOT "${dnnl_INSTALL_DIR}")
      set(dnnl_DIR "${dnnl_ROOT}/lib/cmake/dnnl")
      find_package(dnnl REQUIRED GLOBAL)
      if (dnnl_FOUND)
        message(STATUS "${ZENDNNL_MSG_PREFIX}Found ONEDNN at ${dnnl_ROOT}")
        if(TARGET DNNL::dnnl)
          target_include_directories(DNNL::dnnl
            INTERFACE ${dnnl_ROOT}/include)
        endif()
        include_directories(${dnnl_ROOT}/include)
      else()
        message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}ONEDNN not found at ${dnnl_INSTALL_DIR}")
      endif()
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}ONEDNN seems injected, will not check its presence...")
    endif()
  endif()

  if(ZENDNNL_DEPENDS_LIBXSMM)
    if(NOT ZENDNNL_LIBXSMM_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking LIBXSMM presencce...")
      set(LIBXSMM_INSTALL_DIR "${_install_prefix}/deps/libxsmm")
      find_package(LIBXSMM REQUIRED GLOBAL)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}LIBXSMM seems injected, will not check its presence...")
    endif()
  endif()

  if(ZENDNNL_DEPENDS_PARLOOPER)
    if(NOT ZENDNNL_PARLOOPER_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking PARLOOPER presencce...")
      set(PARLOOPER_INSTALL_DIR "${_install_prefix}/deps/parlooper")
      find_package(PARLOOPER REQUIRED GLOBAL)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}PARLOOPER seems injected, will not check its presence...")
    endif()
  endif()

endmacro()

# dependency injection
macro(enable_dependency_injection _dep _fwk_build_option)
  set(ZENDNNL_EMPTY_STR "")

  set(ZENDNNL_${_dep}_FWK_DIR ${ZENDNNL_EMPTY_STR}
    CACHE PATH "Fwk ${_dep} install path")
  set(ZENDNNL_${_dep}_INJECTED OFF
    CACHE BOOL "${_dep} injected by fwk" FORCE)

  if (${_fwk_build_option})
    string(COMPARE NOTEQUAL "${ZENDNNL_${_dep}_FWK_DIR}" "${ZENDNNL_EMPTY_STR}" FWK_DIR_FOUND)
    if (${FWK_DIR_FOUND})
      set(ZENDNNL_${_dep}_INJECTED ON)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}Framework ${_dep} install path not given, if needed ${_dep} will be built.")
    endif()
  endif()
endmacro()
