# *******************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
  # On MSVC, request the LLVM OpenMP runtime (/openmp:llvm, OpenMP 3.1) instead
  # of the default /openmp (vcomp, OpenMP 2.0). Reasons: (1) ZenDNN uses OpenMP
  # 3.0+ APIs (omp_get/set_max_active_levels) and unsigned parallel-for counters
  # that vcomp rejects; (2) AOCL-DLP is built with /openmp:llvm, so ZenDNN must
  # use the same OpenMP runtime to avoid two runtimes in one process. This
  # variable requires CMake >= 3.30. No effect on non-MSVC toolchains.
  if(MSVC)
    if(CMAKE_VERSION VERSION_LESS "3.30")
      message(FATAL_ERROR
        "ZenDNN MSVC builds require CMake >= 3.30: OpenMP_RUNTIME_MSVC (which "
        "selects the LLVM OpenMP runtime /openmp:llvm) is only honored by CMake "
        ">= 3.30. On older CMake it is silently ignored and MSVC falls back to "
        "/openmp (vcomp), which breaks OpenMP 3.x linking and mixes two OpenMP "
        "runtimes with AOCL-DLP. Use CMake >= 3.30.")
    endif()
    set(OpenMP_RUNTIME_MSVC "llvm")
  endif()
  # find openmp
  find_package(OpenMP REQUIRED GLOBAL)
  # pthreads
  find_package(Threads REQUIRED GLOBAL)
endmacro()

# dependencies
macro(find_build_dependencies  _install_prefix)
  # find aocl utils
  if(ZENDNNL_DEPENDS_AOCLUTILS)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AOCL-UTILS presence...")
    set(AOCLUTILS_INSTALL_DIR "${_install_prefix}/deps/aoclutils")
    # Root hints for both the old and new aocl-utils package names.
    set(aocl-utils_ROOT "${AOCLUTILS_INSTALL_DIR}")
    set(AoclUtils_ROOT  "${AOCLUTILS_INSTALL_DIR}")
    # Transitional dual-packaging support for aocl-utils. Older releases
    # (<= 5.3) install package "aocl-utils" with the au:: namespace
    # (au::aoclutils on Unix, au::libaoclutils on Windows). Newer releases
    # install package "AoclUtils" with the AoclUtils:: namespace. Accept
    # either, then normalize to au::aoclutils so the link lines in
    # src/CMakeLists.txt stay unchanged and the Linux (old-packaging) build is
    # unaffected. REMOVE the old-packaging path once AOCLUTILS_GIT_TAG is
    # bumped to a release shipping the AoclUtils:: package on all platforms
    # (verify that release's installed config name/namespace before removing).
    find_package(aocl-utils QUIET GLOBAL CONFIG
      PATH_SUFFIXES "lib" "lib/CMake" "lib64" "lib64/CMake")
    if(NOT aocl-utils_FOUND)
      find_package(AoclUtils QUIET GLOBAL CONFIG
        PATH_SUFFIXES "lib" "lib/cmake" "lib/CMake" "lib64" "lib64/cmake" "lib64/CMake")
    endif()
    if(aocl-utils_FOUND OR AoclUtils_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found AOCL-UTILS at ${AOCLUTILS_INSTALL_DIR}")
      if(TARGET au::aoclutils)
        # old-packaging Unix: au::aoclutils is a real imported target.
        target_include_directories(au::aoclutils
          INTERFACE ${AOCLUTILS_INSTALL_DIR}/include)
      else()
        # new packaging (AoclUtils::) or old-packaging Windows (au::libaoclutils):
        # alias a real, GLOBAL imported target to au::aoclutils. Prefer the
        # static target because ZenDNNL absorbs aocl-utils via WHOLE_ARCHIVE;
        # newer packages use the bare target name for the shared DLL.
        foreach(_au_real AoclUtils::aoclutils_static AoclUtils::libaoclutils AoclUtils::aoclutils au::libaoclutils)
          if(TARGET ${_au_real})
            target_include_directories(${_au_real}
              INTERFACE ${AOCLUTILS_INSTALL_DIR}/include)
            add_library(au::aoclutils ALIAS ${_au_real})
            break()
          endif()
        endforeach()
      endif()
      include_directories(${AOCLUTILS_INSTALL_DIR}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}AOCL-UTILS dependency not found.")
    endif()
  endif()

  # find json
  if(ZENDNNL_DEPENDS_JSON)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking JSON presence...")
    set(JSON_INSTALL_DIR "${_install_prefix}/deps/json")
    set(nlohmann_json_ROOT "${JSON_INSTALL_DIR}")
    set(nlohmann_json_DIR "${nlohmann_json_ROOT}/share/cmake/nlohmann_json")
    find_package(nlohmann_json REQUIRED GLOBAL)
    if(nlohmann_json_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found JSON at ${nlohmann_json_ROOT}")
      include_directories(${nlohmann_json_ROOT}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}JSON dependency not found.")
    endif()
  endif()

  # find aocl dlp
  if(ZENDNNL_DEPENDS_AOCLDLP)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AOCL-DLP presence...")
    set(AOCLDLP_INSTALL_DIR "${_install_prefix}/deps/aocldlp")
    find_package(AOCLDLP REQUIRED GLOBAL)
  endif()

  # find onednn
  if(ZENDNNL_DEPENDS_ONEDNN)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking ONEDNN presence...")
    set(DNNL_INSTALL_DIR "${_install_prefix}/deps/onednn")
    set(dnnl_ROOT "${DNNL_INSTALL_DIR}")
    set(dnnl_DIR "${dnnl_ROOT}/lib/cmake/dnnl")
    if(EXISTS ${dnnl_DIR})
      message(STATUS "${ZENDNNL_MSG_PREFIX}Finding ONEDNN as a package...")
      find_package(dnnl REQUIRED GLOBAL CONFIG)
      if (dnnl_FOUND)
        message(STATUS "${ZENDNNL_MSG_PREFIX}Found ONEDNN at ${dnnl_ROOT}")
        if(TARGET DNNL::dnnl)
          target_include_directories(DNNL::dnnl
            INTERFACE ${dnnl_ROOT}/include)
        endif()
        include_directories(${dnnl_ROOT}/include)
      else()
        message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}ONEDNN not found at ${DNNL_INSTALL_DIR}")
      endif()
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}Finding ONEDNN as a module...")
      find_package(ZLONEDNN REQUIRED GLOBAL)
      include_directories("${_install_prefix}/deps/onednn/include/")
    endif()
  endif()

  # find libxsmm
  if(ZENDNNL_DEPENDS_LIBXSMM)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking LIBXSMM presence...")
    set(LIBXSMM_INSTALL_DIR "${_install_prefix}/deps/libxsmm")
    find_package(LIBXSMM REQUIRED GLOBAL)
  endif()

  # find parlooper
  if(ZENDNNL_DEPENDS_PARLOOPER)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking PARLOOPER presence...")
    set(PARLOOPER_INSTALL_DIR "${_install_prefix}/deps/parlooper")
    find_package(PARLOOPER REQUIRED GLOBAL)
  endif()

  # find fbgemm
  if(ZENDNNL_DEPENDS_FBGEMM)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking FBGEMM presence...")
    set(FBGEMM_INSTALL_DIR "${_install_prefix}/deps/fbgemm")
    find_package(FBGEMM REQUIRED GLOBAL)
  endif()

endmacro()

# dependencies
macro(find_install_dependencies  _install_prefix)
  # find aocl utils
  if(ZENDNNL_DEPENDS_AOCLUTILS)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AOCL-UTILS presence...")
    set(AOCLUTILS_INSTALL_DIR "${_install_prefix}/deps/aoclutils")
    # Root hints for both the old and new aocl-utils package names.
    set(aocl-utils_ROOT "${AOCLUTILS_INSTALL_DIR}")
    set(AoclUtils_ROOT  "${AOCLUTILS_INSTALL_DIR}")
    # Transitional dual-packaging support for aocl-utils. Older releases
    # (<= 5.3) install package "aocl-utils" with the au:: namespace
    # (au::aoclutils on Unix, au::libaoclutils on Windows). Newer releases
    # install package "AoclUtils" with the AoclUtils:: namespace. Accept
    # either, then normalize to au::aoclutils so the link lines in
    # src/CMakeLists.txt stay unchanged and the Linux (old-packaging) build is
    # unaffected. REMOVE the old-packaging path once AOCLUTILS_GIT_TAG is
    # bumped to a release shipping the AoclUtils:: package on all platforms
    # (verify that release's installed config name/namespace before removing).
    find_package(aocl-utils QUIET GLOBAL CONFIG
      PATH_SUFFIXES "lib" "lib/CMake" "lib64" "lib64/CMake")
    if(NOT aocl-utils_FOUND)
      find_package(AoclUtils QUIET GLOBAL CONFIG
        PATH_SUFFIXES "lib" "lib/cmake" "lib/CMake" "lib64" "lib64/cmake" "lib64/CMake")
    endif()
    if(aocl-utils_FOUND OR AoclUtils_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found AOCL-UTILS at ${AOCLUTILS_INSTALL_DIR}")
      if(TARGET au::aoclutils)
        # old-packaging Unix: au::aoclutils is a real imported target.
        target_include_directories(au::aoclutils
          INTERFACE ${AOCLUTILS_INSTALL_DIR}/include)
      else()
        # new packaging (AoclUtils::) or old-packaging Windows (au::libaoclutils):
        # alias a real, GLOBAL imported target to au::aoclutils. Prefer the
        # static target because ZenDNNL absorbs aocl-utils via WHOLE_ARCHIVE;
        # newer packages use the bare target name for the shared DLL.
        foreach(_au_real AoclUtils::aoclutils_static AoclUtils::libaoclutils AoclUtils::aoclutils au::libaoclutils)
          if(TARGET ${_au_real})
            target_include_directories(${_au_real}
              INTERFACE ${AOCLUTILS_INSTALL_DIR}/include)
            add_library(au::aoclutils ALIAS ${_au_real})
            break()
          endif()
        endforeach()
      endif()
      include_directories(${AOCLUTILS_INSTALL_DIR}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}AOCL-UTILS dependency not found.")
    endif()
  endif()

  # find json
  if(ZENDNNL_DEPENDS_JSON)
    message(STATUS "${ZENDNNL_MSG_PREFIX}Checking JSON presence...")
    set(JSON_INSTALL_DIR "${_install_prefix}/deps/json")
    set(nlohmann_json_ROOT "${JSON_INSTALL_DIR}")
    set(nlohmann_json_DIR "${nlohmann_json_ROOT}/share/cmake/nlohmann_json")
    find_package(nlohmann_json REQUIRED GLOBAL)
    if(nlohmann_json_FOUND)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Found JSON at ${nlohmann_json_ROOT}")
      include_directories(${nlohmann_json_ROOT}/include)
    else()
      message(FATAL_ERROR "${ZENDNNL_MSG_PREFIX}JSON dependency not found.")
    endif()
  endif()

  # find aocl dlp
  if(ZENDNNL_DEPENDS_AOCLDLP)
    if(NOT ZENDNNL_AOCLDLP_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking AOCL-DLP presence...")
      set(AOCLDLP_INSTALL_DIR "${_install_prefix}/deps/aocldlp")
      find_package(AOCLDLP REQUIRED GLOBAL)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}AOCL-DLP seems injected, will not check its presence...")
    endif()
  endif()

  # find onednn
  if(ZENDNNL_DEPENDS_ONEDNN)
    if(NOT ZENDNNL_ONEDNN_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking ONEDNN presence...")
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
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking LIBXSMM presence...")
      set(LIBXSMM_INSTALL_DIR "${_install_prefix}/deps/libxsmm")
      find_package(LIBXSMM REQUIRED GLOBAL)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}LIBXSMM seems injected, will not check its presence...")
    endif()
  endif()

  if(ZENDNNL_DEPENDS_PARLOOPER)
    if(NOT ZENDNNL_PARLOOPER_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking PARLOOPER presence...")
      set(PARLOOPER_INSTALL_DIR "${_install_prefix}/deps/parlooper")
      find_package(PARLOOPER REQUIRED GLOBAL)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}PARLOOPER seems injected, will not check its presence...")
    endif()
  endif()

  if(ZENDNNL_DEPENDS_FBGEMM)
    if(NOT ZENDNNL_FBGEMM_INJECTED)
      message(STATUS "${ZENDNNL_MSG_PREFIX}Checking FBGEMM presence...")
      set(FBGEMM_INSTALL_DIR "${_install_prefix}/deps/fbgemm")
      find_package(FBGEMM REQUIRED GLOBAL)
    else()
      message(STATUS "${ZENDNNL_MSG_PREFIX}FBGEMM seems injected, will not check its presence...")
    endif()
  endif()
  
endmacro()

# dependency injection
macro(enable_dependency_injection _dep)
  set(ZENDNNL_EMPTY_STR "")

  set(ZENDNNL_${_dep}_INJECT_DIR ${ZENDNNL_EMPTY_STR}
    CACHE PATH "Pre-built ${_dep} install path for injection")
  set(ZENDNNL_${_dep}_INJECTED OFF
    CACHE BOOL "${_dep} injected" FORCE)

  string(COMPARE NOTEQUAL "${ZENDNNL_${_dep}_INJECT_DIR}" "${ZENDNNL_EMPTY_STR}" INJECT_DIR_FOUND)
  if (${INJECT_DIR_FOUND})
    set(ZENDNNL_${_dep}_INJECTED ON)
  else()
    message(STATUS "${ZENDNNL_MSG_PREFIX}${_dep} install path not given, if needed ${_dep} will be built.")
  endif()

endmacro()
