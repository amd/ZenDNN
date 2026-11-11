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
#
# Windows-only helper: remove the __declspec(dllexport) markers from the AOCL-DLP
# and AOCL-Utils sources. Both mark their APIs for export even when built as
# static libraries, and ZenDNN links those libraries with WHOLE_ARCHIVE, so any
# DLL that links ZenDNN (e.g. oneDNN's dnnl.dll) re-exports them. On Linux the
# consumer hides them with --exclude-libs; link.exe has no equivalent.
#
# ZenDNN uses only the static AOCL-DLP library, so AOCL-DLP's DLL, which would
# have nothing left to export, is excluded from the build and not installed.
# AOCL-DLP's CMake package config is edited to match, so it offers only the
# static library.
#
# Each edit expects its text exactly once and is a no-op when already applied.
#
# Invoke (run as the PATCH_COMMAND of the AOCL external projects):
#   cmake -DDEPENDENCY=<aocldlp|aoclutils> -DSOURCE_DIR=<dir> -P StripAoclDllExport.cmake

if(NOT DEFINED DEPENDENCY OR NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "StripAoclDllExport: DEPENDENCY and SOURCE_DIR are required")
endif()

function(_strip_aocl_replace file old new)
  set(_path "${SOURCE_DIR}/${file}")
  file(READ "${_path}" _text)
  # The patterns use LF; on Windows file(WRITE) writes CRLF back.
  string(REPLACE "\r" "" _text "${_text}")
  string(FIND "${_text}" "${old}" _pos)
  if(_pos EQUAL -1)
    string(FIND "${_text}" "${new}" _pos)
    if(_pos EQUAL -1)
      message(FATAL_ERROR "StripAoclDllExport: '${old}' not found in ${_path}")
    endif()
    return()
  endif()
  string(REPLACE "${old}" "${new}" _text "${_text}")
  file(WRITE "${_path}" "${_text}")
  message(STATUS "StripAoclDllExport: patched ${file}")
endfunction()

set(_not_exported "/* static-only build: not exported */")

if(DEPENDENCY STREQUAL "aocldlp")
  _strip_aocl_replace("include/classic/dlp_compat.h"
    "#define DLP_CLASSIC_EXPORT __declspec(dllexport)"
    "#define DLP_CLASSIC_EXPORT ${_not_exported}")
  _strip_aocl_replace("CMakeLists.txt"
    "dlp_install(\${PROJECT_NAME})"
    "set_target_properties(\${PROJECT_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)")
  _strip_aocl_replace("CMakeLists.txt"
    "set(AOCL_DLP_REQUIRED_COMPONENTS \"aocl-dlp;aocl-dlp_static\")"
    "set(AOCL_DLP_REQUIRED_COMPONENTS \"aocl-dlp_static\")")
  _strip_aocl_replace("CMakeLists.txt"
    "set(AOCL_DLP_AVAILABLE_COMPONENTS \"aocl-dlp;aocl-dlp_static\")"
    "set(AOCL_DLP_AVAILABLE_COMPONENTS \"aocl-dlp_static\")")
  set(_config "cmake/AoclDlpConfig.cmake.in")
  _strip_aocl_replace("${_config}"
    "if(NOT TARGET AoclDlp::aocl-dlp)"
    "if(NOT TARGET AoclDlp::aocl-dlp_static)")
  string(CONCAT _old_include_dirs
    "if(TARGET AoclDlp::aocl-dlp)\n"
    "    get_target_property(AoclDlp_INCLUDE_DIRS\n"
    "                        AoclDlp::aocl-dlp\n")
  string(CONCAT _new_include_dirs
    "if(TARGET AoclDlp::aocl-dlp_static)\n"
    "    get_target_property(AoclDlp_INCLUDE_DIRS\n"
    "                        AoclDlp::aocl-dlp_static\n")
  _strip_aocl_replace("${_config}"
    "${_old_include_dirs}"
    "${_new_include_dirs}")
  _strip_aocl_replace("${_config}"
    "set(AoclDlp_LIBRARIES AoclDlp::aocl-dlp)"
    "set(AoclDlp_LIBRARIES AoclDlp::aocl-dlp_static)")
  _strip_aocl_replace("${_config}"
    "set(AoclDlp_SHARED_LIBRARY AoclDlp::aocl-dlp)"
    "set(AoclDlp_SHARED_LIBRARY \"\")")
  _strip_aocl_replace("${_config}"
    "message(STATUS \"  Shared library: AoclDlp::aocl-dlp\")"
    "message(STATUS \"  Shared library: not built (static-only)\")")
elseif(DEPENDENCY STREQUAL "aoclutils")
  foreach(_header "SDK/Include/Au/Defs.hh"
                  "SDK/Include/Capi/au/cpuid/au_cpuid_header_only.h")
    _strip_aocl_replace("${_header}"
      "#define AUD_API_EXPORT __declspec(dllexport)"
      "#define AUD_API_EXPORT ${_not_exported}")
  endforeach()
else()
  message(FATAL_ERROR "StripAoclDllExport: unknown DEPENDENCY '${DEPENDENCY}'")
endif()
