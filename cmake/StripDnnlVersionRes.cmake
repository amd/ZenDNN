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
# Windows-only helper: remove compiled resource (*.res) members from a static
# library. oneDNN's static dnnl.lib ships a version resource (version.rc.res).
# ZenDNN links dnnl.lib with WHOLE_ARCHIVE, which force-includes that resource;
# MSVC's manifest-embed re-link then reports it as specified twice and fails
# with LNK1241. Stripping the (purely cosmetic) resource object avoids the
# collision without affecting any code.
#
# The member name is discovered from `lib /LIST` (not hard-coded), so this is
# resilient to oneDNN naming changes, and it is idempotent: if no *.res member
# is present (e.g. already stripped) it is a no-op.
#
# Invoke (run at build time by the oneDNN external project):
#   cmake -DLIB_EXE=<lib.exe> -DTARGET_LIB=<dnnl.lib> -P StripDnnlVersionRes.cmake

if(NOT DEFINED TARGET_LIB)
  message(FATAL_ERROR "StripDnnlVersionRes: TARGET_LIB is required")
endif()

if(NOT EXISTS "${TARGET_LIB}")
  message(STATUS "StripDnnlVersionRes: '${TARGET_LIB}' not found; nothing to do")
  return()
endif()

# Resolve the MSVC archiver: prefer the passed LIB_EXE, else search PATH.
if(NOT LIB_EXE OR NOT EXISTS "${LIB_EXE}")
  find_program(LIB_EXE NAMES lib lib.exe)
endif()
if(NOT LIB_EXE)
  message(WARNING "StripDnnlVersionRes: lib.exe not found; leaving '${TARGET_LIB}' unchanged")
  return()
endif()

# List archive members.
execute_process(
  COMMAND "${LIB_EXE}" /NOLOGO /LIST "${TARGET_LIB}"
  OUTPUT_VARIABLE _members
  RESULT_VARIABLE _list_rc
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _list_rc EQUAL 0)
  message(WARNING "StripDnnlVersionRes: 'lib /LIST' failed (rc=${_list_rc}); skipping")
  return()
endif()

string(REPLACE "\n" ";" _member_lines "${_members}")
set(_removed 0)
foreach(_m IN LISTS _member_lines)
  string(STRIP "${_m}" _m)
  if(_m MATCHES "\\.res$")
    message(STATUS "StripDnnlVersionRes: removing resource member '${_m}' from ${TARGET_LIB}")
    execute_process(
      COMMAND "${LIB_EXE}" /NOLOGO "/REMOVE:${_m}" "${TARGET_LIB}"
      RESULT_VARIABLE _rm_rc)
    if(NOT _rm_rc EQUAL 0)
      message(WARNING "StripDnnlVersionRes: failed to remove '${_m}' (rc=${_rm_rc})")
    else()
      math(EXPR _removed "${_removed}+1")
    endif()
  endif()
endforeach()

message(STATUS "StripDnnlVersionRes: removed ${_removed} resource member(s) from ${TARGET_LIB}")
