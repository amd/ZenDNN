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

# declare a zendnnl dependency
macro(zendnnl_add_dependency )
  set(options INCLUDE_ONLY)
  set(oneValueArgs NAME PATH LIB_SUFFIX INCLUDE_SUFFIX ARCHIVE_FILE ALIAS)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(_zad "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  string(TOUPPER ${_zad_NAME} _ZAD_UNAME)

  if(DEFINED _zad_INCLUDE_SUFFIX)
    set(ZENDNNL_${_ZAD_UNAME}_INC_DIR "${_zad_PATH}/${_zad_INCLUDE_SUFFIX}")
  else()
    set(ZENDNNL_${_ZAD_UNAME}_INC_DIR "${_zad_PATH}/include")
  endif()

  if(DEFINED _zad_LIB_SUFFIX)
    set(ZENDNNL_${_ZAD_UNAME}_LIB_DIR "${_zad_PATH}/${_zad_LIB_SUFFIX}")
  else()
    set(ZENDNNL_${_ZAD_UNAME}_LIB_DIR "${_zad_PATH}/lib")
  endif()

  if(NOT EXISTS ${ZENDNNL_${_ZAD_UNAME}_INC_DIR})
    file(MAKE_DIRECTORY ${ZENDNNL_${_ZAD_UNAME}_INC_DIR})
  endif()

  if(${_zad_INCLUDE_ONLY})
    add_library(zendnnl_${_zad_NAME}_deps INTERFACE IMPORTED GLOBAL)
    add_dependencies(zendnnl_${_zad_NAME}_deps ${_zad_DEPENDS})

    set_target_properties(zendnnl_${_zad_NAME}_deps
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_${_ZAD_UNAME}_INC_DIR}")
  else()

    add_library(zendnnl_${_zad_NAME}_deps STATIC IMPORTED GLOBAL)
    add_dependencies(zendnnl_${_zad_NAME}_deps ${_zad_DEPENDS})

    set_target_properties(zendnnl_${_zad_NAME}_deps
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_${_ZAD_UNAME}_LIB_DIR}/${_zad_ARCHIVE_FILE}"
      INCLUDE_DIRECTORIES "${ZENDNNL_${_ZAD_UNAME}_INC_DIR}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_${_ZAD_UNAME}_INC_DIR}")
  endif()

  add_library(${_zad_ALIAS} ALIAS zendnnl_${_zad_NAME}_deps)
endmacro()

macro(zendnnl_add_option )
  set(options EXECLUDE_FROM_COMMAND_LIST FORCE)
  set(oneValueArgs NAME VALUE TYPE CACHE_STRING COMMAND_LIST)
  set(multiValueArgs "")
  cmake_parse_arguments(_zao "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(${_zao_FORCE})
    set(${_zao_NAME} ${_zao_VALUE} CACHE ${_zao_TYPE} ${_zao_CACHE_STRING} FORCE)
  else()
    set(${_zao_NAME} ${_zao_VALUE} CACHE ${_zao_TYPE} ${_zao_CACHE_STRING})
  endif()

  if (NOT ${_zao_EXECLUDE_FROM_COMMAND_LIST})
    list(APPEND ${_zao_COMMAND_LIST} "-D${_zao_NAME}:${_zao_TYPE}=${_zao_VALUE}")
  endif()
endmacro()
