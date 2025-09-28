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
#include_guard(GLOBAL)

function(prolog_header_file inc_file)
  set(prolog_banner
"/**************************************************************************
* Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
* This is a system generated file.
****************************************************************************/
#ifndef _ZENDNNL_HPP_
#define _ZENDNNL_HPP_\n\n")
  file(APPEND ${inc_file} ${prolog_banner})
endfunction()

function(create_header_file inc_file inc_subdir inc_list)
  #separate_arguments(inc_list NATIVE_COMMAND ${inc_list_str})
  #set(inc_pre_str "#include \"impl/")
  set(inc_pre_str "#include \"")
  string(CONCAT inc_pre_str ${inc_pre_str} ${inc_subdir} "/")
  set(inc_post_str "\"\n")
  foreach(elem ${inc_list})
    string(CONCAT full_str ${inc_pre_str} ${elem} ${inc_post_str})
    file(APPEND ${inc_file} ${full_str})
  endforeach()
endfunction()

function(epilog_header_file inc_file)
  set(epilog_banner "\n#endif\n")
  file(APPEND ${inc_file} ${epilog_banner})
endfunction()

function(deps_flag_header_file inc_file flag value)
  set(deps_flag_str "#define ${flag}  ${value}\n\n")
  file(APPEND ${inc_file} ${deps_flag_str})
endfunction()

function(create_version_header_file inc_file vmajor vminor vpatch)
  set(prolog_banner
"/**************************************************************************
* Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
* This is a system generated file.
****************************************************************************/
#ifndef _ZENDNNL_VERSION_HPP_
#define _ZENDNNL_VERSION_HPP_\n\n")
  file(APPEND ${inc_file} ${prolog_banner})

  set(version_str "#define ZENDNNL_VERSION  ${vmajor}.${vminor}.${vpatch}\n")
  file(APPEND ${inc_file} ${version_str})

  set(version_major_str "#define ZENDNNL_VERSION_MAJOR  ${vmajor}\n")
  file(APPEND ${inc_file} ${version_major_str})

  set(version_minor_str "#define ZENDNNL_VERSION_MINOR  ${vminor}\n")
  file(APPEND ${inc_file} ${version_minor_str})

  set(version_patch_str "#define ZENDNNL_VERSION_PATCH  ${vpatch}\n")
  file(APPEND ${inc_file} ${version_patch_str})

  set(epilog_banner "\n#endif\n")
  file(APPEND ${inc_file} ${epilog_banner})
endfunction()

#prolog_header_file(${INC_FILE_NAME})
#create_header_file(${INC_FILE_NAME} ${INC_SUBDIR} ${INC_FILE_LIST})
