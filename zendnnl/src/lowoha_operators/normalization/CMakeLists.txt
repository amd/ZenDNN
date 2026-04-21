# *******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(lowoha_normalization_sources
  lowoha_normalization.cpp
  lowoha_normalization_utils.cpp
  kernel/reference_kernel.cpp
  kernel/rmsnorm_avx512_kernel.cpp
  kernel/layernorm_avx512_kernel.cpp)

set(lowoha_normalization_public_headers
  lowoha_normalization.hpp
  lowoha_normalization_utils.hpp
  lowoha_normalization_common.hpp)

set(lowoha_normalization_private_headers
  kernel/avx512_utils.hpp
  kernel/reference_kernel.hpp
  kernel/rmsnorm_avx512_kernel.hpp
  kernel/layernorm_avx512_kernel.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PRIVATE ${lowoha_normalization_sources}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${lowoha_normalization_public_headers}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${lowoha_normalization_private_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PRIVATE ${lowoha_normalization_sources}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${lowoha_normalization_public_headers}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${lowoha_normalization_private_headers})
endif()

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

list(APPEND lowoha_normalization_public_headers_list
  ${lowoha_normalization_public_headers})
create_header_file(${zendnnl_inc_file}
  ${inc_subdir}
  "${lowoha_normalization_public_headers_list}")

