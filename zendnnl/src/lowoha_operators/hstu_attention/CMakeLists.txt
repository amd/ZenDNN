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

# common/: work decomposition, scratch, output store and the per-tile driver
# shared by the tiled kernels (internal headers, not installed); tiled/: the
# BF16 kernel policy; reference/: the scalar kernel; hstu_gemm_ck/: HSTU's own
# custom-kernel GEMMs (laid out like sdpa/sdpa_gemm_ck/).
set(lowoha_hstu_attention_sources
    lowoha_hstu_attention.cpp
    common/hstu_attention_pad.cpp
    common/hstu_attention_plan.cpp
    common/hstu_attention_scratch.cpp
    common/hstu_attention_store.cpp
    hstu_gemm_ck/bf16/lowoha_hstu_ck_bf16.cpp
    reference/hstu_attention_reference.cpp
    tiled/hstu_attention_tiled.cpp)

set(lowoha_hstu_attention_public_headers
    lowoha_hstu_attention.hpp
    lowoha_hstu_attention_common.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
                 PRIVATE ${lowoha_hstu_attention_sources}
                 PUBLIC  FILE_SET zendnnl_headers
                 TYPE HEADERS FILES ${lowoha_hstu_attention_public_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
                 PRIVATE ${lowoha_hstu_attention_sources}
                 PUBLIC  FILE_SET zendnnl_headers
                 TYPE HEADERS FILES ${lowoha_hstu_attention_public_headers})
endif()

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
           BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

list(APPEND lowoha_hstu_attention_public_headers_list
     ${lowoha_hstu_attention_public_headers})
create_header_file(${zendnnl_inc_file}
                   ${inc_subdir}
                   "${lowoha_hstu_attention_public_headers_list}")
