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

# lowoha_hstu_attention*: the public API; every internal file is
# lowoha_hstu_<component> (internal headers are not installed).
# common/: internal helpers of the dispatcher and the kernels (mask, kernel
# selection, padded mode, utils); reference/: the scalar kernel; tiled/: the
# tiled kernel's data-type independent driver, planning, scratch, store and
# shared policy blocks, with one policy per data type in tiled/<dtype>/
# (bf16/, int8/); hstu_gemm_ck/: HSTU's own custom-kernel GEMMs, one folder
# per data type (laid out like sdpa/sdpa_gemm_ck/); hstu_quant/: HSTU's own
# dynamic quantizers.
set(lowoha_hstu_attention_sources
    lowoha_hstu_attention.cpp
    common/lowoha_hstu_pad.cpp
    hstu_gemm_ck/bf16/lowoha_hstu_ck_bf16.cpp
    hstu_gemm_ck/int8/lowoha_hstu_ck_int8.cpp
    hstu_quant/lowoha_hstu_quant.cpp
    reference/lowoha_hstu_ref_kernel.cpp
    tiled/lowoha_hstu_tiled_plan.cpp
    tiled/lowoha_hstu_tiled_scratch.cpp
    tiled/lowoha_hstu_tiled_store.cpp
    tiled/bf16/lowoha_hstu_tiled_bf16.cpp
    tiled/int8/lowoha_hstu_tiled_int8.cpp)

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
