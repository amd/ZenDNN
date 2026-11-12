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

# ---------------------------------------------------------------------------
#  The AVX-512 kernels carry their own target pragma (kernel/
#  gdn_avx512_utils.hpp); lowoha_gdn.cpp checks the CPU before dispatching to
#  them, so no per-file ISA flags are needed.
# ---------------------------------------------------------------------------

# kernel/: the native AVX-512 kernels; reference/: the scalar kernel;
# gdn_gemm_ck/: GDN's own custom-kernel GEMMs for the chunked prefill, one
# translation unit per operand type.
set(lowoha_gdn_sources
    lowoha_gdn.cpp
    reference/lowoha_gdn_ref_kernel.cpp
    kernel/gdn_decode_avx512.cpp
    kernel/gdn_prefill_avx512.cpp
    gdn_gemm_ck/bf16/lowoha_gdn_ck_bf16.cpp
    gdn_gemm_ck/f32/lowoha_gdn_ck_f32.cpp)

set(lowoha_gdn_public_headers
    lowoha_gdn.hpp
    lowoha_gdn_common.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
                 PRIVATE ${lowoha_gdn_sources}
                 PUBLIC  FILE_SET zendnnl_headers
                 TYPE HEADERS FILES ${lowoha_gdn_public_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
                 PRIVATE ${lowoha_gdn_sources}
                 PUBLIC  FILE_SET zendnnl_headers
                 TYPE HEADERS FILES ${lowoha_gdn_public_headers})
endif()

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
           BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

list(APPEND lowoha_gdn_public_headers_list
     lowoha_gdn.hpp)
create_header_file(${zendnnl_inc_file}
                   ${inc_subdir}
                   "${lowoha_gdn_public_headers_list}")
