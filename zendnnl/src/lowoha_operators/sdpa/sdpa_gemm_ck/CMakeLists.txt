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
#  Custom-kernel (CK) GEMMs for flash SDPA: BF16 behind zendnn_gemm, INT8
#  behind the dynamic-INT8 Q*K^T and P*V matmuls. ISA target attributes are
#  set per function; the runtime ISA checks are in lowoha_sdpa_ck_common.cpp.
#  Headers are internal (included by flash_sdpa), so none are installed.
# ---------------------------------------------------------------------------

# lowoha_sdpa_ck.hpp: entry points per data type; lowoha_sdpa_ck_common.*:
# modes and data-type independent pieces; <dtype>/: kernels and GEMMs.
set(lowoha_sdpa_gemm_ck_sources
    lowoha_sdpa_ck_common.cpp
    bf16/lowoha_sdpa_ck_bf16.cpp
    int8/lowoha_sdpa_ck_int8.cpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive PRIVATE ${lowoha_sdpa_gemm_ck_sources})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl PRIVATE ${lowoha_sdpa_gemm_ck_sources})
endif()
