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

# Packed variable-length flash SDPA kernels (FP and dynamic INT8). Internal
# only: their entry points are declared in the parent flash_sdpa headers.
set(lowoha_flash_sdpa_varlen_sources
    lowoha_sdpa_flash_cpu_varlen.cpp
    lowoha_sdpa_flash_cpu_varlen_int8.cpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
                 PRIVATE ${lowoha_flash_sdpa_varlen_sources})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
                 PRIVATE ${lowoha_flash_sdpa_varlen_sources})
endif()
