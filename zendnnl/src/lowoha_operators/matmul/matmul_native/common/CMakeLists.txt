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

# Shared infrastructure for Native Matmul
set(matmul_native_common_sources
  cost_model.cpp
  kernel_cache.cpp
  native_pack.cpp
  native_weight_cache.cpp
  native_weight_view.cpp
  native_epilogue.cpp
  native_operator_bridge.cpp
  native_hw.cpp)

set(matmul_native_common_headers
  gemm_descriptor.hpp
  native_pack.hpp
  native_weight_cache.hpp
  native_weight_view.hpp
  native_epilogue.hpp
  native_operator_bridge.hpp
  native_hw.hpp
  native_weight_ends.hpp
  native_int8_scheme.hpp
  cost_model.hpp
  kernel_cache.hpp
  avx512_math.hpp
  native_utils.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PRIVATE ${matmul_native_common_sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${matmul_native_common_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PRIVATE ${matmul_native_common_sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${matmul_native_common_headers})
endif()
