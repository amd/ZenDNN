# *******************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#add_subdirectory(native_kernels)
add_subdirectory(matmul_native)
add_subdirectory(group_matmul)

set(lowoha_matmul_sources
  lowoha_matmul.cpp
  group_matmul_direct.cpp
  ${lowoha_group_matmul_sources}
  lowoha_matmul_utils.cpp
  ggml_weight_unpack.cpp
  partitioning/bmm/bmm_partitioner.cpp
  partitioning/matmul/matmul_partitioner.cpp
  backends/aocl/aocl_kernel.cpp
  tuning/auto_tuner.cpp
  backends/onednn/onednn_kernel.cpp
  quantization/reorder_quantization.cpp)

set(lowoha_matmul_public_headers
  lowoha_common.hpp
  lowoha_matmul.hpp
  lowoha_matmul_utils.hpp
  ${lowoha_group_matmul_public_headers}

  cache/lru_cache.hpp
  cache/zendnnl_key.hpp)

set(lowoha_matmul_private_headers
  backends/libxsmm/libxsmm_utils.hpp
  backends/libxsmm/libxsmm_kernel.hpp
  backends/aocl/aocl_kernel.hpp
  cache/lowoha_cache.hpp
  backends/onednn/onednn_kernel.hpp
  tuning/auto_tuner.hpp
  partitioning/bmm/bmm_partitioner.hpp
  partitioning/matmul/matmul_partitioner.hpp
  brgemm_matmul.hpp
  matmul_kernel_bf16_avx512.hpp
  quantization/reorder_quantization.hpp
  ggml_weight_unpack.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PRIVATE ${lowoha_matmul_sources}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${lowoha_matmul_public_headers}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${lowoha_matmul_private_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PRIVATE ${lowoha_matmul_sources}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${lowoha_matmul_public_headers}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${lowoha_matmul_private_headers})
endif()

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

list(APPEND lowoha_matmul_public_headers_list
  ${lowoha_matmul_public_headers})
create_header_file(${zendnnl_inc_file}
  ${inc_subdir}
  "${lowoha_matmul_public_headers_list}")

