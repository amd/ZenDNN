/********************************************************************************
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

#ifndef _LOWOHA_DISPATCH_KERNEL_HPP
#define _LOWOHA_DISPATCH_KERNEL_HPP

#include "lowoha_embag_common.hpp"
#include "operators/embag/native_kernels/embag_avx512_kernels.hpp"

namespace zendnnl {
namespace lowoha {

/**
 * @brief Dispatch to optimized AVX512 embedding bag kernel
 *
 * Dispatches to the appropriate AVX512 kernel instantiation based on
 * indices, offsets, table, and output data types.
 */
static void dispatch_avx512_kernel(
  const void *table,
  const void *indices,
  const void *offsets,
  const float *weights,
  void *dst,
  const embag_params_t &params) {

  const int64_t embedding_dim = params.embedding_dim;
  const int64_t num_indices = params.num_indices;
  const int64_t num_bags = params.num_bags;
  const int64_t padding_idx = params.padding_idx;
  const bool include_last_offset = params.include_last_offset;

  // Use algo directly since lowoha::embag_algo_t is aliased to ops::embag_algo_t
  const embag_algo_t algo = params.algo;

  const bool is_weights = params.is_weights;
  const int64_t dst_stride = embedding_dim;

  // Dispatch based on indices/offsets types
  // For embedding lookup (algo == none), offsets is nullptr
  const bool is_offsets = (offsets != nullptr);

  if (params.dtypes.indices == data_type_t::s64 &&
      (!is_offsets || params.dtypes.offsets == data_type_t::s64)) {
    if (params.dtypes.table == data_type_t::f32 &&
        params.dtypes.output == data_type_t::f32) {
      zendnnl::ops::embag_avx512_kernel<float, int64_t, int64_t, float>(
        static_cast<const float *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
    }
    else if (params.dtypes.table == data_type_t::bf16 &&
             params.dtypes.output == data_type_t::bf16) {
      zendnnl::ops::embag_avx512_kernel<uint16_t, int64_t, int64_t, uint16_t>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
    }
    else if (params.dtypes.table == data_type_t::bf16 &&
             params.dtypes.output == data_type_t::f32) {
      zendnnl::ops::embag_avx512_kernel<uint16_t, int64_t, int64_t, float>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
    }
    else if (params.dtypes.table == data_type_t::f32 &&
             params.dtypes.output == data_type_t::bf16) {
      zendnnl::ops::embag_avx512_kernel<float, int64_t, int64_t, uint16_t>(
        static_cast<const float *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
    }
    else {
      log_error("embedding_bag_direct: unsupported table and output data types");
    }
  }
  else if (params.dtypes.indices == data_type_t::s32 &&
           (!is_offsets || params.dtypes.offsets == data_type_t::s32)) {
    if (params.dtypes.table == data_type_t::f32 &&
        params.dtypes.output == data_type_t::f32) {
      zendnnl::ops::embag_avx512_kernel<float, int32_t, int32_t, float>(
        static_cast<const float *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
    }
    else if (params.dtypes.table == data_type_t::bf16 &&
             params.dtypes.output == data_type_t::bf16) {
      zendnnl::ops::embag_avx512_kernel<uint16_t, int32_t, int32_t, uint16_t>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
    }
    else if (params.dtypes.table == data_type_t::bf16 &&
             params.dtypes.output == data_type_t::f32) {
      zendnnl::ops::embag_avx512_kernel<uint16_t, int32_t, int32_t, float>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
    }
    else if (params.dtypes.table == data_type_t::f32 &&
             params.dtypes.output == data_type_t::bf16) {
      zendnnl::ops::embag_avx512_kernel<float, int32_t, int32_t, uint16_t>(
        static_cast<const float *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
    }
    else {
      log_error("embedding_bag_direct: unsupported table and output data types");
    }
  }
  else {
    log_error("embedding_bag_direct: unsupported indices/offsets data types");
  }
}

} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_DISPATCH_KERNEL_HPP
