/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "operators/embag/embag_config.hpp"
#include "operators/embag/native_kernels/embag_avx512_kernels.hpp"
#if ZENDNNL_DEPENDS_FBGEMM
  #include "fbgemm_kernel.hpp"
#endif

// can_use_f16_fma_kernel() is provided by embag_avx512_kernels.hpp
using zendnnl::ops::can_use_f16_fma_kernel;

// Forward declarations for AVX2 kernel template instantiations
// These are defined in embag_avx2_kernels.cpp and already compiled
namespace zendnnl {
namespace ops {
template <typename InType, typename IndexType, typename OffsetType, typename OutType>
void embag_avx2_kernel(
  const InType *input,
  const float *weights,
  const IndexType *indices,
  const OffsetType *offsets,
  OutType *dst,
  int64_t width,
  int64_t indsz,
  int64_t offsz,
  int64_t padidx,
  bool is_weights,
  embag_algo_t algo,
  int64_t dst_stride,
  bool include_last_offset);
} // namespace ops
} // namespace zendnnl

namespace zendnnl {
namespace lowoha {
namespace embag {

using zendnnl::common::float16_t;
using zendnnl::ops::can_use_f16_fma_kernel;

/**
 * @brief Dispatch to native AVX512 embedding bag kernel
 *
 * Dispatches to the appropriate native AVX512 kernel instantiation based on
 * indices, offsets, table, and output data types.
 */
static void embag_native_kernel(
  const void *table,
  const void *indices,
  const void *offsets,
  const float *weights,
  void *dst,
  const embag_params_t &params) {

  const uint64_t embedding_dim = params.embedding_dim;
  const uint64_t num_indices = params.num_indices;
  const uint64_t num_bags = params.num_bags;
  const int64_t padding_idx = params.padding_idx;
  const bool include_last_offset = params.include_last_offset;
  const data_type_t table_dtype = params.dtypes.table;
  const bool fp16_scale_bias = params.fp16_scale_bias;

  // Use algo directly since lowoha::embag_algo_t is aliased to ops::embag_algo_t
  const embag_algo_t algo = params.algo;

  const bool is_weights = params.is_weights;
  const int64_t dst_stride = params.dst_stride;

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
#if __GNUC__ >= 12
      zendnnl::ops::embag_avx512_kernel<uint16_t, int64_t, int64_t, uint16_t>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
#else
      zendnnl::ops::embag_avx2_kernel<uint16_t, int64_t, int64_t, uint16_t>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
#endif
    }
    else if (params.dtypes.table == data_type_t::bf16 &&
             params.dtypes.output == data_type_t::f32) {
#if __GNUC__ >= 12
      zendnnl::ops::embag_avx512_kernel<uint16_t, int64_t, int64_t, float>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
#else
      zendnnl::ops::embag_avx2_kernel<uint16_t, int64_t, int64_t, float>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
#endif
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
    else if (params.dtypes.table == data_type_t::f16 &&
             params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_f16_fma_kernel<float16_t, int64_t, int64_t, float16_t>
        (
          static_cast<const float16_t *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_kernel<float16_t, int64_t, int64_t, float16_t>(
          static_cast<const float16_t *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
    }
    else if (params.dtypes.table == data_type_t::f16 &&
             params.dtypes.output == data_type_t::f32) {
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_f16_fma_kernel<float16_t, int64_t, int64_t, float>(
          static_cast<const float16_t *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_kernel<float16_t, int64_t, int64_t, float>(
          static_cast<const float16_t *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
    }
    else if (params.dtypes.table == data_type_t::f32 &&
             params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_f16_fma_kernel<float, int64_t, int64_t, float16_t>(
          static_cast<const float *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_kernel<float, int64_t, int64_t, float16_t>(
          static_cast<const float *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
    }
    else if (params.dtypes.table == data_type_t::s8 &&
             params.dtypes.output == data_type_t::f32) {
      zendnnl::ops::embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t, float>
      (
        static_cast<const int8_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset,
        table_dtype, fp16_scale_bias);
    }
    else if (params.dtypes.table == data_type_t::s8 &&
             params.dtypes.output == data_type_t::bf16) {
      zendnnl::ops::embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t, uint16_t>
      (
        static_cast<const int8_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset,
        table_dtype, fp16_scale_bias);
    }
    else if ((params.dtypes.table == data_type_t::s4 ||
              params.dtypes.table == data_type_t::u4) &&
             params.dtypes.output == data_type_t::f32) {
      zendnnl::ops::embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t, float>
      (
        static_cast<const uint8_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset,
        table_dtype, fp16_scale_bias);
    }
    else if ((params.dtypes.table == data_type_t::s4 ||
              params.dtypes.table == data_type_t::u4) &&
             params.dtypes.output == data_type_t::bf16) {
      zendnnl::ops::embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t, uint16_t>
      (
        static_cast<const uint8_t *>(table), weights,
        static_cast<const int64_t *>(indices),
        static_cast<const int64_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset,
        table_dtype, fp16_scale_bias);
    }
    else if (params.dtypes.table == data_type_t::s8 &&
             params.dtypes.output == data_type_t::f16) {
      using zendnnl::common::float16_t;
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_int8_int4_f16_fma_kernel<false, int8_t, int64_t, int64_t, float16_t>
        (
          static_cast<const int8_t *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset,
          table_dtype, fp16_scale_bias);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t, float16_t>
        (
          static_cast<const int8_t *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset,
          table_dtype, fp16_scale_bias);
      }
    }
    else if ((params.dtypes.table == data_type_t::s4 ||
              params.dtypes.table == data_type_t::u4) &&
             params.dtypes.output == data_type_t::f16) {
      using zendnnl::common::float16_t;
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_int8_int4_f16_fma_kernel<true, uint8_t, int64_t, int64_t, float16_t>
        (
          static_cast<const uint8_t *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset,
          table_dtype, fp16_scale_bias);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t, float16_t>
        (
          static_cast<const uint8_t *>(table), weights,
          static_cast<const int64_t *>(indices),
          static_cast<const int64_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset,
          table_dtype, fp16_scale_bias);
      }
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
#if __GNUC__ >= 12
      zendnnl::ops::embag_avx512_kernel<uint16_t, int32_t, int32_t, uint16_t>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
#else
      zendnnl::ops::embag_avx2_kernel<uint16_t, int32_t, int32_t, uint16_t>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
#endif
    }
    else if (params.dtypes.table == data_type_t::bf16 &&
             params.dtypes.output == data_type_t::f32) {
#if __GNUC__ >= 12
      zendnnl::ops::embag_avx512_kernel<uint16_t, int32_t, int32_t, float>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
#else
      zendnnl::ops::embag_avx2_kernel<uint16_t, int32_t, int32_t, float>(
        static_cast<const uint16_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset);
#endif
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
    else if (params.dtypes.table == data_type_t::f16 &&
             params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_f16_fma_kernel<float16_t, int32_t, int32_t, float16_t>
        (
          static_cast<const float16_t *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_kernel<float16_t, int32_t, int32_t, float16_t>(
          static_cast<const float16_t *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
    }
    else if (params.dtypes.table == data_type_t::f16 &&
             params.dtypes.output == data_type_t::f32) {
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_f16_fma_kernel<float16_t, int32_t, int32_t, float>(
          static_cast<const float16_t *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_kernel<float16_t, int32_t, int32_t, float>(
          static_cast<const float16_t *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
    }
    else if (params.dtypes.table == data_type_t::f32 &&
             params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_f16_fma_kernel<float, int32_t, int32_t, float16_t>(
          static_cast<const float *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_kernel<float, int32_t, int32_t, float16_t>(
          static_cast<const float *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset);
      }
    }
    else if (params.dtypes.table == data_type_t::s8 &&
             params.dtypes.output == data_type_t::f32) {
      zendnnl::ops::embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t, float>
      (
        static_cast<const int8_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset,
        table_dtype, fp16_scale_bias);
    }
    else if (params.dtypes.table == data_type_t::s8 &&
             params.dtypes.output == data_type_t::bf16) {
      zendnnl::ops::embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t, uint16_t>
      (
        static_cast<const int8_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset,
        table_dtype, fp16_scale_bias);
    }
    else if ((params.dtypes.table == data_type_t::s4 ||
              params.dtypes.table == data_type_t::u4) &&
             params.dtypes.output == data_type_t::f32) {
      zendnnl::ops::embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t, float>
      (
        static_cast<const uint8_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<float *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset,
        table_dtype, fp16_scale_bias);
    }
    else if ((params.dtypes.table == data_type_t::s4 ||
              params.dtypes.table == data_type_t::u4) &&
             params.dtypes.output == data_type_t::bf16) {
      zendnnl::ops::embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t, uint16_t>
      (
        static_cast<const uint8_t *>(table), weights,
        static_cast<const int32_t *>(indices),
        static_cast<const int32_t *>(offsets),
        static_cast<uint16_t *>(dst),
        embedding_dim, num_indices, num_bags, padding_idx,
        is_weights, algo, dst_stride, include_last_offset,
        table_dtype, fp16_scale_bias);
    }
    else if (params.dtypes.table == data_type_t::s8 &&
             params.dtypes.output == data_type_t::f16) {
      using zendnnl::common::float16_t;
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_int8_int4_f16_fma_kernel<false, int8_t, int32_t, int32_t, float16_t>
        (
          static_cast<const int8_t *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset,
          table_dtype, fp16_scale_bias);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t, float16_t>
        (
          static_cast<const int8_t *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset,
          table_dtype, fp16_scale_bias);
      }
    }
    else if ((params.dtypes.table == data_type_t::s4 ||
              params.dtypes.table == data_type_t::u4) &&
             params.dtypes.output == data_type_t::f16) {
      using zendnnl::common::float16_t;
#if __GNUC__ >= 12
      if (can_use_f16_fma_kernel()) {
        zendnnl::ops::embag_avx512_int8_int4_f16_fma_kernel<true, uint8_t, int32_t, int32_t, float16_t>
        (
          static_cast<const uint8_t *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset,
          table_dtype, fp16_scale_bias);
      }
      else
#endif
      {
        zendnnl::ops::embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t, float16_t>
        (
          static_cast<const uint8_t *>(table), weights,
          static_cast<const int32_t *>(indices),
          static_cast<const int32_t *>(offsets),
          static_cast<float16_t *>(dst),
          embedding_dim, num_indices, num_bags, padding_idx,
          is_weights, algo, dst_stride, include_last_offset,
          table_dtype, fp16_scale_bias);
      }
    }
    else {
      log_error("embedding_bag_direct: unsupported table and output data types");
    }
  }
  else {
    log_error("embedding_bag_direct: unsupported indices/offsets data types");
  }
}

#if ZENDNNL_DEPENDS_FBGEMM
/**
 * @brief Dispatch to FBGEMM embedding bag kernel
 *
 * Dispatches to the appropriate FBGEMM kernel instantiation based on
 * indices, offsets, table, and output data types.
 */
static void embag_fbgemm_kernel(
  const void *table,
  const void *indices,
  const void *offsets,
  const float *weights,
  void *dst,
  const embag_params_t &params) {

  const data_type_t table_dtype = params.dtypes.table;
  const data_type_t output_dtype = params.dtypes.output;

  // Dispatch based on indices/offsets types (s64 or s32)
  if (params.dtypes.indices == data_type_t::s64 &&
      params.dtypes.offsets == data_type_t::s64) {
    if (table_dtype == data_type_t::f32 && output_dtype == data_type_t::f32) {
      invoke_fbgemm_kernel<false, float, int64_t, int64_t, float>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::bf16 &&
             output_dtype == data_type_t::bf16) {
      invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/true);
    }
    else if (table_dtype == data_type_t::bf16 && output_dtype == data_type_t::f32) {
      invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t, float>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::f32 && output_dtype == data_type_t::bf16) {
      invoke_fbgemm_kernel<false, float, int64_t, int64_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
    }
    else if (table_dtype == data_type_t::f16 && output_dtype == data_type_t::f16) {
      invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::f16 && output_dtype == data_type_t::f32) {
      invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t, float>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::f32 && output_dtype == data_type_t::f16) {
      invoke_fbgemm_kernel<false, float, int64_t, int64_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    // TODO: Explore the feasibility of using FBGEMM for s4/s8 quantized types.
    // s4 and s8 are handled by the native kernel.
    else if (table_dtype == data_type_t::u4 &&
             output_dtype == data_type_t::f32) {
      invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t, float>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::u4 &&
             output_dtype == data_type_t::bf16) {
      invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
    }
    else if (table_dtype == data_type_t::u4 &&
             output_dtype == data_type_t::f16) {
      invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else {
      log_error("embedding_bag_direct: unsupported table/output data types for FBGEMM backend");
    }
  }
  else if (params.dtypes.indices == data_type_t::s32 &&
           params.dtypes.offsets == data_type_t::s32) {
    if (table_dtype == data_type_t::f32 && output_dtype == data_type_t::f32) {
      invoke_fbgemm_kernel<false, float, int32_t, int32_t, float>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::bf16 &&
             output_dtype == data_type_t::bf16) {
      invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/true);
    }
    else if (table_dtype == data_type_t::bf16 && output_dtype == data_type_t::f32) {
      invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t, float>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::f32 && output_dtype == data_type_t::bf16) {
      invoke_fbgemm_kernel<false, float, int32_t, int32_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
    }
    else if (table_dtype == data_type_t::f16 && output_dtype == data_type_t::f16) {
      invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::f16 && output_dtype == data_type_t::f32) {
      invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t, float>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::f32 && output_dtype == data_type_t::f16) {
      invoke_fbgemm_kernel<false, float, int32_t, int32_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::u4 &&
             output_dtype == data_type_t::f32) {
      invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t, float>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else if (table_dtype == data_type_t::u4 &&
             output_dtype == data_type_t::bf16) {
      invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
    }
    else if (table_dtype == data_type_t::u4 &&
             output_dtype == data_type_t::f16) {
      invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t, uint16_t>(
        table, indices, offsets, weights, dst, params,
        /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/false);
    }
    else {
      log_error("embedding_bag_direct: unsupported table/output data types for FBGEMM backend");
    }
  }
  else {
    log_error("embedding_bag_direct: unsupported indices/offsets data types for FBGEMM backend");
  }
}
#endif

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
  embag_params_t &params) {

  kernel_select(params);

  // Update singleton accum_type so the reference kernel can later read which
  // accumulation precision to use to bit-match the chosen backend.
  // - FBGEMM accumulates in F32 internally regardless of dtypes.
  // - Native AVX512 uses F16 FMA only on F16-touching dtype combinations
  //   (f16/f16, f16/f32, f32/f16) when can_use_f16_fma_kernel() is true.
  //
  // TODO(embag-accum-singleton): this set_accum_type write races with
  // sibling threads when dispatch_avx512_kernel is called from
  // group_embedding_bag_direct's #pragma omp parallel region with
  // mixed-dtype groups. See embag_config.hpp set_accum_type doc.
  // Likely fix: make embag_accum_type thread_local.
  zendnnl::ops::embag_config_t &embag_config =
    zendnnl::ops::embag_config_t::instance();

#if ZENDNNL_DEPENDS_FBGEMM
  if (params.kernel == embag_kernel_t::fbgemm && can_use_fbgemm(params)) {
    embag_config.set_accum_type(data_type_t::f32);
    log_info("Using FBGEMM kernel");
    embag_fbgemm_kernel(table, indices, offsets, weights, dst, params);
    return;
  }
#endif

  // Native ZenDNN path: F16 accumulation only when at least one of
  // table/output is F16 AND the F16 FMA kernel is actually available
  // (GCC >= 12 for intrinsics, and can_use_f16_fma_kernel() for HW +
  // ZENDNNL_NATIVE_F32_ACCUM). Otherwise the kernel accumulates in F32.
  [[maybe_unused]] bool is_f16_path = (params.dtypes.table == data_type_t::f16 ||
                                       params.dtypes.output == data_type_t::f16);
#if __GNUC__ >= 12
  bool native_uses_f16_fma = is_f16_path && can_use_f16_fma_kernel();
#else
  bool native_uses_f16_fma = false;
#endif
  embag_config.set_accum_type(native_uses_f16_fma ? data_type_t::f16
                              : data_type_t::f32);

  log_info("Using ZenDNN kernel");
  embag_native_kernel(table, indices, offsets, weights, dst, params);
  return;
}

} // namespace embag
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_DISPATCH_KERNEL_HPP
