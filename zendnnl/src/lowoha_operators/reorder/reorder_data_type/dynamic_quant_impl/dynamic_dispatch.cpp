/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"

#include <vector>

namespace zendnnl {
namespace lowoha {
namespace reorder {

/**
 * @brief Dispatch fused per-token dynamic quantization (native AVX-512 path)
 *
 * Handles all supported src/dst type combinations for per-token (M,1)
 * dynamic quantization using the fused kernel that computes scale/zp and
 * quantizes in a single cache-friendly pass per row.
 *
 * Each kernel manages its own OMP parallel region internally.
 *
 * Supports both f32 and bf16 scale output buffers. When scale.dt is bf16,
 * an intermediate f32 buffer is used for kernel computation, then converted
 * to bf16 on output.
 *
 * @return true if a matching kernel was dispatched, false otherwise
 */
bool dispatch_fused_per_token(const void *src, void *dst,
                                      const reorder_params_t &params,
                                      int64_t M, int64_t N) {
  const auto scale_dt = params.quant_params.scale.dt;
  if (scale_dt != data_type_t::f32 && scale_dt != data_type_t::bf16)
    return false;

  const bool scale_is_bf16 = (scale_dt == data_type_t::bf16);
  std::vector<float> scale_f32_tmp;
  float *scale_f32;

  if (scale_is_bf16) {
    scale_f32_tmp.resize(M);
    scale_f32 = scale_f32_tmp.data();
  } else {
    scale_f32 = static_cast<float *>(params.quant_params.scale.buff);
  }

  const bool is_symmetric = (params.quant_params.zero_point.buff == nullptr);
  bool dispatched = false;

  if (is_symmetric) {
    if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::s8) {
      dynamic_per_token_quant_bf16_s8_native(
          static_cast<const uint16_t *>(src),
          static_cast<int8_t *>(dst), scale_f32, M, N);
      dispatched = true;
    } else if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::s8) {
      dynamic_per_token_quant_f32_s8_native(
          static_cast<const float *>(src),
          static_cast<int8_t *>(dst), scale_f32, M, N);
      dispatched = true;
    }
  } else {
    int32_t *zp_out = static_cast<int32_t *>(params.quant_params.zero_point.buff);
    if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::u8) {
      dynamic_per_token_quant_bf16_u8_native(
          static_cast<const uint16_t *>(src),
          static_cast<uint8_t *>(dst), scale_f32, zp_out, M, N);
      dispatched = true;
    } else if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::u8) {
      dynamic_per_token_quant_f32_u8_native(
          static_cast<const float *>(src),
          static_cast<uint8_t *>(dst), scale_f32, zp_out, M, N);
      dispatched = true;
    }
  }

  if (dispatched && scale_is_bf16) {
    uint16_t *bf16_out = static_cast<uint16_t *>(params.quant_params.scale.buff);
    for (int64_t m = 0; m < M; ++m)
      bf16_out[m] = float_to_bf16(scale_f32[m]);
  }

  return dispatched;
}

/**
 * @brief Dispatch unfused 2-pass per-token dynamic quantization (AVX-512 path)
 *
 * Pass 1: compute per-row scale/zp (parallel over M rows, AVX-512).
 * Pass 2: quantize (parallel over M*N contiguous elements, AVX-512).
 * Better thread utilization than fused kernels when M < num_threads.
 *
 * @return true if a matching kernel was dispatched, false otherwise
 */

bool dispatch_unfused_per_token(const void *src, void *dst,
                                        const reorder_params_t &params,
                                        int64_t M, int64_t N) {
  const auto scale_dt = params.quant_params.scale.dt;
  if (scale_dt != data_type_t::f32 && scale_dt != data_type_t::bf16)
    return false;

  const bool scale_is_bf16 = (scale_dt == data_type_t::bf16);
  std::vector<float> scale_f32_tmp;
  float *scale_f32;

  if (scale_is_bf16) {
    scale_f32_tmp.resize(M);
    scale_f32 = scale_f32_tmp.data();
  } else {
    scale_f32 = static_cast<float *>(params.quant_params.scale.buff);
  }

  const bool is_symmetric = (params.quant_params.zero_point.buff == nullptr);
  bool dispatched = false;

  if (is_symmetric) {
    if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::s8) {
      dynamic_per_token_quant_bf16_s8_unfused_native(
          static_cast<const uint16_t *>(src),
          static_cast<int8_t *>(dst), scale_f32, M, N);
      dispatched = true;
    } else if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::s8) {
      dynamic_per_token_quant_f32_s8_unfused_native(
          static_cast<const float *>(src),
          static_cast<int8_t *>(dst), scale_f32, M, N);
      dispatched = true;
    }
  } else {
    int32_t *zp_out = static_cast<int32_t *>(params.quant_params.zero_point.buff);
    if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::u8) {
      dynamic_per_token_quant_bf16_u8_unfused_native(
          static_cast<const uint16_t *>(src),
          static_cast<uint8_t *>(dst), scale_f32, zp_out, M, N);
      dispatched = true;
    } else if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::u8) {
      dynamic_per_token_quant_f32_u8_unfused_native(
          static_cast<const float *>(src),
          static_cast<uint8_t *>(dst), scale_f32, zp_out, M, N);
      dispatched = true;
    }
  }

  if (dispatched && scale_is_bf16) {
    uint16_t *bf16_out = static_cast<uint16_t *>(params.quant_params.scale.buff);
    for (int64_t m = 0; m < M; ++m)
      bf16_out[m] = float_to_bf16(scale_f32[m]);
  }

  return dispatched;
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
