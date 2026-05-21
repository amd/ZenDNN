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

#include "lowoha_operators/reorder/reorder_data_type/reorder_dtype_dispatch.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace reorder {

using zendnnl::lowoha::matmul::zendnnl_parallel_for;

/**
 * @brief Execute element-wise quantization for 2D matrix (non-per-tensor granularity)
 * 
 * Handles per-channel, per-group, and mixed quantization where each element
 * may use a different scale/zero-point based on its position.
 */
void reorder_granular_scaler_impl_2d(const void *src, void *dst,
                                         const reorder_params_t &params) {
  const int64_t M = params.M();
  const int64_t N = params.N();
  const int64_t total_work = M * N;

  // ── Source-stride support ─────────────────────────────────────────────
  // The dst side is always a contiguous (M × N) buffer (the caller is
  // either reorder_quantization_wrapper, which malloc's the dst
  // contiguously, or the public reorder API, which constrains dst to
  // be contiguous via the absence of `dst_strides` in
  // `reorder_params_t`).  The src side, however, can be strided —
  // most importantly for the fused-MoE Op2 source-quant flow, where
  // the wrapper sets `src_strides = {lda, 1}` because the activated
  // Op1 intermediate is wide (`lda = N_op1 > K_down = N_op1 / 2` for
  // gated activations) so the (M × K_down) sub-slice the reorder
  // needs to read is stride-`N_op1` between rows.  When `src_strides`
  // is unset (or contiguous), `stride_m = N` and `stride_n = 1`, so
  // `src_idx = work_idx` — bit-identical to the legacy contiguous
  // implementation.
  //
  // Without honouring strides here, the loop body's `src[work_idx]`
  // would read sequentially as if every `N` consecutive elements were
  // one row — for the strided case that pulls in garbage columns
  // (`[K_down, N_op1)` of each row, which post-activation are stale
  // raw matmul outputs).  The downstream `dynamic_per_token` AVX path
  // is already guarded against strided src (see
  // `lowoha_reorder.cpp:108-110`), so this scalar fallback is the
  // ONLY path that needs the fix.
  const bool has_src_strides = params.has_src_strides();
  const int64_t stride_m =
      has_src_strides ? params.src_strides[0] : N;
  const int64_t stride_n =
      has_src_strides ? params.src_strides[1] : 1;

  // Hot-path branch-free helper.  Returns the absolute element index
  // into the typed src array for the (i, j) coordinate the work loop
  // is currently processing.  Resolves to `work_idx` on the
  // contiguous fast path because `stride_m == N && stride_n == 1`.
  auto src_idx_for = [stride_m, stride_n](int64_t i, int64_t j) -> int64_t {
    return i * stride_m + j * stride_n;
  };

  // BF16 -> INT8
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::s8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_int8[work_idx] = quantize_bf16_to_s8_scalar(src_bf16[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }
  
  // BF16 -> UINT8
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::u8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_uint8[work_idx] = quantize_bf16_to_u8_scalar(src_bf16[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }
  
  // INT8 -> BF16
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::bf16) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = dequantize_s8_to_bf16_scalar(src_int8[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }
  
  // UINT8 -> BF16
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::bf16) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = dequantize_u8_to_bf16_scalar(src_uint8[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }

  // FP32 -> INT8
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::s8) {
    const float *src_f32 = static_cast<const float *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_int8[work_idx] = quantize_f32_to_s8_scalar(src_f32[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }
  
  // FP32 -> UINT8
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::u8) {
    const float *src_f32 = static_cast<const float *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_uint8[work_idx] = quantize_f32_to_u8_scalar(src_f32[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }
  
  // INT8 -> FP32
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::f32) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = dequantize_s8_to_f32_scalar(src_int8[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }
  
  // UINT8 -> FP32
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::f32) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = dequantize_u8_to_f32_scalar(src_uint8[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }

  // FP32 -> BF16 (with optional scaling)
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::bf16) {
    const float *src_f32 = static_cast<const float *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = convert_f32_to_bf16_scalar(src_f32[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }

  // BF16 -> FP32 (with optional scaling)
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::f32) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = convert_bf16_to_f32_scalar(src_bf16[src_idx_for(i, j)], scale, zp);
      }
    });
    return;
  }
}
/**
 * @brief Execute element-wise quantization for 3D batched matrix (non-per-tensor granularity)
 * 
 * Handles per-channel, per-group, and mixed quantization where each element
 * may use a different scale/zero-point based on its position.
 */
void reorder_granular_scaler_impl_3d(const void *src, void *dst,
                                         const reorder_params_t &params) {
  const int64_t batch = params.batch();
  const int64_t M = params.M();
  const int64_t N = params.N();
  const int64_t matrix_size = M * N;
  const int64_t total_work = batch * matrix_size;

  // ── Source-stride support ─────────────────────────────────────────────
  // Same contract as `reorder_granular_scaler_impl_2d` (see its header
  // comment for the full rationale): dst is contiguous by API design;
  // src can be strided when the caller is the dynamic-quant reorder
  // wrapper feeding from a wide intermediate (e.g. batched fused-MoE
  // Op2 source with a gated activation).  Strides default to the
  // contiguous layout (`stride_batch = M*N`, `stride_m = N`,
  // `stride_n = 1`) when `src_strides` is unset, so the resolved
  // `src_idx_for` matches `work_idx` bit-for-bit on the contiguous
  // fast path.
  const bool has_src_strides = params.has_src_strides();
  const int64_t stride_batch =
      has_src_strides ? params.src_strides[0] : matrix_size;
  const int64_t stride_m =
      has_src_strides ? params.src_strides[1] : N;
  const int64_t stride_n =
      has_src_strides ? params.src_strides[2] : 1;

  auto src_idx_for = [stride_batch, stride_m, stride_n](
      int64_t b, int64_t i, int64_t j) -> int64_t {
    return b * stride_batch + i * stride_m + j * stride_n;
  };

  // BF16 -> INT8
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::s8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);

    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_int8[work_idx] = quantize_bf16_to_s8_scalar(src_bf16[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }
  
  // BF16 -> UINT8
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::u8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_uint8[work_idx] = quantize_bf16_to_u8_scalar(src_bf16[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }
  
  // INT8 -> BF16
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::bf16) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = dequantize_s8_to_bf16_scalar(src_int8[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }
  
  // UINT8 -> BF16
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::bf16) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = dequantize_u8_to_bf16_scalar(src_uint8[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }

  // FP32 -> INT8
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::s8) {
    const float *src_f32 = static_cast<const float *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);

    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_int8[work_idx] = quantize_f32_to_s8_scalar(src_f32[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }
  
  // FP32 -> UINT8
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::u8) {
    const float *src_f32 = static_cast<const float *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_uint8[work_idx] = quantize_f32_to_u8_scalar(src_f32[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }
  
  // INT8 -> FP32
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::f32) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = dequantize_s8_to_f32_scalar(src_int8[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }
  
  // UINT8 -> FP32
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::f32) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = dequantize_u8_to_f32_scalar(src_uint8[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }

  // FP32 -> BF16 (with optional scaling)
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::bf16) {
    const float *src_f32 = static_cast<const float *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = convert_f32_to_bf16_scalar(src_f32[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }

  // BF16 -> FP32 (with optional scaling)
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::f32) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t b = work_idx / matrix_size;
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = convert_bf16_to_f32_scalar(src_bf16[src_idx_for(b, i, j)], scale, zp);
      }
    });
    return;
  }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
