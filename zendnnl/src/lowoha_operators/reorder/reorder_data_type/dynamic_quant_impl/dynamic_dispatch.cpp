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

#include <algorithm>
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
 * Supports f32, bf16, and f16 scale output buffers. When scale.dt is
 * bf16 or f16, an intermediate f32 buffer is used for kernel computation
 * and narrowed to the caller's storage dtype on output (via
 * float_to_bf16 / float16_t::f32_to_f16_val).
 *
 * For f16 source, the FMA precision (F32 vs FP16) is selected by
 * can_use_f16_fma_kernel() (defaults to FP16-FMA when the
 * AVX512-FP16 ISA is available and the library was not built with
 * -DZENDNNL_NATIVE_F32_ACCUM=ON).
 *
 * @return true if a matching kernel was dispatched, false otherwise
 */
bool dispatch_fused_per_token(const void *src, void *dst,
                                      const reorder_params_t &params,
                                      int64_t M, int64_t N) {
  const auto scale_dt = params.quant_params.scale.dt;
  if (scale_dt != data_type_t::f32  &&
      scale_dt != data_type_t::bf16 &&
      scale_dt != data_type_t::f16)
    return false;

  // Scale staging: kernels operate on f32 scale. For bf16/f16 user
  // buffers, stage through an std::vector<float> and narrow to the
  // user's dtype after dispatch.
  const bool scale_needs_narrow = (scale_dt == data_type_t::bf16 ||
                                   scale_dt == data_type_t::f16);
  std::vector<float> scale_f32_tmp;
  float *scale_f32;

  if (scale_needs_narrow) {
    scale_f32_tmp.resize(M);
    scale_f32 = scale_f32_tmp.data();
  } else {
    scale_f32 = static_cast<float *>(params.quant_params.scale.buff);
  }

  const bool is_symmetric = (params.quant_params.zero_point.buff == nullptr);
  bool dispatched = false;

  // FP16-FMA vs F32-FMA selection for f16 source. Cached once per call —
  // can_use_f16_fma_kernel() reads platform info, no env var is consulted.
  const bool f16_use_fp16fma =
      (params.src_dtype == data_type_t::f16) ? can_use_f16_fma_kernel() : false;

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
    } else if (params.src_dtype == data_type_t::f16 && params.dst_dtype == data_type_t::s8) {
      if (f16_use_fp16fma) {
        dynamic_per_token_quant_f16_s8_avx512fp16(
            static_cast<const uint16_t *>(src),
            static_cast<int8_t *>(dst), scale_f32, M, N);
      } else {
        dynamic_per_token_quant_f16_s8_native(
            static_cast<const uint16_t *>(src),
            static_cast<int8_t *>(dst), scale_f32, M, N);
      }
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
    } else if (params.src_dtype == data_type_t::f16 && params.dst_dtype == data_type_t::u8) {
      if (f16_use_fp16fma) {
        dynamic_per_token_quant_f16_u8_avx512fp16(
            static_cast<const uint16_t *>(src),
            static_cast<uint8_t *>(dst), scale_f32, zp_out, M, N);
      } else {
        dynamic_per_token_quant_f16_u8_native(
            static_cast<const uint16_t *>(src),
            static_cast<uint8_t *>(dst), scale_f32, zp_out, M, N);
      }
      dispatched = true;
    }
  }

  if (dispatched && scale_needs_narrow) {
    uint16_t *out = static_cast<uint16_t *>(params.quant_params.scale.buff);
    if (scale_dt == data_type_t::bf16) {
      for (int64_t m = 0; m < M; ++m)
        out[m] = float_to_bf16(scale_f32[m]);
    } else {  // f16: floor-then-narrow via narrow_f32_scale_to_f16
      for (int64_t m = 0; m < M; ++m)
        out[m] = common::narrow_f32_scale_to_f16(scale_f32[m]);
    }
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
 * For FP16 source, the FMA precision is selected at runtime between
 * F32-FMA (AVX-512F + F16C) and FP16-FMA (Strategy A, AVX512-FP16 ISA)
 * via can_use_f16_fma_kernel() — same selector as the fused per-token
 * path.
 *
 * @return true if a matching kernel was dispatched, false otherwise
 */

bool dispatch_unfused_per_token(const void *src, void *dst,
                                        const reorder_params_t &params,
                                        int64_t M, int64_t N) {
  const auto scale_dt = params.quant_params.scale.dt;
  if (scale_dt != data_type_t::f32  &&
      scale_dt != data_type_t::bf16 &&
      scale_dt != data_type_t::f16)
    return false;

  const bool scale_needs_narrow = (scale_dt == data_type_t::bf16 ||
                                   scale_dt == data_type_t::f16);
  std::vector<float> scale_f32_tmp;
  float *scale_f32;

  if (scale_needs_narrow) {
    scale_f32_tmp.resize(M);
    scale_f32 = scale_f32_tmp.data();
  } else {
    scale_f32 = static_cast<float *>(params.quant_params.scale.buff);
  }

  const bool is_symmetric = (params.quant_params.zero_point.buff == nullptr);
  bool dispatched = false;

  // FP16-FMA vs F32-FMA selection (same policy as fused per-token).
  const bool f16_use_fp16fma =
      (params.src_dtype == data_type_t::f16) ? can_use_f16_fma_kernel() : false;

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
    } else if (params.src_dtype == data_type_t::f16 && params.dst_dtype == data_type_t::s8) {
      if (f16_use_fp16fma) {
        dynamic_per_token_quant_f16_s8_unfused_avx512fp16(
            static_cast<const uint16_t *>(src),
            static_cast<int8_t *>(dst), scale_f32, M, N);
      } else {
        dynamic_per_token_quant_f16_s8_unfused_native(
            static_cast<const uint16_t *>(src),
            static_cast<int8_t *>(dst), scale_f32, M, N);
      }
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
    } else if (params.src_dtype == data_type_t::f16 && params.dst_dtype == data_type_t::u8) {
      if (f16_use_fp16fma) {
        dynamic_per_token_quant_f16_u8_unfused_avx512fp16(
            static_cast<const uint16_t *>(src),
            static_cast<uint8_t *>(dst), scale_f32, zp_out, M, N);
      } else {
        dynamic_per_token_quant_f16_u8_unfused_native(
            static_cast<const uint16_t *>(src),
            static_cast<uint8_t *>(dst), scale_f32, zp_out, M, N);
      }
      dispatched = true;
    }
  }

  if (dispatched && scale_needs_narrow) {
    uint16_t *out = static_cast<uint16_t *>(params.quant_params.scale.buff);
    if (scale_dt == data_type_t::bf16) {
      for (int64_t m = 0; m < M; ++m)
        out[m] = float_to_bf16(scale_f32[m]);
    } else {  // f16: floor-then-narrow via narrow_f32_scale_to_f16
      for (int64_t m = 0; m < M; ++m)
        out[m] = common::narrow_f32_scale_to_f16(scale_f32[m]);
    }
  }

  return dispatched;
}

/**
 * @brief Dispatch fused per-group dynamic quantization (native AVX-512 path)
 *
 * Handles 2D contiguous per-group-col scale layout {M, G}.  Each native
 * kernel computes scale/zp and quantizes one contiguous K-direction group.
 */
bool dispatch_fused_per_group(const void *src, void *dst,
                                      const reorder_params_t &params,
                                      int64_t M, int64_t K) {
  const auto scale_dt = params.quant_params.scale.dt;
  if (scale_dt != data_type_t::f32  &&
      scale_dt != data_type_t::bf16 &&
      scale_dt != data_type_t::f16)
    return false;
  if (!is_per_group_col_dims(params.quant_params.scale.dims, params.src_shape))
    return false;

  const int64_t G = get_num_groups_col(params.quant_params.scale.dims);
  if (G <= 1 || K % G != 0)
    return false;

  const int64_t scale_nelems = M * G;
  const bool scale_needs_narrow = (scale_dt == data_type_t::bf16 ||
                                   scale_dt == data_type_t::f16);
  std::vector<float> scale_f32_tmp;
  float *scale_f32;

  if (scale_needs_narrow) {
    scale_f32_tmp.resize(scale_nelems);
    scale_f32 = scale_f32_tmp.data();
  } else {
    scale_f32 = static_cast<float *>(params.quant_params.scale.buff);
  }

  const bool is_symmetric = (params.quant_params.zero_point.buff == nullptr);
  bool dispatched = false;

  // FP16 FMA backend selection (per-group path, see can_use_f16_fma_kernel).
  const bool f16_use_fp16fma =
      (params.src_dtype == data_type_t::f16) ? can_use_f16_fma_kernel() : false;

  if (is_symmetric) {
    if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::s8) {
      dynamic_per_group_quant_bf16_s8_native(
          static_cast<const uint16_t *>(src),
          static_cast<int8_t *>(dst), scale_f32, M, K, G);
      dispatched = true;
    } else if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::s8) {
      dynamic_per_group_quant_f32_s8_native(
          static_cast<const float *>(src),
          static_cast<int8_t *>(dst), scale_f32, M, K, G);
      dispatched = true;
    } else if (params.src_dtype == data_type_t::f16 && params.dst_dtype == data_type_t::s8) {
      if (f16_use_fp16fma) {
        dynamic_per_group_quant_f16_s8_avx512fp16(
            static_cast<const uint16_t *>(src),
            static_cast<int8_t *>(dst), scale_f32, M, K, G);
      } else {
        dynamic_per_group_quant_f16_s8_native(
            static_cast<const uint16_t *>(src),
            static_cast<int8_t *>(dst), scale_f32, M, K, G);
      }
      dispatched = true;
    }
  } else {
    if (params.quant_params.zero_point.dims != params.quant_params.scale.dims)
      return false;
    int32_t *zp_out = static_cast<int32_t *>(params.quant_params.zero_point.buff);
    if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::u8) {
      dynamic_per_group_quant_bf16_u8_native(
          static_cast<const uint16_t *>(src),
          static_cast<uint8_t *>(dst), scale_f32, zp_out, M, K, G);
      dispatched = true;
    } else if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::u8) {
      dynamic_per_group_quant_f32_u8_native(
          static_cast<const float *>(src),
          static_cast<uint8_t *>(dst), scale_f32, zp_out, M, K, G);
      dispatched = true;
    } else if (params.src_dtype == data_type_t::f16 && params.dst_dtype == data_type_t::u8) {
      if (f16_use_fp16fma) {
        dynamic_per_group_quant_f16_u8_avx512fp16(
            static_cast<const uint16_t *>(src),
            static_cast<uint8_t *>(dst), scale_f32, zp_out, M, K, G);
      } else {
        dynamic_per_group_quant_f16_u8_native(
            static_cast<const uint16_t *>(src),
            static_cast<uint8_t *>(dst), scale_f32, zp_out, M, K, G);
      }
      dispatched = true;
    }
  }

  if (dispatched && scale_needs_narrow) {
    uint16_t *out = static_cast<uint16_t *>(params.quant_params.scale.buff);
    if (scale_dt == data_type_t::bf16) {
      for (int64_t i = 0; i < scale_nelems; ++i)
        out[i] = float_to_bf16(scale_f32[i]);
    } else {  // f16: floor-then-narrow via narrow_f32_scale_to_f16
      for (int64_t i = 0; i < scale_nelems; ++i)
        out[i] = common::narrow_f32_scale_to_f16(scale_f32[i]);
    }
  }

  return dispatched;
}

bool dispatch_group_dynamic_per_token(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<void *> &scale,
    const group_dynamic_quant_params_t &params) {
  if (params.dst_dtype != data_type_t::s8) return false;
  if (params.scale_dtype != data_type_t::f32 &&
      params.scale_dtype != data_type_t::bf16 &&
      params.scale_dtype != data_type_t::f16) {
    return false;
  }
  if (params.src_dtype != data_type_t::bf16 &&
      params.src_dtype != data_type_t::f32 &&
      params.src_dtype != data_type_t::f16) {
    return false;
  }

  const size_t num_ops = M.size();
  const bool scale_needs_narrow = (params.scale_dtype == data_type_t::bf16 ||
                                   params.scale_dtype == data_type_t::f16);
  int64_t total_rows = 0;
  for (int m : M) total_rows += std::max(0, m);

  std::vector<float *> scale_f32(num_ops, nullptr);
  std::vector<float> scale_tmp;
  if (scale_needs_narrow) {
    scale_tmp.resize(static_cast<size_t>(total_rows));
    size_t off = 0;
    for (size_t i = 0; i < num_ops; ++i) {
      scale_f32[i] = scale_tmp.data() + off;
      off += static_cast<size_t>(std::max(0, M[i]));
    }
  } else {
    for (size_t i = 0; i < num_ops; ++i) {
      scale_f32[i] = static_cast<float *>(scale[i]);
    }
  }

  if (params.src_dtype == data_type_t::bf16) {
    dynamic_per_token_group_quant_bf16_s8_native(
        src, M, K, lda, dst, dst_lda, scale_f32, params.num_threads);
  } else if (params.src_dtype == data_type_t::f32) {
    dynamic_per_token_group_quant_f32_s8_native(
        src, M, K, lda, dst, dst_lda, scale_f32, params.num_threads);
  } else {
    dynamic_per_token_group_quant_f16_s8_native(
        src, M, K, lda, dst, dst_lda, scale_f32, params.num_threads);
  }

  if (scale_needs_narrow) {
    if (params.scale_dtype == data_type_t::bf16) {
      for (size_t i = 0; i < num_ops; ++i) {
        uint16_t *out = static_cast<uint16_t *>(scale[i]);
        for (int64_t m = 0; m < M[i]; ++m) {
          out[m] = float_to_bf16(scale_f32[i][m]);
        }
      }
    } else {  // f16: floor-then-narrow via narrow_f32_scale_to_f16
      for (size_t i = 0; i < num_ops; ++i) {
        uint16_t *out = static_cast<uint16_t *>(scale[i]);
        for (int64_t m = 0; m < M[i]; ++m) {
          out[m] = common::narrow_f32_scale_to_f16(scale_f32[i][m]);
        }
      }
    }
  }

  return true;
}

bool dispatch_group_dynamic_per_group(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<void *> &scale,
    const group_dynamic_quant_params_t &params) {
  if (params.dst_dtype != data_type_t::s8) return false;
  if (params.scale_dtype != data_type_t::f32 &&
      params.scale_dtype != data_type_t::bf16) {
    return false;
  }
  if (params.src_dtype != data_type_t::bf16 &&
      params.src_dtype != data_type_t::f32) {
    return false;
  }
  const int64_t G = params.num_groups;
  if (G <= 1) return false;

  const size_t num_ops = M.size();
  const bool scale_is_bf16 = (params.scale_dtype == data_type_t::bf16);
  int64_t total_scales = 0;
  for (int m : M) total_scales += static_cast<int64_t>(std::max(0, m)) * G;

  // For bf16 scale output the native kernels write f32 into a scratch
  // buffer (one {M_i, G} block per expert), then we narrow to bf16.
  std::vector<float *> scale_f32(num_ops, nullptr);
  std::vector<float> scale_tmp;
  if (scale_is_bf16) {
    scale_tmp.resize(static_cast<size_t>(total_scales));
    size_t off = 0;
    for (size_t i = 0; i < num_ops; ++i) {
      scale_f32[i] = scale_tmp.data() + off;
      off += static_cast<size_t>(std::max(0, M[i])) * G;
    }
  } else {
    for (size_t i = 0; i < num_ops; ++i) {
      scale_f32[i] = static_cast<float *>(scale[i]);
    }
  }

  if (params.src_dtype == data_type_t::bf16) {
    dynamic_per_group_group_quant_bf16_s8_native(
        src, M, K, lda, dst, dst_lda, scale_f32, G, params.num_threads);
  } else {
    dynamic_per_group_group_quant_f32_s8_native(
        src, M, K, lda, dst, dst_lda, scale_f32, G, params.num_threads);
  }

  if (scale_is_bf16) {
    for (size_t i = 0; i < num_ops; ++i) {
      uint16_t *out = static_cast<uint16_t *>(scale[i]);
      const int64_t n = static_cast<int64_t>(std::max(0, M[i])) * G;
      for (int64_t e = 0; e < n; ++e) {
        out[e] = float_to_bf16(scale_f32[i][e]);
      }
    }
  }

  return true;
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
