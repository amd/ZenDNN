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

#include "lowoha_operators/reorder/reorder_data_type/scalar_impl/scalar_kernels.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"

#include <cstring>
#include <cmath>
#include <limits>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace reorder {

/** Scalar BF16-to-F32 fallback for tail elements that don't fill a vector. */
static inline float bf16_scalar_to_f32(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float r;
  std::memcpy(&r, &bits, sizeof(float));
  return r;
}

static inline void compute_symmetric_scale_from_absmax(float absmax,
                                                        float &scale) {
  if (absmax < 1e-10f) absmax = 1e-10f;
  scale = absmax / 127.0f;
  if (scale < 1e-10f) scale = 1e-10f;
}

static inline void compute_asymmetric_scale_zp(float min_val, float max_val,
                                                float &scale, int32_t &zp) {
  if (min_val == std::numeric_limits<float>::max() &&
      max_val == std::numeric_limits<float>::lowest()) {
    min_val = 0.0f;
    max_val = 0.0f;
  }
  if (max_val <= min_val) max_val = min_val + 1.0f;
  scale = (max_val - min_val) / 255.0f;
  if (scale < 1e-10f) scale = 1e-10f;

  double zp_d = std::round(static_cast<double>(-min_val) /
                            static_cast<double>(scale));
  constexpr double lo = static_cast<double>(std::numeric_limits<int32_t>::min());
  constexpr double hi = static_cast<double>(std::numeric_limits<int32_t>::max());
  if (zp_d < lo) zp = std::numeric_limits<int32_t>::min();
  else if (zp_d > hi) zp = std::numeric_limits<int32_t>::max();
  else zp = static_cast<int32_t>(zp_d);
}
//==============================================================================
//
// Scalar C++ implementations of the same fused per-row logic as the native
// AVX-512 kernels.  Each row is processed in two back-to-back passes
// (statistics + quantize) to keep row data in L1 cache, identical to the
// native path but without SIMD intrinsics.
//
// Used when algo == reference for correctness testing or platforms without
// AVX-512.
//==============================================================================

// --- BF16 -> S8 Symmetric (scalar fused) ---

void dynamic_per_token_quant_bf16_s8_ref(const uint16_t *src, int8_t *dst,
                                          float *scales,
                                          int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;
    int8_t         *row_dst = dst + m * N;

    float absmax = 0.0f;
    for (int64_t j = 0; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      if (std::isfinite(v))
        absmax = std::max(absmax, std::abs(v));
    }

    float scale;
    compute_symmetric_scale_from_absmax(absmax, scale);
    scales[m] = scale;

    for (int64_t j = 0; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      int32_t q = static_cast<int32_t>(std::nearbyint(v / scale));
      q = std::max(-128, std::min(127, q));
      row_dst[j] = static_cast<int8_t>(q);
    }
  }
}

// --- F32 -> S8 Symmetric (scalar fused) ---

void dynamic_per_token_quant_f32_s8_ref(const float *src, int8_t *dst,
                                         float *scales,
                                         int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float *row_src = src + m * N;
    int8_t      *row_dst = dst + m * N;

    float absmax = 0.0f;
    for (int64_t j = 0; j < N; ++j) {
      if (std::isfinite(row_src[j]))
        absmax = std::max(absmax, std::abs(row_src[j]));
    }

    float scale;
    compute_symmetric_scale_from_absmax(absmax, scale);
    scales[m] = scale;

    for (int64_t j = 0; j < N; ++j) {
      int32_t q = static_cast<int32_t>(std::nearbyint(row_src[j] / scale));
      q = std::max(-128, std::min(127, q));
      row_dst[j] = static_cast<int8_t>(q);
    }
  }
}

// --- BF16 -> U8 Asymmetric (scalar fused) ---

void dynamic_per_token_quant_bf16_u8_ref(const uint16_t *src, uint8_t *dst,
                                          float *scales, int32_t *zps,
                                          int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;
    uint8_t        *row_dst = dst + m * N;

    float row_min = std::numeric_limits<float>::max();
    float row_max = std::numeric_limits<float>::lowest();
    for (int64_t j = 0; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      if (std::isfinite(v)) {
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
    }

    float scale;
    int32_t zp;
    compute_asymmetric_scale_zp(row_min, row_max, scale, zp);
    scales[m] = scale;
    zps[m]    = zp;

    for (int64_t j = 0; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      int32_t q = static_cast<int32_t>(std::nearbyint(v / scale)) + zp;
      q = std::max(0, std::min(255, q));
      row_dst[j] = static_cast<uint8_t>(q);
    }
  }
}

// --- F32 -> U8 Asymmetric (scalar fused) ---

void dynamic_per_token_quant_f32_u8_ref(const float *src, uint8_t *dst,
                                         float *scales, int32_t *zps,
                                         int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float *row_src = src + m * N;
    uint8_t     *row_dst = dst + m * N;

    float row_min = std::numeric_limits<float>::max();
    float row_max = std::numeric_limits<float>::lowest();
    for (int64_t j = 0; j < N; ++j) {
      if (std::isfinite(row_src[j])) {
        row_min = std::min(row_min, row_src[j]);
        row_max = std::max(row_max, row_src[j]);
      }
    }

    float scale;
    int32_t zp;
    compute_asymmetric_scale_zp(row_min, row_max, scale, zp);
    scales[m] = scale;
    zps[m]    = zp;

    for (int64_t j = 0; j < N; ++j) {
      int32_t q = static_cast<int32_t>(std::nearbyint(row_src[j] / scale)) + zp;
      q = std::max(0, std::min(255, q));
      row_dst[j] = static_cast<uint8_t>(q);
    }
  }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
