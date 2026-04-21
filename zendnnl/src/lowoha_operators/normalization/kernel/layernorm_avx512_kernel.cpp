/********************************************************************************
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

#include "layernorm_avx512_kernel.hpp"
#include "avx512_utils.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace normalization {

#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512bf16,fma")

using avx512::load16;
using avx512::load16_mask;
using avx512::store16;
using avx512::store16_mask;
using avx512::elem_size;

// =============================================================================
// Layer Norm — single row, processes one [1, norm_size] slice.
//
//   mean    = (1/N) * Σ x[i]
//   var     = (1/N) * Σ x[i]² - mean²          (clamped to ≥ 0)
//   inv_std = 1 / sqrt(var + eps)
//   y[i]   = gamma[i] * (x[i] - mean) * inv_std + beta[i]
//
// Two passes over the row:
//
// Pass 1 — simultaneous sum and sum-of-squares (4×16 = 64 elements/iter):
//   Register map (12 of 32 ZMM used at peak):
//     zmm0  - zmm3  : 4 sum accumulators (sum0–sum3)
//     zmm4  - zmm7  : 4 sum_sq accumulators (sq0–sq3)
//     zmm8  - zmm11 : 4 data loads (s0–s3, 16 FP32 lanes each)
//
//   FP32 path: 4 vmovups(src) + 4 vaddps(sum) + 4 vfmadd231ps(sq) = 12 instr.
//   FMA-bound (Zen4, 2 FMA ports): 4 fmadd / 2 = 2 cycles/iter (best case).
//   Cleanup: 1×16 loop + masked tail.
//
// Pass 2 — normalize + optional gamma/beta (4×16 = 64 elements/iter):
//   With scale+shift:  4 load(γ) + 4 vmulps(γ×inv_std) + 4 load(src) +
//                       4 vsubps(mean) + 4 load(β) + 4 vfmadd(d*g+β) +
//                       4 store = 28 instructions.
//   Pre-multiplying gamma[i] * inv_std saves a vmulps per element.
//   Load-bound: input re-read hits L1 (warm from pass 1).
// =============================================================================

static inline void layer_norm_row_avx512(
  const void  *__restrict__ in_row,
  void        *__restrict__ out_row,
  const void  *__restrict__ gamma,
  const void  *__restrict__ beta,
  uint64_t norm_size,
  float    inv_n,
  float    epsilon,
  bool     use_scale,
  bool     use_shift,
  bool     src_bf16,
  bool     dst_bf16,
  bool     gamma_bf16,
  bool     beta_bf16
) {
  const size_t src_sz = elem_size(src_bf16);
  const size_t dst_sz = elem_size(dst_bf16);
  const size_t g_sz   = elem_size(gamma_bf16);
  const size_t b_sz   = elem_size(beta_bf16);

  // ---- Pass 1: simultaneous sum and sum-of-squares ----
  __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
  __m512 sum2 = _mm512_setzero_ps(), sum3 = _mm512_setzero_ps();
  __m512 sq0  = _mm512_setzero_ps(), sq1  = _mm512_setzero_ps();
  __m512 sq2  = _mm512_setzero_ps(), sq3  = _mm512_setzero_ps();

  uint64_t i = 0;
  const uint64_t vec64 = norm_size & ~63ULL;

  for (; i < vec64; i += 64) {
    const void *p = static_cast<const char *>(in_row) + i * src_sz;
    __m512 s0 = load16(p,                            src_bf16);
    __m512 s1 = load16((const char *)p + 16*src_sz,  src_bf16);
    __m512 s2 = load16((const char *)p + 32*src_sz,  src_bf16);
    __m512 s3 = load16((const char *)p + 48*src_sz,  src_bf16);

    sum0 = _mm512_add_ps(sum0, s0);
    sum1 = _mm512_add_ps(sum1, s1);
    sum2 = _mm512_add_ps(sum2, s2);
    sum3 = _mm512_add_ps(sum3, s3);

    sq0 = _mm512_fmadd_ps(s0, s0, sq0);
    sq1 = _mm512_fmadd_ps(s1, s1, sq1);
    sq2 = _mm512_fmadd_ps(s2, s2, sq2);
    sq3 = _mm512_fmadd_ps(s3, s3, sq3);
  }
  for (; i + 15 < norm_size; i += 16) {
    __m512 s0 = load16(static_cast<const char *>(in_row) + i*src_sz, src_bf16);
    sum0 = _mm512_add_ps(sum0, s0);
    sq0  = _mm512_fmadd_ps(s0, s0, sq0);
  }
  if (i < norm_size) {
    __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
    __m512 s0 = load16_mask(static_cast<const char *>(in_row) + i*src_sz,
                            mask, src_bf16);
    sum0 = _mm512_add_ps(sum0, s0);
    sq0  = _mm512_fmadd_ps(s0, s0, sq0);
  }

  // Horizontal reduce
  sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
  sq0  = _mm512_add_ps(_mm512_add_ps(sq0,  sq1),  _mm512_add_ps(sq2,  sq3));

  float total_sum = _mm512_reduce_add_ps(sum0);
  float total_sq  = _mm512_reduce_add_ps(sq0);

  float mean    = total_sum * inv_n;
  float var     = std::max(0.0f, total_sq * inv_n - mean * mean);
  float inv_std = 1.0f / std::sqrt(var + epsilon);

  __m512 mean_v    = _mm512_set1_ps(mean);
  __m512 inv_std_v = _mm512_set1_ps(inv_std);

  // ---- Pass 2: normalize + optional gamma/beta ----
  i = 0;
  if (use_scale) {
    // y = (x - mean) * (gamma * inv_std) [+ beta]
    for (; i < vec64; i += 64) {
      const void *np = static_cast<const char *>(in_row) + i*src_sz;
      const void *gp = static_cast<const char *>(gamma)  + i*g_sz;
      void       *op = static_cast<char *>(out_row)       + i*dst_sz;

      __m512 g0 = _mm512_mul_ps(load16(gp,                           gamma_bf16),
                                inv_std_v);
      __m512 g1 = _mm512_mul_ps(load16((const char *)gp + 16*g_sz,   gamma_bf16),
                                inv_std_v);
      __m512 g2 = _mm512_mul_ps(load16((const char *)gp + 32*g_sz,   gamma_bf16),
                                inv_std_v);
      __m512 g3 = _mm512_mul_ps(load16((const char *)gp + 48*g_sz,   gamma_bf16),
                                inv_std_v);

      __m512 d0 = _mm512_sub_ps(load16(np,                           src_bf16),
                                mean_v);
      __m512 d1 = _mm512_sub_ps(load16((const char *)np + 16*src_sz, src_bf16),
                                mean_v);
      __m512 d2 = _mm512_sub_ps(load16((const char *)np + 32*src_sz, src_bf16),
                                mean_v);
      __m512 d3 = _mm512_sub_ps(load16((const char *)np + 48*src_sz, src_bf16),
                                mean_v);

      if (use_shift) {
        const void *bp = static_cast<const char *>(beta) + i*b_sz;
        __m512 b0 = load16(bp,                           beta_bf16);
        __m512 b1 = load16((const char *)bp + 16*b_sz,   beta_bf16);
        __m512 b2 = load16((const char *)bp + 32*b_sz,   beta_bf16);
        __m512 b3 = load16((const char *)bp + 48*b_sz,   beta_bf16);
        d0 = _mm512_fmadd_ps(d0, g0, b0);
        d1 = _mm512_fmadd_ps(d1, g1, b1);
        d2 = _mm512_fmadd_ps(d2, g2, b2);
        d3 = _mm512_fmadd_ps(d3, g3, b3);
      }
      else {
        d0 = _mm512_mul_ps(d0, g0);
        d1 = _mm512_mul_ps(d1, g1);
        d2 = _mm512_mul_ps(d2, g2);
        d3 = _mm512_mul_ps(d3, g3);
      }

      store16(op,                           d0, dst_bf16);
      store16((char *)op + 16*dst_sz,       d1, dst_bf16);
      store16((char *)op + 32*dst_sz,       d2, dst_bf16);
      store16((char *)op + 48*dst_sz,       d3, dst_bf16);
    }
    for (; i + 15 < norm_size; i += 16) {
      __m512 g0 = _mm512_mul_ps(
                    load16(static_cast<const char *>(gamma) + i*g_sz, gamma_bf16),
                    inv_std_v);
      __m512 d0 = _mm512_sub_ps(
                    load16(static_cast<const char *>(in_row) + i*src_sz, src_bf16),
                    mean_v);
      if (use_shift) {
        __m512 b0 = load16(static_cast<const char *>(beta) + i*b_sz, beta_bf16);
        d0 = _mm512_fmadd_ps(d0, g0, b0);
      }
      else {
        d0 = _mm512_mul_ps(d0, g0);
      }
      store16(static_cast<char *>(out_row) + i*dst_sz, d0, dst_bf16);
    }
    if (i < norm_size) {
      __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
      __m512 g0 = _mm512_mul_ps(
                    load16_mask(static_cast<const char *>(gamma) + i*g_sz, mask, gamma_bf16),
                    inv_std_v);
      __m512 d0 = _mm512_sub_ps(
                    load16_mask(static_cast<const char *>(in_row) + i*src_sz, mask, src_bf16),
                    mean_v);
      if (use_shift) {
        __m512 b0 = load16_mask(static_cast<const char *>(beta) + i*b_sz, mask,
                                beta_bf16);
        d0 = _mm512_fmadd_ps(d0, g0, b0);
      }
      else {
        d0 = _mm512_mul_ps(d0, g0);
      }
      store16_mask(static_cast<char *>(out_row) + i*dst_sz, d0, mask, dst_bf16);
    }
  }
  else {
    // y = (x - mean) * inv_std [+ beta]
    for (; i < vec64; i += 64) {
      const void *np = static_cast<const char *>(in_row) + i*src_sz;
      void       *op = static_cast<char *>(out_row)       + i*dst_sz;

      __m512 d0 = _mm512_mul_ps(_mm512_sub_ps(load16(np,
                                              src_bf16), mean_v), inv_std_v);
      __m512 d1 = _mm512_mul_ps(_mm512_sub_ps(load16((const char *)np + 16*src_sz,
                                              src_bf16), mean_v), inv_std_v);
      __m512 d2 = _mm512_mul_ps(_mm512_sub_ps(load16((const char *)np + 32*src_sz,
                                              src_bf16), mean_v), inv_std_v);
      __m512 d3 = _mm512_mul_ps(_mm512_sub_ps(load16((const char *)np + 48*src_sz,
                                              src_bf16), mean_v), inv_std_v);

      if (use_shift) {
        const void *bp = static_cast<const char *>(beta) + i*b_sz;
        d0 = _mm512_add_ps(d0, load16(bp,                           beta_bf16));
        d1 = _mm512_add_ps(d1, load16((const char *)bp + 16*b_sz,   beta_bf16));
        d2 = _mm512_add_ps(d2, load16((const char *)bp + 32*b_sz,   beta_bf16));
        d3 = _mm512_add_ps(d3, load16((const char *)bp + 48*b_sz,   beta_bf16));
      }

      store16(op,                           d0, dst_bf16);
      store16((char *)op + 16*dst_sz,       d1, dst_bf16);
      store16((char *)op + 32*dst_sz,       d2, dst_bf16);
      store16((char *)op + 48*dst_sz,       d3, dst_bf16);
    }
    for (; i + 15 < norm_size; i += 16) {
      __m512 d0 = _mm512_mul_ps(
                    _mm512_sub_ps(
                      load16(static_cast<const char *>(in_row) + i*src_sz, src_bf16),
                      mean_v),
                    inv_std_v);
      if (use_shift) {
        d0 = _mm512_add_ps(d0,
                           load16(static_cast<const char *>(beta) + i*b_sz, beta_bf16));
      }
      store16(static_cast<char *>(out_row) + i*dst_sz, d0, dst_bf16);
    }
    if (i < norm_size) {
      __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
      __m512 d0 = _mm512_mul_ps(
                    _mm512_sub_ps(
                      load16_mask(static_cast<const char *>(in_row) + i*src_sz, mask, src_bf16),
                      mean_v),
                    inv_std_v);
      if (use_shift) {
        d0 = _mm512_add_ps(d0,
                           load16_mask(static_cast<const char *>(beta) + i*b_sz, mask, beta_bf16));
      }
      store16_mask(static_cast<char *>(out_row) + i*dst_sz, d0, mask, dst_bf16);
    }
  }
}

// =====================================================================
// Entry point — dispatches LAYER_NORM over the batch with OpenMP.
// =====================================================================

status_t layer_norm_avx512(
  const void *input,
  void       *output,
  const void *gamma,
  const void *beta,
  norm_params &params
) {
  const float inv_n = 1.0f / static_cast<float>(params.norm_size);
  const bool src_bf16   = (params.src_dt   == data_type_t::bf16);
  const bool dst_bf16   = (params.dst_dt   == data_type_t::bf16);
  const bool gamma_bf16 = (params.gamma_dt == data_type_t::bf16);
  const bool beta_bf16  = (params.beta_dt  == data_type_t::bf16);
  const size_t src_sz = src_bf16 ? 2 : 4;
  const size_t dst_sz = dst_bf16 ? 2 : 4;
  const uint64_t N = params.norm_size;
  const int64_t batch = static_cast<int64_t>(params.batch);

  auto row_loop = [&](int64_t b) {
    layer_norm_row_avx512(
      static_cast<const char *>(input)  + b * N * src_sz,
      static_cast<char *>(output)       + b * N * dst_sz,
      gamma, beta, N, inv_n, params.epsilon,
      params.use_scale, params.use_shift,
      src_bf16, dst_bf16, gamma_bf16, beta_bf16);
  };

  if (batch <= 1) {
    for (int64_t b = 0; b < batch; ++b) {
      row_loop(b);
    }
  }
  else {
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
      row_loop(b);
    }
  }
  return status_t::success;
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl
