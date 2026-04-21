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

#include "rmsnorm_avx512_kernel.hpp"
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
// Plain RMS Norm — single row, processes one [1, norm_size] slice.
//
//   rms    = sqrt( (1/N) * Σ input[i]² + eps )
//   out[i] = gamma[i] * input[i] / rms
//
// Register map (9 of 32 ZMM used at peak):
//   zmm0 - zmm3  : 4 FP32 accumulators (acc0–acc3), independent chains
//   zmm4 - zmm7  : 4 data loads (s0–s3, 16 FP32 lanes each)
//   zmm8          : inv_rms broadcast (pass 2)
//   zmm9 - zmm12 : gamma × inv_rms pre-multiplied (g0–g3, pass 2 only)
//   zmm13-zmm31  : free
//
// Pass 1 — sum-of-squares (4×16 = 64 elements/iter):
//   FP32 path: 4 vmovups(src) + 4 vfmadd231ps = 8 instructions.
//   BF16 path: 4 (vmovdqu16 + vpmovzxwd + vpslld) + 4 vfmadd231ps = 16 instr.
//   FMA-bound (Zen4, 2 FMA ports): 4 fmadd / 2 = 2 cycles/iter (best case).
//   128 FP32 FLOPs / 2 cycles = 64 FLOPs/cycle.
//   Cleanup: 1×16 loop + masked tail (no perf impact, runs once).
//
// Pass 2 — normalize + optional gamma (4×16 = 64 elements/iter):
//   With scale: 4 vmovups(γ) + 4 vmulps(γ×inv_rms) + 4 load + 4 vmulps
//               + 4 store = 20 instructions.
//   Without:    4 load + 4 vmulps + 4 store = 12 instructions.
//   Pre-multiplying gamma[i] * inv_rms into g0–g3 saves 4 vmulps/iter.
//   Load-bound: input re-read hits L1 (warm from pass 1).
// =============================================================================

static inline void rms_norm_row_avx512(
  const void  *__restrict__ in_row,
  void        *__restrict__ out_row,
  const void  *__restrict__ gamma,
  uint64_t norm_size,
  float    inv_n,
  float    epsilon,
  bool     use_scale,
  bool     src_bf16,
  bool     dst_bf16,
  bool     gamma_bf16
) {
  const size_t src_sz = elem_size(src_bf16);
  const size_t dst_sz = elem_size(dst_bf16);
  const size_t g_sz   = elem_size(gamma_bf16);

  // ---- Pass 1: sum-of-squares ----
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();

  uint64_t i = 0;
  const uint64_t vec64 = norm_size & ~63ULL;

  for (; i < vec64; i += 64) {
    const void *p = static_cast<const char *>(in_row) + i * src_sz;
    __m512 s0 = load16(p,                    src_bf16);
    __m512 s1 = load16((const char *)p + 16*src_sz, src_bf16);
    __m512 s2 = load16((const char *)p + 32*src_sz, src_bf16);
    __m512 s3 = load16((const char *)p + 48*src_sz, src_bf16);

    acc0 = _mm512_fmadd_ps(s0, s0, acc0);
    acc1 = _mm512_fmadd_ps(s1, s1, acc1);
    acc2 = _mm512_fmadd_ps(s2, s2, acc2);
    acc3 = _mm512_fmadd_ps(s3, s3, acc3);
  }
  for (; i + 15 < norm_size; i += 16) {
    __m512 s0 = load16(static_cast<const char *>(in_row) + i*src_sz, src_bf16);
    acc0 = _mm512_fmadd_ps(s0, s0, acc0);
  }
  if (i < norm_size) {
    __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
    __m512 s0 = load16_mask(static_cast<const char *>(in_row) + i*src_sz, mask,
                            src_bf16);
    acc0 = _mm512_fmadd_ps(s0, s0, acc0);
  }

  acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
  float inv_rms = 1.0f / std::sqrt(_mm512_reduce_add_ps(acc0) * inv_n + epsilon);
  __m512 inv_rms_v = _mm512_set1_ps(inv_rms);

  // ---- Pass 2: normalize + optional gamma ----
  i = 0;
  if (use_scale) {
    for (; i < vec64; i += 64) {
      const void *np = static_cast<const char *>(in_row) + i*src_sz;
      const void *gp = static_cast<const char *>(gamma)  + i*g_sz;
      void *op       = static_cast<char *>(out_row)      + i*dst_sz;
      __m512 g0 = _mm512_mul_ps(load16(gp,                       gamma_bf16),
                                inv_rms_v);
      __m512 g1 = _mm512_mul_ps(load16((const char *)gp+16*g_sz, gamma_bf16),
                                inv_rms_v);
      __m512 g2 = _mm512_mul_ps(load16((const char *)gp+32*g_sz, gamma_bf16),
                                inv_rms_v);
      __m512 g3 = _mm512_mul_ps(load16((const char *)gp+48*g_sz, gamma_bf16),
                                inv_rms_v);
      store16(op,                     _mm512_mul_ps(load16(np,
              src_bf16), g0), dst_bf16);
      store16((char *)op + 16*dst_sz,
              _mm512_mul_ps(load16((const char *)np+16*src_sz, src_bf16), g1), dst_bf16);
      store16((char *)op + 32*dst_sz,
              _mm512_mul_ps(load16((const char *)np+32*src_sz, src_bf16), g2), dst_bf16);
      store16((char *)op + 48*dst_sz,
              _mm512_mul_ps(load16((const char *)np+48*src_sz, src_bf16), g3), dst_bf16);
    }
    for (; i + 15 < norm_size; i += 16) {
      __m512 g0 = _mm512_mul_ps(
                    load16(static_cast<const char *>(gamma)+i*g_sz, gamma_bf16), inv_rms_v);
      store16(static_cast<char *>(out_row) + i*dst_sz,
              _mm512_mul_ps(load16(static_cast<const char *>(in_row)+i*src_sz, src_bf16),
                            g0), dst_bf16);
    }
    if (i < norm_size) {
      __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
      __m512 g0 = _mm512_mul_ps(
                    load16_mask(static_cast<const char *>(gamma)+i*g_sz, mask, gamma_bf16),
                    inv_rms_v);
      store16_mask(static_cast<char *>(out_row) + i*dst_sz,
                   _mm512_mul_ps(load16_mask(static_cast<const char *>(in_row)+i*src_sz, mask,
                                             src_bf16), g0),
                   mask, dst_bf16);
    }
  }
  else {
    for (; i < vec64; i += 64) {
      const void *np = static_cast<const char *>(in_row) + i*src_sz;
      void *op       = static_cast<char *>(out_row)      + i*dst_sz;
      store16(op,                     _mm512_mul_ps(load16(np,
              src_bf16), inv_rms_v), dst_bf16);
      store16((char *)op + 16*dst_sz,
              _mm512_mul_ps(load16((const char *)np+16*src_sz, src_bf16), inv_rms_v),
              dst_bf16);
      store16((char *)op + 32*dst_sz,
              _mm512_mul_ps(load16((const char *)np+32*src_sz, src_bf16), inv_rms_v),
              dst_bf16);
      store16((char *)op + 48*dst_sz,
              _mm512_mul_ps(load16((const char *)np+48*src_sz, src_bf16), inv_rms_v),
              dst_bf16);
    }
    for (; i + 15 < norm_size; i += 16) {
      store16(static_cast<char *>(out_row) + i*dst_sz,
              _mm512_mul_ps(load16(static_cast<const char *>(in_row)+i*src_sz, src_bf16),
                            inv_rms_v), dst_bf16);
    }
    if (i < norm_size) {
      __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
      store16_mask(static_cast<char *>(out_row) + i*dst_sz,
                   _mm512_mul_ps(load16_mask(static_cast<const char *>(in_row)+i*src_sz, mask,
                                             src_bf16), inv_rms_v),
                   mask, dst_bf16);
    }
  }
}

// =============================================================================
// Fused Add + RMS Norm — single row, processes one [1, norm_size] slice.
//
//   residual[i] += input[i]           (in-place update)
//   rms    = sqrt( (1/N) * Σ residual[i]² + eps )
//   out[i] = gamma[i] * residual[i] / rms
//
// Register map (17 of 32 ZMM used at peak in pass 1):
//   zmm0  - zmm3   : 4 FP32 accumulators (acc0–acc3), independent chains
//   zmm4  - zmm7   : 4 input loads (in0–in3)
//   zmm8  - zmm11  : 4 residual loads (r0–r3)
//   zmm12 - zmm15  : 4 sums (s0–s3 = r + in), reused for store + FMA
//   zmm16           : inv_rms broadcast (pass 2)
//   zmm17 - zmm20  : gamma × inv_rms pre-multiplied (g0–g3, pass 2)
//   zmm21 - zmm31  : free (15 registers headroom, no spill risk)
//
// Pass 1 — fused add + sum-of-squares (4×16 = 64 elements/iter):
//   FP32 path: 4 vmovups(in) + 4 vmovups(res) + 4 vaddps + 4 vmovups(store)
//              + 4 vfmadd231ps = 20 instructions.
//   BF16 path: 4 bf16_load(in) + 4 bf16_load(res) + 4 vaddps
//              + 4 bf16_store(res) + 4 vfmadd231ps = 20 high-level ops,
//              ~36 µops (bf16 load = 3 µops, bf16 store = 4 µops each).
//   FMA-bound (Zen4): 4 fmadd / 2 ports = 2 cycles (best case).
//   Store-bound: 4 stores / 1 store port = 4 cycles (likely bottleneck).
//   Effective: ~4–5 cycles/iter.  256 FP32 FLOPs / 5 = 51 FLOPs/cycle.
//   Cleanup: 1×16 loop + masked tail.
//
// Pass 2 — normalize updated residual (4×16 = 64 elements/iter):
//   Identical to plain RMS pass 2 — re-reads residual from L1 (warm from
//   pass 1 stores). With scale: 20 instr/iter. Without: 12 instr/iter.
//   BF16 residual: pass 2 re-reads the rounded BF16 value written in pass 1,
//   matching the reference implementation's precision semantics.
// =============================================================================

static inline void fused_add_rms_row_avx512(
  const void  *__restrict__ in_row,
  void        *__restrict__ out_row,
  void        *__restrict__ res_row,
  const void  *__restrict__ gamma,
  uint64_t norm_size,
  float    inv_n,
  float    epsilon,
  bool     use_scale,
  bool     src_bf16,
  bool     dst_bf16,
  bool     gamma_bf16
) {
  const size_t src_sz = elem_size(src_bf16);
  const size_t dst_sz = elem_size(dst_bf16);
  const size_t g_sz   = elem_size(gamma_bf16);

  // ---- Pass 1: residual += input, accumulate sum-of-squares ----
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();

  uint64_t i = 0;
  const uint64_t vec64 = norm_size & ~63ULL;

  for (; i < vec64; i += 64) {
    const void *ip = static_cast<const char *>(in_row)  + i*src_sz;
    void       *rp = static_cast<char *>(res_row)       + i*src_sz;

    __m512 s0 = _mm512_add_ps(load16(rp,                      src_bf16),
                              load16(ip,                      src_bf16));
    __m512 s1 = _mm512_add_ps(load16((char *)rp + 16*src_sz,   src_bf16),
                              load16((const char *)ip+16*src_sz, src_bf16));
    __m512 s2 = _mm512_add_ps(load16((char *)rp + 32*src_sz,   src_bf16),
                              load16((const char *)ip+32*src_sz, src_bf16));
    __m512 s3 = _mm512_add_ps(load16((char *)rp + 48*src_sz,   src_bf16),
                              load16((const char *)ip+48*src_sz, src_bf16));

    store16(rp,                     s0, src_bf16);
    store16((char *)rp + 16*src_sz,  s1, src_bf16);
    store16((char *)rp + 32*src_sz,  s2, src_bf16);
    store16((char *)rp + 48*src_sz,  s3, src_bf16);

    acc0 = _mm512_fmadd_ps(s0, s0, acc0);
    acc1 = _mm512_fmadd_ps(s1, s1, acc1);
    acc2 = _mm512_fmadd_ps(s2, s2, acc2);
    acc3 = _mm512_fmadd_ps(s3, s3, acc3);
  }
  for (; i + 15 < norm_size; i += 16) {
    __m512 s0 = _mm512_add_ps(
                  load16(static_cast<char *>(res_row) + i*src_sz, src_bf16),
                  load16(static_cast<const char *>(in_row) + i*src_sz, src_bf16));
    store16(static_cast<char *>(res_row) + i*src_sz, s0, src_bf16);
    acc0 = _mm512_fmadd_ps(s0, s0, acc0);
  }
  if (i < norm_size) {
    __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
    __m512 s0 = _mm512_add_ps(
                  load16_mask(static_cast<char *>(res_row) + i*src_sz, mask, src_bf16),
                  load16_mask(static_cast<const char *>(in_row) + i*src_sz, mask, src_bf16));
    store16_mask(static_cast<char *>(res_row) + i*src_sz, s0, mask, src_bf16);
    acc0 = _mm512_fmadd_ps(s0, s0, acc0);
  }

  acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
  float inv_rms = 1.0f / std::sqrt(_mm512_reduce_add_ps(acc0) * inv_n + epsilon);
  __m512 inv_rms_v = _mm512_set1_ps(inv_rms);

  // ---- Pass 2: normalize updated residual + optional gamma ----
  i = 0;
  if (use_scale) {
    for (; i < vec64; i += 64) {
      const void *np = static_cast<char *>(res_row)  + i*src_sz;
      const void *gp = static_cast<const char *>(gamma) + i*g_sz;
      void *op       = static_cast<char *>(out_row)  + i*dst_sz;
      __m512 g0 = _mm512_mul_ps(load16(gp,                       gamma_bf16),
                                inv_rms_v);
      __m512 g1 = _mm512_mul_ps(load16((const char *)gp+16*g_sz, gamma_bf16),
                                inv_rms_v);
      __m512 g2 = _mm512_mul_ps(load16((const char *)gp+32*g_sz, gamma_bf16),
                                inv_rms_v);
      __m512 g3 = _mm512_mul_ps(load16((const char *)gp+48*g_sz, gamma_bf16),
                                inv_rms_v);
      store16(op,                     _mm512_mul_ps(load16(np,
              src_bf16), g0), dst_bf16);
      store16((char *)op + 16*dst_sz,
              _mm512_mul_ps(load16((const char *)np+16*src_sz, src_bf16), g1), dst_bf16);
      store16((char *)op + 32*dst_sz,
              _mm512_mul_ps(load16((const char *)np+32*src_sz, src_bf16), g2), dst_bf16);
      store16((char *)op + 48*dst_sz,
              _mm512_mul_ps(load16((const char *)np+48*src_sz, src_bf16), g3), dst_bf16);
    }
    for (; i + 15 < norm_size; i += 16) {
      __m512 g0 = _mm512_mul_ps(
                    load16(static_cast<const char *>(gamma)+i*g_sz, gamma_bf16), inv_rms_v);
      store16(static_cast<char *>(out_row) + i*dst_sz,
              _mm512_mul_ps(load16(static_cast<char *>(res_row)+i*src_sz, src_bf16), g0),
              dst_bf16);
    }
    if (i < norm_size) {
      __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
      __m512 g0 = _mm512_mul_ps(
                    load16_mask(static_cast<const char *>(gamma)+i*g_sz, mask, gamma_bf16),
                    inv_rms_v);
      store16_mask(static_cast<char *>(out_row) + i*dst_sz,
                   _mm512_mul_ps(load16_mask(static_cast<char *>(res_row)+i*src_sz, mask,
                                             src_bf16), g0),
                   mask, dst_bf16);
    }
  }
  else {
    for (; i < vec64; i += 64) {
      const void *np = static_cast<char *>(res_row)  + i*src_sz;
      void *op       = static_cast<char *>(out_row)  + i*dst_sz;
      store16(op,                     _mm512_mul_ps(load16(np,
              src_bf16), inv_rms_v), dst_bf16);
      store16((char *)op + 16*dst_sz,
              _mm512_mul_ps(load16((const char *)np+16*src_sz, src_bf16), inv_rms_v),
              dst_bf16);
      store16((char *)op + 32*dst_sz,
              _mm512_mul_ps(load16((const char *)np+32*src_sz, src_bf16), inv_rms_v),
              dst_bf16);
      store16((char *)op + 48*dst_sz,
              _mm512_mul_ps(load16((const char *)np+48*src_sz, src_bf16), inv_rms_v),
              dst_bf16);
    }
    for (; i + 15 < norm_size; i += 16) {
      store16(static_cast<char *>(out_row) + i*dst_sz,
              _mm512_mul_ps(load16(static_cast<char *>(res_row)+i*src_sz, src_bf16),
                            inv_rms_v), dst_bf16);
    }
    if (i < norm_size) {
      __mmask16 mask = (__mmask16)((1U << (norm_size - i)) - 1);
      store16_mask(static_cast<char *>(out_row) + i*dst_sz,
                   _mm512_mul_ps(load16_mask(static_cast<char *>(res_row)+i*src_sz, mask,
                                             src_bf16), inv_rms_v),
                   mask, dst_bf16);
    }
  }
}

// =====================================================================
// Dispatches RMS_NORM and FUSED_ADD_RMS_NORM
// =====================================================================

status_t rms_norm_avx512(
  const void *input,
  void       *output,
  void       *residual,
  const void *gamma,
  norm_params &params
) {
  const float inv_n = 1.0f / static_cast<float>(params.norm_size);
  const bool src_bf16   = (params.src_dt == data_type_t::bf16);
  const bool dst_bf16   = (params.dst_dt == data_type_t::bf16);
  const bool gamma_bf16 = (params.gamma_dt == data_type_t::bf16);
  const size_t src_sz = src_bf16 ? 2 : 4;
  const size_t dst_sz = dst_bf16 ? 2 : 4;
  const uint64_t N = params.norm_size;
  const int64_t batch = static_cast<int64_t>(params.batch);
  const bool is_fused = (params.norm_type == norm_type_t::FUSED_ADD_RMS_NORM &&
                         residual);

  auto row_loop = [&](int64_t b) {
    if (!is_fused) {
      rms_norm_row_avx512(
        static_cast<const char *>(input)  + b * N * src_sz,
        static_cast<char *>(output)       + b * N * dst_sz,
        gamma, N, inv_n, params.epsilon,
        params.use_scale, src_bf16, dst_bf16, gamma_bf16);
    }
    else {
      fused_add_rms_row_avx512(
        static_cast<const char *>(input)    + b * N * src_sz,
        static_cast<char *>(output)         + b * N * dst_sz,
        static_cast<char *>(residual)       + b * N * src_sz,
        gamma, N, inv_n, params.epsilon,
        params.use_scale, src_bf16, dst_bf16, gamma_bf16);
    }
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
