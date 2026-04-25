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
 *******************************************************************************/

/// MoE gated activation post-op for group matmul.
///
/// Three variants (matching vLLM CPU MoE):
///   silu_and_mul:    output = silu(gate) * up       (split layout)
///   gelu_and_mul:    output = gelu(gate) * up       (split layout)
///   swiglu_oai_mul:  output = swiglu_oai(gate, up)  (interleaved layout)
///


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include <immintrin.h>
#include <omp.h>

#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"
#include "group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::bfloat16_t;
using zendnnl::memory::data_type_t;

namespace {

constexpr float kSqrtHalf = 0.7071067811865476f;

// AVX-512F availability is a hardware property — cached once per process
// via C++11 magic-static.  Eliminates the ~5 ns per-call singleton lookup
// inside hot activation dispatches (called ~num_threads × num_ops times
// per MoE forward pass).
inline bool avx512f_available() {
  static const bool v =
      zendnnl::common::zendnnl_platform_info().get_avx512f_status();
  return v;
}

// ═══════════════════════════════════════════════════════════════════════
// Scalar helpers (always available)
// ═══════════════════════════════════════════════════════════════════════

inline float sigmoid_scalar(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

inline float silu_scalar(float x) {
  return x * sigmoid_scalar(x);
}

inline float gelu_scalar(float x) {
  return x * 0.5f * (1.0f + std::erf(x * kSqrtHalf));
}

// ═══════════════════════════════════════════════════════════════════════
// Scalar row kernels (fallback for non-AVX-512 platforms)
// ═══════════════════════════════════════════════════════════════════════

void silu_and_mul_row_scalar_f32(float *row, int dim) {
  float *gate = row, *up = row + dim;
  for (int n = 0; n < dim; ++n)
    gate[n] = silu_scalar(gate[n]) * up[n];
}

void silu_and_mul_row_scalar_bf16(bfloat16_t *row, int dim) {
  bfloat16_t *gate = row, *up = row + dim;
  for (int n = 0; n < dim; ++n) {
    float g = static_cast<float>(gate[n]);
    float u = static_cast<float>(up[n]);
    gate[n] = bfloat16_t(silu_scalar(g) * u);
  }
}

void gelu_and_mul_row_scalar_f32(float *row, int dim) {
  float *gate = row, *up = row + dim;
  for (int n = 0; n < dim; ++n)
    gate[n] = gelu_scalar(gate[n]) * up[n];
}

void gelu_and_mul_row_scalar_bf16(bfloat16_t *row, int dim) {
  bfloat16_t *gate = row, *up = row + dim;
  for (int n = 0; n < dim; ++n) {
    float g = static_cast<float>(gate[n]);
    float u = static_cast<float>(up[n]);
    gate[n] = bfloat16_t(gelu_scalar(g) * u);
  }
}

void swiglu_oai_mul_row_scalar_f32(float *row, int dim) {
  const float alpha = 1.702f;
  for (int n = 0; n < dim; ++n) {
    float g = std::max(-7.0f, std::min(row[2 * n], 7.0f));
    float u = std::max(-7.0f, std::min(row[2 * n + 1], 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    row[n] = (1.0f + u) * g * sig;
  }
}

void swiglu_oai_mul_row_scalar_bf16(bfloat16_t *row, int dim) {
  const float alpha = 1.702f;
  for (int n = 0; n < dim; ++n) {
    float g = std::max(-7.0f,
        std::min(static_cast<float>(row[2 * n]), 7.0f));
    float u = std::max(-7.0f,
        std::min(static_cast<float>(row[2 * n + 1]), 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    row[n] = bfloat16_t((1.0f + u) * g * sig);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// AVX-512 primitives (target-attributed, compiled unconditionally)
// ═══════════════════════════════════════════════════════════════════════

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 bf16x16_to_f32(__m256i bf16) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m256i f32_to_bf16x16(__m512 f32) {
  __m512i i32 = _mm512_castps_si512(f32);
  __m512i bias = _mm512_add_epi32(
      _mm512_set1_epi32(0x7FFF),
      _mm512_and_si512(_mm512_srli_epi32(i32, 16), _mm512_set1_epi32(1)));
  return _mm512_cvtepi32_epi16(_mm512_srli_epi32(
      _mm512_add_epi32(i32, bias), 16));
}

// Fast exp(-x): 5th-degree Cephes polynomial for 2^f, exponent clamped.
// Max relative error ~5e-5 — sufficient for sigmoid.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 fast_exp_neg_avx512(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(-1.4426950408889634f);

  __m512 t = _mm512_mul_ps(x, log2e);
  __m512 ti = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEG_INF);
  __m512 f = _mm512_sub_ps(t, ti);

  const __m512 c0 = _mm512_set1_ps(1.0f);
  const __m512 c1 = _mm512_set1_ps(0.6931472f);
  const __m512 c2 = _mm512_set1_ps(0.2402265f);
  const __m512 c3 = _mm512_set1_ps(0.0555042f);
  const __m512 c4 = _mm512_set1_ps(0.0096838f);
  const __m512 c5 = _mm512_set1_ps(0.0013364f);

  __m512 p = _mm512_fmadd_ps(c5, f, c4);
  p = _mm512_fmadd_ps(p, f, c3);
  p = _mm512_fmadd_ps(p, f, c2);
  p = _mm512_fmadd_ps(p, f, c1);
  p = _mm512_fmadd_ps(p, f, c0);

  __m512 ti_clamped = _mm512_max_ps(_mm512_min_ps(ti,
      _mm512_set1_ps(127.0f)), _mm512_set1_ps(-126.0f));
  __m512i ei = _mm512_cvtps_epi32(ti_clamped);
  __m512i exp_bits = _mm512_slli_epi32(
      _mm512_add_epi32(ei, _mm512_set1_epi32(127)), 23);
  __m512 pow2i = _mm512_castsi512_ps(exp_bits);

  __m512 result = _mm512_mul_ps(pow2i, p);
  return _mm512_max_ps(result, _mm512_setzero_ps());
}

// sigmoid = 1/(1+exp(-x)) using rcp14 + 1 Newton-Raphson step.
// rcp14 alone gives ~2^-14 relative error (~6e-5).
// One NR iteration refines to ~2^-28 (~3.7e-9) — well below the
// exp polynomial error (5e-5), so overall sigmoid accuracy is unchanged.
// Cost: rcp14(4c) + fnmadd(0.5c) + mul(0.5c) ≈ 5c  vs  div(14c) = 2.8x faster.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 sigmoid_avx512(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 two = _mm512_set1_ps(2.0f);
  __m512 denom = _mm512_add_ps(one, fast_exp_neg_avx512(x));
  __m512 rcp = _mm512_rcp14_ps(denom);
  // NR: rcp' = rcp * (2 - denom * rcp)
  rcp = _mm512_mul_ps(rcp, _mm512_fnmadd_ps(denom, rcp, two));
  return rcp;
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 silu_avx512(__m512 x) {
  return _mm512_mul_ps(x, sigmoid_avx512(x));
}

// GELU: per-lane scalar std::erf (no AVX-512 erf intrinsic).
// A vectorized tanh-based approximation could be added if GELU becomes
// a hot path; MoE models overwhelmingly use SiLU.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 gelu_avx512(__m512 x) {
  alignas(64) float arr[16];
  _mm512_store_ps(arr, x);
  for (int i = 0; i < 16; ++i)
    arr[i] = arr[i] * 0.5f * (1.0f + std::erf(arr[i] * kSqrtHalf));
  return _mm512_load_ps(arr);
}

// Stride-2 gather index for swiglu_oai F32 (interleaved layout).
alignas(64) static const int32_t kGatherIdx[16] =
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};

// Deinterleave indices for BF16 swiglu_oai via _mm512_permutexvar_epi16.
// Extracts even lanes (gates) or odd lanes (ups) from 32 interleaved BF16 values.
alignas(64) static const uint16_t kDeintGateIdx[32] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
alignas(64) static const uint16_t kDeintUpIdx[32] = {
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// ═══════════════════════════════════════════════════════════════════════
// AVX-512 row kernels (target-attributed)
// ═══════════════════════════════════════════════════════════════════════

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void silu_and_mul_row_avx512_f32(float *row, int dim) {
  float *gate = row, *up = row + dim;
  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512 g = _mm512_loadu_ps(gate + n);
    __m512 u = _mm512_loadu_ps(up + n);
    _mm512_storeu_ps(gate + n, _mm512_mul_ps(silu_avx512(g), u));
  }
  for (; n < dim; ++n)
    gate[n] = silu_scalar(gate[n]) * up[n];
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void silu_and_mul_row_avx512_bf16(bfloat16_t *row, int dim) {
  bfloat16_t *gate = row, *up = row + dim;
  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512 g = bf16x16_to_f32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&gate[n])));
    __m512 u = bf16x16_to_f32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&up[n])));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&gate[n]),
        f32_to_bf16x16(_mm512_mul_ps(silu_avx512(g), u)));
  }
  for (; n < dim; ++n) {
    float g = static_cast<float>(gate[n]);
    float u = static_cast<float>(up[n]);
    gate[n] = bfloat16_t(silu_scalar(g) * u);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void gelu_and_mul_row_avx512_f32(float *row, int dim) {
  float *gate = row, *up = row + dim;
  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512 g = _mm512_loadu_ps(gate + n);
    __m512 u = _mm512_loadu_ps(up + n);
    _mm512_storeu_ps(gate + n, _mm512_mul_ps(gelu_avx512(g), u));
  }
  for (; n < dim; ++n)
    gate[n] = gelu_scalar(gate[n]) * up[n];
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void gelu_and_mul_row_avx512_bf16(bfloat16_t *row, int dim) {
  bfloat16_t *gate = row, *up = row + dim;
  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512 g = bf16x16_to_f32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&gate[n])));
    __m512 u = bf16x16_to_f32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&up[n])));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&gate[n]),
        f32_to_bf16x16(_mm512_mul_ps(gelu_avx512(g), u)));
  }
  for (; n < dim; ++n) {
    float g = static_cast<float>(gate[n]);
    float u = static_cast<float>(up[n]);
    gate[n] = bfloat16_t(gelu_scalar(g) * u);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_mul_row_avx512_f32(float *row, int dim) {
  const __m512 alpha_vec = _mm512_set1_ps(1.702f);
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 clamp_max = _mm512_set1_ps(7.0f);
  const __m512 clamp_min = _mm512_set1_ps(-7.0f);
  const __m512i idx = _mm512_load_si512(kGatherIdx);

  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512 g = _mm512_i32gather_ps(idx, row + 2 * n, 4);
    __m512 u = _mm512_i32gather_ps(idx, row + 2 * n + 1, 4);

    g = _mm512_max_ps(clamp_min, _mm512_min_ps(g, clamp_max));
    u = _mm512_max_ps(clamp_min, _mm512_min_ps(u, clamp_max));

    __m512 sig = sigmoid_avx512(_mm512_mul_ps(g, alpha_vec));
    _mm512_storeu_ps(row + n, _mm512_mul_ps(
        _mm512_add_ps(one, u), _mm512_mul_ps(g, sig)));
  }
  const float alpha = 1.702f;
  for (; n < dim; ++n) {
    float g = std::max(-7.0f, std::min(row[2 * n], 7.0f));
    float u = std::max(-7.0f, std::min(row[2 * n + 1], 7.0f));
    row[n] = (1.0f + u) * g * sigmoid_scalar(g * alpha);
  }
}

// BF16 swiglu: vectorized deinterleave via _mm512_permutexvar_epi16.
// Replaces 16-iteration scalar loop with 1 load + 2 permutes.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_mul_row_avx512_bf16(bfloat16_t *row, int dim) {
  const __m512 alpha_vec = _mm512_set1_ps(1.702f);
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 clamp_max = _mm512_set1_ps(7.0f);
  const __m512 clamp_min = _mm512_set1_ps(-7.0f);
  const __m512i g_idx = _mm512_load_si512(kDeintGateIdx);
  const __m512i u_idx = _mm512_load_si512(kDeintUpIdx);

  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    // Load 32 interleaved BF16 values: [g0,u0,g1,u1,...,g15,u15]
    __m512i raw = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(&row[2 * n]));
    // Deinterleave: extract even lanes (gates) and odd lanes (ups)
    __m512 g = bf16x16_to_f32(
        _mm512_castsi512_si256(_mm512_permutexvar_epi16(g_idx, raw)));
    __m512 u = bf16x16_to_f32(
        _mm512_castsi512_si256(_mm512_permutexvar_epi16(u_idx, raw)));

    g = _mm512_max_ps(clamp_min, _mm512_min_ps(g, clamp_max));
    u = _mm512_max_ps(clamp_min, _mm512_min_ps(u, clamp_max));
    __m512 sig = sigmoid_avx512(_mm512_mul_ps(g, alpha_vec));
    __m512 result = _mm512_mul_ps(
        _mm512_add_ps(one, u), _mm512_mul_ps(g, sig));

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&row[n]),
        f32_to_bf16x16(result));
  }
  const float alpha = 1.702f;
  for (; n < dim; ++n) {
    float g = std::max(-7.0f,
        std::min(static_cast<float>(row[2 * n]), 7.0f));
    float u = std::max(-7.0f,
        std::min(static_cast<float>(row[2 * n + 1]), 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    row[n] = bfloat16_t((1.0f + u) * g * sig);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Out-of-place swiglu_oai tile kernels (src/dst may alias when dst ≤ src)
//
// Used by the ALGO 3 N-tile fused-swiglu-oai path.  Each thread owns an
// interleaved pair-range [col_start, col_end) of the matmul output; after
// a barrier, the thread reads its `pairs = (col_end-col_start)/2` (g, u)
// pairs and writes `pairs` activated values into [col_start/2, col_end/2).
//
// In-place safety: when dst aliases into src's buffer with dst_offset
// ≤ src_offset, the write at position `n` happens AFTER reads at
// positions `2n` and `2n+1`, both of which are ≥ n — so no corruption.
// At dst_offset == src_offset (col_start == 0) the write at position n
// overwrites only positions that are never read again.
// ═══════════════════════════════════════════════════════════════════════

void swiglu_oai_tile_scalar_f32(const float *src, float *dst, int pairs) {
  const float alpha = 1.702f;
  for (int n = 0; n < pairs; ++n) {
    float g = std::max(-7.0f, std::min(src[2 * n], 7.0f));
    float u = std::max(-7.0f, std::min(src[2 * n + 1], 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    dst[n] = (1.0f + u) * g * sig;
  }
}

void swiglu_oai_tile_scalar_bf16(const bfloat16_t *src, bfloat16_t *dst,
                                 int pairs) {
  const float alpha = 1.702f;
  for (int n = 0; n < pairs; ++n) {
    float g = std::max(-7.0f,
        std::min(static_cast<float>(src[2 * n]), 7.0f));
    float u = std::max(-7.0f,
        std::min(static_cast<float>(src[2 * n + 1]), 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    dst[n] = bfloat16_t((1.0f + u) * g * sig);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_tile_avx512_f32(const float *src, float *dst,
                                       int pairs) {
  const __m512 alpha_vec = _mm512_set1_ps(1.702f);
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 clamp_max = _mm512_set1_ps(7.0f);
  const __m512 clamp_min = _mm512_set1_ps(-7.0f);
  const __m512i idx = _mm512_load_si512(kGatherIdx);

  int n = 0;
  for (; n + 16 <= pairs; n += 16) {
    __m512 g = _mm512_i32gather_ps(idx, src + 2 * n, 4);
    __m512 u = _mm512_i32gather_ps(idx, src + 2 * n + 1, 4);
    g = _mm512_max_ps(clamp_min, _mm512_min_ps(g, clamp_max));
    u = _mm512_max_ps(clamp_min, _mm512_min_ps(u, clamp_max));
    __m512 sig = sigmoid_avx512(_mm512_mul_ps(g, alpha_vec));
    _mm512_storeu_ps(dst + n, _mm512_mul_ps(
        _mm512_add_ps(one, u), _mm512_mul_ps(g, sig)));
  }
  const float alpha = 1.702f;
  for (; n < pairs; ++n) {
    float g = std::max(-7.0f, std::min(src[2 * n], 7.0f));
    float u = std::max(-7.0f, std::min(src[2 * n + 1], 7.0f));
    dst[n] = (1.0f + u) * g * sigmoid_scalar(g * alpha);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_tile_avx512_bf16(const bfloat16_t *src,
                                        bfloat16_t *dst, int pairs) {
  const __m512 alpha_vec = _mm512_set1_ps(1.702f);
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 clamp_max = _mm512_set1_ps(7.0f);
  const __m512 clamp_min = _mm512_set1_ps(-7.0f);
  const __m512i g_idx = _mm512_load_si512(kDeintGateIdx);
  const __m512i u_idx = _mm512_load_si512(kDeintUpIdx);

  int n = 0;
  for (; n + 16 <= pairs; n += 16) {
    // Load 32 interleaved BF16 values starting at src[2n].
    __m512i raw = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(&src[2 * n]));
    __m512 g = bf16x16_to_f32(
        _mm512_castsi512_si256(_mm512_permutexvar_epi16(g_idx, raw)));
    __m512 u = bf16x16_to_f32(
        _mm512_castsi512_si256(_mm512_permutexvar_epi16(u_idx, raw)));

    g = _mm512_max_ps(clamp_min, _mm512_min_ps(g, clamp_max));
    u = _mm512_max_ps(clamp_min, _mm512_min_ps(u, clamp_max));
    __m512 sig = sigmoid_avx512(_mm512_mul_ps(g, alpha_vec));
    __m512 result = _mm512_mul_ps(
        _mm512_add_ps(one, u), _mm512_mul_ps(g, sig));

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[n]),
        f32_to_bf16x16(result));
  }
  const float alpha = 1.702f;
  for (; n < pairs; ++n) {
    float g = std::max(-7.0f,
        std::min(static_cast<float>(src[2 * n]), 7.0f));
    float u = std::max(-7.0f,
        std::min(static_cast<float>(src[2 * n + 1]), 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    dst[n] = bfloat16_t((1.0f + u) * g * sig);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Target-attributed execute functions (contain OMP loop — eliminates
// per-row dispatch function call overhead)
// ═══════════════════════════════════════════════════════════════════════

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void execute_act_rows_avx512(
    grp_matmul_gated_act_t act, bool is_f32,
    const std::vector<void *> &dst,
    const std::vector<int64_t> &row_offsets,
    const std::vector<int> &N,
    const std::vector<int> &ldc,
    int64_t total_rows, int num_threads) {

  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int64_t t = 0; t < total_rows; ++t) {
    const int e = static_cast<int>(
        std::upper_bound(row_offsets.begin() + 1, row_offsets.end(), t)
        - row_offsets.begin()) - 1;
    const int m = static_cast<int>(t - row_offsets[e]);
    const int dim = N[e] / 2;

    if (is_f32) {
      float *row = static_cast<float *>(dst[e])
          + static_cast<size_t>(m) * ldc[e];
      switch (act) {
      case grp_matmul_gated_act_t::silu_and_mul:
        silu_and_mul_row_avx512_f32(row, dim); break;
      case grp_matmul_gated_act_t::gelu_and_mul:
        gelu_and_mul_row_avx512_f32(row, dim); break;
      case grp_matmul_gated_act_t::swiglu_oai_mul:
        swiglu_oai_mul_row_avx512_f32(row, dim); break;
      default: break;
      }
    } else {
      auto *row = static_cast<bfloat16_t *>(dst[e])
          + static_cast<size_t>(m) * ldc[e];
      switch (act) {
      case grp_matmul_gated_act_t::silu_and_mul:
        silu_and_mul_row_avx512_bf16(row, dim); break;
      case grp_matmul_gated_act_t::gelu_and_mul:
        gelu_and_mul_row_avx512_bf16(row, dim); break;
      case grp_matmul_gated_act_t::swiglu_oai_mul:
        swiglu_oai_mul_row_avx512_bf16(row, dim); break;
      default: break;
      }
    }
  }
}

void execute_act_rows_scalar(
    grp_matmul_gated_act_t act, bool is_f32,
    const std::vector<void *> &dst,
    const std::vector<int64_t> &row_offsets,
    const std::vector<int> &N,
    const std::vector<int> &ldc,
    int64_t total_rows, int num_threads) {

  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int64_t t = 0; t < total_rows; ++t) {
    const int e = static_cast<int>(
        std::upper_bound(row_offsets.begin() + 1, row_offsets.end(), t)
        - row_offsets.begin()) - 1;
    const int m = static_cast<int>(t - row_offsets[e]);
    const int dim = N[e] / 2;

    if (is_f32) {
      float *row = static_cast<float *>(dst[e])
          + static_cast<size_t>(m) * ldc[e];
      switch (act) {
      case grp_matmul_gated_act_t::silu_and_mul:
        silu_and_mul_row_scalar_f32(row, dim); break;
      case grp_matmul_gated_act_t::gelu_and_mul:
        gelu_and_mul_row_scalar_f32(row, dim); break;
      case grp_matmul_gated_act_t::swiglu_oai_mul:
        swiglu_oai_mul_row_scalar_f32(row, dim); break;
      default: break;
      }
    } else {
      auto *row = static_cast<bfloat16_t *>(dst[e])
          + static_cast<size_t>(m) * ldc[e];
      switch (act) {
      case grp_matmul_gated_act_t::silu_and_mul:
        silu_and_mul_row_scalar_bf16(row, dim); break;
      case grp_matmul_gated_act_t::gelu_and_mul:
        gelu_and_mul_row_scalar_bf16(row, dim); break;
      case grp_matmul_gated_act_t::swiglu_oai_mul:
        swiglu_oai_mul_row_scalar_bf16(row, dim); break;
      default: break;
      }
    }
  }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════

status_t group_matmul_moe_act_execute(
    const grp_matmul_gated_act_params *act_params,
    const std::vector<void *> &dst,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &ldc,
    data_type_t dst_dtype,
    int num_threads) {

  if (act_params == nullptr
      || act_params->act == grp_matmul_gated_act_t::none)
    return status_t::success;

  const int num_ops = static_cast<int>(dst.size());
  if (num_ops == 0) return status_t::success;

  if (static_cast<int>(M.size()) != num_ops
      || static_cast<int>(N.size()) != num_ops
      || static_cast<int>(ldc.size()) != num_ops) {
    log_error("group_matmul_moe_act: M/N/ldc size mismatch with dst");
    return status_t::failure;
  }

  const grp_matmul_gated_act_t act = act_params->act;

  if (act != grp_matmul_gated_act_t::silu_and_mul
      && act != grp_matmul_gated_act_t::gelu_and_mul
      && act != grp_matmul_gated_act_t::swiglu_oai_mul) {
    log_error("group_matmul_moe_act: unsupported activation type ",
              static_cast<int>(act));
    return status_t::failure;
  }

  if (dst_dtype != data_type_t::f32 && dst_dtype != data_type_t::bf16) {
    log_error("group_matmul_moe_act: unsupported dst_dtype (must be f32 or bf16)");
    return status_t::failure;
  }

  const bool is_f32 = (dst_dtype == data_type_t::f32);

  for (int i = 0; i < num_ops; ++i) {
    if (N[i] % 2 != 0) {
      log_error("group_matmul_moe_act: N[", i, "]=", N[i],
                " must be even for gated activation");
      return status_t::failure;
    }
  }

  // Prefix sums over active experts: O(num_ops) instead of O(total_rows).
  std::vector<int64_t> row_offsets(num_ops + 1, 0);
  for (int e = 0; e < num_ops; ++e) {
    const int64_t rows = (dst[e] != nullptr && M[e] > 0) ? M[e] : 0;
    row_offsets[e + 1] = row_offsets[e] + rows;
  }
  const int64_t total_rows = row_offsets[num_ops];
  if (total_rows == 0) return status_t::success;

  // Single runtime dispatch into target-attributed execute function.
  // ZenDNN targets AMD Zen4+ where AVX-512F implies BW/VL/FMA (all are
  // part of the base AVX-512 package on Zen4/5).  Xeon Phi (KNL/KNM)
  // has F without BW/VL but is not a supported platform.  This matches
  // the dispatch pattern in group_matmul_moe_postop.cpp.
  if (avx512f_available())
    execute_act_rows_avx512(act, is_f32, dst, row_offsets, N, ldc,
                            total_rows, num_threads);
  else
    execute_act_rows_scalar(act, is_f32, dst, row_offsets, N, ldc,
                            total_rows, num_threads);

  return status_t::success;
}

// ═══════════════════════════════════════════════════════════════════════
// Single-threaded per-expert activation (for fused ALGO 1/2/4/5 paths)
// ═══════════════════════════════════════════════════════════════════════

void apply_gated_act_inplace(
    grp_matmul_gated_act_t act,
    void *dst, int row_start, int row_end,
    int N, int ldc, data_type_t dst_dtype) {

  if (act == grp_matmul_gated_act_t::none || row_start >= row_end || !dst)
    return;

  const int dim = N / 2;
  const bool is_f32 = (dst_dtype == data_type_t::f32);
  const bool use_avx512 =
      avx512f_available();

  for (int m = row_start; m < row_end; ++m) {
    if (use_avx512) {
      if (is_f32) {
        float *row = static_cast<float *>(dst)
            + static_cast<size_t>(m) * ldc;
        switch (act) {
        case grp_matmul_gated_act_t::silu_and_mul:
          silu_and_mul_row_avx512_f32(row, dim); break;
        case grp_matmul_gated_act_t::gelu_and_mul:
          gelu_and_mul_row_avx512_f32(row, dim); break;
        case grp_matmul_gated_act_t::swiglu_oai_mul:
          swiglu_oai_mul_row_avx512_f32(row, dim); break;
        default: break;
        }
      } else {
        auto *row = static_cast<bfloat16_t *>(dst)
            + static_cast<size_t>(m) * ldc;
        switch (act) {
        case grp_matmul_gated_act_t::silu_and_mul:
          silu_and_mul_row_avx512_bf16(row, dim); break;
        case grp_matmul_gated_act_t::gelu_and_mul:
          gelu_and_mul_row_avx512_bf16(row, dim); break;
        case grp_matmul_gated_act_t::swiglu_oai_mul:
          swiglu_oai_mul_row_avx512_bf16(row, dim); break;
        default: break;
        }
      }
    } else {
      if (is_f32) {
        float *row = static_cast<float *>(dst)
            + static_cast<size_t>(m) * ldc;
        switch (act) {
        case grp_matmul_gated_act_t::silu_and_mul:
          silu_and_mul_row_scalar_f32(row, dim); break;
        case grp_matmul_gated_act_t::gelu_and_mul:
          gelu_and_mul_row_scalar_f32(row, dim); break;
        case grp_matmul_gated_act_t::swiglu_oai_mul:
          swiglu_oai_mul_row_scalar_f32(row, dim); break;
        default: break;
        }
      } else {
        auto *row = static_cast<bfloat16_t *>(dst)
            + static_cast<size_t>(m) * ldc;
        switch (act) {
        case grp_matmul_gated_act_t::silu_and_mul:
          silu_and_mul_row_scalar_bf16(row, dim); break;
        case grp_matmul_gated_act_t::gelu_and_mul:
          gelu_and_mul_row_scalar_bf16(row, dim); break;
        case grp_matmul_gated_act_t::swiglu_oai_mul:
          swiglu_oai_mul_row_scalar_bf16(row, dim); break;
        default: break;
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Per-thread tile application for ALGO 3 fused-swiglu-oai path
// ═══════════════════════════════════════════════════════════════════════

void apply_swiglu_oai_tile_rows(
    void *dst_buf, int M, int col_start, int pairs,
    int ldc, data_type_t dtype) {
  // Preconditions (documented in the header).  Silently no-op on misuse
  // rather than reinterpreting the buffer or mis-halving columns, and
  // assert in debug builds so tests catch the bug early.
  assert(dst_buf != nullptr && "apply_swiglu_oai_tile_rows: dst_buf is null");
  assert((col_start & 1) == 0
         && "apply_swiglu_oai_tile_rows: col_start must be even "
            "(pair-aligned for interleaved gate/up layout)");
  assert((dtype == data_type_t::f32 || dtype == data_type_t::bf16)
         && "apply_swiglu_oai_tile_rows: dtype must be f32 or bf16");
  if (pairs <= 0 || M <= 0 || dst_buf == nullptr) return;
  if ((col_start & 1) != 0) return;  // pair-misaligned — refuse silently in release
  if (dtype != data_type_t::f32 && dtype != data_type_t::bf16) return;

  const bool is_f32 = (dtype == data_type_t::f32);
  const bool use_avx512 = avx512f_available();

  const int dst_col = col_start / 2;  // swiglu halves the output width

  if (is_f32) {
    auto *base = static_cast<float *>(dst_buf);
    for (int m = 0; m < M; ++m) {
      const float *src_row = base + static_cast<size_t>(m) * ldc + col_start;
      float *dst_row       = base + static_cast<size_t>(m) * ldc + dst_col;
      if (use_avx512)
        swiglu_oai_tile_avx512_f32(src_row, dst_row, pairs);
      else
        swiglu_oai_tile_scalar_f32(src_row, dst_row, pairs);
    }
  } else {
    auto *base = static_cast<bfloat16_t *>(dst_buf);
    for (int m = 0; m < M; ++m) {
      const bfloat16_t *src_row = base + static_cast<size_t>(m) * ldc + col_start;
      bfloat16_t *dst_row       = base + static_cast<size_t>(m) * ldc + dst_col;
      if (use_avx512)
        swiglu_oai_tile_avx512_bf16(src_row, dst_row, pairs);
      else
        swiglu_oai_tile_scalar_bf16(src_row, dst_row, pairs);
    }
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
