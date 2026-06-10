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
/// Three variants (compatible with the gated-activation conventions
/// used by major LLM serving stacks):
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
#include "common/float16.hpp"
#include "common/zendnnl_global.hpp"
#include "group_matmul_direct.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_act_avx512.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;
using zendnnl::memory::data_type_t;

namespace {

// Shared AVX-512 vector activation math — single source of truth
// for `fast_exp_neg_avx512`, `sigmoid_avx512`, `silu_avx512`,
// `gelu_avx512`, `swiglu_oai_avx512`, and the `bf16x16_to_f32` /
// `f32_to_bf16x16` / `f16x16_to_f32` / `f32_to_f16x16` cvt helpers.
// Pulled in via using-declarations (rather than `using namespace
// ...;`) so each name is explicit and pinned at the namespace
// boundary; future drift between this file and
// `bf16_microkernel.cpp` is structurally impossible.
using zendnnl::lowoha::matmul::group_matmul_act_avx512::bf16x16_to_f32;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::f32_to_bf16x16;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::f16x16_to_f32;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::f32_to_f16x16;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::fast_exp_neg_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::sigmoid_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::silu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::gelu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::swiglu_oai_avx512;

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

// ── F16 scalar row kernels ────────────────────────────────────────
//
// Storage type is `float16_t` — the canonical F16 element type
// (data_type_t::f16 maps to float16_t via prec_traits, and the
// group_matmul stack carries F16 dst buffers as float16_t*).  This
// matches the bf16 siblings' use of `bfloat16_t*` and avoids the
// strict-aliasing UB of viewing a float16_t buffer through uint16_t*.
// Element access mirrors the bf16 kernels: `static_cast<float>()`
// (float16_t::operator float → f16_to_f32_val) on read and the
// `float16_t(value)` converting constructor (→ f32_to_f16_val) on
// write.  The activation math (silu / gelu / swiglu_oai) runs in F32
// throughout — same pattern as the bf16 siblings.
void silu_and_mul_row_scalar_f16(float16_t *row, int dim) {
  float16_t *gate = row, *up = row + dim;
  for (int n = 0; n < dim; ++n) {
    float g = static_cast<float>(gate[n]);
    float u = static_cast<float>(up[n]);
    gate[n] = float16_t(silu_scalar(g) * u);
  }
}

void gelu_and_mul_row_scalar_f16(float16_t *row, int dim) {
  float16_t *gate = row, *up = row + dim;
  for (int n = 0; n < dim; ++n) {
    float g = static_cast<float>(gate[n]);
    float u = static_cast<float>(up[n]);
    gate[n] = float16_t(gelu_scalar(g) * u);
  }
}

void swiglu_oai_mul_row_scalar_f16(float16_t *row, int dim) {
  const float alpha = 1.702f;
  for (int n = 0; n < dim; ++n) {
    float g = std::max(-7.0f,
        std::min(static_cast<float>(row[2 * n]), 7.0f));
    float u = std::max(-7.0f,
        std::min(static_cast<float>(row[2 * n + 1]), 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    row[n] = float16_t((1.0f + u) * g * sig);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// AVX-512 primitives — single source of truth in
// `group_matmul_act_avx512.hpp`.  Both this file (the separate-pass
// row helpers) and `custom_kernel/ukernel/bf16_microkernel.cpp` (the
// fused-CK in-register store helpers) consume the same FP32 vector
// math via the using-declarations above.  See the header doc-block
// for the cross-path numerical contract.
// ═══════════════════════════════════════════════════════════════════════

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

// FP32 swiglu_oai row helper.  Stride-2 gather from the interleaved
// `[g0, u0, g1, u1, ...]` buffer, then the unified
// `swiglu_oai_avx512(gate, up)` does clamp + (1+u)·g·σ(α·g) on the
// pre-deinterleaved zmm pair.  Scalar tail handles `dim % 16` only.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_mul_row_avx512_f32(float *row, int dim) {
  const __m512i idx = _mm512_load_si512(kGatherIdx);

  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512 g = _mm512_i32gather_ps(idx, row + 2 * n, 4);
    __m512 u = _mm512_i32gather_ps(idx, row + 2 * n + 1, 4);
    _mm512_storeu_ps(row + n, swiglu_oai_avx512(g, u));
  }
  const float alpha = 1.702f;
  for (; n < dim; ++n) {
    float g = std::max(-7.0f, std::min(row[2 * n], 7.0f));
    float u = std::max(-7.0f, std::min(row[2 * n + 1], 7.0f));
    row[n] = (1.0f + u) * g * sigmoid_scalar(g * alpha);
  }
}

// BF16 swiglu_oai row helper.  One 32-lane vmovdqu64 of the
// interleaved buffer, then two `vpermtxvar_epi16` extract gates
// (even lanes) and ups (odd lanes) into separate 16-lane BF16 zmms,
// `bf16x16_to_f32` lifts each to FP32, and the unified
// `swiglu_oai_avx512(gate, up)` does the rest.  Replaces a 16-iter
// scalar loop with 1 load + 2 permutes + the shared math.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_mul_row_avx512_bf16(bfloat16_t *row, int dim) {
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

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&row[n]),
        f32_to_bf16x16(swiglu_oai_avx512(g, u)));
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

// ── F16 AVX-512 row kernels ────────────────────────────────────────
//
// Mirror the BF16 row-kernel pattern line-for-line, swapping
// `bf16x16_to_f32` / `f32_to_bf16x16` for the F16 cvts (`f16x16_to_
// f32` / `f32_to_f16x16`).  Storage type is `float16_t` (the canonical
// F16 element type — same as the bf16 path uses `bfloat16_t`).  SIMD
// load/store go through `__m256i*`/`__m512i*` (may_alias) so the
// element type doesn't affect them; the scalar tail uses
// `static_cast<float>()` / the `float16_t(value)` constructor.  The
// activation math (silu / gelu / swiglu_oai) runs on FP32 lanes
// throughout, so the only difference vs the BF16 path is the cvt.
//
// Target attribute is the SAME as the BF16 siblings
// (`avx512f,avx512bw,avx512vl,fma`).  The F16
// load/store goes through `f16x16_to_f32` / `f32_to_f16x16`, which
// use the AVX-512F `VCVTPH2PS` / `VCVTPS2PH` intrinsics
// (`_mm512_cvtph_ps` / `_mm512_cvtps_ph`); the FP32 activation math
// is AVX-512F and the deinterleave is AVX-512BW.  No AVX-512-FP16
// instruction is emitted, so these kernels run on any AVX-512F host
// and are safe to dispatch behind the `avx512f_available()` gate
// (avoids the SIGILL hazard of an avx512fp16-licensed function being
// reached on a host without AVX-512-FP16).  Note: F16 group_matmul as
// a whole still requires AVX-512-FP16 for the GEMM itself — the public
// `group_matmul_direct` F16 ISA gate enforces that upstream — but the
// activation post-pass does not depend on it.

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void silu_and_mul_row_avx512_f16(float16_t *row, int dim) {
  float16_t *gate = row, *up = row + dim;
  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512 g = f16x16_to_f32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&gate[n])));
    __m512 u = f16x16_to_f32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&up[n])));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&gate[n]),
        f32_to_f16x16(_mm512_mul_ps(silu_avx512(g), u)));
  }
  for (; n < dim; ++n) {
    float g = static_cast<float>(gate[n]);
    float u = static_cast<float>(up[n]);
    gate[n] = float16_t(silu_scalar(g) * u);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void gelu_and_mul_row_avx512_f16(float16_t *row, int dim) {
  float16_t *gate = row, *up = row + dim;
  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512 g = f16x16_to_f32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&gate[n])));
    __m512 u = f16x16_to_f32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&up[n])));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&gate[n]),
        f32_to_f16x16(_mm512_mul_ps(gelu_avx512(g), u)));
  }
  for (; n < dim; ++n) {
    float g = static_cast<float>(gate[n]);
    float u = static_cast<float>(up[n]);
    gate[n] = float16_t(gelu_scalar(g) * u);
  }
}

// F16 swiglu_oai row helper.  Same vpermtxvar_epi16 deinterleave
// pattern as the BF16 sibling — `_mm512_permutexvar_epi16` operates
// on 16-bit lanes regardless of their float interpretation, so the
// gate/up index tables (kDeintGateIdx / kDeintUpIdx) are reused
// verbatim.  Only the load lift (`f16x16_to_f32`) and the final
// cvt-store (`f32_to_f16x16`) differ from the BF16 path.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_mul_row_avx512_f16(float16_t *row, int dim) {
  const __m512i g_idx = _mm512_load_si512(kDeintGateIdx);
  const __m512i u_idx = _mm512_load_si512(kDeintUpIdx);

  int n = 0;
  for (; n + 16 <= dim; n += 16) {
    __m512i raw = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(&row[2 * n]));
    __m512 g = f16x16_to_f32(
        _mm512_castsi512_si256(_mm512_permutexvar_epi16(g_idx, raw)));
    __m512 u = f16x16_to_f32(
        _mm512_castsi512_si256(_mm512_permutexvar_epi16(u_idx, raw)));

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&row[n]),
        f32_to_f16x16(swiglu_oai_avx512(g, u)));
  }
  const float alpha = 1.702f;
  for (; n < dim; ++n) {
    float g = std::max(-7.0f,
        std::min(static_cast<float>(row[2 * n]), 7.0f));
    float u = std::max(-7.0f,
        std::min(static_cast<float>(row[2 * n + 1]), 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    row[n] = float16_t((1.0f + u) * g * sig);
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

// FP32 swiglu_oai OOP tile helper (separate src/dst buffers).
// Same `(g, u)` deinterleave + shared `swiglu_oai_avx512(gate, up)`
// pattern as the in-place row helper above; only the source / dst
// pointers differ.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_tile_avx512_f32(const float *src, float *dst,
                                       int pairs) {
  const __m512i idx = _mm512_load_si512(kGatherIdx);

  int n = 0;
  for (; n + 16 <= pairs; n += 16) {
    __m512 g = _mm512_i32gather_ps(idx, src + 2 * n, 4);
    __m512 u = _mm512_i32gather_ps(idx, src + 2 * n + 1, 4);
    _mm512_storeu_ps(dst + n, swiglu_oai_avx512(g, u));
  }
  const float alpha = 1.702f;
  for (; n < pairs; ++n) {
    float g = std::max(-7.0f, std::min(src[2 * n], 7.0f));
    float u = std::max(-7.0f, std::min(src[2 * n + 1], 7.0f));
    dst[n] = (1.0f + u) * g * sigmoid_scalar(g * alpha);
  }
}

// BF16 swiglu_oai OOP tile helper.  Vectorised 32-lane load +
// vpermtxvar_epi16 deinterleave, then shared
// `swiglu_oai_avx512(gate, up)` + `f32_to_bf16x16` store.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_tile_avx512_bf16(const bfloat16_t *src,
                                        bfloat16_t *dst, int pairs) {
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

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[n]),
        f32_to_bf16x16(swiglu_oai_avx512(g, u)));
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

// ── F16 swiglu_oai OOP tile kernels ────────────────────────────────
//
// Mirror of the BF16 OOP tile pattern: scalar tail handler + AVX-512
// deinterleave-and-fold body.  Storage type is `float16_t` (canonical
// F16 element type); cvt to / from F32 via `static_cast<float>` /
// the `float16_t(value)` constructor (scalar) or `f16x16_to_f32` /
// `f32_to_f16x16`
// (AVX-512F `VCVTPH2PS` / `VCVTPS2PH`).
//
// In-place safety: dst may alias into src's buffer with `dst_offset
// ≤ src_offset` (same alias contract as the BF16 sibling — write at
// position n always happens AFTER reads at positions 2n and 2n+1,
// both of which are ≥ n).  At dst_offset == src_offset (col_start
// == 0) the write at position n overwrites only positions that are
// never read again.

void swiglu_oai_tile_scalar_f16(const float16_t *src, float16_t *dst,
                                int pairs) {
  const float alpha = 1.702f;
  for (int n = 0; n < pairs; ++n) {
    float g = std::max(-7.0f,
        std::min(static_cast<float>(src[2 * n]), 7.0f));
    float u = std::max(-7.0f,
        std::min(static_cast<float>(src[2 * n + 1]), 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    dst[n] = float16_t((1.0f + u) * g * sig);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static void swiglu_oai_tile_avx512_f16(const float16_t *src,
                                       float16_t *dst, int pairs) {
  const __m512i g_idx = _mm512_load_si512(kDeintGateIdx);
  const __m512i u_idx = _mm512_load_si512(kDeintUpIdx);

  int n = 0;
  for (; n + 16 <= pairs; n += 16) {
    __m512i raw = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(&src[2 * n]));
    __m512 g = f16x16_to_f32(
        _mm512_castsi512_si256(_mm512_permutexvar_epi16(g_idx, raw)));
    __m512 u = f16x16_to_f32(
        _mm512_castsi512_si256(_mm512_permutexvar_epi16(u_idx, raw)));

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[n]),
        f32_to_f16x16(swiglu_oai_avx512(g, u)));
  }
  const float alpha = 1.702f;
  for (; n < pairs; ++n) {
    float g = std::max(-7.0f,
        std::min(static_cast<float>(src[2 * n]), 7.0f));
    float u = std::max(-7.0f,
        std::min(static_cast<float>(src[2 * n + 1]), 7.0f));
    float sig = sigmoid_scalar(g * alpha);
    dst[n] = float16_t((1.0f + u) * g * sig);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Target-attributed execute functions (contain OMP loop — eliminates
// per-row dispatch function call overhead)
// ═══════════════════════════════════════════════════════════════════════

// Target attribute is `avx512f,avx512bw,avx512vl,fma`:
// all three dtype branches — including the F16 branch — emit only
// AVX-512F/BW instructions (F16↔F32 cvt via VCVTPH2PS/VCVTPS2PH, F32
// activation math, AVX-512BW deinterleave).  This keeps the function's
// compile-time ISA license aligned with the `avx512f_available()`
// dispatch gate below, so it cannot SIGILL on an AVX-512F host that
// lacks AVX-512-FP16.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void execute_act_rows_avx512(
    grp_matmul_gated_act_t act, data_type_t dst_dtype,
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

    if (dst_dtype == data_type_t::f32) {
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
    } else if (dst_dtype == data_type_t::bf16) {
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
    } else {
      auto *row = static_cast<float16_t *>(dst[e])
          + static_cast<size_t>(m) * ldc[e];
      switch (act) {
      case grp_matmul_gated_act_t::silu_and_mul:
        silu_and_mul_row_avx512_f16(row, dim); break;
      case grp_matmul_gated_act_t::gelu_and_mul:
        gelu_and_mul_row_avx512_f16(row, dim); break;
      case grp_matmul_gated_act_t::swiglu_oai_mul:
        swiglu_oai_mul_row_avx512_f16(row, dim); break;
      default: break;
      }
    }
  }
}

void execute_act_rows_scalar(
    grp_matmul_gated_act_t act, data_type_t dst_dtype,
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

    if (dst_dtype == data_type_t::f32) {
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
    } else if (dst_dtype == data_type_t::bf16) {
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
    } else {
      auto *row = static_cast<float16_t *>(dst[e])
          + static_cast<size_t>(m) * ldc[e];
      switch (act) {
      case grp_matmul_gated_act_t::silu_and_mul:
        silu_and_mul_row_scalar_f16(row, dim); break;
      case grp_matmul_gated_act_t::gelu_and_mul:
        gelu_and_mul_row_scalar_f16(row, dim); break;
      case grp_matmul_gated_act_t::swiglu_oai_mul:
        swiglu_oai_mul_row_scalar_f16(row, dim); break;
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

  // Drive the iteration count off `M.size()` — the matmul-processing
  // count `group_matmul_direct` derived from `params[0].active_matmul`
  // (or `M.size()` itself for legacy callers).  When the framework
  // signals prepack-extras, the caller passes a sliced M while keeping
  // dst[] / N[] / ldc[] at their original size; using `dst.size()`
  // here would walk into the prepack-extras tail and process garbage.
  const int num_ops = static_cast<int>(M.size());
  if (num_ops == 0) return status_t::success;

  // Vector sizes must be at least `num_ops` each.  Anything past
  // num_ops in dst[] / N[] / ldc[] is a prepack-extras placeholder
  // that the dispatch loops below never read.  Strict equality was
  // the legacy contract; relaxing to `<` accepts the new layout
  // without affecting legacy callers (they pass exactly-sized
  // vectors, so the relaxed check still matches strict for them).
  if (static_cast<int>(dst.size()) < num_ops
      || static_cast<int>(N.size()) < num_ops
      || static_cast<int>(ldc.size()) < num_ops) {
    log_error("group_matmul_moe_act: dst/N/ldc must be sized to at least M.size()");
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

  if (dst_dtype != data_type_t::f32
      && dst_dtype != data_type_t::bf16
      && dst_dtype != data_type_t::f16) {
    log_error("group_matmul_moe_act: unsupported dst_dtype (must be f32, bf16, or f16)");
    return status_t::failure;
  }

  for (int i = 0; i < num_ops; ++i) {
    if (N[i] % 2 != 0) {
      log_error("group_matmul_moe_act: N[", i, "]=", N[i],
                " must be even for gated activation");
      return status_t::failure;
    }
  }

  // Prefix sums over active experts: O(num_ops) instead of O(total_rows).
  // Persistent thread-local so the vector capacity grows monotonically
  // and steady-state calls reuse the existing allocation.  resize() to
  // num_ops+1 (idempotent when the capacity is already sufficient) and
  // explicitly seed the first slot to 0 — the prefix-sum loop then
  // overwrites the rest.  Avoids a per-call heap allocation that the
  // separate-pass fused_moe path would otherwise hit on every call
  // (the in-tile fused-activation path doesn't go through this
  // function — it fuses activation in-register).
  static thread_local std::vector<int64_t> row_offsets;
  row_offsets.resize(num_ops + 1);
  row_offsets[0] = 0;
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
    execute_act_rows_avx512(act, dst_dtype, dst, row_offsets, N, ldc,
                            total_rows, num_threads);
  else
    execute_act_rows_scalar(act, dst_dtype, dst, row_offsets, N, ldc,
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

  if (dst_dtype != data_type_t::f32
      && dst_dtype != data_type_t::bf16
      && dst_dtype != data_type_t::f16)
    return;

  const int dim = N / 2;
  const bool use_avx512 = avx512f_available();

  for (int m = row_start; m < row_end; ++m) {
    if (use_avx512) {
      if (dst_dtype == data_type_t::f32) {
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
      } else if (dst_dtype == data_type_t::bf16) {
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
      } else {
        auto *row = static_cast<float16_t *>(dst)
            + static_cast<size_t>(m) * ldc;
        switch (act) {
        case grp_matmul_gated_act_t::silu_and_mul:
          silu_and_mul_row_avx512_f16(row, dim); break;
        case grp_matmul_gated_act_t::gelu_and_mul:
          gelu_and_mul_row_avx512_f16(row, dim); break;
        case grp_matmul_gated_act_t::swiglu_oai_mul:
          swiglu_oai_mul_row_avx512_f16(row, dim); break;
        default: break;
        }
      }
    } else {
      if (dst_dtype == data_type_t::f32) {
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
      } else if (dst_dtype == data_type_t::bf16) {
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
      } else {
        auto *row = static_cast<float16_t *>(dst)
            + static_cast<size_t>(m) * ldc;
        switch (act) {
        case grp_matmul_gated_act_t::silu_and_mul:
          silu_and_mul_row_scalar_f16(row, dim); break;
        case grp_matmul_gated_act_t::gelu_and_mul:
          gelu_and_mul_row_scalar_f16(row, dim); break;
        case grp_matmul_gated_act_t::swiglu_oai_mul:
          swiglu_oai_mul_row_scalar_f16(row, dim); break;
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
  // assert in debug builds so tests catch contract violations early.
  assert(dst_buf != nullptr && "apply_swiglu_oai_tile_rows: dst_buf is null");
  assert((col_start & 1) == 0
         && "apply_swiglu_oai_tile_rows: col_start must be even "
            "(pair-aligned for interleaved gate/up layout)");
  assert((dtype == data_type_t::f32
          || dtype == data_type_t::bf16
          || dtype == data_type_t::f16)
         && "apply_swiglu_oai_tile_rows: dtype must be f32, bf16, or f16");
  if (pairs <= 0 || M <= 0 || dst_buf == nullptr) return;
  if ((col_start & 1) != 0) return;  // pair-misaligned — refuse silently in release
  if (dtype != data_type_t::f32
      && dtype != data_type_t::bf16
      && dtype != data_type_t::f16) return;

  const bool use_avx512 = avx512f_available();

  const int dst_col = col_start / 2;  // swiglu halves the output width

  if (dtype == data_type_t::f32) {
    auto *base = static_cast<float *>(dst_buf);
    for (int m = 0; m < M; ++m) {
      const float *src_row = base + static_cast<size_t>(m) * ldc + col_start;
      float *dst_row       = base + static_cast<size_t>(m) * ldc + dst_col;
      if (use_avx512)
        swiglu_oai_tile_avx512_f32(src_row, dst_row, pairs);
      else
        swiglu_oai_tile_scalar_f32(src_row, dst_row, pairs);
    }
  } else if (dtype == data_type_t::bf16) {
    auto *base = static_cast<bfloat16_t *>(dst_buf);
    for (int m = 0; m < M; ++m) {
      const bfloat16_t *src_row = base + static_cast<size_t>(m) * ldc + col_start;
      bfloat16_t *dst_row       = base + static_cast<size_t>(m) * ldc + dst_col;
      if (use_avx512)
        swiglu_oai_tile_avx512_bf16(src_row, dst_row, pairs);
      else
        swiglu_oai_tile_scalar_bf16(src_row, dst_row, pairs);
    }
  } else {
    auto *base = static_cast<float16_t *>(dst_buf);
    for (int m = 0; m < M; ++m) {
      const float16_t *src_row = base + static_cast<size_t>(m) * ldc + col_start;
      float16_t *dst_row       = base + static_cast<size_t>(m) * ldc + dst_col;
      if (use_avx512)
        swiglu_oai_tile_avx512_f16(src_row, dst_row, pairs);
      else
        swiglu_oai_tile_scalar_f16(src_row, dst_row, pairs);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Out-of-place per-thread tile activation for the I-only fused MoE path
// ═══════════════════════════════════════════════════════════════════════

void apply_swiglu_oai_tile_rows_oop(
    const void *src_buf, int src_ldc, int src_col_start,
    void *dst_buf, int dst_ldc, int dst_col_start,
    int M, int pairs, data_type_t dtype) {
  // Preconditions identical to the in-place sibling, plus: src_buf may
  // alias dst_buf only when the read range [src_col_start ..
  // src_col_start + 2*pairs) is disjoint from the write range
  // [dst_col_start .. dst_col_start + pairs).  The fused MoE path
  // always passes distinct buffers (per-thread scratch vs tight arena),
  // so the alias case isn't exercised today.
  assert(src_buf != nullptr && "apply_swiglu_oai_tile_rows_oop: src_buf is null");
  assert(dst_buf != nullptr && "apply_swiglu_oai_tile_rows_oop: dst_buf is null");
  assert((src_col_start & 1) == 0
         && "apply_swiglu_oai_tile_rows_oop: src_col_start must be even");
  assert((dtype == data_type_t::f32
          || dtype == data_type_t::bf16
          || dtype == data_type_t::f16)
         && "apply_swiglu_oai_tile_rows_oop: dtype must be f32, bf16, or f16");
  if (pairs <= 0 || M <= 0 || src_buf == nullptr || dst_buf == nullptr) return;
  if ((src_col_start & 1) != 0) return;
  if (dtype != data_type_t::f32
      && dtype != data_type_t::bf16
      && dtype != data_type_t::f16) return;

  const bool use_avx512 = avx512f_available();

  if (dtype == data_type_t::f32) {
    const auto *src_base = static_cast<const float *>(src_buf);
    auto       *dst_base = static_cast<float *>(dst_buf);
    for (int m = 0; m < M; ++m) {
      const float *src_row =
          src_base + static_cast<size_t>(m) * src_ldc + src_col_start;
      float *dst_row =
          dst_base + static_cast<size_t>(m) * dst_ldc + dst_col_start;
      if (use_avx512)
        swiglu_oai_tile_avx512_f32(src_row, dst_row, pairs);
      else
        swiglu_oai_tile_scalar_f32(src_row, dst_row, pairs);
    }
  } else if (dtype == data_type_t::bf16) {
    const auto *src_base = static_cast<const bfloat16_t *>(src_buf);
    auto       *dst_base = static_cast<bfloat16_t *>(dst_buf);
    for (int m = 0; m < M; ++m) {
      const bfloat16_t *src_row =
          src_base + static_cast<size_t>(m) * src_ldc + src_col_start;
      bfloat16_t *dst_row =
          dst_base + static_cast<size_t>(m) * dst_ldc + dst_col_start;
      if (use_avx512)
        swiglu_oai_tile_avx512_bf16(src_row, dst_row, pairs);
      else
        swiglu_oai_tile_scalar_bf16(src_row, dst_row, pairs);
    }
  } else {
    const auto *src_base = static_cast<const float16_t *>(src_buf);
    auto       *dst_base = static_cast<float16_t *>(dst_buf);
    for (int m = 0; m < M; ++m) {
      const float16_t *src_row =
          src_base + static_cast<size_t>(m) * src_ldc + src_col_start;
      float16_t *dst_row =
          dst_base + static_cast<size_t>(m) * dst_ldc + dst_col_start;
      if (use_avx512)
        swiglu_oai_tile_avx512_f16(src_row, dst_row, pairs);
      else
        swiglu_oai_tile_scalar_f16(src_row, dst_row, pairs);
    }
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
