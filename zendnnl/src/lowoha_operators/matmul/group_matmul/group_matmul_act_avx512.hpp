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

/// Shared AVX-512 vector activation math for the group_matmul ALGO 3
/// path.
///
/// ── Why this header exists ─────────────────────────────────────────
/// Two consumers in the group_matmul stack need the same FP32 vector
/// activation math:
///
///   1. `group_matmul_moe_act.cpp` (the SEPARATE-PASS fallback)
///      Reads BF16 / FP32 rows from memory, applies gated activation,
///      writes activated cols back.  Reached when the custom kernel
///      is OFF (`ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0`) or refuses (e.g.,
///      silu/gelu + bias) — the dispatcher then runs the activation
///      as a post-pass via `apply_gated_act_inplace`.
///
///   2. `custom_kernel/ukernel/bf16_microkernel.cpp` (the FUSED-CK
///      hot path)
///      Applies the activation in-register on the FP32 accumulator
///      pair `(acc_lo, acc_hi)` produced by the matmul, then stores
///      the half-width BF16 result.  No mid-pipe BF16 round-trip.
///
/// Before this header was extracted, each consumer kept a private
/// copy of `fast_exp_neg`, `sigmoid`, `silu`, `gelu_tanh`, the
/// swiglu_oai (clamp + (1+u)·g·σ(α·g)) form, and the manual RNE
/// FP32→BF16 cvt.  The two copies were byte-identical and the
/// gtest suite enforced that via cross-path comparisons (`tol_act`
/// + `f32_to_bf16x16` bit-equality), but the duplication added a
/// continual risk of drift on every future tweak.  This header
/// makes the FP32 vector math the single source of truth.
///
/// ── What stays file-local ──────────────────────────────────────────
///   * Deinterleave constants (`kGateLaneIdx[16]` / `kUpLaneIdx[16]`
///     in `bf16_microkernel.cpp`; `kGatherIdx[16]`,
///     `kDeintGateIdx[32]` / `kDeintUpIdx[32]` in
///     `group_matmul_moe_act.cpp`).  Different use cases:
///       - microkernel: FP32-lane indices for `vpermt2ps` on the
///         register pair `(acc_lo, acc_hi)` covering 32 lanes of
///         interleaved gate/up accumulators.
///       - moe_act FP32 row helper: FP32-lane indices for
///         `vpgatherdd` from memory at stride 2.
///       - moe_act BF16 row helper: BF16-lane indices for
///         `vpermtxvar_epi16` to deinterleave 32 BF16 values.
///   * Scalar helpers (`silu_scalar`, `gelu_scalar`, …) and scalar-
///     row fallbacks in `group_matmul_moe_act.cpp` — used on non-
///     AVX-512 hardware; outside this header's AVX-512 scope.
///
/// ── Numerical contract (cross-path) ────────────────────────────────
/// The fused-CK path and the separate-pass path now go through the
/// same `fast_exp_neg_avx512`, `sigmoid_avx512`, polynomial
/// coefficients, and `f32_to_bf16x16` integer-RNE sequence.  Any
/// cross-path comparison is bit-equal modulo the elimination of one
/// BF16 round-trip in the fused path (matmul → FP32 acc → activation
/// → BF16 store vs matmul → BF16 → BF16 → FP32 → activation → BF16),
/// which only ever differs in the "more accurate" direction (the
/// fused path skips one lossy rounding).

#ifndef ZENDNNL_LOWOHA_MATMUL_GROUP_MATMUL_ACT_AVX512_HPP
#define ZENDNNL_LOWOHA_MATMUL_GROUP_MATMUL_ACT_AVX512_HPP

#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace group_matmul_act_avx512 {

// ═══════════════════════════════════════════════════════════════════
// BF16 ↔ FP32 conversion
// ═══════════════════════════════════════════════════════════════════

/// `bf16x16_to_f32`: zero-extend each 16-bit BF16 lane to 32 bits and
/// shift left by 16.  Standard BF16 unpack.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 bf16x16_to_f32(__m256i bf16) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

/// `f32_to_bf16x16`: round-to-nearest-even FP32→BF16 via integer
/// add + shift (bias = 0x7FFF + LSB-of-upper-16-bits).
///
/// ── Why the manual sequence (and not `_mm512_cvtneps_pbh`) ──────────
/// The hardware `VCVTNEPS2BF16` instruction is documented as RNE and
/// agrees with this manual sequence on the vast majority of inputs.
/// Half-way (tie) cases can diverge between silicon and the integer
/// sequence depending on the µarch implementation.  In the fused
/// MoE tight-arena path that mid-pipe FP32→BF16 rounding feeds into
/// the Op2 GEMM, which then amplifies any divergence by `√K_down`.
/// Pinning both paths to the same integer-RNE sequence makes the
/// fused-CK output bit-identical to the separate-pass reference at
/// the cvt step, removing one source of cross-path divergence.
/// Cost: one extra integer add + shift per cvt vs a single
/// `VCVTNEPS2BF16` — neutral end-to-end (the Op2 GEMM dominates by
/// ~10×).
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m256i f32_to_bf16x16(__m512 f32) {
  __m512i i32 = _mm512_castps_si512(f32);
  __m512i bias = _mm512_add_epi32(
      _mm512_set1_epi32(0x7FFF),
      _mm512_and_si512(_mm512_srli_epi32(i32, 16),
                       _mm512_set1_epi32(1)));
  return _mm512_cvtepi32_epi16(_mm512_srli_epi32(
      _mm512_add_epi32(i32, bias), 16));
}

// ═══════════════════════════════════════════════════════════════════
// Per-element math primitives
// ═══════════════════════════════════════════════════════════════════

/// `fast_exp_neg_avx512(x) ≈ exp(-x)`.
///
/// 5-term Cephes polynomial for `2^f` (Schraudolph-style), with the
/// integer exponent reconstructed via bit-cast.  Exponent clamped to
/// the IEEE-754 normal range `[-126, 127]`; result clamped to ≥ 0 to
/// defend against subnormal underflow.  Max relative error ~5e-5 —
/// well below BF16 ulp (~7.8e-3 relative) and the
/// `mt::tol_act(/*is_bf16=*/true)` band ({rel=0.15, abs=0.02}).
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 fast_exp_neg_avx512(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(-1.4426950408889634f);

  __m512 t  = _mm512_mul_ps(x, log2e);
  __m512 ti = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEG_INF);
  __m512 f  = _mm512_sub_ps(t, ti);

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

  return _mm512_max_ps(_mm512_mul_ps(pow2i, p), _mm512_setzero_ps());
}

/// `sigmoid_avx512(x) = 1 / (1 + exp(-x))`.
///
/// `rcp14_ps` (~2^-14 relative) refined by one Newton-Raphson step
/// (~2^-28 relative).  The refined relative error is below
/// `fast_exp_neg_avx512`'s ~5e-5, so overall sigmoid accuracy is
/// bounded by the exp polynomial — not by the divide.  rcp14 + NR
/// avoids the high-latency hardware divide.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 sigmoid_avx512(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 two = _mm512_set1_ps(2.0f);
  __m512 denom = _mm512_add_ps(one, fast_exp_neg_avx512(x));
  __m512 rcp = _mm512_rcp14_ps(denom);
  // NR step:  rcp' = rcp * (2 - denom * rcp)
  rcp = _mm512_mul_ps(rcp, _mm512_fnmadd_ps(denom, rcp, two));
  return rcp;
}

// ═══════════════════════════════════════════════════════════════════
// Element-wise gated activations (no `* up` — caller multiplies)
// ═══════════════════════════════════════════════════════════════════

/// `silu_avx512(x) = x * sigmoid(x)`.  Identical to the scalar
/// reference's `silu_scalar`.  Callers apply the `* up` multiplication
/// afterwards (for `silu_and_mul`).
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 silu_avx512(__m512 x) {
  return _mm512_mul_ps(x, sigmoid_avx512(x));
}

/// `gelu_avx512(x)` — vectorised `gelu_tanh` polynomial form (NOT
/// `gelu_erf`).
///
///   gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715·x³)))
///
/// Rewritten via the identity `tanh(y) = 2·sigmoid(2y) − 1`:
///
///   gelu_tanh(x) = x * sigmoid(2y),
///   where 2y = c1·x + c2·x³,
///         c1 = 2·sqrt(2/π)            ≈ 1.5957691,
///         c2 = 2·sqrt(2/π) · 0.044715 ≈ 0.0713548.
///
/// One `sigmoid_avx512` call.  Max delta vs the reference `gelu_erf`
/// ≤ 1.5e-3 across all real x (well-known bound) — ~5× tighter than
/// BF16 ulp.  The `mt::tol_act(/*is_bf16=*/true)` band accepts both
/// forms with ~10× margin.
///
/// Replaces a per-lane `std::erf` loop (16 libc calls per zmm) →
/// ~10× faster on Zen4/5.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 gelu_avx512(__m512 x) {
  const __m512 c1 = _mm512_set1_ps(1.5957691216057308f);
  const __m512 c2 = _mm512_set1_ps(0.0713548097404904f);
  __m512 x2  = _mm512_mul_ps(x, x);
  __m512 c1x = _mm512_mul_ps(c1, x);
  __m512 y2  = _mm512_fmadd_ps(_mm512_mul_ps(c2, x), x2, c1x);
  return _mm512_mul_ps(x, sigmoid_avx512(y2));
}

// ═══════════════════════════════════════════════════════════════════
// Gated activation on pre-deinterleaved (gate, up) FP32 zmm pair
// ═══════════════════════════════════════════════════════════════════

/// `swiglu_oai_avx512(gate, up)` — full swiglu_oai_mul fused form
/// on pre-deinterleaved FP32 zmm inputs.
///
///   out = (1 + clamp(up,  -7, +7))
///       * (clamp(gate, -7, +7) * sigmoid(α · clamp(gate, -7, +7)))
///
/// with α = 1.702 (OpenAI swiglu).  Identical to the scalar reference
/// in `group_matmul_moe_act.cpp::swiglu_oai_mul_row_scalar_*`.
///
/// Callers deinterleave gate/up from their native layout BEFORE
/// invoking, since the deinterleave step is layout-specific:
///   * separate-pass row helper (FP32 dst): stride-2
///     `vpgatherdd` from the `[g0, u0, g1, u1, …]` memory buffer.
///   * separate-pass row helper (BF16 dst): one `vmovdqu64` of the
///     32 interleaved BF16 lanes, then `vpermtxvar_epi16` ×2 to
///     extract gates and ups + `bf16x16_to_f32` ×2.
///   * fused-CK store helper: `vpermt2ps` ×2 on the `(acc_lo,
///     acc_hi)` FP32 accumulator pair, no memory round-trip.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 swiglu_oai_avx512(__m512 gate, __m512 up) {
  const __m512 cmin = _mm512_set1_ps(-7.0f);
  const __m512 cmax = _mm512_set1_ps(+7.0f);
  gate = _mm512_max_ps(_mm512_min_ps(gate, cmax), cmin);
  up   = _mm512_max_ps(_mm512_min_ps(up,   cmax), cmin);
  const __m512 alpha = _mm512_set1_ps(1.702f);
  const __m512 one   = _mm512_set1_ps(1.0f);
  __m512 sig = sigmoid_avx512(_mm512_mul_ps(gate, alpha));
  return _mm512_mul_ps(_mm512_add_ps(one, up),
                       _mm512_mul_ps(gate, sig));
}

}  // namespace group_matmul_act_avx512
}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl

#endif  // ZENDNNL_LOWOHA_MATMUL_GROUP_MATMUL_ACT_AVX512_HPP
