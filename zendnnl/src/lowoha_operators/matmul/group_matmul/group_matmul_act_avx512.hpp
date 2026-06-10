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
// F16 ↔ FP32 conversion
// ═══════════════════════════════════════════════════════════════════
//
/// `f16x16_to_f32`: convert 16 × IEEE 754 binary16 lanes to 16 × FP32
/// via the AVX-512F `VCVTPH2PS` intrinsic (`_mm512_cvtph_ps`).  The
/// input is loaded as a `__m256i` (16 × uint16_t) for storage-ABI
/// parity with the BF16 helpers and consumed directly — no `__m256h`
/// cast.  Using the `__m256i`-typed AVX-512F form (rather than the
/// AVX-512-FP16 `_mm512_cvtxph_ps` on `__m256h`) keeps this helper on
/// the same widely-available intrinsic set as `common/float16.cpp`
/// and avoids a toolchain dependency on `__m256h`/`avx512fp16` for a
/// conversion that the base AVX-512F ISA already provides.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 f16x16_to_f32(__m256i f16) {
  return _mm512_cvtph_ps(f16);
}

/// `f32_to_f16x16`: round-to-nearest-even FP32→F16 cvt via the AVX-512F
/// `VCVTPS2PH` intrinsic (`_mm512_cvtps_ph`).  Stores the 16 × F16
/// lanes as a `__m256i` for storage-ABI parity with the BF16 helpers.
/// The explicit `_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC`
/// immediate pins round-to-nearest-even independent of the runtime
/// MXCSR mode, matching `float16_t::cvt_f32_to_f16_vec` in
/// `common/float16.cpp`.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m256i f32_to_f16x16(__m512 f32) {
  return _mm512_cvtps_ph(f32,
                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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

/// `erf_avx512(x)` — vectorised error function via the
/// Abramowitz & Stegun 7.1.26 rational/exp approximation:
///
///   erf(x) ≈ sign(x) · (1 − P(t) · exp(−x²)),
///   t = 1 / (1 + p·|x|),
///   P(t) = ((((a5·t + a4)·t + a3)·t + a2)·t + a1)·t,
///
/// with constants {p, a1..a5} from A&S 7.1.26.  Note: the published
/// ≤1.5e-7 absolute-error bound assumes an accurate `expf`; this
/// implementation uses `fast_exp_neg_avx512`, so the total error is
/// dominated by that exp approximation (~5e-5 relative).
/// Cost is one `fast_exp_neg_avx512` + one `rcp14_ps` + NR step +
/// five FMAs and a sign mask, comparable to the previous `gelu_tanh` form.
// `avx512dq` is added to the target attribute (in addition to the
// base `avx512f,avx512bw,avx512vl,fma` set used by the rest of this
// header) because the FP-domain bitwise intrinsics
// `_mm512_and_ps` / `_mm512_andnot_ps` / `_mm512_xor_ps` live in
// AVX-512 DQ (header `avx512dqintrin.h`).  All Zen 3+ and
// Skylake-X+ CPUs that ship AVX-512 also ship AVX-512 DQ — the
// pre-existing AVX-512 platform gate elsewhere in the library is
// sufficient to ensure this code path is only entered on
// DQ-capable hardware.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static inline __m512 erf_avx512(__m512 x) {
  const __m512  one  = _mm512_set1_ps(1.0f);
  const __m512  two  = _mm512_set1_ps(2.0f);
  const __m512  p    = _mm512_set1_ps(0.3275911f);
  const __m512  a1   = _mm512_set1_ps( 0.254829592f);
  const __m512  a2   = _mm512_set1_ps(-0.284496736f);
  const __m512  a3   = _mm512_set1_ps( 1.421413741f);
  const __m512  a4   = _mm512_set1_ps(-1.453152027f);
  const __m512  a5   = _mm512_set1_ps( 1.061405429f);

  // Extract sign(x) and |x| via a `-0.0f` float mask.  `-0.0f` has
  // exactly the IEEE-754 sign bit set and all other bits clear, so:
  //   sign(x) = x &  (-0.0f)  ── keeps only the sign bit of x
  //   |x|     = x & ~(-0.0f)  ── clears the sign bit, leaves magnitude
  // Both operations stay in the FP domain (`_mm512_and_ps` /
  // `_mm512_andnot_ps`) — no int-bit constants, no
  // implementation-defined `0x80000000u → int` conversions.  The
  // VANDPS / VANDNPS instructions are the same micro-op family as
  // the int variants on Zen / Skylake, so codegen is unchanged.
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
  __m512 sign  = _mm512_and_ps   (x, sign_mask);
  __m512 absx  = _mm512_andnot_ps(sign_mask, x);

  // t = 1 / (1 + p·|x|), refined by one Newton-Raphson step so the
  // residual is bounded by `fast_exp_neg_avx512`'s ~5e-5 envelope
  // rather than `rcp14_ps`'s ~2⁻¹⁴.
  //
  // Overflow defence (NR step): for catastrophically large `|x|`
  // (`|x| > (FLT_MAX − 1) / p ≈ 1e38`) the FMA `denom = 1 + p·|x|`
  // overflows to `+inf`.  `rcp14_ps(+inf)` returns `0`, and the NR
  // expression `t · (2 − denom · t)` then evaluates as
  // `0 · (2 − inf·0) = 0 · (2 − NaN) = 0 · NaN = NaN` — silently
  // poisoning the polynomial below.  Mask the NR result to `0` on
  // any lane whose `denom` is non-finite (`!(denom < +inf)`, which
  // also catches `denom == NaN`).  The downstream polynomial then
  // collapses to `poly · t = 0`, and the final `1 − poly · ex² = 1`
  // gives the correct erf saturation magnitude (sign re-applied
  // below via XOR).  IEEE `_CMP_LT_OQ` returns false on NaN, so
  // the `denom_finite` mask is `0` for NaN lanes — same fallback.
  const __m512 fp_inf = _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000));
  __m512 denom = _mm512_fmadd_ps(p, absx, one);
  __m512 t     = _mm512_rcp14_ps(denom);
  t = _mm512_mul_ps(t, _mm512_fnmadd_ps(denom, t, two));
  const __mmask16 denom_finite =
      _mm512_cmp_ps_mask(denom, fp_inf, _CMP_LT_OQ);
  t = _mm512_maskz_mov_ps(denom_finite, t);

  // P(t) · t (Horner)
  __m512 poly = _mm512_fmadd_ps(a5, t, a4);
  poly        = _mm512_fmadd_ps(poly, t, a3);
  poly        = _mm512_fmadd_ps(poly, t, a2);
  poly        = _mm512_fmadd_ps(poly, t, a1);
  poly        = _mm512_mul_ps(poly, t);

  // result = 1 − P(t) · exp(−x²); then re-apply sign of x via
  // `_mm512_xor_ps`.  The `sign` mask holds only the sign bit of x,
  // so XOR-ing it into `result` flips the sign iff x was negative —
  // the same bit-level operation as before, just expressed without
  // the int reinterpret.
  //
  // Overflow defence (exp argument): `absx · absx` overflows to
  // `+inf` for `|x| > √FLT_MAX ≈ 1.84e19`.  `fast_exp_neg_avx512`'s
  // exponent-recovery path produces `NaN` for an `+inf` argument
  // (`floor(+inf) = +inf`; subsequent `inf − inf = NaN`).  Force
  // `ex² = 0` on any lane whose `x²` is non-finite — mathematically
  // exact (`exp(−∞) = 0`), and the final `1 − poly · 0 = 1` gives
  // the correct saturation magnitude for `|x| → ∞`.  Same
  // `_CMP_LT_OQ` semantics handle the NaN-propagating case too.
  __m512 x2     = _mm512_mul_ps(absx, absx);
  __m512 ex2    = fast_exp_neg_avx512(x2);
  const __mmask16 x2_finite =
      _mm512_cmp_ps_mask(x2, fp_inf, _CMP_LT_OQ);
  ex2           = _mm512_maskz_mov_ps(x2_finite, ex2);
  __m512 result = _mm512_fnmadd_ps(poly, ex2, one);
  __m512 signed_result = _mm512_xor_ps(result, sign);
  const __mmask16 nan_mask = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
  return _mm512_mask_mov_ps(signed_result, nan_mask, x);
}

/// `gelu_avx512(x)` — vectorised `gelu_erf` form using `erf_avx512`,
/// intended to closely track the scalar reference in
/// `group_matmul_moe_act.cpp::gelu_scalar` (and PyTorch's default
/// `torch.nn.functional.gelu(...)`) within the FP32 / BF16 tolerances:
///   gelu_erf(x) = 0.5 · x · (1 + erf(x / √2))

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 gelu_avx512(__m512 x) {
  const __m512 half       = _mm512_set1_ps(0.5f);
  const __m512 one        = _mm512_set1_ps(1.0f);
  // 1/√2 = √0.5; full-precision constant so the FP32 product
  // matches `M_SQRT1_2` (which the scalar `gelu_scalar` uses
  // implicitly via `std::erf(x * 1/√2)`).
  const __m512 inv_sqrt2  = _mm512_set1_ps(0.7071067811865475f);
  __m512 z   = _mm512_mul_ps(x, inv_sqrt2);
  __m512 e   = erf_avx512(z);
  return _mm512_mul_ps(_mm512_mul_ps(half, x),
                       _mm512_add_ps(one, e));
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
