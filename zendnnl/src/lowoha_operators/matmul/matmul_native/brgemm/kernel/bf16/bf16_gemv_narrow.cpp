/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_narrow.hpp"

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <cstring>

#include "common/zendnnl_global.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::log_error;

namespace {

/// Fast B-pair loader: two 8-byte unaligned loads from B (one per row)
/// followed by a `vpunpcklwd` interleave.  Used in the inner K-loop
/// where we've verified there are at least 4 BF16 elements (= one
/// full 8-byte load per row) worth of in-bounds memory past the
/// current (2kp+1)-th row start.
///
/// Layout in the returned xmm (BF16 lane-pair grouped):
///   lane 0: (B[2kp,   0], B[2kp+1, 0])
///   lane 1: (B[2kp,   1], B[2kp+1, 1])
///   lane 2: (B[2kp,   2], B[2kp+1, 2])
///   lane 3: (B[2kp,   3], B[2kp+1, 3])
/// Unused lanes (N < 4) contain adjacent row/column data that will
/// accumulate into unused acc lanes and get discarded at store time
/// — intentional and harmless.
__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512bf16")))
static inline __m128i load_b_pair_fast(
    const uint16_t *__restrict__ B, int ldb, int kp) {
  const __m128i r0 = _mm_loadl_epi64(
      reinterpret_cast<const __m128i *>(B + static_cast<size_t>(2 * kp)     * ldb));
  const __m128i r1 = _mm_loadl_epi64(
      reinterpret_cast<const __m128i *>(B + static_cast<size_t>(2 * kp + 1) * ldb));
  return _mm_unpacklo_epi16(r0, r1);
}

/// Safe B-pair loader for the trailing K-pair(s) and odd-K tail.
/// Uses a small stack staging buffer so N < 4 stays zero-padded and,
/// crucially, never reads past the caller's B allocation.  The K-loop
/// calls this at most twice per call (last K-pair + odd-K tail) so
/// the store-forward cost is amortised to O(1) vs the K-major cost
/// of the fast path.
///
/// `k1_is_live` is false when 2kp+1 >= K (odd-K tail case), in which
/// case the second row's BF16 is forced to zero (matching the BKC
/// pack's zero-fill convention for the K-odd last element).
template <int N>
__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512bf16")))
static inline __m128i load_b_pair_safe(
    const uint16_t *__restrict__ B, int ldb, int kp, bool k1_is_live) {
  alignas(8) uint16_t b_row0[4] = {0, 0, 0, 0};
  alignas(8) uint16_t b_row1[4] = {0, 0, 0, 0};
  std::memcpy(b_row0, B + static_cast<size_t>(2 * kp)     * ldb,
              N * sizeof(uint16_t));
  if (k1_is_live) {
    std::memcpy(b_row1, B + static_cast<size_t>(2 * kp + 1) * ldb,
                N * sizeof(uint16_t));
  }
  const __m128i r0 = _mm_loadl_epi64(
      reinterpret_cast<const __m128i *>(b_row0));
  const __m128i r1 = _mm_loadl_epi64(
      reinterpret_cast<const __m128i *>(b_row1));
  return _mm_unpacklo_epi16(r0, r1);
}

/// Broadcast one K-pair from A (4 bytes = 2 BF16) to every lane of
/// an xmm register.  Uses the `memcpy(uint32_t) + _mm_set1_epi32`
/// idiom that GCC fuses into a single `vpbroadcastd xmm, [mem]`.
__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512bf16")))
static inline __m128 load_a_pair_broadcast(
    const uint16_t *__restrict__ A, int kp) {
  uint32_t a_pair;
  std::memcpy(&a_pair, A + 2 * kp, sizeof(a_pair));
  return _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(a_pair)));
}

/// Largest kp index for which `load_b_pair_fast` can safely read
/// 8 bytes from row `2kp + 1`.  We need
///   `(2kp + 1) * ldb * 2 + 8 <= K * ldb * 2`
/// i.e. `2kp + 1 + 4 / ldb <= K`.  For N ≤ 4 and ldb ≥ N we have
/// `4 / ldb ≤ 4`, so a conservative-and-correct bound is
///   `2kp + 5 <= K` ⇒ `kp <= (K - 5) / 2`.
/// Callers use the fast path while `kp <= kp_fast_max`, then switch
/// to `load_b_pair_safe` for the remaining 1–2 K-pairs and the
/// odd-K tail.  The bound leaves a 4-row slack so double-buffering
/// `(kp, kp+1)` stays inside the fast region.
inline int kp_fast_max_for_K(int K) {
  const int kp_max = (K - 5) / 2;
  return (kp_max >= 0) ? kp_max : -1;
}

/// Core templated narrow kernel.  N is a template parameter so the
/// memcpy sizes, zero-extension patterns, and final store width all
/// collapse at compile time — no runtime branching inside the
/// K-loop.  Double-buffered over K-pairs (even / odd) to cover the
/// 4-cycle VDPBF16PS latency at 1-cycle issue rate, matching the
/// pattern in the group-matmul custom microkernel.
template <int N>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"), noinline))
static void bf16_gemv_narrow_impl(
    const uint16_t *__restrict__ A, int K,
    const uint16_t *__restrict__ B, int ldb,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16) {

  static_assert(N >= 1 && N <= 4,
                "narrow kernel handles N ∈ [1, 4]");

  // Four independent accumulator chains unroll the K-pair loop by 4
  // to cover the 4-cycle VDPBF16PS xmm latency on Zen 5 — two chains
  // (double-buffer) would saturate issue only if the xmm FMA issues
  // at 1 op/cycle; profiling on representative narrow-N decode
  // shapes (M=1, N=3, K~6000) showed ~1.6× distance from ideal with
  // 2 chains, pointing to a ≤ 0.5 op/cycle issue rate on xmm.  Four
  // chains give the out-of-order engine enough slack to cover either
  // case.
  __m128 acc0 = _mm_setzero_ps();
  __m128 acc1 = _mm_setzero_ps();
  __m128 acc2 = _mm_setzero_ps();
  __m128 acc3 = _mm_setzero_ps();

  const int k_pairs_even = K / 2;
  const int kp_fast_max  = kp_fast_max_for_K(K);

  // ── Fast 4-way unrolled K-pair loop (4 K-pairs per iter) ───────────
  // Runs while all 4 K-pairs (kp, kp+1, kp+2, kp+3) have at least
  // 4 BF16 elements past their last-row start, so 8-byte unaligned
  // loads never cross the caller's B allocation end.
  // NOTE on bf16-lane vector casts: the Intel intrinsics ABI defines
  // `__m128bh` as a distinct vector value type that represents 8
  // packed BF16 lanes (vs `__m128` = 4 FP32, `__m128i` = 128-bit
  // integer).  Going between these types requires a bit-preserving
  // re-interpretation.  C++ `reinterpret_cast` between SIMD vector
  // types is implementation-defined and flagged by some compilers,
  // so we use C-style casts (same pattern the BKC kernel uses —
  // see `bf16_gemv_bkc_nr64_core`) which are a no-op at the
  // generated-code level and are portable across GCC/Clang/MSVC.
  int kp = 0;
  for (; kp + 3 <= kp_fast_max; kp += 4) {
    const __m128   a0 = load_a_pair_broadcast(A, kp);
    const __m128i  b0 = load_b_pair_fast(B, ldb, kp);
    const __m128   a1 = load_a_pair_broadcast(A, kp + 1);
    const __m128i  b1 = load_b_pair_fast(B, ldb, kp + 1);
    const __m128   a2 = load_a_pair_broadcast(A, kp + 2);
    const __m128i  b2 = load_b_pair_fast(B, ldb, kp + 2);
    const __m128   a3 = load_a_pair_broadcast(A, kp + 3);
    const __m128i  b3 = load_b_pair_fast(B, ldb, kp + 3);
    acc0 = _mm_dpbf16_ps(acc0, (__m128bh)a0, (__m128bh)b0);
    acc1 = _mm_dpbf16_ps(acc1, (__m128bh)a1, (__m128bh)b1);
    acc2 = _mm_dpbf16_ps(acc2, (__m128bh)a2, (__m128bh)b2);
    acc3 = _mm_dpbf16_ps(acc3, (__m128bh)a3, (__m128bh)b3);
  }
  // ── Second-pass 2-wide unroll for any 2–3 leftover fast K-pairs ────
  for (; kp + 1 <= kp_fast_max; kp += 2) {
    const __m128   a0 = load_a_pair_broadcast(A, kp);
    const __m128i  b0 = load_b_pair_fast(B, ldb, kp);
    const __m128   a1 = load_a_pair_broadcast(A, kp + 1);
    const __m128i  b1 = load_b_pair_fast(B, ldb, kp + 1);
    acc0 = _mm_dpbf16_ps(acc0, (__m128bh)a0, (__m128bh)b0);
    acc1 = _mm_dpbf16_ps(acc1, (__m128bh)a1, (__m128bh)b1);
  }

  // ── Trailing K-pairs (0–3 remaining), safe staging to avoid OOB ────
  for (; kp < k_pairs_even; ++kp) {
    const __m128   a = load_a_pair_broadcast(A, kp);
    const __m128i  b = load_b_pair_safe<N>(B, ldb, kp, /*k1_is_live=*/true);
    acc0 = _mm_dpbf16_ps(acc0, (__m128bh)a, (__m128bh)b);
  }

  // Combine the four accumulator chains.
  __m128 acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));

  // ── Odd-K tail: one stray row at index K-1 ────────────────────────
  // Force the high BF16 of the A pair and the entire second row of
  // the B pair to zero so VDPBF16PS contributes exactly the live
  // lane.  Same convention as the BKC pack uses for odd-K tails.
  if (K & 1) {
    const int kp_tail = k_pairs_even;  // == (K - 1) / 2
    const uint32_t a_lo = static_cast<uint32_t>(A[K - 1]);
    const __m128   a = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(a_lo)));
    const __m128i  b = load_b_pair_safe<N>(B, ldb, kp_tail, /*k1_is_live=*/false);
    acc = _mm_dpbf16_ps(acc, (__m128bh)a, (__m128bh)b);
  }

  // ── Epilogue: alpha, beta · C_old, bias, fused postop, store ───────
  if (alpha != 1.0f) {
    acc = _mm_mul_ps(acc, _mm_set1_ps(alpha));
  }

  if (beta != 0.0f) {
    __m128 c_old = _mm_setzero_ps();
    if (dst_is_bf16 && C_bf16) {
      alignas(8) uint16_t tmp[4] = {0, 0, 0, 0};
      std::memcpy(tmp, C_bf16, N * sizeof(uint16_t));
      const __m128i raw = _mm_loadl_epi64(
          reinterpret_cast<const __m128i *>(tmp));
      c_old = _mm_castsi128_ps(
          _mm_slli_epi32(_mm_cvtepu16_epi32(raw), 16));
    } else if (C_fp32) {
      alignas(16) float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      std::memcpy(tmp, C_fp32, N * sizeof(float));
      c_old = _mm_load_ps(tmp);
    }
    acc = _mm_fmadd_ps(_mm_set1_ps(beta), c_old, acc);
  }

  if (bias_f != nullptr) {
    alignas(16) float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::memcpy(tmp, bias_f, N * sizeof(float));
    acc = _mm_add_ps(acc, _mm_load_ps(tmp));
  }

  // Fused postop uses the AVX-512F 512-bit variant (only this width
  // is implemented in `apply_fused_postop`).  Widen-apply-extract;
  // runs once per call so the widening cost (~3 cycles) is
  // negligible next to the K-loop.
  //
  // `_mm512_castps128_ps512` is a free type-pun — the upper 12 lanes
  // of the resulting zmm are undefined.  `apply_fused_postop` runs
  // elementwise transcendentals (exp, tanh, …) across all 16 lanes,
  // so feeding undefined data into those lanes can:
  //   * spuriously raise FP exception flags on garbage operands
  //     (denormals, NaNs, large negatives in exp() → +inf), and
  //   * cause minor wasted work in the AVX-512 transcendental units.
  // We zero the upper 12 lanes via `_mm512_maskz_mov_ps` (single
  // `vmovaps zmm{k1}{z}` instruction, ~1 cycle, idle ports) before
  // applying the postop.  The result is bit-identical for the live
  // 4 lanes — the only observable change is "no garbage in upper".
  if (fused_op != fused_postop_t::none) {
    const __m512 wide =
        _mm512_maskz_mov_ps(0x000F, _mm512_castps128_ps512(acc));
    const __m512 applied = apply_fused_postop(wide, fused_op);
    acc = _mm512_castps512_ps128(applied);
  }

  if (dst_is_bf16 && C_bf16) {
    const __m128bh bf = _mm_cvtneps_pbh(acc);
    alignas(8) uint16_t tmp[4];
    // C-style cast for the bf16→int128 lane-type reinterpret; see
    // the K-loop note above for the rationale.
    _mm_storel_epi64(reinterpret_cast<__m128i *>(tmp), (__m128i)bf);
    std::memcpy(C_bf16, tmp, N * sizeof(uint16_t));
  } else if (C_fp32) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, acc);
    std::memcpy(C_fp32, tmp, N * sizeof(float));
  }
}

} // namespace

void bf16_gemv_narrow(
    const uint16_t *A, int K,
    const uint16_t *B, int ldb,
    uint16_t *C_bf16, float *C_fp32,
    const float *bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int N) {
  switch (N) {
  case 1:
    bf16_gemv_narrow_impl<1>(A, K, B, ldb, C_bf16, C_fp32, bias_f,
                             fused_op, alpha, beta, dst_is_bf16);
    break;
  case 2:
    bf16_gemv_narrow_impl<2>(A, K, B, ldb, C_bf16, C_fp32, bias_f,
                             fused_op, alpha, beta, dst_is_bf16);
    break;
  case 3:
    bf16_gemv_narrow_impl<3>(A, K, B, ldb, C_bf16, C_fp32, bias_f,
                             fused_op, alpha, beta, dst_is_bf16);
    break;
  case 4:
    bf16_gemv_narrow_impl<4>(A, K, B, ldb, C_bf16, C_fp32, bias_f,
                             fused_op, alpha, beta, dst_is_bf16);
    break;
  default:
    // Contract violation: caller MUST have filtered on
    // `N >= 1 && N <= kBf16GemvNarrowMaxN` before dispatch.  This
    // branch would leave `C` unwritten and produce silent-wrong
    // results downstream, so we make the misuse loud:
    //   * Debug build → `assert(false)` traps at the source.
    //   * Release build → `log_error()` emits a traceable line so
    //     an integrator sees the contract break in the library log
    //     (the kernel API is `void` with no error return channel).
    // There is intentionally no scalar-GEMV fallback here because
    // the upstream gate in `bf16_gemv_best_algo_impl` +
    // `bf16_gemv_direct` already ensures we never reach this path
    // on a valid dispatch, and a hidden fallback would mask
    // routing bugs that should be fixed at the caller.
    log_error("bf16_gemv_narrow: out-of-range N=", N,
              " (must be 1..kBf16GemvNarrowMaxN=",
              kBf16GemvNarrowMaxN, "); upstream should gate on N. "
              "Destination buffer left UNINITIALISED — downstream "
              "results will be wrong until the caller is fixed.");
    assert(false && "bf16_gemv_narrow dispatched with out-of-range N "
                    "(must be 1..kBf16GemvNarrowMaxN); upstream "
                    "should gate on N.");
    break;
  }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
