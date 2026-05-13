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

/// BF16 custom microkernel implementation — templated on (MR, NV, Act).
///
/// See ukernel.hpp for the role this kernel plays in the group_matmul
/// dispatch stack (both non-fused ALGO 3 and the fused-MoE tight-
/// arena path).
///
/// PER K-PAIR INNER LOOP (VDPBF16PS, AVX512_BF16):
///
///   load Bpacked → NV zmms (b[v] covers cols v*16..(v+1)*16 - 1)
///   for m in 0..MR-1:
///     broadcast A[m, k:k+2] (4 bytes = 2 BF16) into 16 FP32 lanes
///     for v in 0..NV-1:
///       acc[m][v] = vdpbf16ps(acc[m][v], A_reg, b[v])
///
/// The K-pair loop is unrolled by 2 (kk += 2 outer + u ∈ {0,1}
/// inner) so the next K-pair's B-loads and A-broadcasts can issue
/// before the previous pair's FMAs retire.  Same trick used by
/// `bf16_brgemm_ukernel.cpp`.
///
/// EPILOGUE FOR ActKind::swiglu_oai_mul (in-register fusion):
///
///   For each row, the (acc[m][2p], acc[m][2p+1]) pair is
///   deinterleaved into 16 gates and 16 ups via vpermt2ps; swiglu_oai
///   is computed on those vectors and 16 BF16 results are stored to
///   the tight destination.  The epilogue runs NV/2 times per row,
///   producing NV*8 BF16 outputs in total (NV=2 → 16; NV=4 → 32).

#include "ukernel.hpp"

#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

bool avx512bf16_available() {
  static const bool v = []() {
    return zendnnl::lowoha::matmul::native::detect_uarch().avx512bf16;
  }();
  return v;
}

namespace {

// ── Activation epilogue helpers (AVX-512) ───────────────────────────

// Deinterleave indices for vpermt2ps over two source zmms (= 32 FP32
// lanes total).  Even-indexed lanes (0, 2, ..., 30) are gates; odd-
// indexed lanes (1, 3, ..., 31) are ups.
alignas(64) constexpr int32_t kGateLaneIdx[16] =
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) constexpr int32_t kUpLaneIdx[16] =
    {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

// Polynomial exp(-x) — 5th-degree Cephes 2^f approximation.
// Identical numerics to group_matmul_moe_act.cpp's swiglu_oai
// reference path, so the custom kernel matches the existing
// per-element tolerance.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 fast_exp_neg(__m512 x) {
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

// sigmoid = 1 / (1 + exp(-x))  via rcp14 + 1 Newton-Raphson refine.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 sigmoid_fast(__m512 x) {
  const __m512 one   = _mm512_set1_ps(1.0f);
  const __m512 two   = _mm512_set1_ps(2.0f);
  __m512 denom = _mm512_add_ps(one, fast_exp_neg(x));
  __m512 rcp   = _mm512_rcp14_ps(denom);
  return _mm512_mul_ps(rcp, _mm512_fnmadd_ps(denom, rcp, two));
}

// BF16 → FP32 (zero-extend, shift-left 16) for the bias load.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m512 bf16x16_to_fp32(__m256i bf16) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

// FP32 → BF16 (round-to-nearest-even, manual sequence).  Bit-for-bit
// identical to the standard reference path's `f32_to_bf16x16` in
// group_matmul_moe_act.cpp.
//
// Why not the hardware `_mm512_cvtneps_pbh` (VCVTNEPS2BF16):
//   The instruction is documented as round-to-nearest-even and agrees
//   with this manual sequence on the vast majority of inputs.  Half-
//   way (tie) cases can diverge between the silicon and the integer
//   sequence depending on uarch implementation details, and that
//   divergence becomes visible end-to-end on the fused MoE tight-
//   arena path:
//     custom matmul (FP32 acc) → swiglu(g, u) = (1+u)·g·σ(α·g)
//                              → FP32→BF16
//                              → Op2 GEMM amplifies × √K_down
//   Using the manual sequence makes the custom path produce BIT-
//   IDENTICAL BF16 output to the standard reference for the FP32→
//   BF16 step, removing one source of cross-path divergence.  Cost:
//   one extra integer add + shift per cvt vs a single VCVTNEPS2BF16
//   — neutral end-to-end (the Op2 GEMM dominates by ~10×).
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline __m256i fp32_to_bf16x16_rne(__m512 f32) {
  __m512i i32 = _mm512_castps_si512(f32);
  // bias = 0x7FFF + (LSB of upper-16-bits) — implements RNE on the
  // truncated mantissa via integer add + shift.  Identical to the
  // standard reference path so cross-path comparisons are bit-exact.
  __m512i bias = _mm512_add_epi32(
      _mm512_set1_epi32(0x7FFF),
      _mm512_and_si512(_mm512_srli_epi32(i32, 16),
                       _mm512_set1_epi32(1)));
  return _mm512_cvtepi32_epi16(_mm512_srli_epi32(
      _mm512_add_epi32(i32, bias), 16));
}

// Apply swiglu_oai_mul in registers to one (gate, up) pair of FP32
// accumulators and store 16 activated BF16 cols.  `dst_row` is the
// tight destination row pointer (caller pre-offsets for col_start/2
// and the within-row epilogue index `p`).
//
// ── Numerical contract ───────────────────────────────────────────────
// Mathematically this routine is IDENTICAL to the standard reference
// swiglu_oai path in group_matmul_moe_act.cpp: same 5-term Cephes
// exp(-x) polynomial (c0..c5 = {1.0, 0.6931472, 0.2402265, 0.0555042,
// 0.0096838, 0.0013364}), same rcp14 + 1-step Newton-Raphson sigmoid,
// same clamp bounds [-7, 7] on gate/up, same multiply order
// `(1 + up) * (gate * sigmoid(gate * 1.702f))`.
//
// Precision characteristic vs the standard two-pass pipeline:
//   Standard reference: matmul writes BF16 to memory → activation
//     reads BF16 → cvt BF16→FP32 → swiglu → cvt FP32→BF16 → store.
//     The accumulator is rounded to BF16 before the activation sees
//     it.
//   Custom kernel: matmul FP32 accumulator stays in register →
//     swiglu on raw FP32 → cvt FP32→BF16 once → store.  No mid-pipe
//     BF16 rounding.
// Net effect: the custom path skips one lossy BF16 rounding in the
// middle, so its absolute BF16 error is ≤ that of the reference path
// for any (A, B, bias) input.  Cross-path comparisons should expect
// sub-ULP differences in the "better" direction; the gated-act gtest
// tolerance is sized to accept both.
//
// FP32→BF16 conversion uses the manual `fp32_to_bf16x16_rne` (above)
// instead of the hardware `_mm512_cvtneps_pbh` so the cvt step is
// bit-identical to the standard reference.  See the policy block on
// `fp32_to_bf16x16_rne` for the rationale.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline void swiglu_oai_store_pair(
    __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
  const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
  const __m512i up_idx   = _mm512_load_si512(kUpLaneIdx);

  __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
  __m512 up   = _mm512_permutex2var_ps(acc_lo, up_idx,   acc_hi);

  const __m512 cmin = _mm512_set1_ps(-7.0f);
  const __m512 cmax = _mm512_set1_ps(+7.0f);
  gate = _mm512_max_ps(_mm512_min_ps(gate, cmax), cmin);
  up   = _mm512_max_ps(_mm512_min_ps(up,   cmax), cmin);

  const __m512 alpha = _mm512_set1_ps(1.702f);
  const __m512 one   = _mm512_set1_ps(1.0f);
  __m512 sig = sigmoid_fast(_mm512_mul_ps(gate, alpha));
  __m512 r   = _mm512_mul_ps(_mm512_add_ps(one, up),
                             _mm512_mul_ps(gate, sig));

  // Manual RNE FP32→BF16 — bit-identical to the standard reference's
  // f32_to_bf16x16.  See the policy block above the function for why.
  __m256i out = fp32_to_bf16x16_rne(r);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// ─────────────────────────────────────────────────────────────────────
// Templated microkernel — MR ∈ 1..8, NV ∈ {2, 4}, Act ∈ {none, swiglu}.
//
// `noinline` keeps each (MR, NV, Act) specialization as a single
// callable the dispatcher reaches through a function pointer (matches
// the BRGEMM ukernel pattern).  Each specialization pays its own
// MR×NV register footprint; `noinline` prevents the compiler from
// duplicating the body at every dispatch site.
// ─────────────────────────────────────────────────────────────────────
template <int MR, int NV, ActKind Act>
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma"), noinline))
static void ukernel_impl(
    const bfloat16_t *__restrict__ A, int lda,
    const bfloat16_t *__restrict__ Bpacked,
    const void *__restrict__ bias, BiasKind bias_kind,
    bfloat16_t *__restrict__ Cout, int ldc,
    bfloat16_t *__restrict__ Cout_tight, int ldc_tight,
    int K) {

  static_assert(NV == 2 || NV == 4, "NV must be 2 or 4");
  static_assert(Act == ActKind::none || (NV % 2 == 0),
                "swiglu epilogue requires even NV");

  // ── Accumulator-register budget ──────────────────────────────────
  // Base single-buffer footprint per row is `NV` FP32 zmms.  Small-MR
  // specialisations are FMA-latency bound because the number of
  // independent accumulator chains (MR × NV) is below the critical
  // `FMA_latency × FMA_throughput = 4 × 2 = 8` needed to saturate
  // issue on Zen4/5.  Double-buffering splits the accumulators into
  // two parallel sets fed from even / odd K-pairs; the sets are
  // summed at the end of the K-loop.  This doubles the chain count
  // and brings MR=2, MR=3 from latency-bound to issue-bound.  MR=1
  // still has 4 chains (< 8) so remains partially latency-bound, but
  // its effective FMA/cycle doubles.  For MR ≥ 4 single buffering
  // already saturates issue, so kBuffers stays at 1 (avoids spilling
  // the MR=8 × NV=2 specialisation's 16 acc zmms).
  //
  // Register count after double-buffering (MR ≤ 3, NV=2):
  //   MR=1: 2×2 acc + 2 b + 1 a = 7 zmms (22% of 32)
  //   MR=2: 4×2 acc + 2 b + 1 a = 11 zmms (34%)
  //   MR=3: 6×2 acc + 2 b + 1 a = 15 zmms (47%)
  // All safely under the ~70% rule of thumb.
  constexpr int kBuffers = (MR <= 3) ? 2 : 1;
  __m512 acc[kBuffers][MR][NV];

  // ── Bias fold into accumulator init ──────────────────────────────
  // Rather than running a separate `acc += bias` pass after the
  // K-loop, initialise `acc[0]` with the bias vector directly.  The
  // K-loop then accumulates the GEMM contribution ON TOP of the bias
  // seed.  With `kBuffers == 2` only acc[0] carries bias; acc[1]
  // starts at zero so the post-K-loop combine (`acc[0] += acc[1]`)
  // folds the GEMM contribution without doubling bias.
  //
  // Net saving: MR × NV `_mm512_add_ps` ops + the separate bias-load
  // reload after K-loop.  Both the K-loop FMAs and the init are
  // exactly as many ops as before; we just collapse init + post-add
  // into one init.  Zero-risk: when `bias == nullptr` the zero-init
  // branch produces identical accumulator state as the previous code.
  const bool has_bias = (bias != nullptr && bias_kind != BiasKind::none);
  __m512 bias_vec[NV];
  if (has_bias) {
    if (bias_kind == BiasKind::bf16) {
      const auto *bias_bf16 = static_cast<const bfloat16_t *>(bias);
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        __m256i b16 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(bias_bf16 + v * 16));
        bias_vec[v] = bf16x16_to_fp32(b16);
      }
    } else {
      const auto *bias_fp32 = static_cast<const float *>(bias);
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        bias_vec[v] = _mm512_loadu_ps(bias_fp32 + v * 16);
      }
    }
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        acc[0][m][v] = bias_vec[v];
      }
    }
    if constexpr (kBuffers > 1) {
      #pragma GCC unroll 8
      for (int m = 0; m < MR; ++m) {
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
          acc[1][m][v] = _mm512_setzero_ps();
        }
      }
    }
  } else {
    for (int b = 0; b < kBuffers; ++b)
      for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v)
          acc[b][m][v] = _mm512_setzero_ps();
  }

  const int K_pair = K / 2;
  const bool k_odd = (K & 1) != 0;
  // Stride in BF16 elements between consecutive K-pairs at this
  // o-block (= NR cols × VNNI_PAIR).
  constexpr int kp_stride_bf16 = NV * 16 * kVNNIPair;
  // Per-zmm offset (in BF16 elements) within one (o_blk, kp).
  constexpr int v_stride_bf16 = 16 * kVNNIPair;

  // ── K-pair loop: software-pipelined, unrolled by 2 (kk += 2) ─────
  // We stage the next K-pair's B-loads BEFORE issuing the current
  // K-pair's FMAs.  The loads and FMAs then execute concurrently on
  // the CPU's load and FMA ports: loads drain while the 4-cycle FMA
  // pipeline retires the previous pair's VDPBF16PS chain.  Benefits
  // small-MR specialisations most (FMA-latency bound, idle load
  // issue slots), but costs nothing on MR=8 where OoO width alone
  // already overlapped the two.
  //
  // Per outer iteration (one kp += 2 step):
  //   1. stage-load bv_next = B[kp+1]          (overlaps FMAs below)
  //   2. issue FMAs for kp   using bv_cur
  //   3. stage-load bv_cur  = B[kp+2]          (overlaps FMAs below)
  //   4. issue FMAs for kp+1 using bv_next
  //
  // Double-buffering routing (kBuffers == 2):
  //   u=0 (even K-pair) → acc[0]
  //   u=1 (odd  K-pair) → acc[1]
  // Two independent acc sets double the issue-rate utilisation for
  // small-MR specialisations.  For kBuffers == 1 both u feed acc[0] —
  // the compiler elides the unused acc[1].
  // A-row K-pair broadcast: load 4 bytes (= 2 BF16 K-elements) from A
  // and broadcast them to all 16 lanes of a zmm for VDPBF16PS.  We
  // use `std::memcpy(&uint32, ptr); _mm512_set1_epi32(uint32)` — GCC
  // pattern-matches this idiom into a single `vpbroadcastd zmm, [mem]`
  // instruction.  The alternate intrinsic sequence
  // `_mm_loadu_si32(ptr) + _mm512_broadcastd_epi32(...)` looks
  // tempting but was measurably slower on MoE decode workloads
  // (GCC 12 failed to fuse it into the same single broadcast-load).

  int kp = 0;
  if (kp + 1 < K_pair) {
    __m512bh bv_cur[NV];
    // Prolog: load B for kp=0 so the first outer iteration starts
    // with bv_cur already in flight.  Bpacked is 64-byte aligned by
    // `pack.cpp` (std::aligned_alloc(64, ...)) — use the aligned-load
    // intrinsic here and below so any future alignment regression
    // trips immediately rather than silently costing perf.
    {
      const bfloat16_t *bp = Bpacked
          + static_cast<size_t>(kp) * kp_stride_bf16;
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        bv_cur[v] = (__m512bh)_mm512_load_si512(
            reinterpret_cast<const __m512i *>(bp + v * v_stride_bf16));
      }
    }

    constexpr int buf0 = 0;
    constexpr int buf1 = (kBuffers == 2) ? 1 : 0;

    // ── No software prefetch in the K-loop (intentional) ────────────
    // A `_mm_prefetch(B + N K-pairs ahead, _MM_HINT_T0)` per outer
    // iteration was tried on the hypothesis that explicit prefetch
    // would close the IPC gap to AOCL DLP at high thread counts on
    // MoE decode.  Result: a consistent regression across every
    // num_ops bucket and in aggregate on the same workload.
    //
    // Explanation: Zen 4 / Zen 5's hardware prefetcher detects the
    // streaming (kp_stride-strided) B access pattern reliably; the
    // K-loop is not memory-latency bound at the shapes that reach
    // this kernel.  Adding software prefetch consumed load-port
    // issue slots and polluted L1 with addresses the HW unit was
    // already streaming, hurting FMA dispatch.  The remaining IPC
    // gap vs AOCL is in instruction scheduling (compiler-emitted vs
    // hand-tuned asm), not memory hiding — closing it would require
    // an asm rewrite rather than prefetch hints.  Note kept here so
    // a future optimiser does not re-attempt the same fix.

    for (; kp + 1 < K_pair; kp += 2) {
      // ── Stage 1: load bv_next = B[kp+1], overlaps FMAs for kp ──
      __m512bh bv_next[NV];
      {
        const bfloat16_t *bp1 = Bpacked
            + static_cast<size_t>(kp + 1) * kp_stride_bf16;
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
          bv_next[v] = (__m512bh)_mm512_load_si512(
              reinterpret_cast<const __m512i *>(bp1 + v * v_stride_bf16));
        }
      }
      // ── Stage 2: issue FMAs for kp using bv_cur ───────────────
      #pragma GCC unroll 8
      for (int m = 0; m < MR; ++m) {
        uint32_t a_pair;
        std::memcpy(&a_pair,
                    A + static_cast<size_t>(m) * lda + kp * 2,
                    sizeof(a_pair));
        __m512bh av = (__m512bh)_mm512_set1_epi32(
            static_cast<int>(a_pair));
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v)
          acc[buf0][m][v] =
              _mm512_dpbf16_ps(acc[buf0][m][v], av, bv_cur[v]);
      }
      // ── Stage 3: pre-load bv_cur = B[kp+2] for the NEXT outer
      //    iteration (unless this is the last), overlaps FMAs below.
      if (kp + 3 < K_pair) {
        const bfloat16_t *bp2 = Bpacked
            + static_cast<size_t>(kp + 2) * kp_stride_bf16;
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
          bv_cur[v] = (__m512bh)_mm512_load_si512(
              reinterpret_cast<const __m512i *>(bp2 + v * v_stride_bf16));
        }
      }
      // ── Stage 4: issue FMAs for kp+1 using bv_next ────────────
      #pragma GCC unroll 8
      for (int m = 0; m < MR; ++m) {
        uint32_t a_pair;
        std::memcpy(&a_pair,
                    A + static_cast<size_t>(m) * lda + (kp + 1) * 2,
                    sizeof(a_pair));
        __m512bh av = (__m512bh)_mm512_set1_epi32(
            static_cast<int>(a_pair));
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v)
          acc[buf1][m][v] =
              _mm512_dpbf16_ps(acc[buf1][m][v], av, bv_next[v]);
      }
    }
  }
  // Tail: one remaining full K-pair if K_pair was odd.  Tails always
  // feed acc[0] (single stray iteration, no buffer alternation needed).
  if (kp < K_pair) {
    const bfloat16_t *bp = Bpacked
        + static_cast<size_t>(kp) * kp_stride_bf16;
    __m512bh bv[NV];
    #pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
      bv[v] = (__m512bh)_mm512_load_si512(
          reinterpret_cast<const __m512i *>(bp + v * v_stride_bf16));
    }
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      uint32_t a_pair;
      std::memcpy(&a_pair,
                  A + static_cast<size_t>(m) * lda + kp * 2,
                  sizeof(a_pair));
      __m512bh av = (__m512bh)_mm512_set1_epi32(
          static_cast<int>(a_pair));
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v)
        acc[0][m][v] = _mm512_dpbf16_ps(acc[0][m][v], av, bv[v]);
    }
    ++kp;
  }
  // Tail: one remaining single K element (odd K).  The pack already
  // wrote a zero in the second BF16 slot, so a 4-byte load is safe;
  // we still narrow the broadcast to a single live BF16.  Single-
  // element tail keeps the `memcpy` idiom — only runs when K is odd
  // (most production MoE decode shapes use even K and skip this
  // branch entirely), and uses a uint16_t load that a 4-byte
  // unaligned-load intrinsic can't express without reading past A's
  // end.
  //
  // Cold-correctness note: this branch is exercised in gtest by
  // shapes with odd K (K=1, 3, 5, ...).  It produces bit-identical
  // results to an equivalent single-K-iter reference path (same
  // VDPBF16PS instruction, same operand layout; the pack zero-fills
  // the unused BF16 slot so the high-half contribution is 0).  Not
  // executed by typical production MoE decode shapes (even K), but
  // validated by the ALGO 3 random-K gtest parameterization.
  if (k_odd) {
    const bfloat16_t *bp = Bpacked
        + static_cast<size_t>(kp) * kp_stride_bf16;
    __m512bh bv[NV];
    #pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
      bv[v] = (__m512bh)_mm512_load_si512(
          reinterpret_cast<const __m512i *>(bp + v * v_stride_bf16));
    }
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      uint16_t lo;
      std::memcpy(&lo,
                  A + static_cast<size_t>(m) * lda + kp * 2,
                  sizeof(lo));
      __m512bh av = (__m512bh)_mm512_set1_epi32(
          static_cast<int>(static_cast<uint32_t>(lo)));
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v)
        acc[0][m][v] = _mm512_dpbf16_ps(acc[0][m][v], av, bv[v]);
    }
  }

  // ── Combine accumulator buffers (no-op when kBuffers == 1) ───────
  // Bias was folded into `acc[0]`'s init (see above); the combine
  // here sums the GEMM contributions from the two K-pair interleaves
  // so the bias stays applied exactly once.
  if constexpr (kBuffers > 1) {
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v)
        acc[0][m][v] = _mm512_add_ps(acc[0][m][v], acc[1][m][v]);
    }
  }

  // ── Epilogue ─────────────────────────────────────────────────────
  if constexpr (Act == ActKind::swiglu_oai_mul) {
    constexpr int n_pairs = NV / 2;  // 1 for NV=2, 2 for NV=4
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 2
      for (int p = 0; p < n_pairs; ++p) {
        bfloat16_t *dst = Cout_tight
            + static_cast<size_t>(m) * ldc_tight + p * 16;
        swiglu_oai_store_pair(acc[0][m][2 * p], acc[0][m][2 * p + 1], dst);
      }
    }
  } else {
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        bfloat16_t *dst = Cout
            + static_cast<size_t>(m) * ldc + v * 16;
        // Manual RNE FP32→BF16 (see fp32_to_bf16x16_rne policy block
        // above).  Bit-identical to the standard reference path's
        // f32_to_bf16x16, so the act=none custom kernel produces the
        // same BF16 output as the reference — Op2 (in fused MoE)
        // reads the same bytes whether or not the custom kernel was
        // engaged.
        __m256i out = fp32_to_bf16x16_rne(acc[0][m][v]);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), out);
      }
    }
  }
}

} // namespace

// ── Function-pointer table dispatch (mirrors select_bf16_brgemm_kernel) ─
//
// One entry per supported (MR, NV, Act) triple.  Instantiation set:
//   NV=2 (NR=32): MR ∈ {1..8} × Act ∈ {none, swiglu_oai_mul}
//   NV=4 (NR=64): MR ∈ {1..6} × Act ∈ {none, swiglu_oai_mul}
// Total: 28 specializations, all `noinline`.
ukernel_fn_t select_ukernel(int MR, int NV, ActKind act) {
  if (NV == 2) {
    if (act == ActKind::swiglu_oai_mul) {
      switch (MR) {
        case 1: return ukernel_impl<1, 2, ActKind::swiglu_oai_mul>;
        case 2: return ukernel_impl<2, 2, ActKind::swiglu_oai_mul>;
        case 3: return ukernel_impl<3, 2, ActKind::swiglu_oai_mul>;
        case 4: return ukernel_impl<4, 2, ActKind::swiglu_oai_mul>;
        case 5: return ukernel_impl<5, 2, ActKind::swiglu_oai_mul>;
        case 6: return ukernel_impl<6, 2, ActKind::swiglu_oai_mul>;
        case 7: return ukernel_impl<7, 2, ActKind::swiglu_oai_mul>;
        case 8: return ukernel_impl<8, 2, ActKind::swiglu_oai_mul>;
        default: return nullptr;
      }
    }
    switch (MR) {
      case 1: return ukernel_impl<1, 2, ActKind::none>;
      case 2: return ukernel_impl<2, 2, ActKind::none>;
      case 3: return ukernel_impl<3, 2, ActKind::none>;
      case 4: return ukernel_impl<4, 2, ActKind::none>;
      case 5: return ukernel_impl<5, 2, ActKind::none>;
      case 6: return ukernel_impl<6, 2, ActKind::none>;
      case 7: return ukernel_impl<7, 2, ActKind::none>;
      case 8: return ukernel_impl<8, 2, ActKind::none>;
      default: return nullptr;
    }
  }
  if (NV == 4) {
    if (act == ActKind::swiglu_oai_mul) {
      switch (MR) {
        case 1: return ukernel_impl<1, 4, ActKind::swiglu_oai_mul>;
        case 2: return ukernel_impl<2, 4, ActKind::swiglu_oai_mul>;
        case 3: return ukernel_impl<3, 4, ActKind::swiglu_oai_mul>;
        case 4: return ukernel_impl<4, 4, ActKind::swiglu_oai_mul>;
        case 5: return ukernel_impl<5, 4, ActKind::swiglu_oai_mul>;
        case 6: return ukernel_impl<6, 4, ActKind::swiglu_oai_mul>;
        default: return nullptr;
      }
    }
    switch (MR) {
      case 1: return ukernel_impl<1, 4, ActKind::none>;
      case 2: return ukernel_impl<2, 4, ActKind::none>;
      case 3: return ukernel_impl<3, 4, ActKind::none>;
      case 4: return ukernel_impl<4, 4, ActKind::none>;
      case 5: return ukernel_impl<5, 4, ActKind::none>;
      case 6: return ukernel_impl<6, 4, ActKind::none>;
      default: return nullptr;
    }
  }
  return nullptr;
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
