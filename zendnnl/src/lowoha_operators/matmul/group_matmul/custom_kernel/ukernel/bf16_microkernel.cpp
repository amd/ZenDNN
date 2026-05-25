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

#include "bf16_microkernel.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>

#include <immintrin.h>

#include "lowoha_operators/matmul/group_matmul/group_matmul_act_avx512.hpp"
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
//
// The FP32 math primitives (`fast_exp_neg_avx512`, `sigmoid_avx512`,
// `silu_avx512`, `gelu_avx512`, `swiglu_oai_avx512`) and the BF16↔FP32
// cvt helpers (`bf16x16_to_f32`, `f32_to_bf16x16`) live in
// `group_matmul_act_avx512.hpp` — the same header the separate-pass
// `group_matmul_moe_act.cpp` consumes.  Pulled in via using-
// declarations below so the in-register store helpers
// (`swiglu_oai_store_pair`, `silu_and_mul_store_pair`,
// `gelu_and_mul_store_pair`) share bit-for-bit identical math with
// the standard reference's row helpers.  See the header doc-block
// for the cross-path numerical contract.
using zendnnl::lowoha::matmul::group_matmul_act_avx512::bf16x16_to_f32;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::f32_to_bf16x16;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::sigmoid_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::silu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::gelu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::swiglu_oai_avx512;

// Deinterleave indices for vpermt2ps over two source zmms (= 32 FP32
// lanes total).  Even-indexed lanes (0, 2, ..., 30) are gates; odd-
// indexed lanes (1, 3, ..., 31) are ups.  File-local because the
// constants are use-case specific — the moe_act row helpers
// deinterleave from MEMORY (via vpgatherdd / vpermtxvar_epi16) and
// use different index types.
alignas(64) constexpr int32_t kGateLaneIdx[16] =
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) constexpr int32_t kUpLaneIdx[16] =
    {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

// Apply swiglu_oai_mul in registers to one (gate, up) pair of FP32
// accumulators and store 16 activated BF16 cols.  `dst_row` is the
// tight destination row pointer (caller pre-offsets for col_start/2
// and the within-row epilogue index `p`).
//
// ── Structure ────────────────────────────────────────────────────────
//   1. Deinterleave gate/up out of the accumulator pair (`vpermt2ps`).
//   2. Call the shared `swiglu_oai_avx512(gate, up)` from
//      `group_matmul_act_avx512.hpp` — IDENTICAL math to the standard
//      reference's row helpers in `group_matmul_moe_act.cpp` (same
//      5-term Cephes exp(-x), same rcp14 + 1-step Newton-Raphson
//      sigmoid, same clamp bounds [-7, +7], same multiply order
//      `(1 + up) * (gate * sigmoid(gate * 1.702f))`).
//   3. Manual RNE FP32→BF16 via the shared `f32_to_bf16x16` (bit-
//      identical to the reference path's cvt step) and 256-bit store.
//
// ── Precision characteristic vs the standard two-pass pipeline ──────
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
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline void swiglu_oai_store_pair(
    __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
  const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
  const __m512i up_idx   = _mm512_load_si512(kUpLaneIdx);

  __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
  __m512 up   = _mm512_permutex2var_ps(acc_lo, up_idx,   acc_hi);

  __m512 r = swiglu_oai_avx512(gate, up);

  __m256i out = f32_to_bf16x16(r);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// Apply gelu_and_mul in registers to one (gate, up) pair of FP32
// accumulators and store 16 activated BF16 cols.
//
// ── Structure ────────────────────────────────────────────────────────
//   1. Deinterleave gate/up out of the accumulator pair (`vpermt2ps`).
//   2. Call the shared `gelu_avx512(gate)` from
//      `group_matmul_act_avx512.hpp` — same polynomial form used by
//      the standard reference's `gelu_and_mul_row_avx512_*` helpers
//      in `group_matmul_moe_act.cpp`, so both paths produce bit-
//      identical activation output.
//   3. Multiply by `up` and store as BF16 via the shared
//      `f32_to_bf16x16` integer-RNE sequence.
//
// ── Numerical contract ───────────────────────────────────────────────
// Mathematically equivalent (within BF16 tolerance) to the textbook
// `gelu_and_mul` reference:
//
//   reference:  out = gelu_erf(gate) * up
//               gelu_erf(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// The shared `gelu_avx512` uses the well-known `gelu_tanh`
// polynomial approximation:
//
//               y = sqrt(2/pi) * (x + 0.044715 * x^3)
//               gelu_tanh(x) = 0.5 * x * (1 + tanh(y))
//
// rewritten via `tanh(y) = 2·sigmoid(2y) − 1` so the form is one
// `sigmoid_avx512` call:
//
//               gelu_tanh(x) = x * sigmoid(2y)
//               y2 = c1*x + c2*x^3 ; c1 ≈ 1.5957691, c2 ≈ 0.0713548
//               g'  = x * sigmoid_avx512(y2)
//               out = g' * up
//
// Numerical fidelity:
//   * |gelu_tanh − gelu_erf| ≤ 1.5e-3 across all real x (well-known).
//   * BF16 ulp ≈ 7.8e-3 relative; so the gelu_tanh ↔ gelu_erf delta
//     is about 5× tighter than BF16 can represent.
//   * `mt::tol_act(/*is_bf16=*/true)` band is {rel=0.15, abs=0.02},
//     so the kernel passes the existing reference comparison with
//     >10× margin.
//
// Why gelu_tanh (polynomial) instead of gelu_erf (libc):
//   * The reference's "AVX-512" gelu helper
//     (`gelu_and_mul_row_avx512_*` in group_matmul_moe_act.cpp) is
//     itself a thin wrapper around per-lane `std::erf` — it stores
//     the FP32 vector to stack, calls libc 16 times, and reloads.
//     That's the slow path the fused kernel is replacing.
//   * Calling `std::erf` from inside the matmul ukernel would defeat
//     the entire fusion (function-call overhead per output row).
//   * gelu_tanh is the form used by GPT-2/BERT/PaLM; gelu_erf and
//     gelu_tanh are interchangeable for any production model
//     decoding into BF16 dst.
//
// Differences from `silu_and_mul_store_pair`:
//   * Three FMAs to compute `y2 = c1·x + c2·x³` (vs zero in silu).
//   * Otherwise identical: same `sigmoid_avx512`, same `gate * up`
//     finish, same `f32_to_bf16x16` store path.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline void gelu_and_mul_store_pair(
    __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
  const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
  const __m512i up_idx   = _mm512_load_si512(kUpLaneIdx);

  __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
  __m512 up   = _mm512_permutex2var_ps(acc_lo, up_idx,   acc_hi);

  __m512 r = _mm512_mul_ps(gelu_avx512(gate), up);

  __m256i out = f32_to_bf16x16(r);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// Apply silu_and_mul in registers to one (gate, up) pair of FP32
// accumulators and store 16 activated BF16 cols.
//
// ── Structure ────────────────────────────────────────────────────────
//   1. Deinterleave gate/up out of the accumulator pair (`vpermt2ps`).
//   2. Call the shared `silu_avx512(gate)` from
//      `group_matmul_act_avx512.hpp` — same math as the standard
//      reference's `silu_and_mul_row_avx512_*` helpers in
//      `group_matmul_moe_act.cpp`, so both paths produce bit-
//      identical activation output (modulo the elimination of one
//      BF16 round-trip in the fused path; same direction as swiglu).
//   3. Multiply by `up` and store as BF16 via the shared
//      `f32_to_bf16x16` integer-RNE sequence.
//
// ── Numerical contract ───────────────────────────────────────────────
//   out = silu(gate) * up,   silu(x) = x * sigmoid(x)
//
// Differences from `swiglu_oai_store_pair`:
//   * No clamp on gate / up (silu is well-conditioned for any input).
//   * No alpha multiplier on gate (swiglu_oai uses α=1.702; standard
//     silu uses α=1, i.e. no multiplier).
//   * Final multiply is `silu(gate) * up`, not `(1+up) * (gate * sig)`.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline void silu_and_mul_store_pair(
    __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
  const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
  const __m512i up_idx   = _mm512_load_si512(kUpLaneIdx);

  __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
  __m512 up   = _mm512_permutex2var_ps(acc_lo, up_idx,   acc_hi);

  __m512 r = _mm512_mul_ps(silu_avx512(gate), up);

  __m256i out = f32_to_bf16x16(r);
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
template <int MR, int NV, ActKind Act, typename DstT>
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma"), noinline))
static void ukernel_impl(
    const bfloat16_t *__restrict__ A, int lda,
    const bfloat16_t *__restrict__ Bpacked,
    const void *__restrict__ bias, BiasKind bias_kind,
    void *__restrict__ Cout_void, int ldc,
    void *__restrict__ Cout_tight_void, int ldc_tight,
    int K) {

  static_assert(NV == 2 || NV == 4, "NV must be 2 or 4");
  static_assert(Act == ActKind::none || (NV % 2 == 0),
                "gated-activation epilogue requires even NV");
  static_assert(std::is_same<DstT, bfloat16_t>::value
                    || std::is_same<DstT, float>::value,
                "ukernel_impl: DstT must be bfloat16_t or float");
  // The gated-activation pair-pack store helpers
  // (`swiglu_oai_store_pair`, `silu_and_mul_store_pair`,
  // `gelu_and_mul_store_pair`) write 16 BF16 lanes per (gate, up)
  // pair via `_mm256_storeu_si256` — no FP32 counterpart today, and
  // the downstream consumer (Op2 src in fused MoE) reads BF16.
  // Refusing the (gated_act, FP32) tuple at compile time keeps
  // `select_ukernel` from accidentally returning a mis-typed
  // instantiation; runtime dispatch refuses earlier still (see
  // `select_ukernel` in this file + the prepare_for_call gate).
  static_assert(Act == ActKind::none
                    || std::is_same<DstT, bfloat16_t>::value,
                "Gated-activation kinds (swiglu_oai_mul, silu_and_mul, "
                "gelu_and_mul) are BF16-dst only");

  // Reinterpret the dispatcher's `void *` outputs as the templated
  // dst type.  ldc / ldc_tight stay in element units (not bytes) —
  // pointer arithmetic below uses them with implicit DstT scaling.
  DstT *__restrict__ Cout       = static_cast<DstT *>(Cout_void);
  DstT *__restrict__ Cout_tight = static_cast<DstT *>(Cout_tight_void);

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
        bias_vec[v] = bf16x16_to_f32(b16);
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
  // tempting but did not collapse to the same single broadcast-load
  // on the compilers we tested, so the `memcpy` idiom is preferred.

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
    // iteration was tried.  It regressed throughput because the Zen
    // 4 / Zen 5 hardware prefetcher already detects the streaming
    // (kp_stride-strided) B access pattern reliably and the K-loop
    // is not memory-latency bound at the shapes that reach this
    // kernel.  Adding software prefetch consumed load-port issue
    // slots and polluted L1 with addresses the hardware unit was
    // already streaming, which hurt FMA dispatch.  Note kept here
    // so a future contributor does not re-attempt the same fix.

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
  // (typical MoE decode shapes use even K and skip this branch
  // entirely), and uses a uint16_t load that a 4-byte unaligned-
  // load intrinsic can't express without reading past A's end.
  //
  // Cold-correctness note: this branch is exercised in gtest by
  // shapes with odd K (K=1, 3, 5, ...).  It produces bit-identical
  // results to an equivalent single-K-iter reference path (same
  // VDPBF16PS instruction, same operand layout; the pack zero-fills
  // the unused BF16 slot so the high-half contribution is 0).  Not
  // executed by typical even-K shapes, but validated by the ALGO 3
  // random-K gtest parameterization.
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
  // All three gated activations share the pair-pack store contract —
  // they differ only in which helper deinterleaves+activates the
  // (g, u) accumulator pair.  Same loop structure, same dst pointer
  // arithmetic, same BF16-dst constraint (enforced by the
  // static_assert above).
  if constexpr (Act == ActKind::swiglu_oai_mul
                  || Act == ActKind::silu_and_mul
                  || Act == ActKind::gelu_and_mul) {
    constexpr int n_pairs = NV / 2;  // 1 for NV=2, 2 for NV=4
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 2
      for (int p = 0; p < n_pairs; ++p) {
        bfloat16_t *dst = Cout_tight
            + static_cast<size_t>(m) * ldc_tight + p * 16;
        if constexpr (Act == ActKind::swiglu_oai_mul) {
          swiglu_oai_store_pair(acc[0][m][2 * p],
                                acc[0][m][2 * p + 1], dst);
        } else if constexpr (Act == ActKind::silu_and_mul) {
          silu_and_mul_store_pair(acc[0][m][2 * p],
                                   acc[0][m][2 * p + 1], dst);
        } else {  // ActKind::gelu_and_mul
          gelu_and_mul_store_pair(acc[0][m][2 * p],
                                   acc[0][m][2 * p + 1], dst);
        }
      }
    }
  } else if constexpr (std::is_same<DstT, bfloat16_t>::value) {
    // Act = none, BF16 dst — manual RNE FP32→BF16 and a 256-bit
    // store per (m, v) tile.  Bit-identical to the standard
    // reference path's `f32_to_bf16x16` (see policy block above).
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        bfloat16_t *dst = Cout
            + static_cast<size_t>(m) * ldc + v * 16;
        __m256i out = f32_to_bf16x16(acc[0][m][v]);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), out);
      }
    }
  } else {
    // Act = none, FP32 dst — store the FP32 accumulator lanes
    // directly with `_mm512_storeu_ps`.  No conversion, no
    // rounding, so the output is bit-exact w.r.t. the FP32
    // accumulator that produced it (modulo BF16-input quantisation
    // upstream).  Useful for callers that need FP32 dst (e.g.,
    // Op2 of a fused MoE chain that wants FP32 reduce-input).
    static_assert(std::is_same<DstT, float>::value,
                  "Act=none non-bf16 dst path requires DstT == float");
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        float *dst = Cout
            + static_cast<size_t>(m) * ldc + v * 16;
        _mm512_storeu_ps(dst, acc[0][m][v]);
      }
    }
  }
}

} // namespace

// ── Function-pointer table dispatch (mirrors select_bf16_brgemm_kernel) ─
//
// One entry per supported (MR, NV, Act, DstDt) tuple.  Instantiation set:
//   NV=2 (NR=32): MR ∈ {1..8} × {(none, BF16), (none, F32), (swiglu, BF16),
//                                  (silu, BF16), (gelu, BF16)}
//   NV=4 (NR=64): MR ∈ {1..6} × {(none, BF16), (none, F32), (swiglu, BF16),
//                                  (silu, BF16), (gelu, BF16)}
// Total: 8×5 + 6×5 = 70 specializations, all `noinline`.
//
// Any (gated_act, FP32) tuple is intentionally NOT instantiated — the
// in-register pair-pack store helpers write BF16 only, and downstream
// Op2 in fused MoE reads BF16.  The dispatcher refuses those tuples at
// `prepare_for_call` time (see the `select_ukernel` early-out below
// returning nullptr for that case).
ukernel_fn_t select_ukernel(int MR, int NV, ActKind act, DstDt dst_dt) {
  // Refuse the structurally-impossible (gated_act, FP32) combinations
  // before the per-(MR, NV) switch so callers see a clean nullptr.
  const bool is_gated_act = (act == ActKind::swiglu_oai_mul
                              || act == ActKind::silu_and_mul
                              || act == ActKind::gelu_and_mul);
  if (is_gated_act && dst_dt != DstDt::kBf16) {
    return nullptr;
  }
  if (NV == 2) {
    if (act == ActKind::swiglu_oai_mul) {
      switch (MR) {
        case 1: return ukernel_impl<1, 2, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 2: return ukernel_impl<2, 2, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 3: return ukernel_impl<3, 2, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 4: return ukernel_impl<4, 2, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 5: return ukernel_impl<5, 2, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 6: return ukernel_impl<6, 2, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 7: return ukernel_impl<7, 2, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 8: return ukernel_impl<8, 2, ActKind::swiglu_oai_mul, bfloat16_t>;
        default: return nullptr;
      }
    }
    if (act == ActKind::silu_and_mul) {
      switch (MR) {
        case 1: return ukernel_impl<1, 2, ActKind::silu_and_mul, bfloat16_t>;
        case 2: return ukernel_impl<2, 2, ActKind::silu_and_mul, bfloat16_t>;
        case 3: return ukernel_impl<3, 2, ActKind::silu_and_mul, bfloat16_t>;
        case 4: return ukernel_impl<4, 2, ActKind::silu_and_mul, bfloat16_t>;
        case 5: return ukernel_impl<5, 2, ActKind::silu_and_mul, bfloat16_t>;
        case 6: return ukernel_impl<6, 2, ActKind::silu_and_mul, bfloat16_t>;
        case 7: return ukernel_impl<7, 2, ActKind::silu_and_mul, bfloat16_t>;
        case 8: return ukernel_impl<8, 2, ActKind::silu_and_mul, bfloat16_t>;
        default: return nullptr;
      }
    }
    if (act == ActKind::gelu_and_mul) {
      switch (MR) {
        case 1: return ukernel_impl<1, 2, ActKind::gelu_and_mul, bfloat16_t>;
        case 2: return ukernel_impl<2, 2, ActKind::gelu_and_mul, bfloat16_t>;
        case 3: return ukernel_impl<3, 2, ActKind::gelu_and_mul, bfloat16_t>;
        case 4: return ukernel_impl<4, 2, ActKind::gelu_and_mul, bfloat16_t>;
        case 5: return ukernel_impl<5, 2, ActKind::gelu_and_mul, bfloat16_t>;
        case 6: return ukernel_impl<6, 2, ActKind::gelu_and_mul, bfloat16_t>;
        case 7: return ukernel_impl<7, 2, ActKind::gelu_and_mul, bfloat16_t>;
        case 8: return ukernel_impl<8, 2, ActKind::gelu_and_mul, bfloat16_t>;
        default: return nullptr;
      }
    }
    // act == ActKind::none — branch on DstDt for the store epilogue.
    if (dst_dt == DstDt::kF32) {
      switch (MR) {
        case 1: return ukernel_impl<1, 2, ActKind::none, float>;
        case 2: return ukernel_impl<2, 2, ActKind::none, float>;
        case 3: return ukernel_impl<3, 2, ActKind::none, float>;
        case 4: return ukernel_impl<4, 2, ActKind::none, float>;
        case 5: return ukernel_impl<5, 2, ActKind::none, float>;
        case 6: return ukernel_impl<6, 2, ActKind::none, float>;
        case 7: return ukernel_impl<7, 2, ActKind::none, float>;
        case 8: return ukernel_impl<8, 2, ActKind::none, float>;
        default: return nullptr;
      }
    }
    switch (MR) {
      case 1: return ukernel_impl<1, 2, ActKind::none, bfloat16_t>;
      case 2: return ukernel_impl<2, 2, ActKind::none, bfloat16_t>;
      case 3: return ukernel_impl<3, 2, ActKind::none, bfloat16_t>;
      case 4: return ukernel_impl<4, 2, ActKind::none, bfloat16_t>;
      case 5: return ukernel_impl<5, 2, ActKind::none, bfloat16_t>;
      case 6: return ukernel_impl<6, 2, ActKind::none, bfloat16_t>;
      case 7: return ukernel_impl<7, 2, ActKind::none, bfloat16_t>;
      case 8: return ukernel_impl<8, 2, ActKind::none, bfloat16_t>;
      default: return nullptr;
    }
  }
  if (NV == 4) {
    if (act == ActKind::swiglu_oai_mul) {
      switch (MR) {
        case 1: return ukernel_impl<1, 4, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 2: return ukernel_impl<2, 4, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 3: return ukernel_impl<3, 4, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 4: return ukernel_impl<4, 4, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 5: return ukernel_impl<5, 4, ActKind::swiglu_oai_mul, bfloat16_t>;
        case 6: return ukernel_impl<6, 4, ActKind::swiglu_oai_mul, bfloat16_t>;
        default: return nullptr;
      }
    }
    if (act == ActKind::silu_and_mul) {
      switch (MR) {
        case 1: return ukernel_impl<1, 4, ActKind::silu_and_mul, bfloat16_t>;
        case 2: return ukernel_impl<2, 4, ActKind::silu_and_mul, bfloat16_t>;
        case 3: return ukernel_impl<3, 4, ActKind::silu_and_mul, bfloat16_t>;
        case 4: return ukernel_impl<4, 4, ActKind::silu_and_mul, bfloat16_t>;
        case 5: return ukernel_impl<5, 4, ActKind::silu_and_mul, bfloat16_t>;
        case 6: return ukernel_impl<6, 4, ActKind::silu_and_mul, bfloat16_t>;
        default: return nullptr;
      }
    }
    if (act == ActKind::gelu_and_mul) {
      switch (MR) {
        case 1: return ukernel_impl<1, 4, ActKind::gelu_and_mul, bfloat16_t>;
        case 2: return ukernel_impl<2, 4, ActKind::gelu_and_mul, bfloat16_t>;
        case 3: return ukernel_impl<3, 4, ActKind::gelu_and_mul, bfloat16_t>;
        case 4: return ukernel_impl<4, 4, ActKind::gelu_and_mul, bfloat16_t>;
        case 5: return ukernel_impl<5, 4, ActKind::gelu_and_mul, bfloat16_t>;
        case 6: return ukernel_impl<6, 4, ActKind::gelu_and_mul, bfloat16_t>;
        default: return nullptr;
      }
    }
    // act == ActKind::none — branch on DstDt for the store epilogue.
    if (dst_dt == DstDt::kF32) {
      switch (MR) {
        case 1: return ukernel_impl<1, 4, ActKind::none, float>;
        case 2: return ukernel_impl<2, 4, ActKind::none, float>;
        case 3: return ukernel_impl<3, 4, ActKind::none, float>;
        case 4: return ukernel_impl<4, 4, ActKind::none, float>;
        case 5: return ukernel_impl<5, 4, ActKind::none, float>;
        case 6: return ukernel_impl<6, 4, ActKind::none, float>;
        default: return nullptr;
      }
    }
    switch (MR) {
      case 1: return ukernel_impl<1, 4, ActKind::none, bfloat16_t>;
      case 2: return ukernel_impl<2, 4, ActKind::none, bfloat16_t>;
      case 3: return ukernel_impl<3, 4, ActKind::none, bfloat16_t>;
      case 4: return ukernel_impl<4, 4, ActKind::none, bfloat16_t>;
      case 5: return ukernel_impl<5, 4, ActKind::none, bfloat16_t>;
      case 6: return ukernel_impl<6, 4, ActKind::none, bfloat16_t>;
      default: return nullptr;
    }
  }
  return nullptr;
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
