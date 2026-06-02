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

/// DQ-INT8 custom microkernel implementation — templated on
/// (MR, NV, Compute, Act).  See int8_microkernel.hpp for the role
/// the kernel plays in the group_matmul dispatch stack.
///
/// PER K-QUAD INNER LOOP (VPDPBUSD, AVX-512 VNNI):
///
///   load Bpacked → NV zmms (b[v] covers cols v*16..(v+1)*16-1,
///                            each a packed 16×4-byte VNNI block)
///   for m in 0..MR-1:
///     broadcast A[m, kq*4 : kq*4+4] (4 bytes = 4 K-elements) into
///       16 s32 lanes  (XOR-ed with 0x80808080 for Compute=kS8_Sym)
///     for v in 0..NV-1:
///       acc[m][v] = vpdpbusd(acc[m][v], A_reg, b[v])   // s32 += u8 ⋅ s8 ×4
///
/// EPILOGUE (per (m, v) tile):
///
///   1. Compensation correction (one s32 subtract per (m, v) tile)
///      * kS8_Sym  : acc -= 128         * sum_wei[v]
///      * kU8_Asym : acc -= src_zp[m]   * sum_wei[v]
///   2. s32 → f32 via `_mm512_cvtepi32_ps`.
///   3. Multiply by `src_scale[m]` (scalar broadcast) and
///      `wei_scale[v]` (16-lane load).
///   4. Add bias[v] if present (loaded from `bias` per `bias_kind`).
///   5. Either
///        a. store as BF16 to `Cout` (Act = none, full NR cols), OR
///        b. apply gated activation in registers and store the
///           halved BF16 output via the same in-register pair-pack
///           helpers the bf16 microkernel uses (Act = swiglu /
///           silu / gelu_and_mul).
///
/// Why the compensation row instead of a per-tile arithmetic shift:
/// the K-loop hot path stays a pure VPDPBUSD reduction (no extra
/// vector adds or broadcasts), and the comp row is precomputed
/// once at pack time alongside the packed weight bytes (see
/// pack.cpp::pack_int8_vnni).  Same pattern AOCL DLP's
/// `aocl_reorder_s8s8s32os32_sym_quant` uses.

#include "int8_microkernel.hpp"

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

bool avx512vnni_available() {
  static const bool v = []() {
    return zendnnl::lowoha::matmul::native::detect_uarch().avx512vnni;
  }();
  return v;
}

namespace {

// ── Activation epilogue helpers (AVX-512) ───────────────────────────
//
// The FP32 math primitives (`sigmoid_avx512`, `silu_avx512`,
// `gelu_avx512`, `swiglu_oai_avx512`) and the FP32 → BF16 cvt
// (`f32_to_bf16x16`) live in `group_matmul_act_avx512.hpp` — the
// same header the bf16 microkernel and the separate-pass
// `group_matmul_moe_act.cpp` consume.  Pulled in via using
// declarations below so the int8-side in-register store helpers
// share bit-for-bit identical math with the existing bf16-side
// helpers (`swiglu_oai_store_pair`, `silu_and_mul_store_pair`,
// `gelu_and_mul_store_pair` in `bf16_microkernel.cpp`).  Numerical
// contract is therefore the same as bf16's: the in-register pair-
// store epilogue skips one BF16 round-trip vs the standard
// reference path; any cross-path comparison should expect sub-ulp
// differences in the "better" direction.
using zendnnl::lowoha::matmul::group_matmul_act_avx512::f32_to_bf16x16;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::silu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::gelu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::swiglu_oai_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::bf16x16_to_f32;

// Deinterleave indices for vpermt2ps over two source zmms (= 32
// FP32 lanes total).  Even-indexed lanes (0, 2, ..., 30) are gates;
// odd-indexed lanes (1, 3, ..., 31) are ups.  Duplicate of the
// bf16-side file-local constants because both are file-local
// (anonymous namespace) and exposing a shared header just for
// these two 16-int arrays would be over-engineering — same values,
// same alignment, different translation units.
alignas(64) constexpr int32_t kGateLaneIdx[16] =
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) constexpr int32_t kUpLaneIdx[16] =
    {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

// Apply swiglu_oai_mul in registers to one (gate, up) pair of FP32
// accumulators and store 16 activated BF16 cols.  Mirror of the
// bf16-side `swiglu_oai_store_pair` — same math (vpermt2ps
// deinterleave, swiglu_oai math, f32_to_bf16x16 cvt), so any DQ-INT8
// gtest case that asserts cross-path equivalence against the bf16
// CK path or the separate-pass reference passes with the same
// tolerance.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline void swiglu_oai_store_pair_int(
    __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
  const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
  const __m512i up_idx   = _mm512_load_si512(kUpLaneIdx);
  __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
  __m512 up   = _mm512_permutex2var_ps(acc_lo, up_idx,   acc_hi);
  __m512 r = swiglu_oai_avx512(gate, up);
  __m256i out = f32_to_bf16x16(r);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// silu_and_mul in registers — `silu(gate) * up`.  Same pair-pack
// store contract as swiglu_oai; mirror of the bf16-side
// `silu_and_mul_store_pair`.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
static inline void silu_and_mul_store_pair_int(
    __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
  const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
  const __m512i up_idx   = _mm512_load_si512(kUpLaneIdx);
  __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
  __m512 up   = _mm512_permutex2var_ps(acc_lo, up_idx,   acc_hi);
  __m512 r = _mm512_mul_ps(silu_avx512(gate), up);
  __m256i out = f32_to_bf16x16(r);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// gelu_and_mul in registers — `gelu_erf(gate) * up` (same `erf`-
// based polynomial as `bf16_microkernel.cpp::gelu_and_mul_store_pair`).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static inline void gelu_and_mul_store_pair_int(
    __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
  const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
  const __m512i up_idx   = _mm512_load_si512(kUpLaneIdx);
  __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
  __m512 up   = _mm512_permutex2var_ps(acc_lo, up_idx,   acc_hi);
  __m512 r = _mm512_mul_ps(gelu_avx512(gate), up);
  __m256i out = f32_to_bf16x16(r);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// Finish the per-(m, v) dequant: subtract the precomputed s32
// compensation, convert to FP32, scale by per-row `src_scale` and
// per-channel `wei_scale`, and add the optional bias.  Factored out
// of the epilogue so the gated / bf16-dst / f32-dst store paths share
// one definition.  A free `static` function (not a lambda) so it can
// carry its own `target` attribute — a lambda does not inherit
// `ukernel_impl`'s, and the AVX-512 intrinsics would fail to inline.
// The caller selects the compensation term (`128 * sum_wei` for sym,
// `src_zp * sum_wei` for asym) so this helper stays compute-agnostic.
__attribute__((target("avx512f,avx512dq,fma")))
static inline __m512 dequant_finish(
    __m512i acc_s32, __m512i correction,
    __m512 src_scale_m, __m512 wei_scale_v,
    __m512 bias_v, bool has_bias) {
  __m512 f = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc_s32, correction));
  f = _mm512_mul_ps(f, src_scale_m);
  f = _mm512_mul_ps(f, wei_scale_v);
  if (has_bias) f = _mm512_add_ps(f, bias_v);
  return f;
}

// ─────────────────────────────────────────────────────────────────────
// Templated microkernel — MR ∈ 1..8, NV ∈ {2, 4},
// Compute ∈ {kS8_Sym, kU8_Asym}, Act ∈ {none, swiglu, silu, gelu}.
//
// `noinline` keeps each (MR, NV, Compute, Act) specialization as a
// single callable the dispatcher reaches through a function pointer
// (matches the bf16 sibling).  Each specialization pays its own
// MR×NV s32 accumulator register footprint; `noinline` prevents the
// compiler from duplicating the body at every dispatch site.
// ─────────────────────────────────────────────────────────────────────
template <int MR, int NV, IntCompute Compute, ActKind Act,
          DstDt Dst = DstDt::kBf16>
__attribute__((target("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma"),
               noinline))
static void ukernel_impl(
    const uint8_t *__restrict__ A, int lda,
    const int8_t  *__restrict__ Bpacked,
    const void    *__restrict__ src_scale,
    const int32_t *__restrict__ src_zp,
    const void    *__restrict__ wei_scale,
    ScaleKind                   scale_kind,
    const void    *__restrict__ bias, BiasKind bias_kind,
    void          *__restrict__ Cout_void, int ldc,
    void          *__restrict__ Cout_tight_void, int ldc_tight,
    int            K) {

  // Scalar per-row src_scale read with on-load bf16→f32 widening.  The
  // kernel dequantises in f32; f32 scales read directly, bf16 scales
  // convert here (off the hot K-loop).  Mirrors the wei_scale branch
  // in the epilogue and the BiasKind bf16-bias load.
  const auto src_scale_at = [src_scale, scale_kind](int m) -> float {
    return (scale_kind == ScaleKind::kBf16)
        ? static_cast<float>(static_cast<const bfloat16_t *>(src_scale)[m])
        : static_cast<const float *>(src_scale)[m];
  };

  static_assert(NV == 2 || NV == 4, "NV must be 2 or 4");
  static_assert(Act == ActKind::none || (NV % 2 == 0),
                "gated-activation epilogue requires even NV");
  static_assert(Compute == IntCompute::kS8_Sym
                    || Compute == IntCompute::kU8_Asym,
                "Compute must be kS8_Sym or kU8_Asym");
  // Gated-activation kinds write the half-width BF16 output through
  // the in-register pair-store helpers — mirror of the bf16 sibling,
  // FP32 dst is only valid for `Act = none`.  `select_int8_ukernel`
  // returns nullptr for any (gated, kF32) tuple so the dispatcher's
  // `fill_kfn_table_int8` refuses the call cleanly before reaching
  // here; the static_assert is the compile-time backstop.
  static_assert(Act == ActKind::none || Dst == DstDt::kBf16,
                "Gated-activation kinds (swiglu_oai_mul, silu_and_mul, "
                "gelu_and_mul) are BF16-dst only");

  // ── K-quad loop accounting ──────────────────────────────────────
  // K is in K-element units; the pack rounds up to a multiple of
  // kVNNIInt8Quad (= 4) and zero-pads the trailing slot, so a
  // 4-byte VPDPBUSD broadcast is always safe.  We process all
  // `K_quad = ceil(K / 4)` quads — the pack's tail-zero contract
  // guarantees the trailing partial quad contributes zero to the
  // dot product.
  const int K_quad = (K + kVNNIInt8Quad - 1) / kVNNIInt8Quad;
  // Stride in BYTES between consecutive K-quads at this o-block
  // (= NV × 16 cols × kVNNIInt8Quad bytes per col).
  constexpr int kq_stride_bytes = NV * 16 * kVNNIInt8Quad;
  // Per-zmm offset (in BYTES) within one (o_blk, kq).
  constexpr int v_stride_bytes  = 16 * kVNNIInt8Quad;
  // Bytes consumed by all K-quads in this o-block — the
  // compensation row starts immediately after.
  const int weight_bytes_in_oblock = K_quad * kq_stride_bytes;

  // ── Accumulator register budget + double-buffering ─────────────
  // Base single-buffer footprint per row is `NV` s32 zmms.  Small-MR
  // specialisations are FMA-latency bound: the number of independent
  // accumulator chains (MR × NV) is below what is needed to saturate
  // the VPDPBUSD pipeline (5-cycle latency on Zen4 / 4-cycle on Zen5,
  // 2/cycle throughput).  Mirroring the bf16 sibling, double-buffering
  // splits the s32 accumulators into two parallel sets fed from
  // even / odd K-quads; the sets are summed (s32 add) at the end of
  // the K-loop, BEFORE the compensation correction.  This doubles the
  // chain count and brings MR=2, MR=3 from latency-bound to issue-
  // bound.  For MR ≥ 4 single buffering already saturates issue, so
  // kBuffers stays at 1 (avoids spilling the MR=8 × NV=2 spec's
  // 16 acc zmms).
  //
  // Register count after double-buffering (MR ≤ 3, NV=2):
  //   MR=1: 2×2 acc + 2 b + 1 a = 7 zmms
  //   MR=2: 4×2 acc + 2 b + 1 a = 11 zmms
  //   MR=3: 6×2 acc + 2 b + 1 a = 15 zmms
  // All under the ~70% of 32-zmm rule of thumb.
  //
  // Bias is folded into the FP32 epilogue, NOT the s32 accumulator
  // (the s32 path has no clean way to add a float bias), so both
  // buffers simply start at zero — no bias-seed asymmetry like the
  // bf16 sibling has to manage.
  constexpr int kBuffers = (MR <= 3) ? 2 : 1;
  __m512i acc[kBuffers][MR][NV];
  #pragma GCC unroll 2
  for (int b = 0; b < kBuffers; ++b) {
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        acc[b][m][v] = _mm512_setzero_si512();
      }
    }
  }
  // Even K-quads accumulate into buffer 0, odd K-quads into buffer 1
  // (buffer 1 collapses onto 0 when kBuffers == 1 — the compiler
  // elides the unused second set).
  constexpr int buf0 = 0;
  constexpr int buf1 = (kBuffers == 2) ? 1 : 0;

  // For Compute=kS8_Sym the K-broadcast must be biased to u8 so
  // VPDPBUSD's `(unsigned × signed)` operand ordering holds.  We
  // XOR every 4-byte broadcast with `0x80808080` (which converts
  // each s8 byte to (s8 + 128) modulo 256, i.e. the u8 mapping).
  // The resulting `+128 × sum_wei[v]` bias is undone in the
  // epilogue from the compensation row.  For Compute=kU8_Asym the
  // src is already u8 — no XOR.  `constexpr if` removes the XOR
  // entirely from the asym instantiation.
  constexpr int32_t kS8ToU8Bias = static_cast<int32_t>(0x80808080U);
  const __m512i s8_to_u8_bias_vec = _mm512_set1_epi32(kS8ToU8Bias);
  if constexpr (Compute == IntCompute::kU8_Asym) (void)s8_to_u8_bias_vec;

  // ── K-quad loop (software-pipelined, unrolled by 2) ─────────────
  //
  // Mirror of the bf16 sibling's K-pair pipelining: stage-load the
  // next K-quad's B before issuing the current K-quad's FMAs so the
  // loads drain while the 5-cycle VPDPBUSD chain retires.  Benefits
  // small-MR specialisations most (FMA-latency bound, idle load
  // issue slots).
  //
  // A-row K-quad broadcast: load 4 bytes from A and broadcast to
  // 16 s32 lanes for VPDPBUSD.  GCC pattern-matches
  // `memcpy(&u32, ptr); _mm512_set1_epi32(u32)` into a single
  // `vpbroadcastd zmm, [mem]` — same idiom the bf16 sibling uses
  // for its K-pair broadcast.  XOR with the s8→u8 bias when the
  // compute is symmetric.
  int kq = 0;
  if (kq + 1 < K_quad) {
    __m512i bv_cur[NV];
    {
      const int8_t *bp = Bpacked
          + static_cast<size_t>(kq) * kq_stride_bytes;
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        bv_cur[v] = _mm512_load_si512(
            reinterpret_cast<const __m512i *>(bp + v * v_stride_bytes));
      }
    }

    for (; kq + 1 < K_quad; kq += 2) {
      // ── Stage 1: load bv_next = B[kq+1], overlaps FMAs for kq ──
      __m512i bv_next[NV];
      {
        const int8_t *bp1 = Bpacked
            + static_cast<size_t>(kq + 1) * kq_stride_bytes;
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
          bv_next[v] = _mm512_load_si512(
              reinterpret_cast<const __m512i *>(bp1 + v * v_stride_bytes));
        }
      }
      // ── Stage 2: issue FMAs for kq using bv_cur ────────────────
      #pragma GCC unroll 8
      for (int m = 0; m < MR; ++m) {
        uint32_t a_quad;
        std::memcpy(&a_quad,
                    A + static_cast<size_t>(m) * lda + kq * kVNNIInt8Quad,
                    sizeof(a_quad));
        __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad));
        if constexpr (Compute == IntCompute::kS8_Sym) {
          av = _mm512_xor_si512(av, s8_to_u8_bias_vec);
        }
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
          acc[buf0][m][v] = _mm512_dpbusd_epi32(acc[buf0][m][v], av, bv_cur[v]);
        }
      }
      // ── Stage 3: pre-load bv_cur = B[kq+2] for the next outer
      //    iteration (unless this is the last), overlaps FMAs below.
      if (kq + 3 < K_quad) {
        const int8_t *bp2 = Bpacked
            + static_cast<size_t>(kq + 2) * kq_stride_bytes;
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
          bv_cur[v] = _mm512_load_si512(
              reinterpret_cast<const __m512i *>(bp2 + v * v_stride_bytes));
        }
      }
      // ── Stage 4: issue FMAs for kq+1 using bv_next ─────────────
      #pragma GCC unroll 8
      for (int m = 0; m < MR; ++m) {
        uint32_t a_quad;
        std::memcpy(&a_quad,
                    A + static_cast<size_t>(m) * lda
                      + (kq + 1) * kVNNIInt8Quad,
                    sizeof(a_quad));
        __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad));
        if constexpr (Compute == IntCompute::kS8_Sym) {
          av = _mm512_xor_si512(av, s8_to_u8_bias_vec);
        }
        #pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
          acc[buf1][m][v] = _mm512_dpbusd_epi32(acc[buf1][m][v], av, bv_next[v]);
        }
      }
    }
  }
  // Tail: one remaining K-quad if K_quad is odd.  No special partial
  // handling — pack zero-padded the trailing slot.
  if (kq < K_quad) {
    const int8_t *bp = Bpacked
        + static_cast<size_t>(kq) * kq_stride_bytes;
    __m512i bv[NV];
    #pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
      bv[v] = _mm512_load_si512(
          reinterpret_cast<const __m512i *>(bp + v * v_stride_bytes));
    }
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      uint32_t a_quad;
      std::memcpy(&a_quad,
                  A + static_cast<size_t>(m) * lda + kq * kVNNIInt8Quad,
                  sizeof(a_quad));
      __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad));
      if constexpr (Compute == IntCompute::kS8_Sym) {
        av = _mm512_xor_si512(av, s8_to_u8_bias_vec);
      }
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        acc[buf0][m][v] = _mm512_dpbusd_epi32(acc[buf0][m][v], av, bv[v]);
      }
    }
  }

  // ── Combine accumulator buffers (no-op when kBuffers == 1) ──────
  // Sum the even / odd K-quad partial s32 accumulators into buffer 0
  // BEFORE the compensation correction + dequant, so the downstream
  // epilogue reads a single complete s32 reduction per (m, v) tile.
  if constexpr (kBuffers > 1) {
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        acc[buf0][m][v] = _mm512_add_epi32(acc[buf0][m][v], acc[buf1][m][v]);
      }
    }
  }

  // ── Compensation row ───────────────────────────────────────────
  // Stored immediately after the o-block's weight bytes as
  // `int32_t comp[pack_nr]` (one int32 per column).  comp[v_col]
  // = `sum_k wei_s8[k, v_col]`.  See pack.cpp for the layout
  // contract.
  const int32_t *comp_base = reinterpret_cast<const int32_t *>(
      Bpacked + weight_bytes_in_oblock);
  __m512i comp_v[NV];
  #pragma GCC unroll 4
  for (int v = 0; v < NV; ++v) {
    comp_v[v] = _mm512_load_si512(
        reinterpret_cast<const __m512i *>(comp_base + v * 16));
  }

  // ── Compensation correction + dequant + bias + epilogue ────────
  //
  // Per (m, v) tile:
  //   acc_s32 = acc[m][v]
  //          - K_m * comp_v[v]               // sym: K_m = 128, asym: src_zp[m]
  //   acc_f32 = (float)acc_s32
  //          * src_scale[m] * wei_scale[v]
  //          + bias[v]                        // optional
  // For Compute=kS8_Sym we can precompute `128 * comp_v` once per
  // (v) outside the m-loop because K_m is constant — saves one
  // multiply per (m, v) tile.  For Compute=kU8_Asym K_m = src_zp[m]
  // varies per row, so the multiply stays inside the m-loop.
  __m512i comp_sym_scaled[NV];
  if constexpr (Compute == IntCompute::kS8_Sym) {
    const __m512i k_sym = _mm512_set1_epi32(128);
    #pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
      comp_sym_scaled[v] = _mm512_mullo_epi32(k_sym, comp_v[v]);
    }
  } else {
    (void)comp_sym_scaled;
  }

  // Load per-channel wei_scale[v] (16 lanes per v), widening bf16→f32
  // on load when the scales are bf16 (off the hot K-loop, same idiom
  // as the bf16-bias branch below).  f32 scales load directly.
  __m512 wei_scale_v[NV];
  if (scale_kind == ScaleKind::kBf16) {
    const auto *ws = static_cast<const bfloat16_t *>(wei_scale);
    #pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
      wei_scale_v[v] = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(ws + v * 16)));
    }
  } else {
    const auto *ws = static_cast<const float *>(wei_scale);
    #pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
      wei_scale_v[v] = _mm512_loadu_ps(ws + v * 16);
    }
  }

  // Load per-channel bias[v] if present (BF16 or FP32 → FP32 in
  // registers).  Mirror of the bf16 sibling's bias load — the
  // dispatcher resolves `bias_kind` once at `prepare_for_call`
  // time and the branch is outside the per-(m, v) loop below.
  const bool has_bias = (bias != nullptr && bias_kind != BiasKind::none);
  __m512 bias_v[NV];
  if (has_bias) {
    if (bias_kind == BiasKind::bf16) {
      const auto *bias_bf16 = static_cast<const bfloat16_t *>(bias);
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        __m256i b16 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(bias_bf16 + v * 16));
        bias_v[v] = bf16x16_to_f32(b16);
      }
    } else {
      const auto *bias_fp32 = static_cast<const float *>(bias);
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        bias_v[v] = _mm512_loadu_ps(bias_fp32 + v * 16);
      }
    }
  } else {
    (void)bias_v;
  }

  // ── Fused dequant + store epilogue ─────────────────────────────
  // Per (m, v) tile, compute the dequantised FP32 result and store it
  // IMMEDIATELY.  The previous version staged a full
  // `__m512 acc_f32[MR][NV]` array and stored in a second pass; at
  // MR=8 that 16-zmm staging array sat on top of the 16 s32 `acc`
  // zmms + comp/scale/bias temps (~40 zmm) and spilled the 32-zmm
  // register file, generating extra load/store traffic that starved
  // the VPDPBUSD units (measured: int8 issued MORE L1 loads than the
  // bf16 sibling and ran at IPC 1.9 vs 3.2, FP-dispatch-stall 5% vs
  // 41% — i.e. the FMA units were idle, not saturated).  Fusing the
  // dequant into the store keeps at most 1-2 FP32 temps live, so the
  // accumulators stay resident and the kernel becomes FMA-bound like
  // bf16.  `dequant_finish` is the single source of the dequant math;
  // the per-(m, v) compensation term `corr` is selected inline by
  // `Compute` (`128 * comp`, precomputed in `comp_sym_scaled`, for
  // sym; `src_zp[m] * comp` for asym).  The selection stays inline
  // (not a lambda) so it runs under this function's target attribute,
  // and `src_zp[m]` is read only on the asym branch where the
  // dispatcher guarantees a non-null `src_zp`.
  if constexpr (Act == ActKind::swiglu_oai_mul
                  || Act == ActKind::silu_and_mul
                  || Act == ActKind::gelu_and_mul) {
    // Gated acts always write half-width BF16 to the tight dst
    // (Dst == kBf16 enforced by the static_assert at function head).
    bfloat16_t *__restrict__ Cout_tight =
        static_cast<bfloat16_t *>(Cout_tight_void);
    constexpr int n_pairs = NV / 2;  // 1 for NV=2, 2 for NV=4
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      const __m512 src_scale_m = _mm512_set1_ps(src_scale_at(m));
      #pragma GCC unroll 2
      for (int p = 0; p < n_pairs; ++p) {
        __m512i corr_lo, corr_hi;
        if constexpr (Compute == IntCompute::kS8_Sym) {
          corr_lo = comp_sym_scaled[2 * p];
          corr_hi = comp_sym_scaled[2 * p + 1];
        } else {
          const __m512i zp = _mm512_set1_epi32(src_zp[m]);
          corr_lo = _mm512_mullo_epi32(zp, comp_v[2 * p]);
          corr_hi = _mm512_mullo_epi32(zp, comp_v[2 * p + 1]);
        }
        const __m512 lo = dequant_finish(
            acc[buf0][m][2 * p], corr_lo,
            src_scale_m, wei_scale_v[2 * p], bias_v[2 * p], has_bias);
        const __m512 hi = dequant_finish(
            acc[buf0][m][2 * p + 1], corr_hi,
            src_scale_m, wei_scale_v[2 * p + 1], bias_v[2 * p + 1], has_bias);
        bfloat16_t *dst = Cout_tight
            + static_cast<size_t>(m) * ldc_tight + p * 16;
        if constexpr (Act == ActKind::swiglu_oai_mul) {
          swiglu_oai_store_pair_int(lo, hi, dst);
        } else if constexpr (Act == ActKind::silu_and_mul) {
          silu_and_mul_store_pair_int(lo, hi, dst);
        } else {  // ActKind::gelu_and_mul
          gelu_and_mul_store_pair_int(lo, hi, dst);
        }
      }
    }
  } else {
    // Act = none — wide store, BF16 (DstDt::kBf16) or FP32
    // (DstDt::kF32).  The two differ only in the final narrow/store.
    static_assert(Act == ActKind::none,
                  "int8 ukernel only supports {none, swiglu, silu, gelu}");
    #pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
      const __m512 src_scale_m = _mm512_set1_ps(src_scale_at(m));
      #pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        __m512i corr;
        if constexpr (Compute == IntCompute::kS8_Sym) {
          corr = comp_sym_scaled[v];
        } else {
          corr = _mm512_mullo_epi32(_mm512_set1_epi32(src_zp[m]), comp_v[v]);
        }
        const __m512 f = dequant_finish(
            acc[buf0][m][v], corr,
            src_scale_m, wei_scale_v[v], bias_v[v], has_bias);
        if constexpr (Dst == DstDt::kBf16) {
          bfloat16_t *dst =
              static_cast<bfloat16_t *>(Cout_void)
              + static_cast<size_t>(m) * ldc + v * 16;
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst),
                              f32_to_bf16x16(f));
        } else {
          static_assert(Dst == DstDt::kF32,
                        "int8 ukernel act=none dst must be bf16 or f32");
          float *dst = static_cast<float *>(Cout_void)
              + static_cast<size_t>(m) * ldc + v * 16;
          _mm512_storeu_ps(dst, f);
        }
      }
    }
  }
}

}  // namespace

// ── Function-pointer table dispatch ─────────────────────────────
//
// One entry per supported (MR, NV, Compute, Act) tuple.
// Instantiation set:
//   NV=2 (NR=32): MR ∈ {1..8} × Compute ∈ {sym, asym}
//                   × Act ∈ {none, swiglu, silu, gelu}
//   NV=4 (NR=64): MR ∈ {1..6} × Compute ∈ {sym, asym}
//                   × Act ∈ {none, swiglu, silu, gelu}
// Total: (8 + 6) × 2 × 4 = 112 BF16-dst specializations, all
// `noinline`.  The `act=none` cells additionally instantiate an
// FP32-dst variant (`DstDt::kF32`) via `select_none<kF32>`; gated
// acts are BF16-dst only (`select_int8_ukernel` returns nullptr for
// any gated + f32 tuple, mirroring the bf16 sibling).

// Hand-rolled per-(Compute, Act) switch helpers — matches the bf16
// sibling's pattern (no macros, every instantiation explicit).
// NV=2 supports MR ∈ {1..8}; NV=4 caps at MR=6 (register pressure).
//
// The pair of switch helpers is repeated 4× per Compute (one for
// each Act value) and 2× per NV — verbose but the explicit set is
// easy to audit and keeps the dispatcher independent of any macro
// expansion subtleties.

template <ActKind Act, DstDt Dst>
static int8_ukernel_fn_t select_nv2_sym(int MR) {
  switch (MR) {
  case 1: return ukernel_impl<1, 2, IntCompute::kS8_Sym, Act, Dst>;
  case 2: return ukernel_impl<2, 2, IntCompute::kS8_Sym, Act, Dst>;
  case 3: return ukernel_impl<3, 2, IntCompute::kS8_Sym, Act, Dst>;
  case 4: return ukernel_impl<4, 2, IntCompute::kS8_Sym, Act, Dst>;
  case 5: return ukernel_impl<5, 2, IntCompute::kS8_Sym, Act, Dst>;
  case 6: return ukernel_impl<6, 2, IntCompute::kS8_Sym, Act, Dst>;
  case 7: return ukernel_impl<7, 2, IntCompute::kS8_Sym, Act, Dst>;
  case 8: return ukernel_impl<8, 2, IntCompute::kS8_Sym, Act, Dst>;
  default: return nullptr;
  }
}

template <ActKind Act, DstDt Dst>
static int8_ukernel_fn_t select_nv2_asym(int MR) {
  switch (MR) {
  case 1: return ukernel_impl<1, 2, IntCompute::kU8_Asym, Act, Dst>;
  case 2: return ukernel_impl<2, 2, IntCompute::kU8_Asym, Act, Dst>;
  case 3: return ukernel_impl<3, 2, IntCompute::kU8_Asym, Act, Dst>;
  case 4: return ukernel_impl<4, 2, IntCompute::kU8_Asym, Act, Dst>;
  case 5: return ukernel_impl<5, 2, IntCompute::kU8_Asym, Act, Dst>;
  case 6: return ukernel_impl<6, 2, IntCompute::kU8_Asym, Act, Dst>;
  case 7: return ukernel_impl<7, 2, IntCompute::kU8_Asym, Act, Dst>;
  case 8: return ukernel_impl<8, 2, IntCompute::kU8_Asym, Act, Dst>;
  default: return nullptr;
  }
}

template <ActKind Act, DstDt Dst>
static int8_ukernel_fn_t select_nv4_sym(int MR) {
  switch (MR) {
  case 1: return ukernel_impl<1, 4, IntCompute::kS8_Sym, Act, Dst>;
  case 2: return ukernel_impl<2, 4, IntCompute::kS8_Sym, Act, Dst>;
  case 3: return ukernel_impl<3, 4, IntCompute::kS8_Sym, Act, Dst>;
  case 4: return ukernel_impl<4, 4, IntCompute::kS8_Sym, Act, Dst>;
  case 5: return ukernel_impl<5, 4, IntCompute::kS8_Sym, Act, Dst>;
  case 6: return ukernel_impl<6, 4, IntCompute::kS8_Sym, Act, Dst>;
  default: return nullptr;
  }
}

template <ActKind Act, DstDt Dst>
static int8_ukernel_fn_t select_nv4_asym(int MR) {
  switch (MR) {
  case 1: return ukernel_impl<1, 4, IntCompute::kU8_Asym, Act, Dst>;
  case 2: return ukernel_impl<2, 4, IntCompute::kU8_Asym, Act, Dst>;
  case 3: return ukernel_impl<3, 4, IntCompute::kU8_Asym, Act, Dst>;
  case 4: return ukernel_impl<4, 4, IntCompute::kU8_Asym, Act, Dst>;
  case 5: return ukernel_impl<5, 4, IntCompute::kU8_Asym, Act, Dst>;
  case 6: return ukernel_impl<6, 4, IntCompute::kU8_Asym, Act, Dst>;
  default: return nullptr;
  }
}

// Act=none selector for a given (NV, Compute, Dst).  Only `none`
// instantiates the FP32-dst axis (gated kinds are BF16-dst only).
template <DstDt Dst>
static int8_ukernel_fn_t select_none(int MR, int NV, IntCompute compute) {
  if (NV == 2) {
    return (compute == IntCompute::kS8_Sym)
        ? select_nv2_sym<ActKind::none, Dst>(MR)
        : select_nv2_asym<ActKind::none, Dst>(MR);
  }
  if (NV == 4) {
    return (compute == IntCompute::kS8_Sym)
        ? select_nv4_sym<ActKind::none, Dst>(MR)
        : select_nv4_asym<ActKind::none, Dst>(MR);
  }
  return nullptr;
}

int8_ukernel_fn_t select_int8_ukernel(int MR, int NV,
                                      IntCompute compute,
                                      ActKind    act,
                                      DstDt      dst_dt) {
  const bool is_gated_act = (act == ActKind::swiglu_oai_mul
                             || act == ActKind::silu_and_mul
                             || act == ActKind::gelu_and_mul);
  // Gated activations write half-width BF16 — no FP32-dst counterpart
  // (mirror of the bf16 sibling's `select_ukernel` guard).  Refuse the
  // (gated, kF32) tuple so the dispatcher falls back cleanly.
  if (is_gated_act && dst_dt != DstDt::kBf16) {
    return nullptr;
  }

  // Act = none — branch on dst dtype for the store epilogue.
  if (act == ActKind::none) {
    return (dst_dt == DstDt::kF32)
        ? select_none<DstDt::kF32>(MR, NV, compute)
        : select_none<DstDt::kBf16>(MR, NV, compute);
  }

  // Gated activations (BF16 dst only past the guard above).
  if (NV == 2) {
    if (compute == IntCompute::kS8_Sym) {
      switch (act) {
      case ActKind::swiglu_oai_mul: return select_nv2_sym<ActKind::swiglu_oai_mul, DstDt::kBf16>(MR);
      case ActKind::silu_and_mul:   return select_nv2_sym<ActKind::silu_and_mul,   DstDt::kBf16>(MR);
      case ActKind::gelu_and_mul:   return select_nv2_sym<ActKind::gelu_and_mul,   DstDt::kBf16>(MR);
      default:                      return nullptr;
      }
    }
    switch (act) {
    case ActKind::swiglu_oai_mul: return select_nv2_asym<ActKind::swiglu_oai_mul, DstDt::kBf16>(MR);
    case ActKind::silu_and_mul:   return select_nv2_asym<ActKind::silu_and_mul,   DstDt::kBf16>(MR);
    case ActKind::gelu_and_mul:   return select_nv2_asym<ActKind::gelu_and_mul,   DstDt::kBf16>(MR);
    default:                      return nullptr;
    }
  }
  if (NV == 4) {
    if (compute == IntCompute::kS8_Sym) {
      switch (act) {
      case ActKind::swiglu_oai_mul: return select_nv4_sym<ActKind::swiglu_oai_mul, DstDt::kBf16>(MR);
      case ActKind::silu_and_mul:   return select_nv4_sym<ActKind::silu_and_mul,   DstDt::kBf16>(MR);
      case ActKind::gelu_and_mul:   return select_nv4_sym<ActKind::gelu_and_mul,   DstDt::kBf16>(MR);
      default:                      return nullptr;
      }
    }
    switch (act) {
    case ActKind::swiglu_oai_mul: return select_nv4_asym<ActKind::swiglu_oai_mul, DstDt::kBf16>(MR);
    case ActKind::silu_and_mul:   return select_nv4_asym<ActKind::silu_and_mul,   DstDt::kBf16>(MR);
    case ActKind::gelu_and_mul:   return select_nv4_asym<ActKind::gelu_and_mul,   DstDt::kBf16>(MR);
    default:                      return nullptr;
    }
  }
  return nullptr;
}

}  // namespace custom_kernel
}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl
