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

/// FP16 custom microkernel implementation — templated on (MR, NV, Act,
/// DstT).  Sibling of `bf16_microkernel.cpp`; see `f16_microkernel.hpp`
/// for the role this kernel plays in the group_matmul dispatch stack.
///
/// PER K-ELEMENT INNER LOOP (native AVX-512-FP16, `_mm512_fmadd_ph`):
///
///   load Bpacked → NV_h zmms (bv[h] covers cols h*32..(h+1)*32 - 1)
///   for m in 0..MR-1:
///     broadcast A[m, k] into 32 FP16 lanes
///     for h in 0..NV_h-1:
///       acc[m][h] = fmadd_ph(A_reg, bv[h], acc[m][h])
///
/// Unlike the bf16 VDPBF16PS path (which consumes 2 K-elements per
/// instruction via a K-pair pack) the native FP16 FMA consumes ONE
/// K-element per lane, so the K loop is a plain stride-1 walk over
/// `[0, K)` and the pack carries no K-interleave (see
/// `pack_f16_simple_impl` in pack.cpp).  Accumulators are native FP16
/// (`__m512h`, 32 cols/zmm), so `NV_h = NV / 2` zmms cover the
/// NR = NV*16 columns of one o-block.
///
/// EPILOGUE:
///   Each FP16 accumulator is widened to two FP32 zmms via
///   `float16_t::cvt_f16_to_f32_vec`, so the activation math and the
///   FP32→F16 store reuse the SAME primitives
///   (`group_matmul_act_avx512.hpp`) the bf16 sibling and the
///   separate-pass reference use.  For `Act = none` with an F16 dst
///   the accumulator is stored directly via `_mm512_storeu_ph` (no
///   widen); for an FP32 dst it is widened and stored as two FP32
///   halves.  Gated kinds deinterleave the (gate, up) pair out of the
///   widened FP32 lanes (identical `vpermt2ps` indices as the bf16
///   sibling), apply the activation, and store the half-width F16
///   result.
///
/// PRECISION: the accumulation is native FP16 (see the header's
/// "PRECISION NOTE").  Callers needing FP32-accumulate parity disable
/// the F16 custom kernel via `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_F16=0`
/// and fall back to AOCL DLP.

#include "f16_microkernel.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>

#include <immintrin.h>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_act_avx512.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

// TWO INDEPENDENT GATES protect the F16 custom kernel; EITHER one alone
// is sufficient to trigger the AOCL DLP fallback:
//
//   * Compile-time (`ZENDNNL_GRP_F16_CK_AVAILABLE`, below): if the
//     TOOLCHAIN lacks the AVX-512-FP16 intrinsics (`__m512h`,
//     `_mm512_fmadd_ph`, …) and the `_Float16` scalar type, no machine
//     code is emitted for the kernel bodies at all — the TU still
//     builds, but `select_f16_ukernel()` returns nullptr for every
//     slot and `avx512f16_available()` returns false unconditionally.
//   * Runtime (`avx512f16_available()`, below): even on a toolchain
//     that DID compile the kernels, `platform_info::get_avx512_f16_status()`
//     probes the ACTUAL CPU via CPUID (leaf 7 / sub 0 / EDX bit 23);
//     a host without the ISA reports false so the dispatcher refuses.
//
// The compile-time gate reuses the SAME macro `common/float16.hpp` uses
// for its `__m512h` helpers so the two stay in lock-step.  When either
// gate is closed the dispatcher sees `avx512f16_available() == false` /
// `select_f16_ukernel() == nullptr` and routes every F16 call to the
// AOCL DLP fallback.
#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS) \
        || (defined(__GNUC__) && (__GNUC__ >= 12))
#define ZENDNNL_GRP_F16_CK_AVAILABLE 1
#else
#define ZENDNNL_GRP_F16_CK_AVAILABLE 0
#endif

bool avx512f16_available() {
#if ZENDNNL_GRP_F16_CK_AVAILABLE
    // Same platform-info ISA probe the embag / normalization AVX-512-FP16
    // kernels and the `group_matmul_direct` F16 entry use (CPUID leaf 7 /
    // sub 0 / EDX bit 23).  Cached after first call.
    static const bool v = []() {
        return zendnnl::common::zendnnl_platform_info().get_avx512_f16_status();
    }();
    return v;
#else
    // Toolchain without AVX-512-FP16 intrinsics: the kernels below were
    // not compiled, so report unavailable regardless of runtime CPU.
    return false;
#endif
}

#if ZENDNNL_GRP_F16_CK_AVAILABLE

namespace {

// Shared FP32 activation math + the FP32→F16 store cvt, identical to
// the bf16 sibling's using-declarations (the bf16 path pulls the
// BF16 cvts; here we pull the F16 cvt `f32_to_f16x16`).  Reusing the
// same FP32 primitives keeps the F16 fused-CK output on the same
// numerical contract as the separate-pass reference, modulo the
// FP16-accumulate caveat documented in the header.
using zendnnl::lowoha::matmul::group_matmul_act_avx512::bf16x16_to_f32;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::f32_to_f16x16;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::gelu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::silu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::swiglu_oai_avx512;

// Deinterleave indices for vpermt2ps over two source zmms (= 32 FP32
// lanes total) — IDENTICAL to the bf16 sibling.  Even-indexed lanes
// (0, 2, …, 30) are gates; odd-indexed lanes (1, 3, …, 31) are ups.
alignas(64) constexpr int32_t kGateLaneIdx[16]
        = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) constexpr int32_t kUpLaneIdx[16]
        = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

// Convert a `float16_t` (raw uint16 storage) to the `_Float16` scalar
// the `_mm512_set1_ph` broadcast expects.  A bit-copy — no arithmetic
// — so it is exact and the optimiser folds it into the broadcast load.
ZENDNNL_INLINE_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline _Float16 f16_raw_to_scalar(const float16_t &v) {
    _Float16 hv;
    const uint16_t r = v.raw();
    std::memcpy(&hv, &r, sizeof(hv));
    return hv;
}

// Apply swiglu_oai_mul in registers to one FP16 accumulator (32
// interleaved cols) and store 16 activated F16 cols.  Widens to FP32,
// deinterleaves gate/up (`vpermt2ps`), runs the shared
// `swiglu_oai_avx512`, then narrows FP32→F16 via the shared
// `f32_to_f16x16` and stores 256 bits.  Same structure as the bf16
// `swiglu_oai_store_pair`, only the widen step (FP16→FP32) and the
// final cvt (FP32→F16 instead of FP32→BF16) differ.
ZENDNNL_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void swiglu_oai_store_pair_f16(__m512h acc, float16_t *dst_row) {
    __m512 lo, hi;
    float16_t::cvt_f16_to_f32_vec(acc, lo, hi);
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdx);
    __m512 gate = _mm512_permutex2var_ps(lo, gate_idx, hi);
    __m512 up = _mm512_permutex2var_ps(lo, up_idx, hi);
    __m512 r = swiglu_oai_avx512(gate, up);
    __m256i out = f32_to_f16x16(r);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// Apply silu_and_mul in registers — silu(gate) * up.  See the bf16
// sibling's `silu_and_mul_store_pair` for the numerical contract.
ZENDNNL_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void silu_and_mul_store_pair_f16(
        __m512h acc, float16_t *dst_row) {
    __m512 lo, hi;
    float16_t::cvt_f16_to_f32_vec(acc, lo, hi);
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdx);
    __m512 gate = _mm512_permutex2var_ps(lo, gate_idx, hi);
    __m512 up = _mm512_permutex2var_ps(lo, up_idx, hi);
    __m512 r = _mm512_mul_ps(silu_avx512(gate), up);
    __m256i out = f32_to_f16x16(r);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// Apply gelu_and_mul in registers — gelu(gate) * up (gelu_tanh form,
// see the bf16 sibling's `gelu_and_mul_store_pair` for the numerical
// contract).
ZENDNNL_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void gelu_and_mul_store_pair_f16(
        __m512h acc, float16_t *dst_row) {
    __m512 lo, hi;
    float16_t::cvt_f16_to_f32_vec(acc, lo, hi);
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdx);
    __m512 gate = _mm512_permutex2var_ps(lo, gate_idx, hi);
    __m512 up = _mm512_permutex2var_ps(lo, up_idx, hi);
    __m512 r = _mm512_mul_ps(gelu_avx512(gate), up);
    __m256i out = f32_to_f16x16(r);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// ─────────────────────────────────────────────────────────────────────
// Templated microkernel — MR ∈ 1..8, NV ∈ {2, 4}, Act, DstT.
//
// `noinline` keeps each specialization a single callable the
// dispatcher reaches through a function pointer (matches the bf16 /
// int8 siblings).  NV_h = NV / 2 native `__m512h` accumulators per row
// cover the NR = NV*16 columns of one o-block.
// ─────────────────────────────────────────────────────────────────────
template <int MR, int NV, ActKind Act, typename DstT>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static void ukernel_impl(const float16_t *__restrict A, int lda,
        const float16_t *__restrict Bpacked, const void *__restrict bias,
        BiasKind bias_kind, void *__restrict Cout_void, int ldc,
        void *__restrict Cout_tight_void, int ldc_tight, int K) {

    static_assert(NV == 2 || NV == 4, "NV must be 2 or 4");
    static_assert(Act == ActKind::none || (NV % 2 == 0),
            "gated-activation epilogue requires even NV");
    static_assert(std::is_same<DstT, float16_t>::value
                    || std::is_same<DstT, float>::value,
            "ukernel_impl: DstT must be float16_t or float");
    // Gated kinds write 16 F16 lanes per (gate, up) pair via the
    // pair-store helpers (no FP32 counterpart), so refuse (gated, FP32)
    // at compile time — `select_f16_ukernel` refuses it earlier still.
    static_assert(Act == ActKind::none || std::is_same<DstT, float16_t>::value,
            "Gated-activation kinds are F16-dst only");

    DstT *__restrict Cout = static_cast<DstT *>(Cout_void);
    DstT *__restrict Cout_tight = static_cast<DstT *>(Cout_tight_void);

    // NV_h native FP16 accumulators per row (32 cols each).  No double-
    // buffering: FP16 packs 32 cols/zmm so the accumulator footprint is
    // half the bf16 path's and small-MR specialisations stay within the
    // register budget without it.  (A double-buffered variant matching
    // the bf16 sibling's MR≤3 path is a possible latency-bound tuning
    // follow-up.)
    constexpr int NV_h = NV / 2;
    // pack_nr (cols per o-block) in FP16 elements = NV*16 = NV_h*32.
    constexpr int kPackNr = NV * 16;

    __m512h acc[MR][NV_h];

    // ── Bias fold into accumulator init ──────────────────────────────
    // Bias is per-column (NR cols), in BF16, FP32, or F16.  Seed
    // the accumulators with the FP16 bias columns; the K loop accumulates
    // on top.  When no bias is present, zero-init.
    //   * bf16 / fp32 bias is narrowed to FP16 here (one conversion per
    //     tile), in keeping with the native-FP16-accumulate precision
    //     profile of this path.
    //   * f16 bias is loaded directly into the __m512h seed — no
    //     conversion — matching the K-loop weight-load idiom.
    const bool has_bias = (bias != nullptr && bias_kind != BiasKind::none);
    if (has_bias) {
        __m512h bias_vec[NV_h];
        if (bias_kind == BiasKind::bf16) {
            const auto *bias_bf16 = static_cast<const bfloat16_t *>(bias);
#pragma GCC unroll 2
            for (int h = 0; h < NV_h; ++h) {
                __m256i b_lo = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(bias_bf16 + h * 32));
                __m256i b_hi
                        = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                                bias_bf16 + h * 32 + 16));
                bias_vec[h] = float16_t::cvt_f32_to_f16_vec(
                        bf16x16_to_f32(b_lo), bf16x16_to_f32(b_hi));
            }
        } else if (bias_kind == BiasKind::f16) {
            const auto *bias_f16 = static_cast<const float16_t *>(bias);
#pragma GCC unroll 2
            for (int h = 0; h < NV_h; ++h) {
                bias_vec[h] = _mm512_loadu_ph(bias_f16 + h * 32);
            }
        } else { // BiasKind::fp32
            const auto *bias_fp32 = static_cast<const float *>(bias);
#pragma GCC unroll 2
            for (int h = 0; h < NV_h; ++h) {
                __m512 lo = _mm512_loadu_ps(bias_fp32 + h * 32);
                __m512 hi = _mm512_loadu_ps(bias_fp32 + h * 32 + 16);
                bias_vec[h] = float16_t::cvt_f32_to_f16_vec(lo, hi);
            }
        }
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m)
#pragma GCC unroll 2
            for (int h = 0; h < NV_h; ++h)
                acc[m][h] = bias_vec[h];
    } else {
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m)
#pragma GCC unroll 2
            for (int h = 0; h < NV_h; ++h)
                acc[m][h] = _mm512_setzero_ph();
    }

    // ── K loop: native FP16 FMA, one K-element per step ──────────────
    for (int k = 0; k < K; ++k) {
        __m512h bv[NV_h];
        const float16_t *bp = Bpacked + static_cast<size_t>(k) * kPackNr;
#pragma GCC unroll 2
        for (int h = 0; h < NV_h; ++h) {
            bv[h] = _mm512_loadu_ph(bp + h * 32);
        }
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            __m512h av = _mm512_set1_ph(
                    f16_raw_to_scalar(A[static_cast<size_t>(m) * lda + k]));
#pragma GCC unroll 2
            for (int h = 0; h < NV_h; ++h)
                acc[m][h] = _mm512_fmadd_ph(av, bv[h], acc[m][h]);
        }
    }

    // ── Epilogue ─────────────────────────────────────────────────────
    if constexpr (Act == ActKind::swiglu_oai_mul || Act == ActKind::silu_and_mul
            || Act == ActKind::gelu_and_mul) {
// One (gate, up) pair fills one FP16 accumulator (32 interleaved
// cols → 16 outputs); n_pairs == NV_h.
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 2
            for (int p = 0; p < NV_h; ++p) {
                float16_t *dst = Cout_tight + static_cast<size_t>(m) * ldc_tight
                        + p * 16;
                if constexpr (Act == ActKind::swiglu_oai_mul) {
                    swiglu_oai_store_pair_f16(acc[m][p], dst);
                } else if constexpr (Act == ActKind::silu_and_mul) {
                    silu_and_mul_store_pair_f16(acc[m][p], dst);
                } else { // ActKind::gelu_and_mul
                    gelu_and_mul_store_pair_f16(acc[m][p], dst);
                }
            }
        }
    } else if constexpr (std::is_same<DstT, float16_t>::value) {
// Act = none, F16 dst — store the FP16 accumulator (32 cols) per
// NV_h zmm directly; no widen / cvt.
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 2
            for (int h = 0; h < NV_h; ++h) {
                float16_t *dst = Cout + static_cast<size_t>(m) * ldc + h * 32;
                _mm512_storeu_ph(dst, acc[m][h]);
            }
        }
    } else {
        // Act = none, FP32 dst — widen each FP16 accumulator to two FP32
        // halves and store 32 FP32 cols per NV_h zmm.
        static_assert(std::is_same<DstT, float>::value,
                "Act=none non-f16 dst path requires DstT == float");
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 2
            for (int h = 0; h < NV_h; ++h) {
                float *dst = Cout + static_cast<size_t>(m) * ldc + h * 32;
                __m512 lo, hi;
                float16_t::cvt_f16_to_f32_vec(acc[m][h], lo, hi);
                _mm512_storeu_ps(dst, lo);
                _mm512_storeu_ps(dst + 16, hi);
            }
        }
    }
}

} // namespace

#endif // ZENDNNL_GRP_F16_CK_AVAILABLE

// ── Function-pointer table dispatch (mirrors select_ukernel) ─────────
//
// Instantiation set (when the toolchain provides AVX-512-FP16):
//   NV=2 (NR=32): MR ∈ {1..8} × {(none, F16), (none, F32), (swiglu, F16),
//                                 (silu, F16), (gelu, F16)}
//   NV=4 (NR=64): MR ∈ {1..6} × same five
// Total: 8×5 + 6×5 = 70 specializations.  Any (gated_act, FP32) tuple
// is intentionally NOT instantiated — the pair-store helpers write F16
// only — and `select_f16_ukernel` returns nullptr for it so the
// dispatcher refuses the call at `prepare_for_call` time.
//
// On a toolchain WITHOUT AVX-512-FP16 intrinsics the kernel bodies are
// not compiled and this selector unconditionally returns nullptr; the
// dispatcher then falls back to AOCL DLP for every F16 call.
f16_ukernel_fn_t select_f16_ukernel(int MR, int NV, ActKind act, DstDt dst_dt) {
#if ZENDNNL_GRP_F16_CK_AVAILABLE
    const bool is_gated_act = (act == ActKind::swiglu_oai_mul
            || act == ActKind::silu_and_mul || act == ActKind::gelu_and_mul);
    if (is_gated_act && dst_dt != DstDt::kF16) { return nullptr; }
    if (NV == 2) {
        if (act == ActKind::swiglu_oai_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 2, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 2:
                    return ukernel_impl<2, 2, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 3:
                    return ukernel_impl<3, 2, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 4:
                    return ukernel_impl<4, 2, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 5:
                    return ukernel_impl<5, 2, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 6:
                    return ukernel_impl<6, 2, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 7:
                    return ukernel_impl<7, 2, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 8:
                    return ukernel_impl<8, 2, ActKind::swiglu_oai_mul,
                            float16_t>;
                default: return nullptr;
            }
        }
        if (act == ActKind::silu_and_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 2, ActKind::silu_and_mul, float16_t>;
                case 2:
                    return ukernel_impl<2, 2, ActKind::silu_and_mul, float16_t>;
                case 3:
                    return ukernel_impl<3, 2, ActKind::silu_and_mul, float16_t>;
                case 4:
                    return ukernel_impl<4, 2, ActKind::silu_and_mul, float16_t>;
                case 5:
                    return ukernel_impl<5, 2, ActKind::silu_and_mul, float16_t>;
                case 6:
                    return ukernel_impl<6, 2, ActKind::silu_and_mul, float16_t>;
                case 7:
                    return ukernel_impl<7, 2, ActKind::silu_and_mul, float16_t>;
                case 8:
                    return ukernel_impl<8, 2, ActKind::silu_and_mul, float16_t>;
                default: return nullptr;
            }
        }
        if (act == ActKind::gelu_and_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 2, ActKind::gelu_and_mul, float16_t>;
                case 2:
                    return ukernel_impl<2, 2, ActKind::gelu_and_mul, float16_t>;
                case 3:
                    return ukernel_impl<3, 2, ActKind::gelu_and_mul, float16_t>;
                case 4:
                    return ukernel_impl<4, 2, ActKind::gelu_and_mul, float16_t>;
                case 5:
                    return ukernel_impl<5, 2, ActKind::gelu_and_mul, float16_t>;
                case 6:
                    return ukernel_impl<6, 2, ActKind::gelu_and_mul, float16_t>;
                case 7:
                    return ukernel_impl<7, 2, ActKind::gelu_and_mul, float16_t>;
                case 8:
                    return ukernel_impl<8, 2, ActKind::gelu_and_mul, float16_t>;
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
            case 1: return ukernel_impl<1, 2, ActKind::none, float16_t>;
            case 2: return ukernel_impl<2, 2, ActKind::none, float16_t>;
            case 3: return ukernel_impl<3, 2, ActKind::none, float16_t>;
            case 4: return ukernel_impl<4, 2, ActKind::none, float16_t>;
            case 5: return ukernel_impl<5, 2, ActKind::none, float16_t>;
            case 6: return ukernel_impl<6, 2, ActKind::none, float16_t>;
            case 7: return ukernel_impl<7, 2, ActKind::none, float16_t>;
            case 8: return ukernel_impl<8, 2, ActKind::none, float16_t>;
            default: return nullptr;
        }
    }
    if (NV == 4) {
        if (act == ActKind::swiglu_oai_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 4, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 2:
                    return ukernel_impl<2, 4, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 3:
                    return ukernel_impl<3, 4, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 4:
                    return ukernel_impl<4, 4, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 5:
                    return ukernel_impl<5, 4, ActKind::swiglu_oai_mul,
                            float16_t>;
                case 6:
                    return ukernel_impl<6, 4, ActKind::swiglu_oai_mul,
                            float16_t>;
                default: return nullptr;
            }
        }
        if (act == ActKind::silu_and_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 4, ActKind::silu_and_mul, float16_t>;
                case 2:
                    return ukernel_impl<2, 4, ActKind::silu_and_mul, float16_t>;
                case 3:
                    return ukernel_impl<3, 4, ActKind::silu_and_mul, float16_t>;
                case 4:
                    return ukernel_impl<4, 4, ActKind::silu_and_mul, float16_t>;
                case 5:
                    return ukernel_impl<5, 4, ActKind::silu_and_mul, float16_t>;
                case 6:
                    return ukernel_impl<6, 4, ActKind::silu_and_mul, float16_t>;
                default: return nullptr;
            }
        }
        if (act == ActKind::gelu_and_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 4, ActKind::gelu_and_mul, float16_t>;
                case 2:
                    return ukernel_impl<2, 4, ActKind::gelu_and_mul, float16_t>;
                case 3:
                    return ukernel_impl<3, 4, ActKind::gelu_and_mul, float16_t>;
                case 4:
                    return ukernel_impl<4, 4, ActKind::gelu_and_mul, float16_t>;
                case 5:
                    return ukernel_impl<5, 4, ActKind::gelu_and_mul, float16_t>;
                case 6:
                    return ukernel_impl<6, 4, ActKind::gelu_and_mul, float16_t>;
                default: return nullptr;
            }
        }
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
            case 1: return ukernel_impl<1, 4, ActKind::none, float16_t>;
            case 2: return ukernel_impl<2, 4, ActKind::none, float16_t>;
            case 3: return ukernel_impl<3, 4, ActKind::none, float16_t>;
            case 4: return ukernel_impl<4, 4, ActKind::none, float16_t>;
            case 5: return ukernel_impl<5, 4, ActKind::none, float16_t>;
            case 6: return ukernel_impl<6, 4, ActKind::none, float16_t>;
            default: return nullptr;
        }
    }
    return nullptr;
#else
    (void)MR;
    (void)NV;
    (void)act;
    (void)dst_dt;
    return nullptr;
#endif
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
