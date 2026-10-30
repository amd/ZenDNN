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
///   load Bpacked → one zmm per 32-column block (bv[j])
///   for m in 0..MR-1:
///     broadcast A[m, k] into 32 FP16 lanes
///     for j: acc[m][j] = fmadd_ph(A_reg, bv[j], acc[m][j])
///
/// Unlike the bf16 VDPBF16PS path (which consumes 2 K-elements per
/// instruction via a K-pair pack) the native FP16 FMA consumes ONE
/// K-element per lane, so the K loop is a plain stride-1 walk and the
/// pack carries no K-interleave (see `pack_f16_simple_impl` in
/// pack.cpp).  Every `kF16AccumBlockK` K steps the FP16 accumulators
/// are widened and added into FP32 sums, which also carry the bias.
///
/// EPILOGUE (from the FP32 sums):
///   Act = none stores F16 (one FP32→F16 narrowing) or FP32.  Gated
///   kinds deinterleave the (gate, up) pair (identical `vpermt2ps`
///   indices as the bf16 sibling), apply the shared FP32 activation
///   (`group_matmul_act_avx512.hpp`) and store the half-width F16
///   result.

#include "f16_microkernel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <type_traits>

#include <immintrin.h>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_act_avx512.hpp"

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

// Apply swiglu_oai_mul to one 32-column (gate, up)-interleaved FP32 sum
// (`lo` = columns 0..15, `hi` = 16..31) and store 16 activated F16 cols.
// Deinterleaves gate/up (`vpermt2ps`), runs the shared
// `swiglu_oai_avx512`, then narrows FP32→F16 via the shared
// `f32_to_f16x16` and stores 256 bits.  Same structure as the bf16
// `swiglu_oai_store_pair`; only the final cvt (FP32→F16) differs.
ZENDNNL_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void swiglu_oai_store_pair_f16(
        __m512 lo, __m512 hi, float16_t *dst_row) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdx);
    __m512 gate = _mm512_permutex2var_ps(lo, gate_idx, hi);
    __m512 up = _mm512_permutex2var_ps(lo, up_idx, hi);
    __m512 r = swiglu_oai_avx512(gate, up);
    __m256i out = f32_to_f16x16(r);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// silu_and_mul — silu(gate) * up.  See the bf16 sibling's
// `silu_and_mul_store_pair` for the numerical contract.
ZENDNNL_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void silu_and_mul_store_pair_f16(
        __m512 lo, __m512 hi, float16_t *dst_row) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdx);
    __m512 gate = _mm512_permutex2var_ps(lo, gate_idx, hi);
    __m512 up = _mm512_permutex2var_ps(lo, up_idx, hi);
    __m512 r = _mm512_mul_ps(silu_avx512(gate), up);
    __m256i out = f32_to_f16x16(r);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// gelu_and_mul — gelu(gate) * up (gelu_tanh form, see the bf16
// sibling's `gelu_and_mul_store_pair` for the numerical contract).
ZENDNNL_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void gelu_and_mul_store_pair_f16(
        __m512 lo, __m512 hi, float16_t *dst_row) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdx);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdx);
    __m512 gate = _mm512_permutex2var_ps(lo, gate_idx, hi);
    __m512 up = _mm512_permutex2var_ps(lo, up_idx, hi);
    __m512 r = _mm512_mul_ps(gelu_avx512(gate), up);
    __m256i out = f32_to_f16x16(r);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// Widens the 32 FP16 lanes of `v` to FP32 (`lo` = lanes 0..15).  Inline
// intrinsics: an out-of-line convert would make the compiler spill every
// live accumulator around each call.
ZENDNNL_INLINE_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void widen_f16x32(__m512h v, __m512 &lo, __m512 &hi) {
    const __m512i bits = _mm512_castph_si512(v);
    lo = _mm512_cvtph_ps(_mm512_castsi512_si256(bits));
    hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 1));
}

// Stores 32 FP32 lanes (`lo`, `hi`) as F16 at `dst`.
ZENDNNL_INLINE_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void store_f32x32_as_f16(float16_t *dst, __m512 lo, __m512 hi) {
    constexpr int kRne = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(dst), _mm512_cvtps_ph(lo, kRne));
    _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(dst + 16), _mm512_cvtps_ph(hi, kRne));
}

// Widens 16 bias columns starting at `col` to FP32.
ZENDNNL_INLINE_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline __m512 bias16_f32(const void *bias, BiasKind bias_kind, int col) {
    if (bias_kind == BiasKind::bf16) {
        return bf16x16_to_f32(
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                        static_cast<const bfloat16_t *>(bias) + col)));
    }
    if (bias_kind == BiasKind::f16) {
        return _mm512_cvtph_ps(
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                        static_cast<const float16_t *>(bias) + col)));
    }
    return _mm512_loadu_ps(static_cast<const float *>(bias) + col);
}

// ─────────────────────────────────────────────────────────────────────
// Shared FP16 microkernel body — MR rows over NJ 32-column accumulators
// per row.  Column block j of B row k is at `B + j * jstride + k *
// KStride`; its output columns start at j * 32 (j * 16 in the tight
// output of a gated `Act`).
//
// The native FP16 FMAs accumulate K in blocks of `kF16AccumBlockK`;
// each block is widened and added into FP32 sums (`sum`, MR x NJ * 32
// floats on the stack, L1-resident), so the FP16 rounding error of the
// running sum is bounded by one block instead of growing with the whole
// K.  Bias, the epilogue and the store work from the FP32 sums.  With
// `load_c` (Act = none only) the sums start from `Cout` instead of the
// bias, continuing a K-chunked reduction.
// ─────────────────────────────────────────────────────────────────────
template <int MR, int NJ, int KStride, ActivationKind Act, typename DstT>
ZENDNNL_INLINE_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void f16_kernel_body(const float16_t *__restrict A, int lda,
        const float16_t *__restrict B, size_t jstride,
        const void *__restrict bias, BiasKind bias_kind,
        void *__restrict Cout_void, int ldc, void *__restrict Cout_tight_void,
        int ldc_tight, int K, bool load_c) {
    static_assert(
            Act == ActivationKind::none || std::is_same<DstT, float16_t>::value,
            "Gated-activation kinds are F16-dst only");
    static_assert(std::is_same<DstT, float16_t>::value
                    || std::is_same<DstT, float>::value,
            "DstT must be float16_t or float");
    constexpr int kCols = NJ * 32;
    DstT *__restrict Cout = static_cast<DstT *>(Cout_void);
    float16_t *__restrict Cout_tight
            = static_cast<float16_t *>(Cout_tight_void);

    alignas(64) float sum[MR][kCols];
    if (Act == ActivationKind::none && load_c) {
#pragma GCC unroll 12
        for (int m = 0; m < MR; ++m)
#pragma GCC unroll 8
            for (int c = 0; c < kCols; c += 16) {
                const DstT *src = Cout + static_cast<size_t>(m) * ldc + c;
                __m512 v;
                if constexpr (std::is_same<DstT, float16_t>::value) {
                    v = _mm512_cvtph_ps(_mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(src)));
                } else {
                    v = _mm512_loadu_ps(src);
                }
                _mm512_store_ps(&sum[m][c], v);
            }
    } else if (bias != nullptr && bias_kind != BiasKind::none) {
#pragma GCC unroll 8
        for (int c = 0; c < kCols; c += 16) {
            const __m512 b = bias16_f32(bias, bias_kind, c);
#pragma GCC unroll 12
            for (int m = 0; m < MR; ++m)
                _mm512_store_ps(&sum[m][c], b);
        }
    } else {
#pragma GCC unroll 12
        for (int m = 0; m < MR; ++m)
#pragma GCC unroll 8
            for (int c = 0; c < kCols; c += 16)
                _mm512_store_ps(&sum[m][c], _mm512_setzero_ps());
    }

    for (int k0 = 0; k0 < K; k0 += kF16AccumBlockK) {
        const int k1 = std::min(K, k0 + kF16AccumBlockK);
        __m512h acc[MR][NJ];
#pragma GCC unroll 12
        for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
            for (int j = 0; j < NJ; ++j)
                acc[m][j] = _mm512_setzero_ph();
        for (int k = k0; k < k1; ++k) {
            __m512h bv[NJ];
#pragma GCC unroll 4
            for (int j = 0; j < NJ; ++j)
                bv[j] = _mm512_loadu_ph(
                        B + j * jstride + static_cast<size_t>(k) * KStride);
#pragma GCC unroll 12
            for (int m = 0; m < MR; ++m) {
                const __m512h av = _mm512_set1_ph(
                        f16_raw_to_scalar(A[static_cast<size_t>(m) * lda + k]));
#pragma GCC unroll 4
                for (int j = 0; j < NJ; ++j)
                    acc[m][j] = _mm512_fmadd_ph(av, bv[j], acc[m][j]);
            }
        }
#pragma GCC unroll 12
        for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
            for (int j = 0; j < NJ; ++j) {
                __m512 lo, hi;
                widen_f16x32(acc[m][j], lo, hi);
                float *s = &sum[m][j * 32];
                _mm512_store_ps(s, _mm512_add_ps(_mm512_load_ps(s), lo));
                _mm512_store_ps(
                        s + 16, _mm512_add_ps(_mm512_load_ps(s + 16), hi));
            }
    }

#pragma GCC unroll 12
    for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
        for (int j = 0; j < NJ; ++j) {
            const __m512 lo = _mm512_load_ps(&sum[m][j * 32]);
            const __m512 hi = _mm512_load_ps(&sum[m][j * 32 + 16]);
            if constexpr (Act != ActivationKind::none) {
                float16_t *dst = Cout_tight + static_cast<size_t>(m) * ldc_tight
                        + j * 16;
                if constexpr (Act == ActivationKind::swiglu_oai_mul) {
                    swiglu_oai_store_pair_f16(lo, hi, dst);
                } else if constexpr (Act == ActivationKind::silu_and_mul) {
                    silu_and_mul_store_pair_f16(lo, hi, dst);
                } else {
                    gelu_and_mul_store_pair_f16(lo, hi, dst);
                }
            } else {
                DstT *dst = Cout + static_cast<size_t>(m) * ldc + j * 32;
                if constexpr (std::is_same<DstT, float16_t>::value) {
                    store_f32x32_as_f16(dst, lo, hi);
                } else {
                    _mm512_storeu_ps(dst, lo);
                    _mm512_storeu_ps(dst + 16, hi);
                }
            }
        }
}

// ─────────────────────────────────────────────────────────────────────
// Decode microkernel — MR ∈ 1..8, NV ∈ {2, 4}, Act, DstT: NV / 2
// 32-column accumulators per row over one o-block (NR = NV * 16).
//
// `noinline` keeps each specialization a single callable the
// dispatcher reaches through a function pointer (matches the bf16 /
// int8 siblings).
// ─────────────────────────────────────────────────────────────────────
template <int MR, int NV, ActivationKind Act, typename DstT>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static void ukernel_impl(const float16_t *__restrict A, int lda,
        const float16_t *__restrict Bpacked, const void *__restrict bias,
        BiasKind bias_kind, void *__restrict Cout_void, int ldc,
        void *__restrict Cout_tight_void, int ldc_tight, int K) {
    static_assert(NV == 2 || NV == 4, "NV must be 2 or 4");
    f16_kernel_body<MR, NV / 2, NV * 16, Act, DstT>(A, lda, Bpacked, 32, bias,
            bias_kind, Cout_void, ldc, Cout_tight_void, ldc_tight, K, false);
}

// Single-expert prompt microkernel over NB ∈ 1..2 adjacent NR=32
// o-blocks (`b_stride` elements apart) and MR ∈ 1..12 rows: up to 24
// `__m512h` accumulators, against the 8 of the NV=2 `ukernel_impl`
// at MR 8.
template <int MR, int NB, ActivationKind Act, typename DstT>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static void prompt_ukernel_impl(const float16_t *__restrict A, int lda,
        const float16_t *__restrict Bpacked, size_t b_stride,
        const void *__restrict bias, BiasKind bias_kind,
        void *__restrict Cout_void, int ldc, void *__restrict Cout_tight_void,
        int ldc_tight, int K, bool load_c) {
    static_assert(NB >= 1 && NB <= kF16PromptNb, "NB out of range");
    static_assert(MR >= 1 && MR <= kMaxF16PromptMr, "MR out of range");
    f16_kernel_body<MR, NB, 32, Act, DstT>(A, lda, Bpacked, b_stride, bias,
            bias_kind, Cout_void, ldc, Cout_tight_void, ldc_tight, K, load_c);
}

template <int NB, ActivationKind Act, typename DstT, int... MRs>
f16_prompt_ukernel_fn_t pick_prompt_mr(
        int MR, std::integer_sequence<int, MRs...>) {
    f16_prompt_ukernel_fn_t fn = nullptr;
    ((MR == MRs + 1 ? (fn = prompt_ukernel_impl<MRs + 1, NB, Act, DstT>) : fn),
            ...);
    return fn;
}

template <int NB, ActivationKind Act, typename DstT>
f16_prompt_ukernel_fn_t pick_prompt(int MR) {
    return pick_prompt_mr<NB, Act, DstT>(
            MR, std::make_integer_sequence<int, kMaxF16PromptMr> {});
}

template <ActivationKind Act, typename DstT>
f16_prompt_ukernel_fn_t pick_prompt_nb(int MR, int NB) {
    return NB == 1 ? pick_prompt<1, Act, DstT>(MR)
                   : pick_prompt<2, Act, DstT>(MR);
}

template <int MR, int NV>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static void kblock_ukernel_impl(const float16_t *__restrict A, int lda,
        const float16_t *__restrict Bpacked, float *__restrict accum, int ldacc,
        int K, bool add) {
    static_assert(NV == 2 || NV == 4, "NV must be 2 or 4");
    f16_kernel_body<MR, NV / 2, NV * 16, ActivationKind::none, float>(A, lda,
            Bpacked, 32, /*bias=*/nullptr, BiasKind::none, accum, ldacc,
            /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K, add);
}

} // namespace

#endif // ZENDNNL_GRP_F16_CK_AVAILABLE

f16_prompt_ukernel_fn_t select_f16_prompt_ukernel(
        int MR, int NB, ActivationKind act, OutputType dst_dt) {
#if ZENDNNL_GRP_F16_CK_AVAILABLE
    if (MR < 1 || MR > kMaxF16PromptMr || NB < 1 || NB > kF16PromptNb)
        return nullptr;
    switch (act) {
        case ActivationKind::swiglu_oai_mul:
            return dst_dt == OutputType::kF16
                    ? pick_prompt_nb<ActivationKind::swiglu_oai_mul, float16_t>(
                              MR, NB)
                    : nullptr;
        case ActivationKind::silu_and_mul:
            return dst_dt == OutputType::kF16
                    ? pick_prompt_nb<ActivationKind::silu_and_mul, float16_t>(
                              MR, NB)
                    : nullptr;
        case ActivationKind::gelu_and_mul:
            return dst_dt == OutputType::kF16
                    ? pick_prompt_nb<ActivationKind::gelu_and_mul, float16_t>(
                              MR, NB)
                    : nullptr;
        case ActivationKind::none:
            if (dst_dt == OutputType::kF16)
                return pick_prompt_nb<ActivationKind::none, float16_t>(MR, NB);
            if (dst_dt == OutputType::kF32)
                return pick_prompt_nb<ActivationKind::none, float>(MR, NB);
            return nullptr;
        default: return nullptr;
    }
#else
    (void)MR;
    (void)NB;
    (void)act;
    (void)dst_dt;
    return nullptr;
#endif
}

// ── Function-pointer table dispatch (mirrors select_bf16_ukernel) ─────────
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
f16_ukernel_fn_t select_f16_ukernel(
        int MR, int NV, ActivationKind act, OutputType dst_dt) {
#if ZENDNNL_GRP_F16_CK_AVAILABLE
    if (dst_dt != OutputType::kF16 && dst_dt != OutputType::kF32)
        return nullptr;
    const bool is_gated_act = (act == ActivationKind::swiglu_oai_mul
            || act == ActivationKind::silu_and_mul
            || act == ActivationKind::gelu_and_mul);
    if (is_gated_act && dst_dt != OutputType::kF16) { return nullptr; }
    if (NV == 2) {
        if (act == ActivationKind::swiglu_oai_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 2, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 2:
                    return ukernel_impl<2, 2, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 3:
                    return ukernel_impl<3, 2, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 4:
                    return ukernel_impl<4, 2, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 5:
                    return ukernel_impl<5, 2, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 6:
                    return ukernel_impl<6, 2, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 7:
                    return ukernel_impl<7, 2, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 8:
                    return ukernel_impl<8, 2, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                default: return nullptr;
            }
        }
        if (act == ActivationKind::silu_and_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 2, ActivationKind::silu_and_mul,
                            float16_t>;
                case 2:
                    return ukernel_impl<2, 2, ActivationKind::silu_and_mul,
                            float16_t>;
                case 3:
                    return ukernel_impl<3, 2, ActivationKind::silu_and_mul,
                            float16_t>;
                case 4:
                    return ukernel_impl<4, 2, ActivationKind::silu_and_mul,
                            float16_t>;
                case 5:
                    return ukernel_impl<5, 2, ActivationKind::silu_and_mul,
                            float16_t>;
                case 6:
                    return ukernel_impl<6, 2, ActivationKind::silu_and_mul,
                            float16_t>;
                case 7:
                    return ukernel_impl<7, 2, ActivationKind::silu_and_mul,
                            float16_t>;
                case 8:
                    return ukernel_impl<8, 2, ActivationKind::silu_and_mul,
                            float16_t>;
                default: return nullptr;
            }
        }
        if (act == ActivationKind::gelu_and_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 2, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 2:
                    return ukernel_impl<2, 2, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 3:
                    return ukernel_impl<3, 2, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 4:
                    return ukernel_impl<4, 2, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 5:
                    return ukernel_impl<5, 2, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 6:
                    return ukernel_impl<6, 2, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 7:
                    return ukernel_impl<7, 2, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 8:
                    return ukernel_impl<8, 2, ActivationKind::gelu_and_mul,
                            float16_t>;
                default: return nullptr;
            }
        }
        // act == ActivationKind::none — branch on OutputType for the store epilogue.
        if (dst_dt == OutputType::kF32) {
            switch (MR) {
                case 1: return ukernel_impl<1, 2, ActivationKind::none, float>;
                case 2: return ukernel_impl<2, 2, ActivationKind::none, float>;
                case 3: return ukernel_impl<3, 2, ActivationKind::none, float>;
                case 4: return ukernel_impl<4, 2, ActivationKind::none, float>;
                case 5: return ukernel_impl<5, 2, ActivationKind::none, float>;
                case 6: return ukernel_impl<6, 2, ActivationKind::none, float>;
                case 7: return ukernel_impl<7, 2, ActivationKind::none, float>;
                case 8: return ukernel_impl<8, 2, ActivationKind::none, float>;
                default: return nullptr;
            }
        }
        switch (MR) {
            case 1: return ukernel_impl<1, 2, ActivationKind::none, float16_t>;
            case 2: return ukernel_impl<2, 2, ActivationKind::none, float16_t>;
            case 3: return ukernel_impl<3, 2, ActivationKind::none, float16_t>;
            case 4: return ukernel_impl<4, 2, ActivationKind::none, float16_t>;
            case 5: return ukernel_impl<5, 2, ActivationKind::none, float16_t>;
            case 6: return ukernel_impl<6, 2, ActivationKind::none, float16_t>;
            case 7: return ukernel_impl<7, 2, ActivationKind::none, float16_t>;
            case 8: return ukernel_impl<8, 2, ActivationKind::none, float16_t>;
            default: return nullptr;
        }
    }
    if (NV == 4) {
        if (act == ActivationKind::swiglu_oai_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 4, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 2:
                    return ukernel_impl<2, 4, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 3:
                    return ukernel_impl<3, 4, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 4:
                    return ukernel_impl<4, 4, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 5:
                    return ukernel_impl<5, 4, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                case 6:
                    return ukernel_impl<6, 4, ActivationKind::swiglu_oai_mul,
                            float16_t>;
                default: return nullptr;
            }
        }
        if (act == ActivationKind::silu_and_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 4, ActivationKind::silu_and_mul,
                            float16_t>;
                case 2:
                    return ukernel_impl<2, 4, ActivationKind::silu_and_mul,
                            float16_t>;
                case 3:
                    return ukernel_impl<3, 4, ActivationKind::silu_and_mul,
                            float16_t>;
                case 4:
                    return ukernel_impl<4, 4, ActivationKind::silu_and_mul,
                            float16_t>;
                case 5:
                    return ukernel_impl<5, 4, ActivationKind::silu_and_mul,
                            float16_t>;
                case 6:
                    return ukernel_impl<6, 4, ActivationKind::silu_and_mul,
                            float16_t>;
                default: return nullptr;
            }
        }
        if (act == ActivationKind::gelu_and_mul) {
            switch (MR) {
                case 1:
                    return ukernel_impl<1, 4, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 2:
                    return ukernel_impl<2, 4, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 3:
                    return ukernel_impl<3, 4, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 4:
                    return ukernel_impl<4, 4, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 5:
                    return ukernel_impl<5, 4, ActivationKind::gelu_and_mul,
                            float16_t>;
                case 6:
                    return ukernel_impl<6, 4, ActivationKind::gelu_and_mul,
                            float16_t>;
                default: return nullptr;
            }
        }
        if (dst_dt == OutputType::kF32) {
            switch (MR) {
                case 1: return ukernel_impl<1, 4, ActivationKind::none, float>;
                case 2: return ukernel_impl<2, 4, ActivationKind::none, float>;
                case 3: return ukernel_impl<3, 4, ActivationKind::none, float>;
                case 4: return ukernel_impl<4, 4, ActivationKind::none, float>;
                case 5: return ukernel_impl<5, 4, ActivationKind::none, float>;
                case 6: return ukernel_impl<6, 4, ActivationKind::none, float>;
                default: return nullptr;
            }
        }
        switch (MR) {
            case 1: return ukernel_impl<1, 4, ActivationKind::none, float16_t>;
            case 2: return ukernel_impl<2, 4, ActivationKind::none, float16_t>;
            case 3: return ukernel_impl<3, 4, ActivationKind::none, float16_t>;
            case 4: return ukernel_impl<4, 4, ActivationKind::none, float16_t>;
            case 5: return ukernel_impl<5, 4, ActivationKind::none, float16_t>;
            case 6: return ukernel_impl<6, 4, ActivationKind::none, float16_t>;
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

f16_kblock_ukernel_fn_t select_f16_kblock_ukernel(int MR, int NV) {
#if ZENDNNL_GRP_F16_CK_AVAILABLE
    if (NV == 2) {
        switch (MR) {
            case 1: return kblock_ukernel_impl<1, 2>;
            case 2: return kblock_ukernel_impl<2, 2>;
            case 3: return kblock_ukernel_impl<3, 2>;
            case 4: return kblock_ukernel_impl<4, 2>;
            case 5: return kblock_ukernel_impl<5, 2>;
            case 6: return kblock_ukernel_impl<6, 2>;
            case 7: return kblock_ukernel_impl<7, 2>;
            case 8: return kblock_ukernel_impl<8, 2>;
            default: return nullptr;
        }
    }
    if (NV == 4) {
        switch (MR) {
            case 1: return kblock_ukernel_impl<1, 4>;
            case 2: return kblock_ukernel_impl<2, 4>;
            case 3: return kblock_ukernel_impl<3, 4>;
            case 4: return kblock_ukernel_impl<4, 4>;
            case 5: return kblock_ukernel_impl<5, 4>;
            case 6: return kblock_ukernel_impl<6, 4>;
            default: return nullptr;
        }
    }
#else
    (void)MR;
    (void)NV;
#endif
    return nullptr;
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
