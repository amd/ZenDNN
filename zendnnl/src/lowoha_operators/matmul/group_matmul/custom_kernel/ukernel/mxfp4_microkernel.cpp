/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// MXFP4 custom microkernel implementation — templated on (MR, NV, Act) and
/// the source-scale layout; the output type is a run-time argument.
/// See mxfp4_microkernel.hpp for the dequant math and the pack it reads.
///
/// Per K-octet inner loop, one 64-byte packed load → two VPDPBUSD:
///
///   p   = load(Bpacked + ko*oct_stride + v*64)   // 16 cols × 4 bytes
///   ulo = lut[p & 0x0F]                          // 2·W[k = ko*8 + 0..3] + 12
///   uhi = lut[(p >> 4) & 0x0F]                   // 2·W[k = ko*8 + 4..7] + 12
///   for m in 0..MR-1:
///     a0 = bcast(A[m, ko*8 + 0..3])
///     a1 = bcast(A[m, ko*8 + 4..7])
///     sacc[m][v] = vpdpbusd(sacc[m][v], ulo, a0)
///     sacc[m][v] = vpdpbusd(sacc[m][v], uhi, a1)
///
/// E2M1 is a floating grid, so the nibble map is a 16-entry `vpshufb` table
/// rather than a subtract.  At each group boundary (`group_size / 8`
/// octets):
///
///   sacc[m][v] starts at -12 * Σ_{k∈g} A[m,k]   (`src_group_sums`)
///   facc[m][v] += f32(sacc[m][v]) * 2^(e[g][v] - 128)
///
/// With per-block source scales the block's factor is
/// `src_scale[m][g] * 2^(e[g][v] - 128)`, zero where the block's dot is, so
/// an overflowing product cannot turn a zero term into NaN.
///
/// The pack stores e - 1 and the kernel shifts that byte into the f32
/// exponent field.  It lowers an o-block whose scales would overflow f32 by
/// D, stored in the byte after the scale rows; that o-block takes its
/// per-token source scale and 2^D before the final pass, which then scales
/// by one.  Every other per-token o-block takes its source scale in the
/// final pass:
///
///   C[m][v] = facc[m][v] * src_scale[m] + bias[v]  → activation → bf16/f32
///
/// The bias add rounds on its own (`add_bias_rounded`), so every row and
/// column rounds the same way.
///
/// No K tail handling: the dispatcher guarantees K is an exact
/// multiple of the octet and every group a whole number of octets.

#include "mxfp4_microkernel.hpp"
#include "common/zendnnl_compat.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_act_avx512.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

namespace {

// Emits exactly once per process: proof that dispatch entered the compute
// body, not just that it selected and packed the MXFP4 path.  Per-call
// logging here would serialize the OMP team on the logger mutex and destroy
// the workload being diagnosed.
std::atomic<bool> s_mxfp4_ukernel_announced {false};

using zendnnl::lowoha::matmul::group_matmul_act_avx512::act_mode;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::act_mode_t;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::bf16x16_to_f32;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::f32_to_bf16x16;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::gelu_act_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::silu_act_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::swiglu_oai_act_avx512;

// Deinterleave indices for vpermt2ps over two source zmms (32 FP32
// lanes): even lanes are gates, odd lanes are ups.  Duplicated from
// the bf16 / int8 / s4 siblings, all anonymous-namespace local.
alignas(64) constexpr int32_t kGateLaneIdxMxfp4[16]
        = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) constexpr int32_t kUpLaneIdxMxfp4[16]
        = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

// Gated-activation store helpers — same math and pair-pack store as
// the int8 / bf16 / s4 siblings, so results stay directly comparable.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
ZENDNNL_ALWAYS_INLINE static inline void swiglu_oai_store_pair_mxfp4(
        __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row, act_mode_t mode) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdxMxfp4);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdxMxfp4);
    __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
    __m512 up = _mm512_permutex2var_ps(acc_lo, up_idx, acc_hi);
    __m256i out = f32_to_bf16x16(swiglu_oai_act_avx512(gate, up, mode));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
ZENDNNL_ALWAYS_INLINE static inline void silu_and_mul_store_pair_mxfp4(
        __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row, act_mode_t mode) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdxMxfp4);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdxMxfp4);
    __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
    __m512 up = _mm512_permutex2var_ps(acc_lo, up_idx, acc_hi);
    __m256i out
            = f32_to_bf16x16(_mm512_mul_ps(silu_act_avx512(gate, mode), up));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
ZENDNNL_ALWAYS_INLINE static inline void gelu_and_mul_store_pair_mxfp4(
        __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row, act_mode_t mode) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdxMxfp4);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdxMxfp4);
    __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
    __m512 up = _mm512_permutex2var_ps(acc_lo, up_idx, acc_hi);
    __m256i out
            = f32_to_bf16x16(_mm512_mul_ps(gelu_act_avx512(gate, mode), up));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// Load 16 bias columns as f32.  f16 goes through memcpy because
// `float16_t` must not be aliased as an integer type.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
static inline __m512 load_bias_f32_mxfp4(
        const void *bias, BiasKind bias_kind, int v) {
    if (bias_kind == BiasKind::bf16) {
        const auto *b = static_cast<const bfloat16_t *>(bias);
        return bf16x16_to_f32(_mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(b + v * 16)));
    }
    if (bias_kind == BiasKind::f16) {
        __m256i h16;
        const auto *bias_bytes = static_cast<const char *>(bias);
        std::memcpy(&h16, bias_bytes + static_cast<size_t>(v) * sizeof(h16),
                sizeof(h16));
        return _mm512_cvtph_ps(h16);
    }
    return _mm512_loadu_ps(static_cast<const float *>(bias) + v * 16);
}

// The final pass's bias add, rounded on its own: an add with embedded
// rounding is one the compiler cannot contract into an FMA with the scale
// multiply before it.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
static inline __m512 add_bias_rounded(__m512 scaled, __m512 bias) {
    return _mm512_add_round_ps(
            scaled, bias, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

// Templated microkernel — MR ∈ 1..max_mr_for_nv_mxfp4(NV), NV ∈ {2, 4},
// Act ∈ {none, swiglu_oai_mul, silu_and_mul, gelu_and_mul}, and one source
// scale per row or per row and block.  The layout is a template argument
// because a run-time branch in the block fold costs the per-row kernel
// registers at MR = 6.  `noinline` keeps each specialization reachable
// through a function pointer.
template <int MR, int NV, ActivationKind Act, bool PerBlockSrcScale>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma")
static void ukernel_impl_mxfp4(const uint8_t *__restrict A, int lda,
        const int8_t *__restrict Bpacked, const void *__restrict src_scale,
        ScaleKind src_scale_kind, const void *__restrict bias,
        BiasKind bias_kind, void *__restrict Cout_void, OutputType output,
        int ldc, void *__restrict Cout_tight_void, int ldc_tight, int K,
        int group_size, const int32_t *__restrict src_group_sums) {

    static_assert(NV == 2 || NV == 4, "NV must be 2 or 4");
    static_assert(Act == ActivationKind::none || (NV % 2 == 0),
            "gated-activation epilogue requires even NV");

    static const bool s_log = zendnnl::error_handling::apilog_info_enabled();
    if (s_log && !s_mxfp4_ukernel_announced.load(std::memory_order_relaxed)) {
        bool expected = false;
        if (s_mxfp4_ukernel_announced.compare_exchange_strong(expected, true,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
            zendnnl::error_handling::apilog_info(
                    "[GRP_MATMUL.CK.W4A8 UKERNEL] fmt=mxfp4 Executing MXFP4 CK "
                    "microkernel: packed E2M1 -> LUT to biased U8 -> VPDPBUSD "
                    "(unsigned weight, signed source) MR=",
                    MR, " NV=", NV, " NR=", NV * 16, " K=", K,
                    " group_size=", group_size, " G=", K / group_size,
                    " scales=e8m0-in-pack act=",
                    (Act == ActivationKind::swiglu_oai_mul ? "swiglu_oai_mul"
                                    : Act == ActivationKind::silu_and_mul
                                    ? "silu_and_mul"
                                    : Act == ActivationKind::gelu_and_mul
                                    ? "gelu_and_mul"
                                    : "none"),
                    " src_scale=",
                    (PerBlockSrcScale ? "per_group" : "per_token"));
        }
    }

    // Source scale, widened on load when bf16: `m` per token, `m * G + g`
    // per block.
    const auto src_scale_at = [src_scale, src_scale_kind](int index) -> float {
        return (src_scale_kind == ScaleKind::kBf16)
                ? static_cast<float>(
                          static_cast<const bfloat16_t *>(src_scale)[index])
                : static_cast<const float *>(src_scale)[index];
    };
    constexpr bool per_block_src_scale = PerBlockSrcScale;

    // One K-octet is `pack_nr` columns × 4 bytes, i.e. NV zmm loads.
    // The dispatcher's gates guarantee no partial trailing octet.
    const int K_oct = K / kW4Octet;
    const int oct_per_group = group_size / kW4Octet;
    const int G = K / group_size;
    constexpr int oct_stride_bytes = NV * 16 * kW4BytesPerOctetCol;
    constexpr int v_stride_bytes = 16 * kW4BytesPerOctetCol;
    const size_t weight_bytes_in_oblock
            = static_cast<size_t>(K_oct) * oct_stride_bytes;
    // The scale bytes follow the weight slab with `pack_nr` lanes per group,
    // then the byte saying by how much the pack lowered this o-block's
    // exponents (`e8m0_in_pack_max_exp`).
    constexpr int scale_row_stride = NV * 16;
    const uint8_t *__restrict scale_exp_base
            = reinterpret_cast<const uint8_t *>(
                    Bpacked + weight_bytes_in_oblock);
    const uint32_t exp_shift
            = scale_exp_base[static_cast<size_t>(G) * scale_row_stride];
    // 2^e for 0 <= e <= 127, exactly.
    const auto exp2i = [](uint32_t e) {
        float f;
        const uint32_t bits = (127u + e) << 23;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    };
    // A lowered o-block owes its sums 2^exp_shift.  Per block, as much of it
    // as keeps every source scale of the row at most one moves into those
    // scales, so their product with a lowered weight scale cannot underflow;
    // the rest is restored at the end.
    float row_fold[MR];
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        row_fold[m] = 1.0f;
    }
    if constexpr (PerBlockSrcScale) {
        if (exp_shift != 0) {
            for (int m = 0; m < MR; ++m) {
                float largest = 0.0f;
                for (int g = 0; g < G; ++g)
                    largest = std::max(
                            largest, std::fabs(src_scale_at(m * G + g)));
                uint32_t fold = 0;
                while (fold < exp_shift && largest * exp2i(fold + 1) <= 1.0f)
                    ++fold;
                row_fold[m] = exp2i(fold);
            }
        }
    }
    // `vpshufb` indexes the table by the low 4 bits per 128-bit lane.
    const __m512i e2m1_u8_lut = _mm512_broadcast_i32x4(
            _mm_load_si128(reinterpret_cast<const __m128i *>(kE2M1ToBiasedU8)));
    const __m512i mask_0f = _mm512_set1_epi8(0x0F);

    // Persistent f32 accumulator, summed across groups.
    __m512 facc[MR][NV];
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            facc[m][v] = _mm512_setzero_ps();
        }
    }

    for (int g = 0; g < G; ++g) {
        // The +12 weight offset adds 12*Σx per row; the seed removes it.
        __m512i sacc[MR][NV];
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            const __m512i seed = _mm512_set1_epi32(-kE2M1UnsignedBias
                    * src_group_sums[static_cast<size_t>(m) * G + g]);
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                sacc[m][v] = seed;
            }
        }

        const int oct0 = g * oct_per_group;
        for (int ol = 0; ol < oct_per_group; ++ol) {
            const int ko = oct0 + ol;
            const int8_t *__restrict bp
                    = Bpacked + static_cast<size_t>(ko) * oct_stride_bytes;
            const size_t k_base = static_cast<size_t>(ko) * kW4Octet;

            // v-outer / m-inner keeps the two decoded B planes live in
            // two registers across all MR rows; the reverse nesting
            // needs 2*NV planes and overflows the file at NV=4.
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                const __m512i p
                        = _mm512_load_si512(reinterpret_cast<const __m512i *>(
                                bp + v * v_stride_bytes));
                // AVX-512 has no byte-wise shift; the 16-bit shift's
                // borrow across the byte boundary is masked off.
                const __m512i nlo = _mm512_and_si512(p, mask_0f);
                const __m512i nhi
                        = _mm512_and_si512(_mm512_srli_epi16(p, 4), mask_0f);
                const __m512i ulo = _mm512_shuffle_epi8(e2m1_u8_lut, nlo);
                const __m512i uhi = _mm512_shuffle_epi8(e2m1_u8_lut, nhi);
#pragma GCC unroll 8
                for (int m = 0; m < MR; ++m) {
                    const uint8_t *arow
                            = A + static_cast<size_t>(m) * lda + k_base;
                    uint32_t a_q0, a_q1;
                    std::memcpy(&a_q0, arow, sizeof(a_q0));
                    std::memcpy(&a_q1, arow + kVNNIInt8Quad, sizeof(a_q1));
                    const __m512i a0
                            = _mm512_set1_epi32(static_cast<int32_t>(a_q0));
                    const __m512i a1
                            = _mm512_set1_epi32(static_cast<int32_t>(a_q1));
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], ulo, a0);
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], uhi, a1);
                }
            }
        }

        __m512 wscale_v[NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                    scale_exp_base + static_cast<size_t>(g) * scale_row_stride
                    + v * 16));
            wscale_v[v] = _mm512_castsi512_ps(
                    _mm512_slli_epi32(_mm512_cvtepu8_epi32(e), 23));
        }

        if constexpr (per_block_src_scale) {
#pragma GCC unroll 8
            for (int m = 0; m < MR; ++m) {
                const __m512 ssc
                        = _mm512_set1_ps(src_scale_at(m * G + g) * row_fold[m]);
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v) {
                    const __m512 f = _mm512_cvtepi32_ps(sacc[m][v]);
                    const __m512 scale = _mm512_maskz_mul_ps(
                            _mm512_cmpneq_epi32_mask(
                                    sacc[m][v], _mm512_setzero_si512()),
                            ssc, wscale_v[v]);
                    facc[m][v] = _mm512_fmadd_ps(f, scale, facc[m][v]);
                }
            }
        } else {
#pragma GCC unroll 8
            for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v) {
                    const __m512 f = _mm512_cvtepi32_ps(sacc[m][v]);
                    facc[m][v] = _mm512_fmadd_ps(f, wscale_v[v], facc[m][v]);
                }
            }
        }
    }

    // A lowered o-block takes its source scale and the shift back here, and
    // the final pass scales by one, exactly.  Per token, the shift folds into
    // the source scale, so the sums take both in one rounding with no
    // underflow in between; a folded scale past f32's range keeps them
    // apart, the shift last, so the values stay finite.  Per block, the
    // scales are already in `facc` and the shift left after `row_fold`
    // follows.  Every other o-block takes its per-token source scale in the
    // final pass.
    const bool lowered = exp_shift != 0;
    if (lowered) {
        const float restore = exp2i(exp_shift);
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            float ssc = restore / row_fold[m];
            if constexpr (!per_block_src_scale) {
                ssc = src_scale_at(m) * restore;
                if (!std::isfinite(ssc)) {
                    const __m512 token = _mm512_set1_ps(src_scale_at(m));
#pragma GCC unroll 4
                    for (int v = 0; v < NV; ++v) {
                        facc[m][v] = _mm512_mul_ps(
                                _mm512_mul_ps(facc[m][v], token),
                                _mm512_set1_ps(restore));
                    }
                    continue;
                }
            }
            const __m512 folded = _mm512_set1_ps(ssc);
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                facc[m][v] = _mm512_mul_ps(facc[m][v], folded);
            }
        }
    }
    const bool final_src_scale = !lowered && !per_block_src_scale;

    // ── Final pass: per-token scale, bias, activation, store ────────
    const bool has_bias = (bias != nullptr && bias_kind != BiasKind::none);
    alignas(64) float bias_slab[NV * 16];
    const float *__restrict bias_f32 = static_cast<const float *>(bias);
    if (has_bias && bias_kind != BiasKind::fp32) {
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            _mm512_store_ps(bias_slab + v * 16,
                    load_bias_f32_mxfp4(bias, bias_kind, v));
        }
        bias_f32 = bias_slab;
    }

    if constexpr (Act == ActivationKind::swiglu_oai_mul
            || Act == ActivationKind::silu_and_mul
            || Act == ActivationKind::gelu_and_mul) {
        bfloat16_t *__restrict Cout_tight
                = static_cast<bfloat16_t *>(Cout_tight_void);
        const act_mode_t mode = act_mode();
        constexpr int n_pairs = NV / 2;
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            const __m512 ssc
                    = _mm512_set1_ps(final_src_scale ? src_scale_at(m) : 1.0f);
#pragma GCC unroll 2
            for (int p = 0; p < n_pairs; ++p) {
                __m512 lo = _mm512_mul_ps(facc[m][2 * p], ssc);
                __m512 hi = _mm512_mul_ps(facc[m][2 * p + 1], ssc);
                if (has_bias) {
                    lo = add_bias_rounded(
                            lo, _mm512_loadu_ps(bias_f32 + 2 * p * 16));
                    hi = add_bias_rounded(
                            hi, _mm512_loadu_ps(bias_f32 + (2 * p + 1) * 16));
                }
                bfloat16_t *dst = Cout_tight
                        + static_cast<size_t>(m) * ldc_tight + p * 16;
                if constexpr (Act == ActivationKind::swiglu_oai_mul) {
                    swiglu_oai_store_pair_mxfp4(lo, hi, dst, mode);
                } else if constexpr (Act == ActivationKind::silu_and_mul) {
                    silu_and_mul_store_pair_mxfp4(lo, hi, dst, mode);
                } else {
                    gelu_and_mul_store_pair_mxfp4(lo, hi, dst, mode);
                }
            }
        }
    } else {
        static_assert(Act == ActivationKind::none,
                "MXFP4 ukernel supports {none, swiglu, silu, gelu}");
        const bool f32_out = output == OutputType::kF32;
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            const __m512 ssc
                    = _mm512_set1_ps(final_src_scale ? src_scale_at(m) : 1.0f);
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                __m512 f = _mm512_mul_ps(facc[m][v], ssc);
                if (has_bias) {
                    f = add_bias_rounded(f, _mm512_loadu_ps(bias_f32 + v * 16));
                }
                const size_t at = static_cast<size_t>(m) * ldc + v * 16;
                if (f32_out) {
                    _mm512_storeu_ps(static_cast<float *>(Cout_void) + at, f);
                } else {
                    _mm256_storeu_si256(
                            reinterpret_cast<__m256i *>(
                                    static_cast<bfloat16_t *>(Cout_void) + at),
                            f32_to_bf16x16(f));
                }
            }
        }
    }
}

template <ActivationKind Act, bool PerBlock>
static mxfp4_ukernel_fn_t select_mxfp4_nv2(int MR) {
    switch (MR) {
        case 1: return ukernel_impl_mxfp4<1, 2, Act, PerBlock>;
        case 2: return ukernel_impl_mxfp4<2, 2, Act, PerBlock>;
        case 3: return ukernel_impl_mxfp4<3, 2, Act, PerBlock>;
        case 4: return ukernel_impl_mxfp4<4, 2, Act, PerBlock>;
        case 5: return ukernel_impl_mxfp4<5, 2, Act, PerBlock>;
        case 6: return ukernel_impl_mxfp4<6, 2, Act, PerBlock>;
        default: return nullptr;
    }
}

template <ActivationKind Act, bool PerBlock>
static mxfp4_ukernel_fn_t select_mxfp4_nv4(int MR) {
    switch (MR) {
        case 1: return ukernel_impl_mxfp4<1, 4, Act, PerBlock>;
        case 2: return ukernel_impl_mxfp4<2, 4, Act, PerBlock>;
        case 3: return ukernel_impl_mxfp4<3, 4, Act, PerBlock>;
        default: return nullptr;
    }
}

template <ActivationKind Act>
static mxfp4_ukernel_fn_t select_mxfp4_for_act(int MR, int NV, bool per_block) {
    if (per_block)
        return NV == 2 ? select_mxfp4_nv2<Act, true>(MR)
                       : select_mxfp4_nv4<Act, true>(MR);
    return NV == 2 ? select_mxfp4_nv2<Act, false>(MR)
                   : select_mxfp4_nv4<Act, false>(MR);
}

} // namespace

mxfp4_ukernel_fn_t select_mxfp4_ukernel(
        int MR, int NV, ActivationKind act, bool per_block_src_scale) {
    if (NV != 2 && NV != 4) { return nullptr; }
    if (MR < 1 || MR > max_mr_for_nv_mxfp4(NV)) { return nullptr; }
    switch (act) {
        case ActivationKind::none:
            return select_mxfp4_for_act<ActivationKind::none>(
                    MR, NV, per_block_src_scale);
        case ActivationKind::swiglu_oai_mul:
            return select_mxfp4_for_act<ActivationKind::swiglu_oai_mul>(
                    MR, NV, per_block_src_scale);
        case ActivationKind::silu_and_mul:
            return select_mxfp4_for_act<ActivationKind::silu_and_mul>(
                    MR, NV, per_block_src_scale);
        case ActivationKind::gelu_and_mul:
            return select_mxfp4_for_act<ActivationKind::gelu_and_mul>(
                    MR, NV, per_block_src_scale);
        default: return nullptr;
    }
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
