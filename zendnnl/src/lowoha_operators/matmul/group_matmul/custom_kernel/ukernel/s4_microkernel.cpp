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

/// W4A8 custom microkernel implementation — templated on (MR, NV, Act).
/// See s4_microkernel.hpp for the dequant math and pack layout.
///
/// Per K-octet inner loop, one 64-byte packed load → two VPDPBUSD:
///
///   p   = load(Bpacked + ko*oct_stride + v*64)   // 16 cols × 4 bytes
///   blo = (p & 0x0F) - 8                         // W[k = ko*8 + 0..3]
///   bhi = ((p >> 4) & 0x0F) - 8                  // W[k = ko*8 + 4..7]
///   for m in 0..MR-1:
///     a0 = bcast(A[m, ko*8 + 0..3]) ^ 0x80808080
///     a1 = bcast(A[m, ko*8 + 4..7]) ^ 0x80808080
///     sacc[m][v] = vpdpbusd(sacc[m][v], a0, blo)
///     sacc[m][v] = vpdpbusd(sacc[m][v], a1, bhi)
///
/// Both planes come out in VNNI quad order with AND and shift+AND
/// alone — no shuffle or permute — because the pack put `W[k]` and
/// `W[k+4]` in the same byte.  The `- 8` recovers the signed value
/// from the pack's biased nibble.
///
/// At each group boundary (`group_size / 8` octets):
///
///   facc[m][v] += f32(sacc[m][v] - 128*comp[g][v]) * wei_scale[g][v]
///
/// and finally:
///
///   C[m][v] = facc[m][v] * src_scale[m] + bias[v]  → activation → store
///
/// No K tail handling: the dispatcher guarantees K is an exact
/// multiple of the octet and every group a whole number of octets.

#include "s4_microkernel.hpp"
#include "common/zendnnl_compat.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_act_avx512.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

namespace {

// Emits exactly once per process: proof that dispatch entered the
// compute body, not just that it selected and packed the INT4 path.
// Per-call logging here would serialize the OMP team on the logger
// mutex and destroy the workload being diagnosed.
std::atomic<bool> s_s4_ukernel_announced {false};

using zendnnl::lowoha::matmul::group_matmul_act_avx512::bf16x16_to_f32;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::f32_to_bf16x16;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::gelu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::silu_avx512;
using zendnnl::lowoha::matmul::group_matmul_act_avx512::swiglu_oai_avx512;

// Deinterleave indices for vpermt2ps over two source zmms (32 FP32
// lanes): even lanes are gates, odd lanes are ups.  Duplicated from
// the bf16 / int8 siblings, all three anonymous-namespace local.
alignas(64) constexpr int32_t kGateLaneIdxS4[16]
        = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) constexpr int32_t kUpLaneIdxS4[16]
        = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

// Gated-activation store helpers — same math and pair-pack store as
// the int8 / bf16 siblings, so results stay directly comparable.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
static inline void swiglu_oai_store_pair_s4(
        __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdxS4);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdxS4);
    __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
    __m512 up = _mm512_permutex2var_ps(acc_lo, up_idx, acc_hi);
    __m256i out = f32_to_bf16x16(swiglu_oai_avx512(gate, up));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
static inline void silu_and_mul_store_pair_s4(
        __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdxS4);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdxS4);
    __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
    __m512 up = _mm512_permutex2var_ps(acc_lo, up_idx, acc_hi);
    __m256i out = f32_to_bf16x16(_mm512_mul_ps(silu_avx512(gate), up));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
static inline void gelu_and_mul_store_pair_s4(
        __m512 acc_lo, __m512 acc_hi, bfloat16_t *dst_row) {
    const __m512i gate_idx = _mm512_load_si512(kGateLaneIdxS4);
    const __m512i up_idx = _mm512_load_si512(kUpLaneIdxS4);
    __m512 gate = _mm512_permutex2var_ps(acc_lo, gate_idx, acc_hi);
    __m512 up = _mm512_permutex2var_ps(acc_lo, up_idx, acc_hi);
    __m256i out = f32_to_bf16x16(_mm512_mul_ps(gelu_avx512(gate), up));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row), out);
}

// Load 16 scale lanes (f32 direct, bf16 widened).  A free static
// function rather than a lambda so it carries its own target
// attribute — a lambda does not inherit the caller's, and the
// intrinsics would then fail to inline.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
static inline __m512 load_scale16_s4(
        const void *base, size_t elem_off, ScaleKind scale_kind) {
    if (scale_kind == ScaleKind::kBf16) {
        const auto *p = static_cast<const bfloat16_t *>(base) + elem_off;
        return bf16x16_to_f32(
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
    }
    return _mm512_loadu_ps(static_cast<const float *>(base) + elem_off);
}

// Load 16 bias columns as f32.  f16 goes through memcpy because
// `float16_t` must not be aliased as an integer type.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
static inline __m512 load_bias_f32_s4(
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

// Templated microkernel — MR ∈ 1..max_mr_for_nv_s4(NV), NV ∈ {2, 4},
// Act ∈ {none, swiglu_oai_mul, silu_and_mul, gelu_and_mul}.  `noinline`
// keeps each specialization reachable through a function pointer.
template <int MR, int NV, ActKind Act>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma")
static void ukernel_impl_s4(const uint8_t *__restrict A, int lda,
        const int8_t *__restrict Bpacked, const void *__restrict src_scale,
        const void *__restrict wei_scale, ScaleKind scale_kind,
        const void *__restrict bias, BiasKind bias_kind,
        void *__restrict Cout_void, int ldc, void *__restrict Cout_tight_void,
        int ldc_tight, int K, int group_size, int wei_scale_grp_stride) {

    static_assert(NV == 2 || NV == 4, "NV must be 2 or 4");
    static_assert(Act == ActKind::none || (NV % 2 == 0),
            "gated-activation epilogue requires even NV");

    static const bool s_log = zendnnl::error_handling::apilog_info_enabled();
    if (s_log && !s_s4_ukernel_announced.load(std::memory_order_relaxed)) {
        bool expected = false;
        if (s_s4_ukernel_announced.compare_exchange_strong(expected, true,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
            zendnnl::error_handling::apilog_info(
                    "[GRP_MATMUL.CK.W4A8 UKERNEL] Executing INT4 CK "
                    "microkernel: packed S4 -> register S8 unpack -> "
                    "VPDPBUSD",
                    " MR=", MR, " NV=", NV, " NR=", NV * 16, " K=", K,
                    " group_size=", group_size, " G=", K / group_size, " act=",
                    (Act == ActKind::swiglu_oai_mul ? "swiglu_oai_mul"
                                    : Act == ActKind::silu_and_mul
                                    ? "silu_and_mul"
                                    : Act == ActKind::gelu_and_mul
                                    ? "gelu_and_mul"
                                    : "none"));
        }
    }

    // Per-token source scale, widened on load when bf16.
    const auto src_scale_at = [src_scale, scale_kind](int m) -> float {
        return (scale_kind == ScaleKind::kBf16)
                ? static_cast<float>(
                          static_cast<const bfloat16_t *>(src_scale)[m])
                : static_cast<const float *>(src_scale)[m];
    };

    // One K-octet is `pack_nr` columns × 4 bytes, i.e. NV zmm loads.
    // The dispatcher's gates guarantee no partial trailing octet.
    const int K_oct = K / kS4Octet;
    const int oct_per_group = group_size / kS4Octet;
    const int G = K / group_size;
    constexpr int oct_stride_bytes = NV * 16 * kS4BytesPerOctetCol;
    constexpr int v_stride_bytes = 16 * kS4BytesPerOctetCol;
    const size_t weight_bytes_in_oblock
            = static_cast<size_t>(K_oct) * oct_stride_bytes;
    // Compensation rows follow the weight slab, `pack_nr` int32 lanes
    // per group.
    constexpr int comp_row_stride = NV * 16;
    const int32_t *__restrict comp_base = reinterpret_cast<const int32_t *>(
            Bpacked + weight_bytes_in_oblock);

    // Nibble-plane extraction constants and the s8→u8 recentering XOR.
    const __m512i mask_0f = _mm512_set1_epi8(0x0F);
    const __m512i bias_8 = _mm512_set1_epi8(8);
    const __m512i s8_to_u8_bias_vec
            = _mm512_set1_epi32(static_cast<int32_t>(0x80808080U));
    const __m512i k_sym = _mm512_set1_epi32(128);

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
        // s32 accumulator for this group only.
        __m512i sacc[MR][NV];
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                sacc[m][v] = _mm512_setzero_si512();
            }
        }

        const int oct0 = g * oct_per_group;
        for (int ol = 0; ol < oct_per_group; ++ol) {
            const int ko = oct0 + ol;
            const int8_t *__restrict bp
                    = Bpacked + static_cast<size_t>(ko) * oct_stride_bytes;
            const size_t k_base = static_cast<size_t>(ko) * kS4Octet;

            // v-outer / m-inner keeps the two unpacked B planes live in
            // two registers across all MR rows; the reverse nesting
            // needs 2*NV planes and overflows the file at NV=4.
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                const __m512i p
                        = _mm512_load_si512(reinterpret_cast<const __m512i *>(
                                bp + v * v_stride_bytes));
                // AVX-512 has no byte-wise shift; the 16-bit shift's
                // borrow across the byte boundary is masked off.
                const __m512i blo
                        = _mm512_sub_epi8(_mm512_and_si512(p, mask_0f), bias_8);
                const __m512i bhi = _mm512_sub_epi8(
                        _mm512_and_si512(_mm512_srli_epi16(p, 4), mask_0f),
                        bias_8);
#pragma GCC unroll 8
                for (int m = 0; m < MR; ++m) {
                    const uint8_t *arow
                            = A + static_cast<size_t>(m) * lda + k_base;
                    uint32_t a_q0, a_q1;
                    std::memcpy(&a_q0, arow, sizeof(a_q0));
                    std::memcpy(&a_q1, arow + kVNNIInt8Quad, sizeof(a_q1));
                    const __m512i a0 = _mm512_xor_si512(
                            _mm512_set1_epi32(static_cast<int32_t>(a_q0)),
                            s8_to_u8_bias_vec);
                    const __m512i a1 = _mm512_xor_si512(
                            _mm512_set1_epi32(static_cast<int32_t>(a_q1)),
                            s8_to_u8_bias_vec);
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], a0, blo);
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], a1, bhi);
                }
            }
        }

        // `128 * comp[g][v]` undoes the XOR-0x80 source recentering.
        // The multiplier is constant because W4A8 is symmetric, so it
        // hoists out of the m-loop.
        __m512i corr_v[NV];
        __m512 wscale_v[NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m512i comp_gv
                    = _mm512_load_si512(reinterpret_cast<const __m512i *>(
                            comp_base + static_cast<size_t>(g) * comp_row_stride
                            + v * 16));
            corr_v[v] = _mm512_mullo_epi32(k_sym, comp_gv);
            wscale_v[v] = load_scale16_s4(wei_scale,
                    static_cast<size_t>(g) * wei_scale_grp_stride
                            + static_cast<size_t>(v) * 16,
                    scale_kind);
        }

#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                const __m512 f = _mm512_cvtepi32_ps(
                        _mm512_sub_epi32(sacc[m][v], corr_v[v]));
                facc[m][v] = _mm512_fmadd_ps(f, wscale_v[v], facc[m][v]);
            }
        }
    }

    // ── Final pass: per-token scale, bias, activation, store ────────
    const bool has_bias = (bias != nullptr && bias_kind != BiasKind::none);
    alignas(64) float bias_slab[NV * 16];
    const float *__restrict bias_f32 = static_cast<const float *>(bias);
    if (has_bias && bias_kind != BiasKind::fp32) {
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            _mm512_store_ps(
                    bias_slab + v * 16, load_bias_f32_s4(bias, bias_kind, v));
        }
        bias_f32 = bias_slab;
    }

    if constexpr (Act == ActKind::swiglu_oai_mul || Act == ActKind::silu_and_mul
            || Act == ActKind::gelu_and_mul) {
        bfloat16_t *__restrict Cout_tight
                = static_cast<bfloat16_t *>(Cout_tight_void);
        constexpr int n_pairs = NV / 2;
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            const __m512 ssc = _mm512_set1_ps(src_scale_at(m));
#pragma GCC unroll 2
            for (int p = 0; p < n_pairs; ++p) {
                __m512 lo = _mm512_mul_ps(facc[m][2 * p], ssc);
                __m512 hi = _mm512_mul_ps(facc[m][2 * p + 1], ssc);
                if (has_bias) {
                    lo = _mm512_add_ps(
                            lo, _mm512_loadu_ps(bias_f32 + 2 * p * 16));
                    hi = _mm512_add_ps(
                            hi, _mm512_loadu_ps(bias_f32 + (2 * p + 1) * 16));
                }
                bfloat16_t *dst = Cout_tight
                        + static_cast<size_t>(m) * ldc_tight + p * 16;
                if constexpr (Act == ActKind::swiglu_oai_mul) {
                    swiglu_oai_store_pair_s4(lo, hi, dst);
                } else if constexpr (Act == ActKind::silu_and_mul) {
                    silu_and_mul_store_pair_s4(lo, hi, dst);
                } else {
                    gelu_and_mul_store_pair_s4(lo, hi, dst);
                }
            }
        }
    } else {
        static_assert(Act == ActKind::none,
                "s4 ukernel supports {none, swiglu, silu, gelu}");
        bfloat16_t *__restrict Cout = static_cast<bfloat16_t *>(Cout_void);
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            const __m512 ssc = _mm512_set1_ps(src_scale_at(m));
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                __m512 f = _mm512_mul_ps(facc[m][v], ssc);
                if (has_bias) {
                    f = _mm512_add_ps(f, _mm512_loadu_ps(bias_f32 + v * 16));
                }
                bfloat16_t *dst = Cout + static_cast<size_t>(m) * ldc + v * 16;
                _mm256_storeu_si256(
                        reinterpret_cast<__m256i *>(dst), f32_to_bf16x16(f));
            }
        }
    }
}

} // namespace

// Instantiation set: NV=2 (NR=32) MR ∈ {1..6}, NV=4 (NR=64) MR ∈
// {1..3}, each × 4 acts = 36 specializations.

template <ActKind Act>
static s4_ukernel_fn_t select_s4_nv2(int MR) {
    switch (MR) {
        case 1: return ukernel_impl_s4<1, 2, Act>;
        case 2: return ukernel_impl_s4<2, 2, Act>;
        case 3: return ukernel_impl_s4<3, 2, Act>;
        case 4: return ukernel_impl_s4<4, 2, Act>;
        case 5: return ukernel_impl_s4<5, 2, Act>;
        case 6: return ukernel_impl_s4<6, 2, Act>;
        default: return nullptr;
    }
}

template <ActKind Act>
static s4_ukernel_fn_t select_s4_nv4(int MR) {
    switch (MR) {
        case 1: return ukernel_impl_s4<1, 4, Act>;
        case 2: return ukernel_impl_s4<2, 4, Act>;
        case 3: return ukernel_impl_s4<3, 4, Act>;
        default: return nullptr;
    }
}

s4_ukernel_fn_t select_s4_ukernel(int MR, int NV, ActKind act) {
    if (NV != 2 && NV != 4) { return nullptr; }
    if (MR < 1 || MR > max_mr_for_nv_s4(NV)) { return nullptr; }
    if (NV == 2) {
        switch (act) {
            case ActKind::none: return select_s4_nv2<ActKind::none>(MR);
            case ActKind::swiglu_oai_mul:
                return select_s4_nv2<ActKind::swiglu_oai_mul>(MR);
            case ActKind::silu_and_mul:
                return select_s4_nv2<ActKind::silu_and_mul>(MR);
            case ActKind::gelu_and_mul:
                return select_s4_nv2<ActKind::gelu_and_mul>(MR);
            default: return nullptr;
        }
    }
    switch (act) {
        case ActKind::none: return select_s4_nv4<ActKind::none>(MR);
        case ActKind::swiglu_oai_mul:
            return select_s4_nv4<ActKind::swiglu_oai_mul>(MR);
        case ActKind::silu_and_mul:
            return select_s4_nv4<ActKind::silu_and_mul>(MR);
        case ActKind::gelu_and_mul:
            return select_s4_nv4<ActKind::gelu_and_mul>(MR);
        default: return nullptr;
    }
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
