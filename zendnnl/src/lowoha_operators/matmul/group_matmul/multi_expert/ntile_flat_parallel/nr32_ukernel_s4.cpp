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

/// s4 kernels: symmetric s4 weights with one f32 scale per (group, column)
/// against s8 rows with one scale per row or per (row, group).
///
/// Per group, each row's int32 sums start from the group's packed
/// `-128 * sum(w)`, and every K octet adds two VPDPBUSDs: the octet's low
/// nibbles against the row's first four bytes and the high nibbles against
/// the next four, the bytes XOR-ed with 0x80 to make them unsigned.  The
/// group's sums then scale into FP32 totals by the group's weight scales,
/// times the row's scale of the group under per-group source scales.  After
/// the last group the totals take the row's scale (per-row source scales),
/// then the bias.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <immintrin.h>

#include "nr32_kernel_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

namespace {

using namespace kernel;

/// Bytes from one block row (a K octet) to the next.
constexpr int kRow = 128;

// Rounds the source-scale product before the bias add (no FMA contraction).
#if defined(_MSC_VER) && !defined(__clang__)
#define NR32_S4_ROUND(x)
#else
#define NR32_S4_ROUND(x) asm("" : "+v"(x))
#endif

/// Whether a product of group `g`'s weight scales and the source scale of one
/// of the first `rows` rows may overflow or be NaN.  Where it may, a zero sum
/// must add 0, as the custom kernel's does, rather than 0 times the product;
/// where it can't, both give the same bits.
inline bool scale_product_may_overflow(
        const ukernel_args &p, const int rows, const int g) {
    for (int m = 0; m < rows; ++m) {
        const float s = std::fabs(
                p.src_scale[static_cast<size_t>(m) * p.src_groups + g]);
        if (!(s * p.max_scale <= std::numeric_limits<float>::max()))
            return true;
    }
    return false;
}

template <int MR, epilogue Ep, bool kUnaligned, bool kSrcGroups>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma")
static void s4_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 6, "s4 kernels take 1..6 rows");
    static_assert(!kUnaligned || split(Ep),
            "only split epilogues read misaligned columns");
    constexpr int NV = 2;
    const auto *__restrict A = static_cast<const uint8_t *>(p.a);
    const int lda = p.lda;
    const auto *__restrict B = reinterpret_cast<const int8_t *>(p.b);
    const ptrdiff_t next = p.b_next;
    [[maybe_unused]] const auto *__restrict B_hi
            = reinterpret_cast<const int8_t *>(p.b_next_hi);
    [[maybe_unused]] __m512i shift_idx = _mm512_setzero_si512();
    if constexpr (kUnaligned) shift_idx = shift_index32(p.shift);
    const int oct_per_group = p.group_size / 8;
    const int G = p.K / p.group_size;
    const size_t col_group = static_cast<size_t>(p.col_group);

    const __m512i mask_0f = _mm512_set1_epi8(0x0F);
    const __m512i bias_8 = _mm512_set1_epi8(8);
    const __m512i s8_to_u8
            = _mm512_set1_epi32(static_cast<int32_t>(0x80808080U));

    __m512 facc[MR][NV];
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            facc[m][v] = _mm512_setzero_ps();
        }
    }

    for (int g = 0; g < G; ++g) {
        const int32_t *comp = p.comp + static_cast<size_t>(g) * col_group;
        const __m512i seed_v[NV] = {_mm512_loadu_si512(comp),
                _mm512_loadu_si512(comp + p.col_next)};
        __m512i sacc[MR][NV];
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                sacc[m][v] = seed_v[v];
            }
        }

        const int oct0 = g * oct_per_group;
        for (int ol = 0; ol < oct_per_group; ++ol) {
            const int ko = oct0 + ol;
            const int8_t *__restrict bp = B + static_cast<size_t>(ko) * kRow;
            const size_t k_base = static_cast<size_t>(ko) * 8;

            // v outer, m inner: the two unpacked planes stay in registers
            // across all rows.
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                __m512i pv;
                if (v == 0) {
                    pv = _mm512_load_si512(
                            reinterpret_cast<const __m512i *>(bp));
                } else if constexpr (kUnaligned) {
                    pv = _mm512_permutex2var_epi32(
                            _mm512_load_si512(reinterpret_cast<const __m512i *>(
                                    bp + next)),
                            shift_idx,
                            _mm512_load_si512(reinterpret_cast<const __m512i *>(
                                    B_hi + static_cast<size_t>(ko) * kRow)));
                } else {
                    pv = _mm512_load_si512(
                            reinterpret_cast<const __m512i *>(bp + next));
                }
                // No byte-wise shift: the 16-bit shift's borrow across the
                // byte boundary is masked off.
                const __m512i blo = _mm512_sub_epi8(
                        _mm512_and_si512(pv, mask_0f), bias_8);
                const __m512i bhi = _mm512_sub_epi8(
                        _mm512_and_si512(_mm512_srli_epi16(pv, 4), mask_0f),
                        bias_8);
#pragma GCC unroll 8
                for (int m = 0; m < MR; ++m) {
                    const uint8_t *arow
                            = A + static_cast<size_t>(m) * lda + k_base;
                    uint32_t a_q0, a_q1;
                    std::memcpy(&a_q0, arow, sizeof(a_q0));
                    std::memcpy(&a_q1, arow + 4, sizeof(a_q1));
                    const __m512i a0 = _mm512_xor_si512(
                            _mm512_set1_epi32(static_cast<int32_t>(a_q0)),
                            s8_to_u8);
                    const __m512i a1 = _mm512_xor_si512(
                            _mm512_set1_epi32(static_cast<int32_t>(a_q1)),
                            s8_to_u8);
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], a0, blo);
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], a1, bhi);
                }
            }
        }

        const float *scale = p.scale + static_cast<size_t>(g) * col_group;
        const __m512 wscale_v[NV]
                = {_mm512_loadu_ps(scale), _mm512_loadu_ps(scale + p.col_next)};
        if (kSrcGroups && scale_product_may_overflow(p, MR, g)) {
#pragma GCC unroll 8
            for (int m = 0; m < MR; ++m) {
                const __m512 sa = _mm512_set1_ps(
                        p.src_scale[static_cast<size_t>(m) * p.src_groups + g]);
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v) {
                    const __mmask16 nz = _mm512_cmpneq_epi32_mask(
                            sacc[m][v], _mm512_setzero_si512());
                    facc[m][v] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sacc[m][v]),
                            _mm512_maskz_mul_ps(nz, wscale_v[v], sa),
                            facc[m][v]);
                }
            }
            continue;
        }
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            [[maybe_unused]] __m512 sa;
            if constexpr (kSrcGroups) {
                sa = _mm512_set1_ps(
                        p.src_scale[static_cast<size_t>(m) * p.src_groups + g]);
            }
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                const __m512 f = _mm512_cvtepi32_ps(sacc[m][v]);
                if constexpr (kSrcGroups) {
                    facc[m][v] = _mm512_fmadd_ps(
                            f, _mm512_mul_ps(wscale_v[v], sa), facc[m][v]);
                } else {
                    facc[m][v] = _mm512_fmadd_ps(f, wscale_v[v], facc[m][v]);
                }
            }
        }
    }

    const int cols = p.cols;
    const bool has_bias = p.bias != nullptr && p.bias_t != bias_kind::none;
    const __m512 bias_v[NV]
            = {has_bias ? load_bias16(p.bias, p.bias_t, 0, lanes16<Ep>(cols, 0))
                        : _mm512_setzero_ps(),
                    has_bias ? load_bias16(p.bias, p.bias_t, p.col_next,
                                       lanes16<Ep>(cols, 1))
                             : _mm512_setzero_ps()};
    auto *C = static_cast<uint16_t *>(p.c);
    [[maybe_unused]] const act_mode_t mode
            = gated(Ep) ? act_mode() : act_mode_t {};
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        __m512 out[NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            __m512 f = facc[m][v];
            if constexpr (!kSrcGroups) {
                f = _mm512_mul_ps(f, _mm512_set1_ps(p.src_scale[m]));
                NR32_S4_ROUND(f);
            }
            if (has_bias) f = _mm512_add_ps(f, bias_v[v]);
            out[v] = f;
        }
        uint16_t *dst = C + static_cast<size_t>(m) * p.ldc;
        if constexpr (Ep == epilogue::plain) {
            store_bf16(dst, out[0], cols);
            store_bf16(dst + 16, out[1], cols - 16);
        } else {
            store_bf16(dst, gated_out<Ep>(out[0], out[1], mode), cols);
        }
    }
}
// Wide split: the gate block's two vectors at `b`, the up block's at
// `b + b_next`; output column j of the step is act(gate j) * up j.
template <int MR, epilogue Ep, bool kSrcGroups>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma")
static void s4_wide_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 3, "s4 wide kernels take 1..3 rows");
    static_assert(split(Ep), "wide kernels serve split epilogues");
    constexpr int NV = 4;
    const auto *__restrict A = static_cast<const uint8_t *>(p.a);
    const int lda = p.lda;
    const auto *__restrict B = reinterpret_cast<const int8_t *>(p.b);
    const ptrdiff_t next = p.b_next;
    const ptrdiff_t v_off[NV] = {0, 64, next, next + 64};
    const int oct_per_group = p.group_size / 8;
    const int G = p.K / p.group_size;
    const size_t col_group = static_cast<size_t>(p.col_group);
    const int col_of[NV] = {0, 16, p.col_next, p.col_next + 16};

    const __m512i mask_0f = _mm512_set1_epi8(0x0F);
    const __m512i bias_8 = _mm512_set1_epi8(8);
    const __m512i s8_to_u8
            = _mm512_set1_epi32(static_cast<int32_t>(0x80808080U));

    __m512 facc[MR][NV];
#pragma GCC unroll 3
    for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v)
            facc[m][v] = _mm512_setzero_ps();

    for (int g = 0; g < G; ++g) {
        const int32_t *comp = p.comp + static_cast<size_t>(g) * col_group;
        __m512i sacc[MR][NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m512i seed = _mm512_loadu_si512(comp + col_of[v]);
#pragma GCC unroll 3
            for (int m = 0; m < MR; ++m)
                sacc[m][v] = seed;
        }
        const int oct0 = g * oct_per_group;
        for (int ol = 0; ol < oct_per_group; ++ol) {
            const int ko = oct0 + ol;
            const int8_t *__restrict bp = B + static_cast<size_t>(ko) * kRow;
            const size_t k_base = static_cast<size_t>(ko) * 8;
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                const __m512i pv = _mm512_load_si512(
                        reinterpret_cast<const __m512i *>(bp + v_off[v]));
                const __m512i blo = _mm512_sub_epi8(
                        _mm512_and_si512(pv, mask_0f), bias_8);
                const __m512i bhi = _mm512_sub_epi8(
                        _mm512_and_si512(_mm512_srli_epi16(pv, 4), mask_0f),
                        bias_8);
#pragma GCC unroll 3
                for (int m = 0; m < MR; ++m) {
                    const uint8_t *arow
                            = A + static_cast<size_t>(m) * lda + k_base;
                    uint32_t a_q0, a_q1;
                    std::memcpy(&a_q0, arow, sizeof(a_q0));
                    std::memcpy(&a_q1, arow + 4, sizeof(a_q1));
                    const __m512i a0 = _mm512_xor_si512(
                            _mm512_set1_epi32(static_cast<int32_t>(a_q0)),
                            s8_to_u8);
                    const __m512i a1 = _mm512_xor_si512(
                            _mm512_set1_epi32(static_cast<int32_t>(a_q1)),
                            s8_to_u8);
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], a0, blo);
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], a1, bhi);
                }
            }
        }
        const float *scale = p.scale + static_cast<size_t>(g) * col_group;
        [[maybe_unused]] __m512 sa[MR];
        if constexpr (kSrcGroups) {
#pragma GCC unroll 3
            for (int m = 0; m < MR; ++m) {
                sa[m] = _mm512_set1_ps(
                        p.src_scale[static_cast<size_t>(m) * p.src_groups + g]);
            }
        }
        if (kSrcGroups && scale_product_may_overflow(p, MR, g)) {
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                const __m512 ws = _mm512_loadu_ps(scale + col_of[v]);
#pragma GCC unroll 3
                for (int m = 0; m < MR; ++m) {
                    const __mmask16 nz = _mm512_cmpneq_epi32_mask(
                            sacc[m][v], _mm512_setzero_si512());
                    facc[m][v] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sacc[m][v]),
                            _mm512_maskz_mul_ps(nz, ws, sa[m]), facc[m][v]);
                }
            }
            continue;
        }
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m512 ws = _mm512_loadu_ps(scale + col_of[v]);
#pragma GCC unroll 3
            for (int m = 0; m < MR; ++m) {
                const __m512 f = _mm512_cvtepi32_ps(sacc[m][v]);
                if constexpr (kSrcGroups) {
                    facc[m][v] = _mm512_fmadd_ps(
                            f, _mm512_mul_ps(ws, sa[m]), facc[m][v]);
                } else {
                    facc[m][v] = _mm512_fmadd_ps(f, ws, facc[m][v]);
                }
            }
        }
    }

    const int cols = p.cols;
    const int lanes_of[NV] = {cols, cols - 16, cols, cols - 16};
    const bool has_bias = p.bias != nullptr && p.bias_t != bias_kind::none;
    __m512 bias_v[NV];
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
        bias_v[v] = has_bias
                ? load_bias16(p.bias, p.bias_t, col_of[v], lanes_of[v])
                : _mm512_setzero_ps();
    }
    auto *C = static_cast<uint16_t *>(p.c);
    const act_mode_t mode = act_mode();
#pragma GCC unroll 3
    for (int m = 0; m < MR; ++m) {
        __m512 out[NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            __m512 f = facc[m][v];
            if constexpr (!kSrcGroups) {
                f = _mm512_mul_ps(f, _mm512_set1_ps(p.src_scale[m]));
                NR32_S4_ROUND(f);
            }
            if (has_bias) f = _mm512_add_ps(f, bias_v[v]);
            out[v] = f;
        }
        uint16_t *dst = C + static_cast<size_t>(m) * p.ldc;
        store_bf16(dst, gated_out<Ep>(out[0], out[2], mode), cols);
        store_bf16(dst + 16, gated_out<Ep>(out[1], out[3], mode), cols - 16);
    }
}
#undef NR32_S4_ROUND

template <epilogue Ep, bool kSrcGroups>
ukernel_fn pick_wide(const int mr) {
    switch (mr) {
        case 1: return s4_wide_kernel<1, Ep, kSrcGroups>;
        case 2: return s4_wide_kernel<2, Ep, kSrcGroups>;
        case 3: return s4_wide_kernel<3, Ep, kSrcGroups>;
        default: return nullptr;
    }
}

template <epilogue Ep, bool kUnaligned, bool kSrcGroups>
ukernel_fn pick(const int mr) {
    switch (mr) {
        case 1: return s4_kernel<1, Ep, kUnaligned, kSrcGroups>;
        case 2: return s4_kernel<2, Ep, kUnaligned, kSrcGroups>;
        case 3: return s4_kernel<3, Ep, kUnaligned, kSrcGroups>;
        case 4: return s4_kernel<4, Ep, kUnaligned, kSrcGroups>;
        case 5: return s4_kernel<5, Ep, kUnaligned, kSrcGroups>;
        case 6: return s4_kernel<6, Ep, kUnaligned, kSrcGroups>;
        default: return nullptr;
    }
}

template <bool kSrcGroups>
ukernel_fn pick_ep(const int mr, const epilogue ep, const bool unaligned) {
    switch (ep) {
        case epilogue::plain:
            return unaligned ? nullptr
                             : pick<epilogue::plain, false, kSrcGroups>(mr);
        case epilogue::swiglu_pairs:
            return unaligned
                    ? nullptr
                    : pick<epilogue::swiglu_pairs, false, kSrcGroups>(mr);
        case epilogue::silu_split:
            return unaligned
                    ? pick<epilogue::silu_split, true, kSrcGroups>(mr)
                    : pick<epilogue::silu_split, false, kSrcGroups>(mr);
        case epilogue::gelu_split:
            return unaligned
                    ? pick<epilogue::gelu_split, true, kSrcGroups>(mr)
                    : pick<epilogue::gelu_split, false, kSrcGroups>(mr);
    }
    return nullptr;
}

} // namespace

ukernel_fn select_s4_wide(
        const int mr, const epilogue ep, const bool src_groups) {
    switch (ep) {
        case epilogue::silu_split:
            return src_groups ? pick_wide<epilogue::silu_split, true>(mr)
                              : pick_wide<epilogue::silu_split, false>(mr);
        case epilogue::gelu_split:
            return src_groups ? pick_wide<epilogue::gelu_split, true>(mr)
                              : pick_wide<epilogue::gelu_split, false>(mr);
        default: return nullptr;
    }
}

ukernel_fn select_s4(const int mr, const epilogue ep, const bool unaligned,
        const bool src_groups) {
    return src_groups ? pick_ep<true>(mr, ep, unaligned)
                      : pick_ep<false>(mr, ep, unaligned);
}

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
