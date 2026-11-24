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

/// bf16 kernels: VDPBF16PS over K pairs into FP32 accumulators that start
/// from the bias.  Calls of up to three rows split the accumulators into two
/// sets, fed from even and odd K pairs and summed after the K loop, so the
/// FMA chains keep the units busy.  The K loop stages the next pair's
/// weights before issuing the current pair's FMAs.

#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/bf16/bf16.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/kernel/nr32_epilogue.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

namespace {

using namespace kernel;

/// What the kernels store: their row's output type.
constexpr data_type_t kDst = bf16::kFormat.dst;

/// bf16 elements from one block row to the next.
constexpr int kRow = 32 * 2;

/// The two vectors' weights for K pair `kp`.
template <bool kUnaligned>
ZENDNNL_TARGET("avx512f,avx512bf16,avx512bw,avx512vl")
ZENDNNL_ALWAYS_INLINE static inline void load_pair(const uint16_t *B,
        ptrdiff_t next, const uint16_t *B_hi, __m512i shift_idx, int kp,
        __m512bh (&bv)[2]) {
    const uint16_t *bp = B + static_cast<size_t>(kp) * kRow;
    bv[0] = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i *>(bp));
    if constexpr (kUnaligned) {
        bv[1] = (__m512bh)_mm512_permutex2var_epi32(
                _mm512_loadu_si512(
                        reinterpret_cast<const __m512i *>(bp + next)),
                shift_idx,
                _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
                        B_hi + static_cast<size_t>(kp) * kRow)));
    } else {
        (void)B_hi;
        (void)shift_idx;
        bv[1] = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const __m512i *>(bp + next));
    }
}

template <int MR, epilogue Ep, bool kUnaligned>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bf16,avx512bw,avx512vl,avx512dq,fma")
static void bf16_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 8, "bf16 kernels take 1..8 rows");
    static_assert(!kUnaligned || split(Ep),
            "only split epilogues read misaligned columns");
    constexpr int NV = 2;
    const auto *__restrict A = static_cast<const uint16_t *>(p.a);
    const int lda = p.lda;
    const int K = p.K;
    const auto *__restrict B = reinterpret_cast<const uint16_t *>(p.b);
    const ptrdiff_t next = p.b_next / static_cast<ptrdiff_t>(sizeof(uint16_t));
    const auto *__restrict B_hi
            = reinterpret_cast<const uint16_t *>(p.b_next_hi);
    __m512i shift_idx = _mm512_setzero_si512();
    if constexpr (kUnaligned) shift_idx = shift_index32(p.shift);
    const int cols = p.cols;

    constexpr int kBuffers = (MR <= 3) ? 2 : 1;
    __m512 acc[kBuffers][MR][NV];
    const bool has_bias = p.bias != nullptr && p.bias_t != bias_kind::none;
    if (has_bias) {
        const __m512 bias_vec[NV]
                = {load_bias16(p.bias, p.bias_t, 0, lanes16<Ep>(cols, 0)),
                        load_bias16(p.bias, p.bias_t, p.col_next,
                                lanes16<Ep>(cols, 1))};
#pragma GCC unroll 16
        for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                acc[0][m][v] = bias_vec[v];
            }
        }
        if constexpr (kBuffers > 1) {
#pragma GCC unroll 16
            for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v) {
                    acc[1][m][v] = _mm512_setzero_ps();
                }
            }
        }
    } else {
#pragma GCC unroll 2
        for (int b = 0; b < kBuffers; ++b)
#pragma GCC unroll 16
            for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v)
                    acc[b][m][v] = _mm512_setzero_ps();
    }

    const int K_pair = K / 2;
    const bool k_odd = (K & 1) != 0;
    int kp = 0;
    if (kp + 1 < K_pair) {
        __m512bh bv_cur[NV];
        load_pair<kUnaligned>(B, next, B_hi, shift_idx, kp, bv_cur);
        constexpr int buf0 = 0;
        constexpr int buf1 = (kBuffers == 2) ? 1 : 0;
        for (; kp + 1 < K_pair; kp += 2) {
            __m512bh bv_next[NV];
            load_pair<kUnaligned>(B, next, B_hi, shift_idx, kp + 1, bv_next);
#pragma GCC unroll 16
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair, A + static_cast<size_t>(m) * lda + kp * 2,
                        sizeof(a_pair));
                __m512bh av
                        = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v)
                    acc[buf0][m][v]
                            = _mm512_dpbf16_ps(acc[buf0][m][v], av, bv_cur[v]);
            }
            if (kp + 3 < K_pair) {
                load_pair<kUnaligned>(B, next, B_hi, shift_idx, kp + 2, bv_cur);
            }
#pragma GCC unroll 16
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair,
                        A + static_cast<size_t>(m) * lda + (kp + 1) * 2,
                        sizeof(a_pair));
                __m512bh av
                        = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v)
                    acc[buf1][m][v]
                            = _mm512_dpbf16_ps(acc[buf1][m][v], av, bv_next[v]);
            }
        }
    }
    // One remaining full K pair when K_pair is odd: always into set 0.
    if (kp < K_pair) {
        __m512bh bv[NV];
        load_pair<kUnaligned>(B, next, B_hi, shift_idx, kp, bv);
#pragma GCC unroll 16
        for (int m = 0; m < MR; ++m) {
            uint32_t a_pair;
            std::memcpy(&a_pair, A + static_cast<size_t>(m) * lda + kp * 2,
                    sizeof(a_pair));
            __m512bh av = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = _mm512_dpbf16_ps(acc[0][m][v], av, bv[v]);
        }
        ++kp;
    }
    // A single last K element (odd K): the pack's zero second slot makes
    // the 4-byte weight load safe; the source broadcast carries one value.
    if (k_odd) {
        __m512bh bv[NV];
        load_pair<kUnaligned>(B, next, B_hi, shift_idx, kp, bv);
#pragma GCC unroll 16
        for (int m = 0; m < MR; ++m) {
            uint16_t lo;
            std::memcpy(
                    &lo, A + static_cast<size_t>(m) * lda + kp * 2, sizeof(lo));
            __m512bh av = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(static_cast<uint32_t>(lo)));
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = _mm512_dpbf16_ps(acc[0][m][v], av, bv[v]);
        }
    }

    if constexpr (kBuffers > 1) {
#pragma GCC unroll 16
        for (int m = 0; m < MR; ++m) {
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = _mm512_add_ps(acc[0][m][v], acc[1][m][v]);
        }
    }

    auto *C = static_cast<out_elem_t<kDst> *>(p.c);
    [[maybe_unused]] const act_mode_t mode
            = gated(Ep) ? act_mode() : act_mode_t {};
#pragma GCC unroll 16
    for (int m = 0; m < MR; ++m) {
        store_step<Ep, kDst>(C + static_cast<size_t>(m) * p.ldc, acc[0][m][0],
                acc[0][m][1], cols, mode);
    }
}

/// The gate block's two vectors and the up block's two `next` elements on,
/// for K pair `kp`.
ZENDNNL_TARGET("avx512f,avx512bf16,avx512bw,avx512vl")
ZENDNNL_ALWAYS_INLINE static inline void load_quad(
        const uint16_t *B, ptrdiff_t next, int kp, __m512bh (&bv)[4]) {
    const uint16_t *bp = B + static_cast<size_t>(kp) * kRow;
    bv[0] = (__m512bh)_mm512_loadu_si512(bp);
    bv[1] = (__m512bh)_mm512_loadu_si512(bp + 32);
    bv[2] = (__m512bh)_mm512_loadu_si512(bp + next);
    bv[3] = (__m512bh)_mm512_loadu_si512(bp + next + 32);
}

// Wide split: the gate block's two vectors at `b`, the up block's at
// `b + b_next`; output column j of the step is act(gate j) * up j.  With
// `kTwoChains` even K pairs sum into one set of accumulators and odd ones into
// another, both seeded as the narrow kernels seed theirs, so each output sums
// in the order the narrow kernels sum it at the same row count.
template <int MR, epilogue Ep, bool kTwoChains>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bf16,avx512bw,avx512vl,avx512dq,fma")
static void bf16_wide_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 6, "bf16 wide kernels take 1..6 rows");
    static_assert(!kTwoChains || MR <= 3, "two chains fit three rows");
    static_assert(split(Ep), "wide kernels serve split epilogues");
    constexpr int NV = 4;
    constexpr int kBuffers = kTwoChains ? 2 : 1;
    const auto *__restrict A = static_cast<const uint16_t *>(p.a);
    const int lda = p.lda;
    const int K = p.K;
    const auto *__restrict B = reinterpret_cast<const uint16_t *>(p.b);
    const ptrdiff_t next = p.b_next / static_cast<ptrdiff_t>(sizeof(uint16_t));
    const int cols = p.cols;
    const int col_of[NV] = {0, 16, p.col_next, p.col_next + 16};
    const int lanes_of[NV] = {cols, cols - 16, cols, cols - 16};

    __m512 acc[kBuffers][MR][NV];
    const bool has_bias = p.bias != nullptr && p.bias_t != bias_kind::none;
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
        const __m512 b = has_bias
                ? load_bias16(p.bias, p.bias_t, col_of[v], lanes_of[v])
                : _mm512_setzero_ps();
#pragma GCC unroll 6
        for (int m = 0; m < MR; ++m) {
            acc[0][m][v] = b;
            if constexpr (kBuffers > 1) acc[1][m][v] = _mm512_setzero_ps();
        }
    }

    const int K_pair = K / 2;
    int kp = 0;
    for (; kp + 1 < K_pair; kp += 2) {
#pragma GCC unroll 2
        for (int u = 0; u < 2; ++u) {
            const int buf = (kBuffers == 2) ? u : 0;
            __m512bh bv[NV];
            load_quad(B, next, kp + u, bv);
#pragma GCC unroll 6
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair,
                        A + static_cast<size_t>(m) * lda + (kp + u) * 2,
                        sizeof(a_pair));
                const __m512bh av
                        = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v)
                    acc[buf][m][v]
                            = _mm512_dpbf16_ps(acc[buf][m][v], av, bv[v]);
            }
        }
    }
    // One remaining full K pair when K_pair is odd: always into set 0.
    if (kp < K_pair) {
        __m512bh bv[NV];
        load_quad(B, next, kp, bv);
#pragma GCC unroll 6
        for (int m = 0; m < MR; ++m) {
            uint32_t a_pair;
            std::memcpy(&a_pair, A + static_cast<size_t>(m) * lda + kp * 2,
                    sizeof(a_pair));
            const __m512bh av
                    = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = _mm512_dpbf16_ps(acc[0][m][v], av, bv[v]);
        }
        ++kp;
    }
    // A single last K element (odd K), against the pack's zero second slot.
    if ((K & 1) != 0) {
        __m512bh bv[NV];
        load_quad(B, next, kp, bv);
#pragma GCC unroll 6
        for (int m = 0; m < MR; ++m) {
            uint16_t lo;
            std::memcpy(
                    &lo, A + static_cast<size_t>(m) * lda + kp * 2, sizeof(lo));
            const __m512bh av = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(static_cast<uint32_t>(lo)));
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = _mm512_dpbf16_ps(acc[0][m][v], av, bv[v]);
        }
    }

    if constexpr (kBuffers > 1) {
#pragma GCC unroll 6
        for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = _mm512_add_ps(acc[0][m][v], acc[1][m][v]);
    }

    auto *C = static_cast<out_elem_t<kDst> *>(p.c);
    const act_mode_t mode = act_mode();
#pragma GCC unroll 6
    for (int m = 0; m < MR; ++m) {
        store_wide_step<Ep, kDst>(C + static_cast<size_t>(m) * p.ldc,
                acc[0][m][0], acc[0][m][1], acc[0][m][2], acc[0][m][3], cols,
                mode);
    }
}

template <epilogue Ep, bool kTwoChains>
ukernel_fn pick_wide(const int mr) {
    switch (mr) {
        case 1: return bf16_wide_kernel<1, Ep, kTwoChains>;
        case 2: return bf16_wide_kernel<2, Ep, kTwoChains>;
        case 3: return bf16_wide_kernel<3, Ep, kTwoChains>;
        case 4: return kTwoChains ? nullptr : bf16_wide_kernel<4, Ep, false>;
        case 5: return kTwoChains ? nullptr : bf16_wide_kernel<5, Ep, false>;
        case 6: return kTwoChains ? nullptr : bf16_wide_kernel<6, Ep, false>;
        default: return nullptr;
    }
}

template <epilogue Ep, bool kUnaligned>
ukernel_fn pick(const int mr) {
    switch (mr) {
        case 1: return bf16_kernel<1, Ep, kUnaligned>;
        case 2: return bf16_kernel<2, Ep, kUnaligned>;
        case 3: return bf16_kernel<3, Ep, kUnaligned>;
        case 4: return bf16_kernel<4, Ep, kUnaligned>;
        case 5: return bf16_kernel<5, Ep, kUnaligned>;
        case 6: return bf16_kernel<6, Ep, kUnaligned>;
        case 7: return bf16_kernel<7, Ep, kUnaligned>;
        case 8: return bf16_kernel<8, Ep, kUnaligned>;
        default: return nullptr;
    }
}

} // namespace

namespace bf16 {

ukernel_fn narrow(const int mr, const epilogue ep, bool, const bool unaligned,
        scale_groups) {
    switch (ep) {
        case epilogue::plain:
            return unaligned ? nullptr : pick<epilogue::plain, false>(mr);
        case epilogue::swiglu_pairs:
            return unaligned ? nullptr
                             : pick<epilogue::swiglu_pairs, false>(mr);
        case epilogue::silu_split:
            return unaligned ? pick<epilogue::silu_split, true>(mr)
                             : pick<epilogue::silu_split, false>(mr);
        case epilogue::gelu_split:
            return unaligned ? pick<epilogue::gelu_split, true>(mr)
                             : pick<epilogue::gelu_split, false>(mr);
    }
    return nullptr;
}

ukernel_fn wide(const int mr, const epilogue ep, bool, const bool two_chains,
        scale_groups) {
    switch (ep) {
        case epilogue::silu_split:
            return two_chains ? pick_wide<epilogue::silu_split, true>(mr)
                              : pick_wide<epilogue::silu_split, false>(mr);
        case epilogue::gelu_split:
            return two_chains ? pick_wide<epilogue::gelu_split, true>(mr)
                              : pick_wide<epilogue::gelu_split, false>(mr);
        default: return nullptr;
    }
}

} // namespace bf16

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
