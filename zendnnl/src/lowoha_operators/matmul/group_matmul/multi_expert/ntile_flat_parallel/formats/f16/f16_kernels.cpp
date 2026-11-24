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

/// f16 kernels: native AVX-512-FP16 FMAs, one K element per step, into one
/// 32-column FP16 accumulator per row and vector.  Every `kAccumBlockK` K
/// steps the accumulators are widened and added into FP32 sums that start
/// from the bias, which bounds the FP16 rounding of the running sum to one
/// block.  The epilogue works from the FP32 sums and stores f16.
///
/// plain and swiglu_pairs steps are one 32-column vector; split steps are a
/// 32-column gate vector and a 32-column up vector, 32 outputs.

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/f16/f16.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/kernel/nr32_epilogue.hpp"

// The toolchain gate of the custom kernel's FP16 family, which these
// kernels need for `__m512h` and `_Float16`.
#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS) \
        || (defined(__GNUC__) && (__GNUC__ >= 12))
#define ZENDNNL_NR32_F16_KERNELS 1
#else
#define ZENDNNL_NR32_F16_KERNELS 0
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

#if ZENDNNL_NR32_F16_KERNELS

namespace {

using namespace kernel;

/// What the kernels store: their row's output type.
constexpr data_type_t kDst = f16::kFormat.dst;

/// f16 elements from one block row to the next.
constexpr int kRow = 32;
/// K steps between widenings into the FP32 sums.
constexpr int kAccumBlockK = 512;

alignas(64) constexpr int16_t kLane16[32]
        = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

ZENDNNL_INLINE_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline _Float16 f16_scalar(uint16_t raw) {
    _Float16 h;
    std::memcpy(&h, &raw, sizeof(h));
    return h;
}

ZENDNNL_INLINE_TARGET("avx512f,avx512fp16,avx512bw,avx512vl,fma")
static inline void widen_f16x32(__m512h v, __m512 &lo, __m512 &hi) {
    const __m512i bits = _mm512_castph_si512(v);
    lo = _mm512_cvtph_ps(_mm512_castsi512_si256(bits));
    hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 1));
}

template <int MR, epilogue Ep, bool kUnaligned>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512fp16,avx512bw,avx512vl,avx512dq,fma")
static void f16_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 8, "f16 kernels take 1..8 rows");
    static_assert(!kUnaligned || split(Ep),
            "only split epilogues read misaligned columns");
    constexpr int NJ = split(Ep) ? 2 : 1;
    constexpr int kCols = NJ * 32;
    const auto *__restrict A = static_cast<const uint16_t *>(p.a);
    const int lda = p.lda;
    const int K = p.K;
    const auto *__restrict B = reinterpret_cast<const uint16_t *>(p.b);
    const ptrdiff_t next = p.b_next / static_cast<ptrdiff_t>(sizeof(uint16_t));
    [[maybe_unused]] const auto *__restrict B_hi
            = reinterpret_cast<const uint16_t *>(p.b_next_hi);
    [[maybe_unused]] __m512i shift_idx = _mm512_setzero_si512();
    if constexpr (kUnaligned) {
        shift_idx = _mm512_add_epi16(_mm512_load_si512(kLane16),
                _mm512_set1_epi16(static_cast<int16_t>(p.shift)));
    }
    const int cols = p.cols;

    alignas(64) float sum[MR][kCols];
    if (p.bias != nullptr && p.bias_t != bias_kind::none) {
#pragma GCC unroll 8
        for (int c = 0; c < kCols; c += 16) {
            const int off = c % 32;
            const int col = (c < 32 ? 0 : p.col_next) + off;
            const int lanes
                    = (Ep == epilogue::swiglu_pairs ? 2 * cols : cols) - off;
            const __m512 b = load_bias16(p.bias, p.bias_t, col, lanes);
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

    for (int k0 = 0; k0 < K; k0 += kAccumBlockK) {
        const int k1 = std::min(K, k0 + kAccumBlockK);
        __m512h acc[MR][NJ];
#pragma GCC unroll 12
        for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
            for (int j = 0; j < NJ; ++j)
                acc[m][j] = _mm512_setzero_ph();
        for (int k = k0; k < k1; ++k) {
            const uint16_t *bp = B + static_cast<size_t>(k) * kRow;
            __m512h bv[NJ];
            bv[0] = _mm512_loadu_ph(bp);
            if constexpr (NJ == 2) {
                if constexpr (kUnaligned) {
                    bv[1] = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
                            _mm512_loadu_si512(bp + next), shift_idx,
                            _mm512_loadu_si512(
                                    B_hi + static_cast<size_t>(k) * kRow)));
                } else {
                    bv[1] = _mm512_loadu_ph(bp + next);
                }
            }
#pragma GCC unroll 12
            for (int m = 0; m < MR; ++m) {
                const __m512h av = _mm512_set1_ph(
                        f16_scalar(A[static_cast<size_t>(m) * lda + k]));
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

    auto *C = static_cast<out_elem_t<kDst> *>(p.c);
    [[maybe_unused]] const act_mode_t mode
            = gated(Ep) ? act_mode() : act_mode_t {};
#pragma GCC unroll 12
    for (int m = 0; m < MR; ++m) {
        auto *const dst = C + static_cast<size_t>(m) * p.ldc;
        if constexpr (split(Ep)) {
            store_wide_step<Ep, kDst>(dst, _mm512_load_ps(&sum[m][0]),
                    _mm512_load_ps(&sum[m][16]), _mm512_load_ps(&sum[m][32]),
                    _mm512_load_ps(&sum[m][48]), cols, mode);
        } else {
            store_step<Ep, kDst>(dst, _mm512_load_ps(&sum[m][0]),
                    _mm512_load_ps(&sum[m][16]), cols, mode);
        }
    }
}

template <epilogue Ep, bool kUnaligned>
ukernel_fn pick(const int mr) {
    switch (mr) {
        case 1: return f16_kernel<1, Ep, kUnaligned>;
        case 2: return f16_kernel<2, Ep, kUnaligned>;
        case 3: return f16_kernel<3, Ep, kUnaligned>;
        case 4: return f16_kernel<4, Ep, kUnaligned>;
        case 5: return f16_kernel<5, Ep, kUnaligned>;
        case 6: return f16_kernel<6, Ep, kUnaligned>;
        case 7: return f16_kernel<7, Ep, kUnaligned>;
        case 8: return f16_kernel<8, Ep, kUnaligned>;
        default: return nullptr;
    }
}

} // namespace

#endif // ZENDNNL_NR32_F16_KERNELS

namespace f16 {

bool built() {
    return ZENDNNL_NR32_F16_KERNELS != 0;
}

ukernel_fn narrow(const int mr, const epilogue ep, bool, const bool unaligned,
        scale_groups) {
#if ZENDNNL_NR32_F16_KERNELS
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
#else
    (void)mr;
    (void)ep;
    (void)unaligned;
    return nullptr;
#endif
}

} // namespace f16

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
