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

/// The epilogue every format's kernels (formats/<format>/<format>_kernels.cpp)
/// share: bias loads, the gated activations, and the stores of a step's
/// outputs, the one place an output type lands.  Internal to those files.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_KERNEL_NR32_EPILOGUE_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_KERNEL_NR32_EPILOGUE_HPP

#include <cstdint>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_act_avx512.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/kernel/nr32_ukernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {
namespace kernel {

using group_matmul_act_avx512::act_mode;
using group_matmul_act_avx512::act_mode_t;
using group_matmul_act_avx512::bf16x16_to_f32;
using group_matmul_act_avx512::f32_to_bf16x16;
using group_matmul_act_avx512::f32_to_f16x16;
using group_matmul_act_avx512::gelu_act_avx512;
using group_matmul_act_avx512::silu_act_avx512;
using group_matmul_act_avx512::swiglu_oai_act_avx512;

alignas(64) inline constexpr int32_t kGateLane[16]
        = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) inline constexpr int32_t kUpLane[16]
        = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
alignas(64) inline constexpr int32_t kLane[16]
        = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

/// Columns of 16-column vector `v` of a step that hold weight columns,
/// given the step's `cols` outputs.  Lanes past them read padding (or, in
/// a split step's gate vector, up columns) and are never stored.
template <epilogue Ep>
constexpr int lanes16(int cols, int v) {
    if constexpr (Ep == epilogue::plain) {
        return cols - 16 * v;
    } else if constexpr (Ep == epilogue::swiglu_pairs) {
        return 2 * cols - 16 * v;
    } else {
        return cols;
    }
}

inline __mmask16 mask16(int lanes) {
    return lanes >= 16   ? static_cast<__mmask16>(0xFFFF)
            : lanes <= 0 ? static_cast<__mmask16>(0)
                         : static_cast<__mmask16>((1u << lanes) - 1u);
}

/// 16 bias columns from `col` as f32.  The caller's bias ends at the
/// weight's last column, so a partial vector reads only its `lanes`.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
static inline __m512 load_bias16(
        const void *bias, bias_kind kind, int col, int lanes) {
    if (lanes >= 16) {
        if (kind == bias_kind::bf16) {
            return bf16x16_to_f32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                            static_cast<const uint16_t *>(bias) + col)));
        }
        if (kind == bias_kind::f16) {
            return _mm512_cvtph_ps(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                            static_cast<const uint16_t *>(bias) + col)));
        }
        return _mm512_loadu_ps(static_cast<const float *>(bias) + col);
    }
    const __mmask16 k = mask16(lanes);
    if (kind == bias_kind::bf16) {
        return bf16x16_to_f32(_mm256_maskz_loadu_epi16(
                k, static_cast<const uint16_t *>(bias) + col));
    }
    if (kind == bias_kind::f16) {
        return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(
                k, static_cast<const uint16_t *>(bias) + col));
    }
    return _mm512_maskz_loadu_ps(k, static_cast<const float *>(bias) + col);
}

/// Stores the first `lanes` of `v` as bf16.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
static inline void store_bf16(uint16_t *dst, __m512 v, int lanes) {
    const __m256i out = f32_to_bf16x16(v);
    if (lanes >= 16) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), out);
    } else if (lanes > 0) {
        _mm256_mask_storeu_epi16(dst, mask16(lanes), out);
    }
}

/// Stores the first `lanes` of `v` as f16.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
static inline void store_f16(uint16_t *dst, __m512 v, int lanes) {
    const __m256i out = f32_to_f16x16(v);
    if (lanes >= 16) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), out);
    } else if (lanes > 0) {
        _mm256_mask_storeu_epi16(dst, mask16(lanes), out);
    }
}

/// One output per (gate, up) pair: adjacent lanes of `lo` and `hi` for
/// swiglu_pairs, the same lane of each for the split epilogues.
template <epilogue Ep>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
ZENDNNL_ALWAYS_INLINE static inline __m512
        gated_out(__m512 lo, __m512 hi, act_mode_t mode) {
    if constexpr (Ep == epilogue::swiglu_pairs) {
        const __m512i gate_idx = _mm512_load_si512(kGateLane);
        const __m512i up_idx = _mm512_load_si512(kUpLane);
        const __m512 gate = _mm512_permutex2var_ps(lo, gate_idx, hi);
        const __m512 up = _mm512_permutex2var_ps(lo, up_idx, hi);
        return swiglu_oai_act_avx512(gate, up, mode);
    } else if constexpr (Ep == epilogue::silu_split) {
        return _mm512_mul_ps(silu_act_avx512(lo, mode), hi);
    } else {
        static_assert(Ep == epilogue::gelu_split, "gated epilogue expected");
        return _mm512_mul_ps(gelu_act_avx512(lo, mode), hi);
    }
}

/// How an output element of type `Dst` (`format_desc::dst`) is held: 16
/// bits for bf16 and f16.  An FP32 output would be held as `float` here,
/// with its stores in `store_out` (the README lists what else it needs).
template <data_type_t Dst>
struct out_elem;
template <>
struct out_elem<data_type_t::bf16> {
    using type = uint16_t;
};
template <>
struct out_elem<data_type_t::f16> {
    using type = uint16_t;
};
template <data_type_t Dst>
using out_elem_t = typename out_elem<Dst>::type;

/// Stores the first `lanes` of `v` as output type `Dst`
/// (`format_desc::dst`).
template <data_type_t Dst>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
ZENDNNL_ALWAYS_INLINE static inline void store_out(
        out_elem_t<Dst> *dst, __m512 v, int lanes) {
    if constexpr (Dst == data_type_t::f16) {
        store_f16(dst, v, lanes);
    } else {
        static_assert(Dst == data_type_t::bf16, "a 16-bit output type");
        store_bf16(dst, v, lanes);
    }
}

/// Stores row `dst` of a step from its two vectors' f32 results: both
/// (`cols` outputs) for `plain`, else one output per (gate, up) pair.
template <epilogue Ep, data_type_t Dst>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
ZENDNNL_ALWAYS_INLINE static inline void store_step(
        out_elem_t<Dst> *dst, __m512 lo, __m512 hi, int cols, act_mode_t mode) {
    if constexpr (Ep == epilogue::plain) {
        store_out<Dst>(dst, lo, cols);
        store_out<Dst>(dst + 16, hi, cols - 16);
    } else {
        store_out<Dst>(dst, gated_out<Ep>(lo, hi, mode), cols);
    }
}

/// Stores row `dst` of a split step of 32 outputs, `cols` of them, from the
/// f32 results of two gate vectors and two up vectors: a wide kernel's gate
/// and up blocks, or the halves of f16's 32-column gate and up vectors.
template <epilogue Ep, data_type_t Dst>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
ZENDNNL_ALWAYS_INLINE static inline void store_wide_step(out_elem_t<Dst> *dst,
        __m512 gate0, __m512 gate1, __m512 up0, __m512 up1, int cols,
        act_mode_t mode) {
    static_assert(split(Ep), "wide steps serve split epilogues");
    store_out<Dst>(dst, gated_out<Ep>(gate0, up0, mode), cols);
    store_out<Dst>(dst + 16, gated_out<Ep>(gate1, up1, mode), cols - 16);
}

/// The 32-bit lane indices that shift a vector pair down by `shift` lanes.
ZENDNNL_TARGET("avx512f")
static inline __m512i shift_index32(int shift) {
    return _mm512_add_epi32(_mm512_load_si512(kLane), _mm512_set1_epi32(shift));
}

} // namespace kernel
} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_KERNEL_NR32_EPILOGUE_HPP
