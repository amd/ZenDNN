/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

/// Building blocks shared by the data-type policies of the tiled kernel
/// (tiled/<dtype>/), which do not include each other: the Q*K^T tile heights,
/// the opt-in fast silu of the Q*K^T epilogues, the diagonal term every
/// policy adds the same way, and the BF16 P*V over the packed V blocks.

#ifndef LOWOHA_HSTU_TILED_COMMON_HPP
#define LOWOHA_HSTU_TILED_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include "common/bfloat16.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_utils.hpp"
#include "lowoha_operators/hstu_attention/hstu_gemm_ck/lowoha_hstu_ck.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_driver.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Rows of a Q*K^T tile: 8 x 32 columns or 6 x 64 columns (two packed blocks).
constexpr int kMR1 = 8;
constexpr int kMR2 = 6;

/// silu(x) * s for x = c * acc, with t_mul = -c * log2(e) and y_mul = c * s
/// (ZENDNNL_HSTU_ATTENTION_FAST_SILU): x * sigmoid(x) with
/// exp(-x) = 2^t from a degree-3 minimax polynomial of 2^frac(t) and
/// vscalefps for 2^floor(t) (no clamp: it saturates to 0 or inf, and
/// sigmoid to 1 or 0), and rcp14 without a Newton step. Relative error at
/// most 1.2e-4 (`group_matmul_act_avx512::silu_avx512`: 5e-5), well below
/// the 2^-9 rounding of BF16 P and the INT8 P steps.
HSTU_AVX512_INLINE inline __m512 fast_silu_scaled(
        __m512 acc, __m512 t_mul, __m512 y_mul) {
    const __m512 t = _mm512_mul_ps(acc, t_mul);
    const __m512 f = _mm512_sub_ps(t,
            _mm512_roundscale_ps(t, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
    __m512 e = _mm512_fmadd_ps(
            _mm512_set1_ps(0.07802371f), f, _mm512_set1_ps(0.22606677f));
    e = _mm512_fmadd_ps(e, f, _mm512_set1_ps(0.69583452f));
    e = _mm512_fmadd_ps(e, f, _mm512_set1_ps(0.99992502f));
    const __m512 d = _mm512_add_ps(_mm512_set1_ps(1.f), _mm512_scalef_ps(e, t));
    return _mm512_mul_ps(_mm512_mul_ps(acc, y_mul), _mm512_rcp14_ps(d));
}

/// a . b over n BF16 elements, FP32 accumulation.
HSTU_AVX512_NOINLINE inline float dot_bf16(const zendnnl::common::bfloat16_t *a,
        const zendnnl::common::bfloat16_t *b, int64_t n) {
    __m512 acc = _mm512_setzero_ps();
    for (int64_t c = 0; c < n; c += 32) {
        const __mmask32 m
                = __mmask32(range_bits(0, std::min<int64_t>(32, n - c)));
        acc = _mm512_dpbf16_ps(acc,
                (__m512bh)_mm512_maskz_loadu_epi16(m, a + c),
                (__m512bh)_mm512_maskz_loadu_epi16(m, b + c));
    }
    return _mm512_reduce_add_ps(acc);
}

/// y[0, n) += w * x[0, n).
HSTU_AVX512_NOINLINE inline void axpy_bf16(
        float w, const zendnnl::common::bfloat16_t *x, int64_t n, float *y) {
    const __m512 vw = _mm512_set1_ps(w);
    for (int64_t c = 0; c < n; c += 16) {
        const __mmask16 m
                = __mmask16(range_bits(0, std::min<int64_t>(16, n - c)));
        const __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, x + c)), 16));
        _mm512_mask_storeu_ps(y + c, m,
                _mm512_fmadd_ps(vw, xf, _mm512_maskz_loadu_ps(m, y + c)));
    }
}

/// o[0, dv) += w v for the diagonal term of a row outside its band (a target
/// row), from its BF16 q, k and v rows: w = silu(alpha q . k) * scale rounded
/// to BF16 like the P of the BF16 policy, with the exact silu whatever
/// ZENDNNL_HSTU_ATTENTION_FAST_SILU. Every policy adds it through this, so
/// the term does not depend on the data type of the band.
inline void diag_bf16(const zendnnl::common::bfloat16_t *q,
        const zendnnl::common::bfloat16_t *k,
        const zendnnl::common::bfloat16_t *v, int64_t d, int64_t dv,
        float alpha, float scale, float *o) {
    const float x = alpha * dot_bf16(q, k, d);
    const float w = static_cast<float>(
            zendnnl::common::bfloat16_t(x / (1.f + std::exp(-x)) * scale));
    axpy_bf16(w, v, dv, o);
}

/// O[ra, rb) += P[:, plo:phi] V[plo:phi] for a BF16 P tile (rows kBlockN
/// apart) and V packed by `ck::bf16::pack_b` over the block's nb keys, one
/// packed 32-column V block at a time ([ceil(nb/2)][32][2] per block).
inline void bf16_pv(const block_window_t &bw,
        const zendnnl::common::bfloat16_t *p,
        const zendnnl::common::bfloat16_t *v, int64_t dv, float *o) {
    const int64_t v_block = (bw.nb + 1) / 2 * 2 * kNR;
    for (int64_t c = 0; c < dv; c += kNR)
        ck::bf16::gemm_block(bw.rb - bw.ra, int(std::min<int64_t>(kNR, dv - c)),
                bw.phi - bw.plo, p + bw.ra * kBlockN + bw.plo, kBlockN,
                v + c / kNR * v_block + bw.plo * kNR, o + bw.ra * dv + c, dv);
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_TILED_COMMON_HPP
