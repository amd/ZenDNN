/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

template <int MR, int NV>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bf16,fma")
void bf16_brgemm_ukernel(const uint16_t *__restrict A, int lda,
        const uint16_t *__restrict B_vnni, int b_stride, float *__restrict C,
        int ldc, int K, int BK, float beta, const float *__restrict bias,
        fused_postop_t fused_op, uint16_t *__restrict C_bf16, int ldc_bf16) {

    __m512 acc[MR][NV];

    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_mul_ps(
                        bv, _mm512_loadu_ps(C + m * ldc + v * 16));
    } else {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    for (int pc = 0; pc < K; pc += BK) {
        const int kb_orig = std::min(BK, K - pc);
        const uint16_t *a_off = A + pc;
        const uint16_t *b_off = B_vnni + (pc / 2) * b_stride;

        const int k_full_pairs = kb_orig / 2;
        const bool has_odd_tail = (kb_orig & 1) != 0;

        int kk = 0;
        for (; kk + 1 < k_full_pairs; kk += 2) {
            for (int u = 0; u < 2; ++u) {
                const uint16_t *b_kp = b_off + (kk + u) * b_stride;
                __m512bh bv[NV];
                for (int v = 0; v < NV; ++v)
                    bv[v] = (__m512bh)_mm512_loadu_si512(
                            b_kp + v * 16 * VNNI_PAIR);
                for (int m = 0; m < MR; ++m) {
                    uint32_t a_pair;
                    std::memcpy(&a_pair, &a_off[m * lda + 2 * (kk + u)],
                            sizeof(a_pair));
                    __m512bh av = (__m512bh)_mm512_set1_epi32(
                            static_cast<int>(a_pair));
                    for (int v = 0; v < NV; ++v)
                        acc[m][v] = _mm512_dpbf16_ps(acc[m][v], av, bv[v]);
                }
            }
        }
        for (; kk < k_full_pairs; ++kk) {
            const uint16_t *b_kp = b_off + kk * b_stride;
            __m512bh bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = (__m512bh)_mm512_loadu_si512(b_kp + v * 16 * VNNI_PAIR);
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair, &a_off[m * lda + 2 * kk], sizeof(a_pair));
                __m512bh av
                        = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], av, bv[v]);
            }
        }
        if (has_odd_tail) {
            const uint16_t *b_kp = b_off + k_full_pairs * b_stride;
            __m512bh bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = (__m512bh)_mm512_loadu_si512(b_kp + v * 16 * VNNI_PAIR);
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair = static_cast<uint32_t>(
                        a_off[m * lda + 2 * k_full_pairs]);
                __m512bh av
                        = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], av, bv[v]);
            }
        }
    }

    for (int m = 0; m < MR; ++m) {
        for (int v = 0; v < NV; ++v) {
            __m512 val = acc[m][v];
            if (bias) val = _mm512_add_ps(val, _mm512_loadu_ps(bias + v * 16));
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);
            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(val);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(
                                            C_bf16 + m * ldc_bf16 + v * 16),
                        (__m256i)bf);
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, val);
            }
        }
    }
}

// ── Explicit instantiations ──────────────────────────────────────────────
// Same register pressure analysis as the GEMM ukernel (see
// bf16_gemm_ukernel.cpp); BRGEMM has one extra parameter (BK) but
// identical ZMM usage in the K-loop.
//
// Live ZMM count = MR×NV (accumulators) + NV (b vectors) + 1 (a
// broadcast).  AVX-512 has 32 ZMM registers.
//
//   NV=2 (NR=32): register-pressure clean through MR≤12 (27 ZMM); this
//                 file instantiates MR ∈ {1..12}.  select_bf16_brgemm_kernel
//                 selects MR ∈ {1..6}, select_bf16_brgemm_tall_kernel
//                 MR ∈ {1..12}.
//   NV=1 (NR=16): register-pressure clean for practical MR values;
//                 this file currently instantiates/selects MR ∈ {1..6}.
// clang-format off
#define INST(MR, NV) \
    template void bf16_brgemm_ukernel<MR, NV>(const uint16_t *, int, \
            const uint16_t *, int, float *, int, int, int, float, \
            const float *, fused_postop_t, uint16_t *, int);

// NR=32 (NV=2): MR ∈ {1..12}
INST(1, 2)
INST(2, 2) INST(3, 2) INST(4, 2) INST(5, 2) INST(6, 2)
INST(7, 2) INST(8, 2) INST(9, 2) INST(10, 2) INST(11, 2) INST(12, 2)
        // NR=16 (NV=1): MR ∈ {1..6}
        INST(1, 1) INST(2, 1) INST(3, 1) INST(4, 1) INST(5, 1) INST(6, 1)
#undef INST
// clang-format on

ZENDNNL_TARGET("avx512f,avx512bf16,fma")
bf16_brgemm_fn_t select_bf16_brgemm_kernel(int MR, int NR) {
    switch (NR) {
        case 32:
            switch (MR) {
                case 1: return bf16_brgemm_ukernel<1, 2>;
                case 2: return bf16_brgemm_ukernel<2, 2>;
                case 3: return bf16_brgemm_ukernel<3, 2>;
                case 4: return bf16_brgemm_ukernel<4, 2>;
                case 5: return bf16_brgemm_ukernel<5, 2>;
                case 6: return bf16_brgemm_ukernel<6, 2>;
            }
            break;
        case 16:
            switch (MR) {
                case 1: return bf16_brgemm_ukernel<1, 1>;
                case 2: return bf16_brgemm_ukernel<2, 1>;
                case 3: return bf16_brgemm_ukernel<3, 1>;
                case 4: return bf16_brgemm_ukernel<4, 1>;
                case 5: return bf16_brgemm_ukernel<5, 1>;
                case 6: return bf16_brgemm_ukernel<6, 1>;
            }
            break;
    }
    return nullptr;
}

ZENDNNL_TARGET("avx512f,avx512bf16,fma")
bf16_brgemm_fn_t select_bf16_brgemm_tall_kernel(int MR, int NR) {
    if (NR == 32) {
        switch (MR) {
            case 7: return bf16_brgemm_ukernel<7, 2>;
            case 8: return bf16_brgemm_ukernel<8, 2>;
            case 9: return bf16_brgemm_ukernel<9, 2>;
            case 10: return bf16_brgemm_ukernel<10, 2>;
            case 11: return bf16_brgemm_ukernel<11, 2>;
            case 12: return bf16_brgemm_ukernel<12, 2>;
        }
    }
    return select_bf16_brgemm_kernel(MR, NR);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
