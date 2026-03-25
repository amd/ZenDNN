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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/int8_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// INT8 BRGEMM microkernel: MR rows × NV×16 columns.
//
// Accumulates in i32 via vpdpbusd, then dequantizes to fp32:
//   result = (acc_i32 - zp * col_sum) * src_scale * wei_scale + bias
//
// B is in INT8 VNNI layout: groups of 4 consecutive K elements per column.
// b_stride = NR_PACK * INT8_VNNI_GRP = 64 * 4 = 256 bytes per k-quad row.
template<int MR, int NV>
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,avx512vnni,fma"), noinline))
void int8_brgemm_ukernel(
    const uint8_t *__restrict__ A, int lda,
    const int8_t  *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C_fp32, int ldc,
    int K, int BK,
    const int32_t *__restrict__ col_sum,
    int32_t src_zp, float src_scale,
    const float *__restrict__ wei_scale, int wei_scale_count,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    __m512i acc[MR][NV];
    for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v)
            acc[m][v] = _mm512_setzero_si512();

    // ── K-loop: vpdpbusd accumulation ──
    for (int pc = 0; pc < K; pc += BK) {
        const int kb = std::min(BK, K - pc);
        const int kb_padded = (kb + 3) & ~3;
        const int k_quads = kb_padded / 4;
        const uint8_t *a_off = A + pc;
        const int8_t  *b_off = B_vnni + (pc / 4) * b_stride;

        for (int kq = 0; kq < k_quads; ++kq) {
            const int8_t *b_kq = b_off + kq * b_stride;
            __m512i bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = _mm512_loadu_si512(
                    b_kq + v * 16 * INT8_VNNI_GRP);

            for (int m = 0; m < MR; ++m) {
                uint32_t a_quad;
                const int k_abs = 4 * kq;
                if (k_abs + 3 < kb)
                    std::memcpy(&a_quad, &a_off[m * lda + k_abs], 4);
                else {
                    a_quad = 0;
                    const int remain = kb - k_abs;
                    if (remain > 0)
                        std::memcpy(&a_quad, &a_off[m * lda + k_abs], remain);
                }
                __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad));
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_dpbusd_epi32(acc[m][v], av, bv[v]);
            }
        }
    }

    // ── Epilogue: dequantize i32 → fp32 + bias + activation + store ──
    const __m512 v_src_scale = _mm512_set1_ps(src_scale);
    const __m512 v_zp = _mm512_set1_ps(static_cast<float>(src_zp));
    const bool per_channel = (wei_scale_count > 1);

    for (int m = 0; m < MR; ++m) {
        for (int v = 0; v < NV; ++v) {
            __m512 vacc = _mm512_cvtepi32_ps(acc[m][v]);

            // Zero-point correction: acc - zp * col_sum
            __m512 vcs = _mm512_cvtepi32_ps(
                _mm512_loadu_si512(col_sum + v * 16));
            vacc = _mm512_fnmadd_ps(v_zp, vcs, vacc);

            // Dequantize: * src_scale * wei_scale
            __m512 ws = per_channel
                ? _mm512_loadu_ps(wei_scale + v * 16)
                : _mm512_set1_ps(wei_scale[0]);
            vacc = _mm512_mul_ps(vacc, _mm512_mul_ps(v_src_scale, ws));

            // Bias
            if (bias)
                vacc = _mm512_add_ps(vacc, _mm512_loadu_ps(bias + v * 16));

            // Fused activation
            if (fused_op != fused_postop_t::none)
                vacc = apply_fused_postop(vacc, fused_op);

            // Store
            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(vacc);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(
                        C_bf16 + m * ldc_bf16 + v * 16),
                    (__m256i)bf);
            } else {
                _mm512_storeu_ps(C_fp32 + m * ldc + v * 16, vacc);
            }
        }
    }
}

// ── Explicit instantiations ──────────────────────────────────────────
// INT8 vpdpbusd uses i32 accumulators (same register pressure as bf16 fp32 acc).
// MR×NV ZMM registers: MR*NV accumulators + NV B vectors + 1 A broadcast.
// NR=64 (NV=4): MR=1..6 safe (6*4+4+1=29 ZMMs < 32)
// NR=32 (NV=2): MR=1..6
// NR=16 (NV=1): MR=1..6
#define INST(MR, NV) \
    template void int8_brgemm_ukernel<MR,NV>( \
        const uint8_t*, int, const int8_t*, int, float*, int, \
        int, int, const int32_t*, int32_t, float, \
        const float*, int, const float*, fused_postop_t, uint16_t*, int);

INST(1,4) INST(2,4) INST(3,4) INST(4,4) INST(6,4)
INST(1,2) INST(2,2) INST(3,2) INST(4,2) INST(6,2)
INST(1,1) INST(2,1) INST(3,1) INST(4,1) INST(6,1)
#undef INST

__attribute__((target("avx512f,avx512vnni,fma")))
int8_brgemm_fn_t select_int8_brgemm_kernel(int MR, int NR) {
    switch (NR) {
    case 64:
        switch (MR) {
        case 1: return int8_brgemm_ukernel<1, 4>;
        case 2: return int8_brgemm_ukernel<2, 4>;
        case 3: return int8_brgemm_ukernel<3, 4>;
        case 4: return int8_brgemm_ukernel<4, 4>;
        case 6: return int8_brgemm_ukernel<6, 4>;
        }
        break;
    case 32:
        switch (MR) {
        case 1: return int8_brgemm_ukernel<1, 2>;
        case 2: return int8_brgemm_ukernel<2, 2>;
        case 3: return int8_brgemm_ukernel<3, 2>;
        case 4: return int8_brgemm_ukernel<4, 2>;
        case 6: return int8_brgemm_ukernel<6, 2>;
        }
        break;
    case 16:
        switch (MR) {
        case 1: return int8_brgemm_ukernel<1, 1>;
        case 2: return int8_brgemm_ukernel<2, 1>;
        case 3: return int8_brgemm_ukernel<3, 1>;
        case 4: return int8_brgemm_ukernel<4, 1>;
        case 6: return int8_brgemm_ukernel<6, 1>;
        }
        break;
    }
    return nullptr;
}

// ── Tail kernel (dynamic MR/NR for edge tiles) ──────────────────────
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,avx512vnni,fma")))
void int8_brgemm_tail_kernel(
    const uint8_t *__restrict__ A, int lda,
    const int8_t  *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C_fp32, int ldc,
    int K, int BK, int mr_act, int nr_act,
    const int32_t *__restrict__ col_sum,
    int32_t src_zp, float src_scale,
    const float *__restrict__ wei_scale, int wei_scale_count,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    constexpr int MAX_MR = 6;
    constexpr int MAX_NV = 4;
    __m512i acc[MAX_MR][MAX_NV];
    const int nv_act = (nr_act + 15) / 16;

    for (int m = 0; m < mr_act; ++m)
        for (int v = 0; v < nv_act; ++v)
            acc[m][v] = _mm512_setzero_si512();

    for (int pc = 0; pc < K; pc += BK) {
        const int kb = std::min(BK, K - pc);
        const int kb_padded = (kb + 3) & ~3;
        const int k_quads = kb_padded / 4;
        const uint8_t *a_off = A + pc;
        const int8_t  *b_off = B_vnni + (pc / 4) * b_stride;

        for (int kq = 0; kq < k_quads; ++kq) {
            const int8_t *b_kq = b_off + kq * b_stride;
            __m512i bv[MAX_NV];
            for (int v = 0; v < nv_act; ++v)
                bv[v] = _mm512_loadu_si512(b_kq + v * 16 * INT8_VNNI_GRP);
            for (int m = 0; m < mr_act; ++m) {
                uint32_t a_quad;
                const int k_abs = 4 * kq;
                if (k_abs + 3 < kb)
                    std::memcpy(&a_quad, &a_off[m * lda + k_abs], 4);
                else {
                    a_quad = 0;
                    const int remain = kb - k_abs;
                    if (remain > 0)
                        std::memcpy(&a_quad, &a_off[m * lda + k_abs], remain);
                }
                __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad));
                for (int v = 0; v < nv_act; ++v)
                    acc[m][v] = _mm512_dpbusd_epi32(acc[m][v], av, bv[v]);
            }
        }
    }

    // Epilogue with tail masking
    const __m512 v_src_scale = _mm512_set1_ps(src_scale);
    const __m512 v_zp = _mm512_set1_ps(static_cast<float>(src_zp));
    const bool per_channel = (wei_scale_count > 1);

    for (int m = 0; m < mr_act; ++m) {
        for (int v = 0; v < nv_act; ++v) {
            const int n_off = v * 16;
            const int elems = std::min(16, nr_act - n_off);
            if (elems <= 0) break;
            __mmask16 mask = (elems == 16)
                ? __mmask16(0xFFFF)
                : static_cast<__mmask16>((1u << elems) - 1);

            __m512 vacc = _mm512_cvtepi32_ps(acc[m][v]);
            __m512 vcs = _mm512_cvtepi32_ps(
                _mm512_maskz_loadu_epi32(mask, col_sum + n_off));
            vacc = _mm512_fnmadd_ps(v_zp, vcs, vacc);

            __m512 ws = per_channel
                ? _mm512_maskz_loadu_ps(mask, wei_scale + n_off)
                : _mm512_set1_ps(wei_scale[0]);
            vacc = _mm512_mul_ps(vacc, _mm512_mul_ps(v_src_scale, ws));

            if (bias)
                vacc = _mm512_add_ps(vacc,
                    _mm512_maskz_loadu_ps(mask, bias + n_off));

            if (fused_op != fused_postop_t::none)
                vacc = apply_fused_postop(vacc, fused_op);

            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(vacc);
                _mm256_mask_storeu_epi16(
                    C_bf16 + m * ldc_bf16 + n_off, mask, (__m256i)bf);
            } else {
                _mm512_mask_storeu_ps(
                    C_fp32 + m * ldc + n_off, mask, vacc);
            }
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
