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
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_ai/brgemm/intrinsic/bf16/avx512_bf16_brgemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/intrinsic/bf16/avx512_bf16_gemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/brgemm/brgemm_planner.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/postop.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"
#include "common/bfloat16.hpp"

#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

using namespace zendnnl::error_handling;
using zendnnl::ops::matmul_config_t;
using zendnnl::ops::post_op_type_t;

// ============================================================================
// BF16 BRGEMM AVX-512 microkernel: MR x (NV*16) with dpbf16ps
//
// Processes FULL K in one call. FP32 accumulators stay live in ZMM registers
// across ALL K-blocks — no C store/reload between K-blocks.
//
// B is in VNNI format: k-pairs interleaved, NR_PACK=64 wide panels.
// A is BF16 row-major, accessed via vpbroadcastd (pair of BF16 elements).
//
// BK: internal K-block size for cache tiling. The microkernel iterates
// K in BK-sized chunks, advancing A and B pointers per chunk.
// ============================================================================
template<int MR, int NV>
__attribute__((target("avx512f,avx512bf16,fma")))
__attribute__((noinline))
static void bf16_brgemm_ukernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int K, int BK, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

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

    // Batch-reduce: iterate ALL K with accumulators live in registers.
    // B is VNNI-packed with K padded to even; A is raw row-major (NOT padded).
    // We iterate using the original K to avoid out-of-bounds A reads.
    for (int pc = 0; pc < K; pc += BK) {
        const int kb_orig = std::min(BK, K - pc);
        const uint16_t *a_off = A + pc;
        const uint16_t *b_off = B_vnni + (pc / 2) * b_stride;

        const int k_full_pairs = kb_orig / 2;
        const bool has_odd_tail = (kb_orig & 1) != 0;

        // 2x k-pair unrolled main loop
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
                    __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                        static_cast<int>(a_pair));
                    for (int v = 0; v < NV; ++v)
                        acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, bv[v]);
                }
            }
        }
        // Remainder full k-pairs
        for (; kk < k_full_pairs; ++kk) {
            const uint16_t *b_kp = b_off + kk * b_stride;
            __m512bh bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = (__m512bh)_mm512_loadu_si512(
                    b_kp + v * 16 * VNNI_PAIR);
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair, &a_off[m * lda + 2 * kk],
                            sizeof(a_pair));
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, bv[v]);
            }
        }
        // Odd-K tail: load single BF16 from A, zero upper 16 bits.
        // B is already zero-padded for the second element of this pair.
        if (has_odd_tail) {
            const uint16_t *b_kp = b_off + k_full_pairs * b_stride;
            __m512bh bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = (__m512bh)_mm512_loadu_si512(
                    b_kp + v * 16 * VNNI_PAIR);
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair = static_cast<uint32_t>(
                    a_off[m * lda + 2 * k_full_pairs]);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, bv[v]);
            }
        }
    }

    // Single epilogue after ALL K-blocks: bias → activation → store
    for (int m = 0; m < MR; ++m) {
        for (int v = 0; v < NV; ++v) {
            __m512 val = acc[m][v];
            if (bias)
                val = _mm512_add_ps(val, _mm512_loadu_ps(bias + v * 16));
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);
            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(val);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(C_bf16 + m * ldc_bf16 + v * 16),
                    (__m256i)bf);
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, val);
            }
        }
    }
}

// Instantiations: MR=6 with NV=4 (NR=64), NV=2 (NR=32), NV=1 (NR=16)
template void bf16_brgemm_ukernel<6,4>(const uint16_t*, int, const uint16_t*,
    int, float*, int, int, int, float, const float*, fused_postop_t,
    uint16_t*, int);
template void bf16_brgemm_ukernel<6,2>(const uint16_t*, int, const uint16_t*,
    int, float*, int, int, int, float, const float*, fused_postop_t,
    uint16_t*, int);
template void bf16_brgemm_ukernel<6,1>(const uint16_t*, int, const uint16_t*,
    int, float*, int, int, int, float, const float*, fused_postop_t,
    uint16_t*, int);
// Decode MR=1..4 with NV=4 for GEMV-like shapes
template void bf16_brgemm_ukernel<1,4>(const uint16_t*, int, const uint16_t*,
    int, float*, int, int, int, float, const float*, fused_postop_t,
    uint16_t*, int);
template void bf16_brgemm_ukernel<2,4>(const uint16_t*, int, const uint16_t*,
    int, float*, int, int, int, float, const float*, fused_postop_t,
    uint16_t*, int);
template void bf16_brgemm_ukernel<3,4>(const uint16_t*, int, const uint16_t*,
    int, float*, int, int, int, float, const float*, fused_postop_t,
    uint16_t*, int);
template void bf16_brgemm_ukernel<4,4>(const uint16_t*, int, const uint16_t*,
    int, float*, int, int, int, float, const float*, fused_postop_t,
    uint16_t*, int);

using bf16_brgemm_fn_t = void (*)(const uint16_t*, int, const uint16_t*, int,
                                   float*, int, int, int, float, const float*,
                                   fused_postop_t, uint16_t*, int);

static bf16_brgemm_fn_t select_bf16_brgemm_kernel(int MR, int NR) {
    if (NR == 64) {
        switch (MR) {
        case 1: return bf16_brgemm_ukernel<1, 4>;
        case 2: return bf16_brgemm_ukernel<2, 4>;
        case 3: return bf16_brgemm_ukernel<3, 4>;
        case 4: return bf16_brgemm_ukernel<4, 4>;
        case 6: return bf16_brgemm_ukernel<6, 4>;
        }
    } else if (NR == 32) {
        if (MR == 6) return bf16_brgemm_ukernel<6, 2>;
    } else if (NR == 16) {
        if (MR == 6) return bf16_brgemm_ukernel<6, 1>;
    }
    return bf16_brgemm_ukernel<6, 1>;
}

// ============================================================================
// BF16 BRGEMM tail kernel (dynamic MR/NR for edge tiles)
// Same pattern as FP32 BRGEMM tail but with dpbf16ps
// ============================================================================
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
static void bf16_brgemm_tail_kernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int K, int BK, int mr_act, int nr_act, float beta,
    const float *__restrict__ bias, fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    const int nv_full = nr_act / 16;
    const int nr_tail = nr_act % 16;
    const int nv = (nr_act + 15) / 16;
    const __mmask16 tail_mask = nr_tail
        ? static_cast<__mmask16>((1u << nr_tail) - 1) : 0;

    __m512 acc[12][4];
    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < mr_act; ++m) {
            for (int v = 0; v < nv_full; ++v)
                acc[m][v] = _mm512_mul_ps(
                    bv, _mm512_loadu_ps(C + m * ldc + v * 16));
            if (nr_tail)
                acc[m][nv_full] = _mm512_mul_ps(
                    bv, _mm512_maskz_loadu_ps(tail_mask,
                        C + m * ldc + nv_full * 16));
        }
    } else {
        for (int m = 0; m < mr_act; ++m)
            for (int v = 0; v < nv; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    for (int pc = 0; pc < K; pc += BK) {
        const int kb_orig = std::min(BK, K - pc);
        const uint16_t *a_off = A + pc;
        const uint16_t *b_off = B_vnni + (pc / 2) * b_stride;
        const int k_full_pairs = kb_orig / 2;
        const bool has_odd_tail = (kb_orig & 1) != 0;

        for (int kk = 0; kk < k_full_pairs; ++kk) {
            for (int m = 0; m < mr_act; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair, &a_off[m * lda + 2 * kk],
                            sizeof(a_pair));
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < nv; ++v) {
                    __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                        b_off + kk * b_stride + v * 16 * VNNI_PAIR);
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, b_bf16);
                }
            }
        }
        if (has_odd_tail) {
            for (int m = 0; m < mr_act; ++m) {
                uint32_t a_pair = static_cast<uint32_t>(
                    a_off[m * lda + 2 * k_full_pairs]);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < nv; ++v) {
                    __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                        b_off + k_full_pairs * b_stride + v * 16 * VNNI_PAIR);
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, b_bf16);
                }
            }
        }
    }

    // Epilogue
    for (int m = 0; m < mr_act; ++m) {
        for (int v = 0; v < nv_full; ++v) {
            __m512 val = acc[m][v];
            if (bias) val = _mm512_add_ps(val, _mm512_loadu_ps(bias + v * 16));
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);
            if (C_bf16) {
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(C_bf16 + m * ldc_bf16 + v * 16),
                    (__m256i)_mm512_cvtneps_pbh(val));
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, val);
            }
        }
        if (nr_tail) {
            __m512 val = acc[m][nv_full];
            if (bias) val = _mm512_add_ps(val,
                _mm512_maskz_loadu_ps(tail_mask, bias + nv_full * 16));
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);
            if (C_bf16) {
                _mm256_mask_storeu_epi16(
                    C_bf16 + m * ldc_bf16 + nv_full * 16, tail_mask,
                    (__m256i)_mm512_cvtneps_pbh(val));
            } else {
                _mm512_mask_storeu_ps(C + m * ldc + nv_full * 16, tail_mask, val);
            }
        }
    }
}

// ============================================================================
// FP32-to-BF16 output conversion for a tile
// ============================================================================
__attribute__((target("avx512f,avx512bf16")))
static void convert_fp32_to_bf16_tile(
    const float *src_fp32, int ldc_fp32,
    uint16_t *dst_bf16, int ldc_bf16,
    int rows, int cols) {

    for (int m = 0; m < rows; ++m) {
        int n = 0;
        for (; n + 16 <= cols; n += 16) {
            __m512 v = _mm512_loadu_ps(src_fp32 + m * ldc_fp32 + n);
            __m256bh bf = _mm512_cvtneps_pbh(v);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i *>(dst_bf16 + m * ldc_bf16 + n),
                (__m256i)bf);
        }
        for (; n < cols; ++n) {
            // Round-to-nearest-even to match vectorized _mm512_cvtneps_pbh
            float val = src_fp32[m * ldc_fp32 + n];
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            uint32_t lsb = (bits >> 16) & 1;
            uint32_t rounding_bias = 0x7FFF + lsb;
            bits += rounding_bias;
            dst_bf16[m * ldc_bf16 + n] = static_cast<uint16_t>(bits >> 16);
        }
    }
}

// ============================================================================
// Scale tile (alpha != 1)
// ============================================================================
__attribute__((target("avx512f")))
static void scale_tile(float *C, int ldc, int m_count, int n_count, float alpha) {
    __m512 av = _mm512_set1_ps(alpha);
    for (int m = 0; m < m_count; ++m) {
        float *row = C + m * ldc;
        int n = 0;
        for (; n + 15 < n_count; n += 16)
            _mm512_storeu_ps(row + n, _mm512_mul_ps(_mm512_loadu_ps(row + n), av));
        for (; n < n_count; ++n)
            row[n] *= alpha;
    }
}

// ============================================================================
// On-the-fly VNNI pack: pack NR columns of B for K_padded rows into VNNI
// format. Each thread calls this for the NR-wide strip it needs.
// Buffer size: (K_padded/2) * NR * VNNI_PAIR uint16_t values.
// ============================================================================
__attribute__((target("avx512f,avx512bw")))
static void pack_b_vnni_strip(
    const uint16_t *B, int ldb, bool transB,
    int col_start, int nr_act, int K, int K_padded,
    uint16_t *packed) {

    const int k_pairs = K_padded / 2;
    const int out_stride = NR_PACK * VNNI_PAIR;

    for (int kp = 0; kp < k_pairs; ++kp) {
        uint16_t *d = packed + kp * out_stride;
        const int k0 = kp * 2;
        const int k1 = k0 + 1;

        if (!transB && nr_act == NR_PACK) {
            // AVX-512 fast path for full NR_PACK=64 strip
            const uint16_t *row0 = (k0 < K) ? B + k0 * ldb + col_start
                                             : nullptr;
            const uint16_t *row1 = (k1 < K) ? B + k1 * ldb + col_start
                                             : nullptr;
            for (int n = 0; n < NR_PACK; n += 32) {
                __m512i r0 = row0 ? _mm512_loadu_si512(row0 + n)
                                  : _mm512_setzero_si512();
                __m512i r1 = row1 ? _mm512_loadu_si512(row1 + n)
                                  : _mm512_setzero_si512();
                __m512i lo = _mm512_unpacklo_epi16(r0, r1);
                __m512i hi = _mm512_unpackhi_epi16(r0, r1);
                const __m512i idx_lo = _mm512_setr_epi32(
                    0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
                const __m512i idx_hi = _mm512_setr_epi32(
                    8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);
                _mm512_storeu_si512(d + n * VNNI_PAIR,
                    _mm512_permutex2var_epi32(lo, idx_lo, hi));
                _mm512_storeu_si512(d + (n + 16) * VNNI_PAIR,
                    _mm512_permutex2var_epi32(lo, idx_hi, hi));
            }
        } else {
            // Scalar path for partial strips or transposed B
            for (int n = 0; n < nr_act; ++n) {
                uint16_t v0, v1;
                if (!transB) {
                    v0 = (k0 < K) ? B[k0 * ldb + (col_start + n)] : 0;
                    v1 = (k1 < K) ? B[k1 * ldb + (col_start + n)] : 0;
                } else {
                    v0 = (k0 < K) ? B[(col_start + n) * ldb + k0] : 0;
                    v1 = (k1 < K) ? B[(col_start + n) * ldb + k1] : 0;
                }
                d[n * VNNI_PAIR + 0] = v0;
                d[n * VNNI_PAIR + 1] = v1;
            }
            for (int n = nr_act; n < NR_PACK; ++n) {
                d[n * VNNI_PAIR + 0] = 0;
                d[n * VNNI_PAIR + 1] = 0;
            }
        }
    }
}

// ============================================================================
// BF16 BRGEMM thread loop
//
// NO outer K-loop — the microkernel processes full K internally.
// Thread parallelism over MC × NC tiles only.
// When prepacked_b is null, does on-the-fly VNNI packing per NR-strip
// per thread (each thread packs only the columns it needs).
// ============================================================================
static void bf16_brgemm_thread_loop(
    const GemmDescriptor &desc,
    const BrgemmPlan &plan,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params,
    const BF16PrepackedWeight *prepacked_b,
    bool do_otf) {

    const uint16_t *A = static_cast<const uint16_t *>(src);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);

    const int M = desc.M, N = desc.N, K = desc.K;
    const int lda = desc.lda, ldc = desc.ldc;
    const float alpha = desc.alpha;
    const float beta = (desc.alpha != 1.0f && desc.beta != 0.0f)
                       ? (desc.beta / desc.alpha) : desc.beta;
    const int MB = plan.MB, NB = plan.NB, BK = plan.BK;
    const int MR = plan.MR, NR = plan.NR;
    const int num_threads = plan.num_threads;
    const bool has_bias = (bias != nullptr);
    const int vnni_stride = NR_PACK * VNNI_PAIR;

    // Bias conversion (BF16 → FP32 if needed)
    const float *bias_f = nullptr;
    static thread_local float *s_bias_fp32 = nullptr;
    static thread_local size_t s_bias_cap = 0;
    if (has_bias) {
        if (bias_is_bf16) {
            if (s_bias_cap < static_cast<size_t>(N)) {
                std::free(s_bias_fp32);
                s_bias_fp32 = static_cast<float *>(std::aligned_alloc(
                    64, ((N * sizeof(float) + 63) & ~size_t(63))));
                s_bias_cap = s_bias_fp32 ? N : 0;
            }
            if (s_bias_fp32) {
                const uint16_t *bb = static_cast<const uint16_t *>(bias);
                for (int n = 0; n < N; ++n) {
                    uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
                    std::memcpy(&s_bias_fp32[n], &bits, sizeof(float));
                }
                bias_f = s_bias_fp32;
            }
        } else {
            bias_f = static_cast<const float *>(bias);
        }
    }

    // Postop detection
    fused_postop_t fused_op = fused_postop_t::none;
    int fused_idx = -1;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        auto pt = params.postop_[i].po_type;
        if (pt == post_op_type_t::relu && params.postop_[i].alpha == 0.0f) {
            fused_op = fused_postop_t::relu; fused_idx = i; break;
        } else if (pt == post_op_type_t::gelu_tanh) {
            fused_op = fused_postop_t::gelu_tanh; fused_idx = i; break;
        } else if (pt == post_op_type_t::gelu_erf) {
            fused_op = fused_postop_t::gelu_erf; fused_idx = i; break;
        } else if (pt == post_op_type_t::sigmoid) {
            fused_op = fused_postop_t::sigmoid; fused_idx = i; break;
        } else if (pt == post_op_type_t::tanh) {
            fused_op = fused_postop_t::tanh_op; fused_idx = i; break;
        }
    }

    std::vector<matmul_post_op> remaining_postops;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i)
        if (i != fused_idx) remaining_postops.push_back(params.postop_[i]);
    const bool has_remaining_postops = !remaining_postops.empty();

    const bool can_fuse = (alpha == 1.0f);
    const bool can_direct_bf16 = dst_is_bf16 && can_fuse && !has_remaining_postops;

    // FP32 C buffer for BF16 output
    float *C_fp32;
    int ldc_fp32;
    bool need_fp32_buf = dst_is_bf16;
    static thread_local float *s_c_buf = nullptr;
    static thread_local size_t s_c_cap = 0;

    if (need_fp32_buf && !can_direct_bf16) {
        size_t needed = static_cast<size_t>(M) * N;
        if (s_c_cap < needed) {
            std::free(s_c_buf);
            s_c_buf = static_cast<float *>(std::aligned_alloc(
                64, ((needed * sizeof(float) + 63) & ~size_t(63))));
            s_c_cap = s_c_buf ? needed : 0;
        }
        if (!s_c_buf) return;
        C_fp32 = s_c_buf;
        ldc_fp32 = N;
        if (beta != 0.0f) {
            const uint16_t *C_bf16 = static_cast<const uint16_t *>(dst);
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    uint32_t bits = static_cast<uint32_t>(
                        C_bf16[m * ldc + n]) << 16;
                    float val;
                    std::memcpy(&val, &bits, sizeof(val));
                    C_fp32[m * ldc_fp32 + n] = val;
                }
        }
    } else if (!need_fp32_buf) {
        C_fp32 = static_cast<float *>(dst);
        ldc_fp32 = ldc;
    } else {
        C_fp32 = nullptr;
        ldc_fp32 = 0;
    }

    uint16_t *C_bf16_dst = dst_is_bf16
        ? static_cast<uint16_t *>(dst) : nullptr;

    bf16_brgemm_fn_t hot_kernel = select_bf16_brgemm_kernel(MR, NR);

    const bool is_decode = (M <= 4);
    const int actual_MR = is_decode ? M : MR;

    const int ic_tiles = (M + MB - 1) / MB;
    const int jc_tiles = (N + NB - 1) / NB;
    const int total_tiles = ic_tiles * jc_tiles;
    const int active_threads = std::min(num_threads, total_tiles);

    // Raw weight pointer and ldb for on-the-fly packing
    const uint16_t *B_raw = static_cast<const uint16_t *>(weight);
    const int ldb = desc.ldb;
    const bool transB = desc.transB;
    const int K_padded = (K + 1) & ~1;

    auto process_tile = [&](int ic, int jc, uint16_t *otf_buf) {
        const int mb_act = std::min(MB, M - ic);
        const int nb_act = std::min(NB, N - jc);
        const int m_panels = (mb_act + actual_MR - 1) / actual_MR;

        for (int jr = 0; jr < nb_act; jr += NR) {
            const int nr_act = std::min(NR, nb_act - jr);
            const int col = jc + jr;
            const bool full_nr = (nr_act == NR);

            const uint16_t *pb;
            int pb_stride;
            if (prepacked_b) {
                int panel_idx = col / NR_PACK;
                int in_panel_off = col % NR_PACK;
                pb = prepacked_b->get_panel(0, panel_idx)
                     + in_panel_off * VNNI_PAIR;
                pb_stride = vnni_stride;
            } else if (otf_buf) {
                // On-the-fly pack: pack this NR-wide strip into otf_buf
                pack_b_vnni_strip(B_raw, ldb, transB,
                                  col, std::min(NR_PACK, N - col),
                                  K, K_padded, otf_buf);
                pb = otf_buf;
                pb_stride = vnni_stride;
            } else {
                return;
            }

            const float *tile_bias =
                (has_bias && can_fuse) ? (bias_f + col) : nullptr;
            const fused_postop_t tile_fop =
                can_fuse ? fused_op : fused_postop_t::none;

            for (int ip = 0; ip < m_panels; ++ip) {
                const int ir = ip * actual_MR;
                const int mr_act = std::min(actual_MR, mb_act - ir);
                const uint16_t *At = A + (ic + ir) * lda;
                float *Ct = (C_fp32 ? C_fp32 + (ic + ir) * ldc_fp32 + col
                                    : nullptr);

                uint16_t *tile_bf16 = nullptr;
                int tile_ldc_bf16 = 0;
                if (can_direct_bf16) {
                    tile_bf16 = C_bf16_dst + (ic + ir) * ldc + col;
                    tile_ldc_bf16 = ldc;
                }

                if (hot_kernel && full_nr && mr_act == actual_MR) {
                    hot_kernel(At, lda, pb, pb_stride,
                               Ct, ldc_fp32, K, BK, beta,
                               tile_bias, tile_fop,
                               tile_bf16, tile_ldc_bf16);
                } else if (full_nr && mr_act >= 1 && mr_act <= 4) {
                    auto tail_uk = select_bf16_brgemm_kernel(mr_act, NR);
                    if (tail_uk) {
                        tail_uk(At, lda, pb, pb_stride,
                                Ct, ldc_fp32, K, BK, beta,
                                tile_bias, tile_fop,
                                tile_bf16, tile_ldc_bf16);
                    } else {
                        bf16_brgemm_tail_kernel(At, lda, pb, pb_stride,
                                                Ct, ldc_fp32, K, BK,
                                                mr_act, nr_act, beta,
                                                tile_bias, tile_fop,
                                                tile_bf16, tile_ldc_bf16);
                    }
                } else {
                    bf16_brgemm_tail_kernel(At, lda, pb, pb_stride,
                                            Ct, ldc_fp32, K, BK,
                                            mr_act, nr_act, beta,
                                            tile_bias, tile_fop,
                                            tile_bf16, tile_ldc_bf16);
                }
            }
        }

        // Post-tile epilogue
        if (C_fp32) {
            float *Ctile = C_fp32 + ic * ldc_fp32 + jc;
            const int nb_act_e = std::min(NB, N - jc);
            const int mb_act_e = std::min(MB, M - ic);
            if (alpha != 1.0f) {
                scale_tile(Ctile, ldc_fp32, mb_act_e, nb_act_e, alpha);
                if (has_bias)
                    apply_postops_tile(Ctile, ldc_fp32, mb_act_e, nb_act_e,
                                       jc, ic, bias_f, {});
                if (fused_op != fused_postop_t::none && fused_idx >= 0)
                    apply_postops_tile(Ctile, ldc_fp32, mb_act_e, nb_act_e,
                                       jc, ic, nullptr,
                                       {params.postop_[fused_idx]});
            }
            if (has_remaining_postops) {
                apply_postops_tile(Ctile, ldc_fp32, mb_act_e, nb_act_e,
                                   jc, ic, nullptr, remaining_postops);
            }
        }
    };

    // On-the-fly pack buffer size: one NR_PACK-wide strip × K_padded/2 k-pairs
    const size_t otf_buf_size = (prepacked_b || !do_otf) ? 0
        : static_cast<size_t>((K_padded / 2)) * NR_PACK * VNNI_PAIR;

    if (num_threads <= 1) {
        // Single-thread: one reusable otf buffer
        static thread_local uint16_t *s_otf = nullptr;
        static thread_local size_t s_otf_cap = 0;
        uint16_t *otf = nullptr;
        if (do_otf && !prepacked_b && otf_buf_size > 0) {
            if (s_otf_cap < otf_buf_size) {
                std::free(s_otf);
                s_otf = static_cast<uint16_t *>(std::aligned_alloc(
                    64, ((otf_buf_size * sizeof(uint16_t) + 63)
                         & ~size_t(63))));
                s_otf_cap = s_otf ? otf_buf_size : 0;
            }
            otf = s_otf;
        }
        for (int t = 0; t < total_tiles; ++t) {
            int ic = (t / jc_tiles) * MB;
            int jc = (t % jc_tiles) * NB;
            process_tile(ic, jc, otf);
        }
    } else {
        #pragma omp parallel num_threads(active_threads)
        {
            // Per-thread otf buffer (thread_local, reused across calls)
            static thread_local uint16_t *tl_otf = nullptr;
            static thread_local size_t tl_otf_cap = 0;
            uint16_t *otf = nullptr;
            if (do_otf && !prepacked_b && otf_buf_size > 0) {
                if (tl_otf_cap < otf_buf_size) {
                    std::free(tl_otf);
                    tl_otf = static_cast<uint16_t *>(std::aligned_alloc(
                        64, ((otf_buf_size * sizeof(uint16_t) + 63)
                             & ~size_t(63))));
                    tl_otf_cap = tl_otf ? otf_buf_size : 0;
                }
                otf = tl_otf;
            }

            #pragma omp for schedule(static)
            for (int t = 0; t < total_tiles; ++t) {
                int ic = (t / jc_tiles) * MB;
                int jc = (t % jc_tiles) * NB;
                process_tile(ic, jc, otf);
            }
        }
    }

    // FP32 → BF16 output conversion (vectorized, attributed helper)
    if (need_fp32_buf && !can_direct_bf16 && C_fp32) {
        uint16_t *out = static_cast<uint16_t *>(dst);
        convert_fp32_to_bf16_tile(C_fp32, ldc_fp32, out, ldc, M, N);
    }
}

// ============================================================================
// BF16 BRGEMM execute: plan + VNNI B prepack + thread loop
// ============================================================================

void bf16_brgemm_execute(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const bool transB = desc.transB;
    const bool is_weights_const = desc.is_weights_const;
    const int K_padded = (K + 1) & ~1;
    const bool is_decode = (M <= 4);

    // BRGEMM dispatch heuristic: route to GEMM when BRGEMM can't win.
    //
    // BRGEMM's advantage = keeping FP32 accumulators in registers across
    // K-blocks, saving C store/reload at each K-block boundary. This helps
    // when C traffic is a meaningful fraction of total traffic.
    //
    // C traffic per K-block boundary = MR × NR × 8 bytes (store+load)
    // B traffic per tile = K × NR_PACK × 2 bytes
    // Ratio ≈ 8·MR·NR / (BK × NR_PACK × 2)
    //
    // For BF16 with BK ≈ 3000–4000, ratio < 0.3% — negligible.
    // BRGEMM only wins when B panel fits in L2 (reused across K-blocks)
    // AND there are multiple K-blocks. Otherwise GEMM's lighter loop wins.
    //
    // Route to GEMM when:
    //  (a) Decode with K < N (BRGEMM overhead not worth it)
    //  (b) B panel for one NR strip exceeds L2 (K > L2/2 / (NR_PACK × 2))
    {
        int b_panel_limit = (uarch.l2_bytes / 2)
                            / (NR_PACK * static_cast<int>(sizeof(uint16_t)));
        bool b_exceeds_l2 = (K > b_panel_limit);
        bool decode_wide_n = (is_decode && K < N);

        if (decode_wide_n || b_exceeds_l2) {
            bf16_gemm_execute(desc, uarch, src, weight, dst, bias, params);
            return;
        }
    }

    // ── 1. Plan (BF16-specific, mirroring GEMM's proven approach) ──
    BrgemmPlan bplan = plan_brgemm(desc, uarch);
    bplan.NR = 64;

    // Adaptive MR: same logic as GEMM — prefer MR=6 for throughput,
    // but use MR=4 when MR=6 creates badly unbalanced IC tiles.
    if (is_decode) {
        bplan.MR = M;
    } else if (M % 6 == 0 || M >= 18) {
        bplan.MR = 6;
    } else if (M % 4 == 0) {
        bplan.MR = 4;
    } else if (M % 6 <= 3 && M > 12) {
        bplan.MR = 4;
    } else {
        bplan.MR = 6;
    }

    // BK: maximize to keep accumulators live longer (BRGEMM's key advantage).
    // Constraints: A panel (MR×BK×2) in L1, B panel (NR_PACK×BK×2) in L2.
    {
        int kb_a = static_cast<int>(0.8 * uarch.l1d_bytes)
                   / std::max(bplan.MR * 2, 1);
        int kb_b = (uarch.l2_bytes / 2)
                   / (NR_PACK * static_cast<int>(sizeof(uint16_t)));
        int bk_max = std::min(kb_a, kb_b);
        bk_max = std::max(bk_max, 64);
        bk_max = (bk_max + 1) & ~1;
        if (K_padded <= bk_max) {
            bplan.BK = K_padded;
        } else {
            int n_blk = (K_padded + bk_max - 1) / bk_max;
            bplan.BK = ((K_padded + n_blk - 1) / n_blk + 1) & ~1;
        }
    }

    // NB: re-align to NR=64. Keep FP32 planner's NB (usually >= 128)
    // as a floor — larger NB reduces per-tile overhead.
    bplan.NB = std::max((bplan.NB / bplan.NR) * bplan.NR, bplan.NR);
    bplan.NB = std::min(bplan.NB, N);

    // MB: C tile (MB×NB×4) + A tile (MB×BK×2) fits in L2.
    if (!is_decode) {
        int mb_budget = (uarch.l1d_bytes + uarch.l2_bytes)
                        / std::max(bplan.NB * 4 + bplan.BK * 2, 1);
        mb_budget = (mb_budget / bplan.MR) * bplan.MR;
        mb_budget = std::max(mb_budget, bplan.MR);
        bplan.MB = std::min(mb_budget, M);
    } else {
        bplan.MB = M;
    }

    // Load balance: ensure enough tiles for all threads
    if (bplan.num_threads > 1) {
        int jc_tiles = (N + bplan.NB - 1) / bplan.NB;
        int ic_tiles = (M + bplan.MB - 1) / bplan.MB;

        int needed_ic = (bplan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            int m_panels = (M + bplan.MR - 1) / bplan.MR;
            int ppb = std::max(m_panels / needed_ic, 1);
            bplan.MB = ppb * bplan.MR;
            bplan.MB = std::min(bplan.MB, M);
            ic_tiles = (M + bplan.MB - 1) / bplan.MB;
        }

        // Also shrink NB if still not enough tiles
        if (ic_tiles * jc_tiles < bplan.num_threads && bplan.NB > bplan.NR) {
            int needed_jc = (bplan.num_threads + ic_tiles - 1) / ic_tiles;
            int nb_target = (N + needed_jc - 1) / needed_jc;
            nb_target = std::max((nb_target / bplan.NR) * bplan.NR, bplan.NR);
            bplan.NB = std::min(nb_target, bplan.NB);
        }
    }

    if (apilog_info_enabled()) {
        apilog_info("AI BF16 BRGEMM plan: M=", M, " N=", N, " K=", K,
                    " MB=", bplan.MB, " NB=", bplan.NB, " BK=", bplan.BK,
                    " MR=", bplan.MR, " NR=", bplan.NR,
                    " threads=", bplan.num_threads);
    }

    // ── 2. B VNNI prepack ──
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();
    static int32_t s_otf_bpack =
        matmul_config_t::instance().get_otf_bpack();
    const BF16PrepackedWeight *prepacked_b = nullptr;
    const bool can_cache = is_weights_const && (s_weight_cache != 0);

    if (can_cache) {
        PrepackedWeightKey bk{weight, K, N, desc.ldb, transB};
        prepacked_b = BF16PrepackedWeightCache::instance().get_or_prepack(
            bk, static_cast<const uint16_t *>(weight));
    }
    const bool do_otf = (!prepacked_b && s_otf_bpack != 0);

    if (!prepacked_b && !do_otf) {
        bf16_gemm_execute(desc, uarch, src, weight, dst, bias, params);
        return;
    }

    // ── 3. Thread loop ──
    bf16_brgemm_thread_loop(desc, bplan, uarch, src, weight, dst, bias,
                            params, prepacked_b, do_otf);
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
