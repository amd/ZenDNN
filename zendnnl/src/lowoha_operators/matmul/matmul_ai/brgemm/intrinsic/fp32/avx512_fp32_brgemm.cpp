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

#include "lowoha_operators/matmul/matmul_ai/brgemm/intrinsic/fp32/avx512_fp32_brgemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/brgemm/brgemm_planner.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/intrinsic/fp32/avx512_fp32_gemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/postop.hpp"
#include "operators/matmul/matmul_config.hpp"
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

using zendnnl::ops::matmul_config_t;
using zendnnl::ops::post_op_type_t;

// ============================================================================
// BRGEMM AVX-512 FP32 microkernel: MR x (NV*16)
//
// Processes FULL K in one call. Accumulators stay live across K-blocks.
// B pointer is pre-resolved by the caller — microkernel just uses
// pb + kk * pb_stride (same interface as GEMM microkernel per K-step).
//
// The outer BK loop is inside this function. For each BK block, the
// microkernel advances A by BK columns and B by BK rows.
// ============================================================================
template<int MR, int NV>
__attribute__((target("avx512f,fma"), noinline))
static void brgemm_ukernel(
    const float * __restrict__ A, int lda,
    const float * __restrict__ pb, int pb_stride,
    float * __restrict__ C, int ldc,
    int K, int BK, float beta,
    const float * __restrict__ bias, fused_postop_t fused_op) {

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

    // Batch-reduce: iterate ALL K, keeping accumulators live
    for (int pc = 0; pc < K; pc += BK) {
        const int kb = std::min(BK, K - pc);
        const float *a_off = A + pc;        // A offset for this K-block
        const float *b_off = pb + pc * pb_stride; // B offset for this K-block

        // K-loop: 4x unrolled
        int kk = 0;
        for (; kk + 3 < kb; kk += 4) {
            for (int u = 0; u < 4; ++u) {
                const float *bp = b_off + (kk + u) * pb_stride;
                __m512 bv[NV];
                for (int v = 0; v < NV; ++v)
                    bv[v] = _mm512_loadu_ps(bp + v * 16);
                for (int m = 0; m < MR; ++m) {
                    __m512 a = _mm512_set1_ps(a_off[m * lda + kk + u]);
                    for (int v = 0; v < NV; ++v)
                        acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
                }
            }
        }
        for (; kk < kb; ++kk) {
            __m512 bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = _mm512_loadu_ps(b_off + kk * pb_stride + v * 16);
            for (int m = 0; m < MR; ++m) {
                __m512 a = _mm512_set1_ps(a_off[m * lda + kk]);
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
            }
        }
    }

    // Epilogue (ONCE after all K): bias → activation → store
    if (bias) {
        __m512 bias_v[NV];
        for (int v = 0; v < NV; ++v)
            bias_v[v] = _mm512_loadu_ps(bias + v * 16);
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_add_ps(acc[m][v], bias_v[v]);
    }

    if (fused_op != fused_postop_t::none) {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = apply_fused_postop(acc[m][v], fused_op);
    }

    for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v)
            _mm512_storeu_ps(C + m * ldc + v * 16, acc[m][v]);
}

// Explicit instantiation for MR=6, NR=16
template void brgemm_ukernel<6,1>(const float*, int, const float*, int,
    float*, int, int, int, float, const float*, fused_postop_t);

// ============================================================================
// BRGEMM tail kernel (dynamic MR/NR for edge tiles)
// ============================================================================
__attribute__((target("avx512f,avx512bw,fma"), noinline))
static void brgemm_tail_kernel(
    const float * __restrict__ A, int lda,
    const float * __restrict__ pb, int pb_stride,
    float * __restrict__ C, int ldc,
    int K, int BK, int mr_count, int nr_count, float beta,
    const float * __restrict__ bias, fused_postop_t fused_op) {

    const int full_vecs = nr_count / 16;
    const int rem = nr_count & 15;
    const __mmask16 rem_mask = rem ? static_cast<__mmask16>((1u << rem) - 1)
                                   : static_cast<__mmask16>(0);
    const int nv = (nr_count + 15) / 16;

    __m512 acc[12][4];

    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < mr_count; ++m) {
            for (int v = 0; v < full_vecs; ++v)
                acc[m][v] = _mm512_mul_ps(
                    bv, _mm512_loadu_ps(C + m * ldc + v * 16));
            if (rem)
                acc[m][full_vecs] = _mm512_mul_ps(
                    bv, _mm512_maskz_loadu_ps(rem_mask, C + m * ldc + full_vecs * 16));
        }
    } else {
        for (int m = 0; m < mr_count; ++m)
            for (int v = 0; v < nv; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    // Batch-reduce over all K
    for (int pc = 0; pc < K; pc += BK) {
        const int kb = std::min(BK, K - pc);
        const float *a_off = A + pc;
        const float *b_off = pb + pc * pb_stride;

        for (int kk = 0; kk < kb; ++kk) {
            __m512 bv[4];
            for (int v = 0; v < full_vecs; ++v)
                bv[v] = _mm512_loadu_ps(b_off + kk * pb_stride + v * 16);
            if (rem)
                bv[full_vecs] = _mm512_maskz_loadu_ps(
                    rem_mask, b_off + kk * pb_stride + full_vecs * 16);
            for (int m = 0; m < mr_count; ++m) {
                __m512 a = _mm512_set1_ps(a_off[m * lda + kk]);
                for (int v = 0; v < full_vecs; ++v)
                    acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
                if (rem)
                    acc[m][full_vecs] = _mm512_fmadd_ps(a, bv[full_vecs],
                                                         acc[m][full_vecs]);
            }
        }
    }

    // Epilogue
    if (bias) {
        for (int v = 0; v < full_vecs; ++v) {
            __m512 bv = _mm512_loadu_ps(bias + v * 16);
            for (int m = 0; m < mr_count; ++m)
                acc[m][v] = _mm512_add_ps(acc[m][v], bv);
        }
        if (rem) {
            __m512 bv = _mm512_maskz_loadu_ps(rem_mask, bias + full_vecs * 16);
            for (int m = 0; m < mr_count; ++m)
                acc[m][full_vecs] = _mm512_add_ps(acc[m][full_vecs], bv);
        }
    }

    if (fused_op != fused_postop_t::none) {
        for (int m = 0; m < mr_count; ++m)
            for (int v = 0; v < nv; ++v)
                acc[m][v] = apply_fused_postop(acc[m][v], fused_op);
    }

    for (int m = 0; m < mr_count; ++m) {
        for (int v = 0; v < full_vecs; ++v)
            _mm512_storeu_ps(C + m * ldc + v * 16, acc[m][v]);
        if (rem)
            _mm512_mask_storeu_ps(C + m * ldc + full_vecs * 16,
                                  rem_mask, acc[m][full_vecs]);
    }
}

// ============================================================================
// Vectorized alpha scaling
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
// BRGEMM function pointer type (same signature as template kernel)
// ============================================================================
using brgemm_fn_t = void (*)(const float *, int, const float *, int,
                              float *, int, int, int, float,
                              const float *, fused_postop_t);

// ============================================================================
// BRGEMM thread loop
//
// NO PC (K-block) loop — the microkernel handles full K internally.
// Thread parallelism over MC × NC tiles only.
// B pointer is pre-resolved HERE, not inside the microkernel.
// ============================================================================
static void brgemm_thread_loop(
    const GemmDescriptor &desc,
    const BrgemmPlan &plan,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params,
    const PrepackedWeight *prepacked_b) {

    const float *A = static_cast<const float *>(src);
    const float *B = static_cast<const float *>(weight);
    float *C       = static_cast<float *>(dst);
    const float *bias_f = static_cast<const float *>(bias);

    const int M = desc.M, N = desc.N, K = desc.K;
    const int lda = desc.lda, ldb = desc.ldb, ldc = desc.ldc;
    const float alpha = desc.alpha;
    const float beta  = (desc.alpha != 1.0f && desc.beta != 0.0f)
                        ? (desc.beta / desc.alpha) : desc.beta;
    const int MB = plan.MB, NB = plan.NB, BK = plan.BK;
    const int MR = plan.MR, NR = plan.NR;
    const int num_threads = plan.num_threads;
    const bool use_avx512 = uarch.avx512f;
    const bool has_bias = (bias_f != nullptr);
    const bool b_prepacked = (prepacked_b != nullptr);

    // Template kernel for full MR×NR tiles
    brgemm_fn_t hot_kernel = use_avx512 ? brgemm_ukernel<6,1> : nullptr;

    // Postop fusion
    fused_postop_t fused_op = fused_postop_t::none;
    int fused_idx = -1;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        auto pt = params.postop_[i].po_type;
        if (pt == post_op_type_t::relu) {
            if (params.postop_[i].alpha == 0.0f) {
                fused_op = fused_postop_t::relu; fused_idx = i; break;
            }
        } else if (pt == post_op_type_t::gelu_tanh) {
            fused_op = fused_postop_t::gelu_tanh; fused_idx = i; break;
        } else if (pt == post_op_type_t::gelu_erf) {
            fused_op = fused_postop_t::gelu_erf; fused_idx = i; break;
        } else if (pt == post_op_type_t::sigmoid) {
            fused_op = fused_postop_t::sigmoid; fused_idx = i; break;
        } else if (pt == post_op_type_t::tanh) {
            // Don't fuse tanh in BRGEMM kernel — GCC generates incorrect
            // code when avx512_tanh is inlined into the BRGEMM template.
            // Let it go through apply_postops_tile (vectorized separate pass).
            break;
        }
    }

    std::vector<matmul_post_op> remaining_postops;
    remaining_postops.reserve(params.postop_.size());
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        if (i != fused_idx) remaining_postops.push_back(params.postop_[i]);
    }
    const bool has_remaining_postops = !remaining_postops.empty();

    const bool can_fuse = (alpha == 1.0f);
    const int ic_tiles = (M + MB - 1) / MB;
    const int jc_tiles = (N + NB - 1) / NB;
    const int total_tiles = ic_tiles * jc_tiles;

    if (num_threads <= 1) {
        for (int ic_idx = 0; ic_idx < ic_tiles; ++ic_idx) {
            const int ic = ic_idx * MB;
            const int mb_act = std::min(MB, M - ic);
            const int m_panels = (mb_act + MR - 1) / MR;

            for (int jc_idx = 0; jc_idx < jc_tiles; ++jc_idx) {
                const int jc = jc_idx * NB;
                const int nb_act = std::min(NB, N - jc);

                for (int jr = 0; jr < nb_act; jr += NR) {
                    const int nr_act = std::min(NR, nb_act - jr);
                    const int col = jc + jr;
                    const bool full_nr = (nr_act == NR);

                    // Resolve B pointer inline (no lambda)
                    const float *pb;
                    int pb_stride;
                    if (b_prepacked) {
                        int panel_idx = col / NR_PACK;
                        int in_panel_off = col % NR_PACK;
                        pb = prepacked_b->get_panel(0, panel_idx) + in_panel_off;
                        pb_stride = NR_PACK;
                    } else {
                        pb = B + col;
                        pb_stride = ldb;
                    }
                    const float *tile_bias =
                        (has_bias && can_fuse) ? (bias_f + col) : nullptr;
                    const fused_postop_t tile_fop =
                        can_fuse ? fused_op : fused_postop_t::none;

                    for (int ip = 0; ip < m_panels; ++ip) {
                        const int ir = ip * MR;
                        const int mr_act = std::min(MR, mb_act - ir);
                        const float *At = A + (ic + ir) * lda;
                        float *Ct = C + (ic + ir) * ldc + col;

                        if (hot_kernel && full_nr && mr_act == MR) {
                            hot_kernel(At, lda, pb, pb_stride,
                                       Ct, ldc, K, BK, beta,
                                       tile_bias, tile_fop);
                        } else if (use_avx512) {
                            brgemm_tail_kernel(At, lda, pb, pb_stride,
                                               Ct, ldc, K, BK,
                                               mr_act, nr_act, beta,
                                               tile_bias, tile_fop);
                        }

                    }
                }

                // Post-tile: alpha scaling, unfused postops
                float *Ctile = C + ic * ldc + jc;
                if (alpha != 1.0f) {
                    scale_tile(Ctile, ldc, mb_act, nb_act, alpha);
                    if (has_bias)
                        apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                           jc, ic, bias_f, {});
                    if (fused_op != fused_postop_t::none && fused_idx >= 0)
                        apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                           jc, ic, nullptr,
                                           {params.postop_[fused_idx]});
                }
                if (has_remaining_postops) {
                    apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                       jc, ic, nullptr, remaining_postops);
                }
            }
        }
    } else {
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp for schedule(dynamic, 1)
            for (int tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
                const int ic_idx = tile_idx / jc_tiles;
                const int jc_idx = tile_idx % jc_tiles;

                const int ic = ic_idx * MB;
                const int jc = jc_idx * NB;
                const int mb_act = std::min(MB, M - ic);
                const int nb_act = std::min(NB, N - jc);
                const int m_panels = (mb_act + MR - 1) / MR;

                for (int jr = 0; jr < nb_act; jr += NR) {
                    const int nr_act = std::min(NR, nb_act - jr);
                    const int col = jc + jr;
                    const bool full_nr = (nr_act == NR);

                    const float *pb;
                    int pb_stride;
                    if (b_prepacked) {
                        int panel_idx = col / NR_PACK;
                        int in_panel_off = col % NR_PACK;
                        pb = prepacked_b->get_panel(0, panel_idx) + in_panel_off;
                        pb_stride = NR_PACK;
                    } else {
                        pb = B + col;
                        pb_stride = ldb;
                    }
                    const float *tile_bias =
                        (has_bias && can_fuse) ? (bias_f + col) : nullptr;
                    const fused_postop_t tile_fop =
                        can_fuse ? fused_op : fused_postop_t::none;

                    for (int ip = 0; ip < m_panels; ++ip) {
                        const int ir = ip * MR;
                        const int mr_act = std::min(MR, mb_act - ir);
                        const float *At = A + (ic + ir) * lda;
                        float *Ct = C + (ic + ir) * ldc + col;

                        if (hot_kernel && full_nr && mr_act == MR) {
                            hot_kernel(At, lda, pb, pb_stride,
                                       Ct, ldc, K, BK, beta,
                                       tile_bias, tile_fop);
                        } else if (use_avx512) {
                            brgemm_tail_kernel(At, lda, pb, pb_stride,
                                               Ct, ldc, K, BK,
                                               mr_act, nr_act, beta,
                                               tile_bias, tile_fop);
                        }
                    }
                }

                float *Ctile = C + ic * ldc + jc;
                if (alpha != 1.0f) {
                    scale_tile(Ctile, ldc, mb_act, nb_act, alpha);
                    if (has_bias)
                        apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                           jc, ic, bias_f, {});
                    if (fused_op != fused_postop_t::none && fused_idx >= 0)
                        apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                           jc, ic, nullptr,
                                           {params.postop_[fused_idx]});
                }
                if (has_remaining_postops) {
                    apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                       jc, ic, nullptr, remaining_postops);
                }
            }
        } // omp parallel
    }

}

// ============================================================================
// BRGEMM execute: plan + B prepacking + thread loop
// ============================================================================

void brgemm_execute(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int N = desc.N, K = desc.K;
    const bool transB = desc.transB;
    const bool is_weights_const = desc.is_weights_const;

    // ── 1. BRGEMM plan ──
    BrgemmPlan bplan = plan_brgemm(desc, uarch);

    // ── 2. B matrix prepacking ──
    //
    // Three cases:
    //   (a) WEIGHT_CACHE=1 + is_weights_const: one-time prepack, cached forever.
    //   (b) !can_cache + transB=true: per-call packing into NR_PACK panels.
    //   (c) !can_cache + transB=false: prepacked_b = nullptr, direct access.
    //
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();
    static int32_t s_otf_bpack =
        matmul_config_t::instance().get_otf_bpack();
    const PrepackedWeight *prepacked_b = nullptr;
    const bool can_cache = is_weights_const && (s_weight_cache != 0);

    if (can_cache) {
        PrepackedWeightKey bk{weight, K, N, desc.ldb, transB};
        prepacked_b = PrepackedWeightCache::instance().get_or_prepack(
            bk, static_cast<const float *>(weight));
    } else if (transB) {
        static thread_local float *s_tb = nullptr;
        static thread_local size_t s_tb_cap = 0;
        static thread_local PrepackedWeight s_tb_pw;

        const int np = (N + NR_PACK - 1) / NR_PACK;
        const size_t total = static_cast<size_t>(np) * K * NR_PACK;

        if (s_tb_cap < total) {
            std::free(s_tb);
            s_tb = static_cast<float *>(std::aligned_alloc(
                64, ((total * sizeof(float) + 63) & ~size_t(63))));
            s_tb_cap = total;
        }

        if (s_tb) {
            const float *Bf = static_cast<const float *>(weight);
            for (int jp = 0; jp < np; ++jp) {
                const int j0 = jp * NR_PACK;
                const int nr_act = std::min(NR_PACK, N - j0);
                float *dst_p = s_tb + static_cast<size_t>(jp) * K * NR_PACK;
                for (int kk = 0; kk < K; ++kk) {
                    float *d = dst_p + kk * NR_PACK;
                    for (int n = 0; n < nr_act; ++n)
                        d[n] = Bf[(j0 + n) * desc.ldb + kk];
                    for (int n = nr_act; n < NR_PACK; ++n)
                        d[n] = 0.0f;
                }
            }

            s_tb_pw.data = s_tb;
            s_tb_pw.K = K;
            s_tb_pw.N = N;
            s_tb_pw.n_panels = np;
            prepacked_b = &s_tb_pw;
        }
    }

    const bool do_otf = (!prepacked_b && s_otf_bpack != 0);
    if (!prepacked_b && !do_otf) {
        gemm_execute(desc, uarch, src, weight, dst, bias, params);
        return;
    }

    // ── 3. Thread loop ──
    brgemm_thread_loop(desc, bplan, uarch, src, weight, dst, bias, params,
                       prepacked_b);
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
