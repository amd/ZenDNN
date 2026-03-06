/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

//
// FP32 GEMM Looper: MC/NC/KC tile loops + OMP threading + buffer management.
//
// Layer 2 in the Planner/Looper/Kernel architecture.
//

#include "lowoha_operators/matmul/matmul_ai/gemm/looper/fp32_gemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/planner/fp32_gemm_plan.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/kernel/fp32/fp32_gemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/postop.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/fp32_packing.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"

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

static void ai_thread_loop(
    const GemmDescriptor &desc,
    const BlockPlan &plan,
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
    const bool transA = desc.transA;
    const float alpha = desc.alpha;
    // When alpha != 1 and beta != 0, pass beta/alpha to the microkernel.
    // After scale_tile: alpha*(A*B + (beta/alpha)*C_old) = alpha*A*B + beta*C_old.
    const float beta  = (desc.alpha != 1.0f && desc.beta != 0.0f)
                        ? (desc.beta / desc.alpha) : desc.beta;
    const int MB = plan.MB, NB = plan.NB, KB = plan.KB;
    const int MR = plan.MR, NR = plan.NR;
    const int num_threads = plan.num_threads;
    const bool use_avx512 = uarch.avx512f;
    const bool has_bias = (bias_f != nullptr);

    // Detect fusable post-op (first eltwise op in the chain).
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
            fused_op = fused_postop_t::tanh_op; fused_idx = i; break;
        } else if (pt == post_op_type_t::swish) {
            fused_op = fused_postop_t::swish; fused_idx = i; break;
        }
    }

    std::vector<matmul_post_op> remaining_postops;
    remaining_postops.reserve(params.postop_.size());
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        if (i != fused_idx) remaining_postops.push_back(params.postop_[i]);
    }
    const bool has_remaining_postops = !remaining_postops.empty();

    // Pack controls (read once from singleton, cached across calls).
    // Auto-enable A packing when row stride exceeds L1 stride prefetcher
    // limit (~4KB on Zen4/5). For lda=3584 FP32: 14KB stride → pack.
    const bool do_pack_a = transA
                           || (lda * static_cast<int>(sizeof(float)) > 4096);

    ukernel_fn_t hot_ukernel = use_avx512 ? select_ukernel(MR, NR) : nullptr;

    const int jc_tiles = (N + NB - 1) / NB;
    const size_t pa_elems = static_cast<size_t>(((MB + MR - 1) / MR) * MR) * KB;

    // When do_pack_a is true but pa_buf is null (e.g. aligned_alloc failed),
    // fall back to direct A access instead of crashing.
    auto run_loop = [&](float *pa_buf) {
        const bool use_packed_a = do_pack_a && (pa_buf != nullptr);
        for (int jc_idx = 0; jc_idx < jc_tiles; ++jc_idx) {
            const int jc = jc_idx * NB;
            const int nb_act = std::min(NB, N - jc);

            for (int pc = 0; pc < K; pc += KB) {
                const int kb_act = std::min(KB, K - pc);
                const bool is_last_k = (pc + kb_act >= K);

                // B data: prepacked (panel-wide K-contiguous) or direct access
                const bool b_use_prepacked = (prepacked_b != nullptr);

                for (int ic = 0; ic < M; ic += MB) {
                    const int mb_act = std::min(MB, M - ic);

                    // -- Pack A (or skip) --
                    const float *a_base;
                    if (use_packed_a) {
                        pack_a_block(A, pa_buf, ic, pc, M, K, lda, transA,
                                     mb_act, kb_act, MR);
                        a_base = pa_buf;
                    } else {
                        a_base = A + ic * lda + pc;
                    }

                    const float tile_beta = (pc == 0) ? beta : 1.0f;
                    const int m_panels = (mb_act + MR - 1) / MR;

                    for (int jr = 0; jr < nb_act; jr += NR) {
                        const int nr_act = std::min(NR, nb_act - jr);
                        const bool full_nr = (nr_act == NR);

                        const float *pb_tile;
                        int pb_stride;
                        bool panel_crossing = false;
                        int nr_part1 = 0;

                        if (b_use_prepacked) {
                            int col = jc + jr;
                            int panel_idx = col / NR_PACK;
                            int in_panel_off = col % NR_PACK;
                            if (in_panel_off + nr_act <= NR_PACK) {
                                pb_tile = prepacked_b->get_panel(pc, panel_idx) + in_panel_off;
                                pb_stride = NR_PACK;
                            } else {
                                panel_crossing = true;
                                nr_part1 = NR_PACK - in_panel_off;
                                pb_tile = prepacked_b->get_panel(pc, panel_idx) + in_panel_off;
                                pb_stride = NR_PACK;
                            }
                        } else {
                            pb_tile = B + pc * ldb + (jc + jr);
                            pb_stride = ldb;
                        }

                        // Fuse bias/binary/activation only when alpha==1 (epilogue order is correct).
                        // When alpha!=1, bias/binary/activation must come AFTER scale_tile.
                        const bool can_fuse = (alpha == 1.0f);
                        const float *tile_bias =
                            (has_bias && is_last_k && can_fuse) ? (bias_f + jc + jr) : nullptr;
                        const fused_postop_t tile_fop =
                            (is_last_k && can_fuse) ? fused_op : fused_postop_t::none;

                        for (int ip = 0; ip < m_panels; ++ip) {
                            const int ir = ip * MR;
                            const int mr_act = std::min(MR, mb_act - ir);
                            float *Ct = C + (ic + ir) * ldc + (jc + jr);

                            const float *pa;
                            int pa_stride;
                            if (use_packed_a) {
                                pa = a_base + ip * MR * kb_act;
                                pa_stride = kb_act;
                            } else {
                                pa = a_base + ir * lda;
                                pa_stride = lda;
                            }

                            if (!panel_crossing) {
                                if (hot_ukernel && full_nr && mr_act == MR) {
                                    hot_ukernel(pa, pa_stride, pb_tile, pb_stride,
                                                Ct, ldc, kb_act, tile_beta,
                                                tile_bias, tile_fop);
                                } else if (use_avx512) {
                                    avx512_tail_kernel(pa, pa_stride, pb_tile, pb_stride,
                                                       Ct, ldc, kb_act, mr_act, nr_act,
                                                       tile_beta, tile_bias, tile_fop);
                                } else {
                                    scalar_microkernel(pa, pa_stride, pb_tile, pb_stride,
                                                       Ct, ldc, kb_act, mr_act, nr_act,
                                                       tile_beta, tile_bias, tile_fop);
                                }
                            } else {
                                int nr_part2 = nr_act - nr_part1;

                                // Panel crossing only happens with prepacked B
                                int col = jc + jr;
                                int pidx = col / NR_PACK;

                                const float *pb1 = prepacked_b->get_panel(pc, pidx) + (col % NR_PACK);
                                const float *pb2 = prepacked_b->get_panel(pc, pidx + 1);

                                const float *bias1 = tile_bias;
                                avx512_tail_kernel(pa, pa_stride, pb1, NR_PACK,
                                                   Ct, ldc, kb_act, mr_act, nr_part1,
                                                   tile_beta, bias1, tile_fop);

                                const float *bias2 = tile_bias ? (tile_bias + nr_part1) : nullptr;
                                avx512_tail_kernel(pa, pa_stride, pb2, NR_PACK,
                                                   Ct + nr_part1, ldc, kb_act, mr_act, nr_part2,
                                                   tile_beta, bias2, tile_fop);
                            }
                        }
                    }

                    // Post-K-block: alpha scaling, unfused postops
                    if (is_last_k) {
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
                } // ic
            } // pc
        } // jc
    }; // end run_loop lambda (used for single-thread only)

    if (num_threads <= 1) {
        // Single-thread: skip OpenMP, use thread-local reusable buffer
        static thread_local float *s_pa = nullptr;
        static thread_local size_t s_pa_cap = 0;
        if (do_pack_a && s_pa_cap < pa_elems) {
            std::free(s_pa);
            s_pa = static_cast<float *>(
                std::aligned_alloc(64, ((pa_elems * 4 + 63) & ~size_t(63))));
            s_pa_cap = s_pa ? pa_elems : 0;
        }
        run_loop((do_pack_a && s_pa) ? s_pa : nullptr);
    } else {
        const int ic_tiles = (M + MB - 1) / MB;
        const int total_2d_tiles = ic_tiles * jc_tiles;
        const bool b_use_prepacked = (prepacked_b != nullptr);

        // BRGEMM-style: tile-outer, K-inner. Each thread processes ALL
        // K-blocks for its tiles, eliminating inter-K-block barriers.
        const int active_threads = std::min(num_threads, total_2d_tiles);

        const bool skip_pack_a = do_pack_a
            && (static_cast<size_t>(M) * K * sizeof(float)
                <= static_cast<size_t>(uarch.l2_bytes));
        const bool actual_pack_a = do_pack_a && !skip_pack_a;

        #pragma omp parallel num_threads(active_threads)
        {
            static thread_local float *tl_pa = nullptr;
            static thread_local size_t tl_pa_cap = 0;

            #pragma omp for schedule(static)
            for (int tile_idx = 0; tile_idx < total_2d_tiles; ++tile_idx) {
                const int ic_idx = tile_idx / jc_tiles;
                const int jc_idx = tile_idx % jc_tiles;
                const int ic = ic_idx * MB;
                const int jc = jc_idx * NB;
                const int mb_act = std::min(MB, M - ic);
                const int nb_act = std::min(NB, N - jc);
                const int m_panels = (mb_act + MR - 1) / MR;

                for (int pc = 0; pc < K; pc += KB) {
                    const int kb_act = std::min(KB, K - pc);
                    const bool is_last_k = (pc + kb_act >= K);
                    const float tile_beta = (pc == 0) ? beta : 1.0f;

                    const float *a_base;
                    int a_stride_base;
                    if (actual_pack_a) {
                        size_t need = static_cast<size_t>(
                            ((mb_act + MR - 1) / MR) * MR) * kb_act;
                        if (tl_pa_cap < need) {
                            std::free(tl_pa);
                            tl_pa = static_cast<float *>(std::aligned_alloc(
                                64, ((need * 4 + 63) & ~size_t(63))));
                            tl_pa_cap = tl_pa ? need : 0;
                        }
                        if (tl_pa) {
                            pack_a_block(A, tl_pa, ic, pc, M, K, lda, transA,
                                         mb_act, kb_act, MR);
                            a_base = tl_pa;
                            a_stride_base = kb_act;
                        } else {
                            a_base = A + ic * lda + pc;
                            a_stride_base = lda;
                        }
                    } else {
                        a_base = A + ic * lda + pc;
                        a_stride_base = lda;
                    }

                    for (int jr = 0; jr < nb_act; jr += NR) {
                        const int nr_act = std::min(NR, nb_act - jr);
                        const bool full_nr = (nr_act == NR);

                        const float *pb_tile;
                        int pb_stride;
                        bool panel_crossing = false;
                        int nr_part1 = 0;

                        if (b_use_prepacked) {
                            int col = jc + jr;
                            int panel_idx = col / NR_PACK;
                            int in_panel_off = col % NR_PACK;
                            if (in_panel_off + nr_act <= NR_PACK) {
                                pb_tile = prepacked_b->get_panel(pc, panel_idx) + in_panel_off;
                                pb_stride = NR_PACK;
                            } else {
                                panel_crossing = true;
                                nr_part1 = NR_PACK - in_panel_off;
                                pb_tile = prepacked_b->get_panel(pc, panel_idx) + in_panel_off;
                                pb_stride = NR_PACK;
                            }
                        } else {
                            pb_tile = B + pc * ldb + (jc + jr);
                            pb_stride = ldb;
                        }

                        const bool can_fuse = (alpha == 1.0f);
                        const float *tile_bias =
                            (has_bias && is_last_k && can_fuse) ? (bias_f + jc + jr) : nullptr;
                        const fused_postop_t tile_fop =
                            (is_last_k && can_fuse) ? fused_op : fused_postop_t::none;

                        for (int ip = 0; ip < m_panels; ++ip) {
                            const int ir = ip * MR;
                            const int mr_act = std::min(MR, mb_act - ir);
                            float *Ct = C + (ic + ir) * ldc + (jc + jr);

                            const float *pa;
                            int pa_stride;
                            if (actual_pack_a) {
                                pa = a_base + ip * MR * kb_act;
                                pa_stride = kb_act;
                            } else {
                                pa = a_base + ir * a_stride_base;
                                pa_stride = a_stride_base;
                            }

                            if (!panel_crossing) {
                                if (hot_ukernel && full_nr && mr_act == MR) {
                                    hot_ukernel(pa, pa_stride, pb_tile, pb_stride,
                                                Ct, ldc, kb_act, tile_beta,
                                                tile_bias, tile_fop);
                                } else if (use_avx512) {
                                    avx512_tail_kernel(pa, pa_stride, pb_tile, pb_stride,
                                                       Ct, ldc, kb_act, mr_act, nr_act,
                                                       tile_beta, tile_bias, tile_fop);
                                } else {
                                    scalar_microkernel(pa, pa_stride, pb_tile, pb_stride,
                                                       Ct, ldc, kb_act, mr_act, nr_act,
                                                       tile_beta, tile_bias, tile_fop);
                                }
                            } else {
                                int nr_part2 = nr_act - nr_part1;
                                int col = jc + jr;
                                int pidx = col / NR_PACK;

                                const float *pb1 = prepacked_b->get_panel(pc, pidx) + (col % NR_PACK);
                                const float *pb2 = prepacked_b->get_panel(pc, pidx + 1);

                                avx512_tail_kernel(pa, pa_stride, pb1, NR_PACK,
                                                   Ct, ldc, kb_act, mr_act, nr_part1,
                                                   tile_beta, tile_bias, tile_fop);

                                const float *bias2 = tile_bias ? (tile_bias + nr_part1) : nullptr;
                                avx512_tail_kernel(pa, pa_stride, pb2, NR_PACK,
                                                   Ct + nr_part1, ldc, kb_act, mr_act, nr_part2,
                                                   tile_beta, bias2, tile_fop);
                            }
                        }
                    }

                    if (is_last_k) {
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
                } // K-loop (BRGEMM)
            } // omp for
        } // omp parallel
    }
}

// ============================================================================
// FP32 GEMM entry point (restructured: Planner + Looper + Kernel)
// ============================================================================
void gemm_execute(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int K = desc.K, N = desc.N;
    const bool transB = desc.transB;
    const bool is_weights_const = desc.is_weights_const;

    // Layer 1: Planner
    FP32GemmPlan fp = plan_fp32_gemm(desc, uarch, params);

    if (apilog_info_enabled()) {
        apilog_info("AI FP32 GEMM plan: M=", desc.M, " N=", N, " K=", K,
                    " MB=", fp.plan.MB, " NB=", fp.plan.NB, " KB=", fp.plan.KB,
                    " MR=", fp.plan.MR, " NR=", fp.plan.NR,
                    " threads=", fp.plan.num_threads);
    }

    // Weight caching
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();
    const PrepackedWeight *prepacked_b = nullptr;
    const bool can_cache = is_weights_const && (s_weight_cache != 0);

    if (can_cache) {
        PrepackedWeightKey bk{weight, K, N, desc.ldb, transB};
        prepacked_b = PrepackedWeightCache::instance().get_or_prepack(
            bk, static_cast<const float *>(weight));
    } else if (transB || (desc.num_threads > 1 && desc.ldb > NR_PACK)) {
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
                    if (transB) {
                        for (int n = 0; n < nr_act; ++n)
                            d[n] = Bf[(j0 + n) * desc.ldb + kk];
                    } else {
                        for (int n = 0; n < nr_act; ++n)
                            d[n] = Bf[kk * desc.ldb + (j0 + n)];
                    }
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

    // Layer 2: Looper
    ai_thread_loop(desc, fp.plan, uarch, src, weight, dst, bias, params,
                   prepacked_b);
}


} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
