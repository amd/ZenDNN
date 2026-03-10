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

// STL and project headers BEFORE the pragma (these pull in <vector>, <string>)
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/bf16_gemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/bf16_brgemm_plan.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/postop.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"
#include "common/bfloat16.hpp"

#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Enable AVX-512 AFTER all STL headers to avoid always_inline ABI mismatch
#pragma GCC target("avx512f,avx512bf16,avx512bw,avx512vl,fma")

#include <immintrin.h>
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/bf16_packing.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

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
            uint32_t bits;
            std::memcpy(&bits, &src_fp32[m * ldc_fp32 + n], sizeof(bits));
            uint32_t rounding_bias = (bits >> 16) & 1;
            bits += 0x7FFF + rounding_bias;
            dst_bf16[m * ldc_bf16 + n] = static_cast<uint16_t>(bits >> 16);
        }
    }
}

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

    if (has_bias && bias_is_bf16 && !s_bias_fp32) return;

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
        } else if (pt == post_op_type_t::swish) {
            fused_op = fused_postop_t::swish; fused_idx = i; break;
        }
    }

    std::vector<matmul_post_op> remaining_postops;
    std::vector<matmul_post_op> fused_as_postop;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i)
        if (i != fused_idx) remaining_postops.push_back(params.postop_[i]);
    if (fused_idx >= 0)
        fused_as_postop.push_back(params.postop_[fused_idx]);
    const bool has_remaining_postops = !remaining_postops.empty();

    const bool can_fuse = (alpha == 1.0f);
    const bool can_direct_bf16 = dst_is_bf16 && can_fuse && !has_remaining_postops;

    // FP32 C buffer for BF16 output
    float *C_fp32;
    int ldc_fp32;
    bool need_fp32_buf = dst_is_bf16;
    static thread_local float *s_c_buf = nullptr;
    static thread_local size_t s_c_cap = 0;

    if (need_fp32_buf) {
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
    } else {
        C_fp32 = static_cast<float *>(dst);
        ldc_fp32 = ldc;
    }

    uint16_t *C_bf16_dst = dst_is_bf16
        ? static_cast<uint16_t *>(dst) : nullptr;

    bf16_brgemm_fn_t hot_kernel = select_bf16_brgemm_kernel(MR, NR);

    const bool is_decode = (M <= 4);
    const int actual_MR = is_decode ? M : MR;
    // Pre-resolve tail kernel for M-remainder panels (avoids dispatch per tile)
    const int m_tail = M % actual_MR;
    bf16_brgemm_fn_t tail_kernel = (m_tail > 0 && m_tail <= 4)
        ? select_bf16_brgemm_kernel(m_tail, NR) : nullptr;

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
                pack_b_vnni_strip_full(B_raw, ldb, transB,
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
                } else if (full_nr && tail_kernel && mr_act == m_tail) {
                    tail_kernel(At, lda, pb, pb_stride,
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
            }
        }

        // Post-tile epilogue: apply postops + BF16 conversion per-tile
        // while C data is still hot in L1. Avoids costly full-matrix passes.
        if (C_fp32) {
            float *Ctile = C_fp32 + ic * ldc_fp32 + jc;
            const int nb_act_e = std::min(NB, N - jc);
            const int mb_act_e = std::min(MB, M - ic);
            if (alpha != 1.0f) {
                scale_tile(Ctile, ldc_fp32, mb_act_e, nb_act_e, alpha);
                if (has_bias) {
                    static const std::vector<matmul_post_op> empty_ops;
                    apply_postops_tile(Ctile, ldc_fp32, mb_act_e, nb_act_e,
                                       jc, ic, bias_f, empty_ops);
                }
                if (!fused_as_postop.empty())
                    apply_postops_tile(Ctile, ldc_fp32, mb_act_e, nb_act_e,
                                       jc, ic, nullptr, fused_as_postop);
            }
            if (has_remaining_postops) {
                apply_postops_tile(Ctile, ldc_fp32, mb_act_e, nb_act_e,
                                   jc, ic, nullptr, remaining_postops);
            }
            if (need_fp32_buf && !can_direct_bf16) {
                uint16_t *dst_tile = static_cast<uint16_t *>(dst)
                    + ic * ldc + jc;
                convert_fp32_to_bf16_tile(Ctile, ldc_fp32,
                    dst_tile, ldc, mb_act_e, nb_act_e);
            }
        }
    };

    // ────────────────────────────────────────────────────────────────
    // Fast path: prepacked weights + single MC tile + fusable postops.
    // Flat N-parallel loop — no tile dispatch lambda, no OTF checks.
    // ────────────────────────────────────────────────────────────────
    if (prepacked_b && can_fuse && ic_tiles == 1) {
        const int n_strips = (N + NR - 1) / NR;
        const int m_panels = (M + actual_MR - 1) / actual_MR;
        const int act_threads = std::min(num_threads, n_strips);

        #pragma omp parallel for schedule(static) num_threads(act_threads) \
            if(act_threads > 1)
        for (int js = 0; js < n_strips; ++js) {
            const int col = js * NR;
            const int nr_act = std::min(NR, N - col);
            const int panel_idx = col / NR_PACK;
            const int in_panel_off = col % NR_PACK;
            const uint16_t *pb = prepacked_b->get_panel(0, panel_idx)
                                 + in_panel_off * VNNI_PAIR;
            const float *tb = has_bias ? bias_f + col : nullptr;
            const bool full_nr = (nr_act == NR);

            for (int ip = 0; ip < m_panels; ++ip) {
                const int ir = ip * actual_MR;
                const int mr_act = std::min(actual_MR, M - ir);
                const uint16_t *At = A + ir * lda;
                float *Ct = C_fp32 ? C_fp32 + ir * ldc_fp32 + col : nullptr;
                uint16_t *dbf = can_direct_bf16
                    ? C_bf16_dst + ir * ldc + col : nullptr;
                int dbf_ldc = can_direct_bf16 ? ldc : 0;

                if (full_nr && mr_act == actual_MR && hot_kernel) {
                    hot_kernel(At, lda, pb, vnni_stride,
                               Ct, ldc_fp32, K, BK, beta,
                               tb, fused_op, dbf, dbf_ldc);
                } else if (full_nr && mr_act >= 1 && mr_act <= 4) {
                    auto tail_uk = select_bf16_brgemm_kernel(mr_act, NR);
                    if (tail_uk) {
                        tail_uk(At, lda, pb, vnni_stride,
                                Ct, ldc_fp32, K, BK, beta,
                                tb, fused_op, dbf, dbf_ldc);
                    } else {
                        bf16_brgemm_tail_kernel(At, lda, pb, vnni_stride,
                                                Ct, ldc_fp32, K, BK,
                                                mr_act, nr_act, beta,
                                                tb, fused_op, dbf, dbf_ldc);
                    }
                } else {
                    bf16_brgemm_tail_kernel(At, lda, pb, vnni_stride,
                                            Ct, ldc_fp32, K, BK,
                                            mr_act, nr_act, beta,
                                            tb, fused_op, dbf, dbf_ldc);
                }
            }

            // Per-strip postop + BF16 conversion (data hot in L1)
            if (C_fp32) {
                float *strip_c = C_fp32 + col;
                if (has_remaining_postops)
                    apply_postops_tile(strip_c, ldc_fp32, M, nr_act,
                                       col, 0, nullptr, remaining_postops);
                if (need_fp32_buf && !can_direct_bf16) {
                    uint16_t *strip_dst = static_cast<uint16_t *>(dst) + col;
                    convert_fp32_to_bf16_tile(strip_c, ldc_fp32,
                        strip_dst, ldc, M, nr_act);
                }
            }
        }
        return;
    }

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
        // JC-outer, IC-inner: consecutive tiles share the same N column,
        // so B panel stays hot in L2 across M panels.
        for (int t = 0; t < total_tiles; ++t) {
            int jc = (t / ic_tiles) * NB;
            int ic = (t % ic_tiles) * MB;
            process_tile(ic, jc, otf);
        }
    } else {
        #pragma omp parallel num_threads(active_threads)
        {
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
                int jc = (t / ic_tiles) * NB;
                int ic = (t % ic_tiles) * MB;
                process_tile(ic, jc, otf);
            }
        }
    }

    // BF16 conversion is now done per-tile inside process_tile / per-strip
    // in the fast path. No full-matrix pass needed here.
}

// ============================================================================
// BF16 BRGEMM execute: plan + VNNI B prepack + thread loop
// ============================================================================



// ============================================================================
// BF16 BRGEMM entry point (Planner + Looper + Kernel)
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

    {
        int b_panel_limit = (uarch.l2_bytes / 2)
                            / (NR_PACK * static_cast<int>(sizeof(uint16_t)));
        bool b_exceeds_l2 = (K > b_panel_limit);
        bool decode_wide_n = (is_decode && K < N);

        bool small_n = (N < NR_PACK);

        if (decode_wide_n || b_exceeds_l2 || small_n) {
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

    if (!is_decode) {
        int mb_budget = (uarch.l1d_bytes + uarch.l2_bytes)
                        / std::max(bplan.NB * 4 + bplan.BK * 2, 1);
        mb_budget = (mb_budget / bplan.MR) * bplan.MR;
        mb_budget = std::max(mb_budget, bplan.MR);
        bplan.MB = std::min(mb_budget, M);

        // For large M with multi-threading: if the cache-based MB creates
        // badly imbalanced IC tiles, reduce MB for better load balance.
        // Preserve MB >= M for small M (enables the prepacked fast path).
        if (bplan.num_threads > 1 && bplan.MB < M) {
            int m_panels = (M + bplan.MR - 1) / bplan.MR;
            int jc_tiles_now = (N + bplan.NB - 1) / bplan.NB;
            int ic_tiles_now = (M + bplan.MB - 1) / bplan.MB;

            // Check for imbalance: last IC tile much smaller than others
            int last_ic_rows = M - (ic_tiles_now - 1) * bplan.MB;
            bool imbalanced = (last_ic_rows < bplan.MB / 2)
                              || (ic_tiles_now * jc_tiles_now < bplan.num_threads);

            if (imbalanced) {
                int target_tiles = std::max(2 * bplan.num_threads,
                                            bplan.num_threads + jc_tiles_now);
                int needed_ic = (target_tiles + jc_tiles_now - 1) / jc_tiles_now;
                needed_ic = std::max(needed_ic, 2);
                int ppb = std::max(m_panels / needed_ic, 1);
                bplan.MB = ppb * bplan.MR;
                bplan.MB = std::min(bplan.MB, M);
            }
        }
    } else {
        bplan.MB = M;
    }

    if (apilog_info_enabled()) {
        int jt = (N + bplan.NB - 1) / bplan.NB;
        int it = (M + bplan.MB - 1) / bplan.MB;
        apilog_info("Native BF16 BRGEMM plan: M=", M, " N=", N, " K=", K,
                    " MB=", bplan.MB, " NB=", bplan.NB, " BK=", bplan.BK,
                    " MR=", bplan.MR, " NR=", bplan.NR,
                    " ic=", it, " jc=", jt, " tiles=", it*jt,
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

    bf16_brgemm_thread_loop(desc, bplan, uarch, src, weight, dst, bias,
                            params, prepacked_b, do_otf);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
