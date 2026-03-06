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
// BF16 GEMM Looper: MC/NC/KC tile loops + OMP threading + buffer management.
//
// Layer 2 in the Planner/Looper/Kernel architecture.
// Calls: planner (planner/bf16_gemm_plan) for blocking decisions.
//        microkernels (kernel/bf16/bf16_gemm_ukernel) for register-tile compute.
//        packing (common/bf16_packing) for VNNI data reformatting.
//        tile utils (common/tile_utils) for scale/convert operations.
//

// STL and project headers BEFORE the pragma (these pull in <vector>, <string>)
#include "lowoha_operators/matmul/matmul_ai/gemm/looper/bf16_gemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/planner/bf16_gemm_plan.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/kernel/bf16/bf16_gemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/postop.hpp"
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
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/bf16_packing.hpp"

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

// ============================================================================
// BF16 GEMM tile loop (internal)
// ============================================================================
static void bf16_thread_loop(
    const GemmDescriptor &desc,
    const BlockPlan &plan,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params,
    const BF16PrepackedWeight *prepacked_b) {

    const uint16_t *A = static_cast<const uint16_t *>(src);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);

    const int M = desc.M, N = desc.N, K = desc.K;
    const int lda = desc.lda, ldc = desc.ldc;
    const float alpha = desc.alpha;
    const float beta = (desc.alpha != 1.0f && desc.beta != 0.0f)
                       ? (desc.beta / desc.alpha) : desc.beta;
    const int MB = plan.MB, NB = plan.NB, KB = plan.KB;
    const int MR = plan.MR, NR = plan.NR;
    const int num_threads = plan.num_threads;
    const bool has_bias = (bias != nullptr);
    const bool b_prepacked = (prepacked_b != nullptr);

    // On-the-fly packing support: raw B pointer for when weights aren't cached
    const uint16_t *B_raw = static_cast<const uint16_t *>(weight);
    const int ldb_raw = desc.ldb;
    const bool transB_raw = desc.transB;
    const int K_padded_raw = (K + 1) & ~1;

    // Convert bias to FP32 if needed (cached across calls if pointer unchanged).
    // Microkernels always receive FP32 bias pointers.
    static thread_local float *s_bias_fp32 = nullptr;
    static thread_local size_t s_bias_cap = 0;
    static thread_local const void *s_last_bias_ptr = nullptr;
    static thread_local int s_last_bias_N = 0;
    const float *bias_f = nullptr;

    if (has_bias) {
        if (bias_is_bf16) {
            // BF16 bias → convert to FP32 (skip if pointer & N unchanged)
            const bool bias_changed =
                (bias != s_last_bias_ptr) || (N != s_last_bias_N);
            if (bias_changed || s_bias_cap < static_cast<size_t>(N)) {
                if (s_bias_cap < static_cast<size_t>(N)) {
                    std::free(s_bias_fp32);
                    s_bias_fp32 = static_cast<float *>(std::aligned_alloc(
                        64, ((N * sizeof(float) + 63) & ~size_t(63))));
                    s_bias_cap = N;
                }
                if (s_bias_fp32) {
                    const uint16_t *bias_bf16 =
                        static_cast<const uint16_t *>(bias);
                    int n = 0;
                    // Vectorized BF16→FP32: shift left by 16 bits
                    for (; n + 16 <= N; n += 16) {
                        __m256i bf = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(bias_bf16 + n));
                        __m512i wide = _mm512_cvtepu16_epi32(bf);
                        __m512i shifted = _mm512_slli_epi32(wide, 16);
                        _mm512_storeu_ps(s_bias_fp32 + n,
                            _mm512_castsi512_ps(shifted));
                    }
                    // Scalar tail
                    for (; n < N; ++n) {
                        uint32_t bits = static_cast<uint32_t>(bias_bf16[n]) << 16;
                        std::memcpy(&s_bias_fp32[n], &bits, sizeof(float));
                    }
                    s_last_bias_ptr = bias;
                    s_last_bias_N = N;
                }
            }
            bias_f = s_bias_fp32;
        } else {
            // FP32 bias → use directly
            bias_f = static_cast<const float *>(bias);
        }
    }

    // Fused post-op detection
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
        if (i != fused_idx)
            remaining_postops.push_back(params.postop_[i]);
    }
    const bool has_remaining_postops = !remaining_postops.empty();

    const int vnni_stride = NR_PACK * VNNI_PAIR;

    // FP32 output buffer (accumulation always in FP32)
    // If dst is BF16, we write FP32 here then convert at the end.
    // If dst is FP32, we write directly to dst.
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
        if (!s_c_buf) {
            commonlog_error("BF16 GEMM: failed to allocate FP32 C buffer");
            return;
        }
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

    // Select hot kernel based on planner MR and NR
    bf16_ukernel_fn_t hot_kernel = select_bf16_ukernel(MR, NR);

    // Fuse bias/activation only when alpha==1 (epilogue order is correct).
    // When alpha!=1, bias/activation must come AFTER scale_tile.
    const bool can_fuse_epilogue = (alpha == 1.0f);

    // Can we store BF16 directly from microkernel epilogue?
    // Requires: BF16 output + alpha==1 (correct epilogue order) + no unfused postops
    const bool can_direct_bf16 = dst_is_bf16 && can_fuse_epilogue
                                 && !has_remaining_postops;
    // BF16 output pointer (only used when can_direct_bf16)
    uint16_t *C_bf16_dst = dst_is_bf16
        ? static_cast<uint16_t *>(dst) : nullptr;

    // A-packing: auto-enable when row stride exceeds L1 prefetcher limit
    const bool do_pack_a = (lda * static_cast<int>(sizeof(uint16_t)) > 4096);

    // Thread loop: parallel over MC x NC tiles
    if (num_threads <= 1) {
        // Single-threaded path
        for (int pc = 0; pc < K; pc += KB) {
            const int kb_act = std::min(KB, K - pc);
            const int pc_pair = pc / 2;
            const bool is_last_k = (pc + KB >= K);
            const float beta_k = (pc == 0) ? beta : 1.0f;
            const bool can_fuse = is_last_k && can_fuse_epilogue;

            for (int jc = 0; jc < N; jc += NB) {
                const int nb_act = std::min(NB, N - jc);

                for (int ic = 0; ic < M; ic += MB) {
                    const int mb_act = std::min(MB, M - ic);
                    const int m_panels = (mb_act + MR - 1) / MR;

                    for (int jr = 0; jr < nb_act; jr += NR) {
                        const int nr_act = std::min(NR, nb_act - jr);
                        const int col = jc + jr;
                        const bool full_nr = (nr_act == NR);

                        const fused_postop_t tile_fop =
                            can_fuse ? fused_op : fused_postop_t::none;
                        const float *tile_bias =
                            (has_bias && can_fuse) ? (bias_f + col) : nullptr;

                        // Resolve B pointer once per NR-tile (outside M-panel loop)
                        const uint16_t *pb;
                        int pb_stride;
                        if (b_prepacked) {
                            int panel_idx = col / NR_PACK;
                            int in_panel_off = col % NR_PACK;
                            pb = prepacked_b->get_panel(pc_pair, panel_idx)
                                 + in_panel_off * VNNI_PAIR;
                            pb_stride = vnni_stride;
                        } else {
                            static thread_local uint16_t *s_otf = nullptr;
                            static thread_local size_t s_otf_cap = 0;
                            int kb_pad = (kb_act + 1) & ~1;
                            size_t need = static_cast<size_t>(
                                kb_pad / 2) * NR_PACK * VNNI_PAIR;
                            if (s_otf_cap < need) {
                                std::free(s_otf);
                                s_otf = static_cast<uint16_t *>(
                                    std::aligned_alloc(64,
                                        ((need * sizeof(uint16_t) + 63)
                                         & ~size_t(63))));
                                s_otf_cap = s_otf ? need : 0;
                            }
                            if (!s_otf) continue;
                            pack_b_vnni_strip(B_raw, ldb_raw, transB_raw,
                                col, std::min(static_cast<int>(NR_PACK),
                                              N - col),
                                K, K_padded_raw, pc, kb_act, s_otf);
                            pb = s_otf;
                            pb_stride = vnni_stride;
                        }

                        for (int ip = 0; ip < m_panels; ++ip) {
                            const int ir = ip * MR;
                            const int mr_act = std::min(MR, mb_act - ir);
                            const uint16_t *At = A + (ic + ir) * lda + pc;
                            float *Ct = C_fp32 + (ic + ir) * ldc_fp32 + col;

                            uint16_t *tile_bf16 = nullptr;
                            int tile_ldc_bf16 = 0;
                            if (can_direct_bf16 && is_last_k) {
                                tile_bf16 = C_bf16_dst
                                    + (ic + ir) * ldc + col;
                                tile_ldc_bf16 = ldc;
                            }

                            if (hot_kernel && full_nr && mr_act == MR) {
                                hot_kernel(
                                    At, lda, pb, pb_stride,
                                    Ct, ldc_fp32, kb_act, beta_k,
                                    tile_bias, tile_fop,
                                    tile_bf16, tile_ldc_bf16);
                            } else if (full_nr && mr_act >= 1 && mr_act <= 4) {
                                auto tail_uk = select_bf16_ukernel(mr_act, NR);
                                if (tail_uk) {
                                    tail_uk(
                                        At, lda, pb, pb_stride,
                                        Ct, ldc_fp32, kb_act, beta_k,
                                        tile_bias, tile_fop,
                                        tile_bf16, tile_ldc_bf16);
                                } else {
                                    bf16_tail_kernel(
                                        At, lda, pb, pb_stride,
                                        Ct, ldc_fp32, kb_act,
                                        mr_act, nr_act, beta_k,
                                        tile_bias, tile_fop,
                                        tile_bf16, tile_ldc_bf16);
                                }
                            } else {
                                bf16_tail_kernel(
                                    At, lda, pb, pb_stride,
                                    Ct, ldc_fp32, kb_act,
                                    mr_act, nr_act, beta_k,
                                    tile_bias, tile_fop,
                                    tile_bf16, tile_ldc_bf16);
                            }
                        }
                    }
                }
            }

            // Post-tile: alpha scaling, then bias/postops if not fused
            if (is_last_k) {
                if (alpha != 1.0f) {
                    scale_tile(C_fp32, ldc_fp32, M, N, alpha);
                    // Bias and fused postop deferred to here
                    if (has_bias)
                        apply_postops_tile(C_fp32, ldc_fp32, M, N,
                                           0, 0, bias_f, {});
                    if (fused_op != fused_postop_t::none && fused_idx >= 0)
                        apply_postops_tile(C_fp32, ldc_fp32, M, N,
                                           0, 0, nullptr,
                                           {params.postop_[fused_idx]});
                }
                if (has_remaining_postops) {
                    apply_postops_tile(C_fp32, ldc_fp32, M, N,
                                       0, 0, nullptr, remaining_postops);
                }
            }
        }
    } else {
        // BRGEMM-style multi-threaded: tile-outer, K-inner.
        // Each thread owns disjoint M×N tiles and processes ALL K-blocks
        // for its tiles before moving on. This eliminates inter-K-block
        // barriers and keeps C data hot in L1/registers across K-blocks.
        const int ic_tiles = (M + MB - 1) / MB;
        const int jc_tiles = (N + NB - 1) / NB;
        const int total_2d_tiles = ic_tiles * jc_tiles;
        const int active_threads = std::min(num_threads, total_2d_tiles);

        // Skip A-packing for small M when total A fits in L2.
        // Avoids N_threads redundant copies of the same small A matrix.
        const bool skip_pack_a = do_pack_a
            && (static_cast<size_t>(M) * K * sizeof(uint16_t)
                <= static_cast<size_t>(uarch.l2_bytes));
        const bool actual_pack_a = do_pack_a && !skip_pack_a;

        #pragma omp parallel num_threads(active_threads)
        {
            static thread_local uint16_t *tl_pa = nullptr;
            static thread_local size_t tl_pa_cap = 0;

            #pragma omp for schedule(static)
            for (int tid = 0; tid < total_2d_tiles; ++tid) {
                const int ic_idx = tid / jc_tiles;
                const int jc_idx = tid % jc_tiles;
                const int ic = ic_idx * MB;
                const int jc = jc_idx * NB;
                const int mb_act = std::min(MB, M - ic);
                const int nb_act = std::min(NB, N - jc);
                const int m_panels = (mb_act + MR - 1) / MR;

                // BRGEMM K-loop: accumulate ALL K-blocks for this tile
                for (int pc = 0; pc < K; pc += KB) {
                    const int kb_act = std::min(KB, K - pc);
                    const int pc_pair = pc / 2;
                    const bool is_last_k = (pc + KB >= K);
                    const float beta_k = (pc == 0) ? beta : 1.0f;
                    const bool can_fuse = is_last_k && can_fuse_epilogue;

                    // A source for this K-block
                    const uint16_t *a_base;
                    int at_stride_base;
                    if (actual_pack_a) {
                        size_t need = static_cast<size_t>(
                            ((mb_act + MR - 1) / MR) * MR) * kb_act;
                        if (tl_pa_cap < need) {
                            std::free(tl_pa);
                            tl_pa = static_cast<uint16_t *>(std::aligned_alloc(
                                64, ((need * sizeof(uint16_t) + 63) & ~size_t(63))));
                            tl_pa_cap = tl_pa ? need : 0;
                        }
                        if (tl_pa) {
                            pack_a_bf16_block(A, tl_pa, ic, pc, M, K, lda,
                                              mb_act, kb_act, MR);
                            a_base = tl_pa;
                            at_stride_base = kb_act;
                        } else {
                            a_base = A + ic * lda + pc;
                            at_stride_base = lda;
                        }
                    } else {
                        a_base = A + ic * lda + pc;
                        at_stride_base = lda;
                    }

                    for (int jr = 0; jr < nb_act; jr += NR) {
                        const int nr_act = std::min(NR, nb_act - jr);
                        const int col = jc + jr;
                        const bool full_nr = (nr_act == NR);

                        const fused_postop_t tile_fop =
                            can_fuse ? fused_op : fused_postop_t::none;
                        const float *tile_bias =
                            (has_bias && can_fuse) ? (bias_f + col) : nullptr;

                        // Resolve B once per NR-tile (outside M-panel loop)
                        const uint16_t *pb;
                        int pb_stride;
                        if (b_prepacked) {
                            int panel_idx = col / NR_PACK;
                            int in_panel_off = col % NR_PACK;
                            pb = prepacked_b->get_panel(pc_pair, panel_idx)
                                 + in_panel_off * VNNI_PAIR;
                            pb_stride = vnni_stride;
                        } else {
                            static thread_local uint16_t *tl_otf = nullptr;
                            static thread_local size_t tl_otf_cap = 0;
                            int kb_pad = (kb_act + 1) & ~1;
                            size_t need = static_cast<size_t>(
                                kb_pad / 2) * NR_PACK * VNNI_PAIR;
                            if (tl_otf_cap < need) {
                                std::free(tl_otf);
                                tl_otf = static_cast<uint16_t *>(
                                    std::aligned_alloc(64,
                                        ((need * sizeof(uint16_t) + 63)
                                         & ~size_t(63))));
                                tl_otf_cap = tl_otf ? need : 0;
                            }
                            if (!tl_otf) continue;
                            pack_b_vnni_strip(B_raw, ldb_raw,
                                transB_raw, col,
                                std::min(static_cast<int>(NR_PACK),
                                         N - col),
                                K, K_padded_raw, pc, kb_act, tl_otf);
                            pb = tl_otf;
                            pb_stride = vnni_stride;
                        }

                        for (int ip = 0; ip < m_panels; ++ip) {
                            const int ir = ip * MR;
                            const int mr_act = std::min(MR, mb_act - ir);
                            float *Ct = C_fp32 + (ic + ir) * ldc_fp32 + col;

                            const uint16_t *At;
                            int at_stride;
                            if (actual_pack_a) {
                                At = a_base + ip * MR * kb_act;
                                at_stride = kb_act;
                            } else {
                                At = a_base + ir * at_stride_base;
                                at_stride = at_stride_base;
                            }

                            uint16_t *tile_bf16 = nullptr;
                            int tile_ldc_bf16 = 0;
                            if (can_direct_bf16 && is_last_k) {
                                tile_bf16 = C_bf16_dst
                                    + (ic + ir) * ldc + col;
                                tile_ldc_bf16 = ldc;
                            }

                            if (hot_kernel && full_nr && mr_act == MR) {
                                hot_kernel(
                                    At, at_stride, pb, pb_stride,
                                    Ct, ldc_fp32, kb_act, beta_k,
                                    tile_bias, tile_fop,
                                    tile_bf16, tile_ldc_bf16);
                            } else if (full_nr && mr_act >= 1 && mr_act <= 4) {
                                auto tail_uk = select_bf16_ukernel(mr_act, NR);
                                if (tail_uk) {
                                    tail_uk(
                                        At, at_stride, pb, pb_stride,
                                        Ct, ldc_fp32, kb_act, beta_k,
                                        tile_bias, tile_fop,
                                        tile_bf16, tile_ldc_bf16);
                                } else {
                                    bf16_tail_kernel(
                                        At, at_stride, pb, pb_stride,
                                        Ct, ldc_fp32, kb_act,
                                        mr_act, nr_act, beta_k,
                                        tile_bias, tile_fop,
                                        tile_bf16, tile_ldc_bf16);
                                }
                            } else {
                                bf16_tail_kernel(
                                    At, at_stride, pb, pb_stride,
                                    Ct, ldc_fp32, kb_act,
                                    mr_act, nr_act, beta_k,
                                    tile_bias, tile_fop,
                                    tile_bf16, tile_ldc_bf16);
                            }
                        }
                    }

                    // Per-tile epilogue after last K-block
                    if (is_last_k) {
                        float *Ctile = C_fp32 + ic * ldc_fp32 + jc;
                        if (alpha != 1.0f) {
                            scale_tile(Ctile, ldc_fp32, mb_act, nb_act, alpha);
                            if (has_bias)
                                apply_postops_tile(Ctile, ldc_fp32, mb_act, nb_act,
                                                   jc, ic, bias_f, {});
                            if (fused_op != fused_postop_t::none && fused_idx >= 0)
                                apply_postops_tile(Ctile, ldc_fp32, mb_act, nb_act,
                                                   jc, ic, nullptr,
                                                   {params.postop_[fused_idx]});
                        }
                        if (has_remaining_postops) {
                            apply_postops_tile(Ctile, ldc_fp32, mb_act, nb_act,
                                               jc, ic, nullptr, remaining_postops);
                        }
                    }
                } // K-loop (BRGEMM batch-reduce)
            } // omp for
        } // omp parallel
    }

    // Convert FP32 accumulator to BF16 output if not already fused
    if (need_fp32_buf && !can_direct_bf16 && s_c_buf) {
        uint16_t *C_bf16 = static_cast<uint16_t *>(dst);
        convert_fp32_to_bf16_tile(C_fp32, ldc_fp32, C_bf16, ldc, M, N);
    }
}

// ============================================================================
// BF16 GEMM entry point (Planner + Looper + Kernel)
// ============================================================================
void bf16_gemm_execute(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int K = desc.K, N = desc.N;
    const bool transB = desc.transB;
    const bool is_weights_const = desc.is_weights_const;

    // Layer 1: Planner
    BF16GemmPlan bp = plan_bf16_gemm(desc, uarch, params);

    if (apilog_info_enabled()) {
        apilog_info("AI BF16 GEMM plan: M=", desc.M, " N=", N, " K=", K,
                    " MB=", bp.plan.MB, " NB=", bp.plan.NB, " KB=", bp.plan.KB,
                    " MR=", bp.plan.MR, " NR=", bp.plan.NR,
                    " NV=", bp.plan.NR / 16,
                    " path=", bp.path_name,
                    " threads=", bp.plan.num_threads);
    }

    // Weight caching (VNNI prepack)
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();
    const BF16PrepackedWeight *prepacked_b = nullptr;
    const bool can_cache = is_weights_const && (s_weight_cache != 0);

    if (can_cache) {
        PrepackedWeightKey bk{weight, K, N, desc.ldb, transB};
        prepacked_b = BF16PrepackedWeightCache::instance().get_or_prepack(
            bk, static_cast<const uint16_t *>(weight));
    }

    // Layer 2: Looper (tile loops + threading)
    bf16_thread_loop(desc, bp.plan, uarch, src, weight, dst, bias, params,
                     prepacked_b);
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
