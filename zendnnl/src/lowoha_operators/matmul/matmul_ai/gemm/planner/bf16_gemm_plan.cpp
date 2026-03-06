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
// BF16 GEMM Plan: MR/NR selection, KB adjustments, decode path, load balancing.
// Extracted from avx512_bf16_gemm.cpp bf16_gemm_execute() lines 1340-1525.
//

#include "lowoha_operators/matmul/matmul_ai/gemm/planner/bf16_gemm_plan.hpp"
#include "operators/matmul/matmul_config.hpp"

#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

using zendnnl::ops::post_op_type_t;

BF16GemmPlan plan_bf16_gemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const int K_padded = (K + 1) & ~1;

    // Classify activation post-ops for NR selection
    bool has_activation = false;
    bool has_complex_activation = false;
    for (const auto &po : params.postop_) {
        auto pt = po.po_type;
        if (pt == post_op_type_t::relu || pt == post_op_type_t::leaky_relu) {
            has_activation = true;
        } else if (pt == post_op_type_t::gelu_tanh ||
                   pt == post_op_type_t::gelu_erf ||
                   pt == post_op_type_t::sigmoid ||
                   pt == post_op_type_t::tanh ||
                   pt == post_op_type_t::swish ||
                   pt == post_op_type_t::elu) {
            has_activation = true;
            has_complex_activation = true;
            break;
        }
    }

    // Thread-local plan cache
    static thread_local struct {
        int M, N, K, threads; bool transA, transB;
        BlockPlan plan;
    } s_plan_cache = {0, 0, 0, 0, false, false, {}};

    BlockPlan plan;
    if (s_plan_cache.M == M && s_plan_cache.N == N && s_plan_cache.K == K &&
        s_plan_cache.threads == desc.num_threads &&
        s_plan_cache.transA == desc.transA &&
        s_plan_cache.transB == desc.transB) {
        plan = s_plan_cache.plan;
    } else {
        plan = plan_blocks(desc, uarch);
        plan.KB = (plan.KB + 1) & ~1;

        // BF16 KB: minimize K-blocks to reduce synchronization
        {
            int b_full_k_bytes = K_padded * NR_PACK
                                 * static_cast<int>(sizeof(uint16_t));
            if (desc.num_threads <= 1) {
                int a_panel_bytes = 6 * K_padded
                                    * static_cast<int>(sizeof(uint16_t));
                int l1_limit = static_cast<int>(0.8 * uarch.l1d_bytes);
                if (b_full_k_bytes <= uarch.l2_bytes / 2
                    && a_panel_bytes <= l1_limit) {
                    plan.KB = K_padded;
                }
            } else {
                int l2_for_a = uarch.l2_bytes / 2;
                int kb_a = l2_for_a / (6 * static_cast<int>(sizeof(uint16_t)));
                int kb_b = (uarch.l2_bytes / 2)
                           / (NR_PACK * static_cast<int>(sizeof(uint16_t)));
                int kb_max_mt = std::min(kb_a, kb_b);
                kb_max_mt = (kb_max_mt + 1) & ~1;
                if (K_padded <= kb_max_mt) {
                    plan.KB = K_padded;
                } else if (kb_max_mt > plan.KB) {
                    int n_blocks = (K_padded + kb_max_mt - 1) / kb_max_mt;
                    int even_kb = ((K_padded + n_blocks - 1) / n_blocks + 7) & ~7;
                    plan.KB = (even_kb + 1) & ~1;
                }
            }
        }

        s_plan_cache = {M, N, K, desc.num_threads, desc.transA, desc.transB, plan};
    }

    // BF16 MR/NR selection (M-aware + post-op aware)
    const bool is_decode = (M <= 4);
    const char *path_name = "gemm";

    if (is_decode) {
        plan.MR = M;
        plan.MB = M;
        if (N >= 64)       plan.NR = 64;
        else if (N >= 32)  plan.NR = 32;
        else               plan.NR = 16;
        path_name = "decode";
    } else if (N >= 64 && !has_complex_activation) {
        if (M % 6 == 0 || M >= 18) {
            plan.MR = 6;
        } else if (M % 4 == 0) {
            plan.MR = 4;
        } else if (M % 6 <= 3 && M > 12) {
            plan.MR = 4;
        } else {
            plan.MR = 6;
        }
        plan.NR = 64;
        path_name = "gemm-nr64";
    } else if (N >= 32) {
        plan.MR = (M % 6 == 0 || M >= 18) ? 6
                : (M % 4 == 0) ? 4 : 6;
        plan.NR = 32;
    } else {
        plan.MR = (M % 6 == 0 || M >= 18) ? 6
                : (M % 4 == 0) ? 4 : 6;
        plan.NR = 16;
    }

    // Re-align NB to final NR
    plan.NB = std::max(plan.NB / plan.NR * plan.NR, plan.NR);
    plan.NB = std::min(plan.NB, N);

    // Recompute MB: L2-aware when A is packed
    if (!is_decode) {
        bool will_pack_a = (desc.lda * static_cast<int>(sizeof(uint16_t)) > 4096);
        if (will_pack_a) {
            int b_lines_per_krow = ((plan.NR + 15) / 16) * 64;
            int k_pairs_kb = (plan.KB + 1) / 2;
            int b_accessed_bytes = k_pairs_kb * b_lines_per_krow;
            int l2_for_a = std::max(uarch.l2_bytes - b_accessed_bytes, 0);
            plan.MB = std::max(l2_for_a / (plan.KB * 2), plan.MR);
        } else {
            plan.MB = std::max((uarch.l1d_bytes + uarch.l2_bytes)
                               / (plan.NB * 4 + plan.KB * 2), plan.MR);
        }
        plan.MB = plan.MB / plan.MR * plan.MR;
        plan.MB = std::min(plan.MB, M);
    }

    // Load-balance for multi-threaded
    if (!is_decode && plan.num_threads > 1) {
        int jc_tiles = (N + plan.NB - 1) / plan.NB;
        int ic_tiles = (M + plan.MB - 1) / plan.MB;

        int needed_ic = (plan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            int m_panels = (M + plan.MR - 1) / plan.MR;
            int panels_per_block = std::max(m_panels / needed_ic, 1);
            plan.MB = panels_per_block * plan.MR;
            plan.MB = std::min(plan.MB, M);
            ic_tiles = (M + plan.MB - 1) / plan.MB;
        }

        if (ic_tiles * jc_tiles < plan.num_threads && plan.NB > plan.NR) {
            int needed_jc = (plan.num_threads + ic_tiles - 1) / ic_tiles;
            int nb_target = (N + needed_jc - 1) / needed_jc;
            nb_target = std::max((nb_target / plan.NR) * plan.NR, plan.NR);
            plan.NB = std::min(nb_target, plan.NB);
        }
    }

    return BF16GemmPlan{plan, path_name, is_decode,
                        has_activation, has_complex_activation};
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
