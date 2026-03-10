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

#include "lowoha_operators/matmul/matmul_native/gemm/planner/fp32_gemm_plan.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "operators/matmul/matmul_config.hpp"

#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::ops::post_op_type_t;

FP32GemmPlan plan_fp32_gemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;

    // Classify activation post-ops for NR selection
    [[maybe_unused]] bool has_activation = false;
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

        // KB: minimize K-blocks while keeping B panel in L2
        if (desc.num_threads <= 1) {
            int b_bytes = K * NR_PACK * static_cast<int>(sizeof(float));
            if (b_bytes <= uarch.l2_bytes / 2)
                plan.KB = K;
        } else {
            int kb_b = (uarch.l2_bytes / 2)
                       / (NR_PACK * static_cast<int>(sizeof(float)));
            int kb_a = (uarch.l2_bytes / 2)
                       / (6 * static_cast<int>(sizeof(float)));
            int kb_max_mt = std::min(kb_a, kb_b);
            if (K <= kb_max_mt) {
                plan.KB = K;
            } else if (kb_max_mt > plan.KB) {
                int n_blocks = (K + kb_max_mt - 1) / kb_max_mt;
                plan.KB = ((K + n_blocks - 1) / n_blocks + 7) & ~7;
            }
        }

        s_plan_cache = {M, N, K, desc.num_threads, desc.transA, desc.transB, plan};
    }

    // FP32 MR/NR selection
    plan.MR = 6;
    if (N >= 64 && !has_complex_activation) {
        plan.NR = 64;
    } else if (N >= 32) {
        plan.NR = 32;
    } else {
        plan.NR = 16;
    }

    // Re-align NB to final NR
    plan.NB = std::max(plan.NB / plan.NR * plan.NR, plan.NR);
    plan.NB = std::min(plan.NB, N);

    // Recompute MB: L2-aware when A is packed
    {
        bool will_pack_a = desc.transA
                           || (desc.lda * static_cast<int>(sizeof(float)) > 4096);
        if (will_pack_a) {
            int b_lines_per_krow = ((plan.NR + 15) / 16) * 64;
            int b_accessed_bytes = plan.KB * b_lines_per_krow;
            int l2_for_a = std::max(uarch.l2_bytes - b_accessed_bytes, 0);
            plan.MB = std::max(l2_for_a / (plan.KB * 4), plan.MR);
        } else {
            plan.MB = std::max((uarch.l1d_bytes + uarch.l2_bytes)
                               / (plan.NB * 4 + plan.KB * 4), plan.MR);
        }
        plan.MB = plan.MB / plan.MR * plan.MR;
        plan.MB = std::min(plan.MB, M);
    }

    // Load-balance for multi-threaded
    if (plan.num_threads > 1) {
        int jc_tiles = (N + plan.NB - 1) / plan.NB;
        int ic_tiles = (M + plan.MB - 1) / plan.MB;
        int needed_ic = (plan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            int m_panels = (M + plan.MR - 1) / plan.MR;
            int panels_per_block = std::max(m_panels / needed_ic, 1);
            plan.MB = panels_per_block * plan.MR;
            plan.MB = std::min(plan.MB, M);
        }
    }

    return FP32GemmPlan{plan};
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
