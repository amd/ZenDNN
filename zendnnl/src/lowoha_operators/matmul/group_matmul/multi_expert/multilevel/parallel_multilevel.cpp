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
 *******************************************************************************/

/// ALGO 6: CCD-aware multilevel scheduling.

#include "lowoha_operators/matmul/group_matmul/multi_expert/multilevel/parallel_multilevel.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <omp.h>

#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_expert_slice.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ck_enable.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace detail {

// ── ALGO=6: multilevel — CCD-aware adaptive scheduling ──────────────────
//
// (A) Few experts, large M: multi-CCD per expert, all concurrent.
// (B) Few experts + small M, or many experts: round-based, 1 CCD each.
// Uses nested OMP (scoped_active_levels(2)).

void parallel_multilevel(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
        const char **gemm_mode_out) {

    // The executor owns its gemm_mode and writes the concrete regime it ran
    // ("multilevel_concurrent" vs "multilevel_rounds") so the post-exec
    // [GRP_MATMUL.CALL] line reflects the real path.  No-op when nullptr.
    auto set_ml_mode = [&](const char *s) {
        if (gemm_mode_out != nullptr) { *gemm_mode_out = s; }
    };
    // Default to SKIP so a no-op early return (empty call / num_threads<=0)
    // reports exec_algo=0 rather than a real ALGO-6 run; the two regime
    // branches below overwrite it with the executed path.
    set_ml_mode("multilevel_skip");

    const int num_ops = static_cast<int>(M.size());
    if (num_ops == 0 || num_threads <= 0) { return; }

    // NOTE: the all-inactive (every M<=0) call is short-circuited to a "skip"
    // mode by the dispatcher (group_matmul_run_parallel_dispatch) before this
    // executor is ever entered, so no per-regime all-inactive guard is needed
    // here.  The per-slot M<=0 guards below still cover the mixed case.

    // Generic ahead-of-time weight pre-pack for ALGO 6.
    // See sequential_experts above for the contract; identical short-
    // circuits, only the scheduling-algo tag differs.  `num_threads`
    // is forwarded so cross_warm can prefill regime 2 for the upcoming
    // ALGO 3 decode path when CUSTOM_KERNEL=0 (see the comment on the
    // ALGO 1 call site for the full rationale).
    group_matmul_prepack::prepack_for_algo_6(
            group_matmul_prepack::build_prepack_params(weight, K, N, ldb,
                    transB, is_weights_const, params, M,
                    grp_matmul_custom_kernel_effective(params, M, N, fused_act),
                    num_threads,
                    algo3_decode_nr_align(M, N, ldc, fused_act, params),
                    fused_act, act_dtype,
                    /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

    matmul_algo_t algo = resolve_kernel();

    const int ccd_size = std::min(8, num_threads);
    // Ceiling to match flat_m_tile / flat_n_tile: partial last CCD counts as one.
    const int num_ccds = std::max(1, (num_threads + ccd_size - 1) / ccd_size);
    const int max_M = *std::max_element(M.begin(), M.end());

    if (num_ops <= num_ccds && max_M >= ccd_size) {
        // (A) Few experts, large M: multi-CCD per expert, all concurrent.
        set_ml_mode("multilevel_concurrent");
        int64_t total_M = 0;
        for (int i = 0; i < num_ops; ++i) {
            total_M += M[i];
        }
        if (total_M <= 0) { total_M = num_ops; }

        std::vector<int> ccds_per_op(num_ops, 1);
        int remaining = num_ccds - num_ops;
        if (remaining > 0) {
            for (int i = 0; i < num_ops; ++i) {
                int extra = static_cast<int>(
                        static_cast<int64_t>(remaining) * M[i] / total_M);
                ccds_per_op[i] += extra;
            }
            int used = 0;
            for (int i = 0; i < num_ops; ++i) {
                used += ccds_per_op[i];
            }
            for (int i = 0; used < num_ccds; ++i, ++used) {
                ccds_per_op[i % num_ops]++;
            }
        }
        std::vector<int> thr_per_op(num_ops);
        for (int i = 0; i < num_ops; ++i) {
            thr_per_op[i] = ccds_per_op[i] * ccd_size;
        }

        scoped_active_levels guard(2);
        bool short_team_a = false;
#pragma omp parallel num_threads(num_ops)
        {
            // A requested team size is a request, not a guarantee: OMP_DYNAMIC,
            // OMP_THREAD_LIMIT or an already-active outer level can all trim
            // it.  This regime maps expert := tid, so a short team never
            // reaches the high experts and leaves their dst holding whatever
            // it held, with the call still reporting success.  ALGO 3's
            // DecodeD/Rounds fail closed on exactly this; match them rather
            // than letting one deployment get a hard error on decode-shaped
            // calls and silent garbage on prompt-shaped ones.
            if (omp_get_num_threads() < num_ops) {
#pragma omp single
                { short_team_a = true; }
            } else {
                const int i = omp_get_thread_num();
                // Inactive experts (M<=0) are padded placeholder slots that may carry
                // null src/dst/weight pointers; skip them so the slice/activation
                // calls never dereference null (matches the M==0 guards in the
                // sequential / m-tile / n-tile executors).
                if (i < num_ops && M[i] > 0) {
                    execute_expert_slice(layout[i], transA[i], transB[i], M[i],
                            N[i], K[i], alpha[i], src[i], lda[i], weight[i],
                            ldb[i], bias[i], beta[i], dst[i], ldc[i],
                            is_weights_const[i], thr_per_op[i], params[i],
                            resolve_expert_kernel(6, algo, params[i]));
                    if (fused_act != grp_matmul_gated_act_t::none) {
                        // The GEMM above ran on `thr_per_op[i]` nested threads;
                        // leaving the activation at the single-thread default
                        // idled that whole sub-team for the M[i] x N[i] pass.
                        apply_gated_act_inplace(fused_act, dst[i], 0, M[i],
                                N[i], ldc[i], act_dtype, thr_per_op[i]);
                    }
                }
            }
        }
        if (short_team_a) {
            set_ml_mode("error_multilevel_short_team");
            return;
        }
    } else {
        // (B) Round-based, 1 CCD per expert.
        set_ml_mode("multilevel_rounds");
        const int batch = std::min(num_ops, num_ccds);

        scoped_active_levels guard(2);
        bool short_team_b = false;
        for (int round_start = 0; round_start < num_ops; round_start += batch) {
            const int round_end = std::min(num_ops, round_start + batch);
            const int round_size = round_end - round_start;

#pragma omp parallel num_threads(round_size)
            {
                // Same contract as regime A: expert := round_start + tid, so a
                // trimmed team silently skips the tail of every round.
                if (omp_get_num_threads() < round_size) {
#pragma omp single
                    { short_team_b = true; }
                } else {
                    const int slot = omp_get_thread_num();
                    // Inactive experts (M<=0) are padded placeholder slots that may carry
                    // null src/dst/weight pointers; skip them so the slice/activation
                    // calls never dereference null (matches the M==0 guards in the
                    // sequential / m-tile / n-tile executors).
                    if (slot < round_size && M[round_start + slot] > 0) {
                        const int e = round_start + slot;
                        execute_expert_slice(layout[e], transA[e], transB[e],
                                M[e], N[e], K[e], alpha[e], src[e], lda[e],
                                weight[e], ldb[e], bias[e], beta[e], dst[e],
                                ldc[e], is_weights_const[e], ccd_size,
                                params[e],
                                resolve_expert_kernel(6, algo, params[e]));
                        if (fused_act != grp_matmul_gated_act_t::none) {
                            apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                    N[e], ldc[e], act_dtype);
                        }
                    }
                }
            }
            if (short_team_b) { break; }
        }
        if (short_team_b) {
            set_ml_mode("error_multilevel_short_team");
            return;
        }
    }
    return;
}

} // namespace detail

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
