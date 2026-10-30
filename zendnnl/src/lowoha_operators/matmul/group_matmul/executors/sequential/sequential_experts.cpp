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

/// ALGO 1: sequential experts, each on the full team.

#include "lowoha_operators/matmul/group_matmul/executors/sequential/sequential_experts.hpp"

#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_expert_slice.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ck_enable.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace detail {

// ── ALGO=1: sequential — serial over experts ────────────────────────────

void sequential_execute(const std::vector<char> &layout,
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
        const char **execution_path_out) {

    const size_t num_ops = M.size();
    if (num_ops == 0 || num_threads <= 0) { return; }

    // Generic ahead-of-time weight pre-pack for ALGO 1.  Idempotent:
    // short-circuits when `ZENDNNL_GRP_MATMUL_PREPACK=0` or when this
    // thread already warmed the same fingerprint (per-thread cache
    // covers process-lifetime calls of the same model/layer).  Under
    // the uniform-eager semantic, PREPACK=ON warms the firing experts
    // (legacy callers, `total = active = M.size()` after
    // `build_prepack_params`) AND the full prepack-extras pool when
    // the framework opted into `total > active`.  The module owns its
    // own AOCL DLP backend gating via `resolve_matmul_backend()`.
    //
    // `num_threads` is forwarded so `cross_warm` inside prepack.cpp can
    // compute `stable = aocl_stable_n_thr(num_threads, max_N)` and
    // prefill regime 2 (per-tile AOCL) for the upcoming ALGO 3 decode
    // path when CUSTOM_KERNEL=0.  Without it, that branch silently
    // drops to a no-op and decode pays a one-time first-call reorder
    // cost.
    //
    // `nr_align` must be the value decode will actually split on, not a
    // constant: the per-tile cache key embeds `n_tile = aligned_n_split(
    // N, n_thr, ..., nr_align)`, so warming with a different alignment
    // prefills keys the runtime never queries and every tile misses on
    // the first decode call.  Cross-warm used to hardcode 1, which is
    // correct only for the wide arena; a tight fused swiglu splits on 2.
    // The prompt call has `ldc` and `N` in hand, so resolve it here
    // through the same helper decode uses.
    group_matmul_prepack::prepack_sequential(
            group_matmul_prepack::build_prepack_params(weight, K, N, ldb,
                    transB, is_weights_const, params, M,
                    group_matmul_custom_kernel_candidate(
                            params, M, N, fused_act),
                    num_threads,
                    n_tile_decode_nr_align(M, N, ldc, fused_act, params),
                    fused_act, act_dtype,
                    /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

    matmul_algo_t algo = resolve_matmul_backend();

    for (size_t i = 0; i < num_ops; ++i) {
        // Inactive experts (M<=0) are padded placeholder slots that may carry
        // null src/dst/weight pointers (validate_group_matmul_direct_inputs
        // allows null for M==0 because dispatch is expected to short-circuit
        // empty rows); skip them so the slice/activation calls never
        // dereference null.
        if (M[i] <= 0) { continue; }
        execute_expert_slice(layout[i], transA[i], transB[i], M[i], N[i], K[i],
                alpha[i], src[i], lda[i], weight[i], ldb[i], bias[i], beta[i],
                dst[i], ldc[i], is_weights_const[i], num_threads, params[i],
                resolve_expert_backend(1, algo, params[i]));
    }
    // Activation is deferred out of the expert loop rather than applied per
    // expert while dst[i] is still hot.
    //
    // `apply_gated_act_inplace` parallelises over an expert's ROWS, and at
    // decode an expert holds one to a few dozen of them, so a 32-thread
    // region was being opened per expert to run a handful of iterations --
    // once per expert, per layer, per token. The fork and barrier dominated
    // work measured in microseconds, and most threads never got a row.
    //
    // `group_matmul_moe_act_execute` walks the row space flattened across
    // every expert instead, so one region covers all of them and each thread
    // gets a real share. The locality that deferring gives up is small: the
    // whole activated region is num_active * M * N and stays L3-resident at
    // decode sizes.
    if (fused_act != grp_matmul_gated_act_t::none) {
        grp_matmul_gated_act_params act_params {};
        act_params.act = fused_act;
        if (group_matmul_moe_act_execute(
                    &act_params, dst, M, N, ldc, act_dtype, num_threads)
                != status_t::success) {
            // Validation is all-or-nothing across experts, so a refusal here
            // means NOTHING was activated. `sequential_execute` is void, so
            // signal through the execution_path sentinel the caller already
            // matches by `error_` prefix.
            if (execution_path_out != nullptr) {
                *execution_path_out = "error_sequential_act_refused";
            }
            return;
        }
    }
    return;
}

} // namespace detail

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
