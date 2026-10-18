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

/// Per-expert primitives: kernel resolution, `execute_expert_slice` and
/// the per-thread quantization buffer pools.

#ifndef ZENDNNL_GRP_MATMUL_EXPERT_SLICE_HPP
#define ZENDNNL_GRP_MATMUL_EXPERT_SLICE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "common/op_config.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// ──────────────────────────────────────────────────────────────────────
// Shared per-expert primitives.
// ──────────────────────────────────────────────────────────────────────

/// Resolves the effective matmul algo ID from the runtime config,
/// falling back to AOCL DLP blocked (or reference when AOCL is absent)
/// when the config is unset/invalid.
inline matmul_algo_t resolve_kernel() {
    static const matmul_algo_t algo = []() {
        int32_t a = matmul_config_t::instance().get_algo();
        matmul_algo_t kernel;
        if (a <= 0 || a >= static_cast<int32_t>(matmul_algo_t::algo_count)) {
#if ZENDNNL_DEPENDS_AOCLDLP
            kernel = matmul_algo_t::aocl_dlp_blocked;
#else
            kernel = matmul_algo_t::reference;
#endif
        } else {
            kernel = static_cast<matmul_algo_t>(a);
        }
#if !ZENDNNL_DEPENDS_AOCLDLP
        if (kernel == matmul_algo_t::aocl_dlp
                || kernel == matmul_algo_t::aocl_dlp_blocked
                || kernel == matmul_algo_t::batched_sgemm) {
            kernel = matmul_algo_t::reference;
        }
#endif
        return kernel;
    }();
    return algo;
}

/// W4A8 matmul policy:
///   * group AUTO: ALGO 3 is native s4; every other scheduler is simulated;
///   * pinned group ALGO: preserve the inner-kernel choice.
inline matmul_algo_t w4a8_runtime_algo(
        int scheduling_algo, matmul_algo_t inner_kernel) {
    if (get_grp_matmul_algo() == kGrpMatmulAlgoAuto) {
        return scheduling_algo == 3 ? matmul_algo_t::aocl_dlp_blocked
                                    : matmul_algo_t::aocl_dlp;
    }
    return inner_kernel == matmul_algo_t::aocl_dlp_blocked
            ? matmul_algo_t::aocl_dlp_blocked
            : matmul_algo_t::aocl_dlp;
}

/// Resolve the effective inner kernel for one expert under the selected
/// group-matmul scheduler.  Most dtype families preserve `inner_kernel`;
/// scheduler-specific policies (currently W4A8 native-vs-simulated routing)
/// are centralized here so individual executors do not encode dtype policy.
inline matmul_algo_t resolve_expert_kernel(int scheduling_algo,
        matmul_algo_t inner_kernel, const matmul_params &params) {
    // A reordered weight with AOCL (or legacy unset) provenance must remain
    // on AOCL even when the process-wide backend resolves differently.
    // Reinterpreting those blocked bytes as oneDNN/libxsmm/raw is invalid.
    if (params.mem_format_b == 'r'
            && (params.lowoha_algo == matmul_algo_t::none
                    || params.lowoha_algo == matmul_algo_t::aocl_dlp_blocked)) {
        return matmul_algo_t::aocl_dlp_blocked;
    }
    return is_w4a8_config(params)
            ? w4a8_runtime_algo(scheduling_algo, inner_kernel)
            : inner_kernel;
}

/// The per-thread dynamic-quant buffer pool `execute_expert_slice` quantizes
/// through.  Pooled because the per-expert executors (ALGO 1 / 5 / 6) reach it
/// once per expert inside the parallel region, where a stack instance would
/// cost a malloc/free pair each time.  Named rather than function-local so
/// `reset_thread_local_fused_moe_state()` can `release()` it, as with
/// `ntile_flat_parallel::reset_thread_local_scratch()`.
inline reorder_quant_buffers_t &get_thread_local_quant_buffers() {
    static thread_local reorder_quant_buffers_t buffers;
    return buffers;
}

/// The per-thread pools the grouped source quantization fills on the calling
/// thread, one per call site: the two fused-MoE passes and the unfused
/// grouped call.  Named for the same reason as
/// `get_thread_local_quant_buffers`.
enum class group_quant_pool_t : int {
    fused_pass1,
    fused_pass2,
    unfused,
    count
};

inline group_reorder_quant_buffers_t &get_thread_local_group_quant_buffers(
        group_quant_pool_t pool) {
    static thread_local group_reorder_quant_buffers_t
            pools[static_cast<int>(group_quant_pool_t::count)];
    return pools[static_cast<int>(pool)];
}

/// Status-bearing form of `execute_expert_slice`, for callers that can
/// propagate a failure instead of only logging it.  The per-expert source
/// quantization can fail at RUNTIME (a pooled allocation, not just bad
/// metadata a gate could screen), and on that path the destination is left
/// unwritten — so a caller that reports success regardless would publish a
/// stale intermediate.  `matmul_execute` returns void, so the quantization
/// is the only stage that can report anything.
///
/// `execute_expert_slice` remains the void form for the callers that have no
/// way to carry a status out (ALGO 1/6 and the N-tile executors run this
/// inside parallel regions whose contracts are unchanged).
inline status_t execute_expert_slice_checked(char layout, bool transA,
        bool transB, int M, int N, int K, float alpha, const void *src, int lda,
        const void *weight, int ldb, const void *bias, float beta, void *dst,
        int ldc, bool is_weights_const, int num_thr, matmul_params &params,
        matmul_algo_t algo) {

    // Inactive expert (no routed tokens): nothing to compute.  Returning
    // early keeps the per-expert ALGOs (1/5/6) from driving the backend
    // GEMM with M == 0, where tile-count math divides by the row count and
    // traps (SIGFPE).  The M-tile / N-tile ALGOs flatten over rows and skip
    // empty experts implicitly, so this is the only path that needs the
    // guard.  Sparse MoE routing (e.g. 6 of 15 experts firing) relies on it.
    if (M <= 0) { return status_t::success; }

    matmul_batch_params_t bp;
    bp.Batch_A = 1;
    bp.Batch_B = 1;
    matmul_algo_t kernel = algo;

    // Per-expert dynamic-quant fallback path.  This fires only when the
    // source is still bf16/f32 and `params.dynamic_quant == true` (the
    // wrapper's own eligibility gate).  When the caller already ran the
    // grouped `group_dynamic_quant` pre-pass (ZENDNNL_ENABLE_GROUP_DQ on),
    // it rewrote `params.dtypes.src` to s8 and cleared `dynamic_quant`, so
    // this wrapper short-circuits to a no-op and no double-quant occurs.
    // When grouped DQ is disabled (env off) this is the active per-expert
    // quantization path, preserving the legacy behaviour.
    int reordered_lda = lda;
    size_t src_type_size = size_of(params.dtypes.src);
    // Pooled per thread, not constructed per expert: each worker owns its
    // experts and consumes the buffer within one call, so the grow-only
    // reuse is safe.
    reorder_quant_buffers_t &quant_buffers = get_thread_local_quant_buffers();
    if (reorder_quantization_wrapper(src, lda, reordered_lda, src_type_size,
                params, bp, transA, M, K, num_thr, quant_buffers)
            != status_t::success) {
        log_error("execute_expert_slice: reorder_quantization_wrapper failed");
        return status_t::failure;
    }

    matmul_execute(layout, transA, transB, M, N, K, alpha, src,
            params.dynamic_quant ? reordered_lda : lda, weight, ldb, bias, beta,
            dst, ldc, is_weights_const, src_type_size,
            size_of(params.dtypes.dst), num_thr, kernel, params, bp, 0);
    return status_t::success;
}

/// Thin wrapper around matmul_execute that packages per-expert slice
/// arguments into the batch/params objects the kernel expects.
inline void execute_expert_slice(char layout, bool transA, bool transB, int M,
        int N, int K, float alpha, const void *src, int lda, const void *weight,
        int ldb, const void *bias, float beta, void *dst, int ldc,
        bool is_weights_const, int num_thr, matmul_params &params,
        matmul_algo_t algo) {
    // Failure is already logged by the checked form; discarding it here keeps
    // the historical contract of this entry point.
    (void)execute_expert_slice_checked(layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc, is_weights_const,
            num_thr, params, algo);
}

/// ZENDNNL_ENABLE_GROUP_DQ — opt-in/out for the grouped dynamic-quant
/// pre-pass (`group_dynamic_quant`).  Default ON (unset or any value
/// other than a literal "0").  When OFF, group matmul skips the grouped
/// source-quant pre-pass and dynamic quantization falls back to the
/// per-expert `reorder_quantization_wrapper` inside `execute_expert_slice`
/// (legacy path).
///
/// The env read is latched: this sits on the serial critical path once per
/// fused-MoE call, and an int8 MoE decode call is only a few microseconds
/// against a `getenv` that scans the whole environment block.  Latching makes
/// a later `setenv` invisible, so tests use
/// `s_grp_matmul_enable_group_dq_override` (sentinel -1 = no override).
inline bool get_grp_matmul_enable_group_dq() {
    const int ovr = test_api::s_grp_matmul_enable_group_dq_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *env = std::getenv("ZENDNNL_ENABLE_GROUP_DQ");
        if (env == nullptr || env[0] == '\0') return true; // default ON
        return !(env[0] == '0' && env[1] == '\0'); // "0" => OFF
    }();
    return v;
}

// NOTE: The N-tile shared utilities — `sort_indices_by_m`,
// `engage_ntile_custom_kernel`, `ntile_effective_nr_align`,
// `auto_pick_n_order`, `fill_ntile_expert_order` — moved to
// `group_matmul_n_tile.hpp` (Section A.5) so they live alongside
// the N-tile env knobs they read.  Include that header to call
// them — currently only `group_matmul_n_tile.cpp` does.

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GRP_MATMUL_EXPERT_SLICE_HPP
