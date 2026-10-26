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

#ifndef MATMUL_NATIVE_NATIVE_MATMUL_HPP
#define MATMUL_NATIVE_NATIVE_MATMUL_HPP

#include "common/profiler.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// The layout-rule query of a matmul call: a bf16 or f32 source the call
/// quantizes dynamically to u8 / s8 counts as that type, and a {G > 1, N}
/// weight scale as per-group.
native_layout_query_t native_layout_query_for(
        const matmul_params &params, int K, int N, int ldb, bool transB);

/// The native BRGEMM looks at the call: under ZENDNNL_MATMUL_ALGO=11 every
/// call, whatever it selects (a buffer it rewrote in place has no other
/// reader), else a call with lowoha_algo == native_brgemm.
inline bool native_brgemm_selected(const matmul_params &params) {
    return params.lowoha_algo == matmul_algo_t::native_brgemm
            || matmul_config_t::is_native_brgemm_algo();
}

/// The call asks for the native BRGEMM as its kernel: lowoha_algo ==
/// native_brgemm, or no lowoha_algo under ZENDNNL_MATMUL_ALGO=11.
inline bool native_brgemm_requested(const matmul_params &params) {
    return params.lowoha_algo == matmul_algo_t::native_brgemm
            || (params.lowoha_algo == matmul_algo_t::none
                    && matmul_config_t::is_native_brgemm_algo());
}

/// Runs a matmul_direct call under ALGO 11 on the native BRGEMM over the
/// native VNNI layout when the call qualifies:
///   * its weight is a raw ('n') weight decide_native_layout places on the
///     layout, an aocl_dlp_blocked prepack (ZENDNNL_MATMUL_ALGO=11 only)
///     whose pack has an inverse, or a buffer an earlier native call
///     rewrote in place (whatever algorithm the call selects);
///   * it is a single row-major call with a plain source that uses only what
///     the native BRGEMM implements, checked before any work.
/// Returns false when the call does not qualify, with the weight and dst
/// untouched: the caller continues on its other routes.  Returns true
/// otherwise, with status the call's status.  A buffer rewritten in place
/// has no other reader, so a call that cannot read it fails, and so does a
/// call on a raw weight a scalar AOCL in-place reorder rewrote.  Weight
/// cache mode 2 rewrites a buffer in place only under
/// ZENDNNL_MATMUL_ALGO=11; with lowoha_algo == native_brgemm alone it caches
/// out of place.  A call it takes logs the line matmul_direct logs, batch
/// giving its batch fields (batch_count both when null).
bool native_layout_matmul(char layout, bool transA, bool transB, int M, int N,
        int K, float alpha, const void *src, int lda, const void *weight,
        int ldb, const void *bias, float beta, void *dst, int ldc,
        bool is_weights_const, int batch_count, int num_threads,
        matmul_gated_act_t gated_act, matmul_params &params,
        zendnnl::profile::profiler_t *profiler, status_t &status,
        const matmul_batch_params_t *batch = nullptr);

/// Public entry point for the Native matmul kernel (GEMM and BRGEMM).
///
/// Called from matmul_execute() when kernel == native_gemm, or native_brgemm
/// as dynamic dispatch or the auto tuner selects it.
/// Returns true if the operation was handled, false if the caller should
/// fall back to an alternative kernel (e.g., DLP blocked).
/// Shape-aware heuristics may return false for shapes where the Native kernel
/// is known to be slower than the fallback.
bool native_matmul_execute(matmul_algo_t kernel, char layout, bool transA,
        bool transB, int M, int N, int K, float alpha, const void *src, int lda,
        const void *weight, int ldb, const void *bias, float beta, void *dst,
        int ldc, bool is_weights_const, int num_threads, matmul_params &params);

/// Best-algo selector for BF16 M≤8 (GEMV/decode) shapes.
/// Returns the optimal algo (native_brgemm or aocl_dlp_blocked) based
/// on empirical benchmarks on AMD Zen 5.
///   * M>1            → native_brgemm (BRGEMM dominates DLP for M=2-8).
///   * M=1, !transB,
///     N ≤ 4          → native_brgemm (engages pack-free narrow GEMV).
///   * M=1, transB,
///     N ≤ 4          → falls through to the shape-based DLP gates
///                      (narrow kernel can't engage; BKC pad waste
///                      makes it lose to DLP).
///   * other M=1      → shape-based DLP gates for small-N high-K.
matmul_algo_t bf16_gemv_best_algo(
        int M, int N, int K, int num_threads, bool transB);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_NATIVE_MATMUL_HPP
