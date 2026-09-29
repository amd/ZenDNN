/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef _LOWOHA_MATMUL_HPP
#define _LOWOHA_MATMUL_HPP

#include <cmath>
#include <cstring>
#include <vector>

#include "common/zendnnl_api.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/routed_moe/routed_moe.hpp"
#include "operators/matmul/matmul_context.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief Entry function for different backends supported by ZenDNNL
 */
void matmul_kernel_wrapper(char layout, char transA, char transB, int M, int N,
        int K, float alpha, const void *A, int lda, const void *B, int ldb,
        float beta, void *C, int ldc, matmul_data_types &dtypes,
        zendnnl::common::matmul_algo_t &kernel, char mem_format_a,
        char mem_format_b, matmul_params &lowoha_param,
        matmul_batch_params_t &batch_params, const void *bias,
        bool is_weights_const, int num_threads);

/**
 * @brief Execute single Matrix Multiplication (Matmul) for batch_count == 1
 *
 * This function handles all single matrix multiplication scenarios including:
 * - Auto-tuner based kernel selection
 * - LIBXSMM blocked execution with tiling
 * - BRGEMM kernel execution
 * - Standard matmul kernel execution
 */
void matmul_execute(const char layout, const bool transA, const bool transB,
        const int M, const int N, const int K, const float alpha,
        const void *src, const int lda, const void *weight, const int ldb,
        const void *bias, const float beta, void *dst, const int ldc,
        const bool is_weights_const, const size_t src_type_size,
        const size_t out_type_size, const int num_threads,
        matmul_algo_t &kernel, matmul_params &params,
        matmul_batch_params_t &batch_params, unsigned int auto_version);

/**
 * @brief Execute matrix multiplication with automatic kernel selection and optimization
 *
 * This function performs C = alpha * op(A) * op(B) + beta * C + fused post-ops.
 *
 * @param layout           Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA           Whether to transpose matrix A
 * @param transB           Whether to transpose matrix B
 * @param M                Number of rows in A and C
 * @param N                Number of columns in B and C
 * @param K                Number of columns in A and rows in B
 * @param alpha            Scaling factor for A*B
 * @param src              Pointer to matrix A data
 * @param lda              Leading dimension of A
 * @param weight           Pointer to matrix B data
 * @param ldb              Leading dimension of B
 * @param bias             Optional bias vector (can be nullptr)
 * @param beta             Scaling factor for existing C values
 * @param dst              Pointer to matrix C data
 * @param ldc              Leading dimension of C
 * @param is_weights_const Whether the weights are constant (enables caching)
 * @param batch_params  Batch sizes and optional batch strides (read-only).
 * @param params        Matmul configuration (read-only). The library
 *     builds an internal working copy for reorder dispatch, kernel
 *     selection, and post-op normalization so the caller's struct can be
 *     reused across calls without reset.
 *
 * @note Thread safety: This entry point does not mutate the @c matmul_params or
 *       @c matmul_batch_params_t objects themselves. Callers may share them
 *       concurrently only if any writable buffers referenced by them (e.g.,
 *       @c quant_params.src_scale.buff / @c quant_params.src_zp.buff) are
 *       thread-local, or left nullptr so the library allocates per-call scratch.
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */

ZENDNNL_API status_t matmul_direct(const char layout, const bool transA,
        const bool transB, const int M, const int N, const int K,
        const float alpha, const void *src, const int lda, const void *weight,
        const int ldb, const void *bias, const float beta, void *dst,
        const int ldc, const bool is_weights_const,
        const matmul_batch_params_t &batch_params, const matmul_params &params);

/**
 * @brief Execute group matmul operations (e.g. MoE experts)
 *
 * This function performs multiple independent matrix multiplications in sequence.
 * Each operation computes: C[i] = alpha[i] * op(A[i]) * op(B[i]) + beta[i] * C[i] + fused post-ops
 *
 * @param layout           Vector of memory layouts ('r' for row-major, 'c' for column-major)
 * @param transA           Vector of transpose flags for matrix A
 * @param transB           Vector of transpose flags for matrix B
 * @param M                Vector of row counts for A and C
 * @param N                Vector of column counts for B and C
 * @param K                Vector of column counts for A and row counts for B
 * @param alpha            Vector of scaling factors for A*B
 * @param src              Vector of pointers to matrix A data
 * @param lda              Vector of leading dimensions for A
 * @param weight           Vector of pointers to matrix B data
 * @param ldb              Vector of leading dimensions for B
 * @param bias             Vector of optional bias pointers (can contain nullptr)
 * @param beta             Vector of scaling factors for existing C values
 * @param dst              Vector of pointers to matrix C data
 * @param ldc              Vector of leading dimensions for C
 * @param is_weights_const Vector of flags indicating if weights are constant (enables caching)
 * @param params           Vector of per-expert configuration (read-only at the
 *                         API boundary). `group_matmul_direct` copies into
 *                         internal `exec_params` before reorder dispatch and
 *                         kernel execution so the same vector can be reused
 *                         across calls. For MoE workloads the caller may also set the optional
 *                         prepack-extras hint on `params[0]`:
 *                           - `params[0].active_matmul` = number of firing experts
 *                             (must satisfy `active_matmul <= M.size()`).
 *                           - `params[0].total_matmul`  = total expert weight slots
 *                             carried in the call (`>= active_matmul`; `0` means
 *                             "no prepack-extras tail").
 *                         Two input-side sizing patterns are both accepted:
 *                           (a) Compact:  `M.size() == active_matmul`  — input
 *                               vectors carry only the firing experts.
 *                           (b) Padded:   `M.size() == total_matmul`  with
 *                               `M[active_matmul..total_matmul) == 0` placeholders
 *                               — the dispatcher skips the zero-M slots.
 *                         Weight-side sizing depends on whether the caller
 *                         supplies a prepack-extras tail:
 *                           (i)  No tail (`total_matmul == 0` OR
 *                                `total_matmul == active_matmul`): all per-
 *                                expert vectors — weight-side
 *                                (`weight`, `K`, `N`, `ldb`, `transB`,
 *                                `is_weights_const`) and the rest (alpha,
 *                                bias, beta, ldc, params, ...) — need only
 *                                be `>= active_matmul`.
 *                           (ii) With tail (`total_matmul > active_matmul`,
 *                                the rotating-experts MoE case): the six
 *                                weight-side prepack vectors above must be
 *                                `>= total_matmul` so the prepack module
 *                                can warm every advertised expert without
 *                                silent truncation.  All other vectors still
 *                                need only `>= active_matmul`.  Sizes
 *                                shorter than `total_matmul` on the six
 *                                weight-side vectors are rejected by the
 *                                dispatcher up front (no silent
 *                                under-warming).
 *                         The dispatcher always computes only the first
 *                         `active_matmul` GEMMs.  When
 *                         `ZENDNNL_GRP_MATMUL_PREPACK=1` (the default), it
 *                         eagerly pre-warms the weight cache for ALL
 *                         `total_matmul` experts so any future firing hits
 *                         a warm cache.  Leave both fields at `0` for the
 *                         legacy contract: every per-expert vector must be
 *                         exactly `num_ops = M.size()` long and every
 *                         supplied weight fires.  See
 *                         `docs/operator/low_overhead_operator/lowoha_group_matmul_operator.md`
 *                         (Framework prepack-extras contract) for a worked
 *                         example.
 * @param moe_postop       Optional MoE weighted-reduce over pre-gathered expert rows;
 *                         nullptr disables (default). Parallel mode only; see
 *                         group_matmul_moe_postop_params.
 * @param gated_act        Optional gated activation applied in-place after GEMM
 *                         and before moe_postop: dst[:, 0:dim] = act(gate) * up
 *                         where dim = N/2. Requires N even and dst dtype
 *                         f32, bf16, or f16. nullptr disables (default).
 *                         Parallel mode only;
 *                         see grp_matmul_gated_act_params.
 * @param fused_moe        Optional fused MoE parameters describing the full
 *                         Op1 (gate+up) → activation → Op2 (down_proj) block
 *                         in a single call. V1 executes this flow as two
 *                         passes for every GRP_ALGO: Pass 1 runs Op1 plus
 *                         the gated activation via the parallel dispatcher
 *                         (honoring ZENDNNL_GRP_MATMUL_ALGO), Pass 2 runs
 *                         Op2 via the same dispatcher. When moe_postop is
 *                         also provided, the weighted reduce runs afterward
 *                         in its own pass. Deep single-pass Op1→Act→Op2
 *                         chaining is a future optimization.
 *                         nullptr disables (default).
 *
 * @return status_t::success if all operations succeed, status_t::failure if any operation fails
 */
ZENDNNL_API status_t group_matmul_direct(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params,
        const group_matmul_moe_postop_params *moe_postop = nullptr,
        const grp_matmul_gated_act_params *gated_act = nullptr,
        const grp_matmul_fused_moe_params *fused_moe = nullptr);

/**
 * @brief Clear AOCL DLP matmul weight caches (typed LRUs, symquant, woq,
 *        w4a8, GGML unpack, zero-point compensation).
 *
 * No-op when ZenDNNL is built without AOCL-DLP (`ZENDNNL_DEPENDS_AOCLDLP=0`).
 * GGML unpack is cleared only as part of that AOCL-DLP implementation; an
 * AOCL-off build does not drop the GGML LRU from this wrapper.
 *
 * @note Call only during a quiescent window: no in-flight matmul/group_matmul
 *       and no thread holding cached weight pointers. Not safe inside an OMP
 *       parallel region.
 */
ZENDNNL_API void clear_matmul_aocl_weight_caches();

/**
 * @brief Clear the calling thread's AOCL DLP post-op metadata cache.
 *
 * The cache is per-thread and keyed by weight pointer. After this call,
 * subsequent AOCL post-op setup on this thread rebuilds metadata until
 * the cache is repopulated.
 *
 * No-op when ZenDNNL is built without AOCL-DLP (`ZENDNNL_DEPENDS_AOCLDLP=0`).
 *
 * @note Same quiescent-window contract as @ref clear_matmul_aocl_weight_caches.
 *       Only the calling thread is cleared.
 */
ZENDNNL_API void clear_matmul_aocl_postop_metadata_cache();

/**
 * @brief Clear oneDNN blocked matmul weight cache.
 *
 * No-op when ZenDNNL is built without oneDNN (`ZENDNNL_DEPENDS_ONEDNN=0`).
 *
 * @note Same quiescent-window contract as @ref clear_matmul_aocl_weight_caches.
 */
ZENDNNL_API void clear_matmul_onednn_weight_caches();

/**
 * @brief Clear native prepacked weight caches (FP32/BF16/INT8 native paths).
 *
 * @note Same quiescent-window contract as @ref clear_matmul_aocl_weight_caches.
 */
ZENDNNL_API void clear_matmul_native_weight_caches();

/**
 * @brief Clear grp_matmul packed weight caches.
 *
 * Drops the process-wide, pointer-keyed pack arenas used by
 * @c matmul_algo_t::moe_custom_kernel: BF16, DQ-INT8, FP16, and W4A8
 * (S4). Also drops the N-tile f32 weight-scale memo that the INT8
 * custom-kernel path keys by scale-buffer pointer.
 *
 * Under @c ZENDNNL_MATMUL_WEIGHT_CACHE=2, eligible BF16 and INT8
 * calls store an in-place sentinel so a later call reuses the caller's buffer
 * as the packed weight. Clearing removes that sentinel, so the next call
 * repacks from the bytes currently in the buffer. Call this after
 * replacing or freeing those weights, in the same quiescent window as
 * the other clears — not between inferences that still rely on an
 * in-place pack already written into the buffer.
 *
 * Also drops the prepack fingerprint and AUTO mixed-in-place warm
 * latches, so the next @ref group_matmul_direct repacks instead of
 * treating the cleared arenas as already warm. SwiGLU-OAI depends on
 * that re-warm: its ALGO 3 path records a fingerprint for both the
 * custom-kernel pack and the AOCL per-tile reorder.
 *
 * Does not clear routed-MoE packed weights
 * (@ref group_matmul_routed_moe_flush_weight_cache) or LibXSMM
 * blocked weights.
 *
 * @note Same quiescent-window contract as @ref clear_matmul_aocl_weight_caches.
 */
ZENDNNL_API void clear_grp_matmul_weight_caches();

/**
 * @brief Clear AOCL, oneDNN, native, and grp_matmul weight caches,
 *        plus the calling thread's AOCL post-op metadata cache.
 *
 * Preferred entry point when releasing cached reordered weights on
 * model unload, swap, or heap-address reuse. Covers the caches used by
 * @ref matmul_direct and the custom-kernel packs used by
 * @ref group_matmul_direct (including fused FFN / MoE paths that
 * dispatch @c matmul_algo_t::moe_custom_kernel).
 *
 * Under @c ZENDNNL_MATMUL_WEIGHT_CACHE=2 this also removes in-place
 * sentinels. Call only after replacing or releasing those weights; reusing
 * an already-packed buffer after this call would repack the packed bytes.
 *
 * Also drops the prepack fingerprint via
 * @ref clear_grp_matmul_weight_caches. Does not clear routed-MoE
 * packed weights (@ref group_matmul_routed_moe_flush_weight_cache)
 * or LibXSMM blocked weights.
 *
 * @note Same quiescent-window contract as @ref clear_matmul_aocl_weight_caches.
 */
ZENDNNL_API void clear_matmul_weight_caches();

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
