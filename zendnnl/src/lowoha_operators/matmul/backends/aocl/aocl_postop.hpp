/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ZENDNNL_AOCL_POSTOP_HPP_
#define ZENDNNL_AOCL_POSTOP_HPP_

#include "lowoha_operators/matmul/lowoha_common.hpp"

#include "aocl_dlp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Shared scalar constants used as pointer-targets for AOCL/DLP post-op fields
inline constexpr float ONE_F32 = 1.0f;

// Cast a (const) float address into the non-const void* slot expected by the
// AOCL DLP API. The kernels treat these scalars as read-only inputs, so the
// const_cast does not introduce undefined behavior and the underlying storage
// may safely live in read-only memory (e.g. inline constexpr globals).
inline void *get_void_ptr(const float &v) {
    return const_cast<void *>(static_cast<const void *>(&v));
}

// Helper function to compute number of elements from dimension vector.
// Returns 1 for empty dims (per-tensor case) specific for DLP use case,
// or product of all dims.
inline size_t get_num_elements(const std::vector<int64_t> &dims) {
    if (dims.empty()) { return 1; }
    size_t count = 1;
    for (auto d : dims) {
        count *= static_cast<size_t>(d);
    }
    return count;
}

// Collapsed per-token src at M==1: a per-token src scale over a single
// token has one element, so `src_scale_nelems > 1` cannot distinguish it
// from a per-tensor scale.  The weight-scale shape disambiguates — either
// 2-D form ({G>1, N} per-group or {1, N} per-channel) is a scale layout
// the sym-quant reorder can consume, since `sym_quant_group_size` maps
// both to `K / dims[0]`.
//
// Per-channel was previously excluded, which silently dropped every
// single-token DA8W8 expert out of sym-quant: at M==1 the first disjunct
// of the sym-quant predicate is false and the {1, N} weight scale failed
// `has_per_group_wei_scale`'s `dims[0] > 1`.  Those calls reordered into
// the plain s8 blocked LRU instead, holding a second copy of every weight
// that no warm targets — the majority of experts at decode.
//
// Callers must stay in lockstep: this drives `is_s8_sym_quant_scales` in
// aocl_kernel.cpp (which reorder/GEMM runs) AND `is_sym_quant` in
// aocl_postop.cpp (whether the scales are consumed natively by the
// sym-quant GEMM or wired as post-op scale slots).  Disagreement applies
// the scales twice or not at all.
inline bool src_scale_is_collapsed_per_token(
        const matmul_quantization_params_t &quant_params, int M, int N) {
    return quant_params.src_scale.buff != nullptr
            && get_num_elements(quant_params.src_scale.dims) == 1 && M == 1
            && (has_per_group_wei_scale(quant_params, N)
                    || has_per_channel_wei_scale(quant_params, N));
}

/**
 * @brief Creates DLP metadata for post-operations
 *
 * This function initializes and returns a pointer to dlp_metadata_t that
 * encapsulates the post-operation metadata for matrix multiplication
 *
 * @param lowoha_param The parameters containing the post-operations chain specification
 * @param bias Pointer to the bias data to be added (can be nullptr if no bias)
 * @param dtypes Data types structure specifying source, weight, and destination tensor types
 * @param N The number of columns in the output matrix
 * @param K The number of columns in the input matrix / rows in weight matrix
 * @param M The number of rows in the output matrix (used for INT8 zero-point compensation)
 * @param zp_comp_acc Pointer to zero-point compensation buffer (nullptr if no ZP compensation)
 * @param zp_comp_ndim Dimensionality of ZP compensation: 0=none, 1=bias(N), 2=matrix(M*N)
 * @param kernel Algorithm selection for GEMM execution
 * @param weight_ptr Original (pre-reorder) weight buffer pointer (the GEMM
 *                   operand B as supplied by the caller, before any internal
 *                   reorder). Used as the identity component of the per-
 *                   layer cache key, so it must point at the same stable
 *                   buffer for every call that targets a given layer.
 *                   Required; must not be null. This is structurally
 *                   guaranteed by both call sites in aocl_kernel.cpp:
 *                   they pass the matmul B operand, which the AOCL DLP
 *                   kernels themselves dereference further down the same
 *                   wrapper -- a null B would have already crashed
 *                   upstream before reaching this function, so no
 *                   defensive null-check is added here.
 * @param w4a8_algo W4A8 AOCL algo (`none` / `aocl_dlp` / `aocl_dlp_blocked`);
 *                   included in the cache key.
 * @return Pointer to the dlp_metadata_t inside the per-layer holder
 *         (lifetime managed by the per-thread post-op metadata LRU),
 *         OR nullptr for layers that legitimately have no post-op
 *         metadata to wire. Two distinct nullptr-returning paths:
 *           (a) Plain matmul with no post-op chain, not WOQ, not INT8
 *               — re-dispatched on every call (cheap; no holder is
 *               allocated or cached).
 *           (b) BF16-INT8 with src_scale.buff == nullptr (logged
 *               misconfiguration) — a no_metadata-flagged holder is
 *               cached so subsequent calls on the same key short-
 *               circuit to nullptr without re-logging.
 *         Callers MUST tolerate nullptr and run the unfused GEMM
 *         path when it occurs.
 *
 * @throws zendnnl::exception_t If the per-layer holder allocation
 *         fails (std::calloc returns null). No in-function fallback
 *         exists because every subsequent wiring step assumes a live
 *         holder; callers that need a non-throwing path should catch
 *         the exception at the matmul boundary.
 *
 * @param reorder_colsum Optional prepacked per-column weight sums used for
 *        static-INT8 source zero-point compensation.
 * @param neg_src_zp Scale applied to reorder_colsum.
 */
dlp_metadata_t *create_dlp_post_op(const matmul_params &lowoha_param,
        const void *bias, const matmul_data_types &dtypes, int N, int K, int M,
        int32_t *zp_comp_acc, int zp_comp_ndim,
        zendnnl::common::matmul_algo_t kernel, const void *weight_ptr,
        const int32_t *reorder_colsum = nullptr, int32_t neg_src_zp = 0,
        zendnnl::common::matmul_algo_t w4a8_algo
        = zendnnl::common::matmul_algo_t::none);

/**
 * @brief Per-call teardown for the metadata returned by create_dlp_post_op().
 *
 * No-op for the common case — cached holders are owned by the per-thread
 * LRU cache, which frees them with std::free on eviction or thread exit.
 *
 * The non-trivial case is the BF16/INT8 per-token-symmetric quant path
 * (is_bf16_f32_per_token_sym in create_dlp_post_op): that path can't be
 * cached because its inverse-scale array has length = M (unknown at
 * compile time, varies per call) which doesn't fit the cache holder's
 * fixed layout. Such "per-call" holders are flagged inside the holder
 * struct; this function recovers the holder from the metadata pointer
 * (metadata is the first field of the holder), frees the heap-owned
 * inv_scales array, and frees the holder itself.
 *
 * Safe to call with nullptr (matches the no-metadata return contract).
 * MUST be called at every kernel call site that consumes the metadata
 * returned by create_dlp_post_op(), exactly once per create call,
 * AFTER the metadata's last use.
 *
 * @param metadata Pointer returned by create_dlp_post_op(); may be null.
 */
void cleanup_dlp_post_op(dlp_metadata_t *metadata);

/**
 * @brief Clears the calling thread's AOCL DLP post-op metadata cache.
 *
 * The post-op metadata cache is per-thread and indexed by Key_matmul
 * (weight_ptr + N + K + algo + postop_signature). After this call, every
 * subsequent create_dlp_post_op() invocation on this thread takes the
 * cold path (allocate + build + insert) until the cache is repopulated.
 *
 * Intended for use between gtest cases (each test is a fresh "model" with
 * new weight buffers, so the cached holders for the previous test become
 * unreachable). Not meant to be called on the matmul hot path.
 */
void clear_aocl_postop_metadata_cache();

/**
 * @brief Returns the number of holders currently in the calling
 *        thread's AOCL DLP post-op metadata cache.
 *
 * Intended for tests only — production code has no reason to inspect
 * the cache size. Provides an observable signal that is independent of
 * malloc address-reuse, so a test can assert that
 * clear_aocl_postop_metadata_cache() actually evicted the holder
 * (size==0 afterwards) rather than relying on the pointer of the next
 * allocation being different — which it often isn't, because glibc's
 * free-list happily hands a recently-freed slot back to the next
 * same-sized request.
 */
std::size_t get_aocl_postop_metadata_cache_size();

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif //ZENDNNL_AOCL_POSTOP_HPP_
