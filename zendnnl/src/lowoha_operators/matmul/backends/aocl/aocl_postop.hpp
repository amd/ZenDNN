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
#include "lowoha_operators/matmul/lowoha_quant_layout.hpp"

#if ZENDNNL_DEPENDS_AOCLDLP
#include "aocl_dlp.h"
#endif

#include <cmath>
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

// ── Counts and source scales of the AOCL-DLP sym-quant kernel ───────────

/**
 * @brief Element count of quant dims as the AOCL-DLP metadata states it.
 *
 * One for empty dims (per-tensor), else the product of the dims, unchecked:
 * a non-positive or overflowing dimension wraps.
 */
inline size_t get_num_elements(const std::vector<int64_t> &dims) {
    if (dims.empty()) { return 1; }
    size_t count = 1;
    for (auto d : dims) {
        count *= static_cast<size_t>(d);
    }
    return count;
}

/**
 * @brief A per-token source scale over a single token.
 *
 * At M == 1 a per-token scale has one element, so its count cannot tell it
 * from a per-tensor scale.  The weight scale's layout does: either 2-D form
 * ({G > 1, N} per-group or {1, N} per-channel) is one the sym-quant reorder
 * consumes, since `sym_quant_group_size` maps both to `K / dims[0]`.
 */
inline bool src_scale_is_collapsed_per_token(
        const matmul_quantization_params_t &quant_params, int M, int N) {
    return quant_params.src_scale.buff != nullptr
            && get_num_elements(quant_params.src_scale.dims) == 1 && M == 1
            && (has_per_group_wei_scale(quant_params, N)
                    || has_per_channel_wei_scale(quant_params, N));
}

/**
 * @brief A per-tensor source scale over several rows with {G > 1, N} weight
 *        scales.
 *
 * AOCL-DLP applies per-group weight scales only on the sym-quant kernel,
 * which reads one source value per row: the wiring repeats the one value
 * for every row.
 */
inline bool src_scale_is_broadcast_per_token(
        const matmul_quantization_params_t &quant_params, int M, int N) {
    return quant_params.src_scale.buff != nullptr
            && get_num_elements(quant_params.src_scale.dims) == 1 && M > 1
            && has_per_group_wei_scale(quant_params, N);
}

/**
 * @brief The source scale of an s8 source selects the AOCL-DLP sym-quant
 *        kernel: more than one element (per token or per group), a per-token
 *        scale collapsed to one token, or a per-tensor scale with per-group
 *        weight scales.
 *
 * src_count is `get_num_elements(quant_params.src_scale.dims)`.  The kernel
 * choice (aocl_kernel.cpp) and the scale wiring (aocl_postop.cpp) both decide
 * by this: a disagreement applies the scales twice or not at all.
 */
inline bool src_scale_selects_sym_quant(
        const matmul_quantization_params_t &quant_params, size_t src_count,
        int M, int N) {
    return quant_count_granularity(src_count, M)
            != quant_count_granularity_t::per_tensor
            || src_scale_is_collapsed_per_token(quant_params, M, N)
            || src_scale_is_broadcast_per_token(quant_params, M, N);
}

/**
 * @brief The AOCL-DLP route applies the weight scale of an s8-weight call.
 *
 * One value, or one value per column, it applies on any wiring, in the
 * types aocl_dlp_applies_wei_scale_type admits.  Per-group weight scales it
 * applies only on the s8 x s8 sym-quant kernel (no source zero point, f32 /
 * bf16 destination, src_scale_selects_sym_quant), in groups of whole quads
 * with K / group values per column: the SCALE post-op refuses any other
 * length, the sym-quant reorder any other group, and either leaves the
 * destination unwritten.
 *
 * @param src The source type AOCL-DLP multiplies: the compute type of a
 *            source the call quantizes.
 */
inline bool aocl_dlp_applies_wei_scale(
        const matmul_params &params, data_type_t src, int M, int N, int K) {
    const auto &qp = params.quant_params;
    if (params.dtypes.wei != data_type_t::s8 || qp.wei_scale.buff == nullptr)
        return true;
    const size_t count = get_num_elements(qp.wei_scale.dims);
    if (count == 1 || count == static_cast<size_t>(N)) return true;
    if (src != data_type_t::s8 || qp.src_zp.buff != nullptr
            || (params.dtypes.dst != data_type_t::f32
                    && params.dtypes.dst != data_type_t::bf16)
            || !src_scale_selects_sym_quant(
                    qp, get_num_elements(qp.src_scale.dims), M, N))
        return false;
    const int64_t group = sym_quant_group_size(qp, M, N, K);
    return quant_group_size_is_whole_quads(K, group)
            && count == static_cast<size_t>(K / group) * static_cast<size_t>(N);
}

/**
 * @brief The AOCL-DLP sym-quant kernel reads these weight scales: f32 ones,
 *        or ones of the source scale's type; it leaves the destination
 *        unwritten with others.
 */
inline bool sym_quant_reads_wei_scale(
        const matmul_quantization_params_t &quant_params) {
    return quant_params.wei_scale.dt == data_type_t::f32
            || quant_params.wei_scale.dt == quant_params.src_scale.dt;
}

/**
 * @brief The AOCL-DLP route reads the weight scale of an s8-weight call in
 *        its type: the s8 x s8 sym-quant kernel (no source zero point, f32 /
 *        bf16 destination, src_scale_selects_sym_quant) only as
 *        sym_quant_reads_wei_scale says; the SCALE post-ops read the type
 *        they are given.
 *
 * @param src As for aocl_dlp_applies_wei_scale.
 */
inline bool aocl_dlp_applies_wei_scale_type(
        const matmul_params &params, data_type_t src, int M, int N) {
    const auto &qp = params.quant_params;
    if (params.dtypes.wei != data_type_t::s8 || qp.wei_scale.buff == nullptr
            || sym_quant_reads_wei_scale(qp))
        return true;
    return src != data_type_t::s8 || qp.src_zp.buff != nullptr
            || (params.dtypes.dst != data_type_t::f32
                    && params.dtypes.dst != data_type_t::bf16)
            || !src_scale_selects_sym_quant(
                    qp, get_num_elements(qp.src_scale.dims), M, N);
}

/// v is an integer that int32 holds.
inline bool holds_int32(float v) {
    return v >= -2147483648.0f && v < 2147483648.0f && v == std::trunc(v);
}

/**
 * @brief AOCL-DLP's GEMMs of an s8 weight, and of an s4 weight with a u8 /
 *        s8 source, take the call's alpha and beta: those GEMMs sum in int32
 *        and take alpha and beta as int32, so both must be integers that
 *        int32 holds.
 *
 * @param src The source type AOCL-DLP multiplies: the compute type of a
 *            source the call quantizes.
 */
inline bool aocl_dlp_takes_alpha_beta(
        const matmul_params &params, data_type_t src, float alpha, float beta) {
    if (alpha == 1.0f && beta == 0.0f) return true;
    const data_type_t wei = params.dtypes.wei;
    const bool int_src = src == data_type_t::u8 || src == data_type_t::s8;
    if (wei != data_type_t::s8 && !(wei == data_type_t::s4 && int_src))
        return true;
    return holds_int32(alpha) && holds_int32(beta);
}

/**
 * @brief The AOCL-DLP route applies the alpha and beta of a call, which it
 *        takes (aocl_dlp_takes_alpha_beta), as matmul_direct defines them.
 *
 * beta * C of an integer GEMM goes after the scales for per-token or
 * per-group source scales and for per-group weight scales, else into the
 * integer sum before them.  The sym-quant kernels add it after the scales,
 * also for the one source value at M == 1 they take with per-channel weight
 * scales; the kernels of a u8 / s8 source and one source scale value add it
 * to the int32 sum, with C converted to int32 (rounded when it is f32, bf16
 * or f16), so only an integer C (an s32, s8 or u8 destination) is added as
 * it is; those of a bf16 / f32 source refuse a nonzero beta with any source
 * scale and leave the destination unwritten.
 *
 * @param src As for aocl_dlp_takes_alpha_beta.
 */
inline bool aocl_dlp_applies_alpha_beta(const matmul_params &params,
        data_type_t src, int M, int N, float alpha, float beta) {
    if (!aocl_dlp_takes_alpha_beta(params, src, alpha, beta)) return false;
    const data_type_t dst = params.dtypes.dst;
    if (beta == 0.0f || params.dtypes.wei != data_type_t::s8) return true;
    if (src != data_type_t::u8 && src != data_type_t::s8) return false;
    const auto &qp = params.quant_params;
    const size_t src_count = get_num_elements(qp.src_scale.dims);
    if (src == data_type_t::s8 && qp.src_zp.buff == nullptr
            && (dst == data_type_t::f32 || dst == data_type_t::bf16)
            && src_scale_selects_sym_quant(qp, src_count, M, N))
        return src_count > 1 || quant_param_is_per_token(qp.src_scale, M)
                || has_per_group_wei_scale(qp, N);
    return dst == data_type_t::s32 || dst == data_type_t::s8
            || dst == data_type_t::u8;
}

#if ZENDNNL_DEPENDS_AOCLDLP
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
        bool is_weights_const, const int32_t *reorder_colsum = nullptr,
        int32_t neg_src_zp = 0,
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
#endif

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
