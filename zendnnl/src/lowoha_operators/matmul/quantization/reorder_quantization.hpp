/*******************************************************************************
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

#ifndef REORDER_QUANTIZATION_HPP
#define REORDER_QUANTIZATION_HPP

#include <cstdlib>
#include <vector>
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief RAII holder for buffers allocated during reorder quantization.
 *        Non-copyable, move-only; frees on destruction.
 *
 *        May be a POOLED per-thread instance (see
 *        `get_thread_local_quant_buffers()`), in which case destruction is at
 *        thread exit, not end of call: pointers the wrapper rebinds into it
 *        (the returned `src`, `src_scale.buff`, `src_zp.buff`) stay valid only
 *        until the same thread's next quantization.  `release()` frees early.
 */
struct reorder_quant_buffers_t {
    uint8_t *src_buf = nullptr;
    uint8_t *scale_buf = nullptr;
    uint8_t *zp_buf = nullptr;

    // Bytes currently held by each buffer above, which is what makes a POOLED
    // instance reusable.  The pool is reached once per expert per call from
    // inside a parallel region, and the naive `src_buf = malloc(n)` this
    // replaced overwrote a live pointer every time — leaking one block per
    // expert per call, unbounded on a long-lived server.  Tracking the
    // capacity lets `ensure_*` keep the existing block whenever it is already
    // big enough, so a steady-state call allocates nothing at all and the
    // buffer settles at the largest expert's size.
    //
    // A buffer without a capacity would silently reintroduce that leak, so all
    // three carry one.
    size_t src_cap = 0;
    size_t scale_cap = 0;
    size_t zp_cap = 0;

    /// Largest block a pooled buffer keeps once the caller stops needing it,
    /// and the ratio that decides "stops needing".  Retention has to be
    /// BOUNDED, not merely leak-free: a prompt call sizes `src_buf` at
    /// `max_expert_M * K`, and with one pool per worker thread an unbounded
    /// grow-only policy would hold that peak — hundreds of MB across a wide
    /// team — through every later decode token, for the life of the process.
    ///
    /// Shrinking below a generous hysteresis costs at most the malloc/free
    /// pair this pool exists to avoid, and only on a phase change rather than
    /// per call, so the worst case degrades to the pre-pool behaviour while
    /// capping resident memory.
    static constexpr size_t kPoolRetainFloorBytes = 1u << 20; // 1 MiB
    static constexpr size_t kPoolShrinkRatio = 8;

    /// Size `buf` to hold `bytes`, reusing the existing block when it already
    /// fits and is not disproportionately large.
    ///
    /// Allocates BEFORE freeing, so an OOM returns false with the previous
    /// buffer and capacity still valid and the caller can bail safely.
    static bool ensure_buf(uint8_t *&buf, size_t &cap, size_t bytes) {
        const bool fits = (bytes <= cap) && (buf != nullptr);
        const bool oversized = fits && cap > kPoolRetainFloorBytes
                && bytes < cap / kPoolShrinkRatio;
        if (fits && !oversized) { return true; }
        uint8_t *grown = static_cast<uint8_t *>(malloc(bytes));
        if (grown == nullptr) {
            // Keep the oversized block rather than failing the call: it still
            // satisfies `bytes`, and reclaiming memory is not worth an error.
            return fits;
        }
        free(buf);
        buf = grown;
        cap = bytes;
        return true;
    }
    bool ensure_src(size_t bytes) {
        return ensure_buf(src_buf, src_cap, bytes);
    }
    bool ensure_scale(size_t bytes) {
        return ensure_buf(scale_buf, scale_cap, bytes);
    }
    bool ensure_zp(size_t bytes) { return ensure_buf(zp_buf, zp_cap, bytes); }

    /// Free all three buffers and drop the capacities.  For pooled instances,
    /// which otherwise hold their largest-ever request until thread exit — on
    /// a server, a prompt-sized block held through every later decode.
    /// `ensure_*` re-grows afterwards, so this only costs one reallocation.
    void release() {
        free(src_buf);
        free(scale_buf);
        free(zp_buf);
        src_buf = nullptr;
        scale_buf = nullptr;
        zp_buf = nullptr;
        src_cap = 0;
        scale_cap = 0;
        zp_cap = 0;
    }

    reorder_quant_buffers_t() = default;
    ~reorder_quant_buffers_t() {
        free(src_buf);
        free(scale_buf);
        free(zp_buf);
    }

    reorder_quant_buffers_t(const reorder_quant_buffers_t &) = delete;
    reorder_quant_buffers_t &operator=(const reorder_quant_buffers_t &)
            = delete;
    // Capacities move with their buffers: a moved-from object must not claim
    // capacity it no longer owns, or its next `ensure_*` returns freed memory.
    reorder_quant_buffers_t(reorder_quant_buffers_t &&o) noexcept
        : src_buf(o.src_buf)
        , scale_buf(o.scale_buf)
        , zp_buf(o.zp_buf)
        , src_cap(o.src_cap)
        , scale_cap(o.scale_cap)
        , zp_cap(o.zp_cap) {
        o.src_buf = o.scale_buf = o.zp_buf = nullptr;
        o.src_cap = o.scale_cap = o.zp_cap = 0;
    }
    reorder_quant_buffers_t &operator=(reorder_quant_buffers_t &&o) noexcept {
        if (this != &o) {
            free(src_buf);
            free(scale_buf);
            free(zp_buf);
            src_buf = o.src_buf;
            scale_buf = o.scale_buf;
            zp_buf = o.zp_buf;
            src_cap = o.src_cap;
            scale_cap = o.scale_cap;
            zp_cap = o.zp_cap;
            o.src_buf = o.scale_buf = o.zp_buf = nullptr;
            o.src_cap = o.scale_cap = o.zp_cap = 0;
        }
        return *this;
    }
};

struct group_reorder_quant_buffers_t {
    // Per-expert NON-OWNING views into the arenas below (grouped path), or
    // unused (the per-expert fallback path owns its memory via fallback_buf).
    std::vector<uint8_t *> src_buf;
    std::vector<uint8_t *> scale_buf;
    std::vector<reorder_quant_buffers_t> fallback_buf;
    // Single backing allocations for the grouped path: one malloc for all
    // experts' s8 dst rows, one for the scale rows the caller did not
    // pre-allocate — instead of two mallocs PER expert.  `src_buf` /
    // `scale_buf` index into these and must not be freed individually.
    uint8_t *src_arena = nullptr;
    uint8_t *scale_arena = nullptr;

    group_reorder_quant_buffers_t() = default;
    ~group_reorder_quant_buffers_t() {
        free(src_arena);
        free(scale_arena);
    }

    group_reorder_quant_buffers_t(const group_reorder_quant_buffers_t &)
            = delete;
    group_reorder_quant_buffers_t &operator=(
            const group_reorder_quant_buffers_t &)
            = delete;
};

/**
 * @brief Attempt reorder quantization of the source tensor if eligible.
 *
 * Checks whether the dtype combination qualifies for reorder quantization
 * (src is BF16/F32, weight is S8, compute is S8/U8). If eligible, quantizes
 * the source via reorder and updates src/params in-place. If not eligible,
 * returns success with no changes.
 *
 * Also eligible: W4A8 (bf16 src, s4 wei, s8 compute) — same reorder entry;
 * per-token src_scale may be broadcast to per-group shape after quantize.
 *
 * Supports two modes based on params.dynamic_quant:
 * - Dynamic (true):  Computes scale (and zp for u8) on-the-fly.
 * - Static  (false): Uses user-provided scale (and zp for u8) values.
 *
 * Per-token symmetric dynamic quant (bf16/f32 src, s8 weights, s8 compute):
 * controlled by macro ZENDNNL_LOWOHA_DQ_BF16S8 in reorder_quantization.cpp
 * (1 = compute-only scales + bf16s8/f32s8 GEMM; 0 = quantize A to s8 + s8s8_sym_quant).
 * Other granularities still quantize the source to s8/u8.
 *
 * @param[in,out] src              Source data pointer; updated on success
 * @param[in]     lda              Leading dimension of original source
 * @param[in,out] reordered_lda    Updated to contiguous lda on success
 * @param[in,out] src_type_size    Updated to quantized element size on success
 * @param[in,out] params           dtypes.src and quant_params updated on success
 * @param[in,out] batch_params     batch_stride_src updated on success
 * @param[in]     transA           Whether source matrix is transposed
 * @param[in]     M                Number of rows
 * @param[in]     K                Shared (inner) dimension
 * @param[in]     num_threads      Thread count for the reorder operation
 * @param[out]    buffers          RAII holder, grown in place; may be a
 *                                 per-thread pooled instance, in which case
 *                                 the rebound pointers stay valid only until
 *                                 this thread's next call.
 *
 * @return status_t::success  Quantization performed, skipped (not eligible),
 *                            or reorder failed gracefully
 * @return status_t::failure  Validation failed — caller should propagate
 */
status_t reorder_quantization_wrapper(const void *&src, const int lda,
        int &reordered_lda, size_t &src_type_size, matmul_params &params,
        matmul_batch_params_t &batch_params, const bool transA, const int M,
        const int K, const int num_threads, reorder_quant_buffers_t &buffers);

/**
 * @brief Grouped dynamic-quant pre-pass for MoE / group matmul sources.
 *
 * Mutates the supplied @p params vector in place (dtypes, scale buffers,
 * dynamic_quant). The caller must pass a working copy (`exec_params`)
 * built by `group_matmul_direct` — user-owned config vectors are not
 * modified at the API boundary.
 */
status_t group_reorder_quantization_wrapper(
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<bool> &transA, const std::vector<int> &M,
        const std::vector<int> &K, const int num_threads,
        std::vector<matmul_params> &params,
        std::vector<const void *> &quantized_src,
        std::vector<int> &quantized_lda, group_reorder_quant_buffers_t &buffers,
        bool &quantized);

inline bool group_reorder_quantization_required(
        const std::vector<matmul_params> &params, size_t num_ops) {
    if (params.size() < num_ops) { return false; }
    for (size_t i = 0; i < num_ops; ++i) {
        // W4A8 is tested separately: `is_dynamic_quant_config` demands an s8
        // weight, so plain s4 answers false there and would skip the grouped
        // pre-pass entirely.  Source quantization depends only on the src /
        // compute dtypes and the src scale layout — all of which W4A8
        // satisfies — so admit it here and let the per-expert gates in
        // `group_reorder_quantization_wrapper` decide.
        if (is_dynamic_quant_config(params[i]) || is_w4a8_config(params[i])) {
            return true;
        }
    }
    return false;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // REORDER_QUANTIZATION_HPP
