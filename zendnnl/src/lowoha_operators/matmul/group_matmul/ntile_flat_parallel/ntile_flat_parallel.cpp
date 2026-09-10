/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file ntile_flat_parallel.cpp
 * @brief Eligibility analysis, packed-weight cache and executor for the
 *        W8A8 grouped-MoE fast path.
 *
 * The pipeline, once a call is proven eligible:
 *
 *   0.   for `bf16_dynamic`, quantize every grouped source row
 *        bf16 -> biased-u8 once; for `s8_prequantized`, copy the caller's
 *        per-token scale and read the signed-S8 source in place
 *   1.   gate/up: for each (row-tile, N-block) run a dual-accumulator int8
 *        VNNI GEMM against the two halves of the gate/up weight and fold
 *        silu(gate) * up into the epilogue
 *   1.5  requantize the [rows, I] intermediate per row
 *   2.   down projection, writing bf16 either back into the reusable BF16
 *        source (`bf16_dynamic`) or into caller-owned `dst_down`
 *        (`s8_prequantized`) -- exactly where `row_ptrs` points. In S8 mode
 *        `dst_down` is the BF16 view of the same BF16-sized backing whose
 *        leading bytes held the tight S8 source
 *   3.   hand off to the caller's existing weighted-reduce post-op
 *
 * Stages 1 and 2 both read their A operand contiguously: the caller already
 * grouped each expert's rows, so unlike a routing-driven implementation this
 * path never gathers.
 *
 * Framework-neutral: raw pointers, OpenMP, and the AVX-512 primitives in
 * `custom_kernel/ukernel/ntile_flat_parallel_microkernel.hpp`.
 */

#include "lowoha_operators/matmul/group_matmul/ntile_flat_parallel/ntile_flat_parallel.hpp"

#include "lowoha_operators/matmul/group_matmul/custom_kernel/ntile_flat_parallel_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/ntile_flat_parallel_microkernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>
#include <shared_mutex>

#include "common/bfloat16.hpp"
#include "common/op_config.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/lru_cache/lru_cache.hpp"
#include "lowoha_operators/matmul/lru_cache/zendnnl_key.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

namespace {

/// A fast-path execution holds the shared side of this lock from analysis
/// through its final store; the cache flush takes the exclusive side, so it
/// cannot retire a pack while a call is still reading it.
std::shared_mutex &execution_lifecycle_mutex() {
    static std::shared_mutex mutex;
    return mutex;
}

constexpr uint32_t packed_layout_version = 1;

enum class weight_role_t : uint32_t { gate_up = 0, down = 1 };

#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED

// Row-tile quantum handed to one row-dispatch call. The microkernel itself is
// instantiated for 1..4 rows and iterates within each tile.
constexpr int64_t block_m = 32;

// ---------------------------------------------------------------------------
// Packed weight, cached for the weight's life.
//
// One entry covers a whole [E, OC, IC] weight tensor: the block-VNNI quants
// with their int32 compensation. Per-output-channel scales are intentionally
// copied into per-call scratch instead: scale tensors may change independently
// of constant weights, and concurrent calls must never publish over each
// other's epilogue inputs.
// ---------------------------------------------------------------------------
struct packed_weight_t {
    int8_t *quants = nullptr;

    packed_weight_t() = default;
    packed_weight_t(const packed_weight_t &) = delete;
    packed_weight_t &operator=(const packed_weight_t &) = delete;
    ~packed_weight_t() { zendnnl_aligned_free(quants); }
};

using packed_cache_t
        = lru_cache_t<Key_matmul, std::shared_ptr<packed_weight_t>>;

packed_cache_t &packed_cache() {
    // Honour the configured LRU bound exactly. A shared_ptr value still makes
    // eviction safe for in-flight executions, while memory-bounded deployments
    // retain control over how many complete MoE tensors remain resident.
    static packed_cache_t cache(
            common::matmul_config_t::instance().get_lru_cache_capacity());
    return cache;
}

std::mutex &packed_cache_fill_mutex() {
    static std::mutex m;
    return m;
}

Key_matmul make_cache_key(const void *base, weight_role_t role,
        int64_t num_experts, int64_t out_channels, int64_t in_channels,
        int ldb) {
    Key_matmul key;
    key.transpose_weights = true;
    key.weights = base;
    key.m = static_cast<unsigned>(num_experts);
    key.n = static_cast<unsigned>(out_channels);
    key.k = static_cast<unsigned>(in_channels);
    key.ldb = static_cast<unsigned>(ldb);
    key.extra_input_hash = (static_cast<size_t>(role) << 32)
            | static_cast<size_t>(packed_layout_version);
    return key;
}

/// Fetch (packing on first use) the packed form of a whole [E, OC, IC] int8
/// weight tensor.
///
/// Identity is the immutable weight allocation (base address + geometry +
/// role), not sampled content: sampling cannot prove that unsampled bytes did
/// not change. While the cache is enabled, the caller must keep the weight
/// storage live and unchanged until `clear_fused_moe_scratch()` completes in
/// a quiescent window. Scales are copied from the current call and are not part
/// of this lifetime contract. That explicit weight-generation boundary also
/// makes allocator address reuse safe without adding weight reads to this hot
/// lookup.
status_t lookup_packed_weight(const void *base, weight_role_t role,
        int64_t num_experts, int64_t out_channels, int64_t in_channels, int ldb,
        int num_threads, std::shared_ptr<packed_weight_t> &result) {
    const Key_matmul key = make_cache_key(
            base, role, num_experts, out_channels, in_channels, ldb);

    std::shared_ptr<packed_weight_t> found;
    if (packed_cache().try_get(key, found) && found != nullptr) {
        result = std::move(found);
        return status_t::success;
    }

    // Serialize the pack itself so two threads racing on a cold layer do not
    // both pay for it. The LRU has its own lock; this one covers the
    // check-pack-publish sequence.
    std::lock_guard<std::mutex> fill_guard(packed_cache_fill_mutex());
    if (packed_cache().try_get(key, found) && found != nullptr) {
        result = std::move(found);
        return status_t::success;
    }

    const int64_t oc_stride = packed_bytes_per_oc(in_channels);
    size_t expert_channels = 0;
    size_t quant_bytes = 0;
    if (zendnnl_mul_overflow(static_cast<size_t>(num_experts),
                static_cast<size_t>(out_channels), &expert_channels)
            || zendnnl_mul_overflow(expert_channels,
                    static_cast<size_t>(oc_stride), &quant_bytes)) {
        return status_t::memory_bad_size;
    }

    auto entry = std::make_shared<packed_weight_t>();
    entry->quants
            = static_cast<int8_t *>(zendnnl_aligned_alloc(64, quant_bytes));
    if (entry->quants == nullptr) { return status_t::memory_bad_storage; }

    const status_t pack_status = pack_weights(static_cast<const int8_t *>(base),
            entry->quants, num_experts, out_channels, in_channels, num_threads);
    if (pack_status != status_t::success) { return pack_status; }

    packed_cache().add(key, entry);
    result = std::move(entry);
    return status_t::success;
}

/// Promote one current call's scale vector to private f32 scratch.
status_t copy_scale_to_f32(float *dst, const void *scale_buff,
        data_type_t scale_dt, int64_t count) {
    if (scale_dt == data_type_t::bf16) {
        widen_bf16_to_f32(
                dst, static_cast<const uint16_t *>(scale_buff), count);
    } else if (scale_dt == data_type_t::f32) {
        std::memcpy(
                dst, scale_buff, static_cast<size_t>(count) * sizeof(float));
    } else {
        return status_t::unimplemented;
    }
    return status_t::success;
}

// ---------------------------------------------------------------------------
// Scratch, reused across calls at high-water capacity.
// One inference stream per calling thread is this op's existing documented
// assumption, so thread_local both enforces it and keeps every buffer private
// without a lock.
// ---------------------------------------------------------------------------
struct scratch_t {
    // Quantized source and quantized intermediate get their own buffers. One
    // reused block would also work -- the barrier between the passes makes the
    // source dead before the intermediate is written -- but the two row
    // strides (`hidden` vs `inter`) map the same row index to overlapping
    // offsets, so the aliasing would be load-bearing and invisible.
    std::vector<uint8_t> aq_src;
    std::vector<uint8_t> aq_mid;
    std::vector<float> as;
    std::vector<uint16_t> intermediate;
    std::vector<const uint16_t *> row_src;
    std::vector<uint8_t *> row_scale_dst;
    std::vector<uint8_t> row_scale_bf16;
    std::vector<float> gate_up_scales;
    std::vector<float> down_scales;
    std::vector<uint16_t *> slot_dst;
    std::vector<int> slot_ldc;
    std::vector<int32_t> tile_slot;
    std::vector<int64_t> tile_row0;
    std::vector<int32_t> tile_rows;
    std::vector<int64_t> row_off;
    std::vector<int64_t> slot_expert_gate_up;
    std::vector<int64_t> slot_expert_down;
    std::vector<uint8_t> row_seen;
    struct byte_range_t {
        uintptr_t begin = 0;
        uintptr_t end = 0;
    };
    std::vector<byte_range_t> src_ranges;
    std::vector<byte_range_t> dst_ranges;
    std::vector<uintptr_t> written_row_strides;

    template <typename T>
    static T *reserve(std::vector<T> &v, size_t n) {
        if (v.size() < n) { v.resize(n); }
        return v.data();
    }
};

scratch_t &scratch() {
    static thread_local scratch_t s;
    return s;
}

bool make_matrix_range(const void *base, int64_t rows, int64_t row_stride,
        int64_t cols, size_t elem_size, scratch_t::byte_range_t &out) {
    if (base == nullptr || rows <= 0 || row_stride < cols || cols <= 0
            || elem_size == 0) {
        return false;
    }
    size_t row_offset = 0;
    size_t elems = 0;
    size_t bytes = 0;
    if (zendnnl_mul_overflow(static_cast<size_t>(rows - 1),
                static_cast<size_t>(row_stride), &row_offset)
            || zendnnl_add_overflow(
                    row_offset, static_cast<size_t>(cols), &elems)
            || zendnnl_mul_overflow(elems, elem_size, &bytes)) {
        return false;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
    if (bytes > std::numeric_limits<uintptr_t>::max() - begin) { return false; }
    out.begin = begin;
    out.end = begin + bytes;
    return true;
}

inline bool ranges_overlap(
        const scratch_t::byte_range_t &a, const scratch_t::byte_range_t &b) {
    return a.begin < b.end && b.begin < a.end;
}

bool validate_row_permutation(const group_matmul_moe_postop_params &postop,
        const scratch_t::byte_range_t *written_ranges,
        const uintptr_t *written_row_strides, const std::vector<int> &M,
        size_t num_active, const int64_t *row_off, int64_t total_rows,
        uint8_t *seen) {
    std::memset(seen, 0, static_cast<size_t>(total_rows));
    for (int64_t j = 0; j < total_rows; ++j) {
        if (postop.row_ptrs[j] == nullptr) { return false; }
        const uintptr_t addr = reinterpret_cast<uintptr_t>(postop.row_ptrs[j]);

        bool matched = false;
        for (size_t i = 0; i < num_active; ++i) {
            const auto &range = written_ranges[i];
            if (addr < range.begin || addr >= range.end) { continue; }
            const uintptr_t off = addr - range.begin;
            const uintptr_t row_stride = written_row_strides[i];
            if (row_stride == 0 || off % row_stride != 0) { return false; }
            const int64_t row = static_cast<int64_t>(off / row_stride);
            if (row < 0 || row >= M[i]) { return false; }
            const int64_t global = row_off[i] + row;
            if (seen[global] != 0) { return false; }
            seen[global] = 1;
            matched = true;
            break;
        }
        if (!matched) { return false; }
    }
    return true;
}

/// Per-expert canonical indices recovered from a per-expert pointer vector.
///
/// The caller presents the experts in firing order (active ones first), so
/// the vector position is not a stable identity for a cached pack. The
/// tensor's own expert index is, and it is recoverable: the pointers are
/// slices of one [E, OC, IC] tensor, so the lowest address is expert 0 and
/// every other pointer sits at an exact multiple of the expert stride from
/// it. Anything that does not fit that shape is declined.
struct expert_index_map_t {
    const void *base = nullptr;
    std::vector<int64_t> index;
};

bool recover_expert_indices(const std::vector<const void *> &ptrs, size_t count,
        int64_t expert_stride_bytes, expert_index_map_t &out) {
    if (count == 0 || expert_stride_bytes <= 0) { return false; }
    if (ptrs[0] == nullptr) { return false; }
    // The generic API permits independent expert allocations. Compare and
    // subtract their integer representations so a non-contiguous set declines
    // cleanly instead of invoking unrelated-pointer ordering/arithmetic.
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(ptrs[0]);
    for (size_t i = 1; i < count; ++i) {
        if (ptrs[i] == nullptr) { return false; }
        base_addr = std::min(base_addr, reinterpret_cast<uintptr_t>(ptrs[i]));
    }
    out.base = reinterpret_cast<const void *>(base_addr);
    out.index.assign(count, -1);
    // `count` distinct in-range indices over `count` slots is a bijection, so
    // a seen-set of the same size is enough to prove no two slots claim the
    // same expert.
    std::vector<uint8_t> seen(count, 0);
    const uintptr_t stride = static_cast<uintptr_t>(expert_stride_bytes);
    for (size_t i = 0; i < count; ++i) {
        if (ptrs[i] == nullptr) { return false; }
        const uintptr_t addr = reinterpret_cast<uintptr_t>(ptrs[i]);
        if (addr < base_addr) { return false; }
        const uintptr_t off = addr - base_addr;
        if (off % stride != 0) { return false; }
        const uintptr_t expert = off / stride;
        if (expert >= count) { return false; }
        if (seen[static_cast<size_t>(expert)] != 0) { return false; }
        seen[static_cast<size_t>(expert)] = 1;
        out.index[i] = static_cast<int64_t>(expert);
    }
    return true;
}

/// Everything the executor needs, once the call has been proven eligible.
struct plan_t {
    int64_t num_active = 0;
    int64_t num_total = 0;
    int64_t hidden = 0; // K: gate/up reduction, down output width
    int64_t inter = 0; // I: gate/up output width, down reduction
    int64_t num_tokens = 0;
    int64_t topk = 0;
    int64_t total_rows = 0;
    int num_threads = 1;
    expert_index_map_t gate_up_experts;
    expert_index_map_t down_experts;
};

/// Resolved geometry and packed-weight bases, so the two schedules below can
/// share one description of the arithmetic.
struct exec_ctx_t {
    int64_t hidden = 0;
    int64_t inter = 0;
    int64_t gate_up_oc = 0;
    int64_t down_oc = 0;
    int64_t gate_up_oc_stride = 0;
    int64_t down_oc_stride = 0;
    int64_t gate_up_expert_stride = 0;
    int64_t down_expert_stride = 0;
    int64_t nb_gate_up = 0; // N blocks over the gated width
    int64_t nb_down = 0; // N blocks over the hidden width
    const int8_t *packed_gate_up = nullptr;
    const int8_t *packed_down = nullptr;
    const float *gate_up_scale = nullptr;
    const float *down_scale = nullptr;
};

/// Gate/up for one output-channel block: dual-accumulator int8 GEMM against
/// the gate and up halves, with silu(gate) * up folded into the epilogue.
template <bool SignedA>
ZENDNNL_ALWAYS_INLINE inline void gate_up_block(const exec_ctx_t &c,
        int64_t weight_expert, int64_t scale_slot, int64_t nb, const uint8_t *A,
        const float *As, uint16_t *C, int64_t rows, int64_t ldc) {
    const int8_t *base
            = c.packed_gate_up + weight_expert * c.gate_up_expert_stride;
    const int8_t *B0 = base + nb * block_n * c.gate_up_oc_stride;
    const int8_t *B1
            = base + (nb + c.nb_gate_up) * block_n * c.gate_up_oc_stride;
    const float *scale = c.gate_up_scale + scale_slot * c.gate_up_oc;
    tinygemm_gate_up<SignedA>(A, B0, B1, C + nb * block_n, As,
            scale + nb * block_n, scale + (nb + c.nb_gate_up) * block_n,
            reinterpret_cast<const int32_t *>(B0 + block_n * c.hidden),
            reinterpret_cast<const int32_t *>(B1 + block_n * c.hidden), rows,
            c.hidden, c.hidden, block_n, ldc);
}

/// Down projection for one output-channel block, written as bf16 straight to
/// its final destination.
ZENDNNL_ALWAYS_INLINE inline void down_block(const exec_ctx_t &c,
        int64_t weight_expert, int64_t scale_slot, int64_t nb, const uint8_t *A,
        const float *As, uint16_t *C, int64_t rows, int64_t ldc) {
    const int8_t *B = c.packed_down + weight_expert * c.down_expert_stride
            + nb * block_n * c.down_oc_stride;
    tinygemm_down(A, B, C + nb * block_n, As,
            c.down_scale + scale_slot * c.down_oc + nb * block_n,
            reinterpret_cast<const int32_t *>(B + block_n * c.inter), rows,
            c.inter, c.inter, block_n, ldc);
}

#endif // ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED

} // namespace

void flush_packed_weight_cache() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(
            execution_lifecycle_mutex());
#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
    packed_cache().clear();
#endif
}

size_t packed_weight_cache_size() {
#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
    const int size = packed_cache().get_size();
    return size > 0 ? static_cast<size_t>(size) : 0;
#else
    return 0;
#endif
}

void reset_thread_local_scratch() {
#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
    scratch() = scratch_t {};
#endif
}

// ---------------------------------------------------------------------------
// Eligibility + execution
// ---------------------------------------------------------------------------

status_t try_execute(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params,
        const group_matmul_moe_postop_params *moe_postop,
        const grp_matmul_gated_act_params *gated_act,
        const grp_matmul_fused_moe_params *fused_moe) {
#if !ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
    (void)layout;
    (void)transA;
    (void)transB;
    (void)M;
    (void)N;
    (void)K;
    (void)alpha;
    (void)src;
    (void)lda;
    (void)weight;
    (void)ldb;
    (void)bias;
    (void)beta;
    (void)dst;
    (void)ldc;
    (void)is_weights_const;
    (void)params;
    (void)moe_postop;
    (void)gated_act;
    (void)fused_moe;
    return status_t::unimplemented;
#else
    // This path packs both complete expert tensors, so running it without a
    // persistent pack would repack the model on every token. Honour the
    // process-wide cache-disable contract by declining before any output is
    // touched. The caller may continue through main's generic BF16 or
    // caller-prequantized-S8 fused-MoE path.
    if (common::matmul_config_t::instance().get_weight_cache() == 0) {
        return status_t::unimplemented;
    }
    if (common::matmul_config_t::instance().get_lru_cache_capacity() == 0) {
        return status_t::unimplemented;
    }

    // ── Shape of the request ────────────────────────────────────────────
    if (moe_postop == nullptr || gated_act == nullptr || fused_moe == nullptr) {
        return status_t::unimplemented;
    }
    if (gated_act->act != grp_matmul_gated_act_t::silu_and_mul) {
        return status_t::unimplemented;
    }
    if (params.empty() || M.empty() || src.empty()) {
        return status_t::unimplemented;
    }
    if (params[0].active_matmul > 0 && params[0].total_matmul > 0
            && params[0].total_matmul < params[0].active_matmul) {
        return status_t::failure;
    }

    const size_t num_active = params[0].active_matmul > 0
            ? std::min<size_t>(params[0].active_matmul, M.size())
            : M.size();
    const size_t num_total = params[0].total_matmul > num_active
            ? static_cast<size_t>(params[0].total_matmul)
            : num_active;
    if (num_active == 0) { return status_t::unimplemented; }
    // Every expert's rows must arrive in its own buffer: this path indexes
    // src[i] per expert and has no meaning for a shared-source chain.
    if (src.size() != M.size()) { return status_t::unimplemented; }

    // Weight-side vectors must cover the full advertised expert pool, and
    // input-side vectors the active prefix.
    if (weight.size() < num_total || ldb.size() < num_total
            || N.size() < num_total || K.size() < num_total
            || transB.size() < num_total
            || is_weights_const.size() < num_total) {
        return status_t::unimplemented;
    }
    if (lda.size() < num_active || layout.size() < num_active
            || transA.size() < num_active || alpha.size() < num_active
            || beta.size() < num_active || params.size() < num_active) {
        return status_t::unimplemented;
    }
    if (fused_moe->down_weight.size() < num_total
            || fused_moe->N_down.size() < num_total
            || fused_moe->ldb_down.size() < num_total
            || fused_moe->bias_down.size() < num_active
            || fused_moe->down_scale.size() < num_active) {
        return status_t::unimplemented;
    }
    for (size_t i = 0; i < num_active; ++i) {
        if (effective_weight_cache_type(params[i].weight_cache_type) == 0) {
            return status_t::unimplemented;
        }
    }

    const input_mode_t input_mode
            = classify_uniform_input_mode(M, params, num_active);
    if (input_mode == input_mode_t::unsupported
            || input_mode == input_mode_t::mixed) {
        return status_t::unimplemented;
    }
    const bool s8_prequantized = input_mode == input_mode_t::s8_prequantized;

    // Op1 is always library-owned: ALGO 4 produces the gated BF16
    // intermediate privately and never exposes it through `dst`.
    if ((!dst.empty() && dst.size() < num_active)
            || (!ldc.empty() && ldc.size() < num_active)) {
        return status_t::unimplemented;
    }
    if (!dst.empty()) {
        for (size_t i = 0; i < num_active; ++i) {
            if (dst[i] != nullptr) { return status_t::unimplemented; }
        }
    }

    // BF16 mode retains the original in-place W2 destination. S8 mode requires
    // the caller's exact same-base BF16 destination view over the BF16-sized
    // backing whose leading bytes hold the tight S8 source prefix.
    if (!s8_prequantized) {
        if (!fused_moe->dst_down.empty() || !fused_moe->ldc_down.empty()) {
            return status_t::unimplemented;
        }
    } else if (fused_moe->dst_down.size() < num_active
            || fused_moe->ldc_down.size() < num_active) {
        return status_t::unimplemented;
    }

    // ── No bias, no zero point, no unsupported quant mode ───────────────
    if (fused_moe->bias_dt_down != data_type_t::none) {
        return status_t::unimplemented;
    }
    for (size_t i = 0; i < num_active; ++i) {
        if (i < bias.size() && bias[i] != nullptr) {
            return status_t::unimplemented;
        }
        if (i < fused_moe->bias_down.size()
                && fused_moe->bias_down[i] != nullptr) {
            return status_t::unimplemented;
        }
    }
    if (!fused_moe->down_zp.empty()) {
        if (fused_moe->down_zp.size() < num_active) {
            return status_t::unimplemented;
        }
        for (size_t i = 0; i < num_active; ++i) {
            if (!fused_moe->down_zp[i].dims.empty()
                    || fused_moe->down_zp[i].buff != nullptr) {
                return status_t::unimplemented;
            }
        }
    }

    // ── Uniform geometry ────────────────────────────────────────────────
    const int64_t gate_up_oc = N[0];
    const int64_t hidden = K[0];
    const int64_t down_oc = fused_moe->N_down[0];
    if (gate_up_oc <= 0 || hidden <= 0 || down_oc <= 0) {
        return status_t::unimplemented;
    }
    if (gate_up_oc % 2 != 0) { return status_t::unimplemented; }
    const int64_t inter = gate_up_oc / 2;

    // The gate/up reduction is `hidden`, the down reduction is `inter`.
    // BF16 W13 input and the BF16 intermediate feed `quantize_row_u8` (32
    // lanes per step); direct S8 W13 input bypasses that first quantizer. Both
    // GEMMs index N blocks of `block_n`. `down_oc` is the post-op row width.
    if (hidden % block_n != 0 || inter % block_n != 0
            || down_oc % block_n != 0) {
        return status_t::unimplemented;
    }
    if (hidden > max_gemm_reduction || inter > max_gemm_reduction) {
        return status_t::unimplemented;
    }
    // Op2 consumes the activation output, so the down projection must reduce
    // over exactly the gated width, and produce the hidden size back.
    if (down_oc != hidden) { return status_t::unimplemented; }
    if (s8_prequantized) {
        for (size_t i = 0; i < num_active; ++i) {
            if (fused_moe->dst_down[i] == nullptr
                    || fused_moe->dst_down[i] != src[i]
                    || fused_moe->ldc_down[i] != hidden) {
                return status_t::unimplemented;
            }
        }
    }

    for (size_t i = 0; i < num_total; ++i) {
        if (N[i] != gate_up_oc || K[i] != hidden) {
            return status_t::unimplemented;
        }
        if (!transB[i] || !is_weights_const[i]) {
            return status_t::unimplemented;
        }
        // The packer walks each output channel as `in_channels` contiguous
        // bytes, so only tight weight rows qualify.
        if (ldb[i] != hidden) { return status_t::unimplemented; }
        if (weight[i] == nullptr) { return status_t::unimplemented; }
        if (fused_moe->N_down[i] != down_oc
                || fused_moe->ldb_down[i] != inter) {
            return status_t::unimplemented;
        }
        if (fused_moe->down_weight[i] == nullptr) {
            return status_t::unimplemented;
        }
    }

    // ── Per-op dtype / quant contract ───────────────────────────────────
    int64_t total_rows = 0;
    for (size_t i = 0; i < num_active; ++i) {
        const auto &p = params[i];
        if (classify_input_mode(p) != input_mode
                || p.dtypes.dst != data_type_t::bf16
                || p.dtypes.wei != data_type_t::s8
                || p.dtypes.compute != data_type_t::s8
                || p.dtypes.bias != data_type_t::none || !p.postop_.empty()) {
            return status_t::unimplemented;
        }
        if (layout[i] != 'r' || transA[i] || p.mem_format_a != 'n'
                || p.mem_format_b != 'n' || p.packing.pack_format_b != 0) {
            return status_t::unimplemented;
        }
        if (alpha[i] != 1.0f || beta[i] != 0.0f) {
            return status_t::unimplemented;
        }
        if (M[i] <= 0 || src[i] == nullptr) { return status_t::unimplemented; }
        // Rows must be tight in the reduction direction: stage 0 (BF16 mode)
        // or W13 directly (S8 mode) reads `hidden` contiguous values.
        if (lda[i] != hidden) { return status_t::unimplemented; }

        // Symmetric per-output-channel weight scale, per-token activation
        // scale. Anything coarser or finer belongs to the generic path.
        const auto &q = p.quant_params;
        if (q.wei_scale.buff == nullptr) { return status_t::unimplemented; }
        if (q.wei_scale.dims.size() != 2 || q.wei_scale.dims[0] != 1
                || q.wei_scale.dims[1] != gate_up_oc) {
            return status_t::unimplemented;
        }
        if (q.wei_scale.dt != data_type_t::bf16
                && q.wei_scale.dt != data_type_t::f32) {
            return status_t::unimplemented;
        }
        if (q.src_scale.dims.size() != 2 || q.src_scale.dims[0] != M[i]
                || q.src_scale.dims[1] != 1) {
            return status_t::unimplemented;
        }
        if (q.src_scale.dt != data_type_t::bf16
                && q.src_scale.dt != data_type_t::f32) {
            return status_t::unimplemented;
        }
        if (s8_prequantized && q.src_scale.buff == nullptr) {
            return status_t::unimplemented;
        }
        if (!q.wei_zp.dims.empty() || q.wei_zp.buff != nullptr
                || !q.src_zp.dims.empty() || q.src_zp.buff != nullptr
                || (s8_prequantized && q.src_zp.dt != data_type_t::none)
                || !q.dst_zp.dims.empty() || q.dst_zp.buff != nullptr
                || !q.dst_scale.dims.empty() || q.dst_scale.buff != nullptr) {
            return status_t::unimplemented;
        }

        const auto &ds = fused_moe->down_scale[i];
        if (ds.buff == nullptr || ds.dims.size() != 2 || ds.dims[0] != 1
                || ds.dims[1] != down_oc) {
            return status_t::unimplemented;
        }
        if (ds.dt != data_type_t::bf16 && ds.dt != data_type_t::f32) {
            return status_t::unimplemented;
        }

        if (M[i] > std::numeric_limits<int64_t>::max() - total_rows) {
            return status_t::unimplemented;
        }
        total_rows += M[i];
    }

    // ── Post-op contract ────────────────────────────────────────────────
    const int64_t num_tokens = moe_postop->num_tokens;
    const int64_t topk = moe_postop->topk;
    if (num_tokens <= 0 || topk <= 0 || moe_postop->output == nullptr
            || moe_postop->row_ptrs == nullptr) {
        return status_t::unimplemented;
    }
    if (!moe_postop->skip_weighted && moe_postop->topk_weights == nullptr) {
        return status_t::unimplemented;
    }
    if (moe_postop->ldc_output < down_oc) { return status_t::unimplemented; }
    // Every routed (token, slot) pair must be represented exactly once among
    // the grouped rows; that is what makes `row_ptrs` a permutation of the
    // rows this path writes.
    if (num_tokens > std::numeric_limits<int>::max() / topk
            || total_rows != num_tokens * topk) {
        return status_t::unimplemented;
    }

    if (!isa_supported()) { return status_t::unimplemented; }

    // ── Stable per-expert identity for the packed-weight cache ──────────
    plan_t plan;
    plan.num_active = static_cast<int64_t>(num_active);
    plan.num_total = static_cast<int64_t>(num_total);
    plan.hidden = hidden;
    plan.inter = inter;
    plan.num_tokens = num_tokens;
    plan.topk = topk;
    plan.total_rows = total_rows;

    try {
        if (!recover_expert_indices(weight, num_total, gate_up_oc * hidden,
                    plan.gate_up_experts)) {
            return status_t::unimplemented;
        }
        if (!recover_expert_indices(fused_moe->down_weight, num_total,
                    down_oc * inter, plan.down_experts)) {
            return status_t::unimplemented;
        }
    } catch (const std::length_error &) {
        return status_t::memory_bad_size;
    } catch (const std::bad_alloc &) { return status_t::memory_bad_storage; }

    if (static_cast<uint64_t>(total_rows)
            > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return status_t::memory_bad_size;
    }
    const size_t total_rows_size = static_cast<size_t>(total_rows);
    size_t aq_src_elems = 0;
    size_t intermediate_elems = 0;
    size_t gate_up_scale_elems = 0;
    size_t down_scale_elems = 0;
    if (zendnnl_mul_overflow(
                total_rows_size, static_cast<size_t>(hidden), &aq_src_elems)
            || zendnnl_mul_overflow(total_rows_size, static_cast<size_t>(inter),
                    &intermediate_elems)
            || zendnnl_mul_overflow(num_active, static_cast<size_t>(gate_up_oc),
                    &gate_up_scale_elems)
            || zendnnl_mul_overflow(num_active, static_cast<size_t>(down_oc),
                    &down_scale_elems)) {
        return status_t::memory_bad_size;
    }

    std::shared_lock<std::shared_mutex> lifecycle_lock(
            execution_lifecycle_mutex());

    try {
        // Use the actual task-local ICV, not thread_guard's cached process
        // baseline. group_matmul_direct may already have installed a reduced
        // caller cap; restoring the cached baseline here would undo that cap
        // before an eligibility decline enters generic fallback.
        const int32_t omp_mt = omp_get_max_threads();
        plan.num_threads = static_cast<int>(
                resolve_num_threads(params[0].num_threads, omp_mt));
        if (plan.num_threads <= 0) { plan.num_threads = 1; }
        thread_guard tg(plan.num_threads, omp_mt);
        const int nth = plan.num_threads;

        scratch_t &sc = scratch();

        // ── Row bookkeeping: expert-major, in the caller's expert order ──
        int64_t *row_off = scratch_t::reserve(
                sc.row_off, static_cast<size_t>(num_active) + 1);
        row_off[0] = 0;
        for (size_t i = 0; i < num_active; ++i) {
            row_off[i + 1] = row_off[i] + M[i];
        }

        // Build exact matrix spans before any kernel writes. BF16 mode writes
        // back into tight src rows. S8 mode requires ZenTorch's exact
        // same-base reuse contract: a tight S8 [M, hidden] prefix inside a
        // BF16-sized [M, hidden] backing allocation. The raw API cannot prove
        // allocation capacity, so the backing size is the caller's
        // responsibility.
        auto *src_ranges = scratch_t::reserve(sc.src_ranges, num_active);
        for (size_t i = 0; i < num_active; ++i) {
            const size_t src_elem
                    = s8_prequantized ? sizeof(int8_t) : sizeof(uint16_t);
            if (!make_matrix_range(src[i], M[i], hidden, hidden, src_elem,
                        src_ranges[i])) {
                return status_t::unimplemented;
            }
        }

        const scratch_t::byte_range_t *written_ranges = src_ranges;
        uintptr_t *written_row_strides
                = scratch_t::reserve(sc.written_row_strides, num_active);
        size_t tight_bf16_row_bytes = 0;
        if (zendnnl_mul_overflow(static_cast<size_t>(hidden), sizeof(uint16_t),
                    &tight_bf16_row_bytes)
                || tight_bf16_row_bytes == 0) {
            return status_t::unimplemented;
        }
        std::fill_n(written_row_strides, num_active,
                static_cast<uintptr_t>(tight_bf16_row_bytes));
        if (s8_prequantized) {
            auto *dst_ranges = scratch_t::reserve(sc.dst_ranges, num_active);
            for (size_t i = 0; i < num_active; ++i) {
                if (!make_matrix_range(fused_moe->dst_down[i], M[i],
                            fused_moe->ldc_down[i], hidden, sizeof(uint16_t),
                            dst_ranges[i])) {
                    return status_t::unimplemented;
                }
                size_t stride_bytes = 0;
                if (zendnnl_mul_overflow(
                            static_cast<size_t>(fused_moe->ldc_down[i]),
                            sizeof(uint16_t), &stride_bytes)
                        || stride_bytes == 0) {
                    return status_t::unimplemented;
                }
                written_row_strides[i] = static_cast<uintptr_t>(stride_bytes);
            }
            for (size_t i = 0; i < num_active; ++i) {
                for (size_t j = i + 1; j < num_active; ++j) {
                    if (ranges_overlap(dst_ranges[i], dst_ranges[j])) {
                        return status_t::unimplemented;
                    }
                }
            }
            written_ranges = dst_ranges;
        }

        // The reduce writes token rows in parallel after W2. Its output must
        // not alias any BF16 W2 row that another token may still read.
        scratch_t::byte_range_t reduced_output_range;
        if (!make_matrix_range(moe_postop->output, num_tokens,
                    moe_postop->ldc_output, down_oc, sizeof(uint16_t),
                    reduced_output_range)) {
            return status_t::unimplemented;
        }
        for (size_t i = 0; i < num_active; ++i) {
            if (ranges_overlap(written_ranges[i], reduced_output_range)
                    || ranges_overlap(src_ranges[i], reduced_output_range)) {
                return status_t::unimplemented;
            }
        }

        uint8_t *seen = scratch_t::reserve(
                sc.row_seen, static_cast<size_t>(total_rows));
        if (!validate_row_permutation(*moe_postop, written_ranges,
                    written_row_strides, M, num_active, row_off, total_rows,
                    seen)) {
            return status_t::unimplemented;
        }

        // S8 W13 consumes caller-provided per-token scales. Copy them into
        // the contiguous F32 scratch used by the epilogue and validate every
        // value before any intermediate or caller destination is written.
        float *As = scratch_t::reserve(sc.as, static_cast<size_t>(total_rows));
        if (s8_prequantized) {
            for (size_t i = 0; i < num_active; ++i) {
                float *scale_dst = As + row_off[i];
                const auto &src_scale = params[i].quant_params.src_scale;
                if (src_scale.dt == data_type_t::bf16) {
                    widen_bf16_to_f32(scale_dst,
                            static_cast<const uint16_t *>(src_scale.buff),
                            M[i]);
                } else {
                    std::memcpy(scale_dst, src_scale.buff,
                            static_cast<size_t>(M[i]) * sizeof(float));
                }
                for (int64_t m = 0; m < M[i]; ++m) {
                    if (!(scale_dst[m] > 0.f) || !std::isfinite(scale_dst[m])) {
                        return status_t::unimplemented;
                    }
                }
            }
        }

        // Weight scales are current-call data even when weights themselves are
        // constant. Keep them private to this execution so changed scale
        // tensors and concurrent calls cannot observe stale/shared values.
        float *gate_up_scales
                = scratch_t::reserve(sc.gate_up_scales, gate_up_scale_elems);
        float *down_scales
                = scratch_t::reserve(sc.down_scales, down_scale_elems);
        status_t cs = status_t::success;
        for (size_t i = 0; i < num_active; ++i) {
            cs = copy_scale_to_f32(
                    gate_up_scales + i * static_cast<size_t>(gate_up_oc),
                    params[i].quant_params.wei_scale.buff,
                    params[i].quant_params.wei_scale.dt, gate_up_oc);
            if (cs != status_t::success) { return cs; }
            cs = copy_scale_to_f32(
                    down_scales + i * static_cast<size_t>(down_oc),
                    fused_moe->down_scale[i].buff, fused_moe->down_scale[i].dt,
                    down_oc);
            if (cs != status_t::success) { return cs; }
        }

        // ── Packed weights (packed once, cached for the weight's life) ───
        std::shared_ptr<packed_weight_t> pw_gate_up;
        std::shared_ptr<packed_weight_t> pw_down;
        cs = lookup_packed_weight(plan.gate_up_experts.base,
                weight_role_t::gate_up, plan.num_total, gate_up_oc, hidden,
                static_cast<int>(hidden), nth, pw_gate_up);
        if (cs != status_t::success) { return cs; }
        cs = lookup_packed_weight(plan.down_experts.base, weight_role_t::down,
                plan.num_total, down_oc, inter, static_cast<int>(inter), nth,
                pw_down);
        if (cs != status_t::success) { return cs; }

        // ── Resolved arithmetic description, shared by both schedules ───
        exec_ctx_t ctx;
        ctx.hidden = hidden;
        ctx.inter = inter;
        ctx.gate_up_oc = gate_up_oc;
        ctx.down_oc = down_oc;
        ctx.gate_up_oc_stride = packed_bytes_per_oc(hidden);
        ctx.down_oc_stride = packed_bytes_per_oc(inter);
        ctx.gate_up_expert_stride = gate_up_oc * ctx.gate_up_oc_stride;
        ctx.down_expert_stride = down_oc * ctx.down_oc_stride;
        ctx.nb_gate_up = inter / block_n;
        ctx.nb_down = down_oc / block_n;
        ctx.packed_gate_up = pw_gate_up->quants;
        ctx.packed_down = pw_down->quants;
        ctx.gate_up_scale = gate_up_scales;
        ctx.down_scale = down_scales;

        // ── Scratch and per-slot resolution ─────────────────────────────
        uint8_t *Aq_src = nullptr;
        const uint16_t **row_src = nullptr;
        uint8_t **row_scale_dst = nullptr;
        uint8_t *row_scale_bf16 = nullptr;
        if (!s8_prequantized) {
            Aq_src = scratch_t::reserve(sc.aq_src, aq_src_elems);
            row_src = scratch_t::reserve(sc.row_src, total_rows_size);
            row_scale_dst
                    = scratch_t::reserve(sc.row_scale_dst, total_rows_size);
            row_scale_bf16
                    = scratch_t::reserve(sc.row_scale_bf16, total_rows_size);
        }
        uint8_t *Aq_mid = scratch_t::reserve(sc.aq_mid, intermediate_elems);
        uint16_t *intermediate
                = scratch_t::reserve(sc.intermediate, intermediate_elems);
        uint16_t **slot_dst = scratch_t::reserve(
                sc.slot_dst, static_cast<size_t>(num_active));
        int *slot_ldc = scratch_t::reserve(
                sc.slot_ldc, static_cast<size_t>(num_active));
        int64_t *slot_e_gate_up = scratch_t::reserve(
                sc.slot_expert_gate_up, static_cast<size_t>(num_active));
        int64_t *slot_e_down = scratch_t::reserve(
                sc.slot_expert_down, static_cast<size_t>(num_active));

        for (size_t i = 0; i < num_active; ++i) {
            // BF16 mode keeps the original source-reuse destination. S8 mode
            // overwrites the same BF16-sized backing through dst_down only
            // after the complete W13 pass has consumed its tight S8 prefix.
            auto *dst_rows = s8_prequantized
                    ? static_cast<uint16_t *>(fused_moe->dst_down[i])
                    : const_cast<uint16_t *>(
                              static_cast<const uint16_t *>(src[i]));
            slot_dst[i] = dst_rows;
            slot_ldc[i] = s8_prequantized ? fused_moe->ldc_down[i] : hidden;
            slot_e_gate_up[i] = plan.gate_up_experts.index[i];
            slot_e_down[i] = plan.down_experts.index[i];
            if (!s8_prequantized) {
                const auto &src_scale = params[i].quant_params.src_scale;
                auto *scale_base = const_cast<uint8_t *>(
                        static_cast<const uint8_t *>(src_scale.buff));
                const size_t scale_elem_size = src_scale.dt == data_type_t::bf16
                        ? sizeof(uint16_t)
                        : sizeof(float);
                for (int64_t p = 0; p < M[i]; ++p) {
                    const int64_t row = row_off[i] + p;
                    row_src[row] = static_cast<const uint16_t *>(src[i])
                            + p * hidden;
                    row_scale_dst[row] = scale_base == nullptr ? nullptr
                                                               : scale_base
                                    + static_cast<size_t>(p) * scale_elem_size;
                    row_scale_bf16[row]
                            = src_scale.dt == data_type_t::bf16 ? 1 : 0;
                }
            }
        }

        // ── Row tiles, flattened across experts ─────────────────────────
        //
        // The pipeline runs as four passes over one global (row tile, N block)
        // space rather than per-expert. Giving a thread a whole expert would
        // remove the barriers between passes and keep the intermediate in that
        // core's cache, which sounds strictly better and measured 3% slower
        // end to end (decode step 106.4 ms vs 103.4 ms): at decode an expert
        // holds one or two rows, so splitting each expert across threads is
        // what actually balances the layer, and the barriers cost less than
        // the imbalance they avoid.
        int64_t num_tiles = 0;
        for (size_t i = 0; i < num_active; ++i) {
            num_tiles += div_up(M[i], block_m);
        }
        int32_t *tile_slot = scratch_t::reserve(
                sc.tile_slot, static_cast<size_t>(num_tiles));
        int64_t *tile_row0 = scratch_t::reserve(
                sc.tile_row0, static_cast<size_t>(num_tiles));
        int32_t *tile_rows = scratch_t::reserve(
                sc.tile_rows, static_cast<size_t>(num_tiles));
        {
            int64_t b = 0;
            for (size_t i = 0; i < num_active; ++i) {
                for (int64_t s = 0; s < M[i]; s += block_m) {
                    tile_slot[b] = static_cast<int32_t>(i);
                    tile_row0[b] = row_off[i] + s;
                    tile_rows[b] = static_cast<int32_t>(
                            std::min<int64_t>(block_m, M[i] - s));
                    ++b;
                }
            }
        }

        // Every pass runs on the same `nth`-wide team on purpose: resizing a
        // team between passes makes the runtime tear it down and rebuild it
        // mid-pipeline, which costs far more at decode token counts than the
        // fork it would save.
        if (!s8_prequantized) {
#pragma omp parallel for num_threads(nth) schedule(static)
            for (int64_t m = 0; m < total_rows; ++m) {
                quantize_row_u8(Aq_src + m * hidden, As[m], row_src[m], hidden);
                if (row_scale_bf16[m] != 0) {
                    const int16_t scale_bf16
                            = zendnnl::common::bfloat16_t::f32_to_bf16_val(
                                    As[m]);
                    As[m] = zendnnl::common::bfloat16_t::bf16_to_f32_val(
                            scale_bf16);
                    if (row_scale_dst[m] != nullptr) {
                        std::memcpy(row_scale_dst[m], &scale_bf16,
                                sizeof(scale_bf16));
                    }
                } else if (row_scale_dst[m] != nullptr) {
                    std::memcpy(row_scale_dst[m], &As[m], sizeof(As[m]));
                }
            }
        }

        // With `nb` innermost a thread's static chunk walks consecutive
        // output-channel blocks of one expert, so the weight stream stays
        // sequential.
        if (!s8_prequantized) {
#pragma omp parallel for num_threads(nth) schedule(static) collapse(2)
            for (int64_t b = 0; b < num_tiles; ++b) {
                for (int64_t nb = 0; nb < ctx.nb_gate_up; ++nb) {
                    const int64_t slot = tile_slot[b];
                    const int64_t r0 = tile_row0[b];
                    gate_up_block<false>(ctx, slot_e_gate_up[slot], slot, nb,
                            Aq_src + r0 * hidden, As + r0,
                            intermediate + r0 * inter, tile_rows[b], inter);
                }
            }
        } else {
#pragma omp parallel for num_threads(nth) schedule(static) collapse(2)
            for (int64_t b = 0; b < num_tiles; ++b) {
                for (int64_t nb = 0; nb < ctx.nb_gate_up; ++nb) {
                    const int64_t slot = tile_slot[b];
                    const int64_t r0 = tile_row0[b];
                    const int64_t local_row = r0 - row_off[slot];
                    gate_up_block<true>(ctx, slot_e_gate_up[slot], slot, nb,
                            static_cast<const uint8_t *>(src[slot])
                                    + local_row * hidden,
                            As + r0, intermediate + r0 * inter, tile_rows[b],
                            inter);
                }
            }
        }

#pragma omp parallel for num_threads(nth) schedule(static)
        for (int64_t m = 0; m < total_rows; ++m) {
            quantize_row_u8(
                    Aq_mid + m * inter, As[m], intermediate + m * inter, inter);
        }

#pragma omp parallel for num_threads(nth) schedule(static) collapse(2)
        for (int64_t b = 0; b < num_tiles; ++b) {
            for (int64_t nb = 0; nb < ctx.nb_down; ++nb) {
                const int64_t slot = tile_slot[b];
                const int64_t r0 = tile_row0[b];
                const int out_ldc = slot_ldc[slot];
                down_block(ctx, slot_e_down[slot], slot, nb,
                        Aq_mid + r0 * inter, As + r0,
                        slot_dst[slot] + (r0 - row_off[slot]) * out_ldc,
                        tile_rows[b], out_ldc);
            }
        }

        // ── The caller's existing weighted reduction ────────────────────
        return group_matmul_moe_postop_execute(moe_postop,
                static_cast<int>(down_oc), nth, params[0].dtypes.dst);
    } catch (const std::length_error &) {
        return status_t::memory_bad_size;
    } catch (const std::bad_alloc &) { return status_t::memory_bad_storage; }
#endif // ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
}

} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
