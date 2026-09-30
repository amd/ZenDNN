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

#include "lowoha_operators/matmul/lowoha_matmul.hpp"

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/routed_moe/routed_moe_internal.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

struct slot_ref_t {
    int active = -1;
    int row = -1;
};

/**
 * @brief Grow-only, never-zeroed byte scratch.
 *
 * `std::vector<uint8_t>` cannot back the grouped-token surface: `resize`
 * value-initializes, so every fallback call re-zeroed the whole top-k-expanded
 * BF16-sized buffer before overwriting it.  Every consumer here writes each
 * byte it later reads, so the fill is pure waste.  Capacity only ever grows,
 * which also removes the shrink/re-grow refill that a varying token count
 * would otherwise pay on each call.
 */
class byte_scratch_t {
public:
    uint8_t *data() const noexcept { return block_.get(); }

    /// @return false when the host could not supply the block.
    bool reserve(size_t bytes) noexcept {
        if (bytes <= capacity_) { return true; }
        if (bytes > std::numeric_limits<size_t>::max() - (kAlign - 1)) {
            return false;
        }
        size_t grown = capacity_ + capacity_ / 2;
        if (grown < bytes) { grown = bytes; }
        grown = (grown + kAlign - 1) / kAlign * kAlign;
        auto *block
                = static_cast<uint8_t *>(zendnnl_aligned_alloc(kAlign, grown));
        if (block == nullptr) {
            // Retry at the exact request: the 1.5x headroom is an
            // optimization, not a requirement.
            grown = (bytes + kAlign - 1) / kAlign * kAlign;
            block = static_cast<uint8_t *>(
                    zendnnl_aligned_alloc(kAlign, grown));
            if (block == nullptr) { return false; }
        }
        block_.reset(block);
        capacity_ = grown;
        return true;
    }

private:
    // Expert groups start at a multiple of the backing row size and rows are
    // a whole number of elements, so a 64-byte base keeps every grouped row
    // cache-line aligned for the AVX-512 gather/GEMM consumers.
    static constexpr size_t kAlign = 64;
    struct free_deleter_t {
        void operator()(uint8_t *p) const noexcept { zendnnl_aligned_free(p); }
    };
    std::unique_ptr<uint8_t[], free_deleter_t> block_;
    size_t capacity_ = 0;
};

struct grouped_tokens_t {
    std::vector<int> active_experts;
    // Active experts first, followed by every inactive expert. This is the
    // existing framework prepack-extras layout: only the active prefix
    // computes, while the full expert pool remains visible to cache warming.
    std::vector<int> expert_order;
    std::vector<int> counts;
    // Flat expert-major token index per grouped row; active expert `a` owns
    // `[row_base[a], row_base[a] + counts[a])`.  A vector-of-vectors here
    // re-allocated one inner buffer per active expert on every call.
    std::vector<int> token_rows;
    std::vector<size_t> row_base;
    std::vector<slot_ref_t> slots;
    byte_scratch_t storage;
    std::vector<const void *> src;
    // Per-call working sets, kept so their capacity survives the call.
    std::vector<int> expert_to_active;
    std::vector<int> cursors;
};

struct quant_scratch_t {
    byte_scratch_t src_scale;
    byte_scratch_t src_zp;
};

struct projection_result_t {
    byte_scratch_t storage;
    std::vector<void *> dst;
    std::vector<int> ldc;
    quant_scratch_t quant;
};

/**
 * @brief Reused operand vectors for one `group_matmul_direct` invocation.
 *
 * The grouped-Matmul ABI takes seventeen parallel vectors plus a fused-MoE
 * descriptor.  Rebuilding them as locals cost one allocation each per call,
 * and `params` additionally deep-copied every expert's quant-dims and post-op
 * vectors.  Reusing them keeps steady-state allocator traffic at zero.  Only
 * one grouped call is in flight per thread at a time, so a single set is
 * enough for both projections of the unfused path.
 */
struct descriptor_scratch_t {
    std::vector<char> layouts;
    std::vector<bool> trans_a;
    std::vector<bool> trans_b;
    std::vector<int> m;
    std::vector<int> n;
    std::vector<int> k;
    std::vector<float> alpha;
    std::vector<float> beta;
    std::vector<int> lda;
    std::vector<int> ldb;
    std::vector<int> ldc;
    std::vector<const void *> weights;
    std::vector<const void *> biases;
    std::vector<void *> dst;
    std::vector<bool> is_const;
    std::vector<matmul_params> params;
    std::vector<const void *> row_ptrs;
    std::vector<uint8_t> zero_row;
    std::vector<float> dense_weights;
    grp_matmul_fused_moe_params fused;
};

struct generic_scratch_t {
    grouped_tokens_t grouped;
    projection_result_t primary;
    projection_result_t secondary;
    descriptor_scratch_t descriptors;
    group_matmul_projection_params prequantized_primary;
    std::vector<const void *> secondary_src;
    std::vector<int> local_experts;
    std::vector<int> counts;
    std::vector<int8_t> token_quant;
    std::vector<float> token_scale_f32;
    std::vector<uint16_t> token_scale_bf16;
};

generic_scratch_t &generic_scratch() {
    static thread_local generic_scratch_t scratch;
    return scratch;
}

bool routed_moe_enabled() {
    const char *value = std::getenv("ZENDNNL_ENABLE_ROUTED_MOE");
    // Enabled by default.  Only the documented exact value "0" disables the
    // fast executor; unset or unrecognised values retain the default.
    return value == nullptr || value[0] != '0' || value[1] != '\0';
}

/// Grow-only sizing for the typed token scratch: the producer overwrites
/// every live element, so a shrink-then-grow must not re-zero the buffer.
template <typename T>
void reserve_elements(std::vector<T> &buffer, size_t elements) {
    if (buffer.size() < elements) { buffer.resize(elements); }
}

/// Below this much bulk copy, forking an OpenMP team costs more than the
/// copy itself.  Small decode batches stay on the calling thread.
constexpr size_t kParallelCopyBytes = 256u * 1024u;

bool checked_mul_size(size_t a, size_t b, size_t &result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) { return false; }
    result = a * b;
    return true;
}

bool checked_product(
        const std::vector<int64_t> &dims, size_t first, size_t &result) {
    result = 1;
    for (size_t i = first; i < dims.size(); ++i) {
        if (dims[i] <= 0
                || !checked_mul_size(
                        result, static_cast<size_t>(dims[i]), result)) {
            return false;
        }
    }
    return true;
}

size_t packed_elements_bytes(size_t elements, data_type_t dt) {
    if (dt == data_type_t::s4 || dt == data_type_t::u4) {
        return (elements + 1) / 2;
    }
    const size_t element_size = static_cast<size_t>(size_of(dt));
    size_t bytes = 0;
    return element_size != 0 && checked_mul_size(elements, element_size, bytes)
            ? bytes
            : 0;
}

status_t validate_projection(const group_matmul_projection_params &p) {
    if (p.output_size <= 0 || p.input_size <= 0 || p.weight == nullptr
            || p.ldb <= 0) {
        return status_t::op_bad_io;
    }
    const int stored_columns = p.trans_weight ? p.input_size : p.output_size;
    if (p.ldb < stored_columns) { return status_t::memory_bad_stride; }
    if (size_of(p.params.dtypes.src) == 0 || size_of(p.params.dtypes.wei) == 0
            || size_of(p.params.dtypes.dst) == 0) {
        return status_t::memory_bad_quant;
    }
    if (p.bias != nullptr && size_of(p.params.dtypes.bias) == 0) {
        return status_t::memory_bad_quant;
    }
    if (p.beta != 0.0f || p.params.mem_format_b == 'r') {
        return status_t::unimplemented;
    }
    if ((p.params.dtypes.compute == data_type_t::s8
                || p.params.dtypes.compute == data_type_t::u8)
            && (p.params.dtypes.wei == data_type_t::s8
                    || p.params.dtypes.wei == data_type_t::s4
                    || p.params.dtypes.wei == data_type_t::u4)
            && p.params.quant_params.wei_scale.buff == nullptr) {
        return status_t::memory_bad_quant;
    }
    return status_t::success;
}

bool has_token_dependent_postop(const group_matmul_projection_params &p,
        const int num_tokens, const int num_experts) {
    for (const auto &postop : p.params.postop_) {
        if (postop.buff != nullptr && postop.dims.size() >= 2
                && postop.dims[0] == num_tokens
                && postop.dims[0] != num_experts) {
            return true;
        }
    }
    return false;
}

status_t validate_common(const char layout_src, const bool trans_src,
        const void *token_src, const int token_src_ld, const int num_tokens,
        const int num_experts, const int topk, void *moe_output,
        const int moe_output_ld, const group_matmul_projection_params &primary,
        const group_matmul_routing_params &routing,
        const group_matmul_projection_params *secondary,
        const grp_matmul_gated_act_params *gated_act, int &final_width) {
    if (layout_src != 'r' && layout_src != 'R' && layout_src != 'c'
            && layout_src != 'C') {
        return status_t::op_bad_io;
    }
    if (token_src == nullptr || moe_output == nullptr || num_tokens <= 0
            || num_experts <= 0 || topk <= 0 || routing.topk_ids == nullptr) {
        return status_t::op_bad_io;
    }
    if (!routing.skip_weighted && routing.topk_weights == nullptr) {
        return status_t::op_bad_io;
    }
    if (routing.skip_weighted && topk != 1) { return status_t::unimplemented; }
    if ((routing.expert_map == nullptr) != (routing.expert_map_size == 0)) {
        return status_t::op_bad_io;
    }
    if (routing.expert_map_size < 0) { return status_t::op_bad_io; }
    const int ids_stride
            = routing.topk_ids_stride != 0 ? routing.topk_ids_stride : topk;
    const int weights_stride = routing.topk_weights_stride != 0
            ? routing.topk_weights_stride
            : topk;
    if (ids_stride < topk
            || (!routing.skip_weighted && weights_stride < topk)) {
        return status_t::memory_bad_stride;
    }

    status_t status = validate_projection(primary);
    if (status != status_t::success) { return status; }
    if (secondary != nullptr) {
        status = validate_projection(*secondary);
        if (status != status_t::success) { return status; }
    }
    if (has_token_dependent_postop(primary, num_tokens, num_experts)
            || (secondary != nullptr
                    && has_token_dependent_postop(
                            *secondary, num_tokens, num_experts))) {
        return status_t::unimplemented;
    }

    const bool row_major = layout_src == 'r' || layout_src == 'R';
    const int required_ld = row_major
            ? (trans_src ? num_tokens : primary.input_size)
            : (trans_src ? primary.input_size : num_tokens);
    if (token_src_ld < required_ld) { return status_t::memory_bad_stride; }

    const auto act = gated_act != nullptr ? gated_act->act
                                          : grp_matmul_gated_act_t::none;
    int primary_result_width = primary.output_size;
    if (act != grp_matmul_gated_act_t::none) {
        if (primary.output_size % 2 != 0) { return status_t::memory_bad_size; }
        primary_result_width /= 2;
    }
    if (secondary != nullptr && secondary->input_size != primary_result_width) {
        return status_t::memory_bad_size;
    }
    final_width = secondary != nullptr ? secondary->output_size
                                       : primary_result_width;
    if (moe_output_ld < final_width) { return status_t::memory_bad_stride; }

    if (secondary != nullptr
            && primary.params.dtypes.dst != secondary->params.dtypes.src) {
        return status_t::memory_bad_quant;
    }
    return status_t::success;
}

status_t resolve_routes(const int num_tokens, const int num_experts,
        const int topk, const group_matmul_routing_params &routing,
        std::vector<int> &local_experts, std::vector<int> &counts) {
    const int ids_stride
            = routing.topk_ids_stride != 0 ? routing.topk_ids_stride : topk;
    const int id_limit = routing.expert_map != nullptr ? routing.expert_map_size
                                                       : num_experts;
    size_t num_slots = 0;
    if (!checked_mul_size(static_cast<size_t>(num_tokens),
                static_cast<size_t>(topk), num_slots)) {
        return status_t::memory_bad_size;
    }
    local_experts.resize(num_slots);
    counts.assign(static_cast<size_t>(num_experts), 0);
    for (int m = 0; m < num_tokens; ++m) {
        const int32_t *ids
                = routing.topk_ids + static_cast<size_t>(m) * ids_stride;
        for (int k = 0; k < topk; ++k) {
            const int32_t id = ids[k];
            if (id < 0 || id >= id_limit) { return status_t::memory_bad_index; }
            const int local = routing.expert_map != nullptr
                    ? routing.expert_map[id]
                    : id;
            if (local >= num_experts) { return status_t::memory_bad_index; }
            local_experts[static_cast<size_t>(m) * topk + k] = local;
            if (local >= 0) { ++counts[static_cast<size_t>(local)]; }
        }
    }
    return status_t::success;
}

void copy_logical_source_row(uint8_t *dst, const uint8_t *src,
        const char layout_src, const bool trans_src, const int token,
        const int width, const int token_src_ld, const size_t element_size) {
    const bool row_major = layout_src == 'r' || layout_src == 'R';
    if (row_major && !trans_src) {
        std::memcpy(dst,
                src + static_cast<size_t>(token) * token_src_ld * element_size,
                static_cast<size_t>(width) * element_size);
        return;
    }
    for (int k = 0; k < width; ++k) {
        size_t index = 0;
        if (row_major) {
            index = static_cast<size_t>(k) * token_src_ld + token;
        } else if (!trans_src) {
            index = static_cast<size_t>(k) * token_src_ld + token;
        } else {
            index = static_cast<size_t>(token) * token_src_ld + k;
        }
        std::memcpy(dst + static_cast<size_t>(k) * element_size,
                src + index * element_size, element_size);
    }
}

status_t build_grouped_tokens(const char layout_src, const bool trans_src,
        const void *token_src, const int token_src_ld, const int num_tokens,
        const int num_experts, const int topk,
        const group_matmul_projection_params &primary,
        const data_type_t backing_dtype, const std::vector<int> &local_experts,
        const std::vector<int> &counts, grouped_tokens_t &grouped) {
    grouped.active_experts.clear();
    grouped.expert_order.clear();
    grouped.counts.clear();

    auto &expert_to_active = grouped.expert_to_active;
    expert_to_active.assign(static_cast<size_t>(num_experts), -1);
    for (int e = 0; e < num_experts; ++e) {
        if (counts[static_cast<size_t>(e)] == 0) { continue; }
        expert_to_active[static_cast<size_t>(e)]
                = static_cast<int>(grouped.active_experts.size());
        grouped.active_experts.push_back(e);
        grouped.expert_order.push_back(e);
        grouped.counts.push_back(counts[static_cast<size_t>(e)]);
    }
    for (int e = 0; e < num_experts; ++e) {
        if (counts[static_cast<size_t>(e)] == 0) {
            grouped.expert_order.push_back(e);
        }
    }

    const size_t active = grouped.active_experts.size();
    const size_t source_element_size
            = static_cast<size_t>(size_of(primary.params.dtypes.src));
    const size_t backing_element_size
            = static_cast<size_t>(size_of(backing_dtype));
    size_t live_rows = std::accumulate(grouped.counts.begin(),
            grouped.counts.end(), static_cast<size_t>(0));
    size_t source_row_bytes = 0;
    size_t backing_row_bytes = 0;
    size_t total_bytes = 0;
    if (source_element_size == 0 || backing_element_size == 0
            || source_element_size > backing_element_size
            || !checked_mul_size(static_cast<size_t>(primary.input_size),
                    source_element_size, source_row_bytes)
            || !checked_mul_size(static_cast<size_t>(primary.input_size),
                    backing_element_size, backing_row_bytes)
            || !checked_mul_size(live_rows, backing_row_bytes, total_bytes)) {
        return status_t::memory_bad_size;
    }
    if (!grouped.storage.reserve(total_bytes)) {
        return status_t::memory_bad_storage;
    }
    grouped.src.resize(active);
    grouped.row_base.resize(active + 1);
    reserve_elements(grouped.token_rows, live_rows);
    // Every slot below is assigned exactly once, on both the routed and the
    // dropped branch, so a reused tail can never be observed.
    grouped.slots.resize(local_experts.size());

    uint8_t *const storage = grouped.storage.data();
    size_t base = 0;
    for (size_t a = 0; a < active; ++a) {
        grouped.row_base[a] = base;
        grouped.src[a] = storage + base * backing_row_bytes;
        base += static_cast<size_t>(grouped.counts[a]);
    }
    grouped.row_base[active] = base;

    // Pass 1 assigns each routed slot its grouped row.  It is pure integer
    // bookkeeping over num_tokens * topk and stays serial because the row
    // cursors are order-dependent.
    grouped.cursors.assign(active, 0);
    for (int m = 0; m < num_tokens; ++m) {
        for (int k = 0; k < topk; ++k) {
            const size_t slot = static_cast<size_t>(m) * topk + k;
            const int expert = local_experts[slot];
            if (expert < 0) {
                grouped.slots[slot] = slot_ref_t {};
                continue;
            }
            const size_t a = static_cast<size_t>(
                    expert_to_active[static_cast<size_t>(expert)]);
            const int row = grouped.cursors[a]++;
            grouped.slots[slot] = {static_cast<int>(a), row};
            grouped.token_rows[grouped.row_base[a] + static_cast<size_t>(row)]
                    = m;
        }
    }

    // Pass 2 materializes the rows.  This is the bulk of the fallback's
    // non-GEMM cost -- a top-k-expanded copy of the whole token matrix -- and
    // it is embarrassingly parallel once pass 1 has fixed the row map.  The
    // grouped call that follows is itself the top-level parallel region, so
    // this opens one of its own rather than nesting inside anything.
    const auto *source = static_cast<const uint8_t *>(token_src);
    const auto copy_rows = [&](size_t first, size_t last) {
        // row_base is ascending, so the owning expert only ever moves forward.
        size_t a
                = static_cast<size_t>(std::upper_bound(grouped.row_base.begin(),
                                              grouped.row_base.end(), first)
                        - grouped.row_base.begin() - 1);
        for (size_t r = first; r < last; ++r) {
            while (r >= grouped.row_base[a + 1]) {
                ++a;
            }
            copy_logical_source_row(storage
                            + grouped.row_base[a] * backing_row_bytes
                            + (r - grouped.row_base[a]) * source_row_bytes,
                    source, layout_src, trans_src, grouped.token_rows[r],
                    primary.input_size, token_src_ld, source_element_size);
        }
    };

    size_t gathered_bytes = 0;
    const int threads
            = routed_moe::effective_num_threads(primary.params.num_threads);
    if (threads > 1 && live_rows > 1
            && checked_mul_size(live_rows, source_row_bytes, gathered_bytes)
            && gathered_bytes >= kParallelCopyBytes) {
        const int team = static_cast<int>(
                std::min<size_t>(static_cast<size_t>(threads), live_rows));
#pragma omp parallel num_threads(team)
        {
            const int count = omp_get_num_threads();
            const int rank = omp_get_thread_num();
            const size_t first = live_rows * static_cast<size_t>(rank)
                    / static_cast<size_t>(count);
            const size_t last = live_rows * static_cast<size_t>(rank + 1)
                    / static_cast<size_t>(count);
            copy_rows(first, last);
        }
    } else {
        copy_rows(0, live_rows);
    }
    return status_t::success;
}

status_t expert_weight_offset(const group_matmul_projection_params &projection,
        const int expert, size_t &offset) {
    const size_t rows = static_cast<size_t>(projection.trans_weight
                    ? projection.output_size
                    : projection.input_size);
    size_t expert_bytes = 0;
    if (projection.params.packing.pack_format_b == 1) {
        constexpr size_t ggml_group_size = 32;
        if (projection.ldb % static_cast<int>(ggml_group_size) != 0) {
            return status_t::memory_bad_size;
        }
        size_t block_bytes = 0;
        if (projection.params.dtypes.wei == data_type_t::s8) {
            block_bytes = sizeof(uint16_t) + 32 * sizeof(int8_t);
        } else if (projection.params.dtypes.wei == data_type_t::s4) {
            block_bytes = sizeof(uint16_t) + 16 * sizeof(uint8_t);
        } else {
            return status_t::unimplemented;
        }
        size_t blocks = 0;
        if (!checked_mul_size(rows,
                    static_cast<size_t>(projection.ldb) / ggml_group_size,
                    blocks)
                || !checked_mul_size(blocks, block_bytes, expert_bytes)) {
            return status_t::memory_bad_size;
        }
    } else {
        size_t elements = 0;
        if (!checked_mul_size(
                    rows, static_cast<size_t>(projection.ldb), elements)) {
            return status_t::memory_bad_size;
        }
        expert_bytes
                = packed_elements_bytes(elements, projection.params.dtypes.wei);
        if (expert_bytes == 0) { return status_t::memory_bad_size; }
    }
    const size_t expert_stride = projection.wei_buffer_capacity_bytes != 0
            ? projection.wei_buffer_capacity_bytes
            : expert_bytes;
    if (expert_stride < expert_bytes) { return status_t::memory_bad_stride; }
    if (!checked_mul_size(expert_stride, static_cast<size_t>(expert), offset)) {
        return status_t::memory_bad_size;
    }
    return status_t::success;
}

status_t expert_bias_offset(const group_matmul_projection_params &projection,
        const int expert, size_t &offset) {
    if (projection.bias == nullptr) {
        offset = 0;
        return status_t::success;
    }
    size_t bytes = 0;
    if (!checked_mul_size(static_cast<size_t>(projection.output_size),
                static_cast<size_t>(size_of(projection.params.dtypes.bias)),
                bytes)
            || !checked_mul_size(bytes, static_cast<size_t>(expert), offset)) {
        return status_t::memory_bad_size;
    }
    return status_t::success;
}

/**
 * @brief Where expert @p expert starts inside an expert-major quant tensor.
 *
 * Shared by the in-place slicer and the fused down-projection slicer so both
 * agree on what "expert-major" means and neither has to copy a dims vector to
 * find out.  @p expert_major stays false for tensors that are broadcast across
 * experts; those are passed through untouched.
 */
status_t expert_quant_offset(
        const matmul_quantization_params_t::matmul_quant_t &q, const int expert,
        const int num_experts, bool &expert_major, size_t &byte_offset) {
    expert_major = false;
    byte_offset = 0;
    if (q.buff == nullptr || q.dims.size() < 2 || q.dims[0] != num_experts) {
        return status_t::success;
    }
    if (size_of(q.dt) == 0) { return status_t::memory_bad_quant; }
    size_t per_expert = 0;
    if (!checked_product(q.dims, 1, per_expert)) {
        return status_t::memory_bad_size;
    }
    if (!checked_mul_size(
                per_expert, static_cast<size_t>(size_of(q.dt)), byte_offset)
            || !checked_mul_size(
                    byte_offset, static_cast<size_t>(expert), byte_offset)) {
        return status_t::memory_bad_size;
    }
    expert_major = true;
    return status_t::success;
}

status_t slice_expert_quant(matmul_quantization_params_t::matmul_quant_t &q,
        const int expert, const int num_experts, const int output_size) {
    bool expert_major = false;
    size_t byte_offset = 0;
    const status_t status = expert_quant_offset(
            q, expert, num_experts, expert_major, byte_offset);
    if (status != status_t::success || !expert_major) { return status; }
    q.buff = static_cast<const uint8_t *>(q.buff) + byte_offset;
    q.dims.erase(q.dims.begin());
    if (q.dims.size() == 1 && q.dims[0] == output_size) {
        q.dims.insert(q.dims.begin(), 1);
    }
    return status_t::success;
}

/// Same slice, written into a caller-owned down-projection descriptor so the
/// fused path never copy-constructs (and then discards) a dims vector per
/// expert.  @p out keeps its capacity across calls.
status_t slice_expert_quant_into(
        const matmul_quantization_params_t::matmul_quant_t &q, const int expert,
        const int num_experts, const int output_size,
        grp_matmul_fused_moe_params::down_weight_quant_t &out) {
    bool expert_major = false;
    size_t byte_offset = 0;
    const status_t status = expert_quant_offset(
            q, expert, num_experts, expert_major, byte_offset);
    if (status != status_t::success) { return status; }
    out.dt = q.dt;
    out.dims.clear();
    if (!expert_major) {
        out.buff = q.buff;
        out.dims.insert(out.dims.end(), q.dims.begin(), q.dims.end());
        return status_t::success;
    }
    out.buff = static_cast<const uint8_t *>(q.buff) + byte_offset;
    out.dims.insert(out.dims.end(), q.dims.begin() + 1, q.dims.end());
    if (out.dims.size() == 1 && out.dims[0] == output_size) {
        out.dims.insert(out.dims.begin(), 1);
    }
    return status_t::success;
}

status_t gather_source_quant(
        const matmul_quantization_params_t::matmul_quant_t &q,
        const std::vector<int> &token_rows, const size_t rows,
        const int num_tokens, byte_scratch_t &storage) {
    if (q.dims.empty() || q.dims[0] != num_tokens) { return status_t::success; }
    if (size_of(q.dt) == 0) { return status_t::memory_bad_quant; }
    size_t row_elements = 0;
    if (!checked_product(q.dims, 1, row_elements)) {
        return status_t::memory_bad_size;
    }
    size_t row_bytes = 0;
    if (!checked_mul_size(
                row_elements, static_cast<size_t>(size_of(q.dt)), row_bytes)) {
        return status_t::memory_bad_size;
    }
    size_t total_bytes = 0;
    if (!checked_mul_size(rows, row_bytes, total_bytes)) {
        return status_t::memory_bad_size;
    }
    if (!storage.reserve(total_bytes)) { return status_t::memory_bad_storage; }
    uint8_t *const destination = storage.data();
    if (q.buff == nullptr) {
        // The caller left the source-scale buffer for the library to fill.
        // The grouped surface is never zero-filled, so define it here.
        std::memset(destination, 0, total_bytes);
        return status_t::success;
    }
    const auto *source = static_cast<const uint8_t *>(q.buff);
    for (size_t row = 0; row < rows; ++row) {
        std::memcpy(destination + row * row_bytes,
                source + static_cast<size_t>(token_rows[row]) * row_bytes,
                row_bytes);
    }
    return status_t::success;
}

status_t slice_expert_postops(std::vector<matmul_post_op> &postops,
        const int expert, const int num_experts) {
    for (auto &postop : postops) {
        if (postop.buff == nullptr || postop.dims.size() < 2
                || postop.dims[0] != num_experts) {
            continue;
        }
        if (size_of(postop.dtype) == 0) { return status_t::memory_bad_quant; }
        size_t per_expert = 0;
        size_t byte_offset = 0;
        if (!checked_product(postop.dims, 1, per_expert)
                || !checked_mul_size(per_expert,
                        static_cast<size_t>(size_of(postop.dtype)), byte_offset)
                || !checked_mul_size(byte_offset, static_cast<size_t>(expert),
                        byte_offset)) {
            return status_t::memory_bad_size;
        }
        postop.buff = static_cast<uint8_t *>(postop.buff) + byte_offset;
        postop.dims.erase(postop.dims.begin());
    }
    return status_t::success;
}

status_t build_projection_params(
        const group_matmul_projection_params &projection,
        const grouped_tokens_t &grouped, const int num_experts,
        const int num_tokens, std::vector<matmul_params> &params,
        quant_scratch_t &scratch) {
    // `assign`, not `resize`: the vector is reused across calls, so a shorter
    // request must still re-seed every surviving element from `projection`.
    matmul_params grouped_params = projection.params;
    grouped_params.wei_buffer_capacity_bytes
            = projection.wei_buffer_capacity_bytes;
    params.assign(grouped.active_experts.size(), grouped_params);
    const size_t live_rows = grouped.row_base.back();
    status_t status = gather_source_quant(
            projection.params.quant_params.src_scale, grouped.token_rows,
            live_rows, num_tokens, scratch.src_scale);
    if (status != status_t::success) { return status; }

    // Gathered source quant buffers are one contiguous expert-major surface.
    // Assign each active expert its own subrange and row count.
    size_t scale_row_elements = 0;
    const auto &original_scale = projection.params.quant_params.src_scale;
    const bool token_scale = !original_scale.dims.empty()
            && original_scale.dims[0] == num_tokens;
    if (token_scale
            && !checked_product(original_scale.dims, 1, scale_row_elements)) {
        return status_t::memory_bad_size;
    }
    size_t scale_row_bytes = 0;
    if (token_scale
            && !checked_mul_size(scale_row_elements,
                    static_cast<size_t>(size_of(original_scale.dt)),
                    scale_row_bytes)) {
        return status_t::memory_bad_size;
    }

    const auto &original_zp = projection.params.quant_params.src_zp;
    size_t zp_row_elements = 0;
    const bool token_zp
            = !original_zp.dims.empty() && original_zp.dims[0] == num_tokens;
    if (token_zp
            && (!checked_product(original_zp.dims, 1, zp_row_elements)
                    || !checked_mul_size(zp_row_elements,
                            static_cast<size_t>(size_of(original_zp.dt)),
                            zp_row_elements))) {
        return status_t::memory_bad_size;
    }
    if (token_zp) {
        status = gather_source_quant(projection.params.quant_params.src_zp,
                grouped.token_rows, live_rows, num_tokens, scratch.src_zp);
        if (status != status_t::success) { return status; }
    }

    size_t scale_offset = 0;
    size_t zp_offset = 0;
    for (size_t a = 0; a < params.size(); ++a) {
        auto &p = params[a];
        const int expert = grouped.active_experts[a];
        status = slice_expert_quant(p.quant_params.wei_scale, expert,
                num_experts, projection.output_size);
        if (status != status_t::success) { return status; }
        status = slice_expert_quant(p.quant_params.wei_zp, expert, num_experts,
                projection.output_size);
        if (status != status_t::success) { return status; }
        status = slice_expert_quant(p.quant_params.dst_scale, expert,
                num_experts, projection.output_size);
        if (status != status_t::success) { return status; }
        status = slice_expert_quant(p.quant_params.dst_zp, expert, num_experts,
                projection.output_size);
        if (status != status_t::success) { return status; }
        status = slice_expert_postops(p.postop_, expert, num_experts);
        if (status != status_t::success) { return status; }

        if (token_scale) {
            auto &q = p.quant_params.src_scale;
            q.dims[0] = grouped.counts[a];
            q.buff = scratch.src_scale.data() + scale_offset;
            scale_offset
                    += static_cast<size_t>(grouped.counts[a]) * scale_row_bytes;
        }
        if (token_zp) {
            auto &q = p.quant_params.src_zp;
            q.dims[0] = grouped.counts[a];
            q.buff = scratch.src_zp.data() + zp_offset;
            zp_offset
                    += static_cast<size_t>(grouped.counts[a]) * zp_row_elements;
        }
        p.mem_format_a = 'n';
        p.active_matmul = 0;
        p.total_matmul = 0;
    }
    if (!params.empty()) {
        params[0].active_matmul = static_cast<uint32_t>(params.size());
        params[0].total_matmul = static_cast<uint32_t>(num_experts);
    }
    return status_t::success;
}

status_t run_projection(const group_matmul_projection_params &projection,
        const grouped_tokens_t &grouped, const int num_experts,
        const int num_tokens, const std::vector<const void *> &src,
        const int lda, const grp_matmul_gated_act_params *gated_act,
        descriptor_scratch_t &desc, projection_result_t &result) {
    const size_t active = grouped.active_experts.size();
    if (active == 0) { return status_t::success; }
    const size_t total = static_cast<size_t>(num_experts);
    const size_t dst_size
            = static_cast<size_t>(size_of(projection.params.dtypes.dst));
    const size_t total_rows = grouped.row_base.back();
    size_t elements = 0;
    size_t bytes = 0;
    if (!checked_mul_size(total_rows,
                static_cast<size_t>(projection.output_size), elements)
            || !checked_mul_size(elements, dst_size, bytes)) {
        return status_t::memory_bad_size;
    }
    if (!result.storage.reserve(bytes)) { return status_t::memory_bad_storage; }
    result.dst.resize(active);
    result.ldc.assign(active, projection.output_size);

    desc.layouts.assign(active, 'r');
    desc.trans_a.assign(active, false);
    desc.trans_b.assign(total, projection.trans_weight);
    desc.m.assign(grouped.counts.begin(), grouped.counts.end());
    desc.n.assign(total, projection.output_size);
    desc.k.assign(total, projection.input_size);
    desc.alpha.assign(active, projection.alpha);
    desc.lda.assign(active, lda);
    desc.weights.resize(total);
    desc.ldb.assign(total, projection.ldb);
    desc.biases.assign(active, nullptr);
    desc.beta.assign(active, projection.beta);
    desc.is_const.assign(total, projection.weight_is_const);

    uint8_t *const storage = result.storage.data();
    size_t dst_offset = 0;
    for (size_t a = 0; a < active; ++a) {
        result.dst[a] = storage + dst_offset;
        dst_offset += static_cast<size_t>(grouped.counts[a])
                * projection.output_size * dst_size;

        size_t bias_offset = 0;
        status_t status = expert_bias_offset(
                projection, grouped.active_experts[a], bias_offset);
        if (status != status_t::success) { return status; }
        if (projection.bias != nullptr) {
            desc.biases[a] = static_cast<const uint8_t *>(projection.bias)
                    + bias_offset;
        }
    }
    for (size_t slot = 0; slot < total; ++slot) {
        size_t weight_offset = 0;
        status_t status = expert_weight_offset(
                projection, grouped.expert_order[slot], weight_offset);
        if (status != status_t::success) { return status; }
        desc.weights[slot] = static_cast<const uint8_t *>(projection.weight)
                + weight_offset;
    }

    status_t status = build_projection_params(projection, grouped, num_experts,
            num_tokens, desc.params, result.quant);
    if (status != status_t::success) { return status; }

    return group_matmul_direct(desc.layouts, desc.trans_a, desc.trans_b, desc.m,
            desc.n, desc.k, desc.alpha, src, desc.lda, desc.weights, desc.ldb,
            desc.biases, desc.beta, result.dst, result.ldc, desc.is_const,
            desc.params, nullptr, gated_act, nullptr);
}

bool can_use_fused_grouped(const group_matmul_projection_params &primary,
        const group_matmul_projection_params &secondary,
        const group_matmul_routing_params &routing) {
    const auto &p1 = primary.params;
    const auto &p2 = secondary.params;
    return routing.reduce_output
            && primary.trans_weight == secondary.trans_weight
            && secondary.alpha == 1.0f && secondary.beta == 0.0f
            && primary.input_size >= secondary.output_size
            && p1.dtypes.dst == p2.dtypes.src && p2.dtypes.dst == p1.dtypes.dst
            && (p1.dtypes.src == p1.dtypes.dst
                    || (p1.dtypes.src == data_type_t::s8
                            && p1.dtypes.dst == data_type_t::bf16))
            && p1.dtypes.wei == p2.dtypes.wei
            && p1.dtypes.compute == p2.dtypes.compute
            && p1.mem_format_b == p2.mem_format_b
            && p1.packing.pack_format_b == p2.packing.pack_format_b
            && p1.lowoha_algo == p2.lowoha_algo
            && routed_moe::effective_num_threads(p1.num_threads)
            == routed_moe::effective_num_threads(p2.num_threads)
            && p2.postop_.empty();
}

status_t run_fused_grouped(const group_matmul_projection_params &primary,
        const group_matmul_projection_params &secondary,
        const grouped_tokens_t &grouped, const int num_experts,
        const int num_tokens, const int topk, void *moe_output,
        const int moe_output_ld, const group_matmul_routing_params &routing,
        const grp_matmul_gated_act_params *gated_act,
        descriptor_scratch_t &desc, projection_result_t &scratch_result) {
    const size_t active = grouped.active_experts.size();
    const size_t total = static_cast<size_t>(num_experts);
    desc.layouts.assign(active, 'r');
    desc.trans_a.assign(active, false);
    desc.trans_b.assign(total, primary.trans_weight);
    desc.m.assign(grouped.counts.begin(), grouped.counts.end());
    desc.n.assign(total, primary.output_size);
    desc.k.assign(total, primary.input_size);
    desc.alpha.assign(active, primary.alpha);
    desc.lda.assign(active, primary.input_size);
    desc.weights.resize(total);
    desc.ldb.assign(total, primary.ldb);
    desc.biases.assign(active, nullptr);
    desc.beta.assign(active, primary.beta);
    desc.dst.assign(active, nullptr);
    desc.ldc.assign(active, primary.output_size);
    desc.is_const.assign(total, primary.weight_is_const);

    grp_matmul_fused_moe_params &fused = desc.fused;
    fused.down_weight.resize(total);
    fused.N_down.assign(total, secondary.output_size);
    fused.ldb_down.assign(total, secondary.ldb);
    fused.down_wei_buffer_capacity_bytes = secondary.wei_buffer_capacity_bytes;
    fused.bias_down.assign(active, nullptr);
    fused.bias_dt_down = secondary.params.dtypes.bias;
    // The optional vectors are read as "present when non-empty", so a reused
    // descriptor has to drop them explicitly when this call does not use them.
    if (secondary.params.quant_params.wei_scale.buff != nullptr) {
        fused.down_scale.resize(active);
    } else {
        fused.down_scale.clear();
    }
    if (secondary.params.quant_params.wei_zp.buff != nullptr) {
        fused.down_zp.resize(active);
    } else {
        fused.down_zp.clear();
    }
    const bool prequantized_source
            = primary.params.dtypes.src == data_type_t::s8
            && primary.params.dtypes.dst == data_type_t::bf16;
    if (prequantized_source) {
        fused.dst_down.resize(active);
        fused.ldc_down.assign(active, primary.input_size);
        for (size_t a = 0; a < active; ++a) {
            fused.dst_down[a] = const_cast<void *>(grouped.src[a]);
        }
    } else {
        fused.dst_down.clear();
        fused.ldc_down.clear();
    }

    for (size_t a = 0; a < active; ++a) {
        const int expert = grouped.active_experts[a];
        size_t offset = 0;
        status_t status = expert_bias_offset(primary, expert, offset);
        if (status != status_t::success) { return status; }
        if (primary.bias != nullptr) {
            desc.biases[a]
                    = static_cast<const uint8_t *>(primary.bias) + offset;
        }

        status = expert_bias_offset(secondary, expert, offset);
        if (status != status_t::success) { return status; }
        if (secondary.bias != nullptr) {
            fused.bias_down[a]
                    = static_cast<const uint8_t *>(secondary.bias) + offset;
        }

        if (!fused.down_scale.empty()) {
            status = slice_expert_quant_into(
                    secondary.params.quant_params.wei_scale, expert,
                    num_experts, secondary.output_size, fused.down_scale[a]);
            if (status != status_t::success) { return status; }
        }
        if (!fused.down_zp.empty()) {
            status = slice_expert_quant_into(
                    secondary.params.quant_params.wei_zp, expert, num_experts,
                    secondary.output_size, fused.down_zp[a]);
            if (status != status_t::success) { return status; }
        }
    }
    for (size_t slot = 0; slot < total; ++slot) {
        const int expert = grouped.expert_order[slot];
        size_t offset = 0;
        status_t status = expert_weight_offset(primary, expert, offset);
        if (status != status_t::success) { return status; }
        desc.weights[slot]
                = static_cast<const uint8_t *>(primary.weight) + offset;
        status = expert_weight_offset(secondary, expert, offset);
        if (status != status_t::success) { return status; }
        fused.down_weight[slot]
                = static_cast<const uint8_t *>(secondary.weight) + offset;
    }

    status_t status = build_projection_params(primary, grouped, num_experts,
            num_tokens, desc.params, scratch_result.quant);
    if (status != status_t::success) { return status; }

    const size_t output_element_size
            = static_cast<size_t>(size_of(secondary.params.dtypes.dst));
    const size_t zero_bytes
            = static_cast<size_t>(secondary.output_size) * output_element_size;
    // Grows only, and nothing ever writes through it, so it stays all-zero.
    if (desc.zero_row.size() < zero_bytes) { desc.zero_row.resize(zero_bytes); }
    desc.row_ptrs.assign(grouped.slots.size(), desc.zero_row.data());
    for (size_t slot = 0; slot < grouped.slots.size(); ++slot) {
        const slot_ref_t ref = grouped.slots[slot];
        if (ref.active < 0) { continue; }
        desc.row_ptrs[slot]
                = static_cast<const uint8_t *>(
                          grouped.src[static_cast<size_t>(ref.active)])
                + static_cast<size_t>(ref.row) * primary.input_size
                        * output_element_size;
    }

    const float *route_weights = routing.topk_weights;
    if (!routing.skip_weighted && routing.topk_weights_stride != 0
            && routing.topk_weights_stride != topk) {
        reserve_elements(
                desc.dense_weights, static_cast<size_t>(num_tokens) * topk);
        for (int token = 0; token < num_tokens; ++token) {
            std::memcpy(desc.dense_weights.data()
                            + static_cast<size_t>(token) * topk,
                    routing.topk_weights
                            + static_cast<size_t>(token)
                                    * routing.topk_weights_stride,
                    static_cast<size_t>(topk) * sizeof(float));
        }
        route_weights = desc.dense_weights.data();
    }

    group_matmul_moe_postop_params postop;
    postop.num_tokens = num_tokens;
    postop.topk = topk;
    postop.output = moe_output;
    postop.ldc_output = moe_output_ld;
    postop.topk_weights = route_weights;
    postop.skip_weighted = routing.skip_weighted;
    postop.row_ptrs = desc.row_ptrs.data();

    return group_matmul_direct(desc.layouts, desc.trans_a, desc.trans_b, desc.m,
            desc.n, desc.k, desc.alpha, grouped.src, desc.lda, desc.weights,
            desc.ldb, desc.biases, desc.beta, desc.dst, desc.ldc, desc.is_const,
            desc.params, &postop, gated_act, &fused);
}

bool no_quant_buffer(const matmul_quantization_params_t::matmul_quant_t &q) {
    return q.buff == nullptr;
}

bool tight_expert_scale(const matmul_quantization_params_t::matmul_quant_t &q,
        const int num_experts, const int output_size) {
    return q.buff != nullptr
            && (q.dt == data_type_t::f32 || q.dt == data_type_t::bf16)
            && ((q.dims.size() == 2 && q.dims[0] == num_experts
                        && q.dims[1] == output_size)
                    || (q.dims.size() == 3 && q.dims[0] == num_experts
                            && q.dims[1] == 1 && q.dims[2] == output_size));
}

bool fast_reduction_extent_is_valid(const int extent) {
    return extent > 0 && extent % routed_moe::block_n == 0
            && extent <= routed_moe::max_gemm_reduction;
}

bool fast_weight_capacity_is_valid(
        const group_matmul_projection_params &projection) {
    if (projection.wei_buffer_capacity_bytes == 0) { return true; }
    const size_t element_size
            = static_cast<size_t>(size_of(projection.params.dtypes.wei));
    size_t rows = static_cast<size_t>(projection.trans_weight
                    ? projection.output_size
                    : projection.input_size);
    size_t logical_elements = 0;
    size_t logical_bytes = 0;
    return element_size != 0
            && checked_mul_size(
                    rows, static_cast<size_t>(projection.ldb), logical_elements)
            && checked_mul_size(logical_elements, element_size, logical_bytes)
            && projection.wei_buffer_capacity_bytes >= logical_bytes
            && projection.wei_buffer_capacity_bytes % element_size == 0
            && projection.wei_buffer_capacity_bytes / element_size
            <= static_cast<size_t>(std::numeric_limits<int64_t>::max());
}

bool tight_expert_group_scale(
        const matmul_quantization_params_t::matmul_quant_t &q,
        const int num_experts, const int input_size, const int output_size,
        int64_t &group_size) {
    if (q.buff == nullptr
            || (q.dt != data_type_t::f32 && q.dt != data_type_t::bf16)
            || q.dims.size() != 3 || q.dims[0] != num_experts || q.dims[1] <= 0
            || q.dims[2] != output_size
            || input_size % static_cast<int>(q.dims[1]) != 0) {
        return false;
    }
    group_size = input_size / q.dims[1];
    return group_size > 0 && group_size % custom_kernel::kS4Octet == 0;
}

bool fast_common_candidate(const char layout_src, const bool trans_src,
        const int num_experts, const group_matmul_projection_params &primary,
        const group_matmul_routing_params &routing,
        const group_matmul_projection_params *secondary,
        const grp_matmul_gated_act_params *gated_act) {
    if (layout_src != 'r' || trans_src || secondary == nullptr
            || gated_act == nullptr
            || (gated_act->act != grp_matmul_gated_act_t::silu_and_mul
                    && gated_act->act != grp_matmul_gated_act_t::gelu_and_mul)
            || !routing.reduce_output || routing.skip_weighted
            || primary.bias != nullptr || secondary->bias != nullptr
            || !primary.trans_weight || !secondary->trans_weight
            || !primary.weight_is_const || !secondary->weight_is_const
            || primary.alpha != 1.0f || primary.beta != 0.0f
            || secondary->alpha != 1.0f || secondary->beta != 0.0f) {
        return false;
    }
    if (primary.output_size % 2 != 0
            || secondary->input_size != primary.output_size / 2
            || secondary->output_size != primary.input_size
            || !fast_reduction_extent_is_valid(primary.input_size)
            || !fast_reduction_extent_is_valid(secondary->input_size)
            || primary.ldb != primary.input_size
            || secondary->ldb != secondary->input_size
            || primary.params.mem_format_b != 'n'
            || secondary->params.mem_format_b != 'n'
            || primary.params.packing.pack_format_b != 0
            || secondary->params.packing.pack_format_b != 0
            || !primary.params.postop_.empty()
            || !secondary->params.postop_.empty()
            || !fast_weight_capacity_is_valid(primary)
            || !fast_weight_capacity_is_valid(*secondary)) {
        return false;
    }
    return routed_moe::effective_num_threads(primary.params.num_threads)
            == routed_moe::effective_num_threads(secondary->params.num_threads)
            && num_experts > 0;
}

bool fast_w8_candidate(const char layout_src, const bool trans_src,
        const int num_experts, const group_matmul_projection_params &primary,
        const group_matmul_routing_params &routing,
        const group_matmul_projection_params *secondary,
        const grp_matmul_gated_act_params *gated_act) {
    if (!fast_common_candidate(layout_src, trans_src, num_experts, primary,
                routing, secondary, gated_act)) {
        return false;
    }
    const auto &p1 = primary.params;
    const auto &p2 = secondary->params;
    if (p1.dtypes.src != data_type_t::bf16 || p1.dtypes.wei != data_type_t::s8
            || p1.dtypes.dst != data_type_t::bf16
            || p1.dtypes.compute != data_type_t::s8
            || p2.dtypes.src != data_type_t::bf16
            || p2.dtypes.wei != data_type_t::s8
            || p2.dtypes.dst != data_type_t::bf16
            || p2.dtypes.compute != data_type_t::s8 || !p1.dynamic_quant
            || !p2.dynamic_quant
            || !tight_expert_scale(
                    p1.quant_params.wei_scale, num_experts, primary.output_size)
            || !tight_expert_scale(p2.quant_params.wei_scale, num_experts,
                    secondary->output_size)
            || p1.quant_params.wei_scale.dt != p2.quant_params.wei_scale.dt
            || !no_quant_buffer(p1.quant_params.wei_zp)
            || !no_quant_buffer(p2.quant_params.wei_zp)
            || effective_weight_cache_type(p1.weight_cache_type) == 0
            || effective_weight_cache_type(p2.weight_cache_type) == 0) {
        return false;
    }
    return true;
}

bool fast_w4_candidate(const char layout_src, const bool trans_src,
        const int num_tokens, const int num_experts,
        const group_matmul_projection_params &primary,
        const group_matmul_routing_params &routing,
        const group_matmul_projection_params *secondary,
        const grp_matmul_gated_act_params *gated_act, int64_t &gate_group_size,
        int64_t &down_group_size) {
    if (!fast_common_candidate(layout_src, trans_src, num_experts, primary,
                routing, secondary, gated_act)
            || primary.wei_buffer_capacity_bytes != 0
            || secondary->wei_buffer_capacity_bytes != 0
            || primary.input_size % routed_moe::block_n != 0
            || secondary->input_size % routed_moe::block_n != 0
            || primary.input_size > routed_moe::max_gemm_reduction
            || secondary->input_size > routed_moe::max_gemm_reduction) {
        return false;
    }
    const auto &p1 = primary.params;
    const auto &p2 = secondary->params;
    const auto per_token_scale
            = [num_tokens](
                      const matmul_quantization_params_t::matmul_quant_t &q) {
        return q.dt == data_type_t::f32 && q.dims.size() == 2
                && q.dims[0] == num_tokens && q.dims[1] == 1;
    };
    return p1.dtypes.src == data_type_t::bf16
            && p1.dtypes.wei == data_type_t::s4
            && p1.dtypes.dst == data_type_t::bf16
            && p1.dtypes.compute == data_type_t::s8
            && p2.dtypes.src == data_type_t::bf16
            && p2.dtypes.wei == data_type_t::s4
            && p2.dtypes.dst == data_type_t::bf16
            && p2.dtypes.compute == data_type_t::s8 && p1.dynamic_quant
            && p2.dynamic_quant && per_token_scale(p1.quant_params.src_scale)
            && tight_expert_group_scale(p1.quant_params.wei_scale, num_experts,
                    primary.input_size, primary.output_size, gate_group_size)
            && tight_expert_group_scale(p2.quant_params.wei_scale, num_experts,
                    secondary->input_size, secondary->output_size,
                    down_group_size)
            && p1.quant_params.wei_scale.dt == p2.quant_params.wei_scale.dt
            && no_quant_buffer(p1.quant_params.wei_zp)
            && no_quant_buffer(p2.quant_params.wei_zp)
            && no_quant_buffer(p1.quant_params.src_zp)
            && no_quant_buffer(p2.quant_params.src_zp)
            && effective_weight_cache_type(p1.weight_cache_type) != 0
            && effective_weight_cache_type(p2.weight_cache_type) != 0;
}

routed_moe_params make_fast_params(const void *token_src,
        const int token_src_ld, const int num_tokens, const int num_experts,
        const int topk, void *moe_output, const int moe_output_ld,
        const group_matmul_projection_params &primary,
        const group_matmul_routing_params &routing,
        const group_matmul_projection_params &secondary,
        const grp_matmul_gated_act_t act, const bool w4,
        const int64_t gate_group_size = 0, const int64_t down_group_size = 0) {
    routed_moe_params p;
    p.num_tokens = num_tokens;
    p.hidden_size = primary.input_size;
    p.intermediate_size = secondary.input_size;
    p.num_local_experts = num_experts;
    p.topk = topk;
    p.src = token_src;
    p.src_stride = token_src_ld;
    p.src_dt = data_type_t::bf16;
    p.dst = moe_output;
    p.dst_stride = moe_output_ld;
    p.dst_dt = data_type_t::bf16;
    p.gate_up_weight = primary.weight;
    p.down_weight = secondary.weight;
    p.wei_dt = w4 ? data_type_t::s4 : data_type_t::s8;
    if (!w4 && primary.wei_buffer_capacity_bytes != 0) {
        p.gate_up_stride_expert = static_cast<int64_t>(
                primary.wei_buffer_capacity_bytes / sizeof(int8_t));
    }
    if (!w4 && secondary.wei_buffer_capacity_bytes != 0) {
        p.down_stride_expert = static_cast<int64_t>(
                secondary.wei_buffer_capacity_bytes / sizeof(int8_t));
    }
    p.gate_up_scale = primary.params.quant_params.wei_scale.buff;
    p.down_scale = secondary.params.quant_params.wei_scale.buff;
    p.scale_dt = primary.params.quant_params.wei_scale.dt;
    if (w4) {
        p.gate_up_group_size = gate_group_size;
        p.down_group_size = down_group_size;
    }
    p.topk_ids = routing.topk_ids;
    p.topk_ids_stride = routing.topk_ids_stride;
    p.topk_weights = routing.topk_weights;
    p.topk_weights_stride = routing.topk_weights_stride;
    p.expert_map = routing.expert_map;
    p.expert_map_size = routing.expert_map_size;
    p.activation = act == grp_matmul_gated_act_t::gelu_and_mul
            ? routed_moe_activation_t::gelu_and_mul
            : routed_moe_activation_t::silu_and_mul;
    p.quant_scheme = w4
            ? routed_moe_quant_t::sym_per_group_w4a8_dynamic_per_token
            : routed_moe_quant_t::sym_per_oc_w8a8_dynamic_per_token;
    p.num_threads = primary.params.num_threads;
    return p;
}

status_t finish_output(const grouped_tokens_t &grouped,
        const group_matmul_routing_params &routing, const int num_tokens,
        const int topk, void *moe_output, const int moe_output_ld,
        const int final_width, const data_type_t final_dtype,
        const projection_result_t &result, const int num_threads,
        descriptor_scratch_t &desc) {
    const size_t element_size = static_cast<size_t>(size_of(final_dtype));
    const size_t row_bytes = static_cast<size_t>(final_width) * element_size;
    // Grows only, and nothing ever writes through it, so it stays all-zero.
    if (desc.zero_row.size() < row_bytes) { desc.zero_row.resize(row_bytes); }
    desc.row_ptrs.assign(grouped.slots.size(), desc.zero_row.data());
    for (size_t slot = 0; slot < grouped.slots.size(); ++slot) {
        const slot_ref_t ref = grouped.slots[slot];
        if (ref.active < 0) { continue; }
        desc.row_ptrs[slot]
                = static_cast<const uint8_t *>(
                          result.dst[static_cast<size_t>(ref.active)])
                + static_cast<size_t>(ref.row)
                        * result.ldc[static_cast<size_t>(ref.active)]
                        * element_size;
    }

    if (!routing.reduce_output) {
        // Slot output is a straight num_tokens * topk row scatter; the reduced
        // path already runs its equivalent through the threaded MoE post-op,
        // so match it here instead of leaving a serial copy of the whole
        // expanded result.
        auto *output = static_cast<uint8_t *>(moe_output);
        const size_t slots = desc.row_ptrs.size();
        const int threads = routed_moe::effective_num_threads(num_threads);
        size_t scattered_bytes = 0;
        if (threads > 1 && slots > 1
                && checked_mul_size(slots, row_bytes, scattered_bytes)
                && scattered_bytes >= kParallelCopyBytes) {
            const int team = static_cast<int>(
                    std::min<size_t>(static_cast<size_t>(threads), slots));
#pragma omp parallel for schedule(static) num_threads(team)
            for (long long slot = 0; slot < static_cast<long long>(slots);
                    ++slot) {
                std::memcpy(output
                                + static_cast<size_t>(slot) * moe_output_ld
                                        * element_size,
                        desc.row_ptrs[static_cast<size_t>(slot)], row_bytes);
            }
            return status_t::success;
        }
        for (size_t slot = 0; slot < slots; ++slot) {
            std::memcpy(output + slot * moe_output_ld * element_size,
                    desc.row_ptrs[slot], row_bytes);
        }
        return status_t::success;
    }

    const float *weights = routing.topk_weights;
    if (!routing.skip_weighted && routing.topk_weights_stride != 0
            && routing.topk_weights_stride != topk) {
        reserve_elements(
                desc.dense_weights, static_cast<size_t>(num_tokens) * topk);
        for (int m = 0; m < num_tokens; ++m) {
            std::memcpy(
                    desc.dense_weights.data() + static_cast<size_t>(m) * topk,
                    routing.topk_weights
                            + static_cast<size_t>(m)
                                    * routing.topk_weights_stride,
                    static_cast<size_t>(topk) * sizeof(float));
        }
        weights = desc.dense_weights.data();
    }
    group_matmul_moe_postop_params postop;
    postop.num_tokens = num_tokens;
    postop.topk = topk;
    postop.output = moe_output;
    postop.ldc_output = moe_output_ld;
    postop.topk_weights = weights;
    postop.skip_weighted = routing.skip_weighted;
    postop.row_ptrs = desc.row_ptrs.data();
    return group_matmul_moe_postop_execute(
            &postop, final_width, num_threads, final_dtype);
}

status_t run_generic(const char layout_src, const bool trans_src,
        const void *token_src, const int token_src_ld, const int num_tokens,
        const int num_experts, const int topk, void *moe_output,
        const int moe_output_ld, const group_matmul_projection_params &primary,
        const group_matmul_routing_params &routing,
        const group_matmul_projection_params *secondary,
        const grp_matmul_gated_act_params *gated_act, const int final_width,
        const std::vector<int> &local_experts, const std::vector<int> &counts) {
    generic_scratch_t &scratch = generic_scratch();
    group_matmul_projection_params &prequantized_primary
            = scratch.prequantized_primary;
    const group_matmul_projection_params *effective_primary = &primary;
    const void *effective_token_src = token_src;
    int effective_token_src_ld = token_src_ld;
    data_type_t backing_dtype = primary.params.dtypes.src;

    // Match the legacy ZenTorch DA8W8/W4A8 fallback: quantize each unique
    // token once, then duplicate compact S8 rows and their scales into expert
    // groups.  Besides avoiding top-k duplicate quantization, this normalizes
    // the source-scale dtype to the weight-scale dtype.  AOCL's symmetric
    // quantized GEMM requires those two scale-factor types to match (notably
    // for W4A8 with BF16 weight scales).
    const auto &p1 = primary.params;
    const bool can_prequantize_source = secondary != nullptr
            && (layout_src == 'r' || layout_src == 'R') && !trans_src
            && token_src_ld == primary.input_size
            && primary.input_size >= secondary->output_size
            && p1.dtypes.src == data_type_t::bf16
            && (p1.dtypes.wei == data_type_t::s8
                    || p1.dtypes.wei == data_type_t::s4)
            && p1.dtypes.dst == data_type_t::bf16
            && p1.dtypes.compute == data_type_t::s8 && p1.dynamic_quant
            && p1.packing.pack_format_b == 0
            && p1.quant_params.wei_scale.buff != nullptr
            && p1.quant_params.wei_zp.buff == nullptr
            && p1.quant_params.src_zp.buff == nullptr
            && p1.quant_params.src_scale.dims.size() == 2
            && p1.quant_params.src_scale.dims[0] == num_tokens
            && p1.quant_params.src_scale.dims[1] == 1
            && (p1.quant_params.wei_scale.dt == data_type_t::f32
                    || p1.quant_params.wei_scale.dt == data_type_t::bf16)
            && secondary->params.dtypes.src == data_type_t::bf16
            && secondary->params.dtypes.wei == p1.dtypes.wei
            && secondary->params.dtypes.dst == data_type_t::bf16
            && secondary->params.dtypes.compute == data_type_t::s8
            && secondary->params.dynamic_quant
            && secondary->params.quant_params.wei_scale.buff != nullptr
            && zendnnl::common::zendnnl_platform_info().get_avx512f_status()
            && zendnnl::common::zendnnl_platform_info()
                       .get_avx512_bw_vl_status();
    if (can_prequantize_source) {
        size_t quant_elements = 0;
        if (!checked_mul_size(static_cast<size_t>(num_tokens),
                    static_cast<size_t>(primary.input_size), quant_elements)) {
            return status_t::memory_bad_size;
        }
        reserve_elements(scratch.token_quant, quant_elements);
        reserve_elements(
                scratch.token_scale_f32, static_cast<size_t>(num_tokens));
        zendnnl::lowoha::reorder::dynamic_per_token_quant_bf16_s8_native(
                static_cast<const uint16_t *>(token_src),
                scratch.token_quant.data(), scratch.token_scale_f32.data(),
                num_tokens, primary.input_size);

        prequantized_primary = primary;
        auto &effective_params = prequantized_primary.params;
        effective_params.dtypes.src = data_type_t::s8;
        effective_params.dynamic_quant = false;
        auto &source_scale = effective_params.quant_params.src_scale;
        source_scale.dt = p1.quant_params.wei_scale.dt;
        source_scale.dims = {num_tokens, 1};
        if (source_scale.dt == data_type_t::bf16) {
            reserve_elements(
                    scratch.token_scale_bf16, static_cast<size_t>(num_tokens));
            for (int m = 0; m < num_tokens; ++m) {
                scratch.token_scale_bf16[static_cast<size_t>(m)]
                        = zendnnl::lowoha::reorder::float_to_bf16(
                                scratch.token_scale_f32[static_cast<size_t>(
                                        m)]);
            }
            source_scale.buff = scratch.token_scale_bf16.data();
        } else {
            source_scale.buff = scratch.token_scale_f32.data();
        }
        effective_primary = &prequantized_primary;
        effective_token_src = scratch.token_quant.data();
        effective_token_src_ld = primary.input_size;
        backing_dtype = primary.params.dtypes.dst;
    }

    grouped_tokens_t &grouped = scratch.grouped;
    status_t status = build_grouped_tokens(
            can_prequantize_source ? 'r' : layout_src,
            can_prequantize_source ? false : trans_src, effective_token_src,
            effective_token_src_ld, num_tokens, num_experts, topk,
            *effective_primary, backing_dtype, local_experts, counts, grouped);
    if (status != status_t::success) { return status; }
    if (grouped.active_experts.empty()) {
        const size_t element_size = static_cast<size_t>(
                size_of(secondary != nullptr ? secondary->params.dtypes.dst
                                             : primary.params.dtypes.dst));
        const size_t rows = routing.reduce_output
                ? static_cast<size_t>(num_tokens)
                : static_cast<size_t>(num_tokens) * topk;
        for (size_t row = 0; row < rows; ++row) {
            std::memset(static_cast<uint8_t *>(moe_output)
                            + row * moe_output_ld * element_size,
                    0, static_cast<size_t>(final_width) * element_size);
        }
        return status_t::success;
    }

    descriptor_scratch_t &descriptors = scratch.descriptors;
    if (secondary != nullptr
            && can_use_fused_grouped(*effective_primary, *secondary, routing)) {
        return run_fused_grouped(*effective_primary, *secondary, grouped,
                num_experts, num_tokens, topk, moe_output, moe_output_ld,
                routing, gated_act, descriptors, scratch.primary);
    }

    projection_result_t &primary_result = scratch.primary;
    status = run_projection(*effective_primary, grouped, num_experts,
            num_tokens, grouped.src, primary.input_size, gated_act, descriptors,
            primary_result);
    if (status != status_t::success) { return status; }

    const projection_result_t *final_result = &primary_result;
    projection_result_t &secondary_result = scratch.secondary;
    if (secondary != nullptr) {
        scratch.secondary_src.assign(
                primary_result.dst.begin(), primary_result.dst.end());
        status = run_projection(*secondary, grouped, num_experts, num_tokens,
                scratch.secondary_src, primary.output_size, nullptr,
                descriptors, secondary_result);
        if (status != status_t::success) { return status; }
        final_result = &secondary_result;
    }

    const data_type_t final_dtype = secondary != nullptr
            ? secondary->params.dtypes.dst
            : primary.params.dtypes.dst;
    return finish_output(grouped, routing, num_tokens, topk, moe_output,
            moe_output_ld, final_width, final_dtype, *final_result,
            primary.params.num_threads, descriptors);
}

} // namespace

status_t routed_fused_moe_direct(const char layout_src, const bool trans_src,
        const void *token_src, const int token_src_ld, const int num_tokens,
        const int num_experts, const int topk, void *moe_output,
        const int moe_output_ld, const group_matmul_projection_params &primary,
        const group_matmul_routing_params &routing,
        const group_matmul_projection_params *secondary,
        const grp_matmul_gated_act_params *gated_act) {
    int final_width = 0;
    status_t status = validate_common(layout_src, trans_src, token_src,
            token_src_ld, num_tokens, num_experts, topk, moe_output,
            moe_output_ld, primary, routing, secondary, gated_act, final_width);
    if (status != status_t::success) { return status; }

    // Validate ids before either implementation touches expert-indexed data.
    // Both arrays live in the per-thread scratch so the routing pass does not
    // allocate on a path the framework calls once per MoE layer per token.
    generic_scratch_t &scratch = generic_scratch();
    std::vector<int> &local_experts = scratch.local_experts;
    std::vector<int> &counts = scratch.counts;
    status = resolve_routes(
            num_tokens, num_experts, topk, routing, local_experts, counts);
    if (status != status_t::success) { return status; }

    // Sister line to [GRP_MATMUL.*]: names the executor that ran this call.
    static const bool s_route_log = error_handling::apilog_info_enabled();
    const auto act_name = [&]() {
        const auto act = gated_act != nullptr ? gated_act->act
                                              : grp_matmul_gated_act_t::none;
        switch (act) {
            case grp_matmul_gated_act_t::silu_and_mul: return "silu_and_mul";
            case grp_matmul_gated_act_t::gelu_and_mul: return "gelu_and_mul";
            case grp_matmul_gated_act_t::swiglu_oai_mul:
                return "swiglu_oai_mul";
            default: return "none";
        }
    };

    const char *generic_reason = "disabled";
    if (routed_moe_enabled()) {
        generic_reason = "format";
        int64_t gate_group_size = 0;
        int64_t down_group_size = 0;
        const bool w8 = fast_w8_candidate(layout_src, trans_src, num_experts,
                primary, routing, secondary, gated_act);
        const bool w4 = !w8
                && fast_w4_candidate(layout_src, trans_src, num_tokens,
                        num_experts, primary, routing, secondary, gated_act,
                        gate_group_size, down_group_size);
        if (w8 || w4) {
            const auto fast_params = make_fast_params(token_src, token_src_ld,
                    num_tokens, num_experts, topk, moe_output, moe_output_ld,
                    primary, routing, *secondary, gated_act->act, w4,
                    gate_group_size, down_group_size);
            status = routed_moe::execute(fast_params);
            if (status == status_t::success) {
                if (s_route_log) {
                    const bool w4_decode = w4
                            && *std::max_element(counts.begin(), counts.end())
                                    <= routed_moe::max_decode_expert_rows;
                    error_handling::apilog_info("[ROUTED_MOE] path=",
                            w4 ? (w4_decode ? "native_w4_decode"
                                            : "native_w4_prefill")
                               : "fast",
                            " act=", act_name(), " tokens=", num_tokens,
                            " experts=", num_experts, " topk=", topk,
                            " hidden=", fast_params.hidden_size,
                            " intermediate=", fast_params.intermediate_size);
                }
                return status;
            }
            if (status != status_t::isa_unsupported
                    && status != status_t::unimplemented) {
                return status;
            }
            generic_reason = status == status_t::isa_unsupported
                    ? "isa_unsupported"
                    : "unimplemented";
        }
    }
    if (s_route_log) {
        error_handling::apilog_info("[ROUTED_MOE] path=generic reason=",
                generic_reason, " act=", act_name(), " tokens=", num_tokens,
                " experts=", num_experts, " topk=", topk);
    }
    return run_generic(layout_src, trans_src, token_src, token_src_ld,
            num_tokens, num_experts, topk, moe_output, moe_output_ld, primary,
            routing, secondary, gated_act, final_width, local_experts, counts);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
