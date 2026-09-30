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
 * @file routed_moe_execute.cpp
 * @brief Routed-MoE executor: packing cache, routing, scheduling, driver.
 *
 * The whole MoE block runs as one call so the quantized activation can
 * stay resident across output-channel blocks:
 *
 *   0.   quantize src [M, K] bf16 -> uint8 per token, keeping the scale
 *   1.   gate/up: for every (token-block, N-block) tile run a dual
 *        accumulator int8 VNNI GEMM against the two halves of the
 *        gate/up weight and fold act(gate) * up (SiLU or erf-GELU) into
 *        the epilogue,
 *        writing bf16 in *sorted* order
 *   1.5  requantize the [M*topk, N] intermediate per row
 *   2.   down projection, scaled by the router weight and scattered back
 *        to original (token, slot) order
 *   3.   reduce [M, topk, K] -> [M, K]
 *
 * Stage 1 writes in sorted order precisely so stage 2 reads its A
 * operand contiguously and never needs a gather.  The W4 executor fuses
 * stages 2 and 3 when every thread can own whole output-column blocks: each
 * thread adds the router-weighted rows of every routed block into an f32
 * [M, K] surface for its columns, so no [M, topk, K] buffer is written.
 *
 * This translation unit is framework-neutral: raw pointers, OpenMP, and
 * the AVX-512 primitives in routed_moe_kernels.hpp.
 */

#include "lowoha_operators/matmul/routed_moe/routed_moe_internal.hpp"

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/s4_microkernel.hpp"
#include "lowoha_operators/matmul/group_matmul/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/routed_moe/routed_moe_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>
#include <shared_mutex>
#include <unordered_map>

#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace routed_moe {

static status_t pack_weights_strided(const int8_t *src, int8_t *dst,
        int64_t num_experts, int64_t out_channels, int64_t in_channels,
        int64_t expert_stride, int64_t num_threads);

namespace {

// A routed execution holds a shared lock from validation through its final
// store.  Cache flush takes the exclusive side, so it cannot retire a pack
// (or return control to a model-unload caller) while any routed call is active.
std::shared_mutex &execution_lifecycle_mutex() {
    static std::shared_mutex mutex;
    return mutex;
}

#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED

// ---------------------------------------------------------------------------
// Packed-weight cache.
//
// Identity alone is insufficient: frameworks may deliberately use one
// explicit model key for multiple tensors or reload a different geometry
// under it.  Role, complete geometry, source expert stride, dtype and
// packed-layout version are part of equality, and an entry verifies all
// metadata and byte size on every hit.
// ---------------------------------------------------------------------------
enum class packed_tensor_role_t : uint8_t { gate_up = 0, down = 1 };

struct packed_cache_key_t {
    const void *identity = nullptr;
    const void *scale_identity = nullptr;
    packed_tensor_role_t role = packed_tensor_role_t::gate_up;
    int64_t num_experts = 0;
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    int64_t expert_stride = 0;
    int64_t group_size = 0;
    int32_t pack_nr = 0;
    bool interleave_split_halves = false;
    data_type_t dtype = data_type_t::none;
    uint32_t layout_version = 0;

    bool operator==(const packed_cache_key_t &other) const {
        return identity == other.identity
                && scale_identity == other.scale_identity && role == other.role
                && num_experts == other.num_experts
                && out_channels == other.out_channels
                && in_channels == other.in_channels
                && expert_stride == other.expert_stride
                && group_size == other.group_size && pack_nr == other.pack_nr
                && interleave_split_halves == other.interleave_split_halves
                && dtype == other.dtype
                && layout_version == other.layout_version;
    }
};

struct packed_cache_key_hash_t {
    size_t operator()(const packed_cache_key_t &key) const {
        size_t hash = std::hash<const void *> {}(key.identity);
        const auto combine = [&](size_t value) {
            hash ^= value + static_cast<size_t>(0x9e3779b9u) + (hash << 6)
                    + (hash >> 2);
        };
        combine(std::hash<const void *> {}(key.scale_identity));
        combine(std::hash<unsigned> {}(static_cast<unsigned>(key.role)));
        combine(std::hash<int64_t> {}(key.num_experts));
        combine(std::hash<int64_t> {}(key.out_channels));
        combine(std::hash<int64_t> {}(key.in_channels));
        combine(std::hash<int64_t> {}(key.expert_stride));
        combine(std::hash<int64_t> {}(key.group_size));
        combine(std::hash<int32_t> {}(key.pack_nr));
        combine(std::hash<bool> {}(key.interleave_split_halves));
        combine(std::hash<int32_t> {}(static_cast<int32_t>(key.dtype)));
        combine(std::hash<uint32_t> {}(key.layout_version));
        return hash;
    }
};

struct packed_weight_t {
    int8_t *data = nullptr;
    size_t bytes = 0;
    std::vector<float> scales;
    packed_cache_key_t key {};

    ~packed_weight_t() { zendnnl_aligned_free(data); }

    packed_weight_t() = default;
    packed_weight_t(const packed_weight_t &) = delete;
    packed_weight_t &operator=(const packed_weight_t &) = delete;

    bool matches(const packed_cache_key_t &expected, size_t expected_bytes,
            size_t expected_scales = 0) const {
        return data != nullptr && bytes == expected_bytes
                && scales.size() == expected_scales && key == expected;
    }
};

std::mutex &cache_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<packed_cache_key_t, std::shared_ptr<packed_weight_t>,
        packed_cache_key_hash_t> &
cache_store() {
    static std::unordered_map<packed_cache_key_t,
            std::shared_ptr<packed_weight_t>, packed_cache_key_hash_t>
            store;
    return store;
}

struct scale_cache_key_t {
    // The packed-weight/model key is the model-lifetime identity.
    // Quantization scales are immutable metadata owned by that weight and
    // share its flush lifecycle, so a transient scale tensor address must not
    // create a second conversion.
    const void *identity = nullptr;
    packed_tensor_role_t role = packed_tensor_role_t::gate_up;
    size_t elements = 0;
    int64_t num_experts = 0;
    int64_t groups = 0;
    int64_t out_channels = 0;
    data_type_t dtype = data_type_t::none;
    uint32_t layout_version = 0;

    bool operator==(const scale_cache_key_t &other) const {
        return identity == other.identity && role == other.role
                && elements == other.elements
                && num_experts == other.num_experts && groups == other.groups
                && out_channels == other.out_channels && dtype == other.dtype
                && layout_version == other.layout_version;
    }
};

struct scale_cache_key_hash_t {
    size_t operator()(const scale_cache_key_t &key) const {
        size_t hash = std::hash<const void *> {}(key.identity);
        const auto combine = [&](size_t value) {
            hash ^= value + static_cast<size_t>(0x9e3779b9u) + (hash << 6)
                    + (hash >> 2);
        };
        combine(std::hash<unsigned> {}(static_cast<unsigned>(key.role)));
        combine(std::hash<size_t> {}(key.elements));
        combine(std::hash<int64_t> {}(key.num_experts));
        combine(std::hash<int64_t> {}(key.groups));
        combine(std::hash<int64_t> {}(key.out_channels));
        combine(std::hash<int32_t> {}(static_cast<int32_t>(key.dtype)));
        combine(std::hash<uint32_t> {}(key.layout_version));
        return hash;
    }
};

struct converted_scale_t {
    float *data = nullptr;
    size_t elements = 0;
    scale_cache_key_t key {};

    ~converted_scale_t() { zendnnl_aligned_free(data); }

    converted_scale_t() = default;
    converted_scale_t(const converted_scale_t &) = delete;
    converted_scale_t &operator=(const converted_scale_t &) = delete;
};

std::unordered_map<scale_cache_key_t, std::shared_ptr<converted_scale_t>,
        scale_cache_key_hash_t> &
scale_cache_store() {
    static std::unordered_map<scale_cache_key_t,
            std::shared_ptr<converted_scale_t>, scale_cache_key_hash_t>
            store;
    return store;
}

bool checked_scale_elements(int64_t num_experts, int64_t groups,
        int64_t out_channels, size_t &elements) {
    if (num_experts <= 0 || groups <= 0 || out_channels <= 0) { return false; }
    size_t value = static_cast<size_t>(num_experts);
    if (static_cast<uint64_t>(groups)
            > std::numeric_limits<size_t>::max() / value) {
        return false;
    }
    value *= static_cast<size_t>(groups);
    if (static_cast<uint64_t>(out_channels)
            > std::numeric_limits<size_t>::max() / value) {
        return false;
    }
    elements = value * static_cast<size_t>(out_channels);
    return true;
}

status_t lookup_converted_scale(const void *identity, const void *scale,
        packed_tensor_role_t role, int64_t num_experts, int64_t groups,
        int64_t out_channels, data_type_t dtype, uint32_t layout_version,
        int num_threads, std::shared_ptr<const converted_scale_t> &result) {
    if (identity == nullptr || scale == nullptr || dtype != data_type_t::bf16) {
        return status_t::memory_bad_quant;
    }
    size_t elements = 0;
    if (!checked_scale_elements(num_experts, groups, out_channels, elements)) {
        return status_t::memory_bad_size;
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)
            || elements > static_cast<size_t>(
                       std::numeric_limits<long long>::max())) {
        return status_t::memory_bad_size;
    }
    scale_cache_key_t key;
    key.identity = identity;
    key.role = role;
    key.elements = elements;
    key.num_experts = num_experts;
    key.groups = groups;
    key.out_channels = out_channels;
    key.dtype = dtype;
    key.layout_version = layout_version;
    std::lock_guard<std::mutex> guard(cache_mutex());
    auto &store = scale_cache_store();
    const auto found = store.find(key);
    if (found != store.end()) {
        if (found->second == nullptr || found->second->data == nullptr
                || found->second->elements != elements
                || !(found->second->key == key)) {
            return status_t::memory_bad_size;
        }
        result = found->second;
        return status_t::success;
    }

    auto entry = std::make_shared<converted_scale_t>();
    void *raw = zendnnl_aligned_alloc(64, elements * sizeof(float));
    if (raw == nullptr) { return status_t::memory_bad_storage; }
    entry->data = static_cast<float *>(raw);
    entry->elements = elements;
    entry->key = key;
    const auto *src = static_cast<const uint16_t *>(scale);
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (long long i = 0; i < static_cast<long long>(elements); ++i) {
        const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(entry->data + i, &bits, sizeof(bits));
    }
    store.emplace(key, entry);
    result = std::move(entry);
    return status_t::success;
}

status_t lookup_packed_weight(const void *identity, packed_tensor_role_t role,
        const int8_t *weight, int64_t num_experts, int64_t out_channels,
        int64_t in_channels, int64_t expert_stride, data_type_t dtype,
        int num_threads, std::shared_ptr<const packed_weight_t> &result) {
    size_t src_bytes = 0;
    size_t packed_bytes = 0;
    int64_t blocks_per_expert = 0;
    int64_t packed_bytes_per_oc = 0;
    const status_t size_status = checked_pack_sizes(num_experts, out_channels,
            in_channels, src_bytes, packed_bytes, blocks_per_expert,
            packed_bytes_per_oc);
    if (size_status != status_t::success) { return size_status; }
    (void)src_bytes;
    (void)blocks_per_expert;
    (void)packed_bytes_per_oc;

    packed_cache_key_t key;
    key.identity = identity;
    key.role = role;
    key.num_experts = num_experts;
    key.out_channels = out_channels;
    key.in_channels = in_channels;
    key.expert_stride = expert_stride;
    key.dtype = dtype;
    key.layout_version = packed_layout_version;

    std::lock_guard<std::mutex> guard(cache_mutex());
    auto &store = cache_store();
    const auto found = store.find(key);
    if (found != store.end()) {
        if (found->second == nullptr
                || !found->second->matches(key, packed_bytes)) {
            return status_t::memory_bad_size;
        }
        result = found->second;
        return status_t::success;
    }

    auto entry = std::make_shared<packed_weight_t>();
    void *raw = zendnnl_aligned_alloc(64, packed_bytes);
    if (raw == nullptr) { return status_t::memory_bad_storage; }
    entry->data = static_cast<int8_t *>(raw);
    entry->bytes = packed_bytes;
    entry->key = key;
    const status_t pack_status = pack_weights_strided(weight, entry->data,
            num_experts, out_channels, in_channels, expert_stride, num_threads);
    if (pack_status != status_t::success) { return pack_status; }

    store.emplace(key, entry);
    result = std::move(entry);
    return status_t::success;
}

status_t lookup_packed_weight_s4(const void *identity,
        packed_tensor_role_t role, const int8_t *weight, const float *scale,
        int64_t num_experts, int64_t out_channels, int64_t in_channels,
        int64_t group_size, bool interleave_split_halves, int num_threads,
        std::shared_ptr<const packed_weight_t> &result) {
    constexpr int pack_nr = 32;
    const size_t bytes_per_expert = custom_kernel::packed_weight_size_s4(
            static_cast<int>(in_channels), static_cast<int>(out_channels),
            pack_nr, static_cast<int>(group_size));
    const int64_t groups = in_channels / group_size;
    if (bytes_per_expert == 0
            || static_cast<uint64_t>(num_experts)
                    > std::numeric_limits<size_t>::max() / bytes_per_expert) {
        return status_t::memory_bad_size;
    }
    const size_t packed_bytes
            = static_cast<size_t>(num_experts) * bytes_per_expert;

    size_t scale_count = static_cast<size_t>(num_experts);
    if (static_cast<uint64_t>(groups)
            > std::numeric_limits<size_t>::max() / scale_count) {
        return status_t::memory_bad_size;
    }
    scale_count *= static_cast<size_t>(groups);
    if (static_cast<uint64_t>(out_channels)
            > std::numeric_limits<size_t>::max() / scale_count) {
        return status_t::memory_bad_size;
    }
    scale_count *= static_cast<size_t>(out_channels);

    packed_cache_key_t key;
    key.identity = identity;
    key.scale_identity = scale;
    key.role = role;
    key.num_experts = num_experts;
    key.out_channels = out_channels;
    key.in_channels = in_channels;
    key.group_size = group_size;
    key.pack_nr = pack_nr;
    key.interleave_split_halves = interleave_split_halves;
    key.dtype = data_type_t::s4;
    key.layout_version = packed_s4_layout_version;

    {
        std::lock_guard<std::mutex> guard(cache_mutex());
        const auto found = cache_store().find(key);
        if (found != cache_store().end()) {
            if (found->second == nullptr
                    || !found->second->matches(
                            key, packed_bytes, scale_count)) {
                return status_t::memory_bad_size;
            }
            result = found->second;
            return status_t::success;
        }
    }

    // Pack outside the map lock. Different layers and projections may cold
    // start concurrently; only insertion is serialized, and a duplicate
    // loser releases its private shared_ptr.
    auto entry = std::make_shared<packed_weight_t>();
    void *raw = zendnnl_aligned_alloc(64, packed_bytes);
    if (raw == nullptr) { return status_t::memory_bad_storage; }
    entry->data = static_cast<int8_t *>(raw);
    entry->bytes = packed_bytes;
    entry->key = key;
    entry->scales.resize(scale_count);

    const size_t raw_bytes_per_expert = static_cast<size_t>(out_channels)
            * static_cast<size_t>(in_channels) / 2;
    std::atomic<bool> failed {false};
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int64_t e = 0; e < num_experts; ++e) {
        const status_t packed = custom_kernel::prepack_weight_into_s4(
                weight + static_cast<size_t>(e) * raw_bytes_per_expert,
                static_cast<int>(in_channels), static_cast<int>(out_channels),
                static_cast<int>(in_channels), pack_nr, /*transB=*/true,
                interleave_split_halves, static_cast<int>(group_size),
                entry->data + static_cast<size_t>(e) * bytes_per_expert);
        if (packed != status_t::success) {
            failed.store(true, std::memory_order_relaxed);
            continue;
        }

        float *dst_scale = entry->scales.data()
                + static_cast<size_t>(e * groups * out_channels);
        const float *src_scale = scale + e * groups * out_channels;
        const int64_t half = out_channels / 2;
        for (int64_t g = 0; g < groups; ++g) {
            for (int64_t c = 0; c < out_channels; ++c) {
                const int64_t canonical = interleave_split_halves
                        ? ((c & 1) != 0 ? half + (c >> 1) : (c >> 1))
                        : c;
                dst_scale[g * out_channels + c]
                        = src_scale[g * out_channels + canonical];
            }
        }
    }
    if (failed.load(std::memory_order_relaxed)) { return status_t::failure; }

    std::lock_guard<std::mutex> guard(cache_mutex());
    auto &store = cache_store();
    const auto found = store.find(key);
    if (found != store.end()) {
        if (found->second == nullptr
                || !found->second->matches(key, packed_bytes, scale_count)) {
            return status_t::memory_bad_size;
        }
        result = found->second;
        return status_t::success;
    }
    store.emplace(key, entry);
    result = std::move(entry);
    return status_t::success;
}

// ---------------------------------------------------------------------------
// Scratch buffers, reused across calls.
//
// Sized to the largest problem this thread has executed, which is the same
// high-water policy the fused-MoE arena uses.  The executor is documented as
// one inference stream per calling thread, so thread_local both enforces that
// and keeps every buffer private without a lock.
// ---------------------------------------------------------------------------
struct scratch_t {
    std::vector<int32_t> routing;
    std::vector<uint8_t> aq;
    std::vector<float> as;
    std::vector<float> weights;
    std::vector<uint8_t> a_tile;
    std::vector<float> c_tile;
    std::vector<uint16_t> gate_up_out;
    std::vector<uint16_t> down_out;
    std::vector<int64_t> tile_cost_prefix;
    std::vector<float> reduce_acc;

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

// ---------------------------------------------------------------------------
// Sorted / padded routing.
//
//   sorted_ids  flat (token, slot) indices grouped by expert, each expert's
//               run starting at a multiple of block_m
//   expert_ids  expert owning block mb
//   offsets     start of block mb in *compacted* order, so
//               offsets[mb + 1] - offsets[mb] is the block's live row count
//
// Because stage 1 writes its output at offsets[mb], the intermediate ends up
// contiguous per block and stage 2 can read A directly.
//
// numel here is num_tokens * topk (tens of rows at decode), so this is built
// serially: the whole routine is a few hundred operations and a parallel
// version would pay more in barriers than it saves.
//
// Slots whose expert is not resident on this rank (expert_map entry < 0) are
// dropped from the sorted order and recorded in `inactive`, so their
// contribution is zeroed rather than computed against a wrong expert.
// ---------------------------------------------------------------------------
struct routing_t {
    int32_t *sorted_ids = nullptr;
    int32_t *expert_ids = nullptr;
    int32_t *offsets = nullptr;
    int32_t *inactive = nullptr;
    int64_t num_inactive = 0;
    int64_t num_blocks = 0;
};

status_t build_routing(const routed_moe_params &p,
        const checked_problem_sizes_t &sizes, routing_t &r) {
    const int64_t numel = sizes.numel;
    const int64_t num_experts = p.num_local_experts;
    const int64_t id_stride
            = p.topk_ids_stride != 0 ? p.topk_ids_stride : p.topk;
    const int64_t max_padded = sizes.max_padded;
    const int64_t max_blocks = sizes.max_blocks;

    scratch_t &sc = scratch();
    int32_t *base = scratch_t::reserve(sc.routing, sizes.routing_elements);

    r.sorted_ids = base;
    r.expert_ids = r.sorted_ids + max_padded;
    r.offsets = r.expert_ids + max_blocks;
    r.inactive = r.offsets + (max_blocks + 1);
    int32_t *count = r.inactive + numel;
    int32_t *cursor = count + num_experts;

    std::memset(count, 0, static_cast<size_t>(num_experts) * sizeof(int32_t));
    r.num_inactive = 0;

    // Resolve a routing id to a local expert index, or to a negative value
    // when the slot's expert is not resident on this rank.  Returns false on
    // an out-of-range id -- the executor turns that into memory_bad_index
    // rather than indexing off the end of the histogram.
    const auto resolve = [&](int32_t id, int64_t &local) {
        if (p.expert_map != nullptr) {
            if (id < 0 || static_cast<int64_t>(id) >= p.expert_map_size) {
                return false;
            }
            local = p.expert_map[id];
        } else {
            if (id < 0 || static_cast<int64_t>(id) >= num_experts) {
                return false;
            }
            local = id;
        }
        return local < num_experts;
    };

    // First pass: histogram the live slots per expert and collect the slots
    // that have no local expert.
    for (int64_t m = 0; m < p.num_tokens; ++m) {
        const int32_t *row = p.topk_ids + m * id_stride;
        for (int64_t t = 0; t < p.topk; ++t) {
            int64_t local = 0;
            if (!resolve(row[t], local)) { return status_t::memory_bad_index; }
            if (local < 0) {
                r.inactive[r.num_inactive++]
                        = static_cast<int32_t>(m * p.topk + t);
                continue;
            }
            count[local]++;
        }
    }

    int64_t nb = 0;
    int64_t pad = 0;
    int64_t off = 0;
    for (int64_t e = 0; e < num_experts; ++e) {
        const int64_t c = count[e];
        cursor[e] = static_cast<int32_t>(pad);
        if (c == 0) { continue; }
        const int64_t nblk = div_up(c, block_m);
        for (int64_t j = 0; j < nblk; ++j) {
            r.expert_ids[nb] = static_cast<int32_t>(e);
            r.offsets[nb] = static_cast<int32_t>(off);
            off += std::min(block_m, c - j * block_m);
            ++nb;
        }
        pad += nblk * block_m;
    }
    r.offsets[nb] = static_cast<int32_t>(off);
    r.num_blocks = nb;

    // Second pass: place each live slot's flat index into its expert's run.
    // The expert-map indirection is redone rather than cached, because
    // sorted_ids is the destination here and caching into it would alias.
    for (int64_t m = 0; m < p.num_tokens; ++m) {
        const int32_t *row = p.topk_ids + m * id_stride;
        for (int64_t t = 0; t < p.topk; ++t) {
            int64_t local = 0;
            if (!resolve(row[t], local)) { return status_t::memory_bad_index; }
            if (local < 0) { continue; }
            r.sorted_ids[cursor[local]++]
                    = static_cast<int32_t>(m * p.topk + t);
        }
    }
    return status_t::success;
}

// ---------------------------------------------------------------------------
// Tile scheduling.
//
// A near-square thread grid over (MB, NB) keeps each thread's working set
// small, and the N chunking keeps block_n * K of weight resident in L2 while
// sweeping every M block against it -- which is also what lets stage 1 load
// its A rows once per (M block, N chunk) instead of once per tile.
// ---------------------------------------------------------------------------
inline int64_t cache_blocks_for(int64_t chunk_bytes) {
    constexpr int64_t l2_half = 1024 * 1024;
    return std::max<int64_t>(1, l2_half / std::max<int64_t>(1, chunk_bytes));
}

template <typename Func>
inline void loop_2d(int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1,
        int64_t chunk_bytes, const Func &f) {
    const int64_t cb = cache_blocks_for(chunk_bytes);
    for (int64_t nbb = nb0; nbb < nb1; nbb += cb) {
        const int64_t nbe = std::min(nbb + cb, nb1);
        for (int64_t mb = mb0; mb < mb1; ++mb) {
            for (int64_t nb = nbb; nb < nbe; ++nb) {
                f(mb, nb, nb - nbb);
            }
        }
    }
}

bool decode_balance_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("ZENDNNL_ROUTED_MOE_DECODE_BALANCE");
        return value == nullptr || *value == '\0'
                || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool decode_columns_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("ZENDNNL_ROUTED_MOE_DECODE_COLUMNS");
        return value == nullptr || *value == '\0'
                || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Relative cost of one gate/up tile of an M block with `rows` live rows, in
// 1/16ths of the tile's weight stream: the micro-kernel streams the block from
// DRAM with its first row group and re-runs it from L2 for every further one,
// which costs about 1/16 of the stream each.
inline int64_t decode_tile_cost(int64_t rows) {
    return 15 + div_up(rows, gate_up_kernel_rows);
}

/// Contiguous M-block-major tile range [t0, t1) holding thread `tid`'s equal
/// share of a stage's cost.  `prefix[mb]` is the summed per-tile cost of
/// blocks [0, mb); every block has `nb_per_block` tiles.
inline void decode_tile_range(const int64_t *prefix, int64_t num_blocks,
        int64_t nb_per_block, int tid, int nthr, int64_t &t0, int64_t &t1) {
    const int64_t total = prefix[num_blocks] * nb_per_block;
    // First tile whose cost span starts at or after `cost`; num_blocks is a
    // few hundred at most, so a linear scan is cheaper than bookkeeping.
    const auto tile_at = [&](int64_t cost) {
        int64_t mb = 0;
        while (mb < num_blocks && prefix[mb + 1] * nb_per_block <= cost) {
            ++mb;
        }
        if (mb == num_blocks) { return num_blocks * nb_per_block; }
        const int64_t per_tile = prefix[mb + 1] - prefix[mb];
        return mb * nb_per_block
                + div_up(cost - prefix[mb] * nb_per_block, per_tile);
    };
    t0 = tile_at(total * tid / nthr);
    t1 = tile_at(total * (tid + 1) / nthr);
}

/// Near-square factorisation of `nthr` into (rows, cols) biased by the
/// aspect ratio of the (MB, NB) iteration space.
inline void thread_grid(int64_t MB, int64_t NB, int nthr, int &gm, int &gn) {
    gm = std::max(1,
            static_cast<int>(std::ceil(std::sqrt(static_cast<double>(MB)
                    / static_cast<double>(NB) * nthr))));
    for (; gm > 1; --gm) {
        if (nthr % gm == 0) { break; }
    }
    gn = nthr / gm;
}

/// Rows of one block split into ceil(m / max_mr) microkernel tiles of
/// near-equal height (32 -> 6,6,5,5,5,5, not 6,6,6,6,6,2).  Every row's
/// arithmetic is independent of its tile, so the split never changes results.
inline int balanced_tile_rows(int64_t m_size, int64_t tiles, int64_t tile) {
    return static_cast<int>(m_size / tiles + (tile < m_size % tiles ? 1 : 0));
}

bool checked_scratch_elements(int64_t a, int64_t b, int64_t c, size_t &result) {
    const uint64_t size_limit
            = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    if (a < 0 || b < 0 || c < 0 || static_cast<uint64_t>(a) > size_limit
            || static_cast<uint64_t>(b) > size_limit
            || static_cast<uint64_t>(c) > size_limit) {
        return false;
    }
    size_t value = static_cast<size_t>(a);
    const size_t ub = static_cast<size_t>(b);
    const size_t uc = static_cast<size_t>(c);
    if ((ub != 0 && value > std::numeric_limits<size_t>::max() / ub)) {
        return false;
    }
    value *= ub;
    if (uc != 0 && value > std::numeric_limits<size_t>::max() / uc) {
        return false;
    }
    result = value * uc;
    return true;
}

status_t execute_w4(
        const routed_moe_params &p, const checked_problem_sizes_t &sizes) {
    constexpr int pack_nr = 32;
    constexpr int compact_cols_per_gate_block = pack_nr / 2;
    constexpr int s4_max_mr = static_cast<int>(max_s4_kernel_rows);

    const int64_t M = p.num_tokens;
    const int64_t K = p.hidden_size;
    const int64_t N = p.intermediate_size;
    const int64_t topk = p.topk;
    const int64_t numel = sizes.numel;
    const int64_t gate_up_oc = sizes.gate_up_oc;
    const int nth = effective_num_threads(p.num_threads);
    // Every thread must own at least one 32-column output block.
    const bool fused_reduction = K / block_n >= nth;

    const auto *src = static_cast<const uint16_t *>(p.src);
    auto *dst = static_cast<uint16_t *>(p.dst);

    const int64_t gate_groups = K / p.gate_up_group_size;
    const int64_t down_groups = N / p.down_group_size;

    const void *gate_up_key = p.gate_up_cache_key != nullptr
            ? p.gate_up_cache_key
            : p.gate_up_weight;
    const void *down_key
            = p.down_cache_key != nullptr ? p.down_cache_key : p.down_weight;
    const float *gate_up_scale = nullptr;
    const float *down_scale = nullptr;
    std::shared_ptr<const converted_scale_t> converted_gate_up;
    std::shared_ptr<const converted_scale_t> converted_down;
    if (p.scale_dt == data_type_t::f32) {
        gate_up_scale = static_cast<const float *>(p.gate_up_scale);
        down_scale = static_cast<const float *>(p.down_scale);
    } else {
        status_t scale_status = lookup_converted_scale(gate_up_key,
                p.gate_up_scale, packed_tensor_role_t::gate_up,
                p.num_local_experts, gate_groups, gate_up_oc, p.scale_dt,
                packed_s4_layout_version, nth, converted_gate_up);
        if (scale_status != status_t::success) { return scale_status; }
        scale_status = lookup_converted_scale(down_key, p.down_scale,
                packed_tensor_role_t::down, p.num_local_experts, down_groups, K,
                p.scale_dt, packed_s4_layout_version, nth, converted_down);
        if (scale_status != status_t::success) { return scale_status; }
        gate_up_scale = converted_gate_up->data;
        down_scale = converted_down->data;
    }

    std::shared_ptr<const packed_weight_t> pw_gate_up;
    std::shared_ptr<const packed_weight_t> pw_down;
    status_t cache_status = lookup_packed_weight_s4(gate_up_key,
            packed_tensor_role_t::gate_up,
            static_cast<const int8_t *>(p.gate_up_weight), gate_up_scale,
            p.num_local_experts, gate_up_oc, K, p.gate_up_group_size,
            /*interleave_split_halves=*/true, nth, pw_gate_up);
    if (cache_status != status_t::success) { return cache_status; }
    cache_status = lookup_packed_weight_s4(down_key, packed_tensor_role_t::down,
            static_cast<const int8_t *>(p.down_weight), down_scale,
            p.num_local_experts, K, N, p.down_group_size,
            /*interleave_split_halves=*/false, nth, pw_down);
    if (cache_status != status_t::success) { return cache_status; }

    const size_t gate_bytes_per_expert = custom_kernel::packed_weight_size_s4(
            static_cast<int>(K), static_cast<int>(gate_up_oc), pack_nr,
            static_cast<int>(p.gate_up_group_size));
    const size_t down_bytes_per_expert = custom_kernel::packed_weight_size_s4(
            static_cast<int>(N), static_cast<int>(K), pack_nr,
            static_cast<int>(p.down_group_size));
    const size_t gate_oblock_bytes
            = static_cast<size_t>(K / custom_kernel::kS4Octet) * pack_nr
                    * custom_kernel::kS4BytesPerOctetCol
            + static_cast<size_t>(gate_groups) * pack_nr * sizeof(int32_t);
    const size_t down_oblock_bytes
            = static_cast<size_t>(N / custom_kernel::kS4Octet) * pack_nr
                    * custom_kernel::kS4BytesPerOctetCol
            + static_cast<size_t>(down_groups) * pack_nr * sizeof(int32_t);

    const custom_kernel::ActKind gate_act
            = p.activation == routed_moe_activation_t::gelu_and_mul
            ? custom_kernel::ActKind::gelu_and_mul
            : custom_kernel::ActKind::silu_and_mul;
    std::array<custom_kernel::s4_ukernel_fn_t, s4_max_mr + 1> gate_kfn {};
    std::array<custom_kernel::s4_ukernel_fn_t, s4_max_mr + 1> down_kfn {};
    for (int mr = 1; mr <= s4_max_mr; ++mr) {
        gate_kfn[static_cast<size_t>(mr)] = custom_kernel::select_s4_ukernel(mr,
                /*NV=*/2, gate_act, custom_kernel::S4SourceKind::kU8Zp128,
                custom_kernel::S4OutputKind::kBf16);
        down_kfn[static_cast<size_t>(mr)] = custom_kernel::select_s4_ukernel(mr,
                /*NV=*/2, custom_kernel::ActKind::none,
                custom_kernel::S4SourceKind::kU8Zp128,
                custom_kernel::S4OutputKind::kF32);
        if (gate_kfn[static_cast<size_t>(mr)] == nullptr
                || down_kfn[static_cast<size_t>(mr)] == nullptr) {
            return status_t::unimplemented;
        }
    }

    routing_t r;
    const status_t build_status = build_routing(p, sizes, r);
    if (build_status != status_t::success) { return build_status; }
    const int64_t MB = r.num_blocks;
    if (MB == 0) {
        for (int64_t m = 0; m < M; ++m) {
            std::memset(dst + m * p.dst_stride, 0,
                    static_cast<size_t>(K) * sizeof(uint16_t));
        }
        if (test_api::s_capture_native_w4.load(std::memory_order_relaxed)) {
            test_api::s_last_w4_reduction_path.store(
                    test_api::w4_reduction_path_t::all_inactive,
                    std::memory_order_relaxed);
        }
        return status_t::success;
    }

    scratch_t &sc = scratch();
    size_t a_tile_elements = 0;
    size_t c_tile_elements = 0;
    if (!checked_scratch_elements(nth, block_m, K, a_tile_elements)
            || !checked_scratch_elements(
                    nth, block_m, block_n, c_tile_elements)) {
        return status_t::memory_bad_size;
    }
    uint8_t *Aq = scratch_t::reserve(
            sc.aq, std::max(sizes.src_elements, sizes.intermediate_elements));
    float *As = scratch_t::reserve(sc.as, static_cast<size_t>(numel));
    uint8_t *A_tile = scratch_t::reserve(sc.a_tile, a_tile_elements);
    float *C_tile = scratch_t::reserve(sc.c_tile, c_tile_elements);
    uint16_t *gate_up_out
            = scratch_t::reserve(sc.gate_up_out, sizes.intermediate_elements);
    // Fused reduction: [M, K] f32 accumulator. Otherwise: [M, topk, K] bf16
    // per-slot output, reduced over topk afterwards.
    float *reduce_acc = nullptr;
    uint16_t *down_out = nullptr;
    if (fused_reduction) {
        reduce_acc = scratch_t::reserve(sc.reduce_acc, sizes.src_elements);
    } else {
        down_out = scratch_t::reserve(sc.down_out, sizes.down_elements);
    }

    const int64_t weight_stride
            = p.topk_weights_stride != 0 ? p.topk_weights_stride : topk;
    const float *router_weights = p.topk_weights;
    if (weight_stride != topk) {
        float *dense
                = scratch_t::reserve(sc.weights, static_cast<size_t>(numel));
        for (int64_t m = 0; m < M; ++m) {
            std::memcpy(dense + m * topk, p.topk_weights + m * weight_stride,
                    static_cast<size_t>(topk) * sizeof(float));
        }
        router_weights = dense;
    }
    // Inactive slots are absent from the sorted order, so the fused reduction
    // never adds them; the per-slot buffer needs them zeroed.
    for (int64_t i = 0; down_out != nullptr && i < r.num_inactive; ++i) {
        std::memset(down_out + static_cast<int64_t>(r.inactive[i]) * K, 0,
                static_cast<size_t>(K) * sizeof(uint16_t));
    }

    // Stage 0: each original token is quantized once. Routed copies below are
    // already U8+zp128, so the S4 kernel skips its grouped-path XOR.
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        quantize_row_u8(Aq + m * K, As[m], src + m * p.src_stride, K);
    }

    // Stage 1: W13. A 32-column packed block contains 16 interleaved
    // gate/up pairs, so two native NR32 calls produce one compact 32-column
    // intermediate tile while sharing the gathered A rows.
    {
        const int64_t NB = N / block_n;
        const int64_t chunk_bytes = block_n * K;
#pragma omp parallel num_threads(nth)
        {
            const int tid = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            int gm = 1;
            int gn = 1;
            thread_grid(MB, NB, nthr, gm, gn);
            const int im = tid / gn;
            const int in = tid % gn;
            const int64_t bm = div_up(MB, gm);
            const int64_t bn = div_up(NB, gn);
            const int64_t mb0 = im * bm;
            const int64_t mb1 = std::min(MB, mb0 + bm);
            const int64_t nb0 = in * bn;
            const int64_t nb1 = std::min(NB, nb0 + bn);

            if (mb0 < mb1 && nb0 < nb1) {
                uint8_t *ZENDNNL_ROUTED_RESTRICT A = A_tile + tid * block_m * K;
                alignas(64) float As_tile[block_m];
                loop_2d(mb0, mb1, nb0, nb1, chunk_bytes,
                        [&](int64_t mb, int64_t nb, int64_t nb_offset) {
                    const int64_t m_size = r.offsets[mb + 1] - r.offsets[mb];
                    const int32_t expert = r.expert_ids[mb];
                    const int32_t *A_ids = r.sorted_ids + mb * block_m;
                    if (nb_offset == 0) {
                        for (int64_t m = 0; m < m_size; ++m) {
                            const int64_t token = A_ids[m] / topk;
                            std::memcpy(A + m * K, Aq + token * K,
                                    static_cast<size_t>(K));
                            As_tile[m] = As[token];
                        }
                    }

                    const int64_t raw_nb = 2 * nb;
                    const int8_t *B0 = pw_gate_up->data
                            + static_cast<size_t>(expert)
                                    * gate_bytes_per_expert
                            + static_cast<size_t>(raw_nb) * gate_oblock_bytes;
                    const int8_t *B1 = B0 + gate_oblock_bytes;
                    const float *scale_base = pw_gate_up->scales.data()
                            + static_cast<size_t>(
                                    expert * gate_groups * gate_up_oc);
                    const float *Bs0 = scale_base + raw_nb * pack_nr;
                    const float *Bs1 = Bs0 + pack_nr;
                    uint16_t *C
                            = gate_up_out + r.offsets[mb] * N + nb * block_n;

                    const int64_t tiles = div_up(m_size, s4_max_mr);
                    for (int64_t tile = 0, mo = 0; tile < tiles; ++tile) {
                        const int mr = balanced_tile_rows(m_size, tiles, tile);
                        const auto fn = gate_kfn[static_cast<size_t>(mr)];
                        fn(A + mo * K, static_cast<int>(K), B0, As_tile + mo,
                                Bs0, custom_kernel::ScaleKind::kF32, nullptr,
                                custom_kernel::BiasKind::none, nullptr, 0,
                                C + mo * N, static_cast<int>(N),
                                static_cast<int>(K),
                                static_cast<int>(p.gate_up_group_size),
                                static_cast<int>(gate_up_oc));
                        fn(A + mo * K, static_cast<int>(K), B1, As_tile + mo,
                                Bs1, custom_kernel::ScaleKind::kF32, nullptr,
                                custom_kernel::BiasKind::none, nullptr, 0,
                                C + mo * N + compact_cols_per_gate_block,
                                static_cast<int>(N), static_cast<int>(K),
                                static_cast<int>(p.gate_up_group_size),
                                static_cast<int>(gate_up_oc));
                        mo += mr;
                    }
                });
            }
        }
    }

    // Stage 1.5: one requantization per compact sorted W13 row.
    const int64_t live = r.offsets[MB];
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t m = 0; m < live; ++m) {
        quantize_row_u8(Aq + m * N, As[m], gate_up_out + m * N, N);
    }

    // Stage 2: W2 of one (block, 32-column block) tile into F32 tile scratch.
    // Per-group scales remain independent inside the native kernel.
    const int64_t NB2 = K / block_n;
    const auto down_tile = [&](int64_t mb, int64_t nb, float *C) {
        const int64_t m_size = r.offsets[mb + 1] - r.offsets[mb];
        const int32_t expert = r.expert_ids[mb];
        const uint8_t *A = Aq + r.offsets[mb] * N;
        const float *As_block = As + r.offsets[mb];
        const int8_t *B = pw_down->data
                + static_cast<size_t>(expert) * down_bytes_per_expert
                + static_cast<size_t>(nb) * down_oblock_bytes;
        const float *Bs = pw_down->scales.data()
                + static_cast<size_t>(expert * down_groups * K) + nb * block_n;
        const int64_t tiles = div_up(m_size, s4_max_mr);
        for (int64_t tile = 0, mo = 0; tile < tiles; ++tile) {
            const int mr = balanced_tile_rows(m_size, tiles, tile);
            down_kfn[static_cast<size_t>(mr)](A + mo * N, static_cast<int>(N),
                    B, As_block + mo, Bs, custom_kernel::ScaleKind::kF32,
                    nullptr, custom_kernel::BiasKind::none, C + mo * block_n,
                    static_cast<int>(block_n), nullptr, 0, static_cast<int>(N),
                    static_cast<int>(p.down_group_size), static_cast<int>(K));
            mo += mr;
        }
    };

    if (fused_reduction) {
        // Stages 2+3 fused: each thread owns whole 32-column output blocks for
        // every token, runs every routed block against them and adds the
        // router-weighted rows into the f32 accumulator. A token's slots are
        // added in ascending block order whatever the thread count.
#pragma omp parallel num_threads(nth)
        {
            const int tid = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            const int64_t nb0 = NB2 * tid / nthr;
            const int64_t nb1 = NB2 * (tid + 1) / nthr;
            const int64_t c0 = nb0 * block_n;
            const int64_t width = (nb1 - nb0) * block_n;
            if (nb0 < nb1) {
                float *ZENDNNL_ROUTED_RESTRICT C
                        = C_tile + tid * block_m * block_n;
                for (int64_t m = 0; m < M; ++m) {
                    std::memset(reduce_acc + m * K + c0, 0,
                            static_cast<size_t>(width) * sizeof(float));
                }
                for (int64_t mb = 0; mb < MB; ++mb) {
                    const int64_t m_size = r.offsets[mb + 1] - r.offsets[mb];
                    const int32_t *A_ids = r.sorted_ids + mb * block_m;
                    for (int64_t nb = nb0; nb < nb1; ++nb) {
                        down_tile(mb, nb, C);
                        for (int64_t m = 0; m < m_size; ++m) {
                            const int32_t slot = A_ids[m];
                            add_mul_f32(reduce_acc + (slot / topk) * K
                                            + nb * block_n,
                                    C + m * block_n, router_weights[slot],
                                    block_n);
                        }
                    }
                }
                for (int64_t m = 0; m < M; ++m) {
                    f32_to_bf16_row(dst + m * p.dst_stride + c0,
                            reduce_acc + m * K + c0, width);
                }
            }
        }
        if (test_api::s_capture_native_w4.load(std::memory_order_relaxed)) {
            test_api::s_last_w4_reduction_path.store(
                    test_api::w4_reduction_path_t::fused_f32,
                    std::memory_order_relaxed);
        }
    } else {
        // Router-weighted BF16 scatter to (token, slot) order, then a separate
        // top-k reduction.
        const int64_t chunk_bytes = block_n * N / 2;
#pragma omp parallel num_threads(nth)
        {
            const int tid = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            int gm = 1;
            int gn = 1;
            thread_grid(MB, NB2, nthr, gm, gn);
            const int im = tid / gn;
            const int in = tid % gn;
            const int64_t bm = div_up(MB, gm);
            const int64_t bn = div_up(NB2, gn);
            const int64_t mb0 = im * bm;
            const int64_t mb1 = std::min(MB, mb0 + bm);
            const int64_t nb0 = in * bn;
            const int64_t nb1 = std::min(NB2, nb0 + bn);

            if (mb0 < mb1 && nb0 < nb1) {
                float *ZENDNNL_ROUTED_RESTRICT C
                        = C_tile + tid * block_m * block_n;
                loop_2d(mb0, mb1, nb0, nb1, chunk_bytes,
                        [&](int64_t mb, int64_t nb, int64_t) {
                    down_tile(mb, nb, C);
                    const int64_t m_size = r.offsets[mb + 1] - r.offsets[mb];
                    const int32_t *A_ids = r.sorted_ids + mb * block_m;
                    for (int64_t m = 0; m < m_size; ++m) {
                        const int32_t slot = A_ids[m];
                        copy_mul_bf16(down_out + static_cast<int64_t>(slot) * K
                                        + nb * block_n,
                                C + m * block_n, router_weights[slot], block_n);
                    }
                });
            }
        }

#pragma omp parallel for num_threads(nth) schedule(static)
        for (int64_t m = 0; m < M; ++m) {
            sum_rows_bf16(
                    dst + m * p.dst_stride, down_out + m * topk * K, topk, K);
        }
        if (test_api::s_capture_native_w4.load(std::memory_order_relaxed)) {
            test_api::s_last_w4_reduction_path.store(
                    test_api::w4_reduction_path_t::scatter_bf16,
                    std::memory_order_relaxed);
        }
    }

    if (test_api::s_capture_native_w4.load(std::memory_order_relaxed)) {
        test_api::s_native_w4_completed.fetch_add(1, std::memory_order_relaxed);
    }
    return status_t::success;
}

#endif // ZENDNNL_ROUTED_MOE_KERNELS_COMPILED

} // namespace

static status_t pack_weights_strided(const int8_t *src, int8_t *dst,
        int64_t num_experts, int64_t out_channels, int64_t in_channels,
        int64_t expert_stride, int64_t num_threads) {
    if (src == nullptr || dst == nullptr) { return status_t::op_bad_io; }

    size_t src_bytes = 0;
    size_t dst_bytes = 0;
    int64_t blocks_per_expert = 0;
    int64_t row = 0;
    const status_t size_status = checked_pack_sizes(num_experts, out_channels,
            in_channels, src_bytes, dst_bytes, blocks_per_expert, row);
    if (size_status != status_t::success) { return size_status; }
    const size_t logical_expert_bytes
            = src_bytes / static_cast<size_t>(num_experts);
    if (expert_stride < 0
            || static_cast<uint64_t>(expert_stride)
                    < static_cast<uint64_t>(logical_expert_bytes)
            || (num_experts > 1
                    && expert_stride > (std::numeric_limits<int64_t>::max()
                                               - static_cast<int64_t>(
                                                       logical_expert_bytes))
                                    / (num_experts - 1))) {
        return status_t::memory_bad_stride;
    }
    (void)dst_bytes;
    if (!isa_supported()) { return status_t::isa_unsupported; }

#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED
    const int64_t total = num_experts * blocks_per_expert;
    const int32_t requested = num_threads > std::numeric_limits<int32_t>::max()
            ? std::numeric_limits<int32_t>::max()
            : (num_threads < std::numeric_limits<int32_t>::min()
                              ? std::numeric_limits<int32_t>::min()
                              : static_cast<int32_t>(num_threads));
    const int nth = effective_num_threads(requested);

#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t i = 0; i < total; ++i) {
        const int64_t e = i / blocks_per_expert;
        const int64_t nb = i % blocks_per_expert;
        pack_weight_block(dst + e * out_channels * row + nb * block_n * row,
                src + e * expert_stride + nb * block_n * in_channels,
                in_channels);
    }
    return status_t::success;
#else
    (void)num_threads;
    return status_t::isa_unsupported;
#endif
}

status_t pack_weights(const int8_t *src, int8_t *dst, int64_t num_experts,
        int64_t out_channels, int64_t in_channels, int64_t num_threads) {
    int64_t expert_stride = 0;
    if (out_channels <= 0 || in_channels <= 0
            || out_channels
                    > std::numeric_limits<int64_t>::max() / in_channels) {
        return status_t::memory_bad_size;
    }
    expert_stride = out_channels * in_channels;
    return pack_weights_strided(src, dst, num_experts, out_channels,
            in_channels, expert_stride, num_threads);
}

status_t execute(const routed_moe_params &p) {
    std::shared_lock<std::shared_mutex> lifecycle_lock(
            execution_lifecycle_mutex());
    const status_t vs = validate_static(p);
    if (vs != status_t::success) { return vs; }
    const status_t routing_status = validate_routing_ids(p);
    if (routing_status != status_t::success) { return routing_status; }

#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED
    try {
        checked_problem_sizes_t sizes;
        const status_t size_status = checked_problem_sizes(p, sizes);
        if (size_status != status_t::success) { return size_status; }
        if (p.quant_scheme
                == routed_moe_quant_t::sym_per_group_w4a8_dynamic_per_token) {
            return execute_w4(p, sizes);
        }

        const int64_t M = p.num_tokens;
        const int64_t K = p.hidden_size;
        const int64_t N = p.intermediate_size;
        const int64_t topk = p.topk;
        const int64_t numel = sizes.numel;
        const int64_t gate_up_oc = sizes.gate_up_oc;

        const int nth = effective_num_threads(p.num_threads);

        const auto *src = static_cast<const uint16_t *>(p.src);
        auto *dst = static_cast<uint16_t *>(p.dst);

        // ── packed weights (packed once, then cached for the weight's life) ──
        const void *gate_up_key = p.gate_up_cache_key != nullptr
                ? p.gate_up_cache_key
                : p.gate_up_weight;
        const void *down_key = p.down_cache_key != nullptr ? p.down_cache_key
                                                           : p.down_weight;
        const int64_t gate_up_expert_stride = p.gate_up_stride_expert != 0
                ? p.gate_up_stride_expert
                : gate_up_oc * K;
        const int64_t down_expert_stride
                = p.down_stride_expert != 0 ? p.down_stride_expert : K * N;

        std::shared_ptr<const packed_weight_t> pw_gate_up;
        std::shared_ptr<const packed_weight_t> pw_down;
        status_t cache_status = lookup_packed_weight(gate_up_key,
                packed_tensor_role_t::gate_up,
                static_cast<const int8_t *>(p.gate_up_weight),
                p.num_local_experts, gate_up_oc, K, gate_up_expert_stride,
                p.wei_dt, nth, pw_gate_up);
        if (cache_status != status_t::success) { return cache_status; }
        cache_status = lookup_packed_weight(down_key,
                packed_tensor_role_t::down,
                static_cast<const int8_t *>(p.down_weight), p.num_local_experts,
                K, N, down_expert_stride, p.wei_dt, nth, pw_down);
        if (cache_status != status_t::success) { return cache_status; }
        const int8_t *packed_gate_up = pw_gate_up->data;
        const int8_t *packed_down = pw_down->data;
        const float *gate_up_scale = nullptr;
        const float *down_scale = nullptr;
        std::shared_ptr<const converted_scale_t> converted_gate_up;
        std::shared_ptr<const converted_scale_t> converted_down;
        if (p.scale_dt == data_type_t::f32) {
            gate_up_scale = static_cast<const float *>(p.gate_up_scale);
            down_scale = static_cast<const float *>(p.down_scale);
        } else {
            status_t scale_status = lookup_converted_scale(pw_gate_up.get(),
                    p.gate_up_scale, packed_tensor_role_t::gate_up,
                    p.num_local_experts, /*groups=*/1, gate_up_oc, p.scale_dt,
                    packed_layout_version, nth, converted_gate_up);
            if (scale_status != status_t::success) { return scale_status; }
            scale_status = lookup_converted_scale(pw_down.get(), p.down_scale,
                    packed_tensor_role_t::down, p.num_local_experts,
                    /*groups=*/1, K, p.scale_dt, packed_layout_version, nth,
                    converted_down);
            if (scale_status != status_t::success) { return scale_status; }
            gate_up_scale = converted_gate_up->data;
            down_scale = converted_down->data;
        }

        // ── routing ─────────────────────────────────────────────────────────
        routing_t r;
        const status_t build_status = build_routing(p, sizes, r);
        if (build_status != status_t::success) { return build_status; }
        const int64_t MB = r.num_blocks;

        if (MB == 0) {
            for (int64_t m = 0; m < M; ++m) {
                std::memset(dst + m * p.dst_stride, 0,
                        static_cast<size_t>(K) * sizeof(uint16_t));
            }
            return status_t::success;
        }

        scratch_t &sc = scratch();
        size_t a_tile_elements = 0;
        size_t c_tile_elements = 0;
        if (!checked_scratch_elements(nth, block_m, K, a_tile_elements)
                || !checked_scratch_elements(
                        nth, 2 * block_m, block_n, c_tile_elements)) {
            return status_t::memory_bad_size;
        }
        uint8_t *Aq = scratch_t::reserve(sc.aq,
                std::max(sizes.src_elements, sizes.intermediate_elements));
        float *As = scratch_t::reserve(sc.as, static_cast<size_t>(numel));
        uint8_t *A_tile = scratch_t::reserve(sc.a_tile, a_tile_elements);
        float *C_tile = scratch_t::reserve(sc.c_tile, c_tile_elements);
        uint16_t *gate_up_out = scratch_t::reserve(
                sc.gate_up_out, sizes.intermediate_elements);
        uint16_t *down_out
                = scratch_t::reserve(sc.down_out, sizes.down_elements);

        // Router weights indexed by dense flat slot.  Only materialised when the
        // caller's rows are padded; the tight case (every framework we know of)
        // reads the caller's buffer directly.
        const int64_t weight_stride
                = p.topk_weights_stride != 0 ? p.topk_weights_stride : topk;
        const float *router_weights = p.topk_weights;
        if (weight_stride != topk) {
            float *dense = scratch_t::reserve(
                    sc.weights, static_cast<size_t>(numel));
            for (int64_t m = 0; m < M; ++m) {
                std::memcpy(dense + m * topk,
                        p.topk_weights + m * weight_stride,
                        static_cast<size_t>(topk) * sizeof(float));
            }
            router_weights = dense;
        }

        // Slots routed to a non-resident expert contribute zero.
        for (int64_t i = 0; i < r.num_inactive; ++i) {
            std::memset(down_out + static_cast<int64_t>(r.inactive[i]) * K, 0,
                    static_cast<size_t>(K) * sizeof(uint16_t));
        }

        // Row size of the packed weights, and the per-expert / per-block strides
        // that follow from it.
        const int64_t packed_K = packed_row_bytes(K);
        const int64_t packed_N = packed_row_bytes(N);
        const int64_t stride_gate_up = gate_up_oc * packed_K;
        const int64_t stride_down = K * packed_N;

        const int64_t NB1 = N / block_n;
        const int64_t NB2 = K / block_n;

        // ---- stage 0: quantize the hidden states once, per token ------------
        //
        // Every stage below runs on the same `nth`-wide team on purpose. Sizing a
        // stage's team to its own work looks attractive at one token, but changing
        // the requested width between stages makes the runtime tear down and
        // rebuild the team mid-pipeline, which measured several hundred
        // microseconds per call at the token counts that matter -- far more than
        // the fork it avoids.
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int64_t m = 0; m < M; ++m) {
            quantize_row_u8(Aq + m * K, As[m], src + m * p.src_stride, K);
        }

        const bool gelu = p.activation == routed_moe_activation_t::gelu_and_mul;

        // Gate/up tile (mb, nb) with the activation folded in.  `gather`
        // reloads the block's rows into the thread's A buffer; callers reuse
        // them across consecutive N blocks of the same M block.
        const auto gate_up_tile
                = [&](uint8_t *ZENDNNL_ROUTED_RESTRICT A, float *As_tile,
                          int64_t mb, int64_t nb, bool gather) {
            const int64_t m_size = r.offsets[mb + 1] - r.offsets[mb];
            const int32_t expert_id = r.expert_ids[mb];
            const int32_t *A_ids = r.sorted_ids + mb * block_m;

            if (gather) {
                for (int64_t m = 0; m < m_size; ++m) {
                    const int64_t index = A_ids[m] / topk;
                    std::memcpy(
                            A + m * K, Aq + index * K, static_cast<size_t>(K));
                    As_tile[m] = As[index];
                }
            }

            const int8_t *B0 = packed_gate_up + expert_id * stride_gate_up
                    + nb * block_n * packed_K;
            const int8_t *B1 = packed_gate_up + expert_id * stride_gate_up
                    + (nb + NB1) * block_n * packed_K;
            const float *Bs0
                    = gate_up_scale + expert_id * gate_up_oc + nb * block_n;
            const float *Bs1 = gate_up_scale + expert_id * gate_up_oc
                    + (nb + NB1) * block_n;
            const int32_t *Bcomp0
                    = reinterpret_cast<const int32_t *>(B0 + block_n * K);
            const int32_t *Bcomp1
                    = reinterpret_cast<const int32_t *>(B1 + block_n * K);

            uint16_t *C = gate_up_out + r.offsets[mb] * N + nb * block_n;
            if (gelu) {
                tinygemm_gate_up<routed_moe_activation_t::gelu_and_mul>(A, B0,
                        B1, C, As_tile, Bs0, Bs1, Bcomp0, Bcomp1, m_size, K, K,
                        block_n, N);
            } else {
                tinygemm_gate_up<routed_moe_activation_t::silu_and_mul>(A, B0,
                        B1, C, As_tile, Bs0, Bs1, Bcomp0, Bcomp1, m_size, K, K,
                        block_n, N);
            }
        };

        // Down tile (mb, nb), scaled by the router weight and scattered back
        // to (token, slot) order.  A is already contiguous in sorted order.
        const auto down_tile = [&](float *ZENDNNL_ROUTED_RESTRICT C, int64_t mb,
                                       int64_t nb) {
            const int64_t m_size = r.offsets[mb + 1] - r.offsets[mb];
            const int32_t expert_id = r.expert_ids[mb];
            const int32_t *A_ids = r.sorted_ids + mb * block_m;
            const uint8_t *A = Aq + r.offsets[mb] * N;
            const float *As_blk = As + r.offsets[mb];

            const int8_t *B = packed_down + expert_id * stride_down
                    + nb * block_n * packed_N;
            const float *Bs = down_scale + expert_id * K + nb * block_n;
            const int32_t *Bcomp
                    = reinterpret_cast<const int32_t *>(B + block_n * N);

            tinygemm_down(
                    A, B, C, As_blk, Bs, Bcomp, m_size, N, N, block_n, block_n);

            for (int64_t m = 0; m < m_size; ++m) {
                const int32_t index = A_ids[m];
                copy_mul_bf16(down_out + index * K + nb * block_n,
                        C + m * block_n, router_weights[index], block_n);
            }
        };

        // At M <= block_m every routed expert owns exactly one M block, so the
        // grid's L2 reuse across M blocks cannot apply and both GEMM stages
        // are weight streaming.  Each thread then takes a contiguous,
        // M-block-major tile range carrying an equal share of the stage's
        // cost, so hot experts' tiles spread over several threads and no
        // thread holds the stage open while the others idle.
        const bool decode_balance = M <= block_m && decode_balance_enabled();
        int64_t *cost_prefix = nullptr;
        if (decode_balance) {
            cost_prefix = scratch_t::reserve(
                    sc.tile_cost_prefix, static_cast<size_t>(MB + 1));
            cost_prefix[0] = 0;
            for (int64_t mb = 0; mb < MB; ++mb) {
                cost_prefix[mb + 1] = cost_prefix[mb]
                        + decode_tile_cost(r.offsets[mb + 1] - r.offsets[mb]);
            }
        }

        // ---- stage 1: gate/up + fused activation ----------------------------
        {
            const int64_t chunk_bytes = block_n * K * 2;
#pragma omp parallel num_threads(nth)
            {
                const int tid = omp_get_thread_num();
                const int nthr = omp_get_num_threads();
                uint8_t *ZENDNNL_ROUTED_RESTRICT A = A_tile + tid * block_m * K;
                alignas(64) float As_tile[block_m];

                if (decode_balance) {
                    int64_t t0 = 0;
                    int64_t t1 = 0;
                    decode_tile_range(cost_prefix, MB, NB1, tid, nthr, t0, t1);
                    for (int64_t t = t0; t < t1; ++t) {
                        const int64_t nb = t % NB1;
                        gate_up_tile(
                                A, As_tile, t / NB1, nb, t == t0 || nb == 0);
                    }
                } else {
                    int gm = 1;
                    int gn = 1;
                    thread_grid(MB, NB1, nthr, gm, gn);
                    const int im = tid / gn;
                    const int in = tid % gn;

                    const int64_t bm = div_up(MB, gm);
                    const int64_t bn = div_up(NB1, gn);
                    const int64_t mb0 = im * bm;
                    const int64_t mb1 = std::min(MB, mb0 + bm);
                    const int64_t nb0 = in * bn;
                    const int64_t nb1 = std::min(NB1, nb0 + bn);

                    // Gather each block's rows once per N chunk, then reuse
                    // them across every N block in the chunk.
                    if (mb0 < mb1 && nb0 < nb1) {
                        loop_2d(mb0, mb1, nb0, nb1, chunk_bytes,
                                [&](int64_t mb, int64_t nb, int64_t nb_offset) {
                            gate_up_tile(A, As_tile, mb, nb, nb_offset == 0);
                        });
                    }
                }
            }
        }

        // ---- stage 1.5: requantize the intermediate -------------------------
        {
            const int64_t live = r.offsets[MB];
#pragma omp parallel for num_threads(nth) schedule(static)
            for (int64_t m = 0; m < live; ++m) {
                quantize_row_u8(Aq + m * N, As[m], gate_up_out + m * N, N);
            }
        }

        // At decode, when every slot has a resident expert and there is at
        // least one output column block per thread, each thread owns a fixed
        // range of down-projection output columns for all experts.  Every
        // thread then streams the same bytes, and because no other thread
        // writes its columns it also reduces over topk itself: no scatter
        // through a shared buffer and no separate reduction stage.  Each
        // (token, slot) contribution is still rounded to bf16 and the slots
        // summed in order, exactly as stages 2 and 3 do.
        if (decode_balance && decode_columns_enabled() && r.num_inactive == 0
                && NB2 >= nth) {
            // ---- stages 2 and 3 at decode: column-owned down + reduce -------
#pragma omp parallel num_threads(nth)
            {
                const int tid = omp_get_thread_num();
                const int nthr = omp_get_num_threads();
                const int64_t nb0 = NB2 * tid / nthr;
                const int64_t nb1 = NB2 * (tid + 1) / nthr;
                const int64_t cols = (nb1 - nb0) * block_n;
                float *ZENDNNL_ROUTED_RESTRICT C
                        = C_tile + tid * 2 * block_m * block_n;
                // Columns [nb0, nb1) of every (token, slot) row, row stride
                // `cols`; the threads' regions tile down_out exactly.
                uint16_t *ZENDNNL_ROUTED_RESTRICT own
                        = down_out + numel * nb0 * block_n;

                for (int64_t mb = 0; cols > 0 && mb < MB; ++mb) {
                    const int64_t m_size = r.offsets[mb + 1] - r.offsets[mb];
                    const int32_t expert_id = r.expert_ids[mb];
                    const int32_t *A_ids = r.sorted_ids + mb * block_m;
                    const uint8_t *A = Aq + r.offsets[mb] * N;
                    const float *As_blk = As + r.offsets[mb];
                    for (int64_t nb = nb0; nb < nb1; ++nb) {
                        const int8_t *B = packed_down + expert_id * stride_down
                                + nb * block_n * packed_N;
                        const float *Bs
                                = down_scale + expert_id * K + nb * block_n;
                        const int32_t *Bcomp
                                = reinterpret_cast<const int32_t *>(
                                        B + block_n * N);
                        tinygemm_down(A, B, C, As_blk, Bs, Bcomp, m_size, N, N,
                                block_n, block_n);
                        for (int64_t m = 0; m < m_size; ++m) {
                            const int32_t index = A_ids[m];
                            copy_mul_bf16(
                                    own + index * cols + (nb - nb0) * block_n,
                                    C + m * block_n, router_weights[index],
                                    block_n);
                        }
                    }
                }

                for (int64_t m = 0; cols > 0 && m < M; ++m) {
                    sum_rows_bf16(dst + m * p.dst_stride + nb0 * block_n,
                            own + m * topk * cols, topk, cols);
                }
            }
            return status_t::success;
        }

        // ---- stage 2: down projection + weighted scatter --------------------
        {
            const int64_t chunk_bytes = block_n * N;
#pragma omp parallel num_threads(nth)
            {
                const int tid = omp_get_thread_num();
                const int nthr = omp_get_num_threads();
                float *ZENDNNL_ROUTED_RESTRICT C
                        = C_tile + tid * 2 * block_m * block_n;

                if (decode_balance) {
                    int64_t t0 = 0;
                    int64_t t1 = 0;
                    decode_tile_range(cost_prefix, MB, NB2, tid, nthr, t0, t1);
                    for (int64_t t = t0; t < t1; ++t) {
                        down_tile(C, t / NB2, t % NB2);
                    }
                } else {
                    int gm = 1;
                    int gn = 1;
                    thread_grid(MB, NB2, nthr, gm, gn);
                    const int im = tid / gn;
                    const int in = tid % gn;

                    const int64_t bm = div_up(MB, gm);
                    const int64_t bn = div_up(NB2, gn);
                    const int64_t mb0 = im * bm;
                    const int64_t mb1 = std::min(MB, mb0 + bm);
                    const int64_t nb0 = in * bn;
                    const int64_t nb1 = std::min(NB2, nb0 + bn);

                    if (mb0 < mb1 && nb0 < nb1) {
                        loop_2d(mb0, mb1, nb0, nb1, chunk_bytes,
                                [&](int64_t mb, int64_t nb, int64_t nb_offset) {
                            (void)nb_offset;
                            down_tile(C, mb, nb);
                        });
                    }
                }
            }
        }

        // ---- stage 3: reduce over topk --------------------------------------
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int64_t m = 0; m < M; ++m) {
            sum_rows_bf16(
                    dst + m * p.dst_stride, down_out + m * topk * K, topk, K);
        }

        return status_t::success;
    } catch (const std::length_error &) {
        return status_t::memory_bad_size;
    } catch (const std::bad_alloc &) { return status_t::memory_bad_storage; }
#else
    return status_t::isa_unsupported;
#endif
}

void flush_weight_cache() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(
            execution_lifecycle_mutex());
#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED
    std::lock_guard<std::mutex> guard(cache_mutex());
    cache_store().clear();
    scale_cache_store().clear();
#endif
}

} // namespace routed_moe

void group_matmul_routed_moe_flush_weight_cache() {
    routed_moe::flush_weight_cache();
    ntile_flat_parallel::flush_packed_weight_cache();
}

status_t group_matmul_routed_moe_pack_weights(const void *src, void *dst,
        int64_t num_experts, int64_t out_channels, int64_t in_channels,
        data_type_t wei_dt, int num_threads) {
    if (wei_dt != data_type_t::s8) { return status_t::unimplemented; }
    return routed_moe::pack_weights(static_cast<const int8_t *>(src),
            static_cast<int8_t *>(dst), num_experts, out_channels, in_channels,
            num_threads);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
