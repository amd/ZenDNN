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

/// The pack store.  One pack per expert weight, shared by routed layers and
/// grouped calls: the first call of either API to reach a weight packs it,
/// and every later call that reads the same weight, format and shape,
/// whatever its activation, bias or source scales, runs on that pack.  A
/// scaled weight's pack holds its blocks (and sums), so a weight a call names
/// without its scales (a prepack extra) is packed all the same; its scaled
/// pack, on the same blocks, keeps the last weight scales a call gave it.
/// Flushes count generations here, so a call that a flush overtakes
/// publishes nothing.  The store's state is private to store.cpp; callers go
/// through the functions below.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_STORE_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_STORE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/nr32_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/scales.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

/// Why a call fails when an allocation does.  An array, so that it is one
/// object whose address every file sees alike: callers compare it by
/// address.
inline constexpr char kOutOfMemory[] = "out_of_memory";

/// One expert weight: its bytes, format and shape.
struct weight_key {
    const void *weight = nullptr;
    /// The weight's row in `nr32::kFormats`.
    int format = 0;
    int K = 0;
    int N = 0;
    int ldb = 0;
    int group_size = 0;
    bool trans = false;

    bool operator==(const weight_key &o) const {
        return weight == o.weight && format == o.format && K == o.K && N == o.N
                && ldb == o.ldb && group_size == o.group_size
                && trans == o.trans;
    }
};

struct weight_key_hash_t {
    size_t operator()(const weight_key &k) const noexcept {
        size_t h = std::hash<const void *> {}(k.weight);
        for (const int v : {k.format, k.K, k.N, k.ldb, k.group_size,
                     static_cast<int>(k.trans)}) {
            h = (h ^ static_cast<size_t>(v)) * 0x9E3779B97F4A7C15ULL;
        }
        return h;
    }
};

inline size_t pointer_hash(const void *p) {
    uint64_t x = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p))
            + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return static_cast<size_t>(x ^ (x >> 31));
}

/// The weight scales a scaled pack holds.
struct scale_id {
    const void *scale = nullptr;
    data_type_t dt = data_type_t::none;
    int64_t groups = 1;
    bool per_tensor = false;

    bool operator==(const scale_id &o) const {
        return scale == o.scale && dt == o.dt && groups == o.groups
                && per_tensor == o.per_tensor;
    }
};

using expert_pack = std::shared_ptr<const nr32::packed_expert>;

/// A weight's packs: `bare`, whole for a format without scales and its blocks
/// (and sums) alone for a scaled one, and for a scaled one `scaled`, on
/// `bare`'s blocks with the weight scales `scales`.  A call that keeps a pack
/// keeps it alive.
struct weight_packs_t {
    expert_pack bare;
    scale_id scales;
    expert_pack scaled;
};

/// Twice the flushes so far, plus one while a flush runs: it counts up when
/// a flush begins, before it drops anything, and again when it ends.  A call
/// reads the count when it enters the API; after a flush that ran meanwhile,
/// the packs it makes are its own, and it publishes neither them nor its
/// call site.
inline std::atomic<uint64_t> &flushes() {
    static std::atomic<uint64_t> count {0};
    return count;
}

/// Whether a call that read the flush count `since` may publish: no flush
/// ran when it read the count, and none has begun since.
inline bool may_publish(const uint64_t since) {
    return since % 2 == 0 && flushes().load(std::memory_order_relaxed) == since;
}

/// The store's key of `weight`, a weight of `desc`.
inline weight_key key_of(const nr32::weight_desc &desc, const void *weight) {
    weight_key k;
    k.weight = weight;
    k.format = desc.fmt != nullptr ? nr32::format_index(*desc.fmt) : -1;
    k.K = desc.K;
    k.N = desc.N;
    k.ldb = desc.ldb;
    k.group_size = desc.group_size;
    k.trans = desc.trans;
    return k;
}

/// What a scaled pack of the weight scales `scale`, of form `form`, holds.
inline scale_id scale_id_of(const void *scale, const scale_form_t &form) {
    scale_id id;
    id.scale = scale;
    id.dt = form.dt;
    id.groups = form.groups;
    id.per_tensor = form.per_tensor;
    return id;
}

/// The packs of a projection's experts, for a call that began at flush count
/// `since`: of `weights[e]` with, for a scaled format, the weight scales
/// `scales[e]` of form `form`; null where `weights[e]` is, and where a scaled
/// weight comes without scales, which packs the weight for a later call to
/// give its scales.  From the store, or packed now on `num_threads` threads,
/// outside the store's lock: of two calls that pack a weight at once, the
/// first to finish publishes its pack.  After a flush that ran since `since`,
/// packed for the call alone.  Null on success, else why not.
const char *find_packs(const nr32::weight_desc &desc, const scale_form_t &form,
        const std::vector<const void *> &weights,
        const std::vector<const void *> &scales, const uint64_t since,
        const int num_threads, std::vector<expert_pack> &out);

/// What the store holds for each of `weights`, weights of `desc`: empty
/// where it holds nothing.
std::vector<weight_packs_t> stored_packs(const nr32::weight_desc &desc,
        const std::vector<const void *> &weights);

/// Whether there is nothing to pack for the pool of prepack extras
/// `fingerprint` (`extras_fingerprint`): grouped calls have packed it, or a
/// flush has run since `since`, after which its packs would serve no later
/// call.
bool extras_warmed(size_t fingerprint, uint64_t since);

/// Records that the pool `fingerprint` is packed, unless a flush has run
/// since `since`; past `limits().warmed_pools` pools, forgets them all first.
void mark_extras_warmed(size_t fingerprint, uint64_t since);

/// Drops every pack and every pool of prepack extras the store holds.  Only
/// `flush()` may, between its two counts, so that no call publishes a pack
/// over it and no call site keeps a pack the store no longer has.
void clear_store();

/// How many packs and pools of prepack extras the store holds.
size_t pack_count();
size_t warmed_pool_count();

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_STORE_HPP
