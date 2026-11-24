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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/store.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <mutex>
#include <new>
#include <utility>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/call_sites.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

namespace {

// The store's state is never destroyed, so a flush from an exit handler
// finds it whole.
std::shared_mutex &store_mutex() {
    static auto *mutex = new std::shared_mutex;
    return *mutex;
}

std::unordered_map<weight_key, weight_packs_t, weight_key_hash_t> &store() {
    static auto *packs = new std::unordered_map<weight_key, weight_packs_t,
            weight_key_hash_t>;
    return *packs;
}

/// The pools whose prepack extras grouped calls have packed, by
/// `extras_fingerprint`.
std::unordered_set<size_t> &warmed_extras() {
    static auto *warmed = new std::unordered_set<size_t>;
    return *warmed;
}

} // namespace

namespace {

/// A pack counted in `test_api::s_ntile_flat_parallel_live_packs` while
/// alive.
struct counted_pack_t : nr32::packed_expert {
    explicit counted_pack_t(nr32::packed_expert &&pack)
        : nr32::packed_expert(std::move(pack)) {
        test_api::s_ntile_flat_parallel_live_packs.fetch_add(
                1, std::memory_order_relaxed);
    }
    ~counted_pack_t() {
        test_api::s_ntile_flat_parallel_live_packs.fetch_sub(
                1, std::memory_order_relaxed);
    }
    counted_pack_t(const counted_pack_t &) = delete;
    counted_pack_t &operator=(const counted_pack_t &) = delete;
};

/// `pack`, as the store, a call site or a call holds it.
expert_pack hold(nr32::packed_expert &&pack) {
    return std::make_shared<const counted_pack_t>(std::move(pack));
}

} // namespace

namespace {

/// The bytes of a weight of `desc` in the caller's buffer.
size_t weight_bytes(const nr32::weight_desc &desc) {
    const size_t elems = static_cast<size_t>(desc.trans ? desc.N : desc.K)
            * static_cast<size_t>(desc.ldb);
    const size_t bits = static_cast<size_t>(desc.fmt->elem_bits);
    return (elems * bits + 7) / 8;
}

} // namespace

namespace {

/// Whether an AOCL-DLP or oneDNN reorder, a custom-kernel pack or the native
/// BRGEMM at weight cache mode 2 may have rewritten `weight`, a weight of
/// `desc`, in place.
bool packed_in_place(const nr32::weight_desc &desc, const void *weight) {
    return weights_rewritten_in_place(weight)
            || custom_kernel::weight_packed_in_place(weight)
            || native::native_weight_packed_in_place(
                    weight, weight_bytes(desc));
}

/// `bare`, a pack of a weight of `desc`, with the weight scales `scale` of
/// form `form`, into `out`: per-tensor and f16 scales go to the pack as
/// per-column f32 rows.  Null on success, else why not.
const char *add_scales(const nr32::weight_desc &desc, const scale_form_t &form,
        const nr32::packed_expert &bare, const void *scale, expert_pack &out) {
    const bool widen = form.per_tensor || form.dt == data_type_t::f16;
    std::vector<float> wide;
    if (widen) {
        wide.resize(static_cast<size_t>(form.groups) * desc.N);
        for (size_t c = 0; c < wide.size(); ++c)
            wide[c] = scale_at(scale, form.dt, form.per_tensor ? 0 : c);
        scale = wide.data();
    }
    nr32::packed_expert px;
    if (nr32::with_scales(nr32::layout_of(desc), bare, scale,
                widen ? data_type_t::f32 : form.dt, px)
            != status_t::success) {
        return "unsupported_weight";
    }
    out = hold(std::move(px));
    return nullptr;
}

/// `find_packs` through the store, for a call that began at flush count
/// `since`; `stale`, with nothing published, when a flush ran since.
const char *store_packs(const nr32::weight_desc &desc, const scale_form_t &form,
        const std::vector<const void *> &weights,
        const std::vector<const void *> &scales, const uint64_t since,
        const int num_threads, std::vector<expert_pack> &out, bool &stale) {
    const size_t count = weights.size();
    const bool scaled = desc.fmt->scaled();
    out.assign(count, nullptr);
    std::vector<weight_key> keys(count);
    const auto scales_of
            = [&](const size_t e) { return scale_id_of(scales[e], form); };
    // The experts whose pack the store lacks or holds with other scales.
    std::vector<size_t> pending;
    {
        std::shared_lock<std::shared_mutex> lock(store_mutex());
        if (!may_publish(since)) {
            stale = true;
            return nullptr;
        }
        const auto &packs = store();
        for (size_t e = 0; e < count; ++e) {
            if (weights[e] == nullptr) continue;
            const weight_key &k = keys[e] = key_of(desc, weights[e]);
            const auto it = packs.find(k);
            if (it != packs.end()) {
                const weight_packs_t &w = it->second;
                if (!scaled) {
                    out[e] = w.bare;
                    continue;
                }
                if (scales[e] == nullptr) continue;
                if (w.scaled != nullptr && w.scales == scales_of(e)) {
                    out[e] = w.scaled;
                    continue;
                }
            }
            pending.push_back(e);
        }
    }
    if (pending.empty()) return nullptr;

    // The first expert of each weight the store lacks.
    std::vector<size_t> missing;
    {
        std::unordered_set<weight_key, weight_key_hash_t> seen;
        std::shared_lock<std::shared_mutex> lock(store_mutex());
        for (const size_t e : pending) {
            if (store().count(keys[e]) == 0 && seen.insert(keys[e]).second)
                missing.push_back(e);
        }
    }
    std::vector<expert_pack> made(missing.size());
    if (!missing.empty()) {
        for (const size_t e : missing) {
            if (packed_in_place(desc, weights[e]))
                return "weights_packed_in_place";
        }
        std::vector<const void *> pack_weights;
        for (const size_t e : missing)
            pack_weights.push_back(weights[e]);
        nr32::packed_weights pw;
        const status_t packed = nr32::prepack(
                desc, pack_weights, {}, data_type_t::none, num_threads, pw);
        if (packed == status_t::memory_bad_size) return kOutOfMemory;
        if (packed != status_t::success) return "unsupported_weight";
        for (size_t i = 0; i < missing.size(); ++i)
            made[i] = hold(std::move(pw.experts[i]));
    }

    if (test_api::s_ntile_flat_parallel_flush_before_publish.exchange(
                false, std::memory_order_relaxed)) {
        flush();
    }
    const char *unserved = nullptr;
    bool replaced = false;
    {
        std::unique_lock<std::shared_mutex> lock(store_mutex());
        if (!may_publish(since)) {
            stale = true;
            return nullptr;
        }
        auto &packs = store();
        for (size_t i = 0; i < missing.size(); ++i) {
            weight_packs_t &w = packs[keys[missing[i]]];
            if (w.bare == nullptr) w.bare = std::move(made[i]);
        }
        // Only a flush drops a weight, so each pending one is here.
        for (const size_t e : pending) {
            weight_packs_t &w = packs[keys[e]];
            if (!scaled) {
                out[e] = w.bare;
                continue;
            }
            if (scales[e] == nullptr) continue;
            const scale_id id = scales_of(e);
            if (w.scaled == nullptr || !(w.scales == id)) {
                expert_pack px;
                unserved = add_scales(desc, form, *w.bare, scales[e], px);
                if (unserved != nullptr) break;
                replaced = replaced || w.scaled != nullptr;
                w.scaled = std::move(px);
                w.scales = id;
            }
            out[e] = w.scaled;
        }
    }
    // A site must not keep a scaled pack the store has replaced.
    if (replaced) drop_call_sites();
    return unserved;
}

/// `find_packs` for a call a flush overtook: packs made now for the call
/// alone, published nowhere.
const char *own_packs(const nr32::weight_desc &desc, const scale_form_t &form,
        const std::vector<const void *> &weights,
        const std::vector<const void *> &scales, const int num_threads,
        std::vector<expert_pack> &out) {
    const bool scaled = desc.fmt->scaled();
    out.assign(weights.size(), nullptr);
    std::vector<size_t> experts;
    std::vector<const void *> pack_weights;
    for (size_t e = 0; e < weights.size(); ++e) {
        if (weights[e] == nullptr || (scaled && scales[e] == nullptr)) continue;
        if (packed_in_place(desc, weights[e])) return "weights_packed_in_place";
        experts.push_back(e);
        pack_weights.push_back(weights[e]);
    }
    if (experts.empty()) return nullptr;
    nr32::packed_weights pw;
    const status_t packed = nr32::prepack(
            desc, pack_weights, {}, data_type_t::none, num_threads, pw);
    if (packed == status_t::memory_bad_size) return kOutOfMemory;
    if (packed != status_t::success) return "unsupported_weight";
    for (size_t i = 0; i < experts.size(); ++i) {
        const size_t e = experts[i];
        if (!scaled) {
            out[e] = hold(std::move(pw.experts[i]));
            continue;
        }
        const char *unserved
                = add_scales(desc, form, pw.experts[i], scales[e], out[e]);
        if (unserved != nullptr) return unserved;
    }
    return nullptr;
}

} // namespace

const char *find_packs(const nr32::weight_desc &desc, const scale_form_t &form,
        const std::vector<const void *> &weights,
        const std::vector<const void *> &scales, const uint64_t since,
        const int num_threads, std::vector<expert_pack> &out) {
    bool stale = false;
    const char *unserved = store_packs(
            desc, form, weights, scales, since, num_threads, out, stale);
    if (!stale) return unserved;
    return own_packs(desc, form, weights, scales, num_threads, out);
}

std::vector<weight_packs_t> stored_packs(const nr32::weight_desc &desc,
        const std::vector<const void *> &weights) {
    std::vector<weight_packs_t> out(weights.size());
    std::shared_lock<std::shared_mutex> lock(store_mutex());
    for (size_t e = 0; e < weights.size(); ++e) {
        if (weights[e] == nullptr) continue;
        const auto it = store().find(key_of(desc, weights[e]));
        if (it != store().end()) out[e] = it->second;
    }
    return out;
}

bool extras_warmed(const size_t fingerprint, const uint64_t since) {
    std::shared_lock<std::shared_mutex> lock(store_mutex());
    return !may_publish(since) || warmed_extras().count(fingerprint) != 0;
}

void mark_extras_warmed(const size_t fingerprint, const uint64_t since) {
    std::unique_lock<std::shared_mutex> lock(store_mutex());
    if (!may_publish(since)) return;
    if (warmed_extras().size() >= limits().warmed_pools)
        warmed_extras().clear();
    warmed_extras().insert(fingerprint);
}

void clear_store() {
    assert(flushes().load(std::memory_order_relaxed) % 2 == 1);
    std::unique_lock<std::shared_mutex> lock(store_mutex());
    store().clear();
    warmed_extras().clear();
}

size_t pack_count() {
    std::shared_lock<std::shared_mutex> lock(store_mutex());
    return store().size();
}

size_t warmed_pool_count() {
    std::shared_lock<std::shared_mutex> lock(store_mutex());
    return warmed_extras().size();
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
