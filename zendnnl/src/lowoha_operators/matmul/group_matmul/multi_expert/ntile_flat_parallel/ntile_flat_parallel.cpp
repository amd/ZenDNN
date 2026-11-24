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

#include "ntile_flat_parallel.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "nr32_pack.hpp"
#include "nr32_tile.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

namespace {

struct layer_key_hash_t {
    size_t operator()(const layer_key &k) const noexcept {
        size_t h = 0;
        const auto mix
                = [&h](size_t v) { h = (h ^ v) * 0x9E3779B97F4A7C15ULL; };
        for (const projection_key *p : {&k.w13, &k.w2}) {
            mix(std::hash<const void *> {}(p->weight));
            mix(static_cast<size_t>(p->ldb));
            mix(p->expert_stride);
            mix(static_cast<size_t>(p->trans_weight));
        }
        for (const int v : {k.num_experts, k.H, k.I, static_cast<int>(k.act),
                     static_cast<int>(k.wei), static_cast<int>(k.w13_bias),
                     static_cast<int>(k.w2_bias),
                     static_cast<int>(k.w13_src_groups),
                     static_cast<int>(k.w2_src_groups)}) {
            mix(static_cast<size_t>(v));
        }
        return h;
    }
};

/// Target depth of the unit queue per thread.
constexpr int kUnitsPerThread = 2;
/// Rows a W13 unit copies at a time when it shares a run's gather.
constexpr int kGatherRows = 16;
/// Units per thread a prompt call cuts each projection into.
constexpr int kPromptUnitsPerThread = 6;
/// Pauses a unit spends waiting for another thread's rows before it starts
/// yielding the core.
constexpr int kWaitSpinPauses = 4096;
/// A thread's scratch above this size is freed when its call returns.  A
/// prompt's scratch can reach hundreds of MiB, which a later call would
/// otherwise allocate and zero again.
constexpr size_t kScratchKeepBytes = size_t {1} << 30;
/// Bytes past the last s8 row the kernels may read (`nr32::run_tile`).
constexpr size_t kRowSlack = nr32::kRowReadSlack;

bool quantized_weights(const data_type_t wei) {
    return wei == data_type_t::s8 || wei == data_type_t::s4;
}

float wei_bytes_of(const data_type_t wei) {
    return wei == data_type_t::s4 ? 0.5f : wei == data_type_t::s8 ? 1.0f : 2.0f;
}

/// Whether DA8W8 layers run the kernels that read source bytes XOR-ed with
/// 0x80, which skip that XOR on every broadcast: on the CPUs where that
/// outweighs flipping each row once.  W4A8 layers keep the s8-source
/// kernels, whose decode is the faster.
bool u8_src_pays() {
    const int pin = test_api::s_ntile_flat_parallel_u8_src_override.load(
            std::memory_order_relaxed);
    if (pin >= 0) return pin != 0;
    static const bool pays = [] {
        const native::native_hw_t &hw = native::native_hw();
        return hw.vendor == "AuthenticAMD" && hw.family == 0x1A
                && hw.model >= 0x50 && hw.model <= 0x5F;
    }();
    return pays;
}

nr32::epilogue epilogue_of(const grp_matmul_gated_act_t act) {
    switch (act) {
        case grp_matmul_gated_act_t::swiglu_oai_mul:
            return nr32::epilogue::swiglu_pairs;
        case grp_matmul_gated_act_t::silu_and_mul:
            return nr32::epilogue::silu_split;
        case grp_matmul_gated_act_t::gelu_and_mul:
            return nr32::epilogue::gelu_split;
        default: return nr32::epilogue::plain;
    }
}

nr32::bias_kind bias_kind_of(const data_type_t dt) {
    switch (dt) {
        case data_type_t::bf16: return nr32::bias_kind::bf16;
        case data_type_t::f16: return nr32::bias_kind::f16;
        case data_type_t::f32: return nr32::bias_kind::f32;
        default: return nr32::bias_kind::none;
    }
}

status_t decline(const char *reason) {
    apilog_error("[GRP_MATMUL.ALGO4] cannot serve the call: reason=", reason);
    return status_t::unimplemented;
}

/// Why a call fails when an allocation does.
constexpr const char *kOutOfMemory = "out_of_memory";

/// The status of a call ALGO 4 fails for `reason`: `memory_bad_storage` when
/// an allocation failed, else `unimplemented`.
status_t refuse(const char *reason) {
    if (reason == kOutOfMemory) {
        apilog_error(
                "[GRP_MATMUL.ALGO4] cannot serve the call: reason=", reason);
        return status_t::memory_bad_storage;
    }
    return decline(reason);
}

inline void spin_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

// ── The pack store ──────────────────────────────────────────────────────
//
// One pack per expert weight, shared by routed layers and grouped calls:
// the first call of either API to reach a weight packs it, and every later
// call that reads the same weight and shape, whatever its activation, bias
// or source scales, runs on that pack.  An s8 or s4 weight's pack holds its
// blocks and sums, so a weight a call names without its scales (a prepack
// extra) is packed all the same; its scaled pack, on the same blocks, keeps
// the last weight scales a call gave it.

/// One expert weight: its bytes and shape.
struct weight_key {
    const void *weight = nullptr;
    data_type_t wei = data_type_t::none;
    int K = 0;
    int N = 0;
    int ldb = 0;
    int group_size = 0;
    bool trans = false;

    bool operator==(const weight_key &o) const {
        return weight == o.weight && wei == o.wei && K == o.K && N == o.N
                && ldb == o.ldb && group_size == o.group_size
                && trans == o.trans;
    }
};

struct weight_key_hash_t {
    size_t operator()(const weight_key &k) const noexcept {
        size_t h = std::hash<const void *> {}(k.weight);
        for (const int v : {static_cast<int>(k.wei), k.K, k.N, k.ldb,
                     k.group_size, static_cast<int>(k.trans)}) {
            h = (h ^ static_cast<size_t>(v)) * 0x9E3779B97F4A7C15ULL;
        }
        return h;
    }
};

size_t pointer_hash(const void *p) {
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

/// A weight's packs: `bare`, whole for bf16 and f16 and the blocks and sums
/// alone for s8 and s4, and for s8 and s4 `scaled`, on `bare`'s blocks with
/// the weight scales `scales`.  A call that keeps a pack keeps it alive.
struct weight_packs_t {
    expert_pack bare;
    scale_id scales;
    expert_pack scaled;
};

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

/// Twice the flushes so far, plus one while a flush runs: it counts up when
/// a flush begins, before it drops anything, and again when it ends.  A call
/// reads the count when it enters the API; after a flush that ran meanwhile,
/// the packs it makes are its own, and it publishes neither them nor its
/// call site.
std::atomic<uint64_t> &flushes() {
    static std::atomic<uint64_t> count {0};
    return count;
}

/// Serializes flushes, so the count is odd exactly while one runs.
std::mutex &flush_mutex() {
    static auto *mutex = new std::mutex;
    return *mutex;
}

/// Whether a call that read the flush count `since` may publish: no flush
/// ran when it read the count, and none has begun since.
bool may_publish(const uint64_t since) {
    return since % 2 == 0 && flushes().load(std::memory_order_relaxed) == since;
}

/// Drops every grouped call site, and with them the packs only sites keep.
void drop_call_sites();

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

int64_t element_count(const std::vector<int64_t> &dims) {
    int64_t count = 1;
    for (const int64_t d : dims)
        count *= d;
    return count;
}

float scale_at(const void *buff, const data_type_t dt, const size_t i) {
    switch (dt) {
        case data_type_t::bf16:
            return bfloat16_t::bf16_to_f32_val(
                    static_cast<const int16_t *>(buff)[i]);
        case data_type_t::f16:
            return float16_t::f16_to_f32_val(
                    static_cast<const uint16_t *>(buff)[i]);
        default: return static_cast<const float *>(buff)[i];
    }
}

void store_scale(
        void *buff, const data_type_t dt, const size_t i, const float value) {
    switch (dt) {
        case data_type_t::bf16:
            static_cast<int16_t *>(buff)[i]
                    = bfloat16_t::f32_to_bf16_val(value);
            break;
        case data_type_t::f16:
            static_cast<uint16_t *>(buff)[i] = float16_t::f32_to_f16_val(value);
            break;
        default: static_cast<float *>(buff)[i] = value;
    }
}

bool scale_type(const data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::f16;
}

/// How one projection's experts give their weight scales: each a buffer of
/// `groups` rows of N scales, or of one scale for every column.
struct scale_form_t {
    data_type_t dt = data_type_t::none;
    std::vector<int64_t> dims;
    int64_t groups = 1;
    bool per_tensor = false;
};

/// The form of weight scales `dims` of type `dt` for a `K` x `N` weight of
/// type `wei`; false for one a pack cannot take.
bool scale_form(const std::vector<int64_t> &dims, const data_type_t dt,
        const data_type_t wei, const int K, const int N, scale_form_t &out) {
    out = scale_form_t {};
    out.dt = dt;
    out.dims = dims;
    const int64_t count = element_count(dims);
    if (!scale_type(dt) || count <= 0) return false;
    if (count == 1 || count == N) {
        out.per_tensor = count == 1 && N != 1;
        // One group along K, which an s4 pack takes in whole octets.
        return wei != data_type_t::s4 || K % 8 == 0;
    }
    if (count % N != 0) return false;
    out.groups = count / N;
    // A group holds whole K groups of the kernels' dot products.
    const int64_t unit = wei == data_type_t::s4 ? 8 : 4;
    return K % out.groups == 0 && (K / out.groups) % unit == 0;
}

/// The weight's group size for its pack: s4 always, s8 with several groups.
int pack_group_size(
        const data_type_t wei, const int K, const scale_form_t &scales) {
    if (wei == data_type_t::s4) return static_cast<int>(K / scales.groups);
    return scales.groups > 1 ? static_cast<int>(K / scales.groups) : 0;
}

/// The store's key of `weight`, a weight of `desc`.
weight_key key_of(const nr32::weight_desc &desc, const void *weight) {
    weight_key k;
    k.weight = weight;
    k.wei = desc.wei;
    k.K = desc.K;
    k.N = desc.N;
    k.ldb = desc.ldb;
    k.group_size = desc.group_size;
    k.trans = desc.trans;
    return k;
}

/// The bytes of a weight of `desc` in the caller's buffer.
size_t weight_bytes(const nr32::weight_desc &desc) {
    const size_t elems = static_cast<size_t>(desc.trans ? desc.N : desc.K)
            * static_cast<size_t>(desc.ldb);
    switch (desc.wei) {
        case data_type_t::s4: return (elems + 1) / 2;
        case data_type_t::s8: return elems;
        default: return elems * 2;
    }
}

/// What a scaled pack of the weight scales `scale`, of form `form`, holds.
scale_id scale_id_of(const void *scale, const scale_form_t &form) {
    scale_id id;
    id.scale = scale;
    id.dt = form.dt;
    id.groups = form.groups;
    id.per_tensor = form.per_tensor;
    return id;
}

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
    const bool quantized = quantized_weights(desc.wei);
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
                if (!quantized) {
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
            if (!quantized) {
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
    const bool quantized = quantized_weights(desc.wei);
    out.assign(weights.size(), nullptr);
    std::vector<size_t> experts;
    std::vector<const void *> pack_weights;
    for (size_t e = 0; e < weights.size(); ++e) {
        if (weights[e] == nullptr || (quantized && scales[e] == nullptr))
            continue;
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
        if (!quantized) {
            out[e] = hold(std::move(pw.experts[i]));
            continue;
        }
        const char *unserved
                = add_scales(desc, form, pw.experts[i], scales[e], out[e]);
        if (unserved != nullptr) return unserved;
    }
    return nullptr;
}

/// The packs of a projection's experts, for a call that began at flush count
/// `since`: of `weights[e]` with, for s8 and s4, the weight scales
/// `scales[e]` of form `form`; null where `weights[e]` is, and where a
/// quantized weight comes without scales, which packs the weight for a later
/// call to give its scales.  From the store, or packed now on `num_threads`
/// threads, outside the store's lock: of two calls that pack a weight at
/// once, the first to finish publishes its pack.  After a flush that ran
/// since `since`, packed for the call alone.  Null on success, else why not.
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

// ── Routed layers ───────────────────────────────────────────────────────

/// One projection of a routed layer: its experts' packs, from the pack
/// store, and how its tiles read them.
struct projection_state_t {
    nr32::pack_layout layout;
    std::vector<expert_pack> packs;
    /// `packs`' pointers, as the engine reads them.
    std::vector<const nr32::packed_expert *> raw;
    nr32::tile_plan plan;
};

struct layer_state_t {
    std::once_flag packed;
    /// Null when the layer is served, else why not.
    const char *unserved = nullptr;
    /// Both projections run the kernels that read source bytes XOR-ed with
    /// 0x80, so calls flip their s8 rows.
    bool u8_src = false;
    layer_scales scales;
    projection_state_t w13, w2;
};

std::mutex &layer_mutex() {
    static auto *mutex = new std::mutex;
    return *mutex;
}

std::unordered_map<layer_key, std::shared_ptr<layer_state_t>,
        layer_key_hash_t> &
layer_states() {
    static auto *layers = new std::unordered_map<layer_key,
            std::shared_ptr<layer_state_t>, layer_key_hash_t>;
    return *layers;
}

/// What packing a routed layer's projection takes, decided before anything
/// is packed: the weight as its packs read it, and its experts' scales.
struct projection_plan_t {
    nr32::weight_desc desc;
    scale_form_t form;
    std::vector<const void *> scales;
};

/// `p` for all `num_experts` experts, with `groups` weight-scale groups along
/// K for s8 and s4: in `pp`, the weight and its scales; in `out`, the layout
/// and the plan of its tiles for epilogue `ep` and, with `src_groups`, one
/// source scale per row and group.  Null when this CPU runs it, else why not.
const char *plan_projection(projection_state_t &out, projection_plan_t &pp,
        const projection &p, const int num_experts, const data_type_t wei,
        const int64_t groups, const nr32::epilogue ep, const bool u8_src,
        const bool src_groups) {
    const int E = num_experts;
    if (static_cast<int>(p.weight.size()) != E
            || static_cast<int>(p.params.size()) != E) {
        return "weights_unavailable";
    }
    nr32::weight_desc &desc = pp.desc;
    desc.wei = wei;
    desc.K = p.K;
    desc.N = p.N;
    desc.trans = p.trans_weight;
    desc.ldb = p.ldb;
    if (wei == data_type_t::s4 || (wei == data_type_t::s8 && groups > 1)) {
        if (groups <= 0 || p.K % groups != 0) return "unsupported_weight";
        desc.group_size = static_cast<int>(p.K / groups);
    }
    if (!nr32::packable(desc)) return "unsupported_weight";
    pp.scales.assign(static_cast<size_t>(E), nullptr);
    if (quantized_weights(wei)) {
        pp.form.dt = p.params[0].quant_params.wei_scale.dt;
        pp.form.groups = std::max<int64_t>(groups, 1);
        for (int e = 0; e < E; ++e)
            pp.scales[e] = p.params[e].quant_params.wei_scale.buff;
    }
    out.layout = nr32::layout_of(desc);
    if (!nr32::make_plan(out.layout, ep, u8_src, out.plan, src_groups))
        return "no_kernel";
    return nullptr;
}

/// `out`'s packs: `p`'s experts' packs from the pack store, packed now on
/// `num_threads` threads where it lacks them, for a call that began at flush
/// count `since`.  Null on success, else why not.
const char *pack_projection(projection_state_t &out,
        const projection_plan_t &pp, const projection &p, const uint64_t since,
        const int num_threads) {
    const char *unserved = find_packs(pp.desc, pp.form, p.weight, pp.scales,
            since, num_threads, out.packs);
    if (unserved != nullptr) return unserved;
    out.raw.clear();
    for (const expert_pack &pack : out.packs) {
        if (pack == nullptr) return "weights_unavailable";
        out.raw.push_back(pack.get());
    }
    return nullptr;
}

/// The layer's packs and plans, made on its first call and again when its
/// scales change.
std::shared_ptr<const layer_state_t> layer_state(const layer_key &key,
        const layer_scales &scales, const describe_layer &describe,
        const int num_threads) {
    std::shared_ptr<layer_state_t> layer;
    uint64_t since = 0;
    {
        std::lock_guard<std::mutex> guard(layer_mutex());
        since = flushes().load(std::memory_order_relaxed);
        if (since % 2 != 0) {
            // A flush runs: the layer's state is the call's own.
            layer = std::make_shared<layer_state_t>();
            layer->scales = scales;
        } else {
            std::shared_ptr<layer_state_t> &entry = layer_states()[key];
            if (entry == nullptr || !(entry->scales == scales)) {
                entry = std::make_shared<layer_state_t>();
                entry->scales = scales;
            }
            layer = entry;
        }
    }
    std::call_once(layer->packed, [&] {
        layer->u8_src = key.wei == data_type_t::s8 && u8_src_pays();
        projection w13, w2;
        projection_plan_t pp13, pp2;
        layer->unserved = describe(w13, w2);
        if (layer->unserved == nullptr) {
            layer->unserved = plan_projection(layer->w13, pp13, w13,
                    key.num_experts, key.wei, scales.w13_groups,
                    epilogue_of(key.act), layer->u8_src, key.w13_src_groups);
        }
        if (layer->unserved == nullptr) {
            layer->unserved = plan_projection(layer->w2, pp2, w2,
                    key.num_experts, key.wei, scales.w2_groups,
                    nr32::epilogue::plain, layer->u8_src, key.w2_src_groups);
        }
        if (layer->unserved == nullptr) {
            layer->unserved = pack_projection(
                    layer->w13, pp13, w13, since, num_threads);
        }
        if (layer->unserved == nullptr) {
            layer->unserved
                    = pack_projection(layer->w2, pp2, w2, since, num_threads);
        }
        if (layer->unserved != nullptr) {
            layer->w13 = projection_state_t {};
            layer->w2 = projection_state_t {};
        }
        if (test_api::s_capture_ntile_flat_parallel.load(
                    std::memory_order_relaxed)) {
            test_api::s_ntile_flat_parallel_layer_entries.fetch_add(
                    1, std::memory_order_relaxed);
        }
        static const bool s_log = apilog_info_enabled();
        if (s_log) {
            apilog_info("[GRP_MATMUL.ALGO4] layer E=", key.num_experts,
                    " H=", key.H, " I=", key.I,
                    " served=", layer->unserved == nullptr ? 1 : 0);
        }
    });
    if (layer->unserved == kOutOfMemory) {
        // A shortage passes: the next call packs the layer again.
        std::lock_guard<std::mutex> guard(layer_mutex());
        const auto it = layer_states().find(key);
        if (it != layer_states().end() && it->second == layer)
            layer_states().erase(it);
    }
    return layer;
}

/// Rows [row, row + rows) of expert `e`, at most kDecodeMaxM.
struct row_block_t {
    int e;
    int row;
    int rows;
};

/// Blocks [block, block + blocks) of one expert, `rows` rows in all, cut
/// into row runs [run, run + runs): a decode call's group is one block.
struct block_group_t {
    int block;
    int blocks;
    int rows;
    int run = 0;
    int runs = 0;
};

/// Blocks [block, block + blocks) of one group: the rows a unit covers and
/// that a W2 unit waits for.
struct row_run_t {
    int block;
    int blocks;
};

/// Columns [col, col + width) of row runs [run, run + runs) of one group.
struct unit_t {
    int run; // index into the call's row runs
    int runs;
    int col;
    int width;
};

/// The calling thread's buffers, grown to its largest call.
struct scratch_t {
    std::vector<int> offset, fill, row_token, row_of;
    std::vector<row_block_t> blocks;
    std::vector<block_group_t> groups;
    std::vector<row_run_t> runs;
    std::vector<const void *> row_ptrs;
    std::vector<int8_t> tok_rows; // each token's s8 row
    std::vector<float> tok_scale;
    std::vector<uint8_t> rows; // gathered source rows: s8 or 16-bit
    std::vector<int8_t> q2rows; // quantized intermediate rows
    std::vector<float> ascale, s2rows;
    std::vector<uint16_t> inter, stage;
    std::vector<unit_t> queue, planned;
    std::vector<std::atomic<int>> left, ready, gather_next, gather_done;
    std::vector<std::atomic<uint8_t>> claimed;
    /// A grouped call's packs, by operation.
    std::vector<const nr32::packed_expert *> packs13, packs2;

    size_t bytes() const {
        return tok_rows.capacity() + rows.capacity() + q2rows.capacity()
                + (tok_scale.capacity() + ascale.capacity() + s2rows.capacity())
                * sizeof(float)
                + (inter.capacity() + stage.capacity()) * sizeof(uint16_t)
                + (row_token.capacity() + row_of.capacity()) * sizeof(int)
                + row_ptrs.capacity() * sizeof(const void *);
    }
};

scratch_t &scratch() {
    static thread_local scratch_t scratch;
    return scratch;
}

/// Grow-only sizing: every live element is written before it is read.
template <typename T>
void grow(std::vector<T> &v, const size_t n) {
    if (v.size() < n) { v.resize(n); }
}

/// About `target` pieces of `groups` over the call, `g`'s share in proportion
/// to its cost (weight bytes streamed plus rows times weight elements), at
/// most `room`.
int group_pieces(const block_group_t &g, const double total,
        const float wei_bytes, const int target, const int room) {
    const double cost = 16.0 * wei_bytes + g.rows;
    return std::clamp(
            static_cast<int>(std::lround(target * cost / total)), 1, room);
}

double groups_cost(
        const std::vector<block_group_t> &groups, const float wei_bytes) {
    double total = 0.0;
    for (const block_group_t &g : groups)
        total += 16.0 * wei_bytes + g.rows;
    return total;
}

/// Cuts each group into runs of whole blocks: with `split_rows`, as many as
/// keep the source rows and the weights its units read about equal (each unit
/// reads its rows' source and its columns' weights, so sqrt(pieces * rows / N)
/// runs balance the two), else one.
void plan_runs(std::vector<row_run_t> &runs, std::vector<block_group_t> &groups,
        const int N, const int pack_nr, const float wei_bytes, const int target,
        const bool split_rows) {
    const int col_blocks = N / pack_nr;
    const double total = groups_cost(groups, wei_bytes);
    runs.clear();
    for (block_group_t &g : groups) {
        int cuts = 1;
        if (split_rows) {
            const int pieces = group_pieces(
                    g, total, wei_bytes, target, col_blocks * g.blocks);
            cuts = std::clamp(
                    static_cast<int>(std::lround(std::sqrt(
                            static_cast<double>(pieces) * g.rows / N))),
                    1, g.blocks);
        }
        g.run = static_cast<int>(runs.size());
        g.runs = cuts;
        for (int q = 0; q < cuts; ++q) {
            const int r0 = g.block + g.blocks * q / cuts;
            const int r1 = g.block + g.blocks * (q + 1) / cuts;
            runs.push_back({r0, r1 - r0});
        }
    }
}

/// Appends, group by group in the order of `groups`, each group's `N` columns
/// cut into whole `pack_nr` blocks and crossed with runs of its row runs,
/// about `target` units over the call in proportion to each group's cost.
/// With `merge_runs`, a unit covers as many of the group's row runs as keep
/// its source rows and weights balanced for this projection's `N` (the row
/// runs were cut for the W13 projection), else one.
void plan_units(std::vector<unit_t> &out,
        const std::vector<block_group_t> &groups, const int N,
        const int pack_nr, const float wei_bytes, const int target,
        const bool merge_runs) {
    const int col_blocks = N / pack_nr;
    const double total = groups_cost(groups, wei_bytes);
    for (const block_group_t &g : groups) {
        int cuts = g.runs;
        if (merge_runs) {
            const int pieces = group_pieces(
                    g, total, wei_bytes, target, col_blocks * g.runs);
            cuts = std::clamp(
                    static_cast<int>(std::lround(std::sqrt(
                            static_cast<double>(pieces) * g.rows / N))),
                    1, g.runs);
        }
        const int pieces
                = group_pieces(g, total, wei_bytes, target, col_blocks * cuts);
        const int col_cuts
                = std::clamp((pieces + cuts - 1) / cuts, 1, col_blocks);
        for (int p = 0; p < col_cuts; ++p) {
            const int b0 = col_blocks * p / col_cuts;
            const int b1 = col_blocks * (p + 1) / col_cuts;
            for (int q = 0; q < cuts; ++q) {
                const int r0 = g.runs * q / cuts;
                const int r1 = g.runs * (q + 1) / cuts;
                out.push_back({g.run + r0, r1 - r0, b0 * pack_nr,
                        (b1 - b0) * pack_nr});
            }
        }
    }
}

/// Reorders `n` units planned heaviest group first to take turns from the two
/// ends of the plan, so the heaviest groups' units, which keep the cores busy,
/// run beside the lightest groups', which mostly stream weights.
void take_turns(unit_t *units, const int n, std::vector<unit_t> &planned) {
    planned.assign(units, units + n);
    for (int k = 0; k < n; ++k)
        units[k] = planned[k % 2 == 0 ? k / 2 : n - 1 - k / 2];
}

/// The widest run of whole `step` columns whose weights fit the kernels'
/// L2 budget beside one block's rows of `K` elements of `es` bytes.
int l2_chunk_cols(
        const int K, const size_t es, const float wei_bytes, const int step) {
    const double room = static_cast<double>(nr32::l2_budget_bytes())
            - static_cast<double>(kDecodeMaxM) * K * es;
    const int cols = static_cast<int>(room / (K * wei_bytes + 4.0));
    return std::max(cols / step * step, step);
}

/// A call's unit queue and how its units sweep their columns.
struct queue_t {
    bool sweep = false; // a prompt call's
    int n13 = 0; // W13 units, ahead of the W2 units
    int n_units = 0;
    int num_runs = 0;
    int chunk13 = 0;
    int chunk2 = 0;
};

/// Plans a call of `n` experts whose `rows[e]` rows start at `S.offset[e]`:
/// each expert's rows in blocks of at most kDecodeMaxM rows, so an expert of a
/// decode call is one block and one run.  A prompt call's units each cover a
/// run of an expert's blocks and sweep them against one L2-sized chunk of
/// columns at a time, so a unit streams each of its weights once.  A W2 unit
/// waits only for its runs' W13 units.  The unit queue: every W13 unit, then
/// (with `p2`) every W2 unit, heaviest group first, with a prompt call's
/// taking turns from both ends of each projection's units.  `K13` and `K2`
/// are the projections' K, `es` their source bytes per element.
queue_t plan_queue(scratch_t &S, const int *rows, const int n,
        const nr32::tile_plan &p13, const int K13, const nr32::tile_plan *p2,
        const int K2, const size_t es, const float wei_bytes,
        const int num_threads) {
    const bool sweep = *std::max_element(rows, rows + n) > kDecodeMaxM;
    S.blocks.clear();
    S.groups.clear();
    for (int e = 0; e < n; ++e) {
        const int first = static_cast<int>(S.blocks.size());
        for (int r = 0; r < rows[e]; r += kDecodeMaxM) {
            S.blocks.push_back(
                    {e, S.offset[e] + r, std::min(kDecodeMaxM, rows[e] - r)});
            if (!sweep) {
                S.groups.push_back({static_cast<int>(S.blocks.size()) - 1, 1,
                        S.blocks.back().rows});
            }
        }
        if (sweep && rows[e] > 0) {
            S.groups.push_back({first,
                    static_cast<int>(S.blocks.size()) - first, rows[e]});
        }
    }
    std::stable_sort(S.groups.begin(), S.groups.end(),
            [](const block_group_t &a, const block_group_t &b) {
        return a.rows > b.rows;
    });
    const int target
            = (sweep ? kPromptUnitsPerThread : kUnitsPerThread) * num_threads;
    queue_t q;
    q.sweep = sweep;
    q.chunk13 = sweep ? l2_chunk_cols(K13, es, wei_bytes, p13.step) : p13.cols;
    plan_runs(S.runs, S.groups, p13.cols, p13.step, wei_bytes, target, sweep);
    q.num_runs = static_cast<int>(S.runs.size());
    S.queue.clear();
    plan_units(S.queue, S.groups, p13.cols, p13.step, wei_bytes, target,
            /*merge_runs=*/false);
    q.n13 = static_cast<int>(S.queue.size());
    if (p2 != nullptr) {
        q.chunk2
                = sweep ? l2_chunk_cols(K2, es, wei_bytes, p2->step) : p2->cols;
        plan_units(S.queue, S.groups, p2->cols, p2->step, wei_bytes, target,
                sweep);
    }
    q.n_units = static_cast<int>(S.queue.size());
    if (sweep) {
        take_turns(S.queue.data(), q.n13, S.planned);
        take_turns(S.queue.data() + q.n13, q.n_units - q.n13, S.planned);
    }
    if (S.left.size() < static_cast<size_t>(q.num_runs)) {
        const size_t runs = static_cast<size_t>(q.num_runs);
        S.left = std::vector<std::atomic<int>>(runs);
        S.ready = std::vector<std::atomic<int>>(runs);
        S.gather_next = std::vector<std::atomic<int>>(runs);
        S.gather_done = std::vector<std::atomic<int>>(runs);
    }
    for (int r = 0; r < q.num_runs; ++r) {
        S.left[r].store(0, std::memory_order_relaxed);
        S.ready[r].store(0, std::memory_order_relaxed);
        S.gather_next[r].store(0, std::memory_order_relaxed);
        S.gather_done[r].store(0, std::memory_order_relaxed);
    }
    for (int u = 0; u < q.n13; ++u)
        S.left[S.queue[u].run].fetch_add(1, std::memory_order_relaxed);
    if (S.claimed.size() < static_cast<size_t>(q.n_units)) {
        S.claimed = std::vector<std::atomic<uint8_t>>(
                static_cast<size_t>(q.n_units));
    }
    for (int u = 0; u < q.n_units; ++u)
        S.claimed[u].store(0, std::memory_order_relaxed);
    return q;
}

/// Waits, pausing and then yielding, until `flag` reaches `value`.
void wait_for(const std::atomic<int> &flag, const int value) {
    for (int spin = 0; flag.load(std::memory_order_acquire) < value; ++spin) {
        if (spin < kWaitSpinPauses) {
            spin_pause();
        } else {
            std::this_thread::yield();
        }
    }
}

/// Whether a decode call's team of `team` threads runs each thread's own
/// units first: when it has more threads than one L3 has cores, so that it
/// spans several L3s as far as the team size shows, where a call repeated
/// with the same plan then finds each unit's weights in the L3 that read
/// them last.  A smaller team takes units in queue order, which stays
/// balanced when one of its cores runs slow; the rule reads only the team
/// size, so a small team spread over several L3s takes queue order, and a
/// team of SMT threads on one L3 with more threads than its cores, its own
/// units.  `test_api::s_ntile_flat_parallel_own_units_override` pins it.
bool own_units_first(const int team) {
    const int pin = test_api::s_ntile_flat_parallel_own_units_override.load(
            std::memory_order_relaxed);
    if (pin >= 0) return pin != 0;
    static const int l3_cores = [] {
        const native::native_hw_t &hw = native::native_hw();
        return hw.l3_domain_threads / std::max(hw.smt, 1);
    }();
    return l3_cores <= 0 || team > l3_cores;
}

/// Runs, with every thread of the calling parallel region, each unit of the
/// queue `q` once through `run(u)`.  A prompt call's threads, and a decode
/// call's within one L3, take units in queue order.  Otherwise a decode
/// call's thread t of T first runs its own units, those u with u % T == t,
/// its W13 units before its W2 units, so a call repeated with the same plan
/// gives each unit the thread, and the L3, that ran it last; then every
/// thread runs the units still unclaimed.  A W2 unit's wait is on W13 units
/// that are running or that their own thread reaches before any W2 unit, and
/// W13 units never wait.
template <typename Run>
void run_units(scratch_t &S, const queue_t &q, std::atomic<int> &next_unit,
        const Run &run) {
    const int T = omp_get_num_threads(), t = omp_get_thread_num();
    if (q.sweep || !own_units_first(T)) {
        for (;;) {
            const int u = next_unit.fetch_add(1, std::memory_order_relaxed);
            if (u >= q.n_units) break;
            run(u);
        }
        return;
    }
    const auto take = [&S](const int u) {
        return S.claimed[u].load(std::memory_order_relaxed) == 0
                && S.claimed[u].exchange(1, std::memory_order_relaxed) == 0;
    };
    for (int u = t; u < q.n13; u += T) {
        if (take(u)) run(u);
    }
    for (int u = q.n13 + t; u < q.n_units; u += T) {
        if (take(u)) run(u);
    }
    for (int u = 0; u < q.n_units; ++u) {
        if (take(u)) run(u);
    }
}

/// XORs each of `n` bytes with 0x80: an s8 row as the u8-source kernels read.
void flip_to_u8(int8_t *row, const int n) {
    constexpr uint64_t kFlip = 0x8080808080808080ULL;
    auto *bytes = reinterpret_cast<unsigned char *>(row);
    int c = 0;
    for (; c + 8 <= n; c += 8) {
        uint64_t w;
        std::memcpy(&w, bytes + c, sizeof(w));
        w ^= kFlip;
        std::memcpy(bytes + c, &w, sizeof(w));
    }
    for (; c < n; ++c)
        bytes[c] = static_cast<unsigned char>(bytes[c] ^ 0x80u);
}

// ── The engine ──────────────────────────────────────────────────────────
//
// Both APIs run their calls here: a routed call once it has routed its
// tokens to expert-major rows, a grouped call with its operations' rows.

/// A tile's operands besides its columns: its source rows and their scales,
/// its expert's bias row, and where its output rows go.
struct tile_io {
    const void *src = nullptr;
    int lda = 0;
    const float *scale = nullptr;
    const void *bias = nullptr;
    nr32::bias_kind bias_t = nr32::bias_kind::none;
    void *dst = nullptr;
    int ldc = 0;
};

/// Quantizes `len` values to s8 in `groups` equal runs, each with its scale,
/// and XORs the row with 0x80 for the u8-source kernels when `u8`.  One row
/// per call: the quantizer runs a single row on the calling thread, but may
/// fork a team for several.
void quantize_row(const uint16_t *in, int8_t *out, float *scale, const int len,
        const int groups, const bool u8) {
    const int run = len / groups;
    for (int i = 0; i < groups; ++i) {
        reorder::dynamic_per_token_quant_bf16_s8_native(
                in + i * run, out + i * run, scale + i, 1, run);
    }
    if (u8) flip_to_u8(out, len);
}

/// What the engine runs: `n` experts, expert e's `rows[e]` rows from
/// `S.offset[e]` on, through W13 with its plan's epilogue and, when `fused`,
/// W2; then the weighted `reduce` of `reduce_cols` columns, if any.
struct engine_t {
    int n = 0;
    const int *rows = nullptr;
    bool quantized = false;
    bool fused = false;
    bool u8_src = false;
    float wei_bytes = 2.0f;
    int K13 = 0;
    int K2 = 0;
    /// W2's source scales per row.
    int G2 = 1;
    const nr32::pack_layout *l13 = nullptr;
    const nr32::pack_layout *l2 = nullptr;
    const nr32::tile_plan *p13 = nullptr;
    const nr32::tile_plan *p2 = nullptr;
    const nr32::packed_expert *const *packs13 = nullptr;
    const nr32::packed_expert *const *packs2 = nullptr;
    const group_matmul_moe_postop_params *reduce = nullptr;
    int reduce_cols = 0;
    bool reduce_f16 = false;
    int num_threads = 0;
};

/// Runs `E` on the calling thread's scratch, whose `offset` the caller has
/// set.  The front end gives the rows, on every thread of the region:
/// `prologue()`, which may run a worksharing loop, before any unit;
/// `before_w13(run)` before a W13 unit reads row run `run`; `w13(block)` and
/// `w2(block)`, a block's tile operands; and `op1_row(e, row)`, Op1's output
/// row, which a quantized fused call quantizes into the scratch's s8 W2
/// source rows once its run's W13 units are done.
template <typename Front>
status_t run_engine(const engine_t &E, const Front &front) {
    scratch_t &S = scratch();
    const size_t es = E.quantized ? 1 : sizeof(uint16_t);
    const queue_t q = plan_queue(S, E.rows, E.n, *E.p13, E.K13,
            E.fused ? E.p2 : nullptr, E.K2, es, E.wei_bytes, E.num_threads);
    std::atomic<int> next_unit {0};

    // The region reaches the calling thread's buffers through `S`, a
    // reference, so the workers share them.
    const unit_t *const plan = S.queue.data();
    const row_block_t *const blocks = S.blocks.data();
    const row_run_t *const runs = S.runs.data();
    int8_t *const q2rows = S.q2rows.data();
    float *const s2 = S.s2rows.data();

#pragma omp parallel num_threads(E.num_threads)
    {
        front.prologue();

        // Columns [c.col, c.col + c.width) of every block of `c`'s run, one
        // chunk of columns at a time.
        const auto run_w13 = [&](const unit_t &c) {
            front.before_w13(c.run);
            const row_run_t run = runs[c.run];
            for (int col = c.col; col < c.col + c.width; col += q.chunk13) {
                const int width = std::min(q.chunk13, c.col + c.width - col);
                for (int i = run.block; i < run.block + run.blocks; ++i) {
                    const row_block_t b = blocks[i];
                    const tile_io io = front.w13(b);
                    nr32::run_tile(*E.p13, *E.l13, *E.packs13[b.e], b.rows,
                            io.src, io.lda, io.scale, io.bias, io.bias_t,
                            io.dst, io.ldc, col, width);
                }
            }
        };
        const auto run_w2 = [&](const unit_t &c) {
            const int b0 = runs[c.run].block;
            const row_run_t &last = runs[c.run + c.runs - 1];
            for (int col = c.col; col < c.col + c.width; col += q.chunk2) {
                const int width = std::min(q.chunk2, c.col + c.width - col);
                for (int i = b0; i < last.block + last.blocks; ++i) {
                    const row_block_t b = blocks[i];
                    const tile_io io = front.w2(b);
                    nr32::run_tile(*E.p2, *E.l2, *E.packs2[b.e], b.rows, io.src,
                            io.lda, io.scale, io.bias, io.bias_t, io.dst,
                            io.ldc, col, width);
                }
            }
        };

        run_units(S, q, next_unit, [&](const int u) {
            const unit_t c = plan[u];
            if (u < q.n13) {
                run_w13(c);
                if (E.fused
                        && S.left[c.run].fetch_sub(1, std::memory_order_acq_rel)
                                == 1) {
                    const row_run_t run = runs[c.run];
                    const row_block_t &head = blocks[run.block];
                    const row_block_t &tail
                            = blocks[run.block + run.blocks - 1];
                    for (int r = head.row;
                            E.quantized && r < tail.row + tail.rows; ++r) {
                        quantize_row(front.op1_row(head.e, r),
                                q2rows + static_cast<size_t>(r) * E.K2,
                                s2 + static_cast<size_t>(r) * E.G2, E.K2, E.G2,
                                E.u8_src);
                    }
                    S.ready[c.run].store(1, std::memory_order_release);
                }
                return;
            }
            for (int r = c.run; r < c.run + c.runs; ++r) {
                wait_for(S.ready[r], 1);
            }
            run_w2(c);
        });

        if (E.reduce != nullptr) {
#pragma omp barrier
            const int D = E.reduce_cols;
            const int cuts = col_chunks_per_row(
                    E.reduce->num_tokens, D, E.num_threads);
            const int span = col_chunk_span(D, cuts);
#pragma omp for schedule(static)
            for (int it = 0; it < E.reduce->num_tokens * cuts; ++it) {
                const int t = it / cuts;
                const int d0 = (it % cuts) * span;
                if (d0 < D) {
                    const int len = std::min(D, d0 + span) - d0;
                    if (E.reduce_f16) {
                        group_matmul_moe_postop_f16_cols(E.reduce, t, d0, len);
                    } else {
                        group_matmul_moe_postop_bf16_cols(E.reduce, t, d0, len);
                    }
                }
            }
        }
    }
    if (S.bytes() > kScratchKeepBytes) { S = scratch_t {}; }
    return status_t::success;
}

// ── Routed calls ────────────────────────────────────────────────────────

/// A routed call's rows: each (token, slot) pair is a row of its expert, in
/// token order, `S.row_token` naming its token.  Quantized layers quantize
/// each token once and copy its s8 row and scales to its slots; 16-bit
/// layers copy the token's row.  The W13 units that reach a run first share
/// the copies of its rows, a few at a time, so the copies overlap other
/// units' work and no unit waits long.  Op1 writes the scratch's intermediate
/// rows and Op2 its staging rows, which the reduce sums into the output.
struct routed_front {
    scratch_t &S;
    const moe_call &call;
    int H, I;
    size_t es;
    bool quantized, u8_src;
    int G1, G2;
    /// Expert e's biases are row e of the caller's `[E, N]` tensors.
    const char *bias13;
    size_t bias13_row;
    nr32::bias_kind bias13_t;
    const char *bias2;
    size_t bias2_row;
    nr32::bias_kind bias2_t;

    void prologue() const {
        if (!quantized) return;
#pragma omp for schedule(static)
        for (int t = 0; t < call.num_tokens; ++t) {
            quantize_row(call.src + static_cast<size_t>(t) * call.src_ld,
                    S.tok_rows.data() + static_cast<size_t>(t) * H,
                    S.tok_scale.data() + static_cast<size_t>(t) * G1, H, G1,
                    u8_src);
        }
    }

    void before_w13(const int r) const {
        const row_run_t run = S.runs[r];
        const row_block_t &tail = S.blocks[run.block + run.blocks - 1];
        const int first = S.blocks[run.block].row;
        const int count = tail.row + tail.rows - first;
        uint8_t *const rows = S.rows.data();
        for (;;) {
            const int i = S.gather_next[r].fetch_add(
                    kGatherRows, std::memory_order_relaxed);
            if (i >= count) break;
            const int n = std::min(kGatherRows, count - i);
            for (int row = first + i; row < first + i + n; ++row) {
                const int t = S.row_token[row];
                if (quantized) {
                    std::memcpy(rows + static_cast<size_t>(row) * H,
                            S.tok_rows.data() + static_cast<size_t>(t) * H,
                            static_cast<size_t>(H));
                    std::memcpy(S.ascale.data() + static_cast<size_t>(row) * G1,
                            S.tok_scale.data() + static_cast<size_t>(t) * G1,
                            sizeof(float) * G1);
                } else {
                    std::memcpy(rows + static_cast<size_t>(row) * H * es,
                            call.src + static_cast<size_t>(t) * call.src_ld,
                            static_cast<size_t>(H) * es);
                }
            }
            S.gather_done[r].fetch_add(n, std::memory_order_release);
        }
        wait_for(S.gather_done[r], count);
    }

    tile_io w13(const row_block_t &b) const {
        tile_io io;
        io.src = S.rows.data() + static_cast<size_t>(b.row) * H * es;
        io.lda = H;
        io.scale = quantized ? S.ascale.data() + static_cast<size_t>(b.row) * G1
                             : nullptr;
        io.bias = bias13 != nullptr ? bias13 + b.e * bias13_row : nullptr;
        io.bias_t = bias13_t;
        io.dst = S.inter.data() + static_cast<size_t>(b.row) * I;
        io.ldc = I;
        return io;
    }

    tile_io w2(const row_block_t &b) const {
        tile_io io;
        io.src = quantized ? static_cast<const void *>(S.q2rows.data()
                                     + static_cast<size_t>(b.row) * I)
                           : static_cast<const void *>(op1_row(b.e, b.row));
        io.lda = I;
        io.scale = quantized ? S.s2rows.data() + static_cast<size_t>(b.row) * G2
                             : nullptr;
        io.bias = bias2 != nullptr ? bias2 + b.e * bias2_row : nullptr;
        io.bias_t = bias2_t;
        io.dst = S.stage.data() + static_cast<size_t>(b.row) * H;
        io.ldc = H;
        return io;
    }

    const uint16_t *op1_row(int, const int row) const {
        return S.inter.data() + static_cast<size_t>(row) * I;
    }
};

status_t run_call(const layer_key &key, const layer_scales &scales,
        const describe_layer &describe, const moe_call &call) {
    const int E = key.num_experts, H = key.H, I = key.I;
    const int num_tokens = call.num_tokens, topk = call.topk;
    const int num_threads = call.num_threads;
    const int num_pairs = num_tokens * topk;
    const std::shared_ptr<const layer_state_t> layer
            = layer_state(key, scales, describe, num_threads);
    if (layer->unserved != nullptr) { return refuse(layer->unserved); }
    const layer_state_t &L = *layer;
    const int *const counts = call.counts;

    // Expert-major rows: expert e owns rows [offset[e], offset[e+1]), in
    // token order; `row_of` maps each (token, slot) pair to its row for the
    // reduce.
    scratch_t &S = scratch();
    S.offset.assign(static_cast<size_t>(E) + 1, 0);
    for (int e = 0; e < E; ++e)
        S.offset[e + 1] = S.offset[e] + counts[e];
    S.fill.assign(S.offset.begin(), S.offset.end() - 1);
    grow(S.row_token, static_cast<size_t>(num_pairs));
    grow(S.row_of, static_cast<size_t>(num_pairs));
    for (int p = 0; p < num_pairs; ++p) {
        const int r = S.fill[call.local_experts[p]]++;
        S.row_token[r] = p / topk;
        S.row_of[p] = r;
    }

    // Quantized weights read s8 rows of the source and of the intermediate,
    // with G1 and G2 scales per row: one, or one per weight scale group;
    // 16-bit weights read both as they are.  `es` is the bytes per element of
    // either.
    const bool quantized = quantized_weights(key.wei);
    const size_t es = quantized ? 1 : sizeof(uint16_t);
    const int G1 = L.w13.plan.src_groups ? L.w13.layout.groups : 1;
    const int G2 = L.w2.plan.src_groups ? L.w2.layout.groups : 1;
    grow(S.rows, static_cast<size_t>(num_pairs) * H * es + kRowSlack);
    if (quantized) {
        grow(S.tok_rows, static_cast<size_t>(num_tokens) * H);
        grow(S.tok_scale, static_cast<size_t>(num_tokens) * G1);
        grow(S.ascale, static_cast<size_t>(num_pairs) * G1);
        grow(S.q2rows, static_cast<size_t>(num_pairs) * I + kRowSlack);
        grow(S.s2rows, static_cast<size_t>(num_pairs) * G2);
    }
    grow(S.inter, static_cast<size_t>(num_pairs) * I);
    grow(S.stage, static_cast<size_t>(num_pairs) * H);
    grow(S.row_ptrs, static_cast<size_t>(num_pairs));
    for (int p = 0; p < num_pairs; ++p) {
        S.row_ptrs[p] = S.stage.data() + static_cast<size_t>(S.row_of[p]) * H;
    }
    group_matmul_moe_postop_params reduce;
    reduce.num_tokens = num_tokens;
    reduce.topk = topk;
    reduce.output = call.dst;
    reduce.ldc_output = call.dst_ld;
    reduce.topk_weights = call.topk_weights;
    reduce.skip_weighted = call.skip_weighted;
    reduce.row_ptrs = S.row_ptrs.data();

    const char *const bias13 = static_cast<const char *>(call.w13_bias);
    const char *const bias2 = static_cast<const char *>(call.w2_bias);
    const routed_front front {S, call, H, I, es, quantized, L.u8_src, G1, G2,
            bias13,
            bias13 != nullptr
                    ? static_cast<size_t>(size_of(key.w13_bias)) * 2 * I
                    : 0,
            bias_kind_of(key.w13_bias), bias2,
            bias2 != nullptr ? static_cast<size_t>(size_of(key.w2_bias)) * H
                             : 0,
            bias_kind_of(key.w2_bias)};
    engine_t en;
    en.n = E;
    en.rows = counts;
    en.quantized = quantized;
    en.fused = true;
    en.u8_src = L.u8_src;
    en.wei_bytes = wei_bytes_of(key.wei);
    en.K13 = H;
    en.K2 = I;
    en.G2 = G2;
    en.l13 = &L.w13.layout;
    en.l2 = &L.w2.layout;
    en.p13 = &L.w13.plan;
    en.p2 = &L.w2.plan;
    en.packs13 = L.w13.raw.data();
    en.packs2 = L.w2.raw.data();
    en.reduce = &reduce;
    en.reduce_cols = H;
    en.reduce_f16 = key.wei == data_type_t::f16;
    en.num_threads = num_threads;
    const status_t status = run_engine(en, front);
    if (test_api::s_capture_ntile_flat_parallel.load(
                std::memory_order_relaxed)) {
        test_api::s_ntile_flat_parallel_completed.fetch_add(
                1, std::memory_order_relaxed);
        if (L.u8_src) {
            test_api::s_ntile_flat_parallel_u8_src_calls.fetch_add(
                    1, std::memory_order_relaxed);
        }
    }
    return status;
}

// ── Grouped calls ───────────────────────────────────────────────────────

/// How an operation's source scales cover its rows.
enum class source_scales { per_row, per_tensor, per_group, unsupported };

/// The form of source scales `dims` for `M` rows against a weight of `groups`
/// scale groups: `{M, 1}` (or none) per row, `{1}` per tensor, `{M, groups}`
/// per (row, group).
source_scales source_form(
        const std::vector<int64_t> &dims, const int M, const int groups) {
    if (dims.empty()) return source_scales::per_row;
    const int64_t cols = dims.size() >= 2 ? dims.back() : 1;
    if (cols <= 0) return source_scales::unsupported;
    const int64_t rows = element_count(dims) / cols;
    if (cols == 1) {
        return rows == M    ? source_scales::per_row
                : rows == 1 ? source_scales::per_tensor
                            : source_scales::unsupported;
    }
    return groups > 1 && cols == groups && rows == M
            ? source_scales::per_group
            : source_scales::unsupported;
}

/// A grouped call's arguments, as `execute_grouped` takes them.
struct grouped_args {
    int num_ops;
    const std::vector<char> &layout;
    const std::vector<bool> &transA;
    const std::vector<bool> &transB;
    const std::vector<int> &M;
    const std::vector<int> &N;
    const std::vector<int> &K;
    const std::vector<float> &alpha;
    const std::vector<const void *> &src;
    const std::vector<int> &lda;
    const std::vector<const void *> &weight;
    const std::vector<int> &ldb;
    const std::vector<const void *> &bias;
    const std::vector<float> &beta;
    const std::vector<void *> &dst;
    const std::vector<int> &ldc;
    const std::vector<bool> &is_weights_const;
    const std::vector<matmul_params> &params;
    grp_matmul_gated_act_t act;
    const grp_matmul_fused_moe_params *fused;
    const group_matmul_moe_postop_params *reduce;
    int num_threads;
};

/// A grouped call as it runs.
struct grouped_spec_t {
    /// The first operation with rows: the others must match it.
    int first = -1;
    data_type_t wei = data_type_t::none;
    data_type_t src = data_type_t::none;
    data_type_t dst = data_type_t::none;
    bool quantized = false;
    /// The caller's rows are s8 already, with their scales.
    bool src_s8 = false;
    bool u8_src = false;
    nr32::epilogue ep = nr32::epilogue::plain;
    nr32::weight_desc w13, w2;
    scale_form_t scales13, scales2;
    /// Op1's source scales: `per_group`, one per (row, W13 group), or
    /// `per_row` (each caller's s8 operation may give one for all its rows).
    source_scales source = source_scales::per_row;
    /// Op2's source scales are one per (row, W2 group): Op1's are per group
    /// and W2 has several groups.
    bool per_group2 = false;
    /// Op1 writes the library's rows instead of `dst`.
    bool op1_internal = false;
    /// Op2 writes the caller's source rows instead of `dst_down`.
    bool op2_internal = false;
    /// Expert weights the call packs: its operations' and the prepack
    /// extras'.
    int num_packs = 0;
};

/// Every operation with rows has a pointer in `ptrs`, all null or none.
bool all_null(const std::vector<void *> &ptrs, const grouped_args &a) {
    for (int e = 0; e < a.num_ops; ++e) {
        if (a.M[e] > 0 && e < static_cast<int>(ptrs.size())
                && ptrs[e] != nullptr) {
            return false;
        }
    }
    return true;
}

/// Null when ALGO 4 runs the grouped call `a` with rows, `s` then describing
/// it; else why not.
const char *check_grouped(const grouped_args &a, grouped_spec_t &s) {
    const int n = a.num_ops;
    const matmul_params &p0 = a.params[s.first];
    s.wei = p0.dtypes.wei;
    s.src = p0.dtypes.src;
    s.dst = p0.dtypes.dst;
    s.quantized = quantized_weights(s.wei);
    if (s.wei == data_type_t::bf16 || s.wei == data_type_t::f16) {
        if (s.src != s.wei || s.dst != s.wei || p0.dynamic_quant)
            return "unsupported_types";
    } else if (!s.quantized || s.dst != data_type_t::bf16) {
        return "unsupported_types";
    } else if (s.src == data_type_t::s8 && s.wei == data_type_t::s8
            && !p0.dynamic_quant) {
        s.src_s8 = true;
    } else if (s.src != data_type_t::bf16 || !p0.dynamic_quant
            || p0.dtypes.compute != data_type_t::s8) {
        return "unsupported_types";
    }
    s.ep = epilogue_of(a.act);
    const int K = a.K[s.first], N = a.N[s.first];
    if (nr32::gated(s.ep) && N % 2 != 0) return "shape";
    const int out13 = nr32::gated(s.ep) ? N / 2 : N;
    s.w13.wei = s.wei;
    s.w13.K = K;
    s.w13.N = N;
    s.w13.trans = a.transB[s.first];
    s.w13.ldb = a.ldb[s.first];
    const auto &ws0 = p0.quant_params.wei_scale;
    if (s.quantized) {
        if (!scale_form(ws0.dims, ws0.dt, s.wei, K, N, s.scales13))
            return "unsupported_weight_scale";
        s.w13.group_size = pack_group_size(s.wei, K, s.scales13);
        s.source = source_form(p0.quant_params.src_scale.dims, a.M[s.first],
                static_cast<int>(s.scales13.groups));
    }
    s.op1_internal = a.fused != nullptr && all_null(a.dst, a);
    for (int e = 0; e < n; ++e) {
        if (a.M[e] == 0) continue;
        const matmul_params &p = a.params[e];
        const auto &qp = p.quant_params;
        if (a.K[e] != K || a.N[e] != N || a.ldb[e] != s.w13.ldb
                || a.transB[e] != s.w13.trans || p.dtypes.wei != s.wei
                || p.dtypes.src != s.src || p.dtypes.dst != s.dst
                || p.dtypes.compute != p0.dtypes.compute
                || p.dynamic_quant != p0.dynamic_quant) {
            return "non_uniform_operations";
        }
        if ((a.layout[e] != 'r' && a.layout[e] != 'R') || a.transA[e]
                || a.alpha[e] != 1.0f || a.beta[e] != 0.0f) {
            return "unsupported_operation";
        }
        if (!a.is_weights_const[e]) return "non_constant_weights";
        if (effective_weight_cache_type(p.weight_cache_type) < 1)
            return "weight_cache_disabled";
        if (!p.postop_.empty()) return "post_ops";
        if (p.mem_format_b != 'n' || p.packing.pack_format_b != 0)
            return "prepacked_weights";
        if (qp.wei_zp.buff != nullptr || qp.src_zp.buff != nullptr
                || qp.dst_scale.buff != nullptr || qp.dst_zp.buff != nullptr) {
            return "unsupported_quantization";
        }
        if (a.src[e] == nullptr || a.weight[e] == nullptr || a.lda[e] < K)
            return "shape";
        if (a.bias[e] != nullptr
                && bias_kind_of(p.dtypes.bias) == nr32::bias_kind::none) {
            return "unsupported_bias";
        }
        if (!s.op1_internal
                && (static_cast<int>(a.dst.size()) <= e || a.dst[e] == nullptr
                        || static_cast<int>(a.ldc.size()) <= e
                        || a.ldc[e] < out13)) {
            return "shape";
        }
        if (!s.quantized) continue;
        if (qp.wei_scale.buff == nullptr || qp.wei_scale.dt != ws0.dt
                || qp.wei_scale.dims != ws0.dims) {
            return "unsupported_weight_scale";
        }
        // Every operation's source scales group as the first's: per group,
        // or per row (one row's per-tensor scale is its per-row one).  Only
        // the caller's s8 rows come with one scale for all of them.
        source_scales form = source_form(
                qp.src_scale.dims, a.M[e], static_cast<int>(s.scales13.groups));
        if (form == source_scales::per_tensor && !s.src_s8)
            form = source_scales::unsupported;
        const bool per_group = form == source_scales::per_group;
        if (form == source_scales::unsupported
                || per_group != (s.source == source_scales::per_group)
                || (qp.src_scale.buff != nullptr
                        && !scale_type(qp.src_scale.dt))
                || (s.src_s8 && qp.src_scale.buff == nullptr)) {
            return "unsupported_source_scale";
        }
    }
    if (s.source == source_scales::per_tensor)
        s.source = source_scales::per_row;
    if (a.fused != nullptr) {
        const grp_matmul_fused_moe_params &f = *a.fused;
        const size_t ops = static_cast<size_t>(n);
        if (f.down_weight.size() < ops || f.N_down.size() < ops
                || f.ldb_down.size() < ops) {
            return "shape";
        }
        s.w2.wei = s.wei;
        s.w2.K = out13;
        s.w2.N = f.N_down[s.first];
        s.w2.trans = s.w13.trans;
        s.w2.ldb = f.ldb_down[s.first];
        if (s.quantized) {
            if (f.down_scale.size() < ops) return "unsupported_weight_scale";
            const auto &ds0 = f.down_scale[s.first];
            if (!scale_form(ds0.dims, ds0.dt, s.wei, s.w2.K, s.w2.N, s.scales2))
                return "unsupported_weight_scale";
            s.w2.group_size = pack_group_size(s.wei, s.w2.K, s.scales2);
            s.per_group2 = s.source == source_scales::per_group
                    && s.scales2.groups > 1;
        }
        s.op2_internal = all_null(f.dst_down, a);
        if (s.op2_internal && s.src != s.dst) return "unsupported_types";
        for (int e = 0; e < n; ++e) {
            if (a.M[e] == 0) continue;
            if (f.N_down[e] != s.w2.N || f.ldb_down[e] != s.w2.ldb)
                return "non_uniform_operations";
            if (f.down_weight[e] == nullptr) return "shape";
            if (e < static_cast<int>(f.down_zp.size())
                    && f.down_zp[e].buff != nullptr) {
                return "unsupported_quantization";
            }
            const bool scaled = e < static_cast<int>(f.down_scale.size())
                    && f.down_scale[e].buff != nullptr;
            if (s.quantized ? !scaled || f.down_scale[e].dt != s.scales2.dt
                                    || f.down_scale[e].dims != s.scales2.dims
                            : scaled) {
                return "unsupported_weight_scale";
            }
            if (e < static_cast<int>(f.bias_down.size())
                    && f.bias_down[e] != nullptr
                    && bias_kind_of(f.bias_dt_down) == nr32::bias_kind::none) {
                return "unsupported_bias";
            }
            if (s.op2_internal ? a.lda[e] < s.w2.N
                               : static_cast<int>(f.dst_down.size()) <= e
                                    || f.dst_down[e] == nullptr
                                    || static_cast<int>(f.ldc_down.size()) <= e
                                    || f.ldc_down[e] < s.w2.N) {
                return "shape";
            }
        }
    }
    s.u8_src = s.wei == data_type_t::s8 && u8_src_pays();
    s.num_packs = n;
    const matmul_params &lead = a.params[0];
    if (lead.active_matmul > 0 && lead.total_matmul > lead.active_matmul) {
        s.num_packs = static_cast<int>(
                std::min({static_cast<size_t>(lead.total_matmul),
                        a.weight.size(), a.K.size(), a.N.size(), a.ldb.size(),
                        a.transB.size(), a.is_weights_const.size()}));
    }
    return nullptr;
}

/// Whether the call packs expert `e`'s W13 weight: an operation with rows
/// does; another expert does when it matches them, an expert past the call's
/// `params` taken to be of the call's types.
bool packs_w13(const grouped_args &a, const grouped_spec_t &s, const int e) {
    if (a.weight[e] == nullptr) return false;
    if (e < a.num_ops && a.M[e] > 0) return true;
    if (a.K[e] != s.w13.K || a.N[e] != s.w13.N || a.ldb[e] != s.w13.ldb
            || a.transB[e] != s.w13.trans || !a.is_weights_const[e]) {
        return false;
    }
    if (e >= static_cast<int>(a.params.size())) return true;
    const matmul_params &p = a.params[e];
    return p.dtypes.wei == s.wei && p.mem_format_b == 'n'
            && p.packing.pack_format_b == 0;
}

/// Expert `e`'s W13 weight scales when the call gives them in its operations'
/// form, else null (always for 16-bit weights).
const void *scales_w13(
        const grouped_args &a, const grouped_spec_t &s, const int e) {
    if (!s.quantized || e >= static_cast<int>(a.params.size())) return nullptr;
    const auto &q = a.params[e].quant_params.wei_scale;
    return q.dt == s.scales13.dt && q.dims == s.scales13.dims ? q.buff
                                                              : nullptr;
}

/// `packs_w13` for expert `e`'s W2 weight.
bool packs_w2(const grouped_args &a, const grouped_spec_t &s, const int e) {
    const grp_matmul_fused_moe_params &f = *a.fused;
    if (e >= static_cast<int>(f.down_weight.size())
            || f.down_weight[e] == nullptr) {
        return false;
    }
    if (e < a.num_ops && a.M[e] > 0) return true;
    return packs_w13(a, s, e) && e < static_cast<int>(f.N_down.size())
            && e < static_cast<int>(f.ldb_down.size()) && f.N_down[e] == s.w2.N
            && f.ldb_down[e] == s.w2.ldb;
}

/// `scales_w13` for expert `e`'s W2 weight.
const void *scales_w2(
        const grouped_args &a, const grouped_spec_t &s, const int e) {
    const grp_matmul_fused_moe_params &f = *a.fused;
    if (!s.quantized || e >= static_cast<int>(f.down_scale.size()))
        return nullptr;
    const auto &q = f.down_scale[e];
    return q.dt == s.scales2.dt && q.dims == s.scales2.dims ? q.buff : nullptr;
}

/// A fingerprint of a projection's pool: the weights its call names, the
/// operations' and the prepack extras', whatever their order and whichever
/// have rows, and their shape and types.
size_t extras_fingerprint(const nr32::weight_desc &desc,
        const scale_form_t &form, const std::vector<const void *> &actives,
        const std::vector<const void *> &extras) {
    size_t sum = 0;
    for (const auto *pool : {&actives, &extras}) {
        for (const void *w : *pool) {
            if (w != nullptr) sum += pointer_hash(w);
        }
    }
    size_t h = sum;
    const auto mix = [&h](size_t v) { h = (h ^ v) * 0x9E3779B97F4A7C15ULL; };
    for (const int v : {static_cast<int>(desc.wei), desc.K, desc.N, desc.ldb,
                 desc.group_size, static_cast<int>(desc.trans),
                 static_cast<int>(form.dt), static_cast<int>(form.groups)}) {
        mix(static_cast<size_t>(v));
    }
    return h;
}

/// Pool fingerprints kept before they are all dropped.
constexpr size_t kMaxWarmedPools = 4096;

/// The packs of a grouped call's operations with rows (null for the others),
/// from `find_packs` for a call that began at flush count `since`;
/// `weight(e)` and `scales(e)` give expert e's weight (null when the call
/// doesn't pack it) and weight scales.  The call's other experts, its prepack
/// extras, are packed the first time a call names them.
template <typename Weight, typename Scales>
const char *call_packs(const grouped_args &a, const grouped_spec_t &s,
        const nr32::weight_desc &desc, const scale_form_t &form,
        const Weight &weight, const Scales &scales, const uint64_t since,
        std::vector<expert_pack> &out) {
    std::vector<const void *> w(s.num_packs, nullptr), sc(s.num_packs, nullptr);
    std::vector<const void *> extra_w, extra_sc;
    for (int e = 0; e < s.num_packs; ++e) {
        const void *const p = weight(e);
        if (p == nullptr) continue;
        if (e < a.num_ops && a.M[e] > 0) {
            w[e] = p;
            sc[e] = scales(e);
        } else {
            extra_w.push_back(p);
            extra_sc.push_back(scales(e));
        }
    }
    const char *unserved
            = find_packs(desc, form, w, sc, since, a.num_threads, out);
    if (unserved != nullptr) return unserved;
    for (int e = 0; e < s.num_packs; ++e) {
        if (w[e] != nullptr && out[e] == nullptr)
            return "unsupported_weight_scale";
    }
    if (extra_w.empty()) return nullptr;
    const size_t fingerprint = extras_fingerprint(desc, form, w, extra_w);
    {
        // Extras packed after a flush would serve no later call.
        std::shared_lock<std::shared_mutex> lock(store_mutex());
        if (!may_publish(since) || warmed_extras().count(fingerprint) != 0) {
            return nullptr;
        }
    }
    std::vector<expert_pack> extras;
    unserved = find_packs(
            desc, form, extra_w, extra_sc, since, a.num_threads, extras);
    if (unserved != nullptr) return unserved;
    std::unique_lock<std::shared_mutex> lock(store_mutex());
    if (!may_publish(since)) return nullptr;
    if (warmed_extras().size() >= kMaxWarmedPools) warmed_extras().clear();
    warmed_extras().insert(fingerprint);
    return nullptr;
}

/// The packs of a grouped call site: the weights its calls name (their
/// operations' and their prepack extras'), each with the scales of its scaled
/// pack.  A call that names the same weights, in any order, finds its packs
/// here with one lookup instead of one per expert in the store, and the site
/// keeps them alive for it without counting references to each.
struct call_site_t {
    struct slot_t {
        const void *weight = nullptr;
        scale_id scales;
        const nr32::packed_expert *pack = nullptr;
    };
    /// Open-addressed by weight, a power of two of slots each.
    std::vector<slot_t> w13, w2;
    std::vector<expert_pack> keep;
    /// The projections' layouts and tile plans.
    nr32::pack_layout l13, l2;
    nr32::tile_plan p13, p2;
};

const call_site_t::slot_t *find_slot(
        const std::vector<call_site_t::slot_t> &table, const void *weight) {
    if (table.empty() || weight == nullptr) return nullptr;
    const size_t mask = table.size() - 1;
    for (size_t i = pointer_hash(weight) & mask;; i = (i + 1) & mask) {
        if (table[i].weight == weight) return &table[i];
        if (table[i].weight == nullptr) return nullptr;
    }
}

// Never destroyed, as the store.
std::shared_mutex &sites_mutex() {
    static auto *mutex = new std::shared_mutex;
    return *mutex;
}

std::unordered_map<size_t, std::shared_ptr<const call_site_t>> &sites() {
    static auto *sites = new std::unordered_map<size_t,
            std::shared_ptr<const call_site_t>>;
    return *sites;
}

/// How many times the sites were dropped, under `sites_mutex()`.  A call
/// inserts its site only when they weren't dropped since it looked it up.
uint64_t &site_epoch() {
    static uint64_t epoch = 0;
    return epoch;
}

void drop_call_sites() {
    std::unique_lock<std::shared_mutex> lock(sites_mutex());
    sites().clear();
    ++site_epoch();
}

/// Sites kept before they are all dropped.  A site holds its weights' pack
/// pointers and scales, and keeps alive only packs the store holds: the store
/// replacing a scaled pack drops every site.
constexpr size_t kMaxCallSites = 1024;

/// Expert `e`'s W13 and W2 weights, null where the call doesn't pack them.
const void *weight13(const grouped_args &a, const grouped_spec_t &s, int e) {
    return packs_w13(a, s, e) ? a.weight[e] : nullptr;
}
const void *weight2(const grouped_args &a, const grouped_spec_t &s, int e) {
    return a.fused != nullptr && packs_w2(a, s, e) ? a.fused->down_weight[e]
                                                   : nullptr;
}

/// The call's site: the weights it names, whatever their order, their
/// shapes and scale forms, and what its tile plans depend on.
size_t site_key(const grouped_args &a, const grouped_spec_t &s) {
    const size_t downs = a.fused != nullptr ? a.fused->down_weight.size() : 0;
    size_t h = 0;
    for (int e = 0; e < s.num_packs; ++e) {
        const void *const down = static_cast<size_t>(e) < downs
                ? a.fused->down_weight[e]
                : nullptr;
        h += pointer_hash(a.weight[e])
                ^ (pointer_hash(down) * 0x9E3779B97F4A7C15ULL);
    }
    const auto mix = [&h](size_t v) { h = (h ^ v) * 0x9E3779B97F4A7C15ULL; };
    for (const nr32::weight_desc *d : {&s.w13, &s.w2}) {
        mix(weight_key_hash_t {}(key_of(*d, nullptr)));
    }
    for (const scale_form_t *f : {&s.scales13, &s.scales2}) {
        mix(static_cast<size_t>(f->dt));
        mix(static_cast<size_t>(f->groups));
        mix(static_cast<size_t>(f->per_tensor));
    }
    mix(static_cast<size_t>(s.ep));
    mix(static_cast<size_t>(s.u8_src));
    mix(static_cast<size_t>(s.source == source_scales::per_group));
    mix(static_cast<size_t>(s.num_packs));
    return h;
}

/// The packs of the call's operations with rows, from `site`, into the
/// scratch; false when the site lacks one or holds it with other scales.
/// `check_grouped` has given every such operation its scales in the call's
/// forms.
bool site_packs(const call_site_t &site, const grouped_args &a,
        const grouped_spec_t &s, scratch_t &S) {
    const bool fused = a.fused != nullptr;
    S.packs13.assign(static_cast<size_t>(s.num_packs), nullptr);
    if (fused) S.packs2.assign(static_cast<size_t>(s.num_packs), nullptr);
    for (int e = 0; e < a.num_ops; ++e) {
        if (a.M[e] <= 0) continue;
        const call_site_t::slot_t *p = find_slot(site.w13, a.weight[e]);
        if (p == nullptr || p->pack == nullptr
                || (s.quantized
                        && !(p->scales
                                == scale_id_of(
                                        a.params[e].quant_params.wei_scale.buff,
                                        s.scales13)))) {
            return false;
        }
        S.packs13[e] = p->pack;
        if (!fused) continue;
        p = find_slot(site.w2, a.fused->down_weight[e]);
        if (p == nullptr || p->pack == nullptr
                || (s.quantized
                        && !(p->scales
                                == scale_id_of(a.fused->down_scale[e].buff,
                                        s.scales2)))) {
            return false;
        }
        S.packs2[e] = p->pack;
    }
    return true;
}

/// A site of the packs the store holds for the call's weights, and of the
/// call's layouts and tile plans.
std::shared_ptr<const call_site_t> make_site(const grouped_args &a,
        const grouped_spec_t &s, const nr32::pack_layout &l13,
        const nr32::pack_layout &l2, const nr32::tile_plan &p13,
        const nr32::tile_plan &p2) {
    auto site = std::make_shared<call_site_t>();
    site->l13 = l13;
    site->l2 = l2;
    site->p13 = p13;
    site->p2 = p2;
    size_t slots = 8;
    while (slots < 2 * static_cast<size_t>(s.num_packs))
        slots *= 2;
    site->w13.resize(slots);
    if (a.fused != nullptr) site->w2.resize(slots);
    std::shared_lock<std::shared_mutex> lock(store_mutex());
    const auto add
            = [&](std::vector<call_site_t::slot_t> &table,
                      const nr32::weight_desc &desc, const void *weight) {
        if (weight == nullptr) return;
        const auto it = store().find(key_of(desc, weight));
        if (it == store().end()) return;
        const weight_packs_t &w = it->second;
        const expert_pack &pack = s.quantized ? w.scaled : w.bare;
        const size_t mask = table.size() - 1;
        for (size_t i = pointer_hash(weight) & mask;; i = (i + 1) & mask) {
            call_site_t::slot_t &slot = table[i];
            if (slot.weight != nullptr && slot.weight != weight) continue;
            slot.weight = weight;
            slot.scales = w.scales;
            slot.pack = pack.get();
            if (pack != nullptr) site->keep.push_back(pack);
            return;
        }
    };
    for (int e = 0; e < s.num_packs; ++e) {
        add(site->w13, s.w13, weight13(a, s, e));
        if (a.fused != nullptr) add(site->w2, s.w2, weight2(a, s, e));
    }
    return site;
}

/// A grouped call's rows: operation e's are the caller's, `lda[e]` apart.
/// Quantized weights read each row as s8 in the scratch, quantized there (or
/// copied from the caller's s8 rows) before any unit; 16-bit weights read the
/// caller's rows in place.  Op1 writes the caller's `dst`, or the scratch's
/// rows when the call keeps no Op1 output; Op2 writes `dst_down`, or the
/// caller's source rows.
struct grouped_front {
    scratch_t &S;
    const grouped_args &a;
    const grouped_spec_t &s;
    int K, out13, K2, G1, G2;

    /// Row `row` of the call, of operation e, in Op1's output.
    uint16_t *out13_row(const int e, const int row) const {
        return s.op1_internal
                ? S.inter.data() + static_cast<size_t>(row) * out13
                : static_cast<uint16_t *>(a.dst[e])
                        + static_cast<size_t>(row - S.offset[e]) * a.ldc[e];
    }

    int out13_ld(const int e) const {
        return s.op1_internal ? out13 : a.ldc[e];
    }

    void prologue() const {
        if (!s.quantized) return;
        const int n = a.num_ops;
        const int *const offset = S.offset.data();
        int8_t *const rows = reinterpret_cast<int8_t *>(S.rows.data());
        float *const ascale = S.ascale.data();
        const int total = offset[n];
#pragma omp for schedule(static)
        for (int g = 0; g < total; ++g) {
            const int e = static_cast<int>(
                                  std::upper_bound(offset, offset + n + 1, g)
                                  - offset)
                    - 1;
            const int j = g - offset[e];
            int8_t *const row = rows + static_cast<size_t>(g) * K;
            float *const row_scale = ascale + static_cast<size_t>(g) * G1;
            const auto &qs = a.params[e].quant_params.src_scale;
            if (s.src_s8) {
                std::memcpy(row,
                        static_cast<const int8_t *>(a.src[e])
                                + static_cast<size_t>(j) * a.lda[e],
                        static_cast<size_t>(K));
                if (s.u8_src) flip_to_u8(row, K);
                const bool one = element_count(qs.dims) == 1;
                for (int i = 0; i < G1; ++i) {
                    row_scale[i] = scale_at(qs.buff, qs.dt,
                            one ? 0 : static_cast<size_t>(j) * G1 + i);
                }
            } else {
                quantize_row(static_cast<const uint16_t *>(a.src[e])
                                + static_cast<size_t>(j) * a.lda[e],
                        row, row_scale, K, G1, s.u8_src);
                for (int i = 0;
                        qs.buff != nullptr && !qs.dims.empty() && i < G1; ++i) {
                    store_scale(const_cast<void *>(qs.buff), qs.dt,
                            static_cast<size_t>(j) * G1 + i, row_scale[i]);
                }
            }
        }
    }

    void before_w13(int) const {}

    tile_io w13(const row_block_t &b) const {
        const int e = b.e;
        tile_io io;
        io.src = s.quantized
                ? static_cast<const void *>(
                          reinterpret_cast<const int8_t *>(S.rows.data())
                          + static_cast<size_t>(b.row) * K)
                : static_cast<const void *>(
                          static_cast<const uint16_t *>(a.src[e])
                          + static_cast<size_t>(b.row - S.offset[e])
                                  * a.lda[e]);
        io.lda = s.quantized ? K : a.lda[e];
        io.scale = s.quantized
                ? S.ascale.data() + static_cast<size_t>(b.row) * G1
                : nullptr;
        io.bias = a.bias[e];
        io.bias_t = a.bias[e] != nullptr ? bias_kind_of(a.params[e].dtypes.bias)
                                         : nr32::bias_kind::none;
        io.dst = out13_row(e, b.row);
        io.ldc = out13_ld(e);
        return io;
    }

    tile_io w2(const row_block_t &b) const {
        const grp_matmul_fused_moe_params &f = *a.fused;
        const int e = b.e, j = b.row - S.offset[e];
        tile_io io;
        io.src = s.quantized ? static_cast<const void *>(S.q2rows.data()
                                       + static_cast<size_t>(b.row) * K2)
                             : static_cast<const void *>(out13_row(e, b.row));
        io.lda = s.quantized ? K2 : out13_ld(e);
        io.scale = s.quantized
                ? S.s2rows.data() + static_cast<size_t>(b.row) * G2
                : nullptr;
        io.bias = e < static_cast<int>(f.bias_down.size()) ? f.bias_down[e]
                                                           : nullptr;
        io.bias_t = io.bias != nullptr ? bias_kind_of(f.bias_dt_down)
                                       : nr32::bias_kind::none;
        io.dst = s.op2_internal
                ? static_cast<uint16_t *>(const_cast<void *>(a.src[e]))
                        + static_cast<size_t>(j) * a.lda[e]
                : static_cast<uint16_t *>(f.dst_down[e])
                        + static_cast<size_t>(j) * f.ldc_down[e];
        io.ldc = s.op2_internal ? a.lda[e] : f.ldc_down[e];
        return io;
    }

    const uint16_t *op1_row(const int e, const int row) const {
        return out13_row(e, row);
    }
};

status_t run_grouped(
        const grouped_args &a, const grouped_spec_t &s, const uint64_t since) {
    const int n = a.num_ops, num_threads = a.num_threads;
    const bool fused = a.fused != nullptr;

    // The call's packs and plans: from its site, else from the store and
    // planned now, which then make the site for the site's next calls.
    scratch_t &S = scratch();
    const size_t key = site_key(a, s);
    std::shared_ptr<const call_site_t> site;
    uint64_t epoch = 0;
    {
        std::shared_lock<std::shared_mutex> lock(sites_mutex());
        const auto it = sites().find(key);
        if (it != sites().end()) site = it->second;
        epoch = site_epoch();
    }
    std::vector<expert_pack> packs13, packs2;
    if (site == nullptr || !site_packs(*site, a, s, S)) {
        // Whether this CPU runs both projections, before anything is packed.
        if (!nr32::packable(s.w13) || (fused && !nr32::packable(s.w2)))
            return decline("unsupported_weight");
        const nr32::pack_layout l13 = nr32::layout_of(s.w13);
        const nr32::pack_layout l2
                = fused ? nr32::layout_of(s.w2) : nr32::pack_layout {};
        nr32::tile_plan p13, p2;
        if (!nr32::make_plan(l13, s.ep, s.u8_src, p13,
                    s.source == source_scales::per_group)
                || (fused
                        && !nr32::make_plan(l2, nr32::epilogue::plain, s.u8_src,
                                p2, s.per_group2))) {
            return decline("no_kernel");
        }
        const char *unserved
                = call_packs(a, s, s.w13, s.scales13, [&](const int e) {
            return weight13(a, s, e);
        }, [&](const int e) { return scales_w13(a, s, e); }, since, packs13);
        if (unserved != nullptr) return refuse(unserved);
        if (fused) {
            unserved = call_packs(a, s, s.w2, s.scales2, [&](const int e) {
                return weight2(a, s, e);
            }, [&](const int e) { return scales_w2(a, s, e); }, since, packs2);
            if (unserved != nullptr) return refuse(unserved);
        }
        S.packs13.assign(static_cast<size_t>(s.num_packs), nullptr);
        S.packs2.assign(packs2.size(), nullptr);
        for (size_t e = 0; e < packs13.size(); ++e)
            S.packs13[e] = packs13[e].get();
        for (size_t e = 0; e < packs2.size(); ++e)
            S.packs2[e] = packs2[e].get();
        site = make_site(a, s, l13, l2, p13, p2);
        std::unique_lock<std::shared_mutex> lock(sites_mutex());
        if (site_epoch() == epoch && may_publish(since)) {
            if (sites().size() >= kMaxCallSites) sites().clear();
            sites()[key] = site;
        }
    }
    const nr32::pack_layout &l13 = site->l13, &l2 = site->l2;
    const nr32::tile_plan &p13 = site->p13, &p2 = site->p2;
    // Source scales per row of each projection.
    const int G1 = p13.src_groups ? l13.groups : 1;
    const int G2 = fused && p2.src_groups ? l2.groups : 1;

    // Operation e owns rows [offset[e], offset[e+1]) of the call.
    S.offset.assign(static_cast<size_t>(n) + 1, 0);
    for (int e = 0; e < n; ++e)
        S.offset[e + 1] = S.offset[e] + a.M[e];
    const int total = S.offset[n];
    const int K = s.w13.K, out13 = p13.out_cols, K2 = out13;
    if (s.quantized) {
        grow(S.rows, static_cast<size_t>(total) * K + kRowSlack);
        grow(S.ascale, static_cast<size_t>(total) * G1);
        if (fused) {
            grow(S.q2rows, static_cast<size_t>(total) * K2 + kRowSlack);
            grow(S.s2rows, static_cast<size_t>(total) * G2);
        }
    }
    if (s.op1_internal) grow(S.inter, static_cast<size_t>(total) * out13);

    const grouped_front front {S, a, s, K, out13, K2, G1, G2};
    engine_t en;
    en.n = n;
    en.rows = a.M.data();
    en.quantized = s.quantized;
    en.fused = fused;
    en.u8_src = s.u8_src;
    en.wei_bytes = wei_bytes_of(s.wei);
    en.K13 = K;
    en.K2 = K2;
    en.G2 = G2;
    en.l13 = &l13;
    en.l2 = &l2;
    en.p13 = &p13;
    en.p2 = &p2;
    en.packs13 = S.packs13.data();
    en.packs2 = S.packs2.data();
    en.reduce = a.reduce;
    en.reduce_cols = fused ? s.w2.N : out13;
    en.reduce_f16 = s.dst == data_type_t::f16;
    en.num_threads = num_threads;
    const status_t status = run_engine(en, front);
    if (test_api::s_capture_ntile_flat_parallel.load(
                std::memory_order_relaxed)) {
        test_api::s_ntile_flat_parallel_grouped_completed.fetch_add(
                1, std::memory_order_relaxed);
        if (s.u8_src) {
            test_api::s_ntile_flat_parallel_u8_src_calls.fetch_add(
                    1, std::memory_order_relaxed);
        }
    }
    return status;
}

} // namespace

status_t execute(const layer_key &key, const layer_scales &scales,
        const describe_layer &describe, const moe_call &call) {
    if (test_api::s_pause_ntile_flat_parallel.load(std::memory_order_acquire)) {
        test_api::s_ntile_flat_parallel_paused.store(
                true, std::memory_order_release);
        while (test_api::s_pause_ntile_flat_parallel.load(
                std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
    if (key.act == grp_matmul_gated_act_t::none || key.num_experts < 1
            || call.num_tokens <= 0 || call.topk <= 0
            || call.num_threads <= 0) {
        return decline("shape");
    }
    const int num_pairs = call.num_tokens * call.topk;
    for (int p = 0; p < num_pairs; ++p) {
        if (call.local_experts[p] < 0) return decline("non_resident_expert");
        if (call.local_experts[p] >= key.num_experts) return decline("shape");
    }
    try {
        return run_call(key, scales, describe, call);
    } catch (const std::length_error &) {
        return status_t::memory_bad_size;
    } catch (const std::bad_alloc &) { return status_t::memory_bad_storage; }
}

status_t execute_grouped(const size_t num_ops, const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params,
        const grp_matmul_gated_act_t act,
        const grp_matmul_fused_moe_params *fused,
        const group_matmul_moe_postop_params *reduce, const int num_threads,
        const uint64_t since) {
    const grouped_args a {static_cast<int>(num_ops), layout, transA, transB, M,
            N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, act, fused, reduce, num_threads};
    if (a.num_ops < 2 || num_threads <= 0) return decline("shape");
    grouped_spec_t s;
    for (int e = 0; e < a.num_ops; ++e) {
        if (M[e] < 0) return decline("shape");
        if (M[e] > 0 && s.first < 0) s.first = e;
    }
    // The weighted reduce sums rows of the call's last output.
    const int lead = std::max(s.first, 0);
    if (static_cast<size_t>(lead) >= N.size()
            || static_cast<size_t>(lead) >= params.size()
            || (fused != nullptr
                    && static_cast<size_t>(lead) >= fused->N_down.size())) {
        return decline("shape");
    }
    const int reduce_cols = fused != nullptr ? fused->N_down[lead]
                                             : op2_k_for_act(N[lead], act);
    if (reduce != nullptr) {
        const status_t valid = validate_group_matmul_moe_postop(
                reduce, reduce_cols, params[lead].dtypes.dst);
        if (valid != status_t::success) return valid;
    }
    if (s.first < 0) {
        return reduce == nullptr
                ? status_t::success
                : group_matmul_moe_postop_execute(reduce, reduce_cols,
                          num_threads, params[lead].dtypes.dst);
    }
    const char *unserved = check_grouped(a, s);
    if (unserved != nullptr) return decline(unserved);
    try {
        return run_grouped(a, s, since);
    } catch (const std::length_error &) {
        return status_t::memory_bad_size;
    } catch (const std::bad_alloc &) { return status_t::memory_bad_storage; }
}

uint64_t flush_count() {
    return flushes().load(std::memory_order_acquire);
}

void flush() {
    // The count is odd from before anything is dropped until everything is:
    // a call that read it before or during the flush publishes no pack and
    // keeps no layer or site, whatever the order of the steps.
    std::lock_guard<std::mutex> serial(flush_mutex());
    flushes().fetch_add(1, std::memory_order_acq_rel);
    {
        std::unique_lock<std::shared_mutex> lock(store_mutex());
        store().clear();
        warmed_extras().clear();
    }
    if (void (*const during)()
            = test_api::s_ntile_flat_parallel_during_flush.exchange(
                    nullptr, std::memory_order_relaxed)) {
        during();
    }
    {
        std::lock_guard<std::mutex> guard(layer_mutex());
        layer_states().clear();
    }
    drop_call_sites();
    flushes().fetch_add(1, std::memory_order_acq_rel);
}

void reset_thread_local_scratch() {
    scratch() = scratch_t {};
}

} // namespace ntile_flat_parallel

size_t test_api::ntile_flat_parallel_layers() {
    std::lock_guard<std::mutex> guard(ntile_flat_parallel::layer_mutex());
    return ntile_flat_parallel::layer_states().size();
}

size_t test_api::ntile_flat_parallel_packs() {
    std::shared_lock<std::shared_mutex> lock(
            ntile_flat_parallel::store_mutex());
    return ntile_flat_parallel::store().size();
}

size_t test_api::ntile_flat_parallel_call_sites() {
    std::shared_lock<std::shared_mutex> lock(
            ntile_flat_parallel::sites_mutex());
    return ntile_flat_parallel::sites().size();
}

size_t test_api::ntile_flat_parallel_warmed_pools() {
    std::shared_lock<std::shared_mutex> lock(
            ntile_flat_parallel::store_mutex());
    return ntile_flat_parallel::warmed_extras().size();
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
