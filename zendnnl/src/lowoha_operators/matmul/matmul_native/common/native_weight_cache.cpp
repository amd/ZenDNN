/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <condition_variable>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include "common/data_types.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_operator_bridge.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_ends.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::dtype_info;
using zendnnl::error_handling::apilog_info;
using zendnnl::error_handling::apilog_verbose;
using zendnnl::error_handling::apilog_verbose_enabled;
using zendnnl::error_handling::log_error;

namespace {

struct native_key_t {
    const void *weight;
    int K, N, ldb, pack_nr;
    bool transB, interleave;
    data_type_t wei_dt;
    bool aocl_blocked;

    bool operator==(const native_key_t &o) const {
        return weight == o.weight && K == o.K && N == o.N && ldb == o.ldb
                && pack_nr == o.pack_nr && transB == o.transB
                && interleave == o.interleave && wei_dt == o.wei_dt
                && aocl_blocked == o.aocl_blocked;
    }
};

struct native_key_hash_t {
    size_t operator()(const native_key_t &k) const {
        size_t h = std::hash<const void *>()(k.weight);
        auto mix
                = [&h](size_t v) { h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2); };
        mix(static_cast<uint32_t>(k.K));
        mix(static_cast<uint32_t>(k.N));
        mix(static_cast<uint32_t>(k.ldb));
        mix(static_cast<uint32_t>(k.pack_nr));
        mix((k.transB ? 1u : 0u) | (k.interleave ? 2u : 0u)
                | (static_cast<uint32_t>(k.wei_dt) << 2)
                | (k.aocl_blocked ? 1u << 31 : 0u));
        return h;
    }
};

/// A cached layout, with the bytes its pack read from the weight pointer
/// (the raw span, the AOCL prepack or the in-place image) and the digests
/// of their ends at the time.
struct native_cached_pack_t {
    std::unique_ptr<uint8_t, native_aligned_free_t> buf;
    size_t source_bytes = 0;
    weight_ends_t source_ends;
    /// The buffer an in-place rewrite of bytes it was packed from runs on,
    /// when the layout still matched them: it is served without its digests
    /// until the rewrite ends.
    const void *rewritten_by = nullptr;
};

/// A buffer an in-place pack rewrote: the request it was rewritten for
/// (weight = the buffer start), the image bytes and the digests of their
/// two ends.  Reserved while the rewrite runs: the bytes are not the image
/// yet and have no digests, and the buffer is in packing.
struct native_image_t {
    native_raw_weight_t layout;
    size_t bytes = 0;
    weight_ends_t ends;
    bool reserved = false;
    /// While reserved: the bytes of the rewriting call's buffer, within
    /// which lie the layouts the rewrite marks.
    size_t owned = 0;
};

using image_map_t = std::map<std::uintptr_t, native_image_t>;
using pack_map_t = std::unordered_map<native_key_t, native_cached_pack_t,
        native_key_hash_t>;

/// [start, end) of every image, sorted by start; immutable once published.
using image_spans_t = std::vector<std::pair<std::uintptr_t, std::uintptr_t>>;

struct native_cache_state_t {
    /// Shared for lookups, exclusive to change packs, images or packing.
    /// No pack and no in-place rewrite runs under it.
    std::shared_mutex mutex;
    /// Buffers a pack of modes 1 / 2 works on now (out of place, or an
    /// in-place rewrite), outside the mutex.  A call on such a buffer waits
    /// for packed and looks again, so the packs of one buffer run one at a
    /// time and never twice for one layout, while other buffers go on.
    std::unordered_set<const void *> packing;
    std::condition_variable_any packed;
    /// [start, end) of the bytes that packs and other readers read from
    /// weight buffers now, outside the mutex, at every weight cache mode.  An
    /// in-place rewrite reserves its record first, so a read that starts
    /// after it sees the image, then waits until no read overlaps the bytes
    /// it writes.
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> reads;
    /// Rewrites waiting for reads to end: the end of a read then notifies
    /// packed.
    size_t rewrites_waiting = 0;
    pack_map_t packs;
    /// The layouts of entries dropped because their buffer no longer held
    /// the bytes they were packed from.  Calls read a layout outside the
    /// mutex, so a call on another thread may still run on one: they are
    /// freed by clear_native_weight_caches only, and count toward the cache
    /// capacity with packs.
    std::vector<std::unique_ptr<uint8_t, native_aligned_free_t>> retired;
    /// Keyed by buffer start; a pointer inside an image resolves to it.
    image_map_t images;
    std::atomic<size_t> image_count {0};
    /// The spans of images, for a check without the mutex; null when they
    /// could not be copied, and every check then takes the mutex.
    std::shared_ptr<const image_spans_t> spans;
    /// Bumped after spans changes.
    std::atomic<uint64_t> spans_gen {0};
    /// Bumped, under the mutex held exclusively, before a cached layout or
    /// an image record goes and whenever the images change: the memo entries
    /// (thread_memo) of an older generation do not serve.
    std::atomic<uint64_t> generation {1};
    /// In-place rewrites from the reservation of their record to its end;
    /// while one runs, the checks read no weight bytes
    /// (native_weight_check_t).
    /// Changed under the mutex held exclusively.
    std::atomic<size_t> rewrites {0};
};

native_cache_state_t &cache_state() {
    static native_cache_state_t state;
    return state;
}

void *alloc_layout(size_t bytes) {
    return zendnnl_aligned_alloc(64, (bytes + 63) & ~size_t {63});
}

/// Under the mutex held exclusively: the memo entries made so far stop
/// serving.
void bump_generation_locked() {
    cache_state().generation.fetch_add(1, std::memory_order_release);
}

/// Copies the spans of the images for the check without the mutex, under
/// the mutex held exclusively.
void publish_spans_locked() {
    bump_generation_locked();
    native_cache_state_t &state = cache_state();
    std::shared_ptr<image_spans_t> spans;
    try {
        spans = std::make_shared<image_spans_t>();
        spans->reserve(state.images.size());
        for (const auto &it : state.images)
            spans->emplace_back(it.first, it.first + it.second.bytes);
    } catch (const std::bad_alloc &) { spans.reset(); }
    state.spans = std::move(spans);
    // Before the count: a thread that sees the new count sees the new
    // generation too.
    state.spans_gen.fetch_add(1, std::memory_order_release);
    state.image_count.store(state.images.size(), std::memory_order_release);
}

/// The calling thread's copy of the spans of the images, as of spans_gen
/// gen.
struct thread_spans_t {
    uint64_t gen = 0;
    std::shared_ptr<const image_spans_t> spans;
};

thread_spans_t &thread_spans() {
    thread_local thread_spans_t spans;
    return spans;
}

/// The calling thread's copy of the spans, copied again under the mutex
/// held shared when the images changed since it last looked; nullptr while
/// no buffer holds an image.
const thread_spans_t *current_thread_spans() {
    native_cache_state_t &state = cache_state();
    if (state.image_count.load(std::memory_order_acquire) == 0) return nullptr;
    thread_spans_t &tl = thread_spans();
    if (state.spans_gen.load(std::memory_order_acquire) != tl.gen) {
        std::shared_lock<std::shared_mutex> lock(state.mutex);
        tl.spans = state.spans;
        tl.gen = state.spans_gen.load(std::memory_order_relaxed);
    }
    return &tl;
}

/// False only when no span of tl overlaps [p, p + bytes).
bool spans_overlap(const thread_spans_t &tl, const void *p, size_t bytes) {
    if (!tl.spans) return true;
    // The first span that ends past p: it holds p or starts after it.
    const auto at = reinterpret_cast<std::uintptr_t>(p);
    const auto it = std::upper_bound(tl.spans->begin(), tl.spans->end(), at,
            [](std::uintptr_t a,
                    const std::pair<std::uintptr_t, std::uintptr_t> &s) {
        return a < s.second;
    });
    return it != tl.spans->end() && (it->first <= at || it->first - at < bytes);
}

/// False only when no image overlaps [p, p + bytes); takes no lock while
/// the images have not changed since the calling thread last looked.
bool image_may_overlap(const void *p, size_t bytes) {
    const thread_spans_t *tl = current_thread_spans();
    return tl != nullptr && spans_overlap(*tl, p, bytes);
}

/// image_may_overlap without a lock: also true when the images changed
/// since the calling thread last looked.
bool image_may_overlap_unlocked(const void *p, size_t bytes) {
    native_cache_state_t &state = cache_state();
    if (state.image_count.load(std::memory_order_acquire) == 0) return false;
    const thread_spans_t &tl = thread_spans();
    return state.spans_gen.load(std::memory_order_acquire) != tl.gen
            || spans_overlap(tl, p, bytes);
}

/// Forgets the record at it, under the mutex held exclusively.
image_map_t::iterator erase_image_locked(image_map_t::iterator it) {
    it = cache_state().images.erase(it);
    publish_spans_locked();
    return it;
}

/// True when [at, at + bytes) lies in [lo, lo + owned).
bool within(std::uintptr_t at, size_t bytes, std::uintptr_t lo, size_t owned) {
    return at >= lo && at - lo <= owned && bytes <= owned - (at - lo);
}

/// False when an end of [at, at + bytes) that lies in the caller's buffer
/// [lo, lo + owned) no longer has its digest in ends.  Bytes outside the
/// caller's buffer are never read, since they may no longer be mapped: ends
/// outside it count as held.
bool ends_hold(std::uintptr_t at, size_t bytes, const weight_ends_t &ends,
        std::uintptr_t lo, size_t owned) {
    const void *p = reinterpret_cast<const void *>(at);
    const size_t end = std::min(bytes, kWeightEndBytes);
    if (within(at, end, lo, owned)
            && weight_end_digest(p, bytes, false) != ends.head)
        return false;
    return !within(at + bytes - end, end, lo, owned)
            || weight_end_digest(p, bytes, true) == ends.tail;
}

/// A cached layout whose buffer still holds what its pack read, for a
/// caller whose buffer is the owned bytes at the key's weight.
bool pack_holds(
        const native_key_t &k, const native_cached_pack_t &e, size_t owned) {
    const auto at = reinterpret_cast<std::uintptr_t>(k.weight);
    return e.rewritten_by != nullptr
            || ends_hold(at, e.source_bytes, e.source_ends, at, owned);
}

/// Whether the image at it still holds, for a caller whose buffer is
/// [lo, lo + owned): only its ends inside that buffer are checked, so a
/// record whose two ends lie outside it is kept as recorded.  A reserved
/// record holds.  Under the mutex.
bool image_holds(
        image_map_t::const_iterator it, std::uintptr_t lo, size_t owned) {
    const native_image_t &img = it->second;
    return img.reserved || ends_hold(it->first, img.bytes, img.ends, lo, owned);
}

/// The image p points into, under the mutex; interior is set when p lies
/// past its start and stale when it no longer holds for a caller whose
/// buffer is the owned bytes at p.
image_map_t::iterator image_at_locked(
        const void *p, size_t owned, bool &interior, bool &stale) {
    auto &images = cache_state().images;
    const auto at = reinterpret_cast<std::uintptr_t>(p);
    auto it = images.upper_bound(at);
    if (it == images.begin()) return images.end();
    --it;
    if (at - it->first >= it->second.bytes) return images.end();
    interior = at != it->first;
    stale = !image_holds(it, at, owned);
    return it;
}

/// The first image that starts after p and before p + extent, under the
/// mutex: a request from p that reads extent bytes reads into it.
image_map_t::iterator image_after_locked(const void *p, size_t extent) {
    auto &images = cache_state().images;
    const auto at = reinterpret_cast<std::uintptr_t>(p);
    const auto it = images.upper_bound(at);
    return it != images.end() && it->first - at < extent ? it : images.end();
}

void log_stale_image(const native_image_t &img) {
    apilog_info("native weight: the buffer at ", img.layout.weight,
            " no longer holds the in-place image recorded for it; read as "
            "a raw weight");
}

/// The live image p points into, for a caller whose buffer is the owned bytes
/// at p, under the mutex held exclusively; interior is set when p lies past
/// its start.  Records that no longer hold, at p or starting inside the
/// extent bytes a request from p reads, are forgotten; after is set when a
/// live one starts inside them.
const native_image_t *live_image_locked(const void *p, size_t extent,
        size_t owned, bool &interior, bool &after) {
    auto &images = cache_state().images;
    const auto at = reinterpret_cast<std::uintptr_t>(p);
    bool stale = false;
    auto it = image_at_locked(p, owned, interior, stale);
    if (it != images.end() && stale) {
        log_stale_image(it->second);
        erase_image_locked(it);
        it = images.end();
        interior = false;
    }
    if (it != images.end()) return &it->second;
    for (auto next = image_after_locked(p, extent); next != images.end();
            next = image_after_locked(p, extent)) {
        if (image_holds(next, at, owned)) {
            after = true;
            return nullptr;
        }
        log_stale_image(next->second);
        erase_image_locked(next);
    }
    return nullptr;
}

bool same_layout(const native_raw_weight_t &a, const native_raw_weight_t &b) {
    return a.wei_dt == b.wei_dt && a.K == b.K && a.N == b.N && a.ldb == b.ldb
            && a.transB == b.transB && a.pack_nr == b.pack_nr
            && a.interleave_split_halves == b.interleave_split_halves;
}

/// Packs the AOCL inverse of an aocl_blocked w into dst.
status_t pack_from_aocl(
        const native_raw_weight_t &w, void *dst, int num_threads) {
    std::unique_ptr<void, native_aligned_free_t> raw;
    const status_t st = aocl_prepack_unreorder(w.weight, w.K, w.N, w.ldb,
            w.transB, w.wei_dt == data_type_t::s8, num_threads, raw);
    if (st != status_t::success) {
        // Nothing was written: AOCL can still serve its own prepack.
        log_error("native weight: the AOCL blocked weight=", w.weight,
                " was not unreordered (status=", static_cast<int>(st), ")");
        return status_t::unimplemented;
    }
    return native_pack_weight(w.wei_dt, raw.get(), w.K, w.N, w.ldb, w.pack_nr,
            w.transB, w.interleave_split_halves, dst, num_threads);
}

/// Packs w into dst; when its buffer holds image, from the raw weight
/// rebuilt out of the image, which says nothing about bytes beyond it.  The
/// rebuild reads the whole image, so the caller's buffer of owned bytes must
/// cover it.
status_t pack_from_source(const native_raw_weight_t &w,
        const native_image_t *image, size_t owned, void *dst, int num_threads) {
    if (image == nullptr) {
        if (w.aocl_blocked) return pack_from_aocl(w, dst, num_threads);
        return native_pack_weight(w.wei_dt, w.weight, w.K, w.N, w.ldb,
                w.pack_nr, w.transB, w.interleave_split_halves, dst,
                num_threads);
    }
    const native_raw_weight_t &img = image->layout;
    const size_t extent = static_cast<size_t>(img.K) * img.N;
    if (img.wei_dt != data_type_t::bf16 || w.wei_dt != img.wei_dt
            || native_raw_weight_bytes(w.wei_dt, w.K, w.N, w.ldb, w.transB)
                    > extent * sizeof(uint16_t)
            || image->bytes > owned) {
        log_error("native weight: weight=", w.weight,
                " holds an in-place image of a ", dtype_info(img.wei_dt), " ",
                img.K, "x", img.N, " weight (", image->bytes,
                " bytes); the raw weight of this request cannot be rebuilt "
                "from it (wei=",
                dtype_info(w.wei_dt), " K=", w.K, " N=", w.N, " ldb=", w.ldb,
                ", ", owned, " bytes)");
        return status_t::failure;
    }
    std::unique_ptr<void, native_aligned_free_t> raw(
            alloc_layout(extent * sizeof(uint16_t)));
    if (!raw) {
        log_error("native weight: no buffer to rebuild an in-place image (",
                extent * sizeof(uint16_t), " bytes)");
        return status_t::failure;
    }
    native_unpack_bf16(image->layout.weight, img.K, img.N, img.ldb, img.pack_nr,
            img.transB, img.interleave_split_halves, raw.get(), num_threads);
    return native_pack_weight(w.wei_dt, raw.get(), w.K, w.N, w.ldb, w.pack_nr,
            w.transB, w.interleave_split_halves, dst, num_threads);
}

/// Bytes writable at an aocl_blocked weight: the caller's capacity, else
/// the size of its AOCL prepack.
size_t aocl_capacity(const native_raw_weight_t &w, size_t capacity_bytes) {
    if (capacity_bytes != 0) return capacity_bytes;
    return aocl_prepack_bytes(
            w.K, w.N, w.ldb, w.transB, w.wei_dt == data_type_t::s8);
}

/// Reserves the record of an in-place image of used bytes at w's buffer of
/// owned bytes before the buffer is rewritten (a buffer is never rewritten
/// without one), after dropping the records it overlaps; end() when no
/// record can be allocated, and then the buffer must not be rewritten.
/// Under the mutex held exclusively.
image_map_t::iterator reserve_image_locked(
        const native_raw_weight_t &w, size_t used, size_t owned) {
    auto &images = cache_state().images;
    const auto at = reinterpret_cast<std::uintptr_t>(w.weight);
    for (auto it = images.lower_bound(at);
            it != images.end() && it->first - at < used;)
        it = images.erase(it);
    image_map_t::iterator it = images.end();
    try {
        it = images.try_emplace(at).first;
        it->second.layout = w;
        it->second.bytes = used;
        it->second.reserved = true;
        it->second.owned = owned;
    } catch (const std::bad_alloc &) { it = images.end(); }
    publish_spans_locked();
    return it;
}

/// Before an in-place rewrite of the used bytes at weight, in the caller's
/// buffer of owned bytes: marks the cached layouts packed from bytes the
/// rewrite overwrites that still match them.  A layout packed from bytes
/// that reach outside the caller's buffer is not marked.  Under the mutex
/// held exclusively.
void mark_rewritten_layouts_locked(
        const void *weight, size_t used, size_t owned) {
    const auto lo = reinterpret_cast<std::uintptr_t>(weight);
    for (auto &kv : cache_state().packs) {
        native_cached_pack_t &e = kv.second;
        const auto at = reinterpret_cast<std::uintptr_t>(kv.first.weight);
        if (at < lo + used && lo < at + e.source_bytes
                && within(at, e.source_bytes, lo, owned)
                && ends_hold(at, e.source_bytes, e.source_ends, lo, owned))
            e.rewritten_by = weight;
    }
}

/// A layout cached at k while an in-place rewrite of bytes it was packed
/// from waits for its pack's read to end: marked as the rewrite marked the
/// layouts cached before it reserved its record.  Under the mutex held
/// exclusively since the read ended, so the rewrite has not written.
void mark_published_layout_locked(
        const native_key_t &k, native_cached_pack_t &e) {
    const auto &images = cache_state().images;
    const auto at = reinterpret_cast<std::uintptr_t>(k.weight);
    // Records do not overlap: the one before at may hold it.
    auto it = images.upper_bound(at);
    if (it != images.begin()) --it;
    for (; it != images.end() && it->first < at + e.source_bytes; ++it) {
        const native_image_t &img = it->second;
        const std::uintptr_t lo = it->first;
        if (img.reserved && at < lo + img.bytes
                && within(at, e.source_bytes, lo, img.owned)
                && ends_hold(
                        at, e.source_bytes, e.source_ends, lo, img.owned)) {
            e.rewritten_by = img.layout.weight;
            return;
        }
    }
}

/// After the rewrite of weight (done: the buffer was written): the image
/// holds the same weight, so the layouts it marked stay and take the digests
/// of their bytes as they are now.  Clears its marks.  Under the mutex held
/// exclusively.
void refresh_rewritten_layouts_locked(const void *weight, bool done) {
    for (auto &kv : cache_state().packs) {
        native_cached_pack_t &e = kv.second;
        if (e.rewritten_by != weight) continue;
        if (done)
            e.source_ends = weight_ends_of(kv.first.weight, e.source_bytes);
        e.rewritten_by = nullptr;
    }
}

void log_pack(const native_raw_weight_t &w, int32_t wc, const char *where) {
    static const bool s_log = apilog_verbose_enabled();
    if (!s_log) return;
    // A layout of padded columns names its width.
    const std::string padded = w.N % w.pack_nr != 0
            ? " padded_n=" + std::to_string(native_padded_n(w.N, w.pack_nr))
            : std::string();
    apilog_verbose("[MATMUL.NATIVE_BRGEMM PACK] weight=", w.weight,
            " wei=", dtype_info(w.wei_dt), " K=", w.K, " N=", w.N,
            " ldb=", w.ldb, " transB=", w.transB ? 1 : 0,
            " interleave=", w.interleave_split_halves ? 1 : 0,
            " pack_nr=", w.pack_nr, " wc=", wc, " into=", where,
            w.aocl_blocked ? " from=aocl_blocked" : "", padded);
}

using exclusive_lock_t = std::unique_lock<std::shared_mutex>;

/// Holds a buffer in packing while a pack works on it outside the mutex.
struct packing_t {
    native_cache_state_t &state;
    const void *buffer = nullptr;

    /// Under the mutex held exclusively; false when it cannot be allocated.
    bool hold_locked(const void *b) {
        try {
            state.packing.insert(b);
        } catch (const std::bad_alloc &) { return false; }
        buffer = b;
        return true;
    }
    /// Under the mutex held exclusively: the buffer is free again.
    void release_locked() {
        if (buffer == nullptr) return;
        state.packing.erase(buffer);
        buffer = nullptr;
        state.packed.notify_all();
    }
    ~packing_t() {
        if (buffer == nullptr) return;
        exclusive_lock_t guard(state.mutex);
        release_locked();
    }
};

/// Holds bytes of a weight buffer in reads while a pack or another reader
/// reads them outside the mutex.
struct read_hold_t {
    native_cache_state_t &state;
    std::pair<std::uintptr_t, std::uintptr_t> span {0, 0};
    bool held = false;

    /// Under the mutex held exclusively; false when it cannot be allocated.
    bool hold_locked(const void *p, size_t bytes) {
        const auto at = reinterpret_cast<std::uintptr_t>(p);
        try {
            state.reads.emplace_back(at, at + bytes);
        } catch (const std::bad_alloc &) { return false; }
        span = state.reads.back();
        held = true;
        return true;
    }
    /// Under the mutex held exclusively: the bytes are no longer read.
    void release_locked() {
        if (!held) return;
        auto &reads = state.reads;
        const auto it = std::find(reads.begin(), reads.end(), span);
        *it = reads.back();
        reads.pop_back();
        held = false;
        if (state.rewrites_waiting != 0) state.packed.notify_all();
    }
    ~read_hold_t() {
        if (!held) return;
        exclusive_lock_t guard(state.mutex);
        release_locked();
    }
};

/// Under the mutex held exclusively: waits until no read holds bytes of the
/// bytes at weight, which the caller writes next.
void wait_for_reads_locked(
        exclusive_lock_t &lock, const void *weight, size_t bytes) {
    native_cache_state_t &state = cache_state();
    const auto lo = reinterpret_cast<std::uintptr_t>(weight);
    const auto overlapped = [&] {
        for (const auto &s : state.reads)
            if (s.first < lo + bytes && lo < s.second) return true;
        return false;
    };
    if (!overlapped()) return;
    ++state.rewrites_waiting;
    state.packed.wait(lock, [&] { return !overlapped(); });
    --state.rewrites_waiting;
}

/// What the checks of a thread (native_weight_check_t: its memo's, and
/// those of reads outside this cache) tell in-place rewrites: reading is odd
/// while one may read bytes of a weight buffer.  A line of its own: every
/// check of its thread stores to it.
struct alignas(64) memo_reader_t {
    std::atomic<uint64_t> reading {0};
    /// Its thread exited and a later thread may take it; under the mutex of
    /// memo_readers.
    bool free = false;
};

/// The readers of the threads that ran a check.  Never destroyed, and a
/// reader is never freed: a rewrite reads one without the mutex, also after
/// its thread exited, and a thread at its exit may come after the process
/// began to exit.
struct memo_readers_t {
    std::mutex mutex;
    std::vector<memo_reader_t *> list;
};

memo_readers_t &memo_readers() {
    static memo_readers_t *readers = new memo_readers_t();
    return *readers;
}

/// The reader of the calling thread, listed in memo_readers from its first
/// check (one an exited thread left, else a new one) and left there when it
/// exits; nullptr when it cannot be listed.
memo_reader_t *thread_reader() {
    struct owner_t {
        memo_reader_t *reader = nullptr;
        ~owner_t() {
            if (reader == nullptr) return;
            std::lock_guard<std::mutex> guard(memo_readers().mutex);
            reader->free = true;
        }
    };
    thread_local owner_t owner;
    if (owner.reader == nullptr) {
        memo_readers_t &readers = memo_readers();
        std::lock_guard<std::mutex> guard(readers.mutex);
        for (memo_reader_t *r : readers.list) {
            if (!r->free) continue;
            r->free = false;
            owner.reader = r;
            return r;
        }
        std::unique_ptr<memo_reader_t> r(new (std::nothrow) memo_reader_t());
        if (r == nullptr) return nullptr;
        try {
            readers.list.push_back(r.get());
        } catch (const std::bad_alloc &) { return nullptr; }
        owner.reader = r.release();
    }
    return owner.reader;
}

/// Begins a check of reader's thread, which a release store of reading
/// ends: true when it may read weight bytes, no in-place rewrite being
/// counted.
bool begin_check(memo_reader_t &reader) {
    const uint64_t n = reader.reading.load(std::memory_order_relaxed);
    // Before the load of rewrites, as a rewrite counts itself before it
    // loads reading: one of the two sees the other (wait_for_memo_checks).
    reader.reading.store(n + 1, std::memory_order_seq_cst);
    return cache_state().rewrites.load(std::memory_order_seq_cst) == 0;
}

/// Waits until every check that began before the calling rewrite was
/// counted in rewrites has ended; the checks that begin later see the count
/// and read no weight bytes.  Without the mutex, before the rewrite writes;
/// the mutex of memo_readers is held to read the list only, never while a
/// check is waited for: a thread's first memo entry lists its reader under
/// it while holding the cache lock.
void wait_for_memo_checks() {
    memo_readers_t &readers = memo_readers();
    for (size_t i = 0;; ++i) {
        const memo_reader_t *r = nullptr;
        {
            std::lock_guard<std::mutex> guard(readers.mutex);
            if (i == readers.list.size()) return;
            r = readers.list[i];
        }
        const uint64_t n = r->reading.load(std::memory_order_seq_cst);
        if (n % 2 == 0) continue;
        while (r->reading.load(std::memory_order_acquire) == n)
            std::this_thread::yield();
    }
}

/// Rewrites w's buffer in place into the used bytes of its layout, from its
/// AOCL prepack (aocl) or from the raw bf16 or s4 weight.  Entered and left
/// with the mutex held exclusively; the pack and the write-back run outside
/// it, with the buffer in packing and its image record reserved, so calls on
/// the buffer wait and the other calls go on, and once no read of other
/// calls holds bytes it writes and no check reads weight bytes.
/// unimplemented when nothing was written.
status_t rewrite_in_place(exclusive_lock_t &lock, const native_raw_weight_t &w,
        size_t used, size_t owned, size_t bytes, bool aocl, int num_threads) {
    native_cache_state_t &state = cache_state();
    packing_t packing {state};
    if (!packing.hold_locked(w.weight)) return status_t::unimplemented;
    if (reserve_image_locked(w, used, owned) == state.images.end()) {
        log_error("native weight: no record for an in-place image");
        packing.release_locked();
        return aocl ? status_t::failure : status_t::unimplemented;
    }
    state.rewrites.fetch_add(1, std::memory_order_seq_cst);
    mark_rewritten_layouts_locked(w.weight, used, owned);
    wait_for_reads_locked(lock, w.weight, used);
    lock.unlock();
    wait_for_memo_checks();
    status_t st = status_t::success;
    if (aocl) {
        // Weight cache mode 2 is the caller's consent that the weight buffer
        // is writable storage read by this backend only; the AOCL inverse is
        // read out of it before the layout is written.
        st = pack_from_aocl(w, const_cast<void *>(w.weight), num_threads);
    } else {
        std::unique_ptr<void, native_aligned_free_t> tmp(alloc_layout(bytes));
        st = tmp ? native_pack_weight(w.wei_dt, w.weight, w.K, w.N, w.ldb,
                           w.pack_nr, w.transB, w.interleave_split_halves,
                           tmp.get(), num_threads)
                 : status_t::unimplemented;
        // Weight cache mode 2 is the caller's consent that the weight buffer
        // is writable storage read by this backend only.
        if (st == status_t::success)
            std::memcpy(const_cast<void *>(w.weight), tmp.get(), used);
        else
            st = status_t::unimplemented;
    }
    lock.lock();
    const auto it
            = state.images.find(reinterpret_cast<std::uintptr_t>(w.weight));
    refresh_rewritten_layouts_locked(w.weight, st == status_t::success);
    if (it != state.images.end()) {
        if (st == status_t::success) {
            it->second.ends = weight_ends_of(w.weight, used);
            it->second.reserved = false;
        } else {
            erase_image_locked(it);
        }
    }
    state.rewrites.fetch_sub(1, std::memory_order_release);
    packing.release_locked();
    return st;
}

// ── The per-thread layout memo ─────────────────────────────────────────

/// Sets and ways of a thread's memo: at most kMemoSets * kMemoWays layouts.
constexpr size_t kMemoSets = 128;
constexpr size_t kMemoWays = 4;
static_assert((kMemoSets & (kMemoSets - 1)) == 0,
        "the set index is a mask of the hash");

/// A layout a call of the thread found, and the digests its later calls on
/// the key check before they are served it.
struct memo_entry_t {
    native_key_t key {};
    native_layout_t layout; ///< never owned
    /// The bytes at key.weight the digests cover: what the cached layout was
    /// packed from, or the image.
    size_t check_bytes = 0;
    weight_ends_t ends;
    /// The cache generation it was found in; 0 for an empty way.
    uint64_t generation = 0;
    /// layout is the buffer itself, rewritten in place.
    bool image = false;
};

struct memo_set_t {
    /// key.weight of each way, compared first.
    const void *weights[kMemoWays] = {};
    memo_entry_t ways[kMemoWays];
    /// The way the next insert takes when every way holds a current entry.
    uint32_t next = 0;
};

struct layout_memo_t {
    /// The thread's reader (thread_reader), for the checks of the memo.
    memo_reader_t *reader = nullptr;
    memo_set_t sets[kMemoSets];
};

/// The memo of the calling thread, made when make is set (an entry is
/// stored) and freed when the thread exits; nullptr before that, or when it
/// cannot be allocated or its reader listed.
layout_memo_t *thread_memo(bool make) {
    thread_local std::unique_ptr<layout_memo_t> memo;
    if (!memo && make) {
        memo_reader_t *reader = thread_reader();
        if (reader == nullptr) return nullptr;
        memo.reset(new (std::nothrow) layout_memo_t());
        if (memo) memo->reader = reader;
    }
    return memo.get();
}

/// The set of a memo that holds the layouts of weight.
size_t memo_set_index(const void *weight) {
    uint64_t h
            = static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(weight));
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return static_cast<size_t>(h & (kMemoSets - 1));
}

/// The set of the thread's memo that holds the layouts of weight; nullptr
/// while the thread has no memo, unless make makes it.
memo_set_t *memo_set(const void *weight, bool make) {
    layout_memo_t *memo = thread_memo(make);
    if (memo == nullptr) return nullptr;
    return &memo->sets[memo_set_index(weight)];
}

/// The entry of memo for key, of any generation, else nullptr: a key has
/// one way in its set (memo_put_locked).
const memo_entry_t *memo_find(
        const layout_memo_t &memo, const native_key_t &key) {
    const memo_set_t &set = memo.sets[memo_set_index(key.weight)];
    for (size_t i = 0; i < kMemoWays; ++i)
        if (set.weights[i] == key.weight && set.ways[i].key == key)
            return &set.ways[i];
    return nullptr;
}

/// A check of an entry of the calling thread's memo, on the reader the
/// memo holds (native_weight_check_t).
class memo_check_t {
public:
    explicit memo_check_t(memo_reader_t &reader)
        : reader_(reader), ok_(begin_check(reader)) {}
    ~memo_check_t() {
        reader_.reading.store(
                reader_.reading.load(std::memory_order_relaxed) + 1,
                std::memory_order_release);
    }
    memo_check_t(const memo_check_t &) = delete;
    memo_check_t &operator=(const memo_check_t &) = delete;

    bool ok() const { return ok_; }

private:
    memo_reader_t &reader_;
    bool ok_ = false;
};

/// Whether no cached layout or image record has gone, and no image changed,
/// since an entry of generation gen was made, read after its digests: a
/// rewrite in place bumps the generation before it writes the buffer, so an
/// unchanged generation means the digests read no byte it wrote.
bool memo_still_current(uint64_t gen) {
    std::atomic_thread_fence(std::memory_order_acquire);
    return cache_state().generation.load(std::memory_order_relaxed) == gen;
}

/// The entry of memo for an image that starts at weight, found in
/// generation gen, else nullptr.
const memo_entry_t *memo_find_image(
        const layout_memo_t &memo, const void *weight, uint64_t gen) {
    const memo_set_t &set = memo.sets[memo_set_index(weight)];
    for (size_t i = 0; i < kMemoWays; ++i)
        if (set.weights[i] == weight && set.ways[i].generation == gen
                && set.ways[i].image)
            return &set.ways[i];
    return nullptr;
}

/// Remembers for the thread that key is served layout while the check_bytes
/// at key.weight keep the digests ends, as found under the mutex in the
/// cache generation of that moment.
void memo_put_locked(const native_key_t &key, const native_layout_t &layout,
        size_t check_bytes, const weight_ends_t &ends, bool image) {
    memo_set_t *set = memo_set(key.weight, true);
    if (set == nullptr) return;
    const uint64_t gen
            = cache_state().generation.load(std::memory_order_relaxed);
    size_t way = kMemoWays;
    for (size_t i = 0; i < kMemoWays && way == kMemoWays; ++i)
        if (set->weights[i] == key.weight && set->ways[i].key == key) way = i;
    for (size_t i = 0; i < kMemoWays && way == kMemoWays; ++i)
        if (set->ways[i].generation != gen) way = i;
    if (way == kMemoWays) {
        way = set->next;
        set->next = static_cast<uint32_t>((set->next + 1) % kMemoWays);
    }
    memo_entry_t &e = set->ways[way];
    e.key = key;
    e.layout = layout;
    e.layout.owned = nullptr;
    e.check_bytes = check_bytes;
    e.ends = ends;
    e.generation = gen;
    e.image = image;
    set->weights[way] = key.weight;
}

native_key_t key_of(const native_raw_weight_t &w) {
    return {w.weight, w.K, w.N, w.ldb, w.pack_nr, w.transB,
            w.interleave_split_halves, w.wei_dt, w.aocl_blocked};
}

native_raw_weight_t raw_of(const native_key_t &k) {
    native_raw_weight_t w;
    w.wei_dt = k.wei_dt;
    w.weight = k.weight;
    w.K = k.K;
    w.N = k.N;
    w.ldb = k.ldb;
    w.transB = k.transB;
    w.pack_nr = k.pack_nr;
    w.interleave_split_halves = k.interleave;
    w.aocl_blocked = k.aocl_blocked;
    return w;
}

/// Remembers the cached layout of key for the thread, unless an in-place
/// rewrite of its bytes runs.  Under the mutex.
void memo_pack_locked(const native_key_t &key, const native_cached_pack_t &e) {
    if (e.rewritten_by != nullptr) return;
    native_layout_t layout;
    layout.data = e.buf.get();
    memo_put_locked(key, layout, e.source_bytes, e.source_ends, false);
}

/// Remembers for the thread that key is served from the image img at its
/// start.  Under the mutex.
void memo_image_locked(const native_key_t &key, const native_image_t &img) {
    if (img.reserved) return;
    native_layout_t layout;
    layout.data = key.weight;
    memo_put_locked(key, layout, img.bytes, img.ends, true);
}

/// A request of get_native_layout and what its stages share.
struct layout_request_t {
    const native_raw_weight_t &w;
    int32_t weight_cache_mode;
    size_t capacity_bytes;
    int num_threads;
    native_key_t key;
    size_t bytes; ///< native_packed_bytes of the layout
    /// The bytes the request reads at w.weight (the raw span, or the AOCL
    /// prepack).
    size_t extent;
    /// The caller's buffer: extent, or the capacity it states when larger.
    /// The checks read nothing outside it.
    size_t owned;
};

layout_request_t make_request(const native_raw_weight_t &w,
        int32_t weight_cache_mode, size_t capacity_bytes, int num_threads) {
    const size_t extent = w.aocl_blocked
            ? aocl_prepack_bytes(
                      w.K, w.N, w.ldb, w.transB, w.wei_dt == data_type_t::s8)
            : native_raw_weight_bytes(w.wei_dt, w.K, w.N, w.ldb, w.transB);
    return {w, weight_cache_mode, capacity_bytes, num_threads, key_of(w),
            native_packed_bytes(w.wei_dt, w.K, w.N, w.pack_nr), extent,
            std::max(extent, capacity_bytes)};
}

void own_layout(std::unique_ptr<void, native_aligned_free_t> buf,
        native_layout_t &out) {
    out.owned = buf.release();
    out.data = out.owned;
}

/// Packs r into a buffer the caller owns, from image when its buffer holds
/// one.
status_t pack_for_call(const layout_request_t &r, const native_image_t *image,
        native_layout_t &out) {
    std::unique_ptr<void, native_aligned_free_t> buf(alloc_layout(r.bytes));
    if (!buf) {
        // Nothing was written: another route can still serve a raw weight.
        log_error("native weight: no buffer for a ", r.bytes, "-byte layout");
        return status_t::unimplemented;
    }
    const status_t st
            = pack_from_source(r.w, image, r.owned, buf.get(), r.num_threads);
    if (st != status_t::success) return st;
    log_pack(r.w, r.weight_cache_mode, "call");
    own_layout(std::move(buf), out);
    return status_t::success;
}

/// The layout the thread's memo holds for r, when its bytes still keep the
/// digests: what the cache lookup would find, without the lock or the
/// lookup.  A cached layout does not serve weight cache mode 0.  While an
/// in-place rewrite runs, the bytes are not read and the call takes the
/// lookup.
bool find_layout_memo(const layout_request_t &r, native_layout_t &out) {
    layout_memo_t *memo = thread_memo(false);
    if (memo == nullptr) return false;
    const memo_entry_t *e = memo_find(*memo, r.key);
    if (e == nullptr || (!e->image && r.weight_cache_mode == 0)) return false;
    const memo_check_t check(*memo->reader);
    if (!check.ok()) return false;
    const auto at = reinterpret_cast<std::uintptr_t>(r.w.weight);
    if (!ends_hold(at, e->check_bytes, e->ends, at, r.owned)
            || !memo_still_current(e->generation))
        return false;
    out = e->layout;
    return true;
}

/// Whether a reserved image record overlaps [at, at + bytes): an in-place
/// rewrite may be writing there.  Under the mutex.
bool rewrite_overlaps_locked(std::uintptr_t at, size_t bytes) {
    const auto &images = cache_state().images;
    // Records do not overlap: the one before at may reach past it.
    auto it = images.upper_bound(at);
    if (it != images.begin()) --it;
    for (; it != images.end() && it->first < at + bytes; ++it)
        if (it->second.reserved && at < it->first + it->second.bytes)
            return true;
    return false;
}

/// Whether the check of r's cached layout (pack_holds) would read bytes an
/// in-place rewrite may be writing: the rewrite did not mark the layout,
/// and the bytes its check reads overlap the rewrite's record.  The lookup
/// then waits for the rewrite to end.  Under the mutex.
bool pack_check_waits_locked(const layout_request_t &r) {
    native_cache_state_t &state = cache_state();
    if (r.weight_cache_mode == 0
            || state.rewrites.load(std::memory_order_relaxed) == 0)
        return false;
    const auto hit = state.packs.find(r.key);
    if (hit == state.packs.end() || hit->second.rewritten_by != nullptr)
        return false;
    return rewrite_overlaps_locked(reinterpret_cast<std::uintptr_t>(r.w.weight),
            std::min(hit->second.source_bytes, r.owned));
}

/// The cached layout (modes 1 / 2) or the in-place image (may_image) of r
/// that still holds, under the mutex held shared once the buffer is not in
/// packing; false when neither does.
bool find_layout_shared(
        const layout_request_t &r, bool may_image, native_layout_t &out) {
    native_cache_state_t &state = cache_state();
    std::shared_lock<std::shared_mutex> lock(state.mutex);
    state.packed.wait(lock, [&] {
        return state.packing.count(r.w.weight) == 0
                && !pack_check_waits_locked(r);
    });
    if (r.weight_cache_mode != 0) {
        const auto hit = state.packs.find(r.key);
        if (hit != state.packs.end()
                && pack_holds(hit->first, hit->second, r.owned)) {
            out.data = hit->second.buf.get();
            memo_pack_locked(r.key, hit->second);
            return true;
        }
    }
    if (may_image) {
        bool interior = false, stale = false;
        const auto it = image_at_locked(r.w.weight, r.owned, interior, stale);
        if (it != state.images.end() && !interior && !stale
                && same_layout(it->second.layout, r.w)) {
            out.data = r.w.weight;
            memo_image_locked(r.key, it->second);
            return true;
        }
    }
    return false;
}

enum class lookup_t { found, missing, failed };

/// The lookup again under the mutex held exclusively: the cached layout that
/// still holds (one that does not is dropped), else the in-place image, which
/// serves r when it holds r's layout and is returned in image otherwise.
/// failed when r starts inside an image or reads into one.
lookup_t find_layout_exclusive(const layout_request_t &r, native_image_t &image,
        bool &has_image, native_layout_t &out) {
    native_cache_state_t &state = cache_state();
    if (r.weight_cache_mode != 0) {
        const auto hit = state.packs.find(r.key);
        if (hit != state.packs.end()) {
            if (pack_holds(hit->first, hit->second, r.owned)) {
                out.data = hit->second.buf.get();
                memo_pack_locked(r.key, hit->second);
                return lookup_t::found;
            }
            apilog_info("native weight: the buffer at ", r.w.weight,
                    " no longer holds the weight its cached layout was "
                    "packed from; packed again from its bytes");
            try {
                state.retired.push_back(std::move(hit->second.buf));
            } catch (const std::bad_alloc &) {
                log_error(
                        "native weight: no memory to keep a replaced "
                        "layout");
                return lookup_t::failed;
            }
            bump_generation_locked();
            state.packs.erase(hit);
        }
    }
    has_image = false;
    if (state.image_count.load(std::memory_order_relaxed) != 0) {
        bool interior = false, after = false;
        const native_image_t *found = live_image_locked(
                r.w.weight, r.extent, r.owned, interior, after);
        if (interior) {
            log_error("native weight: weight=", r.w.weight,
                    " points inside an in-place image; the raw weight "
                    "cannot be rebuilt from part of it");
            return lookup_t::failed;
        }
        if (after) {
            log_error("native weight: weight=", r.w.weight,
                    " reads into a buffer an in-place pack rewrote; the raw "
                    "weight of that part is gone");
            return lookup_t::failed;
        }
        if (found != nullptr) {
            image = *found;
            has_image = true;
        }
    }
    if (has_image && same_layout(image.layout, r.w)) {
        out.data = r.w.weight;
        memo_image_locked(r.key, image);
        return lookup_t::found;
    }
    return lookup_t::missing;
}

/// Weight cache mode 2 on a buffer with no image: rewrites it in place when
/// its layout fits it (an AOCL prepack, or a contiguous bf16 or s4 weight).
/// done is set when the buffer is rewritten; a failed AOCL rewrite returns
/// its status, a failed bf16 or s4 one leaves r to the out-of-place pack.  A
/// layout of padded columns (N not a multiple of pack_nr) is never written in
/// place.  With the mutex held exclusively (rewrite_in_place).
status_t rewrite_if_fits(exclusive_lock_t &lock, const layout_request_t &r,
        bool &done, native_layout_t &out) {
    const native_raw_weight_t &w = r.w;
    done = false;
    if (w.N % w.pack_nr != 0) return status_t::success;
    const size_t used = native_block_bytes(w.wei_dt, w.K, w.pack_nr)
            * static_cast<size_t>(w.N / w.pack_nr);
    const bool contiguous = w.ldb == (w.transB ? w.K : w.N);
    if (w.aocl_blocked) {
        if (!contiguous || used > aocl_capacity(w, r.capacity_bytes))
            return status_t::success;
        const status_t st = rewrite_in_place(
                lock, w, used, r.owned, r.bytes, true, r.num_threads);
        if (st != status_t::success) return st;
    } else {
        if ((w.wei_dt != data_type_t::bf16 && w.wei_dt != data_type_t::s4)
                || !contiguous)
            return status_t::success;
        const size_t plain
                = native_raw_weight_bytes(w.wei_dt, w.K, w.N, w.ldb, w.transB);
        if (!wei_inplace_fits(used, used, plain, r.capacity_bytes))
            return status_t::success;
        if (rewrite_in_place(
                    lock, w, used, r.owned, r.bytes, false, r.num_threads)
                != status_t::success)
            return status_t::success;
    }
    log_pack(w, r.weight_cache_mode, "in_place");
    out.data = w.weight;
    done = true;
    native_cache_state_t &state = cache_state();
    const auto it
            = state.images.find(reinterpret_cast<std::uintptr_t>(w.weight));
    if (it != state.images.end()) memo_image_locked(r.key, it->second);
    return status_t::success;
}

/// Packs r out of place, from image when its buffer holds one, outside the
/// mutex, which the call holds exclusively on entry, with the bytes it reads
/// held in reads.  Modes 1 / 2 also hold the buffer in packing until the
/// layout is cached (or, past the cache capacity, owned by the caller); mode
/// 0 packs for this call only.
status_t pack_out_of_place(exclusive_lock_t &lock, const layout_request_t &r,
        const native_image_t *image, native_layout_t &out) {
    native_cache_state_t &state = cache_state();
    const native_raw_weight_t &w = r.w;
    const size_t source_bytes = image != nullptr ? image->bytes : r.extent;
    read_hold_t read {state};
    if (!read.hold_locked(w.weight, source_bytes)) {
        log_error("native weight: no memory to pack weight=", w.weight);
        return status_t::failure;
    }
    if (r.weight_cache_mode == 0) {
        lock.unlock();
        return pack_for_call(r, image, out);
    }
    packing_t packing {state};
    if (!packing.hold_locked(w.weight)) {
        read.release_locked();
        log_error("native weight: no memory to pack weight=", w.weight);
        return status_t::failure;
    }
    lock.unlock();
    std::unique_ptr<void, native_aligned_free_t> buf(alloc_layout(r.bytes));
    if (!buf) {
        // Nothing was written: another route can still serve a raw weight.
        log_error("native weight: no buffer for a ", r.bytes, "-byte layout");
        return status_t::unimplemented;
    }
    const weight_ends_t source_ends = image != nullptr
            ? image->ends
            : weight_ends_of(w.weight, r.extent);
    const status_t st
            = pack_from_source(w, image, r.owned, buf.get(), r.num_threads);
    if (st != status_t::success) return st;
    lock.lock();
    read.release_locked();
    if (state.packs.size() + state.retired.size()
            < get_weight_cache_capacity()) {
        try {
            auto slot = state.packs.try_emplace(r.key);
            native_cached_pack_t &entry = slot.first->second;
            if (slot.second) {
                entry.buf.reset(static_cast<uint8_t *>(buf.release()));
                entry.source_bytes = source_bytes;
                entry.source_ends = source_ends;
                mark_published_layout_locked(r.key, entry);
            }
            packing.release_locked();
            out.data = entry.buf.get();
            memo_pack_locked(r.key, entry);
            log_pack(w, r.weight_cache_mode, "cache");
            return status_t::success;
        } catch (const std::bad_alloc &) {
            log_error("native weight: no memory for a cache entry");
            packing.release_locked();
            return status_t::failure;
        }
    }
    packing.release_locked();
    lock.unlock();
    log_pack(w, r.weight_cache_mode, "call");
    own_layout(std::move(buf), out);
    return status_t::success;
}

} // namespace

size_t native_raw_weight_bytes(
        data_type_t wei_dt, int K, int N, int ldb, bool transB) {
    const size_t rows = static_cast<size_t>(transB ? N : K);
    const size_t cols = static_cast<size_t>(transB ? K : N);
    const size_t elements = (rows - 1) * static_cast<size_t>(ldb) + cols;
    // Two s4 elements per byte, the first in the low nibble of byte 0.
    if (wei_dt == data_type_t::s4) return (elements + 1) / 2;
    return elements * zendnnl::common::size_of(wei_dt);
}

status_t get_native_layout(const native_raw_weight_t &w,
        int32_t weight_cache_mode, size_t capacity_bytes, int num_threads,
        native_layout_t &out, const native_image_check_t *checked) {
    out = native_layout_t {};
    if (w.weight == nullptr || w.ldb < (w.transB ? w.K : w.N)
            || weight_cache_mode < 0 || weight_cache_mode > 2) {
        log_error("native weight: no layout for wei=", dtype_info(w.wei_dt),
                " K=", w.K, " N=", w.N, " ldb=", w.ldb, " pack_nr=", w.pack_nr,
                " weight=", w.weight, " weight_cache_mode=", weight_cache_mode);
        return status_t::failure;
    }
    const layout_request_t r
            = make_request(w, weight_cache_mode, capacity_bytes, num_threads);
    // The call's own check found the image of this layout at the buffer,
    // over every byte this lookup would check.
    if (checked != nullptr && checked->ready
            && checked->layout.weight == w.weight
            && same_layout(checked->layout, w)
            && checked->checked_bytes >= r.owned) {
        out.data = w.weight;
        return status_t::success;
    }
    if (find_layout_memo(r, out)) return status_t::success;
    const bool may_image = image_may_overlap(w.weight, r.extent);
    // Mode 0 with no image at the buffer has nothing to find but goes on to
    // the exclusive lookup too, which holds the bytes its pack reads against
    // an in-place rewrite.
    if ((weight_cache_mode != 0 || may_image)
            && find_layout_shared(r, may_image, out))
        return status_t::success;

    native_cache_state_t &state = cache_state();
    exclusive_lock_t lock(state.mutex);
    state.packed.wait(lock, [&] {
        return state.packing.count(w.weight) == 0
                && !pack_check_waits_locked(r);
    });
    native_image_t image;
    bool has_image = false;
    switch (find_layout_exclusive(r, image, has_image, out)) {
        case lookup_t::found: return status_t::success;
        case lookup_t::failed: return status_t::failure;
        case lookup_t::missing: break;
    }
    if (weight_cache_mode == 2 && !has_image) {
        bool done = false;
        const status_t st = rewrite_if_fits(lock, r, done, out);
        if (done || st != status_t::success) return st;
    }
    return pack_out_of_place(lock, r, has_image ? &image : nullptr, out);
}

void native_free_layout(void *owned) {
    if (owned != nullptr) zendnnl_aligned_free(owned);
}

namespace {

/// What the image at it, which holds for the caller, tells a call on weight,
/// remembered for the thread when it starts there.  Under the mutex.
void note_image_locked(image_map_t::const_iterator it, bool interior,
        native_image_check_t &found) {
    if (interior || it->second.reserved) return;
    found.ready = true;
    found.layout = it->second.layout;
    memo_image_locked(key_of(it->second.layout), it->second);
}

} // namespace

bool native_weight_find_image(
        const void *weight, size_t owned_bytes, native_image_check_t &found) {
    found = native_image_check_t {};
    found.checked_bytes = owned_bytes;
    if (weight == nullptr || !image_may_overlap(weight, 1)) return false;
    native_cache_state_t &state = cache_state();
    const auto at = reinterpret_cast<std::uintptr_t>(weight);
    if (layout_memo_t *memo = thread_memo(false)) {
        const memo_entry_t *e = memo_find_image(*memo, weight,
                state.generation.load(std::memory_order_acquire));
        if (e != nullptr) {
            const memo_check_t check(*memo->reader);
            if (check.ok()
                    && ends_hold(at, e->check_bytes, e->ends, at, owned_bytes)
                    && memo_still_current(e->generation)) {
                found.ready = true;
                found.layout = raw_of(e->key);
                return true;
            }
        }
    }
    {
        std::shared_lock<std::shared_mutex> lock(state.mutex);
        bool interior = false, stale = false;
        const auto it = image_at_locked(weight, owned_bytes, interior, stale);
        if (it == state.images.end()) return false;
        if (!stale) {
            note_image_locked(it, interior, found);
            return true;
        }
    }
    exclusive_lock_t lock(state.mutex);
    bool interior = false, stale = false;
    const auto it = image_at_locked(weight, owned_bytes, interior, stale);
    if (it == state.images.end()) return false;
    if (!stale) {
        note_image_locked(it, interior, found);
        return true;
    }
    log_stale_image(it->second);
    erase_image_locked(it);
    return false;
}

bool native_weight_packed_in_place(const void *weight, size_t owned_bytes) {
    native_image_check_t found;
    return native_weight_find_image(weight, owned_bytes, found);
}

native_raw_read_t native_weight_read_raw(const void *weight, size_t extent,
        size_t owned_bytes, const std::function<void()> &read) {
    native_cache_state_t &state = cache_state();
    exclusive_lock_t lock(state.mutex);
    state.packed.wait(lock, [&] { return state.packing.count(weight) == 0; });
    if (state.image_count.load(std::memory_order_relaxed) != 0) {
        bool interior = false, after = false;
        if (live_image_locked(weight, extent, std::max(extent, owned_bytes),
                    interior, after)
                        != nullptr
                || after)
            return native_raw_read_t::image;
    }
    read_hold_t hold {state};
    if (!hold.hold_locked(weight, extent)) return native_raw_read_t::failed;
    lock.unlock();
    read();
    return native_raw_read_t::done;
}

native_weight_check_t::native_weight_check_t(const void *weight, size_t bytes) {
    memo_reader_t *reader = thread_reader();
    if (reader == nullptr) return;
    // The copy of the spans is brought up to date before the check begins:
    // once begun, a check takes no lock, as rewrites wait for it.
    if (bytes != 0) current_thread_spans();
    reading_ = &reader->reading;
    ok_ = begin_check(*reader)
            && (bytes == 0 || !image_may_overlap_unlocked(weight, bytes));
}

void clear_native_weight_caches() {
    native_cache_state_t &state = cache_state();
    exclusive_lock_t lock(state.mutex);
    bump_generation_locked();
    state.packs.clear();
    state.retired.clear();
    state.images.clear();
    publish_spans_locked();
}

size_t native_weight_layouts(const void *weight) {
    if (weight == nullptr) return 0;
    native_cache_state_t &state = cache_state();
    std::shared_lock<std::shared_mutex> lock(state.mutex);
    size_t n = 0;
    for (const auto &kv : state.packs)
        if (kv.first.weight == weight) ++n;
    return n + state.images.count(reinterpret_cast<std::uintptr_t>(weight));
}

size_t native_weight_memo_entries(const void *weight) {
    if (weight == nullptr) return 0;
    const memo_set_t *set = memo_set(weight, false);
    if (set == nullptr) return 0;
    const uint64_t gen
            = cache_state().generation.load(std::memory_order_acquire);
    size_t n = 0;
    for (size_t i = 0; i < kMemoWays; ++i)
        if (set->weights[i] == weight && set->ways[i].generation == gen) ++n;
    return n;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
