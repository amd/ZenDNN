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
#include "lowoha_operators/matmul/matmul_native/common/native_aocl_unreorder.hpp"
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
    pack_map_t packs;
    /// Keyed by buffer start; a pointer inside an image resolves to it.
    image_map_t images;
    std::atomic<size_t> image_count {0};
    /// The spans of images, for a check without the mutex; null when they
    /// could not be copied, and every check then takes the mutex.
    std::shared_ptr<const image_spans_t> spans;
    /// Bumped after spans changes.
    std::atomic<uint64_t> spans_gen {0};
};

native_cache_state_t &cache_state() {
    static native_cache_state_t state;
    return state;
}

void *alloc_layout(size_t bytes) {
    return zendnnl_aligned_alloc(64, (bytes + 63) & ~size_t {63});
}

/// Copies the spans of the images for the check without the mutex, under
/// the mutex held exclusively.
void publish_spans_locked() {
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

/// False only when no image overlaps [p, p + bytes); takes no lock while
/// the images have not changed since the calling thread last looked.
bool image_may_overlap(const void *p, size_t bytes) {
    native_cache_state_t &state = cache_state();
    if (state.image_count.load(std::memory_order_acquire) == 0) return false;
    thread_local uint64_t tl_gen = 0;
    thread_local std::shared_ptr<const image_spans_t> tl_spans;
    if (state.spans_gen.load(std::memory_order_acquire) != tl_gen) {
        std::shared_lock<std::shared_mutex> lock(state.mutex);
        tl_spans = state.spans;
        tl_gen = state.spans_gen.load(std::memory_order_relaxed);
    }
    if (!tl_spans) return true;
    // The first span that ends past p: it holds p or starts after it.
    const auto at = reinterpret_cast<std::uintptr_t>(p);
    const auto it = std::upper_bound(tl_spans->begin(), tl_spans->end(), at,
            [](std::uintptr_t a,
                    const std::pair<std::uintptr_t, std::uintptr_t> &s) {
        return a < s.second;
    });
    return it != tl_spans->end() && (it->first <= at || it->first - at < bytes);
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

/// Reserves the record of an in-place image of used bytes at w's buffer
/// before the buffer is rewritten (a buffer is never rewritten without one),
/// after dropping the records it overlaps; end() when no record can be
/// allocated, and then the buffer must not be rewritten.  Under the mutex
/// held exclusively.
image_map_t::iterator reserve_image_locked(
        const native_raw_weight_t &w, size_t used) {
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
    if (s_log)
        apilog_verbose("[MATMUL.NATIVE_BRGEMM PACK] weight=", w.weight,
                " wei=", dtype_info(w.wei_dt), " K=", w.K, " N=", w.N,
                " ldb=", w.ldb, " transB=", w.transB ? 1 : 0,
                " interleave=", w.interleave_split_halves ? 1 : 0,
                " pack_nr=", w.pack_nr, " wc=", wc, " into=", where,
                w.aocl_blocked ? " from=aocl_blocked" : "");
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

/// Rewrites w's buffer in place into the used bytes of its layout, from its
/// AOCL prepack (aocl) or from the raw bf16 weight.  Entered and left with
/// the mutex held exclusively; the pack and the write-back run outside it,
/// with the buffer in packing and its image record reserved, so calls on the
/// buffer wait and the other calls go on.  unimplemented when nothing was
/// written.
status_t rewrite_in_place(exclusive_lock_t &lock, const native_raw_weight_t &w,
        size_t used, size_t owned, size_t bytes, bool aocl, int num_threads) {
    native_cache_state_t &state = cache_state();
    packing_t packing {state};
    if (!packing.hold_locked(w.weight)) return status_t::unimplemented;
    if (reserve_image_locked(w, used) == state.images.end()) {
        log_error("native weight: no record for an in-place image");
        packing.release_locked();
        return aocl ? status_t::failure : status_t::unimplemented;
    }
    mark_rewritten_layouts_locked(w.weight, used, owned);
    lock.unlock();
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
    packing.release_locked();
    return st;
}

} // namespace

size_t native_raw_weight_bytes(
        data_type_t wei_dt, int K, int N, int ldb, bool transB) {
    const size_t rows = static_cast<size_t>(transB ? N : K);
    const size_t cols = static_cast<size_t>(transB ? K : N);
    return ((rows - 1) * static_cast<size_t>(ldb) + cols)
            * zendnnl::common::size_of(wei_dt);
}

status_t get_native_layout(const native_raw_weight_t &w,
        int32_t weight_cache_mode, size_t capacity_bytes, int num_threads,
        native_layout_t &out) {
    out = native_layout_t {};
    const size_t bytes = native_packed_bytes(w.wei_dt, w.K, w.N, w.pack_nr);
    if (bytes == 0 || w.weight == nullptr || w.ldb < (w.transB ? w.K : w.N)
            || weight_cache_mode < 0 || weight_cache_mode > 2) {
        log_error("native weight: no layout for wei=", dtype_info(w.wei_dt),
                " K=", w.K, " N=", w.N, " ldb=", w.ldb, " pack_nr=", w.pack_nr,
                " weight=", w.weight, " weight_cache_mode=", weight_cache_mode);
        return status_t::failure;
    }
    native_cache_state_t &state = cache_state();
    const native_key_t key {w.weight, w.K, w.N, w.ldb, w.pack_nr, w.transB,
            w.interleave_split_halves, w.wei_dt, w.aocl_blocked};
    // The caller's buffer: the bytes the request reads, or the capacity it
    // states.  The checks below read nothing outside it.
    const size_t extent = w.aocl_blocked
            ? aocl_prepack_bytes(
                      w.K, w.N, w.ldb, w.transB, w.wei_dt == data_type_t::s8)
            : native_raw_weight_bytes(w.wei_dt, w.K, w.N, w.ldb, w.transB);
    const size_t owned = std::max(extent, capacity_bytes);
    const bool may_image = image_may_overlap(w.weight, extent);

    // Hits, under the mutex held shared.
    if (weight_cache_mode != 0 || may_image) {
        std::shared_lock<std::shared_mutex> lock(state.mutex);
        state.packed.wait(
                lock, [&] { return state.packing.count(w.weight) == 0; });
        if (weight_cache_mode != 0) {
            const auto hit = state.packs.find(key);
            if (hit != state.packs.end()
                    && pack_holds(hit->first, hit->second, owned)) {
                out.data = hit->second.buf.get();
                return status_t::success;
            }
        }
        if (may_image) {
            bool interior = false, stale = false;
            const auto it = image_at_locked(w.weight, owned, interior, stale);
            if (it != state.images.end() && !interior && !stale
                    && same_layout(it->second.layout, w)) {
                out.data = w.weight;
                return status_t::success;
            }
        }
    } else {
        // Mode 0 with no image at the buffer: packed for this call only.
        std::unique_ptr<void, native_aligned_free_t> buf(alloc_layout(bytes));
        if (!buf) {
            log_error("native weight: no buffer for a ", bytes, "-byte layout");
            return status_t::unimplemented;
        }
        const status_t st
                = pack_from_source(w, nullptr, owned, buf.get(), num_threads);
        if (st != status_t::success) return st;
        log_pack(w, weight_cache_mode, "call");
        out.owned = buf.release();
        out.data = out.owned;
        return status_t::success;
    }

    exclusive_lock_t lock(state.mutex);
    state.packed.wait(lock, [&] { return state.packing.count(w.weight) == 0; });
    if (weight_cache_mode != 0) {
        const auto hit = state.packs.find(key);
        if (hit != state.packs.end()) {
            if (pack_holds(hit->first, hit->second, owned)) {
                out.data = hit->second.buf.get();
                return status_t::success;
            }
            apilog_info("native weight: the buffer at ", w.weight,
                    " no longer holds the weight its cached layout was "
                    "packed from; packed again from its bytes");
            state.packs.erase(hit);
        }
    }
    native_image_t image;
    bool has_image = false;
    if (state.image_count.load(std::memory_order_relaxed) != 0) {
        bool interior = false, after = false;
        const native_image_t *found
                = live_image_locked(w.weight, extent, owned, interior, after);
        if (interior) {
            log_error("native weight: weight=", w.weight,
                    " points inside an in-place image; the raw weight "
                    "cannot be rebuilt from part of it");
            return status_t::failure;
        }
        if (after) {
            log_error("native weight: weight=", w.weight,
                    " reads into a buffer an in-place pack rewrote; the raw "
                    "weight of that part is gone");
            return status_t::failure;
        }
        if (found != nullptr) {
            image = *found;
            has_image = true;
        }
    }
    if (has_image && same_layout(image.layout, w)) {
        out.data = w.weight;
        return status_t::success;
    }

    const size_t used = native_block_bytes(w.wei_dt, w.K, w.pack_nr)
            * static_cast<size_t>(w.N / w.pack_nr);
    const size_t plain = static_cast<size_t>(w.K) * w.N
            * zendnnl::common::size_of(w.wei_dt);
    const bool contiguous = w.ldb == (w.transB ? w.K : w.N);
    if (weight_cache_mode == 2 && !has_image && w.aocl_blocked && contiguous
            && used <= aocl_capacity(w, capacity_bytes)) {
        const status_t st = rewrite_in_place(
                lock, w, used, owned, bytes, true, num_threads);
        if (st != status_t::success) return st;
        log_pack(w, weight_cache_mode, "in_place");
        out.data = w.weight;
        return status_t::success;
    }
    if (weight_cache_mode == 2 && !has_image && !w.aocl_blocked
            && w.wei_dt == data_type_t::bf16 && contiguous
            && wei_inplace_fits(used, used, plain, capacity_bytes)) {
        const status_t st = rewrite_in_place(
                lock, w, used, owned, bytes, false, num_threads);
        if (st == status_t::success) {
            log_pack(w, weight_cache_mode, "in_place");
            out.data = w.weight;
            return status_t::success;
        }
    }

    // The out-of-place pack runs outside the mutex; in modes 1 / 2 it holds
    // the buffer in packing until its result is published.
    packing_t packing {state};
    if (weight_cache_mode != 0) {
        if (!packing.hold_locked(w.weight)) {
            log_error("native weight: no memory to pack weight=", w.weight);
            return status_t::failure;
        }
    }
    lock.unlock();
    std::unique_ptr<void, native_aligned_free_t> buf(alloc_layout(bytes));
    if (!buf) {
        // Nothing was written: another route can still serve a raw weight.
        log_error("native weight: no buffer for a ", bytes, "-byte layout");
        return status_t::unimplemented;
    }
    size_t source_bytes = 0;
    weight_ends_t source_ends;
    if (weight_cache_mode != 0) {
        source_bytes = has_image ? image.bytes : extent;
        source_ends = has_image ? image.ends : weight_ends_of(w.weight, extent);
    }
    const status_t st = pack_from_source(
            w, has_image ? &image : nullptr, owned, buf.get(), num_threads);
    if (st != status_t::success) return st;
    if (weight_cache_mode != 0) {
        lock.lock();
        if (state.packs.size() < get_weight_cache_capacity()) {
            try {
                auto slot = state.packs.try_emplace(key);
                native_cached_pack_t &entry = slot.first->second;
                if (slot.second) {
                    entry.buf.reset(static_cast<uint8_t *>(buf.release()));
                    entry.source_bytes = source_bytes;
                    entry.source_ends = source_ends;
                }
                packing.release_locked();
                out.data = entry.buf.get();
                log_pack(w, weight_cache_mode, "cache");
                return status_t::success;
            } catch (const std::bad_alloc &) {
                log_error("native weight: no memory for a cache entry");
                packing.release_locked();
                return status_t::failure;
            }
        }
        packing.release_locked();
        lock.unlock();
    }
    log_pack(w, weight_cache_mode, "call");
    out.owned = buf.release();
    out.data = out.owned;
    return status_t::success;
}

void native_free_layout(void *owned) {
    if (owned != nullptr) zendnnl_aligned_free(owned);
}

bool native_weight_packed_in_place(const void *weight, size_t owned_bytes) {
    if (weight == nullptr || !image_may_overlap(weight, 1)) return false;
    native_cache_state_t &state = cache_state();
    {
        std::shared_lock<std::shared_mutex> lock(state.mutex);
        bool interior = false, stale = false;
        const auto it = image_at_locked(weight, owned_bytes, interior, stale);
        if (it == state.images.end()) return false;
        if (!stale) return true;
    }
    exclusive_lock_t lock(state.mutex);
    bool interior = false, stale = false;
    const auto it = image_at_locked(weight, owned_bytes, interior, stale);
    if (it == state.images.end()) return false;
    if (!stale) return true;
    log_stale_image(it->second);
    erase_image_locked(it);
    return false;
}

void clear_native_weight_caches() {
    native_cache_state_t &state = cache_state();
    exclusive_lock_t lock(state.mutex);
    state.packs.clear();
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

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
