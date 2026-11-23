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
#include <cstdlib>
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

#if !defined(_WIN32)
#include <pthread.h>
#endif

#include "common/data_types.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"
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
using zendnnl::error_handling::apilog_warning;
using zendnnl::error_handling::log_error;

namespace {

/// A layout of bytes, 64-byte aligned, from malloc with the offset to its
/// block in the byte before it: a freed layout's block then serves the next
/// layout of its size, where an aligned allocation of a large block leaves
/// a block too small for it and a later layout takes new memory.
void *alloc_layout(size_t bytes) {
    const size_t size = ((bytes + 63) & ~size_t {63}) + 64;
    if (size < bytes) return nullptr;
    unsigned char *block = static_cast<unsigned char *>(std::malloc(size));
    if (block == nullptr) return nullptr;
    unsigned char *p = reinterpret_cast<unsigned char *>(
            (reinterpret_cast<std::uintptr_t>(block) + 64)
            & ~std::uintptr_t {63});
    p[-1] = static_cast<unsigned char>(p - block);
    return p;
}

void free_layout(void *p) {
    if (p == nullptr) return;
    unsigned char *at = static_cast<unsigned char *>(p);
    std::free(at - at[-1]);
}

struct layout_free_t {
    void operator()(void *p) const { free_layout(p); }
};

struct native_key_t {
    const void *weight;
    int K, N, ldb, pack_nr;
    bool transB;
    data_type_t wei_dt;
    bool aocl_blocked;

    bool operator==(const native_key_t &o) const {
        return weight == o.weight && K == o.K && N == o.N && ldb == o.ldb
                && pack_nr == o.pack_nr && transB == o.transB
                && wei_dt == o.wei_dt && aocl_blocked == o.aocl_blocked;
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
        mix((k.transB ? 1u : 0u) | (static_cast<uint32_t>(k.wei_dt) << 2)
                | (k.aocl_blocked ? 1u << 31 : 0u));
        return h;
    }
};

/// A cached layout, with the bytes its pack read from the weight pointer
/// (the raw span, the AOCL prepack or the in-place image) and the digests
/// of their ends at the time.
struct native_cached_pack_t {
    std::unique_ptr<uint8_t, layout_free_t> buf;
    size_t source_bytes = 0;
    weight_ends_t source_ends;
    /// The buffer an in-place rewrite of bytes it was packed from runs on,
    /// when the layout still matched them: it is served without its digests
    /// until the rewrite ends.
    const void *rewritten_by = nullptr;
    /// The serial of the image record it was packed from, or whose rewrite
    /// overwrote the bytes it was packed from while it still matched them;
    /// 0 when none.  While the key's weight lies in a record, at its start or
    /// inside it, only a layout naming that record serves
    /// (serves_in_image_locked).
    uint64_t image_serial = 0;
};

/// Bytes of an image up to which its record keeps every byte, so that a call
/// is checked at each byte of it in its buffer; a larger image keeps the
/// bytes of its two ends.
constexpr size_t kImageKeptBytes = 64 * 1024;

/// Places of lines of kWeightEndBytes between the two ends of an in-place
/// image over kImageKeptBytes, evenly spread (image_sample_offset), whose
/// digests its record keeps where they tell the image from the bytes it
/// replaced: a buffer inside the image that spans more than bytes /
/// kImageSampleLines plus two lines holds one of the places.
constexpr size_t kImageSampleLines = 1024;

/// The sample lines of an image of bytes: each whole line of one of up to
/// kImageKeptBytes, else 1022 to 1024.
size_t image_sample_count(size_t bytes) {
    if (bytes <= kImageKeptBytes) return bytes / kWeightEndBytes;
    return std::min(kImageSampleLines, bytes / kWeightEndBytes - 2);
}

/// Where sample line i of count (image_sample_count) lies in an image of
/// bytes: line i of one of up to kImageKeptBytes, else from the first byte
/// past its head end to the line before its tail end, on 8-byte boundaries,
/// in increasing order.
size_t image_sample_offset(size_t bytes, size_t count, size_t i) {
    if (bytes <= kImageKeptBytes) return i * kWeightEndBytes;
    const size_t span = bytes - 3 * kWeightEndBytes;
    const size_t steps = count - 1;
    const size_t at = span / steps * i + span % steps * i / steps;
    return kWeightEndBytes + (at & ~size_t {7});
}

/// Whether the line of kWeightEndBytes at p repeats one 8-byte word (zeros,
/// or a weight of one value), as freshly cleared or filled memory may.
bool repeats_one_word(const unsigned char *p) {
    uint64_t first = 0;
    std::memcpy(&first, p, sizeof(first));
    for (size_t at = sizeof(first); at < kWeightEndBytes; at += sizeof(first)) {
        uint64_t word = 0;
        std::memcpy(&word, p + at, sizeof(word));
        if (word != first) return false;
    }
    return true;
}

/// A buffer an in-place pack rewrote: the request it was rewritten for
/// (weight = the buffer start), the image bytes, the digests of the lines a
/// call it may serve checks and of the sample lines, and the bytes kept of
/// them.  Reserved while the rewrite runs: the bytes are not the image yet
/// and have no digests, and the buffer is in packing.
struct native_image_t {
    native_raw_weight_t layout;
    size_t bytes = 0;
    /// The image's bytes from check_at that a call it may serve checks, at
    /// their two ends (ends): from the first to past the last line of
    /// kWeightEndBytes where the image differs from the bytes it replaced,
    /// among those a reload of the weight writes back, as the image's own
    /// ends may hold the weight's bytes.
    size_t check_at = 0;
    size_t check_bytes = 0;
    weight_ends_t ends;
    /// The sample lines of an image over kImageKeptBytes
    /// (image_sample_offset), whose bytes kept holds.
    size_t sample_count = 0;
    /// The image's bytes up to kImageKeptBytes, else its head end, its tail
    /// end and its sample lines (kWeightEndBytes each, kept_lines).  A
    /// buffer is not rewritten without them (image_checks_of).
    std::shared_ptr<const uint8_t[]> kept;
    /// A bit for each byte of kept (kWeightEndBytes for each kept line, the
    /// last one of a small image maybe fewer): set where the image differs
    /// from the byte it replaced among those a reload of the weight writes
    /// back, so that the weight reloaded does not hold it.
    std::shared_ptr<const uint8_t[]> informative;
    /// The record covers the bytes from part_at of an image of image_bytes,
    /// the rest of which a call found to hold other bytes; 0 and bytes for a
    /// whole image.
    size_t part_at = 0;
    size_t image_bytes = 0;
    /// A clear forgot the image while the buffer may still hold it, or a
    /// call found other bytes in another part of it, or its image among
    /// other bytes: no call is served from it, and a call on bytes that
    /// still hold it fails.
    bool tomb = false;
    /// A clear met the record as a tombstone already: its buffer is taken
    /// for freed memory, where a call that holds none of its checked bytes
    /// is read as the raw weight it holds.
    bool freed = false;
    bool reserved = false;
    /// Tells the record from every other one, of this buffer or another,
    /// earlier or later (native_cached_pack_t::image_serial); never 0.
    uint64_t serial = 0;
    /// While reserved: the bytes of the rewriting call's buffer, within
    /// which lie the layouts the rewrite marks.
    size_t owned = 0;
};

using image_map_t = std::map<std::uintptr_t, native_image_t>;
using pack_map_t = std::unordered_map<native_key_t, native_cached_pack_t,
        native_key_hash_t>;

struct memo_reader_t;

/// The layout of an entry dropped because its buffer no longer held the
/// bytes it was packed from, with the weight pointer of the entry.  readers
/// holds the reader of every thread listed when it was dropped and the
/// count of its checks then (reclaim_retired_locked); tracked is false when
/// they could not be kept, and the layout then stays until a clear.
struct native_retired_t {
    std::unique_ptr<uint8_t, layout_free_t> buf;
    const void *weight = nullptr;
    std::vector<std::pair<const memo_reader_t *, uint64_t>> readers;
    bool tracked = false;
};

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
    /// mutex, so a call on another thread may still run on one: each is
    /// freed by the first reclaim_retired_locked after no call can read it
    /// (where a layout is dropped, and where a pack is cached or finds the
    /// cache full), or by clear_native_weight_caches, and they count toward
    /// the cache capacity with packs.
    std::vector<native_retired_t> retired;
    /// A thread obtained a cached layout with no reader to count its
    /// lookups (no memory to list one): from then on no dropped layout is
    /// freed until a clear, which frees them all and ends this.
    std::atomic<bool> untracked {false};
    /// Keyed by buffer start; a pointer inside an image resolves to it.
    image_map_t images;
    std::atomic<size_t> image_count {0};
    /// The serial of the last record reserved; under the mutex held
    /// exclusively.
    uint64_t image_serials = 0;
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
    /// while one runs, the memo checks read no weight bytes (memo_check_t).
    /// Changed under the mutex held exclusively.
    std::atomic<size_t> rewrites {0};
};

/// Never destroyed: a call, or a clear from an exit hook, may come after the
/// process began to exit.
native_cache_state_t &cache_state() {
    static native_cache_state_t *state = new native_cache_state_t();
    return *state;
}

/// Under the mutex held exclusively: the memo entries made so far stop
/// serving.  Sequentially consistent with the start of a memo check
/// (begin_check), which stores before the check loads the generation: a
/// thread whose check began after a dropped layout's readers were counted
/// sees the new generation (reclaim_retired_locked).
void bump_generation_locked() {
    cache_state().generation.fetch_add(1, std::memory_order_seq_cst);
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

/// The span of tl that holds at past its start; nullptr when none does or
/// the spans could not be copied.
const std::pair<std::uintptr_t, std::uintptr_t> *span_inside(
        const thread_spans_t &tl, std::uintptr_t at) {
    if (!tl.spans) return nullptr;
    const auto it = std::upper_bound(tl.spans->begin(), tl.spans->end(), at,
            [](std::uintptr_t a,
                    const std::pair<std::uintptr_t, std::uintptr_t> &s) {
        return a < s.second;
    });
    return it != tl.spans->end() && it->first < at ? &*it : nullptr;
}

/// False only when no image overlaps [p, p + bytes); takes no lock while
/// the images have not changed since the calling thread last looked.
bool image_may_overlap(const void *p, size_t bytes) {
    const thread_spans_t *tl = current_thread_spans();
    return tl != nullptr && spans_overlap(*tl, p, bytes);
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

/// True when both ends of [at, at + bytes) lie in the caller's buffer
/// [lo, lo + owned) and keep their digests in ends.
bool ends_held_within(std::uintptr_t at, size_t bytes,
        const weight_ends_t &ends, std::uintptr_t lo, size_t owned) {
    const void *p = reinterpret_cast<const void *>(at);
    const size_t end = std::min(bytes, kWeightEndBytes);
    return within(at, end, lo, owned)
            && within(at + bytes - end, end, lo, owned)
            && weight_end_digest(p, bytes, false) == ends.head
            && weight_end_digest(p, bytes, true) == ends.tail;
}

/// A cached layout whose buffer still holds what its pack read, for a
/// caller whose buffer is the owned bytes at the key's weight.
bool pack_holds(
        const native_key_t &k, const native_cached_pack_t &e, size_t owned) {
    const auto at = reinterpret_cast<std::uintptr_t>(k.weight);
    return e.rewritten_by != nullptr
            || ends_hold(at, e.source_bytes, e.source_ends, at, owned);
}

/// Where the image of the record at it starts: before the record when the
/// record covers a later part of it.
std::uintptr_t image_start(image_map_t::const_iterator it) {
    return it->first - it->second.part_at;
}

/// Narrows the caller's buffer [lo, lo + owned) to the bytes of the record
/// at it; owned 0 when none lies in it.
void within_record(
        image_map_t::const_iterator it, std::uintptr_t &lo, size_t &owned) {
    const std::uintptr_t start = it->first;
    const std::uintptr_t end = start + it->second.bytes;
    if (lo >= end || (lo < start && start - lo >= owned)) {
        owned = 0;
        return;
    }
    const std::uintptr_t to = end - lo <= owned ? end : lo + owned;
    lo = std::max(lo, start);
    owned = to - lo;
}

/// The first sample line of the image of the record at it that starts at lo
/// or past it; the lines that follow lie in a buffer from lo up to the first
/// one that does not.
size_t first_sample_line_from(
        image_map_t::const_iterator it, std::uintptr_t lo) {
    const native_image_t &img = it->second;
    size_t i = 0;
    for (size_t hi = img.sample_count; i < hi;) {
        const size_t mid = i + (hi - i) / 2;
        if (image_start(it)
                        + image_sample_offset(
                                img.image_bytes, img.sample_count, mid)
                < lo)
            i = mid + 1;
        else
            hi = mid;
    }
    return i;
}

/// The lines of kWeightEndBytes a record keeps of an image of bytes, the
/// last one maybe shorter: all of one of up to kImageKeptBytes, else its
/// two ends and its sample lines.
size_t kept_lines(size_t bytes) {
    return bytes <= kImageKeptBytes
            ? (bytes + kWeightEndBytes - 1) / kWeightEndBytes
            : 2 + image_sample_count(bytes);
}

/// Where kept line i (kept_lines) lies in an image of bytes.
size_t kept_line_offset(size_t bytes, size_t i) {
    if (bytes <= kImageKeptBytes || i == 0) return i * kWeightEndBytes;
    if (i == 1) return bytes - kWeightEndBytes;
    return image_sample_offset(bytes, image_sample_count(bytes), i - 2);
}

/// The bytes a record keeps of an image of bytes at p (native_image_t::kept).
std::shared_ptr<const uint8_t[]> image_kept_of(const void *p, size_t bytes) {
    const bool all = bytes <= kImageKeptBytes;
    const size_t lines = kept_lines(bytes);
    uint8_t *kept
            = new (std::nothrow) uint8_t[all ? bytes : lines * kWeightEndBytes];
    if (kept == nullptr) return nullptr;
    const unsigned char *b = static_cast<const unsigned char *>(p);
    if (all) {
        std::memcpy(kept, b, bytes);
    } else {
        for (size_t i = 0; i < lines; ++i)
            std::memcpy(kept + i * kWeightEndBytes,
                    b + kept_line_offset(bytes, i), kWeightEndBytes);
    }
    try {
        return std::shared_ptr<const uint8_t[]>(kept);
    } catch (const std::bad_alloc &) { return nullptr; }
}

/// Whether byte j of the kept bytes of img (native_image_t::informative)
/// differs from the byte its image replaced.
bool informative_at(const native_image_t &img, size_t j) {
    return (img.informative[j / 8] >> (j % 8) & 1) != 0;
}

/// Whether kept bytes [from, to) of img hold a byte that differs from the
/// byte its image replaced.
bool informative_in(const native_image_t &img, size_t from, size_t to) {
    for (size_t j = from; j < to; ++j)
        if (informative_at(img, j)) return true;
    return false;
}

/// Whether kept line i of the record img may tell its image from its weight:
/// a byte of it differs from the byte the image replaced, and it does not
/// repeat one word.
bool kept_line_tells(const native_image_t &img, size_t i) {
    return informative_in(img, i * kWeightEndBytes, (i + 1) * kWeightEndBytes)
            && (img.image_bytes - kept_line_offset(img.image_bytes, i)
                            < kWeightEndBytes
                    || !repeats_one_word(img.kept.get() + i * kWeightEndBytes));
}

/// The bytes of each kept line of an image of bytes at p that differ from
/// the bytes at replaced, of which a reload writes back the first reloaded
/// (native_image_t::informative).
std::shared_ptr<const uint8_t[]> image_informative_of(
        const void *p, const void *replaced, size_t bytes, size_t reloaded) {
    const size_t lines = kept_lines(bytes);
    const size_t mask_bytes = lines * kWeightEndBytes / 8;
    uint8_t *mask = new (std::nothrow) uint8_t[mask_bytes];
    if (mask == nullptr) return nullptr;
    std::memset(mask, 0, mask_bytes);
    const unsigned char *b = static_cast<const unsigned char *>(p);
    const unsigned char *r = static_cast<const unsigned char *>(replaced);
    for (size_t i = 0; i < lines; ++i) {
        const size_t at = kept_line_offset(bytes, i);
        for (size_t k = 0; k < kWeightEndBytes; ++k) {
            const size_t j = at + k;
            if (j < bytes && j < reloaded && b[j] != r[j])
                mask[(i * kWeightEndBytes + k) / 8] = static_cast<uint8_t>(
                        mask[(i * kWeightEndBytes + k) / 8] | 1u << (k % 8));
        }
    }
    try {
        return std::shared_ptr<const uint8_t[]>(mask);
    } catch (const std::bad_alloc &) { return nullptr; }
}

std::atomic<unsigned> g_failing_image_checks {0};

/// Takes one of the failures n counts down, as when no memory is left.
bool take_failing(std::atomic<unsigned> &n) {
    unsigned left = n.load(std::memory_order_relaxed);
    while (left != 0
            && !n.compare_exchange_weak(
                    left, left - 1, std::memory_order_relaxed)) {}
    return left != 0;
}

/// The first line of kWeightEndBytes of the first n bytes, from their start
/// (or from their end when tail, the lines then ending kWeightEndBytes
/// apart from n), where the image at img differs from the bytes at b, one
/// that does not repeat one word if any; n when none differs.
size_t first_differing_line(
        const unsigned char *b, const unsigned char *img, size_t n, bool tail) {
    size_t found = n;
    for (size_t j = 0; j < n / kWeightEndBytes; ++j) {
        const size_t at
                = tail ? n - (j + 1) * kWeightEndBytes : j * kWeightEndBytes;
        if (std::memcmp(b + at, img + at, kWeightEndBytes) == 0) continue;
        if (!repeats_one_word(img + at)) return at;
        if (found == n) found = at;
    }
    return found;
}

/// What image_checks_of found.
enum class image_checks_t { made, same, untold, no_memory };

/// Whether more than half of the kept lines of img, the record of an image
/// of used bytes, that hold a byte where the image differs from the byte it
/// replaced hold a word that tells the image from its weight as image_check
/// reads it: a word with such a byte that is not zero, in a line that does
/// not repeat one word.
bool lines_mostly_tell(const native_image_t &img, size_t used) {
    size_t differing = 0, telling = 0;
    for (size_t i = 0; i < kept_lines(used); ++i) {
        const size_t from = i * kWeightEndBytes;
        const size_t n
                = std::min(kWeightEndBytes, used - kept_line_offset(used, i));
        if (!informative_in(img, from, from + n)) continue;
        ++differing;
        const uint8_t *line = img.kept.get() + from;
        if (n == kWeightEndBytes && repeats_one_word(line)) continue;
        for (size_t w = 0; w < n; w += 8) {
            const size_t m = std::min<size_t>(8, n - w);
            if (informative_in(img, from + w, from + w + m)
                    && std::any_of(line + w, line + w + m,
                            [](uint8_t v) { return v != 0; })) {
                ++telling;
                break;
            }
        }
    }
    return 2 * telling > differing;
}

/// What a record keeps to check the image of used bytes at image, which is
/// to replace the bytes at replaced, of which a reload of its weight writes
/// back the first reloaded: the lines a call it may serve checks
/// (native_image_t::check_at), its kept bytes and, past kImageKeptBytes, the
/// digests of its sample lines.  same when the image equals the replaced
/// bytes at every byte a reload writes: reading the buffer as the weight is
/// right, and a record could not tell the image from the weight reloaded.
/// untold when no more than half of the kept lines where the image differs
/// from them hold a word that tells it from them (lines_mostly_tell): the
/// record would hold too little to tell the image from the weight written
/// back around it.  no_memory when the checks cannot be allocated, and then
/// no buffer may hold the image.
image_checks_t image_checks_of(const void *image, const void *replaced,
        size_t used, size_t reloaded, native_image_t &img) {
    const unsigned char *p = static_cast<const unsigned char *>(image);
    const unsigned char *b = static_cast<const unsigned char *>(replaced);
    const size_t n = std::min(used, reloaded);
    if (n < kWeightEndBytes) {
        if (std::memcmp(b, p, n) == 0) return image_checks_t::same;
        img.check_at = 0;
        img.check_bytes = n;
    } else {
        const size_t head = first_differing_line(b, p, n, false);
        const size_t tail = first_differing_line(b, p, n, true);
        if (head == n && tail == n) return image_checks_t::same;
        // The scans step from 0 and from n, so the two lines may lie in
        // either order.
        const size_t last = head == n ? tail
                : tail == n           ? head
                                      : std::max(head, tail);
        img.check_at = std::min(head, tail);
        img.check_bytes = last + kWeightEndBytes - img.check_at;
    }
    if (take_failing(g_failing_image_checks)) return image_checks_t::no_memory;
    img.ends = weight_ends_of(p + img.check_at, img.check_bytes);
    img.kept = image_kept_of(p, used);
    img.informative = image_informative_of(p, b, used, n);
    img.sample_count = used > kImageKeptBytes ? image_sample_count(used) : 0;
    if (!img.kept || !img.informative) return image_checks_t::no_memory;
    return lines_mostly_tell(img, used) ? image_checks_t::made
                                        : image_checks_t::untold;
}

/// What the checked bytes of an in-place record in a caller's buffer show
/// (image_check).
enum class held_t {
    /// They hold the image, a word or line that tells it from its weight
    /// among them.
    image,
    /// Words or lines that tell the image from its weight hold it; other
    /// checked bytes differ.
    mixed,
    /// Checked bytes differ, and none of the others tells the image from
    /// its weight.
    other,
    /// None differs, and none of them tells the image from its weight.
    same,
    /// None lies in the buffer.
    none,
};

/// What the record at it shows a caller whose buffer is [lo, lo + owned),
/// of which it reads the first reads bytes, the only ones checked with
/// lines.  Without lines (a call the image may serve): the image when the lines
/// that tell it from the weight it replaced (check_at) lie in that buffer
/// and keep their digests; a buffer short of either, or where either
/// differs, is checked as with lines.  With lines: the kept bytes in the
/// buffer (all of an image it kept whole; of a larger one its ends, and its
/// sample lines that may tell it from its weight) word by word, and the
/// two lines by their digests.  A kept word, or the part of one in the
/// buffer, that holds the image and has a byte that differs from the byte
/// it replaced tells the image from its weight, unless its bytes are zero
/// or its line repeats one word, as new zeros or a filled buffer there
/// would; so does a line that keeps its digest.  Bytes outside the buffer,
/// or outside the part of the image the record covers, are not read, as
/// they may no longer be mapped or hold the image.  A reserved record holds
/// its image.  Under the mutex.
held_t image_check(image_map_t::const_iterator it, std::uintptr_t lo,
        size_t owned, size_t reads, bool lines) {
    const native_image_t &img = it->second;
    if (img.reserved) return held_t::image;
    const std::uintptr_t start = image_start(it);
    if (!lines
            && ends_held_within(
                    start + img.check_at, img.check_bytes, img.ends, lo, owned))
        return held_t::image;
    owned = std::min(owned, reads);
    within_record(it, lo, owned);
    if (owned == 0) return held_t::none;
    bool checked = false, differs = false, tells = false;
    // The bytes of kept line i of the image (kept_lines) that lie in the
    // buffer, word by word against the kept bytes.
    const auto kept_line = [&](size_t i) {
        const size_t total = img.image_bytes;
        const size_t at = kept_line_offset(total, i);
        const size_t n = std::min(kWeightEndBytes, total - at);
        const uint8_t *kept = img.kept.get() + i * kWeightEndBytes;
        const std::uintptr_t a = std::max(start + at, lo);
        const std::uintptr_t e = std::min(start + at + n, lo + owned);
        if (a >= e) return;
        const bool repeats = n == kWeightEndBytes && repeats_one_word(kept);
        for (size_t w = (a - start - at) / 8; start + at + 8 * w < e; ++w) {
            const std::uintptr_t ws = start + at + 8 * w;
            const std::uintptr_t x = std::max(ws, a);
            const std::uintptr_t y = std::min({ws + 8, e, start + at + n});
            if (std::memcmp(reinterpret_cast<const void *>(x),
                        kept + (x - start - at), y - x)
                    != 0) {
                checked = differs = true;
                continue;
            }
            checked = true;
            const size_t from = i * kWeightEndBytes + (x - start - at);
            if (repeats || !informative_in(img, from, from + (y - x))) continue;
            if (std::any_of(kept + (x - start - at), kept + (y - start - at),
                        [](uint8_t v) { return v != 0; }))
                tells = true;
        }
    };
    const size_t total = img.image_bytes;
    const size_t count = img.sample_count;
    if (total <= kImageKeptBytes) {
        for (size_t i = (lo - start) / kWeightEndBytes;
                i * kWeightEndBytes < total
                && start + i * kWeightEndBytes < lo + owned;
                ++i)
            kept_line(i);
    } else {
        kept_line(0);
        kept_line(1);
        const void *from = reinterpret_cast<const void *>(start + img.check_at);
        const size_t line = std::min(img.check_bytes, kWeightEndBytes);
        for (const bool at_tail : {false, true}) {
            const std::uintptr_t at = start + img.check_at
                    + (at_tail ? img.check_bytes - line : 0);
            if (!within(at, line, lo, owned)) continue;
            checked = true;
            if (weight_end_digest(from, img.check_bytes, at_tail)
                    != (at_tail ? img.ends.tail : img.ends.head))
                differs = true;
            else if (!repeats_one_word(
                             reinterpret_cast<const unsigned char *>(at)))
                tells = true;
        }
        // A line that cannot tell the image from its weight is not checked,
        // as zeros there or the weight reloaded match it.
        size_t i = first_sample_line_from(it, lo);
        for (i = i > 0 ? i - 1 : 0; i < count; ++i) {
            if (start + image_sample_offset(total, count, i) >= lo + owned)
                break;
            if (kept_line_tells(img, 2 + i)) kept_line(2 + i);
        }
    }
    if (tells) return differs ? held_t::mixed : held_t::image;
    if (differs) return held_t::other;
    return checked ? held_t::same : held_t::none;
}

/// Whether a call whose buffer shows held of the record img is refused or
/// served the image, rather than reading its buffer as a raw weight: not
/// when its checked bytes differ and none of the others tells the image
/// from its weight, nor, on a freed record, when it holds none of them.  A
/// call whose checked bytes hold the image, all or among other bytes, may
/// read it (a view of a buffer that holds it, or whose raw weight was
/// written back in part); one that holds none cannot be told from it.
bool keeps(const native_image_t &img, held_t held) {
    if (held == held_t::other) return false;
    return held != held_t::none || !img.freed;
}

/// Whether a byte image_check checks with lines of the record at it lies in
/// the caller's buffer [lo, lo + owned).
bool image_checked_in(
        image_map_t::const_iterator it, std::uintptr_t lo, size_t owned) {
    const native_image_t &img = it->second;
    const std::uintptr_t start = image_start(it);
    within_record(it, lo, owned);
    const size_t total = img.image_bytes;
    const size_t end = std::min(total, kWeightEndBytes);
    const auto overlaps = [&](size_t at, size_t n) {
        return std::max(start + at, lo) < std::min(start + at + n, lo + owned);
    };
    if (overlaps(0, total <= kImageKeptBytes ? total : end)
            || overlaps(total - end, end))
        return true;
    const size_t line = std::min(img.check_bytes, kWeightEndBytes);
    if (within(start + img.check_at, line, lo, owned)
            || within(start + img.check_at + img.check_bytes - line, line, lo,
                    owned))
        return true;
    size_t i = first_sample_line_from(it, lo);
    for (i = i > 0 ? i - 1 : 0; i < img.sample_count; ++i) {
        const size_t at = image_sample_offset(total, img.sample_count, i);
        if (start + at >= lo + owned) return false;
        if (overlaps(at, kWeightEndBytes) && kept_line_tells(img, 2 + i))
            return true;
    }
    return false;
}

/// The record p points into, under the mutex; interior is set when p lies
/// past its start, and held to what the record shows a caller whose buffer
/// is the owned bytes at p.  A record at p is checked at the lines that
/// tell its image from the weight it replaced only, unless it is a
/// tombstone: a call it may serve reads no more.
image_map_t::iterator image_at_locked(const void *p, size_t owned, size_t reads,
        bool &interior, held_t &held) {
    auto &images = cache_state().images;
    const auto at = reinterpret_cast<std::uintptr_t>(p);
    auto it = images.upper_bound(at);
    if (it == images.begin()) return images.end();
    --it;
    if (at - it->first >= it->second.bytes) return images.end();
    interior = at != it->first;
    held = image_check(it, at, owned, reads, interior || it->second.tomb);
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

/// Makes the record img a tombstone, under the mutex held exclusively: no
/// memo entry serves its image any more.
void entomb_locked(native_image_t &img) {
    if (img.tomb) return;
    img.tomb = true;
    bump_generation_locked();
}

/// Whether bytes [at, at + n) of the image of img may tell it from the
/// weight it replaced: a byte there that its record keeps and that differs
/// from the byte the image replaced, or any byte between the kept lines of
/// an image over kImageKeptBytes, of which its record keeps nothing.
bool part_tells(const native_image_t &img, size_t at, size_t n) {
    const size_t total = img.image_bytes;
    if (total <= kImageKeptBytes) return informative_in(img, at, at + n);
    // The kept lines by offset: the head end, the sample lines, the tail
    // end; sample lines may overlap.
    const size_t lines = kept_lines(total);
    size_t from = at;
    for (size_t k = 0; k < lines && from < at + n; ++k) {
        const size_t i = k == 0 ? 0 : k + 1 < lines ? k + 1 : 1;
        const size_t line = kept_line_offset(total, i);
        if (line + kWeightEndBytes <= from) continue;
        if (line > from) return true;
        const size_t to = std::min(line + kWeightEndBytes, at + n);
        if (informative_in(img, i * kWeightEndBytes + (from - line),
                    i * kWeightEndBytes + (to - line)))
            return true;
        from = to;
    }
    return from < at + n;
}

/// Forgets the record at it over the caller's buffer [lo, lo + owned), which
/// then reads it as a raw weight: the part of the record in the buffer
/// goes, while the rest stays recorded as tombstones (its bytes were not
/// checked, and no call may be served the image any more), but for a rest
/// that holds no word that tells the image from its weight (part_tells):
/// reading it as a raw weight is right, and a reload could not change it.
/// A buffer that holds no byte of the record forgets none.  False when no
/// memory is left to keep the rest: the record stays whole, as a
/// tombstone, and the caller is refused.  Under the mutex held exclusively.
bool forget_image_locked(
        image_map_t::iterator it, std::uintptr_t lo, size_t owned) {
    auto &images = cache_state().images;
    native_image_t &img = it->second;
    const std::uintptr_t start = it->first, end = start + img.bytes;
    within_record(it, lo, owned);
    if (owned == 0) return true;
    const std::uintptr_t hi = lo + owned;
    if (lo == start && hi == end) {
        apilog_info("native weight: the buffer at ", img.layout.weight,
                " no longer holds the in-place image recorded for it; read "
                "as a raw weight");
        erase_image_locked(it);
        return true;
    }
    try {
        if (hi < end && part_tells(img, img.part_at + (hi - start), end - hi)) {
            native_image_t rest = img;
            rest.part_at += hi - start;
            rest.bytes = end - hi;
            rest.tomb = true;
            images.emplace(hi, std::move(rest));
        }
    } catch (const std::bad_alloc &) {
        img.tomb = true;
        publish_spans_locked();
        return false;
    }
    apilog_info("native weight: the bytes at ", reinterpret_cast<void *>(lo),
            " no longer hold the in-place image recorded at ",
            reinterpret_cast<void *>(image_start(it)),
            "; read as a raw weight, the rest of the image stays recorded");
    if (lo > start && part_tells(img, img.part_at, lo - start)) {
        img.bytes = lo - start;
        img.tomb = true;
    } else {
        images.erase(it);
    }
    publish_spans_locked();
    return true;
}

/// Whether the record at it still holds for a caller whose buffer is
/// [lo, lo + owned) and showed held (image_check): the call is refused or
/// served the image.  A record whose checked bytes there show its image
/// among other bytes becomes a tombstone, as its buffer no longer holds
/// the image whole; one that no longer holds is forgotten over the buffer
/// (forget_image_locked).  Under the mutex held exclusively.
bool settle_locked(image_map_t::iterator it, std::uintptr_t lo, size_t owned,
        held_t held) {
    if (held == held_t::mixed) entomb_locked(it->second);
    if (keeps(it->second, held)) return true;
    return !forget_image_locked(it, lo, owned);
}

bool same_layout(const native_raw_weight_t &a, const native_raw_weight_t &b) {
    return a.wei_dt == b.wei_dt && a.K == b.K && a.N == b.N && a.ldb == b.ldb
            && a.transB == b.transB && a.pack_nr == b.pack_nr;
}

/// The record (an image, or a tombstone) whose bytes a request from p reads
/// and that still holds, for a caller whose buffer is the owned bytes at p,
/// under the mutex held exclusively; end() when none.  interior is set when
/// p lies past its start, after when it starts inside the extent bytes the
/// request reads.  A record at p is checked at its sample lines too unless
/// it is an image of served's layout, which the call may be served.
/// Records are settled for the buffer (settle_locked).
image_map_t::iterator live_image_locked(const void *p, size_t extent,
        size_t owned, bool &interior, bool &after,
        const native_raw_weight_t *served) {
    auto &images = cache_state().images;
    const auto at = reinterpret_cast<std::uintptr_t>(p);
    held_t held = held_t::none;
    const size_t reads = std::min(owned, extent);
    auto it = image_at_locked(p, owned, reads, interior, held);
    if (it != images.end() && held == held_t::image && !interior
            && !it->second.tomb
            && (served == nullptr || !same_layout(it->second.layout, *served)))
        held = image_check(it, at, owned, reads, true);
    if (it != images.end() && !settle_locked(it, at, reads, held)) {
        it = images.end();
        interior = false;
    }
    if (it != images.end()) return it;
    for (auto next = image_after_locked(p, extent); next != images.end();
            next = image_after_locked(p, extent)) {
        if (settle_locked(next, at, reads,
                    image_check(next, at, owned, reads, true))) {
            after = true;
            return next;
        }
    }
    return images.end();
}

/// What a check of weight, in a caller's buffer of owned bytes, finds of the
/// record at it, which holds for that buffer: the weight lies past its start
/// (interior, or at the start of a part of an image that a split left) or
/// below it (after).  Under the mutex.
void describe_record(image_map_t::const_iterator it, const void *weight,
        size_t owned, bool interior, bool after, native_image_check_t &found) {
    found.record = reinterpret_cast<const void *>(image_start(it));
    found.record_bytes = it->second.image_bytes;
    found.tomb = it->second.tomb;
    found.interior = interior;
    found.after = after;
    if ((interior || it->second.part_at != 0) && !after && !it->second.reserved)
        found.holds_no_checked_byte = !image_checked_in(
                it, reinterpret_cast<std::uintptr_t>(weight), owned);
}

/// Logs why a request from weight, in a buffer of owned bytes, reads no raw
/// weight: the bytes it reads still hold the record at it, which it starts
/// inside (interior) or below (after).  Under the mutex.
void log_image_refusal(const void *weight, size_t owned,
        image_map_t::const_iterator it, bool interior, bool after) {
    native_image_check_t found;
    describe_record(it, weight, owned, interior, after, found);
    log_error("native weight: weight=", weight,
            " is not read as a raw weight (", native_image_refusal(found),
            "; the image recorded at ", found.record, ", ", found.record_bytes,
            " bytes)");
}

/// The AOCL prepack of an aocl_blocked w is the one of a u8 source.
bool u8s8_prepack(const native_raw_weight_t &w) {
    return native_format(w.wei_dt).cache.foreign_prepack
            == native_prepack_t::u8s8;
}

/// Packs the AOCL inverse of an aocl_blocked w into dst.
status_t pack_from_aocl(
        const native_raw_weight_t &w, void *dst, int num_threads) {
    std::unique_ptr<void, native_aligned_free_t> raw;
    const status_t st = aocl_prepack_unreorder(w.weight, w.K, w.N, w.ldb,
            w.transB, u8s8_prepack(w), num_threads, raw);
    if (st != status_t::success) {
        // Nothing was written: AOCL can still serve its own prepack.
        log_error("native weight: the AOCL blocked weight=", w.weight,
                " was not unreordered (status=", static_cast<int>(st), ")");
        return status_t::unimplemented;
    }
    return native_pack_weight(w.wei_dt, raw.get(), w.K, w.N, w.ldb, w.pack_nr,
            w.transB, dst, num_threads);
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
                w.pack_nr, w.transB, dst, num_threads);
    }
    const native_raw_weight_t &img = image->layout;
    const size_t raw_bytes = static_cast<size_t>(img.K) * img.N
            * (native_geometry(img.wei_dt).elem_bits / 8);
    if (!native_format(img.wei_dt).cache.unpack || w.wei_dt != img.wei_dt
            || native_raw_weight_bytes(w.wei_dt, w.K, w.N, w.ldb, w.transB)
                    > raw_bytes
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
    std::unique_ptr<void, layout_free_t> raw(alloc_layout(raw_bytes));
    if (!raw) {
        log_error("native weight: no buffer to rebuild an in-place image (",
                raw_bytes, " bytes)");
        return status_t::failure;
    }
    native_unpack_weight(img.wei_dt, image->layout.weight, img.K, img.N,
            img.ldb, img.pack_nr, img.transB, raw.get(), num_threads);
    return native_pack_weight(w.wei_dt, raw.get(), w.K, w.N, w.ldb, w.pack_nr,
            w.transB, dst, num_threads);
}

/// Bytes writable at an aocl_blocked weight: the caller's capacity, else
/// the size of its AOCL prepack.
size_t aocl_capacity(const native_raw_weight_t &w, size_t capacity_bytes) {
    if (capacity_bytes != 0) return capacity_bytes;
    return aocl_prepack_bytes(w.K, w.N, w.ldb, w.transB, u8s8_prepack(w));
}

/// The records whose bytes a reservation took (reserve_image_locked), as
/// they were, until its rewrite writes the buffer.
struct displaced_t {
    std::vector<image_map_t::node_type> records;
    /// The last of them reached past the reserved bytes and left its part
    /// there as a tombstone.
    bool split = false;
};

/// Whether a record overlaps [lo, hi), under the mutex.
bool record_overlaps_locked(std::uintptr_t lo, std::uintptr_t hi) {
    const auto &images = cache_state().images;
    auto it = images.lower_bound(lo);
    if (it != images.end() && it->first < hi) return true;
    if (it == images.begin()) return false;
    --it;
    return it->first + it->second.bytes > lo;
}

/// Puts back the records of displaced, taken from the used bytes at at by a
/// reservation that wrote nothing there and whose own record is gone, under
/// the mutex held exclusively: none of their bytes changed.  The tombstone
/// a split left past the used bytes joins its record again while it is as
/// the split left it; otherwise that record comes back over the used bytes
/// only, as a tombstone.  cleared: a clear met the reservation, and each
/// record comes back as that clear left the others (freed: two clears or
/// more).  A record whose bytes another one holds by then stays out.
void restore_displaced_locked(std::uintptr_t at, size_t used,
        displaced_t &displaced, bool cleared, bool freed) {
    auto &images = cache_state().images;
    const std::uintptr_t hi = at + used;
    for (auto &node : displaced.records) {
        native_image_t &img = node.mapped();
        const std::uintptr_t start = node.key(), end = start + img.bytes;
        const bool tomb = img.tomb;
        if (end > hi) {
            const auto rest = images.find(hi);
            if (displaced.split && rest != images.end()
                    && rest->second.serial == img.serial
                    && rest->second.part_at == img.part_at + (hi - start)
                    && rest->second.bytes == end - hi)
                images.erase(rest);
            else if (displaced.split || record_overlaps_locked(hi, end)) {
                img.bytes = hi - start;
                img.tomb = true;
            }
        }
        if (cleared) {
            img.freed = freed || tomb;
            img.tomb = true;
        }
        if (!record_overlaps_locked(start, start + img.bytes))
            images.insert(std::move(node));
    }
    displaced.records.clear();
    displaced.split = false;
}

/// Reserves the record of an in-place image of used bytes at w's buffer of
/// owned bytes before the buffer is rewritten (a buffer is never rewritten
/// without one), after taking the records it overlaps into displaced; end()
/// when no record can be allocated, and then the buffer must not be
/// rewritten and every record is as it was.  Under the mutex held
/// exclusively.
image_map_t::iterator reserve_image_locked(const native_raw_weight_t &w,
        size_t used, size_t owned, displaced_t &displaced) {
    auto &images = cache_state().images;
    const auto at = reinterpret_cast<std::uintptr_t>(w.weight);
    size_t overlapped = 0;
    for (auto it = images.lower_bound(at);
            it != images.end() && it->first - at < used; ++it)
        ++overlapped;
    try {
        displaced.records.reserve(overlapped);
    } catch (const std::bad_alloc &) { return images.end(); }
    for (auto it = images.lower_bound(at);
            it != images.end() && it->first - at < used;) {
        // The bytes of a record past the rewritten ones keep it, as a
        // tombstone.
        const native_image_t &img = it->second;
        const std::uintptr_t end = it->first + img.bytes;
        const size_t skip = at + used - it->first;
        if (end > at + used
                && part_tells(img, img.part_at + skip, end - (at + used))) {
            try {
                native_image_t rest = img;
                rest.part_at += skip;
                rest.bytes = end - (at + used);
                rest.tomb = true;
                images.emplace(at + used, std::move(rest));
                displaced.split = true;
            } catch (const std::bad_alloc &) {
                restore_displaced_locked(at, used, displaced, false, false);
                publish_spans_locked();
                return images.end();
            }
        }
        const auto next = std::next(it);
        displaced.records.push_back(images.extract(it));
        it = next;
    }
    image_map_t::iterator it = images.end();
    try {
        it = images.try_emplace(at).first;
        it->second.layout = w;
        it->second.bytes = used;
        it->second.image_bytes = used;
        it->second.reserved = true;
        it->second.owned = owned;
        it->second.serial = ++cache_state().image_serials;
    } catch (const std::bad_alloc &) {
        restore_displaced_locked(at, used, displaced, false, false);
        it = images.end();
    }
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

/// After the rewrite of weight (done: the buffer was written, its record
/// numbered serial): the image holds the same weight, so the layouts it
/// marked stay, take the digests of their bytes as they are now and name the
/// record.  Clears its marks.  Under the mutex held exclusively.
void refresh_rewritten_layouts_locked(
        const void *weight, bool done, uint64_t serial) {
    for (auto &kv : cache_state().packs) {
        native_cached_pack_t &e = kv.second;
        if (e.rewritten_by != weight) continue;
        if (done) {
            e.source_ends = weight_ends_of(kv.first.weight, e.source_bytes);
            e.image_serial = serial;
        }
        e.rewritten_by = nullptr;
    }
}

/// Whether e, a cached layout at k that holds for its buffer, may serve a
/// request at k.weight: while k.weight lies in a record (at its start or
/// inside it), only when e was packed from that image or carried across its
/// rewrite, or is carried by the rewrite running now.  The bytes of a buffer
/// rewritten again for another weight can match e's digests while the
/// weight differs.  Under the mutex.
bool serves_in_image_locked(
        const native_key_t &k, const native_cached_pack_t &e) {
    const native_cache_state_t &state = cache_state();
    if (e.rewritten_by != nullptr
            || state.image_count.load(std::memory_order_relaxed) == 0)
        return true;
    const auto at = reinterpret_cast<std::uintptr_t>(k.weight);
    auto it = state.images.upper_bound(at);
    if (it == state.images.begin()) return true;
    --it;
    return at - it->first >= it->second.bytes
            || e.image_serial == it->second.serial;
}

/// One warning line per process, at the first in-place rewrite: the buffer
/// a framework handed over holds the native layout from then on.
void note_first_rewrite(const void *weight) {
    static std::atomic<bool> s_noted {false};
    if (s_noted.load(std::memory_order_relaxed)
            || s_noted.exchange(true, std::memory_order_relaxed))
        return;
    apilog_warning(
            "matmul_direct: under ZENDNNL_MATMUL_ALGO=10 the native "
            "BRGEMM rewrote the weight buffer at ",
            weight,
            " in place (weight cache mode 2): from that call the buffer "
            "holds the native layout, which only the native BRGEMM reads; "
            "a weight the framework also reads, saves, updates or shares "
            "takes weight_cache_type 1 or is_weights_const false (logged "
            "once per process)");
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

/// What the memo checks of a thread (memo_check_t) tell in-place rewrites:
/// reading is odd while one may read bytes of a weight buffer.  A line of
/// its own: every check of its thread stores to it.
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

/// The reader of the calling thread (thread_reader), or nullptr.  Trivially
/// destructible: the child of a fork reads it on the thread that forked.
thread_local memo_reader_t *t_reader = nullptr;

#if !defined(_WIN32)
void lock_readers_for_fork();
void unlock_readers_after_fork();
void free_other_readers_after_fork();
#endif

memo_readers_t &memo_readers() {
    static memo_readers_t *readers = [] {
        memo_readers_t *made = new memo_readers_t();
#if !defined(_WIN32)
        pthread_atfork(lock_readers_for_fork, unlock_readers_after_fork,
                free_other_readers_after_fork);
#endif
        return made;
    }();
    return *readers;
}

#if !defined(_WIN32)
/// The mutex of memo_readers is held across a fork: the child finds the
/// list as no thread was changing it.
void lock_readers_for_fork() {
    memo_readers().mutex.lock();
}

void unlock_readers_after_fork() {
    memo_readers().mutex.unlock();
}

/// In the child of a fork only the thread that forked runs: every other
/// listed thread is gone, its check ended and no call of it reads a layout,
/// so its reader goes free, as at its thread's exit.  The forking thread's
/// reader and the counts that dropped layouts noted stay as they are.
void free_other_readers_after_fork() {
    memo_readers_t &readers = memo_readers();
    for (memo_reader_t *r : readers.list) {
        if (r == t_reader || r->free) continue;
        const uint64_t n = r->reading.load(std::memory_order_relaxed);
        r->reading.store(n + (n & 1), std::memory_order_relaxed);
        r->free = true;
    }
    readers.mutex.unlock();
}
#endif

/// While set, a thread with no reader yet is listed none, as when no
/// memory is left (native_fail_reader_listings).
std::atomic<bool> g_fail_reader_listings {false};

/// The reader of the calling thread, listed in memo_readers from its first
/// check (one an exited thread left, else a new one) and left there when it
/// exits; nullptr when it cannot be listed.
memo_reader_t *thread_reader() {
    struct owner_t {
        ~owner_t() {
            if (t_reader == nullptr) return;
            std::lock_guard<std::mutex> guard(memo_readers().mutex);
            t_reader->free = true;
            t_reader = nullptr;
        }
    };
    thread_local owner_t owner;
    (void)owner;
    if (t_reader == nullptr) {
        if (g_fail_reader_listings.load(std::memory_order_relaxed))
            return nullptr;
        memo_readers_t &readers = memo_readers();
        std::lock_guard<std::mutex> guard(readers.mutex);
        for (memo_reader_t *r : readers.list) {
            if (!r->free) continue;
            r->free = false;
            // The calls of the thread that left it have returned: its count
            // passes every count a dropped layout noted, as a later call of
            // that thread would.
            const uint64_t n = r->reading.load(std::memory_order_relaxed);
            r->reading.store(n + 2 + (n & 1), std::memory_order_release);
            t_reader = r;
            return r;
        }
        std::unique_ptr<memo_reader_t> r(new (std::nothrow) memo_reader_t());
        if (r == nullptr) return nullptr;
        try {
            readers.list.push_back(r.get());
        } catch (const std::bad_alloc &) { return nullptr; }
        t_reader = r.release();
    }
    return t_reader;
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

/// The calling thread obtains a cached layout outside its memo: its reader
/// counts the lookup, as a memo check does, so that a layout dropped later
/// waits for this call to end (reclaim_retired_locked).  Under the mutex.
void note_layout_obtained_locked() {
    memo_reader_t *reader = thread_reader();
    if (reader == nullptr) {
        cache_state().untracked.store(true, std::memory_order_relaxed);
        return;
    }
    reader->reading.store(reader->reading.load(std::memory_order_relaxed) + 2,
            std::memory_order_release);
}

/// Counts, for the layout e just dropped, the checks of every listed thread
/// that may have obtained it.  Under the mutex held exclusively, after the
/// generation was bumped.
void note_retired_readers_locked(native_retired_t &e) {
    memo_readers_t &readers = memo_readers();
    std::lock_guard<std::mutex> guard(readers.mutex);
    try {
        e.readers.reserve(readers.list.size());
        for (const memo_reader_t *r : readers.list)
            if (!r->free)
                e.readers.emplace_back(
                        r, r->reading.load(std::memory_order_seq_cst));
        e.tracked = true;
    } catch (const std::bad_alloc &) {
        e.readers.clear();
        e.tracked = false;
    }
}

/// Frees the dropped layouts that no call can read any more.  A call reads
/// a cached layout only after its thread obtained it (a memo check, or a
/// lookup that note_layout_obtained_locked counted), once per call and with
/// no lookup or check of this cache after it in the call, and a thread runs
/// one call at a time: a thread whose count passed the one noted when the
/// layout was dropped (by 1 from an even count, by 2 from an odd one, as a
/// check running then may still obtain it) has begun a later call, and its
/// call that may have read the layout has returned, its team's reads with
/// it.  A thread whose check began after the count sees the new generation
/// and is not served the layout.  Runs only where a layout is dropped and
/// where a pack is cached or finds the cache full, never on a lookup served
/// a cached layout or an image.  Under the mutex held exclusively.
void reclaim_retired_locked() {
    native_cache_state_t &state = cache_state();
    if (state.retired.empty()
            || state.untracked.load(std::memory_order_relaxed))
        return;
    memo_readers_t &readers = memo_readers();
    std::lock_guard<std::mutex> guard(readers.mutex);
    // One pass: the mutexes are held, and an exiting reader can make
    // thousands of entries reclaimable at once.
    const auto unread = [](const native_retired_t &dropped) {
        bool reclaimable = dropped.tracked;
        for (size_t i = 0; reclaimable && i < dropped.readers.size(); ++i) {
            const memo_reader_t *r = dropped.readers[i].first;
            const uint64_t at = dropped.readers[i].second;
            reclaimable = r->free
                    || r->reading.load(std::memory_order_acquire)
                            > at + (at & 1);
        }
        return reclaimable;
    };
    state.retired.erase(
            std::remove_if(state.retired.begin(), state.retired.end(), unread),
            state.retired.end());
}

std::atomic<unsigned> g_failing_retirements {0};

/// Whether the drop of a layout is to fail as when no memory is left
/// (native_fail_next_retirements).
bool take_failing_retirement() {
    return take_failing(g_failing_retirements);
}

/// One warning line per process when the cache holds its capacity of
/// layouts and packs a layout for its call instead.
void note_capacity_reached(uint32_t capacity) {
    static std::atomic<bool> s_noted {false};
    if (s_noted.load(std::memory_order_relaxed)
            || s_noted.exchange(true, std::memory_order_relaxed))
        return;
    apilog_warning(
            "native weight: the native layout cache is at its "
            "capacity of ",
            capacity,
            " layouts (ZENDNNL_LRU_CACHE_CAPACITY; dropped layouts a call may "
            "still read count toward it): weights that do not fit are packed "
            "for each of their calls until a clear makes room (logged once "
            "per process)");
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
    /// The bytes from key.weight + check_at whose ends the digests cover:
    /// what the cached layout was packed from, or the lines of the image a
    /// call it may serve checks (native_image_t::check_at).
    size_t check_at = 0;
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
/// memo holds.
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
    return cache_state().generation.load(std::memory_order_seq_cst) == gen;
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

/// An entry of memo for a cached layout at weight, found in generation gen,
/// else nullptr.
const memo_entry_t *memo_find_layout(
        const layout_memo_t &memo, const void *weight, uint64_t gen) {
    const memo_set_t &set = memo.sets[memo_set_index(weight)];
    for (size_t i = 0; i < kMemoWays; ++i)
        if (set.weights[i] == weight && set.ways[i].generation == gen
                && !set.ways[i].image)
            return &set.ways[i];
    return nullptr;
}

/// Remembers for the thread that key is served layout while the check_bytes
/// at key.weight + check_at keep the digests ends, as found under the mutex
/// in the cache generation of that moment.
void memo_put_locked(const native_key_t &key, const native_layout_t &layout,
        size_t check_at, size_t check_bytes, const weight_ends_t &ends,
        bool image) {
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
    e.check_at = check_at;
    e.check_bytes = check_bytes;
    e.ends = ends;
    e.generation = gen;
    e.image = image;
    set->weights[way] = key.weight;
}

native_key_t key_of(const native_raw_weight_t &w) {
    return {w.weight, w.K, w.N, w.ldb, w.pack_nr, w.transB, w.wei_dt,
            w.aocl_blocked};
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
    w.aocl_blocked = k.aocl_blocked;
    return w;
}

/// Remembers the cached layout of key for the thread, unless an in-place
/// rewrite of its bytes runs.  Under the mutex.
void memo_pack_locked(const native_key_t &key, const native_cached_pack_t &e) {
    if (e.rewritten_by != nullptr) return;
    native_layout_t layout;
    layout.data = e.buf.get();
    memo_put_locked(key, layout, 0, e.source_bytes, e.source_ends, false);
}

/// Remembers for the thread that key is served from the image img at its
/// start.  Under the mutex.
void memo_image_locked(const native_key_t &key, const native_image_t &img) {
    if (img.reserved || img.tomb) return;
    native_layout_t layout;
    layout.data = key.weight;
    memo_put_locked(key, layout, img.check_at, img.check_bytes, img.ends, true);
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
            ? aocl_prepack_bytes(w.K, w.N, w.ldb, w.transB, u8s8_prepack(w))
            : native_raw_weight_bytes(w.wei_dt, w.K, w.N, w.ldb, w.transB);
    return {w, weight_cache_mode, capacity_bytes, num_threads, key_of(w),
            native_packed_bytes(w.wei_dt, w.K, w.N, w.pack_nr), extent,
            std::max(extent, capacity_bytes)};
}

void own_layout(
        std::unique_ptr<void, layout_free_t> buf, native_layout_t &out) {
    out.owned = buf.release();
    out.data = out.owned;
}

/// Packs r into a buffer the caller owns, from image when its buffer holds
/// one.
status_t pack_for_call(const layout_request_t &r, const native_image_t *image,
        native_layout_t &out) {
    std::unique_ptr<void, layout_free_t> buf(alloc_layout(r.bytes));
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
    if (!ends_hold(at + e->check_at, e->check_bytes, e->ends, at, r.owned)
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
                && pack_holds(hit->first, hit->second, r.owned)
                && serves_in_image_locked(hit->first, hit->second)) {
            out.data = hit->second.buf.get();
            memo_pack_locked(r.key, hit->second);
            note_layout_obtained_locked();
            return true;
        }
    }
    if (may_image) {
        bool interior = false;
        held_t held = held_t::none;
        const auto it = image_at_locked(
                r.w.weight, r.owned, r.extent, interior, held);
        if (it != state.images.end() && !interior && held == held_t::image
                && !it->second.tomb && same_layout(it->second.layout, r.w)) {
            out.data = r.w.weight;
            memo_image_locked(r.key, it->second);
            return true;
        }
    }
    return false;
}

enum class lookup_t { found, missing, failed };

/// The lookup again under the mutex held exclusively: the cached layout that
/// still holds and serves in the image that holds r (one that does not is
/// dropped), else the in-place image, which serves r when it holds r's
/// layout and is returned in image otherwise.  failed when r starts inside
/// an image or reads into one.
lookup_t find_layout_exclusive(const layout_request_t &r, native_image_t &image,
        bool &has_image, native_layout_t &out) {
    native_cache_state_t &state = cache_state();
    if (r.weight_cache_mode != 0) {
        const auto hit = state.packs.find(r.key);
        if (hit != state.packs.end()) {
            if (pack_holds(hit->first, hit->second, r.owned)
                    && serves_in_image_locked(hit->first, hit->second)) {
                out.data = hit->second.buf.get();
                memo_pack_locked(r.key, hit->second);
                note_layout_obtained_locked();
                return lookup_t::found;
            }
            apilog_info("native weight: the buffer at ", r.w.weight,
                    " no longer holds the weight its cached layout was "
                    "packed from; packed again from its bytes");
            // The entry keeps the layout until the dropped one that takes
            // it exists: a call on another thread may still read it.
            try {
                if (take_failing_retirement()) throw std::bad_alloc();
                state.retired.emplace_back();
            } catch (const std::bad_alloc &) {
                log_error(
                        "native weight: no memory to keep a replaced "
                        "layout");
                return lookup_t::failed;
            }
            native_retired_t &dropped = state.retired.back();
            dropped.buf = std::move(hit->second.buf);
            dropped.weight = hit->first.weight;
            bump_generation_locked();
            state.packs.erase(hit);
            note_retired_readers_locked(dropped);
            reclaim_retired_locked();
        }
    }
    has_image = false;
    if (state.image_count.load(std::memory_order_relaxed) != 0) {
        bool interior = false, after = false;
        const auto found = live_image_locked(
                r.w.weight, r.extent, r.owned, interior, after, &r.w);
        if (found != state.images.end()) {
            if (found->second.tomb || interior || after) {
                log_image_refusal(r.w.weight, std::min(r.owned, r.extent),
                        found, interior, after);
                return lookup_t::failed;
            }
            image = found->second;
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

/// Caches buf, the layout of r packed from the source_bytes at its weight
/// (from the image numbered image_serial, or 0), whose ends had the digests
/// source_ends, and serves r from it; past the cache capacity r owns it
/// instead.  With the mutex held exclusively, which it releases past the
/// capacity, as it releases packing.
status_t cache_layout_locked(exclusive_lock_t &lock, const layout_request_t &r,
        std::unique_ptr<void, layout_free_t> buf, size_t source_bytes,
        const weight_ends_t &source_ends, uint64_t image_serial,
        packing_t &packing, native_layout_t &out) {
    native_cache_state_t &state = cache_state();
    const uint32_t capacity = get_weight_cache_capacity();
    // Dropped layouts that no call reads any more leave room before the
    // capacity turns the layout away; nothing else would free them here.
    // This call obtained no cached layout, so its thread's earlier calls
    // have returned: its lookup counts first, as a lookup that finds one.
    if (state.packs.size() + state.retired.size() >= capacity) {
        note_layout_obtained_locked();
        reclaim_retired_locked();
    }
    if (state.packs.size() + state.retired.size() < capacity) {
        try {
            auto slot = state.packs.try_emplace(r.key);
            native_cached_pack_t &entry = slot.first->second;
            if (slot.second) {
                entry.buf.reset(static_cast<uint8_t *>(buf.release()));
                entry.source_bytes = source_bytes;
                entry.source_ends = source_ends;
                entry.image_serial = image_serial;
                mark_published_layout_locked(r.key, entry);
            }
            packing.release_locked();
            out.data = entry.buf.get();
            memo_pack_locked(r.key, entry);
            note_layout_obtained_locked();
            reclaim_retired_locked();
            log_pack(r.w, r.weight_cache_mode, "cache");
            return status_t::success;
        } catch (const std::bad_alloc &) {
            log_error("native weight: no memory for a cache entry");
            packing.release_locked();
            return status_t::failure;
        }
    }
    note_capacity_reached(capacity);
    packing.release_locked();
    lock.unlock();
    log_pack(r.w, r.weight_cache_mode, "call");
    own_layout(std::move(buf), out);
    return status_t::success;
}

/// What rewrite_in_place did.
enum class rewrite_t {
    written, ///< the buffer holds the image, recorded
    /// the image would change no byte a reload writes, or its record would
    /// hold too little to tell it from the weight: not written
    same,
    not_written, ///< no memory, or the AOCL prepack not unreordered
};

/// What rewrite_in_place did, and the layout it packed when it wrote
/// nothing after the pack: the layout of the bytes the buffer still holds,
/// whose r.extent bytes had the digests source_ends before the pack.
struct rewrite_result_t {
    rewrite_t done = rewrite_t::not_written;
    std::unique_ptr<void, layout_free_t> layout;
    weight_ends_t source_ends;
};

/// Rewrites the buffer of r's weight in place into the used bytes of its
/// layout, from its AOCL prepack or from the raw weight.  Entered and left
/// with the mutex held exclusively; the pack and the write-back run outside
/// it, with the buffer in packing and its image record reserved, so calls on
/// the buffer wait and the other calls go on, and once no read of other
/// calls holds bytes it writes and no check reads weight bytes.  The image
/// is packed aside and its record's checks made from it and the bytes it
/// replaces first, so an AOCL prepack's rewrite holds the raw weight the
/// AOCL inverse rebuilds and the image at once: without memory for them the
/// buffer is not written, nor when the image would change none of the first
/// r.extent bytes, those a reload of the weight writes back (same): reading
/// the buffer as that weight is right, and a record could not tell it from
/// the weight reloaded; nor when its record would hold too little to tell
/// it from the weight written back around it (untold).  A buffer not written
/// keeps the weight for the
/// out-of-place cache, which takes the layout packed aside, or for AOCL-DLP,
/// and the records its reservation took come back, as its bytes did not
/// change (restore_displaced_locked).
rewrite_result_t rewrite_in_place(
        exclusive_lock_t &lock, const layout_request_t &r, size_t used) {
    const native_raw_weight_t &w = r.w;
    native_cache_state_t &state = cache_state();
    rewrite_result_t result;
    // An AOCL prepack not rewritten is handed back; a bf16, f16 or s4 weight
    // is packed out of place, and the call goes on.
    const auto log_not_written = [&w](const char *what) {
        if (w.aocl_blocked)
            log_error("native weight: ", what, "; the buffer is not rewritten");
        else
            apilog_warning("native weight: ", what,
                    "; the buffer is not rewritten, its layout is packed out "
                    "of place");
    };
    packing_t packing {state};
    if (!packing.hold_locked(w.weight)) return result;
    displaced_t displaced;
    if (reserve_image_locked(w, used, r.owned, displaced)
            == state.images.end()) {
        log_not_written("no memory for the record of an in-place image");
        packing.release_locked();
        return result;
    }
    state.rewrites.fetch_add(1, std::memory_order_seq_cst);
    mark_rewritten_layouts_locked(w.weight, used, r.owned);
    wait_for_reads_locked(lock, w.weight, used);
    lock.unlock();
    wait_for_memo_checks();
    std::unique_ptr<void, layout_free_t> tmp(alloc_layout(r.bytes));
    result.source_ends = weight_ends_of(w.weight, r.extent);
    const status_t packed = !tmp ? status_t::unimplemented
            : w.aocl_blocked
            ? pack_from_aocl(w, tmp.get(), r.num_threads)
            : native_pack_weight(w.wei_dt, w.weight, w.K, w.N, w.ldb, w.pack_nr,
                      w.transB, tmp.get(), r.num_threads);
    rewrite_t done = packed == status_t::success ? rewrite_t::written
                                                 : rewrite_t::not_written;
    native_image_t checks;
    if (done == rewrite_t::written) {
        switch (image_checks_of(tmp.get(), w.weight, used, r.extent, checks)) {
            case image_checks_t::made: break;
            case image_checks_t::same:
            case image_checks_t::untold: done = rewrite_t::same; break;
            case image_checks_t::no_memory:
                log_not_written(
                        "no memory for the checks of an in-place "
                        "image");
                done = rewrite_t::not_written;
                break;
        }
    }
    if (done == rewrite_t::written)
        // Weight cache mode 2 is the caller's consent that the weight buffer
        // is writable storage read by this backend only.
        std::memcpy(const_cast<void *>(w.weight), tmp.get(), used);
    lock.lock();
    const auto it
            = state.images.find(reinterpret_cast<std::uintptr_t>(w.weight));
    refresh_rewritten_layouts_locked(w.weight, done == rewrite_t::written,
            it != state.images.end() ? it->second.serial : 0);
    if (done == rewrite_t::written) {
        // A clear meanwhile made the record a tombstone, and it stays one.
        if (it != state.images.end()) {
            it->second.check_at = checks.check_at;
            it->second.check_bytes = checks.check_bytes;
            it->second.ends = checks.ends;
            it->second.sample_count = checks.sample_count;
            it->second.kept = std::move(checks.kept);
            it->second.informative = std::move(checks.informative);
            it->second.reserved = false;
        }
    } else {
        bool cleared = false, freed = false;
        if (it != state.images.end()) {
            cleared = it->second.tomb;
            freed = it->second.freed;
            state.images.erase(it);
        }
        restore_displaced_locked(reinterpret_cast<std::uintptr_t>(w.weight),
                used, displaced, cleared, freed);
        publish_spans_locked();
    }
    state.rewrites.fetch_sub(1, std::memory_order_release);
    packing.release_locked();
    result.done = done;
    if (done != rewrite_t::written && packed == status_t::success)
        result.layout = std::move(tmp);
    return result;
}

/// Weight cache mode 2 on a buffer with no image: rewrites it in place when
/// its layout fits it (an AOCL prepack, or a contiguous bf16, f16 or s4
/// weight).  done is set when the buffer is rewritten; an AOCL rewrite that
/// wrote nothing for want of memory or of the AOCL inverse is unimplemented
/// (AOCL-DLP can still serve its own prepack), a bf16, f16 or s4 one leaves
/// r to the out-of-place cache, and so does a rewrite that would change none
/// of the bytes a reload writes or whose record would hold too little to
/// tell its image from the weight: the cache takes the layout the rewrite
/// packed, and done is set, or r is packed out of place when there is none.
/// A layout of padded columns (N not a multiple of pack_nr) is never
/// written in place.  With the mutex held exclusively (rewrite_in_place).
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
    } else {
        if (!native_format(w.wei_dt).cache.in_place || !contiguous)
            return status_t::success;
        const size_t plain
                = native_raw_weight_bytes(w.wei_dt, w.K, w.N, w.ldb, w.transB);
        if (!wei_inplace_fits(used, used, plain, r.capacity_bytes))
            return status_t::success;
    }
    rewrite_result_t result = rewrite_in_place(lock, r, used);
    if (result.done == rewrite_t::not_written && w.aocl_blocked)
        return status_t::unimplemented;
    if (result.done != rewrite_t::written) {
        if (!result.layout) return status_t::success;
        done = true;
        packing_t none {cache_state()};
        return cache_layout_locked(lock, r, std::move(result.layout), r.extent,
                result.source_ends, 0, none, out);
    }
    log_pack(w, r.weight_cache_mode, "in_place");
    note_first_rewrite(w.weight);
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
    std::unique_ptr<void, layout_free_t> buf(alloc_layout(r.bytes));
    if (!buf) {
        // Nothing was written: another route can still serve a raw weight.
        log_error("native weight: no buffer for a ", r.bytes, "-byte layout");
        return status_t::unimplemented;
    }
    // A rebuild from an image reads all of it, within the caller's buffer.
    const weight_ends_t source_ends = source_bytes <= r.owned
            ? weight_ends_of(w.weight, source_bytes)
            : weight_ends_t {};
    const status_t st
            = pack_from_source(w, image, r.owned, buf.get(), r.num_threads);
    if (st != status_t::success) return st;
    lock.lock();
    read.release_locked();
    return cache_layout_locked(lock, r, std::move(buf), source_bytes,
            source_ends, image != nullptr ? image->serial : 0, packing, out);
}

} // namespace

size_t native_raw_weight_bytes(
        data_type_t wei_dt, int K, int N, int ldb, bool transB) {
    const size_t rows = static_cast<size_t>(transB ? N : K);
    const size_t cols = static_cast<size_t>(transB ? K : N);
    const size_t elements = (rows - 1) * static_cast<size_t>(ldb) + cols;
    // A format of sub-byte elements packs them in bytes, the first in the
    // low bits of byte 0 (two s4 elements per byte).
    const native_format_desc_t *f = native_format_of(wei_dt);
    if (f != nullptr && f->geometry.elem_bits < 8) {
        const size_t per_byte = static_cast<size_t>(8 / f->geometry.elem_bits);
        return (elements + per_byte - 1) / per_byte;
    }
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
    free_layout(owned);
}

namespace {

/// What the record at it, which holds for the caller whose buffer is the
/// owned bytes at weight, tells a call on weight; an image the call starts
/// at is remembered for the thread.  Under the mutex.
void note_image_locked(image_map_t::const_iterator it, const void *weight,
        size_t owned, bool interior, native_image_check_t &found) {
    describe_record(it, weight, owned, interior, false, found);
    if (interior || it->second.reserved || it->second.tomb) return;
    found.ready = true;
    found.layout = it->second.layout;
    memo_image_locked(key_of(it->second.layout), it->second);
}

} // namespace

bool native_weight_find_image(const void *weight, size_t owned_bytes,
        native_image_check_t &found, bool layouts, size_t read_bytes) {
    found = native_image_check_t {};
    found.checked_bytes = owned_bytes;
    if (weight == nullptr
            || !image_may_overlap(weight, std::max<size_t>(owned_bytes, 1)))
        return false;
    native_cache_state_t &state = cache_state();
    const auto at = reinterpret_cast<std::uintptr_t>(weight);
    if (layout_memo_t *memo = thread_memo(false)) {
        const uint64_t gen = state.generation.load(std::memory_order_acquire);
        const memo_entry_t *e = memo_find_image(*memo, weight, gen);
        const std::pair<std::uintptr_t, std::uintptr_t> *span = nullptr;
        if (e == nullptr && layouts) {
            span = span_inside(thread_spans(), at);
            if (span != nullptr) e = memo_find_layout(*memo, weight, gen);
        }
        if (e != nullptr) {
            const memo_check_t check(*memo->reader);
            // A buffer short of a line the entry checks is checked at the
            // record's bytes it holds.
            if (check.ok()
                    && ends_held_within(at + e->check_at, e->check_bytes,
                            e->ends, at, owned_bytes)
                    && memo_still_current(e->generation)) {
                found.layout = raw_of(e->key);
                if (e->image) {
                    found.ready = true;
                } else {
                    found.interior = true;
                    found.memo_layout = e->layout.data;
                    found.record = reinterpret_cast<const void *>(span->first);
                    found.record_bytes = span->second - span->first;
                }
                return true;
            }
        }
    }
    const size_t reads = std::min(owned_bytes, read_bytes);
    {
        // A record whose image shows among other bytes is made a tombstone
        // under the lock held exclusively.
        std::shared_lock<std::shared_mutex> lock(state.mutex);
        bool interior = false;
        held_t held = held_t::none;
        const auto it
                = image_at_locked(weight, owned_bytes, reads, interior, held);
        if (it != state.images.end()) {
            if (held != held_t::mixed && keeps(it->second, held)) {
                note_image_locked(it, weight, reads, interior, found);
                return true;
            }
        } else {
            const auto next = image_after_locked(weight, reads);
            if (next == state.images.end()) return false;
            held = image_check(next, at, owned_bytes, reads, true);
            if (held != held_t::mixed && keeps(next->second, held)) {
                describe_record(next, weight, reads, false, true, found);
                return true;
            }
        }
    }
    exclusive_lock_t lock(state.mutex);
    bool interior = false;
    held_t held = held_t::none;
    const auto it = image_at_locked(weight, owned_bytes, reads, interior, held);
    if (it != state.images.end() && settle_locked(it, at, reads, held)) {
        note_image_locked(it, weight, reads, interior, found);
        return true;
    }
    for (auto next = image_after_locked(weight, reads);
            next != state.images.end();
            next = image_after_locked(weight, reads)) {
        if (settle_locked(next, at, reads,
                    image_check(next, at, owned_bytes, reads, true))) {
            describe_record(next, weight, reads, false, true, found);
            return true;
        }
    }
    return false;
}

bool native_weight_packed_in_place(
        const void *weight, size_t owned_bytes, size_t read_bytes) {
    native_image_check_t found;
    return native_weight_find_image(
            weight, owned_bytes, found, false, read_bytes);
}

const char *native_image_refusal(const native_image_check_t &found) {
    if (found.tomb && found.holds_no_checked_byte)
        return "the weight lies inside an in-place image that serves no "
               "call, as a weight cache clear forgot it or other bytes "
               "overwrote part of it, and holds none of the bytes its record "
               "checks, so it cannot be told from the image: reload the raw "
               "weight of the image's whole buffer, or, if that buffer was "
               "freed, clear the weight caches once more while no call runs "
               "and reload every other weight a mode 2 call rewrote before "
               "calls resume";
    if (found.tomb)
        return "the bytes it reads still hold an in-place image that serves "
               "no call, as a weight cache clear forgot it or other bytes "
               "overwrote part of it: reload the raw weight of the image's "
               "whole buffer";
    if (found.holds_no_checked_byte)
        return "the weight lies inside an in-place image and holds none of "
               "the bytes its record checks, so it cannot be told from the "
               "image: if the image's buffer was freed, clear the weight "
               "caches twice while no call runs, then reload every other "
               "weight a mode 2 call rewrote";
    if (found.after)
        return "the weight reads into an in-place image whose bytes it reads "
               "still hold it: while no call runs, write the raw weight back "
               "into the image's buffer, clear the weight caches and reload "
               "every weight a mode 2 call rewrote";
    if (found.interior)
        return "the weight lies inside an in-place image whose bytes it "
               "reads still hold it: while no call runs, write the raw "
               "weight back into the image's buffer, clear the weight caches "
               "and reload every weight a mode 2 call rewrote";
    return "the weight's buffer holds an in-place image, which only the "
           "native BRGEMM reads: while no call runs, write the raw weight "
           "back into the image's buffer, clear the weight caches and reload "
           "every weight a mode 2 call rewrote";
}

bool native_weight_retained_layout(const native_raw_weight_t &w,
        int32_t weight_cache_mode, size_t capacity_bytes) {
    if (weight_cache_mode == 0 || w.weight == nullptr
            || !native_layout_fits(w.wei_dt, w.K, w.N, w.pack_nr)
            || w.ldb < (w.transB ? w.K : w.N))
        return false;
    const layout_request_t r
            = make_request(w, weight_cache_mode, capacity_bytes, 1);
    native_cache_state_t &state = cache_state();
    std::shared_lock<std::shared_mutex> lock(state.mutex);
    const auto hit = state.packs.find(r.key);
    // A layout a running rewrite did not mark is not read while the rewrite
    // may write its bytes; it was not carried.
    return hit != state.packs.end() && !pack_check_waits_locked(r)
            && pack_holds(hit->first, hit->second, r.owned)
            && serves_in_image_locked(hit->first, hit->second);
}

native_raw_read_t native_weight_raw_hold_t::hold(const void *weight,
        size_t extent, size_t owned_bytes, native_image_check_t *found) {
    native_cache_state_t &state = cache_state();
    exclusive_lock_t lock(state.mutex);
    state.packed.wait(lock, [&] { return state.packing.count(weight) == 0; });
    if (state.image_count.load(std::memory_order_relaxed) != 0) {
        bool interior = false, after = false;
        const size_t owned = std::max(extent, owned_bytes);
        const auto it = live_image_locked(
                weight, extent, owned, interior, after, nullptr);
        if (it != state.images.end()) {
            if (found != nullptr)
                describe_record(it, weight, std::min(owned, extent), interior,
                        after, *found);
            return native_raw_read_t::image;
        }
    }
    read_hold_t hold {state};
    if (!hold.hold_locked(weight, extent)) return native_raw_read_t::failed;
    span_ = hold.span;
    held_ = true;
    hold.held = false;
    return native_raw_read_t::done;
}

native_weight_raw_hold_t::~native_weight_raw_hold_t() {
    if (!held_) return;
    read_hold_t release {cache_state(), span_, true};
}

void clear_native_weight_caches() {
    native_cache_state_t &state = cache_state();
    exclusive_lock_t lock(state.mutex);
    bump_generation_locked();
    state.packs.clear();
    state.retired.clear();
    // No call runs: no thread without a reader holds a layout any more.
    state.untracked.store(false, std::memory_order_relaxed);
    // A buffer may still hold its image, and may already be freed: each
    // record becomes a tombstone, which reads no byte now, and a tombstone
    // already is taken for freed memory.  A rewrite that a clear meets
    // anyway, outside its contract, may still write its buffer: its record
    // stays, as a tombstone once written.
    for (auto &kv : state.images) {
        kv.second.freed = kv.second.tomb;
        kv.second.tomb = true;
    }
    publish_spans_locked();
}

size_t native_weight_layouts(const void *weight) {
    if (weight == nullptr) return 0;
    native_cache_state_t &state = cache_state();
    std::shared_lock<std::shared_mutex> lock(state.mutex);
    size_t n = 0;
    for (const auto &kv : state.packs)
        if (kv.first.weight == weight) ++n;
    const auto it = state.images.find(reinterpret_cast<std::uintptr_t>(weight));
    return n + (it != state.images.end() && !it->second.tomb ? 1 : 0);
}

size_t native_weight_tombstones(const void *weight) {
    if (weight == nullptr) return 0;
    native_cache_state_t &state = cache_state();
    std::shared_lock<std::shared_mutex> lock(state.mutex);
    const auto it = state.images.find(reinterpret_cast<std::uintptr_t>(weight));
    return it != state.images.end() && it->second.tomb ? 1 : 0;
}

void native_fail_next_retirements(unsigned n) {
    g_failing_retirements.store(n, std::memory_order_relaxed);
}

void native_fail_next_image_checks(unsigned n) {
    g_failing_image_checks.store(n, std::memory_order_relaxed);
}

void native_fail_reader_listings(bool fail) {
    g_fail_reader_listings.store(fail, std::memory_order_relaxed);
}

size_t native_retired_layouts(const void *weight) {
    if (weight == nullptr) return 0;
    native_cache_state_t &state = cache_state();
    std::shared_lock<std::shared_mutex> lock(state.mutex);
    return static_cast<size_t>(std::count_if(state.retired.begin(),
            state.retired.end(),
            [&](const native_retired_t &e) { return e.weight == weight; }));
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
