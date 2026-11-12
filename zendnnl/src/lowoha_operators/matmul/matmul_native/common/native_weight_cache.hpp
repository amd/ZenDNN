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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_CACHE_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_CACHE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// Bytes from the first element of a raw [K, N] weight (or [N, K] when
/// transB), rows ldb elements apart, to past its last one (to past the byte
/// that holds it, for an s4 weight of two elements per byte).
size_t native_raw_weight_bytes(
        data_type_t wei_dt, int K, int N, int ldb, bool transB);

/// A raw weight to read in the native VNNI layout: [K, N] row-major ldb
/// apart, or [N, K] when transB.
struct native_raw_weight_t {
    data_type_t wei_dt = data_type_t::none;
    const void *weight = nullptr;
    int K = 0;
    int N = 0;
    int ldb = 0;
    bool transB = false;
    int pack_nr = 0;
    /// weight holds the raw weight as an aocl_dlp_blocked prepack (bf16, or
    /// s8 packed for a u8 source), which the AOCL inverse rebuilds with ldb.
    bool aocl_blocked = false;
};

/// Where the layout of a raw weight lives.  owned is set when the pack
/// belongs to the caller, who frees it with native_free_layout.
struct native_layout_t {
    const void *data = nullptr;
    void *owned = nullptr;
};

/// What a call's check of its weight buffer for an in-place image found
/// (native_weight_find_image), handed on so that its layout lookup does not
/// check again.
struct native_image_check_t {
    /// The image starts at the weight, its rewrite is done and its ends hold
    /// within the caller's buffer: layout is the layout it holds.
    bool ready = false;
    native_raw_weight_t layout;
    /// The caller's buffer at the weight that the check read: the image's
    /// ends outside it were not checked.
    size_t checked_bytes = 0;
};

/// Resolves w, a weight native_layout_fits takes, to its layout under weight
/// cache mode 0, 1 or 2:
///   0: packed for this call (owned);
///   1: the process cache, keyed by the weight pointer and the whole layout
///      request; packed for this call when ZENDNNL_LRU_CACHE_CAPACITY
///      layouts are already held (entries are never evicted: the caller
///      may hold a layout of any of them).  An entry keeps digests of the
///      two ends of the bytes its pack read (native_weight_ends.hpp); a hit
///      whose buffer no longer matches them drops the entry and packs again
///      from the bytes the buffer holds now.  The dropped entry's layout
///      stays allocated, and held, until clear_native_weight_caches: a call
///      on another thread may still read it.  New bytes equal to the old
///      ones at both ends are served the old layout until
///      clear_native_weight_caches drops it;
///   2: as 1, except that a bf16, f16 or s4 weight whose layout fits its
///      buffer (contiguous ldb, wei_inplace_fits with capacity_bytes; an f16
///      layout is the raw weight's size, an s4 layout when K is a multiple
///      of 8) is rewritten in
///      place and recorded, and so is an aocl_blocked weight with a
///      contiguous ldb whose layout fits capacity_bytes (0: the size of its
///      AOCL prepack).  Other s8 weights, f32 weights and weights of N not a
///      multiple of pack_nr (whose layout holds zero columns past N) are
///      never rewritten.  The cached layouts packed from bytes a rewrite
///      overwrites stay cached: the image holds the same weight.
/// checked is the call's own check of the buffer for an image
/// (native_weight_find_image), when it made one: an image ready there in w's
/// layout is served without a second check when that check read at least
/// the bytes this lookup checks.
/// A buffer rewritten in place is never read as a raw or AOCL weight again
/// while it holds the image: the same layout is served from it, another
/// layout is packed from the raw weight rebuilt out of a bf16 or f16 image
/// (an s8 or s4 image serves its own layout only) when the caller's buffer
/// covers the whole image, and a request that starts inside it, reads past
/// it, reads into it from below or cannot cover it for a rebuild fails.  A
/// record whose buffer no longer holds the image (digests of its two ends,
/// native_weight_ends.hpp: the buffer was overwritten, or freed and reused)
/// is forgotten, and the buffer is read as the raw weight it now holds.
/// These checks read only the caller's buffer, the bytes the request reads
/// or capacity_bytes when larger, as memory outside it may be unmapped: an
/// end outside it, of an image or of the bytes a cached layout was packed
/// from, is not checked.
/// An allocation or AOCL unreorder failure before anything was written is
/// unimplemented.
/// Thread safe; a pack of a missing layout uses up to num_threads threads.
/// Lookups share the cache lock; no pack or rewrite runs under it.  At modes
/// 1 and 2 the packs of one buffer, and its in-place rewrite, run one at a
/// time (a call on that buffer waits, then finds the layout made), while
/// lookups and packs of other buffers go on; the record of a rewrite exists
/// before the buffer is written, so a call that meets the buffer meanwhile
/// sees an image.  A pack at any mode holds the bytes it reads, which the
/// exclusive lookup of its call records, and a rewrite writes none of them
/// until no pack holds them: a call at mode 0 too takes the cache lock
/// exclusively to pack, its packs of one buffer still running side by side.
/// A lookup of a cached layout whose bytes a rewrite overwrites is served it
/// without checking them while the rewrite runs when the rewrite marked it,
/// and otherwise waits for the rewrite to end.
/// A call whose weight lies in no image takes no lock for images while they
/// have not changed since its thread last looked.
/// Each calling thread keeps a memo of the cached layouts and images its
/// calls were served, at most 512 (4 per set of 128 by weight pointer), about
/// 50 KB, made when the thread stores its first entry (a cached layout or
/// image a call was served, or an image its in-place check found) and freed
/// when the thread exits: a later call of the thread on the same request is
/// served from it, after the same digest check of the ends and without the
/// lock or the lookup, as long as no cached layout or image record has gone
/// and no image has been recorded since (a clear, a layout or image found
/// stale, a rewrite in place).  A call served from the memo does not wait
/// for a pack of its buffer.  It reads the digests only while no rewrite in
/// place runs, and a rewrite waits for the memo checks under way before it
/// writes, holding no lock; while a rewrite runs, the calls of every thread
/// take the lookup.
status_t get_native_layout(const native_raw_weight_t &w,
        int32_t weight_cache_mode, size_t capacity_bytes, int num_threads,
        native_layout_t &out, const native_image_check_t *checked = nullptr);

/// Frees native_layout_t::owned; safe with nullptr.
void native_free_layout(void *owned);

/// True when weight points into a buffer an in-place pack rewrote (or is
/// rewriting) and that still holds the image, at its start or inside it.
/// owned_bytes is the caller's buffer at weight: only the ends of the image
/// that lie in it are checked, and an image with neither end in it counts
/// as held.  No lock unless weight lies in an image: one atomic load while
/// no buffer holds one, else a search of the spans of the images that the
/// calling thread copied when they last changed; for a weight inside one,
/// the digests of the image, after the thread's memo or else the shared
/// cache lock and a map lookup.
bool native_weight_packed_in_place(const void *weight, size_t owned_bytes);

/// native_weight_packed_in_place, which also tells found what it found.
bool native_weight_find_image(
        const void *weight, size_t owned_bytes, native_image_check_t &found);

/// What native_weight_read_raw did.
enum class native_raw_read_t : uint8_t {
    done, ///< read ran on the raw weight
    image, ///< the buffer holds an in-place image: read did not run
    failed, ///< no memory to hold the buffer: read did not run
};

/// Runs read, a pack outside this cache that reads the extent bytes of the
/// raw weight at weight, with those bytes held as the packs of
/// get_native_layout hold theirs: no in-place rewrite of bytes among them
/// starts or runs while read runs, and other reads of them go on.  A buffer
/// that holds an image by then (at weight, or starting inside the extent
/// bytes) has no raw weight left, and read does not run.  owned_bytes is the
/// caller's buffer at weight, as for native_weight_find_image.
native_raw_read_t native_weight_read_raw(const void *weight, size_t extent,
        size_t owned_bytes, const std::function<void()> &read);

/// Holds the extent bytes of the raw weight at weight for a reader outside
/// this cache whose read is not one function: from a hold that returns done
/// until the holder is destroyed, as native_weight_read_raw holds them while
/// its read runs.  hold returns image, holding nothing, when the buffer holds
/// an image by then, and failed when there is no memory to hold it.  One
/// hold per holder; while it holds, its thread waits for no rewrite of those
/// bytes (as a native call on the buffer may).
class native_weight_raw_hold_t {
public:
    native_weight_raw_hold_t() = default;
    ~native_weight_raw_hold_t();
    native_weight_raw_hold_t(const native_weight_raw_hold_t &) = delete;
    native_weight_raw_hold_t &operator=(const native_weight_raw_hold_t &)
            = delete;

    native_raw_read_t hold(
            const void *weight, size_t extent, size_t owned_bytes);

private:
    std::pair<std::uintptr_t, std::uintptr_t> span_ {0, 0};
    bool held_ = false;
};

/// A short read of weight bytes outside this cache that holds none of them
/// (native_weight_read_raw holds what it reads), on the calling thread: it
/// may read them only while the check lives and ok().  ok() is false while
/// an in-place rewrite runs, and, with bytes, when an in-place image may
/// overlap the bytes at weight or the images changed as the check began; a
/// rewrite that starts meanwhile writes only once the check has ended.  It
/// may take the cache lock shared before it begins (with bytes, when the
/// images changed since the thread last looked), and none while it lives;
/// what the caller does inside it takes no lock either, as a rewrite waits
/// for it.  A thread runs one at a time, and calls nothing of this cache
/// inside it.
class native_weight_check_t {
public:
    explicit native_weight_check_t(
            const void *weight = nullptr, size_t bytes = 0);
    ~native_weight_check_t() {
        if (reading_ != nullptr)
            reading_->store(reading_->load(std::memory_order_relaxed) + 1,
                    std::memory_order_release);
    }
    native_weight_check_t(const native_weight_check_t &) = delete;
    native_weight_check_t &operator=(const native_weight_check_t &) = delete;

    bool ok() const { return ok_; }

private:
    std::atomic<uint64_t> *reading_ = nullptr;
    bool ok_ = false;
};

/// Drops every cached layout and every in-place record.  Same quiescence
/// contract as clear_all_weight_caches, which calls it; a buffer that still
/// holds an in-place image must not be passed again afterwards.
void clear_native_weight_caches();

/// The layouts cached for the weight pointer weight, plus one when the
/// buffer at weight holds an image an in-place pack wrote: what a call on
/// weight that the native BRGEMM served at weight cache mode 1 or 2 left.
/// A diagnostic; no call reads it.
size_t native_weight_layouts(const void *weight);

/// The calling thread's memo entries for weight that can still serve (made
/// in the current cache generation); 0 when the thread has no memo, which
/// asking does not make.  A diagnostic; no call reads it.
size_t native_weight_memo_entries(const void *weight);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_CACHE_HPP
