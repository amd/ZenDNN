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
    /// The image starts at the weight, its rewrite is done and the lines a
    /// call it serves checks hold within the caller's buffer: layout is the
    /// layout it holds.
    bool ready = false;
    native_raw_weight_t layout;
    /// The caller's buffer at the weight that the check read: the image's
    /// ends outside it were not checked.
    size_t checked_bytes = 0;
    /// The record the check found, when it found one that holds: the start
    /// and bytes of its image, whether it is a tombstone (a clear forgot the
    /// image, or a call found other bytes in another part of it) and whether
    /// the weight lies past its start.
    const void *record = nullptr;
    size_t record_bytes = 0;
    bool tomb = false;
    bool interior = false;
    /// The weight starts below the record, which starts in the caller's
    /// buffer.
    bool after = false;
    /// For a weight past the start of an image: none of the image's checked
    /// bytes lie in the caller's buffer, so the weight cannot be told from
    /// the image.
    bool holds_no_checked_byte = false;
    /// For a weight past the start of an image, found with layouts: the
    /// cached layout of layout's request at the weight that the calling
    /// thread was served since the cache last changed, whose digests hold;
    /// the image was not checked, as that layout serves such a request.
    const void *memo_layout = nullptr;
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
///      stays allocated, and counts toward the capacity, until every thread
///      listed when it was dropped has exited or begun a check of its memo
///      or a counted lookup since, on any weight (a thread whose later calls
///      are all packed for the call, or served an image by their own check,
///      counts none; a call on another thread may still read it until then)
///      and a later call drops a layout or caches one it packed (or finds
///      the cache full), or until a clear: a lookup served a cached layout
///      or an image frees none.  A call obtains its layout
///      once, and looks up or checks nothing of this cache after it.  New
///      bytes equal to the old ones at both ends are served the old layout
///      until clear_native_weight_caches drops it.  Reaching the capacity
///      logs one warning line per process;
///   2: as 1, except that a bf16, f16 or s4 weight whose layout fits its
///      buffer (contiguous ldb, wei_inplace_fits with capacity_bytes; an f16
///      layout is the raw weight's size, an s4 layout when K is a multiple
///      of 8) is rewritten in
///      place and recorded, and so is an aocl_blocked weight with a
///      contiguous ldb whose layout fits capacity_bytes (0: the size of its
///      AOCL prepack).  Other s8 weights, f32 weights and weights of N not a
///      multiple of pack_nr (whose layout holds zero columns past N) are
///      never rewritten.  The cached layouts packed from bytes a rewrite
///      overwrites, that still matched them, stay cached and name its
///      record: the image holds the same weight.  While a request's weight
///      lies in a record (at its start or inside it), only a cached layout
///      packed from that image or naming that record serves it.  A rewrite
///      that finds no memory for the checks its record keeps of the image
///      writes nothing: the bf16, f16 or s4 weight is then cached out of
///      place, and the aocl_blocked one is unimplemented.  Nor does a
///      rewrite whose image would equal the weight's own bytes at every byte
///      a reload of the weight writes (a non-transposed f16 weight of N and
///      ldb 32, an all-zero or constant weight): reading the buffer as the
///      weight is right, and a record could not tell the image from the
///      weight reloaded, so the layout is cached out of place.  Nor does a
///      rewrite whose record would hold too little to tell the image from
///      those bytes: no more than half of the lines it keeps where the image
///      differs from them hold a word that tells the two apart as the checks
///      below read it (a line that repeats one 8-byte word tells nothing,
///      nor does a zero word, as in a bf16 weight of a row of ones and a
///      row of twos), so the layout is cached out of place too.  A rewrite
///      that writes nothing leaves the records of other images over the
///      bytes it would have written as they were.
/// checked is the call's own check of the buffer for an image
/// (native_weight_find_image), when it made one: an image ready there in w's
/// layout is served without a second check when that check read at least
/// the bytes this lookup checks.
/// A buffer rewritten in place is never read as a raw or AOCL weight again
/// while it holds the image: the same layout is served from it, another
/// layout is packed from the raw weight rebuilt out of a bf16 or f16 image
/// (an s8 or s4 image serves its own layout only) when the caller's buffer
/// covers the whole image, and a request that starts inside it (unless a
/// cached layout naming its record serves it), reads past it, reads into it
/// from below or cannot cover it for a rebuild fails.  A
/// record keeps every byte of an image of up to 64 KiB, and of a larger one
/// the bytes of its two 64-byte ends and of up to 1024 64-byte sample lines
/// evenly spread between them; which of their bytes differ from the bytes
/// the image replaced among those a reload writes; and the digests
/// of the first and the last 64-byte line where the image differs from the
/// bytes it replaced among those a reload of the weight writes (an image's
/// own ends may hold the weight's bytes: a non-transposed f16 weight's, an
/// AOCL prepack's, native_weight_ends.hpp).  A request it may serve (at its
/// start, of its layout) checks those two lines; any other request whose
/// bytes it overlaps, and one at its start whose buffer does not hold both
/// lines, checks the kept bytes in its buffer word by word, however few
/// (but for sample lines that repeat one word or hold no such word, as
/// zeros there or the weight reloaded match them), and the two lines by
/// their digests: a kept word, or the part of one in its buffer, that
/// holds the image and has a byte that differs from the byte it replaced
/// tells the image from its weight, unless its bytes are zero or its line
/// repeats one word.  A record whose checked bytes in a
/// request's buffer no longer hold the image, none of them telling it from
/// its weight (the buffer was overwritten, or freed and reused), is
/// forgotten over that buffer, which is read as the raw weight it now
/// holds; the rest of the record stays as a tombstone, as
/// its bytes were not checked (a view's own bytes written back leave the
/// rest of the image recorded), but for a rest whose every byte the record
/// keeps (of a larger image, a rest within its kept lines) and that holds
/// no byte where the image differs from the bytes it replaced, which
/// reading as a raw weight cannot get wrong and a reload could not change;
/// a buffer that holds none of its bytes forgets none.  A request whose
/// checked bytes tell the image among other bytes (part of the raw weight
/// written back: some rows, some columns, between which its rows lie or
/// not) forgets nothing and fails, and the record becomes a tombstone.
/// Equal bytes at every checked place are taken for the image.  A clear
/// turns each record into a tombstone (clear_native_weight_caches): a
/// request whose checked bytes still hold it, all or among other bytes,
/// fails, and one whose checked bytes all differ forgets it over its buffer.
/// These checks read only the caller's buffer, the bytes the request reads
/// or capacity_bytes when larger, as memory outside it may be unmapped: an
/// end or a sample line outside it, of an image or of the bytes a cached
/// layout was packed from, is not checked; of a record's bytes, only those
/// the request reads count, but for the two lines of an image at its start.
/// A request whose buffer holds
/// none of a record's checked bytes (between two sample lines of an image
/// over 64 KiB) cannot be told from the image and fails, until a clear
/// finds the record a tombstone already: its buffer is then taken for
/// freed memory, where such a request is read as the raw weight it holds.
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
/// 54 KB, made when the thread stores its first entry (a cached layout or
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
/// rewriting), or whose image a clear forgot (a tombstone), and that still
/// holds the image, at its start or inside it, or when such a record starts
/// in the caller's buffer, which reads into it from below.  owned_bytes is
/// the caller's buffer at weight, of which the request reads the first
/// read_bytes: only the image's bytes that lie in those are checked (its
/// ends, and for a tombstone or a weight inside it the other bytes of an
/// image up to 64 KiB or the sample lines of a larger one), but for the two
/// lines a call the image may serve checks, which may lie anywhere in the
/// buffer; an image with none of them in it counts as held, a tombstone
/// not.  No lock unless
/// weight lies in a record: one atomic load while no record exists, else a
/// search of the spans of the records that the calling thread copied when
/// they last changed; for a weight inside one, the checks of the image,
/// after the thread's memo or else the shared cache lock and a map lookup.
bool native_weight_packed_in_place(const void *weight, size_t owned_bytes,
        size_t read_bytes = static_cast<size_t>(-1));

/// native_weight_packed_in_place, which also tells found what it found.
/// With layouts, a weight past the start of an image for which the calling
/// thread's memo holds a cached layout made since the cache last changed,
/// whose digests hold, takes no other check (memo_layout): as for every
/// cached layout, its digests are what a later call is served on, and the
/// layout serves only the request it was made for, which the caller
/// compares with its own.
bool native_weight_find_image(const void *weight, size_t owned_bytes,
        native_image_check_t &found, bool layouts = false,
        size_t read_bytes = static_cast<size_t>(-1));

/// True when get_native_layout at weight cache mode 1 or 2 serves w, whose
/// weight lies inside an in-place image, a cached layout: one packed from
/// w's raw bytes that still matched them when the rewrite that wrote the
/// image there (or the rewrite running now) overwrote them, so that the
/// rewrite carried it, and whose digests, taken from the image, still hold.
/// Nothing else serves such a request: the image holds no raw weight for
/// it.  A layout carried across an earlier image of the buffer does not
/// serve, as the bytes of a buffer rewritten again for another weight can
/// match its digests.  False for a w native_layout_fits does not take, which
/// no layout serves.  capacity_bytes as for get_native_layout.
bool native_weight_retained_layout(const native_raw_weight_t &w,
        int32_t weight_cache_mode, size_t capacity_bytes);

/// What a call whose weight is not read as a raw weight, as bytes it reads
/// still hold an in-place record (found, as a check found it), is told to
/// do; every error line that refuses such a call names it with the record's
/// start and bytes: a tombstone asks for the raw weight of the image's
/// whole buffer; a live image for it and a clear, or says that a weight
/// that holds none of its checked bytes cannot be told from it.
const char *native_image_refusal(const native_image_check_t &found);

/// What native_weight_raw_hold_t::hold did.
enum class native_raw_read_t : uint8_t {
    done, ///< the raw weight is held
    image, ///< the buffer holds an in-place image: nothing is held
    failed, ///< no memory to hold the buffer: nothing is held
};

/// Holds the extent bytes of the raw weight at weight for a reader outside
/// this cache, as the packs of get_native_layout hold theirs: from a hold
/// that returns done until the holder is destroyed, no in-place rewrite of
/// bytes among them starts or runs, and other reads of them go on.  hold
/// returns image, holding nothing, when the buffer holds an image by then
/// (at weight, or starting inside the extent bytes), and failed when there
/// is no memory to hold it, which found, when given, then describes (as
/// native_weight_find_image would); owned_bytes is the caller's buffer at
/// weight, as for native_weight_find_image.  One hold per holder; while it
/// holds, its
/// thread waits for no rewrite of those bytes (as a native call on the
/// buffer may).
class native_weight_raw_hold_t {
public:
    native_weight_raw_hold_t() = default;
    ~native_weight_raw_hold_t();
    native_weight_raw_hold_t(const native_weight_raw_hold_t &) = delete;
    native_weight_raw_hold_t &operator=(const native_weight_raw_hold_t &)
            = delete;

    native_raw_read_t hold(const void *weight, size_t extent,
            size_t owned_bytes, native_image_check_t *found = nullptr);

private:
    std::pair<std::uintptr_t, std::uintptr_t> span_ {0, 0};
    bool held_ = false;
};

/// Drops every cached layout, and turns every in-place record into a
/// tombstone, which reads no byte now: a buffer may still hold its image
/// (or be freed); a record that is a tombstone already is taken for freed
/// memory (get_native_layout).  A later request whose checked bytes still
/// hold the image fails until the buffer holds other bytes (the raw weight
/// reloaded);
/// tombstones stay through later clears, with the bytes and digests their
/// records keep (up to 64 KiB each, outside the cache capacity), until a
/// request whose bytes differ forgets them over its buffer.  Same quiescence
/// contract as clear_all_weight_caches, which calls it; an in-place rewrite
/// that a clear meets anyway keeps its record, which becomes a tombstone of
/// the image once the rewrite wrote it (the rewriting call is served), so
/// the calls after it fail rather than read the image as a raw weight.
void clear_native_weight_caches();

/// The layouts cached for the weight pointer weight, plus one when the
/// buffer at weight holds an image an in-place pack wrote: what a call on
/// weight that the native BRGEMM served at weight cache mode 1 or 2 left.
/// A diagnostic; no call reads it.
size_t native_weight_layouts(const void *weight);

/// The layouts dropped from entries of the weight pointer weight that the
/// cache still holds.  A diagnostic; no call reads it.
size_t native_retired_layouts(const void *weight);

/// 1 when a tombstone (the record of an image a clear forgot, or of the part
/// of one outside a buffer a call found other bytes in) starts at weight,
/// else 0.  A diagnostic; no call reads it.
size_t native_weight_tombstones(const void *weight);

/// The calling thread's memo entries for weight that can still serve (made
/// in the current cache generation); 0 when the thread has no memo, which
/// asking does not make.  A diagnostic; no call reads it.
size_t native_weight_memo_entries(const void *weight);

/// The next n drops of a cached layout whose weight changed fail as when no
/// memory is left: the call that finds the layout stale fails and the
/// entry keeps it.  For tests.
void native_fail_next_retirements(unsigned n);

/// The next n in-place rewrites find no memory for the checks their record
/// keeps of the image (its bytes and digests), so their buffers are not
/// written.  For tests.
void native_fail_next_image_checks(unsigned n);

/// While fail is set, a thread that has no reader yet gets none, as when no
/// memory is left to list one: the layouts it obtains go uncounted, and no
/// dropped layout is freed until a clear.  For tests.
void native_fail_reader_listings(bool fail);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_CACHE_HPP
