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

#ifndef MATMUL_NATIVE_KERNEL_CACHE_HPP
#define MATMUL_NATIVE_KERNEL_CACHE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <utility>
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_ends.hpp"
#include <unordered_map>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Column width of the panels the BKC GEMV kernels compute.  The INT8
/// kernel loads a whole panel for a block narrower than that, so an INT8
/// BKC pack is allocated with NR_PACK * INT8_VNNI_GRP guard bytes past its
/// end (bkc_guard).
inline constexpr int NR_PACK = 64;

/// Packing granularity for BKC (Blocked K-Contiguous) GEMV format.
/// BKC pads each block's column count to the next multiple of BKC_NR_PAD
/// instead of NR_PACK, reducing zero-padding waste for non-64-aligned N.
/// The GEMV kernel's 16-wide tail handles any remainder via masking.
/// Must be a power of 2 and divide NR_PACK (so full-panel dispatch is safe).
inline constexpr int BKC_NR_PAD = 16;

/// Cache key for packed weights — independent of microkernel NR.
///
/// Uniqueness: equality and hashing use \c weight_ptr, \c K, \c N, \c ldb,
/// and \c transB. Distinct tensors or layouts produce distinct keys; there is
/// no separate "format id" — if any of these differ, the entry is different.
///
/// IMPORTANT: \c weight_ptr is identity for the tensor. Callers must keep the
/// address stable for the lifetime of cached entries when weights are const;
/// otherwise disable weight caching or clear caches on buffer reuse.
struct PrepackedWeightKey {
    const void *weight_ptr;
    int K, N, ldb;
    bool transB;

    bool operator==(const PrepackedWeightKey &o) const {
        return weight_ptr == o.weight_ptr && K == o.K && N == o.N
                && ldb == o.ldb && transB == o.transB;
    }
};

struct PrepackedWeightKeyHash {
    size_t operator()(const PrepackedWeightKey &k) const {
        size_t h = std::hash<const void *>()(k.weight_ptr);
        auto mix = [](size_t &seed, size_t val) {
            seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        mix(h, static_cast<size_t>(static_cast<uint32_t>(k.K)));
        mix(h, static_cast<size_t>(static_cast<uint32_t>(k.N)));
        mix(h, static_cast<size_t>(static_cast<uint32_t>(k.ldb)));
        mix(h, k.transB ? size_t(1) : size_t(0));
        return h;
    }
};

// ============================================================================
// LRU map for prepacked-weight caches (move-only values).
// ============================================================================

/// Capacity (in entries) for every native weight cache.
/// Sourced from matmul_config_t (env: ZENDNNL_LRU_CACHE_CAPACITY).
/// Default = UINT32_MAX → unbounded growth, identical to behavior
/// before this knob existed.  When set to a finite value the cache
/// enforces the cap by REFUSING to insert new entries, or to replace
/// the value of one, once its entries and the values they replaced
/// (kept until the cache is cleared) reach it (not by evicting
/// existing ones — see `LRUWeightMap::insert()` for the UAF-safety
/// rationale).  Callers that hit a refused
/// insert transparently fall back to their thread-local repack
/// path, so correctness is unaffected; the only cost is the
/// amortised pack hit on every call for the (N+1)th unique weight.
/// Defined in kernel_cache.cpp to avoid including common/op_config.hpp
/// from this header.
uint32_t get_weight_cache_capacity();

/// Generation counter incremented on every cache clear or in-place
/// value replacement.  Defined in kernel_cache.cpp.  Declared early
/// so LRUWeightMap can call it.  Thread-local fast-paths that cache
/// a raw pointer should compare their captured generation against
/// this counter on each call and fall back to `find_and_touch()` on
/// mismatch.  This protects only ACROSS calls — the in-flight call
/// relies on `LRUWeightMap` freeing no value before `clear()`: it
/// never evicts and keeps the values it replaces.
std::atomic<uint64_t> &weight_cache_generation();

/// Move-aware LRU map for weight caches. Caller owns synchronization
/// (the enclosing cache class holds its own std::mutex).
///
/// Capacity policy: EVICTION IS DISABLED.  When a caller goes
/// through `get_or_pack()`, the returned raw
/// pointer is consumed by the GEMV looper AFTER the cache
/// mutex is released — a concurrent `insert()` on another thread
/// that evicted the same entry would call `std::free()` on the
/// packed buffer while the first thread is still reading it, a
/// use-after-free.  The `weight_cache_generation()` counter only
/// guards the thread-local pointer-reuse fast-path across calls;
/// it does NOT keep an in-use entry alive during the current call.
///
/// For the same reason a value replaced under its key is not freed
/// either: it stays in `retired_` until `clear()`.
///
/// Capacity is instead enforced by REFUSING new inserts and
/// replacements once `list_.size() + retired_.size() >=
/// get_weight_cache_capacity()` — `insert()`
/// returns `nullptr` and the caller's `std::unique_ptr<V>` falls
/// out of scope, freeing the just-packed buffer immediately.  The
/// caller of `get_or_pack()` already handles
/// a `nullptr` return by falling back to its thread-local repack
/// path (see `bkc_resolve_packed_B_st` fallback block for an
/// example), so correctness is preserved; the only cost is paying
/// the pack on every call for weights beyond the cap.  This gives
/// the `ZENDNNL_LRU_CACHE_CAPACITY` env knob back its advertised
/// "bound my resident set" semantic without introducing a UAF.
///
/// Steady-state RSS bound = `cap × max(per-entry pack size)`.  For
/// deployments that rotate weights (LoRA hot-swap, recompile
/// cycles, retraining loops) call `clear_all_weight_caches()`
/// during a quiescent window to release everything at once.
///
/// All operations are O(1) amortized.
template <typename V>
class LRUWeightMap {
public:
    using ListItem = std::pair<PrepackedWeightKey, std::unique_ptr<V>>;
    using ListIter = typename std::list<ListItem>::iterator;

    /// Lookup. On hit, moves the entry to MRU and returns the raw pointer.
    /// Returns nullptr on miss. Caller must hold the cache's mutex.
    V *find_and_touch(const PrepackedWeightKey &k) {
        auto it = map_.find(k);
        if (it == map_.end()) return nullptr;
        list_.splice(list_.begin(), list_, it->second);
        return it->second->second.get();
    }

    /// Insert at MRU and return the raw pointer to the just-inserted
    /// value, or nullptr at capacity (below), with v freed.  Caller must
    /// hold the cache's mutex.
    ///
    /// Re-insert (key already present) gives the key the new value at
    /// MRU and does NOT grow the map — preventing orphan list nodes that
    /// would break LRU bookkeeping.  The value it replaces is NOT freed:
    /// a call on another thread may still read it through the pointer
    /// it got before the replacement (outside the mutex), and the
    /// thread-local fast paths (BF16 / INT8 paths, see callers) may
    /// still hold it.  Its node moves to `retired_`, which only
    /// `clear()` frees, and `weight_cache_generation()` is bumped so
    /// the fast paths look the key up again on their next call.
    V *insert(const PrepackedWeightKey &k, std::unique_ptr<V> v) {
        // ── Capacity enforcement (REFUSE-NEW, not evict-old) ──────────
        // Eviction of an existing entry is unsafe because GEMV
        // loopers hold the raw pointer returned by `find_and_touch()`
        // across the entire kernel call AFTER releasing the cache
        // mutex.  A concurrent `std::free` on that pointer would UAF;
        // `weight_cache_generation()` protects only the next-call TL
        // fast path, not the in-flight call.
        //
        // Instead, when the map is at capacity (its entries and the
        // values they replaced) we refuse the new insert or
        // replacement: `v` drops out of scope, the just-packed buffer is
        // freed by the caller's `unique_ptr`, and we return `nullptr`.
        // The caller of `get_or_pack()` already
        // handles `nullptr` as a pack miss and falls back to its
        // thread-local repack path (see `bkc_resolve_packed_B_st` for
        // the canonical fallback pattern) — correctness is preserved,
        // the cost is paying a pack on every call for weights beyond
        // the cap.  Cap = UINT32_MAX (default) means "never refuse",
        // restoring the historical unbounded-growth behaviour.
        const uint32_t cap = get_weight_cache_capacity();
        if (cap != std::numeric_limits<uint32_t>::max()
                && list_.size() + retired_.size() >= cap) {
            return nullptr; // cap hit: unique_ptr destructor frees the pack
        }

        auto existing = map_.find(k);
        list_.emplace_front(k, std::move(v));
        if (existing != map_.end()) {
            retired_.splice(retired_.end(), list_, existing->second);
            existing->second = list_.begin();
            weight_cache_generation().fetch_add(1, std::memory_order_relaxed);
        } else {
            map_[k] = list_.begin();
        }
        return list_.front().second.get();
    }

    /// Drop every entry and every value an entry replaced.  Bumps
    /// weight_cache_generation() when the map was non-empty so any
    /// thread-local fast path that holds a raw entry pointer guarded by
    /// a generation snapshot detects the staleness on its next call
    /// (otherwise the per-cache `clear()` wrappers exposed publicly
    /// would silently invalidate pointers without notifying the TL
    /// paths).
    void clear() {
        const bool had_entries = !list_.empty() || !retired_.empty();
        map_.clear();
        list_.clear();
        retired_.clear();
        if (had_entries) {
            weight_cache_generation().fetch_add(1, std::memory_order_relaxed);
        }
    }

    size_t size() const { return list_.size(); }

private:
    std::list<ListItem> list_; // front = MRU, back = LRU
    /// The values entries replaced, freed only by clear().
    std::list<ListItem> retired_;
    std::unordered_map<PrepackedWeightKey, ListIter, PrepackedWeightKeyHash>
            map_;
};

// ============================================================================
// BF16 Blocked K-contiguous (BKC) VNNI packed weight cache (for BKC GEMV kernel)
// ============================================================================

/// VNNI pair constant: 2 BF16 values per element position per k-pair.
inline constexpr int VNNI_PAIR = 2;

/// Custom deleter for uint16_t aligned_alloc'd memory.
struct AlignedFreeU16Deleter {
    void operator()(uint16_t *p) const { zendnnl_aligned_free(p); }
};

/// BF16 Blocked K-contiguous (BKC) VNNI packed weight buffer.
///
/// Blocked K-contiguous (BKC) packing: B is partitioned into 256-column
/// blocks, each packed with K-contiguous VNNI layout. Within each block,
/// all k-pairs are contiguous with stride = blk_N_padded × VNNI_PAIR.
///
/// Layout: for each k-pair, ALL N columns (padded to BKC_NR_PAD=16) are contiguous:
///   packed[kp * N_padded * 2 + n * 2 + 0] = B[2*kp  ][n]
///   packed[kp * N_padded * 2 + n * 2 + 1] = B[2*kp+1][n]
///
/// N is padded to BKC_NR_PAD (16) boundary with zeros.
/// K is padded to even for VNNI alignment.
///
/// This layout provides sequential memory access when iterating K-outer,
/// N-inner — optimal for M=1 GEMV with L2-resident B.
struct BF16BKCWeight {
    std::unique_ptr<uint16_t[], AlignedFreeU16Deleter> buf;
    const uint16_t *data; ///< Points to buf.get()
    int K; ///< Original K
    int K_padded; ///< K rounded up to even
    int N; ///< Original N
    int N_padded; ///< N rounded up to BKC_NR_PAD (16)
    size_t total; ///< Total buffer size in uint16_t elements
    /// The words of the two ends of the raw weight bytes the pack read.
    weight_end_words_t source_ends;

    /// N-stride in uint16_t units for one k-pair row.
    int n_stride() const { return N_padded * VNNI_PAIR; }
};

/// Thread-safe singleton cache for BF16 BKC packed weight matrices, keyed
/// by PrepackedWeightKey.  An entry serves only while the weight's raw
/// bytes hold the words of their two ends that its pack read
/// (native_weight_ends.hpp): a buffer that holds a new weight is packed
/// again, as the native layouts are (get_native_layout).  A pack it
/// returned stays allocated until clear(), also once a pack of new bytes
/// replaces it under its key: calls read it outside the mutex.
class BF16BKCWeightCache {
public:
    static BF16BKCWeightCache &instance();
    /// The cached pack of key, else nullptr, whatever bytes the raw weight
    /// holds now (a pack serves while they hold its source_ends); reads no
    /// byte of the weight and packs nothing.
    const BF16BKCWeight *find(const PrepackedWeightKey &key);
    /// The cached pack of key that weight, the raw weight (bytes of it),
    /// still serves, else a pack of weight, cached in place of a pack of
    /// other bytes; nullptr when it cannot be cached.
    const BF16BKCWeight *get_or_pack(const PrepackedWeightKey &key,
            const uint16_t *weight, size_t bytes);
    /// Clear all cached entries. Must only be called when no thread is using
    /// any previously returned pointer (e.g., between test cases or after a
    /// global barrier). Violating this causes use-after-free.
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
    BF16BKCWeightCache(const BF16BKCWeightCache &) = delete;
    BF16BKCWeightCache &operator=(const BF16BKCWeightCache &) = delete;

private:
    BF16BKCWeightCache() = default;
    LRUWeightMap<BF16BKCWeight> cache_;
    std::mutex mutex_;
};

// ── INT8 K-contiguous weight cache ──────────────────────────────────────

/// Custom deleter for aligned_alloc'd INT8 memory.
struct AlignedFreeS8Deleter {
    void operator()(int8_t *p) const { zendnnl_aligned_free(p); }
};
struct AlignedFreeI32Deleter {
    void operator()(int32_t *p) const { zendnnl_aligned_free(p); }
};

/// INT8 VNNI group size for vpdpbusd.
inline constexpr int INT8_VNNI_GRP = 4;

/// INT8 K-contiguous VNNI packed weight buffer and the column sums of its
/// zero-point compensation: what the weight alone determines.
///
/// Layout: for each k-quad, ALL N columns (padded to BKC_NR_PAD=16)
/// are contiguous as 4-byte groups (matching vpdpbusd operand layout).
struct INT8KContiguousWeight {
    std::unique_ptr<int8_t[], AlignedFreeS8Deleter> packed_buf;
    std::unique_ptr<int32_t[], AlignedFreeI32Deleter> col_sum_buf;

    const int8_t *data;
    const int32_t *col_sum;

    int K, K_padded, N, N_padded;
    size_t total;
    /// The words of the two ends of the raw weight bytes the pack read.
    weight_end_words_t source_ends;

    int n_stride() const { return N_padded * INT8_VNNI_GRP; }
};

/// Thread-safe singleton cache for INT8 K-contiguous packed weights.
/// Caches packed B + col_sum, keyed by (weight_ptr, K, N, ldb, transB); an
/// entry serves only while the weight's raw bytes hold the words of their
/// two ends that its pack read, and stays allocated until clear(), as in
/// BF16BKCWeightCache.
/// The dequantization of a call (its source scale and zero point, weight
/// scales and bias) is not cached: calls on one weight may carry other
/// values, also in the same buffers.
class INT8KContiguousWeightCache {
public:
    static INT8KContiguousWeightCache &instance();

    /// Get or pack INT8 weights with their column sums: the cached entry of
    /// key that weight, the raw weight (bytes of it), still serves, else a
    /// pack of weight, cached in place of a pack of other bytes; nullptr
    /// when it cannot be cached.
    const INT8KContiguousWeight *get_or_pack(
            const PrepackedWeightKey &key, const int8_t *weight, size_t bytes);

    /// @copydoc BF16BKCWeightCache::clear()
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
    INT8KContiguousWeightCache(const INT8KContiguousWeightCache &) = delete;
    INT8KContiguousWeightCache &operator=(const INT8KContiguousWeightCache &)
            = delete;

private:
    INT8KContiguousWeightCache() = default;
    LRUWeightMap<INT8KContiguousWeight> cache_;
    std::mutex mutex_;
};

/// Clear all weight caches. Must only be called when no thread is using
/// any previously returned pointer (e.g., between test cases or model swap).
/// Defined in kernel_cache.cpp.
void clear_all_weight_caches();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_KERNEL_CACHE_HPP
