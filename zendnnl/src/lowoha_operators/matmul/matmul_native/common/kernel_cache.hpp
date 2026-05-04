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
#include <mutex>
#include <memory>
#include <unordered_map>
#include <list>
#include <limits>
#include <utility>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Canonical panel width for B prepacking.
/// Must be >= max NR across all kernel variants (64).
/// All kernels work with this panel width regardless of their own NR.
/// A kernel with NR=48 loads 3 of 4 ZMMs per k-row; the unused 16 floats
/// are zero-padded and ignored.
inline constexpr int NR_PACK = 64;

/// Packing granularity for BKC (Blocked K-Contiguous) GEMV format.
/// BKC pads each block's column count to the next multiple of BKC_NR_PAD
/// instead of NR_PACK, reducing zero-padding waste for non-64-aligned N.
/// The GEMV kernel's 16-wide tail handles any remainder via masking.
/// Must be a power of 2 and divide NR_PACK (so full-panel dispatch is safe).
inline constexpr int BKC_NR_PAD = 16;

/// Custom deleter for aligned_alloc'd memory.
struct AlignedFreeDeleter {
  void operator()(float *p) const { std::free(p); }
};

/// Prepacked weight buffer: NR_PACK-wide K-contiguous panels.
///
/// Layout:
///   n_panels = ceil(N / NR_PACK)
///   Panel p covers columns [p*NR_PACK .. min((p+1)*NR_PACK, N)-1]
///   Within panel: packed[kk * NR_PACK + nr]
///   Total size: n_panels * K * NR_PACK floats
///
/// This is the optimal AVX-512 layout:
///   - b_stride = NR_PACK = 64 (256 bytes between k-rows)
///   - HW prefetcher handles 256-byte stride perfectly
///   - Micro-tile's B working set (KB × 64 × 4) fits in L2
///   - Any kernel NR (16, 32, 48, 64) reads a prefix of the panel
///   - Decoupled from MR/NR: cache key has no tile dependency
///
/// Indexing:
///   get_panel(pc, panel_idx) → buf + panel_idx * K * NR_PACK + pc * NR_PACK
///   b_stride = NR_PACK
struct PrepackedWeight {
  std::unique_ptr<float[], AlignedFreeDeleter> buf;
  const float *data;   ///< Points to buf.get() (always owns for panel format)
  int K;         ///< K dimension
  int N;         ///< Original N dimension
  int n_panels;    ///< ceil(N / NR_PACK)

  /// Get B panel pointer for K-offset pc, N-panel index.
  const float *get_panel(int pc, int panel_idx) const {
    return data + static_cast<size_t>(panel_idx) * K * NR_PACK + static_cast<size_t>(pc) * NR_PACK;
  }

  /// Row stride for microkernel (constant).
  static constexpr int stride() { return NR_PACK; }
};

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
    return weight_ptr == o.weight_ptr &&
         K == o.K && N == o.N && ldb == o.ldb &&
         transB == o.transB;
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
/// enforces the cap by REFUSING to insert new entries once full
/// (not by evicting existing ones — see `LRUWeightMap::insert()`
/// for the UAF-safety rationale).  Callers that hit a refused
/// insert transparently fall back to their thread-local repack
/// path, so correctness is unaffected; the only cost is the
/// amortised pack hit on every call for the (N+1)th unique weight.
/// Defined in kernel_cache.cpp to avoid including matmul_config.hpp
/// from this header.
uint32_t get_weight_cache_capacity();

/// Generation counter incremented on every cache clear or in-place
/// value replacement.  Defined in kernel_cache.cpp.  Declared early
/// so LRUWeightMap can call it.  Thread-local fast-paths that cache
/// a raw pointer should compare their captured generation against
/// this counter on each call and fall back to `find_and_touch()` on
/// mismatch.  This protects only ACROSS calls — the in-flight call
/// relies on the never-evict property documented in `LRUWeightMap`.
std::atomic<uint64_t> &weight_cache_generation();

/// Move-aware LRU map for weight caches. Caller owns synchronization
/// (the enclosing cache class holds its own std::mutex).
///
/// Capacity policy: EVICTION IS DISABLED.  When a caller goes
/// through `get_or_prepack()` / `get_or_pack()`, the returned raw
/// pointer is consumed by the GEMM/BRGEMM looper AFTER the cache
/// mutex is released — a concurrent `insert()` on another thread
/// that evicted the same entry would call `std::free()` on the
/// packed buffer while the first thread is still reading it, a
/// use-after-free.  The `weight_cache_generation()` counter only
/// guards the thread-local pointer-reuse fast-path across calls;
/// it does NOT keep an in-use entry alive during the current call.
///
/// Capacity is instead enforced by REFUSING new inserts once
/// `list_.size() >= get_weight_cache_capacity()` — `insert()`
/// returns `nullptr` and the caller's `std::unique_ptr<V>` falls
/// out of scope, freeing the just-packed buffer immediately.  The
/// caller of `get_or_prepack()` / `get_or_pack()` already handles
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

  /// Insert at MRU. If size() exceeds capacity, evicts LRU entries
  /// (and bumps weight_cache_generation()). Returns the raw pointer
  /// to the just-inserted value. Caller must hold the cache's mutex.
  ///
  /// Re-insert (key already present) replaces the existing value,
  /// splices its node to MRU, and does NOT grow the map — preventing
  /// orphan list nodes that would break LRU bookkeeping.  Callers
  /// typically check `find_and_touch()` first and only `insert()` on
  /// miss, but the in-place update path keeps the API safe under
  /// concurrent re-pack races (e.g. two threads racing on the same
  /// key after both saw a miss before the first one's insert
  /// completed).
  ///
  /// When a re-insert happens the old `std::unique_ptr<V>` is
  /// destructed as part of the `std::move` assignment — every other
  /// thread that has already cached the OLD raw pointer in its
  /// thread-local fast path (BF16 / INT8 paths, see callers) now
  /// holds a dangling reference.  We therefore bump
  /// `weight_cache_generation()` on this path too: the TL fast paths
  /// compare their cached generation against the global one before
  /// reusing their pointer and fall back to `find_and_touch()` on
  /// mismatch, which returns the new MRU pointer under the cache
  /// mutex.  Without this bump a concurrent re-pack race can leave
  /// the other thread UAFing the freed unique_ptr target.
  V *insert(const PrepackedWeightKey &k, std::unique_ptr<V> v) {
    auto existing = map_.find(k);
    if (existing != map_.end()) {
      // Re-insert of an already-cached key: replace value in-place,
      // splice node to MRU, bump generation.  Note this DOES free
      // the prior unique_ptr target at the `=` assignment; any
      // thread-local fast-path holding the old raw pointer detects
      // the gen bump on its next call and re-fetches.  This is
      // still UAF-free under the cache mutex because only one
      // insert runs at a time per cache and the in-flight callers
      // from other threads compare their gen before touching their
      // cached pointer.
      existing->second->second = std::move(v);
      list_.splice(list_.begin(), list_, existing->second);
      weight_cache_generation().fetch_add(1, std::memory_order_relaxed);
      return list_.front().second.get();
    }

    // ── Capacity enforcement (REFUSE-NEW, not evict-old) ──────────
    // Eviction of an existing entry is unsafe because GEMM/BRGEMM
    // loopers hold the raw pointer returned by `find_and_touch()`
    // across the entire kernel call AFTER releasing the cache
    // mutex.  A concurrent `std::free` on that pointer would UAF;
    // `weight_cache_generation()` protects only the next-call TL
    // fast path, not the in-flight call.
    //
    // Instead, when the map is at capacity we refuse the new
    // insert: `v` drops out of scope, the just-packed buffer is
    // freed by the caller's `unique_ptr`, and we return `nullptr`.
    // The caller of `get_or_prepack()` / `get_or_pack()` already
    // handles `nullptr` as a pack miss and falls back to its
    // thread-local repack path (see `bkc_resolve_packed_B_st` for
    // the canonical fallback pattern) — correctness is preserved,
    // the cost is paying a pack on every call for weights beyond
    // the cap.  Cap = UINT32_MAX (default) means "never refuse",
    // restoring the historical unbounded-growth behaviour.
    const uint32_t cap = get_weight_cache_capacity();
    if (cap != std::numeric_limits<uint32_t>::max()
        && list_.size() >= cap) {
      return nullptr;  // cap hit: unique_ptr destructor frees the pack
    }

    list_.emplace_front(k, std::move(v));
    map_[k] = list_.begin();
    return list_.front().second.get();
  }

  /// Drop every entry.  Bumps weight_cache_generation() when the map
  /// was non-empty so any thread-local fast path that holds a raw
  /// entry pointer guarded by a generation snapshot detects the
  /// staleness on its next call (otherwise the per-cache
  /// `clear()` wrappers exposed publicly would silently invalidate
  /// pointers without notifying the TL paths).
  void clear() {
    const bool had_entries = !list_.empty();
    map_.clear();
    list_.clear();
    if (had_entries) {
      weight_cache_generation().fetch_add(1, std::memory_order_relaxed);
    }
  }

  size_t size() const { return list_.size(); }

 private:
  std::list<ListItem> list_;  // front = MRU, back = LRU
  std::unordered_map<PrepackedWeightKey, ListIter, PrepackedWeightKeyHash> map_;
};

/// Thread-safe singleton cache for prepacked weight matrices.
class PrepackedWeightCache {
public:
  static PrepackedWeightCache &instance();
  const PrepackedWeight *get_or_prepack(
    const PrepackedWeightKey &key, const float *weight);
  /// Clear all cached entries. Must only be called when no thread is using
  /// any previously returned pointer (e.g., between test cases or after a
  /// global barrier). Violating this causes use-after-free.
  void clear() { std::lock_guard<std::mutex> lock(mutex_); cache_.clear(); }
  PrepackedWeightCache(const PrepackedWeightCache &) = delete;
  PrepackedWeightCache &operator=(const PrepackedWeightCache &) = delete;
private:
  PrepackedWeightCache() = default;
  LRUWeightMap<PrepackedWeight> cache_;
  std::mutex mutex_;
};

// ============================================================================
// BF16 VNNI prepacked weight cache
// ============================================================================

/// VNNI pair constant: 2 BF16 values per element position per k-pair.
inline constexpr int VNNI_PAIR = 2;

/// Custom deleter for uint16_t aligned_alloc'd memory.
struct AlignedFreeU16Deleter {
  void operator()(uint16_t *p) const { std::free(p); }
};

/// BF16 VNNI prepacked weight buffer.
///
/// Layout: for each NR_PACK-wide panel and each k-pair:
///   packed[kp * NR_PACK * 2 + n * 2 + 0] = B[2*kp  ][panel*NR_PACK + n]
///   packed[kp * NR_PACK * 2 + n * 2 + 1] = B[2*kp+1][panel*NR_PACK + n]
///
/// K is padded to even for VNNI alignment.
struct BF16PrepackedWeight {
  std::unique_ptr<uint16_t[], AlignedFreeU16Deleter> buf;
  const uint16_t *data;  ///< Points to buf.get()
  int K;           ///< Original K
  int K_padded;    ///< K rounded up to even
  int N;
  int n_panels;

  /// Get B panel pointer for k-pair index kp, N-panel index.
  const uint16_t *get_panel(int kp, int panel_idx) const {
    const int vnni_stride = NR_PACK * VNNI_PAIR;
    const int k_pairs = K_padded / 2;
    return data
        + static_cast<size_t>(panel_idx) * k_pairs * vnni_stride
        + static_cast<size_t>(kp) * vnni_stride;
  }

  /// Row stride in uint16_t units for one k-pair.
  static constexpr int stride() { return NR_PACK * VNNI_PAIR; }
};

/// Thread-safe singleton cache for BF16 VNNI prepacked weight matrices.
class BF16PrepackedWeightCache {
public:
  static BF16PrepackedWeightCache &instance();
  const BF16PrepackedWeight *get_or_prepack(
    const PrepackedWeightKey &key, const uint16_t *weight);
  /// @copydoc PrepackedWeightCache::clear()
  void clear() { std::lock_guard<std::mutex> lock(mutex_); cache_.clear(); }
  BF16PrepackedWeightCache(const BF16PrepackedWeightCache &) = delete;
  BF16PrepackedWeightCache &operator=(const BF16PrepackedWeightCache &) = delete;
private:
  BF16PrepackedWeightCache() = default;
  LRUWeightMap<BF16PrepackedWeight> cache_;
  std::mutex mutex_;
};

// ============================================================================
// BF16 Blocked K-contiguous (BKC) VNNI packed weight cache (for BKC GEMV kernel)
// ============================================================================

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
  const uint16_t *data;  ///< Points to buf.get()
  int K;           ///< Original K
  int K_padded;    ///< K rounded up to even
  int N;           ///< Original N
  int N_padded;    ///< N rounded up to BKC_NR_PAD (16)
  size_t total;    ///< Total buffer size in uint16_t elements

  /// N-stride in uint16_t units for one k-pair row.
  int n_stride() const { return N_padded * VNNI_PAIR; }
};

/// Thread-safe singleton cache for BF16 BKC packed weight matrices.
/// bf16_gemv_direct() checks get_weight_cache() internally and falls back
/// to the thread-local repack path when caching is disabled.
class BF16BKCWeightCache {
public:
  static BF16BKCWeightCache &instance();
  const BF16BKCWeight *get_or_pack(
    const PrepackedWeightKey &key, const uint16_t *weight);
  /// @copydoc PrepackedWeightCache::clear()
  void clear() { std::lock_guard<std::mutex> lock(mutex_); cache_.clear(); }
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
  void operator()(int8_t *p) const { std::free(p); }
};
struct AlignedFreeI32Deleter {
  void operator()(int32_t *p) const { std::free(p); }
};

/// INT8 VNNI group size for vpdpbusd.
inline constexpr int INT8_VNNI_GRP = 4;

/// INT8 K-contiguous VNNI packed weight buffer with precomputed
/// dequantization vectors for zero-point compensation.
///
/// Layout: for each k-quad, ALL N columns (padded to BKC_NR_PAD=16)
/// are contiguous as 4-byte groups (matching vpdpbusd operand layout).
struct INT8KContiguousWeight {
  std::unique_ptr<int8_t[], AlignedFreeS8Deleter>  packed_buf;
  std::unique_ptr<int32_t[], AlignedFreeI32Deleter> col_sum_buf;
  std::unique_ptr<float[], AlignedFreeDeleter>      combined_scale_buf;
  std::unique_ptr<float[], AlignedFreeDeleter>      effective_bias_buf;

  const int8_t  *data;
  const int32_t *col_sum;
  const float   *combined_scale;
  const float   *effective_bias;

  int K, K_padded, N, N_padded;
  size_t total;

  float   cached_src_scale;
  int32_t cached_src_zp;

  int n_stride() const { return N_padded * INT8_VNNI_GRP; }
};

/// Thread-safe singleton cache for INT8 K-contiguous packed weights.
/// Caches packed B + col_sum + combined_scale + effective_bias.
/// Cache key is (weight_ptr, K, N, ldb, transB). Dequant vectors
/// (combined_scale, effective_bias) are baked from src_scale, src_zp,
/// bias, and wei_scale at pack time. On cache hit, src_scale/src_zp
/// are validated; mismatch returns nullptr (caller falls back to
/// thread-local path). bias and wei_scale are assumed constant when
/// is_weights_const=true (same pointer → same values invariant).
class INT8KContiguousWeightCache {
public:
  static INT8KContiguousWeightCache &instance();

  /// Get or pack INT8 weights with precomputed dequant vectors.
  /// Returns cached entry if key matches. Does NOT recompute dequant
  /// if src_scale/src_zp changed — caller must ensure static quant
  /// or use thread-local path for dynamic quant.
  const INT8KContiguousWeight *get_or_pack(
    const PrepackedWeightKey &key,
    const int8_t *weight,
    float src_scale, int32_t src_zp,
    const float *bias,
    const float *wei_scale, int wei_scale_count);

  /// @copydoc PrepackedWeightCache::clear()
  void clear() { std::lock_guard<std::mutex> lock(mutex_); cache_.clear(); }
  INT8KContiguousWeightCache(const INT8KContiguousWeightCache &) = delete;
  INT8KContiguousWeightCache &operator=(const INT8KContiguousWeightCache &) = delete;
private:
  INT8KContiguousWeightCache() = default;
  LRUWeightMap<INT8KContiguousWeight> cache_;
  std::mutex mutex_;
};

// ── INT8 VNNI panel-format prepacked weight cache (for BRGEMM looper) ──

/// INT8 VNNI panel-format prepacked weight buffer + column sums.
///
/// Layout: NR_PACK-wide panels with 4-byte VNNI groups (same as BF16 panels
/// but with int8 elements and 4-byte groups instead of 2-byte pairs).
struct INT8PrepackedWeight {
  std::unique_ptr<int8_t[], AlignedFreeS8Deleter>  packed_buf;
  std::unique_ptr<int32_t[], AlignedFreeI32Deleter> col_sum_buf;

  const int8_t  *data;
  const int32_t *col_sum;
  int K, K_padded, N, n_panels;

  const int8_t *get_panel(int kq, int panel_idx) const {
    const int vnni_stride = NR_PACK * INT8_VNNI_GRP;
    const int k_quads = K_padded / 4;
    return data
        + static_cast<size_t>(panel_idx) * k_quads * vnni_stride
        + static_cast<size_t>(kq) * vnni_stride;
  }

  static constexpr int stride() { return NR_PACK * INT8_VNNI_GRP; }
};

/// Thread-safe global singleton cache for INT8 panel-format packed weights.
/// Used by INT8 BRGEMM looper for cross-thread reuse of packed B.
class INT8PrepackedWeightCache {
public:
  static INT8PrepackedWeightCache &instance();
  const INT8PrepackedWeight *get_or_prepack(
    const PrepackedWeightKey &key, const int8_t *weight);
  /// @copydoc PrepackedWeightCache::clear()
  void clear() { std::lock_guard<std::mutex> lock(mutex_); cache_.clear(); }
  INT8PrepackedWeightCache(const INT8PrepackedWeightCache &) = delete;
  INT8PrepackedWeightCache &operator=(const INT8PrepackedWeightCache &) = delete;
private:
  INT8PrepackedWeightCache() = default;
  LRUWeightMap<INT8PrepackedWeight> cache_;
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
