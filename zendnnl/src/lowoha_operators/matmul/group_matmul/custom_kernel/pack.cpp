/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Note: the `[b0, b1, b2, b3, ...]` interleaved-pair pack format
// implemented below is BF16-specific.  It assumes 16-bit elements and
// the AVX-512-BF16 VDPBF16PS instruction's expectation of a pair-
// interleaved K layout.  Adding int8 support in the future will
// require a different K-interleave stride (4 int8 elements per
// VPDPBUSD lane vs the BF16 pair) and additional packed state
// (per-tensor / per-row / per-group scales and optional zero-points);
// when that lands, this file can either grow a per-dtype packer
// family or be paired with a sibling `pack_int8.cpp`.  The file
// currently stays at `custom_kernel/pack.cpp` because BF16 is the
// only dtype variant.

#include "pack.hpp"

#include <cstdlib>
#include <limits>
#include <mutex>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/lru_cache/lru_cache.hpp"
#include "lowoha_operators/matmul/lru_cache/zendnnl_key.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

using zendnnl::error_handling::apilog_verbose;
using zendnnl::error_handling::apilog_verbose_enabled;
using zendnnl::error_handling::log_error;

namespace {

// VNNI pack of caller's BF16 weight → [O/pack_nr, K_pad/2, pack_nr, 2].
// K_pad = K rounded up to even; trailing K-tail (when K is odd) is
// zero-padded so a 4-byte VDPBF16PS load is always safe.
//
// `transB` selects the caller's logical layout:
//   * false → [K, N] row-major, weight[k][col] = weight[k * ldb + col].
//   * true  → [N, K] row-major (PyTorch nn.Linear convention),
//             weight[col][k] = weight[col * ldb + k].
// `ldb` is the caller's actual row stride (typically N for the [K,N]
// layout and K for the [N,K] layout, but passed explicitly so the
// pack supports non-contiguous tensors / slices correctly).
//
// `interleave_split_halves` selects how each pack column maps back
// to a canonical-weight column:
//   * false (default) — pack column `col` reads canonical column
//     `col`.  Verbatim copy + VNNI re-tile; same OUTPUT for any
//     caller layout that already presents the desired N-order
//     (none / swiglu_oai_mul on already-interleaved input).
//   * true (silu_and_mul / gelu_and_mul fused-CK paths) — caller's
//     weight is in split-halves `[gate_cols | up_cols]` order
//     (N = 2I).  Pack output column `2i+0` reads canonical column
//     `i` (gate i); pack output column `2i+1` reads canonical column
//     `I + i` (up i).  The resulting packed bytes are physically
//     identical to what `swiglu_oai_mul`'s caller-side interleaved
//     input would produce, so the in-register pair-store epilogues
//     (`silu_and_mul_store_pair`, `gelu_and_mul_store_pair`) apply
//     unchanged.  silu and gelu share the SAME permutation; the
//     cache-key bit `kInterleaveSplitMarker` is shared between them
//     so a packed arena warmed for silu can be reused for gelu on
//     the same weight pointer.  Requires N to be even.
//
// The OUTPUT (packed[]) is identical to the swiglu_oai_mul layout
// when interleaving is requested, so the microkernel and every
// downstream consumer remain layout-agnostic.
//
// Implementation note: the `Interleave` template parameter on
// `pack_bf16_vnni_impl` lifts the (loop-invariant) split-halves
// branch out of the innermost loop body — both instantiations
// compile to a tight branch-free pack kernel.  The thin
// `pack_bf16_vnni` wrapper below selects the instantiation once
// per call from the runtime flag.
template <bool Interleave>
inline void pack_bf16_vnni_impl(const bfloat16_t *weight, int K, int N,
                                int ldb, int pack_nr, bool transB,
                                bfloat16_t *packed) {
  const int K_pair = (K + 1) / 2;
  const int n_blocks = N / pack_nr;
  const size_t ldb_z = static_cast<size_t>(ldb);
  // I = N / 2 is the half-width for the interleave path.  Read only
  // when `Interleave` is true; the compiler removes the load in the
  // identity-mapping instantiation.  N evenness is a precondition
  // documented in the header; we don't re-check here to keep the
  // hot loop lean.
  //
  // `if constexpr (!Interleave) (void)I` suppresses `-Wunused-variable`
  // in the `Interleave == false` instantiation (the only branch that
  // references `I` is the `col_canon` ternary's `Interleave?…:col_pack`
  // arm, which the optimiser DCE-removes when the template parameter
  // is false — but the un-optimised, warning-pass-time AST still sees
  // the binding).  `if constexpr` ensures the cast itself is also
  // gone from the `Interleave == true` instantiation, so no code-gen
  // change in either branch.
  const int I = N / 2;
  if constexpr (!Interleave) (void)I;

  for (int o_blk = 0; o_blk < n_blocks; ++o_blk) {
    bfloat16_t *blk_base = packed + static_cast<size_t>(o_blk)
        * K_pair * pack_nr * kVNNIPair;
    const int n_base = o_blk * pack_nr;

    for (int kp = 0; kp < K_pair; ++kp) {
      const int k_lo = kp * 2;
      const int k_hi = k_lo + 1;
      bfloat16_t *kp_base = blk_base
          + static_cast<size_t>(kp) * pack_nr * kVNNIPair;

      for (int n = 0; n < pack_nr; ++n) {
        // Pack output column = n_base + n.  Canonical column is
        // identity when `Interleave` is false; pair-interleave
        // permutation when true.  Either way the compiler folds the
        // mapping into a constant-control sequence per instantiation.
        const int col_pack = n_base + n;
        const int col_canon = Interleave
            ? ((col_pack & 1) ? (I + (col_pack >> 1)) : (col_pack >> 1))
            : col_pack;
        // Row-major addressing in caller's layout.  For transB=false
        // (canonical [K, N]) `k` indexes rows and `col_canon` indexes cols.
        // For transB=true ([N, K] PyTorch layout) `col_canon` indexes rows
        // and `k` indexes cols.  ldb is the caller's row stride in
        // either case, so the same `row * ldb + col` formula works
        // with the row/col labels swapped.
        const size_t lo_off = transB
            ? static_cast<size_t>(col_canon) * ldb_z + k_lo
            : static_cast<size_t>(k_lo) * ldb_z + col_canon;
        kp_base[n * kVNNIPair + 0] = weight[lo_off];
        if (k_hi < K) {
          const size_t hi_off = transB
              ? static_cast<size_t>(col_canon) * ldb_z + k_hi
              : static_cast<size_t>(k_hi) * ldb_z + col_canon;
          kp_base[n * kVNNIPair + 1] = weight[hi_off];
        } else {
          kp_base[n * kVNNIPair + 1] = bfloat16_t(0.0f);
        }
      }
    }
  }
}

void pack_bf16_vnni(const bfloat16_t *weight, int K, int N, int ldb,
                    int pack_nr, bool transB,
                    bool interleave_split_halves,
                    bfloat16_t *packed) {
  if (interleave_split_halves) {
    pack_bf16_vnni_impl<true>(weight, K, N, ldb, pack_nr, transB, packed);
  } else {
    pack_bf16_vnni_impl<false>(weight, K, N, ldb, pack_nr, transB, packed);
  }
}

// ── Shared per-process cache state (see doc-block below) ────────────
// Pulled out of `get_or_pack_weight_bf16()` so
// `clear_custom_kernel_pack_cache()` can reach the same singleton.
// The outer anonymous namespace keeps everything invisible to other
// translation units.
//
// ── Why eviction is DISABLED (UINT32_MAX capacity) ────────────────
// The dispatcher layer (`prepare_for_call`) stores raw packed_ptrs
// in per-call `CallContext` and the OMP region (`dispatch_tile`)
// reads them WITHOUT re-acquiring this mutex.  If eviction were
// allowed to run on a concurrent `add()` call, it would
// `std::free()` a packed buffer that another thread is still
// reading from — a use-after-free in the microkernel.  Giving the
// cache an infinite capacity sidesteps that race without needing
// per-pointer refcounts.  This explicitly overrides any global
// `ZENDNNL_LRU_CACHE_CAPACITY` setting, because the custom kernel
// has no generation-guarded fast-path retry like the outer
// prepack cache (`kernel_cache.hpp`) does.
//
// ── RSS bound in steady state ─────────────────────────────────────
// Live cache footprint is:
//     Σ_unique(weight_ptr, N, K, pack_nr)  K_pad × N × sizeof(BF16)
// where `K_pad = (K + 1) & ~1` (VNNI pair padding).  For typical
// MoE serving with a fixed model:
//     experts × pack_nr_variants × per-expert pack size
// which is the same order of magnitude as the weights already
// resident in RAM.  The cache is model-bounded, not workload-
// bounded, for fixed-weight deployments.
//
// ── Failure mode for weight-rotating workloads ────────────────────
// Frameworks that repeatedly rebuild or reallocate weight tensors
// (e.g. dynamic LoRA hot-swap, retraining loops, compile/decompile
// cycles) will see the cache footprint grow by one entry per
// distinct (weight pointer, N, K, pack_nr) tuple observed — the
// cache cannot distinguish "the old weight at ptr P is gone" from
// "a fresh weight at ptr P is arriving", so old packed buffers
// stay resident.  For those deployments call
// `clear_custom_kernel_pack_cache()` during a known quiescent
// window between workload phases (see the safety note in pack.hpp).
lru_cache_t<Key_matmul, void *> &pack_cache_singleton() {
  static lru_cache_t<Key_matmul, void *> pack_cache(
      std::numeric_limits<uint32_t>::max());
  return pack_cache;
}
std::mutex &pack_mutex_singleton() {
  static std::mutex pack_mutex;
  return pack_mutex;
}

} // namespace

status_t get_or_pack_weight_bf16(
    const bfloat16_t *weight,
    int K, int N, int ldb, int pack_nr,
    bool transB,
    bool interleave_split_halves,
    const bfloat16_t **out_packed,
    bool *was_hit_out,
    bool disable_cache) {

  if (was_hit_out != nullptr) *was_hit_out = false;

  if (weight == nullptr || K <= 0 || N <= 0
      || (pack_nr != kNRMin && pack_nr != kNRMax)
      || (N % pack_nr) != 0
      || ldb <= 0
      || out_packed == nullptr) {
    log_error("custom_kernel pack: invalid arg "
              "(weight, K, N, ldb must be valid; pack_nr in {",
              kNRMin, ",", kNRMax, "}; N %% pack_nr == 0)");
    return status_t::failure;
  }
  // Interleaved-split-halves only valid when N is even (the
  // permutation pairs `(2i, 2i+1)` over `[0, N)`).  Pack_nr is
  // already a power of two divisor of N so this just covers the
  // theoretical case N=2 × odd.
  if (interleave_split_halves && (N & 1)) {
    log_error("custom_kernel pack: interleave_split_halves requires "
              "even N (got N=", N, ")");
    return status_t::failure;
  }
  // Caller stride sanity: must accommodate the LOGICAL layout the
  // pack reads with — N for [K,N], K for [N,K].  Catches a stride
  // that's smaller than the inner row, which would make the pack
  // alias adjacent rows silently.
  const int min_ldb = transB ? K : N;
  if (ldb < min_ldb) {
    log_error("custom_kernel pack: ldb=", ldb,
              " smaller than minimum row stride (",
              transB ? "K=" : "N=", min_ldb,
              " for transB=", (transB ? "true" : "false"), ")");
    return status_t::failure;
  }

  // Pack-buffer size + alignment computation — used by both the
  // disable-cache branch immediately below and the cache-miss
  // branch further down.  Hoisted so the two paths produce
  // physically identical allocations (same alignment, same size,
  // same VNNI K-pair padding).  `K_pair = (K + 1) / 2` matches the
  // odd-K pad-by-zero contract documented at file head.
  const int    K_pair        = (K + 1) / 2;
  const size_t bytes         = static_cast<size_t>(N / pack_nr)
      * K_pair * pack_nr * kVNNIPair * sizeof(bfloat16_t);
  const size_t alignment     = 64;
  const size_t bytes_aligned = (bytes + alignment - 1) & ~(alignment - 1);

  // Cached once on first entry; zero cost per call when API log level
  // is below verbose.  These HIT / MISS / NOCACHE lines fire per-
  // expert and can also be amplified inside OMP regions, so they are
  // gated on the verbose level (`ZENDNNL_API_LOG_LEVEL=4`) — info
  // level (3) stays clean and shows only the consolidated
  // `[GRP_MATMUL.PREPACK]` line emitted from the prepack module.
  // Hoisted above the disable-cache branch (was previously declared
  // inside the cache-path immediately before `find_key`) so both
  // pack routes share the same log-gate without re-querying
  // `apilog_verbose_enabled()` twice per call.  The
  // `static const bool` pattern still lets the compiler treat the
  // log-message construction as dead code in the disabled case.
  static const bool s_pack_log = apilog_verbose_enabled();

  // ── Disable-cache (caller-owned) branch ──────────────────────────
  // Library-wide `ZENDNNL_MATMUL_WEIGHT_CACHE != 1` mode.  The
  // dispatcher (`custom_kernel/dispatch.cpp::prepare_for_call`)
  // walks the active experts and forwards `disable_cache=true` here
  // when the toggle is off; the prepack-side warmer short-circuits
  // earlier so it never reaches this path.
  //
  // Skip the LRU singleton entirely (no `find_key`, no `add`, no
  // mutex acquisition) and always allocate + pack into a fresh
  // 64-byte-aligned buffer.  Lifetime is CALLER-OWNED: the
  // dispatcher records the raw pointer in
  // `CallContext::owned_packed_ptrs[i]` and its destructor
  // (`release_owned_buffers()`) routes the free through
  // `free_owned_packed_weight()` once `dispatch_tile()` has finished
  // consuming it.  This keeps the CK kernel math available under a
  // weight-pointer-churning framework while paying a per-call pack
  // cost (no inter-call amortisation).  `was_hit_out` is forced
  // false above — the pack always runs in this mode.
  //
  // The mutex is intentionally NOT held here: the LRU is not
  // touched, the user weight buffer is read-only, and the freshly
  // allocated destination is private to this call.  Two threads
  // packing the same (weight_ptr, K, N, ...) tuple concurrently
  // would each get their own caller-owned buffer with identical
  // contents — that's the expected semantic when caching is off.
  if (disable_cache) {
    if (s_pack_log) {
      apilog_verbose("[GRP_MATMUL.PACK NOCACHE] weight=", weight,
                     " K=", K, " N=", N, " ldb=", ldb,
                     " transB=", (transB ? 1 : 0),
                     " interleave=", (interleave_split_halves ? 1 : 0),
                     " pack_nr=", pack_nr);
    }
    void *raw = std::aligned_alloc(alignment, bytes_aligned);
    if (raw == nullptr) {
      log_error("custom_kernel pack (disable_cache): aligned_alloc "
                "failed for ", bytes_aligned, " bytes");
      return status_t::failure;
    }
    pack_bf16_vnni(weight, K, N, ldb, pack_nr, transB,
                   interleave_split_halves,
                   static_cast<bfloat16_t *>(raw));
    *out_packed = static_cast<const bfloat16_t *>(raw);
    return status_t::success;
  }

  // Canonical Key_matmul / lru_cache_t pair — same pattern AOCL DLP
  // and oneDNN reorder paths use.  `extra_input_hash` carries the
  // pack-width discriminator AND the transB flag so different NR or
  // layout variants of the same weight pointer do not alias.  See
  // the `pack_cache_singleton()` doc-block above for RSS bounds,
  // eviction rationale, and the weight-rotating-workload failure
  // mode.
  //
  // ldb is folded into Key_matmul::ldb (not into extra_input_hash) so
  // the equality operator distinguishes packs built from different
  // strides over the same `(weight_ptr, K, N, transB)` tuple.  The
  // packed VNNI layout is shape-only (same shape for any ldb), but
  // the SOURCE access pattern depends on ldb — packing the same
  // pointer at two different strides produces two physically different
  // packed buffers.  Without ldb in the key, a caller that re-uses the
  // same weight_ptr with a different ldb (e.g., a sliced view, padded-
  // vs-tight stride, or a re-padded reallocation at the same address)
  // would silently get the first-seen pack, producing wrong rows in
  // the GEMM result.  Including ldb is correctness-critical even if it
  // costs an extra cache entry on the rare callers that legitimately
  // pack the same pointer at multiple strides.
  auto &pack_cache = pack_cache_singleton();

  // Mark the cache slot as belonging to the custom BF16 microkernel
  // (so it doesn't collide with any existing cache that uses the
  // same `Key_matmul`); fold pack_nr, transB, and the split-halves
  // interleave flag into the discriminator.
  //
  // Bit layout of `kCustomKernelAlgoMarker = 0xC0DE0000`:
  //
  //   nibble [28..31] = 0xC = 1100  → bits 30, 31 set
  //   nibble [24..27] = 0x0 = 0000  → all CLEAR (variant-bit zone)
  //   nibble [20..23] = 0xD = 1101  → bits 20, 22, 23 set
  //   nibble [16..19] = 0xE = 1110  → bits 17, 18, 19 set
  //   nibble [ 0..15] = 0x0000      → all CLEAR (variant-bit zone)
  //
  // Variant bits MUST be picked from the CLEAR positions, otherwise
  // an `ALGO_MARKER | flag` OR is a no-op and the cache key fails to
  // distinguish flag-on vs flag-off entries — silently aliasing
  // different physical pack layouts onto the same LRU slot.
  //
  //   * Bit 16 (0x00010000)  — transB flag.  Inside the [16..19]
  //                            nibble but bit 16 itself is clear
  //                            (nibble 0xE = 1110 has LSB clear).
  //   * Bit 24 (0x01000000)  — interleave_split_halves flag.  Picked
  //                            from the all-clear nibble [24..27] so
  //                            future variant bits have a contiguous
  //                            home that's structurally safe.
  //                            Distinguishes packs built with the
  //                            silu/gelu interleave permutation from
  //                            ordinary swiglu_oai_mul / none packs
  //                            of the same weight ptr.  Without
  //                            this bit, switching activation kinds
  //                            mid-process on the same weight
  //                            pointer would silently serve the
  //                            wrong layout from cache.
  //
  // Bit 17 was the previous (buggy) choice for the interleave
  // marker — it's SET in `kCustomKernelAlgoMarker`'s 0xE nibble, so
  // `kCustomKernelAlgoMarker | kInterleaveSplitMarker` was a no-op
  // and interleaved packs aliased non-interleaved packs on the same
  // weight pointer (caught in Copilot review, see commit log).
  // The compile-time check below catches any future regression.
  static constexpr uint32_t kCustomKernelAlgoMarker = 0xC0DE0000U;
  static constexpr uint32_t kTransBMarker          = 0x00010000U;
  static constexpr uint32_t kInterleaveSplitMarker = 0x01000000U;
  // Compile-time invariant: every variant flag must be in a CLEAR
  // bit of `kCustomKernelAlgoMarker` so the OR actually flips the
  // cache key.  Catches any future flag bit that slips into a set
  // position.
  static_assert(
      (kCustomKernelAlgoMarker & kTransBMarker) == 0u,
      "kTransBMarker collides with kCustomKernelAlgoMarker — pick "
      "a clear bit (positions 0-15, 16, 21, or 24-29).");
  static_assert(
      (kCustomKernelAlgoMarker & kInterleaveSplitMarker) == 0u,
      "kInterleaveSplitMarker collides with kCustomKernelAlgoMarker — "
      "pick a clear bit (positions 0-15, 16, 21, or 24-29).");
  const uint32_t variant_bits =
      static_cast<uint32_t>(pack_nr)
      | (transB ? kTransBMarker : 0u)
      | (interleave_split_halves ? kInterleaveSplitMarker : 0u);
  const size_t extra_hash = static_cast<size_t>(
      kCustomKernelAlgoMarker | variant_bits);
  Key_matmul key(weight, static_cast<unsigned>(N),
                 static_cast<unsigned>(K), extra_hash);
  // Fold the caller's row stride into the key so packs built at
  // different strides for the same (weight_ptr, K, N, transB) do not
  // alias.  See the doc-block above for the silent-corruption
  // rationale.
  key.ldb = static_cast<unsigned>(ldb);

  // `s_pack_log` is hoisted above the disable-cache branch — shared
  // by both pack routes (NOCACHE / HIT / MISS) so we never query
  // `apilog_verbose_enabled()` twice per call.
  std::lock_guard<std::mutex> lock(pack_mutex_singleton());

  if (pack_cache.find_key(key)) {
    *out_packed = static_cast<const bfloat16_t *>(pack_cache.get(key));
    if (was_hit_out != nullptr) *was_hit_out = true;
    // Include the full cache key on the HIT line so a model-level
    // log shows exactly which (weight_ptr, K, N, ldb, transB,
    // pack_nr) tuple is being served.  Under the assumption "a fixed
    // model reuses weights across calls" the weight_ptr field
    // should stabilise to a small set of values (= num_experts ×
    // num_weights_per_expert).  If an E2E run shows this pointer
    // cycling through new values on every decode step, the
    // framework is reallocating weights → cache churn is root-caused
    // to weight-lifecycle, not to anything the dispatcher can fix.
    if (s_pack_log) {
      apilog_verbose("[GRP_MATMUL.PACK HIT] weight=", weight,
                     " K=", K, " N=", N, " ldb=", ldb,
                     " transB=", (transB ? 1 : 0),
                     " interleave=", (interleave_split_halves ? 1 : 0),
                     " pack_nr=", pack_nr);
    }
    return status_t::success;
  }

  // Miss — allocate, pack, insert.  64-byte aligned so the
  // microkernel's vmovdqu reads start on a cache-line boundary.
  // Same rationale as the HIT path above — the full key (weight_ptr,
  // K, N, ldb, transB, pack_nr) is on the line so the user can see
  // *which* dimension is rotating.  In a stable-weight workload each
  // unique key should appear in a MISS line exactly once; recurring
  // MISSes for the same (K, N, ldb, transB, pack_nr) with only
  // weight_ptr cycling indicate weight-pointer churn.
  if (s_pack_log) {
    apilog_verbose("[GRP_MATMUL.PACK MISS] weight=", weight,
                   " K=", K, " N=", N, " ldb=", ldb,
                   " transB=", (transB ? 1 : 0),
                   " interleave=", (interleave_split_halves ? 1 : 0),
                   " pack_nr=", pack_nr);
  }
  // `bytes_aligned` + `alignment` are hoisted above the disable-cache
  // branch so cached and caller-owned packs use identical allocations.
  void *raw = std::aligned_alloc(alignment, bytes_aligned);
  if (raw == nullptr) {
    log_error("custom_kernel pack: aligned_alloc failed for ",
              bytes_aligned, " bytes");
    return status_t::failure;
  }

  pack_bf16_vnni(weight, K, N, ldb, pack_nr, transB,
                 interleave_split_halves,
                 static_cast<bfloat16_t *>(raw));
  pack_cache.add(key, raw);

  *out_packed = static_cast<const bfloat16_t *>(raw);
  return status_t::success;
}

// ── Caller-owned packed-weight release ───────────────────────────
// Companion to `get_or_pack_weight_bf16(..., disable_cache=true)`.
// The disable-cache branch above allocates via `std::aligned_alloc`
// and returns the raw pointer without inserting into the LRU; this
// helper free()s the same buffer.  Safe with `nullptr` so callers
// can iterate `owned_packed_ptrs[kMaxExperts]` unconditionally.
//
// MUST NOT be called on cache-served pointers (`disable_cache=false`
// path) — those are owned by the LRU singleton and freed by
// `clear_custom_kernel_pack_cache()`.  The CallContext in
// `dispatch.hpp` is the single authority on which array (`packed_ptrs`
// vs `owned_packed_ptrs`) carries cache-served vs caller-owned
// pointers, so application code never has to make this distinction
// itself.
void free_owned_packed_weight(const bfloat16_t *packed) {
  if (packed == nullptr) return;
  std::free(const_cast<bfloat16_t *>(packed));
}

// See pack.hpp for the quiescent-window safety contract.  The
// implementation calls `lru_cache_t::clear()` under the same
// `pack_mutex_singleton()` that `get_or_pack_weight_bf16()` holds,
// so any racing pack call blocks until the clear completes.
// Combined with the caller's quiescent-window guarantee (no
// in-flight `dispatch_tile` reading a `ctx.packed_ptrs[i]` from a
// previous prepare_for_call), the clear is safe even though the
// cache's eviction-disabled UINT32_MAX capacity is never lowered.
//
// `lru_cache_t::clear()` walks the map and `std::free()`s every
// pointer regardless of the capacity setting — the right primitive
// for this use case.  Previously this function attempted a two-
// step `set_capacity(0)` + `set_capacity(MAX)` toggle, but the
// parametrised `evict(n)` invoked by `set_capacity` has a size_t
// underflow when `capacity_ == 0` (the loop condition
// `size > capacity_ - n` wraps to `size > UINT_MAX - n + 1` and is
// therefore always false), so the old form silently left every
// entry in place.  Using `clear()` directly sidesteps the bug.
void clear_custom_kernel_pack_cache() {
  std::lock_guard<std::mutex> lock(pack_mutex_singleton());
  auto &pack_cache = pack_cache_singleton();
  pack_cache.clear();
  // Rare event (explicit cache clear); still gated for consistency
  // with HIT/MISS so all pack-related APILOG lines share the same
  // enable criterion (gated on the verbose level).
  static const bool s_pack_log = apilog_verbose_enabled();
  if (s_pack_log) {
    apilog_verbose("[GRP_MATMUL.PACK cleared] Pack cache cleared");
  }
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
