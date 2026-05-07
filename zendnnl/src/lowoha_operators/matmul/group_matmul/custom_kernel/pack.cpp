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

using zendnnl::error_handling::apilog_info;
using zendnnl::error_handling::apilog_info_enabled;
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
// The OUTPUT (packed[]) is identical for both transB values — only
// the input access pattern differs.  This means the microkernel and
// every downstream consumer remain layout-agnostic.
void pack_bf16_vnni(const bfloat16_t *weight, int K, int N, int ldb,
                    int pack_nr, bool transB, bfloat16_t *packed) {
  const int K_pair = (K + 1) / 2;
  const int n_blocks = N / pack_nr;
  const size_t ldb_z = static_cast<size_t>(ldb);

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
        const int col = n_base + n;
        // Row-major addressing in caller's layout.  For transB=false
        // (canonical [K, N]) `k` indexes rows and `col` indexes cols.
        // For transB=true ([N, K] PyTorch layout) `col` indexes rows
        // and `k` indexes cols.  ldb is the caller's row stride in
        // either case, so the same `row * ldb + col` formula works
        // with the row/col labels swapped.
        const size_t lo_off = transB
            ? static_cast<size_t>(col) * ldb_z + k_lo
            : static_cast<size_t>(k_lo) * ldb_z + col;
        kp_base[n * kVNNIPair + 0] = weight[lo_off];
        if (k_hi < K) {
          const size_t hi_off = transB
              ? static_cast<size_t>(col) * ldb_z + k_hi
              : static_cast<size_t>(k_hi) * ldb_z + col;
          kp_base[n * kVNNIPair + 1] = weight[hi_off];
        } else {
          kp_base[n * kVNNIPair + 1] = bfloat16_t(0.0f);
        }
      }
    }
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
    const bfloat16_t **out_packed) {

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
  // same `Key_matmul`); fold pack_nr and transB into the discriminator.
  // Bit 16 of the marker is reserved for the transB flag.
  static constexpr uint32_t kCustomKernelAlgoMarker = 0xC0DE0000U;
  static constexpr uint32_t kTransBMarker          = 0x00010000U;
  const uint32_t variant_bits =
      static_cast<uint32_t>(pack_nr) | (transB ? kTransBMarker : 0u);
  const size_t extra_hash = static_cast<size_t>(
      kCustomKernelAlgoMarker | variant_bits);
  Key_matmul key(weight, static_cast<unsigned>(N),
                 static_cast<unsigned>(K), extra_hash);
  // Fold the caller's row stride into the key so packs built at
  // different strides for the same (weight_ptr, K, N, transB) do not
  // alias.  See the doc-block above for the silent-corruption
  // rationale.
  key.ldb = static_cast<unsigned>(ldb);

  // Cached once on first entry; zero cost per call when API log level
  // is below info.  Without this gate each `apilog_info(...)` below
  // would still do one function call + cached-bool check per pack
  // lookup — at ~16k lookups per MoE iteration that adds ~100 µs of
  // pure waste when logging is off.  The `static const bool` pattern
  // lets the compiler treat the whole log-message construction as
  // dead code in the disabled case.
  static const bool s_pack_log = apilog_info_enabled();

  std::lock_guard<std::mutex> lock(pack_mutex_singleton());

  if (pack_cache.find_key(key)) {
    *out_packed = static_cast<const bfloat16_t *>(pack_cache.get(key));
    // Include the full cache key on the HIT line so a model-level
    // log shows exactly which (weight_ptr, K, N, pack_nr) tuple is
    // being served.  Under the assumption "a fixed model reuses
    // weights across calls" the weight_ptr field should stabilise to
    // a small set of values (= num_experts × num_weights_per_expert).
    // If an E2E run shows this pointer cycling through new values on
    // every decode step, the framework is reallocating weights →
    // cache churn is root-caused to weight-lifecycle, not to
    // anything the dispatcher can fix.
    if (s_pack_log) {
      apilog_info("[GRP_MATMUL Level4 pack HIT] weight=", weight,
                  " K=", K, " N=", N, " ldb=", ldb,
                  " transB=", (transB ? 1 : 0),
                  " pack_nr=", pack_nr,
                  " WEIGHT_CACHE_OUT_OF_PLACE");
    }
    return status_t::success;
  }

  // Miss — allocate, pack, insert.  64-byte aligned so the
  // microkernel's vmovdqu reads start on a cache-line boundary.
  // Same rationale as the HIT path above — the key fields are on the
  // line so the user can see *which* dimension is rotating.  In a
  // stable-weight workload each unique key should appear in a MISS
  // line exactly once; recurring MISSes for the same K/N/pack_nr
  // with only weight_ptr cycling indicate weight-pointer churn.
  if (s_pack_log) {
    apilog_info("[GRP_MATMUL Level4 pack MISS] weight=", weight,
                " K=", K, " N=", N, " ldb=", ldb,
                " transB=", (transB ? 1 : 0),
                " pack_nr=", pack_nr,
                " WEIGHT_CACHE_OUT_OF_PLACE");
  }
  const int K_pair = (K + 1) / 2;
  const size_t bytes = static_cast<size_t>(N / pack_nr)
      * K_pair * pack_nr * kVNNIPair * sizeof(bfloat16_t);
  const size_t alignment = 64;
  const size_t bytes_aligned = (bytes + alignment - 1) & ~(alignment - 1);

  void *raw = std::aligned_alloc(alignment, bytes_aligned);
  if (raw == nullptr) {
    log_error("custom_kernel pack: aligned_alloc failed for ",
              bytes_aligned, " bytes");
    return status_t::failure;
  }

  pack_bf16_vnni(weight, K, N, ldb, pack_nr, transB,
                 static_cast<bfloat16_t *>(raw));
  pack_cache.add(key, raw);

  *out_packed = static_cast<const bfloat16_t *>(raw);
  return status_t::success;
}

// See pack.hpp for the quiescent-window safety contract.  The
// implementation is a two-step capacity toggle on the same
// lru_cache_t used by `get_or_pack_weight_bf16()`:
//   1. Lower the capacity to 0 — the LRU's internal `evict()` pass
//      runs on the next mutating op (below) and `std::free()`s every
//      cached packed buffer via `if constexpr(is_pointer)` path.
//   2. Nudge eviction immediately by calling `set_capacity(0)`
//      (which evicts synchronously inside its own mutex).
//   3. Restore capacity to UINT32_MAX so subsequent
//      `get_or_pack_weight_bf16()` hits go back to the eviction-
//      disabled regime.  Without this step the cache would begin
//      evicting packs as callers add new entries — reintroducing
//      the UAF concern that motivated the UINT32_MAX default.
//
// We also hold the outer `pack_mutex_singleton()` for the duration
// so any racing `get_or_pack_weight_bf16()` call blocks until this
// completes; combined with the caller's quiescent-window guarantee,
// no in-flight microkernel can observe a partially-cleared cache.
void clear_custom_kernel_pack_cache() {
  std::lock_guard<std::mutex> lock(pack_mutex_singleton());
  auto &pack_cache = pack_cache_singleton();
  pack_cache.set_capacity(0);  // evict-all, free()s every pointer
  pack_cache.set_capacity(std::numeric_limits<uint32_t>::max());
  // Rare event (explicit cache clear); still gated for consistency
  // with HIT/MISS so all pack-related APILOG lines share the same
  // enable criterion.
  static const bool s_pack_log = apilog_info_enabled();
  if (s_pack_log) {
    apilog_info("[GRP_MATMUL Level4 pack cleared] Pack cache cleared "
                "WEIGHT_CACHE_OUT_OF_PLACE");
  }
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
