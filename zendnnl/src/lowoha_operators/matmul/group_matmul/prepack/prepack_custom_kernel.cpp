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

#include "prepack_custom_kernel.hpp"

#include <algorithm>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace group_matmul_prepack {
namespace custom_kernel {

using zendnnl::common::bfloat16_t;
using zendnnl::ops::matmul_config_t;

namespace ck = zendnnl::lowoha::matmul::custom_kernel;

// ─────────────────────────────────────────────────────────────────────
// Ahead-of-time warm-pack for all expert weights.
//
// Iterates [0, min(total_count, weight.size(), K.size(), N.size(),
// ldb.size(), transB.size())) and calls `get_or_pack_weight_bf16` for
// each well-formed entry.  Idempotent — cached entries are no-ops.
//
// The defensive `min()` against every metadata-vector size guards
// against the framework passing `weight[]` at total_matmul but other
// vectors at active_matmul (or vice-versa).  We treat any out-of-bound
// index as "no metadata" and stop without touching the cache for it.
//
// Per-entry failures (null weight pointer, bad N divisibility, OOM
// during VNNI pack) are counted in `stats.skipped_invalid` and do
// NOT abort the warm-pack — the active-set dispatcher will surface
// any genuine error independently when the caller actually tries to
// dispatch that expert.  Helps avoid a transient framework bug in
// the prepack-only slots from breaking the inference call entirely.
//
// Pre-pack reasons for `skipped_invalid` (parity with the AOCL DLP
// warmer's counter semantics):
//   1) weight ptr null / K|N|ldb non-positive
//   2) plan_pack_nr returns 0 (N not a multiple of 32 or 64)
//   3) ldb below the min row stride (transB ? K : N) — same rule
//      the dispatcher applies at runtime
//   4) is_weights_const flag says "variable weight" — mirrors the
//      AOCL DLP warmer; the CK runtime independently refuses any
//      call with a non-const active expert (see
//      `custom_kernel/dispatch.cpp::prepare_for_call`), so warming
//      a non-const expert here would pollute the cache with an
//      entry the runtime would never consult coherently.  An empty
//      `is_weights_const` vector means "treat every entry as
//      const" (legacy behaviour for callers that don't pass it).
//   5) `get_or_pack_weight_bf16` returns failure (OOM / VNNI pack
//      error)
//
// Body lifted verbatim from `custom_kernel/dispatch.cpp` into this
// file; uses the now-public `ck::dispatch_supported()` and
// `ck::plan_pack_nr()` plus the `ck::get_or_pack_weight_bf16` API in
// `pack.hpp`, so no logic was duplicated or changed.
// ─────────────────────────────────────────────────────────────────────
status_t warm_pack_all_custom_kernel_experts(
    const std::vector<const void *> &weight,
    const std::vector<int>          &K,
    const std::vector<int>          &N,
    const std::vector<int>          &ldb,
    const std::vector<bool>         &transB,
    const std::vector<bool>         &is_weights_const,
    int                              total_count,
    PackProbeStats                  &stats,
    bool                             interleave_split_halves,
    WarmDtypeFamily                  dtype_family) {

  if (total_count <= 0)
    return status_t::success;
  // Per-family ISA gate — mirrors the split gate in
  // `custom_kernel/dispatch.cpp::prepare_for_call`.  bf16 pack warming
  // needs AVX-512 BF16 (VDPBF16PS); the DQ-INT8 family needs AVX-512
  // VNNI (VPDPBUSD).  On Zen 4/5 VNNI is a superset of BF16, but on
  // broader x86 (Cascade Lake / Ice Lake) VNNI exists WITHOUT BF16, so
  // gating the int8 warm on `dispatch_supported()` (BF16) would make
  // the int8 fast path unreachable on a host that can actually run it.
  // Refuse the whole warm only when the family's own ISA is absent —
  // warming an arena the runtime cannot consume is pure waste, and the
  // runtime would refuse the matching call identically.
  if (dtype_family == WarmDtypeFamily::kINT8) {
    if (!ck::avx512vnni_available())
      return status_t::success;
  } else if (!ck::dispatch_supported()) {  // bf16 family needs AVX-512 BF16
    return status_t::success;
  }

  // ── Production-cache gate ────────────────────────────────────────
  // Mirrors the AOCL DLP warmer's entry gate
  // (`prepack_aocl_dlp.cpp::warm_pack_all_aocl_dlp_experts`) and the
  // library-wide weight-cache convention: when
  // `matmul_config_t::get_weight_cache()` (driven by the env knob
  // `ZENDNNL_MATMUL_WEIGHT_CACHE`) is not 1, no path consults the
  // LRU pack cache, so warming would be a pure waste of CPU + memory
  // and would also leave behind cache entries the dispatcher must
  // not read (the CK pack arena is keyed on raw weight pointer; a
  // framework that mutates weight addresses between calls relies on
  // WEIGHT_CACHE=0 specifically to prevent a stale pointer hit from
  // reading freed weights).  The matching runtime gate lives in
  // `custom_kernel/dispatch.cpp::prepare_for_call`, which under
  // `weight_cache_type != 1` keeps CK ENGAGED but switches the per-
  // expert pack call to `get_or_pack_weight_bf16(..., disable_cache=
  // true)` — allocating a fresh aligned buffer, packing, recording
  // it in `CallContext::owned_packed_ptrs[]`, and freeing it after
  // the run via `CallContext::release_owned_buffers()`.  The LRU is
  // never inserted into in that mode, so warming this side would
  // populate keys the runtime never reads.  Earlier revs refused CK
  // outright when WEIGHT_CACHE!=1 and fell back to AOCL DLP (whose
  // cache-off path uses fresh out-of-place reorder + per-call free
  // at `aocl_kernel.cpp::reorderAndCacheWeights` weight_cache_type=0
  // and the matching free at `run_dlp`'s epilogue); that fallback
  // was relaxed by commit 63e0299c to keep CK on the hot path.
  //
  // Short-circuit at entry; `stats` stays zeroed so the PREPACK log
  // line surfaces "no work attempted" rather than "all skipped".
  // The toggle is folded into the prepack fingerprint
  // (`prepack.cpp::fingerprint`), so a process that flips
  // WEIGHT_CACHE=0 → 1 mid-run will re-enter this function on the
  // new state and warm normally.
  const int32_t weight_cache_type =
      matmul_config_t::instance().get_weight_cache();
  if (weight_cache_type != 1) {
    return status_t::success;
  }

  const size_t bound = std::min<size_t>({
      static_cast<size_t>(total_count),
      weight.size(),
      K.size(),
      N.size(),
      ldb.size(),
      transB.size()});

  for (size_t i = 0; i < bound; ++i) {
    ++stats.total_attempted;

    if (weight[i] == nullptr || K[i] <= 0 || N[i] <= 0 || ldb[i] <= 0) {
      ++stats.skipped_invalid;
      continue;
    }

    const int pack_nr = ck::plan_pack_nr(K[i], N[i]);
    if (pack_nr != ck::kNRMin && pack_nr != ck::kNRMax) {
      ++stats.skipped_invalid;
      continue;
    }

    const int min_ldb = transB[i] ? K[i] : N[i];
    if (ldb[i] < min_ldb) {
      ++stats.skipped_invalid;
      continue;
    }

    // Mirror the AOCL DLP warmer's `is_weights_const` gate: when
    // the caller flags expert `i` as variable-weight, skip packing.
    // The CK runtime separately refuses any call containing a non-
    // const active expert, so warming here would always be wasted
    // (and worse, would create a stale cache entry that subsequent
    // calls with mutated weights would silently consume if the CK
    // refusal logic were ever relaxed).  An empty is_weights_const
    // vector means "treat every entry as const" (legacy behaviour
    // for callers that don't pass the field).
    if (!is_weights_const.empty()
        && i < is_weights_const.size()
        && !is_weights_const[i]) {
      ++stats.skipped_invalid;
      continue;
    }

    // Dtype-switch — bf16 and int8 packs share the identical
    // contract (same `(weight, K, N, ldb, pack_nr, transB,
    // interleave_split_halves)` interface), so the warm loop is
    // straight-line except for which pack function to call.  The
    // pack functions populate disjoint LRU singletons keyed by
    // disjoint cache-key markers (see pack.cpp's
    // `kCustomKernelAlgoMarker` vs `kCustomKernelInt8Marker`); a
    // process that warms both families on the same model pays one
    // warm per family with no alias risk.
    bool was_hit = false;
    status_t pst = status_t::success;
    if (dtype_family == WarmDtypeFamily::kINT8) {
      const int8_t *packed_ignored = nullptr;
      pst = ck::get_or_pack_weight_int8(
          static_cast<const int8_t *>(weight[i]),
          K[i], N[i], ldb[i], pack_nr, transB[i],
          /*interleave_split_halves=*/interleave_split_halves,
          &packed_ignored, &was_hit);
    } else {
      const bfloat16_t *packed_ignored = nullptr;
      pst = ck::get_or_pack_weight_bf16(
          static_cast<const bfloat16_t *>(weight[i]),
          K[i], N[i], ldb[i], pack_nr, transB[i],
          /*interleave_split_halves=*/interleave_split_halves,
          &packed_ignored, &was_hit);
    }

    if (pst == status_t::success) {
      ++stats.packed_ok;
      if (was_hit) ++stats.cache_hits;
      else         ++stats.cache_misses;
    } else {
      ++stats.skipped_invalid;
    }
  }

  return status_t::success;
}

} // namespace custom_kernel
} // namespace group_matmul_prepack
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
