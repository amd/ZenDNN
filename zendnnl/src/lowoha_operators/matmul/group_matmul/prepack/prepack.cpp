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

#include "prepack.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_aocl_dlp.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_custom_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace group_matmul_prepack {

using zendnnl::error_handling::apilog_info;
using zendnnl::error_handling::apilog_info_enabled;
using zendnnl::ops::matmul_algo_t;

// ─────────────────────────────────────────────────────────────────────
// Internal helpers — file-local.
// ─────────────────────────────────────────────────────────────────────
namespace {

// Process-wide fingerprint cache (Fix A — see lines 97-126 below for
// the rationale and the thread-safety contract).  `warm_pack_*` is
// idempotent and the underlying caches never evict during regular
// operation, so once we've warmed a (model, layer, weight pointers,
// scheduling algo) combination it stays warm for the lifetime of the
// process and any thread can short-circuit subsequent calls.  Calling
// the helpers again on every per-ALGO entry costs Θ(num_ops_total)
// mutex acquisitions + LRU lookups (each one µs-class); on large
// expert pools that adds a non-trivial per-token tax to the decode
// budget.  This cache gates the whole warm-pack out when the
// fingerprint matches a previously-warmed configuration.
//
// Fingerprint hashes identity-defining inputs that won't change
// across calls of the same MoE block: total_matmul, scheduling
// algo, order-independent XOR over the full per-expert pool
// (`weight_ptr`, `K`, `N`, `ldb`, `transB` over `[0, num_ops_total)`)
// + iteration count, num_threads, nr_align, activation kind,
// activation/bias dtypes, weight-cache state.  Using the per-expert
// XORs (rather than sampling only `[0]`) lets the fingerprint
// detect every shape mutation regardless of which expert index
// changed — the previous `K[0]/N[0]` sampling missed mutations at
// `i > 0` on a stable pointer set.
inline size_t mix_hash(size_t a, size_t b) {
  return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

inline size_t fingerprint(const PrepackParams &p, int scheduling_algo) {
  size_t s = mix_hash(0, static_cast<size_t>(p.num_ops_total));
  s = mix_hash(s, static_cast<size_t>(scheduling_algo));
  // Order-independent reduction over the full weight pool.  The
  // previous implementation sampled three positions (`[0]`, `[n/2]`,
  // `[n-1]`) and folded each through the order-DEPENDENT `mix_hash`,
  // which made the fingerprint flip whenever a caller permuted the
  // weight vector — including the legitimate framework case where
  // the active subset rotates inside a stable expert pool (compact-
  // form callers).  Two consequences for the previous design:
  //   * compact callers paid a re-warm on every rotation even when
  //     the underlying pool was unchanged (Fix-A's process-wide
  //     cache absorbed the multi-thread fan-out, but not the
  //     per-call rotation cost on a single thread);
  //   * three samples hash-collided on different subsets that
  //     happened to agree at `[0] / [n/2] / [n-1]`, allowing rare
  //     incorrect cache reuse.
  // XOR over the full pool fixes both: identical pointer SETS
  // produce identical hashes regardless of permutation, and every
  // pool member contributes its bits so subset changes are reliably
  // observable.  Cost is O(num_ops_total) per call — for a 32-
  // expert MoE block that's ~32 8-byte XORs ≈ 30 ns, dwarfed by
  // the matmul body that follows.  `uintptr_t` is the canonical
  // pointer-to-integer alias (defined in <cstdint>); the previous
  // `reinterpret_cast<size_t>` was implementation-defined on non-
  // LP64 targets.
  //
  // The same rationale extends to the per-expert dim/stride
  // metadata (`K`, `N`, `ldb`, `transB`).  The previous version
  // sampled only `[0]` of `K` and `N`, which is correct for the
  // typical MoE case (uniform K/N within a layer) but breaks two
  // edge cases:
  //   * non-uniform shapes (legitimately supported by the per-
  //     expert vectors): permutation flips `K[0]/N[0]` and forces
  //     a spurious re-warm even though the underlying multiset is
  //     unchanged;
  //   * a caller that mutates `K[i]/N[i]/ldb[i]/transB[i]` for
  //     some `i > 0` while keeping the pointer set identical: the
  //     fingerprint would NOT flip and the warmer would short-
  //     circuit, leaving the new shape un-warmed and re-introducing
  //     the reorder-spike that prepack is designed to absorb.
  // Roll all four into the per-expert XOR reduction (alongside the
  // weight pool above) so the fingerprint sees the entire pool's
  // (ptr, K, N, ldb, transB) shape regardless of order.  Bool-as-
  // size_t coerces 0/1 — XOR'ing it across the pool encodes the
  // count parity of `transB=true` slots, which is sufficient: the
  // dispatcher requires the layout to be uniform-typed within a
  // call, so any `transB` change at all flips the per-expert XOR.
  if (p.weight && !p.weight->empty()) {
    const size_t bound =
        std::min<size_t>(static_cast<size_t>(p.num_ops_total),
                         p.weight->size());
    uintptr_t ptr_xor    = 0;
    uintptr_t ptr_sum    = 0;
    size_t    k_xor      = 0;
    size_t    k_sum      = 0;
    size_t    n_xor      = 0;
    size_t    n_sum      = 0;
    size_t    ldb_xor    = 0;
    size_t    ldb_sum    = 0;
    size_t    transb_xor = 0;
    for (size_t i = 0; i < bound; ++i) {
      const uintptr_t pi = reinterpret_cast<uintptr_t>((*p.weight)[i]);
      ptr_xor ^= pi;
      ptr_sum += pi;
      if (p.K  && i < p.K->size()) {
        const size_t v = static_cast<size_t>((*p.K)[i]);
        k_xor ^= v; k_sum += v;
      }
      if (p.N  && i < p.N->size()) {
        const size_t v = static_cast<size_t>((*p.N)[i]);
        n_xor ^= v; n_sum += v;
      }
      if (p.ldb && i < p.ldb->size()) {
        const size_t v = static_cast<size_t>((*p.ldb)[i]);
        ldb_xor ^= v; ldb_sum += v;
      }
      if (p.transB && i < p.transB->size())
        transb_xor ^= static_cast<size_t>((*p.transB)[i] ? 1u : 0u);
    }
    // XOR is permutation-invariant (the rotation-immunity we want for
    // compact-form active-subset rotations) but loses information on
    // uniform-shape pools: `K^K=0`, and two pointer pairs `(A,B)` and
    // `(A+d, B+d)` give the same XOR `A^B` because the addend `d`
    // cancels.  That collision matters for fused-MoE callers: within
    // a single `group_matmul_direct` invocation the Op1 and Op2
    // dispatchers each call `prepack_for_algo_3` with their own
    // (weight_pool, K_vec, N_vec) — uniform shape, same num_ops_total,
    // same num_threads / nr_align — and the *only* signal that
    // distinguishes their fingerprints is the weight pool's pointer
    // pattern.  Same-size-allocation pools placed in sequence by
    // malloc produce identical pair-wise XORs, so the second pass's
    // `already_warmed` returns true on a stale Op1 entry and Op2's
    // warmer is silently skipped — the failure mode the
    // `TestPrepackKDownSynthesis.BothPassesWarmAllExperts/act_none_E2_N1_64`
    // case surfaced.  Fold the COMMUTATIVE SUM alongside the XOR for
    // every per-expert field so the joint signature requires BOTH
    // the XOR AND the sum to collide — preserving rotation immunity
    // while catching the uniform-shape malloc-aliased-pool case.
    s = mix_hash(s, static_cast<size_t>(ptr_xor));
    s = mix_hash(s, static_cast<size_t>(ptr_sum));
    s = mix_hash(s, k_xor);
    s = mix_hash(s, k_sum);
    s = mix_hash(s, n_xor);
    s = mix_hash(s, n_sum);
    s = mix_hash(s, ldb_xor);
    s = mix_hash(s, ldb_sum);
    s = mix_hash(s, transb_xor);
    // Fold the iteration count in too so an empty-vs-non-empty or
    // size-mismatch case cannot collide with all-zero XORs (e.g.
    // all-null pool, or a pool whose XORs happen to be zero by
    // chance).  Using `bound` (not `weight->size()`) keeps the
    // hash invariant under per-call resizing of the trailing
    // padding region above `num_ops_total`.
    s = mix_hash(s, bound);
  }
  // Per-tile AOCL DLP warm depends on (num_threads, nr_align) — a
  // change in either rotates the per-tile cache keys, so we MUST
  // re-warm.  Fold both into the fingerprint so a process that hot-
  // swaps OMP team size mid-run (rare; possible in mixed-team-size
  // host applications) gets correct cache coverage on the second
  // call after the swap.
  s = mix_hash(s, static_cast<size_t>(p.num_threads));
  s = mix_hash(s, static_cast<size_t>(p.nr_align));
  // Fold every dtype context bit `ck_eligible(p)` reads into the
  // fingerprint so a process that toggles between layers with
  // different (src, wei, dst, act, act_dtype, bias) dtype tuples
  // re-warms on each unique tuple instead of caching the first
  // verdict.  Without these terms, e.g., a process running first
  // with dst=bf16 and then dst=f32 (both legal under
  // `kBF16_BF16_BF16` and `kBF16_BF16_F32` resolve_variant entries)
  // would short-circuit the second call with the first call's
  // eligibility verdict — benign for non-gated act (the pack is
  // dst-agnostic) but the cross_warm regime decision for gated +
  // f32 (refused) differs from gated + bf16 (eligible), and the
  // Fix-B per-tile-AOCL-skip would apply the wrong logic.  Cost:
  // six mix_hash calls (~15 ns), negligible vs the matmul body.
  s = mix_hash(s, static_cast<size_t>(p.src_dtype));
  s = mix_hash(s, static_cast<size_t>(p.wei_dtype));
  s = mix_hash(s, static_cast<size_t>(p.dst_dtype));
  s = mix_hash(s, static_cast<size_t>(p.act));
  s = mix_hash(s, static_cast<size_t>(p.act_dtype));
  s = mix_hash(s, static_cast<size_t>(p.bias_dtype));
  // DQ-INT8 discriminators — a DQ-INT8 warm and a bf16-only warm
  // share `(weight_ptr, K, N, transB)` but pack into PHYSICALLY
  // DIFFERENT arenas (K/4 VNNI-quad + compensation row vs K/2 VNNI
  // pair) keyed by separate LRU singletons.  Without folding both
  // bits into the fingerprint, a process that runs first under
  // bf16 (warms regime 3 with `get_or_pack_weight_bf16`) and then
  // under DQ-INT8 (would have called `get_or_pack_weight_int8`)
  // would short-circuit the second call on the bf16 fingerprint
  // and never warm the int8 pack cache.  Fold both terms here so
  // the two regimes are independent fingerprints from the start.
  s = mix_hash(s, static_cast<size_t>(p.dynamic_quant ? 1u : 0u));
  s = mix_hash(s, static_cast<size_t>(p.compute_dtype));
  // Fold the runtime-mutable weight-cache toggle into the fingerprint.
  // `matmul_config_t::set_weight_cache(...)` can flip this mid-process
  // and BOTH the AOCL DLP warmer (`prepack_aocl_dlp.cpp`) AND the
  // custom-kernel warmer (`prepack_custom_kernel.cpp`) now gate on
  // it at entry: when the env knob `ZENDNNL_MATMUL_WEIGHT_CACHE` is
  // not 1, neither warmer touches its respective cache.  Without
  // this hash term, a process that runs first under WEIGHT_CACHE=0
  // (warmers no-op) and then enables it via `set_weight_cache(1)`
  // would short-circuit on the second call (fingerprint already
  // present) and leave both caches permanently empty for this thread.
  // The custom-kernel runtime in turn switches to caller-owned packs
  // (`custom_kernel/dispatch.cpp::prepare_for_call` →
  // `get_or_pack_weight_bf16(..., disable_cache=true)`) when the
  // toggle is non-1, so the kernel math keeps running on a fresh
  // per-call buffer instead of the LRU singleton; the AOCL DLP
  // runtime independently honours the same toggle via its
  // `weight_cache_type=0` branch (fresh reorder + free per call).
  // Folding the toggle into the hash on every call is one
  // singleton-load (~1 ns) and keeps the hash regime simple.
  s = mix_hash(s, static_cast<size_t>(
      zendnnl::ops::matmul_config_t::instance().get_weight_cache()));
  return s;
}

// Weight-pool IDENTITY fingerprint: hashes ONLY the physical weight pool
// (pointers + per-expert K/N/ldb/transB + expert count) plus the process
// weight-cache mode.  It deliberately OMITS the per-call tuning knobs that
// `fingerprint()` also folds in (scheduling_algo, num_threads, nr_align,
// src/wei/dst dtype, act, act_dtype, bias_dtype, dynamic_quant,
// compute_dtype).
//
// This is the key for the AUTO mixed-in-place warm latch + completion record.
// The in-place mutation is a property of the WEIGHT BUFFER, not of any tuning
// context: a prompt call (ALGO 1/2/4/5) and a decode call (ALGO 3) — or two
// calls differing only in thread count / tile alignment — that SHARE the same
// weight buffer must contend on the SAME latch, otherwise one could read or
// out-of-place-reorder a half-mutated buffer while the other is mid in-place
// mutation.  Keying on the full `fingerprint()` (even with scheduling_algo=0)
// would split them by num_threads/nr_align/dtype context and re-open that
// race.
//
// NOTE: the XOR+SUM reduction below mirrors `fingerprint()` (see its comments
// for the permutation-invariance + malloc-aliasing collision rationale); keep
// the two in sync if the weight-pool hashing changes.
inline size_t weight_pool_fingerprint(const PrepackParams &p) {
  size_t s = mix_hash(0, static_cast<size_t>(p.num_ops_total));
  if (p.weight && !p.weight->empty()) {
    const size_t bound =
        std::min<size_t>(static_cast<size_t>(p.num_ops_total),
                         p.weight->size());
    uintptr_t ptr_xor = 0, ptr_sum = 0;
    size_t k_xor = 0, k_sum = 0, n_xor = 0, n_sum = 0,
           ldb_xor = 0, ldb_sum = 0, transb_xor = 0;
    for (size_t i = 0; i < bound; ++i) {
      const uintptr_t pi = reinterpret_cast<uintptr_t>((*p.weight)[i]);
      ptr_xor ^= pi; ptr_sum += pi;
      if (p.K && i < p.K->size()) {
        const size_t v = static_cast<size_t>((*p.K)[i]); k_xor ^= v; k_sum += v;
      }
      if (p.N && i < p.N->size()) {
        const size_t v = static_cast<size_t>((*p.N)[i]); n_xor ^= v; n_sum += v;
      }
      if (p.ldb && i < p.ldb->size()) {
        const size_t v = static_cast<size_t>((*p.ldb)[i]);
        ldb_xor ^= v; ldb_sum += v;
      }
      if (p.transB && i < p.transB->size())
        transb_xor ^= static_cast<size_t>((*p.transB)[i] ? 1u : 0u);
    }
    s = mix_hash(s, static_cast<size_t>(ptr_xor));
    s = mix_hash(s, static_cast<size_t>(ptr_sum));
    s = mix_hash(s, k_xor);
    s = mix_hash(s, k_sum);
    s = mix_hash(s, n_xor);
    s = mix_hash(s, n_sum);
    s = mix_hash(s, ldb_xor);
    s = mix_hash(s, ldb_sum);
    s = mix_hash(s, transb_xor);
    s = mix_hash(s, bound);
  }
  s = mix_hash(s, static_cast<size_t>(
      zendnnl::ops::matmul_config_t::instance().get_weight_cache()));
  return s;
}

// Process-wide fingerprint set, protected by a mutex.
//
// Was previously `thread_local`.  That worked for benchdnn (single
// caller thread) and the gtest harness, but frameworks that
// dispatch from multiple worker threads (multi-rank serving stacks,
// async pipelines) saw catastrophic redundant warm-pack: each
// worker's thread_local set started empty, so each worker paid the
// "first warm" cost in cache-LOOKUP work on every unique
// (layer, pass) fingerprint — generating a large number of extra
// "Read AOCL cached weights" log lines per benchmark run, a
// corresponding `apilog_info` formatting overhead, and wasted lookup
// cycles inside `reorderAndCacheWeights`.  The
// underlying caches (AOCL DLP LRU, custom-kernel pack arena) are
// process-wide and were already populated by some other thread's
// first warm; only the fingerprint short-circuit was missing.
//
// Moving the set to process scope + mutex resolves it: any thread's
// first encounter with a fingerprint records it once for all other
// threads.  Steady-state cost on the warm path: one `lock_guard`
// + one `unordered_set::insert` (lookup-only after the first call)
// ≈ 80-150 ns.  Vs the ~ms-class matmul body, the overhead is
// imperceptible — and replaces the per-call 256-lookup
// `warm_aocl_n_tile` cost that was previously paid for every
// thread × fingerprint pair.
//
// `clear_fingerprint_cache_for_test()` below acquires the same
// mutex and clears the set; gtest cases that need a fresh state
// across cases call it from their `SetUp()`.
static std::mutex                 s_warmed_fps_mtx;
static std::unordered_set<size_t> s_warmed_fps;

inline bool already_warmed(const PrepackParams &p, int scheduling_algo) {
  const size_t fp = fingerprint(p, scheduling_algo);
  // `insert(fp).second` is true on insertion (NEW fp) and false when
  // the fp was already present.  Negating gives "was-already-warmed".
  std::lock_guard<std::mutex> lk(s_warmed_fps_mtx);
  return !s_warmed_fps.insert(fp).second;
}

// ── Per-fingerprint warm latch (AUTO mixed-in-place mode ONLY) ─────────
// In the default out-of-place mode the warm mutates nothing shared, so
// the cheap insert-and-go `s_warmed_fps` set above is sufficient: a
// concurrent same-fingerprint call may skip warming and compute lazily
// with no correctness risk.
//
// In the AUTO mixed-in-place mode the warm MUTATES the caller's weight
// buffer in place (the AOCL full-weight prompt reorder).  A concurrent
// same-fingerprint call must therefore BLOCK until that warm completes,
// or it would compute on a half-mutated buffer.  This latch implements
// "first caller warms, all other same-fingerprint callers wait": the
// first caller for a fingerprint creates and owns the latch (returned in
// the prelude's `WarmGuard`, signalled when the per-algo warm function
// returns); concurrent callers find the existing latch and wait on it.
struct WarmLatch {
  std::mutex              m;
  std::condition_variable cv;
  bool                    done = false;
};
static std::mutex s_warm_latch_mtx;
// IN-FLIGHT warms only: a fingerprint's latch lives here while its warm is
// running and is ERASED the moment the warm completes (see WarmGuard::signal).
static std::unordered_map<size_t, std::shared_ptr<WarmLatch>> s_warm_latch_map;
// COMPLETED warms: the lightweight (size_t) record that a fingerprint's
// in-place warm has finished.  Guarded by `s_warm_latch_mtx` (same mutex as
// the in-flight map) so the check + map ops are one critical section.  This
// is what bounds memory: once warmed, the heavyweight WarmLatch (mutex+cv) is
// dropped and only an 8-byte fingerprint is retained — so workloads with many
// distinct fingerprints no longer accumulate a latch per fingerprint forever.
static std::unordered_set<size_t> s_warm_done_fps;

// RAII handle: the warming caller holds it for the duration of its warm
// (it lives inside the per-algo function's `PreludeResult`); on
// destruction it marks the latch done and releases any waiters.  Move-only
// so ownership is unambiguous; a default-constructed (null) guard is the
// no-op case for the skip / out-of-place paths.
struct WarmGuard {
  std::shared_ptr<WarmLatch> latch;
  size_t                     fp = 0;  // fingerprint this guard's warm owns
  WarmGuard() = default;
  WarmGuard(std::shared_ptr<WarmLatch> l, size_t fp_)
      : latch(std::move(l)), fp(fp_) {}
  WarmGuard(const WarmGuard &) = delete;
  WarmGuard &operator=(const WarmGuard &) = delete;
  WarmGuard(WarmGuard &&o) noexcept : latch(std::move(o.latch)), fp(o.fp) {}
  WarmGuard &operator=(WarmGuard &&o) noexcept {
    if (this != &o) { signal(); latch = std::move(o.latch); fp = o.fp; }
    return *this;
  }
  ~WarmGuard() { signal(); }
  void signal() {
    if (!latch) return;
    // Record completion and DROP the in-flight latch under the map mutex
    // BEFORE waking waiters: a brand-new same-fingerprint caller then either
    // sees `s_warm_done_fps` (skip, no mutex/cv) or — if it raced in just
    // before this — finds the still-present latch and waits.  Erasing here is
    // what keeps `s_warm_latch_map` bounded to in-flight warms.  Any waiter
    // already blocked holds its own shared_ptr, so the latch stays alive for
    // its `cv.wait` regardless of the erase.
    {
      std::lock_guard<std::mutex> lk(s_warm_latch_mtx);
      s_warm_done_fps.insert(fp);
      s_warm_latch_map.erase(fp);
    }
    {
      std::lock_guard<std::mutex> lk(latch->m);
      latch->done = true;
    }
    latch->cv.notify_all();
    latch.reset();
  }
};

// Reason the prelude short-circuited (when `skip == true`).  Used by
// the skip-path apilog emitter to print a descriptive `state=...`
// field so a level-3 reader can tell "first call, env-disabled" from
// "subsequent call, fingerprint cached".  Order matches the gate
// order inside `prelude()`.
enum class PreludeSkipReason {
  none                 = 0,  ///< `skip == false`; ignore this field
  env_disabled         = 1,  ///< `ZENDNNL_GRP_MATMUL_PREPACK = 0`
  fingerprint_already  = 2,  ///< `already_warmed(...) == true`
};

// Result of the shared opening sequence: one of `skip` (the per-ALGO
// function should return immediately) or a resolved inner kernel.  In the
// AUTO mixed-in-place mode the warming caller also receives a `guard`
// whose destruction (when the per-algo warm function returns) releases
// any concurrent callers waiting on the same fingerprint.
struct PreludeResult {
  bool              skip          = true;
  matmul_algo_t     inner_kernel  = matmul_algo_t::none;
  PreludeSkipReason skip_reason   = PreludeSkipReason::env_disabled;
  WarmGuard         guard;  // non-null only for the warming caller (mixed)
};

inline PreludeResult prelude(const PrepackParams &p, int scheduling_algo) {
  PreludeResult r;
  if (!get_grp_matmul_prepack()) {
    r.skip_reason = PreludeSkipReason::env_disabled;
    return r;
  }
  // NOTE: the previous `if (p.num_ops_total <= p.num_ops_active) return r;`
  // gate was removed.  Under the uniform-eager semantic, PREPACK=ON
  // ALWAYS warms `max(M.size(), total_matmul)` experts up front,
  // regardless of whether the framework opted into the
  // `total_matmul > active_matmul` contract.  Set
  // `ZENDNNL_GRP_MATMUL_PREPACK=0` to restore the lazy-only path.
  const bool mixed_inplace = is_grp_auto_mixed_inplace_active();

  if (!mixed_inplace) {
    // Out-of-place fast path: insert-and-go.  A concurrent skipper may
    // compute lazily; nothing shared is mutated, so there is no race.
    if (already_warmed(p, scheduling_algo)) {
      r.skip_reason = PreludeSkipReason::fingerprint_already;
      return r;
    }
    r.skip         = false;
    r.inner_kernel = resolve_kernel();
    return r;
  }

  // Mixed in-place: serialise via the per-fingerprint latch so a
  // concurrent same-W call never computes (or reads W for its own
  // out-of-place warm) while the in-place warm is still mutating the
  // weight buffer.
  //
  // Key on the WEIGHT-POOL IDENTITY fingerprint (NOT the full `fingerprint()`).
  // In mixed mode the prompt (ALGO 1/2/4/5) and the decode (ALGO 3) prepacks
  // BOTH touch the SAME weight buffers and BOTH do an in-place full-weight
  // AOCL reorder of them: the prompt as its primary warm, the decode via
  // `cross_warm` -> `warm_aocl` (warm_wct=2).  The latch must serialise EVERY
  // call that shares those weights, so its key must depend only on weight
  // identity — `fingerprint()` (even with scheduling_algo=0) also folds in
  // num_threads / nr_align / dtype / act context, which would split same-W
  // callers that differ in any of those into DISTINCT latches and let one
  // read/reorder W while another is mid in-place mutation (the AOCL reorder
  // cache mutex only serialises the AOCL reorder itself, not a concurrent CK
  // pack reading W).  `weight_pool_fingerprint` keys on (weight pool + WC
  // mode) only, so one latch + one completion record covers all of them.
  const size_t fp = weight_pool_fingerprint(p);
  std::shared_ptr<WarmLatch> latch;
  bool i_warm = false;
  {
    std::lock_guard<std::mutex> lk(s_warm_latch_mtx);
    // Already warmed (in-place mutation already done on a prior call)?  Skip
    // with NO latch: the completed set is the only state retained for a
    // warmed fingerprint; its latch was erased when the warm finished.
    if (s_warm_done_fps.count(fp) != 0) {
      r.skip_reason = PreludeSkipReason::fingerprint_already;
      return r;
    }
    auto it = s_warm_latch_map.find(fp);
    if (it == s_warm_latch_map.end()) {
      latch  = std::make_shared<WarmLatch>();
      s_warm_latch_map.emplace(fp, latch);
      i_warm = true;
    } else {
      latch = it->second;
    }
  }
  if (i_warm) {
    // This caller owns the warm; the guard signals the latch (and records
    // completion + erases the in-flight entry) when the per-algo warm
    // function returns (warm + in-place mutation done).
    r.skip         = false;
    r.inner_kernel = resolve_kernel();
    r.guard        = WarmGuard(std::move(latch), fp);
    return r;
  }
  // Another caller is warming (or already finished): block until done,
  // then skip and compute against the now-stable (mutated) buffer.
  {
    std::unique_lock<std::mutex> lk(latch->m);
    latch->cv.wait(lk, [&] { return latch->done; });
  }
  r.skip_reason = PreludeSkipReason::fingerprint_already;
  return r;
}

// AOCL DLP backend wrapper — calls the existing FULL-WEIGHT warmer
// with the per-call vectors taken straight from `PrepackParams`.
// Used by ALGOs 1, 2, 4, 5 (no column tiling) and by ALGO 3's
// fallback paths (STABLE_NTILE off, narrow-N escape, missing thread
// context).  Returns the `packed_ok` count for the apilog probe line.
// Resolve the `is_weights_const` vector for a warm-pack call.  An
// absent vector (`nullptr`) is the documented "treat every expert as
// const" sentinel; forward a shared empty vector so the warmer's
// per-expert gate stays correct.  Shared by all warm_* wrappers.
inline const std::vector<bool> &warm_iwc(const PrepackParams &p) {
  static const std::vector<bool> kEmptyIsConst;
  return (p.is_weights_const != nullptr) ? *p.is_weights_const
                                         : kEmptyIsConst;
}

inline aocl_dlp::AoclDlpPackProbeStats warm_aocl(const PrepackParams &p) {
  aocl_dlp::AoclDlpPackProbeStats st;
  aocl_dlp::warm_pack_all_aocl_dlp_experts(
      *p.weight, *p.K, *p.N, *p.ldb, *p.transB, warm_iwc(p),
      p.num_ops_total, p.wei_dtype, st);
  return st;
}

// DQ-INT8 sym-quant backend wrapper (Gap B).  Calls the s8 sym-quant
// full-weight warmer with the per-call vectors from `PrepackParams`,
// populating the dedicated sym-quant LRU so ALGO 3 decode -> ALGO 1
// prompt cross-warm no longer leaves the first prompt call paying the
// lazy sym-quant reorder.
inline aocl_dlp::AoclDlpPackProbeStats warm_aocl_sym_quant(
    const PrepackParams &p) {
  aocl_dlp::AoclDlpPackProbeStats st;
  aocl_dlp::warm_pack_all_aocl_dlp_experts_sym_quant(
      *p.weight, *p.K, *p.N, *p.ldb, *p.transB, warm_iwc(p),
      p.num_ops_total, p.wei_dtype, st);
  return st;
}

// Per-tile variant for ALGO 3's strict-stable plan.  See
// `warm_pack_all_aocl_dlp_experts_n_tile` doc-comment in
// prepack_aocl_dlp.hpp for the decomposition.
//
// `stable` and `nr_align` are pre-validated by the caller so the
// per-tile path is only entered when:
//   * `ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE = 1` (the planner takes
//     the strict-stable branch), AND
//   * `stable * nr_align <= max_N` (no narrow-N escape — the
//     planner's `ManyExperts` strategy fires, not `Sequential`).
inline aocl_dlp::AoclDlpPackProbeStats warm_aocl_n_tile(
    const PrepackParams &p, int stable, int nr_align_eff) {
  aocl_dlp::AoclDlpPackProbeStats st;
  aocl_dlp::warm_pack_all_aocl_dlp_experts_n_tile(
      *p.weight, *p.K, *p.N, *p.ldb, *p.transB, warm_iwc(p),
      p.num_ops_total, p.wei_dtype,
      p.num_threads, stable, nr_align_eff, st);
  return st;
}

// DQ-INT8 sym-quant PER-TILE wrapper — the int8 sibling of
// `warm_aocl_n_tile`.  Used by ALGO 3 (and its cross-warm) when the
// call is DQ-INT8 (`wei == s8`) and the custom kernel is OFF /
// CK-ineligible, so decode falls back to the AOCL DLP sym-quant
// reorder per N-tile.  Same `stable` / `nr_align_eff` pre-validation
// contract as `warm_aocl_n_tile`.
inline aocl_dlp::AoclDlpPackProbeStats warm_aocl_n_tile_sym_quant(
    const PrepackParams &p, int stable, int nr_align_eff) {
  aocl_dlp::AoclDlpPackProbeStats st;
  aocl_dlp::warm_pack_all_aocl_dlp_experts_n_tile_sym_quant(
      *p.weight, *p.K, *p.N, *p.ldb, *p.transB, warm_iwc(p),
      p.num_ops_total, p.wei_dtype,
      p.num_threads, stable, nr_align_eff, st);
  return st;
}

// Predicate: is this an int8 AOCL-DLP sym-quant warm candidate?  The
// AOCL `s8s8s32os32_sym_quant` reorder is selected by an s8 weight with
// an s8/u8 compute dtype — the DQ-INT8 path whether the src arrives as
// bf16 (runtime hoist, `dynamic_quant=true`) or already-s8
// (`group_dynamic_quant` pre-pass, which CLEARS `dynamic_quant`).  Key
// on `wei==s8 && compute∈{s8,u8}`, NOT on `dynamic_quant` (false post-
// group-DQ — the default production path).  `compute_dtype` is set for
// both forms by `build_prepack_params`.  Used at every AOCL warm site
// to pick the sym-quant warmer over the bf16 one; independent of CK
// eligibility (the caller already decided AOCL, not CK, is the path).
inline bool int8_aocl_warm_candidate(const PrepackParams &p) {
  return p.wei_dtype == data_type_t::s8
      && (p.compute_dtype == data_type_t::s8
          || p.compute_dtype == data_type_t::u8);
}

// Compute max(N[i]) over `[0, num_ops_total)`.  Mirrors what the
// runtime planner's `summarise_topology(...)` produces for
// `topo.max_N`, except the planner only sees active-set N's
// (`num_ops_active`) while this warmer iterates the full
// `[0, num_ops_total)` range — the prepack-extras tail might have
// different N's in pathological framework setups, so we take the
// conservative max here to keep the narrow-N escape decision safe.
inline int compute_max_n(const PrepackParams &p) {
  if (p.N == nullptr || p.N->empty()) return 0;
  const size_t bound = std::min<size_t>(
      static_cast<size_t>(p.num_ops_total), p.N->size());
  int m = 0;
  for (size_t i = 0; i < bound; ++i) {
    m = std::max(m, (*p.N)[i]);
  }
  return m;
}

// Custom-kernel BF16 pack backend wrapper.  Forwards `is_weights_const`
// (when present) so the warmer's per-expert variable-weight skip
// matches the AOCL DLP wrapper above and the runtime CK refusal in
// `custom_kernel/dispatch.cpp::prepare_for_call`.  Empty vector =
// "treat every entry as const" (legacy callers that don't supply
// the field).
//
// `interleave_split_halves` mirrors the dispatcher's flag: for
// `silu_and_mul` and `gelu_and_mul` the canonical W13 weight is in
// split-halves layout and the pack permutes source columns to
// produce the same interleaved arena the in-register fused
// epilogue expects.  Other activations (none, swiglu_oai_mul) pack
// verbatim.  silu and gelu share the same permutation — only the
// kernel's activation math differs.
inline custom_kernel::PackProbeStats warm_custom(const PrepackParams &p) {
  custom_kernel::PackProbeStats st;
  const std::vector<bool> &iwc = warm_iwc(p);
  const bool interleave_split_halves =
      (p.act == grp_matmul_gated_act_t::silu_and_mul)
      || (p.act == grp_matmul_gated_act_t::gelu_and_mul);
  // Family selection — DQ-INT8 fingerprint includes `dynamic_quant`
  // and `compute_dtype`, so this branch is fingerprint-aware and a
  // single process can warm both families across distinct calls
  // (e.g. a bf16-only layer followed by a DQ-INT8 layer) without
  // either warm aliasing the other.  Use the same int8 discriminator
  // as `ck_eligible_int8` / `int8_aocl_warm_candidate`
  // (`wei==s8 && compute in {s8,u8}`) — NOT `dynamic_quant`, which the
  // group_dynamic_quant pre-pass CLEARS (default production path).
  // Keying on `dynamic_quant` here routed grouped-s8 (src=s8,
  // dynamic_quant=false) to the bf16 pack arena, so the int8 CK pack
  // LRU was never warmed for the production decode path.
  const custom_kernel::WarmDtypeFamily family =
      int8_aocl_warm_candidate(p)
          ? custom_kernel::WarmDtypeFamily::kINT8
          : custom_kernel::WarmDtypeFamily::kBF16;
  custom_kernel::warm_pack_all_custom_kernel_experts(
      *p.weight, *p.K, *p.N, *p.ldb, *p.transB, iwc,
      p.num_ops_total, st, interleave_split_halves, family);
  return st;
}

// Test-only accumulator written by `log_pack_probe` and read by the
// `test_api::get_last_invocation_stats()` getter below.  Production
// code never observes it; test code uses it to assert e.g. that Fix B
// skipped the AOCL DLP per-tile warm under CK=1 (the discriminator is
// `aocl.total_attempted` — would be `N × stable_n_thr` if regime 2
// was warmed, drops to `N` (cross-warm regime 1 only) under Fix B).
static std::mutex                   s_last_invocation_mtx;
static test_api::LastInvocationStats s_last_invocation;

// Map `CrossWarmRegime` → human-readable string used by the apilog
// emitter and the `next_HIT_for=[...]` helper below.
inline const char *cross_warm_regime_name(CrossWarmRegime r) {
  switch (r) {
  case CrossWarmRegime::none:                       return "none";
  case CrossWarmRegime::aocl_full_weight:           return "aocl_full_weight";
  case CrossWarmRegime::custom_kernel_pack:         return "custom_kernel_pack";
  case CrossWarmRegime::aocl_per_tile:              return "aocl_per_tile";
  case CrossWarmRegime::aocl_full_weight_sym_quant: return "aocl_full_weight_sym_quant";
  case CrossWarmRegime::aocl_per_tile_sym_quant:    return "aocl_per_tile_sym_quant";
  }
  return "?";
}

// Compose the `next_HIT_for=[...]` field describing which inverse-algo
// runtime call is guaranteed to hit cache given the (primary,
// cross_warm) pair this invocation just warmed.  The strings are kept
// fixed and short for downstream log greps.
//
// Encoding:
//   primary kind            cross_warm regime          → next_HIT_for
//   ck_pack                 aocl_full_weight           → [ALGO_3+CK_decode, ALGO_1+DLP_prompt]
//   ck_pack                 none                       → [ALGO_3+CK_decode]
//   aocl_full_weight        custom_kernel_pack         → [ALGO_3+CK_decode, ALGO_1+DLP_prompt]
//   aocl_full_weight        aocl_per_tile              → [ALGO_3+DLP_decode, ALGO_1+DLP_prompt]
//   aocl_full_weight        none                       → [ALGO_1+DLP_prompt]
//   aocl_per_tile           aocl_full_weight           → [ALGO_3+DLP_decode, ALGO_1+DLP_prompt]
//   aocl_per_tile           none                       → [ALGO_3+DLP_decode]
//   any                     anything else              → fallback: "(see escapes)"
inline const char *next_hit_for_label(const char *primary,
                                      CrossWarmRegime cross) {
  if (primary == nullptr) return "[]";
  if (std::strcmp(primary, "ck_pack") == 0) {
    // BF16 CK pack + bf16 AOCL full-weight cross-warm
    //   → both decode (ALGO 3 CK) and prompt (ALGO 1 DLP) hit.
    // DQ-INT8 CK pack + sym-quant AOCL full-weight cross-warm
    //   → decode hits (ALGO 3 CK pack) AND prompt hits (the
    //     sym-quant LRU is now eagerly warmed via `warm_aocl_sym_quant`
    //     — see the `aocl_full_weight_sym_quant` branch in
    //     `cross_warm`).
    if (cross == CrossWarmRegime::aocl_full_weight
        || cross == CrossWarmRegime::aocl_full_weight_sym_quant) {
      return "[ALGO_3+CK_decode, ALGO_1+DLP_prompt]";
    }
    return "[ALGO_3+CK_decode]";
  }
  // Full-weight primary — bf16 and the DQ-INT8 sym-quant sibling share
  // the same next-hit semantics (the DLP prompt path that hits is the
  // bf16 LRU resp. the sym-quant LRU).
  if (std::strcmp(primary, "aocl_full_weight") == 0
      || std::strcmp(primary, "aocl_full_weight_sym_quant") == 0) {
    switch (cross) {
    case CrossWarmRegime::custom_kernel_pack:
      return "[ALGO_3+CK_decode, ALGO_1+DLP_prompt]";
    case CrossWarmRegime::aocl_per_tile:
    case CrossWarmRegime::aocl_per_tile_sym_quant:
      return "[ALGO_3+DLP_decode, ALGO_1+DLP_prompt]";
    default:
      return "[ALGO_1+DLP_prompt]";
    }
  }
  // Per-tile primary — bf16 and the DQ-INT8 sym-quant sibling alike.
  if (std::strcmp(primary, "aocl_per_tile") == 0
      || std::strcmp(primary, "aocl_per_tile_sym_quant") == 0) {
    return (cross == CrossWarmRegime::aocl_full_weight
            || cross == CrossWarmRegime::aocl_full_weight_sym_quant)
        ? "[ALGO_3+DLP_decode, ALGO_1+DLP_prompt]"
        : "[ALGO_3+DLP_decode]";
  }
  return "[]";
}

// Format a fingerprint as a fixed-width "0xHHHHHHHHHHHHHHHH" string so
// the apilog stream can take it as a plain `const char *`.  Avoids
// fighting the logger's stream-state manipulators (`std::hex` toggles
// the stream globally which would corrupt subsequent numeric fields).
inline std::string format_fingerprint(size_t fp) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "0x%016zx",
                static_cast<std::size_t>(fp));
  return std::string(buf);
}

// Single PREPACK log line shared across the five per-ALGO functions so
// downstream tooling sees the same field set regardless of which
// scheduling algo was warmed.  Also records the per-invocation
// accumulator for the test-only API in `prepack.hpp::test_api::`.
//
// Format (level 3, info):
//   [GRP_MATMUL.PREPACK] for=ALGO_N state=warmed active=A total=T
//     primary=<label>     ck=[hits= misses= skipped=]
//                         aocl=[packed= skipped=]
//     cross_warm=<enabled|disabled> regime=<regime>
//     next_HIT_for=[<paths>]
//     fingerprint=0x<hex>
//
// `primary_label` is supplied by the per-ALGO body — it knows which
// warmer fired first ("ck_pack", "aocl_full_weight", "aocl_per_tile",
// or "none").  `next_HIT_for=[...]` is computed from the
// (primary_label, cross_warm_regime) pair; see the encoding table in
// `next_hit_for_label()` above.
inline void log_pack_probe(int scheduling_algo,
                           const PrepackParams &p,
                           matmul_algo_t inner_kernel,
                           const aocl_dlp::AoclDlpPackProbeStats &st_aocl,
                           const custom_kernel::PackProbeStats   &st_ck,
                           CrossWarmRegime regime,
                           const char *primary_label) {
  // Test-only stats accumulator.  Gated by the `s_capture_last_invocation`
  // atomic so production builds short-circuit on a single relaxed load
  // — no mutex lock, no struct copy, no coherence ping-pong on the
  // hot fused-MoE dispatch path.  Tests opt in for the scope of the
  // test via `LastInvocationCaptureGuard` in `moe_test_utils.hpp`; in
  // that scope the gated branch fires, takes the mutex, and writes
  // through to `s_last_invocation` so the existing
  // `get_last_invocation_stats()` accessor returns the live data.
  // Mirror of the `s_capture_gemm_mode` gate that protects
  // `s_last_group_matmul_direct_gemm_mode` (see the doc-block on that
  // atomic in `group_matmul_parallel_common.hpp`).
  if (test_api::s_capture_last_invocation.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lk(s_last_invocation_mtx);
    s_last_invocation.scheduling_algo   = scheduling_algo;
    s_last_invocation.inner_kernel      = inner_kernel;
    s_last_invocation.aocl              = st_aocl;
    s_last_invocation.ck                = st_ck;
    s_last_invocation.cross_warm_regime = regime;
    s_last_invocation.valid             = true;
  }

  static const bool s_l3_log = apilog_info_enabled();
  if (!s_l3_log) return;
  const bool cross_enabled = (regime != CrossWarmRegime::none);
  const std::string fp_str = format_fingerprint(fingerprint(p, scheduling_algo));
  apilog_info(
      "[GRP_MATMUL.PREPACK] for=ALGO_", scheduling_algo,
      " state=warmed active=", p.num_ops_active,
      " total=", p.num_ops_total,
      " primary=", primary_label,
      " ck=[hits=", st_ck.cache_hits,
      " misses=", st_ck.cache_misses,
      " skipped=", st_ck.skipped_invalid, "]",
      " aocl=[packed=", st_aocl.packed_ok,
      " skipped=", st_aocl.skipped_invalid, "]",
      " cross_warm=", (cross_enabled ? "enabled" : "disabled"),
      " regime=", cross_warm_regime_name(regime),
      " next_HIT_for=", next_hit_for_label(primary_label, regime),
      " fingerprint=", fp_str.c_str());
}

// Companion emitter for the skip path (env-disabled / fingerprint
// already warmed).  Keeps the level-3 stream coherent: a user
// debugging "why am I not seeing a HIT?" can grep for
// `[GRP_MATMUL.PREPACK]` and see one line per prepack invocation
// regardless of whether work was done.
inline void log_pack_probe_skip(int scheduling_algo,
                                const PrepackParams &p,
                                PreludeSkipReason reason) {
  static const bool s_l3_log = apilog_info_enabled();
  if (!s_l3_log) return;
  const char *state =
      (reason == PreludeSkipReason::env_disabled)
          ? "disabled"
          : "skipped_fingerprint";
  const char *note =
      (reason == PreludeSkipReason::env_disabled)
          ? " note=first_runtime_call_pays_lazy_reorder_cost"
          : " note=already_warmed";
  const std::string fp_str = format_fingerprint(fingerprint(p, scheduling_algo));
  apilog_info(
      "[GRP_MATMUL.PREPACK] for=ALGO_", scheduling_algo,
      " state=", state,
      " active=", p.num_ops_active,
      " total=", p.num_ops_total,
      " fingerprint=", fp_str.c_str(),
      note);
}

// Eligibility for the BF16 custom-kernel pack (ALGO 3 only).  Mirrors
// the *static-knowable* refusal gates in
// `custom_kernel/dispatch.cpp::prepare_for_call` for the BF16 family
// only — DQ-INT8 eligibility goes through the sibling
// `ck_eligible_int8` below.  The outer `ck_eligible(p)` ORs the
// two so a call eligible under either family warms the
// corresponding CK pack arena; the two families pack into
// PHYSICALLY DIFFERENT arenas (K/2 pair vs K/4 quad +
// compensation row) keyed on disjoint LRU singletons, so no
// alias-by-cache is possible.
//
// If any of the static-knowable gates fail the runtime falls back
// to AOCL DLP per-tile and lazily reorders the same expert set we'd
// otherwise prefill into the CK pack arena.
//
// Asymmetry matters because the warm-pack module commits memory
// up-front: warming CK under a refused-by-runtime call wastes a
// substantial amount of resident memory across the whole expert ×
// weight × layer × pass space (a packed arena the runtime never
// reads), AND the Fix-B guard in `prepack_for_algo_3` skips the
// per-tile AOCL warm assuming CK will engage — when the runtime
// then falls back, the per-tile entries land lazily during execution
// and pay a first-call latency hit.
//
// Mirrored gates (each refusal in `prepare_for_call` is matched here):
//   * src = bf16, wei = bf16 (refusal: `unsupported_dtype`)
//   * dst ∈ {bf16, f32} — both runtime variants (`kBF16_BF16_BF16`
//     and `kBF16_BF16_F32`) are accepted; pack work is dst-agnostic
//     (the pack format is fixed by NR + kernel layout, dst only
//     affects the post-K epilogue), so a single warm covers both.
//     (refusal: `unsupported_dtype` from `resolve_variant`)
//   * act in {none, swiglu_oai_mul, silu_and_mul, gelu_and_mul}
//     (refusal: `unsupported_activation`).  All three gated kinds
//     have an in-register fused-CK epilogue:
//       - swiglu_oai_mul: caller-side interleaved W13 layout.
//       - silu_and_mul / gelu_and_mul: caller-side canonical
//         split-halves W13 layout; the prepack permutes source
//         columns into the same interleaved layout swiglu uses, so
//         the CK arena is physically identical regardless of caller
//         convention and the in-register pair-store helpers
//         (`silu_and_mul_store_pair`, `gelu_and_mul_store_pair`)
//         apply unchanged.  silu and gelu share the SAME prepack
//         permutation; the cache-key bit `kInterleaveSplitMarker`
//         is shared between them.
//   * (gated-act, f32 dst) refused — every gated-act pair-pack store
//     helper writes BF16 only, so `select_ukernel` returns nullptr
//     for any (gated-act, FP32) tuple and `prepare_for_call` refuses
//     with `kfn_table_fill_failed`.  The caller falls back to AOCL
//     DLP + a separate FP32 activation pass.
//   * (silu_and_mul / gelu_and_mul, +bias) refused — bias-into-init
//     under the prepack-permuted layout would need a permuted
//     `[gate_bias | up_bias]` read; deferred to a planned follow-up.
//     The runtime gate (`split_halves_act_with_bias_not_fused`) and
//     this prepack gate are kept symmetric so a biased silu/gelu
//     call neither warms a CK arena nor skips the AOCL per-tile
//     warm.
//   * act_dtype = bf16 when act != none (refusal: `unsupported_act_dtype`)
//   * bias_dtype in {none, bf16, f32} (refusal: `unsupported_bias_dtype`)
//   * pack_nr ∈ {32, 64} divides N (refusal: `N_not_multiple_of_pack_nr`)
//
// Per-expert refusals (`transA_not_supported`, `null_weight_in_active_
// expert`, `alpha_beta_not_supported`, `ldb_below_min_row_stride`) are
// NOT mirrored: prepack would have to walk the active range to check,
// adding O(num_ops) work to a hot fingerprint-cache short-circuit
// path; the false-positive of warming CK arena when 1-of-N experts
// breaks contract is bounded (max num_ops × pack_size memory) and is
// considered acceptable noise floor.  Frameworks that hit per-expert
// refusals will still see a runtime APILOG `[GRP_MATMUL.CK REFUSED]
// reason=...` line (verbose level) at the first refused call.
inline bool ck_eligible_bf16(const PrepackParams &p) {
  if (!p.custom_kernel_on) return false;
  // BF16 family requires `dynamic_quant == false`.  A
  // dynamic_quant=true call routes to `ck_eligible_int8` instead.
  if (p.dynamic_quant) return false;
  if (p.src_dtype != data_type_t::bf16) return false;
  if (p.wei_dtype != data_type_t::bf16) return false;
  // dst ∈ {bf16, f32} — mirrors `resolve_variant`'s acceptance of
  // both `kBF16_BF16_BF16` and `kBF16_BF16_F32`.  Pack work itself
  // does not depend on dst dtype (the pack format is set by
  // `pack_nr` + the kernel's K-pair layout), so the same packed
  // arena warms both runtime variants.
  if (p.dst_dtype != data_type_t::bf16
      && p.dst_dtype != data_type_t::f32) return false;
  // Gated activations accepted by the CK fused epilogue.  Two
  // physical layouts at the API boundary:
  //
  //   * `swiglu_oai_mul` — caller already provides W13 interleaved
  //     as [g0, u0, g1, u1, ...].  Prepack copies rows verbatim into
  //     the CK pack arena.
  //   * `silu_and_mul` / `gelu_and_mul` — caller provides W13 in
  //     canonical split-halves [gate_cols | up_cols].  Prepack
  //     re-interleaves the source row order during packing
  //     (canonical row `i` → pack row `2i`; canonical row `I+i` →
  //     pack row `2i+1`).  The packed bytes end up physically
  //     identical to the swiglu_oai_mul layout, so the existing
  //     in-register pair-store epilogues
  //     (`silu_and_mul_store_pair`, `gelu_and_mul_store_pair`)
  //     apply unchanged.  silu and gelu share the SAME prepack
  //     interleave permutation — they differ only in the kernel-side
  //     activation math, not in the pack layout.
  //
  // Split-halves silu_and_mul / gelu_and_mul + bias is NOT yet
  // eligible — the bias-into-init epilogue would need to read the
  // canonical [gate_bias | up_bias] in permuted order.  Bias-free
  // silu/gelu (the common case for MoE W13) is handled here;
  // with-bias silu/gelu stays on the separate-pass path until a
  // follow-up adds permuted bias-init.
  //
  // Cross-warm semantics: when a non-ALGO-3 prepack call site
  // invokes cross_warm with act ∈ {silu_and_mul, gelu_and_mul}
  // (e.g., ALGO 1 prompt warming up the decode-time CK arena), the
  // CK warm path receives the same act value, hits this gate, and
  // packs the interleaved layout correctly.  No change to
  // cross_warm's contract.
  const bool split_halves_no_bias =
      (p.act == grp_matmul_gated_act_t::silu_and_mul
       || p.act == grp_matmul_gated_act_t::gelu_and_mul)
      && (p.bias_dtype == data_type_t::none);
  if (p.act != grp_matmul_gated_act_t::swiglu_oai_mul
      && p.act != grp_matmul_gated_act_t::none
      && !split_halves_no_bias) {
    return false;
  }
  // Any gated activation (swiglu_oai_mul, silu_and_mul, gelu_and_mul)
  // is structurally BF16-dst only — the in-register pair-store
  // helpers write 16 BF16 lanes per call.  `select_ukernel` returns
  // nullptr for the (gated_act, FP32) tuple so `prepare_for_call`
  // would refuse with `kfn_table_fill_failed`.  Mirror that refusal
  // here.
  const bool is_gated_act =
      (p.act == grp_matmul_gated_act_t::swiglu_oai_mul)
      || (p.act == grp_matmul_gated_act_t::silu_and_mul)
      || (p.act == grp_matmul_gated_act_t::gelu_and_mul);
  if (is_gated_act && p.dst_dtype != data_type_t::bf16) {
    return false;
  }
  if (p.act != grp_matmul_gated_act_t::none
      && p.act_dtype != data_type_t::bf16) {
    return false;
  }
  if (p.bias_dtype != data_type_t::none
      && p.bias_dtype != data_type_t::bf16
      && p.bias_dtype != data_type_t::f32) {
    return false;
  }
  // Per-expert runtime-gate mirroring.  Each gate below has a
  // matching refusal in `prepare_for_call`; checking the SAME
  // condition at prepack time prevents the warmer from populating
  // CK pack arena entries the runtime would reject.  Without these
  // gates the prepack false-positive-warms regime 3 for shapes the
  // runtime refuses (resident memory waste) AND silently routes
  // the call to AOCL DLP per-tile, paying first-call lazy reorders
  // because Fix-B (CK-on -> skip R2 primary warm) ran on a regime
  // that's never consulted.
  //
  // All three vectors are optional (nullptr / empty = legacy caller
  // didn't supply runtime context; skip the gate — pre-PR
  // behaviour).  Production call sites in `group_matmul_n_tile.cpp`
  // and `group_matmul_dispatch.cpp` pass them explicitly.  Scope is
  // active experts only `[0, num_ops_active)` — under both Compact
  // (`M.size() == active_matmul`) and Padded (`M.size() ==
  // total_matmul` with `M[active..] = 0`) layouts, indices in this
  // range are firing experts; the tail entries (Padded) are zero
  // and are excluded from the runtime's `M[i] > 0` loop, so we
  // match by iterating only the active range here.
  const int n_active = p.num_ops_active;
  for (int i = 0; i < n_active; ++i) {
    // transA — `prepare_for_call` refuses with `transA_not_supported`.
    if (p.transA != nullptr
        && i < static_cast<int>(p.transA->size())
        && (*p.transA)[i]) {
      return false;
    }
    // alpha != 1 / beta != 0 — `alpha_beta_not_supported`.
    if (p.alpha != nullptr
        && i < static_cast<int>(p.alpha->size())
        && (*p.alpha)[i] != 1.0f) {
      return false;
    }
    if (p.beta != nullptr
        && i < static_cast<int>(p.beta->size())
        && (*p.beta)[i] != 0.0f) {
      return false;
    }
    // is_weights_const = false — `non_const_weight_in_active_expert`.
    // Legacy callers leave the vector empty (= "treat every entry
    // as const"); only refuse when the vector is supplied AND the
    // entry is explicitly false.
    if (p.is_weights_const != nullptr
        && !p.is_weights_const->empty()
        && i < static_cast<int>(p.is_weights_const->size())
        && !(*p.is_weights_const)[i]) {
      return false;
    }
  }
  // Delegate the NR-planner decision to the single source of truth in
  // `custom_kernel::plan_pack_nr` (dispatch.cpp) so the prepack-vs-
  // runtime gate stays bit-identical on every shape — including the
  // ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR env override.  Use the first
  // ACTIVE expert's (K, N) — matches `prepare_for_call`'s loop
  // (dispatch.cpp:357-361 `if (M[i] <= 0) continue; ... break;`).
  // The active range is `[0, n_active)`; if a caller produces M[i]=0
  // for some i < n_active (legal under both Compact and Padded
  // layouts) we MUST skip it here too, otherwise prepack's verdict
  // (using K[0]/N[0] unconditionally) and dispatch's verdict (skipping
  // M[0]=0 experts) can disagree on `(K, N)` for callers with non-
  // uniform per-expert shapes.  Falls back to (K[0], N[0]) if every
  // active expert has M=0 (degenerate; runtime would no-op anyway).
  //
  // Guarded against an empty / null K, N vector.  Namespace alias is
  // function-local because the enclosing `group_matmul_prepack` already
  // has its own `custom_kernel` sub-namespace (warm-pack helpers); a
  // file-scope alias would shadow it.
  namespace ck = ::zendnnl::lowoha::matmul::custom_kernel;
  if (p.K == nullptr || p.K->empty()) return false;
  if (p.N == nullptr || p.N->empty()) return false;
  int rep_K = (*p.K)[0];
  int rep_N = (*p.N)[0];
  // M may be empty (legacy callers / direct prepack invocations from
  // tests).  When non-empty, mirror prepare_for_call by skipping
  // zero-M experts when picking the representative.
  if (p.M != nullptr) {
    const int n_active = p.num_ops_active;
    const int sweep = std::min<int>(
        n_active,
        static_cast<int>(std::min({p.K->size(), p.N->size(),
                                   p.M->size()})));
    for (int i = 0; i < sweep; ++i) {
      if ((*p.M)[i] > 0) {
        rep_K = (*p.K)[i];
        rep_N = (*p.N)[i];
        break;
      }
    }
  }
  const int pack_nr = ck::plan_pack_nr(rep_K, rep_N);
  if (pack_nr != ck::kNRMin && pack_nr != ck::kNRMax) return false;
  return true;
}

// Eligibility for the DQ-INT8 custom-kernel pack (ALGO 3 only).
// Mirrors the static-knowable refusal gates that
// `custom_kernel/dispatch.cpp::prepare_for_call` will apply when
// invoked with `dynamic_quant=true` plus the matching
// `compute_dtype`.  Same asymmetry argument as `ck_eligible_bf16`
// applies: warming an int8 arena the runtime would refuse wastes
// memory and disables Fix-B on regime 2.
//
// Mirrored gates (each refusal in `prepare_for_call`'s int8 branch
// is matched here):
//   * Master + sub-knob: `custom_kernel_on` AND
//     `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8 = 1` (a Phase-1 user
//     who wants only bf16 CK can flip the sub-knob off and the
//     warmer will not commit an int8 arena).
//   * dtype tuple: src=bf16, wei=s8, dst=bf16, dynamic_quant=true,
//     compute ∈ {s8, u8}.
//   * Activation: same four (none / swiglu_oai_mul / silu_and_mul
//     / gelu_and_mul) as bf16, with the silu/gelu + bias refusal
//     mirrored.
//   * bias_dtype ∈ {none, bf16, f32}.
//   * pack_nr ∈ {32, 64} divides the representative N (uses
//     `plan_pack_nr` from dispatch.cpp — single source of truth
//     shared with the bf16 family).
//   * Per-expert gates (transA / alpha / beta / is_weights_const)
//     applied identically to the bf16 sibling.
//
// Phase 1 DOES NOT mirror the per-expert quant-granularity gate
// (`check_n_tile_extra`'s per-token src_scale + per-channel
// wei_scale check).  That gate runs in the outer dispatcher and
// is the single source of truth on quant shape; replicating it
// here would couple the prepack module to the quant_params layout
// without reducing the false-positive window meaningfully (the
// runtime CK refusal `weight_pack_failed` would surface
// immediately).
inline bool ck_eligible_int8(const PrepackParams &p) {
  if (!p.custom_kernel_on) return false;
  if (!get_grp_matmul_custom_kernel_int8()) return false;
  // Two int8 entry forms reach the CK microkernel (mirror
  // resolve_variant in custom_kernel/dispatch.cpp):
  //   * runtime hoist     — `dynamic_quant=true` with bf16 src;
  //   * grouped pre-quant — `group_dynamic_quant` produced an s8 src
  //     and CLEARED `dynamic_quant` (the default-on production path).
  // Keying on `dynamic_quant && src==bf16` alone missed the grouped
  // form, so the prepack never warmed the int8 CK pack for it.
  const bool dq_int8_form =
      (p.dynamic_quant && p.src_dtype == data_type_t::bf16)
      || (p.src_dtype == data_type_t::s8);
  if (!dq_int8_form) return false;
  if (p.wei_dtype != data_type_t::s8) return false;
  if (p.dst_dtype != data_type_t::bf16) return false;
  if (p.compute_dtype != data_type_t::s8
      && p.compute_dtype != data_type_t::u8) return false;
  // Activation gate — same four (none / swiglu_oai_mul /
  // silu_and_mul / gelu_and_mul) as bf16; silu_and_mul / gelu_and_mul
  // + bias is refused identically.
  const bool split_halves_no_bias =
      (p.act == grp_matmul_gated_act_t::silu_and_mul
       || p.act == grp_matmul_gated_act_t::gelu_and_mul)
      && (p.bias_dtype == data_type_t::none);
  if (p.act != grp_matmul_gated_act_t::swiglu_oai_mul
      && p.act != grp_matmul_gated_act_t::none
      && !split_halves_no_bias) {
    return false;
  }
  if (p.act != grp_matmul_gated_act_t::none
      && p.act_dtype != data_type_t::bf16) {
    return false;
  }
  if (p.bias_dtype != data_type_t::none
      && p.bias_dtype != data_type_t::bf16
      && p.bias_dtype != data_type_t::f32) {
    return false;
  }
  // Per-expert runtime-gate mirroring — identical to ck_eligible_bf16.
  // The compiler folds the two duplicated bodies; readability wins.
  const int n_active = p.num_ops_active;
  for (int i = 0; i < n_active; ++i) {
    if (p.transA != nullptr
        && i < static_cast<int>(p.transA->size())
        && (*p.transA)[i]) {
      return false;
    }
    if (p.alpha != nullptr
        && i < static_cast<int>(p.alpha->size())
        && (*p.alpha)[i] != 1.0f) {
      return false;
    }
    if (p.beta != nullptr
        && i < static_cast<int>(p.beta->size())
        && (*p.beta)[i] != 0.0f) {
      return false;
    }
    if (p.is_weights_const != nullptr
        && !p.is_weights_const->empty()
        && i < static_cast<int>(p.is_weights_const->size())
        && !(*p.is_weights_const)[i]) {
      return false;
    }
  }
  // Pack-NR planner — shared with the bf16 family.
  namespace ck = ::zendnnl::lowoha::matmul::custom_kernel;
  if (p.K == nullptr || p.K->empty()) return false;
  if (p.N == nullptr || p.N->empty()) return false;
  int rep_K = (*p.K)[0];
  int rep_N = (*p.N)[0];
  if (p.M != nullptr) {
    const int sweep = std::min<int>(
        n_active,
        static_cast<int>(std::min({p.K->size(), p.N->size(),
                                   p.M->size()})));
    for (int i = 0; i < sweep; ++i) {
      if ((*p.M)[i] > 0) {
        rep_K = (*p.K)[i];
        rep_N = (*p.N)[i];
        break;
      }
    }
  }
  // DQ-INT8 CK additionally requires K divisible by 4 (the VNNI K-quad).
  // The packed weight is zero-padded to a K-quad but the hoisted src row
  // is exactly K bytes, so a `K % 4 != 0` tail would over-read src — the
  // runtime dispatcher refuses CK for unaligned K and falls back to AOCL
  // DLP sym-quant (see `custom_kernel/dispatch.cpp::prepare_for_call`,
  // the `int8_K_not_multiple_of_4` gate).  Mirror it here so prepack does
  // NOT mark the call CK-eligible and skip the AOCL warm — otherwise the
  // runtime AOCL fallback would run on a cold cache.  (bf16's K-pair pack
  // is unaffected; this gate is int8-only, hence in `ck_eligible_int8`.)
  if ((rep_K % ck::kVNNIInt8Quad) != 0) return false;
  const int pack_nr = ck::plan_pack_nr(rep_K, rep_N);
  if (pack_nr != ck::kNRMin && pack_nr != ck::kNRMax) return false;
  return true;
}

// Outer `ck_eligible` — true when either the BF16 family or the
// DQ-INT8 family is eligible.  Callers continue to consult this
// single predicate; the family choice is implicit in the fingerprint
// (folded via `dynamic_quant` + `compute_dtype`) and in the
// downstream warmer's per-call dtype switch (see
// `warm_pack_all_custom_kernel_experts` in prepack_custom_kernel.cpp).
inline bool ck_eligible(const PrepackParams &p) {
  return ck_eligible_bf16(p) || ck_eligible_int8(p);
}

// ──────────────────────────────────────────────────────────────────────
// Cross-warm helper
//
// When the auto-select routes prompt → ALGO 1 and decode → ALGO 3,
// a deployment that fires only the prompt phase during warmup
// reaches the first decode call with only the ALGO 1 regime warm —
// the decode's regime 2 (per-tile AOCL) or regime 3 (custom-kernel
// pack) pays a mid-inference prepack spike.
//
// `cross_warm` opportunistically populates the OTHER regime so the
// transition is seamless.  Decision is CUSTOM_KERNEL-aware:
//
//   * From any non-ALGO-3 prepack (1 / 2 / 4 / 5):
//       - CK=1 ⇒ warm custom-kernel pack (regime 3); decode will use it.
//       - CK=0 ⇒ warm per-tile AOCL DLP with nr_align=1 (regime 2);
//                covers the typical Op2 non-tight decode path.
//   * From `prepack_for_algo_3`:
//       - either regime ⇒ warm full-weight AOCL (regime 1); covers
//                the future ALGO 1 prompt path if the framework
//                cycles back to prompt during the same process.
//
// `primary_did_*` flags tell the helper which regimes the calling
// per-ALGO body already warmed, so we don't duplicate work:
//   - `primary_did_aocl_fw=true`  → skip the full-weight AOCL cross-warm
//   - `primary_did_custom=true`   → skip the custom-kernel pack cross-warm
//
// All extra warmer outputs are accumulated into `st_aocl` / `st_ck`
// so the unified `[GRP_MATMUL.PREPACK]` log line reports total work
// done (primary + cross) for this prepack invocation.
//
// Gated by `ZENDNNL_GRP_MATMUL_CROSS_WARM` (default ON) AND only
// active under AUTO scheduling (`ZENDNNL_GRP_MATMUL_ALGO=0`): a pinned
// ALGO runs the same scheduling path for every call, so the
// cross-warm target cache is never consulted and the helper short-
// circuits to a no-op (the fallback path takes a one-time lazy
// reorder instead).
//
// `out_regime` is written to indicate which branch ran: `none` if
// cross-warm was skipped entirely (env off, a pinned ALGO, non-DLP
// inner_kernel, or the structural skip on ALGO 3 where the primary
// already covered the cross-warm target), otherwise the specific
// regime that fired.
// Callers thread `out_regime` into `log_pack_probe(...)` so the
// `[GRP_MATMUL.PREPACK]` line can surface it.
inline void cross_warm(const PrepackParams                  &p,
                       matmul_algo_t                         inner_kernel,
                       int                                   current_algo,
                       bool                                  primary_did_aocl_fw,
                       bool                                  primary_did_custom,
                       aocl_dlp::AoclDlpPackProbeStats      &st_aocl,
                       custom_kernel::PackProbeStats        &st_ck,
                       CrossWarmRegime                      &out_regime) {
  out_regime = CrossWarmRegime::none;
  if (!get_grp_matmul_cross_warm())                          return;
  if (inner_kernel != matmul_algo_t::aocl_dlp_blocked)       return;

  // Auto-select-only gate.  Cross-warm always targets a DIFFERENT
  // scheduling ALGO's reorder cache than the one this prepack
  // invocation serves — it prefills the regime the OTHER inference
  // phase (decode ↔ prompt) would route to in the same process.  That
  // only pays off under AUTO (`ZENDNNL_GRP_MATMUL_ALGO=0`), where the
  // phase selector may legitimately route successive calls to different
  // ALGOs.  When the user PINS a single ALGO (1..5) every call runs
  // that one ALGO, so the cross-warm target cache is never consulted
  // and eagerly packing it is pure warm-up waste (extra CPU during
  // warm-up plus a resident LRU footprint that scales with
  // experts × tiles).  Any pinned-ALGO fallback (e.g. an unsafe-shape
  // safety clamp from a forced ALGO 3 down to ALGO 1) instead pays a
  // one-time lazy reorder on its first cache miss — not performant, but
  // correct and bounded.  No correctness impact either way: the runtime
  // populates whatever it needs on demand.
  if (get_grp_matmul_algo() != 0)                            return;

  if (current_algo == 3) {
    // ALGO 3 → cross-warm the upcoming ALGO 1 prompt path's full-
    // weight AOCL reorder cache.  Family selection:
    //
    //   * BF16 family (default) — `warm_aocl(p)` warms the standard
    //     bf16 AOCL DLP LRU; matches the runtime
    //     `aocl_reorder_bf16bf16f32of32` cache key.
    //   * DQ-INT8 family — the upcoming ALGO 1 prompt under
    //     `dynamic_quant=true` warms the dedicated AOCL sym-quant LRU
    //     (`aocl_reorder_s8s8s32os32_sym_quant`), NOT the bf16 LRU.
    //     `warm_aocl_sym_quant(p)` builds a key byte-identical to the
    //     runtime per-token symmetric shape (`src_grp == K` =>
    //     `extra_input_hash = hash(K)`, `group_size = K`), so the
    //     first post-CK ALGO 1 prompt call hits the warmed slot
    //     instead of paying the lazy reorder.  This brings int8
    //     decode -> prompt cross-warm to bf16 parity (Gap B).
    if (ck_eligible_int8(p)) {
      out_regime = CrossWarmRegime::aocl_full_weight_sym_quant;
      const auto st_extra = warm_aocl_sym_quant(p);
      st_aocl.total_attempted += st_extra.total_attempted;
      st_aocl.packed_ok       += st_extra.packed_ok;
      st_aocl.skipped_invalid += st_extra.skipped_invalid;
      static const bool s_ck_int8_warm = apilog_info_enabled();
      if (s_ck_int8_warm) {
        apilog_info(
            "[GRP_MATMUL.PREPACK CROSS_WARM] regime="
            "aocl_full_weight_sym_quant: int8 CK regime — warmed AOCL "
            "sym-quant LRU eagerly (packed_ok=", st_extra.packed_ok,
            " skipped=", st_extra.skipped_invalid,
            ").  First post-CK ALGO 1 prompt call hits the cache.");
      }
      return;
    }
    // M4 — CK-OFF DQ-INT8: the int8 CK branch above did not fire
    // (`ck_eligible_int8` is false when the int8 sub-knob is off or a
    // per-expert gate refuses), but the upcoming ALGO 1 prompt still
    // reorders the full weight through the AOCL DLP sym-quant path —
    // NOT the bf16 LRU.  Warm the sym-quant full-weight LRU (unless the
    // primary's narrow-N escape already did via M3).
    if (int8_aocl_warm_candidate(p)) {
      if (!primary_did_aocl_fw) {
        const auto st_extra = warm_aocl_sym_quant(p);
        st_aocl.total_attempted += st_extra.total_attempted;
        st_aocl.packed_ok       += st_extra.packed_ok;
        st_aocl.skipped_invalid += st_extra.skipped_invalid;
        out_regime = CrossWarmRegime::aocl_full_weight_sym_quant;
      }
      return;
    }
    // BF16 family — unchanged.  Skip if `prepack_for_algo_3`'s
    // narrow-N escape already ran the full-weight warmer.
    if (!primary_did_aocl_fw) {
      const auto st_extra = warm_aocl(p);
      st_aocl.total_attempted += st_extra.total_attempted;
      st_aocl.packed_ok       += st_extra.packed_ok;
      st_aocl.skipped_invalid += st_extra.skipped_invalid;
      out_regime = CrossWarmRegime::aocl_full_weight;
    }
    return;
  }

  // ALGO 1 / 2 / 4 / 5 → cross-warm the regime the upcoming ALGO 3
  // decode will use, selected by the CUSTOM_KERNEL env knob.
  if (ck_eligible(p)) {
    // CK=1: decode will use custom kernel → warm regime 3.
    if (!primary_did_custom) {
      st_ck = warm_custom(p);
      out_regime = CrossWarmRegime::custom_kernel_pack;
    }
    return;
  }

  // CK=0: decode will use per-tile AOCL DLP → warm regime 2 with
  // nr_align=1 (covers the Op2 non-tight path; Op1 tight under
  // CK=0 uses nr_align=2 and stays lazy on first decode call —
  // option 2b would warm both variants at a much larger extra
  // resident footprint, which we deliberately avoid).
  if (p.num_threads > 0 && get_grp_matmul_aocl_stable_ntile()) {
    constexpr int nr_align_cross = 1;  // backend default for aocl_dlp_blocked
    const int max_N           = compute_max_n(p);
    const int stable          = aocl_stable_n_thr(p.num_threads, max_N);
    const int max_align_slots = std::max(1, max_N / nr_align_cross);
    if (max_N > 0 && stable > 0 && stable <= max_align_slots) {
      // M5 — DQ-INT8 (CK off): the upcoming ALGO 3 decode falls to the
      // AOCL DLP sym-quant reorder per N-tile, so warm the sym-quant
      // per-tile LRU; bf16 otherwise.
      const bool int8_dlp = int8_aocl_warm_candidate(p);
      const auto st_extra = int8_dlp
          ? warm_aocl_n_tile_sym_quant(p, stable, nr_align_cross)
          : warm_aocl_n_tile(p, stable, nr_align_cross);
      st_aocl.total_attempted += st_extra.total_attempted;
      st_aocl.packed_ok       += st_extra.packed_ok;
      st_aocl.skipped_invalid += st_extra.skipped_invalid;
      out_regime = int8_dlp ? CrossWarmRegime::aocl_per_tile_sym_quant
                            : CrossWarmRegime::aocl_per_tile;
    }
  }
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// ALGOs 1, 2, 4, 5 — AOCL DLP only (when inner kernel matches).
//
// The four bodies are identical today; kept as separate symbols so
// the modular contract holds and per-ALGO specialisation is a
// drop-in change, not a refactor.
// ─────────────────────────────────────────────────────────────────────
// Shared body for the AOCL-only scheduling ALGOs (1, 2, 4, 5).  These
// have no per-tile / custom-kernel primary warm of their own: they
// warm the full-weight AOCL DLP cache when the inner kernel is
// `aocl_dlp_blocked`, then defer to `cross_warm` to prefill whatever
// the upcoming decode (ALGO 3) will need.  Only the `scheduling_algo`
// tag differs between them — factored here so the four entry points
// are one-line forwarders (ALGO 3 keeps its own body for the CK
// eligibility logic).
static void prepack_aocl_only_algo(const PrepackParams &p,
                                   int scheduling_algo) {
  auto pre = prelude(p, scheduling_algo);
  if (pre.skip) {
    log_pack_probe_skip(scheduling_algo, p, pre.skip_reason);
    return;
  }
  aocl_dlp::AoclDlpPackProbeStats st_aocl;
  custom_kernel::PackProbeStats   st_ck;
  bool primary_did_aocl_fw = false;
  const char *primary_label = "none";

  // AUTO mixed-in-place ordering: for a prompt-class call (ALGO 1/2/4/5)
  // the primary is the AOCL full-weight reorder, which mutates the weight
  // buffer IN PLACE under mixed mode.  It MUST run AFTER `cross_warm` has
  // packed the decode layout (CK / per-tile) OUT-OF-PLACE from the raw
  // weights — otherwise CK/per-tile would read the already-mutated buffer
  // and corrupt.  So in mixed mode we run cross_warm FIRST, then the
  // in-place full-weight primary LAST.  In normal (out-of-place) mode
  // nothing mutates W, so the historical primary-then-cross order stands.
  const bool mixed_inplace = is_grp_auto_mixed_inplace_active();

  // The full-weight (prompt) primary warm.  bf16 takes the in-place path
  // under mixed mode (via the flag inside `warm_pack_all_aocl_dlp_experts`);
  // int8 sym-quant stays out-of-place (no in-place int8 path).
  auto warm_primary_full_weight = [&]() {
    if (pre.inner_kernel != matmul_algo_t::aocl_dlp_blocked) return;
    if (int8_aocl_warm_candidate(p)) {
      // M1 — DQ-INT8 prompt reorders the full weight through the AOCL DLP
      // sym-quant path (out-of-place), NOT the bf16 LRU.  CK is ALGO-3
      // only, so the prompt phase always uses AOCL DLP.
      st_aocl = warm_aocl_sym_quant(p);
      primary_label = "aocl_full_weight_sym_quant";
    } else {
      st_aocl = warm_aocl(p);
      primary_label = mixed_inplace ? "aocl_full_weight_inplace"
                                    : "aocl_full_weight";
    }
    primary_did_aocl_fw = true;
  };

  CrossWarmRegime cwr = CrossWarmRegime::none;
  if (mixed_inplace) {
    // Out-of-place decode warm first (reads raw W).
    cross_warm(p, pre.inner_kernel, scheduling_algo,
               /*primary_did_aocl_fw=*/false, /*primary_did_custom=*/false,
               st_aocl, st_ck, cwr);
    // The in-place full-weight mutation below DESTROYS the raw weights, so it
    // is only safe once cross_warm has FULLY populated the decode layout the
    // runtime will later read from those raw weights.  cross_warm warmers are
    // best-effort: an expert can be skipped (invalid metadata) or a pack /
    // reorder can fail (OOM) — both reported via `skipped_invalid` (see
    // prepack_aocl_dlp.hpp / prepack_custom_kernel.hpp).  If the decode layout
    // was NOT fully warmed (no regime ran, or any entry was skipped/failed), a
    // later decode cache-miss would reorder from the MUTATED buffer and
    // corrupt.  In that case do NOT mutate: downgrade the process to
    // out-of-place (1) and skip the in-place prompt warm, so this prompt AND
    // all decode run out-of-place from the still-RAW weights.  Capture
    // completeness HERE — `warm_primary_full_weight()` overwrites `st_aocl`
    // with the prompt full-weight stats.
    const bool cross_warm_complete =
        cwr != CrossWarmRegime::none
        && st_ck.skipped_invalid == 0
        && st_aocl.skipped_invalid == 0;
    if (cross_warm_complete) {
      // ... then the in-place full-weight prompt mutation last.
      warm_primary_full_weight();
    } else {
      // Publish WC=1 before clearing the mixed flag (WC==2 is what gates every
      // in-place path; mirrors the dispatch downgrade order).  Leave W RAW.
      zendnnl::ops::matmul_config_t::instance().set_weight_cache(1);
      zendnnl::ops::matmul_config_t::instance().set_grp_auto_mixed_inplace(
          false);
      static std::atomic<bool> s_wc2_xwarm_downgrade_warned{false};
      if (!s_wc2_xwarm_downgrade_warned.exchange(
              true, std::memory_order_relaxed)) {
        zendnnl::error_handling::apilog_warning(
            "[GRP_MATMUL.WEIGHT_CACHE] AUTO mixed-in-place: cross-warm did not "
            "fully populate the decode layout (regime=",
            static_cast<int>(cwr), " ck_skipped=", st_ck.skipped_invalid,
            " aocl_skipped=", st_aocl.skipped_invalid,
            "); skipping the in-place prompt mutation and downgrading "
            "process-wide to out-of-place (weight_cache_type=1) for the rest "
            "of the run so decode never reorders from a mutated buffer.");
      }
    }
  } else {
    warm_primary_full_weight();
    cross_warm(p, pre.inner_kernel, scheduling_algo,
               primary_did_aocl_fw, /*primary_did_custom=*/false,
               st_aocl, st_ck, cwr);
  }
  log_pack_probe(scheduling_algo, p, pre.inner_kernel, st_aocl, st_ck,
                 cwr, primary_label);
}

void prepack_for_algo_1(const PrepackParams &p) {
  prepack_aocl_only_algo(p, /*scheduling_algo=*/1);
}

void prepack_for_algo_2(const PrepackParams &p) {
  prepack_aocl_only_algo(p, /*scheduling_algo=*/2);
}

// ─────────────────────────────────────────────────────────────────────
// ALGO 3 — AOCL DLP + custom-kernel BF16 pack.
//
// flat_n_tile picks the inner kernel per call, so we warm both
// caches when their respective gates hold:
//
//   * AOCL DLP: warmed ONLY under
//     `(inner == aocl_dlp_blocked) && ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE=1`
//     (default-ON).
//
//     Why STABLE_NTILE-gated: ALGO 3's runtime cache key includes
//     SLICED `(weight_ptr_offset, n_tile)` from
//     `aligned_n_split(N[e], n_thr_e, tid, nr_align)`.
//
//     - STABLE_NTILE on  → planner forces a single num_threads-only
//       n_thr per expert, so (col_start, n_tile) stays byte-identical
//       across calls.  We mirror that decomposition with
//       `warm_aocl_n_tile(...)` so every key the runtime builds has
//       a populated cache entry post-warm-up.  When the narrow-N
//       escape fires (`stable > max_N / nr_align`) the planner falls
//       back to Sequential which uses the full-weight key, so we
//       drop to `warm_aocl(...)` for that case.
//
//     - STABLE_NTILE off → legacy dynamic plan with per-call n_thr
//       variability.  (col_start, n_tile) rotates per call so any
//       single warm-pack key set covers only a fraction of the
//       runtime keys.  Skipping the AOCL DLP warm entirely here
//       avoids spending time on reorders the runtime never queries;
//       the legacy mode already accepts cache thrash as its cost
//       (see the doc-comment on `aocl_stable_n_thr` in
//       group_matmul_parallel_common.hpp).
//
//   * Custom kernel: warmed iff custom-kernel env on AND BF16
//     src/wei/dst (independent of `inner` and STABLE_NTILE — the
//     custom-kernel pack cache is shape-keyed on the FULL N, not on
//     a per-tile slice, so warm-pack effectiveness doesn't depend on
//     the planner's tile decomposition).
// ─────────────────────────────────────────────────────────────────────
void prepack_for_algo_3(const PrepackParams &p) {
  auto pre = prelude(p, /*scheduling_algo=*/3);
  if (pre.skip) {
    log_pack_probe_skip(/*scheduling_algo=*/3, p, pre.skip_reason);
    return;
  }

  aocl_dlp::AoclDlpPackProbeStats st_aocl;
  custom_kernel::PackProbeStats   st_ck;
  bool primary_did_aocl_fw = false;
  const char *primary_label = "none";

  // Compute the CK eligibility verdict ONCE up front and reuse it.
  // `ck_eligible(p)` walks `[0, num_ops_active)` per-expert checking
  // transA / alpha / beta / is_weights_const — O(num_ops_active)
  // work.  Previously called twice in this function (once gating the
  // AOCL per-tile warm decision, once gating the CK warm) and a
  // third time inside `cross_warm` for non-ALGO-3 callers.  Caching
  // here drops that to one call per prepack invocation; for a 128-
  // expert MoE block this saves ~10 µs of redundant work on the
  // cold path and avoids any risk of the two calls drifting if a
  // future contributor edits one branch without the other.
  const bool eligible_ck = ck_eligible(p);

  // AOCL DLP per-tile warm is also skipped when the custom kernel
  // will handle the actual compute (`eligible_ck` true): the
  // runtime ALGO 3 path picks the custom kernel for every BF16/BF16/
  // BF16 + alpha=1/beta=0 + supported-act call when `CUSTOM_KERNEL=1`,
  // so the per-tile AOCL DLP LRU entries would be populated but
  // never queried — wasted CPU during warm-up plus a large resident
  // LRU footprint that scales with experts × tiles × layers × passes.
  //
  // If a runtime call falls back from custom kernel to AOCL DLP per-
  // tile (e.g. an expert hits `N % pack_nr != 0`, mixed precision
  // sneaks in, etc.), that one call pays a one-time lazy reorder cost
  // — `run_dlp(...)` lazily inserts into the LRU on miss.  Acceptable
  // trade-off: on the validated shape set the runtime stays on the
  // custom-kernel path and the savings are realised; deployments that
  // DO hit fallback pay a small one-time per-call cost on first miss.
  // No correctness impact either way.
  if (pre.inner_kernel == matmul_algo_t::aocl_dlp_blocked
      && get_grp_matmul_aocl_stable_ntile()
      && p.num_threads > 0 && p.nr_align > 0
      && !eligible_ck) {
    const int max_N = compute_max_n(p);
    const int nr_align_eff = std::max(1, p.nr_align);
    const int stable = aocl_stable_n_thr(p.num_threads, max_N);
    const int max_align_slots = std::max(1, max_N / nr_align_eff);
    if (max_N > 0 && stable > 0 && stable <= max_align_slots) {
      // Strict-stable ManyExperts: per-tile decomposition matches
      // the runtime cache key.  M2 — for DQ-INT8 (CK off / ineligible,
      // gated by `!eligible_ck` above) decode falls to the AOCL DLP
      // sym-quant reorder per N-tile, so warm the sym-quant per-tile
      // LRU, not the bf16 one.
      if (int8_aocl_warm_candidate(p)) {
        st_aocl = warm_aocl_n_tile_sym_quant(p, stable, nr_align_eff);
        primary_label = "aocl_per_tile_sym_quant";
      } else {
        st_aocl = warm_aocl_n_tile(p, stable, nr_align_eff);
        primary_label = "aocl_per_tile";
      }
    } else {
      // Narrow-N escape (`stable * nr_align > max_N`): planner
      // routes to Sequential which uses the full-weight key.  M3 —
      // sym-quant full-weight for DQ-INT8, bf16 otherwise.
      if (int8_aocl_warm_candidate(p)) {
        st_aocl = warm_aocl_sym_quant(p);
        primary_label = "aocl_full_weight_sym_quant";
      } else {
        st_aocl = warm_aocl(p);
        primary_label = "aocl_full_weight";
      }
      primary_did_aocl_fw = true;
    }
  }
  // else: AOCL DLP warm intentionally skipped.  Reasons (in order):
  //   * inner kernel is not AOCL DLP — `run_dlp` won't be invoked.
  //   * STABLE_NTILE is off — legacy dynamic plan, per-tile keys
  //     rotate per call so warming a fixed key set is wasted.
  //   * caller didn't supply a thread context — only happens for
  //     direct prepack invocations from tests / future internal
  //     callers; the `flat_n_tile` production call site always
  //     supplies both num_threads and nr_align.
  //   * `eligible_ck` is true (BF16/BF16/{BF16,F32} +
  //     `CUSTOM_KERNEL=1`) — custom kernel will handle compute, AOCL
  //     DLP cache is wasted memory (see the multi-line note above).

  const bool primary_did_custom = eligible_ck;
  if (primary_did_custom) {
    st_ck = warm_custom(p);
    // CK pack is the dominant primary signal whenever it fires —
    // overrides the AOCL primary_label which would only have been
    // set to "none" since the AOCL warm branch above is gated on
    // `!eligible_ck`.
    primary_label = "ck_pack";
  }

  // Cross-warm regime 1 (full-weight AOCL) so the upcoming ALGO 1
  // prompt path finds its cache warm if the same process toggles
  // between phases.  Skipped when narrow-N escape already populated
  // regime 1, and gated by `ZENDNNL_GRP_MATMUL_CROSS_WARM`.
  CrossWarmRegime cwr = CrossWarmRegime::none;
  cross_warm(p, pre.inner_kernel, /*current_algo=*/3,
             primary_did_aocl_fw, primary_did_custom,
             st_aocl, st_ck, cwr);

  log_pack_probe(/*scheduling_algo=*/3, p, pre.inner_kernel, st_aocl, st_ck,
                 cwr, primary_label);
}

void prepack_for_algo_4(const PrepackParams &p) {
  prepack_aocl_only_algo(p, /*scheduling_algo=*/4);
}

void prepack_for_algo_5(const PrepackParams &p) {
  prepack_aocl_only_algo(p, /*scheduling_algo=*/5);
}

void clear_fingerprint_cache_for_test() {
  {
    std::lock_guard<std::mutex> lk(s_warmed_fps_mtx);
    s_warmed_fps.clear();
  }
  // Also drop the AUTO mixed-in-place warm latches AND the completed-warm
  // record so a fresh test run re-warms (and re-mutates) from raw weights
  // instead of short-circuiting on a stale "already warmed" fingerprint.
  {
    std::lock_guard<std::mutex> lk(s_warm_latch_mtx);
    s_warm_latch_map.clear();
    s_warm_done_fps.clear();
  }
}

namespace test_api {

LastInvocationStats get_last_invocation_stats() {
  std::lock_guard<std::mutex> lk(s_last_invocation_mtx);
  return s_last_invocation;
}

void clear_last_invocation_stats() {
  std::lock_guard<std::mutex> lk(s_last_invocation_mtx);
  s_last_invocation = LastInvocationStats{};
}

} // namespace test_api

} // namespace group_matmul_prepack
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
