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
#include <cstddef>
#include <cstdint>
#include <mutex>
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
    size_t    k_xor      = 0;
    size_t    n_xor      = 0;
    size_t    ldb_xor    = 0;
    size_t    transb_xor = 0;
    for (size_t i = 0; i < bound; ++i) {
      ptr_xor ^= reinterpret_cast<uintptr_t>((*p.weight)[i]);
      if (p.K  && i < p.K->size())
        k_xor ^= static_cast<size_t>((*p.K)[i]);
      if (p.N  && i < p.N->size())
        n_xor ^= static_cast<size_t>((*p.N)[i]);
      if (p.ldb && i < p.ldb->size())
        ldb_xor ^= static_cast<size_t>((*p.ldb)[i]);
      if (p.transB && i < p.transB->size())
        transb_xor ^= static_cast<size_t>((*p.transB)[i] ? 1u : 0u);
    }
    s = mix_hash(s, static_cast<size_t>(ptr_xor));
    s = mix_hash(s, k_xor);
    s = mix_hash(s, n_xor);
    s = mix_hash(s, ldb_xor);
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
  // Fold the activation + bias dtype context into the fingerprint
  // because they participate in `ck_eligible(p)` (mirrors the runtime
  // CK refusal gate).  Without these terms, a process that toggles
  // between two layers with different `act` (e.g. one swiglu_oai_mul,
  // one silu_and_mul) would cache the first call's eligibility
  // verdict and short-circuit re-warming for the second — which has
  // a different verdict.  Cost is two mix_hash calls per fingerprint
  // (~5 ns); negligible vs the matmul body.
  s = mix_hash(s, static_cast<size_t>(p.act));
  s = mix_hash(s, static_cast<size_t>(p.act_dtype));
  s = mix_hash(s, static_cast<size_t>(p.bias_dtype));
  // Fold the runtime-mutable weight-cache toggle into the fingerprint.
  // `matmul_config_t::set_weight_cache(...)` can flip this mid-process
  // and the AOCL DLP warmer's gate reads it on every call.  Without
  // this hash term, a process that runs first under WEIGHT_CACHE=0
  // (warmer no-op) and then enables it via `set_weight_cache(1)`
  // would short-circuit on the second call (fingerprint already
  // present) and leave the cache permanently empty for this thread.
  // The custom-kernel path doesn't read this knob, but folding it
  // unconditionally keeps the hash regime simple and the cost is one
  // singleton-load (~1 ns).
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

// Result of the shared opening sequence: one of `skip` (the per-ALGO
// function should return immediately) or a resolved inner kernel.
struct PreludeResult {
  bool          skip          = true;
  matmul_algo_t inner_kernel  = matmul_algo_t::none;
};

inline PreludeResult prelude(const PrepackParams &p, int scheduling_algo) {
  PreludeResult r;
  if (!get_grp_matmul_prepack())                  return r;  // env OFF
  // NOTE: the previous `if (p.num_ops_total <= p.num_ops_active) return r;`
  // gate was removed.  Under the uniform-eager semantic, PREPACK=ON
  // ALWAYS warms `max(M.size(), total_matmul)` experts up front,
  // regardless of whether the framework opted into the
  // `total_matmul > active_matmul` contract.  Legacy callers
  // (active=total=0 → both fall through to `num_ops_total = M.size()`
  // in `build_prepack_params`) now also exercise the prepack module;
  // they pay a one-time first-iter serial reorder cost in exchange
  // for `do_tile()` cache hits afterwards.  Set
  // `ZENDNNL_GRP_MATMUL_PREPACK=0` to restore the lazy-only path.
  if (already_warmed(p, scheduling_algo))         return r;  // fingerprint hit

  r.skip         = false;
  r.inner_kernel = resolve_kernel();
  return r;
}

// AOCL DLP backend wrapper — calls the existing FULL-WEIGHT warmer
// with the per-call vectors taken straight from `PrepackParams`.
// Used by ALGOs 1, 2, 4, 5 (no column tiling) and by ALGO 3's
// fallback paths (STABLE_NTILE off, narrow-N escape, missing thread
// context).  Returns the `packed_ok` count for the apilog probe line.
inline aocl_dlp::AoclDlpPackProbeStats warm_aocl(const PrepackParams &p) {
  aocl_dlp::AoclDlpPackProbeStats st;
  // Empty `is_weights_const` is the documented "treat as const"
  // sentinel — we forward an empty vector when the caller didn't
  // supply one so the warmer's gate stays correct.
  static const std::vector<bool> kEmptyIsConst;
  const std::vector<bool> &iwc =
      (p.is_weights_const != nullptr) ? *p.is_weights_const : kEmptyIsConst;
  aocl_dlp::warm_pack_all_aocl_dlp_experts(
      *p.weight, *p.K, *p.N, *p.ldb, *p.transB, iwc,
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
  static const std::vector<bool> kEmptyIsConst;
  const std::vector<bool> &iwc =
      (p.is_weights_const != nullptr) ? *p.is_weights_const : kEmptyIsConst;
  aocl_dlp::warm_pack_all_aocl_dlp_experts_n_tile(
      *p.weight, *p.K, *p.N, *p.ldb, *p.transB, iwc,
      p.num_ops_total, p.wei_dtype,
      p.num_threads, stable, nr_align_eff, st);
  return st;
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
inline custom_kernel::PackProbeStats warm_custom(const PrepackParams &p) {
  custom_kernel::PackProbeStats st;
  static const std::vector<bool> kEmptyIsConst;
  const std::vector<bool> &iwc =
      (p.is_weights_const != nullptr) ? *p.is_weights_const : kEmptyIsConst;
  custom_kernel::warm_pack_all_custom_kernel_experts(
      *p.weight, *p.K, *p.N, *p.ldb, *p.transB, iwc,
      p.num_ops_total, st);
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

// Single PROBE log line shared across the five per-ALGO functions so
// downstream tooling sees the same field set regardless of which
// scheduling algo was warmed.  Also records the per-invocation
// accumulator for the test-only API in `prepack.hpp::test_api::`.
inline void log_pack_probe(int scheduling_algo,
                           const PrepackParams &p,
                           matmul_algo_t inner_kernel,
                           const aocl_dlp::AoclDlpPackProbeStats &st_aocl,
                           const custom_kernel::PackProbeStats   &st_ck) {
  // Always record stats — the gate below only controls the log emission,
  // not the accumulator.  Tests benefit even when the apilog level
  // hides the PROBE line.  Cost: one mutex lock + struct copy, ~50 ns.
  {
    std::lock_guard<std::mutex> lk(s_last_invocation_mtx);
    s_last_invocation.scheduling_algo = scheduling_algo;
    s_last_invocation.inner_kernel    = inner_kernel;
    s_last_invocation.aocl            = st_aocl;
    s_last_invocation.ck              = st_ck;
    s_last_invocation.valid           = true;
  }

  static const bool s_l1_log = apilog_info_enabled();
  if (!s_l1_log) return;
  apilog_info(
      "[GRP_MATMUL Level3 PACK_PROBE] sched_algo=", scheduling_algo,
      " inner_kernel=", static_cast<int>(inner_kernel),
      " active=", p.num_ops_active,
      " total=", p.num_ops_total,
      " ck=[hits=", st_ck.cache_hits,
      " misses=", st_ck.cache_misses,
      " skipped=", st_ck.skipped_invalid, "]",
      " aocl=[packed=", st_aocl.packed_ok,
      " skipped=", st_aocl.skipped_invalid, "]");
}

// Eligibility for the BF16 custom-kernel pack (ALGO 3 only).  Mirrors
// the *static-knowable* refusal gates in
// `custom_kernel/dispatch.cpp::prepare_for_call`; if any of those
// fail the runtime falls back to AOCL DLP per-tile and lazily reorders
// the same expert set we'd otherwise prefill into the CK pack arena.
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
//   * src/wei/dst all bf16 (refusal: `unsupported_dtype`)
//   * act in {swiglu_oai_mul, none} (refusal: `unsupported_activation`)
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
// refusals will still see a runtime APILOG `[GRP_MATMUL Level4
// custom_kernel REFUSED] reason=...` line at the first refused call.
inline bool ck_eligible(const PrepackParams &p) {
  if (!p.custom_kernel_on) return false;
  if (p.src_dtype != data_type_t::bf16) return false;
  if (p.wei_dtype != data_type_t::bf16) return false;
  if (p.dst_dtype != data_type_t::bf16) return false;
  if (p.act != grp_matmul_gated_act_t::swiglu_oai_mul
      && p.act != grp_matmul_gated_act_t::none) {
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
  // Delegate the NR-planner decision to the single source of truth in
  // `custom_kernel::plan_pack_nr` (dispatch.cpp) so the prepack-vs-
  // runtime gate stays bit-identical on every shape — including the
  // ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR env override.  Use the first
  // active expert's (K, N) as the representative; the runtime gate
  // does the same (dispatch.cpp::prepare_for_call's loop breaks on
  // the first active expert).  Guarded against an empty / null K, N
  // vector.  Namespace alias is function-local because the enclosing
  // `group_matmul_prepack` already has its own `custom_kernel` sub-
  // namespace (warm-pack helpers); a file-scope alias would shadow it.
  namespace ck = ::zendnnl::lowoha::matmul::custom_kernel;
  if (p.K == nullptr || p.K->empty()) return false;
  if (p.N == nullptr || p.N->empty()) return false;
  const int pack_nr = ck::plan_pack_nr((*p.K)[0], (*p.N)[0]);
  if (pack_nr != ck::kNRMin && pack_nr != ck::kNRMax) return false;
  return true;
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
// so the unified `PACK_PROBE` log line reports total work done
// (primary + cross) for this prepack invocation.
//
// Gated by `ZENDNNL_GRP_MATMUL_CROSS_WARM` (default ON).
inline void cross_warm(const PrepackParams                  &p,
                       matmul_algo_t                         inner_kernel,
                       int                                   current_algo,
                       bool                                  primary_did_aocl_fw,
                       bool                                  primary_did_custom,
                       aocl_dlp::AoclDlpPackProbeStats      &st_aocl,
                       custom_kernel::PackProbeStats        &st_ck) {
  if (!get_grp_matmul_cross_warm())                          return;
  if (inner_kernel != matmul_algo_t::aocl_dlp_blocked)       return;

  if (current_algo == 3) {
    // ALGO 3 → cross-warm regime 1 (full-weight AOCL) for the
    // upcoming ALGO 1 prompt path.  Skip if `prepack_for_algo_3`'s
    // narrow-N escape already ran the full-weight warmer.
    if (!primary_did_aocl_fw) {
      const auto st_extra = warm_aocl(p);
      st_aocl.total_attempted += st_extra.total_attempted;
      st_aocl.packed_ok       += st_extra.packed_ok;
      st_aocl.skipped_invalid += st_extra.skipped_invalid;
    }
    return;
  }

  // ALGO 1 / 2 / 4 / 5 → cross-warm the regime the upcoming ALGO 3
  // decode will use, selected by the CUSTOM_KERNEL env knob.
  if (ck_eligible(p)) {
    // CK=1: decode will use custom kernel → warm regime 3.
    if (!primary_did_custom) {
      st_ck = warm_custom(p);
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
      const auto st_extra =
          warm_aocl_n_tile(p, stable, nr_align_cross);
      st_aocl.total_attempted += st_extra.total_attempted;
      st_aocl.packed_ok       += st_extra.packed_ok;
      st_aocl.skipped_invalid += st_extra.skipped_invalid;
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
void prepack_for_algo_1(const PrepackParams &p) {
  auto pre = prelude(p, /*scheduling_algo=*/1);
  if (pre.skip) return;
  aocl_dlp::AoclDlpPackProbeStats st_aocl;
  custom_kernel::PackProbeStats   st_ck;
  bool primary_did_aocl_fw = false;
  if (pre.inner_kernel == matmul_algo_t::aocl_dlp_blocked) {
    st_aocl = warm_aocl(p);
    primary_did_aocl_fw = true;
  }
  cross_warm(p, pre.inner_kernel, /*current_algo=*/1,
             primary_did_aocl_fw, /*primary_did_custom=*/false,
             st_aocl, st_ck);
  log_pack_probe(/*scheduling_algo=*/1, p, pre.inner_kernel, st_aocl, st_ck);
}

void prepack_for_algo_2(const PrepackParams &p) {
  auto pre = prelude(p, /*scheduling_algo=*/2);
  if (pre.skip) return;
  aocl_dlp::AoclDlpPackProbeStats st_aocl;
  custom_kernel::PackProbeStats   st_ck;
  bool primary_did_aocl_fw = false;
  if (pre.inner_kernel == matmul_algo_t::aocl_dlp_blocked) {
    st_aocl = warm_aocl(p);
    primary_did_aocl_fw = true;
  }
  cross_warm(p, pre.inner_kernel, /*current_algo=*/2,
             primary_did_aocl_fw, /*primary_did_custom=*/false,
             st_aocl, st_ck);
  log_pack_probe(/*scheduling_algo=*/2, p, pre.inner_kernel, st_aocl, st_ck);
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
//       avoids spending ~tens of ms on reorders the runtime never
//       queries; the legacy A/B mode already accepts cache thrash
//       as its cost (see the doc-comment on `aocl_stable_n_thr` in
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
  if (pre.skip) return;

  aocl_dlp::AoclDlpPackProbeStats st_aocl;
  custom_kernel::PackProbeStats   st_ck;
  bool primary_did_aocl_fw = false;

  // AOCL DLP per-tile warm is also skipped when the custom kernel
  // will handle the actual compute (`ck_eligible(p)` true): the
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
      && !ck_eligible(p)) {
    const int max_N = compute_max_n(p);
    const int nr_align_eff = std::max(1, p.nr_align);
    const int stable = aocl_stable_n_thr(p.num_threads, max_N);
    const int max_align_slots = std::max(1, max_N / nr_align_eff);
    if (max_N > 0 && stable > 0 && stable <= max_align_slots) {
      // Strict-stable ManyExperts: per-tile decomposition matches
      // the runtime cache key.
      st_aocl = warm_aocl_n_tile(p, stable, nr_align_eff);
    } else {
      // Narrow-N escape (`stable * nr_align > max_N`): planner
      // routes to Sequential which uses the full-weight key.
      st_aocl = warm_aocl(p);
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
  //   * `ck_eligible(p)` is true (BF16/BF16/BF16 + `CUSTOM_KERNEL=1`)
  //     — custom kernel will handle compute, AOCL DLP cache is
  //     wasted memory (see the multi-line note immediately above).

  const bool primary_did_custom = ck_eligible(p);
  if (primary_did_custom) {
    st_ck = warm_custom(p);
  }

  // Cross-warm regime 1 (full-weight AOCL) so the upcoming ALGO 1
  // prompt path finds its cache warm if the same process toggles
  // between phases.  Skipped when narrow-N escape already populated
  // regime 1, and gated by `ZENDNNL_GRP_MATMUL_CROSS_WARM`.
  cross_warm(p, pre.inner_kernel, /*current_algo=*/3,
             primary_did_aocl_fw, primary_did_custom,
             st_aocl, st_ck);

  log_pack_probe(/*scheduling_algo=*/3, p, pre.inner_kernel, st_aocl, st_ck);
}

void prepack_for_algo_4(const PrepackParams &p) {
  auto pre = prelude(p, /*scheduling_algo=*/4);
  if (pre.skip) return;
  aocl_dlp::AoclDlpPackProbeStats st_aocl;
  custom_kernel::PackProbeStats   st_ck;
  bool primary_did_aocl_fw = false;
  if (pre.inner_kernel == matmul_algo_t::aocl_dlp_blocked) {
    st_aocl = warm_aocl(p);
    primary_did_aocl_fw = true;
  }
  cross_warm(p, pre.inner_kernel, /*current_algo=*/4,
             primary_did_aocl_fw, /*primary_did_custom=*/false,
             st_aocl, st_ck);
  log_pack_probe(/*scheduling_algo=*/4, p, pre.inner_kernel, st_aocl, st_ck);
}

void prepack_for_algo_5(const PrepackParams &p) {
  auto pre = prelude(p, /*scheduling_algo=*/5);
  if (pre.skip) return;
  aocl_dlp::AoclDlpPackProbeStats st_aocl;
  custom_kernel::PackProbeStats   st_ck;
  bool primary_did_aocl_fw = false;
  if (pre.inner_kernel == matmul_algo_t::aocl_dlp_blocked) {
    st_aocl = warm_aocl(p);
    primary_did_aocl_fw = true;
  }
  cross_warm(p, pre.inner_kernel, /*current_algo=*/5,
             primary_did_aocl_fw, /*primary_did_custom=*/false,
             st_aocl, st_ck);
  log_pack_probe(/*scheduling_algo=*/5, p, pre.inner_kernel, st_aocl, st_ck);
}

void clear_fingerprint_cache_for_test() {
  std::lock_guard<std::mutex> lk(s_warmed_fps_mtx);
  s_warmed_fps.clear();
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
