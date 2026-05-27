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

/// `group_matmul/prepack/prepack.hpp` — generic per-scheduling-ALGO
/// weight pre-pack module.
///
/// Each of the five scheduling ALGOs (1=sequential_experts,
/// 2=flat_m_tile, 3=flat_n_tile, 4=parallel_multilevel,
/// 5=parallel_per_expert) calls its matching
/// `prepack_for_algo_X(...)` as the first action of its body.  Each
/// per-ALGO function:
///
///   1. Short-circuits when `ZENDNNL_GRP_MATMUL_PREPACK=0`.  This is
///      the documented escape hatch that restores the strict
///      pre-PR / lazy-only behaviour for callers that prefer
///      first-iter latency over up-front warm-up.
///   2. Short-circuits when the process-wide fingerprint cache holds
///      a previously-warmed configuration (idempotent across calls
///      and across threads — see Fix A in prepack.cpp).
///   3. Picks the inner kernel via `resolve_kernel()` and eagerly
///      warms its cache for `num_ops_total` experts (resolved by
///      `build_prepack_params`: framework-hint `total_matmul` when
///      set, else `active_matmul` when only that is set, else
///      `M.size()` for legacy callers).  AOCL DLP blocked is the
///      only inner kernel this module knows about today; oneDNN /
///      libxsmm / native are no-ops, preserving "old functionality"
///      for those backends.
///   4. ALGO 3 additionally warms the BF16 custom-kernel pack cache
///      when `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1` and the dtype
///      triple is BF16/BF16/BF16 — flat_n_tile picks between custom
///      and AOCL per call, so we warm both caches.
///
/// Uniform-eager semantic (PR change-log):
///
///   The earlier semantic gated step 3 on
///   `num_ops_total > num_ops_active` so legacy callers without the
///   framework `total_matmul` contract saw a no-op.  That gate was
///   removed: PREPACK=ON now ALWAYS warms `num_ops_total` experts.
///   Three regimes share this single code path
///   (see `build_prepack_params` for the active/total resolution
///   that mirrors the dispatcher's framework-opt-in contract):
///
///     * Framework-hint regime, full (`active > 0 && total >= active`)
///       — warms the full `total` set, including the prepack-extras
///       tail.  Production MoE rotating-experts is the design target.
///
///     * Framework-hint regime, active-only (`active > 0 && total == 0`)
///       — warms exactly `active_matmul` experts (no extras).  PREPACK
///       reports `active = active_matmul` regardless of whether the
///       caller used Compact (`M.size() == active`) or Padded
///       (`M.size() == total` with placeholders) input layout.
///
///     * Legacy regime (`active = total = 0`) — warms `M.size()`
///       experts (every entry of `M` fires).  One-time first-iter
///       serial reorder cost; steady-state cache hits afterwards.
///       Set `ZENDNNL_GRP_MATMUL_PREPACK=0` to opt back into the
///       lazy-only path (pre-PR behaviour).
///
/// The prepack module sees ONE matmul per call.  When the upstream
/// caller is fused-MoE the dispatcher is invoked twice (Pass 1 with
/// gate+up weights, Pass 2 with down_proj weights); each pass
/// independently warms its own weights via the per-ALGO function it
/// lands on.  The fingerprint cache holds separate entries keyed on
/// the weight pointers + scheduling ALGO so neither pass blocks the
/// other and both short-circuit on subsequent calls.

#ifndef ZENDNNL_GROUP_MATMUL_PREPACK_HPP
#define ZENDNNL_GROUP_MATMUL_PREPACK_HPP

#include <atomic>
#include <vector>

#include "common/data_types.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
// Backend probe-stats structs surface in `test_api::LastInvocationStats`
// below.  Production callers never reach for either of these; pulling
// them in here keeps the test-only API self-contained without forcing
// every consumer to include the two backend headers themselves.
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_aocl_dlp.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_custom_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace group_matmul_prepack {

using zendnnl::common::data_type_t;

/// Per-call inputs to the per-ALGO prepack functions.  Describes ONE
/// matmul operation's per-expert metadata; there is no Op1 / Op2
/// distinction at this layer.  Callers (the five ALGO bodies) build
/// this from their local arguments and pass it through.
struct PrepackParams {
  // Per-expert weight metadata for this matmul.
  const std::vector<const void *> *weight           = nullptr;
  const std::vector<int>          *K                = nullptr;
  const std::vector<int>          *N                = nullptr;
  const std::vector<int>          *ldb              = nullptr;
  const std::vector<bool>         *transB           = nullptr;

  // Optional per-expert M.  When supplied, `ck_eligible(p)` mirrors
  // the runtime's first-active-expert selection for the pack_nr
  // representative — `prepare_for_call` (custom_kernel/dispatch.cpp)
  // skips experts with `M[i] <= 0` when picking the representative
  // `(K, N)`.  Without it, `ck_eligible` falls back to `(K[0], N[0])`
  // which can disagree with `prepare_for_call` when `M[0] == 0` and
  // per-expert (K, N) are non-uniform.  Default `nullptr` preserves
  // legacy behaviour (sample index 0).
  // (No file-line citation deliberately — the dispatcher's
  // first-active-expert pick has moved across past refactors and a
  // line-precise comment goes stale fast.  Grep for the
  // `M[i] <= 0`-guarded representative pick in
  // `prepare_for_call` if the citation needs to be re-confirmed.)
  const std::vector<int>          *M                = nullptr;

  // Optional: when present (and non-empty), AOCL warmer skips experts
  // with `is_weights_const[i] == false` (matches `run_dlp(...)`'s
  // gate).  Empty / nullptr = legacy "treat every entry as const".
  const std::vector<bool>         *is_weights_const = nullptr;

  // Optional per-expert runtime context — mirrored here so
  // `ck_eligible(p)` can match the gates `prepare_for_call` checks
  // per-call AND prevent the prepack from false-positive-warming
  // the CK pack arena on shapes the runtime will refuse.
  //
  // ── What each gate refuses at the runtime ──────────────────────
  //   * `transA[i] == true`            -> `transA_not_supported`
  //                                       (CK reads src row-major
  //                                       only).
  //   * `alpha[i] != 1.0f` or
  //     `beta[i]  != 0.0f`             -> `alpha_beta_not_supported`.
  //   * `is_weights_const[i] == false` -> `non_const_weight_in_active_expert`
  //                                       (CK pack cache cannot honour
  //                                       mutable weight).
  //
  // ── Why "all three optional" ───────────────────────────────────
  // Legacy prepack callers — and `cross_warm` siblings — pre-date
  // these fields.  Default `nullptr` means "no runtime context
  // available; skip the gate".  The CK pack arena that those
  // callers warm may still be served at runtime; the worst case
  // is the pre-fix behaviour (warm an entry the runtime never
  // reads).  Production call sites that DO have the context
  // (`group_matmul_n_tile.cpp`, `group_matmul_parallel.cpp`) pass
  // the vectors and get the bit-symmetric prepack/runtime contract
  // — no wasted warm on any call the runtime would refuse.
  //
  // The vectors are indexed per active expert `i ∈ [0, num_ops_active)`.
  // The warmer applies each gate only to active experts (matches
  // the `prepare_for_call`'s `M[i] > 0` loop), so a Padded layout
  // (`M.size() == total_matmul` with `M[active..] = 0`) does not
  // refuse CK on a stale tail entry.
  const std::vector<bool>         *transA = nullptr;
  const std::vector<float>        *alpha  = nullptr;
  const std::vector<float>        *beta   = nullptr;

  // Dtype context (read once from `params[0].dtypes` by the caller).
  data_type_t src_dtype = data_type_t::none;
  data_type_t wei_dtype = data_type_t::none;
  data_type_t dst_dtype = data_type_t::none;

  // Active / total slicing.  Under the uniform-eager semantic the
  // per-ALGO functions warm `num_ops_total` experts whenever
  // `ZENDNNL_GRP_MATMUL_PREPACK=1` (the default) — there's no
  // `total <= active` short-circuit anymore.  The two fields exist
  // so the framework-hint regime (`total > active`) warms the
  // prepack-extras tail; `num_ops_active` is the firing-expert
  // count reported in the PREPACK log line.  See
  // `build_prepack_params` for the active/total resolution that
  // mirrors the dispatcher's framework-opt-in contract, and the
  // file-level doc-block ("Uniform-eager semantic") for the
  // `ZENDNNL_GRP_MATMUL_PREPACK=0` escape hatch.
  int num_ops_active = 0;
  int num_ops_total  = 0;

  // ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL — only consulted by
  // `prepack_for_algo_3`.  Caller passes the cached env value via
  // `get_grp_matmul_custom_kernel()`.
  bool custom_kernel_on = false;

  // Gated activation kind for this dispatcher invocation, mirrored
  // here so `ck_eligible(p)` can match the runtime CK refusal gate
  // in `custom_kernel/dispatch.cpp::prepare_for_call`.
  //
  // Runtime CK acceptance matrix (as of the silu/gelu fused-CK PR):
  //   * `none`                                — always accepted
  //                                              (plain matmul tile).
  //   * `swiglu_oai_mul` + BF16 dst           — accepted; caller-
  //                                              side interleaved
  //                                              W13.
  //   * `silu_and_mul`   + BF16 dst + no bias — accepted; prepack
  //                                              re-interleaves
  //                                              split-halves W13.
  //   * `gelu_and_mul`   + BF16 dst + no bias — accepted (same
  //                                              prepack permutation
  //                                              as silu).
  //   * gated-act + FP32 dst, or gated-act + bias — refused at runtime;
  //                                              `ck_eligible` mirrors
  //                                              the refusal here.
  //
  // Warming the CK pack arena under refused activations populates
  // entries the runtime never reads (a substantial waste of resident
  // memory on many-experts MoE) AND silently routes the call to AOCL
  // DLP per-tile, adding a large number of lazy reorders at first
  // execution.  See prepack/prepack.cpp::ck_eligible for the full
  // symmetry contract.
  // Default `none` means "no gated activation"; legacy callers that
  // don't fill this field opt out of the activation gate (the rest
  // of `ck_eligible` still applies).
  grp_matmul_gated_act_t act = grp_matmul_gated_act_t::none;

  // Gated-activation intermediate dtype.  Runtime CK requires bf16
  // when `act != none`; for plain GEMM (`act == none`) the field is
  // ignored.  Default `none` means "no activation in flight" (legacy).
  data_type_t act_dtype = data_type_t::none;

  // Per-expert bias dtype (read once from `params[0].dtypes.bias` by
  // the caller).  Runtime CK supports `none / bf16 / f32` only; warming
  // CK pack arena under any other bias dtype prefills entries the
  // runtime never reads.  Default `none` is the safe no-bias case.
  data_type_t bias_dtype = data_type_t::none;

  // Per-call OMP team size — taken straight from the dispatcher's
  // entry-API `num_threads` argument.  Only consumed by
  // `prepack_for_algo_3` to compute
  // `aocl_stable_n_thr(num_threads)` at warm time so per-tile cache
  // keys match what `do_tile()` will build at run time under the
  // strict-stable plan (`ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE=1`).
  // Zero (default) means "no thread context" — `prepack_for_algo_3`
  // intentionally SKIPS the AOCL DLP warm-pack in this case.  A
  // full-weight fallback would prefill cache entries the runtime
  // never queries (the runtime's per-tile keys depend on
  // num_threads + nr_align, which we don't know here), so the
  // skip avoids wasting CPU on misaligned reorders.  ALGOs 1, 2, 4,
  // 5 don't use this field — their warmer is full-weight by design
  // and runs unconditionally on the AOCL DLP path.
  int num_threads = 0;

  // Per-thread N-slice alignment that ALGO 3's `aligned_n_split`
  // will use at run time.  Only consumed by `prepack_for_algo_3`.
  // For the AOCL DLP path it's typically 1 (`backend_n_align` for
  // `aocl_dlp_blocked`) or 2 (when `tight_fused_epilogue` widens it
  // to keep gate/up pairs on the same thread); for the custom-kernel
  // path it widens to `kctx.pack_nr` (32 or 64).
  //
  // Zero (default) is the "no align context" sentinel — the
  // production ALGO 3 call site always resolves the runtime value
  // via `ntile_effective_nr_align(...)` and forwards a positive
  // number.  `prepack_for_algo_3` intentionally SKIPS the AOCL DLP
  // warm-pack when `nr_align == 0` (and again when `num_threads ==
  // 0`) — without both, the per-tile cache keys this warmer would
  // build (which embed `n_tile = aligned_n_split(N, n_thr, ..., nr_align)`)
  // do not match what `do_tile()` queries at run time, so the warm
  // would prefill useless entries.  The `std::max(1, p.nr_align)`
  // clamp inside the warmer is defence-in-depth for the
  // already-positive case — it does not turn a zero into a one for
  // the gate's purposes.
  //
  // ALGOs 1, 2, 4, 5 don't read this field — their full-weight
  // warmer doesn't decompose by tile and doesn't need an alignment.
  int nr_align = 0;
};

/// Inline helper that pulls the typical per-call locals (weight, K,
/// N, ldb, transB, is_weights_const, params, M) into a
/// `PrepackParams` ready for `prepack_for_algo_X`.  Lives in the
/// header so each ALGO body's call site stays a single line; the
/// helper is templated on the params element type so it works for
/// `std::vector<matmul_params>` and `std::vector<grp_matmul_params>`
/// alike (both expose `.dtypes`, `.active_matmul`, `.total_matmul`).
///
/// `num_threads` and `nr_align` default to 0 — only ALGO 3 (flat_n_tile)
/// supplies them so the AOCL DLP warmer can mirror the strict-stable
/// per-tile decomposition.  ALGOs 1, 2, 4, 5 use the full-weight
/// warmer regardless and the defaults are correct for them.
///
/// `act` / `act_dtype` default to `none` — only the fused-MoE / gated-
/// act dispatch sites supply real values so `ck_eligible(p)` can match
/// the runtime CK refusal gate's activation check.  Legacy callers
/// that don't fill them stay on the safe side (CK eligible only when
/// the BF16 dtype gates pass; per-call activation refusal at runtime
/// remains the catch-all).
template <typename ParamsVec>
inline PrepackParams build_prepack_params(
    const std::vector<const void *> &weight,
    const std::vector<int>          &K,
    const std::vector<int>          &N,
    const std::vector<int>          &ldb,
    const std::vector<bool>         &transB,
    const std::vector<bool>         &is_weights_const,
    const ParamsVec                 &params,
    const std::vector<int>          &M,
    bool                             custom_kernel_on,
    int                              num_threads = 0,
    int                              nr_align    = 0,
    grp_matmul_gated_act_t           act         = grp_matmul_gated_act_t::none,
    data_type_t                      act_dtype   = data_type_t::none,
    // Optional per-expert runtime context — when non-null, mirrors
    // the runtime CK refusal gates in `prepare_for_call` (transA,
    // alpha != 1, beta != 0, is_weights_const = false).  Legacy
    // callers leave these unset (nullptr) and `ck_eligible` skips
    // the corresponding checks — same as pre-PR behaviour.
    const std::vector<bool>         *transA = nullptr,
    const std::vector<float>        *alpha  = nullptr,
    const std::vector<float>        *beta   = nullptr) {
  PrepackParams p;
  p.weight           = &weight;
  p.K                = &K;
  p.N                = &N;
  p.ldb              = &ldb;
  p.transB           = &transB;
  p.M                = &M;
  p.is_weights_const = &is_weights_const;
  p.transA           = transA;
  p.alpha            = alpha;
  p.beta             = beta;

  if (!params.empty()) {
    p.src_dtype  = params[0].dtypes.src;
    p.wei_dtype  = params[0].dtypes.wei;
    p.dst_dtype  = params[0].dtypes.dst;
    p.bias_dtype = params[0].dtypes.bias;
  }

  // Mirror the dispatcher's active/total contract
  // (`group_matmul_direct.cpp::framework_opt_in`):
  //
  //   * `active_matmul > 0` is THE framework opt-in signal — the
  //     dispatcher gates its entire opt-in path
  //     (`framework_opt_in = params[0].active_matmul > 0`) on exactly
  //     this field.  The dispatcher accepts both Compact
  //     (`M.size() == active_matmul`) and Padded (`M.size() ==
  //     total_matmul` with `M[active..]=0`) input layouts; in the
  //     Padded form `M.size()` over-counts the firing experts.  Use
  //     `active_matmul` here so the PREPACK log line and any
  //     downstream diagnostic that reads `num_ops_active` reflects
  //     the true firing count regardless of which layout the caller
  //     used.
  //   * `active_matmul == 0` (legacy) means "no opt-in, every entry
  //     in `M` fires" — fall back to `M.size()`.
  //   * `total_matmul` is honoured ONLY when `active_matmul > 0`
  //     (i.e. only inside the opt-in regime).  The two fields are a
  //     pair: a legacy caller (`active_matmul == 0`) may leave
  //     `total_matmul` at any value (stale memory, copy-paste from
  //     an opt-in caller, ...) and the dispatcher will ignore it —
  //     this module must do the same, otherwise:
  //       - the fingerprint cache key (`prepack.cpp::fingerprint`,
  //         first input is `num_ops_total`) would carve out distinct
  //         entries for callers that the dispatcher treats as
  //         identical;
  //       - the PREPACK log line would report a `total=N` that
  //         exceeds the count the dispatcher actually runs;
  //       - the warmer would iterate `[0, total_matmul)` over weight
  //         vectors that legacy-mode strict size validation requires
  //         to be exactly `M.size()` long, producing misleading
  //         `skipped_invalid` counts after the `min({...})` clamp in
  //         `warm_pack_all_aocl_dlp_experts`.
  //     Within the opt-in regime, `total_matmul == 0` means "no
  //     prepack-extras tail" and `num_ops_total` falls back to
  //     `num_ops_active = active_matmul` (warmer walks exactly the
  //     firing experts).
  const bool has_active_hint =
      !params.empty() && params[0].active_matmul > 0;
  const bool has_total_hint  =
      has_active_hint && params[0].total_matmul > 0;
  p.num_ops_active = has_active_hint
                     ? static_cast<int>(params[0].active_matmul)
                     : static_cast<int>(M.size());
  p.num_ops_total  = has_total_hint
                     ? static_cast<int>(params[0].total_matmul)
                     : p.num_ops_active;

  p.custom_kernel_on = custom_kernel_on;
  p.num_threads      = num_threads;
  p.nr_align         = nr_align;
  p.act              = act;
  p.act_dtype        = act_dtype;
  return p;
}

// ── Per-ALGO entry points ────────────────────────────────────────────
//
// Each of these is a no-op when:
//   * the env knob `ZENDNNL_GRP_MATMUL_PREPACK` is OFF, OR
//   * the process-wide fingerprint cache holds a previously-warmed
//     configuration (idempotent across calls and across threads —
//     see Fix A in prepack.cpp), OR
//   * the inner kernel resolved by `resolve_kernel()` is not one this
//     module knows about (oneDNN / libxsmm / native).
//
// Otherwise eagerly warms `p.num_ops_total` experts before the
// matmul kicks off.  `num_ops_total` is resolved in
// `build_prepack_params` from `params[0].total_matmul` (framework-
// hint regime) or, when that is unset, from `params[0].active_matmul`
// or `M.size()` (legacy callers).  By construction this is always
// `>= M.size()` for supported call patterns, so the warmed set
// covers every firing expert plus the prepack-extras tail.  See the
// file-level doc-block above for the uniform-eager semantic.
//
// Functionally, ALGOs 1, 2, 4, 5 are identical (warm AOCL DLP iff
// inner == aocl_dlp_blocked).  ALGO 3 additionally warms the BF16
// custom-kernel pack cache when its eligibility predicate holds.
// Five separate symbols are kept because the modular contract is
// "each ALGO has its own prepack function" — if a future ALGO needs
// special-cased warm-pack behaviour, the per-ALGO body is the place.

void prepack_for_algo_1(const PrepackParams &p);  // sequential_experts
void prepack_for_algo_2(const PrepackParams &p);  // flat_m_tile
void prepack_for_algo_3(const PrepackParams &p);  // flat_n_tile
void prepack_for_algo_4(const PrepackParams &p);  // parallel_multilevel
void prepack_for_algo_5(const PrepackParams &p);  // parallel_per_expert

/// Clear the process-wide fingerprint cache.  Test-only API used by
/// gtest cases that need a clean fingerprint state to avoid false
/// "already warmed" matches caused by heap address reuse across
/// independent test cases (the cache key includes weight pointers,
/// and freeing+reallocating buffers can land at the same heap
/// addresses in adjacent tests).  Production code should NEVER call
/// this — the fingerprint cache is process-wide and grows
/// monotonically to a few entries (one per (model, layer,
/// sched_algo) combination), so no clearing is needed at runtime.
///
/// Thread-safe: acquires the same mutex `already_warmed` uses.
void clear_fingerprint_cache_for_test();

// ───────────────────────────────────────────────────────────────────────
// Test-only introspection API.
//
// Gtest sections need to assert what each `prepack_for_algo_X`
// invocation actually warmed (e.g. did Fix B skip the AOCL DLP
// per-tile warm under CK=1?  did Fix D's cross-warm fire from ALGO 1
// under CK=0 to populate regime 2?).  Probe-stats counters
// `AoclDlpPackProbeStats` / `PackProbeStats` are already accumulated
// across the primary warm and any cross-warm inside the per-ALGO
// function, but the final values get consumed by the PREPACK
// apilog line and discarded.  This API captures the most recent
// invocation's accumulated stats into a process-wide accumulator that
// tests can read.
//
// Strictly test-only.  Production code never calls these.  Runtime
// overhead per `log_pack_probe` call: one struct assignment + one
// mutex lock — negligible vs the actual matmul body, but the API is
// gated by the function name suffix so production callers never
// reach for it by accident.
//
// CAPTURE GATE — `s_capture_last_invocation` (atomic bool, default
// false):
//   Production builds never set this flag, so the store path in
//   `log_pack_probe` short-circuits on a single relaxed load of a
//   cache-line-shared `false` value (no mutex acquire, no struct
//   copy, no coherence traffic).  Tests arm the flag (via
//   `LastInvocationCaptureGuard` in `moe_test_utils.hpp`) for the
//   test's scope, in which case the gated branch DOES fire and
//   takes the mutex + writes through to `s_last`.  Without this
//   gate the unconditional mutex lock in `log_pack_probe` ran on
//   every dispatcher call — measurable cost on hot fused-MoE serving
//   paths that have no use for the test hook.  Mirror of the
//   `s_capture_gemm_mode` gate in `group_matmul_parallel_common.hpp`.
// ───────────────────────────────────────────────────────────────────────

/// Which cross_warm regime fired alongside the primary warm.  Recorded
/// per invocation in `LastInvocationStats` and surfaced in the
/// `[GRP_MATMUL.PREPACK]` apilog line as two separate fields:
/// `cross_warm=<enabled|disabled>` (the env state, mirrors the
/// `ZENDNNL_GRP_MATMUL_CROSS_WARM` knob) and `regime=<regime>` (this
/// enum's stringified value).  Together they let a user debugging a
/// HIT/MISS trace see at a glance whether cross-warm was on AND
/// which backend path got opportunistically warmed for the OTHER
/// algo regime.
///
/// The four states map to the four reachable code paths in
/// `cross_warm()`:
///   * `none`               — env `ZENDNNL_GRP_MATMUL_CROSS_WARM=0`, OR
///                            inner_kernel != aocl_dlp_blocked, OR
///                            the structural skip path on ALGO 3 where
///                            the primary already covered the
///                            cross-warm target.
///   * `aocl_full_weight`   — fired from `prepack_for_algo_3`: covers
///                            the upcoming ALGO 1 prompt path.
///   * `custom_kernel_pack` — fired from a non-ALGO-3 prepack with
///                            `ck_eligible(p)==true`: covers the
///                            upcoming ALGO 3 + CK decode path.
///   * `aocl_per_tile`      — fired from a non-ALGO-3 prepack with
///                            CK ineligible: covers the upcoming
///                            ALGO 3 + DLP decode path (per-tile cache
///                            with nr_align=1).
enum class CrossWarmRegime {
  none               = 0,
  aocl_full_weight   = 1,
  custom_kernel_pack = 2,
  aocl_per_tile      = 3,
};

namespace test_api {

/// Snapshot of the most recent `prepack_for_algo_X` invocation's
/// observable side-effects.  Mirrors what the PREPACK apilog line
/// reports, but in a struct that gtest can `EXPECT_*` on.
struct LastInvocationStats {
  /// Scheduling ALGO whose per-ALGO function was called (1..5).
  int scheduling_algo = 0;

  /// Inner kernel resolved by `prelude(...)::resolve_kernel()` for
  /// this invocation.  Lets tests distinguish "AOCL DLP warm path
  /// was eligible (= `aocl_dlp_blocked`)" from "inner kernel is
  /// oneDNN / libxsmm / native, no AOCL DLP warm path taken".
  zendnnl::ops::matmul_algo_t inner_kernel =
      zendnnl::ops::matmul_algo_t::none;

  /// Accumulated AOCL DLP probe stats across primary warm
  /// (`warm_aocl_n_tile` or `warm_aocl`) AND any cross-warm
  /// contributions (`warm_aocl` from `cross_warm`).  Tests
  /// differentiate Fix B's CK=1 skip from CK=0 warm by comparing
  /// `total_attempted`:
  ///
  ///   * CK=1 + BF16 + ALGO 3 + Fix B: per-tile warm skipped,
  ///     cross-warm regime 1 contributes `num_experts` entries.
  ///   * CK=0 + ALGO 3: per-tile warm contributes
  ///     `num_experts × stable_n_thr`, cross-warm regime 1 adds
  ///     another `num_experts`.
  aocl_dlp::AoclDlpPackProbeStats aocl{};

  /// Accumulated custom-kernel probe stats.  Same accumulation
  /// model as `aocl` above.
  custom_kernel::PackProbeStats ck{};

  /// Which `cross_warm` regime ran for this invocation.  `none` when
  /// the cross-warm helper was skipped (env off, non-DLP inner kernel,
  /// or the primary already covered the cross-warm target).  Surfaces
  /// in the `[GRP_MATMUL.PREPACK]` apilog line as `regime=<regime>`,
  /// paired with the env-state field `cross_warm=<enabled|disabled>`
  /// — see `CrossWarmRegime`'s doc-block above for the field layout.
  CrossWarmRegime cross_warm_regime = CrossWarmRegime::none;

  /// True after a `prepack_for_algo_X` invocation that took the
  /// non-skip path through `prelude()`.  False after
  /// `clear_last_invocation_stats()` or before any prepack call.
  bool valid = false;
};

/// Return a snapshot of the most recent invocation's stats.
/// Thread-safe; returns a value copy under a mutex.
LastInvocationStats get_last_invocation_stats();

/// Reset the accumulator to its default-constructed state.  Call from
/// `SetUp()` in gtest fixtures that need a clean baseline.
void clear_last_invocation_stats();

/// Capture gate for `log_pack_probe` — see the file-level CAPTURE
/// GATE doc-block above.  Tests flip this to `true` for the scope of
/// the test (via `LastInvocationCaptureGuard`); production code never
/// touches it.  `inline` so the single definition lives in the header
/// (same pattern as `s_capture_gemm_mode`).
inline std::atomic<bool> s_capture_last_invocation{false};

} // namespace test_api

} // namespace group_matmul_prepack
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PREPACK_HPP
