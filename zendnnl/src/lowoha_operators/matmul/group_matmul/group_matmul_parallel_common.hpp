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

/// Library-internal helpers shared by the ALGO 1..5 parallel paths.
///
/// Each ALGO implementation (`sequential_experts`, `flat_m_tile`,
/// `flat_n_tile`, `parallel_multilevel`, `parallel_per_expert`) is
/// split into its own translation unit so the files stay small and
/// ownership is clear.  This header hosts the bits they all need:
///   - env-driven feature flags
///   - the `resolve_kernel()` / `execute_expert_slice()` primitives
///   - tile-size constants referenced by ALGO 0 auto-select
///   - forward declarations of the per-strategy entry points
///
/// None of these symbols are part of the public ZenDNN API.

#ifndef ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
#define ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include "custom_kernel/dispatch.hpp"
#include "group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "operators/matmul/matmul_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Match the original group_matmul_parallel.cpp using-declarations so
// source files including this header can refer to matmul_algo_t,
// matmul_config_t, post_op_type_t, etc. without namespace prefixes.
using namespace zendnnl::ops;
using zendnnl::common::size_of;

// Short string renderer for `grp_matmul_gated_act_t` — used by APILOG
// lines and gemm_mode_out strings; keeps activation names consistent
// across executors.
inline const char *act_name(grp_matmul_gated_act_t a) {
  switch (a) {
  case grp_matmul_gated_act_t::none:
    return "none";
  case grp_matmul_gated_act_t::silu_and_mul:
    return "silu_and_mul";
  case grp_matmul_gated_act_t::gelu_and_mul:
    return "gelu_and_mul";
  case grp_matmul_gated_act_t::swiglu_oai_mul:
    return "swiglu_oai_mul";
  }
  return "?";
}

// Tile-size + weight-class constants shared by ALGO 0 auto-select and
// the tile kernels.  Cutoffs separate per-CCD-L3-resident shapes
// (small) from L3-tight shapes (medium) from DRAM-streaming (large).
inline constexpr int    kDecodeMaxM   = 32;                // per-expert M ≤ this → "decode"
inline constexpr int    kMinNTile     = 512;               // prompt-path per-thread N
inline constexpr int    kDecodeNTile  = 256;               // decode-path per-thread N
inline constexpr size_t kSmallWeight  = 16UL * 1024UL * 1024UL;  // 16 MB / expert
inline constexpr size_t kMediumWeight = 64UL * 1024UL * 1024UL;  // 64 MB / expert

// Op2's K-dimension as a function of the fused activation.  Gated
// activations (swiglu/silu/gelu_and_mul) collapse the [gate, up] pair
// into half the columns, so Op2 sees K_down = N/2.  Without an
// activation Op1's full output flows into Op2, so K_down = N.  The
// caller's `down_weight[i]` must be shaped accordingly:
//   * act != none → [N/2, N_down] row-major (or [N_down, N/2] transB).
//   * act == none → [N,   N_down] row-major (or [N_down, N  ] transB).
//
// Shared between the dispatcher's Phase-F validator
// (`group_matmul_direct.cpp::validate_group_matmul_direct_inputs`)
// and the fused-MoE execute path (`group_matmul_fused_moe.cpp`) so
// both apply the same `ldb_down` minimum.  Was previously a private
// helper in `group_matmul_fused_moe.cpp`'s anonymous namespace; the
// validator was independently using `N[i] / 2` unconditionally,
// which under-restricted `ldb_down` for `act == none` callers.
inline int op2_k_for_act(int n_op1, grp_matmul_gated_act_t act) {
  return (act == grp_matmul_gated_act_t::none) ? n_op1 : (n_op1 / 2);
}

// ──────────────────────────────────────────────────────────────────────
// Env-driven feature flags.
// ──────────────────────────────────────────────────────────────────────

/// ZENDNNL_GRP_MATMUL_ALGO = "1".."5" force a specific ALGO, "0"/unset
/// = auto-select.  Intentionally NOT cached — gtests toggle mid-process
/// via AlgoEnvGuard (test_group_matmul.cpp).
inline int get_grp_matmul_algo() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
  return (env && env[0] >= '1' && env[0] <= '5') ? (env[0] - '0') : 0;
}

// ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT = { "0", "1" } — cached, default ON.
//   ALGO 3 folds a supported gated activation into the per-thread
//   epilogue (saves a second OMP pass over dst).  Adds one OMP barrier
//   between matmul-write and activation-read for correctness.  Net
//   win on every benchmarked workload; env retained as A/B escape
//   hatch.  Mid-process env changes have no effect (static const).
inline bool get_grp_n_tile_fused_act() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT");
    if (e == nullptr || e[0] == '\0') return true;  // default: ON
    return e[0] != '0';
  }();
  return v;
}

/// True when ALGO 3 can fuse `act` into the per-thread epilogue.
/// Extend here when new activation kinds gain epilogue support.
inline bool a3_can_fuse_act(grp_matmul_gated_act_t act) {
  return act == grp_matmul_gated_act_t::swiglu_oai_mul;
}

// ZENDNNL_GRP_MATMUL_N_ROUNDS = { 0, 1, 2, 3 } — cached, default 1.
//   ALGO 3 ManyExperts round-mode selection.  Internal tuning knob.
//     0 = auto: planner picks single-round / multi-round / balanced
//               via cost-model on wall time.
//     1 = force single-round (all experts in one round, n_thr =
//         num_threads / num_ops).  Falls back to balanced when
//         num_threads < num_ops.  CURRENT DEFAULT: production
//         sweeps showed single-round dominates on the target MoE
//         envelope at high thread counts; the auto cost-model
//         occasionally picked balanced/multi-round at boundaries
//         where single-round was within noise but had simpler
//         cache-key behaviour.  Pin to single-round so the
//         planner's choice is shape-independent and the AOCL DLP
//         per-tile cache key set stays stable across decode
//         iterations.  Set "0" to restore the original auto
//         behaviour for A/B comparisons or shape exploration.
//     2 = force multi-round legacy fixed-shape (batch experts × ccd_size
//         threads each, possibly wasteful tail round).
//     3 = force balanced (n_rounds = ceil(num_ops / target_batch),
//         experts evenly redistributed across rounds).
//   Mid-process env changes have no effect; relaunch for A/B.
inline int get_grp_n_rounds_mode() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_ROUNDS");
    if (e == nullptr || e[0] == '\0') return 1;
    const int parsed = std::atoi(e);
    if (parsed == 1) return 1;
    if (parsed == 2) return 2;
    if (parsed == 3) return 3;
    return 0;
  }();
  return v;
}

// `kDecodeTileAbOn` documents the production decode-tile-AB
// behaviour as an unconditional constant: when max_M ≤ kDecodeMaxM,
// FewExperts/ManyExperts use kDecodeNTile (256) instead of kMinNTile
// (512) as the per-thread N-tile bound — doubles max_n_thr for
// decode-shape down_proj.  Was previously a getter
// (`get_grp_n_decode_tile_ab`) preserving function shape for a
// future experimental env knob, but it never read any env and has
// stayed unconditionally `true` since introduction.  Replaced with
// a `constexpr` so the compiler folds it at the call site
// (`group_matmul_n_tile.cpp::plan_group_n_tile`); a future env knob
// can be re-introduced cleanly by replacing this constant with the
// getter when needed.
inline constexpr bool kDecodeTileAbOn = true;

// ZENDNNL_GRP_MATMUL_N_ORDER = { 0..4 } — cached, default 0 (auto).
//   Permutation of experts walked by ALGO 3 FewExperts/ManyExperts.
//     0 = auto: shape-aware picker (auto_pick_n_order); resolved
//         sub-mode is logged in [Level3 flat_n_tile] APILOG.
//     1 = ascending  — by M, lightest first.
//     2 = descending — by M, heaviest first; minimises
//                      Σ max_M_per_round under fixed-batch rounds.
//     3 = pair-balanced — desc, then interleave largest with smallest
//                         (heavy/light alternation).  Caveat: long-
//                         tailed M still 2-bucket sum-imbalanced;
//                         use mode 4 for those.
//     4 = balanced-spread — prefix-sum-balanced: any K-way consecutive
//                           split yields Σ M per chunk ≈ total / K
//                           (heavies evenly distributed throughout).
//   Mid-process env changes have no effect; relaunch for A/B.
inline int get_grp_matmul_n_order() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_ORDER");
    if (e == nullptr || e[0] == '\0') return 0;
    const int parsed = std::atoi(e);
    if (parsed >= 0 && parsed <= 4) return parsed;
    return 0;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT = { "0", "1" } — cached, default 1.
//   Fused MoE Op1 → act → Op2 arena layout: tight [M, I] when 1,
//   wide [M, 2I] when "0".  Tight halves Op2's src DRAM traffic and
//   is neutral-or-positive on every measured workload.  The dispatcher
//   only engages tight when ALL of: act=swiglu_oai_mul, Op1 internal-
//   alloc on, ALGO 3 selected (auto or env-forced), shape-adaptive
//   picker agrees — otherwise falls back to wide silently regardless
//   of this env.  Note: ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT is NOT a
//   tight-engagement gate — when the fused-MoE picker hands the
//   dispatcher a tight destination (ldc < N), the dispatcher auto-
//   enables ALGO 3 fused activation regardless of N_TILE_FUSED_ACT
//   (tight is a correctness constraint on the writer, not a perf
//   toggle).  See `pick_fused_moe_want_tight` in
//   group_matmul_fused_moe.cpp for the full predicate.  Set "0" here
//   to force wide (debug / layout-regression bisection).
inline int get_grp_matmul_fused_moe_tight() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT");
    if (e == nullptr || e[0] == '\0') return 1;  // default: force-tight
    return (e[0] == '0') ? 0 : 1;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL = { "0", "1" } — cached, default ON.
//   Master switch for the hand-rolled AVX-512-BF16 microkernel
//   (custom_kernel/).  ON: ALGO 3 flat_n_tile dispatches per-tile
//   GEMM through VDPBF16PS, with swiglu_oai_mul applied in-register
//   (writes activated I cols at caller's ldc — covers both wide and
//   tight fused-MoE layouts in one path).  OFF: per-tile GEMM goes
//   through the standard AOCL DLP / BRGEMM dispatch, fused activation
//   runs as a separate per-tile pass.
//
//   Why default ON: production-sweep results across the target MoE
//   envelope showed the custom kernel a consistent win — fewer
//   weight-reorder spikes (the pack arena is eviction-immune,
//   unlike the AOCL DLP LRU under capacity pressure), in-register
//   swiglu_oai_mul fusion that halves Op2 src DRAM traffic on the
//   tight layout, and bit-identical FP32→BF16 conversion to the
//   reference path.  The dispatcher refuses cleanly and falls
//   back to the standard AOCL DLP path for any expert that
//   violates the kernel's contract (non-bf16, transA, alpha≠1,
//   β≠0, N % pack_nr ≠ 0, non-const weights, etc. — see
//   `custom_kernel/dispatch.cpp::prepare_for_call` for the full
//   gate cascade), so callers outside the supported envelope see
//   no behaviour change.  Callers that want the pre-flip path
//   (e.g. for A/B perf comparison or to side-step the CK pack
//   arena's resident memory cost on memory-constrained hosts)
//   set this knob to "0".
inline bool get_grp_matmul_custom_kernel() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL");
    if (e == nullptr || e[0] == '\0') return true;  // default: ON
    return e[0] != '0';
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_CROSS_WARM = { "0", "1" } — cached, default ON.
//   When ON, each `prepack_for_algo_X` opportunistically populates the
//   cache regime that auto-select would route the OTHER phase to in the
//   same process, so a deployment that fires only prompt during warmup
//   still arrives at decode with both regimes warm — no first-decode-
//   call prepack spike.
//
//   Cross-warm decision is `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL`-aware:
//
//     * `CK=1` (custom-kernel on, production decode path):
//         - prompt → ALGO 1 (full-weight AOCL)
//                  + cross-warm regime 3 (custom-kernel pack)
//         - decode → ALGO 3 + custom kernel (regimes 2 + 3)
//                  + cross-warm regime 1 (full-weight AOCL)
//       Memory cost on many-experts MoE at high thread counts is
//       sizeable (full-weight + custom-kernel pack).  Regime 2
//       (per-tile AOCL) is still populated by `prepack_for_algo_3` if
//       `STABLE_NTILE=1` — Fix B (a separate planned commit) skips
//       it when CK is on.
//
//     * `CK=0` (AOCL DLP for both phases):
//         - prompt → ALGO 1 (full-weight AOCL)
//                  + cross-warm regime 2 (per-tile AOCL with nr_align=1)
//         - decode → ALGO 3 + AOCL DLP (regime 2)
//                  + cross-warm regime 1 (full-weight AOCL)
//       Memory cost is dominated by the full-weight + per-tile sum
//       and can be substantial on many-experts MoE at high thread
//       counts.  The cross-warmed regime 2 uses nr_align=1 (Op2
//       non-tight path).  Op1 tight (nr_align=2 under CK=0) still
//       pays a one-time lazy warm on its first decode call — full
//       coverage of both nr_align variants is a separate option
//       (2b) not yet implemented.
//
//   OFF — reverts to the strict per-ALGO regime populated by
//   `prepack_for_algo_X` itself: each ALGO only warms what it would
//   itself use at runtime.  Use this when memory is constrained or
//   for A/B comparison with the pre-cross-warm behaviour.
//
//   Independent of `ZENDNNL_GRP_MATMUL_PREPACK`: the master knob
//   short-circuits everything to a no-op; this knob only controls
//   the cross-regime fan-out when the master is ON.
inline bool get_grp_matmul_cross_warm() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CROSS_WARM");
    if (e == nullptr || e[0] == '\0') return true;   // default: On
    return e[0] != '0';
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_PREPACK = { "0", "1" } — cached, default ON.
//   Master switch for the ahead-of-time weight prepack module
//   (group_matmul/prepack/).  Single uniform semantic:
//
//   ON  — each scheduling-ALGO body invokes its matching
//         `prepack_for_algo_X(...)` as the first action, which
//         eagerly warms the inner-kernel weight cache for
//         `p.num_ops_total` experts BEFORE the matmul kicks off.
//         `p.num_ops_total` is resolved by
//         `group_matmul_prepack::build_prepack_params` from the
//         framework-hint fields, in priority order:
//
//           a) `params[0].total_matmul`  when set, or
//           b) `params[0].active_matmul` when set, or
//           c) `M.size()`                 (legacy fall-back).
//
//         By construction `p.num_ops_total >= M.size()` for every
//         supported call pattern, so the warmed set covers every
//         firing expert plus any prepack-extras tail.  Two regimes
//         share this single code path:
//
//           * Framework-hint regime (`params[0].total_matmul >
//             params[0].active_matmul`):  prepack warms the full
//             `total` set, including the prepack-extras tail of
//             experts that aren't firing this call but may fire on
//             a future call (the production MoE rotating-experts
//             use case the module was designed for).
//
//           * Active-only regime (`active_matmul > 0 &&
//             total_matmul == 0`): no rotating-experts hint, so
//             `build_prepack_params` resolves `num_ops_total` to
//             `active_matmul`; the warmer prefills exactly the
//             firing experts and skips the prepack-extras tail.
//
//           * Legacy / no-hint regime (`active=total=0` →
//             `build_prepack_params` resolves both to `M.size()`):
//             prepack warms exactly the firing experts up front.
//             This is a one-time first-iter serial reorder cost
//             paid in exchange for `do_tile()` cache hits in
//             subsequent iterations of the same configuration
//             (subsequent calls short-circuit via the per-thread
//             fingerprint cache).  Steady-state throughput is
//             identical to the lazy path; first-iter latency is
//             measurably higher (N × reorder_per_expert vs the
//             ~one-reorder parallel-cache-fill the lazy path
//             achieved).  Callers that care about first-iter
//             latency more than they care about steady-state
//             determinism should set this knob to "0".
//
//   OFF — every per-ALGO function short-circuits at entry.  No
//         warm-pack runs.  AOCL DLP / custom-kernel caches still
//         populate lazily inside `run_dlp(...)` / `prepare_for_call`
//         on first miss.  Behaviour is identical to a build without
//         the prepack module compiled in (the original pre-PR
//         library semantics).
//
//   Why default ON: a single coherent semantic is easier to reason
//   about than a conditional gate.  Production deployments that
//   integrate the framework `total_matmul` contract get the
//   prepack-extras benefit out of the box; deployments that don't
//   integrate (legacy / unit tests / single-shot inference) get
//   eager warm-up of the firing experts (small one-time cost) and
//   warm caches for everything afterwards.  Callers that need the
//   strict "no behaviour change vs pre-PR" guarantee set this knob
//   to "0" — that path is also covered by the env-matrix gtests in
//   `group_matmul/test_prepack.cpp` ([26]-[28]).
inline bool get_grp_matmul_prepack() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_PREPACK");
    if (e == nullptr || e[0] == '\0') return true;   // default: On
    return e[0] != '0';
  }();
  return v;
}

// ──────────────────────────────────────────────────────────────────────
// AOCL-path stable N-tile (strict, num_threads-only)
//
// AOCL DLP / BRGEMM / oneDNN reorder caches key on the per-thread
// slice — `(transB, K, n_tile, ldb, B + col_start·elem, algo)` —
// NOT on the full weight.  When `n_thr_per_expert` varies between
// calls (active-expert filtering, batch-size shifts, …), col_start
// and n_tile rotate, the cache thrashes, and under churn the LRU
// can free entries another thread is still reading via raw pointer
// → use-after-free → garbage rows.  The custom kernel sidesteps
// this (its pack cache is shape-keyed), so this whole subsystem is
// non-custom only.
//
// Mitigation: for non-custom dispatch under ALGO 3 flat_n_tile,
// pin the per-expert thread count to a `num_threads`-only formula:
//
//     stable = max(1, num_threads / kAoclTargetConcurrentSlots)
//
// `stable` depends ONLY on `num_threads` and the env-static
// kAoclTargetConcurrentSlots — invariant across MoE routing,
// expert filtering, N shifts, and num_ops shifts.  The planner
// (`plan_group_n_tile`) then forces `n_thr_fixed = stable` and
// `batch_size = num_threads / stable` so every expert team has
// exactly `stable` threads in every round.  `col_start` and
// `n_tile` therefore stay byte-identical across calls → AOCL
// cache reaches a steady hit-rate post-warmup, regardless of any
// caller-side variation.
//
// Why the formula does not include an N-dependent density floor:
// an earlier `by_density = N / kAoclBlisNc` term protected thin-N
// shapes above AOCL's NC=128 amortisation point, but for variable-
// N MoE callers it re-introduced an N-dependence into `stable`,
// rotating the cache key per expert and undermining the stability
// contract.  Narrow-N protection is now handled by a planner-side
// escape: when `stable * nr_align > max_N` (the regime where
// `aligned_n_split` cannot produce stable aligned slices),
// `plan_group_n_tile` routes the call to Sequential which uses
// the full thread team per expert and bypasses tile-level cache
// keys entirely.
//
// Trade-off: at low num_ops (num_ops × stable < num_threads),
// some threads idle in the strict-stable plan.  Accepted as the
// cost of the cache-stability guarantee — for typical MoE decode
// workloads, the per-call cache-hit savings (tens of milliseconds
// of avoided reorders) dominate the per-call thread-utilisation
// loss (sub-ms).
//
// `participating_n_thr` (group_matmul_n_tile.cpp) retains secondary
// clamps by `align_cap = N / nr_align` and `team_size` as defence-
// in-depth: the strict-stable planner already guarantees
// `team_size == stable` and `align_cap >= stable` (else the
// narrow-N escape fires), so the clamps are no-ops in the strict-
// stable plan — they only fire if a future planner regression
// breaks an invariant, in which case they degrade gracefully to
// dynamic-tile behaviour rather than silently corrupting output.
// ──────────────────────────────────────────────────────────────────────

// ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE = { "0", "1" } — cached, default ON.
//   "0" restores the legacy dynamic plan topology (cache thrash)
//   for A/B benchmarking.
inline bool get_grp_matmul_aocl_stable_ntile() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE");
    if (e == nullptr || e[0] == '\0') return true;
    return e[0] != '0';
  }();
  return v;
}

inline constexpr int kAoclTargetConcurrentSlots = 16;   // team-budget divisor
inline constexpr int kAoclBlisNc                = 128;  // BLIS-bf16 inner-N block

// ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS = positive int — cached, default 16.
//   Team-budget divisor.  Lower (e.g. 8) for few-expert deployments;
//   raise for many-expert deployments where reducing per-expert fan-
//   out helps.  Non-positive → default.
inline int get_grp_matmul_aocl_target_slots() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS");
    if (e == nullptr || e[0] == '\0') return kAoclTargetConcurrentSlots;
    const int parsed = std::atoi(e);
    return (parsed > 0) ? parsed : kAoclTargetConcurrentSlots;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_AOCL_BLIS_NC = positive int — cached, default 128.
//   AOCL density floor: each thread's slice must be ≥ NC cols for
//   reorder amortisation to win.  Lower → more parallelism per
//   expert / thinner slices; higher → wider slices / fewer slots.
//   Non-positive → default.
inline int get_grp_matmul_aocl_blis_nc() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_BLIS_NC");
    if (e == nullptr || e[0] == '\0') return kAoclBlisNc;
    const int parsed = std::atoi(e);
    return (parsed > 0) ? parsed : kAoclBlisNc;
  }();
  return v;
}

// Per-expert thread count for the AOCL DLP / BRGEMM / oneDNN execute
// path inside ALGO 3 flat_n_tile.  See the strict-stable doc-block
// above for the cache-stability contract and rationale.
//
// Depends ONLY on `num_threads`.  An earlier formula included a
// `by_density = N / kAoclBlisNc` term which re-introduced an
// N-dependence into the cache key.  Narrow-N protection (where
// `by_density` was needed in the first place) is now handled by
// `plan_group_n_tile`'s narrow-N escape — calls that
// can't produce stable-aligned tiles are routed to Sequential
// instead.
//
// The `N` parameter is retained for source-level compatibility with
// existing callers; it is intentionally unused.
inline int aocl_stable_n_thr(int num_threads, int /*N*/) {
  if (num_threads <= 0) return 1;
  return std::max(1, num_threads / get_grp_matmul_aocl_target_slots());
}

// ──────────────────────────────────────────────────────────────────────
// Custom microkernel sub-knobs (only consumed when
// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1).  All cached as static const.
// ──────────────────────────────────────────────────────────────────────

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR = { unset, "32", "64" } — cached.
//   Pack/microkernel NR override.  Auto (unset) → 32 (cleanest
//   register budget, MR=8 on NV=2).  64 doubles N-lanes per zmm at
//   MR cap 6 — worth trying on prompt shapes.  Other values → auto.
inline int get_grp_matmul_custom_kernel_nr() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR");
    if (e == nullptr || e[0] == '\0') {
      return 0;
    }
    int parsed = std::atoi(e);
    return (parsed == 32 || parsed == 64) ? parsed : 0;
  }
  ();
  return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT = { "0", "1" } — cached, default OFF.
//   Per-expert L2-friendly subtile_cols (vs. one m_max-sized value
//   for the whole call).  Noise-floor on typical MoE decode shapes;
//   may help on large-L2 hosts or workloads with extreme M variance.
inline bool get_grp_matmul_custom_kernel_subtile_per_expert() {
  static const bool v = []() {
    const char *e = std::getenv(
                      "ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT");
    return (e != nullptr && e[0] != '\0' && e[0] != '0');
  }
  ();
  return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE = { unset, multiple of 32 } — cached.
//   Override the ALGO 3 outer N-tile minimum (default kDecodeNTile=256).
//   Use 128 for high threads/num_ops (more thread participation), 512
//   for prompt-class (wider tiles amortise kernel-call overhead).
//   Non-multiples of 32 → ignored (silently safe vs typos).
inline int get_grp_matmul_custom_kernel_n_tile() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE");
    if (e == nullptr || e[0] == '\0') return 0;
    const int parsed = std::atoi(e);
    return (parsed > 0 && (parsed % 32) == 0) ? parsed : 0;
  }
  ();
  return v;
}

/// N alignment the inner kernel prefers for each per-thread slice
/// (1 = no alignment).  ALGO 3's column partitioner (aligned_n_split)
/// honours it when the slowest thread stays within 2× of the fastest
/// after rounding; otherwise it falls back to the unaligned even split.
inline int backend_n_align(matmul_algo_t algo) {
  switch (algo) {
  case matmul_algo_t::native_brgemm:
  case matmul_algo_t::native_gemm:
    return 64;
  default:
    return 1;
  }
}

/// Aligned column-slice partitioner for ALGO 3.  Returns {col_start,
/// col_end} for `tid` of `n_thr` over [0, N).
///
/// Aligned branch: per-thread = aligned_per_thr (a multiple of `align`),
/// last thread takes the remainder.  Engaged only when last ≥
/// aligned_per_thr/2 (BLIS-style 2× imbalance bound).  Search walks
/// DOWN in `align` quanta from ceil(N/n_thr) rounded up to align,
/// so feasible alignments aren't rejected just because the first
/// candidate over-sized the last slice (e.g. N=2880, n_thr=11,
/// align=64 → 256 cols/thread fits, 320 doesn't).
///
/// Falls back to even split (N*tid/n_thr) when n_thr<=1 or align<=1
/// or no aligned slice meets the 2× bound.
inline std::pair<int, int> aligned_n_split(int N, int n_thr, int tid,
    int align) {
  // Hardened against pathological inputs: n_thr<=0 used to hit
  // even_split's divide-by-zero (`N * tid / n_thr`).
  if (n_thr <= 0 || N <= 0) {
    return std::make_pair(0, 0);
  }

  auto even_split = [&]() {
    const int s = static_cast<int>(static_cast<int64_t>(N) * tid / n_thr);
    const int e = static_cast<int>(
                    static_cast<int64_t>(N) * (tid + 1) / n_thr);
    return std::make_pair(s, e);
  };

  if (align <= 1 || n_thr <= 1) {
    return even_split();
  }

  // Walk slice size down in `align` quanta from ceil(N/n_thr) until
  // the imbalance bound holds.  Cost is at most a handful of
  // iterations in practice (slice sizes converge in 1-2 steps for
  // realistic N / n_thr / align triples).
  //
  // Intermediates promoted to int64_t to keep the products
  // aligned_per_thr * (n_thr - 1) and aligned_per_thr * tid free of
  // signed overflow (UB) for any (N, n_thr) combination representable
  // as int.
  const int64_t even_per_thr =
    (static_cast<int64_t>(N) + n_thr - 1) / n_thr;
  for (int64_t aligned_per_thr =
         ((even_per_thr + align - 1) / align) * align;
       aligned_per_thr >= align;
       aligned_per_thr -= align) {
    const int64_t n_full = aligned_per_thr * (n_thr - 1);
    const int64_t last = static_cast<int64_t>(N) - n_full;
    if (last > 0 && last * 2 >= aligned_per_thr) {
      const int64_t s = aligned_per_thr * tid;
      const int64_t e =
        (tid < n_thr - 1) ? aligned_per_thr * (tid + 1) : N;
      return std::make_pair(static_cast<int>(s), static_cast<int>(e));
    }
  }
  return even_split();
}

/// 32 MB of L3 per CCD on Zen 3 / 4 / 5 classic-CCD topologies.
inline constexpr size_t kL3PerCcdBytes = 32UL * 1024UL * 1024UL;

/// Aggregate L3 the planner uses to bound the experts-per-round
/// budget (ALGO 3 N-tile, ALGO 4 multilevel).  `num_ccds` comes from
/// summarise_topology() so this stays consistent with how the rest
/// of the planner partitions the team.
inline size_t get_grp_l3_total_bytes(int num_ccds) {
  return static_cast<size_t>(std::max(1, num_ccds)) * kL3PerCcdBytes;
}

// ──────────────────────────────────────────────────────────────────────
// Shared per-expert primitives.
// ──────────────────────────────────────────────────────────────────────

/// Resolves the effective matmul algo ID from the runtime config,
/// falling back to AOCL DLP blocked when the config is unset/invalid.
inline matmul_algo_t resolve_kernel() {
  static const matmul_algo_t algo = []() {
    int32_t a = matmul_config_t::instance().get_algo();
    if (a <= 0 || a >= static_cast<int32_t>(matmul_algo_t::algo_count)) {
      return matmul_algo_t::aocl_dlp_blocked;
    }
    return static_cast<matmul_algo_t>(a);
  }
  ();
  return algo;
}

/// Thin wrapper around matmul_execute that packages per-expert slice
/// arguments into the batch/params objects the kernel expects.
inline void execute_expert_slice(
  char layout, bool transA, bool transB,
  int M, int N, int K, float alpha,
  const void *src, int lda,
  const void *weight, int ldb,
  const void *bias, float beta,
  void *dst, int ldc,
  bool is_weights_const, int num_thr,
  matmul_params &params,
  matmul_algo_t algo) {

  matmul_batch_params_t bp;
  bp.Batch_A = 1;
  bp.Batch_B = 1;
  matmul_algo_t kernel = algo;

  int         reordered_lda = lda;
  size_t      src_type_size = size_of(params.dtypes.src);
  reorder_quant_buffers_t quant_buffers;
  if (reorder_quantization_wrapper(src, lda, reordered_lda, src_type_size,
                                   params, bp, transA, M, K,
                                   num_thr, quant_buffers) != status_t::success) {
    log_error("execute_expert_slice: reorder_quantization_wrapper failed");
    return;
  }

  matmul_execute(layout, transA, transB,
                 M, N, K, alpha, src,
                 params.dynamic_quant ? reordered_lda : lda,
                 weight, ldb, bias, beta, dst, ldc,
                 is_weights_const, src_type_size,
                 size_of(params.dtypes.dst),
                 num_thr, kernel, params, bp, 0);
}

// ──────────────────────────────────────────────────────────────────────
// N-tile shared utilities (consumed by group_matmul_n_tile.cpp; kept
// shared so a future additional N-tile executor can pick them up).
// ──────────────────────────────────────────────────────────────────────

/// Sort `indices[0..n)` by `M[idx]` (asc or desc).  Heap-free; caller
/// owns the buffer and guarantees `indices.size() >= n` and `M.size() >= n`.
inline void sort_indices_by_m(int *indices, int n,
                              const std::vector<int> &M, bool ascending) {
  for (int i = 0; i < n; ++i) {
    indices[i] = i;
  }
  std::sort(indices, indices + n, [&M, ascending](int a, int b) {
    return ascending ? (M[a] < M[b]) : (M[a] > M[b]);
  });
}

/// Env-gated BF16 custom-microkernel engagement for an N-tile executor.
/// Caller stack-allocates a fresh `CallContext` and reads `kctx.enabled`
/// on return.  Default OFF (env=1 to opt in).  Even with env=1, the
/// dispatcher's contract check (bf16, no transA, α=1, β=0, N % pack_nr,
/// supported act/bias dtypes, is_weights_const = true / empty for every
/// active expert) decides whether the path can actually run; caller
/// falls back to its standard path when kctx.enabled=false.
/// Supported `act`: none, swiglu_oai_mul (fused gate+up → halved out).
inline void engage_ntile_custom_kernel(
    grp_matmul_gated_act_t act,
    data_type_t src_dtype,
    data_type_t wei_dtype,
    data_type_t dst_dtype,
    data_type_t act_dtype,
    data_type_t bias_dtype,
    const std::vector<bool>          &transA,
    const std::vector<bool>          &transB,
    const std::vector<int>           &M,
    const std::vector<int>           &N,
    const std::vector<int>           &K,
    const std::vector<int>           &ldb,
    const std::vector<float>         &alpha,
    const std::vector<float>         &beta,
    const std::vector<const void *>  &weight,
    const std::vector<bool>          &is_weights_const,
    custom_kernel::CallContext       &kctx) {
  if (!get_grp_matmul_custom_kernel()) return;
  custom_kernel::prepare_for_call(
      act, src_dtype, wei_dtype, dst_dtype, act_dtype, bias_dtype,
      transA, transB, M, N, K, ldb, alpha, beta, weight,
      is_weights_const, kctx);
}

/// Effective N-column alignment for the per-thread split:
///   max(backend_nr, kctx.pack_nr if enabled, 2 if pair_aligned).
/// `backend_nr` from backend_n_align(algo); `pair_aligned`=true when
/// activation requires even col boundaries (e.g. swiglu_oai_mul must
/// keep gate+up pairs on the same thread).
inline int ntile_effective_nr_align(
  int backend_nr,
  const custom_kernel::CallContext &kctx,
  bool pair_aligned) {
  int a = backend_nr;
  if (kctx.enabled) {
    a = std::max(a, kctx.pack_nr);
  }
  if (pair_aligned) {
    a = std::max(a, 2);
  }
  return a;
}

// Heap-free upper bound on num_ops for N-tile expert ordering.
// Matches GroupNTilePlan::kMaxExperts.
inline constexpr int kNTileMaxExperts = 256;

/// Auto-pick N_ORDER from num_ops.  Returns 3 (pair_balanced) or 0
/// (walk input order); explicit modes 1/2/4 only reachable via env.
///
/// Rule was derived from an internal MoE-decode sweep across a wide
/// num_ops × shape × mode grid.  Pair-balanced ordering improves
/// throughput at the low- and high-num_ops ends of the range:
///   * Few-experts (≤ ~kSmallExpertsCutoff): the bin-packing benefit
///     of pair-balanced outweighs the walk-input alternative.
///   * Many-experts (≥ ~kLargeExpertsCutoff): pair-balanced is again
///     dominant and the alternative shows essentially no wins.
///   * Mid-band: the two strategies trade wins/losses, so we default
///     to walk-input (lower overhead and zero permutation cost).
/// The cutoffs are calibrated against the dispatcher's stable-N-tile
/// plan and should be re-evaluated together if that planner changes.
inline int auto_pick_n_order(int num_ops) {
  if (num_ops <= 18) {
    return 3;  // few-experts regime — pair-balanced is the better default
  }
  if (num_ops >= 26) {
    return 3;  // many-experts regime — pair-balanced is the better default
  }
  return 0;    // mid-band — walk input order
}

/// Write `out[0..out_size)` with expert indices ordered per
/// `get_grp_matmul_n_order()`.  `out_size = 0` signals "walk input
/// order, ignore out" (auto-mode resolved to no permutation).
///
/// Heap-free: stack array of kNTileMaxExperts for the desc-sort temp;
/// beyond that the ordering is skipped (correct, just unsorted).
/// Mode 4 is O(num_ops²) ≤ 64K comparisons — well under 10 µs.
///
/// `auto_resolved_out` (optional): when env mode = 0, the resolved
/// concrete sub-mode is written here for APILOG diagnostics.
inline void fill_ntile_expert_order(
  int *out, int &out_size, int max_size,
  const std::vector<int> &M, int num_ops,
  int *auto_resolved_out = nullptr) {

  if (num_ops <= 0 || num_ops > max_size
      || num_ops > kNTileMaxExperts) {
    out_size = 0;
    return;
  }

  int order = get_grp_matmul_n_order();

  // Mode 0 — auto: shape-aware sub-mode selection.
  if (order == 0) {
    order = auto_pick_n_order(num_ops);
    if (auto_resolved_out != nullptr) {
      *auto_resolved_out = order;
    }
    if (order == 0) {
      // Auto chose walk-input; leave out empty.
      out_size = 0;
      return;
    }
  }

  // Modes 1 (ascending) and 2 (descending) are direct sorts.
  if (order == 1 || order == 2) {
    const bool ascending = (order == 1);
    sort_indices_by_m(out, num_ops, M, ascending);
    out_size = num_ops;
    return;
  }

  // Mode 3 — pair-balanced: descending sort, then interleave
  // (largest, smallest, 2nd-largest, 2nd-smallest, …) so each round
  // sees a mix of heavy and light experts.
  if (order == 3) {
    std::array<int, kNTileMaxExperts> sorted_desc{};
    sort_indices_by_m(sorted_desc.data(), num_ops, M,
                      /*ascending=*/false);
    int lo = 0, hi = num_ops - 1, o = 0;
    while (lo <= hi) {
      out[o++] = sorted_desc[lo++];
      if (lo <= hi) {
        out[o++] = sorted_desc[hi--];
      }
    }
    out_size = num_ops;
    return;
  }

  // Mode 4 — balanced-spread: at each output position p, pick the
  // remaining (sorted-desc) item whose M brings the running prefix
  // sum closest to the ideal line `y = (p + 1) * total / num_ops`.
  //
  // Result: for any K, splitting the output into K equal-length
  // consecutive chunks yields Σ M per chunk ≈ total / K.  Heavy
  // experts land at positions ≈ i × num_ops / num_heavies, so the
  // round scheduler sees at most ONE heavy expert per CCX slot for
  // typical (batch_size, ccd_size) choices.
  {
    std::array<int, kNTileMaxExperts> sorted_desc{};
    sort_indices_by_m(sorted_desc.data(), num_ops, M,
                      /*ascending=*/false);

    int64_t total = 0;
    for (int i = 0; i < num_ops; ++i) {
      total += M[i];
    }

    std::array<bool, kNTileMaxExperts> used{};  // zero-init
    int64_t cum = 0;
    for (int p = 0; p < num_ops; ++p) {
      // Error metric: |target_scaled − num_ops × new_cum| where
      // target_scaled = (p + 1) × total.  Integer-only; scales
      // cancel out across candidates.
      const int64_t target_scaled =
        static_cast<int64_t>(p + 1) * total;
      int best_j = -1;
      int64_t best_err = std::numeric_limits<int64_t>::max();
      for (int j = 0; j < num_ops; ++j) {
        if (used[j]) {
          continue;
        }
        const int64_t new_cum = cum + M[sorted_desc[j]];
        const int64_t err = std::llabs(
                              target_scaled - static_cast<int64_t>(num_ops) * new_cum);
        if (err < best_err) {
          best_err = err;
          best_j = j;
        }
      }
      used[best_j] = true;
      out[p] = sorted_desc[best_j];
      cum += M[sorted_desc[best_j]];
    }
    out_size = num_ops;
    return;
  }
}

// ──────────────────────────────────────────────────────────────────────
// Tile-strategy entry points — library-internal linkage; the
// dispatcher in group_matmul_parallel.cpp forwards to these.
// ──────────────────────────────────────────────────────────────────────

/// ALGO 2 — M-tile parallel GEMM.  Defined in group_matmul_m_tile.cpp.
void flat_m_tile(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K, const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  int num_threads);

/// ALGO 3 — N-tile parallel GEMM, with optional fused-swiglu-oai
/// epilogue.  Defined in group_matmul_n_tile.cpp.
///
/// `gemm_mode_out` (optional) receives a static string naming the
/// concrete path that ran: `"flat_n_tile"`, `"flat_n_tile_custom"`,
/// `"flat_n_tile_fused_swiglu_oai"`, or `"flat_n_tile_fused_swiglu_oai_custom"`.
/// The caller is expected to thread this through to its own
/// gemm_mode_out so benchdnn / profiler output reveals whether the
/// custom BF16 microkernel engaged.
void flat_n_tile(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K, const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  int num_threads,
  grp_matmul_gated_act_t fused_act = grp_matmul_gated_act_t::none,
  data_type_t act_dtype = data_type_t::none,
  const char **gemm_mode_out = nullptr);

/// Peek at the ALGO the dispatcher would pick for this call (1=seq,
/// 2=m_tile, 3=n_tile, 4=multilevel, 5=per_expert).  Mirrors the
/// dispatcher's full gating: ZENDNNL_GRP_MATMUL_ALGO override, m/n
/// tile-safety checks, auto_select_algo on env=0.  Pure observer
/// (no side-effects); used by the fused-MoE entry to choose tight
/// vs wide arena before committing the buffer layout.
int select_grp_matmul_algo(
  const std::vector<char> &layout,
  const std::vector<int> &M,
  const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<matmul_params> &params,
  int num_threads);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif  // ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
