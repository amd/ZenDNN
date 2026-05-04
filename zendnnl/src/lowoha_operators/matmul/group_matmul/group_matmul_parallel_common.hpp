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

// ──────────────────────────────────────────────────────────────────────
// Display / logging helpers shared across group_matmul sources.
//
// `act_name()` renders a `grp_matmul_gated_act_t` as a short literal
// for APILOG lines and gemm_mode strings.  Centralised here so every
// fused / non-fused executor logs the same names (e.g. "swiglu_oai_mul"
// not "swiglu_oai") and new activation kinds only need one edit.
// ──────────────────────────────────────────────────────────────────────
inline const char *act_name(grp_matmul_gated_act_t a) {
  switch (a) {
    case grp_matmul_gated_act_t::none:           return "none";
    case grp_matmul_gated_act_t::silu_and_mul:   return "silu_and_mul";
    case grp_matmul_gated_act_t::gelu_and_mul:   return "gelu_and_mul";
    case grp_matmul_gated_act_t::swiglu_oai_mul: return "swiglu_oai_mul";
  }
  return "?";
}

// ──────────────────────────────────────────────────────────────────────
// Shared tile-size constants.
//
// Used by both ALGO 0 auto-select (to score ntile/mtile viability) and
// the tile kernels themselves (to decide per-thread min-work).
// ──────────────────────────────────────────────────────────────────────

/// Upper bound on per-expert M for a shape to be considered "decode".
/// Above this threshold ALGO 0 switches to prompt-oriented heuristics
/// and the tile kernels use different per-thread minimums.
inline constexpr int kDecodeMaxM = 32;

/// Minimum N-columns per thread in prompt-path N-tile.
inline constexpr int kMinNTile = 512;

/// Minimum N-columns per thread in decode-path N-tile.
inline constexpr int kDecodeNTile = 256;

/// MoE weight-class thresholds (bytes per expert, max_N × max_K × dtype).
/// Used by ALGO 0's auto-select dispatcher *and* by ALGO 3's planner
/// internal fallback so the two paths agree on what is "small" /
/// "medium" / "large".  The cut-offs separate per-CCD-L3-resident
/// shapes (small) from L3-tight shapes (medium) from DRAM-streaming
/// shapes (large).
inline constexpr size_t kSmallWeight  = 16UL * 1024UL * 1024UL;  // 16 MB
inline constexpr size_t kMediumWeight = 64UL * 1024UL * 1024UL;  // 64 MB

// ──────────────────────────────────────────────────────────────────────
// Env-driven feature flags.
// ──────────────────────────────────────────────────────────────────────

/// ZENDNNL_GRP_MATMUL_ALGO: force a specific ALGO (1..5), or 0/unset
/// to let ALGO 0 auto-select.
///
/// Intentionally NOT cached — gtests toggle this mid-process via
/// AlgoEnvGuard (see test_group_matmul.cpp) to parameterize the
/// ALGO-under-test across test cases within a single binary run.
inline int get_grp_matmul_algo() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
  return (env && env[0] >= '1' && env[0] <= '5') ? (env[0] - '0') : 0;
}

// ──────────────────────────────────────────────────────────────────────
// ALGO 3 fused gated-activation epilogue
//
//   The N-tile path can fold a gated activation into its per-thread
//   epilogue, saving a second OMP pass over dst.  Whether a given
//   activation is fusable depends on whether each thread's tile
//   contains the inputs it needs to compute its outputs locally; the
//   set of supported activations is enumerated in a3_can_fuse_act().
//
//   Default OFF.  The fused path adds an `omp barrier` between matmul
//   write and activation read for correctness; that cost can outweigh
//   the savings of the fused epilogue on some shape / thread-count
//   combinations.  Callers opt in via:
//
//     ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT=1
//
// Per-call lookup — intentionally not cached so deployments can
// toggle the flag mid-process and gtests can exercise both paths.
// ──────────────────────────────────────────────────────────────────────
inline bool get_grp_n_tile_fused_act() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT");
  if (env == nullptr || env[0] == '\0') return false;
  return env[0] != '0';
}

/// Returns true when ALGO 3 can run a per-thread fused epilogue for
/// `act`.  Add new activation kinds here as the per-thread epilogue
/// adds support for them.
inline bool a3_can_fuse_act(grp_matmul_gated_act_t act) {
  return act == grp_matmul_gated_act_t::swiglu_oai_mul;
}

// ──────────────────────────────────────────────────────────────────────
// ALGO 3 ManyExperts round-mode selection
//
// The ManyExperts planner has two execution shapes when num_ops
// exceeds one round:
//
//   * Multi-round: experts processed in ceil(num_ops / batch) rounds
//     of `batch` experts each, every expert getting `ccd_size`
//     threads.  The last (tail) round is wasteful when num_ops mod
//     batch is small — most of the thread team idles waiting for the
//     few tail experts.
//   * Single-round: all `num_ops` experts processed in one OMP round
//     at `num_threads / num_ops` threads per expert.  Avoids the tail
//     overhead but each expert runs at fewer threads.
//
// Trade-off: single-round is preferred when the tail would be
// wasteful AND the lower per-expert thread count is still close to
// ccd_size; otherwise multi-round is preferred because it gives each
// expert the full ccd_size compute budget.
//
// `get_grp_n_rounds_mode()` returns:
//
//   0 / unset  AUTO (default).  The planner builds three candidate
//              scheduling shapes and picks the one with the lowest
//              cost-model wall time:
//                A) single-round   — all num_ops experts in one
//                                    round, n_thr = num_threads /
//                                    num_ops (capped by ccd_size and
//                                    max_tiles).  Wall ∝ 1 / n_thr.
//                B) multi-round    — fixed batch = compute_target_
//                                    batch, fixed n_thr = ccd_size.
//                                    The pre-balanced-rounds default.
//                C) balanced       — n_rounds = ceil(num_ops /
//                                    target_batch), balanced_batch =
//                                    ceil(num_ops / n_rounds), n_thr
//                                    proportional per round (capped
//                                    by max_tiles).  Saturates the
//                                    tail by redistributing experts
//                                    evenly across rounds.
//              Cost model assumes uniform M per round; when
//              ZENDNNL_GRP_MATMUL_N_ORDER ∈ {2, 3} (descending /
//              pair-balanced) is set, balanced and multi-round see
//              the heavy-M expert dominate every round, so all
//              walls scale by the same M factor and the picker is
//              correct on relative cost.  With order = 0 (default)
//              the assumption holds exactly on uniform-M workloads
//              (the common decode shape).
//   1          Force single-round.  Useful for A/B vs auto.  Falls
//              back to balanced when num_threads < num_ops.
//   2          Force multi-round (legacy fixed-shape).  Useful as a
//              regression baseline.
//   3          Force balanced.  Useful for A/B vs auto.
//
// Per-call lookup, ~50 ns per group_matmul call.
//
//   ZENDNNL_GRP_MATMUL_N_ROUNDS={0,1,2,3}
//
// SCOPE: Internal tuning / debugging knob, not a stable user-facing
// API.  Default behaviour (auto, mode 0) is the only mode production
// deployments should depend on.  Modes 1-3 exist to let CI / perf
// engineers force a specific scheduling shape for A/B comparison and
// may be removed or renumbered in future releases.  Keep this
// distinction clear when documenting in PR descriptions and release
// notes.
// ──────────────────────────────────────────────────────────────────────
inline int get_grp_n_rounds_mode() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_N_ROUNDS");
  if (env == nullptr || env[0] == '\0') return 0;
  const int v = std::atoi(env);
  if (v == 1) return 1;       // force single-round
  if (v == 2) return 2;       // force multi-round (legacy fixed-shape)
  if (v == 3) return 3;       // force balanced
  return 0;                   // auto
}

// ──────────────────────────────────────────────────────────────────────
// ALGO 3 production scheduling decisions.
//
// This helper gates a scheduling refinement that has settled into an
// always-on production default.  The function shape is preserved (as
// opposed to inlining the constant at every call site) so a future
// experiment can plumb an env / config knob here without touching
// callers in group_matmul_n_tile.cpp.
//
//   decode_tile_ab When max_M ≤ kDecodeMaxM, use kDecodeNTile (256)
//                 instead of kMinNTile (512) as the per-thread N-tile
//                 bound for FewExperts and ManyExperts paths.  Doubles
//                 the max_n_thr cap so decode-shape down_proj can use
//                 more threads per expert.
// ──────────────────────────────────────────────────────────────────────
inline bool get_grp_n_decode_tile_ab() { return true; }

// ──────────────────────────────────────────────────────────────────────
// ALGO 3 expert ordering policy — ZENDNNL_GRP_MATMUL_N_ORDER
//
// Controls how `fill_sorted_expert_order()` builds the permutation of
// experts that the FewExperts / ManyExperts executor walks.
//
// Modes:
//   0  auto         — DEFAULT.  Shape-aware selection derived from
//                    the GPT-OSS MoE decode benchmark sweep: for
//                    skewed-M workloads with num_ops in the "sweet
//                    spot", auto picks pair_balanced (mode 3); for
//                    mid-range num_ops where pair_balanced
//                    empirically regresses, auto falls back to
//                    walking experts in input order.  The concrete
//                    picker lives next to the executor in
//                    `auto_pick_n_order()` (group_matmul_n_tile.cpp)
//                    so the policy can evolve as more workload data
//                    is collected; the APILOG line for flat_n_tile
//                    reports the sub-mode auto chose for each call.
//   1  ascending    — sort by M increasing (lightest experts first).
//                    Useful for testing tail-latency hypotheses;
//                    under the barrier-based executor total
//                    Σ max_M_per_round equals descending, but
//                    downstream stages consuming Op1 outputs see
//                    the lightest experts finish first.
//   2  descending   — sort by M decreasing (heaviest experts first).
//                    Theoretically optimal for minimising
//                    Σ max_M_per_round under fixed-batch rounds:
//                    the heavy-M expert dominates round 0 (where
//                    its max already sets the wall) and the tail
//                    rounds drop to the smallest-M experts.
//   3  pair-balanced — descending sort, then interleave largest with
//                    smallest so adjacent positions alternate
//                    (heavy, light, heavy, light, …).  Each round
//                    contains a mix of heavy and light experts,
//                    which can even out per-round wall time on
//                    skewed-M shapes when cache / scheduling effects
//                    dominate the simple max(M) wall model.
//                    Caveat: when the M distribution is long-tailed
//                    (few heavies + many very-small experts), the
//                    "heavy series" exhausts in the first half of
//                    the output and the second half is all tail-
//                    lights — so a 2-way bucket split is still
//                    sum-imbalanced.  Use mode 4 for workloads where
//                    this matters.
//   4  balanced-spread — prefix-sum-balanced: the output list is
//                    arranged so that for ANY K, splitting into K
//                    equal-length consecutive chunks yields
//                    Σ M per chunk ≈ total / K.  Algorithm: sort
//                    desc, then for each position p pick the
//                    remaining item whose M brings the running
//                    prefix sum closest to (p+1) * avg.
//                    Effect: heavy experts are spread roughly
//                    evenly throughout the output list (at
//                    positions ≈ i * N / num_heavies for i = 0 .. ).
//                    The round scheduler then automatically sees
//                    approximately equal Σ M per round AND at most
//                    one heavy expert on any given CCX (because
//                    heavies are far apart in the walk order).
//                    Works well on skewed / long-tailed M profiles
//                    where pair-balanced fails the 2-bucket sum
//                    balance property.
//
// Per-call lookup (~50 ns per group_matmul call, intentionally not
// cached so deployments can A/B mid-process and gtests can exercise
// each mode).
//
// SCOPE: Tuning / experimentation knob.  Default (0, auto) is the
// shape-aware picker that routes between the explicit modes based on
// workload characteristics.  Explicit modes 1 / 2 / 3 / 4 are for
// forced A/B comparison and may be extended or renumbered as new
// orderings are added.
// ──────────────────────────────────────────────────────────────────────
inline int get_grp_matmul_n_order() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_N_ORDER");
  if (env == nullptr || env[0] == '\0') return 0;   // default: auto
  const int v = std::atoi(env);
  if (v == 0) return 0;                              // auto
  if (v == 1) return 1;                              // ascending
  if (v == 2) return 2;                              // descending
  if (v == 3) return 3;                              // pair-balanced
  if (v == 4) return 4;                              // balanced-spread
  return 0;                                          // fallback: auto
}

// ──────────────────────────────────────────────────────────────────────
// Fused MoE Op1 → activation → Op2 path selector
//   ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT = { 0, 1 }
//
// Internal-alloc arena layout selector for the fused MoE Op1
// (gate+up GEMM + activation) → Op2 (down_proj GEMM) flow.  Tri-state
// knob read once per call via `get_grp_matmul_fused_moe_tight`:
//
//   unset / empty   (auto, default)   — picker in group_matmul_fused_moe.cpp
//                   decides tight vs wide per call based on shape +
//                   algo viability.  Today the picker requests tight
//                   whenever the dispatcher would route Op1 to ALGO 3
//                   (N-tile) with swiglu_oai_mul fused activation,
//                   because flat_n_tile handles both layouts and the
//                   tight layout halves Op2's src DRAM traffic.
//   "0"             force-wide — ignore the picker and always allocate
//                   the classic [M, 2I] arena.  Useful for bisecting
//                   tight-path regressions against the pre-unification
//                   wide baseline.
//   any other       force-tight — ignore the picker and always request
//                   tight [M, I].  The fused-MoE entry still blocks
//                   tight on un-gated layouts (missing swiglu, missing
//                   internal-alloc, forced ALGO 1/2/4/5, fused-act
//                   disabled) — those cases silently fall back to
//                   wide regardless of this env.
//
// Tight layout only produces correct output when:
//   * act == swiglu_oai_mul (activation halves N, enables the tight
//     write pattern),
//   * internal_alloc is engaged (library owns arena stride),
//   * dispatcher routes Op1 to ALGO 3 (only flat_n_tile implements
//     the per-thread-scratch + out-of-place swiglu path needed for
//     tight; ALGO 1 / 2 / 4 / 5 assume wide layout and overrun the
//     tight arena).
//   * ALGO 3 fused-act env is enabled (see get_grp_n_tile_fused_act).
// The picker enforces each of these; this knob only affects the
// final pick when all of them are satisfied.
// ──────────────────────────────────────────────────────────────────────
/// Tri-state tight-arena env value.  Returns -1 for unset (auto),
/// 0 for force-wide, 1 for force-tight.  The picker in the fused-MoE
/// entry translates these into the actual per-call layout decision.
inline int get_grp_matmul_fused_moe_tight() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT");
  if (env == nullptr || env[0] == '\0') return -1;
  return (env[0] == '0') ? 0 : 1;
}

// ──────────────────────────────────────────────────────────────────────
// Custom BF16 microkernel — ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL = { 0, 1 }
//
//   0  (default) Use the standard DLP / BRGEMM dispatch via
//      execute_expert_slice() for per-tile GEMM.  Any fused
//      activation runs as a separate per-tile pass or through the
//      legacy in-place compaction (flat_n_tile's swiglu_oai
//      epilogue).
//
//   1  Use the hand-rolled BF16 microkernel (see
//      group_matmul/custom_kernel/).  Per-tile GEMM runs through
//      VDPBF16PS on AVX-512; when the caller asks for
//      swiglu_oai_mul fusion the activation is applied in-register
//      and written directly in place of the standard wide-GEMM +
//      in-place-compact sequence — no L2 scratch round-trip.  The
//      microkernel honours the caller's ldc so a single code path
//      covers both the wide layout (ldc = N, activated I cols in
//      the first half of each row) and the tight layout (ldc = N/2,
//      the fused-MoE internal-alloc arena).  Falls back to the
//      standard path when the shape / dtype / activation does not
//      match the microkernel's contract (non-bf16 dst, transposed
//      weights, alpha≠1 / beta≠0, N not a multiple of 32/64,
//      unsupported bias dtype).
//
// Single engagement point: flat_n_tile (ALGO 3, group_matmul_n_tile.cpp).
// Covers both plain group_matmul and fused-MoE Op1/Op2 because the
// fused-MoE entry routes through the parallel dispatcher, which picks
// flat_n_tile for its ALGO 3 workloads.  Per-tile activations:
//   - act=none : plain group GEMM; downstream callers may run
//     a separate silu / gelu pass or a moe_postop reduce on
//     the wide output.
//   - act=swiglu_oai_mul (with ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT=1):
//     in-register fusion.  Writes activated I cols at the caller's
//     ldc — stays byte-compatible with the existing wide layout
//     AND the tight fused-MoE layout under the same code path.
// The silu / gelu fused epilogues are not implemented in the
// microkernel yet and stay on the standard path (caller's separate
// activation pass handles those).
// ──────────────────────────────────────────────────────────────────────
inline bool get_grp_matmul_custom_kernel() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL");
  if (env == nullptr || env[0] == '\0') return false;
  return env[0] != '0';
}

// ──────────────────────────────────────────────────────────────────────
// Custom microkernel sub-knobs — all tuning / debug levers that only
// affect the custom BF16 kernel path (engaged via the master switch
// `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL` above).  Grouped here so every
// group_matmul env lives in one file; the readers cache their values
// (`static const` singletons) because, unlike the master switches,
// these sub-knobs are set once at process start for a deployment and
// never toggled mid-run.
// ──────────────────────────────────────────────────────────────────────

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR = { unset, "32", "64" }
//   Pack / microkernel NR override.  Auto (unset / empty) prefers 32
//   (cleanest register budget, largest MR=8 on NV=2).  NR=64 trades
//   MR cap down to 6 for double N-lanes per zmm; worth trying on
//   prompt-class shapes where MR saturation is less critical.  Any
//   other value is ignored and falls back to auto.
inline int get_grp_matmul_custom_kernel_nr() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR");
    if (e == nullptr || e[0] == '\0') return 0;
    int parsed = std::atoi(e);
    return (parsed == 32 || parsed == 64) ? parsed : 0;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT = { "0" (default), "1" }
//   Per-expert L2-friendly subtile_cols sizing (vs. a single m_max-
//   sized value used for every expert in a call).  Default OFF —
//   measured noise-floor on GPT-OSS decode.  Possibly useful on
//   workloads with extreme M variance across experts, or on CPUs
//   with larger L2 where the A-panel-dominated budget math leaves
//   more headroom for wider B strips on small-M experts.
inline bool get_grp_matmul_custom_kernel_subtile_per_expert() {
  static const bool v = []() {
    const char *e = std::getenv(
        "ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT");
    return (e != nullptr && e[0] != '\0' && e[0] != '0');
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE = { unset, positive int }
//   Override for the ALGO 3 outer N-tile minimum (`kDecodeNTile` in
//   flat_n_tile's planner).  Sets the per-thread minimum column
//   count; the actual per-expert tile is then `max(N/team_size,
//   n_tile_min)` rounded by the backend's NR alignment.  Accepted
//   values should be a multiple of 32 to align cleanly with
//   NR=32 packing: 128 (narrow, many subtile passes), 256
//   (current default — balanced), 512 (prompt-style — fewer wider
//   tiles).  Any other value (or unset) falls back to `kDecodeNTile`.
//
// Rationale: when `threads / num_ops` is high (small ops count on
// big thread pool), the 256-col default gives each thread too few
// subtile passes for good L2 reuse — a narrower tile (128) would
// increase thread participation.  When `threads / num_ops` is
// low, the default is fine or a wider tile amortises the kernel
// invocation overhead better.  Exposed as env first so deployment
// tuning can pick the right value before committing to an
// auto-picker.
inline int get_grp_matmul_custom_kernel_n_tile() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE");
    if (e == nullptr || e[0] == '\0') return 0;
    int parsed = std::atoi(e);
    // Accept positive multiples of 32 only; other values fall back to
    // the planner's default so users typo'ing don't silently wreck
    // the schedule.
    return (parsed > 0 && (parsed % 32) == 0) ? parsed : 0;
  }();
  return v;
}

// ──────────────────────────────────────────────────────────────────────
// Backend-aware N-tile alignment (ALGO 3).
//
// When ALGO 3 splits the N dimension across threads, each backend may
// prefer per-thread slices to start on a particular alignment.  This
// header surfaces a single integer per backend via backend_n_align();
// the per-thread column partitioner (aligned_n_split, below) honors
// it when feasible and falls back to an even split when alignment
// would push the slowest thread more than 2× heavier than the fastest.
//
// The values themselves are owned by the backends — group_matmul does
// not encode any kernel-internal register tile, blocking, or ukernel
// dispatch logic here.  A backend that does not benefit from
// alignment (or has been measured to regress under it) returns 1 and
// the partitioner uses the unaligned even split.
// ──────────────────────────────────────────────────────────────────────

/// Returns the N alignment the inner kernel prefers per-thread
/// slices to be a multiple of, or 1 when no alignment is needed.
inline int backend_n_align(matmul_algo_t algo) {
  switch (algo) {
    case matmul_algo_t::native_brgemm:
    case matmul_algo_t::native_gemm:
      return 64;
    default:
      return 1;
  }
}

/// Aligned column-slice partitioner used by ALGO 3.
///
/// Returns {col_start, col_end} for thread `tid` of `n_thr` over a
/// column range [0, N).  When alignment is feasible (= the last thread
/// still has at least half of `aligned_per_thr` cols) the partition is
///   - [aligned_per_thr * tid,  aligned_per_thr * (tid+1))   for tid < n_thr-1
///   - [aligned_per_thr * (n_thr-1), N)                       for the last tid
/// Otherwise it falls back to the legacy even split (N*tid/n_thr).
///
/// Imbalance bound: in the aligned branch the slowest thread does
/// `aligned_per_thr` cols and the last thread does `last` ≥
/// aligned_per_thr/2 cols — wall-clock skew is at most 2×, the BLIS-
/// style upper bound for accepting an alignment-driven partition.
///
/// Search policy: starts from `ceil(N/n_thr)` rounded UP to `align`,
/// then walks DOWN in `align`-quanta steps until the imbalance bound
/// is met.  This avoids rejecting feasible alignments just because
/// the ceil-rounded slice happens to be too large for the last
/// thread (e.g. N=2880, n_thr=11, align=64: ceil rounds 262 up to
/// 320 which leaves last=−320, but 256 cols/thread fits with last=320
/// and is the right answer).
inline std::pair<int, int> aligned_n_split(int N, int n_thr, int tid,
                                           int align) {
  // Hardened against pathological inputs: n_thr<=0 used to hit
  // even_split's divide-by-zero (`N * tid / n_thr`).
  if (n_thr <= 0 || N <= 0)
    return std::make_pair(0, 0);

  auto even_split = [&]() {
    const int s = static_cast<int>(static_cast<int64_t>(N) * tid / n_thr);
    const int e = static_cast<int>(
        static_cast<int64_t>(N) * (tid + 1) / n_thr);
    return std::make_pair(s, e);
  };

  if (align <= 1 || n_thr <= 1)
    return even_split();

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

/// L3 cache slice attached to each CCD on the Zen 3 / Zen 4 / Zen 5
/// classic-CCD topology that the planner targets (8 cores per CCD,
/// 32 MB of L3 per CCD).
inline constexpr size_t kL3PerCcdBytes = 32UL * 1024UL * 1024UL;

/// Aggregate L3 capacity available to the running thread team, used
/// by the ALGO 3 N-tile and ALGO 4 multilevel planners to bound the
/// number of experts batched into a single L3-resident "round".
///
/// Returns `num_ccds × kL3PerCcdBytes`.  Scales linearly with the CCD
/// count derived from `num_threads`, so the same constant works across
/// the full SKU range without per-machine tuning.  `num_ccds` is
/// computed by `summarise_topology()` in group_matmul_n_tile.cpp from
/// the actual `num_threads` the kernel was given; passing it in keeps
/// L3 sizing consistent with the thread-team partitioning the rest of
/// the planner sees.
///
/// History: a previous revision honored a `ZENDNNL_GRP_L3_TOTAL_MB`
/// env override.  Removed: every classic-CCD SKU we target (Zen 3 /
/// Zen 4 / Zen 5) ships 32 MB of L3 per CCD, so the override never
/// changed the planner's decisions in practice and the runtime
/// `getenv` cost on every group_matmul call was avoidable.  Container
/// CPU-pinned scenarios still derive `num_ccds` correctly from the
/// thread-team size, so the formula remains valid there.  If a future
/// SKU lands with a different L3-per-CCD ratio the constant
/// `kL3PerCcdBytes` is the single source of truth.
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
    if (a <= 0 || a >= static_cast<int32_t>(matmul_algo_t::algo_count))
      return matmul_algo_t::aocl_dlp_blocked;
    return static_cast<matmul_algo_t>(a);
  }();
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
  matmul_execute(layout, transA, transB,
      M, N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
      is_weights_const, size_of(params.dtypes.src),
      size_of(params.dtypes.dst),
      num_thr, kernel, params, bp, 0);
}

// ──────────────────────────────────────────────────────────────────────
// N-tile shared utilities
//
// These helpers live here because they were historically shared by
// two N-tile executors (the non-fused `flat_n_tile` and a specialised
// fused-MoE Op1 executor).  Post-unification only `flat_n_tile`
// (group_matmul_n_tile.cpp) consumes them — the fused-MoE entry now
// routes through the parallel dispatcher, which forwards to
// `flat_n_tile`.  Kept in the shared header so a future additional
// N-tile executor can pick them up without duplication.
// ──────────────────────────────────────────────────────────────────────

/// Sort `[indices, indices + n)` by `M[idx]` in ascending or descending
/// order.  Heap-free; operates on a caller-owned flat buffer (stack
/// array or preallocated vector).  No bounds checking — the caller
/// guarantees `indices` holds at least `n` entries and `M.size() >= n`.
inline void sort_indices_by_m(int *indices, int n,
                              const std::vector<int> &M, bool ascending) {
  for (int i = 0; i < n; ++i) indices[i] = i;
  std::sort(indices, indices + n, [&M, ascending](int a, int b) {
    return ascending ? (M[a] < M[b]) : (M[a] > M[b]);
  });
}

/// Env-gated BF16 custom-microkernel engagement for an N-tile executor.
///
/// Usage: each caller stack-allocates a fresh `custom_kernel::CallContext`
/// and hands it to this helper; on return `kctx.enabled` tells the
/// caller whether the custom path is live for this call.  Engagement
/// is OFF by default; `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1` opts in.
/// Even when opted in, the dispatcher's own contract check (bf16 dst,
/// no transB, alpha=1 / beta=0, N % pack_nr == 0) decides whether
/// the per-tile path can actually run.  The caller falls back to its
/// standard path when `kctx.enabled` is false.
///
/// `act` is the activation the custom kernel should fuse in-register:
///   * `none`           — plain GEMM tile.
///   * `swiglu_oai_mul` — fused gate+up → halved output in-register.
/// Other activations (silu / gelu) stay on the standard execute_expert_slice
/// path; callers pass `act=none` here and run a separate activation pass.
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
    const std::vector<float>         &alpha,
    const std::vector<float>         &beta,
    const std::vector<const void *>  &weight,
    custom_kernel::CallContext       &kctx) {
  if (!get_grp_matmul_custom_kernel()) return;
  custom_kernel::prepare_for_call(
      act, src_dtype, wei_dtype, dst_dtype, act_dtype, bias_dtype,
      transA, transB, M, N, K, alpha, beta, weight, kctx);
}

/// Effective N-column alignment for an N-tile executor's per-thread
/// split, combining three constraints:
///   * `backend_nr`    — backend-preferred alignment (from
///                       `backend_n_align(algo)`): 64 for
///                       native_brgemm / native_gemm, 1 for every
///                       other backend (AOCL DLP, LibXSMM, oneDNN).
///   * `kctx.pack_nr`  — when the custom kernel engaged, its pack/
///                       microkernel NR (32 or 64).  Disregarded when
///                       `kctx.enabled` is false.
///   * `pair_aligned`  — true when activation requires even column
///                       boundaries per thread (swiglu_oai_mul consumes
///                       gate+up pairs and must not split a pair
///                       across threads).
/// Returned value is max(backend_nr, optional pack_nr, pair floor).
inline int ntile_effective_nr_align(
    int backend_nr,
    const custom_kernel::CallContext &kctx,
    bool pair_aligned) {
  int a = backend_nr;
  if (kctx.enabled) a = std::max(a, kctx.pack_nr);
  if (pair_aligned) a = std::max(a, 2);
  return a;
}

// ──────────────────────────────────────────────────────────────────────
// Expert ordering for N-tile executors (V1 flat_n_tile + V2 fused MoE)
//
// Both N-tile executors walk a permutation of experts; the permutation
// is chosen by the `ZENDNNL_GRP_MATMUL_N_ORDER` env knob and computed
// here as a shared primitive.  V1 wraps this into its `GroupNTilePlan`
// struct; V2 calls it directly and applies a "default-to-descending"
// shim for the auto-resolved-to-walk-input case (V2's barrier-free
// round scheduler requires heaviest-M experts in round 0).
// ──────────────────────────────────────────────────────────────────────

/// Shared upper bound on num_ops for heap-free N-tile expert ordering.
/// Kept at 256 so both V1's `GroupNTilePlan::kMaxExperts` and V2's
/// local cap agree on a single scalar.
inline constexpr int kNTileMaxExperts = 256;

/// Auto-pick a concrete N_ORDER sub-mode from workload shape.
///
/// Rule derived from the GPT-OSS MoE decode 128-thread benchmark sweep
/// (115 unique shapes × 5 modes), summarised as win/loss counts for
/// pair_balanced (mode 3) vs walk-input (no sort):
///
///    num_ops range   |  mode 3 wins >2%  |  mode 3 losses >2%
///    ──────────────────────────────────────────────────────────
///    num_ops ≤ 18    |       ~58 %       |       ~12 %       ← strong win
///    19 ≤ ops ≤ 25   |       ~26 %       |       ~35 %       ← regression band
///    num_ops ≥ 26    |       ~42 %       |        ~0 %       ← strong win, zero losses
///
/// So auto picks pair_balanced (mode 3) on `num_ops ≤ 18 ∨ num_ops ≥ 26`
/// and falls back to walk-input (return 0) for the 19..25 mid-band
/// where pair_balanced empirically costs more than it delivers.
///
/// Sentinel: returns 3 (pair_balanced) or 0 ("caller walks input order").
/// Explicit modes 1 / 2 / 4 are only reachable via env override.
inline int auto_pick_n_order(int num_ops) {
  if (num_ops <= 18) return 3;   // sweet spot: few active experts
  if (num_ops >= 26) return 3;   // strong win + zero losses in data
  return 0;                      // mid-band: walk input order
}

/// Populate `out[0..out_size)` with expert indices ordered by the
/// policy returned by `get_grp_matmul_n_order()`:
///
///   0  auto             — DEFAULT.  Shape-aware via auto_pick_n_order():
///                         returns pair_balanced (mode 3) for the
///                         data-backed sweet spot, otherwise writes
///                         out_size = 0 meaning "walk input order".
///   1  ascending        — sort by M increasing
///   2  descending       — sort by M decreasing
///   3  pair-balanced    — descending then interleave largest with
///                         smallest (anti-correlated pairing)
///   4  balanced-spread  — prefix-sum-balanced: any K-way consecutive
///                         split yields Σ M per chunk ≈ total / K
///
/// Generic (no plan-struct dependency); both V1 (wraps this via
/// `fill_sorted_expert_order(plan, ...)`) and V2 (direct consumer
/// with an auto→desc fallback shim) call this helper.
///
/// Parameters:
///   out               — caller-owned buffer of at least `max_size` ints.
///   out_size          — written: number of valid slots in `out`
///                       (0 signals "walk input order, ignore out").
///   max_size          — capacity bound on `out`.
///   M                 — expert token-counts vector.
///   num_ops           — number of experts (must ≤ max_size).
///   auto_resolved_out — optional out-param: when env mode is 0 (auto),
///                       the resolved concrete sub-mode is written
///                       here for diagnostics / APILOG.  Pass nullptr
///                       if the caller doesn't need it.
///
/// Heap-free: uses a stack-allocated `std::array<int, kNTileMaxExperts>`
/// for the descending-sort temp buffer.  Beyond `kNTileMaxExperts`
/// the ordering is skipped (perf-only; falling through to input
/// order is correct).  O(num_ops²) in mode 4 (balanced-spread),
/// bounded ≤ 64K comparisons — well under 10 µs.
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
    if (auto_resolved_out != nullptr) *auto_resolved_out = order;
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
      if (lo <= hi) out[o++] = sorted_desc[hi--];
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
    for (int i = 0; i < num_ops; ++i) total += M[i];

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
        if (used[j]) continue;
        const int64_t new_cum = cum + M[sorted_desc[j]];
        const int64_t err = std::llabs(
            target_scaled - static_cast<int64_t>(num_ops) * new_cum);
        if (err < best_err) { best_err = err; best_j = j; }
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
// Tile-strategy entry points (defined in their own .cpp files).
//
// These functions have external (library-internal) linkage so that
// group_matmul_parallel.cpp (dispatcher) can call them.  They are NOT
// part of any public header; they remain library-private.
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

/// Peek at the ALGO `group_matmul_run_parallel_dispatch` would pick
/// for this (layout, shapes, params, num_threads) combination.
/// Returns an integer in {1..5}:
///   1 = sequential_experts, 2 = flat_m_tile, 3 = flat_n_tile,
///   4 = parallel_multilevel, 5 = parallel_per_expert.
///
/// Mirrors the exact gating the dispatcher uses internally:
///   * reads `ZENDNNL_GRP_MATMUL_ALGO` (forced override, 1..5);
///   * runs `check_m_tile_safe` / `check_n_tile_extra` and falls back
///     to ALGO 1 when the forced override is unsafe for the shapes;
///   * calls the `auto_select_algo` heuristic when env=0 (auto).
///
/// Intended for pre-dispatch decisions that depend on knowing which
/// executor will run — currently consumed by the fused-MoE entry to
/// choose between tight (ALGO-3-only) and wide arena allocation
/// before it commits the buffer layout.
///
/// Pure observer: no side-effects on `params`, no OMP regions, no
/// env writes.  O(num_ops) cost, safe to call repeatedly.
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
