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

/// Fused MoE: Op1(gate+up) → activation → Op2(down_proj) → optional
/// weighted-reduce post-op, all in one API call.
///
/// Unified dispatch architecture:
///   * Op1 + activation and Op2 both route through
///     `group_matmul_run_parallel_dispatch`, which selects the ALGO
///     via `ZENDNNL_GRP_MATMUL_ALGO` (1..5 manual, 0 auto) and the
///     inner BLAS kernel via `ZENDNNL_MATMUL_ALGO`.  All 5 strategies
///     remain selectable for both passes.
///   * The activation is fused inline by the dispatcher when the
///     chosen strategy supports it (ALGO 1/2/4/5 always; ALGO 3 for
///     swiglu_oai_mul either when `ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT=1`
///     on wide layout, or auto-enabled on tight layout regardless of
///     the env flag).  Otherwise a separate-pass activation runs
///     in-place on the wide arena.
///
/// Adaptive arena layout (internal-alloc mode only):
///   * `pick_fused_moe_want_tight()` decides per call whether to
///     allocate a tight [M, I] arena or the classic wide [M, 2I].
///   * Tight layout is requested when ALL of the following hold:
///       - act == swiglu_oai_mul (activation halves N),
///       - dispatcher would route Op1 to ALGO 3 (only flat_n_tile
///         implements tight handling — ALGO 1/2/4/5 would overrun),
///       - `ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT` is unset (auto)
///         or forces tight (any non-zero).  Forcing 0 selects wide.
///   * Important: for the tight path, fused activation on ALGO 3
///     is AUTO-ENABLED by `group_matmul_run_parallel_dispatch` when
///     it detects `ldc < N`, independent of the
///     `ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT` env flag.  Tight layout
///     is a correctness contract for the caller's buffer (a
///     separate-pass swiglu would overrun the [M, I] arena), so it
///     must fuse regardless of the perf-tuning env.  This is why
///     the auto-fuse is not a gate in `pick_fused_moe_want_tight()`.
///     See `group_matmul_parallel.cpp::a3_fuses` for the corresponding
///     dispatcher logic.
///   * When tight is selected the Op1 arena holds `sum_i M[i]·I[i]·
///     dst_elem` bytes (half of the wide case) and `op1_ldc[i] = I[i]`.
///     Op2 then reads the activated output at tight stride — halves
///     Op2's src DRAM traffic vs the wide layout.
///   * When tight is NOT selected the legacy wide [M, 2I] arena is
///     used unchanged.  Non-internal-alloc callers always run wide.
///
/// Op2 output mode (see grp_matmul_fused_moe_params doc-block in the
/// public header for full semantics):
///   * Legacy / caller-allocated : caller fills both `dst[]` (Op1 dst,
///     entry API) and `fused.dst_down[]` (Op2 dst); the library
///     writes into them.  Non-internal-alloc callers always run wide.
///   * Internal-alloc + src-reuse : caller leaves BOTH `dst[]` (all
///     nullptr / empty) AND `fused.dst_down` empty.  The library
///     allocates Op1 scratch in a persistent thread-local arena
///     (sized to the high-water mark, no per-call allocator traffic
///     on the steady state) and runs Op2 reading from the scratch
///     and writing back into the caller's `src[]` buffer (in-place
///     reuse).  Caller reads Op2 output from `src[]` after the call.
///   * Mixed (one filled, one empty) is rejected by the validator.

#include <cstdlib>
#include <string>
#include <vector>

#include "detect_internal_alloc.hpp"
#include "group_matmul_direct.hpp"
#include "group_matmul_parallel_common.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::op_instrumentation;
using zendnnl::common::size_of;

namespace {

// `op2_k_for_act(n_op1, act)` lives in `group_matmul_parallel_common.hpp`
// (shared with the dispatcher's Phase-F validator so both apply the
// same `ldb_down` minimum).  See the helper's doc-block there for
// the full contract.

// Per-thread persistent Op1 arena used by fused-MoE internal-alloc.
// Owns a single 64-byte-aligned slab whose capacity monotonically
// grows to the high-water mark this thread has seen.  Per-expert
// pointers are tightly packed byte-offsets into the slab (only the
// base is 64B-aligned; per-expert first rows fall wherever the
// previous expert's footprint ended).  Freed by the destructor on
// thread exit; freed + reallocated when a call needs more than the
// current capacity.
//
// Hoisted out of the dispatch function body so the persistent-arena
// pattern is visible at file scope and can be reviewed / replaced
// without reading the 500-line dispatch body.
struct FusedMoEArena {
  void *buf = nullptr;
  size_t cap = 0;
  ~FusedMoEArena() { std::free(buf); }
};

// Per-thread persistent Op2 setup scratch.  Holds the working arrays
// that Op2 dispatch needs (`K_down`, `alpha_down`, …), the Op1 / Op2
// dst pointer arrays (internal-alloc mode), and the per-expert Op1
// ldc vector (populated only when the tight layout is requested;
// wide mode uses `N` directly).  Vector capacity persists across
// calls — `resize()` only shrinks the logical size, never the
// underlying allocation — so after the first call all per-call
// traffic is O(num_ops) field writes, no allocator traffic on the
// steady state.
struct FusedMoEScratch {
  std::vector<int> K_down;
  std::vector<float> alpha_down;
  std::vector<float> beta_down;
  std::vector<bool> transA_down;
  std::vector<const void *> src_down;
  std::vector<matmul_params> params_down;
  std::vector<void *> op1_dst_internal;   // populated in internal-alloc mode
  std::vector<void *> op2_dst_internal;   // populated in internal-alloc mode
  std::vector<int> op1_ldc_local;         // populated only when `want_tight`
                                          // (= N[i] / 2 per expert)
};

// ──────────────────────────────────────────────────────────────────────
// Adaptive arena-layout picker.
//
// Decides per call whether to request the tight [M, I] arena (half the
// wide [M, 2I] footprint) for Op1's internal-alloc output.  Returns
// `true` to request tight, `false` for wide.  All gates are O(num_ops)
// or cheaper; the function has no side effects.
//
// Correctness gates (all must pass for tight to be considered):
//
//   1. `op1_internal` — only when the library owns the Op1 arena.
//      Caller-allocated Op1 paths supply their own dst[] with
//      caller-chosen ldc; the tight arena layout is library-private,
//      so it's never visible to caller-allocated paths.
//
//   2. `act == swiglu_oai_mul` — tight's compaction is the swiglu
//      gate·up fold; silu/gelu produce a half-layout activation that
//      flat_n_tile does not support as an OOP writer.
//
//   3. `env_algo ∈ {0, 3}` — tight requires Op1 to run in flat_n_tile
//      (only strategy that writes per-thread scratch + OOP swiglu into
//      the tight arena).  A caller forcing ALGO 1/2/4/5 explicitly
//      asked for a non-N-tile strategy; silently flipping them
//      violates intent.
//
//      `get_grp_n_tile_fused_act()` is NOT a gate here — when the
//      fused-MoE picker hands the dispatcher a tight destination
//      (ldc < N), the dispatcher auto-enables ALGO 3 fused activation
//      regardless of the env flag (tight layout is a correctness
//      constraint, not a perf toggle).  See
//      `group_matmul_run_parallel_dispatch` in group_matmul_parallel.cpp.
//
//   4. Env override: unset (auto) and force-tight request tight,
//      force-wide (=0) rejects.
//
//   5. `select_grp_matmul_algo()` actually would return 3 for this
//      (shapes, params, num_threads).  Without this, auto-select
//      could pick ALGO 1 (small N, non-N-tile-viable) on the tight
//      arena and overrun it.  This is the shape-adaptive half of the
//      picker — tight is only safe when the planner agrees.
//
// Performance policy once all gates pass:
//   * env = 1 (force-tight) or unset (auto): return true.
//   * env = 0 (force-wide): already rejected in gate (4).
//
// Today the auto policy is "tight whenever safe": tight halves Op2's
// Op1-src DRAM traffic and flat_n_tile's full planner (DecodeD, pair-
// balanced rounds, N_ORDER, multi-round batching) adapts to any
// num_ops so there is no scheduler deficit vs the wide path.  If a
// future shape regresses, plug a `num_ops`-keyed threshold here.
inline bool pick_fused_moe_want_tight(
    bool op1_internal,
    grp_matmul_gated_act_t act,
    int env_algo,
    const std::vector<char> &layout,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &K,
    const std::vector<matmul_params> &params,
    int num_threads) {
  if (!op1_internal) return false;
  if (act != grp_matmul_gated_act_t::swiglu_oai_mul) return false;
  if (env_algo != 0 && env_algo != 3) return false;
  const int env_tight = get_grp_matmul_fused_moe_tight();
  if (env_tight == 0) return false;
  // env_tight == 1 (default / forced).  The getter was historically
  // tri-state with -1 meaning "auto", but auto collapsed to the same
  // behaviour as 1 (the dispatcher re-validates every gate below)
  // and was merged into the default.  Confirm the dispatcher would
  // actually route Op1 to flat_n_tile for this shape before
  // committing to the tight layout.
  const int algo_would =
      select_grp_matmul_algo(layout, M, N, K, params, num_threads);
  return algo_would == 3;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
// Primary entry: Op1+Act → Op2 (→ optional weighted reduce post-op)
// ═══════════════════════════════════════════════════════════════════════

status_t group_matmul_fused_moe_execute(
    const grp_matmul_fused_moe_params &fused,
    grp_matmul_gated_act_t act, data_type_t act_dtype,
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    const     std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads,
    const char **gemm_mode_out,
    const group_matmul_moe_postop_params *moe_postop) {

  const size_t num_ops = M.size();

  // ── Always-on safety guards ─────────────────────────────────────────
  // This function is declared in an internal header and has exactly
  // one known caller (`group_matmul_direct`), but `group_matmul_direct`
  // only size-checks the non-fused vectors always-on; the fused-MoE
  // specific vectors (down_weight / N_down / ldb_down / bias_down)
  // plus the per-expert stride invariants are nowhere else enforced
  // outside the diagnostic path.  So this block runs always-on as
  // the primary defense against OOB pointer arithmetic in the
  // dispatch body below.  The three classes it covers:
  //
  //   (1) primary-vector emptiness — params[0] / N[0] / etc. are
  //       dereferenced unconditionally below.
  //   (2) per-vector size consistency — every required vector must
  //       be sized to num_ops (with the dst / ldc / dst_down /
  //       ldc_down exception for the internal-alloc + src-reuse
  //       mode, where the caller leaves those empty and the library
  //       owns Op1 dst and reuses src as Op2 output).
  //   (3) per-expert dimension, leading-stride, and required-pointer
  //       sanity for active experts — non-negative M, positive N/K,
  //       even N (required by the swiglu half-split), lda/ldb/ldc/
  //       ldb_down/ldc_down all large enough for the row-major
  //       access pattern, and non-null src/weight/down_weight
  //       pointers (plus dst/dst_down in legacy caller-allocated
  //       mode).  These prevent OOB pointer arithmetic or null
  //       dereference later.  Per-expert pointer null checks are
  //       always-on here because no other layer validates
  //       `fused.down_weight[i]` outside the diagnostic gate.
  //   (4) internal-alloc dtype safety — cross-expert dst dtype
  //       uniformity and per-expert matched-precision (src == dst).
  //       Both feed sizing math for the single Op1 arena slab; a
  //       divergent expert would overrun its slice or corrupt the
  //       caller's src buffer on the Op2 in-place write.
  //
  //   (5) cross-expert N_down uniformity (only when moe_postop is
  //       engaged) — the weighted-reduce stage below uses
  //       `fused.N_down[0]` as the common D for every expert; a
  //       divergent expert would cause OOB reads past its row.
  //       Correctness-critical, therefore always-on.
  //
  // Silent-wrong-result scalars (bias dtype declaration, activation
  // dtype bucket, mixed-state dst[] iteration when legacy mode is
  // engaged) move under ZENDNNL_DIAGNOSTICS_ENABLE below because
  // they are either already covered by group_matmul_direct's
  // phase-D/F validator or produce wrong numbers without corrupting
  // memory.
  //
  if (num_ops == 0) return status_t::failure;

  // ── Op1 / Op2 buffer-source detection — INDEPENDENT ──────────────
  //
  // Each side (Op1's `dst[]` for the gate+up output, Op2's
  // `fused.dst_down[]` for the down_proj output) is detected on its
  // own so callers can mix any of the four combinations:
  //
  //   dst[] supplied  | dst_down[] supplied | Op1 destination     | Op2 destination
  //   ─────────────── | ─────────────────── | ─────────────────── | ───────────────────
  //   no              | no                  | library Op1 arena   | in-place src reuse
  //   no              | yes                 | library Op1 arena   | caller's dst_down
  //   yes             | no                  | caller's dst        | in-place src reuse
  //   yes             | yes                 | caller's dst        | caller's dst_down
  //
  // "Supplied" means the vector is non-empty AND every entry in the
  // active range `[0, num_ops)` is non-null.  An empty vector or an
  // all-null active range means library-managed (the per-side
  // internal flag is true); any mixed null/non-null active range is
  // a hard caller-contract break (some experts would silently route
  // to internal-alloc, others to caller-allocated) and is rejected
  // up front by a full O(num_ops) sweep.  Sweeping only the active
  // range lets a prepack-extras tail (where non-fired slots are
  // typically nullptr placeholders) coexist with caller-allocated
  // active slots — see commit 9bb7776a for the active/total contract
  // these helpers assume.
  //
  // `pick_fused_moe_want_tight()` uses only `op1_internal` (Op1
  // arena layout depends on whether the library owns it; Op2 buffer
  // source is orthogonal).
  using group_matmul_internal::detect_internal_alloc;
  using group_matmul_internal::internal_alloc_mode;
  auto run_detect = [&](const std::vector<void *> &v,
                        const char *name,
                        bool *out_internal) -> status_t {
    const status_t st = detect_internal_alloc(
        v, num_ops, /*fused_moe_present=*/true,
        internal_alloc_mode::sweep_active, out_internal);
    if (st != status_t::success) {
      log_error("group_matmul_fused_moe: ", name, " has a mixed "
                "null/non-null state — every active entry must be "
                "either null (library-managed) or non-null "
                "(caller-allocated).");
    }
    return st;
  };
  bool op1_internal = false;
  bool op2_internal = false;
  if (run_detect(dst, "dst", &op1_internal)
      != status_t::success) return status_t::failure;
  if (run_detect(fused.dst_down, "fused.dst_down", &op2_internal)
      != status_t::success) return status_t::failure;
  // Vector sizes — must hold AT LEAST `num_ops` entries each (the
  // active matmul-processing count).  Anything past `num_ops` is the
  // framework's prepack-extras tail and is never read by the
  // dispatch loops below.  Strict equality (`!=`) was the original
  // contract; relaxing to `<` accepts the new active+extras layout
  // without affecting legacy callers who pass exactly-sized vectors.
  // Per-call mandatory vectors (always required regardless of which
  // side is internal-alloc).  Down-side weight metadata is required
  // here even when op2_internal — the dispatcher still needs the
  // weight pointers, dimensions, and stride to drive Op2 itself.
  if (layout.size() < num_ops || transA.size() < num_ops
      || transB.size() < num_ops || N.size() < num_ops
      || K.size() < num_ops || src.size() < num_ops
      || weight.size() < num_ops || lda.size() < num_ops
      || ldb.size() < num_ops || params.size() < num_ops
      || alpha.size() < num_ops || beta.size() < num_ops
      || bias.size() < num_ops || is_weights_const.size() < num_ops
      || fused.down_weight.size() < num_ops
      || fused.N_down.size() < num_ops
      || fused.ldb_down.size() < num_ops
      || fused.bias_down.size() < num_ops)
    return status_t::failure;
  // Op1 dst/ldc — when caller-allocated must reach `num_ops`; when
  // library-managed (op1_internal) the vectors may be empty (library
  // owns) or sized to at least `num_ops` (caller passed all-null
  // placeholders, possibly with prepack-extras tail).  Reject the
  // in-between case (non-empty AND undersized) as a malformed
  // caller intent that would walk past the vector end below.
  if (op1_internal) {
    if (!dst.empty() && dst.size() < num_ops) return status_t::failure;
    if (!ldc.empty() && ldc.size() < num_ops) return status_t::failure;
  } else {
    if (dst.size() < num_ops || ldc.size() < num_ops)
      return status_t::failure;
  }
  // Op2 dst_down/ldc_down — same shape as Op1 above, gated on
  // op2_internal.
  if (op2_internal) {
    if (!fused.dst_down.empty() && fused.dst_down.size() < num_ops)
      return status_t::failure;
    if (!fused.ldc_down.empty() && fused.ldc_down.size() < num_ops)
      return status_t::failure;
  } else {
    if (fused.dst_down.size() < num_ops || fused.ldc_down.size() < num_ops)
      return status_t::failure;
  }

  // Hoist the Op1 dst element-size used in two places below (arena
  // sizing and per-expert offset computation).  Computed once after
  // size-consistency guards above guarantee params[0] is valid.
  // Only Op1's destination depends on this — Op2 writes either to
  // caller-supplied dst_down or to src (in-place reuse), neither of
  // which goes through the library arena.
  const size_t dst_elem_internal =
      op1_internal ? size_of(params[0].dtypes.dst) : 0;

  // Cache once at entry — the same env read is consulted again inside
  // the custom-kernel-enable APILOG below.  A single `getenv` per call
  // instead of one-per-readsite.
  const int env_algo_fused = get_grp_matmul_algo();
  const bool custom_kernel_en = get_grp_matmul_custom_kernel();

  // ── Adaptive layout picker (wide vs tight Op1 arena) ────────────────
  // The picker resolves the full wide-vs-tight decision for Op1,
  // including env-force overrides and N-tile viability.  See
  // `pick_fused_moe_want_tight` above for the full gate list.  The
  // decision is committed here and used for:
  //   * Op1 arena size         — tight halves the per-expert byte budget.
  //   * `op1_ldc[e]`           — tight sets = N[e]/2; wide keeps = N[e].
  //   * Op2 src stride (= Op1 ldc) — flows through naturally.
  // When tight is selected the dispatcher routes Op1 to flat_n_tile,
  // which detects the tight caller layout from `ldc < N` and takes
  // its per-thread-scratch + OOP swiglu branch (commit 1 of this
  // series).
  // Tight Op1 layout is purely an Op1-arena property — only meaningful
  // when the library owns the Op1 destination (op1_internal).
  const bool want_tight = pick_fused_moe_want_tight(
      op1_internal, act, env_algo_fused,
      layout, M, N, K, params, num_threads);

  // ── APILOG: one line per fused_moe call summarising path decisions.
  // Caller sees arena layout, per-side internal-alloc state, act
  // kind, custom-kernel flag, and the env ALGO — full dispatch trail
  // when `ZENDNNL_API_LOG_LEVEL=3`.  apilog_info_enabled() is cached
  // after the first call so the gate check is free when logging is
  // off.
  static const bool s_apilog = apilog_info_enabled();
  if (s_apilog) {
    apilog_info("[GRP_MATMUL Level2 fused_moe] arena=",
                (want_tight ? "tight" : "wide"),
                " op1_internal=", op1_internal,
                " op2_internal=", op2_internal,
                " act=", act_name(act),
                " env_algo=", env_algo_fused,
                " env_tight=", get_grp_matmul_fused_moe_tight(),
                " num_ops=", (int)num_ops,
                " custom_kernel_env=", custom_kernel_en);
  }

  // ── Always-on: per-expert dim / stride / pointer / N-evenness sweep
  // One pass serves three purposes:
  //   (1) primary OOB-prevention for the dispatch body — any stride
  //       or dim violation returns `failure` before we build pointer
  //       offsets from it.
  //   (2) active-expert null-pointer checks.
  //   (3) accumulates `total_M` and (in internal-alloc mode) the Op1
  //       slab byte budget so the all-zero short-circuit and arena
  //       sizing below reuse those values without a second sweep.
  // Silent-wrong-result scalars (bias dtype, act dtype) and the
  // cross-expert uniformity re-checks move under the diagnostic gate
  // below.
  int64_t total_M = 0;
  size_t total_bytes_internal = 0;
  for (size_t i = 0; i < num_ops; ++i) {
    // Core dimensions.
    if (M[i] < 0 || N[i] <= 0 || K[i] <= 0) return status_t::failure;
    // N must be even ONLY when a gated activation is fused (swiglu /
    // silu / gelu_and_mul collapse pairs of cols).  For act=none Op1
    // output flows into Op2 verbatim, so any N is admissible.
    if (act != grp_matmul_gated_act_t::none && (N[i] & 1) != 0)
      return status_t::failure;
    if (fused.N_down[i] <= 0) return status_t::failure;

    // Op1 leading strides (row-major).  Each row m starts at
    // src + m*lda, weight col c starts at weight + (transB ? c*ldb : c),
    // dst row m starts at dst + m*ldc.  These bounds prevent the
    // GEMM kernel from indexing past the row.
    if (lda[i] < K[i]) return status_t::failure;
    if (ldb[i] < (transB[i] ? K[i] : N[i])) return status_t::failure;
    // Op2 weight ldb — `K_down` (the row count of the down_weight
    // matrix Op2 reads) is N/2 for gated activations, N for act=none.
    // See `op2_k_for_act` above for the contract.
    const int K_down = op2_k_for_act(N[i], act);
    if (fused.ldb_down[i] < (transB[i] ? K_down : fused.N_down[i]))
      return status_t::failure;

    // Op1 output-side check: when caller-allocated, ldc must
    // accommodate the wide write per row.  When library-managed
    // (op1_internal), ldc is unused — the arena writes at row width
    // = N (wide) or N/2 (tight).
    if (!op1_internal) {
      if (ldc[i] < N[i]) return status_t::failure;
    }
    // Op2 output-side check: when caller-allocated, ldc_down must
    // accommodate N_down per row.  When library-managed
    // (op2_internal), Op2 writes BACK into src with stride lda, so
    // lda must accommodate N_down instead.
    if (!op2_internal) {
      if (fused.ldc_down[i] < fused.N_down[i]) return status_t::failure;
    } else if (M[i] > 0) {
      if (lda[i] < fused.N_down[i]) return status_t::failure;
    }

    // Cross-expert dst-dtype uniformity / matched-precision are
    // both always-on when op1_internal (Op1 arena is sized once
    // using `dst_elem_internal` = sizeof(params[0].dtypes.dst); any
    // larger expert dtype would overrun its per-expert slice) AND
    // when op2_internal (Op2 writes BACK into src using the dst
    // element size; mismatched src/dst element sizes overrun the
    // src row pitch).  When BOTH are external-alloc, the caller
    // owns both buffer footprints and we trust them to size things
    // correctly per expert.
    if ((op1_internal || op2_internal) && M[i] > 0) {
      if (params[i].dtypes.dst != params[0].dtypes.dst) return status_t::failure;
      if (params[i].dtypes.src != params[i].dtypes.dst) return status_t::failure;
    }

    if (M[i] > 0) {
      if (src[i] == nullptr || weight[i] == nullptr) return status_t::failure;
      if (fused.down_weight[i] == nullptr) return status_t::failure;
      if (!op1_internal && dst[i] == nullptr) return status_t::failure;
      if (!op2_internal && fused.dst_down[i] == nullptr)
        return status_t::failure;
      total_M += M[i];
      if (op1_internal)
        total_bytes_internal +=
            static_cast<size_t>(M[i]) * N[i] * dst_elem_internal;
    }
  }

  // Cross-expert N_down uniformity (when moe_postop is engaged) —
  // promoted to always-on because the weighted-reduce stage
  // downstream uses `fused.N_down[0]` as the common D for every
  // expert.  If expert i has `N_down[i] < N_down[0]`, the moe_postop
  // reader would stride past the end of expert i's rows → OOB read,
  // memory corruption, or garbage in the output.  This is a
  // correctness-critical invariant, not a silent-wrong-result path,
  // so it stays outside the diagnostic gate.  `group_matmul_direct`
  // has its own always-on copy of this check in phase G; the
  // duplicate here defends the code path for any future caller that
  // invokes `group_matmul_fused_moe_execute()` without going through
  // `group_matmul_direct`'s validator.
  if (moe_postop != nullptr) {
    for (size_t i = 1; i < num_ops; ++i)
      if (fused.N_down[i] != fused.N_down[0]) return status_t::failure;
  }

  // ── Diagnostic-only full input validation ──────────────────────────
  // These checks either (a) produce silent wrong results rather than
  // OOB, or (b) are already covered always-on by `group_matmul_direct`
  // phase-A guards.  Keeping them under `op_instrumentation::validate`
  // keeps production dispatch at O(num_ops) × {few loads, branches}
  // for the main sweep above and pays one predicted-not-taken branch
  // for the rest.
  status_t val = op_instrumentation::validate([&]() {
    // Mixed-state diagnostic catch — the up-front detection above
    // already keys on the active range; this is a defence-in-depth
    // check that confirms the same per-side invariants when
    // diagnostics is enabled.
    //
    // (Cross-expert dst dtype uniformity and matched-precision are
    // enforced always-on in the primary sweep above — they bound
    // the Op1 arena slab sizing and the Op2 in-place write
    // footprint, both of which are corruption vectors rather than
    // silent-wrong-result paths.)
    if (op1_internal) {
      const size_t dst_sweep = std::min<size_t>(num_ops, dst.size());
      for (size_t i = 0; i < dst_sweep; ++i)
        if (dst[i] != nullptr) return status_t::failure;
    }
    if (op2_internal) {
      const size_t dst_down_sweep =
          std::min<size_t>(num_ops, fused.dst_down.size());
      for (size_t i = 0; i < dst_down_sweep; ++i)
        if (fused.dst_down[i] != nullptr) return status_t::failure;
    }
    // Bias + act dtype contracts: silent-wrong-result paths.  A bias
    // buffer without a declared dtype would be misread as FP32 by the
    // post-op kernel; a gated activation with a non-{f32, bf16} act
    // dtype would have its layout misinterpreted.
    bool any_bias_down_d = false;
    for (size_t i = 0; i < num_ops; ++i)
      if (fused.bias_down[i] != nullptr) { any_bias_down_d = true; break; }
    if (any_bias_down_d && fused.bias_dt_down == data_type_t::none)
      return status_t::failure;
    if (act != grp_matmul_gated_act_t::none
        && act_dtype != data_type_t::f32
        && act_dtype != data_type_t::bf16)
      return status_t::failure;
    // (Cross-expert N_down uniformity for moe_postop is enforced
    // always-on in the primary sweep above — it bounds the
    // weighted-reduce stage's OOB-read risk, not a silent-wrong-
    // result path.)
    return status_t::success;
  });
  if (val != status_t::success) return val;

  // No active work (every expert has M=0): return success without
  // spawning OMP regions or touching the Op2 dispatch.
  if (total_M == 0) {
    if (gemm_mode_out) *gemm_mode_out = "fused_moe_skip";
    return status_t::success;
  }

  // ── Arena sizing per layout ────────────────────────────────────────
  //   wide  bytes = sum_i(M[i] · N[i]   · dst_elem)     (full [M, 2I])
  //   tight bytes = sum_i(M[i] · N[i]/2 · dst_elem)     (post-act I only)
  //
  // total_bytes_internal was accumulated as the wide footprint above;
  // halve it for the tight layout since flat_n_tile writes the
  // activated I-wide output directly (no wide intermediate in the
  // global arena).  N evenness is enforced by always-on validation so
  // the divide is exact.
  size_t arena_bytes = total_bytes_internal;
  if (want_tight) arena_bytes /= 2;

  // Persistent thread-local Op1 arena (internal-alloc mode) — struct
  // is hoisted to file-scope anonymous namespace above.  Monotonically
  // grows to the high-water mark this thread has seen; freed by its
  // destructor on thread exit.  Worker-pool footprint:
  //   num_threads × largest_seen_arena_bytes
  // — bounded by the largest fused MoE call the framework ever issues.
  static thread_local FusedMoEArena arena;

  if (op1_internal && arena_bytes > arena.cap) {
    std::free(arena.buf);
    arena.buf = nullptr;
    arena.cap = 0;
    void *tmp = nullptr;
    if (posix_memalign(&tmp, 64, arena_bytes) != 0 || tmp == nullptr)
      return status_t::failure;
    arena.buf = tmp;
    arena.cap = arena_bytes;
  }

  // Persistent thread-local Op2 setup scratch — struct hoisted above.
  static thread_local FusedMoEScratch scratch;

  // Populate the Op1 per-expert pointer / stride scratch in ONE sweep
  // over num_ops — the internal-alloc `op1_dst_internal` pointers and
  // (in tight mode) the `op1_ldc_local = N/2` vector are written inside
  // the same loop iteration, so each slot's cache lines are touched
  // exactly once.  Per-expert row width depends on the layout:
  //   * wide  : N[i]   cols/row (wide raw GEMM output, swiglu writes
  //             activated I cols into the first half in place).
  //   * tight : N[i]/2 cols/row (already-activated I-wide output,
  //             flat_n_tile's per-thread-scratch + OOP swiglu path).
  // Inactive (M <= 0) slots get an explicit nullptr — avoids the
  // pre-zero of every slot that `assign(num_ops, nullptr)` would do.
  if (op1_internal) scratch.op1_dst_internal.resize(num_ops);
  if (want_tight)   scratch.op1_ldc_local.resize(num_ops);
  if (op1_internal || want_tight) {
    char *base = op1_internal
        ? static_cast<char *>(arena.buf)   // may be nullptr if arena_bytes == 0
        : nullptr;
    size_t cursor = 0;
    for (size_t i = 0; i < num_ops; ++i) {
      const int row_cols = want_tight ? (N[i] / 2) : N[i];
      if (want_tight) scratch.op1_ldc_local[i] = row_cols;
      if (op1_internal) {
        if (M[i] <= 0 || base == nullptr) {
          scratch.op1_dst_internal[i] = nullptr;
        } else {
          scratch.op1_dst_internal[i] = base + cursor;
          cursor += static_cast<size_t>(M[i]) * row_cols * dst_elem_internal;
        }
      }
    }
  }

  // Op1 dst / ldc for Pass 1 dispatch:
  //   op1_internal + tight  : library arena, op1_ldc[i] = N[i]/2.
  //   op1_internal + wide   : library arena, op1_ldc[i] = N[i].
  //   caller-allocated      : caller's dst / ldc (wide by contract —
  //                           N >= ldc enforced by the always-on
  //                           validator above; tight ldc would have
  //                           been refused by `pick_fused_moe_want_tight`
  //                           which gates tight on op1_internal).
  const std::vector<void *> &op1_dst =
      op1_internal ? scratch.op1_dst_internal : dst;
  const std::vector<int> &op1_ldc =
      want_tight ? scratch.op1_ldc_local
                 : (op1_internal ? N : ldc);

  // ── Pass 1: Op1 (gate+up) + activation ────────────────────────────
  // Single route for all layouts:
  //   * `group_matmul_run_parallel_dispatch` picks ALGO 1..5 (or auto)
  //     per `ZENDNNL_GRP_MATMUL_ALGO` and the safety gates; inner BLAS
  //     kernel honours `ZENDNNL_MATMUL_ALGO`.
  //   * Wide layout: any of ALGO 1..5 runs; activation is fused inline
  //     for ALGO 1/2/4/5 always, and for ALGO 3 when
  //     `ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT=1` (otherwise a separate
  //     pass runs below).  Env flag is a pure perf knob here — the
  //     wide arena has room for either fused or separate-pass output.
  //   * Tight layout: picker ensures the dispatcher will route to
  //     ALGO 3 (flat_n_tile).  The dispatcher auto-enables fused
  //     activation for the tight path regardless of the
  //     `N_TILE_FUSED_ACT` env setting — the [M, I] arena has no
  //     room for a separate-pass swiglu, so fusion is a correctness
  //     contract, not a tuning knob.  flat_n_tile detects `ldc < N`
  //     and engages its per-thread-scratch + OOP swiglu branch;
  //     `act_fused` comes back true unconditionally.
  const char *pass1_mode = nullptr;
  const bool act_fused = group_matmul_run_parallel_dispatch(
      layout, transA, transB, M, N, K, alpha,
      src, lda, weight, ldb, bias, beta, op1_dst, op1_ldc,
      is_weights_const, params, num_threads, &pass1_mode,
      act, act_dtype);

  // Separate-pass activation when the dispatcher cannot fuse (e.g.
  // ALGO 3 + silu_and_mul / gelu_and_mul).  Never fires in tight mode
  // — the picker already required ALGO 3 + swiglu + fused-act enabled.
  if (act != grp_matmul_gated_act_t::none && !act_fused) {
    grp_matmul_gated_act_params act_p;
    act_p.act = act;
    status_t act_st = group_matmul_moe_act_execute(
        &act_p, op1_dst, M, N, op1_ldc, act_dtype, num_threads);
    if (act_st != status_t::success) return act_st;
  }

  // ── Pass 2: Op2 (down_proj) — setup ───────────────────────────────
  // Source = activated Op1 output in dst[:, 0:K_down] with stride ldc.
  //
  // Two-phase scratch setup for zero per-call allocator traffic on the
  // steady state:
  //
  //   (1) Grow-only `resize(n, value)` for `alpha_down` / `beta_down` /
  //       `transA_down`: the Op2 constants are `1.0f / 0.0f / false`
  //       on every call.  New slots (when num_ops grows beyond the
  //       previous high-water mark) are initialised with the constant;
  //       existing slots keep the constant from earlier calls because
  //       the per-expert loop below no longer writes them.
  //
  //   (2) Per-expert write loop for `K_down` / `src_down` / `params_down`
  //       (and `op2_dst_internal` in internal-alloc mode).  The per-
  //       expert data touched for slot i stays on the same cache lines
  //       across all writes inside the iteration, so num_ops × constant
  //       writes are linear-scan friendly.
  //
  // params_down policy: `resize(num_ops)` reuses existing matmul_params
  // slots across calls (avoids the destruct + construct cost of
  // assign(num_ops, matmul_params{}) for a type that carries postop_,
  // quant_params, plugin_op).  We only reset the fields the Op2 dispatch
  // could mutate (lowoha_algo) or that vary per call (dtypes,
  // num_threads); mem_format_a / mem_format_b / dynamic_quant / postop_
  // are left at their matmul_params() default values (see slot layout
  // comment inside the loop below).
  //
  // K_down is sized to `N.size()` rather than `num_ops` so the Pass-2
  // prepack reads a fully-populated K vector across `[0, num_ops_total)`.
  // The per-ALGO `prepack_for_algo_X` invoked from inside
  // `group_matmul_run_parallel_dispatch` for Pass 2 iterates
  // `min(num_ops_total, weight.size, K.size, N.size, ldb.size, transB.size)`;
  // every other vector (fused.down_weight / N_down / ldb_down / transB)
  // is sized to `total_matmul` per the framework's prepack-extras
  // contract, so K_down must match or the warmer truncates to
  // `num_ops` and the [num_ops, num_ops_total) tail of Op2 weights
  // never gets warmed — defeating the spike-elimination goal.
  // The [num_ops, N.size()) tail is unused for the actual Op2
  // dispatch (M.size() = num_ops bounds that loop) but available
  // for the prepack iteration.
  scratch.K_down.resize(N.size());
  for (size_t i = 0; i < N.size(); ++i) {
    scratch.K_down[i] = op2_k_for_act(N[i], act);
  }

  // alpha_down / beta_down / transA_down carry the same Op2 constants
  // on every call (1.0f / 0.0f / false).  Use `resize(n, value)` so
  // NEW slots are initialized with the constant; EXISTING slots keep
  // the constant from earlier calls (the per-expert loop below no
  // longer writes them).  After the first call these three are
  // zero-touch per call.  These are dispatch-side-only (the prepack
  // doesn't read them), so num_ops sizing is sufficient.
  scratch.alpha_down  .resize(num_ops, 1.0f);
  scratch.beta_down   .resize(num_ops, 0.0f);
  scratch.transA_down .resize(num_ops, false);
  scratch.src_down.resize(num_ops);
  scratch.params_down.resize(num_ops);
  if (op2_internal) scratch.op2_dst_internal.resize(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    // Op2 src = activated Op1 output (in op1_dst at op1_ldc stride:
    // N for wide layouts, N/2 for tight layout).
    scratch.src_down[i]    = op1_dst[i];

    // params_down slot layout:
    //   * `lowoha_algo` is both an input hint (`none` = let dispatcher
    //     pick via matmul_config) AND an output (dispatcher writes the
    //     chosen kernel back, see lowoha_matmul_utils.cpp:753).  Must
    //     be reset to `none` every call so a dispatcher pick from an
    //     earlier call does not force the same kernel on the next.
    //   * `dtypes` / `num_threads` vary per-call with caller's Op1
    //     params + fused.bias_dt_down.
    //   * `mem_format_a / mem_format_b / dynamic_quant / postop_`
    //     are NEVER mutated by the Op2 dispatch path on this scratch
    //     (verified by grep across lowoha_operators); the slot's
    //     default-constructed values (`'n'` / `'n'` / `false` / empty)
    //     match the required dispatch contract, so no per-call reset
    //     is issued here — `resize()`-grown slots are already correct
    //     and existing slots are never dirtied.
    matmul_params &p = scratch.params_down[i];
    p.lowoha_algo  = matmul_algo_t::none;
    p.dtypes.src   = params[i].dtypes.dst;
    p.dtypes.wei   = params[i].dtypes.wei;
    p.dtypes.dst   = params[i].dtypes.dst;
    p.dtypes.bias  = fused.bias_dt_down;
    p.num_threads  = params[i].num_threads;
    // active_matmul / total_matmul propagate from the outer params
    // so the Pass-2 per-ALGO prepack inside `group_matmul_run_parallel_dispatch`
    // sees the full active/total contract.  Without this, the
    // default-constructed `params_down[i]` slot has both fields = 0,
    // `build_prepack_params` falls back to `num_ops_total = num_ops_active`,
    // and Pass 2 warms only `num_ops_active` Op2 weights — leaving
    // the prepack-extras tail of Op2 weights cold while the
    // framework-hint regime warmed the full pool for Op1.  The
    // dispatch path would then pack the missing Op2 entries
    // on-demand on the first call that routes an inactive expert
    // into the firing set, defeating the spike-elimination goal
    // (and breaking the symmetry between the two passes — Op1 warm,
    // Op2 cold).
    p.active_matmul = params[i].active_matmul;
    p.total_matmul  = params[i].total_matmul;

    if (op2_internal) {
      // Op2 writes back into the caller's `src[]` buffer (in-place
      // reuse) with stride `lda[]`.  const_cast is well-defined
      // because the caller signalled op2_internal by clearing
      // fused.dst_down, which implies accepting src reuse as the
      // Op2 output (validator already checked lda[i] >= N_down[i]).
      scratch.op2_dst_internal[i] = const_cast<void *>(src[i]);
    }
  }

  const std::vector<void *> &op2_dst =
      op2_internal ? scratch.op2_dst_internal : fused.dst_down;
  const std::vector<int> &op2_ldc =
      op2_internal ? lda : fused.ldc_down;

  // ── Pass 2: Op2 (down_proj) dispatch ────────────────────────────────
  // Single route: `group_matmul_run_parallel_dispatch` with `act=none`.
  // Honours `ZENDNNL_GRP_MATMUL_ALGO` (1..5) and `ZENDNNL_MATMUL_ALGO`
  // for inner BLAS, and routes through the custom BF16 microkernel
  // inside flat_n_tile when `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1` and
  // the ALGO 3 path is selected.
  //
  // Op2's lda (= op1_ldc) per Op1 layout × activation:
  //   wide  + gated act  — lda=N,    K_down=N/2 (reads first half;
  //                                  skips up cols compacted by act).
  //   wide  + act=none   — lda=N,    K_down=N   (full pass-through).
  //   tight + gated act  — lda=N/2,  K_down=N/2 (perfectly packed).
  //   (tight + act=none is precluded by `pick_fused_moe_want_tight`,
  //    which gates tight on swiglu_oai_mul.)
  const char *pass2_mode = nullptr;
  group_matmul_run_parallel_dispatch(
      layout, scratch.transA_down, transB, M, fused.N_down, scratch.K_down,
      scratch.alpha_down,
      scratch.src_down, op1_ldc,
      fused.down_weight, fused.ldb_down,
      fused.bias_down, scratch.beta_down,
      op2_dst, op2_ldc,
      is_weights_const, scratch.params_down, num_threads, &pass2_mode,
      grp_matmul_gated_act_t::none, act_dtype);

  // ── Optional MoE post-op: weighted reduce over per-expert outputs ───
  // The post-op is the natural "Stage 4" of the fused MoE pipeline (Op1
  // → activation → Op2 → weighted reduce), so when the caller supplies
  // moe_postop we run it here rather than forcing them to issue a
  // separate call after we return.  D = fused.N_down[0] — the dispatcher
  // already validated N_down is uniform across experts when both
  // moe_postop and fused_moe are active.
  if (moe_postop != nullptr) {
    const int D_down = fused.N_down[0];
    status_t postop_st = group_matmul_moe_postop_execute(
        moe_postop, D_down, num_threads, params[0].dtypes.dst);
    if (postop_st != status_t::success)
      return postop_st;
  }

  // Compose gemm_mode string revealing Op1 / Op2 / Op-mode / postop
  // dispatch for profiler / apilog.  Thread-local `std::string` so
  // the caller's `const char *` stays valid until the next call on
  // this thread; `clear()` preserves the internal buffer capacity
  // across calls (no re-allocation on the steady state).  C++ string
  // `append` chain is used in place of snprintf to follow the
  // library's "no C-style formatted output" convention.
  if (gemm_mode_out != nullptr) {
    static thread_local std::string mode_buf;
    mode_buf.clear();
    mode_buf.reserve(64);
    mode_buf.append("fused_moe_2pass");
    // intalloc tag reflects which side(s) the library is managing.
    // Empty tag means both Op1 and Op2 are caller-allocated (the
    // legacy "all-buffers-from-caller" path).
    if (op1_internal && op2_internal) mode_buf.append("_intalloc");
    else if (op1_internal)            mode_buf.append("_intalloc_op1");
    else if (op2_internal)            mode_buf.append("_intalloc_op2");
    if (want_tight)                   mode_buf.append("_tight");
    mode_buf.append("(op1=");
    mode_buf.append(pass1_mode != nullptr ? pass1_mode : "?");
    mode_buf.append(",op2=");
    mode_buf.append(pass2_mode != nullptr ? pass2_mode : "?");
    mode_buf.append(")");
    if (moe_postop != nullptr) mode_buf.append("+postop");
    *gemm_mode_out = mode_buf.c_str();
  }
  return status_t::success;
}

// ═══════════════════════════════════════════════════════════════════════
// Legacy ABI-preserving overload (no moe_postop parameter).  Forwards to
// the primary entry with moe_postop = nullptr.  Kept as a separate non-
// inline exported symbol so binaries built against the pre-postop
// version of the library continue to find their mangled name.
// ═══════════════════════════════════════════════════════════════════════
status_t group_matmul_fused_moe_execute(
    const grp_matmul_fused_moe_params &fused,
    grp_matmul_gated_act_t act, data_type_t act_dtype,
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
    const char **gemm_mode_out) {
  return group_matmul_fused_moe_execute(
      fused, act, act_dtype, layout, transA, transB,
      M, N, K, alpha, src, lda, weight, ldb, bias, beta,
      dst, ldc, is_weights_const, params, num_threads,
      gemm_mode_out, /*moe_postop=*/nullptr);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
