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
/// This translation unit is the THIN DISPATCHER for the fused-MoE
/// public entry point.  The public entry
/// `group_matmul_fused_moe_execute()` is a short orchestrator that:
///
///   1. Detects per-side internal-alloc state (Op1 dst / Op2 dst).
///   2. Validates inputs via `validate_fused_moe_inputs()`.
///   3. Picks the Op1 arena layout (wide vs tight) via
///      `pick_fused_moe_want_tight()`.
///   4. Sets up the Op1 arena + per-expert pointers via
///      `setup_op1_arena_and_layout()`.
///   5. Populates the Op2 dispatch scratch via
///      `setup_op2_dispatch_scratch()`.
///   6. Tries vertical-fusion FIRST (M-tile pipeline; the executor
///      accepts ONE of three regimes on both halves: BF16
///      end-to-end, WOQ-INT4 s4/u4 weights, OR DQ-INT8 per-token-
///      symmetric on s8 weights) via `try_flat_m_tile_pipeline_bf16()`
///      from `group_matmul_m_tile.hpp`.  If that engages, both Op1
///      and Op2 have been computed by the pipeline executor and the
///      profiler `gemm_mode` reflects one of `vertical_fusion_bf16`,
///      `vertical_fusion_woq`, or `vertical_fusion_dqint8` depending
///      on the weight dtype + dynamic-quant flag.
///   7. Otherwise runs the legacy two-pass via
///      `run_fused_moe_legacy_two_pass()`: Op1+act through
///      `group_matmul_run_parallel_dispatch` (which internally picks
///      ALGO 1..5 / flat_n_tile / flat_m_tile / etc.), then Op2
///      through the same dispatcher with `act=none`.
///   8. Runs an optional MoE weighted-reduce post-op (Stage 4).
///   9. Composes the gemm_mode string for profiler / apilog.
///
/// All backend-specific code lives in the per-ALGO translation units:
///   * `group_matmul_m_tile.cpp`  — `flat_m_tile`,
///                                   `flat_m_tile_pipeline_bf16`,
///                                   `try_flat_m_tile_pipeline_bf16`.
///   * `group_matmul_n_tile.cpp`  — `flat_n_tile`.
///   * `group_matmul_dispatch.cpp`— `group_matmul_run_parallel_dispatch`
///                                   (ALGO 1..5 routing).
/// This file owns only the fused-MoE-specific glue: validation, arena
/// management, Op2-dispatch-scratch population, and the dispatch fork.
///
/// Adaptive arena layout (internal-alloc mode only):
///   * `pick_fused_moe_want_tight()` decides per call whether to
///     allocate a tight [M, I] arena or the classic wide [M, 2I].
///   * Tight is requested when (a) `op1_internal` is true,
///     (b) `act` admits a fused per-thread epilogue
///     (`a3_can_fuse_act(act, CUSTOM_KERNEL)`),
///     (c) `env_algo ∈ {0, 3}`, (d) the env override allows it,
///     AND (e) `select_grp_matmul_algo()` agrees to route to ALGO 3.
///   * When tight is selected the Op1 arena holds `sum_i M[i]·I[i]·
///     dst_elem` bytes (half of the wide case) and `op1_ldc[i] = I[i]`.
///     Op2 then reads the activated output at tight stride — halves
///     Op2's src DRAM traffic vs the wide layout.
///   * The dispatcher in `group_matmul_dispatch.cpp::a3_fuses` auto-
///     enables fused activation when it detects `ldc < N`, regardless
///     of the `N_TILE_FUSED_ACT` env flag — tight layout is a
///     correctness contract, not a perf toggle.
///
/// Op2 output mode (see grp_matmul_fused_moe_params doc-block in the
/// public header for full semantics):
///   * Legacy / caller-allocated : caller fills both `dst[]` (Op1 dst,
///     entry API) and `fused.dst_down[]` (Op2 dst); the library
///     writes into them.  Non-internal-alloc callers always run wide.
///   * Internal-alloc + src-reuse : caller leaves BOTH `dst[]` (all
///     nullptr / empty) AND `fused.dst_down` empty.  The library
///     allocates Op1 scratch in a persistent thread-local arena
///     (sized to the high-water mark) and runs Op2 reading from the
///     scratch and writing back into the caller's `src[]` buffer.
///   * Mixed (one filled, one empty) is rejected by the validator.

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include <omp.h>

#include "detect_internal_alloc.hpp"
#include "group_matmul_direct.hpp"
#include "m_tile/group_matmul_m_tile.hpp"  // try_flat_m_tile_pipeline_bf16
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

// ═══════════════════════════════════════════════════════════════════════
// File-private types (persistent thread-local scratch).
// ═══════════════════════════════════════════════════════════════════════

namespace {

// Per-thread persistent Op1 arena used by fused-MoE internal-alloc.
// Owns a single 64-byte-aligned slab whose capacity monotonically
// grows to the high-water mark this thread has seen.  Per-expert
// pointers are tightly packed byte-offsets into the slab (only the
// base is 64B-aligned; per-expert first rows fall wherever the
// previous expert's footprint ended).  Freed by the destructor on
// thread exit; freed + reallocated when a call needs more than the
// current capacity.
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

// ───────────────────────────────────────────────────────────────────────
// File-scope thread-local accessors.
// ───────────────────────────────────────────────────────────────────────
//
// Both the Op1 arena (raw `posix_memalign` slab) and the Op2 dispatch
// scratch (set of growing `std::vector`s) need to be reachable from
// OUTSIDE `group_matmul_fused_moe_execute` so a public clear API can
// drop them between workload phases (long-running model servers, OMP
// pool shared with the host process, etc.).  The pre-PR layout buried
// these as function-local statics, which made them visible ONLY to
// the executor — there was no host-visible knob to bound the high-
// water-mark footprint.
//
// Accessor pattern: a `Meyers singleton`-style returning a reference
// to the current thread's instance.  Calling `get_…()` on a thread
// that has not previously called it triggers the default-construct,
// which is cheap (zeroed POD-ish state).  Side benefit: the executor
// keeps using local references (`arena`, `scratch`) and reads the
// same way as before — no per-call cost.
inline FusedMoEArena &get_thread_local_arena() {
  static thread_local FusedMoEArena arena;
  return arena;
}
inline FusedMoEScratch &get_thread_local_scratch() {
  static thread_local FusedMoEScratch scratch;
  return scratch;
}

// Per-thread reset.  MUST be called on the SAME thread that owns the
// scratch (TLS visibility).  The public `clear_fused_moe_scratch()`
// API below spawns an OMP parallel region so every worker in the
// current OMP pool hits this on its own TLS.
//
// Reset semantics:
//   * Arena    : `free(buf)` + reset to nullptr/0.  Next call re-
//                allocates from scratch (a one-shot cost).
//   * Scratch  : swap each `std::vector` with a freshly-default-
//                constructed empty temporary.  This is the canonical
//                C++ "force capacity release" idiom — the temporary's
//                destructor at end-of-scope frees the original storage
//                deterministically.  `clear()` alone only resets
//                logical size (capacity retained); `shrink_to_fit()`
//                is a non-binding REQUEST per the C++ standard and
//                some allocators / libstdc++ configurations may
//                no-op it, defeating the purpose of this API.
//
// Cheap when nothing has been allocated (the arena ptr is null, the
// vectors are empty), so it is safe to call unconditionally on every
// worker.
inline void reset_thread_local_fused_moe_state() {
  FusedMoEArena &arena = get_thread_local_arena();
  std::free(arena.buf);
  arena.buf = nullptr;
  arena.cap = 0;

  FusedMoEScratch &s = get_thread_local_scratch();
  // Deterministic dealloc: swap with empty temporary → temporary's
  // dtor frees the old buffer at end of statement.  See doc-block
  // above for why `shrink_to_fit()` is NOT used here.
  std::vector<int>{}.swap(s.K_down);
  std::vector<float>{}.swap(s.alpha_down);
  std::vector<float>{}.swap(s.beta_down);
  std::vector<bool>{}.swap(s.transA_down);
  std::vector<const void *>{}.swap(s.src_down);
  std::vector<matmul_params>{}.swap(s.params_down);
  std::vector<void *>{}.swap(s.op1_dst_internal);
  std::vector<void *>{}.swap(s.op2_dst_internal);
  std::vector<int>{}.swap(s.op1_ldc_local);
}

// ═══════════════════════════════════════════════════════════════════════
// Adaptive arena-layout picker.
// ═══════════════════════════════════════════════════════════════════════
//
// Decides per call whether to request the tight [M, I] arena (half the
// wide [M, 2I] footprint) for Op1's internal-alloc output.  Returns
// `true` to request tight, `false` for wide.  All gates are O(num_ops)
// or cheaper; the function has no side effects.
//
// Correctness gates (all must pass for tight to be considered):
//
//   1. `op1_internal` — only when the library owns the Op1 arena.
//      Caller-allocated Op1 paths supply their own dst[] with caller-
//      chosen ldc; the tight arena layout is library-private.
//
//   2. `a3_can_fuse_act(act, CUSTOM_KERNEL)` — the activation admits
//      ALGO 3's per-thread fused epilogue (in-register store_pair for
//      CK, or the standard backend's OOP swiglu writer when CK is
//      off).  Today: `swiglu_oai_mul` (both backends) and
//      `silu_and_mul` / `gelu_and_mul` (CK only — the standard
//      backend's wide-arena helper is swiglu-only).
//
//   3. `env_algo ∈ {0, 3}` — tight requires Op1 to run in flat_n_tile;
//      a caller forcing ALGO 1/2/4/5 explicitly asked for a non-N-tile
//      strategy and silently flipping them violates intent.
//
//   4. Env override `ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT`: unset (auto)
//      and force-tight request tight, force-wide (=0) rejects.
//
//   5. `select_grp_matmul_algo()` actually would return 3 for this
//      (shapes, params, num_threads).  Without this, auto-select
//      could pick ALGO 1 (small N, non-N-tile-viable) on the tight
//      arena and overrun it.
//
// Today the auto policy is "tight whenever safe" — tight halves Op2's
// Op1-src DRAM traffic and flat_n_tile's planner adapts to any num_ops.
// If a future shape regresses, plug a `num_ops`-keyed threshold here.
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
  if (!a3_can_fuse_act(act, get_grp_matmul_custom_kernel())) return false;
  if (env_algo != 0 && env_algo != 3) return false;
  if (get_grp_matmul_fused_moe_tight() == 0) return false;
  const int algo_would =
      select_grp_matmul_algo(layout, M, N, K, params, num_threads);
  return algo_would == 3;
}

// ═══════════════════════════════════════════════════════════════════════
// Input validation.
// ═══════════════════════════════════════════════════════════════════════
//
// `validate_fused_moe_inputs` runs the entire input-shape contract in
// one place.  Three classes:
//
//   (1) primary-vector emptiness + per-vector size consistency — every
//       required vector must be sized to at least `num_ops` (with the
//       dst / ldc / dst_down / ldc_down exceptions in internal-alloc
//       mode).  Strict equality was the original contract; relaxing
//       to `<` accepts the prepack-extras tail layout without
//       affecting legacy callers.
//
//   (2) per-expert dimension / leading-stride / required-pointer
//       sanity for active experts — non-negative M, positive N/K,
//       even N (required by the swiglu half-split), lda/ldb/ldc/
//       ldb_down/ldc_down all large enough for the row-major access
//       pattern, and non-null src/weight/down_weight pointers (plus
//       dst/dst_down in legacy caller-allocated mode).
//
//   (3) internal-alloc dtype safety — cross-expert dst dtype
//       uniformity and per-expert matched-precision (src == dst) when
//       either side is internal-alloc.  Both feed sizing math for the
//       Op1 arena slab.
//
//   (4) cross-expert N_down uniformity (only when moe_postop is
//       engaged) — the weighted-reduce stage uses `fused.N_down[0]`
//       as the common D for every expert; a divergent expert would
//       cause OOB reads past its row.  Correctness-critical, always-on.
//
// Silent-wrong-result paths (mixed-state dst[] iteration, bias dtype
// declaration, activation dtype bucket) run under the
// `op_instrumentation::validate` diagnostic gate (default ON; bypassed
// only when explicitly set to "0") because they either are already
// covered by `group_matmul_direct`'s phase-D/F validator or produce
// wrong numbers without corrupting memory.
//
// On success: writes `*out_total_M` (sum of M[i] across active
// experts) and `*out_total_bytes_internal` (Op1 arena byte budget for
// the wide layout — caller halves it when tight is selected).  Both
// outputs are uninitialised on failure.
inline status_t validate_fused_moe_inputs(
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
    const std::vector<matmul_params> &params,
    const group_matmul_moe_postop_params *moe_postop,
    bool op1_internal, bool op2_internal,
    size_t dst_elem_internal,
    int64_t *out_total_M,
    size_t *out_total_bytes_internal) {
  const size_t num_ops = M.size();

  // Vector sizes — must hold AT LEAST `num_ops` entries each.  Anything
  // past `num_ops` is the framework's prepack-extras tail and is never
  // read by the dispatch loops downstream.
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
  // Op2 weight quant is optional: empty `down_scale` / `down_zp` means
  // "Op2 weight un-quantized".  When non-empty, each MUST cover every
  // active expert — a partial vector would silently leave the tail
  // experts un-quantized.
  if (!fused.down_scale.empty() && fused.down_scale.size() < num_ops)
    return status_t::failure;
  if (!fused.down_zp.empty() && fused.down_zp.size() < num_ops)
    return status_t::failure;
  // Op1 dst/ldc — when caller-allocated must reach `num_ops`; when
  // library-managed (op1_internal) the vectors may be empty or sized
  // to at least `num_ops` (caller passed all-null placeholders).
  if (op1_internal) {
    if (!dst.empty() && dst.size() < num_ops) return status_t::failure;
    if (!ldc.empty() && ldc.size() < num_ops) return status_t::failure;
  } else {
    if (dst.size() < num_ops || ldc.size() < num_ops)
      return status_t::failure;
  }
  if (op2_internal) {
    if (!fused.dst_down.empty() && fused.dst_down.size() < num_ops)
      return status_t::failure;
    if (!fused.ldc_down.empty() && fused.ldc_down.size() < num_ops)
      return status_t::failure;
  } else {
    if (fused.dst_down.size() < num_ops
        || fused.ldc_down.size() < num_ops)
      return status_t::failure;
  }

  // Per-expert sweep — covers classes (2) and (3) above plus
  // accumulates total_M / total_bytes_internal for the caller.
  int64_t total_M = 0;
  size_t total_bytes_internal = 0;
  for (size_t i = 0; i < num_ops; ++i) {
    if (M[i] < 0 || N[i] <= 0 || K[i] <= 0) return status_t::failure;
    // N must be even ONLY when a gated activation is fused (swiglu /
    // silu / gelu_and_mul collapse pairs of cols).  For act=none Op1
    // output flows into Op2 verbatim, so any N is admissible.
    if (act != grp_matmul_gated_act_t::none && (N[i] & 1) != 0)
      return status_t::failure;
    if (fused.N_down[i] <= 0) return status_t::failure;

    if (lda[i] < K[i]) return status_t::failure;
    if (ldb[i] < (transB[i] ? K[i] : N[i])) return status_t::failure;
    const int K_down = op2_k_for_act(N[i], act);
    if (fused.ldb_down[i] < (transB[i] ? K_down : fused.N_down[i]))
      return status_t::failure;

    if (!op1_internal) {
      if (ldc[i] < N[i]) return status_t::failure;
    }
    if (!op2_internal) {
      if (fused.ldc_down[i] < fused.N_down[i]) return status_t::failure;
    } else if (M[i] > 0) {
      // ── op2_internal contract: src[] is REUSED as Op2's dst ────────
      //
      // When the caller signals `op2_internal` (empty / all-null
      // `fused.dst_down`), Pass-2 writes its M×N_down output back
      // into `src[i]` with row stride `lda[i]`.  The caller's
      // allocation MUST cover `M[i] · lda[i] · src_elem` bytes —
      // i.e. the WIDEST row stride is what bounds the allocation.
      //
      // Two correctness gates ZenDNN can enforce:
      //
      //   (G1) `lda[i] >= max(K[i], N_down[i])`.  The row stride must
      //        be wide enough for the larger of the two passes that
      //        write into the buffer (Op1 reads K[i] cols per row;
      //        Op2 writes N_down[i] cols per row, both at stride
      //        `lda[i]`).
      //
      //   (G2) `lda[i] == K[i]` OR the caller has explicitly opted
      //        into the asymmetric layout by setting `lda[i] >=
      //        max(K[i], N_down[i])`.  We can't detect under-
      //        allocation directly — but we CAN reject the common
      //        silent-bug shape: an asymmetric MoE (`N_down != K`)
      //        with `op2_internal=true` AND `lda[i] == K[i]` (the
      //        "natural Op1 stride") — that combination guarantees
      //        Pass-2 will overrun the caller's allocation if the
      //        caller sized src[] for Op1 only.
      //
      // (G1) is the legacy check (preserved verbatim below).  (G2)
      // is new: it elevates the validator from "wide-enough stride"
      // to "consistent stride AND wide enough", which catches the
      // gpt-oss / Mixtral / typical-MoE silent-corruption path where
      // K_in == hidden_dim but N_down can be smaller (rare) or
      // larger (with bias projections / future variants).  When this
      // gate trips, the validator emits a single `log_error` so the
      // caller sees a clear failure instead of a downstream
      // `std::bad_array_new_length` or a corrupted activation.
      //
      // OUT OF SCOPE for the validator: detecting cases where the
      // caller passed a correctly-wide `lda` but UNDER-ALLOCATED
      // `src[i]` (e.g. `lda[i] = N_down`, `src[i]` sized to
      // `M*K*elem`).  No defensive check can spot that without an
      // allocation introspection API — it remains a caller-contract
      // requirement.
      if (lda[i] < fused.N_down[i]) {
        log_error("group_matmul_fused_moe: op2_internal requires "
                  "lda[", i, "] >= fused.N_down[", i, "] (got lda=",
                  lda[i], ", N_down=", fused.N_down[i], ").  src[",
                  i, "] is reused as Op2's destination and is "
                  "written with row stride lda; the stride must be "
                  "wide enough for the Op2 output columns.  Either "
                  "(a) widen lda and allocate src[] for "
                  "M*lda*src_elem bytes, or (b) pass an explicit "
                  "fused.dst_down[] (caller-allocated Op2 dst).");
        return status_t::failure;
      }
    }

    // Cross-expert dst-dtype uniformity / matched-precision are
    // always-on when either side is internal-alloc (Op1 arena slab
    // sizing and Op2 in-place write footprint both depend on
    // params[0].dtypes.dst).
    if ((op1_internal || op2_internal) && M[i] > 0) {
      if (params[i].dtypes.dst != params[0].dtypes.dst)
        return status_t::failure;
      if (params[i].dtypes.src != params[i].dtypes.dst)
        return status_t::failure;
    }

    if (M[i] > 0) {
      if (src[i] == nullptr || weight[i] == nullptr)
        return status_t::failure;
      if (fused.down_weight[i] == nullptr) return status_t::failure;
      if (!op1_internal && dst[i] == nullptr) return status_t::failure;
      if (!op2_internal && fused.dst_down[i] == nullptr)
        return status_t::failure;
      total_M += M[i];
      if (op1_internal) {
        // Overflow-safe per-expert byte computation:
        //   per_expert_bytes = M[i] * N[i] * dst_elem_internal
        // followed by an overflow-safe running sum into
        // total_bytes_internal.  Both `M[i]` and `N[i]` have been
        // validated >= 0 / > 0 above, so the casts to size_t are
        // well-defined.  Two distinct overflow gates:
        //   (a) per-expert product — pathological caller passing
        //       huge M/N (e.g. INT_MAX × INT_MAX × 8 wraps size_t).
        //   (b) running sum — a long expert list with individually
        //       reasonable per-expert footprints whose total still
        //       wraps (no realistic shape hits this on a 64-bit
        //       host, but the gate is cheap and defends against a
        //       caller bug pumping garbage).
        // Either trip drops to `status_t::failure`; the arena is
        // never asked to size beyond size_t-representable bytes,
        // so `posix_memalign` cannot be fed a wrapped count that
        // succeeds-but-is-too-small (= silent heap corruption).
        const size_t m_sz = static_cast<size_t>(M[i]);
        const size_t n_sz = static_cast<size_t>(N[i]);
        size_t per_expert_bytes = 0;
        if (__builtin_mul_overflow(m_sz, n_sz, &per_expert_bytes))
          return status_t::failure;
        if (__builtin_mul_overflow(per_expert_bytes, dst_elem_internal,
                                   &per_expert_bytes))
          return status_t::failure;
        if (__builtin_add_overflow(total_bytes_internal, per_expert_bytes,
                                   &total_bytes_internal))
          return status_t::failure;
      }
    }
  }

  // Cross-expert N_down uniformity (when moe_postop is engaged).  The
  // duplicate of `group_matmul_direct`'s phase-G check defends the
  // path for any future caller that bypasses that validator.
  if (moe_postop != nullptr) {
    for (size_t i = 1; i < num_ops; ++i)
      if (fused.N_down[i] != fused.N_down[0]) return status_t::failure;
  }

  // Diagnostic-only validators — silent-wrong-result paths only.  See
  // doc-block on the validator above for what stays always-on.
  const status_t val = op_instrumentation::validate([&]() {
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
    bool any_bias_down = false;
    for (size_t i = 0; i < num_ops; ++i)
      if (fused.bias_down[i] != nullptr) { any_bias_down = true; break; }
    if (any_bias_down && fused.bias_dt_down == data_type_t::none)
      return status_t::failure;
    if (act != grp_matmul_gated_act_t::none
        && act_dtype != data_type_t::f32
        && act_dtype != data_type_t::bf16)
      return status_t::failure;
    return status_t::success;
  });
  if (val != status_t::success) return val;

  *out_total_M = total_M;
  *out_total_bytes_internal = total_bytes_internal;
  return status_t::success;
}

// ═══════════════════════════════════════════════════════════════════════
// Op1 arena + per-expert pointer / stride setup.
// ═══════════════════════════════════════════════════════════════════════
//
// Sizes and (re-)allocates the persistent thread-local Op1 arena
// (only when `op1_internal` AND we need more than the high-water
// mark).  Then populates `scratch.op1_dst_internal[]` and (when
// tight is requested) `scratch.op1_ldc_local[]` in a single sweep
// over num_ops.  Returns the (op1_dst, op1_ldc) view-pair the
// dispatch fork below will pass to the executors.
//
// `arena_bytes_wide` is the wide-layout budget that
// `validate_fused_moe_inputs()` accumulated; halved here when
// `want_tight` is set.
//
// On allocation failure returns `status_t::failure`; on success
// writes the view-pair through the out-parameters.  Both views point
// into either caller-supplied vectors or into `scratch`'s persistent
// storage, so they stay valid until the next call on this thread.
inline status_t setup_op1_arena_and_layout(
    FusedMoEArena &arena,
    FusedMoEScratch &scratch,
    bool op1_internal,
    bool want_tight,
    size_t arena_bytes_wide,
    size_t dst_elem_internal,
    const std::vector<int> &N,
    const std::vector<int> &M,
    const std::vector<int> &ldc_caller,
    const std::vector<void *> &dst_caller,
    const std::vector<void *> *&out_op1_dst,
    const std::vector<int>  *&out_op1_ldc) {
  const size_t num_ops = M.size();

  size_t arena_bytes = arena_bytes_wide;
  if (want_tight) arena_bytes /= 2;

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

  // Populate Op1 per-expert pointer / stride scratch.  Per-expert row
  // width depends on the layout:
  //   * wide  : N[i]   cols/row (raw GEMM output; swiglu writes
  //             activated I cols into the first half in place).
  //   * tight : N[i]/2 cols/row (already-activated I-wide output via
  //             flat_n_tile's per-thread-scratch + OOP swiglu path).
  // Inactive (M <= 0) slots get an explicit nullptr.
  if (op1_internal) scratch.op1_dst_internal.resize(num_ops);
  if (want_tight)   scratch.op1_ldc_local.resize(num_ops);
  if (op1_internal || want_tight) {
    char *base = op1_internal
                     ? static_cast<char *>(arena.buf)
                     : nullptr;
    size_t cursor = 0;
    for (size_t i = 0; i < num_ops; ++i) {
      const int row_cols = want_tight ? (N[i] / 2) : N[i];
      if (want_tight) scratch.op1_ldc_local[i] = row_cols;
      if (op1_internal) {
        if (M[i] <= 0 || base == nullptr) {
          scratch.op1_dst_internal[i] = nullptr;
        } else {
          // Overflow-safe per-expert slab accumulation.  Sister to
          // the validator's pre-flight overflow gate — that gate
          // computed the WIDE total; here we incrementally build
          // per-expert offsets and must independently confirm that
          // `cursor + (M*row_cols*elem)` stays representable.  In
          // tight mode `row_cols = N/2`, so the per-expert footprint
          // is half the validator's wide computation — strictly
          // smaller, but we still re-check because the multiplier
          // chain is different.  A trip aborts with `failure` BEFORE
          // any thread proceeds past `setup_op1_arena_and_layout`,
          // so the executors never see a wrap-around pointer.
          scratch.op1_dst_internal[i] = base + cursor;
          const size_t m_sz       = static_cast<size_t>(M[i]);
          const size_t row_sz     = static_cast<size_t>(row_cols);
          size_t per_expert_bytes = 0;
          if (__builtin_mul_overflow(m_sz, row_sz, &per_expert_bytes))
            return status_t::failure;
          if (__builtin_mul_overflow(per_expert_bytes, dst_elem_internal,
                                     &per_expert_bytes))
            return status_t::failure;
          if (__builtin_add_overflow(cursor, per_expert_bytes, &cursor))
            return status_t::failure;
        }
      }
    }
    // The arena was sized by the validator using the same per-expert
    // formula (wide; halved in this function for tight) — assert the
    // invariant in debug builds.  If the planner ever produces a
    // cursor > arena.cap, the next Op1 GEMM would write past the slab
    // boundary, so this is correctness-critical.  In release builds
    // the gate above + the arena-bytes math in the caller cover the
    // same property; the assert is a belt-and-braces during develop-
    // ment.
    if (op1_internal) {
      assert(cursor <= arena.cap
             && "Op1 arena overflow: cumulative per-expert footprint "
                "exceeds arena capacity (sizing math regression).");
    }
  }

  // Op1 dst / ldc views for Pass 1 dispatch:
  //   op1_internal + tight  : library arena, op1_ldc[i] = N[i]/2.
  //   op1_internal + wide   : library arena, op1_ldc[i] = N[i].
  //   caller-allocated      : caller's dst / ldc (wide by contract).
  out_op1_dst = op1_internal ? &scratch.op1_dst_internal : &dst_caller;
  out_op1_ldc = want_tight ? &scratch.op1_ldc_local
                           : (op1_internal ? &N : &ldc_caller);
  return status_t::success;
}

// ═══════════════════════════════════════════════════════════════════════
// Op2 dispatch scratch setup.
// ═══════════════════════════════════════════════════════════════════════
//
// Two-phase scratch population for zero per-call allocator traffic on
// the steady state:
//
//   (1) Grow-only `resize(n, value)` for `alpha_down` / `beta_down` /
//       `transA_down`: the Op2 constants are `1.0f / 0.0f / false`
//       on every call.  New slots are initialised with the constant;
//       existing slots keep the constant from earlier calls.
//
//   (2) Per-expert write loop for `K_down` / `src_down` / `params_down`
//       (and `op2_dst_internal` in internal-alloc mode).
//
// `K_down` is sized to `N.size()` rather than `num_ops` so the Pass-2
// prepack reads a fully-populated K vector across the prepack-extras
// tail (otherwise the warmer truncates to `num_ops` and the tail of
// Op2 weights never gets warmed).
//
// Per-call quant-field reset is essential: the persistent thread-local
// `scratch.params_down` retains whatever was written on the previous
// call — a stale buffer pointer from a freed caller-side scale tensor
// would crash the next call.
//
// Op2 inherits Op1's `dynamic_quant` flag and `dtypes.compute` so the
// down_proj runs through the same dispatch path as the gate+up GEMM.
// Per-group src_scale (`dims = {M, ngroups>1}`) is rejected here
// because Op1.K != Op2.K — ngroups can't transfer; callers must use
// per-token (`{M, 1}`) instead.  Returns `status_t::failure` on that
// rejection.
inline status_t setup_op2_dispatch_scratch(
    FusedMoEScratch &scratch,
    const grp_matmul_fused_moe_params &fused,
    grp_matmul_gated_act_t act,
    size_t num_ops,
    const std::vector<int> &N,
    const std::vector<const void *> &src,
    const std::vector<int> &lda,
    const std::vector<matmul_params> &params,
    const std::vector<void *> &op1_dst,
    bool op2_internal) {
  // num_ops MUST be the ACTIVE matmul count (== M.size()) — caller
  // derives it from M.size() and passes it explicitly so this
  // function CANNOT silently drift to params.size().  Under the
  // framework prepack-extras contract `params` is sized to
  // `total_matmul` (>= active), so deriving num_ops from
  // params.size() here would walk `op1_dst` / `src` /
  // `fused.down_scale` / `fused.down_zp` past their active-sized
  // .size() and copy garbage `dims` vectors into `params_down`,
  // which the next std::vector copy on the hot path turns into a
  // `new T[garbage_size]` → `std::bad_array_new_length` crash.
  // The K_down loop below is the ONE intentional iteration over
  // `N.size()` (the total Op2 weight count) — see its inline comment.

  // `K_down` is sized to `N.size()` (the total-matmul Op2 K-vector)
  // rather than `num_ops` so the Pass-2 prepack reads a fully-
  // populated K vector across the prepack-extras tail (otherwise the
  // warmer truncates to `num_ops` and the tail of Op2 weights never
  // gets warmed).  N[i] is well-defined for all i in [0, N.size())
  // — the framework populates N for every total-matmul slot, firing
  // or not — so this is the only loop in this function that legally
  // iterates the total range.  Execution never reads K_down past
  // num_ops; the [num_ops, N.size()) tail is consumed by the
  // prepack module only.
  scratch.K_down.resize(N.size());
  for (size_t i = 0; i < N.size(); ++i) {
    scratch.K_down[i] = op2_k_for_act(N[i], act);
  }

  // Op2 dispatch-side-only constants — zero-touch per call after the
  // first call (the per-expert loop below no longer writes them).
  scratch.alpha_down  .resize(num_ops, 1.0f);
  scratch.beta_down   .resize(num_ops, 0.0f);
  scratch.transA_down .resize(num_ops, false);
  scratch.src_down.resize(num_ops);
  scratch.params_down.resize(num_ops);
  if (op2_internal) scratch.op2_dst_internal.resize(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    scratch.src_down[i] = op1_dst[i];

    // `lowoha_algo` is both input hint and output — must be reset to
    // `none` every call so a dispatcher pick from an earlier call does
    // not force the same kernel on the next.
    matmul_params &p = scratch.params_down[i];
    p.lowoha_algo  = matmul_algo_t::none;
    p.dtypes.src   = params[i].dtypes.dst;
    p.dtypes.wei   = params[i].dtypes.wei;
    p.dtypes.dst   = params[i].dtypes.dst;
    p.dtypes.bias  = fused.bias_dt_down;
    p.num_threads  = params[i].num_threads;
    p.dynamic_quant  = params[i].dynamic_quant;
    p.dtypes.compute = params[i].dtypes.compute;

    // Reset everything first, then fill in just the wei_scale /
    // wei_zp (from the caller-facing fields) and the inherited
    // src_scale dims (when dynamic_quant is on).
    p.quant_params = matmul_quantization_params_t{};
    if (!fused.down_scale.empty()) {
      p.quant_params.wei_scale.buff = fused.down_scale[i].buff;
      p.quant_params.wei_scale.dt   = fused.down_scale[i].dt;
      p.quant_params.wei_scale.dims = fused.down_scale[i].dims;
    }
    if (!fused.down_zp.empty()) {
      p.quant_params.wei_zp.buff = fused.down_zp[i].buff;
      p.quant_params.wei_zp.dt   = fused.down_zp[i].dt;
      p.quant_params.wei_zp.dims = fused.down_zp[i].dims;
    }
    if (p.dynamic_quant) {
      const auto &scale_dims = params[i].quant_params.src_scale.dims;
      if (scale_dims.size() == 2 && scale_dims[1] > 1) {
        log_error("group_matmul_fused_moe: per-group src_scale on "
                  "params[", i, "] (dims={", scale_dims[0], ",",
                  scale_dims[1], "}) unsupported; use per-token "
                  "({M, 1}).");
        return status_t::failure;
      }
      if (params[i].quant_params.src_zp.dt != data_type_t::none) {
        const auto &zp_dims = params[i].quant_params.src_zp.dims;
        if (zp_dims.size() == 2 && zp_dims[1] > 1) {
          log_error("group_matmul_fused_moe: per-group src_zp on "
                    "params[", i, "] (dims={", zp_dims[0], ",",
                    zp_dims[1], "}) unsupported; use per-token "
                    "({M, 1}).");
          return status_t::failure;
        }
      }
      p.quant_params.src_scale.buff = nullptr;
      p.quant_params.src_scale.dt   = params[i].quant_params.src_scale.dt;
      p.quant_params.src_scale.dims = params[i].quant_params.src_scale.dims;
      if (params[i].quant_params.src_zp.dt != data_type_t::none) {
        p.quant_params.src_zp.buff = nullptr;
        p.quant_params.src_zp.dt   = params[i].quant_params.src_zp.dt;
        p.quant_params.src_zp.dims = params[i].quant_params.src_zp.dims;
      }
    }
    // active_matmul / total_matmul propagate so the Pass-2 per-ALGO
    // prepack sees the full active/total contract and warms the
    // prepack-extras tail.
    p.active_matmul = params[i].active_matmul;
    p.total_matmul  = params[i].total_matmul;

    if (op2_internal) {
      // const_cast is well-defined because the caller signalled
      // op2_internal by clearing fused.dst_down, which implies
      // accepting src reuse as the Op2 output.
      scratch.op2_dst_internal[i] = const_cast<void *>(src[i]);
    }
  }
  return status_t::success;
}

// ═══════════════════════════════════════════════════════════════════════
// Per-path dispatch wrappers.
// ═══════════════════════════════════════════════════════════════════════
//
// The vertical-fusion wrapper `try_flat_m_tile_pipeline_bf16` lives
// in `group_matmul_m_tile.cpp` (Section C.2) — see its doc-block
// there for the engagement contract.  Only the legacy two-pass
// wrapper stays here because it is ALGO-agnostic: it forwards each
// pass through `group_matmul_run_parallel_dispatch`, which internally
// picks ALGO 1..5 based on shape and env knobs.

// Legacy two-pass MoE dispatch.  Pass 1 = Op1 (W13 + optional gated
// activation) via `group_matmul_run_parallel_dispatch`.  Pass 2 = Op2
// (W2 down-projection) via the same dispatcher with `act=none`.
//
// When Pass 1's dispatcher cannot fuse the activation (e.g. ALGO 3 +
// silu_and_mul / gelu_and_mul on the wide arena), a separate-pass
// activation runs between Pass 1 and Pass 2.
inline status_t run_fused_moe_legacy_two_pass(
    grp_matmul_gated_act_t act,
    data_type_t act_dtype,
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &op1_dst, const std::vector<int> &op1_ldc,
    const grp_matmul_fused_moe_params &fused,
    FusedMoEScratch &scratch,
    const std::vector<void *> &op2_dst, const std::vector<int> &op2_ldc,
    const std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads,
    const char *&pass1_mode, const char *&pass2_mode) {
  // Pass 1: Op1 (gate+up) + activation.  The dispatcher picks ALGO
  // 1..5 (or auto) per `ZENDNNL_GRP_MATMUL_ALGO` and the safety
  // gates; inner BLAS kernel honours `ZENDNNL_MATMUL_ALGO`.  For
  // tight layout the dispatcher auto-enables fused activation
  // regardless of `N_TILE_FUSED_ACT` (correctness contract).
  const bool act_fused = group_matmul_run_parallel_dispatch(
      layout, transA, transB, M, N, K, alpha,
      src, lda, weight, ldb, bias, beta, op1_dst, op1_ldc,
      is_weights_const, params, num_threads, &pass1_mode,
      act, act_dtype);

  // Separate-pass activation when the dispatcher cannot fuse (e.g.
  // ALGO 3 + silu_and_mul / gelu_and_mul on wide arena).  Never
  // fires in tight mode.
  if (act != grp_matmul_gated_act_t::none && !act_fused) {
    grp_matmul_gated_act_params act_p;
    act_p.act = act;
    const status_t act_st = group_matmul_moe_act_execute(
        &act_p, op1_dst, M, N, op1_ldc, act_dtype, num_threads);
    if (act_st != status_t::success) return act_st;
  }

  // Pass 2: Op2 (down_proj) dispatch.  Same dispatcher with `act=none`.
  // Op2's lda (= op1_ldc) per Op1 layout × activation:
  //   wide  + gated act  — lda=N,    K_down=N/2.
  //   wide  + act=none   — lda=N,    K_down=N.
  //   tight + gated act  — lda=N/2,  K_down=N/2.
  group_matmul_run_parallel_dispatch(
      layout, scratch.transA_down, transB, M, fused.N_down,
      scratch.K_down, scratch.alpha_down,
      scratch.src_down, op1_ldc,
      fused.down_weight, fused.ldb_down,
      fused.bias_down, scratch.beta_down,
      op2_dst, op2_ldc,
      is_weights_const, scratch.params_down, num_threads, &pass2_mode,
      grp_matmul_gated_act_t::none, act_dtype);
  return status_t::success;
}

// ═══════════════════════════════════════════════════════════════════════
// gemm_mode composition.
// ═══════════════════════════════════════════════════════════════════════
//
// Composes a single string describing which fused-MoE path ran for
// profiler / benchdnn / apilog readers.  Top-level tag distinguishes
// `fused_moe_vertical` (single fused executor) from `fused_moe_2pass`
// (Pass 1 + sep-act + Pass 2); intalloc and tight tags reflect which
// side(s) the library managed; the trailing `(op1=…,op2=…)` reveals
// the underlying executor identifiers reported by each pass.
//
// Returns a `const char *` whose lifetime is tied to a thread-local
// `std::string` — valid until the next call to this function on the
// same thread.
inline const char *compose_fused_moe_gemm_mode(
    bool vertical_fusion_engaged,
    bool op1_internal,
    bool op2_internal,
    bool want_tight,
    const char *pass1_mode,
    const char *pass2_mode,
    bool has_postop) {
  static thread_local std::string mode_buf;
  mode_buf.clear();
  mode_buf.reserve(64);
  mode_buf.append(vertical_fusion_engaged
                      ? "fused_moe_vertical"
                      : "fused_moe_2pass");
  if (op1_internal && op2_internal) mode_buf.append("_intalloc");
  else if (op1_internal)            mode_buf.append("_intalloc_op1");
  else if (op2_internal)            mode_buf.append("_intalloc_op2");
  if (want_tight)                   mode_buf.append("_tight");
  mode_buf.append("(op1=");
  mode_buf.append(pass1_mode != nullptr ? pass1_mode : "?");
  mode_buf.append(",op2=");
  mode_buf.append(pass2_mode != nullptr ? pass2_mode : "?");
  mode_buf.append(")");
  if (has_postop) mode_buf.append("+postop");
  return mode_buf.c_str();
}

} // namespace (end file-private helpers)

// ═══════════════════════════════════════════════════════════════════════
// Primary entry: Op1+Act → Op2 (→ optional weighted reduce post-op)
// ═══════════════════════════════════════════════════════════════════════
//
// Orchestrator only — every step is a helper above or a backend
// executor in a sibling translation unit.  Read top-to-bottom to see
// the fused-MoE pipeline flow.

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
  if (num_ops == 0) return status_t::failure;

  // ── Step 1: detect per-side internal-alloc state ───────────────────
  // Each side is detected independently so callers can mix any of the
  // four (op1_internal, op2_internal) combinations.  Mixed null/non-
  // null active range on either side is rejected up front by the
  // detector (the per-side internal flag means "all-null active range").
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
  if (run_detect(dst, "dst", &op1_internal) != status_t::success)
    return status_t::failure;
  if (run_detect(fused.dst_down, "fused.dst_down", &op2_internal)
      != status_t::success)
    return status_t::failure;

  // ── Step 2: validate inputs ────────────────────────────────────────
  const size_t dst_elem_internal =
      op1_internal ? size_of(params[0].dtypes.dst) : 0;
  int64_t total_M = 0;
  size_t total_bytes_internal = 0;
  {
    const status_t v = validate_fused_moe_inputs(
        fused, act, act_dtype, layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc, is_weights_const,
        params, moe_postop, op1_internal, op2_internal,
        dst_elem_internal, &total_M, &total_bytes_internal);
    if (v != status_t::success) return v;
  }

  // No active work (every expert has M=0): return success without
  // spawning OMP regions or touching the Op2 dispatch.
  if (total_M == 0) {
    if (gemm_mode_out) *gemm_mode_out = "fused_moe_skip";
    return status_t::success;
  }

  // ── Step 3: pick wide-vs-tight Op1 arena layout ────────────────────
  const int env_algo_fused = get_grp_matmul_algo();
  const bool custom_kernel_en = get_grp_matmul_custom_kernel();
  const bool want_tight = pick_fused_moe_want_tight(
      op1_internal, act, env_algo_fused,
      layout, M, N, K, params, num_threads);

  // EXEC APILOG — one line per fused_moe call summarising arena
  // layout, per-side internal-alloc state, act-fusion choice, and
  // the W13 write width.  apilog_info_enabled() is cached after the
  // first call so the gate check is free when logging is off.
  static const bool s_apilog = apilog_info_enabled();
  if (s_apilog) {
    const int log_fused_moe_tight = get_grp_matmul_fused_moe_tight();
    const bool act_is_gated = (act != grp_matmul_gated_act_t::none);
    const char *w13_write_elems = act_is_gated
                                      ? (want_tight ? "I" : "2I")
                                      : "N";
    apilog_info("[GRP_MATMUL.EXEC] op=fused_moe arena=",
                (want_tight ? "tight" : "loose"),
                " op1_internal=", (op1_internal ? "yes" : "no"),
                " op2_internal=", (op2_internal ? "yes" : "no"),
                " act=", act_name(act),
                " act_in_register=",
                ((want_tight && act_is_gated) ? "yes" : "no"),
                " W13_write_elems_per_row=", w13_write_elems,
                " op2_dst_reuse=",
                (op2_internal ? "src_inplace" : "caller_dst_down"),
                " env_algo=", env_algo_fused,
                " env_tight=", log_fused_moe_tight,
                " custom_kernel_env=",
                (custom_kernel_en ? "on" : "off"),
                " num_ops=", (int)num_ops);
  }

  // ── Step 4: Op1 arena + per-expert pointer / stride setup ──────────
  // Thread-local scratch surfaces now live in file-scope accessors so
  // `clear_fused_moe_scratch()` can reach them via an OMP team sweep.
  // Functional behaviour is identical to a function-local static (one
  // instance per thread, persistent for the thread's lifetime); the
  // indirection cost is zero after the first call on a given thread
  // (returns by reference to a static thread_local).
  FusedMoEArena   &arena   = get_thread_local_arena();
  FusedMoEScratch &scratch = get_thread_local_scratch();
  const std::vector<void *> *op1_dst_p = nullptr;
  const std::vector<int>    *op1_ldc_p = nullptr;
  {
    const status_t s = setup_op1_arena_and_layout(
        arena, scratch, op1_internal, want_tight,
        total_bytes_internal, dst_elem_internal,
        N, M, ldc, dst, op1_dst_p, op1_ldc_p);
    if (s != status_t::success) return s;
  }
  const std::vector<void *> &op1_dst = *op1_dst_p;
  const std::vector<int>    &op1_ldc = *op1_ldc_p;

  // ── Step 5: Op2 dispatch scratch population ────────────────────────
  {
    // Pass `num_ops` (= M.size() = ACTIVE matmul count) explicitly so
    // the setup loop is bounded by the active range, not by the
    // framework's prepack-extras-tail `params.size()`.  See the
    // doc-block on setup_op2_dispatch_scratch() for the active/total
    // contract.
    const status_t s = setup_op2_dispatch_scratch(
        scratch, fused, act, num_ops,
        N, src, lda, params, op1_dst, op2_internal);
    if (s != status_t::success) return s;
  }
  const std::vector<void *> &op2_dst =
      op2_internal ? scratch.op2_dst_internal : fused.dst_down;
  const std::vector<int> &op2_ldc =
      op2_internal ? lda : fused.ldc_down;

  // ── Step 6: per-path dispatch fork ─────────────────────────────────
  // Try vertical fusion FIRST.  The eligibility gate inside
  // `try_flat_m_tile_pipeline_bf16` (defined in m_tile.cpp) checks
  // env knob, dtype regime on both passes (BF16 end-to-end OR
  // WOQ-INT4 s4/u4 weights OR DQ-INT8 per-token-symmetric on s8
  // weights), supported activation set, and `check_m_tile_safe` on
  // Op1 + synthesized Op2.  When it returns `false` NO writes have
  // been made to op1_dst / op2_dst, so the legacy two-pass below
  // overwrites cleanly.
  //
  // The three regimes share the SAME executor — see the doc-block
  // on `flat_m_tile_pipeline_bf16` in `group_matmul_m_tile.cpp`
  // for the per-regime memory-management notes (DQ-INT8 adds two
  // RAII-owned `std::vector<reorder_quant_buffers_t>` allocations
  // on the dispatcher stack: per-expert Op1 src hoist + per-thread
  // Stage 2b re-quant scratch; both freed deterministically when
  // the executor returns).
  //
  // Pre-dispatch apilog tags emitted on EACH entry so a crash inside
  // either executor surfaces in the log immediately before the fault
  // (the gemm_mode composition at Step 8 only runs on successful
  // completion).  Lets triage tell VF-vs-legacy without re-running
  // under gdb / ASAN.
  if (s_apilog) {
    apilog_info("[GRP_MATMUL.EXEC] op=fused_moe enter=vertical_fusion_attempt");
  }
  const char *pass1_mode = nullptr;
  const char *pass2_mode = nullptr;
  bool vertical_fusion_engaged = try_flat_m_tile_pipeline_bf16(
      layout, transA, scratch.transA_down, transB,
      M, N, K, alpha,
      src, lda, weight, ldb, bias, beta,
      op1_dst, op1_ldc, /*dst_w13_is_caller_alloc=*/!op1_internal,
      fused.N_down, scratch.K_down, scratch.alpha_down,
      fused.down_weight, fused.ldb_down,
      fused.bias_down, scratch.beta_down,
      op2_dst, op2_ldc,
      act, act_dtype,
      is_weights_const, params, scratch.params_down, num_threads);
  if (vertical_fusion_engaged) {
    if (s_apilog) {
      apilog_info("[GRP_MATMUL.EXEC] op=fused_moe exit=vertical_fusion_ok");
    }
    // Differentiate BF16 end-to-end / WOQ-INT4 / DQ-INT8 in the
    // profiler / apilog so per-route timings can be partitioned
    // downstream.  The eligibility wrapper guarantees both halves
    // share the same regime, so a single probe of
    // `params[0].dtypes.wei` (with `dynamic_quant` to distinguish
    // DQ-INT8 from a hypothetical static-INT8 placeholder) suffices.
    const data_type_t wei0 =
        (!params.empty()) ? params[0].dtypes.wei : data_type_t::none;
    const bool is_woq_wei =
        (wei0 == data_type_t::s4 || wei0 == data_type_t::u4);
    const bool is_dqint8_wei =
        (wei0 == data_type_t::s8)
        && (!params.empty()) && params[0].dynamic_quant;
    if (is_dqint8_wei)        pass1_mode = "vertical_fusion_dqint8";
    else if (is_woq_wei)      pass1_mode = "vertical_fusion_woq";
    else                      pass1_mode = "vertical_fusion_bf16";
    pass2_mode = pass1_mode;
  } else {
    if (s_apilog) {
      apilog_info("[GRP_MATMUL.EXEC] op=fused_moe enter=legacy_two_pass");
    }
    const status_t s = run_fused_moe_legacy_two_pass(
        act, act_dtype, layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta,
        op1_dst, op1_ldc, fused, scratch,
        op2_dst, op2_ldc, is_weights_const, params, num_threads,
        pass1_mode, pass2_mode);
    if (s != status_t::success) return s;
    if (s_apilog) {
      apilog_info("[GRP_MATMUL.EXEC] op=fused_moe exit=legacy_two_pass_ok");
    }
  }

  // ── Step 7: optional MoE post-op (weighted reduce) ─────────────────
  // The post-op is the natural "Stage 4" of the fused MoE pipeline
  // (Op1 → activation → Op2 → weighted reduce).  D = fused.N_down[0]
  // — the validator already confirmed N_down is uniform across
  // experts when moe_postop is engaged.
  if (moe_postop != nullptr) {
    const int D_down = fused.N_down[0];
    const status_t postop_st = group_matmul_moe_postop_execute(
        moe_postop, D_down, num_threads, params[0].dtypes.dst);
    if (postop_st != status_t::success) return postop_st;
  }

  // ── Step 8: compose gemm_mode for profiler / apilog ────────────────
  if (gemm_mode_out != nullptr) {
    *gemm_mode_out = compose_fused_moe_gemm_mode(
        vertical_fusion_engaged, op1_internal, op2_internal,
        want_tight, pass1_mode, pass2_mode,
        /*has_postop=*/moe_postop != nullptr);
  }
  return status_t::success;
}

// ═══════════════════════════════════════════════════════════════════════
// Legacy ABI-preserving overload (no moe_postop parameter).  Forwards
// to the primary entry with moe_postop = nullptr.  Kept as a separate
// non-inline exported symbol so binaries built against the pre-postop
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

// ═══════════════════════════════════════════════════════════════════════
// Public scratch-release API.
// ═══════════════════════════════════════════════════════════════════════
//
// See doc-block on the declaration in `group_matmul_direct.hpp` for
// semantics + limitations.  Implementation orchestrates an OMP
// parallel region so each worker in the current OMP pool calls
// `reset_thread_local_fused_moe_state()` against its OWN TLS — there
// is no shared-state path that one thread can use to reach another
// thread's `thread_local` instance.
//
// The team size is the OMP runtime's current `max_threads` — the
// natural sweep granularity.  If the host application configured a
// smaller team via `omp_set_num_threads(n)`, only those `n` workers
// will be touched; threads outside the active OMP pool retain their
// scratch until process exit (which is the expected POSIX TLS
// behaviour).
//
// SAFETY: caller MUST NOT be inside an OMP parallel region.  We
// detect that via `omp_in_parallel()` and silently no-op in that
// case (calling `omp parallel` from inside another would either
// nest or serialise, depending on `OMP_NESTED`; neither is the
// intent of this API).
void clear_fused_moe_scratch() {
  if (omp_in_parallel()) return;
  #pragma omp parallel
  {
    reset_thread_local_fused_moe_state();
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
