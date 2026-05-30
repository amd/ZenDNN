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

/// ALGO 3 — N-tile parallel GEMM for grouped expert matmul, with
/// optional fused swiglu_oai epilogue (see
/// group_matmul_parallel_common.hpp).
///
/// Self-contained translation unit.  Exposes `flat_n_tile` to the
/// dispatcher via the common header.  All other helpers are private.
///
/// File layout
/// ───────────
///   Section A  `GroupNTileContext` and its two per-thread tile
///              primitives (`do_tile`, `apply_swiglu_oai`).  The
///              planner-shaped POD types (Strategy enum, Topology,
///              Plan, RoundCandidates, RoundPick, PerThreadScratch)
///              live in the companion header `group_matmul_n_tile.hpp`
///              so planner-only gtests can include them without
///              dragging in this TU's OMP executors.
///
///   Section B  Planner — pure decision layer.  Inspects the inputs
///              (M/N/K, num_ops, num_threads, dtype, fused-act) and
///              picks one of the strategies plus its block / tile /
///              thread parameters.  No side effects, no OMP.
///
///   Section C  Strategy executors — one function per parallel pattern
///              that consumes a `GroupNTilePlan` produced by Section B
///              and a `GroupNTileContext` and runs the actual matmul
///              calls.
///
///   Section D  Public entry point `flat_n_tile()` — thin orchestrator
///              that builds context, calls the planner, then dispatches
///              to the matching strategy executor.  `gemm_mode_label`
///              (anonymous-namespace helper just above) renders the
///              static label string emitted via `gemm_mode_out`.
///
/// Adding new tuning (e.g. INT8, larger MoE shapes, new dtype):
///   * pick new tile sizes / thread budgets only in Section B;
///   * if a new parallel pattern is needed, add a strategy + its
///     executor in Section C and route to it from the planner;
///   * Section A primitives (do_tile / apply_swiglu_oai) stay reusable.

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../group_matmul_parallel_common.hpp"
#include "group_matmul_n_tile.hpp"            // (re-includes planner header)
#include "../custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "../prepack/prepack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using zendnnl::common::bfloat16_t;

namespace {

// =====================================================================
// Column-slice helper for per-channel weight quantization metadata
// =====================================================================
//
// N-tile slices columns of B (and dst), so per-channel weight scales
// / zero-points need to be re-anchored to each thread's column range
// `[col_start, col_start + n_tile)` before the kernel call.
//
// Layouts handled here:
//
//   * Per-tensor (`dims = {}`, `{1}`, or any shape whose product is
//     1) — single value applies to all columns; no-op.
//
//   * Per-channel (`dims = {N}` or `{1, N}`) — one value per output
//     column, laid out contiguously.  The slice is a contiguous
//     `n_tile`-long sub-array, so advance `buff` by
//     `col_start * size_of(dt)` and rewrite the trailing dim to
//     `n_tile`.  The backend (e.g. `cvt_4bit_to_bf16` in
//     `aocl_kernel.cpp`) detects per-channel via `qsize == N`, so
//     once we present `dims_back == n_tile` and the sliced buffer
//     the kernel's `compute_quant_offset` returns `col`
//     (`∈ [0, n_tile)`) and indexes the sliced buffer correctly.
//
// Per-group on K (`{G, N}` with `G > 1`) is NOT handled here —
// `check_n_tile_extra` rejects it from the per-token-dynamic scope
// the gate currently allows.  Supporting per-group would require a
// per-thread `G × n_tile` repack scratch (because the column slice
// is `G` non-contiguous strips in the original buffer); the
// machinery is intentionally absent from this helper until a
// deployment needs it.
//
// Other shapes silently no-op (defensive — upstream gate already
// filters them out).
//
// The helper takes a non-const reference because the call site
// operates on a `thread_local` copy of `params[e]`, never the
// caller's shared `params[]` vector — so the in-place mutation is
// safe.
inline void offset_quant_by_col(
    matmul_quantization_params_t::matmul_quant_t &q,
    int col_start, int n_tile) {
  if (q.buff == nullptr || q.dims.empty()) return;
  int64_t nelems = 1;
  for (auto d : q.dims) {
    if (d <= 0) return;
    nelems *= d;
  }
  if (nelems <= 1) return;  // Per-tensor — no slicing.

  // Per-channel detection: rank-1 `{N}` or rank-2 `{1, N}`.  The
  // back-dim carries the column count in both cases.  Any other
  // shape is left untouched — `check_n_tile_extra` is the
  // authoritative gate for what reaches this helper.
  const auto &dims = q.dims;
  const bool per_channel_1d = (dims.size() == 1);
  const bool per_channel_2d = (dims.size() == 2 && dims[0] == 1);
  if (!(per_channel_1d || per_channel_2d)) return;

  const size_t elem = size_of(q.dt);
  if (elem == 0) return;

  q.buff = static_cast<const uint8_t *>(q.buff)
      + static_cast<size_t>(col_start) * elem;
  q.dims.back() = static_cast<int64_t>(n_tile);
}

// =====================================================================
// Section A — Data structures
// =====================================================================
//
// The planner-shaped types (`PerThreadScratch`, `grow_scratch`,
// `GroupNTileStrategy`, `GroupNTileTopology`, `GroupNTilePlan`) live
// in `group_matmul_n_tile.hpp` so future planner-only gtests can
// include them without dragging in this TU's executors.  Only
// `GroupNTileContext` stays here because it captures the caller's
// `std::vector<…> &` inputs by reference (impl-only, owned by one
// `flat_n_tile()` invocation).

// Hoisted source-side dynamic-quant state per expert.  Populated by
// `flat_n_tile`'s pre-OMP hoist loop for every expert that has
// `params[e].dynamic_quant == true`: a single-shot
// `reorder_quantization_wrapper` runs ahead of the parallel region
// (with the full thread team driving the internal reorder kernel)
// and the resulting S8 source + per-token / per-group scale buffer
// are then SHARED — read-only — by every per-tile thread inside the
// OMP region.  Without hoisting, each N-tile thread would
// independently allocate its own M × K S8 scratch and either race
// on the caller's scale buffer (when caller pre-allocated it) or
// duplicate the (M, K) reorder `num_threads` times per call.
//
// Lifetime: the backing `reorder_quant_buffers_t` RAII vector lives
// on `flat_n_tile`'s stack and outlives the OMP region; the OMP
// region only reads from those buffers.  All `do_tile()` does is
// pull the substitute `src_ptr` / `lda` / dtype / scale-and-zp out
// of the slot and overwrite its thread-local `tile_params` so the
// wrapper inside `execute_expert_slice` sees `dtypes.src == s8` and
// short-circuits (its `eligible` check requires src dtype to still
// be bf16 / f32 — the hoist has already rewritten it).
struct HoistedSrcQuant {
  const void *src_ptr = nullptr;
  int lda = 0;
  data_type_t src_dtype = data_type_t::none;
  matmul_quantization_params_t::matmul_quant_t src_scale;
  matmul_quantization_params_t::matmul_quant_t src_zp;
  bool valid = false;
};

// Bundle every reference / dtype-size the executors and per-thread
// tile primitives need.  Cuts the per-call argument list down to
// `(plan, ctx, e, local_tid, team_size, min_n_tile)`.  The two
// methods below are the unified per-thread primitives:
//
//   do_tile()           — one expert × one N-slice matmul.
//   apply_swiglu_oai()  — per-thread fused swiglu_oai epilogue
//                         (rows split, not columns; see method
//                         body for the correctness argument).
struct GroupNTileContext {
  const std::vector<char> &layout;
  const std::vector<bool> &transA;
  const std::vector<bool> &transB;
  const std::vector<int> &M;
  const std::vector<int> &N;
  const std::vector<int> &K;
  const std::vector<float> &alpha;
  const std::vector<const void *> &src;
  const std::vector<int> &lda;
  const std::vector<const void *> &weight;
  const std::vector<int> &ldb;
  const std::vector<const void *> &bias;
  const std::vector<float> &beta;
  const std::vector<void *> &dst;
  const std::vector<int> &ldc;
  const std::vector<bool> &is_weights_const;
  std::vector<matmul_params> &params;

  grp_matmul_gated_act_t fused_act;
  data_type_t act_dtype;

  size_t wei_elem;
  size_t dst_elem;
  size_t bias_elem;

  // Custom BF16 microkernel hook.  `use_custom` is the sticky decision
  // taken at `flat_n_tile` entry (single-threaded) — non-null + enabled
  // means every `do_tile()` dispatches through
  // `custom_kernel::dispatch_tile()` instead of `execute_expert_slice()`.
  // The custom path writes to the same destination at the same ldc
  // contract (wide OR tight — the microkernel honors the caller's
  // ldc), so the surrounding planner / swiglu epilogue / post-op
  // machinery is unchanged.
  //
  // Scope in non-fused flat_n_tile: act=none only.  When the fused
  // swiglu epilogue is also active (Op1 of fused MoE with custom
  // kernel) the microkernel fuses the activation in-register and
  // writes I cols directly at the caller's ldc — the per-thread
  // scratch + OOP path below is skipped.
  //
  // Caveat: only `do_tile()`-based executors (DecodeD, FewExperts,
  // ManyExperts) route through the custom kernel.  The Sequential
  // executor (picked when N is too small to usefully split across
  // threads) calls `execute_expert_slice` directly and ignores this
  // hook.  In practice that's a non-issue because the shapes routing
  // to Sequential also tend to fail the custom kernel's
  // `N % pack_nr == 0` contract, so `use_custom` is already false
  // there.  Left as-is to keep Sequential a pure BLAS delegation.
  bool use_custom = false;
  const custom_kernel::CallContext *kctx = nullptr;

  // Alloc-fail flag set by the tight-fused-epilogue branch of
  // `do_tile()` when a per-thread scratch grow fails.  Checked after
  // the OMP region exits so the failure propagates to the caller as
  // an error instead of silently producing wrong output.  Pointer
  // (not owned) so the ctx struct itself stays copyable-by-reference
  // across the parallel region.
  std::atomic<int> *alloc_fail = nullptr;

  // Per-expert hoisted source dynamic-quant state.  Non-null when
  // `flat_n_tile` ran the pre-OMP hoist loop (i.e. at least one
  // active expert has `params[e].dynamic_quant == true`); each
  // active expert's slot is populated with `valid = true` if the
  // wrapper successfully replaced its bf16/f32 src with an S8 buffer
  // and computed/captured the corresponding scale (and zp) buffer.
  // `do_tile()` and `execute_sequential()` check the per-expert
  // `valid` flag and, when set, substitute the hoisted state into
  // their `tile_params` / src pointer / lda before calling
  // `execute_expert_slice` — see the struct's doc-block above for
  // the full lifetime + correctness argument.  Pointer (not owned)
  // because the backing vector lives on `flat_n_tile`'s stack; the
  // OMP region only reads it.
  const std::vector<HoistedSrcQuant> *hoisted_src_quant = nullptr;

  // Returns the number of threads that share the work for expert `e`
  // in a team of `team_size`.  Used by `do_tile()` (column split for
  // matmul) and `apply_swiglu_oai()` (row split for the in-place
  // swiglu epilogue).
  //
  // INVARIANT: do_tile() and apply_swiglu_oai() MUST agree on the
  // returned value so every column written has a row-reader and
  // vice versa.  Centralising it here makes that automatic.
  //
  // Two branches, picked by the planner-set state below:
  //
  //   AOCL strict-stable branch
  //     Gate:    `!use_custom && stable[e] > 0`.
  //     Returns: `stable[e]` clamped to
  //              `[1, min(team_size, N[e] / nr_align)]`.
  //     Why:     `stable[e]` comes from `aocl_stable_n_thr(num_threads)`,
  //              which is num_threads-only and shape-independent —
  //              the planner also forces `team_size == stable[e]` for
  //              every expert, so the AOCL reorder cache key
  //              `(col_start, n_tile)` is byte-identical across
  //              calls.  Under the strict-stable plan all three
  //              `min({...})` operands equal `stable[e]` and the
  //              clamp is a no-op; it is left in as defence-in-depth
  //              so a future planner regression degrades gracefully
  //              to dynamic-tile behaviour (some cache thrash, no
  //              corruption) rather than reopening a silent-
  //              miscompute window.
  //
  //   Dynamic-tile branch (default)
  //     Gate:    anything not matching above — `use_custom` (any
  //              CK plan, including Phase B's remainder-distribute
  //              where the executor passes `n_thr_e` as team_size),
  //              OR `!use_custom` with `stable[e] == 0` (legacy
  //              non-strict AOCL, env opt-out).
  //     Returns: `max(1, min(team_size, N[e] / min_n_tile))`.
  //     Why:     CK's pack cache is shape-keyed (full-N pack per
  //              expert), and legacy non-strict AOCL accepts cache
  //              thrash by design — neither needs `(col_start,
  //              n_tile)` to stay byte-identical across calls.
  //
  // See the doc-block in group_matmul_parallel_common.hpp for the
  // cache-stability contract that motivates the strict-stable branch.
  inline int participating_n_thr(const GroupNTilePlan &plan,
                                 int e, int team_size,
                                 int min_n_tile) const {
    // Defence-in-depth bounds check: `stable_n_thr_per_expert` is a
    // stack-resident `std::array<int16_t, kMaxExperts=256>`.  The
    // strict-stable planner only populates indices in
    // `[0, min(num_ops, kMaxExperts))`, and `plan_group_n_tile`
    // additionally routes any caller with `num_ops > kMaxExperts` to
    // Sequential — so this branch should never see `e >= kMaxExperts`
    // in production.  We still guard the read here so a future
    // regression in the upstream gate cannot reopen an OOB-read
    // window: experts past `kMaxExperts` silently fall through to the
    // dynamic-tile branch instead of touching invalid memory.
    if (!use_custom
        && e >= 0
        && e < GroupNTilePlan::kMaxExperts
        && plan.stable_n_thr_per_expert[e] > 0) {
      const int nr_align_safe = std::max(1, plan.nr_align);
      const int align_cap     = std::max(1, N[e] / nr_align_safe);
      const int clamped       = std::min({
          static_cast<int>(plan.stable_n_thr_per_expert[e]),
          align_cap,
          team_size});
      return std::max(1, clamped);
    }
    return std::max(1, std::min(team_size, N[e] / min_n_tile));
  }

  // Per-thread N-slice of expert e's matmul.  The column split is
  // NR-aligned (see aligned_n_split() in the common header).
  // Body lives out-of-line just below the struct.
  inline void do_tile(const GroupNTilePlan &plan,
                      int e, int local_tid, int team_size,
                      int min_n_tile) const;

  // Per-thread fused swiglu_oai epilogue.  Splits by rows so reads /
  // writes stay on a thread's own row slice and never alias another
  // thread's write — see the body for the full correctness argument.
  // Body lives out-of-line just below the struct.
  inline void apply_swiglu_oai(const GroupNTilePlan &plan,
                               int e, int local_tid, int team_size,
                               int min_n_tile) const;
};

// ---------------------------------------------------------------------
// GroupNTileContext::do_tile
// ---------------------------------------------------------------------
//
// Per-thread N-slice of expert e's matmul.  The column split is
// NR-aligned (see aligned_n_split() in the common header).
//
// Note for the fused swiglu_oai path: the matmul split is NOT
// forced onto even / pair boundaries.  Pair alignment used to be a
// correctness constraint when the epilogue was column-parallel
// (both halves of each (g, u) pair had to be produced by the same
// thread).  The current epilogue (`apply_swiglu_oai`, also out-of-line
// below) splits by rows instead, so a thread reads pair
// (col 2k, col 2k+1) from any thread's matmul writes within its own
// row range — both reads are guaranteed visible by the OMP barrier
// between matmul and activation in the executors, regardless of which
// thread wrote each column.  Every dst cell is still produced by
// exactly one thread, so per-cell numerics are unchanged.
inline void GroupNTileContext::do_tile(const GroupNTilePlan &plan,
                                       int e, int local_tid, int team_size,
                                       int min_n_tile) const {
  if (M[e] <= 0) return;
  const int n_thr = participating_n_thr(plan, e, team_size, min_n_tile);
  // Coverage trip-wire: n_thr > team_size would mean aligned_n_split
  // produces more slots than the executor has threads, leaving the
  // surplus slots' dst columns uncomputed (silent corruption).
  assert(n_thr <= team_size
         && "do_tile: n_thr > team_size; aligned_n_split would "
            "leave dst cols uncomputed");
  if (local_tid >= n_thr) return;

  const auto split =
      aligned_n_split(N[e], n_thr, local_tid, plan.nr_align);
  const int col_start = split.first;
  const int col_end   = split.second;
  const int n_tile = col_end - col_start;
  if (n_tile <= 0) return;

  // ── Custom BF16 microkernel fast path ─────────────────────────────
  // When enabled at flat_n_tile entry the kernel replaces the
  // standard execute_expert_slice for this tile.  Writes to the
  // caller's [M, N] destination at the caller's ldc (wide or tight —
  // the microkernel honors any ldc); when fused_epilogue=swiglu the
  // activation is fused in-register and the kernel writes I cols at
  // the tight ldc.  Either way, no per-thread scratch is needed.
  if (use_custom && kctx != nullptr) {
    // bias[e] is passed as `const void *` — the dispatcher resolves
    // the dtype via `kctx->bias_kind` (bf16 or fp32) and branches
    // the load path inside the ukernel.
    custom_kernel::dispatch_tile(
        *kctx, e,
        M[e], K[e], n_tile, col_start,
        src[e], lda[e],
        bias[e],
        static_cast<bfloat16_t *>(dst[e]), ldc[e]);
    return;
  }

  // Weight: slice columns of op(B).  Shared by both the wide path
  // below and the tight-scratch path further down.
  const size_t wei_off = transB[e]
      ? static_cast<size_t>(col_start) * ldb[e] * wei_elem
      : static_cast<size_t>(col_start) * wei_elem;
  const auto *w = static_cast<const char *>(weight[e]) + wei_off;

  // bias: offset by col_start if present.  Same slicing rules for
  // the wide and tight paths (bias is a [N]-wide row).
  const void *b = nullptr;
  if (bias[e] != nullptr)
    b = static_cast<const char *>(bias[e])
        + static_cast<size_t>(col_start) * bias_elem;

  static thread_local matmul_params tile_params;
  tile_params = params[e];

  // ── Hoisted dynamic-quant source substitution ──────────────────────
  // When `flat_n_tile` ran the pre-OMP hoist loop for this expert,
  // swap the bf16/f32 caller src for the shared S8 reorder result
  // and rewrite this thread's `tile_params` so the wrapper inside
  // `execute_expert_slice` sees `dtypes.src == s8` and short-circuits
  // (otherwise every N-tile thread would re-run the wrapper on the
  // full (M, K) source — racing on the caller-shared scale buffer
  // and duplicating the reorder work `num_threads` times per call).
  // `src_for_tile` / `lda_for_tile` default to the caller's vectors
  // when no hoist happened (legacy bf16 / static-quant / WOQ paths).
  const void *src_for_tile = src[e];
  int lda_for_tile = lda[e];
  if (hoisted_src_quant != nullptr
      && static_cast<size_t>(e) < hoisted_src_quant->size()
      && (*hoisted_src_quant)[e].valid) {
    const auto &h = (*hoisted_src_quant)[e];
    src_for_tile = h.src_ptr;
    lda_for_tile = h.lda;
    tile_params.dtypes.src = h.src_dtype;
    tile_params.quant_params.src_scale = h.src_scale;
    tile_params.quant_params.src_zp = h.src_zp;
  }

  // ── Column-slice per-channel weight quantization metadata ───────────
  // N-tile slices columns of B; per-channel weight scales /
  // zero-points (`{N}` or `{1, N}`) need their pointer advanced by
  // `col_start × elem_size` and the trailing dim rewritten to
  // `n_tile`.  Per-tensor scales are a no-op.  Per-group `{G, N}`
  // wei is rejected by `check_n_tile_extra` (the current scope only
  // admits dynamic INT8 per-token + per-channel wei).
  //
  // Source-side scales reach the kernel in one of two forms — both
  // N-independent and untouched by this slicer:
  //   (a) the `{M, 1}` dims left over from the caller when static
  //       src quant was supplied (not currently in scope, but the
  //       slicer would no-op either way);
  //   (b) the hoisted dynamic-quant result substituted above (the
  //       `HoistedSrcQuant` block) — also N-independent.
  offset_quant_by_col(tile_params.quant_params.wei_scale,
                      col_start, n_tile);
  offset_quant_by_col(tile_params.quant_params.wei_zp,
                      col_start, n_tile);

  // ── Tight-fused-epilogue path (non-custom + swiglu + tight dst) ────
  // Caller's dst is a tight [M, I]-layout buffer (ldc < N).  The
  // classic matmul-then-in-place-compact pattern can't run here (no
  // room for 2I cols in dst).  Switch to per-thread-scratch + OOP
  // activation:
  //
  //   1. matmul the thread's N-tile slice into a thread-local
  //      scratch buffer (stride = n_tile, holds wide 2I cols for
  //      this column range only),
  //   2. read pairs from scratch and write `n_tile/2` activated
  //      cols into caller's tight dst at col offset `col_start/2`.
  //
  // Barrier-free: each thread's scratch is private, and its writes
  // to caller's dst land on disjoint column ranges across threads
  // (scratch cols [col_start, col_end) → dst cols
  // [col_start/2, col_end/2)).  `execute_rounds` / `execute_decode_d`
  // skip the post-matmul barrier + `apply_swiglu_oai()` pass when
  // `plan.tight_fused_epilogue` is true.
  if (plan.tight_fused_epilogue) {
    // STRUCTURAL CONSTRAINT — do_tile's per-thread tight branch is
    // swiglu-only.  The split-halves gated acts (silu_and_mul,
    // gelu_and_mul) cannot use this code path: per-thread scratch
    // covers `[col_start, col_start + n_tile)` of the LOGICAL output,
    // and for split-halves that range is EITHER in the gate half
    // [0, I) OR in the up half [I, N) — never both — so the OOP
    // pair-pack helper would pair gate-with-gate (wrong) or
    // up-with-up (wrong).  flat_n_tile's entry routes silu/gelu +
    // tight + CK-refused to the Sequential strategy (which has a
    // per-expert wide scratch + `apply_gated_act_inplace` + memcpy
    // fallback path) before this code is reached.  The assertion
    // catches any future code change that bypasses that routing.
    assert(fused_act == grp_matmul_gated_act_t::swiglu_oai_mul
           && "do_tile's tight branch is swiglu-only; split-halves "
              "gated acts (silu_and_mul, gelu_and_mul) MUST route "
              "through the Sequential fallback in flat_n_tile when "
              "CK does not engage.  See the "
              "`TIGHT_SPLIT_HALVES_FALLBACK` block in flat_n_tile.");
    assert((n_tile % 2) == 0
           && "tight_fused_epilogue requires even n_tile (pair-aligned)");

    static thread_local PerThreadScratch scratch;
    const size_t need_bytes =
        static_cast<size_t>(M[e]) * n_tile * dst_elem;
    if (!grow_scratch(scratch, need_bytes)) {
      if (alloc_fail) alloc_fail->store(1, std::memory_order_relaxed);
      return;
    }

    // Matmul into scratch at stride n_tile.  `beta` is passed as-is
    // so the caller's beta contract (overwrite / accumulate) still
    // holds at the scratch granularity — but note that "accumulate"
    // semantics onto the tight dst are undefined in tight mode
    // because scratch starts empty each call; the internal-alloc
    // arena always passes beta=0 here.
    execute_expert_slice(layout[e], transA[e], transB[e],
        M[e], n_tile, K[e], alpha[e],
        src_for_tile, lda_for_tile, w, ldb[e],
        b, beta[e], scratch.buf, n_tile,
        is_weights_const[e], 1, tile_params, plan.algo);

    // OOP activation: compact 2I → I into caller's tight dst at the
    // matching column range.
    const int pairs = n_tile / 2;
    apply_swiglu_oai_tile_rows_oop(
        scratch.buf, /*src_ldc=*/n_tile, /*src_col_start=*/0,
        dst[e], /*dst_ldc=*/ldc[e], /*dst_col_start=*/col_start / 2,
        M[e], pairs, act_dtype);
    return;
  }

  // ── Wide path (caller's ldc ≥ N) ───────────────────────────────────
  // Classic matmul-into-caller's-dst; fused epilogue (if any) runs
  // later via `apply_swiglu_oai()` after an OMP barrier.

  // dst: column offset within each row (ldc unchanged).
  auto *d = static_cast<char *>(dst[e])
      + static_cast<size_t>(col_start) * dst_elem;

  // Per-channel weight quant (`{N}` / `{1, N}`) has already been
  // column-sliced into `tile_params` above; per-tensor wei quant and
  // M-indexed (`{M, *}`) src quant pass through unchanged because
  // they have no N-axis dependency.  Dynamic source quant has
  // already been hoisted into `src_for_tile` / `lda_for_tile` /
  // `tile_params` above — its wrapper short-circuits inside
  // `execute_expert_slice` (`dtypes.src` is now s8).  Per-group
  // weight quant (`{G, N}`) and post-ops with buffers
  // (binary_add/mul) remain blocked by `check_n_tile_extra` — they
  // would each require additional per-thread repack machinery not
  // present here.
  execute_expert_slice(layout[e], transA[e], transB[e],
      M[e], n_tile, K[e], alpha[e],
      src_for_tile, lda_for_tile, w, ldb[e],
      b, beta[e], d, ldc[e],
      is_weights_const[e], 1, tile_params, plan.algo);
}

// ---------------------------------------------------------------------
// GroupNTileContext::apply_swiglu_oai
// ---------------------------------------------------------------------
//
// Swiglu_oai per-thread fused epilogue.  The caller MUST place an
// OMP barrier between the matmul (`do_tile`) and this call so every
// thread's matmul writes are globally visible before any thread
// starts reading for activation.
//
// Correctness note — why we split M, not N:
//
//   The matmul split was column-wise (thread t owns cols
//   [2·p_start_t, 2·p_end_t)).  Splitting the epilogue the same way
//   would create a cross-thread write-after-read race on the
//   in-place compaction: thread t writes compact output cols
//   [p_start_t, p_end_t) while thread t' < t still needs to read
//   its own pair cols [2·p_start_{t'}, 2·p_start_t) — and the
//   t-write range starts at p_start_t < 2·p_start_t, so the two
//   ranges overlap.
//
//   Splitting by M instead makes every thread own a disjoint row
//   slice of the (M × N) output.  Reads and writes stay on that
//   thread's own rows, so no cross-thread aliasing is possible on
//   any column.  The in-place compaction within one row is still
//   safe because writing col n happens after reads at cols 2n, 2n+1
//   (both ≥ n) — see the in-place safety note in swiglu_oai_tile_*.
//
//   Cache-locality note: threads no longer re-use the exact column
//   range their matmul populated.  The activation is light element-
//   wise arithmetic, so any remaining hit goes through L3.
//
//   When M[e] < n_thr some threads get m_slice == 0 and no-op — the
//   outer omp parallel region barrier still lets them exit cleanly.
inline void GroupNTileContext::apply_swiglu_oai(const GroupNTilePlan &plan,
                                                int e, int local_tid,
                                                int team_size,
                                                int min_n_tile) const {
  if (M[e] <= 0) return;
  // Same `n_thr` as do_tile() — they MUST agree (see comment on
  // participating_n_thr()) so every matmul column has a row-reader.
  const int n_thr = participating_n_thr(plan, e, team_size, min_n_tile);
  assert(n_thr <= team_size
         && "apply_swiglu_oai: n_thr > team_size; row-split would "
            "leave dst rows un-activated");
  if (local_tid >= n_thr) return;

  // swiglu_oai requires even N (gate+up = 2 * intermediate_dim).
  // The dispatcher in group_matmul_dispatch.cpp enforces this for
  // the fused path; assert defensively to catch any future caller
  // that bypasses the dispatcher (otherwise we'd silently leave the
  // odd trailing column un-compacted in the activation output).
  assert(N[e] % 2 == 0
         && "apply_swiglu_oai: N must be even for swiglu_oai_mul");

  // Row split: this thread owns rows [m_start, m_end) of expert e's
  // (M × N) output and applies the full-width compaction in place.
  const int m_start = static_cast<int>(
      static_cast<int64_t>(M[e]) * local_tid / n_thr);
  const int m_end = static_cast<int>(
      static_cast<int64_t>(M[e]) * (local_tid + 1) / n_thr);
  const int m_slice = m_end - m_start;
  if (m_slice <= 0) return;

  char *row_base = static_cast<char *>(dst[e])
      + static_cast<size_t>(m_start) * ldc[e] * dst_elem;
  const int pairs = N[e] / 2;
  apply_swiglu_oai_tile_rows(row_base, m_slice, /*col_start=*/0, pairs,
                             ldc[e], act_dtype);
}

// =====================================================================
// Section B — Planner
// =====================================================================
//
// Pure decision layer for ALGO 3.  All tile / thread / batch choices
// for the supported dtype + shape regimes live here, so future dtype
// tuning (e.g. INT8) is a one-place change.

// Build the topology summary once at the top of flat_n_tile.  Single
// cache-friendly pass over M / N / K populates `max_M`, `max_N`,
// `max_K`, and `min_M_active`; `wei_per_expert = max_N * max_K * elem`
// is also precomputed so downstream helpers don't each re-derive it.
inline GroupNTileTopology summarise_topology(
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, int num_threads, size_t wei_elem) {
  GroupNTileTopology t{};
  t.num_ops = static_cast<int>(M.size());
  t.num_threads = num_threads;
  t.ccd_size = std::min(8, num_threads);
  // Ceiling so a partial last CCD (e.g., 126 threads → 16 CCDs, last
  // = 6 cores) still counts — keeps num_ccds consistent with
  // flat_m_tile's planner.
  t.num_ccds = std::max(1, (num_threads + t.ccd_size - 1) / t.ccd_size);

  // Seed running maxima with element 0; sentinel-based init for
  // `min_M_active` so an all-empty input falls through to the
  // `min_M_active = max_M` fallback after the loop.
  t.max_M = M[0];
  t.max_N = N[0];
  t.max_K = K[0];
  t.min_M_active = std::numeric_limits<int>::max();
  for (int i = 0; i < t.num_ops; ++i) {
    if (M[i] > t.max_M) t.max_M = M[i];
    if (N[i] > t.max_N) t.max_N = N[i];
    if (K[i] > t.max_K) t.max_K = K[i];
    if (M[i] > 0 && M[i] < t.min_M_active) t.min_M_active = M[i];
  }
  if (t.min_M_active == std::numeric_limits<int>::max())
    t.min_M_active = t.max_M;

  t.wei_elem = wei_elem;
  t.wei_per_expert = static_cast<size_t>(t.max_N)
                   * static_cast<size_t>(t.max_K) * wei_elem;
  return t;
}

// L3-aware batch budget: how many experts' weights (max_N × max_K)
// fit into the aggregate L3 of the running thread team.  Returns
// `topo.num_ops` when the weights themselves are zero-size (defensive
// fallback for degenerate inputs).
inline int compute_l3_batch(const GroupNTileTopology &topo) {
  if (topo.wei_per_expert == 0) return topo.num_ops;
  const size_t kL3Total = get_grp_l3_total_bytes(topo.num_ccds);
  return std::max(1,
      static_cast<int>(kL3Total / topo.wei_per_expert));
}

// Target batch_size for FewExperts / ManyExperts.  Defaults to the
// L3-aware `min(num_ops, l3_batch)`, but BUMPS up to the team-
// saturating value `num_threads / ccd_size` when:
//
//   1. L3 was binding `target` strictly below the team-saturating
//      value (`l3_batch < num_threads/ccd_size`), AND
//   2. We have at least that many experts (`num_threads/ccd_size <=
//      num_ops`), AND
//   3. The bumped working set is within a small overshoot of
//      aggregate L3 capacity (see `kL3OvershootDen` below).
//
// Check (3) is the safety net: shapes whose per-round working set
// would massively overflow L3 (very large weights) keep the strict
// L3-aware target and avoid victim-cache thrashing, while shapes
// whose per-round working set sits at or just over L3 take the bump
// because the small overshoot is cheaper than leaving threads idle.
inline int compute_target_batch(const GroupNTileTopology &topo,
                                int l3_batch) {
  // Allow a small overshoot above L3 before refusing the bump;
  // tuned to absorb sub-tile alignment slack without inviting
  // victim-cache thrashing on large-weight regimes.
  constexpr size_t kL3OvershootDen = 10;
  const int batch_team_saturating = topo.num_threads / topo.ccd_size;
  int target = std::min(topo.num_ops, std::max(1, l3_batch));
  if (l3_batch < batch_team_saturating
      && batch_team_saturating <= topo.num_ops) {
    const size_t bumped_weight =
        static_cast<size_t>(batch_team_saturating) * topo.wei_per_expert;
    const size_t kL3 = get_grp_l3_total_bytes(topo.num_ccds);
    if (bumped_weight <= kL3 + kL3 / kL3OvershootDen) {
      target = batch_team_saturating;
    }
  }
  return target;
}


// (`sort_indices_by_m` is defined in group_matmul_parallel_common.hpp
//  and is also used by the fused-MoE Op1 executor.)

// Thin wrapper around the generic `fill_ntile_expert_order()` helper
// (defined in group_matmul_parallel_common.hpp, shared with the
// fused-MoE Op1 executor).  Plumbs the result into the stack-
// resident fields on `GroupNTilePlan`: `expert_order[]`,
// `expert_order_size`, and `auto_resolved_order` (the concrete sub-
// mode the auto-picker resolved when env = 0, for APILOG
// transparency).
//
// See the common helper for the full N_ORDER semantics, mode table,
// and auto-picker rationale.
inline void fill_sorted_expert_order(GroupNTilePlan &plan,
                                     const std::vector<int> &M,
                                     int num_ops) {
  fill_ntile_expert_order(plan.expert_order.data(),
                          plan.expert_order_size,
                          GroupNTilePlan::kMaxExperts,
                          M, num_ops,
                          &plan.auto_resolved_order);
}

// Effective per-thread decode N-tile size.  Honors the optional
// `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE` env override (common.hpp,
// cached) when non-zero; otherwise falls back to the compile-time
// `kDecodeNTile = 256` default.  Deployment tuning knob — smaller
// values (e.g. 128) increase subtile-pass count per thread for small
// num_ops / large thread pools; larger values (e.g. 512) reduce
// kernel invocation overhead on wider per-thread N-slices.
inline int effective_decode_n_tile() {
  const int ov = get_grp_matmul_custom_kernel_n_tile();
  return (ov > 0) ? ov : kDecodeNTile;
}

// ── Auto-select mirror ───────────────────────────────────────────────
// Returns true when ALGO 0's auto-selector (`auto_select_algo` in
// `group_matmul_dispatch.cpp`) would have picked ALGO 1
// (sequential_experts) for this shape rather than ALGO 3.  The
// planner uses this to route to `execute_sequential` so that calls
// pinned with `ZENDNNL_GRP_MATMUL_ALGO=3` behave identically to
// auto-pick (env=0) on shapes where ALGO 1 is the better choice —
// only the gemm_mode label distinguishes the two paths
// (`flat_n_tile_sequential` vs `sequential_experts`) for telemetry.
//
// Mirror of `auto_select_algo`'s decision tree, simplified to the
// portions reachable here (we know we're already in flat_n_tile, so
// `n_tile_safe` is implicitly true and Rule 0 is handled by the
// R3 capacity gate that runs alongside this check):
//
//   Rule 1  num_ops ≥ num_threads        → ALGO 3 (Qwen-class)
//   Rule 2  num_ops ≤ kFewExpertsAlgo1   → ALGO 1 (Mixtral-class)
//   Rule 3  9 ≤ num_ops < num_threads,
//           prompt-class (max_M > kDecodeMaxM)  → ALGO 1
//           decode-class (max_M ≤ kDecodeMaxM)  → ALGO 3
//
// Critically: Rule 1 takes precedence over Rule 3's M-driven check,
// so Qwen-style prompt (128 experts × max_M=2048 on 64-128 thread
// hosts) stays on ALGO 3 — the prompt → ALGO 1 routing only fires
// when num_ops < num_threads.  This is the same priority order
// `auto_select_algo` applies; mirroring it preserves the explicit
// Qwen-prompt win without any per-shape carve-outs here.
//
// Replaces the legacy R1 (`num_ops ≤ 3`) and R2 (large-weight
// regime) ad-hoc perf gates: those tried to identify shapes where
// ALGO 3's column-parallel approach was specifically poor, but
// since the auto-selector ALREADY routes those shapes to ALGO 1
// upstream, the gates only ever fired under forced env=3.  The new
// mirror replaces them with a faithful replay of auto-select's
// decision so env=3 = env=0 for the strategy choice.
inline bool auto_select_would_pick_algo1(
    const GroupNTileTopology &topo) {
  // Rule 1: num_ops ≥ num_threads → ALGO 3 (Qwen-class).
  if (topo.num_ops >= topo.num_threads) return false;
  // Rule 2: num_ops ≤ 8 → ALGO 1 (Mixtral-class).
  if (topo.num_ops <= kFewExpertsAlgo1) return true;
  // Rule 3 (M-driven): prompt → ALGO 1, decode → ALGO 3.
  return (topo.max_M > kDecodeMaxM);
}

// ── Viability check (PERF heuristic, not structural) ─────────────────
// N-tiling only helps when N is large enough to create useful tiles
// AND the rounds path actually needs to split N across threads.  Two
// regimes:
//
//   * Per-expert team (`team_size_est = num_threads / num_ops ≥ 2`):
//     each expert gets a team of `team_size_est` threads that must
//     split N.  The per-thread tile must be ≥ kMinNTile (prompt) or
//     kDecodeNTile (decode) to keep the microkernel efficient.
//     Below that, AOCL DLP packing the full row once and saturating
//     the team (Sequential) wins.
//
//   * One-thread-per-expert (`team_size_est ≤ 1`, Qwen-class):
//     `num_ops ≥ num_threads/2` so the rounds executor assigns
//     ≤ 1 thread per expert and the per-thread tile equals the full
//     expert width N.  There is NO N-split, hence NO per-thread
//     tile-min to satisfy — N-tile viability is trivially TRUE.
//     Without this carve-out, prompt-class Qwen3-30B-A3B Op1
//     (`num_ops=121..128`, `num_threads=128`, `max_N=1536`,
//     `max_M>>kDecodeMaxM`) fails the prompt min-tile gate
//     (`1536/kMinNTile=3 tiles < ccd_size/2=4`) and silently falls
//     back to Sequential — even though the rounds executor would
//     run one thread per expert with the full 1536-wide tile.
//     Production measurement (Qwen3-30B-A3B prompt, 128 threads,
//     121 active experts, skewed M): Sequential averaged 122 ms /
//     call (max 943 ms) vs ~tens-of-ms for CK rounds.
//
// Used by `plan_group_n_tile` ONLY when `n_tile_strategy == 0` (auto
// mode) — under explicit `n_tile_strategy = {1, 2}` the user has
// opted into N-tile and viability is a soft hint, not a gate (see
// the precedence diagram in `plan_group_n_tile`).
inline bool ntile_viable(const GroupNTileTopology &topo) {
  const int team_size_est = topo.num_threads / std::max(1, topo.num_ops);
  // Qwen-class carve-out: no per-expert team split → no tile-min
  // requirement.  `max_N > 0` is a defence-in-depth structural check
  // (zero-N callers are rejected upstream by the dispatcher).
  if (team_size_est <= 1) return topo.max_N > 0;
  const int viability_min_tile =
      (topo.max_M <= kDecodeMaxM) ? effective_decode_n_tile() : kMinNTile;
  const int tiles_available = topo.max_N / viability_min_tile;
  const int min_useful = (topo.num_ops > topo.num_ccds)
      ? std::max(1, topo.ccd_size / 2)
      : std::max(1, team_size_est / 2);
  return tiles_available >= min_useful;
}

// ── (D) Decode parallel ──────────────────────────────────────────────
// Equal CCD-sized team per expert, all concurrent, no barriers.
// Eligible only on decode-class shapes with enough experts, fitting
// in CCDs, and balanced M distribution (skew_ratio ≤ 4).  High M-
// skew would leave large-M experts' surplus threads idle (capped by
// their N-tile count) while small-M experts bottleneck.
//
// Returns true when the plan is finalised as DecodeD; false when the
// caller should fall through to the rounds-based strategies.
inline bool try_decode_d_plan(const GroupNTileTopology &topo,
                              GroupNTilePlan &plan) {
  if (topo.max_M > kDecodeMaxM) return false;
  const int decode_n_tile = effective_decode_n_tile();
  const int team_size_est = topo.num_threads / std::max(1, topo.num_ops);
  const int skew_ratio = (topo.min_M_active > 0)
      ? (topo.max_M / topo.min_M_active) : topo.max_M;
  const bool eligible =
         topo.num_ops >= 6
      && topo.num_ops <= topo.num_ccds
      && topo.min_M_active >= 3
      && skew_ratio <= 4
      && topo.max_N / decode_n_tile <= team_size_est;
  if (!eligible) return false;

  // Equal thread allocation per expert, capped at the available
  // N-tile count — M-proportional allocation would just leave
  // large-M experts idle while small-M becomes the bottleneck.
  const int max_tiles = topo.max_N / decode_n_tile;
  const int thr_per_expert = std::max(1,
      std::min(topo.num_threads / topo.num_ops, max_tiles));
  plan.strategy = GroupNTileStrategy::DecodeD;
  plan.min_n_tile = decode_n_tile;
  plan.decode_thr_per_expert = thr_per_expert;
  plan.decode_total_threads = topo.num_ops * thr_per_expert;
  return true;
}

// Force-DecodeD path — used when the user pins
// `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1` and explicitly asks the
// planner to bypass the perf-eligibility heuristics
// (`try_decode_d_plan`'s `max_M ≤ 32`, `num_ops ∈ [6, num_ccds]`,
// `min_M_active ≥ 3`, `skew_ratio ≤ 4`, `max_N / decode_n_tile ≤
// team_size_est`).  Useful for benchmarking DecodeD on shapes the
// auto-heuristic would normally route to Rounds or to ALGO 1
// (e.g. Qwen-style prompt with `num_ops > num_ccds`, where the
// eligibility's `num_ops ≤ num_ccds` gate refuses).
//
// One structural floor remains: `num_threads >= num_ops`.  DecodeD's
// executor opens `#pragma omp parallel num_threads(num_ops ×
// thr_per_expert)` and maps `tid / thr_per_expert → expert`.  If the
// team is smaller than `num_ops` we'd over-subscribe the OMP team
// and the integer-division mapping would collide on the wrap, so
// return false here and let the caller fall through to Rounds.
//
// `thr_per_expert` for the force path is `max(1, num_threads/num_ops)`
// — the eligibility-path's additional `max_tiles` cap is dropped so
// per-thread N slices can go below `decode_n_tile` (the user is
// asking us to ignore that perf threshold).
inline bool force_decode_d_plan(const GroupNTileTopology &topo,
                                GroupNTilePlan &plan) {
  if (topo.num_threads < topo.num_ops) return false;
  const int decode_n_tile = effective_decode_n_tile();
  const int thr_per_expert = std::max(1,
      topo.num_threads / std::max(1, topo.num_ops));
  plan.strategy = GroupNTileStrategy::DecodeD;
  plan.min_n_tile = decode_n_tile;
  plan.decode_thr_per_expert = thr_per_expert;
  plan.decode_total_threads = topo.num_ops * thr_per_expert;
  return true;
}

// ── (A) FewExperts plan — L3-aware adaptive batching ─────────────────
// Instead of running all experts concurrently with few threads each
// (L3-thrashing for large weights), process in batches sized to
// maximise threads per expert AND keep concurrent weights in L3.
// `compute_target_batch` bumps the batch up toward num_threads/ccd_size
// when the bumped working set still fits L3, so every round saturates
// the thread team.
inline void build_few_experts_plan(const GroupNTileTopology &topo,
                                   int ab_min_tile,
                                   GroupNTilePlan &plan) {
  const int l3_batch = compute_l3_batch(topo);
  const int batch_size = compute_target_batch(topo, l3_batch);
  const int max_n_thr = std::max(1, topo.max_N / ab_min_tile);

  plan.strategy   = GroupNTileStrategy::FewExperts;
  plan.min_n_tile = ab_min_tile;
  plan.batch_size = batch_size;
  plan.max_n_thr  = max_n_thr;           // n_thr_fixed = 0 → per-round
}

// ── (B) ManyExperts round-scheduler candidates ───────────────────────
// Three competing shapes differ only in how per-round thread count
// and round size are resolved:
//   S_A  Single-round: one round of num_ops experts, n_thr per expert
//        capped at ccd_size and the N-tile count.  Eligible only when
//        num_threads ≥ num_ops (every expert gets ≥1 thread).
//   S_B  Multi-round (legacy): fixed n_thr = ccd_size, batch = capped
//        L3-aware target.  walk_wall = n_rounds / n_thr.
//   S_C  Balanced-rounds: same n_rounds as multi but spreads experts
//        evenly across rounds (round sizes differ by ≤ 1) and scales
//        n_thr proportionally (capped by max_tiles).  Saturates the
//        tail.
//
// The picker consumes a plain `RoundCandidates` (declared in
// group_matmul_n_tile.hpp) so `plan_group_n_tile` stays a thin
// orchestrator and the candidate-building logic is isolated.
inline RoundCandidates build_round_candidates(
    const GroupNTileTopology &topo, int ab_min_tile) {

  RoundCandidates c{};
  c.max_tiles    = std::max(1, topo.max_N / ab_min_tile);
  const int l3_batch   = compute_l3_batch(topo);
  const int target_batch = compute_target_batch(topo, l3_batch);
  c.capped_batch = std::max(1, std::min(target_batch, topo.num_threads));

  // S_A single-round.
  c.n_thr_single = std::max(1,
      std::min({topo.ccd_size, c.max_tiles,
                topo.num_threads / std::max(1, topo.num_ops)}));
  c.single_eligible = (topo.num_threads >= topo.num_ops);
  c.wall_single = c.single_eligible
      ? 1.0 / static_cast<double>(c.n_thr_single)
      : std::numeric_limits<double>::infinity();

  // S_B multi-round.
  c.n_thr_multi = std::max(1, std::min({topo.ccd_size, c.max_tiles,
                                        topo.num_threads / c.capped_batch}));
  c.batch_multi = std::max(1,
      std::min(c.capped_batch, topo.num_threads / c.n_thr_multi));
  c.n_rounds_multi =
      (topo.num_ops + c.batch_multi - 1) / c.batch_multi;
  c.wall_multi = static_cast<double>(c.n_rounds_multi)
      / static_cast<double>(c.n_thr_multi);

  // S_C balanced-rounds.
  const int n_rounds_bal = c.n_rounds_multi;
  c.balanced_batch = std::max(1,
      (topo.num_ops + n_rounds_bal - 1) / n_rounds_bal);
  c.wall_balanced = 0.0;
  int processed = 0;
  for (int r = 0; r < n_rounds_bal; ++r) {
    const int rs = std::min(c.balanced_batch, topo.num_ops - processed);
    if (rs <= 0) break;
    const int thr = std::max(1,
        std::min(topo.num_threads / rs, c.max_tiles));
    c.wall_balanced += 1.0 / static_cast<double>(thr);
    processed += rs;
  }
  return c;
}

// Pick S_A / S_B / S_C either via forced env (ZENDNNL_GRP_MATMUL_N_ROUNDS)
// or by AUTO cost model:
//   AUTO: lowest wall among the eligible set.
//     * num_threads ≤ 64 → {single, multi, balanced}
//     * num_threads > 64 → {single, multi}  (balanced excluded because
//                         thr_per_expert > ccd_size spans CCDs and
//                         misses the DLP kernel's blocking factor).
//   Ties resolve to multi (keeps thr = ccd_size, the DLP sweet spot).
//   Forced single falls back to balanced when single is infeasible
//   (num_threads < num_ops).
//
// SCOPE: this cost model is consulted only on the legacy / custom-
// kernel path of `plan_group_n_tile`.  The default non-custom AOCL
// path takes the strict-stable shortcut (n_thr_fixed = stable,
// batch_size = num_threads/stable) and does NOT call this function.
// Empirical observations below apply specifically to the custom
// BF16 microkernel running with shape-keyed pack cache.
// (`RoundPick` enum lives in group_matmul_n_tile.hpp.)
inline RoundPick pick_round_strategy(const GroupNTileTopology &topo,
                                     const RoundCandidates &c) {
  const int rounds_mode = get_grp_n_rounds_mode();
  if (rounds_mode == 1) {
    if (c.single_eligible) return RoundPick::Single;
    // Force-Single is infeasible (num_threads < num_ops): fall back
    // to Balanced.  Emit a one-shot warning so an A/B benchmarker
    // running with `ZENDNNL_GRP_MATMUL_N_ROUNDS=1` knows the env
    // override didn't take effect for this call.  The gate uses an
    // `std::atomic<bool>` + `compare_exchange_strong` so concurrent
    // planner invocations (e.g. multiple application threads each
    // calling group_matmul) emit the warning exactly once across the
    // process — a plain `static bool` would race here in release
    // builds, with both readers seeing `false` and emitting the
    // warning twice (or, with sufficiently bad interleaving, not at
    // all).  Subsequent calls follow the documented fallback
    // silently.
    static const bool s_log_fallback = apilog_warning_enabled();
    static std::atomic<bool> s_warned{false};
    bool expected = false;
    if (s_log_fallback
        && s_warned.compare_exchange_strong(
               expected, true, std::memory_order_relaxed)) {
      apilog_warning(
          "[GRP_MATMUL.PLAN WARN] N_ROUNDS=1 forced single-round"
          " infeasible (num_threads=", topo.num_threads,
          " < num_ops=", topo.num_ops,
          "); using RoundPick::Balanced instead.  This warning fires"
          " once per process; subsequent calls follow the same"
          " documented fallback silently.");
    }
    return RoundPick::Balanced;
  }
  if (rounds_mode == 2) return RoundPick::Multi;
  if (rounds_mode == 3) return RoundPick::Balanced;

  const bool consider_balanced = (topo.num_threads <= 64);

  // ── L3-spill penalty on Single (auto cost model) ─────────────────
  // The plain `wall_single = 1 / n_thr_single` metric assumes the
  // per-expert weight footprint is L3-resident, which is true only
  // when the entire concurrent set of `num_ops` experts fits in
  // aggregate L3.  In Single mode every expert dispatches at the same
  // time (no rounds), so the live-set is `num_ops × wei_per_expert`
  // — for fused-MoE decode with several active experts and weights
  // in the multi-tens-of-MB range, this typically exceeds aggregate
  // L3 and every call streams weights from DRAM.  Multi and Balanced amortise the
  // spill across rounds whose individual working sets are sized by
  // `compute_l3_batch` to fit, so their `wall_*` already reflect the
  // L3-resident regime.
  //
  // Penalise `wall_single` by the spill ratio so the picker treats a
  // 2× DRAM-bound Single as twice as costly as a perfectly-fitting
  // Single.  The factor is conservative (linear in spill) — actual
  // DRAM bandwidth is shared across CCDs and contention isn't strictly
  // linear, but the linear approximation is sufficient to flip the
  // pick toward Balanced/Multi in the regime where Single's wall_*
  // is unrealistic.
  double wall_single_adj = c.wall_single;
  if (c.single_eligible && topo.wei_per_expert > 0) {
    const size_t single_concurrent =
        static_cast<size_t>(topo.num_ops) * topo.wei_per_expert;
    const size_t l3_total = get_grp_l3_total_bytes(topo.num_ccds);
    if (single_concurrent > l3_total) {
      const double spill =
          static_cast<double>(single_concurrent)
        / static_cast<double>(std::max<size_t>(l3_total, 1));
      wall_single_adj = c.wall_single * spill;
    }
  }

  // ── Thin-single-round correction ─────────────────────────────────
  // Even with L3 fitting, Single's wall = 1/thr undercounts when
  // `thr_per_expert < ccd_size` because the DLP kernel's NR-blocking
  // is tuned for ccd_size-wide teams and partial-tile misalignment
  // plus cross-CCD coordination add fixed-cost overhead the model
  // doesn't capture.  Effect is most visible at the cliff `n_thr_single
  // ∈ [ccd_size*3/4, ccd_size)` for many-experts MoE decode: the model
  // picks Single while Multi/Balanced is the better choice in practice;
  // at or near ccd_size the model is correct and Single is the winner.
  // Trigger condition: thin Single + few rounds (Multi/Balanced fits
  // in a 1-2 round tail).  Above 2 rounds the cost model's tail
  // approximation is sound and we trust the wall_* values.
  //
  // Tie-breaker: at ≤64t Balanced is in the eligible set and is
  // typically equal-or-better than Multi (more thr/expert per round
  // when team_size > ccd_size on a single-CCD-shape).  At >64t
  // Balanced is excluded (cross-CCD coord overhead dominates), so
  // the correction lands on Multi as the original logic.
  if (c.single_eligible) {
    const int thr_single =
        std::max(1, topo.num_threads / std::max(1, topo.num_ops));
    const int ccd_floor = std::max(1, (topo.ccd_size * 3) / 4);
    const bool single_thin = (thr_single < ccd_floor);
    if (single_thin && c.n_rounds_multi > 0 && c.n_rounds_multi <= 2) {
      if (consider_balanced && c.wall_balanced <= c.wall_multi) {
        return RoundPick::Balanced;
      }
      return RoundPick::Multi;
    }
  }

  if (c.single_eligible && wall_single_adj < c.wall_multi
      && (!consider_balanced || wall_single_adj <= c.wall_balanced)) {
    return RoundPick::Single;
  }
  if (consider_balanced && c.wall_balanced < c.wall_multi) {
    return RoundPick::Balanced;
  }
  return RoundPick::Multi;
}

// ── AUTO adaptive multi-tier thread allocator ──────────────────────────
// Engaged by `ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD=0`.
//
// Builds an asymmetric, M-skew-aware per-expert thread plan inside the
// existing ALGO 3 ManyExperts Single-round shape.  The output is
// written to `plan.stable_n_thr_per_expert[]` (consumed by the CK
// executor's prefix-sum scan), exactly like the legacy single-threshold
// path — only the way the array is populated differs.
//
// Three-tier policy with budget-aware compression:
//   * `T_high = max(M_p95, M_max * 0.40)`  → up to 8 threads / expert
//   * `T_mid  = max(M_p75, M_max * 0.20)`  → up to 4 threads / expert
//   * `T_low  = max(M_p50, M_max * 0.10)`  → up to 2 threads / expert
//   * everyone else                        → 1 thread / expert
//
// Eligibility gates (return `false` → fall back to Phase B silently):
//   1. `num_active < kMinExpertsForTier` — too few experts to gain.
//   2. `num_active >= num_threads` — no extras-budget at all.
//   3. `M_max / max(1, M_mean_active) < kMinSkew` — distribution is
//      flat; tiering would add overhead without converting threads
//      into wall-time savings.
//   4. `per_expert_cap_hybrid < 2` — every tier would clip to 1 thread.
//
// Budget math (each call):
//   preferred_extras = n_high * 7 + n_mid * 3 + n_low * 1
//   available_extras = num_threads - num_active_experts
//   if preferred ≤ available: assign full target; water-fill leftover
//                             into high tier by M-weight.
//   else:                     scale = available / preferred; round
//                             extras down, then M-weighted rounding
//                             water-fill consumes the remainder.
//
// Per-expert cap (always enforced AFTER tier allocation):
//   cap[e] = min(ccd_size, max_tiles, N[e] / ab_min_tile)
//
// Returns true iff the path applied (plan was populated and
// `per_expert_remainder` set).  Caller is expected to short-circuit
// the Phase B fallback in that case.
inline bool apply_adaptive_tiers(const GroupNTileTopology &topo,
                                 const RoundCandidates &c,
                                 int ab_min_tile,
                                 GroupNTilePlan &plan,
                                 const std::vector<int> &M,
                                 const std::vector<int> &N) {
  // Eligibility threshold constants.  Empirically derived from the
  // Qwen3-30B-A3B prompt sweep — values are intentionally conservative
  // so the path stays a no-op on workloads that don't match the M-skew
  // bottleneck pattern (e.g. decode shapes with M_max ≤ 20).
  constexpr int    kMinExpertsForTier = 8;
  constexpr double kMinSkew           = 2.5;     // M_max / M_mean threshold

  // Decode safety guard — defence-in-depth.  The caller in
  // `apply_round_pick` already gates the AUTO entry on
  // `max_M > kDecodeMaxM`, but mirror the same check here so any
  // future direct caller (test harness, alternative dispatcher) cannot
  // engage AUTO on a decode-class shape and re-introduce the decode
  // regression that motivated the gate.  See the comment block in
  // `apply_round_pick` for the rationale.
  if (topo.max_M <= kDecodeMaxM) return false;

  const int nops = topo.num_ops;
  if (nops < kMinExpertsForTier) return false;

  const int per_expert_cap_hybrid =
      std::min(topo.ccd_size, c.max_tiles);
  if (per_expert_cap_hybrid < 2) return false;

  // Build (M, e, cap) entries; count active experts and compute
  // M_max / M_total in a single pass.
  struct HE { int M; int e; int cap; };
  HE pairs[GroupNTilePlan::kMaxExperts];
  int n_active = 0;
  int m_max    = 0;
  int64_t m_sum = 0;
  for (int e = 0; e < nops; ++e) {
    const int m =
        (e < static_cast<int>(M.size()) && M[e] > 0) ? M[e] : 0;
    const int my_N =
        (e < static_cast<int>(N.size())) ? N[e] : 0;
    const int my_cap = std::min(per_expert_cap_hybrid,
        my_N / std::max(1, ab_min_tile));
    pairs[e].M   = m;
    pairs[e].e   = e;
    pairs[e].cap = std::max(1, my_cap);
    if (m > 0) {
      ++n_active;
      m_sum += m;
      if (m > m_max) m_max = m;
    }
  }
  if (n_active < kMinExpertsForTier) return false;
  // Insufficient budget: every active expert already wants 1 thread.
  // Without extras to redistribute the tier path can't help.
  const int extras_budget = topo.num_threads - n_active;
  if (extras_budget <= 0) return false;
  const double m_mean = static_cast<double>(m_sum) /
                        static_cast<double>(n_active);
  if (m_mean <= 0.0 ||
      static_cast<double>(m_max) / m_mean < kMinSkew) return false;

  // Sort by M descending (heavy-first); inactive (M=0) trails.
  std::sort(pairs, pairs + nops,
      [](const HE &a, const HE &b) {
        if (a.M != b.M) return a.M > b.M;
        return a.e < b.e;
      });

  // Compute percentile breakpoints over the ACTIVE experts (the first
  // `n_active` entries of the sorted array, M-descending).  P95 is
  // the M at index `floor(n_active * 0.05)`, etc.
  auto pct_idx = [&](double pct) -> int {
    int idx = static_cast<int>(std::floor(n_active * (1.0 - pct)));
    if (idx < 0)         idx = 0;
    if (idx >= n_active) idx = n_active - 1;
    return idx;
  };
  const int M_p95 = pairs[pct_idx(0.95)].M;
  const int M_p75 = pairs[pct_idx(0.75)].M;
  const int M_p50 = pairs[pct_idx(0.50)].M;

  // Tier thresholds — max(percentile, M_max * fraction).  The
  // fraction guard prevents pathological cases (e.g. all-mediums-no-
  // heavies) from sliding the tier boundaries to zero.
  const int T_high = std::max(M_p95, static_cast<int>(m_max * 0.40));
  const int T_mid  = std::max(M_p75, static_cast<int>(m_max * 0.20));
  const int T_low  = std::max(M_p50, static_cast<int>(m_max * 0.10));

  // Per-tier target threads (capped per-expert downstream).
  constexpr int kTgtHigh = 8;
  constexpr int kTgtMid  = 4;
  constexpr int kTgtLow  = 2;

  // Classify each active expert; count tier sizes.  We keep tier
  // boundaries strict (≥), so the classification is monotone in M
  // (high → mid → low → baseline) along the M-descending sort.
  int tier[GroupNTilePlan::kMaxExperts] = {0};  // 3=high, 2=mid, 1=low, 0=base
  int n_high = 0, n_mid = 0, n_low = 0;
  for (int i = 0; i < n_active; ++i) {
    const int m = pairs[i].M;
    int t = 0;
    if      (m >= T_high) { t = 3; ++n_high; }
    else if (m >= T_mid)  { t = 2; ++n_mid; }
    else if (m >= T_low)  { t = 1; ++n_low; }
    tier[i] = t;
  }
  // If the high tier is empty AND mid tier is empty, the workload
  // doesn't satisfy the M-skew premise of the optimisation despite
  // passing the broad skew gate above.  Fall back to Phase B.
  if (n_high == 0 && n_mid == 0) return false;

  // Compute preferred extras (above the implicit 1-thread baseline).
  const int preferred_extras = n_high * (kTgtHigh - 1)
                             + n_mid  * (kTgtMid  - 1)
                             + n_low  * (kTgtLow  - 1);
  if (preferred_extras <= 0) return false;

  // Per-tier ROUNDED extras after budget compression.  Use double
  // arithmetic for the scaling and a single rounding pass.
  double scale = 1.0;
  if (preferred_extras > extras_budget) {
    scale = static_cast<double>(extras_budget) /
            static_cast<double>(preferred_extras);
  }
  const int ext_high = static_cast<int>(std::floor((kTgtHigh - 1) * scale));
  const int ext_mid  = static_cast<int>(std::floor((kTgtMid  - 1) * scale));
  const int ext_low  = static_cast<int>(std::floor((kTgtLow  - 1) * scale));

  // Initial allocation: 1 baseline + tier extras (clipped by cap).
  int alloc[GroupNTilePlan::kMaxExperts] = {0};
  int used_extras = 0;
  for (int i = 0; i < n_active; ++i) {
    int t = tier[i];
    int e = (t == 3) ? (1 + ext_high)
         : (t == 2) ? (1 + ext_mid)
         : (t == 1) ? (1 + ext_low)
                    : 1;
    if (e > pairs[i].cap) e = pairs[i].cap;
    alloc[i] = e;
    used_extras += (e - 1);
  }

  // Water-fill the rounding leftover (extras_budget - used_extras),
  // M-weighted, longest-job-first.  Reuses the same lhs * best_rhs
  // cross-multiply pattern as the legacy single-threshold path.
  int remaining = extras_budget - used_extras;
  while (remaining > 0) {
    int     best_i   = -1;
    int64_t best_lhs = -1;
    int64_t best_rhs =  1;
    for (int i = 0; i < n_active; ++i) {
      if (alloc[i] >= pairs[i].cap) continue;
      const int64_t lhs = static_cast<int64_t>(pairs[i].M);
      const int64_t rhs = static_cast<int64_t>(alloc[i] + 1);
      if (best_i < 0 || lhs * best_rhs > best_lhs * rhs) {
        best_i   = i;
        best_lhs = lhs;
        best_rhs = rhs;
      }
    }
    if (best_i < 0) break;        // every active expert at its cap
    alloc[best_i] += 1;
    remaining     -= 1;
  }

  // Commit to `stable_n_thr_per_expert[]`.  Inactive experts (M==0)
  // keep their default-zero slot — the prefix-sum scan skips them.
  for (int i = 0; i < n_active; ++i) {
    plan.stable_n_thr_per_expert[pairs[i].e] =
        static_cast<int16_t>(alloc[i]);
  }
  // Inactive entries (sorted last) — zero out defensively to avoid
  // stale values from a prior populate path that ran on the same
  // plan object before the dispatcher decided to retry.
  for (int i = n_active; i < nops; ++i) {
    plan.stable_n_thr_per_expert[pairs[i].e] = 0;
  }
  plan.per_expert_remainder = true;
  plan.n_thr_fixed          = 0;  // executor uses per-expert scan only

  static const bool s_log = apilog_info_enabled();
  if (s_log) {
    apilog_info(
        "[GRP_MATMUL.PLAN.AUTO_TIER] adaptive_tiers applied"
        " n_active=", n_active,
        " n_high=", n_high, " n_mid=", n_mid, " n_low=", n_low,
        " M_max=", m_max,
        " T_high=", T_high, " T_mid=", T_mid, " T_low=", T_low,
        " preferred_extras=", preferred_extras,
        " available_extras=", extras_budget,
        " scale=", scale,
        " heaviest_alloc=", alloc[0],
        " num_threads=", topo.num_threads,
        " per_expert_cap=", per_expert_cap_hybrid);
  }
  return true;
}

inline void apply_round_pick(const GroupNTileTopology &topo,
                             const RoundCandidates &c,
                             RoundPick pick,
                             int ab_min_tile,
                             GroupNTilePlan &plan,
                             const std::vector<int> &M,
                             const std::vector<int> &N,
                             bool use_custom) {
  plan.strategy   = GroupNTileStrategy::ManyExperts;
  plan.min_n_tile = ab_min_tile;
  switch (pick) {
    case RoundPick::Single: {
      plan.batch_size  = topo.num_ops;
      plan.n_thr_fixed = c.n_thr_single;

      // ── Three-mode HYBRID dispatch (env-gated) ─────────────────────
      // `ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD` selects between:
      //
      //   -1  DISABLED — skip the entire HYBRID block; fall through to
      //                  Phase B base+1.  Default; matches the legacy
      //                  behaviour when the env was unset.
      //    0  AUTO     — engage `apply_adaptive_tiers()` (planner-
      //                  driven 3-tier policy that auto-scales tier
      //                  thresholds to M_max and budget-compresses
      //                  per num_threads).  Falls back to Phase B on
      //                  eligibility failure.
      //   >0  MANUAL   — legacy single-threshold water-fill (heavy iff
      //                  `M[e] > threshold`).  Unchanged code path
      //                  below.
      //
      // All three modes share the same executor consumer
      // (`stable_n_thr_per_expert[]` + `per_expert_remainder = true`);
      // only the way the array is populated differs.  Same CK-only
      // gate (`use_custom`) and per-expert-cap precheck as Phase B.
      //
      // PHASE GATE (decode safety) — HYBRID is engaged ONLY on prompt-
      // class shapes (`max_M > kDecodeMaxM`).  Decode-class calls
      // (`max_M <= kDecodeMaxM`) bypass both AUTO and MANUAL and fall
      // through to Phase B's base+1 remainder distribution, regardless
      // of the env value.  Rationale:
      //   * Qwen decode (per-expert M ≤ 28) gains nothing measurable
      //     from giving heavies 4-8 threads — extra threads add OMP /
      //     N-slice overhead without compute headroom (sweep shows
      //     `HYBRID=0` ~flat, `HYBRID=8/16` ~+30-40% slower).
      //   * Unified E2E processes set a single env for the whole run;
      //     this gate lets `HYBRID=0` ship for prompt without
      //     touching the decode plan.
      // Phase B (base+1 remainder distribution further down) is NOT
      // the HYBRID feature — it's the planner's general thread-
      // saturation step and stays enabled for decode unconditionally.
      const int hybrid_mode = get_grp_matmul_n_tile_heavy_threshold();
      const int per_expert_cap_hybrid =
          std::min(topo.ccd_size, c.max_tiles);
      const bool is_prompt_class = (topo.max_M > kDecodeMaxM);
      bool hybrid_applied = false;

      // AUTO path — planner-driven adaptive tiers.  Prompt-only.
      if (use_custom
          && hybrid_mode == 0
          && is_prompt_class
          && per_expert_cap_hybrid >= 2) {
        hybrid_applied = apply_adaptive_tiers(topo, c, ab_min_tile,
                                              plan, M, N);
      }

      // MANUAL path — legacy single-threshold water-fill.  Prompt-only.
      // `heavy_threshold` is local-scoped so the existing body below
      // references it unchanged.
      const int heavy_threshold = hybrid_mode;
      if (!hybrid_applied
          && use_custom
          && heavy_threshold > 0
          && is_prompt_class
          && per_expert_cap_hybrid >= 2) {
        struct HE { int M; int e; int cap; bool heavy; };
        HE pairs[GroupNTilePlan::kMaxExperts];
        const int nops = topo.num_ops;
        int n_heavy = 0;
        int n_light_active = 0;
        for (int e = 0; e < nops; ++e) {
          const int m =
              (e < static_cast<int>(M.size()) && M[e] > 0) ? M[e] : 0;
          const int my_N =
              (e < static_cast<int>(N.size())) ? N[e] : 0;
          const int my_cap = std::min(per_expert_cap_hybrid,
              my_N / std::max(1, ab_min_tile));
          pairs[e].M     = m;
          pairs[e].e     = e;
          pairs[e].cap   = std::max(1, my_cap);
          pairs[e].heavy = (m > heavy_threshold);
          if (m > 0) {
            if (pairs[e].heavy) ++n_heavy;
            else                ++n_light_active;
          }
        }
        // Need both heavy AND light to extract value from the split
        // (all-heavy / all-light reduces to symmetric Phase B).  Also
        // need ≥ 2 threads/heavy on average to beat the existing top-N
        // base+1 promotion — falls through otherwise.
        const int heavy_budget = std::max(0,
            topo.num_threads - n_light_active);
        if (n_heavy > 0 && n_light_active > 0
            && heavy_budget >= 2 * n_heavy) {
          // Sort heavy-first (by M descending); light experts trail in
          // M-desc order.  Inactive experts (M == 0) sort last.
          std::sort(pairs, pairs + nops,
              [](const HE &a, const HE &b) {
                if (a.heavy != b.heavy) return a.heavy;
                if (a.M     != b.M)     return a.M > b.M;
                return a.e < b.e;
              });
          // Initial allocation: every heavy gets 1 thread; light/
          // inactive get nothing yet (set below).
          int alloc[GroupNTilePlan::kMaxExperts] = {0};
          for (int i = 0; i < n_heavy; ++i) alloc[i] = 1;
          int remaining = heavy_budget - n_heavy;
          // Water-fill: at each step, give the next thread to the
          // heavy expert whose `M / (alloc + 1)` ratio is largest
          // among those not yet capped.  Equivalent to a greedy
          // longest-job-first scheduler over the marginal thread.
          while (remaining > 0) {
            int best_i = -1;
            // Use 64-bit comparison to avoid overflow on M ~ 4096 ×
            // capped allocations.
            int64_t best_lhs = -1;
            int64_t best_rhs = 1;
            for (int i = 0; i < n_heavy; ++i) {
              if (alloc[i] >= pairs[i].cap) continue;
              const int64_t lhs =
                  static_cast<int64_t>(pairs[i].M);
              const int64_t rhs =
                  static_cast<int64_t>(alloc[i] + 1);
              // Compare lhs/rhs vs best_lhs/best_rhs without
              // dividing (cross-multiply, branch-friendly).
              if (best_i < 0 || lhs * best_rhs > best_lhs * rhs) {
                best_i   = i;
                best_lhs = lhs;
                best_rhs = rhs;
              }
            }
            if (best_i < 0) break;  // every heavy at its cap
            alloc[best_i] += 1;
            remaining -= 1;
          }
          // Populate `stable_n_thr_per_expert[]`.  Heavy slots from
          // the water-fill, light slots get exactly 1 thread, inactive
          // slots stay at zero (the prefix-sum scan skips them).
          for (int i = 0; i < n_heavy; ++i) {
            plan.stable_n_thr_per_expert[pairs[i].e] =
                static_cast<int16_t>(alloc[i]);
          }
          for (int i = n_heavy; i < nops; ++i) {
            const int slot = (pairs[i].M > 0) ? 1 : 0;
            plan.stable_n_thr_per_expert[pairs[i].e] =
                static_cast<int16_t>(slot);
          }
          plan.per_expert_remainder = true;
          plan.n_thr_fixed = 0;  // tell executor to use per-expert
                                 // scan exclusively
          hybrid_applied = true;
          static const bool s_log = apilog_info_enabled();
          if (s_log) {
            apilog_info(
                "[GRP_MATMUL.PLAN.HINT] hybrid_m_split enabled "
                "n_heavy=", n_heavy,
                " n_light_active=", n_light_active,
                " heavy_threshold=", heavy_threshold,
                " heavy_budget=", heavy_budget,
                " per_expert_cap=", per_expert_cap_hybrid,
                " heaviest_M=", pairs[0].M,
                " heaviest_alloc=", alloc[0],
                " num_threads=", topo.num_threads);
          }
        }
      }
      if (hybrid_applied) break;  // skip Phase B remainder fallback

      // ── Phase B (T4-simple): remainder-distribute heaviest-first ──
      // With `n_thr_single = min(ccd_size, max_tiles,
      // num_threads/num_ops)`, an integer-division remainder leaves
      // `num_threads - n_thr_single * num_ops` threads IDLE under the
      // executor's uniform-tpe mapping (`tid / thr_per_expert`).
      //
      // Concrete examples (CK path, ZENDNNL_GRP_MATMUL_N_ROUNDS=1):
      //   64t / num_ops=18 → base=3, 18 × 3 = 54 used, 10 IDLE.
      //   64t / num_ops=14 → base=4, 14 × 4 = 56 used,  8 IDLE.
      //   64t / num_ops=32 → base=2, 32 × 2 = 64 used,  0 IDLE.
      //   128t/ num_ops=18 → base=7, 18 × 7 = 126 used, 2 IDLE.
      //
      // At medium num_ops the loss is up to ~16% of total thread
      // budget — visible end-to-end as a sub-linear 64t/128t scaling
      // ratio (~2.08x vs ideal 2.0x).  Distribute the surplus to the
      // M-heaviest experts (where extra threads convert most
      // efficiently into wall-time savings via finer N-tile splits)
      // so all threads contribute productive work.  When `remainder
      // == 0` or `base + 1` would breach the per-expert cap, skip the
      // distribution and fall through to the uniform `n_thr_fixed`
      // path (zero behaviour change at 128t / num_ops=32 and similar
      // perfect-division shapes).
      //
      // Gated to the CK path only (`use_custom == true`).  Two
      // separate cases would otherwise reach this code with
      // `use_custom == false`:
      //   1. Strict-stable AOCL plan — never reaches `apply_round_pick`
      //      because `plan_group_n_tile` returns early after its own
      //      uniform population.  Not a concern here.
      //   2. Legacy non-strict AOCL (`AOCL_STABLE_NTILE=0` opt-out) —
      //      DOES reach `apply_round_pick`.  Populating
      //      `stable_n_thr_per_expert[]` there would make
      //      `participating_n_thr` take its strict-stable branch with
      //      NON-uniform per-expert values, repurposing what was
      //      meant as a CK-only optimisation as a fake strict-stable
      //      plan.  The legacy AOCL caller opted out of cache
      //      stability for A/B benchmarking; reintroducing a
      //      per-expert override there is outside the documented
      //      contract.  Gate prevents the leak.
      //
      // Safe for CK only — the CK pack cache is shape-keyed
      // (full-N pack per expert), so per-expert n_thr variation does
      // not break cache stability.  At runtime the executor detects
      // the populated array via `plan.per_expert_remainder` (set
      // below) and switches from `tid / tpe` mapping to a per-round
      // prefix-sum lookup.
      //
      // Per-expert eligibility filter.  `(base + 1) <= per_expert_cap`
      // checks against the GLOBAL `max_tiles = max_N / ab_min_tile`.
      // For uniform-N MoE workloads (typical) every expert has
      // `N[e] = max_N` and is therefore eligible.  For non-uniform-N
      // callers, an expert with `N[e] / ab_min_tile < base + 1`
      // cannot absorb the extra thread — `participating_n_thr`'s
      // dynamic-tile clamp `min(team_size, N[e] / min_n_tile)` would
      // cap it back down to its tile capacity and `do_tile`'s
      // `local_tid >= n_thr` early-return would idle the surplus
      // thread anyway.  Filtering ineligible experts out of the
      // base+1 recipient set keeps the documented "distribute
      // surplus to experts that can productively use them" semantics
      // and avoids re-introducing the very thread-idleness Phase B
      // exists to eliminate.
      const int base = c.n_thr_single;
      const int total_used = base * topo.num_ops;
      const int remainder = topo.num_threads - total_used;
      const int per_expert_cap = std::min(topo.ccd_size, c.max_tiles);
      if (use_custom
          && remainder > 0
          && remainder < topo.num_ops
          && (base + 1) <= per_expert_cap) {
        struct ME { int M; int e; bool eligible; };
        ME pairs[GroupNTilePlan::kMaxExperts];
        const int nops = topo.num_ops;
        const int n_for_extra = base + 1;
        for (int e = 0; e < nops; ++e) {
          pairs[e].M =
              (e < static_cast<int>(M.size()) && M[e] > 0) ? M[e] : 0;
          pairs[e].e = e;
          const int my_N =
              (e < static_cast<int>(N.size())) ? N[e] : 0;
          const int my_cap = my_N / std::max(1, ab_min_tile);
          pairs[e].eligible = (my_cap >= n_for_extra);
        }
        // Sort eligible-first, then by M descending within each
        // group.  Break ties on the original expert index so the
        // ordering stays deterministic without depending on
        // `std::stable_sort` (which can heap-allocate for buffered
        // merge — kept off the hot per-call planner path; the rest
        // of the N-tile sort utilities in
        // `group_matmul_parallel_common.hpp::sort_indices_by_m` are
        // similarly heap-free `std::sort`).  After the sort, slots
        // [0, eligible_count) are the M-heaviest eligible experts;
        // slots [eligible_count, nops) are the ineligible ones,
        // M-descending among themselves.
        std::sort(pairs, pairs + nops,
            [](const ME &a, const ME &b) {
              if (a.eligible != b.eligible) return a.eligible;
              if (a.M != b.M) return a.M > b.M;
              return a.e < b.e;
            });
        int eligible_count = 0;
        for (int i = 0; i < nops; ++i) {
          if (!pairs[i].eligible) break;
          ++eligible_count;
        }
        if (eligible_count > 0) {
          // Cap base+1 recipients at min(remainder, eligible_count).
          // If fewer than `remainder` experts are eligible, the extra
          // remainder threads idle — same outcome as pre-Phase-B but
          // never worse than the baseline.
          const int extras = std::min(remainder, eligible_count);
          for (int i = 0; i < nops; ++i) {
            const int e = pairs[i].e;
            const int n = (i < extras) ? (base + 1) : base;
            plan.stable_n_thr_per_expert[e] = static_cast<int16_t>(n);
          }
          plan.per_expert_remainder = true;
        }
        // else: no eligible recipients — leave the array zero and
        // `per_expert_remainder` false so the executor takes the
        // uniform O(1) `tid / tpe` mapping with `n_thr_fixed = base`
        // for every expert (identical to pre-Phase-B behaviour).
      }
      break;
    }
    case RoundPick::Multi:
      plan.batch_size  = c.batch_multi;
      plan.n_thr_fixed = c.n_thr_multi;
      break;
    case RoundPick::Balanced:
      plan.batch_size  = c.balanced_batch;
      plan.n_thr_fixed = 0;             // proportional in execute_rounds
      plan.max_n_thr   = c.max_tiles;   // cap by N-tile count
      break;
  }
}

// Strategy decision + parameter computation for ALGO 3.
//
// Two-path planner — pick the path up front based on whether the
// custom BF16 microkernel will run:
//
//   ┌────────────────────────────────────────────────────────────┐
//   │ AOCL DLP / oneDNN_blocked / native_brgemm   (!use_custom)  │
//   ├────────────────────────────────────────────────────────────┤
//   │ STRICT-STABLE plan.  Tile partition is fixed by             │
//   │ `num_threads` alone; every expert team has exactly         │
//   │ `stable = num_threads / kAoclTargetConcurrentSlots`        │
//   │ threads.  Cache key (col_start, n_tile) is invariant       │
//   │ across calls → AOCL reorder cache reaches a steady hit-    │
//   │ rate post-warmup.  Cost model (Single/Multi/Balanced) NOT  │
//   │ consulted on this path.                                    │
//   ├────────────────────────────────────────────────────────────┤
//   │ Custom BF16 microkernel                  (use_custom=true) │
//   ├────────────────────────────────────────────────────────────┤
//   │ DYNAMIC plan via cost model (try_decode_d / FewExperts /   │
//   │ ManyExperts with Single/Multi/Balanced picks).             │
//   │ Pack cache is shape-keyed (full-N pack per expert), so     │
//   │ tile-level cache stability is not required.                │
//   └────────────────────────────────────────────────────────────┘
//
// Both paths share the up-front fail-fast checks (viability, R3
// capacity, optional auto-select mirror) and the narrow-N escape:
// in either path, shapes that cannot run an N-tile plan OR that
// auto-select would have routed to ALGO 1 (when the user hasn't
// explicitly forced ntile via the strategy knob) fall through to
// Sequential.
//
// Sequential is reached for three reasons:
//
//   1. STRUCTURAL — `!ntile_viable` (N too small to split usefully)
//      OR R3 (num_ops exceeds the plan's fixed-size per-expert
//      arrays).  Memory safety / kernel correctness; non-negotiable
//      regardless of the strategy knob.
//
//   2. AUTO-MIRROR — `auto_select_algo` (the env=0 selector in
//      `group_matmul_dispatch.cpp`) would have picked ALGO 1 for
//      this shape rather than ALGO 3.  We mirror that decision so
//      forced `ZENDNNL_GRP_MATMUL_ALGO=3` runs the same strategy
//      auto-pick would have.  ONLY consulted under
//      `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=0 (auto)`; values 1 and 2
//      are explicit user intent to run ntile and skip this gate
//      (useful for benchmarking ntile vs ALGO 1 on shapes auto-
//      select would otherwise route away).  Replaces the legacy R1
//      (`num_ops ≤ 3`) and R2 (large-weight) ad-hoc perf gates with
//      a faithful replay of auto-select's three-rule tree (Rules 1,
//      2, 3 — see `auto_select_would_pick_algo1` for the mirror).
//
//   3. F3 NARROW-N ESCAPE — only on the AOCL strict-stable path,
//      when `stable > max_N / nr_align` would force aligned_n_split
//      to fall back to its unaligned even-split, breaking the
//      kernel's nr-alignment contract.
//
// Helper map (in execution order):
//   0. n_tile_strategy read            — `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY`
//                                        snapshot.  Drives the
//                                        auto-mirror gate (skipped
//                                        under values 1/2), the
//                                        force-DecodeD attempt
//                                        (value 1 only), and the
//                                        DecodeD heuristic attempt
//                                        on the CK path (value 0
//                                        only).
//   1. ntile_viable                    — Sequential if false (1).
//   2. R3 capacity guard               — Sequential if true (1).
//   3. auto_select_would_pick_algo1    — Sequential if true AND
//                                        n_tile_strategy == 0 (2).
//                                        Skipped under values 1/2:
//                                        explicit user intent runs
//                                        ntile even on Mixtral-class
//                                        / prompt-class shapes.
//   4. force_decode_d_plan             — value 1 ONLY.  Attempted
//                                        BEFORE the AOCL strict-
//                                        stable branch and the CK
//                                        cost-model branch so the
//                                        force semantics actually
//                                        win on both `use_custom`
//                                        and `!use_custom` paths.
//                                        Falls through to the
//                                        regular planner only when
//                                        structurally infeasible
//                                        (`num_threads < num_ops`).
//   5. strict-stable AOCL plan         — non-custom path; has its own
//                                        F3 narrow-N escape (3).
//   6. try_decode_d_plan               — value 0 ONLY (heuristic).
//                                        Custom path, decode-class.
//                                        Skipped under values 1/2
//                                        (1 already tried force
//                                        above; 2 skips DecodeD
//                                        outright).
//   7. build_few_experts_plan          — custom path, (A) num_ops ≤ num_ccds.
//   8. build_round_candidates
//      + pick_round_strategy
//      + apply_round_pick              — custom path, (B) many-experts.
inline GroupNTilePlan plan_group_n_tile(
    const GroupNTileTopology &topo,
    matmul_algo_t algo, int nr_align, bool fused_epilogue,
    bool use_custom_at_plan_time,
    const std::vector<int> &M,
    const std::vector<int> &N) {

  GroupNTilePlan plan{};
  plan.algo = algo;
  plan.num_threads = topo.num_threads;
  plan.nr_align = nr_align;
  plan.fused_epilogue = fused_epilogue;

  // `kDecodeTileAbOn` is the production constant in
  // `group_matmul_parallel_common.hpp` — folded at compile time so the
  // decode-class min-tile bump (kDecodeNTile vs kMinNTile, below) has
  // zero runtime cost.
  constexpr bool decode_tile_ab_on = kDecodeTileAbOn;

  // Sequential fallback gates.  Precedence (tightest first):
  //
  //   * R3 — STRUCTURAL.  Capacity guard: GroupNTilePlan carries
  //     fixed-size stack arrays sized to `kMaxExperts =
  //     kNTilePlanMaxExperts = 256` (currently `expert_order` and
  //     `stable_n_thr_per_expert`).  When `num_ops > kMaxExperts`
  //     the strict-stable populator would only write the first 256
  //     entries, leaving executors that index `[0, num_ops)`
  //     reading either zeros (silently disabling the stable path
  //     for late experts) or — if the read site forgets to bounds-
  //     check — accessing memory past the array.  Both are fragile,
  //     so route the call to Sequential.  Sequential walks experts
  //     via `for (e=0; e<num_ops; ++e)` with no fixed-size lookup
  //     arrays, so it is safe at any num_ops.  Auto-select's rule 0
  //     capacity carve-out already routes `num_ops > kMaxExperts`
  //     shapes to ALGO 5 upstream; R3 here is the second-line
  //     defence for forced env=3 with `num_ops > kMaxExperts`.
  //     FIRES REGARDLESS OF `n_tile_strategy`.
  //
  //   * AUTO-MIRROR — PERF (auto mode only).  The auto-selector
  //     (`auto_select_algo` in `group_matmul_dispatch.cpp`) would
  //     have picked ALGO 1 for this shape if env were 0.  We mirror
  //     that decision so forced `ZENDNNL_GRP_MATMUL_ALGO=3` runs
  //     the same strategy auto-pick would have, with the gemm_mode
  //     label (`flat_n_tile_sequential` vs `sequential_experts`)
  //     distinguishing the two paths for telemetry.  Replaces the
  //     legacy R1 (`num_ops ≤ 3`) and R2 (large-weight regime)
  //     ad-hoc gates.  See `auto_select_would_pick_algo1` above for
  //     the three-rule mirror.
  //     FIRES ONLY WHEN `n_tile_strategy == 0` (auto).  Under
  //     explicit env=1/2 the user has opted out of the auto perf
  //     preference and we run N-tile.
  //
  //   * VIABILITY — PERF (auto mode only).  `ntile_viable` says the
  //     shape is too thin for an efficient per-thread N-split (see
  //     the heuristic doc-block on `ntile_viable` above).  Same
  //     auto-only gating as auto-mirror: under explicit env=1/2
  //     the user accepts whatever cost a thin N gives them — we run
  //     N-tile.  Historically this gate ALSO fired under force_ntile
  //     and silently demoted Qwen3-30B-A3B prompt (`num_ops=121-128`,
  //     `N=1536`, `max_M>>32`) onto Sequential at ~122 ms / call —
  //     a regression that violated the documented env contract.  Now
  //     gated behind `!force_ntile` and a PLAN.HINT line announces
  //     when the env overrode the viability hint.
  //
  // Other STRUCTURAL gates that may demote to Sequential further
  // down this function (independent of the early-return below):
  //
  //   * F3 narrow-N escape — kernel correctness (`aligned_n_split`
  //     alignment contract).  Only reachable on the strict-stable
  //     AOCL path (`!use_custom_at_plan_time && stable env=1`); not
  //     bypassable by `n_tile_strategy`.
  //
  //   * tight_split_halves CK refusal — memory safety (silu/gelu +
  //     tight caller without an OOP swiglu helper).  Handled
  //     post-plan in `flat_n_tile`; not bypassable.
  const int n_tile_strategy = get_grp_n_tile_strategy();
  const bool force_ntile = (n_tile_strategy != 0);

  const bool viable = ntile_viable(topo);
  const bool r3 = (topo.num_ops > GroupNTilePlan::kMaxExperts);
  // Auto-mirror and viability are PERF heuristics, gated behind
  // `!force_ntile`.  Values 1 and 2 are explicit user intent to run
  // N-tile and we honour that — Sequential under force_ntile is
  // reserved for genuinely structural reasons (R3, F3 narrow-N,
  // tight split-halves CK refusal).
  const bool auto_mirror = !force_ntile
                           && auto_select_would_pick_algo1(topo);
  const bool unviable_in_auto = !force_ntile && !viable;
  if (r3 || auto_mirror || unviable_in_auto) {
    plan.strategy = GroupNTileStrategy::Sequential;
    static const bool s_fb_log = apilog_info_enabled();
    if (s_fb_log) {
      const char *reason =
          r3              ? "R3_num_ops_exceeds_plan_capacity"
        : auto_mirror     ? "auto_mirror_picks_algo1"
                          : "ntile_unviable(N_too_small_for_team_split)";
      // Sub-reason for auto_mirror: which rule of the auto-selector
      // fired.  Helps readers tell "Mixtral path" (Rule 2) from
      // "gpt-oss prompt" (Rule 3) at a glance in the L3 log.
      const char *auto_sub = auto_mirror
          ? (topo.num_ops <= kFewExpertsAlgo1
                 ? "rule2_few_experts"
                 : "rule3_prompt_M")
          : "";
      apilog_info("[GRP_MATMUL.PLAN.FALLBACK] strategy=Sequential "
                  "reason=", reason,
                  (auto_mirror ? " auto_sub=" : ""),
                  (auto_mirror ? auto_sub      : ""),
                  " num_ops=", topo.num_ops,
                  " plan_capacity=", GroupNTilePlan::kMaxExperts,
                  " max_M=", topo.max_M,
                  " max_N=", topo.max_N,
                  " max_K=", topo.max_K,
                  " wei_per_expert_MB=", (topo.wei_per_expert >> 20),
                  " num_threads=", topo.num_threads,
                  " num_ccds=", topo.num_ccds,
                  " n_tile_strategy=", n_tile_strategy);
    }
    return plan;
  }

  // Env-honoured-over-heuristic announcement.  Fires when the user
  // explicitly set `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY={1,2}` and the
  // viability heuristic would otherwise have demoted to Sequential.
  // Surfaces "the env contract was honoured" in the L3 trail so
  // production operators can confirm in one grep that the strategy
  // env is in effect on the shape they're tuning.
  if (force_ntile && !viable) {
    static const bool s_hint_log = apilog_info_enabled();
    if (s_hint_log) {
      apilog_info("[GRP_MATMUL.PLAN.HINT] "
                  "n_tile_strategy=", n_tile_strategy,
                  " honoured over ntile_viable=false (env wins over "
                  "perf heuristic).  num_ops=", topo.num_ops,
                  " max_M=", topo.max_M,
                  " max_N=", topo.max_N,
                  " max_K=", topo.max_K,
                  " num_threads=", topo.num_threads);
    }
  }

  // For paths (A), (B), and the strict-stable AOCL path: when max_M is
  // small (decode-class shape), use the smaller decode-n-tile as
  // min-tile so max_n_thr is high enough to saturate all threads.
  // See `kDecodeTileAbOn` in group_matmul_parallel_common.hpp for the
  // rationale.  `effective_decode_n_tile()` honors the optional
  // `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE` override.
  //
  // Hoisted to a local so the value (and its underlying cached env
  // probe) is computed once for this `plan_group_n_tile` call and
  // reused at all downstream sites (currently `ab_min_tile` plus the
  // force-decode_d HINT log).  Cheap — `effective_decode_n_tile()`
  // already short-circuits on a cached snapshot — but the local
  // makes the "single read per plan" intent explicit.
  const int decode_n_tile_snapshot = effective_decode_n_tile();
  const int ab_min_tile = (topo.max_M <= kDecodeMaxM && decode_tile_ab_on)
      ? decode_n_tile_snapshot : kMinNTile;

  // ── Force-DecodeD path (knob value 1) ──────────────────────────────
  // When the user explicitly forces DecodeD, attempt it BEFORE any
  // other strategy could run — including the AOCL strict-stable plan
  // below, which would otherwise pick ManyExperts on the non-custom
  // path and silently override the user's request.  The user
  // accepted the consequence of cache-key thrash by setting env=1
  // on a non-CK workload.  Structural floor: `num_threads >= num_ops`;
  // if false we fall through to the regular planner branches below
  // (closest-equivalent Rounds strategy).
  if (n_tile_strategy == 1) {
    if (force_decode_d_plan(topo, plan)) {
      static const bool s_log_force = apilog_info_enabled();
      if (s_log_force) {
        apilog_info(
            "[GRP_MATMUL.PLAN.HINT] "
            "n_tile_strategy=decode_d FORCED — bypassed perf-"
            "eligibility heuristic and AOCL strict-stable.  "
            "num_ops=", topo.num_ops,
            " num_threads=", topo.num_threads,
            " thr_per_expert=", plan.decode_thr_per_expert,
            " max_M=", topo.max_M,
            " max_N=", topo.max_N,
            " decode_n_tile=", decode_n_tile_snapshot);
      }
      return plan;
    }
    // Structural infeasibility: num_threads < num_ops.  Fall through
    // to the regular planner branches below (AOCL strict-stable
    // ManyExperts when !use_custom, or the CK cost-model path).
    static const bool s_log_fb = apilog_info_enabled();
    if (s_log_fb) {
      apilog_info(
          "[GRP_MATMUL.PLAN.HINT] "
          "n_tile_strategy=decode_d requested but num_threads < num_ops "
          "(structurally infeasible — DecodeD would over-subscribe the "
          "OMP team).  Falling through to Rounds.  "
          "num_ops=", topo.num_ops,
          " num_threads=", topo.num_threads);
    }
  }

  // ── AOCL strict-stable plan (path 1 of 2; see header above) ────────
  // Forces `team_size == stable` for every expert in every round,
  // making the AOCL reorder cache key invariant across calls.
  //
  // Narrow-N escape: when `stable > max_N / nr_align` we cannot have
  // `stable` aligned partitions of N (aligned_n_split would fall
  // back to its unaligned even-split, breaking the kernel's
  // nr-alignment contract).  Route to Sequential instead.
  //
  // DecodeD is skipped on this path: its `thr_per_expert =
  // num_threads / num_ops` is num_ops-dependent, which would
  // re-introduce shape sensitivity into the cache key.  The
  // single-round ManyExperts shape (when num_ops ≤ batch_max) gives
  // the same parallelism plus a sub-µs end-of-region barrier.
  if (!use_custom_at_plan_time && get_grp_matmul_aocl_stable_ntile()) {
    const int stable = aocl_stable_n_thr(topo.num_threads, topo.max_N);
    const int max_align_slots = std::max(1,
        topo.max_N / std::max(1, nr_align));
    if (stable > max_align_slots) {
      // Narrow-N escape: route to Sequential.
      plan.strategy = GroupNTileStrategy::Sequential;
      static const bool s_log_narrow = apilog_info_enabled();
      if (s_log_narrow) {
        apilog_info(
            "[GRP_MATMUL.PLAN.FALLBACK] strategy=Sequential "
            "reason=F3_narrow_N_escape "
            "stable=", stable,
            " max_align_slots=", max_align_slots,
            " nr_align=", nr_align,
            " max_N=", topo.max_N,
            " num_threads=", topo.num_threads);
      }
      return plan;
    }

    const int batch_max = std::max(1, topo.num_threads / stable);
    plan.strategy   = GroupNTileStrategy::ManyExperts;
    plan.min_n_tile = ab_min_tile;
    plan.batch_size  = std::min(topo.num_ops, batch_max);
    plan.n_thr_fixed = stable;
    plan.max_n_thr   = stable;  // metadata for APILOG; executor consults
                                // n_thr_fixed when > 0

    // Self-gating on ZENDNNL_GRP_MATMUL_N_ORDER (mode 0 = off, no-op).
    fill_sorted_expert_order(plan, M, topo.num_ops);

    // Populate stable_n_thr_per_expert so participating_n_thr's
    // safety clamps (defence-in-depth — see comment above the
    // function in this file) return `stable` directly.  Under the
    // strict-stable plan these clamps are no-ops because the planner
    // guarantees team_size == stable; if a future regression breaks
    // that, the clamps degrade gracefully to dynamic-tile behaviour.
    for (int e = 0; e < topo.num_ops && e < GroupNTilePlan::kMaxExperts;
         ++e) {
      if (M[e] > 0) {
        plan.stable_n_thr_per_expert[e] = static_cast<int16_t>(stable);
      }
    }
    return plan;
  }

  // ── Custom-kernel path (path 2 of 2): cost-model strategy ──────────
  // Reached when `use_custom_at_plan_time` (the BF16 microkernel
  // engaged at flat_n_tile entry) OR when the AOCL stable env knob
  // is OFF (legacy A/B mode).  Pack cache is shape-keyed, so the
  // cost model is free to optimise wall time without cache-key
  // constraints.

  // Heuristic DecodeD attempt (knob value 0).  Bypassed under values
  // 1 (force-DecodeD already attempted upfront above) and 2 (skip
  // DecodeD entirely — go straight to Rounds).
  if (n_tile_strategy == 0) {
    if (try_decode_d_plan(topo, plan)) return plan;
  }

  if (topo.num_ops <= topo.num_ccds) {
    // (A) Few experts: L3-aware adaptive batching, proportional
    // thr_per_expert per round (via n_thr_fixed = 0 in the executor).
    build_few_experts_plan(topo, ab_min_tile, plan);
  } else {
    // (B) Many experts: barrier-synchronised rounds.  Build the three
    // candidate shapes, pick by cost model (or force via env), and
    // commit the winner's parameters to the plan.
    const RoundCandidates c = build_round_candidates(topo, ab_min_tile);
    const RoundPick pick = pick_round_strategy(topo, c);
    apply_round_pick(topo, c, pick, ab_min_tile, plan, M, N,
                     use_custom_at_plan_time);
  }

  // Self-gating on ZENDNNL_GRP_MATMUL_N_ORDER (mode 0 = off, no-op).
  fill_sorted_expert_order(plan, M, topo.num_ops);

  // `stable_n_thr_per_expert` population on this branch is conditional:
  //   * Custom-kernel + Single round with `num_threads % num_ops != 0`
  //     and `base + 1 <= min(ccd_size, max_tiles)`: `apply_round_pick`
  //     populates it with a remainder-distribution (M-heaviest
  //     experts get `base + 1` threads, the rest get `base`) so the
  //     executor saturates the full thread team instead of leaving
  //     `num_threads % num_ops` slots idle.  Phase B / T4-simple from
  //     the 64-core optimisation work.  participating_n_thr's
  //     `!use_custom` gate prevents the per-expert array from being
  //     read on the CK path — instead the executor's prefix-sum
  //     lookup feeds the per-expert team_size into participating_n_thr
  //     directly, and the dynamic-tile clamp inside that function
  //     returns the same value.
  //   * Custom-kernel + non-Single (Multi / Balanced / FewExperts /
  //     DecodeD): no population — uniform `n_thr_fixed` is already
  //     correct for those patterns.
  //   * Legacy non-strict AOCL (env=0): no population — caller opted
  //     out of cache stability for A/B perf comparison.
  // The strict-stable AOCL branch (above) populates the field
  // unconditionally with uniform `stable` (the only configuration
  // that requires byte-identical cache keys across calls).

  return plan;
}

// =====================================================================
// Section C — Strategy executors
// =====================================================================
//
// Each executor consumes the planner's `GroupNTilePlan` and a
// `GroupNTileContext` and runs the actual GEMM calls.  All OMP and
// barriers live here.

// (F) Sequential — N too small for tiling.  Runs one expert at a time
// with the full thread team per kernel.  Two flows:
//
//   (1) Wide caller (ldc[e] >= N[e]) OR no fused activation: matmul
//       directly into caller's dst; if `plan.fused_epilogue`, run an
//       in-place activation pass on the wide buffer afterward.
//
//   (2) Tight caller + fused swiglu (ldc[e] < N[e] AND fused_epilogue):
//       caller's dst is `[M, I]` (e.g. fused-MoE internal-alloc tight
//       arena).  The matmul produces `2I` cols (gate+up) which can't
//       fit in caller's tight dst — AOCL would refuse with
//       `ldc < N`.  Allocate a thread-local wide scratch of size
//       `M[e] * N[e] * dst_elem`, matmul into scratch at stride
//       `N[e]`, then OOP swiglu compacts to caller's tight dst at
//       its halved stride.  Same shape as the do_tile() tight branch
//       but executed serially per expert (no col split).
//
// The tight branch is rare in practice — it only fires when a tight-
// arena fused-MoE call has shapes that fail flat_n_tile's viability
// check (small N, large weight + few experts, num_ops <= 3).
inline void execute_sequential(const GroupNTilePlan &plan,
                               GroupNTileContext &ctx) {
  const int num_ops = static_cast<int>(ctx.M.size());
  for (int e = 0; e < num_ops; ++e) {
    if (ctx.M[e] <= 0) continue;
    static thread_local matmul_params local_params;
    local_params = ctx.params[e];

    // Tight-caller fallback path for Sequential: any gated activation
    // with `ldc < N` requires a wide [M, N] scratch (the matmul writes
    // 2I cols per row, but the caller's dst stride is only I).
    //
    // Three execution sub-cases:
    //   * swiglu_oai_mul — the OOP tile-row helper exists and folds
    //     `(scratch[M, 2I] -> dst[M, I])` in one pass per row.
    //     Used preferentially for swiglu since it's the path the
    //     standard backend has been using since the fused-MoE wrapper
    //     was introduced.
    //   * silu_and_mul / gelu_and_mul — no OOP tile-row helper today,
    //     so we apply the activation in-place on the wide scratch
    //     (writes activated cols [0, I) of scratch in-place via
    //     `apply_gated_act_inplace`) and then memcpy I cols per row
    //     into the tight dst.  Same memory-traffic profile as a
    //     hypothetical OOP helper would have (read 2I, write I), one
    //     extra in-flight loop trip per row for the memcpy.
    //
    // Without this branch, the Sequential strategy's wide-path matmul
    // would execute with ldc < N, overrunning rows of the caller's
    // tight dst — a silent corruption of the activation output when
    // the planner happens to pick Sequential on a tight-caller frame
    // (typically very small M / num_ops <= 3 shapes).
    // Hoisted dynamic-quant source substitution.  Sequential runs
    // one expert at a time with the full thread team, so the
    // wrapper inside `execute_expert_slice` would NOT race if we
    // skipped this — but we honour the hoisted state when present
    // so the planner-level decision ("dynamic-quant src has been
    // hoisted for the entire flat_n_tile call") stays uniform across
    // all strategies and the redundant per-expert wrapper call is
    // saved.  Identical substitution semantics to `do_tile()`.
    const void *src_for_call = ctx.src[e];
    int lda_for_call = ctx.lda[e];
    if (ctx.hoisted_src_quant != nullptr
        && static_cast<size_t>(e) < ctx.hoisted_src_quant->size()
        && (*ctx.hoisted_src_quant)[e].valid) {
      const auto &h = (*ctx.hoisted_src_quant)[e];
      src_for_call = h.src_ptr;
      lda_for_call = h.lda;
      local_params.dtypes.src = h.src_dtype;
      local_params.quant_params.src_scale = h.src_scale;
      local_params.quant_params.src_zp = h.src_zp;
    }

    const bool tight_caller = plan.fused_epilogue
                              && ctx.ldc[e] < ctx.N[e];
    if (tight_caller) {
      assert((ctx.N[e] % 2) == 0
             && "Sequential tight: N must be even (gate+up pair)");
      static thread_local PerThreadScratch scratch;
      const size_t need_bytes =
          static_cast<size_t>(ctx.M[e]) * ctx.N[e] * ctx.dst_elem;
      if (!grow_scratch(scratch, need_bytes)) {
        if (ctx.alloc_fail)
          ctx.alloc_fail->store(1, std::memory_order_relaxed);
        return;
      }
      execute_expert_slice(ctx.layout[e], ctx.transA[e], ctx.transB[e],
          ctx.M[e], ctx.N[e], ctx.K[e], ctx.alpha[e],
          src_for_call, lda_for_call, ctx.weight[e], ctx.ldb[e],
          ctx.bias[e], ctx.beta[e], scratch.buf, ctx.N[e],
          ctx.is_weights_const[e], plan.num_threads, local_params,
          plan.algo);
      if (ctx.fused_act == grp_matmul_gated_act_t::swiglu_oai_mul) {
        const int pairs = ctx.N[e] / 2;
        apply_swiglu_oai_tile_rows_oop(
            scratch.buf, /*src_ldc=*/ctx.N[e], /*src_col_start=*/0,
            ctx.dst[e], /*dst_ldc=*/ctx.ldc[e], /*dst_col_start=*/0,
            ctx.M[e], pairs, ctx.act_dtype);
      } else {
        // silu_and_mul / gelu_and_mul: apply activation in-place on
        // the wide scratch (writes activated cols [0, N/2) per row,
        // leaves cols [N/2, N) as garbage by the public-API contract),
        // then memcpy the activated I cols into the tight dst.
        apply_gated_act_inplace(
            ctx.fused_act, scratch.buf, /*row_start=*/0, ctx.M[e],
            ctx.N[e], /*ldc=*/ctx.N[e], ctx.act_dtype);
        const int I = ctx.N[e] / 2;
        const size_t row_bytes = static_cast<size_t>(I) * ctx.dst_elem;
        const size_t scratch_stride =
            static_cast<size_t>(ctx.N[e]) * ctx.dst_elem;
        const size_t dst_stride =
            static_cast<size_t>(ctx.ldc[e]) * ctx.dst_elem;
        for (int m = 0; m < ctx.M[e]; ++m) {
          std::memcpy(static_cast<char *>(ctx.dst[e]) + m * dst_stride,
                      static_cast<const char *>(scratch.buf)
                          + m * scratch_stride,
                      row_bytes);
        }
      }
      continue;
    }

    // Wide path (default).
    execute_expert_slice(ctx.layout[e], ctx.transA[e], ctx.transB[e],
        ctx.M[e], ctx.N[e], ctx.K[e], ctx.alpha[e],
        src_for_call, lda_for_call, ctx.weight[e], ctx.ldb[e],
        ctx.bias[e], ctx.beta[e], ctx.dst[e], ctx.ldc[e],
        ctx.is_weights_const[e], plan.num_threads, local_params, plan.algo);
    if (plan.fused_epilogue) {
      apply_gated_act_inplace(ctx.fused_act, ctx.dst[e], 0, ctx.M[e],
                              ctx.N[e], ctx.ldc[e], ctx.act_dtype);
    }
  }
}

// (D) DecodeD — small-M, balanced, ≤ num_ccds experts.
//
// Equal CCD-sized team per expert; one flat OMP region.  The planner
// sizes `decode_total_threads = num_ops × decode_thr_per_expert`
// exactly, so every thread in the region has a valid expert
// (`e = tid / thr_per_expert ∈ [0, num_ops)`) — no idle-thread guard
// needed.  Fused activation runs after a single barrier so all matmul
// writes are visible before any thread reads them back.
inline void execute_decode_d(const GroupNTilePlan &plan,
                             GroupNTileContext &ctx) {
  const int thr_per_expert = plan.decode_thr_per_expert;
  const int total_threads = plan.decode_total_threads;
  // Respect ZENDNNL_GRP_MATMUL_N_ORDER even on DecodeD: ordering is
  // perf-neutral here (DecodeD has no rounds, all experts are
  // processed concurrently), but keeping the indirection consistent
  // with execute_rounds means a user A/B-testing N_ORDER sees the
  // env knob applied uniformly across all ALGO 3 strategies.  Walk-
  // input remains the default for num_ops in the auto-mode walk-
  // input band (see auto_pick_n_order).
  const bool sort_on = (plan.expert_order_size > 0);

  #pragma omp parallel num_threads(total_threads)
  {
    const int tid = omp_get_thread_num();
    const int local_expert = tid / thr_per_expert;
    const int local_tid = tid % thr_per_expert;
    const int e = sort_on ? plan.expert_order[local_expert]
                          : local_expert;

    ctx.do_tile(plan, e, local_tid, thr_per_expert, plan.min_n_tile);

    // Fused activation: barrier so every thread's matmul write is
    // globally visible before any thread reads it back for its
    // swiglu_oai epilogue.  Non-fused mode has no barrier here
    // (matches legacy behaviour exactly).
    //
    // Skipped in two cases (activation is already fused into do_tile
    // above — see its body):
    //   * `ctx.use_custom` — the custom BF16 microkernel applies
    //     swiglu in registers and writes activated I cols directly.
    //   * `plan.tight_fused_epilogue` — the non-custom tight-dst
    //     branch runs matmul → scratch → OOP swiglu → tight dst,
    //     all per-thread with disjoint dst column ranges.
    // A second activation pass would reinterpret already-activated
    // bytes as raw (gate, up) pairs and corrupt the result.  Skipping
    // the barrier is safe here: neither skipped path has cross-thread
    // writes to the caller's dst, and the implicit end-of-parallel-
    // region barrier synchronises everything before return.
    if (plan.fused_epilogue && !ctx.use_custom
        && !plan.tight_fused_epilogue) {
      #pragma omp barrier
      ctx.apply_swiglu_oai(plan, e, local_tid, thr_per_expert,
                           plan.min_n_tile);
    }
  }
}

// (A) FewExperts / (B) ManyExperts — round-based tile execution.
// The two strategies share this executor and differ only in how
// `thr_per_expert` is sized per round:
//   FewExperts (n_thr_fixed == 0): proportional to
//     `num_threads / round_size`, capped by `max_n_thr` so threads
//     never exceed the N-tile count.
//   ManyExperts (n_thr_fixed > 0): fixed value across all rounds.
//
// Round synchronisation: each round body ends with an unconditional
// `#pragma omp barrier`.  Two reasons:
//
//   1. L3-resident batching contract.  `compute_target_batch()` sizes
//      `batch_size` so that `batch_size × weight_per_expert` fits in
//      the team's aggregate L3 (with a small overshoot cap controlled
//      by `kL3OvershootDen`).  Without a barrier between rounds,
//      threads on small-M experts of round
//      k race ahead into round k+1 while threads on large-M experts
//      are still finishing round k.  At that point up to
//      `2 × batch_size` experts' weights are concurrently live —
//      double the L3 footprint the planner sized for.  The barrier
//      keeps round-k and round-k+1 working sets disjoint in time.
//
//   2. Fused-mode matmul → activation ordering.  The activation
//      reads back per-expert matmul writes, so all round-k do_tile
//      writes must be globally visible before any thread starts
//      reading.  This is enforced by an inner barrier between
//      do_tile and apply_swiglu_oai.  The end-of-round barrier
//      additionally prevents fast threads from starting round k+1's
//      matmul while slow threads are still in round k's activation.
//
// Cost: one full-team sync per round.  Trades a few sync points for
// the L3-thrash protection described above.
inline void execute_rounds(const GroupNTilePlan &plan,
                           GroupNTileContext &ctx) {
  const int num_ops = static_cast<int>(ctx.M.size());
  const int batch_size = plan.batch_size;
  const int min_n_tile = plan.min_n_tile;
  const int n_thr_fixed = plan.n_thr_fixed;
  const int max_n_thr = plan.max_n_thr;
  const bool sort_on = (plan.expert_order_size > 0);

  // Pre-compute the round structure ONCE per call (same values
  // would otherwise be recomputed by every thread on every iteration
  // inside the OMP region: 128 threads × N rounds = N × 128 redundant
  // mod/div/min ops per call).  Stack-allocated array — 4 × int per
  // round × kMaxRounds = 1 KB max, no allocator traffic.
  //
  // For the FewExperts path (`n_thr_fixed == 0`), `thr_per_expert`
  // varies per round (depends on `round_size`); for ManyExperts
  // (`n_thr_fixed > 0`) it's the same on every round but we treat
  // both uniformly via the precomputed array.
  struct RoundInfo {
    int round_start;
    int round_size;
    int thr_per_expert;
    int round_threads;
  };
  static constexpr int kMaxRounds = kNTilePlanMaxExperts;  // same upper bound as expert order + planner
  RoundInfo rounds[kMaxRounds];
  const int n_rounds = (num_ops + batch_size - 1) / batch_size;
  // Defensive runtime check: a debug-only `assert` is not enough
  // here — `RoundInfo rounds[kMaxRounds]` is stack-allocated, and
  // `n_rounds > kMaxRounds` would write past the end (UB → memory
  // corruption) in release builds.  Trips for pathological inputs
  // (e.g., 8 threads × num_ops > 256, with N_ROUNDS=2 forcing
  // batch_multi=1) that were previously silently undefined.
  if (n_rounds > kMaxRounds) {
    // Gate the variadic argument formatting on the (cached) err-sink
    // enable.  `apilog_error_enabled()` is defined via
    // `LOGGER_ENABLED_MACRO(api, error)` in
    // `zendnnl/src/common/zendnnl_global.hpp` (lines 210-213) and
    // tracks the err level itself — gating on
    // `apilog_warning_enabled()` would suppress ERROR logs when the
    // user runs at `ZENDNNL_API_LOG_LEVEL=error` (warnings off).
    // Avoids the per-arg expression evaluation when nothing is
    // listening — a few cycles per refused dispatch in release
    // builds, but free and consistent with the other error-path
    // call sites we gate this commit.
    static const bool s_err_log = apilog_error_enabled();
    if (s_err_log) {
      apilog_error(
          "[execute_rounds] n_rounds=", n_rounds,
          " exceeds kMaxRounds=", kMaxRounds,
          " (num_ops=", num_ops, " batch_size=", batch_size, ")"
          " — refusing to run flat_n_tile rounds path; caller's dst"
          " is left untouched.  Increase kNTilePlanMaxExperts or route"
          " through a different ALGO 3 strategy.");
    }
    return;
  }
  assert(n_rounds <= kMaxRounds);

  // Per-expert n_thr detection (Phase B / T4-simple).  When the
  // CK Single round populates `stable_n_thr_per_expert[]` with a
  // NON-uniform remainder-distribution (M-heaviest experts get
  // `base+1`, the rest get `base`), `plan.per_expert_remainder` is
  // set true by `apply_round_pick`.  In that case we build each
  // round's `round_threads` from a sum over the per-expert array and
  // the OMP body resolves `tid → (expert, local_tid)` via a per-
  // round prefix-sum scan.
  //
  // Otherwise we fall back to the uniform `tpe` mapping with
  // `tid / tpe` arithmetic — preserving the original fast path
  // bit-for-bit.  This explicitly covers:
  //   * The strict-stable AOCL plan, which DOES populate
  //     `stable_n_thr_per_expert[]` (uniformly with `stable`, so
  //     `participating_n_thr`'s safety re-clamp returns `stable`
  //     directly), but with `per_expert_remainder == false` here so
  //     the executor takes the O(1) uniform path — no need to scan a
  //     uniform array O(round_size) times per round per thread.
  //   * Multi / Balanced / FewExperts / DecodeD on the CK path —
  //     no per-expert population happens for those.
  //   * Legacy non-strict AOCL (`AOCL_STABLE_NTILE=0`) — Phase B's
  //     CK-only gate (see `apply_round_pick`) leaves the array zero.
  const bool per_expert_thr = plan.per_expert_remainder;

  for (int r = 0; r < n_rounds; ++r) {
    const int rs  = r * batch_size;
    const int re  = std::min(num_ops, rs + batch_size);
    const int rsz = re - rs;
    if (per_expert_thr) {
      // Sum per-expert n_thr to get the round's active thread count.
      // `thr_per_expert = 0` is the sentinel that tells the OMP body
      // to use prefix-sum mapping instead of `tid / tpe`.
      int sum_thr = 0;
      for (int local_e = 0; local_e < rsz; ++local_e) {
        const int e = sort_on
            ? plan.expert_order[rs + local_e]
            : (rs + local_e);
        sum_thr +=
            static_cast<int>(plan.stable_n_thr_per_expert[e]);
      }
      rounds[r] = {rs, rsz, /*thr_per_expert=*/0, sum_thr};
    } else {
      const int tpe = (n_thr_fixed > 0)
          ? n_thr_fixed
          : std::min(plan.num_threads / rsz, max_n_thr);
      rounds[r] = {rs, rsz, tpe, rsz * tpe};
    }
  }

  #pragma omp parallel num_threads(plan.num_threads)
  {
    const int tid = omp_get_thread_num();

    for (int r = 0; r < n_rounds; ++r) {
      const RoundInfo &ri = rounds[r];

      int e = -1;
      int local_tid = -1;
      int n_thr_e = 0;
      if (tid < ri.round_threads) {
        if (ri.thr_per_expert > 0) {
          // Uniform fast path (no per-expert distribution): O(1)
          // mapping by div/mod, identical to the pre-Phase-B
          // behaviour.
          const int local_expert = tid / ri.thr_per_expert;
          local_tid = tid % ri.thr_per_expert;
          e = sort_on
              ? plan.expert_order[ri.round_start + local_expert]
              : (ri.round_start + local_expert);
          n_thr_e = ri.thr_per_expert;
        } else {
          // Per-expert path: linear prefix-sum scan over the round's
          // experts.  Cost is `O(round_size)` per thread — for the
          // typical MoE decode envelope (`round_size ≤ 32`) that's
          // ~32 int-compares per thread per round, dwarfed by the
          // kernel work that follows.  Cache: `stable_n_thr_per_expert`
          // is stack-resident inside the plan, hot in every thread's
          // L1 after the first scan.
          int cumulative = 0;
          for (int local_e = 0; local_e < ri.round_size; ++local_e) {
            const int e_cand = sort_on
                ? plan.expert_order[ri.round_start + local_e]
                : (ri.round_start + local_e);
            const int n =
                static_cast<int>(plan.stable_n_thr_per_expert[e_cand]);
            if (tid < cumulative + n) {
              e = e_cand;
              local_tid = tid - cumulative;
              n_thr_e = n;
              break;
            }
            cumulative += n;
          }
        }
        ctx.do_tile(plan, e, local_tid, n_thr_e, min_n_tile);
      }
      // Fused-mode matmul → activation ordering: every thread's
      // matmul writes must be globally visible before any thread
      // reads them for swiglu_oai.
      //
      // Skipped in two cases (activation is fused into do_tile; see
      // execute_decode_d for the detailed reasoning):
      //   * `ctx.use_custom` — in-register fused.
      //   * `plan.tight_fused_epilogue` — per-thread scratch + OOP.
      if (plan.fused_epilogue && !ctx.use_custom
          && !plan.tight_fused_epilogue) {
        #pragma omp barrier
        if (e >= 0) {
          ctx.apply_swiglu_oai(plan, e, local_tid, n_thr_e,
                               min_n_tile);
        }
      }
      // End-of-round barrier (see function-level comment): preserves
      // the planner's L3 batching contract by preventing fast threads
      // (small-M experts) from starting round k+1 while slow threads
      // are still in round k.  Unconditional in both fused and non-
      // fused modes — activation work is also M-proportional, so the
      // same imbalance applies.
      #pragma omp barrier
    }
  }
}

// =====================================================================
// gemm_mode_label — string label exposed via `gemm_mode_out`
// =====================================================================
//
// Returns the static literal that benchdnn / profilers print in the
// "kernel/gemm mode" column for this flat_n_tile call.  Fifteen
// values across the (strategy × fused × tight × custom × act-kind)
// cube:
//
//   strategy == Sequential                          → flat_n_tile_sequential
//   non-fused, standard backend                     → flat_n_tile
//   non-fused, custom kernel                        → flat_n_tile_custom
//   fused swiglu, wide,  standard backend           → flat_n_tile_fused_swiglu_oai
//   fused swiglu, wide,  custom kernel              → flat_n_tile_fused_swiglu_oai_custom
//   fused swiglu, tight, standard backend           → flat_n_tile_fused_swiglu_oai_tight
//   fused swiglu, tight, custom kernel              → flat_n_tile_fused_swiglu_oai_tight_custom
//   fused silu_and_mul, wide,  standard backend     → flat_n_tile_fused_silu_and_mul
//   fused silu_and_mul, wide,  custom kernel        → flat_n_tile_fused_silu_and_mul_custom
//   fused silu_and_mul, tight, standard backend     → flat_n_tile_fused_silu_and_mul_tight
//   fused silu_and_mul, tight, custom kernel        → flat_n_tile_fused_silu_and_mul_tight_custom
//   fused gelu_and_mul, wide,  standard backend     → flat_n_tile_fused_gelu_and_mul
//   fused gelu_and_mul, wide,  custom kernel        → flat_n_tile_fused_gelu_and_mul_custom
//   fused gelu_and_mul, tight, standard backend     → flat_n_tile_fused_gelu_and_mul_tight
//   fused gelu_and_mul, tight, custom kernel        → flat_n_tile_fused_gelu_and_mul_tight_custom
//
// Sequential bypasses the custom kernel and the fused-epilogue / tight
// machinery, so it gets a distinct label regardless of the upstream
// flags — keeps profiler / benchdnn output honest about what actually
// ran.
//
// The "wide / tight standard backend" silu_and_mul and gelu_and_mul
// labels are included for completeness but are unreachable in
// production today: when `fused_act ∈ {silu_and_mul, gelu_and_mul}`
// and the custom kernel is OFF, the parallel dispatcher already
// translates silu/gelu → none upstream (see the CK gate inside
// `a3_can_fuse_act`).  The labeler defends itself with the
// `use_custom == false` branch so a future standard-backend
// silu/gelu OOP writer or an unusual code path still gets a sane
// label.
//
// `plan_tight_fused_epilogue` is the plan's per-call switch (the
// non-custom scratch+OOP writer).  `entry_tight_fused_epilogue` is the
// `ldc[0] < N[0]` detection at flat_n_tile entry.  We label as
// "tight" when EITHER of these is true while fused_epilogue holds:
// the non-custom branch sets `plan.tight_fused_epilogue`, the custom
// branch leaves it false but the kernel always writes compacted I
// cols when activation is gated, so `(entry_tight && use_custom)`
// is the equivalent signal there.
inline const char *gemm_mode_label(GroupNTileStrategy strategy,
                                   grp_matmul_gated_act_t fused_act,
                                   bool fused_epilogue,
                                   bool plan_tight_fused_epilogue,
                                   bool entry_tight_fused_epilogue,
                                   bool use_custom) {
  if (strategy == GroupNTileStrategy::Sequential) {
    return "flat_n_tile_sequential";
  }
  if (fused_epilogue) {
    const bool tight =
        plan_tight_fused_epilogue
        || (entry_tight_fused_epilogue && use_custom);
    if (fused_act == grp_matmul_gated_act_t::silu_and_mul) {
      if (tight) {
        return use_custom
            ? "flat_n_tile_fused_silu_and_mul_tight_custom"
            : "flat_n_tile_fused_silu_and_mul_tight";
      }
      return use_custom
          ? "flat_n_tile_fused_silu_and_mul_custom"
          : "flat_n_tile_fused_silu_and_mul";
    }
    if (fused_act == grp_matmul_gated_act_t::gelu_and_mul) {
      if (tight) {
        return use_custom
            ? "flat_n_tile_fused_gelu_and_mul_tight_custom"
            : "flat_n_tile_fused_gelu_and_mul_tight";
      }
      return use_custom
          ? "flat_n_tile_fused_gelu_and_mul_custom"
          : "flat_n_tile_fused_gelu_and_mul";
    }
    // Default fused-epilogue label set is swiglu_oai_mul.
    if (tight) {
      return use_custom
          ? "flat_n_tile_fused_swiglu_oai_tight_custom"
          : "flat_n_tile_fused_swiglu_oai_tight";
    }
    return use_custom
        ? "flat_n_tile_fused_swiglu_oai_custom"
        : "flat_n_tile_fused_swiglu_oai";
  }
  return use_custom ? "flat_n_tile_custom" : "flat_n_tile";
}

} // namespace

// =====================================================================
// Section D — Public entry
// =====================================================================
//
// flat_n_tile is the single entry point exposed to the dispatcher.
// It only wires the inputs into a `GroupNTileContext`, asks the
// planner for a strategy + parameters, and dispatches to the matching
// executor.  All workload-specific knobs are concentrated in the
// planner (Section B); per-strategy execution lives in Section C.
//
// Adding a new dtype tuning (e.g. INT8): adjust the planner's tile
// constants / thread budgets / strategy thresholds — Sections C and D
// stay unchanged.
//
// fused_act:
//   none           → legacy N-tile only; the caller runs any gated
//                    activation as a separate pass afterward.
//   swiglu_oai_mul → N-tile + per-thread interleaved-pair epilogue.
//   (other values are treated as none; see a3_can_fuse_act.)
// act_dtype:
//   Element type of the output buffer when fusing; unused when
//   fused_act == none.
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
    grp_matmul_gated_act_t fused_act,
    data_type_t act_dtype,
    const char **gemm_mode_out) {

  const int num_ops = static_cast<int>(M.size());
  if (num_ops == 0 || num_threads <= 0) return;

  // Engage the per-thread fused epilogue only for activations whose
  // interleaved layout puts complete (g, u) pairs on every thread's
  // tile.  Today:
  //   * `swiglu_oai_mul` — caller-side interleaved input.
  //   * `silu_and_mul`   — split-halves input; prepack permutes
  //     source columns so the CK pack arena physically matches the
  //     swiglu_oai_mul layout, and the in-register epilogue applies
  //     via `silu_and_mul_store_pair`.
  //   * `gelu_and_mul`   — same prepack-permuted layout as silu;
  //     the in-register epilogue applies via
  //     `gelu_and_mul_store_pair` (gelu_tanh polynomial form,
  //     within BF16 tolerance of the reference's gelu_erf).
  // Everything else falls through the legacy path (separate post-pass
  // activation over the wide [M, N] arena).
  const bool fused_epilogue =
      (fused_act == grp_matmul_gated_act_t::swiglu_oai_mul)
      || (fused_act == grp_matmul_gated_act_t::silu_and_mul)
      || (fused_act == grp_matmul_gated_act_t::gelu_and_mul);

  // Tight-dst detection for the fused-epilogue path.  Caller's dst is
  // a tight [M, I]-layout buffer when ldc < N (the activation halves
  // N, so I = N/2).  Inferred from expert 0's stride; the symmetric
  // uniform-layout guard immediately below re-verifies that every
  // OTHER active expert agrees with that inference.
  const bool tight_fused_epilogue =
      fused_epilogue && !M.empty() && ldc[0] < N[0];
  // Always-on SYMMETRIC uniform-layout guard (defense-in-depth).
  //
  // The caller boundary (`validate_group_matmul_direct_inputs`)
  // rejects mixed tight/wide callers, but `flat_n_tile` can also be
  // reached from the fused-MoE executor and future internal paths,
  // so re-verify here.  Two failure modes must be caught:
  //
  //   1. tight_fused_epilogue == true (inferred from ldc[0] < N[0])
  //      AND some later active expert has ldc[e] >= N[e]:
  //      the executors' tight path writes N/2 cols at each expert's
  //      stride; that wide expert's second half is left untouched
  //      — silent wrong result downstream.
  //
  //   2. tight_fused_epilogue == false (inferred wide from ldc[0]
  //      >= N[0]) AND some later active expert has ldc[e] < N[e]:
  //      the wide path writes N cols at each expert's stride; that
  //      tight expert has rows of physical length ldc[e] < N, so
  //      writing N cols overruns the next row — OOB / memory
  //      corruption.
  //
  // Check both directions: every active expert's local tight/wide
  // classification must match the global one inferred from ldc[0].
  // Refusal is the only memory-safe release-mode reaction; we log
  // loudly via apilog_error so the (should-never-happen-in-practice)
  // reach gets a filable signal rather than an unexplained crash /
  // corruption downstream.  The gate is `fused_epilogue` (not the
  // narrower `tight_fused_epilogue`) so the wide-inferred case in
  // failure mode (2) is also caught.
  // Single cached `apilog_error_enabled()` probe shared by both
  // bail-out sites in this validator loop AND the alloc-fail apilog
  // at end of function.  Gating directly on the err level (not on
  // `apilog_warning_enabled()`) ensures ERROR-only runs
  // (`ZENDNNL_API_LOG_LEVEL=error`) still emit these abort-class
  // messages.  Skips the variadic argument evaluation when no
  // sink is listening — these are abort-class paths so the cost is
  // a one-time `mov+test` in the hot fused-MoE call.
  static const bool s_flat_n_tile_err_log = apilog_error_enabled();

  if (fused_epilogue) {
    for (int e = 0; e < num_ops; ++e) {
      if (M[e] <= 0) continue;
      const bool e_is_tight = (ldc[e] < N[e]);
      if (e_is_tight != tight_fused_epilogue) {
        if (s_flat_n_tile_err_log) {
          apilog_error(
              "[flat_n_tile] mixed tight/wide ldc across experts at e=", e,
              " (ldc[e]=", ldc[e], ", N[e]=", N[e],
              ", local=", (e_is_tight ? "tight" : "wide"), ")"
              "; layout inferred ",
              (tight_fused_epilogue ? "tight" : "wide"),
              " from ldc[0]=", ldc[0], " vs N[0]=", N[0],
              ".  Refusing to run — the caller-boundary validator "
              "should have rejected this combination upstream.  The "
              "caller's dst buffer(s) are unmodified by this call.");
        }
        return;
      }
      // Tight-path divisibility precondition: the executors store
      // exactly N/2 activated cols per row, so N must be even and
      // ldc must equal N/2 (any larger stride would leave gaps; any
      // smaller would overrun).  Skipped on the wide path where ldc
      // simply needs to be >= N (validator-checked upstream).
      if (e_is_tight && (ldc[e] != N[e] / 2 || (N[e] % 2) != 0)) {
        if (s_flat_n_tile_err_log) {
          apilog_error(
              "[flat_n_tile] tight expert e=", e, " violates the "
              "tight-arena stride contract: ldc[e]=", ldc[e],
              " must equal N[e]/2=", (N[e] / 2),
              " and N[e]=", N[e], " must be even.  Refusing to run.");
        }
        return;
      }
    }
  }

  const matmul_algo_t algo = resolve_kernel();
  int nr_align = backend_n_align(algo);

  const size_t wei_elem = size_of(params[0].dtypes.wei);
  const size_t dst_elem = size_of(params[0].dtypes.dst);
  // `bias_elem` is used by do_tile() to offset a non-null bias pointer
  // by col_start.  When the caller didn't declare a bias dtype
  // (`dtypes.bias == none`) the bias pointer is also null on every
  // expert, so `bias_elem` is never actually dereferenced — the
  // `sizeof(float)` fallback is just a safe non-zero placeholder that
  // keeps the arithmetic well-defined and doesn't produce a divide-
  // by-zero / shift-by-zero anywhere downstream.  A real bias dtype
  // overrides this with the correct element width.
  const size_t bias_elem = (params[0].dtypes.bias != data_type_t::none)
      ? size_of(params[0].dtypes.bias) : sizeof(float);

  // ── Custom BF16 microkernel opt-in (ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1) ──
  // Engage for both the non-fused path (act=none — plain matmul tile)
  // AND for the inline gated-act fused epilogue (act ∈
  // {swiglu_oai_mul, silu_and_mul, gelu_and_mul}).
  //
  // All three gated acts write the activated `[M, :I]` output at
  // stride `ldc = 2I` from a half-width pair-pack store helper.
  // swiglu_oai_mul uses the caller-supplied interleaved W13 directly;
  // silu_and_mul and gelu_and_mul re-interleave canonical split-
  // halves W13 at prepack time so the CK arena physically matches
  // the swiglu layout (silu and gelu share the same prepack
  // permutation; only the kernel-side activation math differs).
  //
  // What we pass to `prepare_for_call`:
  //   * act=none      when `fused_epilogue=false` (plain GEMM tile).
  //   * act=fused_act when `fused_epilogue=true`  (one of the three
  //                   gated kinds the CK dispatcher accepts;
  //                   refusal is logged by `prepare_for_call` and
  //                   the call falls back to standard backend).
  // `engage_ntile_custom_kernel` (group_matmul_parallel_common.hpp)
  // does the env check, dispatcher hand-off, and contract gating in
  // one helper shared with the fused-MoE Op1 executor.  When it
  // returns with `kctx.enabled = false` (env off, or dispatcher
  // refused — alpha != 1, beta != 0, non-bf16 dst, transA, +bias on
  // silu/gelu, etc.) we stay on the standard execute_expert_slice
  // path, with the tight split-halves fallback above forcing
  // Sequential routing for silu/gelu so the activation is applied
  // correctly.
  const grp_matmul_gated_act_t custom_act = fused_epilogue
      ? fused_act
      : grp_matmul_gated_act_t::none;
  custom_kernel::CallContext kctx;
  engage_ntile_custom_kernel(
      custom_act,
      /*src_dtype=*/params[0].dtypes.src,
      /*wei_dtype=*/params[0].dtypes.wei,
      /*dst_dtype=*/params[0].dtypes.dst,
      act_dtype,
      /*bias_dtype=*/params[0].dtypes.bias,
      transA, transB, M, N, K, ldb, alpha, beta, weight,
      is_weights_const, kctx);

  // Custom-kernel engagement guard for the wide swiglu path.
  //
  // Correctness: the custom swiglu ukernel writes only the COMPACTED
  // half-width activated result (cols [0, I) with stride ldc).  In
  // tight fused-epilogue mode (ldc == I) that matches the caller's
  // buffer exactly — every allocated column is filled.  In wide mode
  // (ldc == 2I, the caller owns a full [M, 2I] matmul buffer), the
  // second half of each row [I, 2I) is NEVER written by the custom
  // kernel, so it retains whatever bytes the caller's buffer held
  // before the call (zeros if memset, or stale data otherwise).
  //
  // That is a problem when a downstream stage reduces or otherwise
  // consumes cols [I, 2I): e.g. a caller chaining
  // `group_matmul_direct(gated_act=swiglu) + moe_postop` with
  // `moe_D = N[0] = 2I` would sum the untouched second half into its
  // weighted-reduce result, producing garbage.  The non-custom wide
  // path has subtly different but equally wrong semantics (it leaves
  // raw matmul FP32→BF16 bytes in the second half), so either way
  // that caller combination is not well-defined — but we choose NOT
  // to introduce the custom kernel into that code path so it keeps
  // using the existing (deterministic, if semantically wrong)
  // behavior.  Callers that need swiglu + postop correctness should
  // either (a) use fused_moe mode (postop D = N_down, second half
  // never read), or (b) use tight layout (no second half to worry
  // about).
  //
  // Tight layout is detected at the flat_n_tile entry point via
  // `tight_fused_epilogue = fused_epilogue && ldc[0] < N[0]`.
  const bool use_custom =
      kctx.enabled
      && (!fused_epilogue || tight_fused_epilogue);

  // Log the two distinct "kctx.enabled but use_custom=false" paths so
  // the operator shows the downgrade root cause.  A silent kernel=
  // standard line would leave debuggers guessing whether the env is
  // off, the dispatcher refused, or the wide-swiglu guard fired.
  if (kctx.enabled && !use_custom) {
    static const bool s_skip_log = apilog_info_enabled();
    if (s_skip_log) {
      apilog_info("[GRP_MATMUL.PLAN.SKIP_CUSTOM] reason="
                  "wide_swiglu_correctness_guard "
                  "(fused_epilogue=1 tight=0 → custom writes "
                  "compacted [M,I] into caller's [M,2I] buffer, "
                  "leaving cols [I,2I) uninitialised; downstream "
                  "moe_postop on full 2I would reduce garbage). "
                  "FALLBACK to kernel=standard wide matmul + "
                  "separate activation pass.");
    }
  }

  // Widen the per-thread N-slice floor to the custom kernel's pack_nr
  // when engaged (no-op otherwise).  Two `pair_aligned` regimes:
  //   * Wide fused epilogue (ldc ≥ N, non-custom): activation runs as
  //     a separate row-split pass AFTER the matmul OMP barrier, so
  //     per-thread column boundaries can be odd without corrupting
  //     swiglu pairs (activation reads whole rows).
  //   * Tight fused epilogue (ldc < N, non-custom): activation is
  //     fused into `do_tile()` as per-thread scratch + OOP swiglu;
  //     the OOP writer packs at `col_start / 2`, so `col_start` MUST
  //     be even (pair-aligned) across all threads' splits.
  // When the custom kernel is engaged, `kctx.pack_nr` (32 or 64) is
  // already even, so pair-alignment is implicit regardless of layout.
  const bool tight_pair_align = tight_fused_epilogue && !use_custom;
  nr_align = ntile_effective_nr_align(
      nr_align, kctx, /*pair_aligned=*/tight_pair_align);

  // Generic ahead-of-time weight pre-pack for ALGO 3.  Warms both
  // the AOCL DLP reorder cache and (when BF16 + custom-kernel env
  // on) the BF16 custom-kernel pack cache, since flat_n_tile picks
  // between the two per call.  Idempotent across calls.  Under the
  // uniform-eager semantic, PREPACK=ON (the default) warms the
  // firing experts even when the framework hasn't opted into the
  // `total > active` contract — legacy callers see a one-time
  // first-iter serial reorder cost in exchange for warm caches on
  // every subsequent call.  Set `ZENDNNL_GRP_MATMUL_PREPACK=0` to
  // restore the strict pre-PR / lazy-only behaviour.
  //
  // Placed after `nr_align` is finalised (post `kctx` engagement) so
  // the AOCL DLP per-tile warmer can mirror `do_tile()`'s
  // `aligned_n_split(N[e], stable, tid, nr_align)` exactly — the
  // per-tile cache key includes `n_tile` (= sliced N), which depends
  // on `nr_align`, so an early call (pre-kctx) would warm a key set
  // the runtime never queries when the dispatcher widens nr_align to
  // the custom kernel's pack_nr or the tight-pair-align floor.
  group_matmul_prepack::prepack_for_algo_3(
      group_matmul_prepack::build_prepack_params(
          weight, K, N, ldb, transB, is_weights_const, params, M,
          get_grp_matmul_custom_kernel(),
          /*num_threads=*/num_threads,
          /*nr_align=*/nr_align,
          fused_act, act_dtype,
          /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

  // APILOG moved below after the plan is built so the log line can
  // surface both the env-selected and the auto-resolved N_ORDER
  // sub-mode (when env=0/auto) in one place.
  //
  // `gemm_mode_out` is also written after the plan is built (below)
  // so it reflects the strategy the planner actually picked.  The
  // Sequential strategy (picked when N is too small for a useful tile
  // split) calls execute_expert_slice directly and BYPASSES the custom
  // kernel entirely — so reporting `*_custom` in that case would
  // mislead profilers / benchdnn into thinking the microkernel ran
  // when it did not.  The post-plan labelling downgrades to
  // `flat_n_tile_sequential` in that case regardless of use_custom.

  scoped_active_levels guard(1);

  // Per-thread scratch alloc-fail flag, set inside the tight-branch
  // of do_tile on an unrecoverable `posix_memalign` failure.  Checked
  // once after the OMP region exits so the caller gets a clear error
  // instead of silently-wrong output.  Zero-initialised; allocated
  // here (and passed into ctx by pointer) so the atomic lives at a
  // stable address across the parallel region.
  std::atomic<int> alloc_fail{0};

  // ── Hoisted dynamic-quant source reorder (per-expert, pre-OMP) ─────
  // For every active expert with `params[e].dynamic_quant == true`,
  // run `reorder_quantization_wrapper` here — ONCE, single-threaded
  // over experts but with the full thread team driving the internal
  // reorder kernel — and stash the resulting (S8 src, src_scale,
  // src_zp) tuple in `hoisted[e]`.  The per-tile OMP threads inside
  // `do_tile()` then read from those shared buffers (read-only
  // across the team), avoiding both the caller-shared-buffer race
  // and the `num_threads × O(M × K)` duplicated reorder work that
  // would otherwise happen if each thread ran the wrapper inside
  // `execute_expert_slice`.
  //
  // `hoist_buffers` owns the malloc'd S8 / scale / zp buffers via
  // RAII; both vectors live on this function's stack for the entire
  // duration of the OMP region below, so the shared reads remain
  // valid.  When a slot stays `.valid = false` (no dynamic_quant on
  // that expert, or the wrapper short-circuited because the dtype
  // combo wasn't eligible), `do_tile` and `execute_sequential` fall
  // back to the caller's original `src[e]` / `lda[e]`.
  //
  // Reached only when `check_n_tile_extra` accepted `dynamic_quant`,
  // which is paired with `check_m_tile_safe`'s row-local granularity
  // gate (`src_scale.dims[0] == M[i]`).  After both gates the only
  // surviving src granularity is per-token `{M[i], 1}` (including
  // the `M[i] == 1` decode case where dims are `{1, 1}` and the
  // hoist runs a single-row reorder).  Per-tensor on M > 1 inputs,
  // per-column, per-channel-on-src, and per-group `{M[i], G}` on K
  // (which `check_m_tile_safe` would accept but `check_n_tile_extra`
  // rejects) all route to ALGO 1 instead.
  std::vector<reorder_quant_buffers_t> hoist_buffers(num_ops);
  std::vector<HoistedSrcQuant> hoisted(num_ops);
  bool any_hoist = false;
  for (int e = 0; e < num_ops; ++e) {
    if (M[e] <= 0 || !params[e].dynamic_quant) continue;

    // Single-shot reorder: build a local `shadow` of `params[e]`
    // because the wrapper mutates its `matmul_params &` argument
    // (`dtypes.src`, `quant_params.src_scale.buff`, etc.).  We do
    // NOT want to touch the caller's shared `params[e]` — the per-
    // tile threads will copy from it independently and then layer
    // the hoisted state on top via `tile_params = params[e]`
    // followed by the `(*hoisted_src_quant)[e]` overrides.
    matmul_params shadow = params[e];
    const void *src_e = src[e];
    int reordered_lda = lda[e];
    size_t src_type_size = size_of(shadow.dtypes.src);
    matmul_batch_params_t bp;
    bp.Batch_A = 1;
    bp.Batch_B = 1;

    const status_t s = reorder_quantization_wrapper(
        src_e, lda[e], reordered_lda, src_type_size,
        shadow, bp, transA[e], M[e], K[e],
        num_threads, hoist_buffers[e]);

    if (s != status_t::success) {
      // Validation failure (bad dims, missing required buf for an
      // asymmetric u8 quant, etc.).  Without a usable hoisted src
      // we cannot proceed — the per-thread wrapper inside
      // `execute_expert_slice` would either fail the same way or
      // race on caller buffers.  Surface the error and abort the
      // call; the caller's dst is left untouched (no OMP work has
      // started yet).
      apilog_error(
          "[flat_n_tile] hoisted dynamic-quant source reorder failed "
          "for expert ", e, " — aborting call; caller's dst is "
          "untouched.  See preceding `reorder_quantization_wrapper` "
          "error for the granularity / dtype mismatch.");
      return;
    }

    // `reorder_quantization_wrapper` returns success without doing
    // anything when the dtype combo isn't eligible (e.g. caller set
    // `dynamic_quant=true` but `dtypes.wei != s8`).  Detect the
    // no-op case via the unchanged src dtype and leave the slot
    // invalid; `do_tile` will fall through to the caller's bf16/f32
    // src and the per-thread wrapper will short-circuit identically.
    if (shadow.dtypes.src != params[e].dtypes.src) {
      hoisted[e].valid     = true;
      hoisted[e].src_ptr   = src_e;
      hoisted[e].lda       = reordered_lda;
      hoisted[e].src_dtype = shadow.dtypes.src;
      hoisted[e].src_scale = shadow.quant_params.src_scale;
      hoisted[e].src_zp    = shadow.quant_params.src_zp;
      any_hoist = true;
    }
  }

  GroupNTileContext ctx{
      layout, transA, transB,
      M, N, K, alpha,
      src, lda, weight, ldb, bias, beta, dst, ldc,
      is_weights_const, params,
      fused_act, act_dtype,
      wei_elem, dst_elem, bias_elem,
      use_custom, use_custom ? &kctx : nullptr,
      &alloc_fail,
      any_hoist ? &hoisted : nullptr
  };

  const GroupNTileTopology topo =
      summarise_topology(M, N, K, num_threads, wei_elem);
  // Pass `use_custom` so the planner can short-circuit to the
  // strict-stable plan for the non-custom path.
  // The custom path keeps the legacy cost-model picks because its
  // pack cache is shape-keyed, not tile-keyed.
  GroupNTilePlan plan =
      plan_group_n_tile(topo, algo, nr_align, fused_epilogue,
                        use_custom, M, N);
  // The tight-dst switch is orthogonal to the planner's strategy /
  // threading decisions: it only toggles how each thread writes its
  // final swiglu output (scratch + OOP vs in-place).  Set after the
  // plan is built so the planner's shape-driven choices are unaffected.
  // Sequential handles tight callers via its own scratch+OOP path
  // (see execute_sequential()) so this flag is set whenever the
  // caller is tight and the custom kernel isn't engaged.  The flag is
  // consumed by DecodeD / FewExperts / ManyExperts to route do_tile
  // through the per-thread scratch + OOP swiglu code; Sequential
  // detects the same `ldc[e] < N[e]` condition locally and doesn't
  // depend on this flag.
  plan.tight_fused_epilogue = tight_fused_epilogue && !use_custom;

  // ── Tight split-halves fallback (silu_and_mul / gelu_and_mul) ────
  // Correctness gate for the CK-refusal path on split-halves gated
  // activations with a tight caller (`ldc < N`).
  //
  // ── Why the per-thread do_tile path is unsafe here ──────────────
  // do_tile's tight branch was originally swiglu-only.  Its per-
  // thread scratch holds `[M, n_tile]` covering columns
  // `[col_start, col_start + n_tile)` of the LOGICAL output.  For
  // `swiglu_oai_mul` (interleaved layout) those cols always come in
  // adjacent (gate, up) pairs within the slice, so the OOP pair-pack
  // helper can deinterleave and fold gate*up locally.
  //
  // For `silu_and_mul` / `gelu_and_mul` the W13 layout is canonical
  // split-halves `[gate_cols=0..I) | up_cols=[I..N)]`.  Each thread's
  // `n_tile` slice lands EITHER entirely in the gate half OR
  // entirely in the up half — never both — so the per-thread scratch
  // does not have the pair of values the activation needs.  Applying
  // any in-place gated-act helper on `scratch[M, n_tile]` would
  // either pair gate-with-gate (wrong) or up-with-up (wrong).  The
  // CK in-register fused epilogue sidesteps this by having the
  // prepack pre-interleave the weight columns — so the kernel sees
  // the swiglu_oai_mul layout regardless of caller-side convention.
  // When CK refuses (bias on silu/gelu, transA, alpha ≠ 1, etc.) the
  // prepack-interleaved arena is absent and the per-thread tight
  // branch has no correct path forward.
  //
  // ── The fix ─────────────────────────────────────────────────────
  // Force the Sequential strategy for this specific combination.
  // `execute_sequential` allocates a per-expert wide
  // `[M, N]` scratch, runs the matmul wide (so both gate and up
  // halves are present on the same buffer), applies the activation
  // in-place via `apply_gated_act_inplace` (which dispatches to the
  // AVX-512 silu / gelu row helpers), then memcpys the activated
  // half into the caller's tight `[M, I]` dst.  See the matching
  // branch in `execute_sequential` for the row-by-row copy.
  //
  // ── Production envelope ─────────────────────────────────────────
  // The only realistic production trigger for this fallback is
  // `silu_and_mul / gelu_and_mul + bias` (the CK fused epilogue
  // refuses biased calls; bias-into-init under the prepack-permuted
  // layout is a planned follow-up).  Qwen3, Mixtral, DBRX, DeepSeek
  // are all bias-free on W13 so the fallback never fires on those.
  // Test-only refusal triggers (transA = true, alpha != 1, non-const
  // weight) are covered by the same Sequential routing.
  //
  // ── Cost ────────────────────────────────────────────────────────
  // Sequential is single-expert-at-a-time but each expert can still
  // use the full thread pool internally (via `execute_expert_slice`).
  // The cross-expert parallelism is lost but the activation is
  // applied correctly — correctness over speed for a refusal path.
  if (plan.tight_fused_epilogue
      && (fused_act == grp_matmul_gated_act_t::silu_and_mul
          || fused_act == grp_matmul_gated_act_t::gelu_and_mul)) {
    plan.strategy = GroupNTileStrategy::Sequential;
    // Clear the tight-fused flag so a future do_tile invocation
    // (e.g. via a downstream caller that misroutes back here) can't
    // accidentally re-enter the unsafe swiglu-only OOP path.
    plan.tight_fused_epilogue = false;
    static const bool s_fallback_log = apilog_info_enabled();
    if (s_fallback_log) {
      apilog_info("[GRP_MATMUL.PLAN.FALLBACK] tight_split_halves "
                  "act=", act_name(fused_act),
                  " reason=CK refused on tight split-halves caller "
                  "(likely silu/gelu + bias, or per-call gate mismatch); "
                  "routing to Sequential strategy with wide scratch + "
                  "apply_gated_act_inplace + tight memcpy.");
    }
  }

  // ── Test-only snapshot of the finalised plan ─────────────────────
  // White-box hook for `gtests/group_matmul/test_algos.cpp` to
  // assert on Phase B's heaviest-first / eligibility-filter output
  // (per-expert thread counts and `per_expert_remainder` flag) —
  // the end-to-end correctness comparison cannot observe these
  // because the per-expert thread distribution does not change the
  // final GEMM values.  See the `test_api::` block in
  // `group_matmul_n_tile.hpp` for the snapshot struct definition.
  //
  // CONCURRENCY MODEL — EXPLICITLY SINGLE-THREAD-ONLY.
  //   The capture flag (`s_capture_phase_b`) is atomic so that the
  //   production read path can do a cheap relaxed load without
  //   tearing.  The snapshot payload itself (`s_last_phase_b_snapshot`)
  //   is a plain non-atomic global, written field-by-field below.
  //   A concurrent reader that polls `snap.valid` from another
  //   thread while flat_n_tile writes the payload would observe a
  //   data race even with the `valid`-published-last ordering: the
  //   payload writes lack release semantics and the reader's
  //   `valid` load lacks acquire semantics.  Use exclusively from
  //   single-threaded gtests (`PhaseBCaptureGuard` armed on the
  //   test thread, snapshot read after the dispatcher returns).
  //   Promoting to cross-thread use would require an acquire-load
  //   on the reader side plus release-store on the publish side,
  //   or wrapping the snapshot in `std::atomic` / a mutex.
  //
  // Production hot-path cost — kept negligible by the two-step probe:
  //   In production `s_capture_phase_b == false`, so the outer
  //   relaxed load returns a value cached in each core's L1 in the
  //   Shared coherence state — no cache-line invalidation, no
  //   contention even with many concurrent dispatchers.  Only when
  //   a test arms capture does the inner `exchange(false, acq_rel)`
  //   fire, which performs the one-shot disarm.  The atomic RMW
  //   therefore costs O(1) per capture cycle (test-only) instead of
  //   O(1) per group_matmul call (production).
  if (test_api::s_capture_phase_b.load(std::memory_order_relaxed)
      && test_api::s_capture_phase_b.exchange(
          false, std::memory_order_acq_rel)) {
    auto &snap = test_api::s_last_phase_b_snapshot;
    // Publish payload before flipping `valid`.  Single-thread-only
    // (see CONCURRENCY MODEL above) so this is invariant-style
    // discipline rather than a cross-thread memory-ordering
    // guarantee — but the discipline makes the future cross-thread
    // refactor easier (only the load/store ordering needs to change,
    // not the field-write sequence).
    snap.per_expert_remainder = plan.per_expert_remainder;
    snap.strategy             = plan.strategy;
    snap.batch_size           = plan.batch_size;
    snap.n_thr_fixed          = plan.n_thr_fixed;
    snap.num_ops_active       = num_ops;
    snap.stable_n_thr_per_expert = plan.stable_n_thr_per_expert;
    snap.valid                = true;   // published last (single-thread invariant)
  }

  // Surface the concrete path to the caller via `gemm_mode_out`.
  // Static literals only — see `gemm_mode_label` above for the
  // full set of possible values and the labelling rationale.
  if (gemm_mode_out != nullptr) {
    *gemm_mode_out = gemm_mode_label(plan.strategy,
                                     fused_act,
                                     fused_epilogue,
                                     plan.tight_fused_epilogue,
                                     tight_fused_epilogue,
                                     use_custom);
  }

  // ── APILOG: one line per flat_n_tile call ──────────────────────────
  // Shows the act kind, whether the custom kernel engaged, the pack
  // parameters, AND the N_ORDER sub-mode actually used — either the
  // env-selected value or, when env=0 (auto), the sub-mode the
  // auto-picker resolved from shape.  apilog_info_enabled() is
  // cached; the check is free when the log level is below info.
  // Variadic `apilog_info` composes via the library's stringstream
  // logger; no C-style formatted output.
  static const bool s_apilog = apilog_info_enabled();
  if (s_apilog) {
    // Hoist the three env-cache getters that the variadic apilog
    // line below references — `apilog_info(...)` evaluates each
    // argument expression once at the call site, but parking them
    // in named locals up front makes the call's read-set explicit
    // (each underlying getter still costs a single relaxed atomic
    // load now that they all cache + override, so the savings are
    // microscopic — clarity is the deliverable).  Same hoist
    // pattern we applied to `decode_n_tile_snapshot` in the planner.
    const int  log_n_tile_heavy_thr    = get_grp_matmul_n_tile_heavy_threshold();
    const int  log_aocl_target_slots   = get_grp_matmul_aocl_target_slots();
    const int  log_aocl_blis_nc        = get_grp_matmul_aocl_blis_nc();
    const int env_order = get_grp_matmul_n_order();
    const bool is_auto_resolved =
        (env_order == 0 && plan.auto_resolved_order >= 0);
    const char *strategy_name =
        (plan.strategy == GroupNTileStrategy::Sequential)   ? "Sequential"
      : (plan.strategy == GroupNTileStrategy::DecodeD)      ? "DecodeD"
      : (plan.strategy == GroupNTileStrategy::FewExperts)   ? "FewExperts"
      : (plan.strategy == GroupNTileStrategy::ManyExperts)  ? "ManyExperts"
      :                                                      "unknown";
    // `path` distinguishes the AOCL strict-stable plan from the
    // custom-kernel cost-model plan — see the path-overview header
    // above `plan_group_n_tile`.  Sequential is reachable from
    // either path's fail-fast / narrow-N escape so it labels itself.
    const bool aocl_strict =
        !use_custom && plan.stable_n_thr_per_expert[0] > 0;
    const char *path_name = (plan.strategy == GroupNTileStrategy::Sequential)
        ? "sequential_fallback"
        : (use_custom        ? "custom_dynamic"
        :  aocl_strict       ? "aocl_strict_stable"
        :                       "aocl_legacy_costmodel");
    apilog_info("[GRP_MATMUL.PLAN] flat_n_tile strategy=", strategy_name,
                " path=", path_name,
                " kernel=", (use_custom ? "custom" : "standard"),
                " act=", act_name(fused_act),
                " fused_epilogue=", (fused_epilogue ? "yes" : "no"),
                " tight=", (plan.tight_fused_epilogue ? "yes" : "no"),
                " batch_size=", plan.batch_size,
                " thr_per_expert=", plan.decode_thr_per_expert,
                " n_thr_fixed=", plan.n_thr_fixed,
                " max_n_thr=", plan.max_n_thr,
                " pack_nr=", (use_custom ? kctx.pack_nr : 0),
                " subtile_cols=", (use_custom ? kctx.subtile_cols : 0),
                " min_n_tile=", plan.min_n_tile,
                " num_ops=", num_ops,
                " num_threads=", num_threads,
                " n_order_env=", env_order,
                " n_order_used=",
                is_auto_resolved ? plan.auto_resolved_order : env_order,
                is_auto_resolved ? " (auto)" : "",
                " stable_n_thr[0]=",
                static_cast<int>(plan.stable_n_thr_per_expert[0]),
                " n_tile_heavy_thr=",
                log_n_tile_heavy_thr,
                " per_expert_remainder=",
                (plan.per_expert_remainder ? "yes" : "no"),
                " aocl_target_slots=",
                log_aocl_target_slots,
                " aocl_blis_nc=",
                log_aocl_blis_nc);
  }

  switch (plan.strategy) {
    case GroupNTileStrategy::Sequential:
      execute_sequential(plan, ctx);
      break;
    case GroupNTileStrategy::DecodeD:
      execute_decode_d(plan, ctx);
      break;
    case GroupNTileStrategy::FewExperts:
    case GroupNTileStrategy::ManyExperts:
      execute_rounds(plan, ctx);
      break;
  }

  // Post-exec failure check for the tight-scratch path.  The
  // OMP-region-internal alloc failure (rare: exhausted per-thread
  // heap in posix_memalign) is benign for threads that did succeed —
  // they wrote their own disjoint dst columns correctly — but the
  // failing thread's columns are undefined.  Elevate to apilog_error
  // so benchdnn / torch-side observers catch the incident; no
  // exception because the surrounding group_matmul API is noexcept
  // and partial-correct output is still safer than undefined behaviour
  // on the caller's dst (which they own).
  if (alloc_fail.load(std::memory_order_relaxed) != 0) {
    if (s_flat_n_tile_err_log) {
      apilog_error(
          "[flat_n_tile] per-thread scratch allocation failed in the "
          "tight-fused-epilogue path; some dst column ranges may be "
          "undefined.  Consider disabling internal-alloc tight mode "
          "(ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT=0) if this recurs.");
    }
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
