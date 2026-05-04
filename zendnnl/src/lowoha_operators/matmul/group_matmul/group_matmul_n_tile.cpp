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
///   Section A  Data structures — Strategy enum, Topology, Plan, and
///              the GroupNTileContext that bundles the global expert
///              vectors with the two per-thread tile primitives
///              (`do_tile`, `apply_swiglu_oai`).
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
///              to the matching strategy executor.
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
#include <cstdlib>
#include <limits>
#include <vector>

#include <omp.h>

#include "group_matmul_parallel_common.hpp"
#include "custom_kernel/dispatch.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using zendnnl::common::bfloat16_t;

namespace {

// =====================================================================
// Per-thread scratch buffer — used by the tight-fused-epilogue path
// =====================================================================
//
// When `fused_epilogue=swiglu` AND the caller's dst is a tight
// [M, I]-layout buffer (ldc < N), the classic matmul-then-in-place-
// compact pattern no longer fits (matmul writes 2I cols but dst only
// has I cols per row).  In that case `do_tile()` switches to the
// per-thread flow that the fused-MoE V2 executor pioneered:
//
//   1. matmul the thread's N-tile slice into a thread-local scratch
//      buffer (wide, `n_tile` cols, ldc=n_tile),
//   2. run `apply_swiglu_oai_tile_rows_oop()` from that scratch into
//      the caller's tight dst at halved col offset `col_start / 2`.
//
// This is barrier-free: each thread writes to its own scratch then to
// a disjoint column range of the caller's tight dst — no cross-thread
// reads needed inside the fused-epilogue step.
//
// Lifetime: `static thread_local` inside the OMP region; monotonically
// grows to the high-water mark this thread has seen.  Freed by the
// destructor on thread exit.  Per-thread, so no contention.
//
// The buffer is 64-byte aligned so the custom kernel's `vmovdqu`
// reads start on a cache-line boundary.
struct PerThreadScratch {
  void *buf = nullptr;
  size_t cap = 0;
  ~PerThreadScratch() { std::free(buf); }
};

// Grow a per-thread scratch to at least `need` bytes, 64-byte aligned.
// Returns false on alloc failure (caller signals via alloc_fail atomic
// + post-OMP-region check).
inline bool grow_scratch(PerThreadScratch &s, size_t need) {
  if (need <= s.cap) return true;
  std::free(s.buf);
  s.buf = nullptr;
  s.cap = 0;
  void *tmp = nullptr;
  if (posix_memalign(&tmp, 64, need) != 0 || tmp == nullptr) return false;
  s.buf = tmp;
  s.cap = need;
  return true;
}

// =====================================================================
// Section A — Data structures
// =====================================================================

// Execution patterns the planner can pick.  When N-tile is not
// viable we run experts sequentially with the full thread team per
// kernel (equivalent to ALGO 1's behaviour), which is the safest
// fallback for shapes where N is too small to split usefully across
// threads.
enum class GroupNTileStrategy {
  // (F)  N-tile not viable.  Run experts sequentially with
  //      `num_threads` per kernel.
  Sequential,

  // (D)  Decode parallel — small max_M, balanced M, num_ops ≤ num_ccds.
  //      Each expert gets an equal CCD-sized team.
  DecodeD,

  // (A)  Few experts (num_ops ≤ num_ccds, non-decode):
  //      L3-aware batches, proportional thr_per_expert per round.
  FewExperts,

  // (B)  Many experts (num_ops > num_ccds):
  //      L3-aware barrier-synchronized rounds, fixed n_thr/expert.
  ManyExperts,
};

// Inputs / topology summary used by the planner — only what's needed
// to decide; no expert vectors are inspected at strategy level.
struct GroupNTileTopology {
  int num_ops;
  int num_threads;
  int ccd_size;          // = min(8, num_threads)
  int num_ccds;          // ceil(num_threads / ccd_size)
  int max_M;
  int max_N;
  int max_K;
  int min_M_active;      // smallest positive M, or max_M if all empty
  size_t wei_elem;       // weight bytes per element
  size_t wei_per_expert; // = max_N * max_K * wei_elem, precomputed once
                         // so compute_l3_batch / compute_target_batch /
                         // the R2 weight-class gate don't each re-derive
                         // it from the three inputs.
};

// Knobs the strategy executors consume.  Most fields are filled only
// for the strategy they belong to; the rest stay zero / default.
struct GroupNTilePlan {
  // Always-valid fields
  GroupNTileStrategy strategy = GroupNTileStrategy::Sequential;
  matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked;
  int  num_threads = 1;
  int  nr_align = 1;
  bool fused_epilogue = false;

  // Per-thread N-slice floor (== `min_n_tile` argument of
  // ctx.do_tile).  Set for DecodeD / FewExperts / ManyExperts.
  int min_n_tile = 1;

  // (D) DecodeD parameters
  int decode_thr_per_expert = 0;
  int decode_total_threads = 0;

  // (A) FewExperts / (B) ManyExperts shared "rounds" parameters.
  // Threads per expert in a round is `n_thr_fixed` if non-zero
  // (ManyExperts), else min(num_threads / round_size, max_n_thr)
  // (FewExperts — proportional to round_size, capped by N-tile count).
  int batch_size = 0;
  int n_thr_fixed = 0;
  int max_n_thr = 0;

  // Optional permutation of expert indices used by FewExperts /
  // ManyExperts when assigning experts to rounds.  When
  // `expert_order_size == 0` callers use input order; when > 0 the
  // first `expert_order_size` slots of `expert_order` hold the
  // sorted permutation.  Populated by `fill_sorted_expert_order()`
  // when the planner enables M-descending sort: round time is
  // dominated by max(M) within the round, so sorting puts the
  // heaviest experts in the first round (where they would dominate
  // anyway) and pushes the lightest into the last round, minimising
  // sum_round(max_M).
  //
  // Stack-allocated to keep the hot path heap-free.  `kMaxExperts`
  // sets the upper bound on the in-place expert sort; workloads with
  // more experts skip the sort (perf optimisation, not a correctness
  // requirement) and fall through to input-order assignment.
  static constexpr int kMaxExperts = 256;
  std::array<int, kMaxExperts> expert_order{};
  int expert_order_size = 0;

  // When the env `ZENDNNL_GRP_MATMUL_N_ORDER` is 0 (auto), the picker
  // resolves a concrete sub-mode and stashes it here for APILOG
  // transparency.  Left at -1 when the env was explicitly set
  // (no auto-resolution performed).
  int auto_resolved_order = -1;

  // Tight-dst fused-epilogue switch.  Set to true at `flat_n_tile`
  // entry when `fused_epilogue=swiglu` AND the caller's dst is a
  // tight [M, I]-layout buffer (ldc[0] < N[0]).  Triggers the
  // per-thread-scratch + out-of-place swiglu flow inside `do_tile`;
  // when false, the classic matmul-then-in-place-compact wide flow
  // runs.
  //
  // Contract when true: all experts have `ldc[e] == N[e] / 2`
  // (the fused-MoE caller allocates a uniform-stride arena).
  // `execute_*` skip the barrier + `apply_swiglu_oai()` pass because
  // the activation is already fused into `do_tile`.
  bool tight_fused_epilogue = false;
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

  // Number of threads that actually share the work for expert e in a
  // team of `team_size`.  Capped at the number of N-tiles available
  // (N[e] / min_n_tile) so each participant has at least `min_n_tile`
  // columns to chew on; floored at 1.
  //
  // CRITICAL invariant: this formula is shared by `do_tile()` (matmul
  // column split) and `apply_swiglu_oai()` (activation row split).
  // The two MUST agree on which threads participate so that:
  //   1. Every column written by matmul has a corresponding row-split
  //      reader in the activation, and vice versa.
  //   2. Threads with `local_tid >= participating_n_thr` skip BOTH
  //      matmul and activation.
  // Centralising the formula here makes any future tweak land in
  // both code paths automatically.
  inline int participating_n_thr(int e, int team_size,
                                 int min_n_tile) const {
    return std::max(1, std::min(team_size, N[e] / min_n_tile));
  }

  // Per-thread N-slice of expert e's matmul.  The column split is
  // NR-aligned (see aligned_n_split() in the common header).
  //
  // Note for the fused swiglu_oai path: the matmul split is NOT
  // forced onto even / pair boundaries.  Pair alignment used to be a
  // correctness constraint when the epilogue was column-parallel
  // (both halves of each (g, u) pair had to be produced by the same
  // thread).  The current epilogue (apply_swiglu_oai below) splits
  // by rows instead, so a thread reads pair (col 2k, col 2k+1) from
  // any thread's matmul writes within its own row range — both reads
  // are guaranteed visible by the OMP barrier between matmul and
  // activation in the executors, regardless of which thread wrote
  // each column.  Every dst cell is still produced by exactly one
  // thread, so per-cell numerics are unchanged.
  inline void do_tile(const GroupNTilePlan &plan,
                      int e, int local_tid, int team_size,
                      int min_n_tile) const {
    if (M[e] <= 0) return;
    const int n_thr = participating_n_thr(e, team_size, min_n_tile);
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

    // ── Tight-fused-epilogue path (non-custom + swiglu + tight dst) ────
    // Caller's dst is a tight [M, I]-layout buffer (ldc < N).  The
    // classic matmul-then-in-place-compact pattern can't run here (no
    // room for 2I cols in dst).  Switch to per-thread-scratch + OOP
    // activation, cribbed from the fused MoE V2 executor:
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
      assert(fused_act == grp_matmul_gated_act_t::swiglu_oai_mul
             && "tight_fused_epilogue requires swiglu_oai_mul");
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
          src[e], lda[e], w, ldb[e],
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

    // Quantization is blocked upstream by n_tile_safe (quant dims
    // metadata cannot be safely column-sliced without updating dims
    // to match n_tile).  Post-ops with buffers (binary_add/mul) are
    // also blocked by n_tile_safe.  Only buffer-free element-wise
    // activations reach this path.
    execute_expert_slice(layout[e], transA[e], transB[e],
        M[e], n_tile, K[e], alpha[e],
        src[e], lda[e], w, ldb[e],
        b, beta[e], d, ldc[e],
        is_weights_const[e], 1, tile_params, plan.algo);
  }

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
  inline void apply_swiglu_oai(int e, int local_tid, int team_size,
                               int min_n_tile) const {
    if (M[e] <= 0) return;
    // Same formula as do_tile() — see participating_n_thr() above for
    // why this MUST stay in lockstep with the matmul split.
    const int n_thr = participating_n_thr(e, team_size, min_n_tile);
    if (local_tid >= n_thr) return;

    // swiglu_oai requires even N (gate+up = 2 * intermediate_dim).
    // The dispatcher in group_matmul_parallel.cpp enforces this for
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
};

// =====================================================================
// Section B — Planner
// =====================================================================
//
// Pure decision layer for ALGO 3.  All tile / thread / batch choices
// for the supported dtype + shape regimes live here, so future dtype
// tuning (e.g. INT8) is a one-place change.

// Build the topology summary once at the top of flat_n_tile.
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
  t.max_M = *std::max_element(M.begin(), M.end());
  t.max_N = *std::max_element(N.begin(), N.end());
  t.max_K = *std::max_element(K.begin(), K.end());
  t.min_M_active = t.max_M;
  for (int i = 0; i < t.num_ops; ++i)
    if (M[i] > 0) t.min_M_active = std::min(t.min_M_active, M[i]);
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
//   3. The bumped working set is within 10% of aggregate L3 capacity.
//
// Check (3) is the safety net: shapes whose per-round working set
// would massively overflow L3 (very large weights) keep the strict
// L3-aware target and avoid victim-cache thrashing, while shapes
// whose per-round working set sits at or just over L3 take the bump
// because the small overshoot is cheaper than leaving threads idle.
inline int compute_target_batch(const GroupNTileTopology &topo,
                                int l3_batch) {
  const int batch_team_saturating = topo.num_threads / topo.ccd_size;
  int target = std::min(topo.num_ops, std::max(1, l3_batch));
  if (l3_batch < batch_team_saturating
      && batch_team_saturating <= topo.num_ops) {
    const size_t bumped_weight =
        static_cast<size_t>(batch_team_saturating) * topo.wei_per_expert;
    const size_t kL3 = get_grp_l3_total_bytes(topo.num_ccds);
    if (bumped_weight <= kL3 + kL3 / 10) {       // ≤ 110% of L3
      target = batch_team_saturating;
    }
  }
  return target;
}


// (`sort_indices_by_m` is defined in group_matmul_parallel_common.hpp
//  and shared with the V2 fused-MoE Op1 executor.)

// V1-side thin wrapper around the generic `fill_ntile_expert_order()`
// helper (defined in group_matmul_parallel_common.hpp, shared with the
// V2 fused-MoE Op1 executor).  V1 plumbs the result into the stack-
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

// ── Viability check ──────────────────────────────────────────────────
// N-tiling only helps when N is large enough to create useful tiles.
// For few-expert path (A): need tiles ≥ team_size / 2.
// For round path   (B): need tiles ≥ ccd_size / 2 (each round-expert
//                       gets ccd_size threads; below half the team
//                       would idle).
// Use the stricter of the two checks.
inline bool ntile_viable(const GroupNTileTopology &topo) {
  const int team_size_est = topo.num_threads / std::max(1, topo.num_ops);
  const int viability_min_tile =
      (topo.max_M <= kDecodeMaxM) ? effective_decode_n_tile() : kMinNTile;
  const int tiles_available = topo.max_N / viability_min_tile;
  const int min_useful = (topo.num_ops > topo.num_ccds)
      ? std::max(1, topo.ccd_size / 2)
      : std::max(1, team_size_est / 2);
  return tiles_available >= min_useful;
}

// Self-fallback regimes — let a caller force ALGO 3 deployment-wide
// and still get ALGO 1's Sequential plan on shapes where N-tile is
// provably worse, without the framework needing to consult ALGO 0's
// auto-select.  Returns `true` when the plan should be Sequential
// (and sets plan.strategy accordingly).  Uses the same constants
// ALGO 0 uses (kMediumWeight, kDecodeMaxM) so the two code paths
// agree.
//
// (R1) Few experts (≤ 3): the per-expert matmul is large enough that
//      one-expert-at-a-time with the full thread team beats column-
//      parallel across 2-3 experts.  Matches ALGO 0's
//      `if (num_ops <= 3) return 1;` gate at every weight class.
//
// (R2) Large-weight DRAM-streaming (wei_per_expert > kMediumWeight):
//      AOCL DLP's internal panel blocking with the full thread team
//      is near-optimal.  ALGO 3 only beats it in a narrow regime —
//      prompt-class shapes (max_M > kDecodeMaxM) with enough experts
//      (num_ops ≥ 5).  Outside that regime, fall back to Sequential.
inline bool self_fallback_to_sequential(const GroupNTileTopology &topo) {
  // (R1)
  if (topo.num_ops <= 3) return true;
  // (R2) — ntile_viable is the caller's precondition; this is the
  // remaining ALGO-3-wins gate for large weights.
  if (topo.wei_per_expert > kMediumWeight
      && (topo.num_ops < 5 || topo.max_M <= kDecodeMaxM)) return true;
  return false;
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
// The numbers the picker consumes are packaged into this plain
// struct so `plan_group_n_tile` stays a thin orchestrator and the
// candidate-building logic is isolated.
struct RoundCandidates {
  // Shared derivations
  int max_tiles = 0;
  int capped_batch = 0;

  // Per-candidate parameters
  int    n_thr_single     = 0;
  bool   single_eligible  = false;
  double wall_single      = 0.0;

  int    n_thr_multi      = 0;
  int    batch_multi      = 0;
  int    n_rounds_multi   = 0;
  double wall_multi       = 0.0;

  int    balanced_batch   = 0;
  double wall_balanced    = 0.0;
};

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
enum class RoundPick { Single, Multi, Balanced };

inline RoundPick pick_round_strategy(const GroupNTileTopology &topo,
                                     const RoundCandidates &c) {
  const int rounds_mode = get_grp_n_rounds_mode();
  if (rounds_mode == 1)
    return c.single_eligible ? RoundPick::Single : RoundPick::Balanced;
  if (rounds_mode == 2) return RoundPick::Multi;
  if (rounds_mode == 3) return RoundPick::Balanced;

  // ── Thin-single-round correction ─────────────────────────────────
  // The simple wall = 1/thr cost model for single-round assumes
  // perfect linear scaling as thr_per_expert drops, which breaks
  // below ccd_size:
  //   * thr_per_expert < ccd_size means each expert's thread team
  //     spans less than one CCD's worth of cores, but the DLP
  //     kernel's NR-blocking factor is tuned for ccd_size-wide
  //     teams.
  //   * Cross-CCD coordination + partial-tile misalignment add
  //     overheads the cost model doesn't capture.
  //
  // Empirically on 128-thread Zen5 (GPT-OSS decode sweep):
  //     ops=22..24 → n_thr_single = 128/ops = 5 (below ccd_size=8);
  //                  cost model picks single, reality measures
  //                  multi-round 2-5% faster at n_thr_multi = 8.
  //     ops=17..21 → n_thr_single = 6..7 (at or near ccd_size);
  //                  cost model is correct, single wins in reality.
  // So the cliff is at n_thr_single ∈ [ccd_size*3/4, ccd_size).
  // Below that (< 6 on 8-wide CCD), prefer multi when feasible.
  // The `n_rounds_multi <= 2` guard keeps the correction bounded
  // to the small-tail regime: the cost model over-counts the
  // light-tail round's wall (a short tail of light experts under
  // descending / pair-balanced order costs less than the model's
  // 1/thr), but beyond 2 rounds the tail effect dilutes and the
  // model's prediction is sound.
  if (c.single_eligible) {
    const int thr_single =
        std::max(1, topo.num_threads / std::max(1, topo.num_ops));
    const int ccd_floor = std::max(1, (topo.ccd_size * 3) / 4);
    const bool single_thin = (thr_single < ccd_floor);
    if (single_thin && c.n_rounds_multi > 0 && c.n_rounds_multi <= 2) {
      return RoundPick::Multi;
    }
  }

  const bool consider_balanced = (topo.num_threads <= 64);
  if (c.single_eligible && c.wall_single < c.wall_multi
      && (!consider_balanced || c.wall_single <= c.wall_balanced)) {
    return RoundPick::Single;
  }
  if (consider_balanced && c.wall_balanced < c.wall_multi) {
    return RoundPick::Balanced;
  }
  return RoundPick::Multi;
}

inline void apply_round_pick(const GroupNTileTopology &topo,
                             const RoundCandidates &c,
                             RoundPick pick,
                             int ab_min_tile,
                             GroupNTilePlan &plan) {
  plan.strategy   = GroupNTileStrategy::ManyExperts;
  plan.min_n_tile = ab_min_tile;
  switch (pick) {
    case RoundPick::Single:
      plan.batch_size  = topo.num_ops;
      plan.n_thr_fixed = c.n_thr_single;
      break;
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
// Thin orchestrator — defers each decision to a named helper above:
//   1. ntile_viable             — falls back to Sequential if false.
//   2. self_fallback_to_sequential  — (R1/R2) gates.
//   3. try_decode_d_plan        — decode-class parallel, if eligible.
//   4. build_few_experts_plan   — (A) when num_ops ≤ num_ccds.
//   5. build_round_candidates
//      + pick_round_strategy
//      + apply_round_pick       — (B) many-experts round scheduler.
//
// The order is kept stable so that a future dtype-specific override
// only needs to swap tile constants or add a new strategy without
// changing the existing thresholds.
inline GroupNTilePlan plan_group_n_tile(
    const GroupNTileTopology &topo,
    matmul_algo_t algo, int nr_align, bool fused_epilogue,
    const std::vector<int> &M) {

  GroupNTilePlan plan{};
  plan.algo = algo;
  plan.num_threads = topo.num_threads;
  plan.nr_align = nr_align;
  plan.fused_epilogue = fused_epilogue;

  const bool decode_tile_ab_on = get_grp_n_decode_tile_ab();

  // Viability + R1/R2 self-fallbacks — both route to Sequential.
  if (!ntile_viable(topo) || self_fallback_to_sequential(topo)) {
    plan.strategy = GroupNTileStrategy::Sequential;
    return plan;
  }

  // Decode-class shapes may take the DecodeD fast path.
  if (try_decode_d_plan(topo, plan)) return plan;

  // For paths (A) and (B): when max_M is small (decode-class shape),
  // use the smaller decode-n-tile as min-tile so max_n_thr is high
  // enough to saturate all threads.  See `decode_tile_ab` in
  // group_matmul_parallel_common.hpp for the rationale.
  // `effective_decode_n_tile()` honors the optional
  // `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE` override.
  const int ab_min_tile = (topo.max_M <= kDecodeMaxM && decode_tile_ab_on)
      ? effective_decode_n_tile() : kMinNTile;

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
    apply_round_pick(topo, c, pick, ab_min_tile, plan);
  }

  // Self-gating on ZENDNNL_GRP_MATMUL_N_ORDER (mode 0 = off, no-op).
  fill_sorted_expert_order(plan, M, topo.num_ops);
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
// with the full thread team per kernel, then runs any fused activation
// as a second pass (no tile-local epilogue, since the matmul did not
// split N).
inline void execute_sequential(const GroupNTilePlan &plan,
                               GroupNTileContext &ctx) {
  const int num_ops = static_cast<int>(ctx.M.size());
  for (int e = 0; e < num_ops; ++e) {
    if (ctx.M[e] <= 0) continue;
    static thread_local matmul_params local_params;
    local_params = ctx.params[e];
    execute_expert_slice(ctx.layout[e], ctx.transA[e], ctx.transB[e],
        ctx.M[e], ctx.N[e], ctx.K[e], ctx.alpha[e],
        ctx.src[e], ctx.lda[e], ctx.weight[e], ctx.ldb[e],
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

  #pragma omp parallel num_threads(total_threads)
  {
    const int tid = omp_get_thread_num();
    const int e = tid / thr_per_expert;
    const int local_tid = tid % thr_per_expert;

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
      ctx.apply_swiglu_oai(e, local_tid, thr_per_expert, plan.min_n_tile);
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
//      the team's aggregate L3 (with a 10 % overshoot cap).  Without
//      a barrier between rounds, threads on small-M experts of round
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

  #pragma omp parallel num_threads(plan.num_threads)
  {
    const int tid = omp_get_thread_num();

    for (int round_start = 0; round_start < num_ops;
         round_start += batch_size) {
      const int round_end = std::min(num_ops, round_start + batch_size);
      const int round_size = round_end - round_start;
      const int thr_per_expert = (n_thr_fixed > 0)
          ? n_thr_fixed
          : std::min(plan.num_threads / round_size, max_n_thr);
      const int round_threads = round_size * thr_per_expert;

      int e = -1;
      int local_tid = -1;
      if (tid < round_threads) {
        const int local_expert = tid / thr_per_expert;
        local_tid = tid % thr_per_expert;
        e = sort_on ? plan.expert_order[round_start + local_expert]
                    : (round_start + local_expert);
        ctx.do_tile(plan, e, local_tid, thr_per_expert, min_n_tile);
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
          ctx.apply_swiglu_oai(e, local_tid, thr_per_expert, min_n_tile);
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
  // tile.  Everything else falls through the legacy path.
  const bool fused_epilogue =
      (fused_act == grp_matmul_gated_act_t::swiglu_oai_mul);

  // Tight-dst detection for the fused-epilogue path.  Caller's dst is
  // a tight [M, I]-layout buffer when ldc < N (the activation halves
  // N, so I = N/2).  Checked on the first expert only: the fused-MoE
  // caller allocates a uniform-stride arena across experts; a debug
  // assert below catches mismatches.  When active, the executors take
  // the per-thread-scratch + OOP swiglu path in `do_tile()`.
  const bool tight_fused_epilogue =
      fused_epilogue && !M.empty() && ldc[0] < N[0];
  // Always-on uniform-layout guard (defense-in-depth).  The caller
  // boundary (`validate_group_matmul_direct_inputs`) rejects mixed
  // tight/wide callers, but `flat_n_tile` can also be reached from
  // the fused-MoE executor and future internal paths, so re-verify
  // here.  If any active expert has an ldc that disagrees with the
  // inferred tight-from-ldc[0] state, ANY write path we could run
  // (tight-compact, wide-full, or wide-then-separate-pass-swiglu)
  // is unsafe:
  //   * tight-compact writes N/2 cols at each expert's stride; the
  //     wide experts get their second half left untouched — silent
  //     wrong result downstream.
  //   * wide writes N cols at each expert's stride; a tight expert
  //     with stride N/2 has rows of physical length N/2, so writing
  //     N cols straddles the next row — OOB / memory corruption.
  // The only memory-safe release-mode reaction is to refuse the
  // call.  We log loudly via apilog_error so the (should-never-
  // happen-in-practice) reach gets a filable signal rather than
  // an unexplained crash / corruption downstream.
  if (tight_fused_epilogue) {
    for (int e = 0; e < num_ops; ++e) {
      if (M[e] <= 0) continue;
      if (ldc[e] != N[e] / 2 || (N[e] % 2) != 0) {
        apilog_error(
            "[flat_n_tile] mixed tight/wide ldc across experts at e=", e,
            " (ldc[e]=", ldc[e], ", N[e]=", N[e],
            "); tight_fused_epilogue inferred from ldc[0]=", ldc[0],
            " < N[0]=", N[0],
            " but this expert disagrees.  Refusing to run — caller-"
            "boundary validation should have rejected this combo "
            "(please file a bug).  The caller's dst buffer(s) are "
            "unmodified by this call.");
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
  // Engage for both the non-fused path (act=none / silu / gelu — the
  // GEMM itself runs with act=none and the caller does any non-swiglu
  // activation as a separate pass) AND for the inline swiglu fused
  // epilogue (act=swiglu_oai_mul).  The custom swiglu path writes the
  // activated `[M, :I]` output at stride `ldc = 2I` — physically
  // identical to the layout the standard in-place `apply_swiglu_oai`
  // compaction produces, so downstream readers see the same bytes.
  //
  // What we pass to `prepare_for_call`:
  //   * act=none   when `fused_epilogue=false` (plain GEMM tile).
  //   * act=swiglu when `fused_epilogue=true`  (fused activation).
  // Anything else (silu / gelu) reaches flat_n_tile with
  // fused_epilogue=false so it falls in the first bucket — the
  // caller's separate silu/gelu pass runs on the wide GEMM output.
  // `engage_ntile_custom_kernel` (group_matmul_parallel_common.hpp)
  // does the env check, dispatcher hand-off, and contract gating in
  // one helper shared with the V2 fused-MoE Op1 executor.  When it
  // returns with `kctx.enabled = false` (env off, or dispatcher
  // refused — alpha != 1, beta != 0, non-bf16 dst, transB, etc.) we
  // stay on the standard execute_expert_slice path.
  const grp_matmul_gated_act_t custom_act = fused_epilogue
      ? grp_matmul_gated_act_t::swiglu_oai_mul
      : grp_matmul_gated_act_t::none;
  custom_kernel::CallContext kctx;
  engage_ntile_custom_kernel(
      custom_act,
      /*src_dtype=*/params[0].dtypes.src,
      /*wei_dtype=*/params[0].dtypes.wei,
      /*dst_dtype=*/params[0].dtypes.dst,
      act_dtype,
      /*bias_dtype=*/params[0].dtypes.bias,
      transA, transB, M, N, K, alpha, beta, weight, kctx);

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

  GroupNTileContext ctx{
      layout, transA, transB,
      M, N, K, alpha,
      src, lda, weight, ldb, bias, beta, dst, ldc,
      is_weights_const, params,
      fused_act, act_dtype,
      wei_elem, dst_elem, bias_elem,
      use_custom, use_custom ? &kctx : nullptr,
      &alloc_fail
  };

  const GroupNTileTopology topo =
      summarise_topology(M, N, K, num_threads, wei_elem);
  GroupNTilePlan plan =
      plan_group_n_tile(topo, algo, nr_align, fused_epilogue, M);
  // The tight-dst switch is orthogonal to the planner's strategy /
  // threading decisions: it only toggles how each thread writes its
  // final swiglu output (scratch + OOP vs in-place).  Set after the
  // plan is built so the planner's shape-driven choices are unaffected.
  // Sequential strategy + tight is not supported (its sequential path
  // calls execute_expert_slice directly at caller's ldc — which is too
  // narrow to fit the 2I matmul output).  In practice the fused-MoE
  // tight callers always route to DecodeD / FewExperts / ManyExperts
  // because their shapes are N-tile viable; assert defensively.
  plan.tight_fused_epilogue =
      tight_fused_epilogue && !use_custom
      && plan.strategy != GroupNTileStrategy::Sequential;
  assert((!tight_fused_epilogue || use_custom
          || plan.strategy != GroupNTileStrategy::Sequential)
         && "tight_fused_epilogue unsupported on Sequential strategy");

  // ── Surface the concrete path to the caller (gemm_mode_out) ────────
  // Seven possible strings: Sequential + six (fused_epilogue ×
  // use_custom × tight).  Static literals only — pointer is written
  // once per call, no allocation.  The Sequential strategy bypasses
  // the custom kernel and the fused-epilogue / tight machinery, so it
  // gets a distinct label regardless of the upstream flags — keeps
  // profiler / benchdnn output honest about what actually ran.
  if (gemm_mode_out != nullptr) {
    if (plan.strategy == GroupNTileStrategy::Sequential) {
      *gemm_mode_out = "flat_n_tile_sequential";
    } else if (fused_epilogue) {
      if (plan.tight_fused_epilogue || (tight_fused_epilogue && use_custom)) {
        // Tight path — either the non-custom scratch+OOP writer
        // (plan.tight_fused_epilogue=true) or the custom-kernel tight
        // writer (use_custom + tight_fused_epilogue; the custom
        // kernel always writes compacted I-cols when activation is
        // swiglu regardless of the plan flag).
        *gemm_mode_out = use_custom
            ? "flat_n_tile_fused_swiglu_oai_tight_custom"
            : "flat_n_tile_fused_swiglu_oai_tight";
      } else {
        *gemm_mode_out = use_custom
            ? "flat_n_tile_fused_swiglu_oai_custom"
            : "flat_n_tile_fused_swiglu_oai";
      }
    } else {
      *gemm_mode_out = use_custom ? "flat_n_tile_custom" : "flat_n_tile";
    }
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
    const int env_order = get_grp_matmul_n_order();
    const bool is_auto_resolved =
        (env_order == 0 && plan.auto_resolved_order >= 0);
    apilog_info("[flat_n_tile] act=", act_name(fused_act),
                " fused_epilogue=", fused_epilogue,
                " tight=", plan.tight_fused_epilogue,
                " kernel=", (use_custom ? "custom" : "standard"),
                " pack_nr=", (use_custom ? kctx.pack_nr : 0),
                " subtile_cols=", (use_custom ? kctx.subtile_cols : 0),
                " min_n_tile=", plan.min_n_tile,
                " num_ops=", num_ops,
                " num_threads=", num_threads,
                " n_order_env=", env_order,
                " n_order_used=",
                is_auto_resolved ? plan.auto_resolved_order : env_order,
                is_auto_resolved ? " (auto)" : "");
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
    apilog_error(
        "[flat_n_tile] per-thread scratch allocation failed in the "
        "tight-fused-epilogue path; some dst column ranges may be "
        "undefined.  Consider disabling internal-alloc tight mode "
        "(ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT=0) if this recurs.");
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
