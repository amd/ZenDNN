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
///      generic ALGO {1,2,3,5,6} / flat_n_tile / flat_m_tile / etc.), then Op2
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
///                                   (generic ALGO routing).
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

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>
#include "common/zendnnl_compat.hpp"

#include <omp.h>

#include "detect_internal_alloc.hpp"
#include "group_matmul_direct.hpp"
#include "group_matmul_parallel_common.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "m_tile/group_matmul_m_tile.hpp" // try_flat_m_tile_pipeline_bf16
#include "ntile_flat_parallel/ntile_flat_parallel.hpp"

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
    ~FusedMoEArena() { zendnnl_aligned_free(buf); }
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
    std::vector<void *> op1_dst_internal; // populated in internal-alloc mode
    std::vector<void *> op2_dst_internal; // populated in internal-alloc mode
    std::vector<int> op1_ldc_local; // populated only when `want_tight`
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
    zendnnl_aligned_free(arena.buf);
    arena.buf = nullptr;
    arena.cap = 0;

    FusedMoEScratch &s = get_thread_local_scratch();
    // Deterministic dealloc: swap with empty temporary → temporary's
    // dtor frees the old buffer at end of statement.  See doc-block
    // above for why `shrink_to_fit()` is NOT used here.
    std::vector<int> {}.swap(s.K_down);
    std::vector<float> {}.swap(s.alpha_down);
    std::vector<float> {}.swap(s.beta_down);
    std::vector<bool> {}.swap(s.transA_down);
    std::vector<const void *> {}.swap(s.src_down);
    std::vector<matmul_params> {}.swap(s.params_down);
    std::vector<void *> {}.swap(s.op1_dst_internal);
    std::vector<void *> {}.swap(s.op2_dst_internal);
    std::vector<int> {}.swap(s.op1_ldc_local);
    ntile_flat_parallel::reset_thread_local_scratch();
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
//      a caller forcing ALGO 1/2/5/6 explicitly asked for a non-N-tile
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
// `resolved_algo` is the ALREADY-resolved (and safety-clamped) ALGO for
// this call — `select_grp_matmul_algo(...)`'s output — passed in by the
// caller so it is computed exactly once per fused-MoE call and shared
// with the vertical-fusion gate (avoids a second O(num_ops) selection
// pass and a duplicate `[GRP_MATMUL.ALGO WARN]` on clamped shapes).
inline bool pick_fused_moe_want_tight(bool op1_internal,
        grp_matmul_gated_act_t act, int env_algo, int resolved_algo) {
    if (!op1_internal) return false;
    if (!a3_can_fuse_act(act, get_grp_matmul_custom_kernel())) return false;
    if (env_algo != 0 && env_algo != 3) return false;
    if (get_grp_matmul_fused_moe_tight() == 0) return false;
    return resolved_algo == 3;
}

// True when `fused.dst_down[]` aliases `src[]` in a way only the two-pass
// ordering can serve.  Two-pass drains src into the Op1 arena before Pass 2
// writes anything.  Vertical fusion instead hands each thread one M-slice
// and runs Op1-read -> act -> Op2-write inside it with no intervening
// barrier (`flat_m_tile_pipeline_bf16`), and rows are owned by different
// threads.  So the ONLY safe overlap is exact in-place reuse, where row r
// writes precisely onto row r's own bytes: same base AND same row stride.
// Any other overlap lets one thread's Op2 write land on a row another
// thread has not read yet, in either direction (a lower slice's write can
// outrun its own reader, and a higher slice's write can fall back onto a
// lower row still owned by another thread).  Every src is tested against
// every dst so an Op2 output covering a DIFFERENT expert's source is caught.
// A transposed Op1 has M as its fast dimension, so every slice touches
// almost the whole src buffer (`m_tile` offsets src by `row_start*elem`,
// not `row_start*lda*elem`): the per-row frontier model does not apply, so
// such an op spans K rows and never earns the in-place exemption.
inline bool fused_moe_src_op2dst_hazard(const std::vector<int> &M,
        const std::vector<int> &K, const std::vector<int> &lda,
        const std::vector<bool> &transA, const std::vector<const void *> &src,
        const std::vector<void *> &op2_dst, const std::vector<int> &op2_ldc,
        const std::vector<int> &N_down,
        const std::vector<matmul_params> &params, size_t num_ops) {
    const auto active = [&](size_t i) {
        return M[i] > 0 && src[i] != nullptr && op2_dst[i] != nullptr;
    };
    // Spans below feed an address-range comparison, so arithmetic that wraps
    // must never be able to shrink a range into a false "disjoint" verdict.
    // Any overflow trips this flag and the whole query answers "hazard",
    // which costs such a (already pathological) shape only its vertical
    // fusion.
    bool unsafe_arith = false;
    const auto mul = [&](size_t a, size_t b) {
        size_t r = 0;
        if (zendnnl_mul_overflow(a, b, &r)) unsafe_arith = true;
        return r;
    };
    const auto add = [&](size_t a, size_t b) {
        size_t r = 0;
        if (zendnnl_add_overflow(a, b, &r)) unsafe_arith = true;
        return r;
    };
    // Bytes each op reads from src[i] / writes to op2_dst[i], modelled as
    // the STRIDED accessed interval: (rows - 1) full strides plus the
    // columns actually touched on the last row.  Counting `rows * ld`
    // instead would fold the final row's trailing padding into the range.
    //
    // This interval is an ENCLOSING hull, not the accessed set: it also
    // covers each row's interior padding.  So it answers "definitely
    // disjoint" exactly, but its "overlapping" verdict is only a
    // candidate — two operands can share a hull while touching disjoint
    // column ranges of the same rows.  The canonical case is a padded
    // layout that writes the unused half of each row (non-transposed
    // K=64, lda=128, `dst_down = src + 64`): at M=1 the hulls separate,
    // but from M=2 on they interleave and the hull test alone would deny
    // vertical fusion to a provably safe caller.  `strided_sets_overlap`
    // below settles those candidates exactly.
    //
    // A transposed src is a [K, lda] buffer with M indexing COLUMNS, so
    // its footprint is (K-1) full rows plus M elements — not K*lda, which
    // understates it whenever lda < M and would hide an alias past the
    // K*lda mark.
    const auto strided_span
            = [&](size_t rows, size_t ld, size_t last_cols, size_t elem) {
        if (rows == 0 || last_cols == 0) return static_cast<size_t>(0);
        return mul(add(mul(rows - 1, ld), last_cols), elem);
    };
    const auto src_span = [&](size_t i) {
        // Op1 reads K columns per row (M columns per row when transposed,
        // where the roles of the two extents swap).
        const size_t ld = static_cast<size_t>(lda[i]);
        const size_t rows = static_cast<size_t>(transA[i] ? K[i] : M[i]);
        const size_t last_cols
                = static_cast<size_t>(transA[i] ? M[i] : std::max(K[i], 0));
        return strided_span(rows, ld, last_cols, size_of(params[i].dtypes.src));
    };
    const auto dst_span = [&](size_t i) {
        // Op2 writes N_down columns per row at stride op2_ldc.  `N_down` is
        // sized to the ACTIVE range by the caller, so clamp defensively
        // rather than indexing past it.
        const size_t cols = (i < N_down.size())
                ? static_cast<size_t>(std::max(N_down[i], 0))
                : static_cast<size_t>(op2_ldc[i]);
        return strided_span(static_cast<size_t>(M[i]),
                static_cast<size_t>(op2_ldc[i]), cols,
                size_of(params[i].dtypes.dst));
    };

    // Exact intersection of two strided row sets, for the case that
    // actually occurs in a padded MoE layout: both operands non-transposed
    // and walking the SAME row stride.  Anything else keeps the
    // conservative hull verdict.
    //
    // Model each set as rows of touched bytes at a fixed stride S:
    //   src: r in [0, Rs)   ->  [ r*S,       r*S + Cs )
    //   dst: q in [0, Rd)   ->  [ D + q*S,   D + q*S + Cd )
    // with D the signed base delta.  A src row and a dst row intersect iff
    // their offsets differ by less than the respective widths, and the
    // difference only ever depends on k = q - r:
    //   intersect(k)  <=>  -Cd < D + k*S < Cs
    // and k is realisable iff some row pair exists, i.e. -Rs < k < Rd.
    // So the sets overlap iff an integer k satisfies both — a couple of
    // divisions, no loop over rows.
    const auto floor_div = [](ptrdiff_t a, ptrdiff_t b) { // b > 0
        ptrdiff_t q = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
        return q;
    };
    const auto strided_sets_overlap
            = [&](ptrdiff_t D, ptrdiff_t S, ptrdiff_t Rs, ptrdiff_t Cs,
                      ptrdiff_t Rd, ptrdiff_t Cd) {
        if (S <= 0 || Cs <= 0 || Cd <= 0 || Rs <= 0 || Rd <= 0) return true;
        // Smallest k with D + k*S > -Cd, largest k with D + k*S < Cs.
        const ptrdiff_t k_min = floor_div(-Cd - D, S) + 1;
        const ptrdiff_t k_max = -floor_div(-(Cs - D), S) - 1;
        // Realisable row offsets.
        const ptrdiff_t k_lo = std::max(k_min, -(Rs - 1));
        const ptrdiff_t k_hi = std::min(k_max, Rd - 1);
        return k_lo <= k_hi;
    };

    uintptr_t src_lo = UINTPTR_MAX, src_hi = 0;
    uintptr_t dst_lo = UINTPTR_MAX, dst_hi = 0;
    for (size_t i = 0; i < num_ops; ++i) {
        if (!active(i)) continue;
        const uintptr_t s = reinterpret_cast<uintptr_t>(src[i]);
        const uintptr_t d = reinterpret_cast<uintptr_t>(op2_dst[i]);
        const uintptr_t s_end = s + src_span(i);
        const uintptr_t d_end = d + dst_span(i);
        if (s_end < s || d_end < d) unsafe_arith = true; // address wrap
        src_lo = std::min(src_lo, s);
        src_hi = std::max(src_hi, s_end);
        dst_lo = std::min(dst_lo, d);
        dst_hi = std::max(dst_hi, d_end);
    }
    if (unsafe_arith) return true;
    // O(num_ops) pre-filter: disjoint unions prove no pair can overlap.  It
    // is only a pre-filter — separate src and dst allocations may interleave
    // on the heap, so overlapping unions do NOT imply an aliased pair.  Pay
    // for the exact pairwise scan only in that (rare) case, so the common
    // non-aliasing caller is not charged O(num_ops^2) and, more importantly,
    // is not denied vertical fusion by a false positive.
    if (!(dst_lo < src_hi && src_lo < dst_hi)) return false;
    for (size_t i = 0; i < num_ops; ++i) {
        if (!active(i)) continue;
        const uintptr_t s = reinterpret_cast<uintptr_t>(src[i]);
        const uintptr_t s_end = s + src_span(i);
        for (size_t j = 0; j < num_ops; ++j) {
            if (!active(j)) continue;
            const uintptr_t d = reinterpret_cast<uintptr_t>(op2_dst[j]);
            const uintptr_t d_end = d + dst_span(j);
            if (d >= s_end || s >= d_end) continue; // hulls disjoint
            // Overlapping: safe only as this op's own exact in-place reuse.
            if (i == j && !transA[i] && d == s
                    && static_cast<size_t>(op2_ldc[i])
                                    * size_of(params[i].dtypes.dst)
                            == static_cast<size_t>(lda[i])
                                    * size_of(params[i].dtypes.src))
                continue;
            // Hulls overlap, but they may still touch disjoint columns of
            // the same rows.  Settle it exactly when both operands are
            // non-transposed and share a row stride; otherwise keep the
            // conservative verdict.  Erring toward "hazard" only costs
            // vertical fusion, while a wrong "safe" is a data race.
            const size_t s_elem = size_of(params[i].dtypes.src);
            const size_t d_elem = size_of(params[j].dtypes.dst);
            const ptrdiff_t s_stride = static_cast<ptrdiff_t>(lda[i]) * s_elem;
            const ptrdiff_t d_stride
                    = static_cast<ptrdiff_t>(op2_ldc[j]) * d_elem;
            if (!transA[i] && s_stride == d_stride) {
                const ptrdiff_t delta
                        = static_cast<ptrdiff_t>(d) - static_cast<ptrdiff_t>(s);
                const ptrdiff_t s_cols
                        = static_cast<ptrdiff_t>(std::max(K[i], 0)) * s_elem;
                const ptrdiff_t d_cols
                        = static_cast<ptrdiff_t>(j < N_down.size()
                                          ? std::max(N_down[j], 0)
                                          : op2_ldc[j])
                        * d_elem;
                if (!strided_sets_overlap(delta, s_stride,
                            static_cast<ptrdiff_t>(M[i]), s_cols,
                            static_cast<ptrdiff_t>(M[j]), d_cols))
                    continue;
            }
            return true;
        }
    }
    return false;
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
//       uniformity when either side is internal-alloc (it feeds the Op1
//       arena slab sizing), plus per-expert `src == dst` dtype equality
//       (gate (G3)) for `op2_internal` only, where Op2's write footprint
//       lands in the caller's src[].
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
        const grp_matmul_fused_moe_params &fused, grp_matmul_gated_act_t act,
        data_type_t act_dtype, const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params,
        const group_matmul_moe_postop_params *moe_postop, bool op1_internal,
        bool op2_internal, size_t dst_elem_internal, int64_t *out_total_M,
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
            || fused.N_down.size() < num_ops || fused.ldb_down.size() < num_ops
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
        if (fused.dst_down.size() < num_ops || fused.ldc_down.size() < num_ops)
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

        // ── Op2 weight-scale metadata, checked BEFORE Op1 runs ─────────
        //
        // `setup_op2_dispatch_scratch` gives Op2 Op1's weight dtype
        // verbatim (`p.dtypes.wei = params[i].dtypes.wei`), so a quantized
        // Op1 weight means Op2 also runs a quantized GEMM and needs its
        // own weight scale.  The vector-size checks above bound only the
        // LENGTH of `down_scale`, and only when it is non-empty, which
        // leaves two holes: an ABSENT `down_scale` on a quantized down
        // weight, and a PRESENT one whose entry for this expert is null or
        // shaped for a different K/N.  In both cases `params_down` reaches
        // dispatch with `dynamic_quant=true` and no usable weight scale;
        // nothing corrupts memory and the dispatch chain still returns
        // success, so W2 can come back unwritten or unscaled with no error
        // for the caller to see.  Fail closed here instead.
        //
        // Active experts only: an `M[i] == 0` slot runs no Op2 GEMM, so
        // demanding scale metadata for it would reject callers that leave
        // the inactive tail of a padded expert pool default-constructed.
        if (M[i] > 0) {
            const data_type_t wei_dt = params[i].dtypes.wei;
            const bool op2_wei_quantized = (wei_dt == data_type_t::s8
                    || wei_dt == data_type_t::u8 || wei_dt == data_type_t::s4
                    || wei_dt == data_type_t::u4);
            if (op2_wei_quantized) {
                if (fused.down_scale.empty()) return status_t::failure;
                const auto &ws = fused.down_scale[i];
                if (ws.buff == nullptr) return status_t::failure;
                if (ws.dt != data_type_t::f32 && ws.dt != data_type_t::bf16)
                    return status_t::failure;
                // Granularities the Op2 path actually consumes, measured
                // against THIS pass's K_down / N_down rather than Op1's
                // K_in / N — the two differ in general, which is the whole
                // reason Op2 derives its own group count from these dims.
                //   per-channel: {N_down} or {1, N_down}
                //   per-group:   {G2, N_down} or {1, G2, N_down},
                //                with K_down divisible by G2
                const auto &d = ws.dims;
                const int64_t n_down = static_cast<int64_t>(fused.N_down[i]);
                const int64_t k_down = static_cast<int64_t>(K_down);
                bool dims_ok = false;
                if (d.size() == 1) {
                    dims_ok = (d[0] == n_down);
                } else if (d.size() == 2) {
                    dims_ok = (d[1] == n_down) && d[0] >= 1
                            && (k_down % d[0] == 0);
                } else if (d.size() == 3) {
                    dims_ok = (d[0] == 1) && (d[2] == n_down) && d[1] >= 1
                            && (k_down % d[1] == 0);
                }
                if (!dims_ok) return status_t::failure;
            }
        }

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
            // Three correctness gates ZenDNN can enforce:
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
            //   (G3) `dtypes.src == dtypes.dst`.  One integer `lda[i]`
            //        addresses both passes in different element sizes — Op1
            //        reads row m at `m·lda·src_elem`, Op2 writes it at
            //        `m·lda·dst_elem` — so a NARROWER src (e.g. s8 src +
            //        bf16 dst) both overruns the caller's allocation and,
            //        because `src[]` holds per-expert bases into ONE grouped
            //        buffer, lets expert e's output land in expert e+1's
            //        SOURCE rows — corruption that needs no concurrency,
            //        since that region is another expert's input.
            //        Over-allocating does not help: the spacing, not the
            //        total size, is what overlaps.  A WIDER src is
            //        spacing-safe, but it leaves a mixed src/dst fused-MoE
            //        configuration that dispatch does not compute, so it is
            //        rejected too and the predicate is plain equality rather
            //        than element-size parity.  A pre-quantized s8 source
            //        therefore needs a caller-allocated `fused.dst_down[]`;
            //        `op1_internal` stays available since the Op1 arena is
            //        sized from dst alone.
            //
            // (G1) is the legacy check (preserved verbatim below).  (G2)
            // elevates the validator from "wide-enough stride" to
            // "consistent stride AND wide enough".  (G3) replaces the
            // blanket per-expert `src == dst` check that used to run for
            // `op1_internal` too.
            //
            // A correctly-wide `lda` over an UNDER-ALLOCATED `src[i]` stays
            // out of scope — undetectable without allocation introspection,
            // so it remains a caller-contract requirement.
            if (lda[i] < fused.N_down[i]) {
                log_error(
                        "group_matmul_fused_moe: op2_internal requires "
                        "lda[",
                        i, "] >= fused.N_down[", i, "] (got lda=", lda[i],
                        ", N_down=", fused.N_down[i], ").  src[", i,
                        "] is reused as Op2's destination and is "
                        "written with row stride lda; the stride must be "
                        "wide enough for the Op2 output columns.  Either "
                        "(a) widen lda and allocate src[] for "
                        "M*lda*src_elem bytes, or (b) pass an explicit "
                        "fused.dst_down[] (caller-allocated Op2 dst).");
                return status_t::failure;
            }

            if (params[i].dtypes.src != params[i].dtypes.dst) {
                log_error(
                        "group_matmul_fused_moe: op2_internal requires "
                        "dtypes.src == dtypes.dst on params[",
                        i,
                        "] (got src=", static_cast<int>(params[i].dtypes.src),
                        " (", size_of(params[i].dtypes.src),
                        "B), dst=", static_cast<int>(params[i].dtypes.dst),
                        " (", size_of(params[i].dtypes.dst), "B)).  src[", i,
                        "] is reused as Op2's destination: a NARROWER source "
                        "element makes Op2 overrun both the caller's "
                        "allocation and the next expert's rows, and a WIDER "
                        "one is a mixed src/dst fused-MoE configuration that "
                        "the dispatch does not compute.  Pass an explicit "
                        "fused.dst_down[] (caller-allocated Op2 dst) "
                        "instead.");
                return status_t::failure;
            }
        }

        // Cross-expert dst-dtype uniformity is always-on when either side
        // is internal-alloc: the Op1 arena slab and the Op2 in-place write
        // footprint are both sized from `params[0].dtypes.dst`.  Per-expert
        // src==dst is NOT implied — that is an op2_internal-only
        // requirement, now (G3) above.
        if ((op1_internal || op2_internal) && M[i] > 0) {
            if (params[i].dtypes.dst != params[0].dtypes.dst)
                return status_t::failure;
        }

        // (G4) Supported-Op1-tuple gate for internal Op1 allocation.
        //
        // (G3) above constrains `src == dst` only under `op2_internal`,
        // where one integer `lda` has to address both passes.  An internal
        // Op1 arena has no such spacing constraint, so relaxing (G3) off
        // `op1_internal` was correct for SPACING — but it also stopped
        // rejecting tuples the dispatch cannot compute at all.  An f32 src
        // with a bf16 dst is the case in point: spacing-safe, passes every
        // other check, and AOCL implements only the f32/f32 -> f32 branch,
        // so the call returns `success` having written NEITHER pass.  A
        // silent success over an untouched buffer is a worse outcome than a
        // diagnostic, so admit only what dispatch actually computes:
        //
        //   * `src == dst` — the classic bf16/bf16 and f32/f32 regimes.
        //     WOQ (s4/u4 weight) and library-side dynamic-quant INT8 both
        //     live here too: their source stays float and equals dst; only
        //     `dtypes.wei` / `dtypes.compute` differ.
        //   * pre-quantized s8 — s8 src + s8 wei with a float dst, the
        //     configuration this change adds.  Op1 consumes the caller's s8
        //     rows plus `src_scale.buff` directly, so src != dst is
        //     intended rather than an unsupported mix.
        //
        // Scoped to `op1_internal` because that is the mode whose gate was
        // relaxed; a fully caller-allocated call keeps its prior behaviour.
        if (op1_internal && M[i] > 0
                && params[i].dtypes.src != params[i].dtypes.dst) {
            const bool prequant_s8_op1 = params[i].dtypes.src == data_type_t::s8
                    && params[i].dtypes.wei == data_type_t::s8
                    && (params[i].dtypes.dst == data_type_t::bf16
                            || params[i].dtypes.dst == data_type_t::f32)
                    && params[i].quant_params.src_scale.buff != nullptr;
            if (!prequant_s8_op1) {
                log_error(
                        "group_matmul_fused_moe: op1_internal requires "
                        "dtypes.src == dtypes.dst on params[",
                        i,
                        "] unless the call is the pre-quantized s8 form "
                        "(src=s8, wei=s8, dst=bf16/f32, non-null "
                        "quant_params.src_scale.buff).  Got src=",
                        static_cast<int>(params[i].dtypes.src),
                        ", wei=", static_cast<int>(params[i].dtypes.wei),
                        ", dst=", static_cast<int>(params[i].dtypes.dst),
                        ".  A mixed src/dst tuple is not a dispatch the "
                        "library computes: it would return success having "
                        "written neither pass.  Match src to dst, or supply "
                        "the pre-quantized s8 source with its scale.");
                return status_t::failure;
            }
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
                if (zendnnl_mul_overflow(m_sz, n_sz, &per_expert_bytes))
                    return status_t::failure;
                if (zendnnl_mul_overflow(per_expert_bytes, dst_elem_internal,
                            &per_expert_bytes))
                    return status_t::failure;
                if (zendnnl_add_overflow(total_bytes_internal, per_expert_bytes,
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
            const size_t dst_down_sweep
                    = std::min<size_t>(num_ops, fused.dst_down.size());
            for (size_t i = 0; i < dst_down_sweep; ++i)
                if (fused.dst_down[i] != nullptr) return status_t::failure;
        }
        bool any_bias_down = false;
        for (size_t i = 0; i < num_ops; ++i)
            if (fused.bias_down[i] != nullptr) {
                any_bias_down = true;
                break;
            }
        if (any_bias_down && fused.bias_dt_down == data_type_t::none)
            return status_t::failure;
        if (act != grp_matmul_gated_act_t::none && act_dtype != data_type_t::f32
                && act_dtype != data_type_t::bf16
                && act_dtype != data_type_t::f16)
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
inline status_t setup_op1_arena_and_layout(FusedMoEArena &arena,
        FusedMoEScratch &scratch, bool op1_internal, bool want_tight,
        size_t arena_bytes_wide, size_t dst_elem_internal,
        const std::vector<int> &N, const std::vector<int> &M,
        const std::vector<int> &ldc_caller,
        const std::vector<void *> &dst_caller,
        const std::vector<void *> *&out_op1_dst,
        const std::vector<int> *&out_op1_ldc) {
    const size_t num_ops = M.size();

    size_t arena_bytes = arena_bytes_wide;
    if (want_tight) arena_bytes /= 2;

    if (op1_internal && arena_bytes > arena.cap) {
        zendnnl_aligned_free(arena.buf);
        arena.buf = nullptr;
        arena.cap = 0;
        void *tmp = nullptr;
        if (zendnnl_posix_memalign(&tmp, 64, arena_bytes) != 0
                || tmp == nullptr)
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
    if (want_tight) scratch.op1_ldc_local.resize(num_ops);
    if (op1_internal || want_tight) {
        char *base = op1_internal ? static_cast<char *>(arena.buf) : nullptr;
        size_t cursor = 0;
        for (size_t i = 0; i < num_ops; ++i) {
            const int row_cols = want_tight ? (N[i] / 2) : N[i];
            if (want_tight) scratch.op1_ldc_local[i] = row_cols;
            if (op1_internal) {
                if (M[i] <= 0 || base == nullptr) {
                    scratch.op1_dst_internal[i] = nullptr;
                } else {
                    // Overflow-safe per-expert slab accumulation.  The
                    // validator gated the WIDE total; this multiplier chain is
                    // different (tight halves `row_cols`), so re-check that
                    // `cursor + M*row_cols*elem` stays representable.  A trip
                    // fails before any executor sees a wrapped pointer.
                    scratch.op1_dst_internal[i] = base + cursor;
                    const size_t m_sz = static_cast<size_t>(M[i]);
                    const size_t row_sz = static_cast<size_t>(row_cols);
                    size_t per_expert_bytes = 0;
                    if (zendnnl_mul_overflow(m_sz, row_sz, &per_expert_bytes))
                        return status_t::failure;
                    if (zendnnl_mul_overflow(per_expert_bytes,
                                dst_elem_internal, &per_expert_bytes))
                        return status_t::failure;
                    if (zendnnl_add_overflow(cursor, per_expert_bytes, &cursor))
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
// Op2 inherits Op1's `dtypes.compute` so the down_proj runs through the
// same dispatch path as the gate+up GEMM, but `dynamic_quant` is DERIVED
// from Op2's own dtypes, not inherited: Op2's source is always the float
// Op1 output, so a pre-quantized (s8) Op1 src must not disable Op2's
// quantization.
// Per-group src_scale (`dims = {M, ngroups>1}`) cannot inherit Op1's
// group count because Op1.K != Op2.K; instead Op2's group count is
// derived from the paired down-projection weight scale ({G2, N_down}),
// yielding an Op2 source scale of `{M, G2}` quantized independently
// per pass.  (Vertical fusion's inline requant is per-token only, so
// per-group routes to the two-pass legacy path.)  Per-group source
// zero-points remain unsupported and return `status_t::failure`.
inline status_t setup_op2_dispatch_scratch(FusedMoEScratch &scratch,
        const grp_matmul_fused_moe_params &fused, grp_matmul_gated_act_t act,
        size_t num_ops, const std::vector<int> &N,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<matmul_params> &params,
        const std::vector<void *> &op1_dst, bool op2_internal) {
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
    scratch.alpha_down.resize(num_ops, 1.0f);
    scratch.beta_down.resize(num_ops, 0.0f);
    scratch.transA_down.resize(num_ops, false);
    scratch.src_down.resize(num_ops);
    scratch.params_down.resize(num_ops);
    if (op2_internal) scratch.op2_dst_internal.resize(num_ops);

    for (size_t i = 0; i < num_ops; ++i) {
        scratch.src_down[i] = op1_dst[i];

        // `lowoha_algo` is both input hint and output — must be reset to
        // `none` every call so a dispatcher pick from an earlier call does
        // not force the same kernel on the next.
        matmul_params &p = scratch.params_down[i];
        p.lowoha_algo = matmul_algo_t::none;
        p.dtypes.src = params[i].dtypes.dst;
        p.dtypes.wei = params[i].dtypes.wei;
        p.dtypes.dst = params[i].dtypes.dst;
        p.dtypes.bias = fused.bias_dt_down;
        p.num_threads = params[i].num_threads;
        p.weight_cache_type = params[i].weight_cache_type;
        // `dtypes.compute` carries over, but `dynamic_quant` is DERIVED:
        // Op2's source is always the float Op1 output, so it needs a source
        // quant pass exactly when its own compute dtype is int8.  Inheriting
        // would leave Op2's s8 GEMM unscaled under a pre-quantized Op1 src.
        p.dtypes.compute = params[i].dtypes.compute;
        const bool op2_int8_compute = (p.dtypes.compute == data_type_t::s8
                || p.dtypes.compute == data_type_t::u8);
        // bf16/f32 only, matching `is_dynamic_quant_config`: the reorder
        // admits no other source dtype, so listing f16 here would set a flag
        // the reorder ignores and hand Op2's s8 GEMM an unquantized source.
        const bool op2_src_is_float = (p.dtypes.src == data_type_t::bf16
                || p.dtypes.src == data_type_t::f32);
        p.dynamic_quant = op2_int8_compute && op2_src_is_float;
        // GGML-packed down weights are unpacked + AOCL sym-quant-reordered by
        // the caller (group_matmul_direct) into the same layout as Op1's
        // weight, and `down_scale[i]` carries the resulting {K_down/32, N_down}
        // scale.  Inherit Op1's reorder/pack flags so Op2 consumes the
        // already-reordered down_weight instead of re-reordering plain bytes.
        // Gated on Op1 being packed → exact no-op for the non-GGML path.
        if (params[i].packing.pack_format_b == 1) {
            p.mem_format_b = params[i].mem_format_b; // 'r' (reordered)
            p.packing.pack_format_b = params[i].packing.pack_format_b;
        } else {
            // Plain (non-GGML) down weight: reset the reorder/pack flags
            // EXPLICITLY.  `scratch.params_down` is a persistent thread-local
            // reused across calls (resize is a no-op at steady size), so a stale
            // `mem_format_b == 'r'` / `pack_format_b == 1` left by a PRIOR GGML
            // fused call on this thread would otherwise leak into this plain call
            // — mis-routing the Op2 dispatch (e.g. the AOCL kernel would treat the
            // weight as pre-reordered) and tripping `check_m_tile_safe`'s
            // row-major gate so vertical fusion silently declines.
            p.mem_format_b = 'n';
            p.packing.pack_format_b = 0;
        }

        // Reset everything first, then fill in just the wei_scale /
        // wei_zp (from the caller-facing fields) and the inherited
        // src_scale dims (when dynamic_quant is on).
        p.quant_params = matmul_quantization_params_t {};
        if (!fused.down_scale.empty()) {
            p.quant_params.wei_scale.buff = fused.down_scale[i].buff;
            p.quant_params.wei_scale.dt = fused.down_scale[i].dt;
            p.quant_params.wei_scale.dims = fused.down_scale[i].dims;
        }
        if (!fused.down_zp.empty()) {
            p.quant_params.wei_zp.buff = fused.down_zp[i].buff;
            p.quant_params.wei_zp.dt = fused.down_zp[i].dt;
            p.quant_params.wei_zp.dims = fused.down_zp[i].dims;
        }
        if (p.dynamic_quant) {
            const auto &scale_dims = params[i].quant_params.src_scale.dims;
            const bool op1_per_group
                    = (scale_dims.size() == 2 && scale_dims[1] > 1);
            p.quant_params.src_scale.buff = nullptr;
            p.quant_params.src_scale.dt = params[i].quant_params.src_scale.dt;
            if (op1_per_group) {
                // Op1 quantizes its source [M, K_in] per-group, but Op2's source
                // is the Op1 output [M, K_down] with K_down != K_in, so Op1's
                // group count cannot transfer.  Derive Op2's OWN group count from
                // the paired down-projection weight scale: a per-group weight has
                // dims {G2, N_down} (or {1, G2, N_down}), i.e. K_down split into
                // G2 groups, so the matching Op2 source scale is {M, G2} with
                // group_size = K_down / G2 (equal to the weight's group size).
                // The per-pass DQ (grouped pre-pass or per-expert fallback) then
                // quantizes the Op2 source per-group independently of Op1.
                const int64_t op1_M = scale_dims[0];
                int64_t g2 = 1;
                if (!fused.down_scale.empty()) {
                    const auto &wsd = fused.down_scale[i].dims;
                    if (wsd.size() == 2 && wsd[0] > 1)
                        g2 = wsd[0]; // {G2, N}
                    else if (wsd.size() == 3 && wsd[1] > 1)
                        g2 = wsd[1]; // {1, G2, N}
                }
                p.quant_params.src_scale.dims = {op1_M, g2};
            } else {
                p.quant_params.src_scale.dims
                        = params[i].quant_params.src_scale.dims;
            }
            // Source zero-point flows only for asymmetric quant.  Per-group
            // source zero-points on the fused down-proj are still unsupported
            // (the down-proj re-quant path is symmetric s8 only).
            if (params[i].quant_params.src_zp.dt != data_type_t::none) {
                const auto &zp_dims = params[i].quant_params.src_zp.dims;
                if (zp_dims.size() == 2 && zp_dims[1] > 1) {
                    log_error(
                            "group_matmul_fused_moe: per-group src_zp on "
                            "params[",
                            i, "] (dims={", zp_dims[0], ",", zp_dims[1],
                            "}) unsupported; use per-token "
                            "({M, 1}).");
                    return status_t::failure;
                }
                p.quant_params.src_zp.buff = nullptr;
                p.quant_params.src_zp.dt = params[i].quant_params.src_zp.dt;
                p.quant_params.src_zp.dims = params[i].quant_params.src_zp.dims;
            }
        }
        // active_matmul / total_matmul propagate so the Pass-2 per-ALGO
        // prepack sees the full active/total contract and warms the
        // prepack-extras tail.
        p.active_matmul = params[i].active_matmul;
        p.total_matmul = params[i].total_matmul;

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
// picks a generic ALGO from {1,2,3,5,6} based on shape and env knobs.

// Legacy two-pass MoE dispatch.  Pass 1 = Op1 (W13 + optional gated
// activation) via `group_matmul_run_parallel_dispatch`.  Pass 2 = Op2
// (W2 down-projection) via the same dispatcher with `act=none`.
//
// When Pass 1's dispatcher cannot fuse the activation (e.g. ALGO 3 +
// silu_and_mul / gelu_and_mul on the wide arena), a separate-pass
// activation runs between Pass 1 and Pass 2.
//
// Source dynamic quantization is selected by ZENDNNL_ENABLE_GROUP_DQ:
//   * on (default): each pass group-quantizes its source up front via
//     `group_reorder_quantization_wrapper`, rewriting the per-pass
//     params copy to s8 + clearing dynamic_quant, so the per-expert
//     `reorder_quantization_wrapper` inside `execute_expert_slice`
//     short-circuits to a no-op (no double quant).
//   * off: the grouped pre-pass is skipped and dynamic quant flows
//     through the per-expert path inside `execute_expert_slice`
//     (legacy behaviour).
// This is the NON-vertical-fusion fallback path; the vertical-fusion
// executor (tried first by the caller) does its own in-pipeline quant
// and is never reached when it engages.
inline status_t run_fused_moe_legacy_two_pass(grp_matmul_gated_act_t act,
        data_type_t act_dtype, const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &op1_dst, const std::vector<int> &op1_ldc,
        const grp_matmul_fused_moe_params &fused, FusedMoEScratch &scratch,
        const std::vector<void *> &op2_dst, const std::vector<int> &op2_ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        const char *&pass1_mode, const char *&pass2_mode) {
    const bool enable_group_dq = get_grp_matmul_enable_group_dq();

    // Pass 1 source group dynamic quant (opt-in; default on).  Quantizes
    // every expert's BF16/F32 src to S8 in one grouped pass (with a
    // per-expert fallback for shapes the grouped kernel doesn't cover)
    // and rewrites `params` (library exec_params from group_matmul_direct;
    // caller config is unchanged).  Op2 scratch is built from the same
    // exec vector before Pass 1 mutates it.
    std::vector<const void *> pass1_quant_src;
    std::vector<int> pass1_quant_lda;
    group_reorder_quant_buffers_t pass1_quant_buffers;
    bool pass1_group_quantized = false;
    if (enable_group_dq) {
        const status_t pass1_quant_st = group_reorder_quantization_wrapper(src,
                lda, transA, M, K, num_threads, params, pass1_quant_src,
                pass1_quant_lda, pass1_quant_buffers, pass1_group_quantized);
        if (pass1_quant_st != status_t::success) return pass1_quant_st;
    }

    // Per-op timing instrumentation (diagnostic; OFF by default).  When
    // ZENDNNL_GRP_MATMUL_OPTIME=1 we wrap each pass's executor with a
    // wall-clock timer and emit one parseable [GRP_MATMUL.OPTIME] line per
    // op per call (op=1 covers Op1 matmul + any separate activation pass;
    // op=2 covers Op2).  The line carries this op's num_ops (total expert
    // pool = M.size(), consistent with the other [GRP_MATMUL.*] lines),
    // active_ops (the M[i]>0 experts that actually fire), and the per-op
    // N/K maxima, so per-op latency can be correlated with shape offline.
    // Covers only this legacy two-pass path (not the fused
    // vertical-fusion path).  Emitted via apilog at info level (requires
    // ZENDNNL_API_LOG_LEVEL=3); the measured region excludes the emit, so
    // logging cost does not perturb the timing.
    static const bool s_optime = []() {
        // The OPTIME line is emitted via apilog_info; if info logging is
        // off the output is dropped, so gate the whole diagnostic on it to
        // avoid paying the timer + per-op reductions for nothing (apilog's
        // own arguments are evaluated before it can short-circuit).
        if (!apilog_info_enabled()) return false;
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_OPTIME");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    const double t_op1_start = s_optime ? omp_get_wtime() : 0.0;

    // Pass 1: Op1 (gate+up) + activation.  The dispatcher picks ALGO
    // 1..5 (or auto) per `ZENDNNL_GRP_MATMUL_ALGO` and the safety
    // gates; inner BLAS kernel honours `ZENDNNL_MATMUL_ALGO`.  For
    // tight layout the dispatcher auto-enables fused activation
    // regardless of `N_TILE_FUSED_ACT` (correctness contract).  When
    // grouped DQ did not run, `params` still carries dynamic_quant and
    // `execute_expert_slice` quantizes per expert.
    const bool act_fused
            = group_matmul_run_parallel_dispatch(layout, transA, transB, M, N,
                    K, alpha, pass1_group_quantized ? pass1_quant_src : src,
                    pass1_group_quantized ? pass1_quant_lda : lda, weight, ldb,
                    bias, beta, op1_dst, op1_ldc, is_weights_const, params,
                    num_threads, &pass1_mode, act, act_dtype);

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
    if (s_optime) {
        // Close the timing window FIRST, before any metadata work, so the
        // measured region is well-defined and excludes the reductions /
        // logging below (argument-evaluation order is unspecified, so the
        // elapsed read must not live in the apilog_info arg list).
        const double t_op1_ms = (omp_get_wtime() - t_op1_start) * 1.0e3;
        // N/K can vary per expert (validator allows it), so report the max
        // across experts rather than element 0 — a single representative
        // that does not mislead offline correlation when shapes differ.
        // The reductions run only on the (off-by-default) diagnostic path.
        const int n_max = N.empty() ? 0 : *std::max_element(N.begin(), N.end());
        const int k_max = K.empty() ? 0 : *std::max_element(K.begin(), K.end());
        const int active_ops = static_cast<int>(
                std::count_if(M.begin(), M.end(), [](int m) { return m > 0; }));
        apilog_info("[GRP_MATMUL.OPTIME] op=1 ms=", t_op1_ms,
                " num_ops=", static_cast<int>(M.size()),
                " active_ops=", active_ops, " N_max=", n_max, " K_max=", k_max,
                " act_fused=", (act_fused ? 1 : 0),
                " mode=", (pass1_mode != nullptr ? pass1_mode : "?"));
    }

    // Pass 2 source group dynamic quant (same opt-in gate).  Runs AFTER
    // Op1 + activation so the activated Op1 output (`scratch.src_down`,
    // read at `op1_ldc` stride) is the source being quantized.  When
    // disabled, Op2 dynamic quant is handled per-expert in
    // `execute_expert_slice` (params_down still carries dynamic_quant).
    std::vector<const void *> pass2_quant_src;
    std::vector<int> pass2_quant_lda;
    group_reorder_quant_buffers_t pass2_quant_buffers;
    bool pass2_group_quantized = false;
    if (enable_group_dq) {
        const status_t pass2_quant_st
                = group_reorder_quantization_wrapper(scratch.src_down, op1_ldc,
                        scratch.transA_down, M, scratch.K_down, num_threads,
                        scratch.params_down, pass2_quant_src, pass2_quant_lda,
                        pass2_quant_buffers, pass2_group_quantized);
        if (pass2_quant_st != status_t::success) return pass2_quant_st;
    }

    // ── Pass 2: Op2 (down_proj) dispatch ────────────────────────────────
    // Single route: `group_matmul_run_parallel_dispatch` with `act=none`.
    // Honours generic `ZENDNNL_GRP_MATMUL_ALGO` values {1,2,3,5,6} and
    // `ZENDNNL_MATMUL_ALGO`
    // for inner BLAS, and routes through the custom BF16 microkernel
    // inside flat_n_tile when `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1` and
    // the ALGO 3 path is selected.
    //
    // Op2's lda (= op1_ldc) per Op1 layout × activation:
    //   wide  + gated act  — lda=N,    K_down=N/2.
    //   wide  + act=none   — lda=N,    K_down=N.
    //   tight + gated act  — lda=N/2,  K_down=N/2.
    const double t_op2_start = s_optime ? omp_get_wtime() : 0.0;
    group_matmul_run_parallel_dispatch(layout, scratch.transA_down, transB, M,
            fused.N_down, scratch.K_down, scratch.alpha_down,
            pass2_group_quantized ? pass2_quant_src : scratch.src_down,
            pass2_group_quantized ? pass2_quant_lda : op1_ldc,
            fused.down_weight, fused.ldb_down, fused.bias_down,
            scratch.beta_down, op2_dst, op2_ldc, is_weights_const,
            scratch.params_down, num_threads, &pass2_mode,
            grp_matmul_gated_act_t::none, act_dtype);
    if (s_optime) {
        // Close the timing window FIRST (see Op1 note) so the measured
        // region excludes the reductions / logging below.
        const double t_op2_ms = (omp_get_wtime() - t_op2_start) * 1.0e3;
        // Op2 N_down / K_down are likewise per-expert; report the max.
        const int n_max = fused.N_down.empty()
                ? 0
                : *std::max_element(fused.N_down.begin(), fused.N_down.end());
        const int k_max = scratch.K_down.empty()
                ? 0
                : *std::max_element(
                          scratch.K_down.begin(), scratch.K_down.end());
        const int active_ops = static_cast<int>(
                std::count_if(M.begin(), M.end(), [](int m) { return m > 0; }));
        apilog_info("[GRP_MATMUL.OPTIME] op=2 ms=", t_op2_ms,
                " num_ops=", static_cast<int>(M.size()),
                " active_ops=", active_ops, " N_max=", n_max, " K_max=", k_max,
                " act_fused=", 0,
                " mode=", (pass2_mode != nullptr ? pass2_mode : "?"));
    }
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
inline const char *compose_fused_moe_gemm_mode(bool vertical_fusion_engaged,
        bool op1_internal, bool op2_internal, bool want_tight,
        const char *pass1_mode, const char *pass2_mode, bool has_postop) {
    static thread_local std::string mode_buf;
    mode_buf.clear();
    mode_buf.reserve(64);
    mode_buf.append(
            vertical_fusion_engaged ? "fused_moe_vertical" : "fused_moe_2pass");
    if (op1_internal && op2_internal)
        mode_buf.append("_intalloc");
    else if (op1_internal)
        mode_buf.append("_intalloc_op1");
    else if (op2_internal)
        mode_buf.append("_intalloc_op2");
    if (want_tight) mode_buf.append("_tight");
    mode_buf.append("(op1=");
    mode_buf.append(pass1_mode != nullptr ? pass1_mode : "?");
    mode_buf.append(",op2=");
    mode_buf.append(pass2_mode != nullptr ? pass2_mode : "?");
    mode_buf.append(")");
    if (has_postop) mode_buf.append("+postop");
    return mode_buf.c_str();
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
// Primary entry: Op1+Act → Op2 (→ optional weighted reduce post-op)
// ═══════════════════════════════════════════════════════════════════════
//
// Orchestrator only — every step is a helper above or a backend
// executor in a sibling translation unit.  Read top-to-bottom to see
// the fused-MoE pipeline flow.

status_t group_matmul_fused_moe_execute(
        const grp_matmul_fused_moe_params &fused, grp_matmul_gated_act_t act,
        data_type_t act_dtype, const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
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
    auto run_detect = [&](const std::vector<void *> &v, const char *name,
                              bool *out_internal) -> status_t {
        const status_t st
                = detect_internal_alloc(v, num_ops, /*fused_moe_present=*/true,
                        internal_alloc_mode::sweep_active, out_internal);
        if (st != status_t::success) {
            log_error("group_matmul_fused_moe: ", name,
                    " has a mixed "
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
    const size_t dst_elem_internal
            = op1_internal ? size_of(params[0].dtypes.dst) : 0;
    int64_t total_M = 0;
    size_t total_bytes_internal = 0;
    {
        const status_t v = validate_fused_moe_inputs(fused, act, act_dtype,
                layout, transA, transB, M, N, K, alpha, src, lda, weight, ldb,
                bias, beta, dst, ldc, is_weights_const, params, moe_postop,
                op1_internal, op2_internal, dst_elem_internal, &total_M,
                &total_bytes_internal);
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
    // Resolve (and safety-clamp) the ALGO for this call ONCE — shared by
    // both the tight-arena decision below and the vertical-fusion gate at
    // Step 8.  NOTE: this is NOT simply `env_algo_fused`: even a pinned
    // generic env algo ({1,2,3,5,6}) is clamped by m_tile_safe /
    // n_tile_safe inside
    // `select_grp_matmul_algo`, so e.g. a pinned ALGO 2 on an m-tile-unsafe
    // shape resolves to 1 and vertical fusion must NOT engage.
    const int resolved_algo
            = select_grp_matmul_algo(layout, M, N, K, params, num_threads);
    const bool want_tight = pick_fused_moe_want_tight(
            op1_internal, act, env_algo_fused, resolved_algo);

    // EXEC APILOG — one line per fused_moe call summarising arena
    // layout, per-side internal-alloc state, act-fusion choice, and
    // the W13 write width.  apilog_info_enabled() is cached after the
    // first call so the gate check is free when logging is off.
    static const bool s_apilog = apilog_info_enabled();
    if (s_apilog) {
        const int log_fused_moe_tight = get_grp_matmul_fused_moe_tight();
        const bool act_is_gated = (act != grp_matmul_gated_act_t::none);
        const char *w13_write_elems
                = act_is_gated ? (want_tight ? "I" : "2I") : "N";
        apilog_info("[GRP_MATMUL.EXEC] op=fused_moe arena=",
                (want_tight ? "tight" : "loose"),
                " op1_internal=", (op1_internal ? "yes" : "no"),
                " op2_internal=", (op2_internal ? "yes" : "no"),
                " act=", act_name(act), " act_in_register=",
                ((want_tight && act_is_gated) ? "yes" : "no"),
                " W13_write_elems_per_row=", w13_write_elems, " op2_dst_reuse=",
                (op2_internal ? "src_inplace" : "caller_dst_down"),
                " env_algo=", env_algo_fused,
                " env_tight=", log_fused_moe_tight,
                " custom_kernel_env=", (custom_kernel_en ? "on" : "off"),
                " num_ops=", (int)num_ops);
    }

    // ── Step 4: Op1 arena + per-expert pointer / stride setup ──────────
    // Thread-local scratch surfaces now live in file-scope accessors so
    // `clear_fused_moe_scratch()` can reach them via an OMP team sweep.
    // Functional behaviour is identical to a function-local static (one
    // instance per thread, persistent for the thread's lifetime); the
    // indirection cost is zero after the first call on a given thread
    // (returns by reference to a static thread_local).
    FusedMoEArena &arena = get_thread_local_arena();
    FusedMoEScratch &scratch = get_thread_local_scratch();
    const std::vector<void *> *op1_dst_p = nullptr;
    const std::vector<int> *op1_ldc_p = nullptr;
    {
        const status_t s = setup_op1_arena_and_layout(arena, scratch,
                op1_internal, want_tight, total_bytes_internal,
                dst_elem_internal, N, M, ldc, dst, op1_dst_p, op1_ldc_p);
        if (s != status_t::success) return s;
    }
    const std::vector<void *> &op1_dst = *op1_dst_p;
    const std::vector<int> &op1_ldc = *op1_ldc_p;

    // ── Step 5: Op2 dispatch scratch population ────────────────────────
    {
        // Pass `num_ops` (= M.size() = ACTIVE matmul count) explicitly so
        // the setup loop is bounded by the active range, not by the
        // framework's prepack-extras-tail `params.size()`.  See the
        // doc-block on setup_op2_dispatch_scratch() for the active/total
        // contract.
        const status_t s = setup_op2_dispatch_scratch(scratch, fused, act,
                num_ops, N, src, lda, params, op1_dst, op2_internal);
        if (s != status_t::success) return s;
    }
    const std::vector<void *> &op2_dst
            = op2_internal ? scratch.op2_dst_internal : fused.dst_down;
    const std::vector<int> &op2_ldc = op2_internal ? lda : fused.ldc_down;

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
    // Vertical fusion is an M-tile (ALGO 2) executor, NOT a separate
    // ALGO — it slots into the M-tile branch.  Only engage it when the
    // RESOLVED algo for this call is ALGO 2: under a pinned env algo
    // ({1,2,3,5,6}) that is exactly the pinned value; under AUTO (env 0) it is
    // the auto-selector's per-phase choice (prompt -> 2, decode -> 3 by
    // default).  This keeps vertical fusion inside the ALGO-2 decision
    // tree and stops it from overriding a pinned ALGO 1/3/5/6 (e.g. an
    // ALGO-3 N-tile decode run, where it previously still *attempted*
    // before falling through to legacy two-pass).  Uses the same
    // resolver `pick_fused_moe_want_tight` consults for its ALGO-3 tight
    // check, so the gate agrees with the per-GEMM dispatch the legacy
    // two-pass below will pick.  `resolved_algo` was computed once at
    // Step 3 (reused here — same safety-clamped value).
    // VF is additionally declined on a src[]/Op2-dst alias only two-pass
    // can serve; exact in-place reuse is unaffected.  Gated on the resolved
    // algo AND on vertical fusion being enabled at all (the knob defaults
    // to DISABLED), since the hazard verdict has no other consumer — so no
    // caller pays for the scan on the default path.
    const bool vf_knob_on = (get_grp_matmul_m_tile_vertical_fusion() != -1);
    const bool src_op2dst_hazard = (resolved_algo == 2) && vf_knob_on
            && fused_moe_src_op2dst_hazard(M, K, lda, transA, src, op2_dst,
                    op2_ldc, fused.N_down, params, num_ops);
    if (src_op2dst_hazard && s_apilog) {
        apilog_info(
                "[GRP_MATMUL.EXEC] op=fused_moe vertical_fusion=declined "
                "reason=src_op2dst_alias");
    }
    const bool vf_algo_allowed = (resolved_algo == 2) && !src_op2dst_hazard;

    const char *pass1_mode = nullptr;
    const char *pass2_mode = nullptr;
    bool vertical_fusion_engaged = false;
    if (vf_algo_allowed) {
        if (s_apilog) {
            apilog_info(
                    "[GRP_MATMUL.EXEC] op=fused_moe "
                    "enter=vertical_fusion_attempt");
        }
        vertical_fusion_engaged = try_flat_m_tile_pipeline_bf16(layout, transA,
                scratch.transA_down, transB, M, N, K, alpha, src, lda, weight,
                ldb, bias, beta, op1_dst, op1_ldc,
                /*dst_w13_is_caller_alloc=*/!op1_internal, fused.N_down,
                scratch.K_down, scratch.alpha_down, fused.down_weight,
                fused.ldb_down, fused.bias_down, scratch.beta_down, op2_dst,
                op2_ldc, act, act_dtype, is_weights_const, params,
                scratch.params_down, num_threads);
    }
    if (vertical_fusion_engaged) {
        if (s_apilog) {
            apilog_info(
                    "[GRP_MATMUL.EXEC] op=fused_moe exit=vertical_fusion_ok");
        }
        // Differentiate BF16 end-to-end / WOQ-INT4 / DQ-INT8 in the
        // profiler / apilog so per-route timings can be partitioned
        // downstream.  The eligibility wrapper guarantees both halves
        // share the same regime, so a single probe of
        // `params[0].dtypes.wei` (with `dynamic_quant` to distinguish
        // DQ-INT8 from a hypothetical static-INT8 placeholder) suffices.
        const data_type_t wei0
                = (!params.empty()) ? params[0].dtypes.wei : data_type_t::none;
        const bool is_woq_wei
                = (wei0 == data_type_t::s4 || wei0 == data_type_t::u4);
        const bool is_dqint8_wei = (wei0 == data_type_t::s8)
                && (!params.empty()) && params[0].dynamic_quant;
        if (is_dqint8_wei)
            pass1_mode = "vertical_fusion_dqint8";
        else if (is_woq_wei)
            pass1_mode = "vertical_fusion_woq";
        else
            pass1_mode = "vertical_fusion_bf16";
        pass2_mode = pass1_mode;
    } else {
        if (s_apilog) {
            apilog_info("[GRP_MATMUL.EXEC] op=fused_moe enter=legacy_two_pass");
        }
        const status_t s = run_fused_moe_legacy_two_pass(act, act_dtype, layout,
                transA, transB, M, N, K, alpha, src, lda, weight, ldb, bias,
                beta, op1_dst, op1_ldc, fused, scratch, op2_dst, op2_ldc,
                is_weights_const, params, num_threads, pass1_mode, pass2_mode);
        if (s != status_t::success) return s;
        if (s_apilog) {
            apilog_info(
                    "[GRP_MATMUL.EXEC] op=fused_moe exit=legacy_two_pass_ok");
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
        *gemm_mode_out = compose_fused_moe_gemm_mode(vertical_fusion_engaged,
                op1_internal, op2_internal, want_tight, pass1_mode, pass2_mode,
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
        const grp_matmul_fused_moe_params &fused, grp_matmul_gated_act_t act,
        data_type_t act_dtype, const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        const char **gemm_mode_out) {
    return group_matmul_fused_moe_execute(fused, act, act_dtype, layout, transA,
            transB, M, N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, num_threads, gemm_mode_out,
            /*moe_postop=*/nullptr);
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
    { reset_thread_local_fused_moe_state(); }
    ntile_flat_parallel::flush_packed_weight_cache();
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
