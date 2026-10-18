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

/// N-split helpers shared by the executors: the AOCL stable N-tile knobs,
/// backend and decode N alignment, `aligned_n_split` and L3 sizes.

#ifndef ZENDNNL_GRP_MATMUL_N_SPLIT_HPP
#define ZENDNNL_GRP_MATMUL_N_SPLIT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "common/op_config.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

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
// workloads, the per-call cache-hit savings from avoided reorders
// dominate the per-call thread-utilisation loss from the idle
// threads in a strict-stable plan.
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
//   "0" restores the legacy dynamic plan topology (cache thrash).
inline bool get_grp_matmul_aocl_stable_ntile() {
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE");
        if (e == nullptr || e[0] == '\0') return true;
        return e[0] != '0';
    }();
    return v;
}

inline constexpr int kAoclTargetConcurrentSlots = 16; // team-budget divisor
inline constexpr int kAoclBlisNc = 128; // BLIS-bf16 inner-N block

// ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS = positive int — cached, default 16.
//   Team-budget divisor.  Lower (e.g. 8) for few-expert deployments;
//   raise for many-expert deployments where reducing per-expert fan-
//   out helps.  Non-positive → default.
inline int get_grp_matmul_aocl_target_slots() {
    // Strict env parsing — non-numeric input falls back to default.
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return kAoclTargetConcurrentSlots;
        return (parsed > 0) ? parsed : kAoclTargetConcurrentSlots;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_AOCL_BLIS_NC = positive int — cached, default 128.
//   Informational / telemetry only as of the strict-stable cache-key
//   simplification.  The original `aocl_stable_n_thr` formula included
//   a `by_density = N / kAoclBlisNc` term that protected thin-N
//   shapes above AOCL's NC=128 amortisation point; that term was
//   removed because it re-introduced an N-dependence into the per-
//   expert thread count and rotated the AOCL DLP cache key per call.
//   Narrow-N protection is now handled by the planner's narrow-N
//   escape (see `aocl_stable_n_thr` and the F3 escape comment in
//   `plan_group_n_tile`).
//
//   The env value is parsed (strict) and emitted in the
//   `[GRP_MATMUL.PLAN] flat_n_tile ...` apilog line so external
//   telemetry can correlate user-set tuning with the planner's
//   actual choices; setting it does NOT change planning behaviour.
//   Non-positive → default.  Kept as a getter (rather than removed
//   outright) so reintroducing a density floor in future is a one-
//   line change in `aocl_stable_n_thr` and we don't churn the env
//   surface area.
inline int get_grp_matmul_aocl_blis_nc() {
    // Strict env parsing — non-numeric input falls back to default.
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_BLIS_NC");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return kAoclBlisNc;
        return (parsed > 0) ? parsed : kAoclBlisNc;
    }();
    return v;
}

// NOTE: `get_grp_matmul_n_tile_heavy_threshold()`
// (ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD) moved to
// `group_matmul_n_tile.hpp` (Section A.4) together with the rest of
// the N-tile env getters.  Include that header to call it.  The
// three-mode (DISABLED / AUTO / MANUAL) doc-block lives at the new
// home.

// NOTE: All `get_grp_matmul_m_tile_*` getters (hybrid /
// slice_target / hybrid_min_max_m / hybrid_min_skew /
// hybrid_lights_per_thread / vertical_fusion / pipeline_scratch_kb)
// moved to `group_matmul_m_tile.hpp` (Section H.3) together with
// the override atoms they read.  Include that header to pull them
// in.

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

/// N alignment the inner kernel prefers for each per-thread slice
/// (1 = no alignment).  ALGO 3's column partitioner (aligned_n_split)
/// honours it when the slowest thread stays within 2× of the fastest
/// after rounding; otherwise it falls back to the unaligned even split.
inline int backend_n_align(matmul_algo_t algo) {
    switch (algo) {
        case matmul_algo_t::native_brgemm:
        case matmul_algo_t::native_gemm: return 64;
        default: return 1;
    }
}

/// The per-thread N-slice alignment an upcoming ALGO 3 decode will split
/// on, resolved from a prompt-shaped call so `cross_warm` can warm
/// per-tile AOCL DLP keys the decode will actually query.
///
/// The per-tile cache key embeds `n_tile = aligned_n_split(N, n_thr, ...,
/// nr_align)`, so warming at a different alignment prefills keys the
/// runtime never looks up and every tile misses on the first decode call.
///
/// This mirrors `flat_n_tile`'s own derivation and must keep mirroring it:
///
///   * the dispatcher hands ALGO 3 `a3_can_fuse_act(act, CK) ? act : none`,
///     so silu / gelu with the custom kernel off arrive as `none` and the
///     arena is wide;
///   * a fused epilogue with `ldc[0] < N[0]` is the tight arena, whose OOP
///     writer packs at `col_start / 2` and so needs an even `col_start` on
///     every thread;
///   * with the custom kernel engaged `pack_nr` (32 or 64) is already even,
///     and that regime is warmed by the custom-kernel warmer instead, so
///     only the CK-off case has to be right here.
///
/// Returns the `aocl_dlp_blocked` backend alignment (1), widened to 2 for
/// the tight pair-aligned case.  This only selects which keys get warmed;
/// a wrong answer costs a first-call reorder, never correctness.
/// `ck_enabled` must be the EFFECTIVE per-call verdict, not the master env
/// knob: `grp_matmul_custom_kernel_effective`, which folds in the family
/// sub-toggle.  Passing the raw env mispredicts int8 or f16 whenever their
/// sub-knob is off, because those reach decode with the custom kernel
/// disabled and therefore split on the tight alignment.
///
/// `decode_arena_tight` is the arena the DECODE call will use, which is not
/// observable from the prompt call's `ldc`.  Under AUTO a prompt that
/// resolves to ALGO 1 gets a wide library-owned Op1 arena, because
/// `pick_fused_moe_want_tight` requires `resolved_algo == 3`, while the
/// decode call gets a tight one.  Reading `ldc[0] < N[0]` here would
/// therefore invert the prediction on exactly the fused-MoE layers this
/// exists to serve.
/// Callers that know the decode arena (the fused-MoE entry) pass it; callers
/// that own their own destination and keep one stride across both phases
/// pass their observed tightness.
inline int algo3_decode_nr_align(
        grp_matmul_gated_act_t act, bool ck_enabled, bool decode_arena_tight) {
    const int backend_nr = backend_n_align(matmul_algo_t::aocl_dlp_blocked);
    // Mirrors `ntile_effective_nr_align`: the custom kernel widens to its
    // own pack_nr and owns the pairing, so only the CK-off tight case needs
    // the pair alignment.  When the custom kernel will run, the per-tile
    // AOCL keys are not what decode queries at all -- the custom-kernel pack
    // arena is -- so the value here is immaterial and `backend_nr` is right.
    const bool tight_pair_align
            = !ck_enabled && decode_arena_tight && a3_can_fuse_act(act, false);
    return tight_pair_align ? std::max(backend_nr, 2) : backend_nr;
}

/// Decode-arena hint, published by the fused-MoE entry for the duration of
/// one Op1 dispatch on the SAME thread.
///
/// The fused-MoE entry is the only place that can answer "will the decode
/// call get a tight arena?", because it owns the arena and can evaluate
/// `pick_fused_moe_want_tight(..., resolved_algo = 3)` regardless of what
/// this particular call resolved to.  The prompt-side prepack lives several
/// frames down inside the ALGO executors, and threading a parameter there
/// would touch the dispatcher signature and every executor.  A scoped
/// thread-local is the narrower change; the RAII guard means it cannot
/// outlive the dispatch that set it, and nothing reads it off-thread.
inline thread_local bool tls_decode_arena_tight = false;
inline thread_local bool tls_decode_arena_known = false;

class scoped_decode_arena_hint {
public:
    explicit scoped_decode_arena_hint(bool tight)
        : prev_tight_(tls_decode_arena_tight)
        , prev_known_(tls_decode_arena_known) {
        tls_decode_arena_tight = tight;
        tls_decode_arena_known = true;
    }
    ~scoped_decode_arena_hint() {
        tls_decode_arena_tight = prev_tight_;
        tls_decode_arena_known = prev_known_;
    }
    scoped_decode_arena_hint(const scoped_decode_arena_hint &) = delete;
    scoped_decode_arena_hint &operator=(const scoped_decode_arena_hint &)
            = delete;

private:
    bool prev_tight_;
    bool prev_known_;
};

/// Aligned column-slice partitioner for ALGO 3.  Returns {col_start,
/// col_end} for `tid` of `n_thr` over [0, N).
///
/// Legacy mode gives `n_thr - 1` threads one aligned slice and the last
/// thread the remainder, searching downward in alignment quanta until the
/// last slice is at least half the common width. It falls back to an
/// unaligned even split when no such width exists.
///
/// `balance_remainder` is the single-expert custom-kernel mode. It distributes
/// complete `align` blocks as evenly as possible, gives any sub-block tail to
/// the last active thread, and returns an empty [N,N) range for surplus
/// threads when there are fewer aligned blocks than requested workers.
/// Multi-expert and standard-backend callers retain legacy partitioning.
///
/// Falls back to even split (N*tid/n_thr) when n_thr<=1 or align<=1
/// or no aligned slice meets the 2× bound.
inline std::pair<int, int> aligned_n_split(
        int N, int n_thr, int tid, int align, bool balance_remainder = false) {
    // Hardened against pathological inputs: n_thr<=0 used to hit
    // even_split's divide-by-zero (`N * tid / n_thr`).
    if (n_thr <= 0 || N <= 0) { return std::make_pair(0, 0); }

    auto even_split = [&]() {
        const int s = static_cast<int>(static_cast<int64_t>(N) * tid / n_thr);
        const int e
                = static_cast<int>(static_cast<int64_t>(N) * (tid + 1) / n_thr);
        return std::make_pair(s, e);
    };

    if (align <= 1 || n_thr <= 1) { return even_split(); }

    // Spread the remainder only for the explicitly opted-in single-expert CK
    // route. Capping the active workers by the aligned-block count keeps every
    // non-empty range pack-aligned; extra workers return [N,N) and no-op.
    if (balance_remainder) {
        const int64_t blocks = static_cast<int64_t>(N) / align;
        if (blocks > 0) {
            const int64_t active_thr = std::min<int64_t>(n_thr, blocks);
            if (tid >= active_thr) return std::make_pair(N, N);
            const int64_t base = blocks / active_thr;
            const int64_t n_big = blocks % active_thr;
            const int64_t start_blk = base * tid + (tid < n_big ? tid : n_big);
            const int64_t cnt = base + (tid < n_big ? 1 : 0);
            const int64_t s = start_blk * align;
            const int64_t e = (tid == active_thr - 1)
                    ? static_cast<int64_t>(N)
                    : (start_blk + cnt) * align;
            return std::make_pair(static_cast<int>(s), static_cast<int>(e));
        }
    }

    // Legacy aligned split. Intermediates are int64_t so products remain free
    // of signed overflow for every int-representable input.
    const int64_t even_per_thr = (static_cast<int64_t>(N) + n_thr - 1) / n_thr;
    for (int64_t aligned_per_thr = ((even_per_thr + align - 1) / align) * align;
            aligned_per_thr >= align; aligned_per_thr -= align) {
        const int64_t n_full = aligned_per_thr * (n_thr - 1);
        const int64_t last = static_cast<int64_t>(N) - n_full;
        if (last > 0 && last * 2 >= aligned_per_thr) {
            const int64_t s = aligned_per_thr * tid;
            const int64_t e
                    = (tid < n_thr - 1) ? aligned_per_thr * (tid + 1) : N;
            return std::make_pair(static_cast<int>(s), static_cast<int>(e));
        }
    }
    return even_split();
}

/// Wrapper around aligned_n_split with optional even boundary snapping for
/// packed-s4 N-tile slices.
inline std::pair<int, int> n_split_for_tile(int N, int n_thr, int tid,
        int align, bool even_boundaries, bool balance_remainder = false) {
    auto s = aligned_n_split(N, n_thr, tid, align, balance_remainder);
    if (!even_boundaries) { return s; }
    s.first &= ~1;
    s.second = (tid == n_thr - 1) ? N : (s.second & ~1);
    return s;
}

/// Byte offset of column `col_start` in a nibble-packed s4 B matrix.
inline size_t packed_s4_col_byte_off(int col_start, int ldb, bool transB) {
    const size_t nib = transB
            ? static_cast<size_t>(col_start) * static_cast<size_t>(ldb)
            : static_cast<size_t>(col_start);
    return nib >> 1;
}

/// 32 MB of L3 per CCD on current-generation classic-CCD CPU topologies.
inline constexpr size_t kL3PerCcdBytes = 32UL * 1024UL * 1024UL;

/// Aggregate L3 the planner uses to bound the experts-per-round
/// budget (ALGO 3 N-tile, ALGO 6 multilevel).  `num_ccds` comes from
/// summarise_topology() so this stays consistent with how the rest
/// of the planner partitions the team.
inline size_t get_grp_l3_total_bytes(int num_ccds) {
    return static_cast<size_t>(std::max(1, num_ccds)) * kL3PerCcdBytes;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GRP_MATMUL_N_SPLIT_HPP
