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

/// ALGO 3 — N-tile parallel GEMM, public interface header.
///
/// This header carries the planner-shaped data types AND the public
/// surface (env override atoms, env getters, `n_tile_execute()` forward
/// declaration) that the `n_tile_execute()` translation unit, planner
/// gtests, and the top-level dispatcher need to share.  It is
/// library-internal: not part of the public ZenDNN API and not meant
/// for inclusion outside `src/lowoha_operators/matmul/group_matmul/`
/// (plus the `gtests/group_matmul/` test files that exercise N-tile
/// path overrides).
///
/// Layout — section banners match the .cpp where applicable:
///
///   N-tile shared constants `kDecodeNTile` and `kNTilePlanMaxExperts`
///   live in the companion planner header
///   `group_matmul_n_tile_planner.hpp` (top of that file, near the
///   `GroupNTilePlan` definition that consumes `kNTilePlanMaxExperts`
///   at compile time).  This header re-includes the planner so every
///   N-tile consumer (executor, dispatcher, gtests) sees them through
///   `n_tile/group_matmul_n_tile.hpp`.
///
///   Section A.0  AlignedTileScratch — per-thread aligned heap buffer
///                used by the tight-fused-epilogue path of `execute_tile()`.
///                Lives here so a future memory-audit gtest can
///                exercise `grow_scratch()` directly.
///
///   Section A.1  Strategy enum + planner-shaped types (Topology,
///                Plan, RoundCandidates, RoundPick).  All POD-ish:
///                only fundamental types and `std::array`, no
///                references or owning resources.  Safe to include
///                from gtests for planner property tests.
///
///   Section A.2  test_api::PhaseBSnapshot capture (existing).  Locks
///                down the heaviest-first / eligibility-filter
///                behaviour of `apply_round_pick`'s Phase B body for
///                the end-to-end correctness test in `test_algos.cpp`.
///
///   Section A.3  test_api::s_grp_*_n_tile_* override atoms.
///                Sentinel-`-1` ("no override") or sentinel-`INT_MIN`
///                atomics that shadow the env-cache.  Flipped by
///                gtests (via RAII helpers in
///                `gtests/group_matmul/moe_test_utils.hpp`) and
///                long-running services that want to switch
///                configuration without re-launching the process.
///
///   Section A.4  N-tile env getters: `get_grp_n_tile_*()` /
///                `get_grp_matmul_n_*()`.  Cached-static-const +
///                atomic-override read pattern; see
///                `parse_env_int_strict` (in
///                `group_matmul_parallel_common.hpp`) for the strict
///                env parsing convention.
///
///   Section A.5  N-tile shared utilities (`sort_indices_by_m`,
///                `prepare_n_tile_custom_kernel`,
///                `n_tile_nr_align`, `auto_pick_n_order`,
///                `fill_ntile_expert_order`) — free functions at
///                namespace scope (NOT in `test_api`) consumed by
///                `group_matmul_n_tile.cpp` and any future additional
///                N-tile executor.  Previously lived in
///                `group_matmul_parallel_common.hpp`.
///
///   Section A.6  Forward declaration for the N-tile executor
///                `n_tile_execute()`.  Defined in
///                `group_matmul_n_tile.cpp`.
///
/// What stays in `group_matmul_n_tile.cpp`:
///   * `GroupNTileContext`  — captures the caller's `std::vector<…> &`
///                            inputs by reference; private impl detail
///                            of one OMP region.
///   * Planner functions    — pure decision logic on the structs above.
///   * Strategy executors   — OMP-parallel matmul drivers.
///   * `n_tile_execute()`      — body of the public entry declared in
///                            Section A.5 above.
///
/// Dependency direction:
///   * This header depends on `group_matmul_parallel_common.hpp` for
///     shared types (`matmul_params`, `grp_matmul_gated_act_t`,
///     `data_type_t`) and the `parse_env_int_strict` helper, and on
///     `group_matmul_n_tile_planner.hpp` for the planner output types
///     and the test-only Phase-B capture machinery.
///   * `../group_matmul_parallel_common.hpp` does NOT include this
///     header — consumers of N-tile interfaces include this file
///     explicitly.  One-way dependency, mirrors
///     `../m_tile/group_matmul_m_tile.hpp`.

#ifndef ZENDNNL_GROUP_MATMUL_N_TILE_HPP
#define ZENDNNL_GROUP_MATMUL_N_TILE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>
#include "common/zendnnl_compat.hpp"

#include "common/op_config.hpp" // matmul_algo_t
// `group_matmul_parallel_common.hpp` provides:
//   * `parse_env_int_strict`            — strict env parsing helper
//   * `matmul_params` / `data_type_t`   — used by `n_tile_execute()` decl
//   * `grp_matmul_gated_act_t`          — fused-act argument type
//   * the inline `using namespace zendnnl::common;` declaration
//
// NOTE: `kNTilePlanMaxExperts` previously came from
// `group_matmul_parallel_common.hpp`; it now lives in the companion
// planner header `group_matmul_n_tile_planner.hpp` (re-included
// below) with the other N-tile-specific constants so `GroupNTilePlan`
// can size its stack arrays at struct-definition time.  Consumers
// that include this public N-tile header transitively see the
// symbol via that re-include — no separate include is needed.
#include "../../common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/common/per_thread_scratch.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_knobs.hpp"
#include "lowoha_operators/matmul/group_matmul/selection/call_route.hpp"
// `group_matmul_n_tile_planner.hpp` provides the planner output
// types (`GroupNTileStrategy`, `GroupNTileTopology`, `GroupNTilePlan`,
// `RoundCandidates`, `RoundPick`) plus the test-only Phase-B capture
// machinery (`PhaseBSnapshot`, `s_capture_phase_b`,
// `s_last_phase_b_snapshot`).  Re-included here so any TU that
// includes the public N-tile interface transitively gets the planner
// surface — symmetric with the M-tile pattern in
// `../m_tile/group_matmul_m_tile.hpp`.
#include "group_matmul_n_tile_planner.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Bring `matmul_algo_t` into scope so the `algo` field on
// `GroupNTilePlan` reads as `matmul_algo_t` (matches the unqualified
// usage in the rest of the group_matmul code, which gets the same
// using-decl via `group_matmul_parallel_common.hpp`).
using zendnnl::common::matmul_algo_t;

// N-tile shared constants (`kDecodeNTile`, `kNTilePlanMaxExperts`)
// live in the companion planner header
// `group_matmul_n_tile_planner.hpp` (re-included via line above), at
// the top of that file.  They must be defined there — not here — so
// `GroupNTilePlan`'s stack arrays (sized by `kNTilePlanMaxExperts` at
// struct definition time) see the constant before the struct body is
// parsed.  This header re-includes the planner unconditionally, so
// every consumer of the N-tile public surface (executor, dispatcher,
// gtests, fused-MoE) still observes the constants through
// `n_tile/group_matmul_n_tile.hpp`.

// NOTE: Section A.0 (`AlignedTileScratch`, `grow_scratch`) moved to
// `common/per_thread_scratch.hpp`, included above.

// =====================================================================
// Sections A.1 / A.2 — PLANNER — moved to
//                       `group_matmul_n_tile_planner.hpp` (Sections
//                       P.1 – P.5)
// =====================================================================
//
// The strategy enum (`GroupNTileStrategy`), topology summary
// (`GroupNTileTopology`), per-call plan (`GroupNTilePlan`),
// multi-round cost-model types (`RoundCandidates` / `RoundPick`),
// and the test-only Phase-B capture machinery (`PhaseBSnapshot`,
// `s_capture_phase_b`, `s_last_phase_b_snapshot`) all live in the
// companion planner header so future N-tile planner optimizations
// land in a single, isolated optimization surface — symmetric with
// the M-tile pattern (see `../m_tile/group_matmul_m_tile_planner.hpp`).
//
// This file re-includes the planner header at the top of the
// `#include` block, so every TU that pulls in the public N-tile
// interface continues to see all of the above symbols unchanged.

// NOTE: Sections A.3 (the N-tile test-only override atoms) and A.4 (the
// N-tile env getters) moved to `n_tile_knobs.hpp`, included above.

// =====================================================================
// Section A.5 — N-tile shared utilities (consumed by
//               group_matmul_n_tile.cpp and a future additional N-tile
//               executor; previously lived in
//               `group_matmul_parallel_common.hpp`).
// =====================================================================
//
// Free functions exposed at namespace scope (NOT inside `test_api`):
// these are real library primitives the production N-tile executor
// calls into, not test hooks.  They moved here from
// `group_matmul_parallel_common.hpp` together with the env knobs they
// read (`get_grp_matmul_n_order` for `fill_ntile_expert_order`,
// `get_grp_matmul_custom_kernel` (which stays in parallel_common.hpp)
// for `prepare_n_tile_custom_kernel`).

/// Sort `indices[0..n)` by `M[idx]` (asc or desc).  Heap-free; caller
/// owns the buffer and guarantees `indices.size() >= n` and `M.size() >= n`.
inline void sort_indices_by_m(
        int *indices, int n, const std::vector<int> &M, bool ascending) {
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }
    std::sort(indices, indices + n, [&M, ascending](int a, int b) {
        return ascending ? (M[a] < M[b]) : (M[a] > M[b]);
    });
}

/// Env-gated custom-microkernel engagement for an N-tile executor.
/// Covers BOTH compute regimes:
///
///   * BF16 (`dynamic_quant=false`, default): bf16×bf16→bf16 (or
///     bf16×bf16→f32) — the original `prepare_n_tile_custom_kernel`
///     contract.
///   * DQ-INT8 (`dynamic_quant=true`, `compute_dtype∈{s8,u8}`):
///     hoist-quantised src(s8 or u8) × pre-packed wei(s8) → bf16,
///     per-token src_scale + optional src_zp, per-channel wei_scale.
///
/// Caller stack-allocates a fresh `CallContext` and reads `kctx.enabled`
/// on return.  Controlled by `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL` (master,
/// cached, default ON) AND `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8` (int8
/// sub-toggle, cached, default ON — only consulted for DQ-INT8 calls;
/// bf16 calls ignore it).  So both bf16 and DQ-INT8 reach CK by default,
/// and the master switch means the same thing for every dtype.  Set the
/// int8 sub-toggle to "0" to route DQ-INT8 back to the AOCL DLP sym-quant
/// path instead.  Master "0" disables CK for all dtypes.
/// W4A8 (s4) has no sub-toggle of its own — the master switch governs
/// it, as it does the FP16 family.
///
/// Even with the envs enabled, the dispatcher's per-call contract
/// check (dtype tuple, no transA, α=1, β=0, N % pack_nr, supported
/// act/bias dtypes, is_weights_const = true / empty for every active
/// expert) decides whether the path can actually run; caller falls
/// back to its standard path when kctx.enabled=false.
///
/// Supported `act`: none, swiglu_oai_mul (fused gate+up → halved out),
/// silu_and_mul, gelu_and_mul (canonical split-halves; CK arena
/// pack-permuted at prepack time).
inline void prepare_n_tile_custom_kernel(grp_matmul_gated_act_t act,
        data_type_t src_dtype, data_type_t wei_dtype, data_type_t dst_dtype,
        data_type_t act_dtype, data_type_t bias_dtype,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<int> &ldb,
        const std::vector<float> &alpha, const std::vector<float> &beta,
        const std::vector<const void *> &weight,
        const std::vector<bool> &is_weights_const,
        custom_kernel::CallContext &kctx, bool dynamic_quant = false,
        data_type_t compute_dtype = data_type_t::none,
        // Per-expert "weight already CK-VNNI-packed" signal, forwarded to
        // `prepare_for_call` (built from the caller's per-expert
        // `mem_format_b == 'r'`).  Empty ⇒ no prepacked experts.
        const std::vector<bool> &weights_prepacked = {},
        // W4A8 only; 0 elsewhere, which is the "not an s4 call" signal.
        int group_size = 0, int wei_scale_grp_stride = 0,
        // Explicit AOCL-DLP-blocked provenance. These weights are normalized
        // to raw only inside a validated CK preparation.
        const std::vector<bool> &weights_aocl_prepacked = {},
        int32_t num_threads = 0, int32_t weight_cache_mode = -1,
        // Forwarded to `prepare_for_call`: false when the call will not
        // run the kernel even if the context prepares.
        bool allow_in_place = true,
        const std::vector<size_t> &wei_capacities = {}) {
    if (!get_grp_matmul_custom_kernel()) { return; }
    // Master CK env is ON; gate the DQ-INT8 sub-toggle separately so
    // operators can toggle int8 without disabling bf16.  The int8 CK path
    // arrives two ways and BOTH must honour the sub-toggle:
    //   * runtime hoist  — `dynamic_quant=true` with s8 weights
    //     (bf16 src quantized per-tile);
    //   * grouped pre-quant — `group_dynamic_quant` already produced an
    //     s8 src and CLEARED `dynamic_quant`, so detect it via
    //     `src=s8 && wei=s8`.  Without this, a grouped-s8 call would
    //     engage CK even with the int8 sub-toggle OFF (inconsistent with
    //     `ck_eligible_int8` / prepack which honour the sub-toggle).
    // Bf16 calls (`src=bf16, wei=bf16`) never satisfy either clause.
    const bool is_dq_int8_call = wei_dtype == data_type_t::s8
            && (dynamic_quant || src_dtype == data_type_t::s8
                    || src_dtype == data_type_t::u8);
    if (is_dq_int8_call && !get_grp_matmul_custom_kernel_int8()) return;
    // FP16 CK sub-toggle — independent from the master + int8 knobs so
    // operators can A/B the native AVX-512-FP16 fast path without
    // disturbing the bf16 / int8 paths.  An f16×f16 call honours it;
    // bf16 / int8 calls never satisfy this clause.
    const bool is_f16_call
            = (src_dtype == data_type_t::f16 && wei_dtype == data_type_t::f16);
    if (is_f16_call && !get_grp_matmul_custom_kernel_f16()) return;
    custom_kernel::prepare_for_call(act, src_dtype, wei_dtype, dst_dtype,
            act_dtype, bias_dtype, transA, transB, M, N, K, ldb, alpha, beta,
            weight, is_weights_const, kctx, dynamic_quant, compute_dtype,
            weights_prepacked, group_size, wei_scale_grp_stride,
            weights_aocl_prepacked, num_threads, weight_cache_mode,
            allow_in_place, wei_capacities);
}

/// Effective N-column alignment for the per-thread split:
///   max(backend_nr, kctx.pack_nr if the custom kernel will RUN, 2 if
///   pair_aligned).
/// `backend_nr` is 1 (the backends group matmul runs take any column
/// split); `pair_aligned`=true when activation requires even col
/// boundaries (e.g. swiglu_oai_mul must keep gate+up pairs on the same
/// thread).
///
/// `ck_will_run` is the resolved `use_custom`, NOT `kctx.enabled`.  The two
/// differ whenever the context prepared successfully but the call was then
/// disqualified -- a per-group `{G, N}` weight scale, or the wide fused
/// swiglu guard.  Widening to `pack_nr` in those cases was wrong twice
/// over: the tiles are handed to AOCL DLP, which has no pack_nr contract,
/// and the coarser split produced a per-tile cache key set that no warmer
/// builds, so every tile missed on the first decode call.  Keying on
/// `use_custom` makes the split, the warm and the backend agree.
inline int n_tile_nr_align(int backend_nr,
        const custom_kernel::CallContext &kctx, bool pair_aligned,
        bool ck_will_run) {
    int a = backend_nr;
    if (ck_will_run && kctx.enabled) { a = std::max(a, kctx.pack_nr); }
    if (pair_aligned) { a = std::max(a, 2); }
    return a;
}

// `kNTileMaxExperts` previously duplicated `kNTilePlanMaxExperts` as
// a separate `= 256` constant for the expert-ordering helpers below.
// Removed to keep `kNTilePlanMaxExperts` (defined in the companion
// planner header `group_matmul_n_tile_planner.hpp`, re-included by
// this file at the top) as the SINGLE source of truth for the N-tile
// expert capacity.  Callers below now reference
// `kNTilePlanMaxExperts` directly so a future bump cannot leave the
// order helpers out of sync with the auto-selector and the planner's
// stack-array sizing.

/// Auto-pick N_ORDER from num_ops.  Returns 3 (pair_balanced) or 0
/// (walk input order); explicit modes 1/2/4 only reachable via env.
///
/// The ordering choice depends on num_ops:
///   * Few-experts (≤ ~kSmallExpertsCutoff): pair-balanced ordering's
///     bin-packing keeps per-round max-M balanced.
///   * Many-experts (≥ ~kLargeExpertsCutoff): pair-balanced applies
///     again.
///   * Mid-band: default to walk-input (lower overhead, zero
///     permutation cost).
/// The cutoffs are calibrated against the dispatcher's stable-N-tile
/// plan and should be re-evaluated together if that planner changes.
inline int auto_pick_n_order(int num_ops) {
    if (num_ops <= 18) {
        return 3; // few-experts regime — pair-balanced is the better default
    }
    if (num_ops >= 26) {
        return 3; // many-experts regime — pair-balanced is the better default
    }
    return 0; // mid-band — walk input order
}

/// Write `out[0..out_size)` with expert indices ordered per
/// `get_grp_matmul_n_order()`.  `out_size = 0` signals "walk input
/// order, ignore out" (auto-mode resolved to no permutation).
///
/// Heap-free: stack array of kNTilePlanMaxExperts for the desc-sort
/// temp; beyond that the ordering is skipped (correct, just unsorted).
/// Mode 4 is O(num_ops²) ≤ 64K comparisons — low overhead vs the GEMM body.
///
/// `auto_resolved_out` (optional): when env mode = 0, the resolved
/// concrete sub-mode is written here for APILOG diagnostics.
inline void fill_ntile_expert_order(int *out, int &out_size, int max_size,
        const std::vector<int> &M, int num_ops,
        int *auto_resolved_out = nullptr) {

    if (num_ops <= 0 || num_ops > max_size || num_ops > kNTilePlanMaxExperts) {
        out_size = 0;
        return;
    }

    int order = get_grp_matmul_n_order();

    // Mode 0 — auto: shape-aware sub-mode selection.
    if (order == 0) {
        order = auto_pick_n_order(num_ops);
        if (auto_resolved_out != nullptr) { *auto_resolved_out = order; }
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
        std::array<int, kNTilePlanMaxExperts> sorted_desc {};
        sort_indices_by_m(sorted_desc.data(), num_ops, M,
                /*ascending=*/false);
        int lo = 0, hi = num_ops - 1, o = 0;
        while (lo <= hi) {
            out[o++] = sorted_desc[lo++];
            if (lo <= hi) { out[o++] = sorted_desc[hi--]; }
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
        std::array<int, kNTilePlanMaxExperts> sorted_desc {};
        sort_indices_by_m(sorted_desc.data(), num_ops, M,
                /*ascending=*/false);

        int64_t total = 0;
        for (int i = 0; i < num_ops; ++i) {
            total += M[i];
        }

        std::array<bool, kNTilePlanMaxExperts> used {}; // zero-init
        int64_t cum = 0;
        for (int p = 0; p < num_ops; ++p) {
            // Error metric: |target_scaled − num_ops × new_cum| where
            // target_scaled = (p + 1) × total.  Integer-only; scales
            // cancel out across candidates.
            const int64_t target_scaled = static_cast<int64_t>(p + 1) * total;
            int best_j = -1;
            int64_t best_err = std::numeric_limits<int64_t>::max();
            for (int j = 0; j < num_ops; ++j) {
                if (used[j]) { continue; }
                const int64_t new_cum = cum + M[sorted_desc[j]];
                const int64_t err = std::llabs(target_scaled
                        - static_cast<int64_t>(num_ops) * new_cum);
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

// =====================================================================
// Section A.6 — n_tile_execute() forward declaration
// =====================================================================

/// ALGO 3 — N-tile parallel GEMM, with optional fused-swiglu-oai
/// epilogue.  Defined in group_matmul_n_tile.cpp.
///
/// `execution_path_out` (optional) receives a static string naming the
/// concrete path that ran: `"n_tile_execute"`, `"flat_n_tile_custom"`,
/// `"flat_n_tile_fused_swiglu_oai"`, or `"flat_n_tile_fused_swiglu_oai_custom"`.
/// The caller is expected to thread this through to its own
/// execution_path_out so benchdnn / profiler output reveals whether the
/// custom microkernel engaged.
///
/// An ALGO-3 MoE call, or one whose weights only the custom kernel can
/// read, with more than `kNTilePlanMaxExperts` experts runs as consecutive
/// expert slices of at most that many; `execution_path_out` then names the last
/// slice's path, or the first failing slice's `error_` sentinel, after which
/// no further slice runs.
///
/// `route` carries the dispatcher's per-call verdicts
/// (`grp_matmul_call_route`): the MoE family, the custom-kernel weight-cache
/// mode, and the licence to write that pack in place while AUTO mixed
/// in-place mode is active (`prepare_auto_weight_cache`).  Without
/// one, the executor classifies the call itself and grants no licence.
void n_tile_execute(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        grp_matmul_gated_act_t fused_act = grp_matmul_gated_act_t::none,
        data_type_t act_dtype = data_type_t::none,
        const char **execution_path_out = nullptr,
        const std::vector<void *> *w4a8_s8_weights = nullptr,
        const grp_matmul_call_route *route = nullptr);

/// @brief Drop every memoized f32 weight-scale view.
///
/// The memo is pointer-keyed and process-wide, so it carries the same
/// heap-address-reuse hazard as the CK pack arenas: a freed scale buffer
/// whose address is recycled would otherwise hit a stale entry.  Tests
/// call this via `reset_grp_matmul_caches()`; a server rotating weights
/// can call it in a quiescent window.  Not safe to call concurrently with
/// an in-flight `n_tile_execute`.
///
/// Named without a `_for_test` suffix deliberately, matching
/// `clear_custom_kernel_pack_cache_bf16()`: the hazard it clears is a
/// production one (framework allocators recycle weight addresses across
/// model reload / adapter rotation), and a framework integration has no
/// other way to invalidate the memo while
/// `ZENDNNL_MATMUL_WEIGHT_CACHE` is at its default of 1.
void clear_weight_scale_f32_cache();

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_N_TILE_HPP
