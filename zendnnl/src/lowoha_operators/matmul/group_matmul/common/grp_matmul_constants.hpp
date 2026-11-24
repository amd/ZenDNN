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

/// Group-matmul constants: ALGO ids, phase classification, tile and
/// expert-count thresholds, abort sentinels and execution_path decoding.

#ifndef ZENDNNL_GRP_MATMUL_CONSTANTS_HPP
#define ZENDNNL_GRP_MATMUL_CONSTANTS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/op_config.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Match the original group_matmul_dispatch.cpp using-declarations so
// source files including this header can refer to matmul_algo_t,
// matmul_config_t, post_op_type_t, etc. without namespace prefixes.
using namespace zendnnl::common;
using zendnnl::common::size_of;

// Group-matmul selector identities.  These are deliberately independent of
// `matmul_algo_t`: that enum names inner GEMM kernels, while these values name
// whole-call grouped scheduling/interception modes.
//
// Selector 4 is ALGO 4, the N-tile flat-parallel MoE executor; selector 6
// owns the generic
// multilevel scheduler.
inline constexpr int kGrpMatmulAlgoAuto = 0;
inline constexpr int kGrpMatmulAlgoMultilevel = 6;
inline constexpr int kGrpMatmulAlgoNTileFlatParallel = 4;
// One whole expert per thread, no intra-expert split.  Named for the same
// reason as its siblings above: the selector compares against it in several
// files, and a bare `5` there says nothing about the strategy it selects.
inline constexpr int kGrpMatmulAlgoExpertParallel = 5;

// Team size for the short passes around the grouped GEMMs (gated activation,
// weighted reduce, tight-dst copy, StaticExpertTeams, ALGO 5): a region opens either
// the caller's whole team or a serialized region, never a team in between.
// A serialized region leaves the pool alone, while any other size makes the
// runtime rebuild the pool there and again at the next GEMM region.
//
// Row-wise passes that run with fewer rows than threads split each row into
// column chunks until the team has work.  Chunk edges are multiples of
// `kColChunk` columns: a multiple of the 16-lane vector width keeps every
// column on the same vector-or-scalar path as a whole-row pass, so chunked
// output is bit-identical, and 64 keeps neighbouring chunks off each other's
// output cache lines.
inline constexpr int kColChunk = 64;

inline int col_chunks_per_row(int64_t rows, int width, int num_threads) {
    if (num_threads <= 1 || rows <= 0 || rows >= num_threads) return 1;
    const int64_t want = (num_threads + rows - 1) / rows;
    return static_cast<int>(
            std::max<int64_t>(1, std::min<int64_t>(want, width / kColChunk)));
}

inline int col_chunk_span(int width, int chunks) {
    const int per = (width + chunks - 1) / chunks;
    return (per + kColChunk - 1) / kColChunk * kColChunk;
}

inline constexpr bool is_grp_matmul_generic_algo(int algo) {
    return algo == 1 || algo == 2 || algo == 3
            || algo == kGrpMatmulAlgoExpertParallel
            || algo == kGrpMatmulAlgoMultilevel;
}

inline constexpr bool is_grp_matmul_phase_algo(int algo) {
    return algo == kGrpMatmulAlgoAuto || is_grp_matmul_generic_algo(algo);
}

inline constexpr bool is_grp_matmul_requested_algo(int algo) {
    return is_grp_matmul_phase_algo(algo)
            || algo == kGrpMatmulAlgoNTileFlatParallel;
}

// Short string renderer for `grp_matmul_gated_act_t` — used by APILOG
// lines and execution_path_out strings; keeps activation names consistent
// across executors.
inline const char *act_name(grp_matmul_gated_act_t a) {
    switch (a) {
        case grp_matmul_gated_act_t::none: return "none";
        case grp_matmul_gated_act_t::silu_and_mul: return "silu_and_mul";
        case grp_matmul_gated_act_t::gelu_and_mul: return "gelu_and_mul";
        case grp_matmul_gated_act_t::swiglu_oai_mul: return "swiglu_oai_mul";
    }
    return "?";
}

// Tile-size + weight-class constants shared by ALGO 0 auto-select and
// the tile kernels.  Cutoffs separate per-CCD-L3-resident shapes
// (small) from L3-tight shapes (medium) from DRAM-streaming (large).
inline constexpr int kDecodeMaxM = 64; // per-expert M ≤ this → "decode"
inline constexpr int kMinNTile = 512; // prompt-path per-thread N

/// Runtime phase used by AUTO group-matmul policy and the ALGO 4 request.
///
/// A call is decode when the maximum M in its active operation prefix is at
/// most `kDecodeMaxM`; otherwise it is prompt. Empty prefixes are decode (their
/// effective max M is zero), although normal dispatch short-circuits them
/// before phase-specific execution.
enum class grp_matmul_phase { decode, prompt };

inline constexpr grp_matmul_phase classify_grp_matmul_phase(int max_active_m) {
    return max_active_m <= kDecodeMaxM ? grp_matmul_phase::decode
                                       : grp_matmul_phase::prompt;
}

inline int max_active_grp_matmul_m(
        const std::vector<int> &M, size_t active_prefix) {
    const size_t count = std::min(active_prefix, M.size());
    if (count == 0) return 0;
    return *std::max_element(
            M.begin(), M.begin() + static_cast<std::ptrdiff_t>(count));
}

/// Classify only the first `active_prefix` entries: trailing M entries may
/// describe prepack-only experts and must not turn a decode call into prompt.
inline grp_matmul_phase classify_grp_matmul_phase(
        const std::vector<int> &M, size_t active_prefix) {
    return classify_grp_matmul_phase(max_active_grp_matmul_m(M, active_prefix));
}

inline grp_matmul_phase classify_grp_matmul_phase(const std::vector<int> &M) {
    return classify_grp_matmul_phase(M, M.size());
}

inline constexpr const char *grp_matmul_phase_name(grp_matmul_phase phase) {
    return phase == grp_matmul_phase::decode ? "decode" : "prompt";
}

// NOTE: the single-op scope predicates (`is_single_dense_expert`,
// `is_dense_ffn_decode`) moved to `single_expert/single_expert_policy.hpp`.

// DQ-INT8 sibling of `kMinNTile`.  Today set equal to the bf16
// value so the dtype-aware split is a structural no-op.  Kept as an
// independent constant so the int8 family can be retuned without
// perturbing the bf16 prompt thresholds.
inline constexpr int kMinNTileInt8 = 512;

// NOTE: `kDecodeNTile` (decode-path per-thread N, default 256) now
// lives in `executors/n_tile/group_matmul_n_tile_planner.hpp` (re-included by
// the public N-tile header `executors/n_tile/group_matmul_n_tile.hpp`) with
// the rest of the N-tile-specific constants.  Consumers (dispatcher,
// N-tile executor, custom-kernel dispatch, gtests) include the
// public N-tile header anyway to reach `n_tile_execute()`, so the
// symbol remains visible to them transparently.
//
// NOTE: `kMediumWeight` (64 MB / expert, "referenced by N-tile
// planner") was unused throughout the tree — no production or test
// site read it on either `origin/main` or this branch — so it was
// dropped instead of moved.  Re-introduce only when a consumer needs
// it.

/// Few-experts threshold for ALGO 0 auto-select rule 2.  Workloads
/// with `num_ops ≤ kFewExpertsAlgo1` AND `num_ops < num_threads`
/// (rule 1's strict `>=` would otherwise win on a ≤ 8-thread host)
/// pin to ALGO 1 (sequential experts with full-team AOCL DLP) for
/// prompt AND decode.  Targets few-expert layers, where the per-expert
/// weight footprint is large enough that the full-team sequential path
/// is a better fit than N-tile's column slices on a thin per-expert
/// thread budget.
///
/// The `num_ops < num_threads` precondition holds for a few-expert
/// layer on typical multi-CCD hosts.  An 8-expert workload on a
/// ≤ 8-thread host instead falls to rule 1 and routes to ALGO 3.
/// Documented in `auto_select_algo`'s rule precedence comment in
/// `group_matmul_dispatch.cpp`.
inline constexpr int kFewExpertsAlgo1 = 8;

/// Total-expert ceiling for Rule 0.5's decode occupancy policy. When the
/// global ALGO is AUTO and the decode phase is not explicitly pinned, calls
/// at or below this threshold choose between ALGO 3 and ALGO 1 using active
/// expert occupancy. Larger expert pools inherit the normal decode default.
inline constexpr int kFewExpertsDecodeThreshold = 8;

/// Threads-per-active-expert factor for the few-expert DECODE arrow
/// (Rule 0.5 in `auto_select_algo`).  ALGO 3 splits each expert's N across
/// the team, so it needs enough ACTIVE experts to be worth its round-based
/// schedule: the rule takes ALGO 3 when
/// `active_ops * kDecodeNTileThreadFactor >= num_threads`, i.e. when there is
/// at most this many threads per active expert.  Below that the team is too
/// wide for the expert count and full-team sequential (ALGO 1) wins — the
/// Mixtral-class shape (8 experts, topk=2 → 2 active on a 32+ thread team)
/// lands there.  Dtype-agnostic by design; only the `n_tile_safe` legality
/// clamp differs per dtype.
inline constexpr int kDecodeNTileThreadFactor = 4;

// ── Abort-class execution_path sentinels ─────────────────────────────────────
// The group_matmul executors are void and the surrounding API is noexcept, so
// a condition that leaves the caller's dst wrong is reported by writing one of
// these strings through `execution_path_out`.  `group_matmul_direct` and the
// fused-MoE dispatcher turn any mode matching `execution_path_is_error` into
// `status_t::failure`, so the caller never consumes a dst the library knows is
// bad.
//
// New sentinels MUST keep the `error_` prefix: the check is by prefix so a
// sentinel added in an executor fails the call closed even if a translation
// site is not updated.
inline constexpr const char *kGrpMatmulErrPrefix = "error_";
inline constexpr const char *kGrpMatmulErrPrepackedNoCk
        = "error_prepacked_no_ck";
/// An active expert's weight buffer holds the image an earlier in-place
/// custom-kernel pack (WEIGHT_CACHE=2) wrote over it, and the resolved route
/// does not run the custom kernel.  Any other reader would take the image for
/// the raw weight.
inline constexpr const char *kGrpMatmulErrPackedInPlaceNoCk
        = "error_packed_in_place_no_ck";
/// WC=2 cannot safely serve a BF16/FP16 MoE layer (or an INT8 one with a
/// declared writable capacity) when a phase override splits a non-AUTO global
/// pin across different weight-layout owners.
inline constexpr const char *kGrpMatmulErrWc2PhaseSplitUnsupported
        = "error_wc2_global_phase_split_unsupported";
/// ALGO 3 could not leave dst defined: a per-thread scratch allocation failed,
/// or a fused-epilogue layout invariant was violated.  `n_tile_execute` logs the
/// specific reason; the sentinel only has to fail the call.
inline constexpr const char *kGrpNTileErrUndefinedDst
        = "error_ntile_undefined_dst";
/// The caller passed a tight destination (`ldc < N`) with a gated activation,
/// but the resolved route has no tight-aware writer, so applying the
/// activation would walk past the end of the caller's buffer.
inline constexpr const char *kGrpMatmulErrTightNoFusedWriter
        = "error_tight_dst_no_fused_writer";
/// A weight-cache lookup missed on a buffer an earlier in-place (WC=2) reorder
/// had already rewritten, so the AOCL backend refused to reorder it a second
/// time (the raw weights are gone and the result would be silently wrong).
/// The backend can only count the refusal -- `run_dlp` returns void and runs
/// inside OMP regions -- so the dispatcher compares the count across the call
/// and raises this, which fails the call rather than returning the dst that
/// the refused reorder left undefined.
inline constexpr const char *kGrpMatmulErrMutatedWeightMiss
        = "error_mutated_weight_miss";
/// Simulated W4A8 on grouped ALGO 3 (and AUTO, when that route is live)
/// caches the s4→s8 conversion by weight address. A non-constant expert
/// would reuse that conversion after the caller rewrites the same buffer,
/// so the call is refused before GEMM.
inline constexpr const char *kGrpMatmulErrW4a8NonConst
        = "error_w4a8_nonconst_weight";
/// The custom kernel, MXFP4's only backend, did not serve every expert of an
/// f4_e2m1 call, so the call fails rather than reach a backend that would
/// read the codes as another type.
inline constexpr const char *kGrpMatmulErrMxfp4NoCk = "error_mxfp4_no_ck";

inline bool execution_path_is_error(const char *mode) {
    return mode != nullptr
            && std::strncmp(mode, kGrpMatmulErrPrefix,
                       std::strlen(kGrpMatmulErrPrefix))
            == 0;
}

// ── Executed-ALGO from execution_path ────────────────────────────────────────
// Maps the executor-written `execution_path` string (the authoritative record of
// what ACTUALLY ran) to the ALGO that actually executed ({1,2,3,5,6}), so the
// post-exec `[GRP_MATMUL.CALL]` line can surface `exec_algo=` alongside
// `mode=`.  Together with the pre-exec `[GRP_MATMUL.ALGO] chosen=` selection
// line, this makes any selection-vs-execution divergence explicit instead of
// silent (e.g. a forced ALGO 2 that clamps to sequential-full-team reports
// `chosen=ALGO_2 ... exec_algo=1 mode=flat_m_tile_seq_clamp`).
//
// Returns 0 for null / unrecognised modes OR for explicit no-op markers
// (`*_skip` — nothing executed), and a generic scheduler ID for a recognised
// executed path.
// So `exec_algo=0` means "no GEMM ran or mode not understood", NOT "ALGO 0".
// ORDER MATTERS: the `flat_m_tile_seq_clamp` special case (ALGO-1 behaviour
// wearing an ALGO-2 mode prefix) and the `*_skip` markers must be checked
// before the generic `flat_m_tile` / `m_tile_execute` / `multilevel` /
// `fused_moe` prefixes they share.  Fused composites
// (`fused_moe_*(op1=..,op2=..)`) derive the algo from the Op1 sub-mode; the
// full per-op detail stays in the `mode=` string itself.
inline int executed_algo_from_execution_path(const char *mode) {
    if (mode == nullptr) return 0;
    auto starts = [&](const char *p) {
        return std::strncmp(mode, p, std::strlen(p)) == 0;
    };
    // Abort-class sentinels: the call is being failed, so no ALGO produced a
    // usable result.  Checked first — an `error_` mode must never be reported
    // as a successful execution of the path that raised it.
    if (execution_path_is_error(mode)) return 0;
    // No-op / nothing-executed markers map to 0 (must precede the generic
    // prefixes they share, e.g. "flat_m_tile_skip" before "flat_m_tile").
    if (starts("skip")) return 0; // whole-call no-op (all M<=0)
    if (starts("flat_m_tile_skip")) return 0; // empty / no active expert
    if (starts("multilevel_skip")) return 0;
    if (starts("fused_moe_skip")) return 0;
    if (starts("flat_m_tile_seq_clamp")) return 1; // sequential full-team
    if (starts("sequential")) return 1; // "sequential" / "..._experts"
    // Keep accepting the legacy flat_* execution-path vocabulary while
    // producers migrate independently to the executor-oriented names.
    if (starts("m_tile_execute") || starts("flat_m_tile")) return 2;
    // Must precede the generic `vertical_fusion` arm, which would otherwise
    // claim it by prefix and report an ALGO 5 execution as ALGO 2.
    if (starts("vertical_fusion_expert_parallel")) {
        return kGrpMatmulAlgoExpertParallel;
    }
    if (starts("vertical_fusion")) return 2; // M-tile fused pipeline
    if (starts("n_tile_execute") || starts("flat_n_tile")) return 3;
    if (starts("ntile_flat_parallel")) return kGrpMatmulAlgoNTileFlatParallel;
    if (starts("multilevel")) return kGrpMatmulAlgoMultilevel;
    if (starts("per_expert")) return kGrpMatmulAlgoExpertParallel;
    if (starts("fused_moe")) {
        // Composite: derive from the Op1 executor sub-mode.
        const char *op1 = std::strstr(mode, "op1=");
        return (op1 != nullptr) ? executed_algo_from_execution_path(op1 + 4)
                                : 0;
    }
    return 0;
}

// NOTE: `kNTilePlanMaxExperts` (max experts the ALGO 3 N-tile planner
// can represent, = 256) now lives in
// `executors/n_tile/group_matmul_n_tile_planner.hpp` (re-included by the
// public N-tile header `executors/n_tile/group_matmul_n_tile.hpp`) with the
// other N-tile-specific constants.  The auto-selector still reads it
// for its rule-0 capacity exit to ALGO 1, which an ALGO-3 MoE layer
// skips because ALGO 3 slices such a call; the dispatcher
// includes the public N-tile header directly to call `n_tile_execute()`,
// so the symbol remains visible unchanged.

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

// NOTE: `m_tile_supported` (M-tile ALGO 2 structural eligibility
// predicate) moved to `executors/m_tile/group_matmul_m_tile.hpp` (Section H.5)
// so the M-tile-only gate sits next to the M-tile executor it
// protects.  Both callers — the legacy dispatcher
// (`group_matmul_execute`) and the MoE vertical-fusion
// dispatcher fork (`group_matmul_fused_moe_execute`) — already
// include `executors/m_tile/group_matmul_m_tile.hpp` to reach `m_tile_execute()`
// / `try_fused_m_tile_core()`, so they see the predicate
// unchanged.

/// True when ALGO 3 can fuse `act` into the per-thread epilogue.
/// Extend here when new activation kinds gain epilogue support.
///
/// Currently supports:
///   * `swiglu_oai_mul` — interleaved-input path (caller-side
///     interleaved W13).  Both the CK in-register epilogue
///     (`swiglu_oai_store_pair`) and the standard-backend
///     wide-arena helper (`apply_swiglu_oai_tile_rows`) handle it,
///     so the fused path is supported regardless of `use_custom_kernel`.
///   * `silu_and_mul` — split-halves input (canonical W13).  Only
///     the CK in-register epilogue (`silu_and_mul_store_pair`)
///     handles it today; the standard backend's
///     `apply_swiglu_oai_tile_rows` is swiglu-only and has no silu
///     sibling, so this kind is fusible only when CK is engaged.
///   * `gelu_and_mul` — split-halves input (canonical W13).  Same
///     story as silu: only the CK in-register epilogue
///     (`gelu_and_mul_store_pair`) implements the fused form, with
///     a `gelu_tanh` polynomial approximation that matches the
///     reference's `gelu_erf` to within BF16 tolerance.  The
///     standard backend's wide-arena helper is swiglu-only, so
///     gelu fused requires `use_custom_kernel=true`.
///
/// Standard-backend silu / gelu tile helpers are a planned follow-up;
/// until then, callers with CK off fall back to the separate-pass
/// path (`act_fused = false`) by way of this gate returning `false`.
inline bool n_tile_can_fuse_activation(
        grp_matmul_gated_act_t act, bool use_custom_kernel) {
    if (act == grp_matmul_gated_act_t::swiglu_oai_mul) return true;
    if (act == grp_matmul_gated_act_t::silu_and_mul) return use_custom_kernel;
    if (act == grp_matmul_gated_act_t::gelu_and_mul) return use_custom_kernel;
    return false;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GRP_MATMUL_CONSTANTS_HPP
