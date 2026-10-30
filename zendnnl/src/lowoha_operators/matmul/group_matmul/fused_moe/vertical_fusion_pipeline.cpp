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

/// Fused-MoE vertical-fusion pipeline executor (W13 -> gated activation ->
/// W2 per M-tile slice) and its eligibility gate.

#include "lowoha_operators/matmul/group_matmul/fused_moe/vertical_fusion_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <vector>

#include <omp.h>

#include "lowoha_operators/matmul/group_matmul/common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/common/per_thread_scratch.hpp" // AlignedTileScratch + grow_scratch
#include "lowoha_operators/matmul/group_matmul/executors/m_tile/group_matmul_m_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/m_tile/m_tile_internal.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"
// DQ-INT8 vertical fusion (Section C, DQINT8 regime) uses the standard
// pre-OMP per-expert src-quantization hoist mirrored from the N-tile
// executor.  Both the RAII buffer holder (`reorder_quant_buffers_t`)
// and the bf16/f32 → s8 reorder entry (`reorder_quantization_wrapper`)
// live in this header; the per-thread Stage 2b re-quant kernel
// (`dynamic_per_token_quant_bf16_s8_native`) lives in the reorder
// dynamic-kernels header included below.
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp" // float_to_bf16
#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

// ── A.6  DQ-INT8 vertical-fusion pre-OMP hoist primitives ──────────────
//
// Used ONLY by the vertical-fusion executor (Section C) when its
// eligibility gate enters the `kRegimeDQINT8` arm.  Two small
// utilities:
//
//   `HoistedSrcQuant_mtile` — read-only per-expert handle to the
//        shared bf16→s8 quantized source produced by the pre-OMP
//        hoist loop.  Mirror of the same-named struct in the N-tile
//        executor's anonymous namespace
//        (`group_matmul_n_tile.cpp::HoistedSrcQuant`, line ~179).
//        Duplicated here intentionally — extracting to a shared
//        header would couple two otherwise-independent executors;
//        the struct is tiny (6 fields, no methods).
//
//   `dqint8_compact_and_requant_slice` — Stage 2b helper that takes
//        a row-strided post-activation bf16 tile in the per-thread
//        scratch and writes (i) a contiguous bf16 tile sized
//        `[slice_M × K_w2]` into a pre-allocated compact buffer
//        and (ii) the per-row symmetric s8 quantization output +
//        per-row f32 scale, using the existing reorder kernel
//        `dynamic_per_token_quant_bf16_s8_native`.  The compaction
//        step is required because the per-token quant kernel
//        assumes a contiguous `[M, N]` source layout, but our
//        post-activation tile lives at stride `N_w13[e]` (the
//        pre-activation column count) with the live cols sitting in
//        `[0, K_w2[e])` per row.  All three output buffers are
//        owned by RAII `reorder_quant_buffers_t` slots allocated
//        pre-OMP at the dispatcher (one slot per thread, sized to
//        that thread's max slice).
//
// Lifetime: identical to the N-tile pattern.  Backing buffers live
// in stack-scoped `std::vector<reorder_quant_buffers_t>` on the
// executor's stack; the OMP region only reads from them (Stage 1
// reads the per-expert hoisted s8 src; Stage 2b writes the
// per-thread compact bf16 / s8 / scale slots; Stage 3 reads them
// back).  When the executor returns, both vectors go out of scope
// and every `~reorder_quant_buffers_t` frees its `src_buf /
// scale_buf / zp_buf` deterministically — no per-slice mallocs,
// no thread_local persistence.
struct HoistedSrcQuant_mtile {
    const void *src_ptr = nullptr;
    int lda = 0;
    data_type_t src_dtype = data_type_t::none;
    matmul_quantization_params_t::matmul_quant_t src_scale;
    matmul_quantization_params_t::matmul_quant_t src_zp;
    bool valid = false;
};

// Number of K-groups encoded in a source-scale tensor: per-group `{M, G>1}`
// returns G; per-token `{M, 1}`, per-tensor, or empty returns 1.  Used by the
// vertical-fusion pipeline to size / drive the Op2 re-quant and Stage-3 src
// scale dims uniformly for both granularities (G==1 is the per-token path).
inline int mtile_src_group_count(
        const matmul_quantization_params_t::matmul_quant_t &q) {
    if (q.dims.size() == 2 && q.dims[1] > 1) {
        return static_cast<int>(q.dims[1]);
    }
    return 1;
}

inline void dqint8_compact_and_requant_slice(
        const void *sc_buf, // post-activation bf16 tile (strided)
        int slice_M, int k_w2, int row_stride_bf16_elems,
        uint8_t *compact_bf16_buf, // [slice_M × k_w2] bf16 contig
        int8_t *int8_buf, // [slice_M × k_w2] s8 contig
        float *scale_buf, // [slice_M × src_groups] f32 scratch
        int src_groups, // 1 = per-token {M,1}; G>1 = per-group {M,G}
        data_type_t scale_dt) { // GEMM-side src-scale dtype: f32 or bf16
    // (1) Compact the row-strided post-act tile to a contiguous
    // [slice_M, k_w2] bf16 buffer.  Per-row memcpy of `k_w2 × 2 B`
    // from `sc_buf + m × row_stride_bf16_elems × 2 B` — L1/L2 hot
    // because the activation just wrote it.
    const size_t row_bytes = static_cast<size_t>(k_w2) * 2u;
    const size_t src_row_stride_bytes
            = static_cast<size_t>(row_stride_bf16_elems) * 2u;
    for (int m = 0; m < slice_M; ++m) {
        std::memcpy(compact_bf16_buf + static_cast<size_t>(m) * row_bytes,
                static_cast<const uint8_t *>(sc_buf)
                        + static_cast<size_t>(m) * src_row_stride_bytes,
                row_bytes);
    }
    // (2) Fused symmetric bf16→s8 + f32 scale write on the contiguous tile.
    //   * per-token  (src_groups == 1): one absmax/scale per row → {slice_M}.
    //   * per-group  (src_groups  > 1): per-K-group absmax → {slice_M, G},
    //     pairing with the per-group {G, N} down-weight scale.  Row-local, so
    //     the M-tile per-thread slicing stays race-free.
    if (src_groups > 1) {
        zendnnl::lowoha::reorder::dynamic_per_group_quant_bf16_s8_native(
                reinterpret_cast<const uint16_t *>(compact_bf16_buf), int8_buf,
                scale_buf, static_cast<int64_t>(slice_M),
                static_cast<int64_t>(k_w2), static_cast<int64_t>(src_groups));
    } else {
        zendnnl::lowoha::reorder::dynamic_per_token_quant_bf16_s8_native(
                reinterpret_cast<const uint16_t *>(compact_bf16_buf), int8_buf,
                scale_buf, static_cast<int64_t>(slice_M),
                static_cast<int64_t>(k_w2));
    }
    // (3) Narrow f32 → bf16 in place when the GEMM expects bf16 src scales
    // (the GGML / AOCL-reordered weight carries a bf16 {G,N} wei_scale, and
    // the AOCL sym-quant kernel requires the A/B scale dtypes to agree).  The
    // native quant kernels only emit f32, so convert the small [slice_M×G]
    // scratch.  Forward in-place is safe: the bf16 write at byte offset 2·i
    // never overruns the not-yet-read f32 at byte offset 4·i.  f32 callers
    // (per-token VF, plain per-group with f32 wei_scale) skip this untouched.
    if (scale_dt == data_type_t::bf16) {
        const int64_t n = static_cast<int64_t>(slice_M)
                * static_cast<int64_t>(src_groups);
        // Store the bf16 result through a byte pointer (uint8_t may alias any
        // type) so the forward in-place narrowing stays well-defined: a
        // `uint16_t *` view of the `float *` buffer would violate strict
        // aliasing and let the compiler reorder the overlapping loads/stores.
        uint8_t *out_bytes = reinterpret_cast<uint8_t *>(scale_buf);
        for (int64_t i = 0; i < n; ++i) {
            const uint16_t bf16
                    = zendnnl::lowoha::reorder::float_to_bf16(scale_buf[i]);
            std::memcpy(out_bytes + static_cast<size_t>(2 * i), &bf16,
                    sizeof(bf16));
        }
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
// SECTION C — Vertical-fusion pipeline M-tile executor
// ═══════════════════════════════════════════════════════════════════════
//
// Vertical-fusion: a SINGLE OMP region that runs the entire MoE FFN
// inner pipeline (W13 matmul → gated activation → W2 matmul) per
// row-disjoint M-tile slice.  Each thread keeps the `(slice_M × 2·I)`
// bf16 intermediate in its own scratch buffer across all three stages
// — no global barrier between stages, no DRAM round-trip of the
// intermediate.  Engaged from `group_matmul_fused_moe_execute` when
// the eligibility gate passes AND
// `ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION ∈ {0 AUTO, 1 FORCED}`.
//
// Capture tag:
//   * `kVerticalFusionBF16`   = 4  when bf16 weights on both halves.
//   * `kVerticalFusionWOQ`    = 5  when int4 (s4/u4) weights on both
//                                  halves (bf16 src + bf16 dst still).
//   * `kVerticalFusionDQINT8` = 6  when per-token symmetric DQ-INT8
//                                  on both halves (bf16 src + s8 wei
//                                  + bf16 dst).  Adds a pre-OMP
//                                  per-expert Op1 src hoist (bf16→s8
//                                  via `reorder_quantization_wrapper`,
//                                  mirroring the N-tile hoist) and a
//                                  per-thread Stage 2b re-quant of
//                                  the post-activation tile (bf16→s8
//                                  via `dynamic_per_token_quant_bf16_
//                                  s8_native`).  All three buffer
//                                  classes (Op1 hoisted s8 + scale,
//                                  Op2 per-thread compact bf16 + s8 +
//                                  scale) are RAII-owned at the
//                                  dispatcher scope so the
//                                  `~reorder_quant_buffers_t`
//                                  destructor frees them on executor
//                                  return — zero per-slice mallocs.
// All three regimes share this same executor — the per-stage work
// differs only in (a) which AOCL DLP kernel routes inside
// `execute_expert_slice` for Stages 1 and 3, and (b) whether Stage 2b
// runs (DQ-INT8 only).  The per-thread slice plan, scratch sizing
// math for the bf16 intermediate, and stage layout (W13 → gated-act
// → W2) are identical because the staging tile element size is bf16
// (2 B) in all three regimes (WOQ dequantizes weights inside the
// kernel; DQ-INT8 re-quantizes the bf16 post-act tile to s8 between
// Stages 2 and 3 — the s8 buffer is OUT-of-band in a separate
// per-thread RAII scratch, not in the bf16 staging tile).
//
// ═══════════════════════════════════════════════════════════════════════
//
// ── C.1  try_fused_m_tile_core (BF16 / WOQ-INT4 / DQ-INT8) ─────────
//
// Single OMP region that fuses (W13 matmul → gated activation → W2
// matmul) over row-disjoint M-tile slices.  Each thread runs all three
// stages on its own slice rows using a thread-local scratch buffer; no
// global barrier between stages, and the intermediate `(slice_M × 2·I)`
// bf16 buffer never round-trips through DRAM.  Engaged from
// `group_matmul_fused_moe_execute` when the eligibility gate passes
// AND `ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION ∈ {0 AUTO, 1 FORCED}`.
//
// Name note — the `_bf16` suffix reflects the STAGING / ACTIVATION
// dtype (the per-thread intermediate tile is bf16 in ALL THREE
// regimes the executor accepts), NOT the weight dtype.  WOQ-INT4
// callers route through this same executor; the weight dequantization
// is internal to the AOCL DLP kernel.  DQ-INT8 callers also route
// through this same executor; the s8 buffers used at Stages 1 and 3
// live OUT-of-band in pre-OMP RAII scratches, distinct from the bf16
// staging tile.  Name kept for source-compat with prior commits and
// the dispatcher fork-point in `group_matmul_fused_moe.cpp`.
//
// Return contract:
//   * `true`  — pipeline engaged AND completed successfully.  Caller
//               MUST skip the legacy two-pass; dst_w2 (and optionally
//               dst_w13 in spill mode) holds the final result.
//   * `false` — pipeline DID NOT engage (any of: empty inputs,
//               round-based regime, multi-tier would have engaged,
//               shallow-M fallback signaled, scratch budget insufficient,
//               per-thread allocation failed, DQ-INT8 pre-OMP hoist
//               failed).  Caller falls back to the legacy two-pass.
//               IMPORTANT correctness contract: the `false`-path
//               guarantees NO partial writes have been made to dst_w2
//               / dst_w13 — every alloc failure is caught BEFORE any
//               thread proceeds past the OMP barrier that separates
//               scratch acquisition from actual work, and the
//               DQ-INT8 pre-OMP hoist runs single-threaded BEFORE
//               the OMP region opens.
//
// REGIME SCOPE — the executor's caller-side eligibility wrapper
// (`try_fused_m_tile`) must verify that BOTH halves
// agree on ONE of these regimes:
//
//   (A) BF16 end-to-end — `dtypes.{src,wei,dst} = bf16` and
//       `dynamic_quant == false`.
//
//   (B) WOQ-INT4 — `dtypes.{src,dst} = bf16`, `dtypes.wei ∈
//       {s4, u4}`, `dynamic_quant == false`,
//       `quant_params.wei_scale.buff != nullptr`, and
//       `is_weights_const[*] == true` for every active expert.
//       Activation stays bf16 across all three stages — the WOQ
//       kernel dequantizes weights internally and emits bf16 tiles
//       into the same per-thread scratch that BF16 callers use.
//
//   (C) DQ-INT8 — `dtypes.src = bf16`, `dtypes.wei = s8`,
//       `dtypes.dst = bf16`, `dtypes.compute = s8`,
//       `dynamic_quant == true`, per-token symmetric src quant
//       (`quant_params.src_scale.dims == {M[i], 1}`,
//       `src_scale.dt = f32`).  Symmetric only for v1 (no
//       `src_zp`); u8/asymmetric is a future extension.  The
//       executor adds two pre-OMP allocations on the dispatcher
//       stack, both RAII-owned via `std::vector<
//       reorder_quant_buffers_t>`:
//
//         (i) Per-expert Op1 src hoist (`hoist_buffers_w13`).
//             One `reorder_quantization_wrapper` call per active
//             expert quantizes the bf16 src to s8 + per-row scale
//             ONCE before the OMP region opens; per-thread slices
//             of that expert then read from the shared s8 buffer
//             (read-only) instead of re-running the reorder
//             per-thread.  Mirrors the N-tile hoist at
//             `group_matmul_n_tile.cpp:2933-3019`.
//
//         (ii) Per-thread Stage 2b re-quant scratch
//              (`per_thread_2b`).  One slot per thread, sized to
//              that thread's max slice's `slice_M × K_w2 × 3 +
//              slice_M × 4` bytes (compact bf16 + s8 + scales).
//              Allocated up front so the inner per-slice loop
//              never enters an allocator.
//
//       Both vectors live on the executor's stack; on return,
//       every `~reorder_quant_buffers_t` frees its `src_buf /
//       scale_buf / zp_buf` deterministically.  No per-slice
//       mallocs and no `thread_local` persistence (DQ-INT8 does
//       NOT extend the `thread_local AlignedTileScratch` that holds
//       the bf16 intermediate; it allocates fresh-per-call RAII
//       buffers instead).
//
// The executor itself does NOT re-check dtypes — the wrapper above
// guarantees regime uniformity across both halves and across active
// experts within each half.  The scratch sizing uses `inter_elem =
// size_of(params_w13[0].dtypes.dst)` which is bf16 (= 2 bytes) in
// ALL THREE regimes.  A regime-A/regime-B/regime-C mix across the
// two halves is rejected by the wrapper above — the unified scratch
// sizing only fits when both halves agree on bf16 staging dtype +
// the same Stage 2b plumbing.
//
// Memory contract (relevant to "what gets freed"):
//
//   * BF16 staging (`sc`): function-local `static thread_local
//     AlignedTileScratch` that owns its buffer via aligned malloc and
//     grows monotonically to each OMP worker's high-water slice
//     footprint.  Freed ONLY on the owning OMP worker thread's
//     exit; NOT released by `clear_fused_moe_scratch()` (that API
//     sweeps the file-scope fused-MoE TLS surfaces it can reach,
//     not function-local statics inside this executor — see the
//     LIMITATION note on the public `clear_fused_moe_scratch()`
//     doc-block in `group_matmul_direct.hpp`).  Unchanged across
//     all three regimes.
//
//   * WOQ pre-op metadata: stack-local per `execute_expert_slice`
//     call; caller-owned wei_scale / wei_zp buffers.  No new
//     allocation.
//
//   * DQ-INT8 pre-OMP scratches: stack-scoped `std::vector
//     <reorder_quant_buffers_t>` on this function's stack; freed
//     at function return via the vectors' destructors.  Zero
//     per-slice mallocs.
//
// PER-STAGE WORK (only the DQ-INT8-specific bits noted; BF16 / WOQ
// are unchanged):
//
//   Stage 1 (W13 matmul):  DQ-INT8 overlays `w13_local.dtypes.src
//                          = s8`, `quant_params.src_scale = h.src_
//                          scale` (slice-local row offset already
//                          applied by `offset_quant_by_row`).  The
//                          per-thread reorder wrapper inside
//                          `execute_expert_slice` sees `dtypes.src
//                          == s8` and short-circuits; the AOCL DLP
//                          s8s8→bf16 kernel produces the bf16
//                          staging tile.
//   Stage 2 (gated act):   bf16 in-place 2I → I on the staging tile
//                          (unchanged — same `apply_gated_act_
//                          inplace`).
//   Stage 2b (DQ-INT8):    `dqint8_compact_and_requant_slice`
//                          compacts the row-strided post-act tile
//                          to a contiguous `[slice_M × K_w2]` bf16
//                          buffer, then runs `dynamic_per_token_
//                          quant_bf16_s8_native` → s8 + per-row
//                          scale.  All three output buffers are
//                          pre-allocated per-thread RAII slots.
//   Stage 3 (W2 matmul):   DQ-INT8 overlays `w2_local.dtypes.src
//                          = s8`, `src_scale.{buff,dt,dims} =
//                          (scale_buf, f32, {slice_M, 1})`.  Pass
//                          the s8 buffer as src with `lda = K_w2[e]`
//                          (compacted to tight stride in Stage 2b).
//
// `bool dst_w13_is_caller_alloc`:  when true, after Stage 2 each thread
// spills its post-activation `(slice_M × I)` bf16 tile from scratch to
// the caller's dst_w13 buffer (stride `ldc_w13[e]`).  When false (the
// `use_internal_alloc=1` mode), the spill is skipped — Op1's library
// arena is never read after the pipeline returns, so writing it is
// pure overhead.  This is the "auto-pick" semantics requested by the
// vertical-fusion design.
bool try_fused_m_tile_core(const std::vector<char> &layout,
        const std::vector<bool> &transA, // Op1 src transposition
        const std::vector<bool> &transA_w2, // Op2 src transposition (must
        // be all-`false` — Op2 reads
        // the row-major scratch tile)
        const std::vector<bool> &transB, // shared by Op1 / Op2
        // Op1 (W13) inputs
        const std::vector<int> &M, const std::vector<int> &N_w13,
        const std::vector<int> &K_in, const std::vector<float> &alpha_w13,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight_w13,
        const std::vector<int> &ldb_w13,
        const std::vector<const void *> &bias_w13,
        const std::vector<float> &beta_w13, const std::vector<void *> &dst_w13,
        const std::vector<int> &ldc_w13, bool dst_w13_is_caller_alloc,
        // Op2 (W2) inputs
        const std::vector<int> &N_w2, const std::vector<int> &K_w2,
        const std::vector<float> &alpha_w2,
        const std::vector<const void *> &weight_w2,
        const std::vector<int> &ldb_w2,
        const std::vector<const void *> &bias_w2,
        const std::vector<float> &beta_w2, const std::vector<void *> &dst_w2,
        const std::vector<int> &ldc_w2,
        // shared
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params_w13,
        std::vector<matmul_params> &params_w2, int num_threads) {

    const int num_ops = static_cast<int>(M.size());
    if (num_ops == 0 || num_threads <= 0) return false;

    // ── Regime detection (tri-state: BF16 / WOQ / DQ-INT8) ──
    //
    // Probed ONCE from `params_w13[0]` because the eligibility wrapper
    // (`try_fused_m_tile`) guarantees both halves agree on
    // the same regime AND cross-expert dtype uniformity within each
    // half (via `m_tile_supported`).  Used by the budget gate, the
    // tag store, the pre-OMP hoist branch, and the per-stage overlays
    // inside the OMP region.
    enum RegimeTag { kRegBF16 = 0, kRegWOQ = 1, kRegDQINT8 = 2 };
    const RegimeTag regime = [&]() -> RegimeTag {
        const auto wei0 = params_w13[0].dtypes.wei;
        if (wei0 == data_type_t::s4 || wei0 == data_type_t::u4) return kRegWOQ;
        if (wei0 == data_type_t::s8) return kRegDQINT8;
        return kRegBF16;
    }();

    // ── Generic prepack for BOTH Op1 and Op2 weights ──
    // ALGO 2 prepack is idempotent (cached per-weight pointer in the
    // pack cache) and short-circuits when there is nothing to do.  We
    // call it here for both passes BEFORE the OMP region so the inner
    // BRGEMM / oneDNN kernels see warm packed weights from the very
    // first slice; legacy two-pass already does this implicitly by
    // calling `m_tile_execute` twice, once per pass.
    group_matmul_prepack::prepack_m_tile(
            group_matmul_prepack::build_prepack_params(weight_w13, K_in, N_w13,
                    ldb_w13, transB, is_weights_const, params_w13, M,
                    group_matmul_custom_kernel_candidate(
                            params_w13, M, N_w13, fused_act),
                    num_threads,
                    n_tile_decode_nr_align(
                            M, N_w13, ldc_w13, fused_act, params_w13),
                    fused_act, act_dtype,
                    /*transA=*/&transA, /*alpha=*/&alpha_w13,
                    /*beta=*/&beta_w13));
    group_matmul_prepack::prepack_m_tile(
            group_matmul_prepack::build_prepack_params(weight_w2, K_w2, N_w2,
                    ldb_w2, transB, is_weights_const, params_w2, M,
                    group_matmul_custom_kernel_candidate(
                            params_w2, M, N_w2, grp_matmul_gated_act_t::none),
                    num_threads, /*nr_align=*/0,
                    /*fused_act=*/grp_matmul_gated_act_t::none,
                    /*act_dtype=*/data_type_t::none,
                    /*transA=*/&transA_w2, /*alpha=*/&alpha_w2,
                    /*beta=*/&beta_w2));

    matmul_algo_t algo = resolve_matmul_backend();

    // Element sizes.  `inter_elem` is the scratch element size — by
    // contract (see PHASE 1 SCOPE above) this is `size_of(bf16) = 2`.
    // `src_elem` is Op1's src dtype (also bf16 in phase 1) and
    // `dst_elem_w2` is Op2's dst dtype.
    const size_t src_elem = size_of(params_w13[0].dtypes.src);
    const size_t inter_elem = size_of(params_w13[0].dtypes.dst);
    const size_t dst_elem_w2 = size_of(params_w2[0].dtypes.dst);

    scoped_active_levels guard(1);

    const int kSliceTarget = get_grp_matmul_m_tile_slice_target();

    // ── Phase 1: count active experts + early bail ──
    int active_ops = 0;
    for (int i = 0; i < num_ops; ++i)
        if (M[i] > 0) ++active_ops;
    if (active_ops == 0) return true; // claim success: nothing to do

    // Round-based regime is the `active_ops > num_threads` branch in
    // `m_tile_execute` — every thread runs at most one whole expert with
    // `team=1`.  Vertical fusion at M-tile granularity has no slicing
    // win there (each thread already owns its expert end-to-end) and
    // would just double the per-expert scratch footprint.  Bail to
    // legacy two-pass; this regime is rare on production MoE shapes
    // (≥ num_threads active experts simultaneously).
    if (active_ops > num_threads) return false;

    std::vector<int> active_pos(num_ops, -1);
    {
        int next_pos = 0;
        for (int i = 0; i < num_ops; ++i) {
            if (M[i] > 0) active_pos[i] = next_pos++;
        }
    }

    // ── CCD topology (mirrors `m_tile_execute`) ──
    const int cores_per_ccd = std::min(8, num_threads);
    const int num_ccds
            = std::max(1, (num_threads + cores_per_ccd - 1) / cores_per_ccd);

    // ── Multi-tier hybrid prediction (mirrors `m_tile_execute`) ──
    //
    // Vertical fusion phase 1 only covers the single-tier mapping.
    // Multi-tier hybrid (engaged on skewed-many-expert Qwen3-class
    // prompt frames) splits the thread budget into a light pool +
    // heavy pool with different per-pool slice strategies; fusing
    // W13→act→W2 across that split requires a separate executor design
    // that's deferred to phase 2+.  If the legacy gates would have
    // engaged multi-tier on the Op1 shape, bail to legacy two-pass.
    //
    // The prediction is a literal copy of the gating logic in
    // `m_tile_execute` (search "Multi-tier hybrid" above) so the decision
    // stays in sync — any future tuning of the multi-tier gates will
    // re-route those workloads back through the legacy path
    // automatically.  Heavy-underfill perf guard (`ht_assigned_final *
    // 4 >= heavy_pool * 3`) is NOT replayed here: when it fails the
    // legacy path falls through to single-tier itself, which the
    // pipeline executor can also handle.
    if (get_grp_matmul_m_tile_hybrid() == 0) {
        int max_M = 0;
        int64_t sum_M_total = 0;
        for (int i = 0; i < num_ops; ++i) {
            if (M[i] > 0) {
                sum_M_total += M[i];
                if (M[i] > max_M) max_M = M[i];
            }
        }
        const int avg_M = static_cast<int>(sum_M_total / active_ops);
        const int kHybridMinMaxM = get_grp_matmul_m_tile_hybrid_min_max_m();
        const int kHybridMinSkewX = get_grp_matmul_m_tile_hybrid_min_skew();
        const int kLightsPerThread
                = get_grp_matmul_m_tile_hybrid_lights_per_thread();
        const int min_actives = num_threads / 2;
        const int min_lights = num_threads / 8;
        const bool gate_skew = (max_M >= kHybridMinMaxM) && (avg_M > 0)
                && (static_cast<int64_t>(max_M)
                        >= static_cast<int64_t>(kHybridMinSkewX) * avg_M);
        if (active_ops >= min_actives && gate_skew) {
            const int light_cut = std::max(8, avg_M / 4);
            int n_light = 0, n_heavy = 0;
            int64_t heavy_M_sum = 0;
            for (int i = 0; i < num_ops; ++i) {
                if (M[i] <= 0) continue;
                if (M[i] <= light_cut)
                    ++n_light;
                else {
                    ++n_heavy;
                    heavy_M_sum += M[i];
                }
            }
            const int candidate_light_pool = std::min(cores_per_ccd,
                    static_cast<int>(std::max<int64_t>(1,
                            (static_cast<int64_t>(n_light) + kLightsPerThread
                                    - 1)
                                    / kLightsPerThread)));
            const int candidate_heavy_pool = num_threads - candidate_light_pool;
            if (n_light > 0 && n_light >= min_lights && n_heavy > 0
                    && heavy_M_sum > 0 && candidate_light_pool < num_threads
                    && candidate_heavy_pool >= n_heavy) {
                // Multi-tier would have engaged in `m_tile_execute`.  Bail.
                return false;
            }
        }
    }

    // ── Single-tier Phase-1b/2/3 plan (shared with `m_tile_execute`) ──
    // Pass short_circuit_on_shallow_m=true: this executor bails to the legacy
    // two-pass on the shallow-M signal (just below) and never reads the tid
    // mapping, so let the planner skip the throwaway Phase-2/3 work.
    const auto plan = make_m_tile_single_tier_plan(M, active_pos, num_ops,
            num_threads, active_ops, kSliceTarget, cores_per_ccd, num_ccds,
            /*short_circuit_on_shallow_m=*/true);

    // Shallow-M fallback bails out: full-team-per-expert sequential GEMM
    // doesn't fuse vertically at M-tile granularity (no row-disjoint
    // slice plan inside a single expert).  Legacy two-pass handles it.
    if (plan.shallow_m_fallback) return false;

    // ── Per-thread scratch sizing + budget gate ──
    //
    // Scratch budget caps `slice_M` per expert so each thread's
    // `(slice_M × N_w13) × inter_elem` staging buffer fits in L2.
    // Default 512 KB fits within a 1 MB per-core L2 with headroom
    // for W13 + W2 weight blocks the inner kernel co-loads.
    //
    // We compute the EXACT byte count per thread up front (slice_M is
    // known once the planner returns).  If any thread's slice exceeds
    // the budget, eligibility fails — even forcing slice_M=1 still
    // exceeds budget on pathological shapes (e.g. a large intermediate
    // dim I=14336 needs `2 * 14336 * 2 = 56 KB` per row, well under any
    // sensible budget — so the practical bail-out trigger is when the
    // caller sets the budget unreasonably low).  Bail returns false
    // BEFORE any OMP work, so dst is untouched and the legacy two-pass
    // overwrite is safe.
    //
    // UNBOUNDED budget (`kMTilePipelineScratchKbUnbounded`, env/override
    // value -1): the gate is disabled by treating the budget as the
    // full size_t range, so `bytes > scratch_budget_bytes` is never
    // true and vertical fusion engages regardless of slice size.  This
    // is the "always run vertical fusion" escape hatch — see the
    // knob's doc-block in group_matmul_m_tile.hpp for the L2-spill
    // perf caveat.
    const int scratch_budget_kb = get_grp_matmul_m_tile_pipeline_scratch_kb();
    const size_t scratch_budget_bytes
            = (scratch_budget_kb == kMTilePipelineScratchKbUnbounded)
            ? std::numeric_limits<size_t>::max()
            : static_cast<size_t>(scratch_budget_kb) * 1024u;

    std::vector<size_t> per_thread_bytes(num_threads, 0);
    // DQ-INT8 Stage 2b: per-thread sizing for the (compact bf16 + s8 +
    // f32-scale) RAII slot allocated below.  Sized exactly to each
    // thread's max slice (single slice per thread per call in this
    // executor — no slice growth loop), so the allocation can be done
    // once pre-OMP and reused across the per-stage work inside the
    // OMP region.  All three vectors stay zero-sized when regime is
    // BF16 or WOQ; the inner OMP body checks `regime == kRegDQINT8`
    // before touching them.
    std::vector<size_t> per_thread_2b_compact_bytes(num_threads, 0);
    std::vector<size_t> per_thread_2b_int8_bytes(num_threads, 0);
    std::vector<size_t> per_thread_2b_scale_bytes(num_threads, 0);
    for (int tid = 0; tid < num_threads; ++tid) {
        const int e = plan.tid_to_expert[tid];
        if (e < 0) continue;
        const int local = plan.tid_to_local[tid];
        const int team = plan.tid_to_team[tid];
        if (team <= 0) continue;
        const int row_start
                = static_cast<int>(static_cast<int64_t>(M[e]) * local / team);
        const int row_end = static_cast<int>(
                static_cast<int64_t>(M[e]) * (local + 1) / team);
        const int slice_M = row_end - row_start;
        if (slice_M <= 0) continue;
        size_t bytes = static_cast<size_t>(slice_M)
                * static_cast<size_t>(N_w13[e]) * inter_elem;
        if (regime == kRegDQINT8) {
            // Stage 2b out-of-band scratch.  `K_w2[e]` is the post-act
            // active-col count — = N_w13[e]/2 for gated activations
            // (caller-enforced; see fused-MoE dispatcher), = N_w13[e]
            // when `fused_act == none`.
            const size_t k_w2_e = static_cast<size_t>(K_w2[e]);
            // Op2 source-scale group count: per-token (1) or per-group (G>1).
            // The Stage-2b re-quant writes {slice_M × G} f32 scales, so size the
            // scratch slot for G (G==1 collapses to the per-row per-token case).
            const int g2_e = mtile_src_group_count(
                    params_w2[e].quant_params.src_scale);
            const size_t compact_b
                    = static_cast<size_t>(slice_M) * k_w2_e * 2u; // bf16
            const size_t int8_b
                    = static_cast<size_t>(slice_M) * k_w2_e * 1u; // s8
            const size_t scale_b = static_cast<size_t>(slice_M)
                    * static_cast<size_t>(g2_e)
                    * sizeof(float); // {slice_M × G} f32
            per_thread_2b_compact_bytes[tid] = compact_b;
            per_thread_2b_int8_bytes[tid] = int8_b;
            per_thread_2b_scale_bytes[tid] = scale_b;
            bytes += compact_b + int8_b + scale_b;
        }
        if (bytes > scratch_budget_bytes) {
            // One thread's slice doesn't fit the budget — bail out.
            return false;
        }
        per_thread_bytes[tid] = static_cast<size_t>(slice_M)
                * static_cast<size_t>(N_w13[e]) * inter_elem;
    }

    // ── Capture-gated branch tag (test hook) ──
    //
    // Tag is stored AT the gate-pass point — before the per-thread alloc
    // attempts inside the OMP region.  On the rare path where one
    // thread's `grow_scratch` fails mid-OMP, every thread observes
    // `alloc_fail = true` past the barrier, skips its dst-writing block
    // (no partial writes — see the contract in the OMP body), and the
    // function returns `false`.  The dispatcher in
    // `group_matmul_fused_moe.cpp` then falls back to the legacy
    // two-pass, which itself routes through `m_tile_execute` and OVERWRITES
    // this tag with one of {kRoundBased, kMultiTier, kShallowMFallback,
    // kPhase2Single}.
    //
    // Therefore the transient `kVerticalFusion* + return=false` state
    // is NOT externally observable by callers / tests — the final tag
    // visible after the public-API call is always the executor that
    // actually produced the result.  This self-correcting contract is
    // why we store the tag here (cheapest position — single store outside
    // any hot loop) rather than at end-of-function after success.
    //
    // Same self-correcting guarantee covers the new DQ-INT8 pre-OMP
    // hoist below (between the tag store and the OMP region): if the
    // hoist fails for any expert we return `false`, the dispatcher
    // falls back to legacy two-pass, and the tag is overwritten by
    // the legacy executor's branch tag.
    //
    // Dtype-specific tag selection — the eligibility wrapper
    // `try_fused_m_tile` guarantees both halves share the
    // same regime, so the locally-cached `regime` enum above is
    // sufficient to discriminate; we do NOT need to re-probe params
    // here.  The three tags differ only in (a) which AOCL DLP kernel
    // runs at each stage and (b) whether Stage 2b runs — the
    // per-thread slice plan, OMP region, and bf16 scratch sizing are
    // identical.
    if (test_api::s_capture_m_tile_path.load(std::memory_order_relaxed)) {
        const int tag = (regime == kRegDQINT8)
                ? test_api::m_tile_path_tag::kVerticalFusionDQINT8
                : (regime == kRegWOQ)
                ? test_api::m_tile_path_tag::kVerticalFusionWOQ
                : test_api::m_tile_path_tag::kVerticalFusionBF16;
        test_api::s_last_m_tile_path.store(tag, std::memory_order_relaxed);
    }

    // ── DQ-INT8 pre-OMP per-expert Op1 src hoist + per-thread Stage 2b ──
    //
    // ONLY runs in the `kRegDQINT8` regime; both vectors stay default-
    // constructed (empty) and are no-ops at destruction in the other
    // regimes.  See the doc-block on `try_fused_m_tile_core` above
    // (REGIME SCOPE / Memory contract) for the lifetime argument.
    //
    // `hoist_buffers_w13[e].src_buf` owns the per-expert s8 quantized
    // source via `~reorder_quant_buffers_t`; `hoisted_w13[e]` is a
    // read-only view substituted into `w13_local` inside Stage 1.
    //
    // `per_thread_2b[tid]` holds the three Stage 2b slots (compact
    // bf16 in `src_buf`, s8 in `scale_buf` — name carries no semantic
    // meaning here, just a free uint8_t slot — and per-row f32 scale
    // in `zp_buf`).  Re-using the struct's three pointer slots avoids
    // adding a fourth sibling type for what is structurally the same
    // RAII contract (three malloc'd buffers, deterministic free at
    // scope exit).
    std::vector<reorder_quant_buffers_t> hoist_buffers_w13;
    std::vector<HoistedSrcQuant_mtile> hoisted_w13;
    std::vector<reorder_quant_buffers_t> per_thread_2b;
    if (regime == kRegDQINT8) {
        hoist_buffers_w13.resize(num_ops);
        hoisted_w13.resize(num_ops);
        for (int e = 0; e < num_ops; ++e) {
            if (M[e] <= 0 || !params_w13[e].dynamic_quant) continue;

            // Shadow the caller's params_w13[e] — the wrapper mutates its
            // matmul_params& argument (dtypes.src, src_scale.buff, etc.).
            // The caller's vector is shared with the legacy fallback path,
            // so we MUST NOT touch it; per-thread overlay below copies
            // from caller params_w13[e] and then layers the hoisted state
            // on top, matching the N-tile execute_tile pattern at
            // `group_matmul_n_tile.cpp:443-454`.
            matmul_params shadow = params_w13[e];
            const void *src_e = src[e];
            int reordered_lda = lda[e];
            size_t src_type_size = size_of(shadow.dtypes.src);
            matmul_batch_params_t bp;
            bp.Batch_A = 1;
            bp.Batch_B = 1;

            const status_t s = reorder_quantization_wrapper(src_e, lda[e],
                    reordered_lda, src_type_size, shadow, bp, transA[e], M[e],
                    K_in[e], num_threads, hoist_buffers_w13[e]);

            if (s != status_t::success) {
                // Validation failure (bad src_scale dims / dt, etc.) — same
                // failure mode as the N-tile hoist; bail to legacy two-pass.
                // Hoist buffers allocated so far are freed by the vector
                // destructors when this function returns.  dst_w13 and
                // dst_w2 have NOT been touched yet (OMP region is below),
                // so the legacy fallback can overwrite them cleanly.
                log_error(
                        "try_fused_m_tile_core (DQ-INT8): pre-OMP src "
                        "hoist failed for expert ",
                        e, "; falling back to legacy two-pass.");
                return false;
            }

            // Wrapper succeeded but may have short-circuited (dtype combo
            // outside its `eligible` filter — shouldn't happen given the
            // gate, but defended for parity with the N-tile pattern).
            // Detect via the unchanged src dtype: when no reorder
            // happened, `shadow.dtypes.src == params_w13[e].dtypes.src`
            // and we leave the slot `valid = false`; the per-thread
            // overlay below then falls through to the caller's bf16 src.
            // Practically this means the kernel runs with bf16 src + s8
            // wei, which the AOCL DLP path handles via its internal
            // wrapper (correct but defeats the hoist's perf win — log
            // INFO so production deployments can spot the configuration
            // mismatch).
            if (shadow.dtypes.src != params_w13[e].dtypes.src) {
                hoisted_w13[e].valid = true;
                hoisted_w13[e].src_ptr = src_e;
                hoisted_w13[e].lda = reordered_lda;
                hoisted_w13[e].src_dtype = shadow.dtypes.src;
                hoisted_w13[e].src_scale = shadow.quant_params.src_scale;
                hoisted_w13[e].src_zp = shadow.quant_params.src_zp;
            }
        }

        // Per-thread Stage 2b allocation.  Single-threaded pre-OMP so
        // no allocator contention; sized once from the planner output
        // above.  On any malloc failure we bail to legacy; vectors
        // allocated so far are freed at function return.
        per_thread_2b.resize(num_threads);
        for (int tid = 0; tid < num_threads; ++tid) {
            const size_t cb = per_thread_2b_compact_bytes[tid];
            const size_t ib = per_thread_2b_int8_bytes[tid];
            const size_t sb = per_thread_2b_scale_bytes[tid];
            if (cb == 0 && ib == 0 && sb == 0) continue; // idle thread
            // `.src_buf` carries the compact bf16 tile; `.scale_buf` the
            // s8 quantized tile; `.zp_buf` the per-row f32 scales.  See
            // the comment on the `per_thread_2b` declaration above for
            // the naming rationale (three malloc slots, no semantic
            // tie-in to the field names beyond ordering).
            auto &pt = per_thread_2b[tid];
            pt.src_buf = static_cast<uint8_t *>(std::malloc(cb));
            pt.scale_buf = static_cast<uint8_t *>(std::malloc(ib));
            pt.zp_buf = static_cast<uint8_t *>(std::malloc(sb));
            if ((cb > 0 && !pt.src_buf) || (ib > 0 && !pt.scale_buf)
                    || (sb > 0 && !pt.zp_buf)) {
                log_error(
                        "try_fused_m_tile_core (DQ-INT8): per-thread "
                        "Stage 2b alloc failed for tid ",
                        tid, " (compact=", cb, " int8=", ib, " scale=", sb,
                        "); falling back to legacy two-pass.");
                return false;
            }
        }
    }

    // ── OMP region: per-thread W13 → activation → W2 pipeline ──
    //
    // Correctness contract: every thread allocates its scratch first,
    // then a barrier synchronises so all threads either proceed with
    // work (alloc OK) or all skip (any thread's alloc failed).  This
    // ensures the `false`-return path from this function NEVER leaves
    // dst with partial writes.
    std::atomic<bool> alloc_fail {false};

#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const int e = plan.tid_to_expert[tid];
        const size_t need = per_thread_bytes[tid];
        static thread_local AlignedTileScratch sc;
        if (e >= 0 && need > 0) {
            if (!grow_scratch(sc, need)) {
                alloc_fail.store(true, std::memory_order_relaxed);
            }
        }

// Memory-ordering barrier: every thread's alloc attempt finishes
// before any thread enters the work block below.  Without this
// barrier a thread that succeeded in alloc could race ahead and
// start writing dst_w2 before a later-failing thread sets
// alloc_fail, breaking the "no partial writes on false" contract.
#pragma omp barrier

        if (!alloc_fail.load(std::memory_order_relaxed) && e >= 0 && need > 0) {
            const int local = plan.tid_to_local[tid];
            const int team = plan.tid_to_team[tid];
            const int row_start = static_cast<int>(
                    static_cast<int64_t>(M[e]) * local / team);
            const int row_end = static_cast<int>(
                    static_cast<int64_t>(M[e]) * (local + 1) / team);
            const int slice_M = row_end - row_start;

            // ── STAGE 1: W13 matmul into per-thread scratch ──
            //
            // Slice the per-token / per-group src quant params and any
            // row-varying binary postops, exactly as `execute_m_tile`
            // does for the legacy Op1 pass.  The `static thread_local`
            // matmul_params copy is reused across calls — declared in
            // the OMP region so each worker has its own instance.
            static thread_local matmul_params w13_local;
            w13_local = params_w13[e];
            w13_local.num_threads = 1;

            // Row-offset binary post-op buffers (mirrors execute_m_tile).
            // 1D broadcast and {1,N} row-broadcast are unchanged; 2D/3D
            // row-varying tensors offset by row_start × effective_ld.
            for (auto &po : w13_local.postop_) {
                if ((po.po_type == post_op_type_t::binary_add
                            || po.po_type == post_op_type_t::binary_mul)
                        && po.buff != nullptr) {
                    const bool is_broadcast_1d = (po.dims.size() <= 1)
                            || (po.dims.size() == 2 && po.dims[0] == 1);
                    if (!is_broadcast_1d) {
                        const int eff_ld = (po.leading_dim > 0) ? po.leading_dim
                                                                : N_w13[e];
                        const size_t po_elem = size_of(po.dtype);
                        po.buff = static_cast<uint8_t *>(po.buff)
                                + static_cast<size_t>(row_start) * eff_ld
                                        * po_elem;
                    }
                }
            }

            // ── DQ-INT8 Stage 1 overlay ──
            //
            // When regime is DQ-INT8 and the pre-OMP hoist produced a
            // valid s8 src for this expert, substitute the shared s8
            // buffer + its full-M src_scale / src_zp dims into
            // `w13_local`.  Mirrors the N-tile `execute_tile()` substitution
            // at `group_matmul_n_tile.cpp:443-454` so the per-thread
            // reorder wrapper inside `execute_expert_slice` sees
            // `dtypes.src == s8` and short-circuits (avoids racing on
            // the shared scale buffer and the `num_threads ×
            // O(M × K)` duplicated reorder work that would happen if
            // every M-tile thread re-ran the wrapper).
            //
            // The substitution lands BEFORE the per-slice
            // `offset_quant_by_row` calls below so the slice-local
            // dims[0] rewrite is applied to the hoisted (full-M) dims,
            // not the caller's (which has been overwritten by the
            // wrapper in the shadow copy but NOT in `params_w13[e]`
            // itself; we deliberately avoided mutating the caller's
            // params so the legacy fallback path stays clean).
            const void *src_for_w13 = src[e];
            int lda_for_w13 = lda[e];
            size_t src_elem_for_w13 = src_elem;
            if (regime == kRegDQINT8 && hoisted_w13[e].valid) {
                const auto &h = hoisted_w13[e];
                src_for_w13 = h.src_ptr;
                lda_for_w13 = h.lda;
                src_elem_for_w13 = size_of(h.src_dtype); // = 1 (s8)
                w13_local.dtypes.src = h.src_dtype;
                w13_local.quant_params.src_scale = h.src_scale;
                w13_local.quant_params.src_zp = h.src_zp;
            }

            // Per-token / per-group src quantization row offsets + dims[0]
            // rewrite (slice-local view).  Inert for BF16 / WOQ callers
            // because `params_w13[e].dynamic_quant == false` and
            // `quant_params.src_scale.dims` is empty there.  For DQ-INT8
            // the overlay above placed full-M dims into `w13_local`; this
            // rewrites the slice-local dims[0] view to `slice_M` so the
            // dynamic-quant reorder dispatcher's per-token gate matches
            // the sliced src_shape (see `offset_quant_by_row` doc-block).
            offset_quant_by_row(
                    w13_local.quant_params.src_scale, row_start, slice_M);
            offset_quant_by_row(
                    w13_local.quant_params.src_zp, row_start, slice_M);

            const size_t src_off = transA[e]
                    ? static_cast<size_t>(row_start) * src_elem_for_w13
                    : static_cast<size_t>(row_start) * lda_for_w13
                            * src_elem_for_w13;
            const void *src_slice
                    = static_cast<const char *>(src_for_w13) + src_off;

            execute_expert_slice(layout[e], transA[e], transB[e], slice_M,
                    N_w13[e], K_in[e], alpha_w13[e], src_slice, lda_for_w13,
                    weight_w13[e], ldb_w13[e], bias_w13[e], beta_w13[e], sc.buf,
                    /*ldc=*/N_w13[e], is_weights_const[e], /*num_thr=*/1,
                    w13_local, resolve_expert_backend(2, algo, w13_local));

            // ── STAGE 2: in-place gated activation on scratch (2I → I) ──
            if (fused_act != grp_matmul_gated_act_t::none) {
                apply_gated_act_inplace(fused_act, sc.buf,
                        /*row_start=*/0, /*row_end=*/slice_M,
                        /*N=*/N_w13[e], /*ldc=*/N_w13[e], act_dtype);
            }

            // ── (Optional) Op1 dst spill — caller-alloc mode only ──
            //
            // When the caller allocated dst_w13 (`dst_w13_is_caller_alloc`)
            // we copy the post-act `(slice_M × I)` tile from scratch to
            // the caller's dst_w13 buffer.  In internal-alloc mode the
            // caller never reads dst_w13 after this call returns, so the
            // copy is pure overhead — the "auto-pick" semantics from the
            // vertical-fusion design.
            //
            // Spill width depends on the activation:
            //   gated (silu/gelu/swiglu_oai)   →  I  cols  = N_w13[e] / 2
            //   none                            → N_w13[e] cols (full Op1 dst).
            if (dst_w13_is_caller_alloc && !dst_w13.empty()
                    && dst_w13[e] != nullptr) {
                const int spill_cols
                        = (fused_act != grp_matmul_gated_act_t::none)
                        ? (N_w13[e] / 2)
                        : N_w13[e];
                char *dst_w13_slice = static_cast<char *>(dst_w13[e])
                        + static_cast<size_t>(row_start) * ldc_w13[e]
                                * inter_elem;
                const char *sc_base = static_cast<const char *>(sc.buf);
                const size_t row_bytes
                        = static_cast<size_t>(spill_cols) * inter_elem;
                for (int m = 0; m < slice_M; ++m) {
                    std::memcpy(dst_w13_slice
                                    + static_cast<size_t>(m) * ldc_w13[e]
                                            * inter_elem,
                            sc_base
                                    + static_cast<size_t>(m) * N_w13[e]
                                            * inter_elem,
                            row_bytes);
                }
            }

            // ── STAGE 2b: DQ-INT8 only — bf16→s8 re-quant of post-act tile ──
            //
            // Compacts the row-strided post-activation tile (active cols
            // `[0, K_w2[e])` at stride `N_w13[e]`) into a contiguous
            // `[slice_M × K_w2[e]]` bf16 buffer, then runs per-token
            // symmetric `dynamic_per_token_quant_bf16_s8_native` → s8 +
            // per-row f32 scale.  Three output buffers (compact bf16, s8,
            // scale) come from the pre-allocated `per_thread_2b[tid]`
            // RAII slot — zero in-OMP mallocs.  See
            // `dqint8_compact_and_requant_slice` for the per-row memcpy
            // + AVX-512 quant kernel call.
            //
            // Cache locality: `sc.buf` is L1/L2-hot from the activation
            // that just wrote it.  The compact memcpy reads from `sc.buf`
            // (hot) and writes to `compact_bf16` (cold but local to this
            // thread); the s8 quant kernel reads from `compact_bf16`
            // (just-warm) and writes to `int8_buf` + `scale_f32` (cold,
            // local).  The W2 matmul below then reads `int8_buf` warm.
            //
            // Inert for BF16 / WOQ — the locals stay nullptr and Stage 3
            // routes through the bf16 `sc.buf` path unchanged.
            int8_t *int8_buf_for_w2 = nullptr;
            float *scale_f32_for_w2 = nullptr;
            if (regime == kRegDQINT8) {
                auto &pt = per_thread_2b[tid];
                uint8_t *compact_bf16_buf = pt.src_buf;
                int8_buf_for_w2 = reinterpret_cast<int8_t *>(pt.scale_buf);
                scale_f32_for_w2 = reinterpret_cast<float *>(pt.zp_buf);
                const int g2_w2 = mtile_src_group_count(
                        params_w2[e].quant_params.src_scale);
                // Match the GEMM's src-scale dtype to the Op2 weight-scale dtype so
                // AOCL sym-quant sees agreeing A/B scale types: f32 for per-token /
                // plain per-group, bf16 for GGML-reordered weights.  Anything else
                // (incl. `none`) defaults to f32 — the historical per-token behaviour.
                const data_type_t op2_scale_dt
                        = (params_w2[e].quant_params.wei_scale.dt
                                  == data_type_t::bf16)
                        ? data_type_t::bf16
                        : data_type_t::f32;
                dqint8_compact_and_requant_slice(sc.buf, slice_M, K_w2[e],
                        /*row_stride_bf16_elems=*/N_w13[e], compact_bf16_buf,
                        int8_buf_for_w2, scale_f32_for_w2,
                        /*src_groups=*/g2_w2, /*scale_dt=*/op2_scale_dt);
            }

            // ── STAGE 3: W2 matmul from scratch tile to dst_w2 slice ──
            //
            // BF16 / WOQ regimes: scratch is row-major `[slice_M, N_w13]`
            // with stride `N_w13[e]` between rows.  The gated activation
            // collapses the column count from `N_w13` to `I = N_w13/2`
            // IN PLACE but does NOT change the row stride — so the W2
            // kernel reads its `K_w2[e]` cols (= `I` for gated, =
            // `N_w13` for none) at `lda = N_w13[e]`.
            //
            // DQ-INT8 regime: src is the tight-strided s8 tile produced
            // by Stage 2b above.  Layout is contiguous `[slice_M ×
            // K_w2[e]]` with `lda = K_w2[e]` — the bf16 scratch's
            // pre-activation `N_w13[e]` stride does NOT apply here.
            //
            // Op2 always uses `transA_w2[e] = false` because we own the
            // scratch layout (the dispatcher enforces this at the call
            // site).  The same `transB[e]` / `layout[e]` as Op1 is reused
            // for the weight side.
            static thread_local matmul_params w2_local;
            w2_local = params_w2[e];
            w2_local.num_threads = 1;

            // Row-offset binary post-op buffers on Op2's params (no-op
            // for vanilla MoE since the dispatcher leaves Op2's postops
            // empty, but kept here so user-fed Op2 postops slice cleanly
            // — same loop as for Op1 above).
            for (auto &po : w2_local.postop_) {
                if ((po.po_type == post_op_type_t::binary_add
                            || po.po_type == post_op_type_t::binary_mul)
                        && po.buff != nullptr) {
                    const bool is_broadcast_1d = (po.dims.size() <= 1)
                            || (po.dims.size() == 2 && po.dims[0] == 1);
                    if (!is_broadcast_1d) {
                        const int eff_ld = (po.leading_dim > 0) ? po.leading_dim
                                                                : N_w2[e];
                        const size_t po_elem = size_of(po.dtype);
                        po.buff = static_cast<uint8_t *>(po.buff)
                                + static_cast<size_t>(row_start) * eff_ld
                                        * po_elem;
                    }
                }
            }
            // Per-token / per-group src quant offsets for Op2.  Inert in
            // BF16 / WOQ regimes (`dynamic_quant == false`).  For DQ-INT8
            // the caller-supplied `params_w2[e].quant_params.src_scale.dims`
            // are full-M `{M[i], 1}` (the user's Op2 src_scale is sized
            // to the full expert M); this rewrites the slice-local
            // dims[0] view to `slice_M` so the wrapper's per-token gate
            // matches what we hand it AFTER the overlay below replaces
            // the buffer pointer.  The overlay then OVERWRITES the
            // dims[0] view again with the slice-local `{slice_M, 1}`
            // tied to the Stage 2b scale buffer — both calls land on the
            // same slice_M, so the order is observationally invariant
            // (the second wins).  Kept as separate steps for diff
            // readability: this block exists for static/dynamic-quant
            // generality, the overlay below for the DQ-INT8 fast path.
            offset_quant_by_row(
                    w2_local.quant_params.src_scale, row_start, slice_M);
            offset_quant_by_row(
                    w2_local.quant_params.src_zp, row_start, slice_M);

            // ── DQ-INT8 Stage 3 overlay ──
            //
            // Route the W2 matmul through the s8 tile produced by Stage
            // 2b above.  Source is contiguous `[slice_M × K_w2[e]]` at
            // `lda = K_w2[e]` (tight stride — the compact step
            // intentionally drops the bf16 scratch's pre-activation
            // `N_w13[e]` stride).  Slice-local `src_scale = {slice_M, 1}`
            // tied to the per-thread `scale_f32_for_w2` buffer.
            // Symmetric (no `src_zp`).
            //
            // The per-thread `reorder_quantization_wrapper` inside
            // `execute_expert_slice` will see `dtypes.src == s8` and
            // short-circuit (same self-disabling contract as N-tile
            // execute_tile's hoisted-state substitution).  The kernel
            // dispatch then routes to the AOCL DLP s8s8 → bf16 path.
            const void *src_for_w2 = sc.buf;
            int lda_for_w2 = N_w13[e];
            if (regime == kRegDQINT8) {
                src_for_w2 = int8_buf_for_w2;
                lda_for_w2 = K_w2[e];
                w2_local.dtypes.src = data_type_t::s8;
                w2_local.quant_params.src_scale.buff = scale_f32_for_w2;
                // f32 for per-token / plain per-group, bf16 for GGML-reordered Op2 —
                // Stage 2b wrote the scratch in this dtype just above.
                w2_local.quant_params.src_scale.dt
                        = (params_w2[e].quant_params.wei_scale.dt
                                  == data_type_t::bf16)
                        ? data_type_t::bf16
                        : data_type_t::f32;
                // {slice_M, G}: per-token (G==1) or per-group (G>1, pairing with the
                // {G, N} down-weight scale).  run_dlp derives the AOCL sym-quant
                // group_size = K_w2 / G from this dims pair — no separate plumbing.
                w2_local.quant_params.src_scale.dims
                        = {static_cast<int64_t>(slice_M),
                                static_cast<int64_t>(mtile_src_group_count(
                                        params_w2[e].quant_params.src_scale))};
                // Symmetric DQ-INT8 — clear any inherited src_zp state so
                // the kernel's eligibility check doesn't misroute to an
                // asymmetric path.
                w2_local.quant_params.src_zp
                        = matmul_quantization_params_t::matmul_quant_t {};
            }

            char *dst_w2_slice = static_cast<char *>(dst_w2[e])
                    + static_cast<size_t>(row_start) * ldc_w2[e] * dst_elem_w2;

            execute_expert_slice(layout[e], transA_w2[e], transB[e], slice_M,
                    N_w2[e], K_w2[e], alpha_w2[e], src_for_w2, lda_for_w2,
                    weight_w2[e], ldb_w2[e], bias_w2[e], beta_w2[e],
                    dst_w2_slice, ldc_w2[e], is_weights_const[e], /*num_thr=*/1,
                    w2_local, resolve_expert_backend(2, algo, w2_local));
        }
    } // end #pragma omp parallel

    if (alloc_fail.load(std::memory_order_relaxed)) {
        log_error(
                "try_fused_m_tile_core: per-thread scratch alloc "
                "failed; falling back to legacy two-pass.");
        return false;
    }
    return true;
}

// ── C.3  flat_m_tile_pipeline_int8 (phase 2 — TODO) ────────────────────
//
// PHASE 2 (dynamic INT8 vertical fusion) lands as a separate executor
// next to `try_fused_m_tile_core` above.  Sketch:
//
//   * Same per-thread slice plan as Section C.1.
//   * Pre-OMP hoist (analogous to N-tile, see
//     `group_matmul_n_tile.cpp:2934-2983`): allocate per-thread
//     `std::vector<reorder_quant_buffers_t>` at dispatcher scope
//     (RAII; ~`~reorder_quant_buffers_t()` frees src/scale/zp at
//     vector destruction — no per-slice mallocs inside the OMP
//     region).
//   * Stage 1: `aocl_gemm_s8s8s32obf16_sym_quant` (s8 src + s8 wei
//              → bf16 scratch).
//   * Stage 2: `apply_gated_act_inplace` on bf16 scratch (unchanged).
//   * Stage 2b (NEW): per-row bf16 → s8 re-quant via
//                     `dynamic_per_token_quant_bf16_s8_native`,
//                     writing the per-row scale into the per-thread
//                     `reorder_quant_buffers_t.scale_buf` slot
//                     allocated up front.
//   * Stage 3: same `aocl_gemm_s8s8s32obf16_sym_quant` for W2.
//
// Eligibility gate (in `try_fused_m_tile` above) will
// be extended with a third regime for dynamic-int8 dtypes; the
// current gate explicitly rejects `dynamic_quant=true`.  No
// phase-2 code in this commit.

// ─────────────────────────────────────────────────────────────────────
// Section C.2 — Eligibility-gated wrapper around the pipeline executor.
// ─────────────────────────────────────────────────────────────────────
//
// `try_fused_m_tile` bundles the M-tile-pipeline-specific
// eligibility gate (env knob, dtype-regime check on both halves —
// BF16 end-to-end OR WOQ-INT4 OR DQ-INT8 per-token-symmetric,
// supported activation set, `m_tile_supported` on Op1 + synthesized
// Op2, WOQ-only `is_weights_const[*] == true` requirement, and
// DQ-INT8-only per-token-symmetric structural checks) with the
// actual `try_fused_m_tile_core` invocation.
//
// Lives next to the executor (rather than at the caller in
// `group_matmul_fused_moe.cpp`) so the engagement contract — which
// gates feed which planner inputs — stays at the same scroll position
// as the executor's bail-out contract.  Future tightening of the gate
// (asymmetric DQ-INT8 u8 support, additional activation set, etc.) is
// a one-file diff.
bool try_fused_m_tile(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transA_w2,
        const std::vector<bool> &transB, const std::vector<int> &M,
        const std::vector<int> &N_w13, const std::vector<int> &K_in,
        const std::vector<float> &alpha_w13,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight_w13,
        const std::vector<int> &ldb_w13,
        const std::vector<const void *> &bias_w13,
        const std::vector<float> &beta_w13, const std::vector<void *> &dst_w13,
        const std::vector<int> &ldc_w13, bool dst_w13_is_caller_alloc,
        const std::vector<int> &N_w2, const std::vector<int> &K_w2,
        const std::vector<float> &alpha_w2,
        const std::vector<const void *> &weight_w2,
        const std::vector<int> &ldb_w2,
        const std::vector<const void *> &bias_w2,
        const std::vector<float> &beta_w2, const std::vector<void *> &dst_w2,
        const std::vector<int> &ldc_w2, grp_matmul_gated_act_t fused_act,
        data_type_t act_dtype, const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params_w13,
        std::vector<matmul_params> &params_w2, int num_threads) {
    // Env knob short-circuit FIRST — cheapest gate; -1 (DISABLED)
    // bypasses every other check.  AUTO (0) and FORCED (1) both
    // continue to the data-shape gates below.
    if (get_grp_matmul_m_tile_vertical_fusion() == -1) return false;

    // Phase 1 eligibility — accept ONE of three dtype regimes per
    // matmul half (both halves must share the same regime by way of
    // the AOCL-DLP path they each route to; the predicate below is
    // applied symmetrically and `m_tile_supported` further enforces
    // cross-expert dtype uniformity within each half):
    //
    //   (A) BF16 end-to-end — `dtypes.{src,wei,dst} = bf16` and
    //       `dynamic_quant == false`.  Original phase-1 contract;
    //       scratch tile element size = 2 B (bf16).
    //
    //   (B) WOQ INT4 — `dtypes.{src,dst} = bf16`, `dtypes.wei ∈
    //       {s4, u4}`, `dynamic_quant == false`, per-channel
    //       `quant_params.wei_scale.buff != nullptr`.  Weight
    //       dequantization happens inside the AOCL DLP WOQ pre-op
    //       per-tile, so the slice-resident pipeline is unchanged
    //       (src + dst + scratch all bf16; weight access pattern is
    //       N-broadcast over the slice's M rows).  `is_weights_const`
    //       must hold for every active expert (the WOQ fast path
    //       caches the dequant prepack, see
    //       `aocl_postop.cpp::is_woq`).
    //
    //   (C) DQ-INT8 per-token symmetric — `dtypes.src = bf16`,
    //       `dtypes.wei = s8`, `dtypes.dst = bf16`,
    //       `dtypes.compute = s8`, `dynamic_quant == true`,
    //       per-token src quant granularity
    //       (`quant_params.src_scale.dims == {M[i], 1}`, `dt = f32`)
    //       — already enforced for Op1 by `m_tile_supported`'s
    //       row-locality predicate and re-asserted symmetrically
    //       for Op2 here.  Per-channel wei quant is permitted; the
    //       AOCL DLP s8s8 → bf16 kernel reads `wei_scale` from
    //       `quant_params.wei_scale.buff`.  Symmetric only for v1
    //       (`compute == s8`, no `src_zp`); u8/asymmetric is a
    //       future extension and is rejected explicitly to avoid
    //       silently dropping the zero-point in Stage 2b.
    //
    // Mixed-dtype calls (e.g. W13 bf16 / W2 dynamic-int8, or W13
    // bf16 / W2 woq-s8) are rejected because the executor's unified
    // scratch model only fits when both halves share the same
    // bf16-element staging tile + the same Stage 2b plumbing
    // (present in DQ-INT8 only).  Probing only [0] on each side is
    // correct: `m_tile_supported` below enforces cross-expert dtype
    // uniformity within each half.
    if (params_w13.empty() || params_w2.empty()) return false;

    // `classify_half` accepts a single matmul half under regime (A),
    // (B), or (C).  Returns the regime tag so the outer gate can
    // require both halves agree on which one.
    enum HalfRegime {
        kRegimeNone = 0,
        kRegimeBF16 = 1,
        kRegimeWOQ = 2,
        kRegimeDQINT8 = 3,
    };
    auto classify_half
            = [](const matmul_params &p, const std::vector<int> &Mvec,
                      int half_num_ops) -> HalfRegime {
        if (p.dtypes.dst != data_type_t::bf16) return kRegimeNone;
        if (p.dtypes.src != data_type_t::bf16) return kRegimeNone;
        if (p.dtypes.wei == data_type_t::bf16) {
            if (p.dynamic_quant) return kRegimeNone;
            return kRegimeBF16;
        }
        if (p.dtypes.wei == data_type_t::s4
                || p.dtypes.wei == data_type_t::u4) {
            if (p.dynamic_quant) return kRegimeNone;
            // Per-channel wei_scale must be present so AOCL DLP can wire
            // the WOQ pre-op.  Without it the dispatch falls through
            // `is_non_quant_src_int8` and the kernel produces all-zero
            // output — see the docstring in `aocl_postop.cpp::is_woq`.
            if (p.quant_params.wei_scale.buff == nullptr) return kRegimeNone;
            return kRegimeWOQ;
        }
        if (p.dtypes.wei == data_type_t::s8) {
            // DQ-INT8: require symmetric dynamic quant on the src side, in either
            // per-token {M,1} or per-group {M,G>1} granularity.
            // `m_tile_supported` already enforces row-locality
            // (`src_scale.dims[0] == M[i]`) for Op1; we replay the
            // structural shape check here so Op2 (which `check_m_tile_
            // safe` will also visit below) is held to the same
            // contract before we commit to the regime.
            if (!p.dynamic_quant) return kRegimeNone;
            if (p.dtypes.compute != data_type_t::s8)
                return kRegimeNone; // sym only
            if (p.quant_params.src_scale.dims.size() != 2) return kRegimeNone;
            // Accept per-token {M,1} AND per-group {M,G>1}.  The pipeline drives
            // the Op2 re-quant + Stage-3 src-scale dims off G uniformly (G==1 is
            // the per-token fast path) and AOCL DLP's sym-quant consumes the
            // matching {G,N} weight scale.  G is range-checked against K (K%G==0)
            // at the call site below, where the per-pass K is in scope.
            if (p.quant_params.src_scale.dims[1] < 1) return kRegimeNone;
            // Source-scale dtype: f32 (per-token VF / plain per-group) OR bf16
            // (GGML-reordered weights carry a bf16 {G,N} wei_scale and the AOCL
            // sym-quant kernel requires the A/B scale dtypes to agree).  The Op1
            // hoist emits scales in `src_scale.dt`, and Stage 2b narrows the Op2
            // re-quant to match the wei dtype — so both granularities are served.
            if (p.quant_params.src_scale.dt != data_type_t::f32
                    && p.quant_params.src_scale.dt != data_type_t::bf16) {
                return kRegimeNone;
            }
            // Wei scale required for the AOCL DLP s8s8→bf16 kernel; the
            // bf16 dst path reads it for the per-channel dequant of the
            // s32 accumulator.  Asymmetric src_zp is rejected at the
            // compute-dtype check above.
            if (p.quant_params.wei_scale.buff == nullptr) return kRegimeNone;
            // Sanity guard: the caller's full-M src_scale.dims[0] must
            // match this half's M[i] when `m_tile_supported` will
            // accept it.  Replayed here so the regime classification
            // cannot succeed on a malformed [0]-only sample where
            // `M[0] == 0` (round-based bail handles it but the regime
            // would still be tagged DQINT8, polluting the dispatcher's
            // profiler-mode string).
            if (half_num_ops > 0 && Mvec[0] > 0
                    && p.quant_params.src_scale.dims[0]
                            != static_cast<int64_t>(Mvec[0])) {
                return kRegimeNone;
            }
            return kRegimeDQINT8;
        }
        return kRegimeNone;
    };

    const int num_ops_int = static_cast<int>(M.size());
    const HalfRegime regime_w13 = classify_half(params_w13[0], M, num_ops_int);
    const HalfRegime regime_w2 = classify_half(params_w2[0], M, num_ops_int);
    if (regime_w13 == kRegimeNone || regime_w2 == kRegimeNone) return false;
    // Both halves must agree on the regime.  Allowing a mixed
    // (BF16, WOQ), (BF16, DQ-INT8), or (WOQ, DQ-INT8) call would
    // require per-stage scratch element-size + Stage 2b plumbing
    // differentiation (today's `inter_elem = dtypes.dst` is read
    // once at the top of `try_fused_m_tile_core`, and Stage 2b
    // only runs in the DQ-INT8 arm); mixed-regime workloads are not
    // produced by today's MoE call sites.
    if (regime_w13 != regime_w2) return false;

    // Per-group source quant requires K to divide evenly into G groups.  Op1
    // groups over K_in and Op2 over K_w2 — independent counts, each derived
    // from that pass's paired weight scale ({G1,N_w13} / {G2,N_w2}).  A
    // non-dividing G would make the per-group re-quant kernel silently
    // truncate group_size = K / G; reject so the call falls back to the
    // two-pass, which validates each pass on its own.  G==1 (per-token) is a
    // no-op here.
    if (regime_w13 == kRegimeDQINT8 && num_ops_int > 0) {
        const int g1
                = mtile_src_group_count(params_w13[0].quant_params.src_scale);
        const int g2
                = mtile_src_group_count(params_w2[0].quant_params.src_scale);
        if (g1 > 1 && K_in[0] % g1 != 0) return false;
        if (g2 > 1 && K_w2[0] % g2 != 0) return false;
    }

    const bool act_supported = (fused_act == grp_matmul_gated_act_t::none)
            || (fused_act == grp_matmul_gated_act_t::silu_and_mul)
            || (fused_act == grp_matmul_gated_act_t::gelu_and_mul)
            || (fused_act == grp_matmul_gated_act_t::swiglu_oai_mul);
    if (!act_supported) return false;

    if (!m_tile_supported(layout, M, params_w13, num_ops_int)) return false;
    if (!m_tile_supported(layout, M, params_w2, num_ops_int)) return false;
    // A weight an earlier in-place custom-kernel pack rewrote is still marked
    // 'n', so `m_tile_supported` cannot see it.  Declining sends the call to
    // the two-pass, whose dispatcher fails it closed.
    if (custom_kernel::any_weight_packed_in_place(weight_w13, M)
            || custom_kernel::any_weight_packed_in_place(weight_w2, M)) {
        return false;
    }

    // WOQ-only check: AOCL DLP's WOQ fast path caches the dequant
    // prepack on the const-weight side (see `aocl_postop.cpp::416-479`
    // setup_woq_pre_ops).  Without is_weights_const the prepack
    // cache key is invalid and the WOQ pre-op can't be wired safely;
    // the call would silently degrade.  Reject early so the caller
    // falls back to the legacy two-pass which validates this itself.
    // BF16 end-to-end (regime A) and DQ-INT8 (regime C) place no
    // such constraint — the existing `m_tile_execute` accepts non-const
    // weights for both and the pipeline executor inherits that
    // behaviour.  (DQ-INT8 in fact has its OWN constant-weight cache
    // via the AOCL DLP s8s8→bf16 prepack, but does not require
    // `is_weights_const` at the gate; the executor will just miss
    // the cache on non-const-weight workloads.)
    if (regime_w13 == kRegimeWOQ) {
        for (bool wc : is_weights_const) {
            if (!wc) return false;
        }
    }

    // All structural gates passed — hand off to the pipeline executor.
    // `dst_w13_is_caller_alloc` is threaded through verbatim: when the
    // caller owns the W13 destination, the post-act tile in thread-
    // local scratch is consumed directly by Stage 3's W2 matmul AND
    // also spilled to `dst_w13[]` for downstream readers; when the
    // library owns it, the spill is skipped (Op1's library arena would
    // be a write-and-immediately-discard path).
    return try_fused_m_tile_core(layout, transA, transA_w2, transB, M, N_w13,
            K_in, alpha_w13, src, lda, weight_w13, ldb_w13, bias_w13, beta_w13,
            dst_w13, ldc_w13, dst_w13_is_caller_alloc, N_w2, K_w2, alpha_w2,
            weight_w2, ldb_w2, bias_w2, beta_w2, dst_w2, ldc_w2, fused_act,
            act_dtype, is_weights_const, params_w13, params_w2, num_threads);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
