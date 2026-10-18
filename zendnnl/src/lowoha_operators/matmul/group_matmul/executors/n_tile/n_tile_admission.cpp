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

/// ALGO 3 admission check for the quantization and post-op contract.

#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_admission.hpp"

#include <cstdint>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace detail {

// N-tile (ALGO 3) slices columns of B, so the executor must either re-anchor
// N-indexed metadata onto each `[col_start, col_start + n_tile)` window or
// select a full-N fallback. This helper admits only the configurations for
// which `group_matmul_n_tile.cpp` has one of those safe paths.
//
// SCOPE NOTE — what `n_tile_safe = false` actually does to a
// quantised workload's routing.
//
//   This helper only computes `n_tile_safe`; it does NOT make the
//   final ALGO decision.  `select_grp_matmul_algo` consults
//   `n_tile_safe` ONLY at the ALGO 3 decision points:
//
//     * Forced `env_algo == 3`: rejected → falls back to ALGO 1.
//     * Auto-select (`env_algo == 0`): every ALGO 3 answer (see
//       `auto_select_algo` below) honours `n_tile_safe`, so a
//       quantised input outside the supported sub-set below runs on
//       ALGO 1.
//
//   Other ALGOs are unaffected by this helper:
//     * Forced `env_algo ∈ {1, 2, 5, 6}` is respected as-is
//       (m_tile_safe is checked separately for ALGO 2; ALGO 1/5/6
//       have no tile-safety gate).
//
//   Quantized N-tile accepts dynamic row-local INT8, static per-tensor
//   U8xS8, and W4A8 contracts. Dynamic sources use `{M,1}` per-token or
//   `{M,G}` per-group scales paired with per-channel `{N}` / `{1,N}` or
//   per-group `{G,N}` weight scales. Static U8xS8 uses scalar source
//   scale/zero-point and scalar or per-channel symmetric weight scales.
//
//   Source-side dynamic quantization is hoisted once per expert before the
//   OMP region. N-indexed metadata is sliced or repacked by column. Weight
//   zero-points, malformed dimensions, unsupported dtypes/granularity, and
//   incompatible packed layouts reject N-tile and retain the full-N fallback.
//   Per-channel slices retain the indexing contract consumed by the AOCL DLP
//   and native INT8 kernels. Weight zero-points remain unsupported by both the
//   CK microkernel and the AOCL sym-quant fallback. The API has no dynamic
//   weight-quantization form; AOCL DLP and native kernels consume caller-
//   quantized weights.
//
//   Static U8xS8 additionally accepts one buffer-free ReLU/sigmoid or the
//   exact F32 `[M,N]` binary-multiply then binary-add chain. CK offsets the
//   binary buffers by tile column; a runtime CK refusal switches to full-N
//   Sequential. Gated activation cannot be combined with these post-ops.
//   There are three gated activation forms, all restricted to BF16 output.
//
// PRECONDITION: the caller has already run `check_m_tile_safe` and
// confirmed it returned true.  This helper intentionally does NOT
// re-run those checks — the orchestrator `select_grp_matmul_algo`
// always calls M-tile first and only invokes this when m_tile_safe is
// true, so a second pass would just be duplicated work.
bool check_n_tile_extra(const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<matmul_params> &params, int num_ops,
        grp_matmul_gated_act_t fused_act) {
    // Per-token source side: dims must be exactly `{M[i], 1}`
    // (rank-2, first dim equals this expert's row count, scalar
    // across K).  The `M[i]` match is the key row-locality signal —
    // it excludes per-tensor / per-column / per-channel-on-src
    // layouts where the first dim is 1 (or empty) while accepting
    // the single-row decode case `M[i] == 1` with dims `{1, 1}`.
    //
    // Dynamic-quant input reaches this gate with `buff == nullptr` and
    // is hoisted by flat_n_tile. Grouped dynamic-quant reaches this gate
    // after pre-quantizing to S8, so `buff != nullptr` and the same
    // per-token dims describe the ready-to-use scale buffer.
    // Per-group source side: dims `{M[i], G}` with G > 1 (one scale per
    // K-group, row-local).  Pairs with a `{G, N}` per-group weight scale.
    // The source scale is N-independent, so N-tile column slicing leaves it
    // whole; `do_tile` slices the WEIGHT scale to `{G, n_tile}`.  Reaches
    // this gate either as dynamic (`buff == nullptr`, hoisted by flat_n_tile)
    // or grouped-pre-quantized S8 (`buff != nullptr`).
    auto is_per_group_src
            = [](const matmul_quantization_params_t::matmul_quant_t &q,
                      int M_expert) -> bool {
        return q.dims.size() == 2 && q.dims[0] == static_cast<int64_t>(M_expert)
                && q.dims[1] > 1;
    };

    // Per-group weight side: dims `{G, N}` with G > 1 and N > 1, non-null
    // buff (statically quantised by the caller).  do_tile column-slices the
    // {G, N} scale into a contiguous {G, n_tile} per-thread scratch for the
    // AOCL sym-quant GEMM.
    auto is_per_group_wei =
            [](const matmul_quantization_params_t::matmul_quant_t &q) -> bool {
        return q.buff != nullptr && q.dims.size() == 2 && q.dims[0] > 1
                && q.dims[1] > 1;
    };
    // Same active-range constraint as `check_m_tile_safe` above —
    // tail slots carry framework prepack metadata, not real per-call
    // state, and would falsely flip n-tile-safe to false.
    for (int i = 0; i < num_ops; ++i) {
        // Inactive experts (M==0) do no compute.  The grouped / fallback DQ
        // pre-pass clears their `dynamic_quant` and leaves the original bf16
        // src (no per-token src_scale), so evaluating them here would hit the
        // `!dynamic_quant && !grouped_s8_src` source-side reject below and
        // veto ALGO 3 for the WHOLE call — even though every ACTIVE expert is
        // a valid grouped-s8 / dynamic-INT8 shape.  Skip them (matches the
        // first-active reference in `check_m_tile_safe`).
        if (M[i] == 0) { continue; }
        const auto &qp = params[i].quant_params;
        const bool static_u8s8 = is_static_u8s8_quant_config(params[i]);

        // Detect any quant intent on this expert.  If every quant field
        // is empty AND `dynamic_quant` is false, this is a pure
        // non-quantised call and there's nothing to gate — ALGO 3 is
        // free to run.
        const bool static_int8_dtype = !params[i].dynamic_quant
                && params[i].dtypes.wei == data_type_t::s8
                && (params[i].dtypes.src == data_type_t::s8
                        || params[i].dtypes.src == data_type_t::u8);
        const bool any_quant = qp.wei_scale.buff != nullptr
                || qp.wei_zp.buff != nullptr || qp.src_scale.buff != nullptr
                || qp.src_zp.buff != nullptr || qp.dst_scale.buff != nullptr
                || qp.dst_zp.buff != nullptr || params[i].dynamic_quant
                || static_int8_dtype;

        if (any_quant) {
            // W4A8 (s4 weight): accepted on the N-tile path when is_w4a8_config
            // passes AND the shape is a valid per-group layout that flat_n_tile
            // can column-slice.  Inner aocl_dlp expands s4→s8 into the W4A8
            // LRU and rewrites tile_params to flow through sym-quant GEMM;
            // inner aocl_dlp_blocked slices packed s4 directly.
            // u4 remains rejected (no symmetric W4A8 support).
            if (params[i].dtypes.wei == data_type_t::u4) { return false; }
            if (is_w4a8_config(params[i])) {
#if !ZENDNNL_DEPENDS_AOCLDLP
                // W4A8 N-tile needs AOCL-DLP (s4→s8 plain cache, per-tile
                // sym-quant reorder, broadcast_w4a8_src_scale).  Without AOCL
                // the ALGO-1 path falls through to the reference W4A8 kernel.
                return false;
#else
                // ALGO 3 needs raw packed s4 (mem_format_b='n'); native
                // prepack is full-N only.
                if (params[i].mem_format_b != 'n') { return false; }
#endif
                const bool w4a8_src_ok
                        = quant_param_is_per_token(qp.src_scale, M[i])
                        || is_per_group_src(qp.src_scale, M[i]);
                const bool w4a8_wei_ok = is_per_group_wei(qp.wei_scale);
                if (!w4a8_src_ok || !w4a8_wei_ok) { return false; }
                continue;
            }

            if (static_u8s8) {
                if (!static_u8s8_quant_supported(params[i], N[i])) return false;
            } else {
                // Source side: accept either (a) dynamic BF16/F32 input that
                // flat_n_tile will hoist, or (b) already grouped-quantized S8
                // input with a ready per-token source scale buffer.
                const bool grouped_s8_src = !params[i].dynamic_quant
                        && params[i].dtypes.src == data_type_t::s8
                        && qp.src_scale.buff != nullptr;
                if (!params[i].dynamic_quant && !grouped_s8_src) {
                    return false;
                }

                // Source dims: `{M[i], 1}` per-token (incl. the `M[i] == 1`
                // decode case `{1, 1}`) OR `{M[i], G}` per-group (G > 1).
                // For grouped_s8_src the scale buffer must be non-null (checked
                // above). For dynamic input, nullness is a hoist-allocation
                // contract, not a per-token-scope contract.
                const bool src_per_token
                        = quant_param_is_per_token(qp.src_scale, M[i]);
                const bool src_per_group = is_per_group_src(qp.src_scale, M[i]);
                if (!src_per_token && !src_per_group) { return false; }
                // Per-group is symmetric-only on the N-tile path (the AOCL
                // sym-quant kernel and source reorder are both symmetric).
                if (src_per_group && qp.src_zp.buff != nullptr) return false;
                if (grouped_s8_src && qp.src_zp.buff != nullptr) return false;
                if (src_per_token && qp.src_zp.buff != nullptr
                        && !quant_param_is_per_token(qp.src_zp, M[i]))
                    return false;

                // Weight side: per-channel `{N}` / `{1, N}` (per-token src)
                // OR per-group `{G, N}` (per-group src).
                const bool wei_per_channel
                        = quant_param_is_per_channel(qp.wei_scale, N[i]);
                const bool wei_per_group = is_per_group_wei(qp.wei_scale);
                if (!wei_per_channel && !wei_per_group) return false;
                if (src_per_group && !wei_per_group) return false;
                if (wei_per_group && params[i].mem_format_b != 'n')
                    return false;
                if (qp.wei_zp.buff != nullptr) return false;
            }
        }

        // Exact static-U8xS8 `[M,N]` mul+add is handled by CK, with a full-N
        // Sequential fallback when CK cannot run. Other buffer-bearing
        // post-ops remain unsafe for N slicing. A gated activation cannot be
        // combined with a static post-op because the CK gated epilogue does
        // not apply that second operation.
        if (static_u8s8 && !params[i].postop_.empty()
                && fused_act != grp_matmul_gated_act_t::none) {
            return false;
        }
        if (!(static_u8s8
                    && int8_ck_binary_postops_supported(
                            params[i], M[i], N[i]))) {
            for (const auto &po : params[i].postop_) {
                if (po.buff != nullptr) { return false; }
            }
        }

        // `do_tile` column-slices `wei_scale` and `wei_zp` for each tile but
        // never touches `dst_scale` / `dst_zp`, and the AOCL post-op layer
        // infers their granularity from the buffer length against the N it is
        // handed -- which is `n_tile` on a tile call.  A per-channel `{N}`
        // dst scale would therefore reach every tile pointing at element 0,
        // so each tile would dequantise its own columns with the FIRST
        // `n_tile` scales.  Per-tensor (a single element) is N-independent
        // and safe; anything wider has no per-tile meaning here, so decline
        // and let a full-N ALGO serve it.
        const auto &dsc = params[i].quant_params.dst_scale;
        const auto &dzp = params[i].quant_params.dst_zp;
        const int64_t dsc_nelems = dsc.buff != nullptr
                ? checked_quant_param_num_elements(dsc.dims)
                : 0;
        const int64_t dzp_nelems = dzp.buff != nullptr
                ? checked_quant_param_num_elements(dzp.dims)
                : 0;
        if (dsc_nelems < 0 || dzp_nelems < 0 || dsc_nelems > 1
                || dzp_nelems > 1) {
            return false;
        }
    }
    return true;
}

} // namespace detail

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
