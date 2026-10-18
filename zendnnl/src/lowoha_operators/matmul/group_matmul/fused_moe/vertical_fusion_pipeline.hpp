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

/// Fused-MoE vertical-fusion pipeline: W13 -> gated activation -> W2 at
/// M-tile slice granularity, and its eligibility-gated wrapper.

#ifndef ZENDNNL_GROUP_MATMUL_VERTICAL_FUSION_PIPELINE_HPP
#define ZENDNNL_GROUP_MATMUL_VERTICAL_FUSION_PIPELINE_HPP

#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/// ALGO 2 — vertical-fusion pipeline (W13 → gated act → W2) at
/// M-tile slice granularity.  Tri-regime: BF16 end-to-end, WOQ-INT4
/// (s4 / u4 weights), and DQ-INT8 per-token symmetric (s8 weights
/// with bf16 src/dst).  Defined in `vertical_fusion_pipeline.cpp`.
/// See the doc-block on the function for the
/// engagement contract, return semantics, and per-regime memory
/// management notes.
///
/// Returns `true` when the pipeline engaged AND completed
/// successfully (caller skips legacy two-pass); `false` when it
/// declined to engage (round-based / multi-tier / wide-N regime,
/// scratch budget exhausted, per-thread alloc failure, or — in the
/// DQ-INT8 regime — pre-OMP source-quant hoist failure) — in
/// which case dst_w2 and dst_w13 are guaranteed to be untouched
/// and the caller MUST run the legacy two-pass.
bool flat_m_tile_pipeline_bf16(const std::vector<char> &layout,
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
        std::vector<matmul_params> &params_w2, int num_threads);

/// Eligibility-gated wrapper around `flat_m_tile_pipeline_bf16`.
///
/// Returns `true` iff ALL of the following hold, in which case the
/// pipeline has produced both Op1 (W13) and Op2 (W2) outputs and the
/// caller MUST NOT re-run a legacy two-pass over the same buffers:
///
///   * Env knob `ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION != -1`
///     (cheapest gate — short-circuits before any data inspection).
///   * BOTH params_w13[0] and params_w2[0] are in the SAME dtype
///     regime, where the eligible regimes are:
///       (A) BF16 end-to-end — `src/wei/dst == bf16` and
///           `dynamic_quant == false`.
///       (B) WOQ-INT4 — `src/dst == bf16`, `wei ∈ {s4, u4}`,
///           `dynamic_quant == false`, `quant_params.wei_scale.buff
///           != nullptr`, and `is_weights_const[*] == true` for
///           every active expert (the AOCL DLP WOQ fast path caches
///           the dequant prepack on the const-weight side).
///       (C) DQ-INT8 per-token symmetric — `src/dst == bf16`,
///           `wei == s8`, `compute == s8`, `dynamic_quant == true`,
///           `quant_params.src_scale.dims == {M[i], 1}`,
///           `src_scale.dt == f32`, `quant_params.wei_scale.buff
///           != nullptr` (per-channel wei scale required by the
///           AOCL DLP s8s8 → bf16 kernel).  Symmetric only —
///           asymmetric u8 src_zp is rejected at the `compute`
///           dtype check.
///     A mixed-regime call (e.g. BF16 W13 + WOQ-INT4 W2, or BF16
///     W13 + DQ-INT8 W2) is rejected because the executor's
///     scratch uses a single staging element size AND the Stage 2b
///     re-quant runs only in the DQ-INT8 arm; mixed-half scratch
///     sizing / Stage-2b plumbing is not in scope for the unified
///     executor today.
///   * `fused_act` is in the supported set: `none`, `silu_and_mul`,
///     `gelu_and_mul`, or `swiglu_oai_mul` (the four activations
///     `apply_gated_act_inplace` handles on the thread-local scratch).
///   * Both Op1 and synthesized Op2 matmul shapes pass
///     `check_m_tile_safe` (row-major, dtype uniformity, no packed-B,
///     etc. — see the predicate's doc-block in Section H.5 below).
///   * `flat_m_tile_pipeline_bf16` itself committed (single-tier
///     planner agreed, scratch budget fits including any DQ-INT8
///     Stage 2b bytes, per-thread alloc succeeded, and — in the
///     DQ-INT8 regime — the pre-OMP per-expert src hoist succeeded
///     for every active expert).
///
/// When this returns `false`, NO writes have been made to either
/// `dst_w13` or `dst_w2`, so the caller can safely fall through to a
/// legacy two-pass dispatch over the same buffers.
bool try_flat_m_tile_pipeline_bf16(const std::vector<char> &layout,
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
        std::vector<matmul_params> &params_w2, int num_threads);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_VERTICAL_FUSION_PIPELINE_HPP
