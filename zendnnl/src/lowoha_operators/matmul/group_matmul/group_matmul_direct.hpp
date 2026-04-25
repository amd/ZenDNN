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

/**
 * @file group_matmul_direct.hpp
 * @brief Group matmul: MoE post-op, parallel dispatch, and direct-path helpers.
 *
 * The public entry @c group_matmul_direct is declared in lowoha_matmul.hpp;
 * include that header for the stable API. This header holds shared declarations
 * for group_matmul_parallel.cpp and group_matmul_moe_postop.cpp (and types used by
 * group_matmul_direct.cpp, which implements @c group_matmul_direct).
 */

#ifndef LOWOHA_GROUP_MATMUL_DIRECT_HPP
#define LOWOHA_GROUP_MATMUL_DIRECT_HPP

#include <cstdint>
#include <vector>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// --- MoE post-op: weighted-reduce after parallel expert GEMMs ---

/**
 * @brief Optional MoE post-op parameters for parallel group matmul.
 *
 * When supplied to @c group_matmul_direct (parallel mode only), the post-op
 * performs a weighted-reduce over pre-gathered expert output rows:
 *
 *   For each token t and hidden dim d:
 *     output[t, d] = Σ_k  topk_weights[t, k] * row_ptrs[t*topk+k][d]
 *
 * When @c skip_weighted is true, all weights are implicitly 1.0.
 *
 * The caller builds @c row_ptrs during the token-to-expert scatter step,
 * fusing scatter + gather in one pass on the frontend side.  Each entry
 * is a pointer to the start of a D-wide row in an expert dst buffer.
 * This design allows future fusion with the expert GEMM kernels.
 *
 * Constraints checked by validation:
 *   - @c row_ptrs and @c output must be non-null.
 *   - @c num_tokens > 0 and @c topk > 0.
 *   - @c ldc_output >= D (hidden dimension).
 *   - dst dtype must be FP32 or BF16.
 *   - @c topk_weights required unless @c skip_weighted is true.
 */
struct group_matmul_moe_postop_params {
  /// Number of input tokens (rows in the output buffer).
  int num_tokens = 0;

  /// Number of experts selected per token.
  int topk = 0;

  /// Output buffer: row-major [num_tokens, ldc_output].
  /// First D columns of each row are written (FP32 or BF16).
  void *output = nullptr;

  /// Leading dimension of the output buffer (>= D).
  int ldc_output = 0;

  /// Routing weights: tightly packed [num_tokens, topk] (row-major).
  /// Entry [t * topk + k] is the weight for token t's k-th expert.
  /// Required unless @c skip_weighted is true.
  const float *topk_weights = nullptr;

  /// When true, every routing weight is implicitly 1.0 and
  /// @c topk_weights may be nullptr (plain gather-sum, no weighting).
  bool skip_weighted = false;

  /// Pre-gathered row pointers: flat array of size num_tokens * topk.
  /// Entry row_ptrs[t * topk + k] points to the start of a D-wide row
  /// (FP32 or BF16) in an expert dst buffer — the row that contributes
  /// to token t's k-th expert slot.
  ///
  /// The caller builds this during token-to-expert scatter:
  ///   row_ptrs[t * topk + k] = dst[expert_id] + row_j * ldc[expert_id]
  /// (with appropriate element-size scaling for the dst dtype).
  const void **row_ptrs = nullptr;
};

status_t validate_group_matmul_moe_postop(
    const group_matmul_moe_postop_params *postop,
    int D,
    data_type_t dst_elem);

status_t group_matmul_moe_postop_execute(
    const group_matmul_moe_postop_params *postop,
    int D,
    int num_threads,
    data_type_t dst_elem);

// --- MoE gated activation: fused act(gate) * up after GEMM ---

/**
 * @brief Gated activation type for MoE fused gate+up GEMM.
 *
 * Applied after the GEMM that uses fused [gate_W | up_W] weights.
 * The GEMM output dst[M, 2*dim] is split into gate[:, 0:dim] and
 * up[:, dim:2*dim].  The activation computes in-place:
 *   dst[:, 0:dim] = act(gate) * up
 * The second half (up columns) becomes garbage after activation.
 * The caller passes ldc=2*dim to the subsequent down_proj GEMM as lda.
 */
enum class grp_matmul_gated_act_t : int {
  none = 0,           ///< No gated activation (down_proj or unfused).
  silu_and_mul = 1,   ///< SiLU(gate) * up — Mixtral, Llama, Qwen.
  gelu_and_mul = 2,   ///< GELU(gate) * up — some GPT variants.
  swiglu_oai_mul = 3  ///< SwigluOAI — interleaved gate/up layout (GPT-OSS).
};

/**
 * @brief Parameters for MoE gated activation post-op.
 *
 * Single struct (not per-expert) — all experts in a group_matmul call
 * use the same activation type.  The activation operates on each expert's
 * dst buffer in-place.
 */
struct grp_matmul_gated_act_params {
  grp_matmul_gated_act_t act;  ///< Activation type (or none).

  grp_matmul_gated_act_params() : act(grp_matmul_gated_act_t::none) {}
};

/**
 * @brief Apply gated activation to all experts' dst buffers.
 *
 * For each expert e: dst[e][:, 0:dim] = act(dst[e][:, 0:dim]) * dst[e][:, dim:2*dim]
 * where dim = N[e] / 2.
 *
 * @param act_params  Activation type (nullptr or act==none → no-op).
 * @param dst         Per-expert output buffers from GEMM [M_e, N_e].
 * @param M           Per-expert row counts.
 * @param N           Per-expert column counts (must be even: N = 2*dim).
 * @param ldc         Per-expert leading dimensions of dst.
 * @param dst_dtype   Data type of dst buffers (FP32 or BF16).
 * @param num_threads OMP thread count for parallel execution.
 */
status_t group_matmul_moe_act_execute(
    const grp_matmul_gated_act_params *act_params,
    const std::vector<void *> &dst,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &ldc,
    data_type_t dst_dtype,
    int num_threads);

/**
 * @brief Apply gated activation in-place on a row range of a single expert.
 *
 * Single-threaded — designed to be called from within OMP parallel regions
 * (ALGO 2 M-tile, ALGO 1/4/5 per-expert) for fused activation.
 *
 * @param act       Activation type (none is a no-op).
 * @param dst       Expert output buffer [M, ldc].
 * @param row_start First row to process (inclusive).
 * @param row_end   Last row to process (exclusive).
 * @param N         Total columns (must be even: N = 2*dim).
 * @param ldc       Leading dimension of dst.
 * @param dst_dtype Data type of dst (f32 or bf16).
 */
void apply_gated_act_inplace(
    grp_matmul_gated_act_t act,
    void *dst, int row_start, int row_end,
    int N, int ldc, data_type_t dst_dtype);

/**
 * @brief Apply swiglu_oai_mul to an M x pairs interleaved tile, in-place.
 *
 * Used by the ALGO 3 N-tile fused-swiglu-oai path.  Each OMP thread owns a
 * pair-aligned column range [col_start, col_start + 2*pairs) of the matmul
 * output.  After an OMP barrier separates matmul from activation, this
 * function reads the thread's `pairs` interleaved (g, u) pairs from those
 * columns of every row and writes `pairs` activated values into columns
 * [col_start/2, col_start/2 + pairs) of the same rows.
 *
 * In-place safety: at col_start == 0 the read-ahead property (read 2n, 2n+1
 * before writing n) prevents self-corruption; at col_start > 0 the write
 * range is strictly to the left of the read range and cannot stomp it.
 * Cross-thread correctness relies on the caller placing a barrier between
 * the matmul writes and the first call into this function.
 *
 * Only swiglu_oai_mul is supported (interleaved gate/up layout).  Legacy
 * split-halves activations (silu_and_mul / gelu_and_mul) are still handled
 * via the separate-pass path.
 *
 * @param dst_buf    Expert output buffer (base pointer to row 0, col 0).
 * @param M          Number of rows in the expert's output.
 * @param col_start  First column in the interleaved buffer (must be even).
 * @param pairs      Number of (g, u) pairs this thread owns.
 * @param ldc        Leading dimension of dst_buf (elements, not bytes).
 * @param dtype      Buffer element type (f32 or bf16).
 */
void apply_swiglu_oai_tile_rows(
    void *dst_buf, int M, int col_start, int pairs,
    int ldc, data_type_t dtype);

// --- Fused MoE: Op1(gate+up) → activation → Op2(down_proj) in one pass ---

/**
 * @brief Parameters for fused MoE execution (Op1 → Act → Op2).
 *
 * When provided to group_matmul_direct, the entire MoE block is executed
 * as a single API call.
 *
 * V1 current implementation — always two-pass for all GRP_ALGO values:
 *     Pass 1: Op1 (gate+up) + gated activation via parallel dispatch.
 *     Pass 2: Op2 (down_proj)                  via parallel dispatch.
 *   Both passes honor ZENDNNL_GRP_MATMUL_ALGO.  The dispatcher may fuse
 *   the gated activation into Pass 1's epilogue (all activations on
 *   ALGO 1/2/4/5, and swiglu_oai_mul on ALGO 3); otherwise a separate
 *   activation sub-pass is applied to Pass 1's output before Pass 2.
 *   A future version may add per-expert deep fusion (Op1→Act→Op2
 *   chained in L1/L2/L3) for further cache-locality wins.
 *
 * The weighted-reduce (moe_postop) always runs in a separate pass after
 * Op2 since it requires all experts' outputs to be complete.
 *
 * All vectors must have size num_ops (one entry per expert).
 *
 * Constraints:
 *   - N must be even (K_down = N / 2 for each expert).
 *   - dst_down[i] must be at least [M[i], N_down[i]].
 *   - Op2 inherits the caller-provided per-expert layout from Op1;
 *     transA=false is hardcoded for Op2, transB is inherited from Op1
 *     (typically both false for MoE row-major weights).
 *   - When act==none, Op2 reads the raw first-half columns of Op1 output.
 *     For correct MoE semantics, gated_act should be enabled.
 */
struct grp_matmul_fused_moe_params {
  /// Per-expert down_proj weights [dim, N_down[i]].
  /// Must use same dtype as Op1 weights (params[i].dtypes.wei).
  std::vector<const void *> down_weight;
  std::vector<int> N_down;                ///< Per-expert output columns of down_proj.
  std::vector<int> ldb_down;              ///< Leading dimension of down_weight per expert.
  std::vector<const void *> bias_down;    ///< Per-expert bias for down_proj (nullptr OK).
  data_type_t bias_dt_down = data_type_t::none;  ///< Bias dtype for Op2 (none = no bias).
  /// Per-expert down_proj output [M, N_down[i]].
  /// Written with Op1 dst dtype (params[i].dtypes.dst).
  std::vector<void *> dst_down;
  std::vector<int> ldc_down;              ///< Leading dimension of dst_down per expert.
};

/**
 * @brief Execute fused MoE: Op1(gate+up) → activation → Op2(down_proj).
 *
 * V1 runs the flow as two passes of group_matmul_run_parallel_dispatch;
 * both passes honor ZENDNNL_GRP_MATMUL_ALGO.  See grp_matmul_fused_moe_params
 * (above) for the full design note.
 *
 * @param gemm_mode_out If non-null, receives a string literal for logging.
 */
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
    const char **gemm_mode_out = nullptr);

// --- Parallel expert dispatch ---

/**
 * @brief Run independent expert GEMMs (parallel group matmul path).
 *
 * @param gemm_mode_out  If non-null, receives a static string literal for logging
 *                       ("sequential_experts", "flat_m_tile", "flat_n_tile",
 *                       "multilevel", or "per_expert").
 */
/**
 * @return true if the gated activation was fused into the ALGO's epilogue
 *         (caller should skip the separate activation pass).  false if
 *         the caller must apply activation separately.  Today:
 *           ALGO 1 / 2 / 4 / 5 — always fuse any supported activation.
 *           ALGO 3 N-tile      — fuses swiglu_oai_mul only (interleaved
 *                                layout); silu_and_mul / gelu_and_mul are
 *                                returned unfused because their split-
 *                                halves layout does not colocate (g, u)
 *                                pairs on the same thread's N-tile.
 */
bool group_matmul_run_parallel_dispatch(
    const std::vector<char> &layout,
    const std::vector<bool> &transA,
    const std::vector<bool> &transB,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &K,
    const std::vector<float> &alpha,
    const std::vector<const void *> &src,
    const std::vector<int> &lda,
    const std::vector<const void *> &weight,
    const std::vector<int> &ldb,
    const std::vector<const void *> &bias,
    const std::vector<float> &beta,
    const std::vector<void *> &dst,
    const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads,
    const char **gemm_mode_out,
    grp_matmul_gated_act_t fused_act = grp_matmul_gated_act_t::none,
    data_type_t act_dtype = data_type_t::none);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
