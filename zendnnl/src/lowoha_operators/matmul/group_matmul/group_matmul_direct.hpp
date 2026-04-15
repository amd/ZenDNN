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

// --- Parallel expert dispatch ---

/**
 * @brief Run independent expert GEMMs (parallel group matmul path).
 *
 * @param gemm_mode_out  If non-null, receives a static string literal for logging
 *                       ("sequential_experts", "flat_ccd_m_tile", "flat_ccd_n_tile",
 *                       "multilevel", or "per_expert").
 */
void group_matmul_run_parallel_dispatch(
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
    const char **gemm_mode_out);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
