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

/// Fused MoE: Op1(gate+up) → activation → Op2(down_proj) in one API call.
///
/// V1 implementation: always two-pass for all ALGOs.
///   Pass 1: Op1 + gated activation via group_matmul_run_parallel_dispatch.
///   Pass 2: Op2 (down_proj)         via group_matmul_run_parallel_dispatch.
///
/// Both passes run through the same parallel dispatcher, so the caller's
/// ZENDNNL_GRP_MATMUL_ALGO override is honored for Pass 1 AND Pass 2.  Pass 2
/// is not forced to ALGO 0; it simply sees different dimensions (M,N_down,
/// K_down = N/2), which usually leads the auto-select path to pick a
/// different concrete strategy than Pass 1 but the override applies either
/// way.
///
/// Deep fusion (Op1→Act→Op2 per expert/tile) is a future optimization.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;

// ═══════════════════════════════════════════════════════════════════════
// Two-pass: Op1+Act via parallel dispatch, then Op2 via parallel dispatch
// ═══════════════════════════════════════════════════════════════════════

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
    const     std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads,
    const char **gemm_mode_out) {

  const size_t num_ops = M.size();

  // ── Input validation ──────────────────────────────────────────────
  // group_matmul_direct performs full validation before calling here,
  // but this function is declared in a header and could be invoked
  // directly, so guard against mismatched vectors, invalid dimensions,
  // and inconsistent bias configuration.  All checks are O(num_ops),
  // run once per call, and are negligible vs GEMM cost.
  if (num_ops == 0) return status_t::failure;
  if (layout.size() != num_ops || transA.size() != num_ops ||
      transB.size() != num_ops || N.size() != num_ops ||
      K.size() != num_ops || src.size() != num_ops ||
      weight.size() != num_ops || dst.size() != num_ops ||
      lda.size() != num_ops || ldb.size() != num_ops ||
      ldc.size() != num_ops || params.size() != num_ops ||
      alpha.size() != num_ops || beta.size() != num_ops ||
      bias.size() != num_ops || is_weights_const.size() != num_ops ||
      fused.down_weight.size() != num_ops ||
      fused.N_down.size() != num_ops ||
      fused.dst_down.size() != num_ops ||
      fused.ldb_down.size() != num_ops ||
      fused.ldc_down.size() != num_ops ||
      fused.bias_down.size() != num_ops)
    return status_t::failure;

  // Per-expert consistency: dimensions, leading strides, pointers, and
  // bias dtype.  Accumulate total_M to detect the all-zero-work case.
  int64_t total_M = 0;
  bool any_bias_down = false;
  for (size_t i = 0; i < num_ops; ++i) {
    // Core dimensions: M non-negative, N/K positive, N even (K_down = N/2).
    if (M[i] < 0 || N[i] <= 0 || K[i] <= 0) return status_t::failure;
    if ((N[i] & 1) != 0) return status_t::failure;
    if (fused.N_down[i] <= 0) return status_t::failure;
    // Op1 leading dimensions must satisfy the per-op row-major layout.
    // group_matmul_direct already enforces these for the Op1 side, but
    // this function is public (declared in group_matmul_direct.hpp) so
    // we re-check here in case a caller bypasses the dispatcher.
    const int K_down = N[i] / 2;  // K for Op2 = N/2 after gated activation
    if (lda[i] < K[i]) return status_t::failure;
    if (ldb[i] < (transB[i] ? K[i] : N[i])) return status_t::failure;
    if (ldc[i] < N[i]) return status_t::failure;
    // Op2 (down_proj) leading dimensions: same row-major rules with
    // K_down = N[i]/2 and N_down = fused.N_down[i].  Src for Op2 is the
    // Op1 dst buffer with stride ldc[i] (not a separate Op2 lda), so
    // ldc[i] must be >= K_down; already implied by ldc[i] >= N[i] ≥
    // 2·K_down, no extra check needed.
    if (fused.ldb_down[i] < (transB[i] ? K_down : fused.N_down[i]))
      return status_t::failure;
    if (fused.ldc_down[i] < fused.N_down[i])
      return status_t::failure;
    // Active experts (M>0) need non-null buffers for both ops.
    if (M[i] > 0) {
      if (src[i] == nullptr || weight[i] == nullptr || dst[i] == nullptr)
        return status_t::failure;
      if (fused.down_weight[i] == nullptr || fused.dst_down[i] == nullptr)
        return status_t::failure;
    }
    if (fused.bias_down[i] != nullptr) any_bias_down = true;
    total_M += M[i];
  }
  // Non-null bias without a valid dtype would let the AOCL backend
  // silently treat the buffer as FP32 (potential wrong results or OOB reads).
  if (any_bias_down && fused.bias_dt_down == data_type_t::none)
    return status_t::failure;
  // A gated activation with dtype=none would pass through to the
  // swiglu/silu/gelu tile kernels which reinterpret the buffer as
  // f32/bf16.  Reject here so the error surfaces in a visible place
  // rather than as a segfault deep in the activation path.
  if (act != grp_matmul_gated_act_t::none
      && act_dtype != data_type_t::f32
      && act_dtype != data_type_t::bf16)
    return status_t::failure;

  // No active work: every expert has M=0.  Return success without
  // spawning OMP regions or touching the Op2 dispatch.
  if (total_M == 0) {
    if (gemm_mode_out) *gemm_mode_out = "fused_moe_skip";
    return status_t::success;
  }

  // ── Pass 1: Op1 (gate+up) + activation ────────────────────────────
  // Runs through the parallel dispatcher, which honors
  // ZENDNNL_GRP_MATMUL_ALGO.  The dispatcher returns act_fused=true
  // when it has already applied the activation inline:
  //   - ALGO 1 / 2 / 4 / 5: always fuse (per-expert or per-M-tile).
  //   - ALGO 3 N-tile: fuses only swiglu_oai_mul (interleaved layout);
  //     legacy silu_and_mul / gelu_and_mul still fall through to the
  //     separate-pass branch below.
  const char *pass1_mode = nullptr;
  const bool act_fused = group_matmul_run_parallel_dispatch(
      layout, transA, transB, M, N, K, alpha,
      src, lda, weight, ldb, bias, beta, dst, ldc,
      is_weights_const, params, num_threads, &pass1_mode,
      act, act_dtype);

  // Separate-pass activation for dispatches that cannot fuse the
  // activation inline (e.g., ALGO 3 with silu_and_mul / gelu_and_mul).
  if (act != grp_matmul_gated_act_t::none && !act_fused) {
    grp_matmul_gated_act_params act_p;
    act_p.act = act;
    status_t act_st = group_matmul_moe_act_execute(
        &act_p, dst, M, N, ldc, act_dtype, num_threads);
    if (act_st != status_t::success) return act_st;
  }

  // ── Pass 2: Op2 (down_proj) ───────────────────────────────────────
  // Source = activated Op1 output in dst[:, 0:dim] with stride ldc.
  std::vector<int> K_down(num_ops);
  for (size_t i = 0; i < num_ops; ++i)
    K_down[i] = N[i] / 2;

  std::vector<float> alpha_down(num_ops, 1.0f);
  std::vector<float> beta_down(num_ops, 0.0f);
  std::vector<bool> transA_down(num_ops, false);

  std::vector<const void *> src_down(num_ops);
  for (size_t i = 0; i < num_ops; ++i)
    src_down[i] = dst[i];

  // Op2 params: source dtype = Op1 output dtype (critical for mixed precision).
  std::vector<matmul_params> params_down(num_ops);
  for (size_t i = 0; i < num_ops; ++i) {
    params_down[i] = matmul_params{};
    params_down[i].dtypes.src = params[i].dtypes.dst;
    params_down[i].dtypes.wei = params[i].dtypes.wei;
    params_down[i].dtypes.dst = params[i].dtypes.dst;
    params_down[i].dtypes.bias = fused.bias_dt_down;
    params_down[i].num_threads = params[i].num_threads;
  }

  // Pass 2 also runs through group_matmul_run_parallel_dispatch and
  // therefore honors the same ZENDNNL_GRP_MATMUL_ALGO override.  The
  // dispatcher may pick a different concrete strategy from Pass 1 because
  // Op2 dimensions differ (M, N_down, K_down = N/2).
  const char *pass2_mode = nullptr;
  group_matmul_run_parallel_dispatch(
      layout, transA_down, transB, M, fused.N_down, K_down, alpha_down,
      src_down, ldc,  // lda for Op2 = ldc of Op1 (activated stride)
      fused.down_weight, fused.ldb_down,
      fused.bias_down, beta_down,
      fused.dst_down, fused.ldc_down,
      is_weights_const, params_down, num_threads, &pass2_mode,
      grp_matmul_gated_act_t::none, act_dtype);

  // Compose gemm_mode string revealing Op1 and Op2 dispatch modes for
  // profiler/apilog.  Uses a thread_local buffer so the caller's
  // const char* remains valid until the next call on the same thread.
  if (gemm_mode_out != nullptr) {
    static thread_local char mode_buf[96];
    std::snprintf(mode_buf, sizeof(mode_buf),
                  "fused_moe_2pass(op1=%s,op2=%s)",
                  pass1_mode != nullptr ? pass1_mode : "?",
                  pass2_mode != nullptr ? pass2_mode : "?");
    *gemm_mode_out = mode_buf;
  }
  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
