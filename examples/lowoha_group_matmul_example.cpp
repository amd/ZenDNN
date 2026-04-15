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
 ******************************************************************************/

/// Group MatMul examples — FP32, BF16, and MoE weighted-reduce post-op.
///
/// Demonstrates the group_matmul_direct API for:
///   1. FP32 parallel group GEMM (multiple independent matmuls).
///   2. BF16 parallel group GEMM.
///   3. BF16 group GEMM with MoE post-op (weighted-reduce over expert outputs).
///
/// The MoE post-op example simulates a Mixture-of-Experts layer:
///   - 4 experts, topk=2, 8 tokens
///   - Each token is routed to 2 experts with routing weights
///   - After expert GEMMs, the post-op reduces:
///       output[t, d] = Σ_k  weight[t,k] * expert_output[row_ptrs[t*topk+k]][d]

#include "lowoha_group_matmul_example.hpp"

#include <cmath>
#include <vector>

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::matmul;
using zendnnl::interface::testlog_info;
using zendnnl::interface::testlog_error;
using zendnnl::error_handling::exception_t;

// Helper: fill a float buffer with a simple pattern.
static void fill_f32(std::vector<float> &buf, float val) {
  for (auto &x : buf) x = val;
}

// Helper: convert float to BF16 with round-to-nearest-even.
static uint16_t f32_to_bf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

// Helper: convert BF16 to float.
static float bf16_to_f32(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Helper: fill a BF16 buffer with a constant value.
static void fill_bf16(std::vector<uint16_t> &buf, float val) {
  uint16_t bval = f32_to_bf16(val);
  for (auto &x : buf) x = bval;
}

// ---------------------------------------------------------------------------
// Example 1: FP32 parallel group GEMM
// ---------------------------------------------------------------------------

int group_matmul_fp32_example() {
  testlog_info("** group_matmul FP32 example: 3 parallel GEMMs");

  try {
    const int NUM_OPS = 3;
    std::vector<int> Ms = {16, 32, 8};
    std::vector<int> Ns = {64, 32, 128};
    std::vector<int> Ks = {32, 64, 16};

    std::vector<std::vector<float>> src(NUM_OPS), wei(NUM_OPS), dst(NUM_OPS);
    for (int i = 0; i < NUM_OPS; ++i) {
      src[i].resize(Ms[i] * Ks[i]);
      wei[i].resize(Ks[i] * Ns[i]);
      dst[i].resize(Ms[i] * Ns[i], 0.f);
      fill_f32(src[i], 1.f);
      fill_f32(wei[i], 1.f);
    }

    std::vector<char> layouts(NUM_OPS, 'r');
    std::vector<bool> transAs(NUM_OPS, false), transBs(NUM_OPS, false);
    std::vector<float> alphas(NUM_OPS, 1.f), betas(NUM_OPS, 0.f);
    std::vector<bool> wconst(NUM_OPS, false);
    std::vector<int> ldas = Ks, ldbs = Ns, ldcs = Ns;

    std::vector<const void *> sp(NUM_OPS), wp(NUM_OPS);
    std::vector<const void *> bp(NUM_OPS, nullptr);
    std::vector<void *> dp(NUM_OPS);
    for (int i = 0; i < NUM_OPS; ++i) {
      sp[i] = src[i].data();
      wp[i] = wei[i].data();
      dp[i] = dst[i].data();
    }

    std::vector<matmul_params> params(NUM_OPS);
    for (int i = 0; i < NUM_OPS; ++i) {
      params[i].dtypes.src = data_type_t::f32;
      params[i].dtypes.wei = data_type_t::f32;
      params[i].dtypes.dst = data_type_t::f32;
    }

    status_t st = group_matmul_direct(
        layouts, transAs, transBs, Ms, Ns, Ks, alphas,
        sp, ldas, wp, ldbs, bp, betas, dp, ldcs, wconst, params, nullptr);

    if (st != status_t::success) {
      testlog_error("FP32 group_matmul failed");
      return NOT_OK;
    }

    // Verify: C = 1*1*K = K for each element.
    for (int i = 0; i < NUM_OPS; ++i) {
      float expected = static_cast<float>(Ks[i]);
      for (int j = 0; j < Ms[i] * Ns[i]; ++j) {
        if (std::abs(dst[i][j] - expected) > 1e-3f) {
          testlog_error("FP32 verify failed: op=", i, " idx=", j);
          return NOT_OK;
        }
      }
    }
    testlog_info("FP32 group_matmul verified OK.");
  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }
  return OK;
}

// ---------------------------------------------------------------------------
// Example 2: BF16 parallel group GEMM
// ---------------------------------------------------------------------------

int group_matmul_bf16_example() {
  testlog_info("** group_matmul BF16 example: 3 parallel GEMMs");

  try {
    const int NUM_OPS = 3;
    std::vector<int> Ms = {16, 32, 8};
    std::vector<int> Ns = {64, 32, 128};
    std::vector<int> Ks = {32, 64, 16};

    std::vector<std::vector<uint16_t>> src(NUM_OPS), wei(NUM_OPS), dst(NUM_OPS);
    for (int i = 0; i < NUM_OPS; ++i) {
      src[i].resize(Ms[i] * Ks[i]);
      wei[i].resize(Ks[i] * Ns[i]);
      dst[i].resize(Ms[i] * Ns[i], 0);
      fill_bf16(src[i], 1.f);
      fill_bf16(wei[i], 1.f);
    }

    std::vector<char> layouts(NUM_OPS, 'r');
    std::vector<bool> transAs(NUM_OPS, false), transBs(NUM_OPS, false);
    std::vector<float> alphas(NUM_OPS, 1.f), betas(NUM_OPS, 0.f);
    std::vector<bool> wconst(NUM_OPS, true);
    std::vector<int> ldas = Ks, ldbs = Ns, ldcs = Ns;

    std::vector<const void *> sp(NUM_OPS), wp(NUM_OPS);
    std::vector<const void *> bp(NUM_OPS, nullptr);
    std::vector<void *> dp(NUM_OPS);
    for (int i = 0; i < NUM_OPS; ++i) {
      sp[i] = src[i].data();
      wp[i] = wei[i].data();
      dp[i] = dst[i].data();
    }

    std::vector<matmul_params> params(NUM_OPS);
    for (int i = 0; i < NUM_OPS; ++i) {
      params[i].dtypes.src = data_type_t::bf16;
      params[i].dtypes.wei = data_type_t::bf16;
      params[i].dtypes.dst = data_type_t::bf16;
    }

    status_t st = group_matmul_direct(
        layouts, transAs, transBs, Ms, Ns, Ks, alphas,
        sp, ldas, wp, ldbs, bp, betas, dp, ldcs, wconst, params, nullptr);

    if (st != status_t::success) {
      testlog_error("BF16 group_matmul failed");
      return NOT_OK;
    }

    // Verify: each element ≈ K (BF16 precision).
    for (int i = 0; i < NUM_OPS; ++i) {
      float expected = static_cast<float>(Ks[i]);
      for (int j = 0; j < Ms[i] * Ns[i]; ++j) {
        float got = bf16_to_f32(dst[i][j]);
        if (std::abs(got - expected) > expected * 0.05f) {
          testlog_error("BF16 verify failed: op=", i, " idx=", j,
                        " expected=", expected, " got=", got);
          return NOT_OK;
        }
      }
    }
    testlog_info("BF16 group_matmul verified OK.");
  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }
  return OK;
}

// ---------------------------------------------------------------------------
// Example 3: BF16 group GEMM with MoE weighted-reduce post-op
// ---------------------------------------------------------------------------

int group_matmul_moe_postop_example() {
  testlog_info("** group_matmul MoE post-op example: 4 experts, topk=2, 8 tokens");

  try {
    // MoE configuration.
    const int NUM_EXPERTS = 4;
    const int TOPK = 2;
    const int NUM_TOKENS = 8;
    const int K = 32;
    const int N = 64;

    // Each expert processes all tokens (uniform routing for simplicity).
    // In production, M per expert varies based on router decisions.
    const int M_PER_EXPERT = NUM_TOKENS;

    std::vector<int> Ms(NUM_EXPERTS, M_PER_EXPERT);
    std::vector<int> Ns(NUM_EXPERTS, N);
    std::vector<int> Ks(NUM_EXPERTS, K);

    // Allocate expert src/weight/dst buffers (BF16).
    std::vector<std::vector<uint16_t>> src(NUM_EXPERTS), wei(NUM_EXPERTS);
    std::vector<std::vector<uint16_t>> dst_expert(NUM_EXPERTS);
    for (int i = 0; i < NUM_EXPERTS; ++i) {
      src[i].resize(M_PER_EXPERT * K);
      wei[i].resize(K * N);
      dst_expert[i].resize(M_PER_EXPERT * N, 0);
      fill_bf16(src[i], 1.f);
      // Each expert has weight = (i+1) so outputs are distinguishable.
      fill_bf16(wei[i], static_cast<float>(i + 1));
    }

    // group_matmul API vectors.
    std::vector<char> layouts(NUM_EXPERTS, 'r');
    std::vector<bool> transAs(NUM_EXPERTS, false), transBs(NUM_EXPERTS, false);
    std::vector<float> alphas(NUM_EXPERTS, 1.f), betas(NUM_EXPERTS, 0.f);
    std::vector<bool> wconst(NUM_EXPERTS, true);
    std::vector<int> ldas = Ks, ldbs = Ns, ldcs = Ns;

    std::vector<const void *> sp(NUM_EXPERTS), wp(NUM_EXPERTS);
    std::vector<const void *> bp(NUM_EXPERTS, nullptr);
    std::vector<void *> dp(NUM_EXPERTS);
    for (int i = 0; i < NUM_EXPERTS; ++i) {
      sp[i] = src[i].data();
      wp[i] = wei[i].data();
      dp[i] = dst_expert[i].data();
    }

    std::vector<matmul_params> params(NUM_EXPERTS);
    for (int i = 0; i < NUM_EXPERTS; ++i) {
      params[i].dtypes.src = data_type_t::bf16;
      params[i].dtypes.wei = data_type_t::bf16;
      params[i].dtypes.dst = data_type_t::bf16;
    }

    // ── Build the MoE post-op ──
    // Simulate routing: token t goes to experts (t % NUM_EXPERTS) and
    // ((t+1) % NUM_EXPERTS), with equal weights 0.5.

    // row_ptrs: flat array of size num_tokens * topk.
    // row_ptrs[t * topk + k] = pointer to the row in expert dst buffer
    // that contributes to token t's k-th expert slot.
    std::vector<const void *> row_ptrs(NUM_TOKENS * TOPK);
    std::vector<float> topk_weights(NUM_TOKENS * TOPK, 0.5f);

    for (int t = 0; t < NUM_TOKENS; ++t) {
      int expert_0 = t % NUM_EXPERTS;
      int expert_1 = (t + 1) % NUM_EXPERTS;
      // Each expert has M_PER_EXPERT rows. Token t maps to row t.
      row_ptrs[t * TOPK + 0] = static_cast<const uint16_t *>(dp[expert_0])
          + static_cast<size_t>(t) * N;
      row_ptrs[t * TOPK + 1] = static_cast<const uint16_t *>(dp[expert_1])
          + static_cast<size_t>(t) * N;
    }

    // MoE output buffer: [num_tokens, N] in BF16.
    std::vector<uint16_t> moe_output(NUM_TOKENS * N, 0);

    // Fill the post-op struct.
    group_matmul_moe_postop_params moe;
    moe.num_tokens = NUM_TOKENS;
    moe.topk = TOPK;
    moe.output = moe_output.data();
    moe.ldc_output = N;
    moe.topk_weights = topk_weights.data();
    moe.skip_weighted = false;
    moe.row_ptrs = row_ptrs.data();

    // Execute: expert GEMMs + fused MoE weighted-reduce.
    status_t st = group_matmul_direct(
        layouts, transAs, transBs, Ms, Ns, Ks, alphas,
        sp, ldas, wp, ldbs, bp, betas, dp, ldcs, wconst, params, &moe);

    if (st != status_t::success) {
      testlog_error("MoE group_matmul failed");
      return NOT_OK;
    }

    // Verify MoE output.
    // Expert i output: each element = 1.0 * (i+1) * K = K*(i+1).
    // Token t routes to experts (t%4) and ((t+1)%4) with weight 0.5 each.
    // Expected: 0.5 * K * (expert_0+1) + 0.5 * K * (expert_1+1)
    bool ok = true;
    for (int t = 0; t < NUM_TOKENS && ok; ++t) {
      int e0 = t % NUM_EXPERTS;
      int e1 = (t + 1) % NUM_EXPERTS;
      float expected = 0.5f * K * (e0 + 1) + 0.5f * K * (e1 + 1);
      for (int d = 0; d < N && ok; ++d) {
        float got = bf16_to_f32(moe_output[t * N + d]);
        if (std::abs(got - expected) > expected * 0.1f) {
          testlog_error("MoE verify failed: t=", t, " d=", d,
                        " expected=", expected, " got=", got);
          ok = false;
        }
      }
    }

    if (ok) {
      testlog_info("MoE post-op verified OK. Tokens reduced from ",
                   NUM_EXPERTS, " experts with topk=", TOPK);
    } else {
      return NOT_OK;
    }
  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }
  return OK;
}

} // namespace examples
} // namespace zendnnl
