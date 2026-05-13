/********************************************************************************
# * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

/// @file test_quant.cpp
/// @brief Quantized group_matmul gtest section.  Owned:
///
///   [9] TestGroupMatmulQuant - dedicated fixture for quantized group-matmul
///                              (WOQ S4/U4, INT8 per-group / per-token /
///                              dynamic-quant, sym-quant variants).
///                              Driven by `quant_matmul_test`
///                              (`GroupQuantMatmulType` parameter struct
///                              defined in `group_matmul_test_helpers.hpp`).
///
/// Split from `test_group_matmul.cpp` during the gtests folder refactor;
/// see `group_matmul/README.md` for the file layout overview.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "gtest_utils.hpp"
#include "group_matmul_test_helpers.hpp"
#include "moe_test_utils.hpp"

// ============================================================================
// [9] TestGroupMatmulQuant: dedicated fixture for quantized group-matmul
//
// The shared `MatmulType` random grid (alpha, beta in U[0,10]; random
// transA/transB; arbitrary K) drives the quantized kernels into regimes
// where no useful tolerance can be derived for activated comparison --
// silu(g)*u-style outputs inherit an O(sqrt(K)*alpha) noise term that
// swamps any rel/abs bound for large alpha.  Per the long-standing TODO
// at the bottom of this file, the quantized suites move to a dedicated
// fixture with hardcoded alpha=1, beta=0, no-transpose, K aligned to 4,
// and the compatible-algos algo selector baked into
// `GroupQuantMatmulType`.
//
// Within the fixture, alpha / beta / transA / transB are `static
// constexpr` members so the migrated test bodies use them as plain
// member names (no rename needed) and the un-activated comparison
// tolerances scale exactly as before for alpha=1.
// ============================================================================

class TestGroupMatmulQuant
    : public ::testing::TestWithParam<GroupQuantMatmulType> {
 protected:
  virtual void SetUp() {
    GroupQuantMatmulType params = GetParam();
    srand(static_cast<unsigned int>(seed));
    m            = params.matmul_m;
    k            = params.matmul_k;
    n            = params.matmul_n;
    algo         = params.algo;
    num_threads  = params.num_threads;
    source_dtype       = params.source_dtype;
    output_dtype       = params.output_dtype;
    weight_granularity = params.weight_granularity;
    omp_set_num_threads(num_threads);
    num_ops = 2 + (rand() % 4);
    log_info("GroupMatmulQuant test: m=", m, " k=", k, " n=", n,
             " (alpha=1 beta=0 no-transpose) num_ops=", num_ops,
             " num_threads=", num_threads);
  }
  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {
    clear_matmul_test_caches();
  }

  uint64_t m, k, n;
  matmul_algo_t algo{};
  int32_t num_threads{};
  data_type_t source_dtype{};
  data_type_t output_dtype{};
  quant_granularity_t weight_granularity{};
  size_t num_ops{};
  tensor_factory_t tensor_factory{};

  // Pinned by construction ? these are the constraints the TODO calls for.
  // Members rather than locals so the migrated test bodies (which read
  // `alpha`, `beta`, `transA`, `transB`) keep working unchanged.
  static constexpr float alpha = 1.0f;
  static constexpr float beta  = 0.0f;
  static constexpr bool  transA = false;
  static constexpr bool  transB = false;
};

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param WOQ_BF16_S4 user-defined name of test
 *  @brief Test to validate group_matmul_direct weight-only quantization with
 *         BF16 activation and signed 4-bit (S4) weights. Randomly selects
 *         per-tensor, per-channel, or per-group scale granularity and
 *         scale dtype (F32/BF16). Output dtype is randomly BF16 or F32.
 *         Skips oneDNN algo paths.
 */
TEST_P(TestGroupMatmulQuant, WOQ_BF16_S4) {
  if (algo == matmul_algo_t::onednn || algo == matmul_algo_t::onednn_blocked) {
    GTEST_SKIP() << "WOQ_BF16_S4 is not supported for oneDNN-based algorithms.";
  }

  int quant_combo = rand() % 3;

  std::vector<uint64_t> scale_size;
  uint64_t group_size = 0;
  uint64_t num_groups = 1;

  std::vector<uint64_t> valid_group_sizes;
  for (uint64_t gs = 2; gs <= k; gs += 2) {
    if (k % gs == 0) {
      valid_group_sizes.push_back(gs);
    }
  }
  bool has_valid_group_size = !valid_group_sizes.empty() || (k % 2 == 0);
  if (valid_group_sizes.empty() && k % 2 == 0) {
    valid_group_sizes.push_back(k);
  }
  if (!has_valid_group_size && quant_combo == 2) {
    quant_combo = 1;
  }

  switch (quant_combo) {
  case 0:
    scale_size = {1, 1};
    break;
  case 1:
    scale_size = {1, n};
    break;
  case 2: {
    uint64_t gs = valid_group_sizes[rand() % valid_group_sizes.size()];
    group_size = gs;
    num_groups = k / group_size;
    scale_size = {num_groups, n};
    break;
  }
  }

  auto scale_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops), wei_scale(num_ops);

  auto out_dt = rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32;

  for (size_t i = 0; i < num_ops; ++i) {
    wei_scale[i] = tensor_factory.uniform_dist_tensor(scale_size, scale_dtype, 2.0);
    wt[i]      = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::s4, 7.0,
                 transB, wei_scale[i]);
    inp[i]     = tensor_factory.uniform_dist_tensor({m, k}, data_type_t::bf16, 2.0,
                 transA);
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
  }

  // Random gated activation: paired with the WOQ kernel exercises
  // bf16-src + s4-wei + (bf16/f32)-dst dispatch through the gated
  // epilogue.  `pick_random_gated_act` returns `none` when N is odd
  // since gated activations require even N.  Activation must agree
  // across experts (the validator enforces uniform dst dtype, so we
  // pick once outside the loop).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xAC54u);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, alpha,
                    beta, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, alpha, beta);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    // For activated runs apply the scalar gated-activation reference
    // to out_ref in-place (modifies cols [0:N/2]) and compare only the
    // activated half ? cols [N/2:N] are "don't care" per the kernel
    // contract (TestGroupMatmulAlgoCustom comment).  Activated values
    // amplify noise multiplicatively, so use the looser
    // `compare_activated_2D` bound instead of `compare_tensor_2D_matrix`.
    if (act_type != grp_matmul_gated_act_t::none) {
      const float eps = (out_dt == data_type_t::bf16) ? epsilon_bf16
                        : epsilon_woq;
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2, k,
                                             alpha, eps, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, k,
                                 out_dt == data_type_t::bf16 ? rtol_bf16 : rtol_woq,
                                 out_dt == data_type_t::bf16 ? epsilon_bf16 : epsilon_woq,
                                 ok, false, alpha, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param WOQ_BF16_U4 user-defined name of test
 *  @brief Test to validate group_matmul_direct weight-only quantization with
 *         BF16 activation and unsigned 4-bit (U4) weights. Exercises
 *         asymmetric quantization with zero-point support. Randomly selects
 *         per-tensor, per-channel, or per-group scale/zp granularity, zp
 *         dtype (BF16/S8), scale dtype (F32/BF16), and output dtype
 *         (BF16/F32).
 */
TEST_P(TestGroupMatmulQuant, WOQ_BF16_U4) {
  int quant_combo = (rand() + k) % 4;
  bool random_zp_domain = (rand() + k) % 2 == 0;

  std::vector<uint64_t> scale_size, zp_size;
  uint64_t group_size = 0;
  uint64_t num_groups = 1;

  std::vector<uint64_t> valid_group_sizes;
  for (uint64_t gs = 2; gs <= k; gs += 2) {
    if (k % gs == 0) {
      valid_group_sizes.push_back(gs);
    }
  }
  bool has_valid_group_size = !valid_group_sizes.empty() || (k % 2 == 0);
  if (valid_group_sizes.empty() && k % 2 == 0) {
    valid_group_sizes.push_back(k);
  }
  if (!has_valid_group_size && (quant_combo == 2 || quant_combo == 3)) {
    quant_combo = 1;
  }

  switch (quant_combo) {
  case 0:
    scale_size = {1, 1};
    zp_size = {1, 1};
    break;
  case 1:
    scale_size = {1, n};
    zp_size = {1, 1};
    break;
  case 2:
    scale_size = {1, n};
    zp_size = {1, n};
    break;
  case 3: {
    uint64_t gs = valid_group_sizes[rand() % valid_group_sizes.size()];
    group_size = gs;
    num_groups = k / group_size;
    scale_size = {num_groups, n};
    zp_size    = {num_groups, n};
    break;
  }
  }

  auto scale_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
  auto out_dt = rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32;

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops),
      w_scale(num_ops), w_zp(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    w_scale[i] = tensor_factory.uniform_dist_tensor(scale_size, scale_dtype, 2.0);
    w_zp[i]    = tensor_factory.uniform_dist_tensor(zp_size,
                 random_zp_domain ? data_type_t::bf16 : data_type_t::s8,
                 random_zp_domain ? 2.0 : 25.0);
    wt[i]      = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::u4,
                 15.0, transB, w_scale[i], w_zp[i]);
    inp[i]     = tensor_factory.uniform_dist_tensor({m, k}, data_type_t::bf16, 2.0,
                 transA);
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
  }

  // Random gated activation (asymmetric U4 WOQ + gated epilogue).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACE4u);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, alpha,
                    beta, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, alpha, beta);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      const float eps = (out_dt == data_type_t::bf16) ? epsilon_bf16
                        : epsilon_woq;
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2, k,
                                             alpha, eps, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, k,
                                 out_dt == data_type_t::bf16 ? rtol_bf16 : rtol_woq,
                                 out_dt == data_type_t::bf16 ? epsilon_bf16 : epsilon_woq,
                                 ok, false, alpha, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8 user-defined name of test
 *  @brief Test to validate group_matmul_direct INT8 (S8 weight) kernel support.
 *         Uses parameterized source_dtype/output_dtype and weight_granularity
 *         (per-tensor or per-channel) to exercise asymmetric dynamic
 *         quantization paths. Constrains alpha=1, beta=0.
 */
TEST_P(TestGroupMatmulQuant, INT8) {
  std::vector<uint64_t> wei_scale_size = (weight_granularity ==
                                          quant_granularity_t::tensor) ?
                                         std::vector<uint64_t> {1, 1} :
                                         std::vector<uint64_t> {1, n};
  data_type_t ref_dt = (output_dtype == data_type_t::f32) ? data_type_t::f32
                       : data_type_t::bf16;

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({k, n}, ref_dt, 25.0, transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8, {
    static_cast<int64_t>(wei_scale_size[0]),
      static_cast<int64_t>(wei_scale_size[1])
    },
    data_type_t::f32, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }

    auto src_ref = tensor_factory.uniform_dist_tensor({m, k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, source_dtype,
  {1, 1}, data_type_t::f32, src_scale, src_zp,
  &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    tensor_t dst_scale, dst_zp;
    if (output_dtype != data_type_t::f32 && output_dtype != data_type_t::bf16) {
      auto dst_ref = tensor_factory.uniform_dist_tensor({m, n}, ref_dt, 2.0);
      if (quant_params_compute(tensor_factory, dst_ref, ref_dt, output_dtype,
      {1, 1}, data_type_t::f32, dst_scale, dst_zp) != status_t::success) {
        FAIL() << "destination scale/zp computation failed";
      }
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, output_dtype, 2.0,
                 false, dst_scale, dst_zp);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, output_dtype, 2.0,
                 false, dst_scale, dst_zp);
  }

  // Random gated activation ? only when dst is f32/bf16.  Quantized
  // dst dtypes (s8/u8/...) are rejected by the gated_act validator.
  // INT8 GEMMs always run with alpha=1, beta=0 (see kernel call below),
  // so the bound here is informational; real bounding is per-suite.
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACE8u);
  const bool act_dtype_ok = (output_dtype == data_type_t::f32
                             || output_dtype == data_type_t::bf16);
  grp_matmul_gated_act_t act_type = act_dtype_ok
                                    ? moe_test_utils::pick_random_gated_act(
                                        static_cast<uint64_t>(n), act_rng)
                                    : grp_matmul_gated_act_t::none;
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2, k,
                                             1.0f, epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, k,
                                 rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_SYM_QUANT_PER_GROUP_BF16 user-defined name of test
 *  @brief Test to validate group_matmul_direct symmetric INT8 quantization
 *         with per-group scaling and BF16 output. K is rounded to a multiple
 *         of 4 and group_size is randomly selected from valid divisors.
 *         Both source and weight are symmetrically quantized to S8.
 */
TEST_P(TestGroupMatmulQuant, INT8_SYM_QUANT_PER_GROUP_BF16) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  std::vector<uint64_t> valid_gs;
  for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
    if (sym_k % gs == 0) {
      valid_gs.push_back(gs);
    }
  }
  if (valid_gs.empty()) {
    GTEST_SKIP() << "No valid group_size for K=" << sym_k;
  }
  std::mt19937 local_rng(m ^ k ^ n ^ 0xBF16);
  uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];

  data_type_t ref_dt = data_type_t::bf16;
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  uint64_t ngroups = sym_k / group_size;
  std::vector<int64_t> wei_sd = {static_cast<int64_t>(ngroups), static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), static_cast<int64_t>(ngroups)};

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                   transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                             wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                   transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                             src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::bf16, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::bf16, 2.0);
  }

  // Random gated activation paired with the symmetric INT8 + per-group
  // GEMM (dst dtype is bf16, satisfying gated_act's f32/bf16 requirement).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACBEu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f, epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_bf16, epsilon_bf16, ok, false, 1.0f);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_SYM_QUANT_PER_GROUP_F32 user-defined name of test
 *  @brief Test to validate group_matmul_direct symmetric INT8 quantization
 *         with per-group scaling and F32 output. Same quantization setup as
 *         INT8_SYM_QUANT_PER_GROUP_BF16 but accumulates into FP32.
 */
TEST_P(TestGroupMatmulQuant, INT8_SYM_QUANT_PER_GROUP_F32) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  std::vector<uint64_t> valid_gs;
  for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
    if (sym_k % gs == 0) {
      valid_gs.push_back(gs);
    }
  }
  if (valid_gs.empty()) {
    GTEST_SKIP() << "No valid group_size for K=" << sym_k;
  }
  std::mt19937 local_rng(m ^ k ^ n ^ 0xF320);
  uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];

  data_type_t ref_dt = data_type_t::f32;
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  uint64_t ngroups = sym_k / group_size;
  std::vector<int64_t> wei_sd = {static_cast<int64_t>(ngroups), static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), static_cast<int64_t>(ngroups)};

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 2.0,
                   transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                             wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                   transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                             src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
  }

  // Random gated activation paired with the symmetric INT8 + per-group
  // GEMM (dst dtype is f32, satisfying gated_act's f32/bf16 requirement).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACFEu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f, epsilon_f32, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_f32, epsilon_f32, ok, false, 1.0f);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_SYM_QUANT_PER_TOKEN_BF16 user-defined name of test
 *  @brief Test to validate group_matmul_direct symmetric INT8 quantization
 *         with per-token source scaling and per-channel weight scaling,
 *         producing BF16 output. Exercises the token-wise quantization path
 *         used in inference-time activation quantization.
 */
TEST_P(TestGroupMatmulQuant, INT8_SYM_QUANT_PER_TOKEN_BF16) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  data_type_t ref_dt = data_type_t::bf16;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xBF17);
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                   transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                             wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                   transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                             src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::bf16, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::bf16, 2.0);
  }

  // Random gated activation paired with the symmetric INT8 + per-token
  // GEMM (dst dtype is bf16).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACBDu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f, epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_SYM_QUANT_PER_TOKEN_F32 user-defined name of test
 *  @brief Test to validate group_matmul_direct symmetric INT8 quantization
 *         with per-token source scaling and per-channel weight scaling,
 *         producing F32 output. Same quantization setup as
 *         INT8_SYM_QUANT_PER_TOKEN_BF16 but accumulates into FP32.
 */
TEST_P(TestGroupMatmulQuant, INT8_SYM_QUANT_PER_TOKEN_F32) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  data_type_t ref_dt = data_type_t::f32;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xF321);
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                   transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                             wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                   transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                             src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
  }

  // Random gated activation paired with the symmetric INT8 + per-token
  // GEMM (dst dtype is f32).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACFDu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f, epsilon_f32, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_f32, epsilon_f32, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_DYNAMIC_GEMM_BF16 user-defined name of test
 *  @brief Test to validate group_matmul_direct dynamic quantization GEMM path
 *         with BF16 activation, S8 weights, and BF16 output. The source
 *         tensor carries pre-allocated scale buffers for runtime quantization.
 *         Randomly selects per-token or per-group source scaling.
 */
TEST_P(TestGroupMatmulQuant, INT8_DYNAMIC_GEMM_BF16) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  data_type_t test_dt = data_type_t::bf16;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xBF18);
  bool use_per_group = (local_rng() % 2 == 0);
  uint64_t group_size = 0, ngroups = 0;

  if (use_per_group) {
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
      if (sym_k % gs == 0) {
        valid_gs.push_back(gs);
      }
    }
    if (valid_gs.empty()) {
      use_per_group = false;
    }
    else {
      group_size = valid_gs[local_rng() % valid_gs.size()];
      ngroups = sym_k / group_size;
    }
  }

  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_scale_dims;
  std::vector<uint64_t> src_scale_shape;
  if (use_per_group) {
    wei_scale_dims = {static_cast<int64_t>(ngroups), static_cast<int64_t>(n)};
    src_scale_shape = {m, ngroups};
  }
  else {
    wei_scale_dims = {1, static_cast<int64_t>(n)};
    src_scale_shape = {m, 1};
  }

  std::vector<tensor_t> inp(num_ops), inp_ref(num_ops),
      wt_s8(num_ops), wt_ref(num_ops),
      bias(num_ops), out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wt_ref_t = tensor_factory.uniform_dist_tensor({sym_k, n}, test_dt, 2.0);
    tensor_t wt_s8_t, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wt_ref_t, test_dt, data_type_t::s8,
                             wei_scale_dims, scale_dt, wei_scale, wei_zp, &wt_s8_t) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_scale = tensor_factory.zero_tensor(src_scale_shape, scale_dt);
    auto inp_t     = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt, 2.0,
                     transA, src_scale, tensor_t());
    auto inp_ref_t = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt, 2.0,
                     transA);

    inp[i]     = inp_t;
    inp_ref[i] = inp_ref_t;
    wt_s8[i]   = wt_s8_t;
    wt_ref[i]  = wt_ref_t;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
  }

  // Random gated activation paired with the dynamic INT8 quant GEMM
  // (dst dtype is bf16).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xADCBu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt_s8, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp_ref[i], wt_ref[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f,
                                             16 * epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_bf16, 16 * epsilon_bf16, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_DYNAMIC_GEMM_F32 user-defined name of test
 *  @brief Test to validate group_matmul_direct dynamic quantization GEMM path
 *         with F32 activation, S8 weights, and F32 output. Same dynamic
 *         quantization setup as INT8_DYNAMIC_GEMM_BF16 but operates entirely
 *         in FP32 precision.
 */
TEST_P(TestGroupMatmulQuant, INT8_DYNAMIC_GEMM_F32) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  data_type_t test_dt = data_type_t::f32;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xF322);
  bool use_per_group = (local_rng() % 2 == 0);
  uint64_t group_size = 0, ngroups = 0;

  if (use_per_group) {
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
      if (sym_k % gs == 0) {
        valid_gs.push_back(gs);
      }
    }
    if (valid_gs.empty()) {
      use_per_group = false;
    }
    else {
      group_size = valid_gs[local_rng() % valid_gs.size()];
      ngroups = sym_k / group_size;
    }
  }

  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_scale_dims;
  std::vector<uint64_t> src_scale_shape;
  if (use_per_group) {
    wei_scale_dims = {static_cast<int64_t>(ngroups), static_cast<int64_t>(n)};
    src_scale_shape = {m, ngroups};
  }
  else {
    wei_scale_dims = {1, static_cast<int64_t>(n)};
    src_scale_shape = {m, 1};
  }

  std::vector<tensor_t> inp(num_ops), inp_ref(num_ops),
      wt_s8(num_ops), wt_ref(num_ops),
      bias(num_ops), out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wt_ref_t = tensor_factory.uniform_dist_tensor({sym_k, n}, test_dt, 2.0);
    tensor_t wt_s8_t, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wt_ref_t, test_dt, data_type_t::s8,
                             wei_scale_dims, scale_dt, wei_scale, wei_zp, &wt_s8_t) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_scale = tensor_factory.zero_tensor(src_scale_shape, scale_dt);
    auto inp_t     = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt, 2.0,
                     transA, src_scale, tensor_t());
    auto inp_ref_t = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt, 2.0,
                     transA);

    inp[i]     = inp_t;
    inp_ref[i] = inp_ref_t;
    wt_s8[i]   = wt_s8_t;
    wt_ref[i]  = wt_ref_t;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
  }

  // Random gated activation paired with the dynamic INT8 quant GEMM
  // (dst dtype is f32).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xADCFu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt_s8, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp_ref[i], wt_ref[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f,
                                             18 * epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_bf16, 18 * epsilon_bf16, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);

// TODO: Add per-ALGO coverage tests for ZENDNNL_GRP_MATMUL_ALGO=1..5.
// get_grp_matmul_algo() reads the env var on every call (no caching),
// so setenv/putenv can switch algos within a single test process.
// Exercise all dispatch paths (1=sequential, 2=flat_ccd_m_tile,
// 3=flat_ccd_n_tile, 4=multilevel, 5=per_expert). Add in a follow-up PR.
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulQuant, TestGroupMatmulQuant,
                         ::testing::ValuesIn(quant_matmul_test));

