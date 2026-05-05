/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "gtest_utils.hpp"
#include "common/bfloat16.hpp"


/** @brief TestMatmul is a test class to handle parameters */
class TestMatmul : public ::testing::TestWithParam<MatmulType> {
 protected:
  /** @brief SetUp is to initialize test parameters
   *
   *  This method is a standard and is used in googletests to initialize parameters
   *  for each test and also acts as fixutres i.e. handling the common part of
   *  each test.
   *
   * */
  virtual void SetUp() {
    MatmulType params = GetParam();
    srand(static_cast<unsigned int>(seed));
    m            = params.matmul_m;
    k            = params.matmul_k;
    n            = params.matmul_n;
    transA       = params.transA;
    transB       = params.transB;
    alpha        = params.alpha;
    beta         = params.beta;
    source_dtype = params.source_dtype;
    output_dtype = params.output_dtype;
    weight_granularity = params.weight_granularity;
    po_types = params.po_types;
    use_LOWOHA = params.use_LOWOHA;
    algo = params.algo;
    num_threads = params.num_threads;
    omp_set_num_threads(num_threads);
    log_info("m: ", m, " k: ", k, " n: ", n, " TransA: ", transA, " TransB: ",
             transB, " alpha: ", alpha, " beta: ", beta,
             " postops: ", postOpTypesToStr(po_types), " algo: ", static_cast<int>(algo),
             " num_threads: ", num_threads);
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {
    clear_matmul_test_caches();
  }
  uint64_t m,k,n;
  std::vector<post_op_type_t> po_types;
  bool     transA, transB;
  tensor_factory_t tensor_factory{};
  float alpha, beta;
  bool use_LOWOHA;
  data_type_t source_dtype, output_dtype;
  quant_granularity_t weight_granularity;
  matmul_algo_t algo;
  int32_t num_threads;
};

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate matmul F32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul,F32_F32) {
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f32, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f32, 2.0, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n},
                            data_type_t::f32, 2.0);

  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo,
                            alpha, beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);
  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked ||
                                 algo == matmul_algo_t::native_gemm ||
                                 algo == matmul_algo_t::native_brgemm);
  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_f32,
                             epsilon_f32, is_test_successful, enable_f32_relaxation,
                             alpha);

  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param WOQ_BF16_S4 user-defined name of test according to test
 *  @brief Test to validate matmul WOQ (Weight-Only Quantization) with BF16 input
 *         and S4 weights wrt Reference kernel
 */
TEST_P(TestMatmul, WOQ_BF16_S4) {
  if (algo == matmul_algo_t::onednn || algo == matmul_algo_t::onednn_blocked) {
    GTEST_SKIP();
  }
  // Test WOQ with different scale/zp granularity combinations:
  // Combination 0: scale=per-tensor - {1,1}
  // Combination 1: scale=per-channel - {1,n}
  // Combination 2: scale=per-group - {G,n}

  int quant_combo = rand() % 3;

  std::vector<uint64_t> scale_size;
  uint64_t group_size = 0;
  uint64_t num_groups = 1;
  std::string scale_granularity;

  // Calculate group size for per-group cases
  // DLP only supports even group sizes
  // Find all even divisors of K as valid group sizes
  std::vector<uint64_t> valid_group_sizes;
  for (uint64_t gs = 2; gs <= k; gs += 2) {  // Only even numbers
    if (k % gs == 0) {
      valid_group_sizes.push_back(gs);
    }
  }
  // If no even divisors found, fall back to K only if K is even
  // Otherwise, fall back to per-channel (combo 1) for per-group cases
  bool has_valid_group_size = !valid_group_sizes.empty() || (k % 2 == 0);
  if (valid_group_sizes.empty() && k % 2 == 0) {
    valid_group_sizes.push_back(k);
  }
  // If K is odd and no even divisors, downgrade per-group to per-channel
  if (!has_valid_group_size && quant_combo == 2) {
    quant_combo = 1;  // Fall back to per-channel
  }

  switch (quant_combo) {
  case 0:
    // scale=per-tensor
    scale_size = {1, 1};
    scale_granularity = "per-tensor";
    break;

  case 1:
    // scale=per-channel
    scale_size = {1, n};
    scale_granularity = "per-channel";
    break;

  case 2: {
    // scale=per-group
    // Randomly select one of the valid group sizes (safe here since we checked has_valid_group_size)
    uint64_t valid_group_size = valid_group_sizes[rand() %
                                       valid_group_sizes.size()];
    group_size = valid_group_size;
    num_groups = k / group_size;
    scale_size = {num_groups, n};
    scale_granularity = "per-group";
    break;
  }
  }

  auto scale_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
  auto wei_scale = tensor_factory.uniform_dist_tensor(scale_size, scale_dtype,
                   2.0);

  // Log test configuration
  std::string group_info = quant_combo == 2
                           ? " group_size=" + std::to_string(group_size) + " num_groups=" + std::to_string(
                             num_groups)
                           : "";
  log_info("WOQ test: scale=", scale_granularity, "[", scale_size[0], ",",
           scale_size[1], "]",
           " scale_dtype=", (scale_dtype == data_type_t::f32 ? "f32" : "bf16"),
           group_info);

  // Create S4 quantized weight tensor with scale
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::s4, 7.0, transB, wei_scale);

  // BF16 input tensor
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 2.0, transA);

  // Bias tensor (optional, can be BF16 or F32)
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);

  // Binary tensor for post-ops if needed
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});

  auto output_dtype       = rand() % 2 == 0 ? data_type_t::bf16 :
                            data_type_t::f32;
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0);

  // Run kernel test and reference test
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor,
                            output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha, beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref,
                            po_types, binary_tensors, use_LOWOHA, algo, alpha, beta);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                             output_dtype == data_type_t::bf16 ? rtol_bf16 : rtol_woq,
                             output_dtype == data_type_t::bf16 ? epsilon_bf16 : epsilon_woq,
                             is_test_successful, false, alpha, true);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param WOQ_BF16_U4 user-defined name of test according to test
 *  @brief Test to validate matmul WOQ (Weight-Only Quantization) with BF16 input
 *         and U4 weights wrt Reference kernel
 */
TEST_P(TestMatmul, WOQ_BF16_U4) {
  // Test WOQ with different scale/zp granularity combinations:
  // Combination 0: scale=per-tensor,  zp=per-tensor  -> {1,1}, {1,1}
  // Combination 1: scale=per-channel, zp=per-tensor  -> {1,n}, {1,1}
  // Combination 2: scale=per-channel,   zp=per-channel  -> {1,n}, {1,n}
  // Combination 3: scale=per-group,   zp=per-group   -> {G,n}, {G,n}

  int quant_combo = (rand() + k) % 4;
  bool random_zp_domain = (rand() + k) % 2 == 0;

  std::vector<uint64_t> scale_size;
  std::vector<uint64_t> zp_size;
  uint64_t group_size = 0;
  uint64_t num_groups = 1;
  std::string scale_granularity, zp_granularity;

  // Calculate group size for per-group cases
  // DLP only supports even group sizes
  // Find all even divisors of K as valid group sizes
  std::vector<uint64_t> valid_group_sizes;
  for (uint64_t gs = 2; gs <= k; gs += 2) {  // Only even numbers
    if (k % gs == 0) {
      valid_group_sizes.push_back(gs);
    }
  }
  // If no even divisors found, fall back to K only if K is even
  // Otherwise, fall back to per-channel (combo 1) for per-group cases
  bool has_valid_group_size = !valid_group_sizes.empty() || (k % 2 == 0);
  if (valid_group_sizes.empty() && k % 2 == 0) {
    valid_group_sizes.push_back(k);
  }
  // If K is odd and no even divisors, downgrade per-group to per-channel
  if (!has_valid_group_size && (quant_combo == 2 || quant_combo == 3)) {
    quant_combo = 1;  // Fall back to per-channel
  }

  switch (quant_combo) {
  case 0:
    // scale=per-tensor, zp=per-tensor
    scale_size = {1, 1};
    zp_size = {1, 1};
    scale_granularity = "per-tensor";
    zp_granularity = "per-tensor";
    break;

  case 1:
    // scale=per-channel, zp=per-tensor
    scale_size = {1, n};
    zp_size = {1, 1};
    scale_granularity = "per-channel";
    zp_granularity = "per-tensor";
    break;

  case 2:
    // scale=per-channel, zp=per-channel
    scale_size = {1, n};
    zp_size = {1, n};
    scale_granularity = "per-channel";
    zp_granularity = "per-channel";
    break;

  case 3: {
    // scale=per-group, zp=per-group
    // Randomly select one of the valid group sizes (safe here since we checked has_valid_group_size)
    uint64_t valid_group_size = valid_group_sizes[rand() %
                                       valid_group_sizes.size()];
    group_size = valid_group_size;
    num_groups = k / group_size;
    scale_size = {num_groups, n};
    zp_size = {num_groups, n};
    scale_granularity = "per-group";
    zp_granularity = "per-group";
    break;
  }
  }

  auto scale_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
  auto wei_scale = tensor_factory.uniform_dist_tensor(scale_size, scale_dtype,
                   2.0);
  auto wei_zp = tensor_factory.uniform_dist_tensor(zp_size, random_zp_domain ?
                data_type_t::bf16 : data_type_t::s8, random_zp_domain ? 2.0 : 25.0);

  // Log test configuration
  std::string group_info = quant_combo > 2
                           ? " group_size=" + std::to_string(group_size) + " num_groups=" + std::to_string(
                             num_groups)
                           : "";
  log_info("WOQ test: scale=", scale_granularity, "[", scale_size[0], ",",
           scale_size[1], "]",
           " zp=", zp_granularity, "[", zp_size[0], ",", zp_size[1], "]",
           " scale_dtype=", (scale_dtype == data_type_t::f32 ? "f32" : "bf16"),
           group_info);

  // Create U4 quantized weight tensor with scale and zero point
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::u4, 15.0, transB, wei_scale, wei_zp);

  // BF16 input tensor
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 2.0, transA);

  // Bias tensor (optional, can be BF16 or F32)
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);

  // Binary tensor for post-ops if needed
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});

  auto output_dtype       = rand() % 2 == 0 ? data_type_t::bf16 :
                            data_type_t::f32;
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0);

  // Run kernel test and reference test
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor,
                            output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha, beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref,
                            po_types, binary_tensors, use_LOWOHA, algo, alpha, beta);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                             output_dtype == data_type_t::bf16 ? rtol_bf16 : rtol_woq,
                             output_dtype == data_type_t::bf16 ? epsilon_bf16 : epsilon_woq,
                             is_test_successful, false, alpha, true);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param BF16_F32 user-defined name of test according to test
 *  @brief Test to validate matmul BF16outputF32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, BF16_F32) {
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 2.0, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);
  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked ||
                                 algo == matmul_algo_t::native_gemm ||
                                 algo == matmul_algo_t::native_brgemm);
  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_f32,
                             epsilon_f32, is_test_successful, enable_f32_relaxation,
                             alpha);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param BF16_BF16 user-defined name of test according to test
 *  @brief Test to validate matmul BF16outputBF16 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, BF16_BF16) {
  // TODO: Extend support for test cases with a wider range of values.
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 2.0, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::bf16, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::bf16, 2.0);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_bf16,
                             epsilon_bf16, is_test_successful, false, alpha);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param F16_F16 user-defined name of test according to test
 *  @brief Test to validate matmul F16 input/weight/output
 */
TEST_P(TestMatmul, F16_F16) {
  bool disable_bias = false;
  if (algo == matmul_algo_t::aocl_dlp ||
      algo == matmul_algo_t::aocl_dlp_blocked) {
    log_info("Post-ops/bias are not supported for F16_F16 with AOCL-DLP kernel; disabling all post-ops and bias (po_types={none})");
    po_types = {post_op_type_t::none};
    disable_bias = true;
  }
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f16, 2.0, transA);
  auto bias_tensor        = disable_bias ? tensor_t() :
                            tensor_factory.uniform_dist_tensor({1, n},
                                data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f16, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f16, 2.0);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha,
                            beta);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k, rtol_bf16,
                             epsilon_bf16, is_test_successful, false, alpha);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param F16_F32 user-defined name of test according to test
 *  @brief Test to validate matmul F16 input/weight with F32 output
 */
TEST_P(TestMatmul, F16_F32) {
  if (algo == matmul_algo_t::aocl_dlp ||
      algo == matmul_algo_t::aocl_dlp_blocked) {
    GTEST_SKIP() << "F16_F32 is not supported with AOCL-DLP kernel";
  }
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f16, 2.0, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n},
                            data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha,
                            beta);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k, rtol_f32,
                             epsilon_f32, is_test_successful, false, alpha);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmulStride parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate matmul F32 aocl kernel support wrt Reference kernel
 *  with strided tensors.
 *
 */
TEST_P(TestMatmul,F32_F32_Stride) {
  size_t stride_in_inc           = rand() % 50;
  size_t stride_wt_inc           = rand() % 50;
  size_t stride_dst_inc          = rand() % 50;
  std::vector<size_t> stride_in  = {m,k};
  std::vector<size_t> stride_wt  = {k,n};
  std::vector<size_t> stride_dst = {m,n + stride_dst_inc};
  if (transB) {
    stride_wt[0] += stride_wt_inc;
  }
  else {
    stride_wt[1] += stride_wt_inc;
  }
  if (transA) {
    stride_in[0] += stride_in_inc;
  }
  else {
    stride_in[1] += stride_in_inc;
  }
  auto weight_tensor      = tensor_factory.uniform_dist_strided_tensor({k, n},
                            stride_wt, data_type_t::f32, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::f32, 2.0, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n},
                            data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f32, 2.0);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);
  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked ||
                                 algo == matmul_algo_t::native_gemm ||
                                 algo == matmul_algo_t::native_brgemm);
  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_f32,
                             epsilon_f32, is_test_successful, enable_f32_relaxation,
                             alpha);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmulStride parameterized test class to initialize parameters
 *  @param BF16_F32 user-defined name of test
 *  @brief Test to validate matmul BF16 input, F32 output aocl kernel support
 *  wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestMatmul,BF16_F32_Stride) {
  size_t stride_in_inc           = rand() % 50;
  size_t stride_wt_inc           = rand() % 50;
  size_t stride_dst_inc          = rand() % 50;
  std::vector<size_t> stride_in  = {m,k};
  std::vector<size_t> stride_wt  = {k,n};
  std::vector<size_t> stride_dst = {m,n + stride_dst_inc};
  if (transB) {
    stride_wt[0] += stride_wt_inc;
  }
  else {
    stride_wt[1] += stride_wt_inc;
  }
  if (transA) {
    stride_in[0] += stride_in_inc;
  }
  else {
    stride_in[1] += stride_in_inc;
  }
  auto weight_tensor      = tensor_factory.uniform_dist_strided_tensor({k, n},
                            stride_wt, data_type_t::bf16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::bf16, 2.0, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f32, 2.0);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked ||
                                 algo == matmul_algo_t::native_gemm ||
                                 algo == matmul_algo_t::native_brgemm);
  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_f32,
                             epsilon_f32, is_test_successful, enable_f32_relaxation,
                             alpha);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmulStride parameterized test class to initialize parameters
 *  @param BF16_BF16 user-defined name of test
 *  @brief Test to validate matmul BF16 input, BF16 output aocl kernel support
 *  wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestMatmul,BF16_BF16_Stride) {
  size_t stride_in_inc           = rand() % 50;
  size_t stride_wt_inc           = rand() % 50;
  size_t stride_dst_inc          = rand() % 50;
  std::vector<size_t> stride_in  = {m,k};
  std::vector<size_t> stride_wt  = {k,n};
  std::vector<size_t> stride_dst = {m,n + stride_dst_inc};
  if (transB) {
    stride_wt[0] += stride_wt_inc;
  }
  else {
    stride_wt[1] += stride_wt_inc;
  }
  if (transA) {
    stride_in[0] += stride_in_inc;
  }
  else {
    stride_in[1] += stride_in_inc;
  }
  auto weight_tensor      = tensor_factory.uniform_dist_strided_tensor({k, n},
                            stride_wt, data_type_t::bf16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::bf16, 2.0, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::bf16, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::bf16, 2.0);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_bf16,
                             epsilon_bf16, is_test_successful, false, alpha);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmulStride parameterized test class to initialize parameters
 *  @param F16_F16_Stride user-defined name of test
 *  @brief Test to validate matmul F16 input, F16 output kernel support
 *  wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestMatmul, F16_F16_Stride) {
  if (algo == matmul_algo_t::aocl_dlp ||
      algo == matmul_algo_t::aocl_dlp_blocked) {
    GTEST_SKIP() << "F16_F16_Stride is not supported with AOCL-DLP kernel";
  }
  size_t stride_in_inc           = rand() % 50;
  size_t stride_wt_inc           = rand() % 50;
  size_t stride_dst_inc          = rand() % 50;
  std::vector<size_t> stride_in  = {m,k};
  std::vector<size_t> stride_wt  = {k,n};
  std::vector<size_t> stride_dst = {m,n + stride_dst_inc};
  if (transB) {
    stride_wt[0] += stride_wt_inc;
  }
  else {
    stride_wt[1] += stride_wt_inc;
  }
  if (transA) {
    stride_in[0] += stride_in_inc;
  }
  else {
    stride_in[1] += stride_in_inc;
  }
  auto weight_tensor      = tensor_factory.uniform_dist_strided_tensor({k, n},
                            stride_wt, data_type_t::f16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::f16, 2.0, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n},
                            data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f16, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f16, 2.0);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA, algo, alpha,
                            beta);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types, binary_tensors,
                            use_LOWOHA, algo, alpha, beta);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k, rtol_bf16,
                             epsilon_bf16, is_test_successful, false, alpha);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param INT8 user-defined name of test according to test
 *  @brief Test to validate matmul INT8 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, INT8) {
  // TODO: Extend support for test cases with a wider range of values.
  std::vector<uint64_t> wei_scale_size = (weight_granularity ==
                                          quant_granularity_t::tensor) ?
                                         std::vector<uint64_t> {1, 1} :
                                         std::vector<uint64_t> {1, n};

  data_type_t ref_dt = (output_dtype == data_type_t::f32) ? data_type_t::f32
                       : data_type_t::bf16;

  auto wei_ref = tensor_factory.uniform_dist_tensor({k, n}, ref_dt, 25.0, transB);
  tensor_t weight_tensor, wei_scale, wei_zp;
  if (quant_params_compute(tensor_factory, wei_ref,
                           ref_dt, data_type_t::s8,
{static_cast<int64_t>(wei_scale_size[0]), static_cast<int64_t>(wei_scale_size[1])},
  data_type_t::f32, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
    FAIL() << "weight dynamic quantization failed";
  }

  auto src_ref = tensor_factory.uniform_dist_tensor({m, k}, ref_dt, 25.0, transA);
  tensor_t input_tensor, src_scale, src_zp;
  if (quant_params_compute(tensor_factory, src_ref,
                           ref_dt, source_dtype,
{1, 1}, data_type_t::f32, src_scale, src_zp,
&input_tensor) != status_t::success) {
    FAIL() << "source dynamic quantization failed";
  }
  tensor_t dst_scale, dst_zp;
  if (output_dtype != data_type_t::f32 && output_dtype != data_type_t::bf16) {
    auto dst_ref = tensor_factory.uniform_dist_tensor({m, n}, ref_dt, 2.0);
    if (quant_params_compute(tensor_factory, dst_ref,
                             ref_dt, output_dtype,
    {1, 1}, data_type_t::f32, dst_scale, dst_zp) != status_t::success) {
      FAIL() << "destination scale/zp computation failed";
    }
  }
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0, false, dst_scale, dst_zp);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0, false, dst_scale, dst_zp);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_types, binary_tensors,
                            use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_types,
                            binary_tensors, use_LOWOHA, algo, 1.0, 0.0);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k, rtol_bf16,
                             epsilon_bf16, is_test_successful, false, 1.0f, true);
  }


  EXPECT_TRUE(is_test_successful);
}

/** @brief Test INT8 sym_quant: per-group source scale, bf16 output */
TEST_P(TestMatmul, INT8_SYM_QUANT_PER_GROUP_BF16) {
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

  source_dtype = data_type_t::s8;
  use_LOWOHA = true;

  data_type_t ref_dt = data_type_t::bf16;
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  uint64_t num_groups = sym_k / group_size;
  std::vector<int64_t> wei_sd = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), static_cast<int64_t>(num_groups)};

  auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                 transB);
  tensor_t weight_tensor, wei_scale, wei_zp;
  if (quant_params_compute(tensor_factory, wei_ref, ref_dt,
                           data_type_t::s8,
                           wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
    FAIL() << "weight dynamic quantization failed";
  }

  auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                 transA);
  tensor_t input_tensor, src_scale, src_zp;
  if (quant_params_compute(tensor_factory, src_ref, ref_dt,
                           data_type_t::s8,
                           src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
    FAIL() << "source dynamic quantization failed";
  }

  auto bias_tensor   = tensor_factory.uniform_dist_tensor({1, n},
                       rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::bf16, 2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::bf16, 2.0);

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor,
                        bias_tensor, output_tensor, po_types, binary_tensors,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_types,
                        binary_tensors, use_LOWOHA, algo, 1.0, 0.0);
  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                             rtol_bf16, epsilon_bf16, ok, false, 1.0f);
  }
  EXPECT_TRUE(ok);
}

/** @brief Test INT8 sym_quant: per-group source scale, f32 output */
TEST_P(TestMatmul, INT8_SYM_QUANT_PER_GROUP_F32) {
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

  source_dtype = data_type_t::s8;
  use_LOWOHA = true;

  data_type_t ref_dt = data_type_t::f32;
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  uint64_t num_groups = sym_k / group_size;
  std::vector<int64_t> wei_sd = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), static_cast<int64_t>(num_groups)};
  // Keeping the range to 2.0 to avoid accuracy drops in INT8 matmul
  auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 2.0,
                 transB);
  tensor_t weight_tensor, wei_scale, wei_zp;
  if (quant_params_compute(tensor_factory, wei_ref, ref_dt,
                           data_type_t::s8,
                           wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
    FAIL() << "weight dynamic quantization failed";
  }

  auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                 transA);
  tensor_t input_tensor, src_scale, src_zp;
  if (quant_params_compute(tensor_factory, src_ref, ref_dt,
                           data_type_t::s8,
                           src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
    FAIL() << "source dynamic quantization failed";
  }

  auto bias_tensor   = tensor_factory.uniform_dist_tensor({1, n},
                       rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);

  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::f32, 2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::f32, 2.0);

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor,
                        bias_tensor, output_tensor, po_types, binary_tensors,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_types,
                        binary_tensors, use_LOWOHA, algo, 1.0, 0.0);
  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                             rtol_f32, epsilon_f32, ok, false, 1.0f);
  }
  EXPECT_TRUE(ok);
}

/** @brief Test INT8 sym_quant with GGML Q8_0 packed weights.
 *  Combines INT8_SYM_QUANT_PER_GROUP_BF16 and F32 variants:
 *    - Randomly picks bf16 or f32 for both input/output
 *    - K is always a multiple of 32 (GGML Q8_0 group size)
 *    - Weights are repacked into GGML Q8_0 blocked format and passed
 *      with pack_format_b = 1 so the API unpacks them internally
 */
TEST_P(TestMatmul, INT8_PER_GROUP_GGML_PACKED) {
  uint64_t sym_k = (k / 32) * 32;
  if (sym_k == 0) {
    sym_k = 32;
  }

  std::mt19937 local_rng(m ^ k ^ n ^ 0xBB01);
  bool use_bf16 = (local_rng() % 2 == 0);
  data_type_t ref_dt = use_bf16 ? data_type_t::bf16 : data_type_t::f32;
  data_type_t out_dt = ref_dt;

  source_dtype = data_type_t::s8;
  use_LOWOHA = true;

  uint64_t num_groups = sym_k / 32;
  data_type_t scale_dt = data_type_t::bf16;

  std::vector<int64_t> wei_sd = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), static_cast<int64_t>(num_groups)};

  auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 2.0,
                 false);
  tensor_t weight_tensor, wei_scale, wei_zp;
  if (quant_params_compute(tensor_factory, wei_ref, ref_dt,
                           data_type_t::s8, wei_sd, scale_dt,
                           wei_scale, wei_zp, &weight_tensor) != status_t::success) {
    FAIL() << "weight dynamic quantization failed";
  }

  auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                 transA);
  tensor_t input_tensor, src_scale, src_zp;
  if (quant_params_compute(tensor_factory, src_ref, ref_dt,
                           data_type_t::s8, src_sd, scale_dt,
                           src_scale, src_zp, &input_tensor) != status_t::success) {
    FAIL() << "source dynamic quantization failed";
  }

  const int8_t *raw_wt = static_cast<const int8_t *>(
                           weight_tensor.get_raw_handle_unsafe());

  int64_t M_pack = static_cast<int64_t>(n);
  int64_t K_pack = static_cast<int64_t>(sym_k);
  int64_t ng = K_pack / 32;
  size_t num_scales = static_cast<size_t>(num_groups * n);

  const auto *raw_scl_bf16 = static_cast<const uint16_t *>(
                               wei_scale.get_raw_handle_unsafe());
  std::vector<float> scl_f32(num_scales);
  for (size_t i = 0; i < num_scales; i++) {
    uint32_t bits = static_cast<uint32_t>(raw_scl_bf16[i]) << 16;
    std::memcpy(&scl_f32[i], &bits, sizeof(float));
  }

  std::vector<int8_t> wt_nk(sym_k * n);
  for (uint64_t ki = 0; ki < sym_k; ki++) {
    for (uint64_t ni = 0; ni < n; ni++) {
      wt_nk[ni * sym_k + ki] = raw_wt[ki * n + ni];
    }
  }

  size_t packed_size = static_cast<size_t>(M_pack * ng * 34);
  std::vector<uint8_t> packed_buf(packed_size);
  repack_weights_q8_0(wt_nk.data(), scl_f32.data(), M_pack, K_pack,
                      packed_buf.data());

  auto packed_weight_tensor = tensor_factory.copy_tensor({sym_k, n},
                              data_type_t::s8,
                              std::make_pair(packed_size, static_cast<void *>(packed_buf.data())),
                              true, false);

  auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
                     rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});

  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n}, out_dt,
                           2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n}, out_dt,
                           2.0);

  log_info("INT8_SYM_QUANT_GGML_PACKED: dtype=",
           use_bf16 ? "bf16" : "f32",
           " K=", sym_k, " groups=", num_groups);

  status_t status     = matmul_kernel_test(input_tensor, packed_weight_tensor,
                        bias_tensor, output_tensor, po_types, binary_tensors,
                        use_LOWOHA, algo, 1.0, 0.0, 1);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_types,
                        binary_tensors, use_LOWOHA, algo, 1.0, 0.0);

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                             rtol_bf16, epsilon_bf16, ok, false, 1.0f,
                             true);
  }
  EXPECT_TRUE(ok);
}

/** @brief Test INT8 sym_quant: per-token source scale, bf16 output */
TEST_P(TestMatmul, INT8_SYM_QUANT_PER_TOKEN_BF16) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  source_dtype = data_type_t::s8;
  use_LOWOHA = true;

  data_type_t ref_dt = data_type_t::bf16;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xBF17);
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

  auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                 transB);
  tensor_t weight_tensor, wei_scale, wei_zp;
  if (quant_params_compute(tensor_factory, wei_ref, ref_dt,
                           data_type_t::s8,
                           wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
    FAIL() << "weight dynamic quantization failed";
  }

  auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                 transA);
  tensor_t input_tensor, src_scale, src_zp;
  if (quant_params_compute(tensor_factory, src_ref, ref_dt,
                           data_type_t::s8,
                           src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
    FAIL() << "source dynamic quantization failed";
  }

  auto bias_tensor   = tensor_factory.uniform_dist_tensor({1, n},
                       rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::bf16, 2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::bf16, 2.0);

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor,
                        bias_tensor, output_tensor, po_types, binary_tensors,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_types,
                        binary_tensors, use_LOWOHA, algo, 1.0, 0.0);
  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                             rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
  }
  EXPECT_TRUE(ok);
}

/** @brief Test INT8 sym_quant: per-token source scale, f32 output */
TEST_P(TestMatmul, INT8_SYM_QUANT_PER_TOKEN_F32) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  source_dtype = data_type_t::s8;
  use_LOWOHA = true;

  data_type_t ref_dt = data_type_t::f32;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xF321);
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

  auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                 transB);
  tensor_t weight_tensor, wei_scale, wei_zp;
  if (quant_params_compute(tensor_factory, wei_ref, ref_dt,
                           data_type_t::s8,
                           wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
    FAIL() << "weight dynamic quantization failed";
  }

  auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                 transA);
  tensor_t input_tensor, src_scale, src_zp;
  if (quant_params_compute(tensor_factory, src_ref, ref_dt,
                           data_type_t::s8,
                           src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
    FAIL() << "source dynamic quantization failed";
  }

  auto bias_tensor   = tensor_factory.uniform_dist_tensor({1, n},
                       rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::f32, 2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::f32, 2.0);

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor,
                        bias_tensor, output_tensor, po_types, binary_tensors,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_types,
                        binary_tensors, use_LOWOHA, algo, 1.0, 0.0);
  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                             rtol_f32, epsilon_f32, ok, false, 1.0f, true);
  }
  EXPECT_TRUE(ok);
}

/** @brief Test INT8 dynamic GEMM: bf16 src, s8 weight (dynamically quantized), bf16 dst vs bf16 reference */
TEST_P(TestMatmul, INT8_DYNAMIC_GEMM_BF16) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  use_LOWOHA = true;
  data_type_t test_dt = data_type_t::bf16;

  std::mt19937 local_rng(m ^ k ^ n ^ 0xBF18);
  bool use_per_group = (local_rng() % 2 == 0);
  uint64_t group_size = 0;
  uint64_t num_groups = 0;

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
      num_groups = sym_k / group_size;
    }
  }

  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;

  std::vector<int64_t> wei_scale_dims;
  std::vector<uint64_t> src_scale_shape;
  if (use_per_group) {
    wei_scale_dims = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
    src_scale_shape = {m, num_groups};
  }
  else {
    wei_scale_dims = {1, static_cast<int64_t>(n)};
    src_scale_shape = {m, 1};
  }

  auto weight_tensor_ref = tensor_factory.uniform_dist_tensor({sym_k, n},
                           test_dt, 2.0);
  tensor_t weight_tensor_s8, wei_scale, wei_zp;
  if (quant_params_compute(tensor_factory, weight_tensor_ref, test_dt,
                           data_type_t::s8,
                           wei_scale_dims, scale_dt,
                           wei_scale, wei_zp, &weight_tensor_s8) != status_t::success) {
    FAIL() << "weight dynamic quantization failed";
  }

  auto src_scale      = tensor_factory.zero_tensor(src_scale_shape,
                        scale_dt);
  auto input_tensor   = tensor_factory.uniform_dist_tensor({m, sym_k},
                        test_dt, 2.0, transA, src_scale, tensor_t());
  auto input_tensor_ref = tensor_factory.uniform_dist_tensor({m, sym_k},
                          test_dt, 2.0, transA);
  auto bias_tensor    = tensor_factory.uniform_dist_tensor({1, n},
                        rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n}, test_dt,
                           2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n}, test_dt,
                           2.0);

  log_info("INT8_DYNAMIC_GEMM_BF16: ", use_per_group ? "per-group" : "per-token",
           use_per_group ? " group_size=" + std::to_string(group_size) : "",
           " scale_dt=", scale_dt == data_type_t::f32 ? "f32" : "bf16");

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor_s8,
                        bias_tensor, output_tensor, po_types, binary_tensors,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor_ref,
                        weight_tensor_ref, bias_tensor, output_tensor_ref, po_types,
                        binary_tensors, use_LOWOHA, algo, 1.0, 0.0);
  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                             rtol_bf16, 16 *epsilon_bf16, ok, false, 1.0f, true);
  }
  EXPECT_TRUE(ok);
}

/** @brief Test INT8 dynamic GEMM: f32 src, s8 weight (dynamically quantized), f32 dst vs f32 reference */
TEST_P(TestMatmul, INT8_DYNAMIC_GEMM_F32) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  use_LOWOHA = true;
  data_type_t test_dt = data_type_t::f32;

  std::mt19937 local_rng(m ^ k ^ n ^ 0xF322);
  bool use_per_group = (local_rng() % 2 == 0);
  uint64_t group_size = 0;
  uint64_t num_groups = 0;

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
      num_groups = sym_k / group_size;
    }
  }

  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;

  std::vector<int64_t> wei_scale_dims;
  std::vector<uint64_t> src_scale_shape;
  if (use_per_group) {
    wei_scale_dims = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
    src_scale_shape = {m, num_groups};
  }
  else {
    wei_scale_dims = {1, static_cast<int64_t>(n)};
    src_scale_shape = {m, 1};
  }

  auto weight_tensor_ref = tensor_factory.uniform_dist_tensor({sym_k, n},
                           test_dt, 2.0);
  tensor_t weight_tensor_s8, wei_scale, wei_zp;
  if (quant_params_compute(tensor_factory, weight_tensor_ref, test_dt,
                           data_type_t::s8,
                           wei_scale_dims, scale_dt,
                           wei_scale, wei_zp, &weight_tensor_s8) != status_t::success) {
    FAIL() << "weight dynamic quantization failed";
  }

  auto src_scale      = tensor_factory.zero_tensor(src_scale_shape,
                        scale_dt);
  auto input_tensor   = tensor_factory.uniform_dist_tensor({m, sym_k},
                        test_dt, 2.0, transA, src_scale, tensor_t());
  auto input_tensor_ref = tensor_factory.uniform_dist_tensor({m, sym_k},
                          test_dt, 2.0, transA);
  auto bias_tensor    = tensor_factory.uniform_dist_tensor({1, n},
                        rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po_types, {m, n});
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n}, test_dt,
                           2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n}, test_dt,
                           2.0);

  log_info("INT8_DYNAMIC_GEMM_F32: ", use_per_group ? "per-group" : "per-token",
           use_per_group ? " group_size=" + std::to_string(group_size) : "",
           " scale_dt=", scale_dt == data_type_t::f32 ? "f32" : "bf16");

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor_s8,
                        bias_tensor, output_tensor, po_types, binary_tensors,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor_ref,
                        weight_tensor_ref, bias_tensor, output_tensor_ref, po_types,
                        binary_tensors, use_LOWOHA, algo, 1.0, 0.0);
  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    // TODO: Update the tolerace calcuation
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                             rtol_bf16, 18 *epsilon_bf16, ok, false, 1.0f, true);
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param BF16_INT8 user-defined name of test according to test
 *  @brief Test to validate matmul INT8 aocl kernel support wrt Reference kernel
 */
//  TEST_P(TestMatmul, BF16_INT8) {

//   // if (true) {
//   //   GTEST_SKIP();
//   // }
//   output_dtype = data_type_t::bf16;
//   source_dtype = data_type_t::bf16;
//   use_LOWOHA = true;
//   algo = algo != matmul_algo_t::aocl_dlp ? matmul_algo_t::aocl_dlp_blocked : algo;

//   bool is_u8_source       = false;// Currently, only s8 source conversion is supported by DLP
//   auto wei_scale          = tensor_factory.uniform_dist_tensor({1, 1},
//                             data_type_t::f32, 0.6);
//   auto src_scale          = tensor_factory.uniform_tensor({1, 1},
//                             data_type_t::f32, 0.2);
//   auto src_zp             = is_u8_source ? tensor_factory.uniform_tensor({1, 1},
//                             data_type_t::s32, 16) : tensor_t();

//   auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
//                             data_type_t::s8, 25.0, transB, wei_scale);
//   auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
//                             source_dtype, 2.0, transA, src_scale, src_zp);
//   auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::bf16, 1.0);
//   auto binary_tensor      = is_binary_postop(po_type) ?
//                             tensor_factory.uniform_dist_tensor({m, n},
//                                 data_type_t::f32, 2.0) : tensor_t();
//   auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
//                             output_dtype, 2.0, false);
//   auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
//                             output_dtype, 2.0, false);
//   status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
//                             bias_tensor, output_tensor, po_type, binary_tensor,
//                             use_LOWOHA, algo, 1.0, 0.0);
//   status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
//                             weight_tensor, bias_tensor, output_tensor_ref, po_type,
//                             binary_tensor, use_LOWOHA, algo, 1.0, 0.0);

//   bool is_test_successful =
//     (status == status_t::success && ref_status == status_t::success);
//   if (is_test_successful) {
//     compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_bf16,
//                              epsilon_bf16, is_test_successful);
//   }

//   EXPECT_TRUE(is_test_successful);
// }

// /** @fn TEST_P
//  *  @param TestMatmul parameterized test class to initialize Matmul parameters
//  *  @param F32_INT8 user-defined name of test according to test
//  *  @brief Test to validate matmul INT8 aocl kernel support wrt Reference kernel
//  */
//  TEST_P(TestMatmul, F32_INT8) {

//   // if (true) {
//   //   GTEST_SKIP();
//   // }
//   output_dtype = data_type_t::f32;
//   source_dtype = data_type_t::f32;
//   use_LOWOHA = true;
//   algo = algo != matmul_algo_t::aocl_dlp ? matmul_algo_t::aocl_dlp_blocked : algo;

//   bool is_u8_source       = false;// Currently, only s8 source conversion is supported by DLP
//   auto wei_scale          = tensor_factory.uniform_dist_tensor({1, n},
//                             data_type_t::f32, 0.6);
//   auto src_scale          = tensor_factory.uniform_tensor({1, 1},
//                             data_type_t::f32, 0.2);
//   auto src_zp             = is_u8_source ? tensor_factory.uniform_tensor({1, 1},
//                             data_type_t::s32, 16) : tensor_t();

//   auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
//                             data_type_t::s8, 25.0, transB, wei_scale);
//   auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
//                             source_dtype, 2.0, transA, src_scale, src_zp);
//   auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::bf16, 1.0);
//   auto binary_tensor      = is_binary_postop(po_type) ?
//                             tensor_factory.uniform_dist_tensor({m, n},
//                                 data_type_t::f32, 2.0) : tensor_t();
//   auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
//                             output_dtype, 2.0, false);
//   auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
//                             output_dtype, 2.0, false);
//   status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
//                             bias_tensor, output_tensor, po_type, binary_tensor,
//                             use_LOWOHA, algo, 1.0, 0.0);
//   status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
//                             weight_tensor, bias_tensor, output_tensor_ref, po_type,
//                             binary_tensor, use_LOWOHA, algo, 1.0, 0.0);

//   bool is_test_successful =
//     (status == status_t::success && ref_status == status_t::success);
//   if (is_test_successful) {
//     compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_bf16,
//                              epsilon_bf16, is_test_successful);
//   }

//   EXPECT_TRUE(is_test_successful);
// }

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Matmul parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Matmul, TestMatmul,
                         ::testing::ValuesIn(matmul_test));

// Group matmul tests (TestGroupMatmul, TestGatedAct, TestFusedMoE) moved to
// test_group_matmul.cpp to keep this file focused on single-op matmul tests.