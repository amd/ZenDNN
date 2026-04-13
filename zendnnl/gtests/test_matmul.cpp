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
    po_type = params.po_type;
    use_LOWOHA = params.use_LOWOHA;
    algo = params.algo;
    num_threads = params.num_threads;
    omp_set_num_threads(num_threads);
    log_info("m: ", m, " k: ", k, " n: ", n, " TransA: ", transA, " TransB: ",
             transB, " alpha: ", alpha, " beta: ", beta,
             " postop: ", postOpsToStr(po_type), " algo: ", static_cast<int>(algo),
             " num_threads: ", num_threads);
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}
  uint64_t m,k,n;
  post_op_type_t po_type;
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

  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo,
                            alpha, beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();

  auto output_dtype       = rand() % 2 == 0 ? data_type_t::bf16 :
                            data_type_t::f32;
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0);

  // Run kernel test and reference test
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor,
                            output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha, beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref,
                            po_type, binary_tensor, use_LOWOHA, algo, alpha, beta);

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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();

  auto output_dtype       = rand() % 2 == 0 ? data_type_t::bf16 :
                            data_type_t::f32;
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0);

  // Run kernel test and reference test
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor,
                            output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha, beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref,
                            po_type, binary_tensor, use_LOWOHA, algo, alpha, beta);

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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::bf16, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::bf16, 2.0);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
    log_info("Post-ops/bias are not supported for F16_F16 with AOCL-DLP kernel; disabling post-ops and bias (po_type=none)");
    po_type = post_op_type_t::none;
    disable_bias = true;
  }
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f16, 2.0, transA);
  auto bias_tensor        = disable_bias ? tensor_t() :
                            tensor_factory.uniform_dist_tensor({1, n},
                                data_type_t::f32, 2.0);
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f16, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f16, 2.0);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f32, 2.0);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f32, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f32, 2.0);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::bf16, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::bf16, 2.0);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f16, 2.0);
  auto output_tensor_ref  = tensor_factory.uniform_dist_strided_tensor({m, n},
                            stride_dst, data_type_t::f16, 2.0);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");

  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type, binary_tensor,
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0, false, dst_scale, dst_zp);
  auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
                            output_dtype, 2.0, false, dst_scale, dst_zp);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_type, binary_tensor,
                            use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_type,
                            binary_tensor, use_LOWOHA, algo, 1.0, 0.0);

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
  auto binary_tensor = is_binary_postop(po_type)
                       ? tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0)
                       : tensor_t();
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::bf16, 2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::bf16, 2.0);

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor,
                        bias_tensor, output_tensor, po_type, binary_tensor,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_type,
                        binary_tensor, use_LOWOHA, algo, 1.0, 0.0);
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
  auto binary_tensor = is_binary_postop(po_type)
                       ? tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0)
                       : tensor_t();
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::f32, 2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::f32, 2.0);

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor,
                        bias_tensor, output_tensor, po_type, binary_tensor,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_type,
                        binary_tensor, use_LOWOHA, algo, 1.0, 0.0);
  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                             rtol_f32, epsilon_f32, ok, false, 1.0f);
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
  auto binary_tensor = is_binary_postop(po_type)
                       ? tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0)
                       : tensor_t();
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::bf16, 2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::bf16, 2.0);

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor,
                        bias_tensor, output_tensor, po_type, binary_tensor,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_type,
                        binary_tensor, use_LOWOHA, algo, 1.0, 0.0);
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
  auto binary_tensor = is_binary_postop(po_type)
                       ? tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0)
                       : tensor_t();
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::f32, 2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n},
                           data_type_t::f32, 2.0);

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor,
                        bias_tensor, output_tensor, po_type, binary_tensor,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor,
                        weight_tensor, bias_tensor, output_tensor_ref, po_type,
                        binary_tensor, use_LOWOHA, algo, 1.0, 0.0);
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
  auto binary_tensor  = is_binary_postop(po_type)
                        ? tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0)
                        : tensor_t();
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n}, test_dt,
                           2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n}, test_dt,
                           2.0);

  log_info("INT8_DYNAMIC_GEMM_BF16: ", use_per_group ? "per-group" : "per-token",
           use_per_group ? " group_size=" + std::to_string(group_size) : "",
           " scale_dt=", scale_dt == data_type_t::f32 ? "f32" : "bf16");

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor_s8,
                        bias_tensor, output_tensor, po_type, binary_tensor,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor_ref,
                        weight_tensor_ref, bias_tensor, output_tensor_ref, po_type,
                        binary_tensor, use_LOWOHA, algo, 1.0, 0.0);
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
  auto binary_tensor  = is_binary_postop(po_type)
                        ? tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0)
                        : tensor_t();
  auto output_tensor     = tensor_factory.uniform_dist_tensor({m, n}, test_dt,
                           2.0);
  auto output_tensor_ref = tensor_factory.uniform_dist_tensor({m, n}, test_dt,
                           2.0);

  log_info("INT8_DYNAMIC_GEMM_F32: ", use_per_group ? "per-group" : "per-token",
           use_per_group ? " group_size=" + std::to_string(group_size) : "",
           " scale_dt=", scale_dt == data_type_t::f32 ? "f32" : "bf16");

  status_t status     = matmul_kernel_test(input_tensor, weight_tensor_s8,
                        bias_tensor, output_tensor, po_type, binary_tensor,
                        use_LOWOHA, algo, 1.0, 0.0);
  status_t ref_status = matmul_forced_ref_kernel_test(input_tensor_ref,
                        weight_tensor_ref, bias_tensor, output_tensor_ref, po_type,
                        binary_tensor, use_LOWOHA, algo, 1.0, 0.0);
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

/** @brief TestGroupMatmul is a test class for group_matmul_direct API */
class TestGroupMatmul : public ::testing::TestWithParam<MatmulType> {
 protected:
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
    algo         = params.algo;
    num_threads  = params.num_threads;
    omp_set_num_threads(num_threads);
    // Deterministic: srand(seed) above seeds rand(), so num_ops is
    // reproducible for a given seed across CI runs.
    num_ops = 2 + (rand() % 4);
    log_info("GroupMatmul test: m=", m, " k=", k, " n=", n,
             " transA=", transA, " transB=", transB,
             " alpha=", alpha, " beta=", beta,
             " num_ops=", num_ops, " num_threads=", num_threads);
  }

  virtual void TearDown() {}

  uint64_t m, k, n;
  bool transA, transB;
  tensor_factory_t tensor_factory{};
  float alpha, beta;
  matmul_algo_t algo;
  int32_t num_threads;
  size_t num_ops;
};

/** @fn TEST_P
 *  @param TestGroupMatmul parameterized test class
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate group_matmul_direct F32 kernel support.
 *         Deterministically enables MoE post-op for ~half the parameter
 *         combinations (topk=2, uniform weights), exercising both the plain
 *         parallel path and the fused weighted-reduce path.
 *
 *         Determinism: m, n, k are from the parameterized test data and
 *         num_ops is derived from srand(seed) in SetUp(), so the formula
 *         produces the same enable_moe decision for every CI run.
 */
TEST_P(TestGroupMatmul, F32_F32) {
  const int D = static_cast<int>(n);
  const int num_tokens = static_cast<int>(m);
  const int topk = 2;
  const bool enable_moe = ((m + n + k + num_ops) % 2 == 1)
                           && (num_ops >= static_cast<size_t>(topk));

  std::vector<char> layouts(num_ops, 'r');
  std::vector<bool> transAs(num_ops, transA);
  std::vector<bool> transBs(num_ops, transB);
  std::vector<int> Ms(num_ops, static_cast<int>(m));
  std::vector<int> Ns(num_ops, static_cast<int>(n));
  std::vector<int> Ks(num_ops, static_cast<int>(k));
  std::vector<float> alphas(num_ops, alpha);
  std::vector<float> betas(num_ops, beta);
  std::vector<int> ldas(num_ops);
  std::vector<int> ldbs(num_ops);
  std::vector<int> ldcs(num_ops);
  std::vector<bool> is_weights_consts(num_ops, false);

  std::vector<tensor_t> input_tensors(num_ops);
  std::vector<tensor_t> weight_tensors(num_ops);
  std::vector<tensor_t> bias_tensors(num_ops);
  std::vector<tensor_t> output_tensors(num_ops);
  std::vector<tensor_t> output_tensors_ref(num_ops);

  std::vector<const void *> srcs(num_ops);
  std::vector<const void *> weights_vec(num_ops);
  std::vector<const void *> biases(num_ops);
  std::vector<void *> dsts(num_ops);
  std::vector<matmul_params> params(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    input_tensors[i] = tensor_factory.uniform_dist_tensor({m, k},
                       data_type_t::f32, 2.0, transA);
    weight_tensors[i] = tensor_factory.uniform_dist_tensor({k, n},
                        data_type_t::f32, 2.0, transB);
    bias_tensors[i] = tensor_factory.uniform_dist_tensor({1, n},
                      data_type_t::f32, 2.0);
    output_tensors[i] = tensor_factory.uniform_dist_tensor({m, n},
                        data_type_t::f32, 2.0);
    output_tensors_ref[i] = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);

    ldas[i] = transA ? static_cast<int>(m) : static_cast<int>(k);
    ldbs[i] = transB ? static_cast<int>(k) : static_cast<int>(n);
    ldcs[i] = static_cast<int>(n);

    srcs[i] = input_tensors[i].get_raw_handle_unsafe();
    weights_vec[i] = weight_tensors[i].get_raw_handle_unsafe();
    biases[i] = bias_tensors[i].get_raw_handle_unsafe();
    dsts[i] = output_tensors[i].get_raw_handle_unsafe();

    params[i].dtypes.src = data_type_t::f32;
    params[i].dtypes.wei = data_type_t::f32;
    params[i].dtypes.dst = data_type_t::f32;
    params[i].dtypes.bias = data_type_t::f32;
    params[i].num_threads = num_threads;
  }

  const int num_slots = num_tokens * topk;
  std::vector<float> moe_weights(static_cast<size_t>(num_slots),
                                  1.f / static_cast<float>(topk));
  std::vector<float> moe_output(static_cast<size_t>(num_tokens * D), 0.f);
  std::vector<const void *> moe_row_ptrs_storage(static_cast<size_t>(num_slots));

  group_matmul_moe_postop_params moe;
  group_matmul_moe_postop_params *moe_ptr = nullptr;

  if (enable_moe) {
    moe.num_tokens = num_tokens;
    moe.topk = topk;
    moe.output = moe_output.data();
    moe.ldc_output = D;
    const bool use_skip_weighted = ((m + k + num_ops) % 3 == 0);
    moe.topk_weights = use_skip_weighted ? nullptr : moe_weights.data();
    moe.skip_weighted = use_skip_weighted;

    // Expert assignment: token t's k-th slot uses expert = (t+kk) % num_ops.
    // This distributes slots across ALL experts, not just 0..topk-1.
    for (int t = 0; t < num_tokens; ++t) {
      for (int kk = 0; kk < topk; ++kk) {
        const int slot = t * topk + kk;
        const size_t expert = static_cast<size_t>((t + kk) % num_ops);
        const auto *base = static_cast<const float *>(dsts[expert]);
        moe_row_ptrs_storage[static_cast<size_t>(slot)] =
            base + static_cast<size_t>(t) * static_cast<size_t>(ldcs[expert]);
      }
    }
    moe.row_ptrs = moe_row_ptrs_storage.data();
    moe_ptr = &moe;
    log_info("  MoE post-op enabled: num_tokens=", num_tokens,
             " topk=", topk, " num_slots=", num_slots);
  }

  status_t status = group_matmul_direct(
                      layouts, transAs, transBs,
                      Ms, Ns, Ks, alphas,
                      srcs, ldas,
                      weights_vec, ldbs,
                      biases, betas,
                      dsts, ldcs,
                      is_weights_consts,
                      params,
                      moe_ptr);

  // Reference: run individual matmuls
  post_op_type_t po_type = post_op_type_t::none;
  bool use_LOWOHA = false;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    tensor_t binary_tensor = tensor_factory.zero_tensor({1, 1}, data_type_t::f32);
    ref_status = matmul_forced_ref_kernel_test(input_tensors[i],
                 weight_tensors[i], bias_tensors[i], output_tensors_ref[i],
                 po_type, binary_tensor, use_LOWOHA, algo, alphas[i], betas[i]);
  }

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful && !enable_moe) {
    for (size_t i = 0; i < num_ops && is_test_successful; ++i) {
      compare_tensor_2D_matrix(output_tensors[i], output_tensors_ref[i],
                               m, n, k, rtol_f32, epsilon_f32,
                               is_test_successful, false, alpha);
    }
  }

  if (is_test_successful && enable_moe) {
    const bool sw = moe.skip_weighted;
    for (int t = 0; t < num_tokens; ++t) {
      for (int d = 0; d < D; ++d) {
        float acc = 0.f;
        for (int kk = 0; kk < topk; ++kk) {
          const size_t expert = static_cast<size_t>((t + kk) % num_ops);
          const auto *ref_base = static_cast<const float *>(
              output_tensors_ref[expert].get_raw_handle_unsafe());
          const float val = ref_base[static_cast<size_t>(t) * static_cast<size_t>(n) + d];
          const float w = sw ? 1.f : moe_weights[static_cast<size_t>(t * topk + kk)];
          acc += w * val;
        }
        const float got = moe_output[static_cast<size_t>(t) * static_cast<size_t>(D) + d];
        if (std::abs(acc - got) > 2 * epsilon_f32 + 2 * rtol_f32 * std::abs(acc)) {
          log_error("MoE output mismatch at token=", t, " d=", d,
                    ": expected=", acc, " got=", got);
          is_test_successful = false;
          break;
        }
      }
      if (!is_test_successful) break;
    }
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestGroupMatmul parameterized test class
 *  @param BF16_F32 user-defined name of test
 *  @brief Test to validate group_matmul_direct BF16 input F32 output kernel support.
 *         Randomly enables MoE post-op (FP32 weighted-reduce path).
 */
TEST_P(TestGroupMatmul, BF16_F32) {
  const int D = static_cast<int>(n);
  const int num_tokens = static_cast<int>(m);
  const int topk = 2;
  const bool enable_moe = ((m + n + k + num_ops) % 2 == 1)
                           && (num_ops >= static_cast<size_t>(topk));

  std::vector<char> layouts(num_ops, 'r');
  std::vector<bool> transAs(num_ops, transA);
  std::vector<bool> transBs(num_ops, transB);
  std::vector<int> Ms(num_ops, static_cast<int>(m));
  std::vector<int> Ns(num_ops, static_cast<int>(n));
  std::vector<int> Ks(num_ops, static_cast<int>(k));
  std::vector<float> alphas(num_ops, alpha);
  std::vector<float> betas(num_ops, beta);
  std::vector<int> ldas(num_ops), ldbs(num_ops), ldcs(num_ops);
  std::vector<bool> is_weights_consts(num_ops, false);

  std::vector<tensor_t> input_tensors(num_ops), weight_tensors(num_ops),
      bias_tensors(num_ops), output_tensors(num_ops), output_tensors_ref(num_ops);

  std::vector<const void *> srcs(num_ops), weights_vec(num_ops), biases(num_ops);
  std::vector<void *> dsts(num_ops);
  std::vector<matmul_params> params(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    input_tensors[i] = tensor_factory.uniform_dist_tensor({m, k},
                       data_type_t::bf16, 2.0, transA);
    weight_tensors[i] = tensor_factory.uniform_dist_tensor({k, n},
                        data_type_t::bf16, 2.0, transB);
    bias_tensors[i] = tensor_factory.uniform_dist_tensor({1, n},
                      data_type_t::f32, 2.0);
    output_tensors[i] = tensor_factory.uniform_dist_tensor({m, n},
                        data_type_t::f32, 2.0);
    output_tensors_ref[i] = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::f32, 2.0);

    ldas[i] = transA ? static_cast<int>(m) : static_cast<int>(k);
    ldbs[i] = transB ? static_cast<int>(k) : static_cast<int>(n);
    ldcs[i] = static_cast<int>(n);

    srcs[i] = input_tensors[i].get_raw_handle_unsafe();
    weights_vec[i] = weight_tensors[i].get_raw_handle_unsafe();
    biases[i] = bias_tensors[i].get_raw_handle_unsafe();
    dsts[i] = output_tensors[i].get_raw_handle_unsafe();

    params[i].dtypes.src = data_type_t::bf16;
    params[i].dtypes.wei = data_type_t::bf16;
    params[i].dtypes.dst = data_type_t::f32;
    params[i].dtypes.bias = data_type_t::f32;
    params[i].num_threads = num_threads;
  }

  const int num_slots = num_tokens * topk;
  std::vector<float> moe_weights(static_cast<size_t>(num_slots),
                                  1.f / static_cast<float>(topk));
  std::vector<float> moe_output(static_cast<size_t>(num_tokens * D), 0.f);
  std::vector<const void *> moe_row_ptrs_storage(static_cast<size_t>(num_slots));

  group_matmul_moe_postop_params moe;
  group_matmul_moe_postop_params *moe_ptr = nullptr;

  if (enable_moe) {
    moe.num_tokens = num_tokens;
    moe.topk = topk;
    moe.output = moe_output.data();
    moe.ldc_output = D;
    moe.topk_weights = moe_weights.data();
    moe.skip_weighted = false;
    for (int t = 0; t < num_tokens; ++t) {
      for (int kk = 0; kk < topk; ++kk) {
        const int slot = t * topk + kk;
        const size_t expert = static_cast<size_t>((t + kk) % num_ops);
        const auto *base = static_cast<const float *>(dsts[expert]);
        moe_row_ptrs_storage[static_cast<size_t>(slot)] =
            base + static_cast<size_t>(t) * static_cast<size_t>(ldcs[expert]);
      }
    }
    moe.row_ptrs = moe_row_ptrs_storage.data();
    moe_ptr = &moe;
  }

  status_t status = group_matmul_direct(
      layouts, transAs, transBs, Ms, Ns, Ks, alphas,
      srcs, ldas, weights_vec, ldbs, biases, betas, dsts, ldcs,
      is_weights_consts, params, moe_ptr);

  post_op_type_t po_type = post_op_type_t::none;
  bool use_LOWOHA = false;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    tensor_t binary_tensor = tensor_factory.zero_tensor({1, 1}, data_type_t::f32);
    ref_status = matmul_forced_ref_kernel_test(input_tensors[i],
                 weight_tensors[i], bias_tensors[i], output_tensors_ref[i],
                 po_type, binary_tensor, use_LOWOHA, algo, alphas[i], betas[i]);
  }

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful && !enable_moe) {
    for (size_t i = 0; i < num_ops && is_test_successful; ++i) {
      compare_tensor_2D_matrix(output_tensors[i], output_tensors_ref[i],
                               m, n, k, rtol_f32, epsilon_f32,
                               is_test_successful, false, alpha);
    }
  }

  if (is_test_successful && enable_moe) {
    for (int t = 0; t < num_tokens && is_test_successful; ++t) {
      for (int d = 0; d < D && is_test_successful; ++d) {
        float acc = 0.f;
        for (int kk = 0; kk < topk; ++kk) {
          const size_t expert = static_cast<size_t>((t + kk) % num_ops);
          const auto *ref_base = static_cast<const float *>(
              output_tensors_ref[expert].get_raw_handle_unsafe());
          acc += moe_weights[static_cast<size_t>(t * topk + kk)] *
                 ref_base[static_cast<size_t>(t) * n + d];
        }
        const float got = moe_output[static_cast<size_t>(t) * D + d];
        if (std::abs(acc - got) > 2 * epsilon_f32 + 2 * rtol_f32 * std::abs(acc)) {
          log_error("MoE BF16_F32 mismatch at t=", t, " d=", d,
                    ": expected=", acc, " got=", got);
          is_test_successful = false;
        }
      }
    }
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestGroupMatmul parameterized test class
 *  @param BF16_BF16 user-defined name of test
 *  @brief Test to validate group_matmul_direct BF16 input BF16 output kernel support.
 *         Deterministically enables MoE post-op (BF16 weighted-reduce path)
 *         for ~half the parameter combinations — see F32_F32 comment for details.
 */
TEST_P(TestGroupMatmul, BF16_BF16) {
  const int D = static_cast<int>(n);
  const int num_tokens = static_cast<int>(m);
  const int topk = 2;
  const bool enable_moe = ((m + n + k + num_ops) % 2 == 1)
                           && (num_ops >= static_cast<size_t>(topk));

  std::vector<char> layouts(num_ops, 'r');
  std::vector<bool> transAs(num_ops, transA);
  std::vector<bool> transBs(num_ops, transB);
  std::vector<int> Ms(num_ops, static_cast<int>(m));
  std::vector<int> Ns(num_ops, static_cast<int>(n));
  std::vector<int> Ks(num_ops, static_cast<int>(k));
  std::vector<float> alphas(num_ops, alpha);
  std::vector<float> betas(num_ops, beta);
  std::vector<int> ldas(num_ops), ldbs(num_ops), ldcs(num_ops);
  std::vector<bool> is_weights_consts(num_ops, false);

  std::vector<tensor_t> input_tensors(num_ops), weight_tensors(num_ops),
      bias_tensors(num_ops), output_tensors(num_ops), output_tensors_ref(num_ops);

  std::vector<const void *> srcs(num_ops), weights_vec(num_ops), biases(num_ops);
  std::vector<void *> dsts(num_ops);
  std::vector<matmul_params> params(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    input_tensors[i] = tensor_factory.uniform_dist_tensor({m, k},
                       data_type_t::bf16, 2.0, transA);
    weight_tensors[i] = tensor_factory.uniform_dist_tensor({k, n},
                        data_type_t::bf16, 2.0, transB);
    bias_tensors[i] = tensor_factory.uniform_dist_tensor({1, n},
                      data_type_t::bf16, 2.0);
    output_tensors[i] = tensor_factory.uniform_dist_tensor({m, n},
                        data_type_t::bf16, 2.0);
    output_tensors_ref[i] = tensor_factory.uniform_dist_tensor({m, n},
                            data_type_t::bf16, 2.0);

    ldas[i] = transA ? static_cast<int>(m) : static_cast<int>(k);
    ldbs[i] = transB ? static_cast<int>(k) : static_cast<int>(n);
    ldcs[i] = static_cast<int>(n);

    srcs[i] = input_tensors[i].get_raw_handle_unsafe();
    weights_vec[i] = weight_tensors[i].get_raw_handle_unsafe();
    if (algo == matmul_algo_t::libxsmm ||
        algo == matmul_algo_t::libxsmm_blocked) {
      biases[i] = nullptr;
    } else {
      biases[i] = bias_tensors[i].get_raw_handle_unsafe();
    }
    dsts[i] = output_tensors[i].get_raw_handle_unsafe();

    params[i].dtypes.src = data_type_t::bf16;
    params[i].dtypes.wei = data_type_t::bf16;
    params[i].dtypes.dst = data_type_t::bf16;
    params[i].dtypes.bias = data_type_t::bf16;
    params[i].num_threads = num_threads;
  }

  const int num_slots = num_tokens * topk;
  std::vector<float> moe_weights(static_cast<size_t>(num_slots),
                                  1.f / static_cast<float>(topk));
  std::vector<uint16_t> moe_output(static_cast<size_t>(num_tokens * D), 0);
  std::vector<const void *> moe_row_ptrs_storage(static_cast<size_t>(num_slots));

  group_matmul_moe_postop_params moe;
  group_matmul_moe_postop_params *moe_ptr = nullptr;

  if (enable_moe) {
    moe.num_tokens = num_tokens;
    moe.topk = topk;
    moe.output = moe_output.data();
    moe.ldc_output = D;
    moe.topk_weights = moe_weights.data();
    moe.skip_weighted = false;
    for (int t = 0; t < num_tokens; ++t) {
      for (int kk = 0; kk < topk; ++kk) {
        const int slot = t * topk + kk;
        const size_t expert = static_cast<size_t>((t + kk) % num_ops);
        const auto *base = static_cast<const uint16_t *>(dsts[expert]);
        moe_row_ptrs_storage[static_cast<size_t>(slot)] =
            base + static_cast<size_t>(t) * static_cast<size_t>(ldcs[expert]);
      }
    }
    moe.row_ptrs = moe_row_ptrs_storage.data();
    moe_ptr = &moe;
  }

  status_t status = group_matmul_direct(
      layouts, transAs, transBs, Ms, Ns, Ks, alphas,
      srcs, ldas, weights_vec, ldbs, biases, betas, dsts, ldcs,
      is_weights_consts, params, moe_ptr);

  post_op_type_t po_type = post_op_type_t::none;
  bool use_LOWOHA = false;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    tensor_t binary_tensor = tensor_factory.zero_tensor({1, 1}, data_type_t::bf16);
    ref_status = matmul_forced_ref_kernel_test(input_tensors[i],
                 weight_tensors[i], bias_tensors[i], output_tensors_ref[i],
                 po_type, binary_tensor, use_LOWOHA, algo, alphas[i], betas[i]);
  }

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful && !enable_moe) {
    for (size_t i = 0; i < num_ops && is_test_successful; ++i) {
      compare_tensor_2D_matrix(output_tensors[i], output_tensors_ref[i],
                               m, n, k, rtol_bf16, epsilon_bf16,
                               is_test_successful, false, alpha);
    }
  }

  if (is_test_successful && enable_moe) {
    for (int t = 0; t < num_tokens && is_test_successful; ++t) {
      for (int d = 0; d < D && is_test_successful; ++d) {
        float acc = 0.f;
        for (int kk = 0; kk < topk; ++kk) {
          const size_t expert = static_cast<size_t>((t + kk) % num_ops);
          const auto *ref_base = static_cast<const uint16_t *>(
              output_tensors_ref[expert].get_raw_handle_unsafe());
          const uint16_t raw = ref_base[static_cast<size_t>(t) * n + d];
          acc += moe_weights[static_cast<size_t>(t * topk + kk)] *
                 zendnnl::common::bfloat16_t::bf16_to_f32_val(
                     static_cast<int16_t>(raw));
        }
        const uint16_t got_raw = moe_output[static_cast<size_t>(t) * D + d];
        const float got = zendnnl::common::bfloat16_t::bf16_to_f32_val(
                              static_cast<int16_t>(got_raw));
        if (std::abs(acc - got) > 2 * epsilon_bf16 + 2 * rtol_bf16 * std::abs(acc)) {
          log_error("MoE BF16_BF16 mismatch at t=", t, " d=", d,
                    ": expected=", acc, " got=", got);
          is_test_successful = false;
        }
      }
    }
  }

  EXPECT_TRUE(is_test_successful);
}

// TODO: INT8, WOQ, and dynamic-quant group matmul tests require a dedicated
// test fixture with constrained parameters (no transpose, alpha=1, beta=0,
// aligned K, compatible algos). The shared matmul_test params are not valid
// for quantized group matmul. Add in a follow-up PR with INT8-specific params.

// TODO: Add per-ALGO coverage tests that set ZENDNNL_GRP_MATMUL_ALGO=1..4
// via setenv/putenv before calling group_matmul_direct, to ensure all
// dispatch paths (sequential, per_expert, multilevel, flat_ccd_m_slice)
// are exercised and don't regress. Add in a follow-up PR.

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers group_matmul_direct parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(GroupMatmul, TestGroupMatmul,
                         ::testing::ValuesIn(matmul_test));
