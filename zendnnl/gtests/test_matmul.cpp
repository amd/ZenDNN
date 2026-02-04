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
  uint32_t num_threads;
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
                                 algo == matmul_algo_t::libxsmm_blocked);
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
  // Combination 0: scale=per-tensor,  zp=per-tensor  -> {1,1}, {1,1}
  // Combination 1: scale=per-channel, zp=per-tensor  -> {1,n}, {1,1}
  // Combination 2: scale=per-tensor,   zp=per-channel  -> {1,1}, {1,n}
  // Combination 3: scale=per-group,   zp=per-group   -> {G,n}, {G,n}

  int quant_combo = rand() % 4;

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
    // scale=per-tensor, zp=per-channel
    scale_size = {1, 1};
    zp_size = {1, n};
    scale_granularity = "per-tensor";
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
  auto wei_zp = tensor_factory.uniform_tensor(zp_size, data_type_t::s8, 0);

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

  // Create S4 quantized weight tensor with scale and zero point
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::s4, 7.0, transB, wei_scale, wei_zp);

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
                                 algo == matmul_algo_t::libxsmm_blocked);
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
                                 algo == matmul_algo_t::libxsmm_blocked);
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
                                 algo == matmul_algo_t::libxsmm_blocked);
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
  auto wei_scale          = tensor_factory.uniform_dist_tensor(wei_scale_size,
                            data_type_t::f32, 0.2);
  auto src_scale          = tensor_factory.uniform_dist_tensor({1, 1},
                            data_type_t::f32, 0.3);
  auto dst_scale          = !(output_dtype == data_type_t::f32 ||
                              output_dtype == data_type_t::bf16) ? tensor_factory.uniform_dist_tensor({1, 1},
                                  data_type_t::f32, 2) : tensor_t();

  auto src_zp             = (source_dtype == data_type_t::u8) ?
                            tensor_factory.uniform_tensor({1, 1},
                                data_type_t::s32, 16) : tensor_t();
  auto wei_zp             = (weight_granularity == quant_granularity_t::tensor) ?
                            tensor_factory.uniform_tensor({1, 1},
                                data_type_t::s32, 16) : tensor_t();
  auto dst_zp             = (output_dtype == data_type_t::u8) ?
                            tensor_factory.uniform_tensor({1, 1},
                                data_type_t::s32, 53) : tensor_t();
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::s8, 25.0, transB, wei_scale, wei_zp);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            source_dtype, 25.0, transA, src_scale, src_zp);
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
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_bf16,
                             epsilon_bf16, is_test_successful, false, 1.0f);
  }

  EXPECT_TRUE(is_test_successful);
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
