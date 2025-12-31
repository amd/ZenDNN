/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
class TestBatchMatmul : public ::testing::TestWithParam<BatchMatmulType> {
protected:
  /** @brief SetUp is to initialize test parameters
   *
   *  This method is a standard and is used in googletests to initialize parameters
   *  for each test and also acts as fixutres i.e. handling the common part of
   *  each test.
   *
   * */
  virtual void SetUp() {
    BatchMatmulType params = GetParam();
    batch_size = params.batch_size;
    m          = params.mat.matmul_m;
    n          = params.mat.matmul_n;
    k          = params.mat.matmul_k;
    transA     = params.mat.transA;
    transB     = params.mat.transB;
    alpha      = params.mat.alpha;
    beta       = params.mat.beta;
    if (!cmd_post_op.empty()) {
      auto it = find_if(po_arr.begin(), po_arr.end(),
      [&](const std::pair<std::string, post_op_type_t> &po) {
        return po.first == cmd_post_op;
      });
      po_index = it != po_arr.end() ? distance(po_arr.begin(), it) : po_size;
    }
    else {
      po_index = params.mat.po_index;
    }
    algo = params.mat.algo;
    use_LOWOHA = params.mat.use_LOWOHA;
    if (algo == matmul_algo_t::aocl_dlp_blocked) {
      algo = matmul_algo_t::aocl_dlp;
    }
    log_info("batch_size: ",batch_size, " m: ",m, " k: ",k, " n: ", n, " TransA: ",
             transA, " TransB: ", transB, " po_index: ",po_index, " algo: ",
             static_cast<int>(algo));
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}
  uint64_t batch_size;
  uint64_t m, n, k;
  uint32_t po_index;
  bool     transA, transB;
  tensor_factory_t tensor_factory{};
  float alpha, beta;
  bool use_LOWOHA;
  matmul_algo_t algo;
};

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param 4D_INVALID user-defined name of test
 *  @brief Test to validate there is no support to 4D batchmatmul in AOCL
 */
TEST_P(TestBatchMatmul,4D_INVALID) {
  //INPUT {GC,MB,M,K}
  auto dummy_group_count  = 1U;
  auto input_tensor       = tensor_factory.uniform_dist_tensor({dummy_group_count, batch_size, m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {GC,MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({dummy_group_count, batch_size, k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {GC,MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({dummy_group_count, batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_failed = (status != status_t::success);
  EXPECT_TRUE(is_test_failed);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param OUTPUT_LESS_THAN_3D_INVALID user-defined name of test
 *  @brief Test to validate there is no support to batchmatmul output size < 3
 */
TEST_P(TestBatchMatmul,OUTPUT_LESS_THAN_3D_INVALID) {
  //INPUT {M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {M,N}
  auto output_tensor      = tensor_factory.zero_tensor({m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_failed = (status != status_t::success);
  EXPECT_TRUE(is_test_failed);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param OUTPUT_ONLY_3D user-defined name of test
 *  @brief Test to validate that there is no support for batchmatmul when
 *  input and weights both have 2-dimesions but output has 3-dimensions.
 */
TEST_P(TestBatchMatmul,OUTPUT_ONLY_3D_INVALID) {
  //INPUT {M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_failed = (status != status_t::success);
  EXPECT_TRUE(is_test_failed);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param INPUT_LESS_THAN_2D_INVALID user-defined name of test
 *  @brief Test to validate there is no support to batchmatmul input size < 2
 */
TEST_P(TestBatchMatmul,INPUT_LESS_THAN_2D_INVALID) {
  //INPUT {K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({k},
                            data_type_t::f32, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_failed = (status != status_t::success);
  EXPECT_TRUE(is_test_failed);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param WEIGHT_LESS_THAN_2D_INVALID user-defined name of test
 *  @brief Test to validate there is no support to batchmatmul weight size < 2
 */
TEST_P(TestBatchMatmul,WEIGHT_LESS_THAN_2D_INVALID) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_failed = (status != status_t::success);
  EXPECT_TRUE(is_test_failed);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param DIFFERENT_BATCH_INVALID user-defined name of test
 *  @brief Test to validate there is no support for different batch-size of batchmatmul
 *  of inputVSouput and weightVSoutput
 */
TEST_P(TestBatchMatmul,DIFFERENT_BATCH_INVALID) {
  int add_bs_inp          = rand()%2;
  int add_bs_wei          = 1 - add_bs_inp;
  //INPUT {MB1,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size+add_bs_inp, m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {MB2,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size+add_bs_wei, k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_failed = (status != status_t::success);
  EXPECT_TRUE(is_test_failed);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param DIFFERENT_ROW_INVALID user-defined name of test
 *  @brief Test to validate there is no support for different rows of batchmatmul
 *  between input and ouput
 */
TEST_P(TestBatchMatmul,DIFFERENT_ROW_INVALID) {
  int add_row_inp          = 1 + rand()%10;
  //INPUT {MB,M1,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m+add_row_inp, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_failed = (status != status_t::success);
  EXPECT_TRUE(is_test_failed);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param DIFFERENT_COL_INVALID user-defined name of test
 *  @brief Test to validate there is no support for different col of batchmatmul
 *  between weight and ouput
 */
TEST_P(TestBatchMatmul,DIFFERENT_COL_INVALID) {
  int add_col_wei          = 1 + rand()%10;
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {MB,K,N1}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n + add_col_wei},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_failed = (status != status_t::success);
  EXPECT_TRUE(is_test_failed);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param F32_3D user-defined name of test
 *  @brief Test to validate batchmatmul F32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestBatchMatmul,F32_3D) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);

  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked);

  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_f32, epsilon_f32,
                             is_test_successful, enable_f32_relaxation);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param F32_2D_WEI user-defined name of test
 *  @brief Test to validate batchmatmul(2D Weight) F32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestBatchMatmul,F32_2D_WEI) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);

  bool is_test_failed = (status != status_t::success);
  if (is_test_failed && (algo == matmul_algo_t::onednn ||
                         algo == matmul_algo_t::onednn_blocked)) {
    EXPECT_TRUE(is_test_failed);
    return;
  }

  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);


  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked);
  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_f32, epsilon_f32,
                             is_test_successful, enable_f32_relaxation);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param F32_2D_INP user-defined name of test
 *  @brief Test to validate batchmatmul(2D Input) F32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestBatchMatmul,F32_2D_INP) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f32, 2.0, transA);
  //WEI {K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::f32, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n},
                            data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);

  bool is_test_failed = (status != status_t::success);
  if (is_test_failed && (algo == matmul_algo_t::onednn ||
                         algo == matmul_algo_t::onednn_blocked)) {
    EXPECT_TRUE(is_test_failed);
    return;
  }

  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);

  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked);
  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_f32, epsilon_f32,
                             is_test_successful, enable_f32_relaxation);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param BF16_F32_3D user-defined name of test
 *  @brief Test to validate batchmatmul BF16 aocl kernel support wrt Reference kernel
 *  @todo: BF16 comparison logic and reference correctness to handle BF16 tensor
 */
TEST_P(TestBatchMatmul,BF16_F32_3D) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m, k},
                            data_type_t::bf16, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::bf16, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n}, rand() %
                            2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);

  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked);
  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_f32, epsilon_f32,
                             is_test_successful, enable_f32_relaxation);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param BF16_F32_2D_WEI user-defined name of test
 *  @brief Test to validate batchmatmul(2D Weight) BF16 aocl kernel support wrt Reference kernel
 *  @todo: BF16 comparison logic and reference correctness to handle BF16 tensor
 */
TEST_P(TestBatchMatmul,BF16_F32_2D_WEI) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m, k},
                            data_type_t::bf16, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n}, rand() %
                            2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);

  bool is_test_failed = (status != status_t::success);
  if (is_test_failed && (algo == matmul_algo_t::onednn ||
                         algo == matmul_algo_t::onednn_blocked)) {
    EXPECT_TRUE(is_test_failed);
    return;
  }

  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);

  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked);
  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_f32, epsilon_f32,
                             is_test_successful, enable_f32_relaxation);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param BF16_F32_2D_INP user-defined name of test
 *  @brief Test to validate batchmatmul(2D Input) BF16 aocl kernel support wrt Reference kernel
 *  @todo: BF16 comparison logic and reference correctness to handle BF16 tensor
 */
TEST_P(TestBatchMatmul,BF16_F32_2D_INP) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::bf16, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n}, rand() %
                            2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);

  bool is_test_failed = (status != status_t::success);
  if (is_test_failed && (algo == matmul_algo_t::onednn ||
                         algo == matmul_algo_t::onednn_blocked)) {
    EXPECT_TRUE(is_test_failed);
    return;
  }

  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);

  bool enable_f32_relaxation  = (algo == matmul_algo_t::libxsmm ||
                                 algo == matmul_algo_t::libxsmm_blocked);
  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_f32, epsilon_f32,
                             is_test_successful, enable_f32_relaxation);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param BF16_BF16_3D user-defined name of test
 *  @brief Test to validate batchmatmul BF16 aocl kernel support wrt Reference kernel
 *  @todo: BF16 comparison logic and reference correctness to handle BF16 tensor
 */
TEST_P(TestBatchMatmul,BF16_BF16_3D) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m, k},
                            data_type_t::bf16, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::bf16, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n}, rand() %
                            2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::bf16);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::bf16);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);
  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_bf16, epsilon_bf16,
                             is_test_successful);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param BF16_BF16_2D_WEI user-defined name of test
 *  @brief Test to validate batchmatmul(2D Weight) BF16 aocl kernel support wrt Reference kernel
 *  @todo: BF16 comparison logic and reference correctness to handle BF16 tensor
 */
TEST_P(TestBatchMatmul,BF16_BF16_2D_WEI) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({batch_size, m, k},
                            data_type_t::bf16, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n}, rand() %
                            2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::bf16);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::bf16);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);

  bool is_test_failed = (status != status_t::success);
  if (is_test_failed && (algo == matmul_algo_t::onednn ||
                         algo == matmul_algo_t::onednn_blocked)) {
    EXPECT_TRUE(is_test_failed);
    return;
  }

  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);
  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_bf16, epsilon_bf16,
                             is_test_successful);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestBatchMatmul parameterized test class to initialize parameters
 *  @param BF16_BF16_2D_INP user-defined name of test
 *  @brief Test to validate batchmatmul(2D Input) BF16 aocl kernel support wrt Reference kernel
 *  @todo: BF16 comparison logic and reference correctness to handle BF16 tensor
 */
TEST_P(TestBatchMatmul,BF16_BF16_2D_INP) {
  //INPUT {MB,M,K}
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 2.0, transA);
  //WEI {MB,K,N}
  auto weight_tensor      = tensor_factory.uniform_dist_tensor({batch_size, k, n},
                            data_type_t::bf16, 2.0, transB);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, 1, n}, rand() %
                            2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
  //Binary Tensor {}
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ? tensor_factory.uniform_dist_tensor({m, n},
                                 data_type_t::f32, 2.0) : tensor_t();
  //OUTPUT {MB,M,N}
  auto output_tensor      = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::bf16);
  auto output_tensor_ref  = tensor_factory.zero_tensor({batch_size, m, n},
                            data_type_t::bf16);
  status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
                            bias_tensor, output_tensor, po_index, binary_tensor, use_LOWOHA, algo, alpha,
                            beta);

  bool is_test_failed = (status != status_t::success);
  if (is_test_failed && (algo == matmul_algo_t::onednn ||
                         algo == matmul_algo_t::onednn_blocked)) {
    EXPECT_TRUE(is_test_failed);
    return;
  }

  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor,
                            weight_tensor, bias_tensor, output_tensor_ref, po_index, binary_tensor,
                            use_LOWOHA, algo, alpha,
                            beta);
  bool is_test_successful = (status == status_t::success &&
                             ref_status == status_t::success);
  if (is_test_successful) {
    compare_tensor_3D_matrix(output_tensor, output_tensor_ref, batch_size,
                             m, n, k, rtol_bf16, epsilon_bf16,
                             is_test_successful);
  }
  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Matmul parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(BatchMatmul, TestBatchMatmul,
                         ::testing::ValuesIn(batchmatmul_test));
