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


/** @brief TestReorder is a test class to handle parameters */
class TestReorder : public ::testing::TestWithParam<MatmulType> {
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
    m = params.matmul_m;
    k = params.matmul_k;
    n = params.matmul_n;
    transA   = params.transA;
    transB   = params.transB;
    if (gtest_argc >= 3) {
      auto it = find_if(po_arr.begin(), po_arr.end(),
      [&](const std::pair<std::string, post_op_type_t> &po) {
        return po.first == gtest_argv[2];
      });
      po_index = it != po_arr.end() ? distance(po_arr.begin(), it) : po_size;
    }
    else {
      po_index = params.po_index;
    }
    inplace_reorder = rand() % 2;
    log_info("m: ",m, " k: ",k, " n: ",n," po_index: ",po_index, " reorder: ",
             inplace_reorder ? "In Place" : "Out of Place");
    bias     = tensor_factory.uniform_dist_tensor({n}, rand() % 2 == 0 ?
               data_type_t::bf16 : data_type_t::f32, 2.0);
    bias.set_name("bias");
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}
  uint64_t m,k,n;
  bool transA, transB;
  uint32_t po_index;
  bool inplace_reorder;
  tensor_factory_t tensor_factory{};
  tensor_t bias;
};

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate Reorder + Matmul F32 aocl kernel support wrt Matmul F32 aocl
 */
TEST_P(TestReorder,F32_F32) {
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f32, 2.0);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f32, 2.0);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias,
                            output_tensor_ref, po_index,
                            binary_tensor);
  void *weights_buff      = nullptr;

  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias,
                            output_tensor,
                            po_index, binary_tensor);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_F32_TOL,
                      is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);

  if (weights_buff) {
    free(weights_buff);
  }
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize Matmul parameters
 *  @param BF16_F32 user-defined name of test according to test
 *  @brief Test to validate Reorder + Matmul BF16(Inp, Wei) F32(Out)
 *  aocl kernel support wrt Matmul aocl
 */
TEST_P(TestReorder, BF16_F32) {
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 2.0);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 2.0);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias,
                            output_tensor_ref, po_index,
                            binary_tensor);
  void *weights_buff      = nullptr;

  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias,
                            output_tensor,
                            po_index, binary_tensor);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_F32_TOL,
                      is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);

  if (weights_buff) {
    free(weights_buff);
  }
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize Matmul parameters
 *  @param BF16_BF16 user-defined name of test according to test
 *  @brief Test to validate Reorder + Matmul BF16(Inp, Wei) BF16(Out)
 *  aocl kernel support wrt Matmul aocl
 */
TEST_P(TestReorder, BF16_BF16) {
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 2.0);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 2.0);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias,
                            output_tensor_ref, po_index,
                            binary_tensor);
  void *weights_buff      = nullptr;

  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias,
                            output_tensor,
                            po_index,
                            binary_tensor);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_BF16_TOL,
                      is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);

  if (weights_buff) {
    free(weights_buff);
  }
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate Reorder + strided matmul F32 aocl kernel support
 *  wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestReorder,F32_F32_Stride) {
  size_t stride_in_inc          = rand() % 50;
  size_t stride_wt_inc          = rand() % 50;
  std::vector<size_t> stride_in = {m,k};
  std::vector<size_t> stride_wt = {k,n};
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
  auto weights            = tensor_factory.uniform_dist_strided_tensor({k, n},
                            stride_wt, data_type_t::f32, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::f32, 2.0, transA);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias,
                            output_tensor_ref,
                            po_index, binary_tensor);
  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder,&weights_buff);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias,
                            output_tensor,
                            po_index,
                            binary_tensor);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_F32_TOL,
                      is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_F32 user-defined name of test
 *  @brief Test to validate Reorder + strided matmul BF16 input, F32 output
 *  aocl kernel support wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestReorder,BF16_F32_Stride) {
  size_t stride_in_inc          = rand() % 50;
  size_t stride_wt_inc          = rand() % 50;
  std::vector<size_t> stride_in = {m,k};
  std::vector<size_t> stride_wt = {k,n};
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
  auto weights            = tensor_factory.uniform_dist_strided_tensor({k, n},
                            stride_wt, data_type_t::bf16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::bf16, 2.0, transA);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias,
                            output_tensor_ref,
                            po_index, binary_tensor);
  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias,
                            output_tensor,
                            po_index,
                            binary_tensor);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_F32_TOL,
                      is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_BF16 user-defined name of test
 *  @brief Test to validate Reorder + strided matmul BF16 input, BF16 output
 *  aocl kernel support wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestReorder,BF16_BF16_Stride) {
  size_t stride_in_inc          = rand() % 50;
  size_t stride_wt_inc          = rand() % 50;
  std::vector<size_t> stride_in = {m,k};
  std::vector<size_t> stride_wt = {k,n};
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
  auto weights            = tensor_factory.uniform_dist_strided_tensor({k, n},
                            stride_wt, data_type_t::bf16, 2.0, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::bf16, 2.0, transA);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 2.0) : tensor_t();
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias,
                            output_tensor_ref,
                            po_index, binary_tensor);
  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias,
                            output_tensor,
                            po_index,
                            binary_tensor);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_BF16_TOL,
                      is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Reorder parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Reorder, TestReorder,
                         ::testing::ValuesIn(matmul_test));
