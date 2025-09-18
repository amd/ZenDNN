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
class TestReorder : public ::testing::TestWithParam<ReorderType> {
 protected:
  /** @brief SetUp is to initialize test parameters
   *
   *  This method is a standard and is used in googletests to initialize parameters
   *  for each test and also acts as fixutres i.e. handling the common part of
   *  each test.
   *
   * */
  virtual void SetUp() {
    ReorderType params = GetParam();
    m        = params.mat.matmul_m;
    n        = params.mat.matmul_n;
    k        = params.mat.matmul_k;
    transA   = params.mat.transA;
    transB   = params.mat.transB;
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
    inplace_reorder = params.inplace_reorder;
    use_LOWOHA = 0; // TODO: Enable LOWOHA support
    source_dtype = params.mat.source_dtype;
    algo = params.mat.algo;
    log_info("m: ",m, " k: ",k, " n: ",n," po_index: ",po_index, " reorder: ",
             inplace_reorder ? "In Place" : "Out of Place");
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}
  uint64_t m, k, n;
  bool transA, transB;
  uint32_t po_index;
  bool inplace_reorder;
  data_type_t source_dtype;
  bool use_LOWOHA;
  matmul_algo_t algo;
  tensor_factory_t tensor_factory{};
};

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate Reorder + Matmul F32 aocl kernel support wrt Matmul F32 aocl
 */
TEST_P(TestReorder,F32_F32) {
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f32, 1.0f);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f32, 1.0f);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n},
                            data_type_t::f32, 1.0f);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias_tensor,
                            output_tensor_ref, po_index,
                            binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    GTEST_SKIP();
    if (weights_buff) {
      free(weights_buff);
    }
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_index, binary_tensor, use_LOWOHA, algo);

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
                            data_type_t::bf16, 1.0f);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 1.0f);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 1.0f);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias_tensor,
                            output_tensor_ref, po_index,
                            binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    GTEST_SKIP();
    if (weights_buff) {
      free(weights_buff);
    }
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_index, binary_tensor, use_LOWOHA, algo);

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
                            data_type_t::bf16, 1.0f);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 1.0f);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 1.0f);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias_tensor,
                            output_tensor_ref, po_index, binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    GTEST_SKIP();
    if (weights_buff) {
      free(weights_buff);
    }
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_index, binary_tensor, use_LOWOHA, algo);

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
 *  @param F32 user-defined name of test according to test
 *  @brief Test to validate Reorder Contiguous to Blocked and vice-versa
 *  with aocl kernel.
 */
TEST_P(TestReorder, F32) {
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f32, 2.0f);
  auto weights_ref        = tensor_factory.zero_tensor({k, n}, data_type_t::f32);
  void *weights_ptr       = weights.get_raw_handle_unsafe();
  void *weights_ptr_ref   = weights_ref.get_raw_handle_unsafe();

  std::memcpy(weights_ptr_ref, weights_ptr, k * n * sizeof(float));
  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff, source_dtype);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }

  void *weights_buffer    = nullptr;
  auto [unreorder_weights,
        unreorder_status] = reorder_kernel_test(reorder_weights, inplace_reorder,
                            &weights_buffer, source_dtype);
  if (unreorder_status == status_t::unimplemented) {
    if (weights_buffer) {
      free(weights_buffer);
    }
    GTEST_SKIP();
  }

  bool is_test_successful =
    (reorder_status == status_t::success && unreorder_status == status_t::success);

  compare_tensor_2D(unreorder_weights, weights_ref, k, n, REORDER_TOL,
                    is_test_successful);
  EXPECT_TRUE(is_test_successful);

  if (weights_buff) {
    free(weights_buff);
  }
  if (weights_buffer) {
    free(weights_buffer);
  }
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16 user-defined name of test according to test
 *  @brief Test to validate Reorder Contiguous to Blocked and vice-versa
 *  with aocl kernel.
 */
TEST_P(TestReorder, BF16) {
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 2.0f);
  auto weights_ref        = tensor_factory.zero_tensor({k, n}, data_type_t::bf16);
  void *weights_ptr       = weights.get_raw_handle_unsafe();
  void *weights_ptr_ref   = weights_ref.get_raw_handle_unsafe();

  std::memcpy(weights_ptr_ref, weights_ptr, k * n * sizeof(int16_t));
  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff, source_dtype);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }

  void *weights_buffer    = nullptr;
  auto [unreorder_weights,
        unreorder_status] = reorder_kernel_test(reorder_weights, inplace_reorder,
                            &weights_buffer, source_dtype);
  if (unreorder_status == status_t::unimplemented) {
    if (weights_buffer) {
      free(weights_buffer);
    }
    GTEST_SKIP();
  }

  bool is_test_successful =
    (reorder_status == status_t::success && unreorder_status == status_t::success);

  compare_tensor_2D(unreorder_weights, weights_ref, k, n, REORDER_TOL,
                    is_test_successful);
  EXPECT_TRUE(is_test_successful);

  if (weights_buff) {
    free(weights_buff);
  }
  if (weights_buffer) {
    free(weights_buffer);
  }
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param S8 user-defined name of test according to test
 *  @brief Test to validate Reorder Contiguous to Blocked and vice-versa
 *  with aocl kernel.
 */
TEST_P(TestReorder, S8) {
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::s8, 2.0f);
  auto weights_ref        = tensor_factory.zero_tensor({k, n}, data_type_t::s8);
  void *weights_ptr       = weights.get_raw_handle_unsafe();
  void *weights_ptr_ref   = weights_ref.get_raw_handle_unsafe();

  std::memcpy(weights_ptr_ref, weights_ptr, k * n * sizeof(int8_t));
  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff, source_dtype);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }

  void *weights_buffer    = nullptr;
  auto [unreorder_weights,
        unreorder_status] = reorder_kernel_test(reorder_weights, inplace_reorder,
                            &weights_buffer, source_dtype);
  if (unreorder_status == status_t::unimplemented) {
    if (weights_buffer) {
      free(weights_buffer);
    }
    GTEST_SKIP();
  }

  bool is_test_successful =
    (reorder_status == status_t::success && unreorder_status == status_t::success);

  compare_tensor_2D(unreorder_weights, weights_ref, k, n, REORDER_TOL,
                    is_test_successful);
  EXPECT_TRUE(is_test_successful);

  if (weights_buff) {
    free(weights_buff);
  }
  if (weights_buffer) {
    free(weights_buffer);
  }
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F32_F32_Stride user-defined name of test
 *  @brief Test to validate Reorder + strided matmul F32 aocl kernel support
 *  wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestReorder, F32_F32_Stride) {
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
                            stride_wt, data_type_t::f32, 1.0f, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::f32, 1.0f, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n},
                            data_type_t::f32, 1.0f);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias_tensor, output_tensor_ref,
                            po_index, binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    GTEST_SKIP();
    if (weights_buff) {
      free(weights_buff);
    }
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_index, binary_tensor, use_LOWOHA, algo);
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
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_F32_Stride user-defined name of test
 *  @brief Test to validate Reorder + strided matmul BF16 input, F32 output
 *  aocl kernel support wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestReorder, BF16_F32_Stride) {
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
                            stride_wt, data_type_t::bf16, 1.0f, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::bf16, 1.0f, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 1.0f);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias_tensor, output_tensor_ref,
                            po_index, binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    GTEST_SKIP();
    if (weights_buff) {
      free(weights_buff);
    }
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_index, binary_tensor, use_LOWOHA, algo);
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
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_BF16_Stride user-defined name of test
 *  @brief Test to validate Reorder + strided matmul BF16 input, BF16 output
 *  aocl kernel support wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestReorder, BF16_BF16_Stride) {
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
                            stride_wt, data_type_t::bf16, 1.0f, transB);
  auto input_tensor       = tensor_factory.uniform_dist_strided_tensor({m, k},
                            stride_in, data_type_t::bf16, 1.0f, transA);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 1.0f);
  auto binary_tensor      = (po_index < po_arr.size() &&
                             is_binary_postop(po_arr[po_index].first)) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias_tensor, output_tensor_ref,
                            po_index, binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    GTEST_SKIP();
    if (weights_buff) {
      free(weights_buff);
    }
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor,
                            output_tensor,
                            po_index, binary_tensor, use_LOWOHA, algo);
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

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Reorder parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Reorder, TestReorder,
                         ::testing::ValuesIn(reorder_test));
