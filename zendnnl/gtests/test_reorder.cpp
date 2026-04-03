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
    use_LOWOHA = params.is_lowoha_test;

    if (!use_LOWOHA) {
      // Initialize regular reorder parameters
      m        = params.mat.matmul_m;
      n        = params.mat.matmul_n;
      k        = params.mat.matmul_k;
      transA   = params.mat.transA;
      transB   = params.mat.transB;
      po_type = params.mat.po_type;
      algo = matmul_algo_t::aocl_dlp;
      inplace_reorder = params.inplace_reorder;
      source_dtype = params.mat.source_dtype;
      num_threads = params.mat.num_threads;
      omp_set_num_threads(num_threads);
      log_info("m: ", m, " k: ", k, " n: ", n," postop: ", postOpsToStr(po_type),
               " reorder: ",
               inplace_reorder ? "In Place" : "Out of Place", " num_threads: ", num_threads);
    }
    else {
      // Initialize LOWOHA params
      lowoha_params = params;
      omp_set_num_threads(lowoha_params.num_threads);
    }
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}

  uint64_t m, k, n;
  bool transA, transB;
  post_op_type_t po_type;
  bool inplace_reorder;
  data_type_t source_dtype;
  bool use_LOWOHA;
  matmul_algo_t algo;
  int32_t num_threads;
  tensor_factory_t tensor_factory{};

  ReorderType lowoha_params;
};

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate Reorder + Matmul F32 aocl kernel support wrt Matmul F32 aocl
 */
TEST_P(TestReorder,F32_F32) {
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::f32, 1.0f);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::f32, 1.0f);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n},
                            data_type_t::f32, 1.0f);
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias_tensor,
                            output_tensor_ref, po_type,
                            binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_type, binary_tensor, use_LOWOHA, algo);

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
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 1.0f);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 1.0f);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 1.0f);
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias_tensor,
                            output_tensor_ref, po_type,
                            binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_type, binary_tensor, use_LOWOHA, algo);

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
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
  auto weights            = tensor_factory.uniform_dist_tensor({k, n},
                            data_type_t::bf16, 1.0f);
  auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
                            data_type_t::bf16, 1.0f);
  auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, rand() % 2
                            == 0 ? data_type_t::bf16 : data_type_t::f32, 1.0f);
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  status_t ref_status     = matmul_kernel_test(input_tensor, weights, bias_tensor,
                            output_tensor_ref, po_type, binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_type, binary_tensor, use_LOWOHA, algo);

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
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
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
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
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
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
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
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias_tensor, output_tensor_ref,
                            po_type, binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_type, binary_tensor, use_LOWOHA, algo);
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
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::f32);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias_tensor, output_tensor_ref,
                            po_type, binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor, output_tensor,
                            po_type, binary_tensor, use_LOWOHA, algo);
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
  if (use_LOWOHA) {
    GTEST_SKIP();
  }
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
  auto binary_tensor      = is_binary_postop(po_type) ?
                            tensor_factory.uniform_dist_tensor({m, n},
                                data_type_t::f32, 1.0f) : tensor_t();
  auto output_tensor_ref  = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);

  log_info("transA:", transA, " transB:", transB, " strided_inp:{", stride_in[0],
           ",", stride_in[1], "} strided_wt:{", stride_wt[0], ",", stride_wt[1],"}");
  status_t ref_status     = matmul_forced_ref_kernel_test(input_tensor, weights,
                            bias_tensor, output_tensor_ref,
                            po_type, binary_tensor, use_LOWOHA, algo);

  void *weights_buff      = nullptr;
  auto [reorder_weights, reorder_status] = reorder_kernel_test(weights,
      inplace_reorder, &weights_buff);
  if (reorder_status == status_t::unimplemented) {
    if (weights_buff) {
      free(weights_buff);
    }
    GTEST_SKIP();
  }
  auto output_tensor      = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  status_t status         = matmul_kernel_test(input_tensor, reorder_weights,
                            bias_tensor,
                            output_tensor,
                            po_type, binary_tensor, use_LOWOHA, algo);
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

//==============================================================================
// LOWOHA Reorder Tests (Quantization/Dequantization/Type Conversion)
//==============================================================================

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_QUANT_DEQUANT user-defined name of test
 *  @brief Round-trip test: BF16 quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, BF16_QUANT_DEQUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = (std::rand() % 2 == 0);
  data_type_t src_dtype = data_type_t::bf16;
  data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, false, true);
  log_info("Quantization: ",
           is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, quant_dtype);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  // ZP constrained to [55, 183] to prevent saturation with min_scale=0.04
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = 55 + (std::abs(zp_ptr[i]) % 129);
    }
  }

  // ---- Step 1: Quantization (BF16 → S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = quant_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params);
  if (quant_status != status_t::success) {
    log_error("Quantization (BF16 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → BF16) using same scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = quant_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> BF16) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original BF16 with dequantized BF16 ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_QUANT_DEQUANT user-defined name of test
 *  @brief Round-trip test: FP32 quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, FP32_QUANT_DEQUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = (std::rand() % 2 == 0);
  data_type_t src_dtype = data_type_t::f32;
  data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, false, true);
  log_info("Quantization: ",
           is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, quant_dtype);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  // ZP constrained to [55, 183] to prevent saturation with min_scale=0.04
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = 55 + (std::abs(zp_ptr[i]) % 129);
    }
  }

  // ---- Step 1: Quantization (FP32 → S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = quant_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params);
  if (quant_status != status_t::success) {
    log_error("Quantization (FP32 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → FP32) using same scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = quant_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> FP32) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original FP32 with dequantized FP32 ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_BF16_CVT user-defined name of test
 *  @brief Round-trip test: FP32 <-> BF16 type conversion without scale/zp
 */
TEST_P(TestReorder, FP32_BF16_CVT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::f32 :
                          data_type_t::bf16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::bf16 :
                          data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, false);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);

  // No scale/zp tensors for simple conversion
  tensor_t scale_tensor;
  tensor_t zp_tensor;

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  ReorderType cmp_params = lowoha_params;
  cmp_params.dst_dtype = data_type_t::bf16;
  bool is_test_successful = true;
  compare_lowoha_reorder_output(bwd_tensor, src_tensor, cmp_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_BF16_CVT_SCALED user-defined name of test
 *  @brief Round-trip test: FP32 <-> BF16 type conversion with scale/zp
 */
TEST_P(TestReorder, FP32_BF16_CVT_SCALED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::f32 :
                          data_type_t::bf16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::bf16 :
                          data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  auto zp_tensor = tensor_factory.uniform_dist_tensor(
                     quant_shape, data_type_t::s32, 64.0f);

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) with scale/zp ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt (scaled) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) with scale/zp ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt (scaled) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, bwd_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

// TODO: Enable strided tests when strided API is supported
// /** @fn TEST_P
//  *  @param TestReorder parameterized test class to initialize parameters
//  *  @param BF16_QUANT_DEQUANT_STRIDED user-defined name of test
//  *  @brief Round-trip test: strided BF16 quantization (S8/U8) and dequantization
//  */
// TEST_P(TestReorder, BF16_QUANT_DEQUANT_STRIDED) {
//   if (!use_LOWOHA) {
//     GTEST_SKIP();
//   }
//   bool is_symmetric = (std::rand() % 2 == 0);
//   data_type_t src_dtype = data_type_t::bf16;
//   data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
//   bool use_row_padding = (std::rand() % 2 == 0);
//   log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, true, true);
//   log_info("Quantization: ",
//            is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

//   std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
//   std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params,
//                                       use_row_padding);
//   std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

//   auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
//                       shape, strided_shape, src_dtype, 2.0f);
//   auto quant_tensor = tensor_factory.zero_tensor(shape, quant_dtype);

//   auto scale_tensor = tensor_factory.uniform_dist_tensor(
//                         quant_shape, data_type_t::f32, 0.2f);
//   float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
//   size_t scale_nelem = scale_tensor.get_nelem();
//   for (size_t i = 0; i < scale_nelem; ++i) {
//     scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
//   }

//   // ZP constrained to [55, 183] to prevent saturation with min_scale=0.04
//   tensor_t zp_tensor;
//   if (!is_symmetric) {
//     zp_tensor = tensor_factory.uniform_dist_tensor(
//                   quant_shape, data_type_t::s32, 128.0f);
//     int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
//     size_t zp_nelem = zp_tensor.get_nelem();
//     for (size_t i = 0; i < zp_nelem; ++i) {
//       zp_ptr[i] = 55 + (std::abs(zp_ptr[i]) % 129);
//     }
//   }

//   lowoha_params.src_dtype = src_dtype;
//   lowoha_params.dst_dtype = quant_dtype;
//   lowoha_params.use_strided_src = true;
//   lowoha_params.lowoha_algo = reorder_algo_t::native;

//   status_t quant_status = lowoha_reorder_kernel_test(
//                             src_tensor, quant_tensor, scale_tensor, zp_tensor,
//                             lowoha_params);
//   if (quant_status != status_t::success) {
//     log_error("Quantization (strided) failed");
//     EXPECT_TRUE(false);
//     return;
//   }

//   auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

//   ReorderType dequant_params = lowoha_params;
//   dequant_params.src_dtype = quant_dtype;
//   dequant_params.dst_dtype = src_dtype;
//   dequant_params.use_strided_src = false;
//   dequant_params.lowoha_algo = reorder_algo_t::native;

//   status_t dequant_status = lowoha_reorder_kernel_test(
//                               quant_tensor, dequant_tensor, scale_tensor,
//                               zp_tensor, dequant_params);
//   if (dequant_status != status_t::success) {
//     log_error("Dequantization (strided) failed");
//     EXPECT_TRUE(false);
//     return;
//   }

//   bool is_test_successful = true;
//   compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
//                               lowoha_params, is_test_successful);
//   EXPECT_TRUE(is_test_successful);
// }

// /** @fn TEST_P
//  *  @param TestReorder parameterized test class to initialize parameters
//  *  @param FP32_QUANT_DEQUANT_STRIDED user-defined name of test
//  *  @brief Round-trip test: strided FP32 quantization (S8/U8) and dequantization
//  */
// TEST_P(TestReorder, FP32_QUANT_DEQUANT_STRIDED) {
//   if (!use_LOWOHA) {
//     GTEST_SKIP();
//   }
//   bool is_symmetric = (std::rand() % 2 == 0);
//   data_type_t src_dtype = data_type_t::f32;
//   data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
//   bool use_row_padding = (std::rand() % 2 == 0);
//   log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, true, true);
//   log_info("Quantization: ",
//            is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

//   std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
//   std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params,
//                                       use_row_padding);
//   std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

//   auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
//                       shape, strided_shape, src_dtype, 2.0f);
//   auto quant_tensor = tensor_factory.zero_tensor(shape, quant_dtype);

//   auto scale_tensor = tensor_factory.uniform_dist_tensor(
//                         quant_shape, data_type_t::f32, 0.2f);
//   float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
//   size_t scale_nelem = scale_tensor.get_nelem();
//   for (size_t i = 0; i < scale_nelem; ++i) {
//     scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
//   }

//   // ZP constrained to [55, 183] to prevent saturation with min_scale=0.04
//   tensor_t zp_tensor;
//   if (!is_symmetric) {
//     zp_tensor = tensor_factory.uniform_dist_tensor(
//                   quant_shape, data_type_t::s32, 128.0f);
//     int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
//     size_t zp_nelem = zp_tensor.get_nelem();
//     for (size_t i = 0; i < zp_nelem; ++i) {
//       zp_ptr[i] = 55 + (std::abs(zp_ptr[i]) % 129);
//     }
//   }

//   // ---- Step 1: Quantization (FP32 → S8/U8) with strided source ----
//   lowoha_params.src_dtype = src_dtype;
//   lowoha_params.dst_dtype = quant_dtype;
//   lowoha_params.use_strided_src = true;
//   lowoha_params.lowoha_algo = reorder_algo_t::native;

//   status_t quant_status = lowoha_reorder_kernel_test(
//                             src_tensor, quant_tensor, scale_tensor, zp_tensor,
//                             lowoha_params);
//   if (quant_status != status_t::success) {
//     log_error("Quantization (strided) failed");
//     EXPECT_TRUE(false);
//     return;
//   }

//   // ---- Step 2: Dequantization (S8/U8 → FP32) using same scale/zp ----
//   auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

//   ReorderType dequant_params = lowoha_params;
//   dequant_params.src_dtype = quant_dtype;
//   dequant_params.dst_dtype = src_dtype;
//   dequant_params.use_strided_src = false;
//   dequant_params.lowoha_algo = reorder_algo_t::native;

//   status_t dequant_status = lowoha_reorder_kernel_test(
//                               quant_tensor, dequant_tensor, scale_tensor,
//                               zp_tensor, dequant_params);
//   if (dequant_status != status_t::success) {
//     log_error("Dequantization (strided) failed");
//     EXPECT_TRUE(false);
//     return;
//   }

//   // ---- Step 3: Compare original vs dequantized output ----
//   bool is_test_successful = true;
//   compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
//                               lowoha_params, is_test_successful);
//   EXPECT_TRUE(is_test_successful);
// }

// /** @fn TEST_P
//  *  @param TestReorder parameterized test class to initialize parameters
//  *  @param FP32_BF16_CVT_STRIDED user-defined name of test
//  *  @brief Round-trip test: strided FP32 <-> BF16 conversion with scale/zp
//  */
// TEST_P(TestReorder, FP32_BF16_CVT_STRIDED) {
//   if (!use_LOWOHA) {
//     GTEST_SKIP();
//   }
//   data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::f32 :
//                           data_type_t::bf16;
//   data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::bf16 :
//                           data_type_t::f32;
//   bool use_row_padding = (std::rand() % 2 == 0);
//   log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, true, true);

//   std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
//   std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params,
//                                       use_row_padding);
//   std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

//   auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
//                       shape, strided_shape, src_dtype, 2.0f);

//   auto scale_tensor = tensor_factory.uniform_dist_tensor(
//                         quant_shape, data_type_t::f32, 0.2f);
//   float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
//   size_t scale_nelem = scale_tensor.get_nelem();
//   for (size_t i = 0; i < scale_nelem; ++i) {
//     scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
//   }

//   auto zp_tensor = tensor_factory.uniform_dist_tensor(
//                      quant_shape, data_type_t::s32, 64.0f);

//   // ---- Step 1: Forward conversion (src_dtype → dst_dtype) with strided src ----
//   auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

//   lowoha_params.src_dtype = src_dtype;
//   lowoha_params.dst_dtype = dst_dtype;
//   lowoha_params.use_strided_src = true;
//   lowoha_params.lowoha_algo = reorder_algo_t::native;

//   status_t fwd_status = lowoha_reorder_kernel_test(
//                           src_tensor, fwd_tensor, scale_tensor, zp_tensor,
//                           lowoha_params);
//   if (fwd_status != status_t::success) {
//     log_error("Forward cvt (strided) failed");
//     EXPECT_TRUE(false);
//     return;
//   }

//   // ---- Step 2: Backward conversion (dst_dtype → src_dtype) ----
//   auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

//   ReorderType bwd_params = lowoha_params;
//   bwd_params.src_dtype = dst_dtype;
//   bwd_params.dst_dtype = src_dtype;
//   bwd_params.use_strided_src = false;
//   bwd_params.lowoha_algo = reorder_algo_t::native;

//   status_t bwd_status = lowoha_reorder_kernel_test(
//                           fwd_tensor, bwd_tensor, scale_tensor,
//                           zp_tensor, bwd_params);
//   if (bwd_status != status_t::success) {
//     log_error("Backward cvt (strided) failed");
//     EXPECT_TRUE(false);
//     return;
//   }

//   // ---- Step 3: Compare original vs round-trip output ----
//   bool is_test_successful = true;
//   compare_lowoha_quant_output(src_tensor, bwd_tensor, scale_tensor,
//                               lowoha_params, is_test_successful);
//   EXPECT_TRUE(is_test_successful);
// }

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_DYN_QUANT user-defined name of test
 *  @brief Round-trip test: FP32 dynamic quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, FP32_DYN_QUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = (std::rand() % 2 == 0);
  data_type_t src_dtype = data_type_t::f32;
  data_type_t compute_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, compute_dtype, false, true);
  log_info("Dynamic Quantization: ",
           is_symmetric ? "Symmetric (S8)" : "Asymmetric (U8)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, compute_dtype);

  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f32);
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::s32);
  }

  // ---- Step 1: Dynamic Quantization (F32 → S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = compute_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params, /*dynamic_quant=*/true);
  if (quant_status != status_t::success) {
    log_error("Dynamic quantization (F32 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → F32) using computed scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = compute_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8", " -> F32) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original F32 with dequantized F32 ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_DYN_QUANT user-defined name of test
 *  @brief Round-trip test: BF16 dynamic quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, BF16_DYN_QUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = (std::rand() % 2 == 0);
  data_type_t src_dtype = data_type_t::bf16;
  data_type_t compute_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, compute_dtype, false, true);
  log_info("Dynamic Quantization: ",
           is_symmetric ? "Symmetric (S8)" : "Asymmetric (U8)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, compute_dtype);

  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f32);
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::s32);
  }

  // ---- Step 1: Dynamic Quantization (BF16 → S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = compute_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params, /*dynamic_quant=*/true);
  if (quant_status != status_t::success) {
    log_error("Dynamic quantization (BF16 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → BF16) using computed scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = compute_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> BF16) failed");
    EXPECT_TRUE(false);
    return;
  }

  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Reorder parameterized test suite (includes both regular and LOWOHA tests)
 */
INSTANTIATE_TEST_SUITE_P(Reorder, TestReorder,
                         ::testing::ValuesIn(reorder_test));
