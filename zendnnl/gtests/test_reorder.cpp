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


/** @brief TestReorder is a test class to handle parameters for both
 *  regular reorder and LOWOHA reorder (quantization/dequantization) tests */
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
      log_info("m: ", m, " k: ", k, " n: ", n," postop: ", postOpsToStr(po_type),
               " reorder: ",
               inplace_reorder ? "In Place" : "Out of Place", " num_threads: ", num_threads);
    }
    else {
      // Initialize LOWOHA params (always available since all params are LOWOHA)
      lowoha_params = params;
      omp_set_num_threads(lowoha_params.num_threads);
      log_info("M: ", lowoha_params.M, " N: ", lowoha_params.N,
               " batch: ", lowoha_params.batch,
               " src_dtype: ", dtype_info(lowoha_params.src_dtype),
               " dst_dtype: ", dtype_info(lowoha_params.dst_dtype),
               " granularity: ", static_cast<int>(lowoha_params.granularity),
               " num_groups: ", lowoha_params.num_groups,
               " use_strided_src: ", lowoha_params.use_strided_src,
               " lowoha_algo: ", static_cast<int>(lowoha_params.lowoha_algo),
               " num_threads: ", lowoha_params.num_threads);
    }
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}

  // Regular reorder parameters
  uint64_t m, k, n;
  bool transA, transB;
  post_op_type_t po_type;
  bool inplace_reorder;
  data_type_t source_dtype;
  bool use_LOWOHA;
  matmul_algo_t algo;
  uint32_t num_threads;
  tensor_factory_t tensor_factory{};

  // LOWOHA-specific parameters (always initialized since all params have LOWOHA data)
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
  omp_set_num_threads(num_threads);
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
  omp_set_num_threads(num_threads);
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
  omp_set_num_threads(num_threads);
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
  omp_set_num_threads(num_threads);
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
  omp_set_num_threads(num_threads);
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
  omp_set_num_threads(num_threads);
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
  omp_set_num_threads(num_threads);
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
  omp_set_num_threads(num_threads);
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
  omp_set_num_threads(num_threads);
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
 *  @param TestReorder parameterized test class
 *  @param BF16_QUANT test name
 *  @brief Test to validate LOWOHA BF16 to S8/U8 quantization.
 */
TEST_P(TestReorder, BF16_QUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = data_type_t::bf16;
  data_type_t dst_dtype = (std::rand() % 2 == 0) ? data_type_t::s8 :
                          data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  // Create source tensor
  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 100.0f);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // Create scale tensor
  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 10.0f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.01f + std::fabs(scale_ptr[i]);
  }

  // Create zero-point tensor
  tensor_t zp_tensor;
  if (dst_dtype == data_type_t::u8) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = std::abs(zp_ptr[i]);
    }
  }
  else {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 64.0f);
  }

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param FP32_QUANT test name
 *  @brief Test to validate LOWOHA FP32 to S8/U8 quantization.
 */
TEST_P(TestReorder, FP32_QUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = data_type_t::f32;
  data_type_t dst_dtype = (std::rand() % 2 == 0) ? data_type_t::s8 :
                          data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  // Create source tensor
  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 100.0f);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // Create scale tensor
  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 10.0f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.01f + std::fabs(scale_ptr[i]);
  }

  // Create zero-point tensor
  tensor_t zp_tensor;
  if (dst_dtype == data_type_t::u8) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = std::abs(zp_ptr[i]);
    }
  }
  else {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 64.0f);
  }

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param BF16_DEQUANT test name
 *  @brief Test to validate LOWOHA S8/U8 to BF16 dequantization.
 */
TEST_P(TestReorder, BF16_DEQUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::s8 :
                          data_type_t::u8;
  data_type_t dst_dtype = data_type_t::bf16;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  // Create source tensor
  float src_range = (src_dtype == data_type_t::s8) ? 127.0f : 255.0f;
  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype,
                    src_range);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // Create scale tensor
  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 10.0f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.01f + std::fabs(scale_ptr[i]);
  }

  // Create zero-point tensor
  tensor_t zp_tensor;
  if (src_dtype == data_type_t::u8) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = std::abs(zp_ptr[i]);
    }
  }
  else {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 64.0f);
  }

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param FP32_DEQUANT test name
 *  @brief Test to validate LOWOHA S8/U8 to FP32 dequantization.
 */
TEST_P(TestReorder, FP32_DEQUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::s8 :
                          data_type_t::u8;
  data_type_t dst_dtype = data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  // Create source tensor
  float src_range = (src_dtype == data_type_t::s8) ? 127.0f : 255.0f;
  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype,
                    src_range);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // Create scale tensor
  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 10.0f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.01f + std::fabs(scale_ptr[i]);
  }

  // Create zero-point tensor
  tensor_t zp_tensor;
  if (src_dtype == data_type_t::u8) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = std::abs(zp_ptr[i]);
    }
  }
  else {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 64.0f);
  }

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param FP32_BF16_CONV test name
 *  @brief Test to validate LOWOHA FP32 <-> BF16 type conversion without scale/zp.
 */
TEST_P(TestReorder, FP32_BF16_CONV) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::f32 :
                          data_type_t::bf16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::bf16 :
                          data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, false);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);

  // Create source tensor
  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 100.0f);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // No scale/zp tensors for simple conversion
  tensor_t scale_tensor;
  tensor_t zp_tensor;

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param FP32_BF16_CONV_SCALED test name
 *  @brief Test to validate LOWOHA FP32 <-> BF16 type conversion with scale/zp.
 */
TEST_P(TestReorder, FP32_BF16_CONV_SCALED) {
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

  // Create source tensor
  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 100.0f);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // Create scale tensor
  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 10.0f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.01f + std::fabs(scale_ptr[i]);
  }

  // Create zero-point tensor
  auto zp_tensor = tensor_factory.uniform_dist_tensor(
                     quant_shape, data_type_t::s32, 64.0f);

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param QUANT_STRIDED test name
 *  @brief Test to validate LOWOHA quantization with strided source memory.
 */
TEST_P(TestReorder, QUANT_STRIDED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::bf16 :
                          data_type_t::f32;
  data_type_t dst_dtype = (std::rand() % 2 == 0) ? data_type_t::s8 :
                          data_type_t::u8;
  bool use_row_padding = (std::rand() % 2 == 0);
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, true, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params,
                                      use_row_padding);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  // Create strided source tensor
  auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
                      shape, strided_shape, src_dtype, 100.0f);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // Create scale tensor
  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 10.0f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.01f + std::fabs(scale_ptr[i]);
  }

  // Create zero-point tensor
  tensor_t zp_tensor;
  if (dst_dtype == data_type_t::u8) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = std::abs(zp_ptr[i]);
    }
  }
  else {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 64.0f);
  }

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = true;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param DEQUANT_STRIDED test name
 *  @brief Test to validate LOWOHA dequantization with strided source memory.
 */
TEST_P(TestReorder, DEQUANT_STRIDED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::s8 :
                          data_type_t::u8;
  data_type_t dst_dtype = (std::rand() % 2 == 0) ? data_type_t::bf16 :
                          data_type_t::f32;
  bool use_row_padding = (std::rand() % 2 == 0);
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, true, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params,
                                      use_row_padding);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  // Create strided source tensor
  float src_range = (src_dtype == data_type_t::s8) ? 127.0f : 255.0f;
  auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
                      shape, strided_shape, src_dtype, src_range);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // Create scale tensor
  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 10.0f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.01f + std::fabs(scale_ptr[i]);
  }

  // Create zero-point tensor
  tensor_t zp_tensor;
  if (src_dtype == data_type_t::u8) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = std::abs(zp_ptr[i]);
    }
  }
  else {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 64.0f);
  }

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = true;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param FP32_BF16_CONV_STRIDED test name
 *  @brief Test to validate LOWOHA FP32 <-> BF16 conversion with strided source memory.
 */
TEST_P(TestReorder, FP32_BF16_CONV_STRIDED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = (std::rand() % 2 == 0) ? data_type_t::f32 :
                          data_type_t::bf16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::bf16 :
                          data_type_t::f32;
  bool use_row_padding = (std::rand() % 2 == 0);
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, true, false);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params,
                                      use_row_padding);

  // Create strided source tensor
  auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
                      shape, strided_shape, src_dtype, 100.0f);

  // Create destination tensors
  auto dst_tensor = tensor_factory.zero_tensor(shape, dst_dtype);
  auto dst_ref_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  // No scale/zp tensors for simple conversion
  tensor_t scale_tensor;
  tensor_t zp_tensor;

  // Set test-specific parameters
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = true;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  // Execute native kernel
  status_t status = lowoha_reorder_kernel_test(
                      src_tensor, dst_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (status != status_t::success) {
    log_error("LOWOHA reorder (native) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Execute reference kernel
  lowoha_params.lowoha_algo = reorder_algo_t::reference;
  status_t ref_status = lowoha_reorder_kernel_test(
                          src_tensor, dst_ref_tensor, scale_tensor, zp_tensor, lowoha_params);
  if (ref_status != status_t::success) {
    log_error("LOWOHA reorder (reference) execution failed");
    EXPECT_TRUE(false);
    return;
  }

  // Compare results
  bool is_test_successful = true;
  compare_lowoha_reorder_output(dst_tensor, dst_ref_tensor, lowoha_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

//==============================================================================
// LOWOHA Dynamic Quantization Tests (Quantize + Dequantize Round-Trip)
//==============================================================================

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param FP32_DYN_QUANT test name
 *  @brief Test to validate LOWOHA dynamic quantization of F32 input.
 *
 *  Flow:
 *    1. Create F32 source tensor with random data
 *    2. Dynamic quantization: F32 → S8 (symmetric) or U8 (asymmetric)
 *       - API computes scale and zero-point from the source data at runtime
 *       - API quantizes the data to S8/U8 using the computed parameters
 *    3. Dequantization: S8/U8 → F32 using the computed scale/zero-point
 *    4. Compare original F32 input with dequantized F32 output
 *       - Tolerance is based on the quantization step size (scale/2)
 */
TEST_P(TestReorder, FP32_DYN_QUANT) {

  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  // Choose symmetric (S8, no zp) or asymmetric (U8, with zp) quantization
  bool is_symmetric = (std::rand() % 2 == 0);
  data_type_t src_dtype = data_type_t::f32;
  data_type_t compute_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, compute_dtype, false, true);
  log_info("Dynamic Quantization: ",
           is_symmetric ? "Symmetric (S8)" : "Asymmetric (U8)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_dynamic_quant_shape(lowoha_params);

  // Create source tensor (F32 input data)
  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 100.0f);

  // Create quantized output tensor (S8 or U8)
  auto quant_tensor = tensor_factory.zero_tensor(shape, compute_dtype);

  // Create scale output tensor (will be filled by dynamic quantization API)
  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f32);

  // Create zero-point output tensor (only for asymmetric/U8; empty for symmetric/S8)
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::s32);
  }

  // ---- Step 1: Dynamic Quantization (F32 → S8/U8) ----
  // The API computes scale (and zp for U8) from source data, then quantizes
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
  dequant_params.lowoha_algo = reorder_algo_t::reference;

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
  compare_lowoha_dyn_quant_output(src_tensor, dequant_tensor, scale_tensor,
                                  lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class
 *  @param BF16_DYN_QUANT test name
 *  @brief Test to validate LOWOHA dynamic quantization of BF16 input.
 *
 *  Flow:
 *    1. Create BF16 source tensor with random data
 *    2. Dynamic quantization: BF16 → S8 (symmetric) or U8 (asymmetric)
 *       - API computes scale and zero-point from the source data at runtime
 *       - API quantizes the data to S8/U8 using the computed parameters
 *    3. Dequantization: S8/U8 → BF16 using the computed scale/zero-point
 *    4. Compare original BF16 input with dequantized BF16 output
 *       - Tolerance accounts for both quantization step size and BF16 precision
 */
TEST_P(TestReorder, BF16_DYN_QUANT) {

  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  // Choose symmetric (S8, no zp) or asymmetric (U8, with zp) quantization
  bool is_symmetric = (std::rand() % 2 == 0);
  data_type_t src_dtype = data_type_t::bf16;
  data_type_t compute_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, compute_dtype, false, true);
  log_info("Dynamic Quantization: ",
           is_symmetric ? "Symmetric (S8)" : "Asymmetric (U8)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_dynamic_quant_shape(lowoha_params);

  // Create source tensor (BF16 input data)
  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 10.0f);

  // Create quantized output tensor (S8 or U8)
  auto quant_tensor = tensor_factory.zero_tensor(shape, compute_dtype);

  // Create scale output tensor (will be filled by dynamic quantization API)
  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f32);

  // Create zero-point output tensor (only for asymmetric/U8; empty for symmetric/S8)
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::s32);
  }

  // ---- Step 1: Dynamic Quantization (BF16 → S8/U8) ----
  // The API computes scale (and zp for U8) from source data, then quantizes
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
  dequant_params.lowoha_algo = reorder_algo_t::reference;

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
  compare_lowoha_dyn_quant_output(src_tensor, dequant_tensor, scale_tensor,
                                  lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Reorder parameterized test suite (includes both regular and LOWOHA tests)
 */
INSTANTIATE_TEST_SUITE_P(Reorder, TestReorder,
                         ::testing::ValuesIn(reorder_test));
