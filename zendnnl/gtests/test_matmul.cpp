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
    m = params.matmul_m;
    k = params.matmul_k;
    n = params.matmul_n;
    po_index = (gtest_argc >= 3) ?
               (po_map.find(gtest_argv[2])==po_map.end()?po_size:po_map.at(gtest_argv[2]))
               : params.po_index;
    log_info("m: ",m, " k: ",k, " n: ",n," po_index: ",po_index);
    bias    = tensor_factory.uniform_dist_tensor({n}, data_type_t::f32, 2.0);
    bias.set_name("bias");
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}
  uint64_t m,k,n;
  uint32_t po_index;
  tensor_factory_t tensor_factory{};
  tensor_t bias;
};

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize parameters
 *  @param F32 user-defined name of test
 *  @brief Test to validate matmul F32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul,F32) {
  auto weights = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::f32,
                 2.0);
  auto input_tensor = tensor_factory.uniform_dist_tensor({m, k}, data_type_t::f32,
                      2.0);
  auto output_tensor = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  auto output_tensor_ref = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  matmul_kernel_test(input_tensor, weights, bias, output_tensor, po_index);
  matmul_forced_ref_kernel_test(input_tensor, weights, bias, output_tensor_ref,
                                po_index);
  bool flag=false;
  compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_F32_TOL, flag);
  EXPECT_EQ(flag,false);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param BF16_F32 user-defined name of test according to test
 *  @brief Test to validate matmul BF16outputF32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, BF16_F32) {
  auto weights = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::bf16,
                 2.0);
  auto weights_ref = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::f32,
                     2.0);
  auto input_tensor = tensor_factory.uniform_dist_tensor({m, k},
                      data_type_t::bf16, 2.0);
  auto input_tensor_ref = tensor_factory.uniform_dist_tensor({m, k},
                          data_type_t::f32, 2.0);
  auto output_tensor = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  auto output_tensor_ref = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  matmul_kernel_test(input_tensor, weights, bias, output_tensor, po_index);
  matmul_forced_ref_kernel_test(input_tensor_ref, weights_ref, bias,
                                output_tensor_ref, po_index);
  bool flag=false;
  compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_BF16_TOL,
                    flag);
  EXPECT_EQ(flag,false);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param BF16_BF16 user-defined name of test according to test
 *  @brief Test to validate matmul BF16outputBF16 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, BF16_BF16) {
  auto weights = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::bf16,
                 2.0);
  auto weights_ref = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::f32,
                     2.0);
  auto input_tensor = tensor_factory.uniform_dist_tensor({m, k},
                      data_type_t::bf16, 2.0);
  auto input_tensor_ref = tensor_factory.uniform_dist_tensor({m, k},
                          data_type_t::f32, 2.0);
  auto output_tensor = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
  auto output_tensor_ref = tensor_factory.zero_tensor({m, n}, data_type_t::f32);
  matmul_kernel_test(input_tensor, weights, bias, output_tensor, po_index);
  matmul_forced_ref_kernel_test(input_tensor_ref, weights_ref, bias,
                                output_tensor_ref, po_index);
  bool flag=false;
  compare_tensor_2D(output_tensor, output_tensor_ref, m, n, MATMUL_BF16_TOL,
                    flag);
  EXPECT_EQ(flag,false);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Matmul parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Matmul, TestMatmul,
                         ::testing::ValuesIn(matmul_test));
