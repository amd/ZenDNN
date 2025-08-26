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

/** @brief TestEmbag is a test class to handle parameters */
class TestEmbag : public ::testing::TestWithParam<EmbagType> {
 protected:
  /** @brief SetUp is to initialize test parameters
   *
   *  This method is a standard and is used in googletests to initialize parameters
   *  for each test and also acts as fixtures i.e. handling the common part of
   *  each test.
   *
   * */
  virtual void SetUp() {
    EmbagType params = GetParam();
    num_embeddings     = params.num_embeddings;
    embedding_dim      = params.embedding_dim;
    num_bags           = params.num_bags;
    num_indices        = params.num_indices;
    algo               = params.algo;
    padding_index      = params.padding_index;
    include_last_offset = params.include_last_offset;
    is_weights         = params.is_weights;
    scatter_stride     = params.scatter_stride;

    log_info("num_embeddings: ", num_embeddings, " embedding_dim: ", embedding_dim,
             " num_bags: ", num_bags, " num_indices: ", num_indices,
             " algo: ", static_cast<int>(algo), " padding_index: ", padding_index,
             " include_last_offset: ", include_last_offset, " is_weights: ", is_weights);
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}

  uint64_t num_embeddings, embedding_dim, num_bags, num_indices;
  embag_algo_t algo;
  int64_t padding_index;
  bool include_last_offset, is_weights;
  int64_t scatter_stride;
  tensor_factory_t tensor_factory{};
};

//TODO: Implement single test for all datatypes and iterate over input and o/p data type
/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate embag F32 kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, F32_F32) {
  auto table_tensor      = tensor_factory.uniform_dist_tensor({num_embeddings, embedding_dim},
                           data_type_t::f32, 2.0f);
  auto indices_tensor    = tensor_factory.random_indices_tensor({num_indices},
                           num_embeddings);
  uint64_t offsets_size  = include_last_offset ? num_bags + 1 : num_bags;
  auto offsets_tensor    = tensor_factory.random_offsets_tensor({offsets_size},
                           num_indices, include_last_offset);
  auto weights_tensor    = is_weights ? tensor_factory.uniform_dist_tensor({num_indices},
                           data_type_t::f32, 2.0f) : tensor_t();
  auto output_tensor     = tensor_factory.zero_tensor({num_bags, embedding_dim},
                           data_type_t::f32);
  auto output_tensor_ref = tensor_factory.zero_tensor({num_bags, embedding_dim},
                           data_type_t::f32);

  status_t status         = embag_kernel_test(table_tensor, indices_tensor,
                            offsets_tensor, weights_tensor, output_tensor,
                            algo, padding_index, include_last_offset, is_weights,
                            scatter_stride);
  status_t ref_status     = embag_forced_ref_kernel_test(table_tensor,
                            indices_tensor,
                            offsets_tensor, weights_tensor, output_tensor_ref,
                            algo, padding_index, include_last_offset, is_weights,
                            scatter_stride);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, num_bags, embedding_dim,
                      EMBAG_F32_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param F32_BF16 user-defined name of test
 *  @brief Test to validate embag F32 kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, F32_BF16) {
  auto table_tensor      = tensor_factory.uniform_dist_tensor({num_embeddings, embedding_dim},
                           data_type_t::f32, 2.0f);
  auto indices_tensor    = tensor_factory.random_indices_tensor({num_indices},
                           num_embeddings);
  uint64_t offsets_size  = include_last_offset ? num_bags + 1 : num_bags;
  auto offsets_tensor    = tensor_factory.random_offsets_tensor({offsets_size},
                           num_indices, include_last_offset);
  auto weights_tensor    = is_weights ? tensor_factory.uniform_dist_tensor({num_indices},
                           data_type_t::f32, 2.0f) : tensor_t();
  auto output_tensor     = tensor_factory.zero_tensor({num_bags, embedding_dim},
                           data_type_t::bf16);
  auto output_tensor_ref = tensor_factory.zero_tensor({num_bags, embedding_dim},
                           data_type_t::bf16);

  status_t status         = embag_kernel_test(table_tensor, indices_tensor,
                            offsets_tensor, weights_tensor, output_tensor,
                            algo, padding_index, include_last_offset, is_weights,
                            scatter_stride);
  status_t ref_status     = embag_forced_ref_kernel_test(table_tensor,
                            indices_tensor,
                            offsets_tensor, weights_tensor, output_tensor_ref,
                            algo, padding_index, include_last_offset, is_weights,
                            scatter_stride);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, num_bags, embedding_dim,
                      EMBAG_F32_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param BF16_F32 user-defined name of test
 *  @brief Test to validate embag BF16 input F32 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, BF16_F32) {
  auto table_tensor      = tensor_factory.uniform_dist_tensor({num_embeddings, embedding_dim},
                           data_type_t::bf16, 2.0);
  auto indices_tensor    = tensor_factory.random_indices_tensor({num_indices},
                           num_embeddings);
  uint64_t offsets_size  = include_last_offset ? num_bags + 1 : num_bags;
  auto offsets_tensor    = tensor_factory.random_offsets_tensor({offsets_size},
                           num_indices, include_last_offset);
  auto weights_tensor    = is_weights ? tensor_factory.uniform_dist_tensor({num_indices},
                           data_type_t::f32, 2.0f) : tensor_t();
  auto output_tensor     = tensor_factory.zero_tensor({num_bags, embedding_dim},
                           data_type_t::f32);
  auto output_tensor_ref = tensor_factory.zero_tensor({num_bags, embedding_dim},
                           data_type_t::f32);

  status_t status         = embag_kernel_test(table_tensor, indices_tensor,
                            offsets_tensor, weights_tensor, output_tensor,
                            algo, padding_index, include_last_offset, is_weights,
                            scatter_stride);
  status_t ref_status     = embag_forced_ref_kernel_test(table_tensor,
                            indices_tensor,
                            offsets_tensor, weights_tensor, output_tensor_ref,
                            algo, padding_index, include_last_offset, is_weights,
                            scatter_stride);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, num_bags, embedding_dim,
                      EMBAG_F32_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param BF16_BF16 user-defined name of test
 *  @brief Test to validate embag BF16 input BF16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, BF16_BF16) {
  auto table_tensor      = tensor_factory.uniform_dist_tensor({num_embeddings, embedding_dim},
                           data_type_t::bf16, 2.0);
  auto indices_tensor    = tensor_factory.random_indices_tensor({num_indices},
                           num_embeddings);
  uint64_t offsets_size  = include_last_offset ? num_bags + 1 : num_bags;
  auto offsets_tensor    = tensor_factory.random_offsets_tensor({offsets_size},
                           num_indices, include_last_offset);
  auto weights_tensor    = is_weights ? tensor_factory.uniform_dist_tensor({num_indices},
                           data_type_t::f32, 2.0f) : tensor_t();
  auto output_tensor     = tensor_factory.zero_tensor({num_bags, embedding_dim},
                           data_type_t::bf16);
  auto output_tensor_ref = tensor_factory.zero_tensor({num_bags, embedding_dim},
                           data_type_t::bf16);

  status_t status         = embag_kernel_test(table_tensor, indices_tensor,
                            offsets_tensor, weights_tensor, output_tensor,
                            algo, padding_index, include_last_offset, is_weights,
                            scatter_stride);
  status_t ref_status     = embag_forced_ref_kernel_test(table_tensor,
                            indices_tensor,
                            offsets_tensor, weights_tensor, output_tensor_ref,
                            algo, padding_index, include_last_offset, is_weights,
                            scatter_stride);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, num_bags, embedding_dim,
                      EMBAG_BF16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Embag parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(EmbeddingBag, TestEmbag,
                         ::testing::ValuesIn(embag_test));
