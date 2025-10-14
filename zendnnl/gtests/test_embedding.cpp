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

/** @brief TestEmbedding is a test class to handle parameters */
class TestEmbedding : public ::testing::TestWithParam<EmbeddingType> {
 protected:
  /** @brief SetUp is to initialize test parameters
   *
   *  This method is a standard and is used in googletests to initialize parameters
   *  for each test and also acts as fixtures i.e. handling the common part of
   *  each test.
   *
   * */
  virtual void SetUp() {
    EmbeddingType params = GetParam();
    num_embeddings     = params.num_embeddings;
    embedding_dim      = params.embedding_dim;
    num_indices        = params.num_indices;
    padding_index      = params.padding_index;
    is_weights         = params.is_weights;

    log_info("num_embeddings: ", num_embeddings, " embedding_dim: ", embedding_dim,
             " num_indices: ", num_indices, " padding_index: ", padding_index,
             " is_weights: ", is_weights);
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {}

  uint64_t num_embeddings, embedding_dim, num_indices;
  int64_t padding_index;
  bool is_weights;
  tensor_factory_t tensor_factory{};
};

//TODO: Implement single test for all datatypes and iterate over input and o/p data type
/** @fn TEST_P
 *  @param TestEmbedding parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate embedding F32 input F32 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbedding, F32_F32) {
  auto table_tensor      = tensor_factory.uniform_dist_tensor({num_embeddings, embedding_dim},
                           data_type_t::f32, 2.0f);
  auto indices_tensor    = tensor_factory.random_indices_tensor({num_indices},
                           num_embeddings);
  auto weights_tensor    = is_weights ? tensor_factory.uniform_dist_tensor({num_indices},
                           data_type_t::f32, 2.0f) : tensor_t();
  auto output_tensor     = tensor_factory.zero_tensor({num_indices, embedding_dim},
                           data_type_t::f32);
  auto output_tensor_ref = tensor_factory.zero_tensor({num_indices, embedding_dim},
                           data_type_t::f32);

  status_t status         = embedding_kernel_test(table_tensor, indices_tensor,
                            weights_tensor, output_tensor, padding_index, is_weights);
  status_t ref_status     = embedding_forced_ref_kernel_test(table_tensor,
                            indices_tensor, weights_tensor,
                            output_tensor_ref, padding_index, is_weights);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, num_indices, embedding_dim,
                      EMBAG_F32_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbedding parameterized test class to initialize parameters
 *  @param F32_BF16 user-defined name of test
 *  @brief Test to validate embedding F32 input BF16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbedding, F32_BF16) {
  auto table_tensor      = tensor_factory.uniform_dist_tensor({num_embeddings, embedding_dim},
                           data_type_t::f32, 2.0f);
  auto indices_tensor    = tensor_factory.random_indices_tensor({num_indices},
                           num_embeddings);
  auto weights_tensor    = is_weights ? tensor_factory.uniform_dist_tensor({num_indices},
                           data_type_t::f32, 2.0f) : tensor_t();
  auto output_tensor     = tensor_factory.zero_tensor({num_indices, embedding_dim},
                           data_type_t::bf16);
  auto output_tensor_ref = tensor_factory.zero_tensor({num_indices, embedding_dim},
                           data_type_t::bf16);

  status_t status         = embedding_kernel_test(table_tensor, indices_tensor,
                            weights_tensor, output_tensor,
                            padding_index, is_weights);
  status_t ref_status     = embedding_forced_ref_kernel_test(table_tensor,
                            indices_tensor, weights_tensor,
                            output_tensor_ref, padding_index, is_weights);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);
  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, num_indices, embedding_dim,
                      EMBAG_BF16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbedding parameterized test class to initialize parameters
 *  @param BF16_F32 user-defined name of test
 *  @brief Test to validate embedding BF16 input F32 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbedding, BF16_F32) {
  auto table_tensor      = tensor_factory.uniform_dist_tensor({num_embeddings, embedding_dim},
                           data_type_t::bf16, 2.0f);
  auto indices_tensor    = tensor_factory.random_indices_tensor({num_indices},
                           num_embeddings);
  auto weights_tensor    = is_weights ? tensor_factory.uniform_dist_tensor({num_indices},
                           data_type_t::f32, 2.0f) : tensor_t();
  auto output_tensor     = tensor_factory.zero_tensor({num_indices, embedding_dim},
                           data_type_t::f32);
  auto output_tensor_ref = tensor_factory.zero_tensor({num_indices, embedding_dim},
                           data_type_t::f32);

  status_t status         = embedding_kernel_test(table_tensor, indices_tensor,
                            weights_tensor, output_tensor,
                            padding_index, is_weights);
  status_t ref_status     = embedding_forced_ref_kernel_test(table_tensor,
                            indices_tensor, weights_tensor,
                            output_tensor_ref, padding_index, is_weights);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, num_indices, embedding_dim,
                      EMBAG_F32_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbedding parameterized test class to initialize parameters
 *  @param BF16_BF16 user-defined name of test
 *  @brief Test to validate embedding BF16 input BF16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbedding, BF16_BF16) {
  auto table_tensor      = tensor_factory.uniform_dist_tensor({num_embeddings, embedding_dim},
                           data_type_t::bf16, 2.0f);
  auto indices_tensor    = tensor_factory.random_indices_tensor({num_indices},
                           num_embeddings);
  auto weights_tensor    = is_weights ? tensor_factory.uniform_dist_tensor({num_indices},
                           data_type_t::f32, 2.0f) : tensor_t();
  auto output_tensor     = tensor_factory.zero_tensor({num_indices, embedding_dim},
                           data_type_t::bf16);
  auto output_tensor_ref = tensor_factory.zero_tensor({num_indices, embedding_dim},
                           data_type_t::bf16);

  status_t status         = embedding_kernel_test(table_tensor, indices_tensor,
                            weights_tensor, output_tensor, padding_index, is_weights);
  status_t ref_status     = embedding_forced_ref_kernel_test(table_tensor,
                            indices_tensor, weights_tensor,
                            output_tensor_ref, padding_index, is_weights);
  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_2D(output_tensor, output_tensor_ref, num_indices, embedding_dim,
                      EMBAG_BF16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Embedding parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Embedding, TestEmbedding,
                         ::testing::ValuesIn(embedding_test));
