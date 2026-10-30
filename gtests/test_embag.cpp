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

#include "gtest_utils.hpp"
#include <gtest/gtest.h>

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
        num_embeddings = params.num_embeddings;
        embedding_dim = params.embedding_dim;
        num_bags = params.num_bags;
        num_indices = params.num_indices;
        algo = params.algo;
        padding_index = params.padding_index;
        include_last_offset = params.include_last_offset;
        is_weights = params.is_weights;
        indices_dtype = params.indices_dtype;
        offsets_dtype = params.offsets_dtype;
        fp16_scale_bias = params.fp16_scale_bias;
        strided = params.strided;
        use_LOWOHA = params.use_LOWOHA;
        // DUT kernel is a function of this case's already-seeded params so
        // `--seed` replays the same kernel without consuming extra rand()
        // (isolated --gtest_filter=.../N stays consistent). Reference stays
        // the oracle path in each TEST_P. ZENDNNL_EMBAG_ALGO, when set, wins
        // over the hash: leave kernel as none so kernel_select reads the env.
        const char *embag_algo_env = std::getenv("ZENDNNL_EMBAG_ALGO");
        if (embag_algo_env && embag_algo_env[0] != '\0') {
            kernel = embag_kernel_t::none;
        } else {
            switch ((params.num_embeddings + params.embedding_dim
                            + params.num_bags + params.num_indices
                            + static_cast<uint64_t>(params.algo)
                            + params.is_weights + params.strided)
                    % 3) {
                case 0: kernel = embag_kernel_t::none; break;
                case 1: kernel = embag_kernel_t::native; break;
                default: kernel = embag_kernel_t::fbgemm; break;
            }
        }
        // LOWOHA-only mode: tests are masked when the user explicitly selects the
        // regular (non-LOWOHA) API. Skip with a message asking the user to use the
        // LOA (LOWOHA) API. Run this guard *before* any global side effects
        // (e.g. omp_set_num_threads) so skipped tests don't mutate process state
        // that subsequent test suites rely on.
        if (!use_LOWOHA) {
            GTEST_SKIP() << "Skipping: please use LOA (LOWOHA) API. "
                         << "Omit --lowoha or pass --lowoha true to run these "
                            "tests.";
        }
        num_threads = params.num_threads;
        omp_set_num_threads(num_threads);

        log_info("num_embeddings: ", num_embeddings,
                " embedding_dim: ", embedding_dim, " num_bags: ", num_bags,
                " num_indices: ", num_indices,
                " algo: ", static_cast<int>(algo),
                " padding_index: ", padding_index,
                " include_last_offset: ", include_last_offset,
                " is_weights: ", is_weights, " strided: ", strided,
                " use_LOWOHA: ", use_LOWOHA, " num_threads: ", num_threads,
                " kernel: ", kernel_to_string(kernel));
    }

    /** @brief TearDown is used to free resource used in test */
    virtual void TearDown() {}

    uint64_t num_embeddings, embedding_dim, num_bags, num_indices;
    embag_algo_t algo;
    int64_t padding_index;
    bool include_last_offset, is_weights, fp16_scale_bias;
    data_type_t indices_dtype, offsets_dtype;
    bool use_LOWOHA, strided;
    int32_t num_threads;
    embag_kernel_t kernel;
    tensor_factory_t tensor_factory {};
};

//TODO: Implement single test for all datatypes and iterate over input and o/p data type
/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate embag F32 kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, F32_F32) {
    auto table_tensor = tensor_factory.uniform_dist_tensor(
            {num_embeddings, embedding_dim}, data_type_t::f32, 2.0f);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f32, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f32, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_F32_TOL, is_test_successful);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param F32_BF16 user-defined name of test
 *  @brief Test to validate embag F32 kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, F32_BF16) {
    auto table_tensor = tensor_factory.uniform_dist_tensor(
            {num_embeddings, embedding_dim}, data_type_t::f32, 2.0f);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::bf16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::bf16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_BF16_TOL, is_test_successful);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param BF16_F32 user-defined name of test
 *  @brief Test to validate embag BF16 input F32 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, BF16_F32) {
    auto table_tensor = tensor_factory.uniform_dist_tensor(
            {num_embeddings, embedding_dim}, data_type_t::bf16, 2.0f);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f32, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f32, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_F32_TOL, is_test_successful);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param BF16_BF16 user-defined name of test
 *  @brief Test to validate embag BF16 input BF16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, BF16_BF16) {
    auto table_tensor = tensor_factory.uniform_dist_tensor(
            {num_embeddings, embedding_dim}, data_type_t::bf16, 2.0f);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::bf16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::bf16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_BF16_TOL, is_test_successful);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param F32_F16 user-defined name of test
 *  @brief Test to validate embag F32 input F16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, F32_F16) {
    auto table_tensor = tensor_factory.uniform_dist_tensor(
            {num_embeddings, embedding_dim}, data_type_t::f32, 2.0f);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_F16_TOL, is_test_successful);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param F16_F32 user-defined name of test
 *  @brief Test to validate embag F16 input F32 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, F16_F32) {
    auto table_tensor = tensor_factory.uniform_dist_tensor(
            {num_embeddings, embedding_dim}, data_type_t::f16, 2.0f);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f32, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f32, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_F16_TOL, is_test_successful);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param F16_F16 user-defined name of test
 *  @brief Test to validate embag F16 input F16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, F16_F16) {
    auto table_tensor = tensor_factory.uniform_dist_tensor(
            {num_embeddings, embedding_dim}, data_type_t::f16, 2.0f);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_F16_TOL, is_test_successful);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param INT8_F32 user-defined name of test
 *  @brief Test to validate embag INT8 input F32 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, INT8_F32) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::s8, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f32, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f32, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (no_int8_int4_kernel(status)) {
        free(table_tensor.get_raw_handle_unsafe());
        table_tensor.reset();
        GTEST_SKIP() << "INT8/INT4 not supported: requires AVX-512F and "
                        "AVX512-BF16";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param INT8_BF16 user-defined name of test
 *  @brief Test to validate embag INT8 input BF16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, INT8_BF16) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::s8, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::bf16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::bf16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (no_int8_int4_kernel(status)) {
        free(table_tensor.get_raw_handle_unsafe());
        table_tensor.reset();
        GTEST_SKIP() << "INT8/INT4 not supported: requires AVX-512F and "
                        "AVX512-BF16";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param INT8_F16 user-defined name of test
 *  @brief Test to validate embag S8 input F16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, INT8_F16) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::s8, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param S4_F32 user-defined name of test
 *  @brief Test to validate embag S4 input F32 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, S4_F32) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::s4, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f32, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f32, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (no_int8_int4_kernel(status)) {
        free(table_tensor.get_raw_handle_unsafe());
        table_tensor.reset();
        GTEST_SKIP() << "INT8/INT4 not supported: requires AVX-512F and "
                        "AVX512-BF16";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param S4_BF16 user-defined name of test
 *  @brief Test to validate embag S4 input BF16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, S4_BF16) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::s4, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::bf16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::bf16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (no_int8_int4_kernel(status)) {
        free(table_tensor.get_raw_handle_unsafe());
        table_tensor.reset();
        GTEST_SKIP() << "INT8/INT4 not supported: requires AVX-512F and "
                        "AVX512-BF16";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param S4_F16 user-defined name of test
 *  @brief Test to validate embag S4 input F16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, S4_F16) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::s4, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param U4_F32 user-defined name of test
 *  @brief Test to validate embag U4 input F32 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, U4_F32) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::u4, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f32, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f32, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (no_int8_int4_kernel(status)) {
        free(table_tensor.get_raw_handle_unsafe());
        table_tensor.reset();
        GTEST_SKIP() << "INT8/INT4 not supported: requires AVX-512F and "
                        "AVX512-BF16";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param U4_BF16 user-defined name of test
 *  @brief Test to validate embag U4 input BF16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, U4_BF16) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::u4, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::bf16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::bf16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (no_int8_int4_kernel(status)) {
        free(table_tensor.get_raw_handle_unsafe());
        table_tensor.reset();
        GTEST_SKIP() << "INT8/INT4 not supported: requires AVX-512F and "
                        "AVX512-BF16";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestEmbag parameterized test class to initialize parameters
 *  @param U4_F16 user-defined name of test
 *  @brief Test to validate embag U4 input F16 output kernel support wrt Reference kernel
 */
TEST_P(TestEmbag, U4_F16) {
    auto table_tensor = tensor_factory.quantized_embedding_tensor_random(
            {num_embeddings, embedding_dim}, data_type_t::u4, "table",
            fp16_scale_bias);
    auto indices_tensor = tensor_factory.random_indices_tensor(
            {num_indices}, num_embeddings, indices_dtype);
    uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;
    auto offsets_tensor = tensor_factory.random_offsets_tensor(
            {offsets_size}, num_indices, offsets_dtype, include_last_offset);
    auto weights_tensor = is_weights
            ? tensor_factory.uniform_dist_tensor(
                      {num_indices}, data_type_t::f32, 2.0f)
            : tensor_t();
    auto output_tensor = tensor_factory.zero_tensor({num_bags, embedding_dim},
            data_type_t::f16, tensor_t(), tensor_t(), strided);
    auto output_tensor_ref
            = tensor_factory.zero_tensor({num_bags, embedding_dim},
                    data_type_t::f16, tensor_t(), tensor_t(), strided);

    status_t status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor, algo, padding_index,
            include_last_offset, is_weights, fp16_scale_bias, kernel,
            use_LOWOHA);
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }
    status_t ref_status = embag_kernel_test(table_tensor, indices_tensor,
            offsets_tensor, weights_tensor, output_tensor_ref, algo,
            padding_index, include_last_offset, is_weights, fp16_scale_bias,
            embag_kernel_t::reference, use_LOWOHA);
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D(output_tensor, output_tensor_ref, num_bags,
                embedding_dim, EMBAG_INT4_TOL, is_test_successful);
    }
    //free this table pointer after use
    free(table_tensor.get_raw_handle_unsafe());
    table_tensor.reset();
    EXPECT_TRUE(is_test_successful);
}

#if ZENDNNL_DEPENDS_FBGEMM
/** @fn TEST
 *  @brief FBGEMM rejects an index outside [0, num_embeddings) and stops
 *         before writing the rest of its output, so the LOWOHA call must
 *         return status_t::failure. The same call with valid indices
 *         succeeds.
 */
TEST(EmbagFbgemm, RejectedIndexReturnsFailure) {
    constexpr uint64_t rows = 16, dim = 8, bags = 2;
    std::vector<float> table(rows * dim, 1.0f);
    std::vector<int64_t> indices = {0, 1, 2, 3};
    std::vector<int64_t> offsets = {0, 2};
    std::vector<float> out(bags * dim, 0.0f);

    embag_params_t params;
    params.dtypes.table = data_type_t::f32;
    params.dtypes.output = data_type_t::f32;
    params.algo = embag_algo_t::sum;
    params.num_embeddings = rows;
    params.embedding_dim = dim;
    params.num_indices = indices.size();
    params.num_bags = bags;
    params.dst_stride = dim;
    params.kernel = embag_kernel_t::fbgemm;

    ASSERT_EQ(embedding_bag_direct(table.data(), indices.data(), offsets.data(),
                      nullptr, out.data(), params),
            status_t::success);
    // Each bag sums two rows of ones.
    EXPECT_TRUE(std::all_of(
            out.begin(), out.end(), [](float v) { return v == 2.0f; }));

    indices[3] = static_cast<int64_t>(rows);
    EXPECT_EQ(embedding_bag_direct(table.data(), indices.data(), offsets.data(),
                      nullptr, out.data(), params),
            status_t::failure);
}

/** @fn TEST
 *  @brief FBGEMM has no padding index, so a sum with padding_idx set must
 *         skip the padding row even when FBGEMM is requested.
 */
TEST(EmbagFbgemm, PaddedSumSkipsPaddingRow) {
    constexpr uint64_t rows = 4, dim = 16, bags = 1;
    // Row r holds r + 1.
    std::vector<float> table(rows * dim);
    for (uint64_t r = 0; r < rows; ++r) {
        std::fill_n(table.begin() + r * dim, dim, static_cast<float>(r + 1));
    }
    std::vector<int64_t> indices = {0, 1, 2};
    std::vector<int64_t> offsets = {0};
    std::vector<float> out(bags * dim, 0.0f);

    embag_params_t params;
    params.dtypes.table = data_type_t::f32;
    params.dtypes.output = data_type_t::f32;
    params.algo = embag_algo_t::sum;
    params.num_embeddings = rows;
    params.embedding_dim = dim;
    params.num_indices = indices.size();
    params.num_bags = bags;
    params.padding_idx = 0;
    params.dst_stride = dim;
    params.kernel = embag_kernel_t::fbgemm;

    ASSERT_EQ(embedding_bag_direct(table.data(), indices.data(), offsets.data(),
                      nullptr, out.data(), params),
            status_t::success);
    // Rows 1 and 2 hold 2 and 3; adding padding row 0 would give 6.
    EXPECT_TRUE(std::all_of(
            out.begin(), out.end(), [](float v) { return v == 5.0f; }));
}
#endif

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Embag parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(
        EmbeddingBag, TestEmbag, ::testing::ValuesIn(embag_test));

namespace {

// Sets ZENDNNL_EMBAG_ISA for its lifetime. Not thread-safe: setenv/getenv
// are racy across threads. Fine for the gtests binary since tests run
// serially within the process.
class scoped_embag_isa {
public:
    explicit scoped_embag_isa(const char *value) {
        if (const char *prev = std::getenv(kEnvName)) {
            prev_value_ = prev;
            had_prev_ = true;
        }
        setenv(kEnvName, value, /*overwrite=*/1);
    }
    ~scoped_embag_isa() {
        if (had_prev_) {
            setenv(kEnvName, prev_value_.c_str(), /*overwrite=*/1);
        } else {
            unsetenv(kEnvName);
        }
    }
    scoped_embag_isa(const scoped_embag_isa &) = delete;
    scoped_embag_isa &operator=(const scoped_embag_isa &) = delete;

private:
    static constexpr const char *kEnvName = "ZENDNNL_EMBAG_ISA";
    std::string prev_value_;
    bool had_prev_ = false;
};

bool can_run_avx2_embag_kernel() {
    const platform_info_t &platform = zendnnl_platform_info();
    return platform.get_avx2_status();
}

} // namespace

/** @fn TEST
 *  @brief With ZENDNNL_EMBAG_ISA=avx2 the LOWOHA path runs the AVX2 kernel
 *         that CPUs without AVX-512 use. Compare it with the reference kernel
 *         for every type it covers.
 */
TEST(EmbagIsa, Avx2KernelMatchesReference) {
    if (!can_run_avx2_embag_kernel()) { GTEST_SKIP() << "Requires AVX2."; }
    // 37 columns: four full 8-wide blocks plus a tail.
    constexpr uint64_t num_embeddings = 100, embedding_dim = 37;
    constexpr uint64_t num_bags = 8, num_indices = 40;
    tensor_factory_t factory;
    const data_type_t dtypes[] = {data_type_t::f32, data_type_t::bf16};
    const auto dtype_name = [](data_type_t dtype) {
        return dtype == data_type_t::f32 ? "f32" : "bf16";
    };
    // embag_algo_t::none is the embedding lookup.
    const struct {
        embag_algo_t algo;
        const char *name;
    } algos[] = {{embag_algo_t::sum, "sum"}, {embag_algo_t::mean, "mean"},
            {embag_algo_t::max, "max"}, {embag_algo_t::none, "lookup"}};

    for (data_type_t table_dtype : dtypes) {
        for (data_type_t output_dtype : dtypes) {
            for (const auto &algo_case : algos) {
                const embag_algo_t algo = algo_case.algo;
                const bool lookup = (algo == embag_algo_t::none);
                const uint64_t rows = lookup ? num_indices : num_bags;
                auto table = factory.uniform_dist_tensor(
                        {num_embeddings, embedding_dim}, table_dtype, 2.0f);
                auto indices = factory.random_indices_tensor(
                        {num_indices}, num_embeddings, data_type_t::s64);
                auto offsets = factory.random_offsets_tensor({num_bags},
                        num_indices, data_type_t::s64,
                        /*include_last_offset_=*/false);
                tensor_t weights;
                auto output = factory.zero_tensor(
                        {rows, embedding_dim}, output_dtype);
                auto output_ref = factory.zero_tensor(
                        {rows, embedding_dim}, output_dtype);

                const auto run = [&](tensor_t &out, embag_kernel_t kernel) {
                    return lookup ? embedding_kernel_test(table, indices,
                                            weights, out, /*padding_index=*/-1,
                                            /*is_weights=*/false,
                                            /*fp16_scale_bias=*/true, kernel,
                                            /*use_LOWOHA=*/true)
                                  : embag_kernel_test(table, indices, offsets,
                                            weights, out, algo,
                                            /*padding_index=*/-1,
                                            /*include_last_offset=*/false,
                                            /*is_weights=*/false,
                                            /*fp16_scale_bias=*/true, kernel,
                                            /*use_LOWOHA=*/true);
                };
                status_t status;
                {
                    scoped_embag_isa isa("avx2");
                    status = run(output, embag_kernel_t::native);
                }
                const status_t ref_status
                        = run(output_ref, embag_kernel_t::reference);

                bool is_test_successful = (status == status_t::success
                        && ref_status == status_t::success);
                if (is_test_successful) {
                    compare_tensor_2D(output, output_ref, rows, embedding_dim,
                            output_dtype == data_type_t::f32 ? EMBAG_F32_TOL
                                                             : EMBAG_BF16_TOL,
                            is_test_successful);
                }
                EXPECT_TRUE(is_test_successful)
                        << dtype_name(table_dtype) << " table, "
                        << dtype_name(output_dtype) << " output, "
                        << algo_case.name;
            }
        }
    }
}

/** @fn TEST
 *  @brief A wide embedding with fewer bags than threads gives the AVX2 kernel
 *         workers with no row. Only the workers with a bag may allocate
 *         accumulator scratch, and the result must still be right.
 */
TEST(EmbagIsa, Avx2SparseBatchWideEmbedding) {
    if (!can_run_avx2_embag_kernel()) { GTEST_SKIP() << "Requires AVX2."; }
    scoped_embag_isa isa("avx2");

    // 65536 columns is 8192 accumulator registers (256 KiB) per worker.
    // The width is not a multiple of 8 so the tail block is covered too.
    constexpr uint64_t rows = 8, dim = 65536 + 3, num_indices = 3;
    constexpr int32_t many_threads = 48;
    std::vector<float> table(rows * dim);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < dim; ++c) {
            table[r * dim + c] = static_cast<float>(r + 1)
                    + static_cast<float>(c % 7) * 0.25f;
        }
    }
    const std::vector<int64_t> indices = {1, 4, 6};
    const std::vector<int64_t> offsets = {0};

    embag_params_t params;
    params.dtypes.table = data_type_t::f32;
    params.dtypes.output = data_type_t::f32;
    params.algo = embag_algo_t::sum;
    params.num_embeddings = rows;
    params.embedding_dim = dim;
    params.num_indices = num_indices;
    params.num_bags = 1;
    params.dst_stride = dim;
    params.kernel = embag_kernel_t::native;
    params.num_threads = many_threads;

    std::vector<float> out(dim, -1.0f);
    ASSERT_EQ(embedding_bag_direct(table.data(), indices.data(), offsets.data(),
                      nullptr, out.data(), params),
            status_t::success);
    for (uint64_t c = 0; c < dim; ++c) {
        float expected = 0.0f;
        for (int64_t idx : indices) {
            expected += table[idx * dim + c];
        }
        ASSERT_NEAR(out[c], expected, 1e-4f) << "column " << c;
    }
}

/** @fn TEST
 *  @brief When the AVX-512 kernels may not run, types the AVX2 kernel does
 *         not cover return status_t::isa_unsupported instead of executing
 *         AVX-512 code, and group_embedding_bag_direct fails before running
 *         any of its tables.
 */
TEST(EmbagIsa, NoKernelReturnsIsaUnsupportedBeforeRunning) {
    if (!can_run_avx2_embag_kernel()) { GTEST_SKIP() << "Requires AVX2."; }
    scoped_embag_isa isa("avx2");

    constexpr uint64_t rows = 16, dim = 8, num_indices = 4;
    std::vector<float> f32_table(rows * dim, 1.0f);
    // INT8 rows carry their FP32 scale and bias after the data. No kernel
    // runs in this test, so the contents do not matter.
    std::vector<int8_t> s8_table(rows * (dim + 2 * sizeof(float)), 1);
    std::vector<int64_t> indices = {0, 1, 2, 3};
    std::vector<int64_t> offsets = {0};
    std::vector<float> f32_out(dim, -1.0f);
    std::vector<float> s8_out(dim, -1.0f);

    embag_params_t f32_params;
    f32_params.dtypes.table = data_type_t::f32;
    f32_params.dtypes.output = data_type_t::f32;
    f32_params.algo = embag_algo_t::sum;
    f32_params.num_embeddings = rows;
    f32_params.embedding_dim = dim;
    f32_params.num_indices = num_indices;
    f32_params.num_bags = 1;
    f32_params.dst_stride = dim;
    f32_params.kernel = embag_kernel_t::native;
    embag_params_t s8_params = f32_params;
    s8_params.dtypes.table = data_type_t::s8;
    s8_params.fp16_scale_bias = false;

    EXPECT_EQ(embedding_bag_direct(s8_table.data(), indices.data(),
                      offsets.data(), nullptr, s8_out.data(), s8_params),
            status_t::isa_unsupported);

    EXPECT_EQ(group_embedding_bag_direct({f32_table.data(), s8_table.data()},
                      {indices.data(), indices.data()},
                      {offsets.data(), offsets.data()}, {nullptr, nullptr},
                      {f32_out.data(), s8_out.data()}, {f32_params, s8_params}),
            status_t::isa_unsupported);
    // The FP32 table has a kernel but must not have run.
    EXPECT_TRUE(std::all_of(f32_out.begin(), f32_out.end(),
            [](float v) { return v == -1.0f; }));
}
