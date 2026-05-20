/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file test_group_embedding.cpp
/// @brief Lookup-mode tests for group_embedding_bag_direct.
///
///   TestGroupEmbedding - F32/BF16/F16 + INT8/S4/U4 with F32/BF16/F16
///                        outputs, algo=none and offsets=nullptr per
///                        table, compared against the per-table
///                        forced-reference embedding kernel.

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "group_embag_test_helpers.hpp"

// =============================================================================
// TestGroupEmbedding - lookup-mode correctness
// =============================================================================

class TestGroupEmbedding : public ::testing::TestWithParam<GroupEmbagType> {
 protected:
  void SetUp() override {
    GroupEmbagType params = GetParam();
    num_embeddings  = params.num_embeddings;
    embedding_dim   = params.embedding_dim;
    num_indices     = params.num_indices;
    padding_index   = params.padding_index;
    is_weights      = params.is_weights;
    indices_dtype   = params.indices_dtype;
    fp16_scale_bias = params.fp16_scale_bias;
    group_size      = params.group_size;
    thread_algo     = params.thread_algo;
    num_threads     = params.num_threads;
    omp_set_num_threads(num_threads);

    log_info("GroupEmbedding test: num_embeddings=", num_embeddings,
             " embedding_dim=", embedding_dim,
             " num_indices=", num_indices,
             " group_size=", group_size,
             " thread_algo=", static_cast<int>(thread_algo),
             " num_threads=", num_threads);
  }
  void TearDown() override {}

  uint64_t num_embeddings, embedding_dim, num_indices;
  int64_t padding_index;
  bool is_weights, fp16_scale_bias;
  data_type_t indices_dtype;
  size_t group_size;
  eb_thread_algo_t thread_algo;
  int32_t num_threads;
  tensor_factory_t tensor_factory{};

  // Float-table lookup body.  F16 cases skip cleanly on hosts without
  // AVX-512 FP16 via the kernel's isa_unsupported status.
  void run_float_lookup_test(data_type_t table_dt, data_type_t output_dt,
                             float tol) {
    GroupTensors g = build_group_embedding_tensors(
                       tensor_factory, group_size, num_embeddings,
                       embedding_dim, num_indices, padding_index,
                       is_weights, fp16_scale_bias,
                       table_dt, output_dt, indices_dtype);

    status_t status = group_embag_kernel_test(
                        g.tables, g.indices, g.offsets, g.weights, g.outputs,
                        g.algos, g.padding_idxs, g.include_last_offsets,
                        g.fp16_scale_bias, thread_algo);

    if (status == status_t::isa_unsupported) {
      GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }

    status_t ref_status = group_embag_forced_ref_kernel_test(
                            g.tables, g.indices, g.offsets, g.weights,
                            g.outputs_ref, g.algos, g.padding_idxs,
                            g.include_last_offsets, g.fp16_scale_bias);

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
      ok = compare_group_outputs(g, num_indices, embedding_dim, tol);
    }
    EXPECT_TRUE(ok);
  }

  // Quantized-table lookup body; free()'s table buffers before
  // return.  F16-output cases skip cleanly on hosts without AVX-512
  // FP16 via the kernel's isa_unsupported status.
  void run_quant_lookup_test(data_type_t table_dt, data_type_t output_dt,
                             float tol) {
    GroupTensors g = build_group_embedding_quant_tensors(
                       tensor_factory, group_size, num_embeddings,
                       embedding_dim, num_indices, padding_index,
                       is_weights, fp16_scale_bias,
                       table_dt, output_dt, indices_dtype);

    status_t status = group_embag_kernel_test(
                        g.tables, g.indices, g.offsets, g.weights, g.outputs,
                        g.algos, g.padding_idxs, g.include_last_offsets,
                        g.fp16_scale_bias, thread_algo);

    if (status == status_t::isa_unsupported) {
      free_quant_tables(g);
      GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }

    status_t ref_status = group_embag_forced_ref_kernel_test(
                            g.tables, g.indices, g.offsets, g.weights,
                            g.outputs_ref, g.algos, g.padding_idxs,
                            g.include_last_offsets, g.fp16_scale_bias);

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
      ok = compare_group_outputs(g, num_indices, embedding_dim, tol);
    }
    free_quant_tables(g);
    EXPECT_TRUE(ok);
  }
};

TEST_P(TestGroupEmbedding, F32_F32) {
  run_float_lookup_test(data_type_t::f32, data_type_t::f32, EMBAG_F32_TOL);
}
TEST_P(TestGroupEmbedding, F32_BF16) {
  run_float_lookup_test(data_type_t::f32, data_type_t::bf16, EMBAG_BF16_TOL);
}
TEST_P(TestGroupEmbedding, BF16_F32) {
  run_float_lookup_test(data_type_t::bf16, data_type_t::f32, EMBAG_F32_TOL);
}
TEST_P(TestGroupEmbedding, BF16_BF16) {
  run_float_lookup_test(data_type_t::bf16, data_type_t::bf16, EMBAG_BF16_TOL);
}
TEST_P(TestGroupEmbedding, F32_F16) {
  run_float_lookup_test(data_type_t::f32, data_type_t::f16, EMBAG_F16_TOL);
}
TEST_P(TestGroupEmbedding, F16_F32) {
  run_float_lookup_test(data_type_t::f16, data_type_t::f32, EMBAG_F16_TOL);
}
TEST_P(TestGroupEmbedding, F16_F16) {
  run_float_lookup_test(data_type_t::f16, data_type_t::f16, EMBAG_F16_TOL);
}
TEST_P(TestGroupEmbedding, INT8_F32) {
  run_quant_lookup_test(data_type_t::s8, data_type_t::f32, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbedding, INT8_BF16) {
  run_quant_lookup_test(data_type_t::s8, data_type_t::bf16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbedding, INT8_F16) {
  run_quant_lookup_test(data_type_t::s8, data_type_t::f16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbedding, S4_F32) {
  run_quant_lookup_test(data_type_t::s4, data_type_t::f32, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbedding, S4_BF16) {
  run_quant_lookup_test(data_type_t::s4, data_type_t::bf16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbedding, S4_F16) {
  run_quant_lookup_test(data_type_t::s4, data_type_t::f16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbedding, U4_F32) {
  run_quant_lookup_test(data_type_t::u4, data_type_t::f32, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbedding, U4_BF16) {
  run_quant_lookup_test(data_type_t::u4, data_type_t::bf16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbedding, U4_F16) {
  run_quant_lookup_test(data_type_t::u4, data_type_t::f16, EMBAG_INT4_TOL);
}

INSTANTIATE_TEST_SUITE_P(GroupEmbedding, TestGroupEmbedding,
                         ::testing::ValuesIn(group_embag_test));
