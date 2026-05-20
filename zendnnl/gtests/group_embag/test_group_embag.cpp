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

/// @file test_group_embag.cpp
/// @brief Bag-mode tests for group_embedding_bag_direct.
///
///   TestGroupEmbag - F32/BF16/F16 + INT8/S4/U4 with F32/BF16/F16
///                    outputs (sum/mean/max), compared against the
///                    per-table forced-reference kernel.  thread_algo
///                    is randomized via GroupEmbagType so all four
///                    schedulers are exercised across runs.

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "group_embag_test_helpers.hpp"

// =============================================================================
// TestGroupEmbag - bag-mode correctness
// =============================================================================

class TestGroupEmbag : public ::testing::TestWithParam<GroupEmbagType> {
 protected:
  void SetUp() override {
    GroupEmbagType params = GetParam();
    num_embeddings      = params.num_embeddings;
    embedding_dim       = params.embedding_dim;
    num_indices         = params.num_indices;
    num_bags            = params.num_bags;
    algo                = params.algo;
    padding_index       = params.padding_index;
    include_last_offset = params.include_last_offset;
    is_weights          = params.is_weights;
    indices_dtype       = params.indices_dtype;
    offsets_dtype       = params.offsets_dtype;
    fp16_scale_bias     = params.fp16_scale_bias;
    group_size          = params.group_size;
    thread_algo         = params.thread_algo;
    num_threads         = params.num_threads;
    omp_set_num_threads(num_threads);

    log_info("GroupEmbag test: num_embeddings=", num_embeddings,
             " embedding_dim=", embedding_dim,
             " num_indices=", num_indices, " num_bags=", num_bags,
             " algo=", static_cast<int>(algo),
             " group_size=", group_size,
             " thread_algo=", static_cast<int>(thread_algo),
             " num_threads=", num_threads);
  }
  void TearDown() override {}

  uint64_t num_embeddings, embedding_dim, num_indices, num_bags;
  embag_algo_t algo;
  int64_t padding_index;
  bool include_last_offset, is_weights, fp16_scale_bias;
  data_type_t indices_dtype, offsets_dtype;
  size_t group_size;
  eb_thread_algo_t thread_algo;
  int32_t num_threads;
  tensor_factory_t tensor_factory{};

  // Float-table body.  F16 cases skip cleanly on hosts without
  // AVX-512 FP16 via the kernel's isa_unsupported status.
  void run_float_group_test(data_type_t table_dt, data_type_t output_dt,
                            float tol) {
    GroupTensors g = build_group_embag_tensors(
                       tensor_factory, group_size, num_embeddings, embedding_dim,
                       num_indices, num_bags, algo, padding_index,
                       include_last_offset, is_weights, fp16_scale_bias,
                       table_dt, output_dt, indices_dtype, offsets_dtype);

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
      ok = compare_group_outputs(g, num_bags, embedding_dim, tol);
    }
    EXPECT_TRUE(ok);
  }

  // Quantized-table body.  free()'s heap-allocated table buffers
  // before return; `quantized_embedding_tensor_random` returns a
  // buffer the tensor_t does not own.  F16-output cases skip cleanly
  // on hosts without AVX-512 FP16 via the kernel's isa_unsupported
  // status.
  void run_quant_group_test(data_type_t table_dt, data_type_t output_dt,
                            float tol) {
    GroupTensors g = build_group_embag_quant_tensors(
                       tensor_factory, group_size, num_embeddings,
                       embedding_dim, num_indices, num_bags, algo,
                       padding_index, include_last_offset, is_weights,
                       fp16_scale_bias, table_dt, output_dt,
                       indices_dtype, offsets_dtype);

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
      ok = compare_group_outputs(g, num_bags, embedding_dim, tol);
    }
    free_quant_tables(g);
    EXPECT_TRUE(ok);
  }
};

TEST_P(TestGroupEmbag, F32_F32) {
  run_float_group_test(data_type_t::f32, data_type_t::f32, EMBAG_F32_TOL);
}
TEST_P(TestGroupEmbag, F32_BF16) {
  run_float_group_test(data_type_t::f32, data_type_t::bf16, EMBAG_BF16_TOL);
}
TEST_P(TestGroupEmbag, BF16_F32) {
  run_float_group_test(data_type_t::bf16, data_type_t::f32, EMBAG_F32_TOL);
}
TEST_P(TestGroupEmbag, BF16_BF16) {
  run_float_group_test(data_type_t::bf16, data_type_t::bf16, EMBAG_BF16_TOL);
}
TEST_P(TestGroupEmbag, F32_F16) {
  run_float_group_test(data_type_t::f32, data_type_t::f16, EMBAG_F16_TOL);
}
TEST_P(TestGroupEmbag, F16_F32) {
  run_float_group_test(data_type_t::f16, data_type_t::f32, EMBAG_F16_TOL);
}
TEST_P(TestGroupEmbag, F16_F16) {
  run_float_group_test(data_type_t::f16, data_type_t::f16, EMBAG_F16_TOL);
}
TEST_P(TestGroupEmbag, INT8_F32) {
  run_quant_group_test(data_type_t::s8, data_type_t::f32, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbag, INT8_BF16) {
  run_quant_group_test(data_type_t::s8, data_type_t::bf16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbag, INT8_F16) {
  run_quant_group_test(data_type_t::s8, data_type_t::f16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbag, S4_F32) {
  run_quant_group_test(data_type_t::s4, data_type_t::f32, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbag, S4_BF16) {
  run_quant_group_test(data_type_t::s4, data_type_t::bf16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbag, S4_F16) {
  run_quant_group_test(data_type_t::s4, data_type_t::f16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbag, U4_F32) {
  run_quant_group_test(data_type_t::u4, data_type_t::f32, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbag, U4_BF16) {
  run_quant_group_test(data_type_t::u4, data_type_t::bf16, EMBAG_INT4_TOL);
}
TEST_P(TestGroupEmbag, U4_F16) {
  run_quant_group_test(data_type_t::u4, data_type_t::f16, EMBAG_INT4_TOL);
}

INSTANTIATE_TEST_SUITE_P(GroupEmbag, TestGroupEmbag,
                         ::testing::ValuesIn(group_embag_test));
