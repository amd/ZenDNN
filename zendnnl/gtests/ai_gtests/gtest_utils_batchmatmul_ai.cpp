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

#include "gtest_utils_batchmatmul_ai.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include "operators/matmul/matmul_ref_kernel.hpp"

using namespace zendnnl::memory;
using namespace zendnnl::common;
using namespace zendnnl::ops;
using namespace zendnnl::error_handling;

namespace ai_gtests {

// Enum for test case counts
enum class TestCaseCount : int {
  TINY = 5,
  SMALL = 5,
  MEDIUM = 10,
  LARGE = 10,
  RECTANGULAR = 10,
  SKINNY = 8,
  MIN = 3,
  MAX = 3,
  XL_BATCH = 3,
  XXL_BATCH = 2,
  DEFAULT = 3
};

// Add tensor_map_type typedef for local use
using tensor_map_type = std::map<std::string, tensor_t>;

// Random number generation for parameter generation
static std::mt19937 batch_param_rng(
  std::chrono::steady_clock::now().time_since_epoch().count());

// Helper function to generate random dimensions within a range
static uint64_t generate_random_batch_dim(uint64_t min_dim, uint64_t max_dim) {
  std::uniform_int_distribution<uint64_t> dist(min_dim, max_dim);
  return dist(batch_param_rng);
}

// -----------------------------------------------------------------------------
// BatchMatmulTestUtils implementation
// -----------------------------------------------------------------------------

bool BatchMatmulTestUtils::validate_batch_dimensions(uint64_t batch_size,
    uint64_t m, uint64_t n,
    uint64_t k) {
  return (batch_size >= 1 && batch_size <= AI_MAX_DIM &&
          AITestUtils::validate_dimensions(m, n, k));
}

bool BatchMatmulTestUtils::is_valid_broadcast_config(bool broadcast_weights,
    bool broadcast_input) {
  // Both cannot be broadcasted simultaneously in typical batch matmul
  // At least one must be 3D (batched)
  return !(broadcast_weights && broadcast_input);
}

std::vector<uint64_t> BatchMatmulTestUtils::get_input_dims(uint64_t batch_size,
    uint64_t m, uint64_t k,
    bool broadcast_input) {
  if (broadcast_input) {
    return {m, k};  // 2D input (broadcasted)
  }
  return {batch_size, m, k};  // 3D input (batched)
}

std::vector<uint64_t> BatchMatmulTestUtils::get_weight_dims(uint64_t batch_size,
    uint64_t k, uint64_t n,
    bool broadcast_weights) {
  if (broadcast_weights) {
    return {k, n};  // 2D weights (broadcasted)
  }
  return {batch_size, k, n};  // 3D weights (batched)
}

std::vector<uint64_t> BatchMatmulTestUtils::get_output_dims(uint64_t batch_size,
    uint64_t m, uint64_t n) {
  return {batch_size, m, n};  // Output is always 3D (batched)
}

std::vector<uint64_t> BatchMatmulTestUtils::get_bias_dims(uint64_t n) {
  return {1, 1, n};  // Bias is always 3D with batch and m dimensions as 1
}

bool BatchMatmulTestUtils::compare_batch_tensors(const tensor_t &test_tensor,
    const tensor_t &ref_tensor,
    uint64_t k,
    float rel_tolerance,
    float epsilon) {
  // Reuse the matmul comparison logic for batch matmul
  return AITestUtils::compare_sampled_tensors_matmul(test_tensor, ref_tensor,
         k, rel_tolerance, epsilon);
}

status_t BatchMatmulTestUtils::run_reference_batchmatmul(
  tensor_t &input,
  tensor_t &weights,
  tensor_t &bias,
  tensor_t &output,
  const PostOpConfig &post_op_config,
  std::vector<tensor_t> &binary_postop_tensors) {

  try {
    // Prepare input and output maps for the reference kernel
    tensor_map_type inputs;
    tensor_map_type outputs;
    inputs["matmul_input"] = input;
    outputs["matmul_output"] = output;

    // Create context and set parameters
    tensor_t weights_copy = weights;
    tensor_t bias_copy = bias;
    weights_copy.set_name("weights");
    bias_copy.set_name("bias");

    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights_copy)
                          .set_param("bias", bias_copy);

    for (const auto &post_op_type : post_op_config.post_ops) {
      post_op_t post_op{post_op_type};
      matmul_context = matmul_context.set_post_op(post_op);
    }

    matmul_context = matmul_context.create();
    if (!matmul_context.check()) {
      return status_t::failure;
    }

    // Bind binary post-op tensors using the actual tensor names from the context
    size_t binary_tensor_idx = 0;
    for (size_t i = 0; i < post_op_config.post_ops.size(); ++i) {
      auto post_op_type = post_op_config.post_ops[i];
      if ((post_op_type == post_op_type_t::binary_add ||
           post_op_type == post_op_type_t::binary_mul)
          && binary_tensor_idx < binary_postop_tensors.size()) {
        std::string tensor_name;
        try {
          if (post_op_type == post_op_type_t::binary_add) {
            tensor_name = matmul_context.get_post_op(i).binary_add_params.tensor_name;
          }
          else {
            tensor_name = matmul_context.get_post_op(i).binary_mul_params.tensor_name;
          }
        }
        catch (...) {
          tensor_name = "binary_post_op_tensor";
        }
        inputs[tensor_name] = binary_postop_tensors[binary_tensor_idx];
        ++binary_tensor_idx;
      }
    }

    // Create the matmul operator and force reference kernel
    auto matmul_operator = matmul_operator_t()
                           .set_name("batchmatmul_forced_ref_operator")
                           .set_context(matmul_context)
                           .create();

    if (matmul_operator.is_bad_object()) {
      return status_t::failure;
    }

    // Set all inputs
    matmul_operator = matmul_operator.set_input("matmul_input", input);
    for (auto &kv : inputs) {
      matmul_operator = matmul_operator.set_input(kv.first, kv.second);
    }
    matmul_operator = matmul_operator.set_output("matmul_output", output);

    // Force reference kernel
    matmul_operator = matmul_operator.set_forced_kernel("reference");

    // Execute
    status_t status = matmul_operator.execute();
    return status;
  }
  catch (const std::exception &e) {
    std::cerr << "[AI_BATCH_REF] Exception in run_reference_batchmatmul: "
              << e.what() << std::endl;
    return status_t::failure;
  }
  catch (...) {
    std::cerr << "[AI_BATCH_REF] Unknown exception in run_reference_batchmatmul"
              << std::endl;
    return status_t::failure;
  }
}

// -----------------------------------------------------------------------------
// BatchMatmulParameterGenerator implementation
// -----------------------------------------------------------------------------

std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, std::string>>
BatchMatmulParameterGenerator::get_fixed_dimensions_for_minimal_tests() {
  return {
    // Core shapes - essential for functional coverage
    {2, 4, 4, 4, "tiny_square"},
    {2, 3, 5, 4, "tiny_rectangular"},
    {4, 32, 32, 32, "small_square"},
    {8, 64, 64, 64, "medium_square"},
    {4, 128, 128, 128, "large_square"},
    {8, 64, 32, 48, "rectangular_1"},
    {4, 32, 64, 48, "rectangular_2"},
    {2, 42, 42, 42, "non_power_of_2"},
    {4, 512, 4, 4, "skinny_tall"},
    {4, 4, 512, 4, "skinny_wide"},
    {4, 4, 4, 512, "skinny_deep"},
    // Representative large batch tests
    {64, 16, 16, 16, "xlarge_batch_tiny_matrix"},
    {256, 8, 8, 8, "xxxlarge_batch_tiny_matrix"}
  };
}

BatchMatmulParamsAI
BatchMatmulParameterGenerator::generate_random_params_for_accuracy_subcategory(
  const std::string &category,
  DataTypeCombination data_combo,
  const PostOpConfig &post_op_config,
  bool expect_success) {

  uint64_t batch_size = 1, m = 1, n = 1, k = 1;
  bool broadcast_weights = false;
  bool broadcast_input = false;

  // Helper to generate dimensions
  auto generate_dims = [&](uint64_t min_m, uint64_t max_m,
                           uint64_t min_n, uint64_t max_n,
                           uint64_t min_k, uint64_t max_k,
                           BatchSizeDimensions min_batch, BatchSizeDimensions max_batch,
  bool square = false) {
    batch_size = generate_random_batch_dim(static_cast<uint64_t>(min_batch),
                                           static_cast<uint64_t>(max_batch));
    m = generate_random_batch_dim(min_m, max_m);
    n = square ? m : generate_random_batch_dim(min_n, max_n);
    k = generate_random_batch_dim(min_k, max_k);

    // Randomly decide broadcasting (10% chance for each)
    std::uniform_int_distribution<int> broadcast_dist(0, 9);
    if (broadcast_dist(batch_param_rng) == 0) {
      broadcast_weights = true;
    }
    if (broadcast_dist(batch_param_rng) == 0 && !broadcast_weights) {
      broadcast_input = true;
    }
  };

  if (category == "tiny_square") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      BatchSizeDimensions::TINY_MIN, BatchSizeDimensions::TINY_MAX, true);
  }
  else if (category == "tiny_rectangular") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      BatchSizeDimensions::TINY_MIN, BatchSizeDimensions::TINY_MAX, false);
  }
  else if (category == "small_square") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      BatchSizeDimensions::SMALL_MIN, BatchSizeDimensions::SMALL_MAX, true);
  }
  else if (category == "medium_square") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MIN),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MAX),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MIN),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MAX),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MIN),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MAX),
      BatchSizeDimensions::MEDIUM_MIN, BatchSizeDimensions::MEDIUM_MAX, true);
  }
  else if (category == "large_square") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::LARGE_MIN),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MAX),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MIN),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MAX),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MIN),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MAX),
      BatchSizeDimensions::LARGE_MIN, BatchSizeDimensions::LARGE_MAX, true);
  }
  else if (category == "rectangular") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::RECT_MIN),
      static_cast<uint64_t>(MatrixDimensions::RECT_MAX),
      static_cast<uint64_t>(MatrixDimensions::RECT_MIN),
      static_cast<uint64_t>(MatrixDimensions::RECT_MAX),
      static_cast<uint64_t>(MatrixDimensions::RECT_MIN),
      static_cast<uint64_t>(MatrixDimensions::RECT_MAX),
      BatchSizeDimensions::MEDIUM_MIN, BatchSizeDimensions::MEDIUM_MAX, false);
  }
  else if (category == "skinny") {
    int shape_type = generate_random_batch_dim(0, 2);
    if (shape_type == 0) {  // tall
      generate_dims(
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        BatchSizeDimensions::SKINNY_MIN, BatchSizeDimensions::SKINNY_MAX, false);
    }
    else if (shape_type == 1) {  // wide
      generate_dims(
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        BatchSizeDimensions::SKINNY_MIN, BatchSizeDimensions::SKINNY_MAX, false);
    }
    else {  // deep
      generate_dims(
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        BatchSizeDimensions::SKINNY_MIN, BatchSizeDimensions::SKINNY_MAX, false);
    }
  }
  else if (category == "min") {
    // Minimal dimensions - min of all dimensions
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      BatchSizeDimensions::MIN_BATCH, BatchSizeDimensions::MIN_BATCH, false);
  }
  else if (category == "max") {
    // Maximum dimensions - approaching AI_MAX_DIM limits
    // Use smaller batch sizes for very large matrices to avoid memory issues
    generate_dims(
      AI_MAX_DIM / 4, AI_MAX_DIM / 2,
      AI_MAX_DIM / 4, AI_MAX_DIM / 2,
      AI_MAX_DIM / 4, AI_MAX_DIM / 2,
      BatchSizeDimensions::MIN_BATCH, BatchSizeDimensions::MAX_BATCH, false);
  }
  else if (category == "xl_batch_tiny") {
    // XL batch size (1024) with tiny matrices
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      BatchSizeDimensions::XL_MIN, BatchSizeDimensions::XL_MAX, false);
  }
  else if (category == "xl_batch_small") {
    // XL batch size (1024) with small matrices
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      BatchSizeDimensions::XL_MIN, BatchSizeDimensions::XL_MAX, false);
  }
  else if (category == "xxl_batch_tiny") {
    // XXL batch size (2048-4096) with tiny matrices
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      BatchSizeDimensions::XXL_MIN, BatchSizeDimensions::XXL_MAX, false);
  }
  else if (category == "xxl_batch_small") {
    // XXL batch size (2048-4096) with small matrices
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      BatchSizeDimensions::XXL_MIN, BatchSizeDimensions::XXL_MAX, false);
  }
  else {
    // Default case
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      BatchSizeDimensions::TINY_MIN, BatchSizeDimensions::TINY_MAX, false);
  }

  return create_param(batch_size, m, n, k, data_combo, TestCategory::ACCURACY,
                      post_op_config, broadcast_weights, broadcast_input,
                      expect_success);
}

std::vector<BatchMatmulParamsAI>
BatchMatmulParameterGenerator::generate_comprehensive_test_suite() {
  std::vector<BatchMatmulParamsAI> all_params;
  add_accuracy_params(all_params);
  add_boundary_params(all_params);
  add_edge_case_params(all_params);
  add_invalid_params(all_params);
  add_broadcast_params(all_params);
  return all_params;
}

std::vector<BatchMatmulParamsAI>
BatchMatmulParameterGenerator::generate_minimal_test_suite() {
  std::vector<BatchMatmulParamsAI> minimal_params;
  auto post_op_configs = AITestUtils::get_all_post_op_configs();

  // Add fixed dim accuracy params for minimal testing
  add_minimal_accuracy_params(minimal_params);

  // Add boundary tests
  minimal_params.push_back(create_param(1, 1, 1, 1,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::BOUNDARY,
                                        post_op_configs[0]));
  minimal_params.push_back(create_param(8, 8, 8, 8,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::BOUNDARY,
                                        post_op_configs[0]));

  // Add edge cases
  // Very large batch with tiny matrix
  minimal_params.push_back(create_param(256, 4, 4, 4,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));
  // Prime number batch
  minimal_params.push_back(create_param(7, 32, 32, 32,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));
  // Power-of-2 boundary (2^6 - 1)
  minimal_params.push_back(create_param(63, 16, 16, 16,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));
  // Single dimension edge case
  minimal_params.push_back(create_param(1, 1024, 1024, 1,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));
  // Batch size 512 tests
  minimal_params.push_back(create_param(512, 4, 4, 4,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));
  minimal_params.push_back(create_param(512, 16, 16, 16,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));
  // Batch size 1024 tests
  minimal_params.push_back(create_param(1024, 4, 4, 4,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));
  minimal_params.push_back(create_param(1024, 16, 16, 16,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));
  minimal_params.push_back(create_param(1024, 32, 32, 32,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::EDGE_CASE,
                                        post_op_configs[0], false, false, true));

  // Add invalid cases
  // Zero batch size
  minimal_params.push_back(create_param(0, 32, 32, 32,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::INVALID,
                                        post_op_configs[0], false, false, false));
  // Zero m dimension
  minimal_params.push_back(create_param(4, 0, 32, 32,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::INVALID,
                                        post_op_configs[0], false, false, false));
  // Zero n dimension
  minimal_params.push_back(create_param(4, 32, 0, 32,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::INVALID,
                                        post_op_configs[0], false, false, false));
  // Zero k dimension
  minimal_params.push_back(create_param(4, 32, 32, 0,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::INVALID,
                                        post_op_configs[0], false, false, false));
  // Oversized batch
  minimal_params.push_back(create_param(AI_MAX_DIM + 1, 32, 32, 32,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::INVALID,
                                        post_op_configs[0], false, false, false));
  // Oversized m dimension
  minimal_params.push_back(create_param(4, AI_MAX_DIM + 1, 32, 32,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::INVALID,
                                        post_op_configs[0], false, false, false));
  // Invalid broadcast configuration (both input and weights broadcasted)
  minimal_params.push_back(create_param(4, 8, 8, 8,
                                        DataTypeCombination::F32_F32_F32,
                                        TestCategory::INVALID,
                                        post_op_configs[0], true, true, false));

  return minimal_params;
}

std::vector<BatchMatmulParamsAI>
BatchMatmulParameterGenerator::generate_category_specific_params(
  TestCategory category) {
  std::vector<BatchMatmulParamsAI> params;
  switch (category) {
  case TestCategory::ACCURACY:
    add_accuracy_params(params);
    break;
  case TestCategory::BOUNDARY:
    add_boundary_params(params);
    break;
  case TestCategory::EDGE_CASE:
    add_edge_case_params(params);
    break;
  case TestCategory::INVALID:
    add_invalid_params(params);
    break;
  case TestCategory::REFERENCE_KERNEL:
    generate_reference_kernel_exhaustive_params(params);
    break;
  default:
    break;
  }
  return params;
}

void BatchMatmulParameterGenerator::add_minimal_accuracy_params(
  std::vector<BatchMatmulParamsAI> &params) {

  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  auto fixed_dims = get_fixed_dimensions_for_minimal_tests();

  for (auto data_combo : ParameterGenerator::supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      for (const auto& [batch, m, n, k, desc] : fixed_dims) {
        if (AITestUtils::is_aocl_kernel_supported(
              AITestUtils::get_input_dtype(data_combo),
              AITestUtils::get_weight_dtype(data_combo),
              AITestUtils::get_output_dtype(data_combo),
              post_op_config.post_ops)) {
          params.push_back(create_param(batch, m, n, k, data_combo,
                                        TestCategory::ACCURACY,
                                        post_op_config, false, false, true));
        }
      }
    }
  }
}

void BatchMatmulParameterGenerator::add_accuracy_params(
  std::vector<BatchMatmulParamsAI> &params) {

  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  const std::vector<std::string> categories = {
    "tiny_square", "tiny_rectangular", "small_square", "medium_square",
    "large_square", "rectangular", "skinny", "min", "max",
    "xl_batch_tiny", "xl_batch_small", "xxl_batch_tiny", "xxl_batch_small"
  };

  // Reduced test counts for batch matmul (more expensive than regular matmul)
  auto get_max_cases = [](const std::string& category) -> TestCaseCount {
    if (category == "tiny_square" || category == "tiny_rectangular") {
      return TestCaseCount::TINY;
    }
    if (category == "small_square") {
      return TestCaseCount::SMALL;
    }
    if (category == "medium_square" || category == "large_square") {
      return TestCaseCount::MEDIUM;
    }
    if (category == "rectangular") {
      return TestCaseCount::RECTANGULAR;
    }
    if (category == "skinny") {
      return TestCaseCount::SKINNY;
    }
    if (category == "min") {
      return TestCaseCount::MIN;
    }
    if (category == "max") {
      return TestCaseCount::MAX;
    }
    if (category == "xl_batch_tiny" || category == "xl_batch_small") {
      return TestCaseCount::XL_BATCH;
    }
    if (category == "xxl_batch_tiny" || category == "xxl_batch_small") {
      return TestCaseCount::XXL_BATCH;
    }
    return TestCaseCount::DEFAULT;
  };

  for (const auto &category : categories) {
    const int max_cases = static_cast<int>(get_max_cases(category));
    for (auto data_combo : ParameterGenerator::supported_combinations) {
      for (const auto &post_op_config : post_op_configs) {
        if (AITestUtils::is_aocl_kernel_supported(
              AITestUtils::get_input_dtype(data_combo),
              AITestUtils::get_weight_dtype(data_combo),
              AITestUtils::get_output_dtype(data_combo),
              post_op_config.post_ops)) {
          for (int i = 0; i < max_cases; i++) {
            params.push_back(generate_random_params_for_accuracy_subcategory(
                               category, data_combo, post_op_config, true));
          }
        }
      }
    }
  }
}

void BatchMatmulParameterGenerator::add_boundary_params(
  std::vector<BatchMatmulParamsAI> &params) {

  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, std::string>>
  boundary_dims = {
    {1, 1, 1, 1, "minimal_all"},
    {1, 1, 32, 32, "minimal_batch"},
    {32, 1, 32, 32, "minimal_m"},
    {32, 32, 1, 32, "minimal_n"},
    {32, 32, 32, 1, "minimal_k"},
    {8, 8, 8, 8, "simd_boundary"},
    {16, 16, 16, 16, "avx_boundary"}
  };

  for (auto data_combo : ParameterGenerator::supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      for (const auto& [batch, m, n, k, desc] : boundary_dims) {
        if (AITestUtils::is_aocl_kernel_supported(
              AITestUtils::get_input_dtype(data_combo),
              AITestUtils::get_weight_dtype(data_combo),
              AITestUtils::get_output_dtype(data_combo),
              post_op_config.post_ops)) {
          params.push_back(create_param(batch, m, n, k, data_combo,
                                        TestCategory::BOUNDARY,
                                        post_op_config, false, false, true));
        }
      }
    }
  }
}

void BatchMatmulParameterGenerator::add_edge_case_params(
  std::vector<BatchMatmulParamsAI> &params) {

  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>> edge_dims = {
    {1, 1, 128, 128}, {1, 128, 1, 128}, {1, 128, 128, 1},
    {64, 1, 1, 1}, {1, 1024, 1024, 1}, {1, 1, 1024, 1024},
    {32, 1024, 1, 1024}
  };

  for (auto data_combo : ParameterGenerator::supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      for (const auto &tup : edge_dims) {
        uint64_t batch = std::get<0>(tup);
        uint64_t m = std::get<1>(tup);
        uint64_t n = std::get<2>(tup);
        uint64_t k = std::get<3>(tup);
        if (AITestUtils::is_aocl_kernel_supported(
              AITestUtils::get_input_dtype(data_combo),
              AITestUtils::get_weight_dtype(data_combo),
              AITestUtils::get_output_dtype(data_combo),
              post_op_config.post_ops)) {
          params.push_back(create_param(batch, m, n, k, data_combo,
                                        TestCategory::EDGE_CASE,
                                        post_op_config, false, false, true));
        }
      }
    }
  }
}

void BatchMatmulParameterGenerator::add_invalid_params(
  std::vector<BatchMatmulParamsAI> &params) {

  auto post_op_configs = AITestUtils::get_all_post_op_configs();

  for (auto data_combo : ParameterGenerator::supported_combinations) {
    for (auto post_op_config : post_op_configs) {
      if (AITestUtils::is_aocl_kernel_supported(
            AITestUtils::get_input_dtype(data_combo),
            AITestUtils::get_weight_dtype(data_combo),
            AITestUtils::get_output_dtype(data_combo),
            post_op_config.post_ops)) {
        // Invalid batch size
        params.push_back(create_param(0, 32, 32, 32, data_combo,
                                      TestCategory::INVALID,
                                      post_op_config, false, false, false));
        // Invalid dimensions
        params.push_back(create_param(4, 0, 32, 32, data_combo,
                                      TestCategory::INVALID,
                                      post_op_config, false, false, false));
        params.push_back(create_param(4, 32, 0, 32, data_combo,
                                      TestCategory::INVALID,
                                      post_op_config, false, false, false));
        params.push_back(create_param(4, 32, 32, 0, data_combo,
                                      TestCategory::INVALID,
                                      post_op_config, false, false, false));
        // Oversized dimensions
        params.push_back(create_param(AI_MAX_DIM + 1, 32, 32, 32, data_combo,
                                      TestCategory::INVALID,
                                      post_op_config, false, false, false));
      }
    }
  }

  // Invalid broadcasting configurations (both input and weights broadcasted)
  params.push_back(create_param(4, 8, 8, 8, DataTypeCombination::F32_F32_F32,
                                TestCategory::INVALID, post_op_configs[0],
                                true, true, false));
}

void BatchMatmulParameterGenerator::add_broadcast_params(
  std::vector<BatchMatmulParamsAI> &params) {

  auto post_op_configs = AITestUtils::get_all_post_op_configs();

  // Test broadcasting scenarios
  for (auto data_combo : ParameterGenerator::supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      if (AITestUtils::is_aocl_kernel_supported(
            AITestUtils::get_input_dtype(data_combo),
            AITestUtils::get_weight_dtype(data_combo),
            AITestUtils::get_output_dtype(data_combo),
            post_op_config.post_ops)) {
        // Broadcast weights (2D weights, 3D input)
        params.push_back(create_param(4, 32, 32, 32, data_combo,
                                      TestCategory::ACCURACY,
                                      post_op_config, true, false, true));
        // Broadcast input (3D weights, 2D input)
        params.push_back(create_param(4, 32, 32, 32, data_combo,
                                      TestCategory::ACCURACY,
                                      post_op_config, false, true, true));
      }
    }
  }
}

void BatchMatmulParameterGenerator::generate_reference_kernel_exhaustive_params(
  std::vector<BatchMatmulParamsAI> &params) {

  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, std::string>>
  ref_dims = {
    {2, 4, 4, 4, "tiny_square"},
    {2, 4, 3, 2, "tiny_rectangular"},
    {4, 32, 32, 32, "small_square"},
    {8, 64, 64, 64, "medium_square"},
    {4, 128, 128, 128, "large_square"},
    {8, 32, 64, 32, "rectangular_1"},
    {4, 64, 32, 64, "rectangular_2"},
    {2, 96, 96, 96, "non_power_of_2"},
    {4, 256, 4, 4, "skinny_tall"},
    {4, 4, 256, 4, "skinny_wide"},
    {4, 4, 4, 256, "skinny_deep"}
  };

  for (auto data_combo : ParameterGenerator::supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      for (const auto& [batch, m, n, k, desc] : ref_dims) {
        if (AITestUtils::is_reference_implementation_supported(
              AITestUtils::get_input_dtype(data_combo),
              AITestUtils::get_weight_dtype(data_combo),
              AITestUtils::get_output_dtype(data_combo),
              post_op_config.post_ops)) {
          params.push_back(create_param(batch, m, n, k, data_combo,
                                        TestCategory::REFERENCE_KERNEL,
                                        post_op_config, false, false, true));
        }
      }
    }
  }
}

BatchMatmulParamsAI BatchMatmulParameterGenerator::create_param(
  uint64_t batch_size, uint64_t m, uint64_t n, uint64_t k,
  DataTypeCombination combo,
  TestCategory category,
  const PostOpConfig &post_op_config,
  bool broadcast_weights,
  bool broadcast_input,
  bool expect_success) {

  BatchMatmulParamsAI param;
  param.batch_size = batch_size;
  param.m = m;
  param.n = n;
  param.k = k;
  param.data_types = combo;
  param.category = category;
  param.post_op_config = post_op_config;
  param.broadcast_weights = broadcast_weights;
  param.broadcast_input = broadcast_input;
  param.expect_success = expect_success;

  static std::atomic<uint64_t> param_counter{0};

  auto dtype_to_str = [](data_type_t dt) {
    switch (dt) {
    case data_type_t::f32:
      return "f32";
    case data_type_t::bf16:
      return "bf16";
    case data_type_t::s8:
      return "s8";
    case data_type_t::s4:
      return "s4";
    case data_type_t::u8:
      return "u8";
    case data_type_t::s32:
      return "s32";
    default:
      return "unk";
    }
  };

  std::string input_dtype_str = dtype_to_str(AITestUtils::get_input_dtype(combo));
  std::string weight_dtype_str = dtype_to_str(AITestUtils::get_weight_dtype(
                                   combo));
  std::string output_dtype_str = dtype_to_str(AITestUtils::get_output_dtype(
                                   combo));

  std::string broadcast_str = "";
  if (broadcast_weights) {
    broadcast_str += "_bw";
  }
  if (broadcast_input) {
    broadcast_str += "_bi";
  }

  param.test_name = "b" + std::to_string(batch_size) +
                    "_m" + std::to_string(m) +
                    "_n" + std::to_string(n) +
                    "_k" + std::to_string(k) +
                    "_in_" + input_dtype_str +
                    "_wt_" + weight_dtype_str +
                    "_out_" + output_dtype_str +
                    "_" + post_op_config.config_name +
                    broadcast_str +
                    "_" + std::to_string(param_counter.fetch_add(1));

  return param;
}

} // namespace ai_gtests
