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

#ifndef _GTEST_UTILS_BATCHMATMUL_AI_HPP_
#define _GTEST_UTILS_BATCHMATMUL_AI_HPP_

#include "gtest_utils_ai.hpp"

namespace ai_gtests {

/** @brief Batch size dimension categories for batch matmul */
enum class BatchSizeDimensions : uint64_t {
  // Batch size ranges
  TINY_MIN = 1,
  TINY_MAX = 4,
  SMALL_MIN = 2,
  SMALL_MAX = 8,
  MEDIUM_MIN = 2,
  MEDIUM_MAX = 16,
  LARGE_MIN = 2,
  LARGE_MAX = 8,
  SKINNY_MIN = 2,
  SKINNY_MAX = 8,
  MIN_BATCH = 1,
  MAX_BATCH = 4,
  XL_MIN = 1024,
  XL_MAX = 1024,
  XXL_MIN = 2048,
  XXL_MAX = 4096
};

/** @brief AI-specific batch matmul parameter structure */
struct BatchMatmulParamsAI {
  uint64_t batch_size;  // Batch dimension
  uint64_t m, n, k;     // Matrix dimensions
  DataTypeCombination data_types;
  TestCategory category;
  PostOpConfig post_op_config;
  bool broadcast_weights;  // If true, weights are 2D (broadcasted across batch)
  bool broadcast_input;    // If true, input is 2D (broadcasted across batch)
  bool expect_success;
  std::string test_name;

  BatchMatmulParamsAI() : batch_size(1), m(1), n(1), k(1),
    data_types(DataTypeCombination::F32_F32_F32),
    category(TestCategory::ACCURACY),
    broadcast_weights(false),
    broadcast_input(false),
    expect_success(true),
    test_name("") {}
};

/** @brief Batch matmul-specific utility functions */
class BatchMatmulTestUtils {
 public:
  // Dimension validation
  static bool validate_batch_dimensions(uint64_t batch_size, uint64_t m,
                                        uint64_t n, uint64_t k);

  // Broadcasting validation
  static bool is_valid_broadcast_config(bool broadcast_weights,
                                        bool broadcast_input);

  // Tensor comparison for batch matmul
  static bool compare_batch_tensors(const tensor_t &test_tensor,
                                    const tensor_t &ref_tensor,
                                    uint64_t k,
                                    float rel_tolerance,
                                    float epsilon);

  // Reference implementation for batch matmul
  static status_t run_reference_batchmatmul(
    tensor_t &input,
    tensor_t &weights,
    tensor_t &bias,
    tensor_t &output,
    const PostOpConfig &post_op_config,
    std::vector<tensor_t> &binary_postop_tensors);

  // Get expected tensor dimensions based on broadcasting
  static std::vector<uint64_t> get_input_dims(uint64_t batch_size, uint64_t m,
      uint64_t k, bool broadcast_input);
  static std::vector<uint64_t> get_weight_dims(uint64_t batch_size, uint64_t k,
      uint64_t n, bool broadcast_weights);
  static std::vector<uint64_t> get_output_dims(uint64_t batch_size, uint64_t m,
      uint64_t n);
  static std::vector<uint64_t> get_bias_dims(uint64_t n);
};

/** @brief Batch matmul parameter generator */
class BatchMatmulParameterGenerator {
 public:
  static std::vector<BatchMatmulParamsAI> generate_comprehensive_test_suite();
  static std::vector<BatchMatmulParamsAI> generate_minimal_test_suite();
  static std::vector<BatchMatmulParamsAI> generate_category_specific_params(
    TestCategory category);

 private:
  static BatchMatmulParamsAI generate_random_params_for_accuracy_subcategory(
    const std::string &category,
    DataTypeCombination data_combo,
    const PostOpConfig &post_op_config,
    bool expect_success);

  static std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, std::string>>
      get_fixed_dimensions_for_minimal_tests();

  static void add_minimal_accuracy_params(std::vector<BatchMatmulParamsAI>
                                          &params);
  static void add_accuracy_params(std::vector<BatchMatmulParamsAI> &params);
  static void add_boundary_params(std::vector<BatchMatmulParamsAI> &params);
  static void add_edge_case_params(std::vector<BatchMatmulParamsAI> &params);
  static void add_invalid_params(std::vector<BatchMatmulParamsAI> &params);
  static void add_broadcast_params(std::vector<BatchMatmulParamsAI> &params);
  static void generate_reference_kernel_exhaustive_params(
    std::vector<BatchMatmulParamsAI> &params);

  static BatchMatmulParamsAI create_param(
    uint64_t batch_size, uint64_t m, uint64_t n, uint64_t k,
    DataTypeCombination data_types,
    TestCategory category,
    const PostOpConfig &post_op_config,
    bool broadcast_weights = false,
    bool broadcast_input = false,
    bool expect_success = true);
};

} // namespace ai_gtests

#endif // _GTEST_UTILS_BATCHMATMUL_AI_HPP_
