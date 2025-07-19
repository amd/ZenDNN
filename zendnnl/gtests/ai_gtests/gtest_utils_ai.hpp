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

#ifndef _GTEST_UTILS_AI_HPP_
#define _GTEST_UTILS_AI_HPP_

#include <string>
#include <random>
#include <algorithm>
#include <variant>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <atomic>
#include <omp.h>
#include "memory/tensor.hpp"
#include "common/zendnnl_global.hpp"
#include "operators/matmul/matmul_context.hpp"
#include "operators/matmul/matmul_operator.hpp"

// Configurable dimension macros for easy modification
#define AI_MIN_DIM 1
#define AI_MAX_DIM 10000
#define AI_MAX_VALIDATION_ELEMENTS 10000

// Accuracy tolerance macros for output data types
#define AI_MATMUL_TOLERANCE_F32 1e-3f
#define AI_MATMUL_TOLERANCE_BF16 1e-2f
#define AI_MATMUL_TOLERANCE_S8 1e-2f
#define AI_MATMUL_TOLERANCE_DEFAULT 1e-5f

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;
using namespace zendnnl::ops;

namespace ai_gtests {

/** @brief Test categories for comprehensive coverage */
enum class TestCategory {
  ACCURACY,      // Standard accuracy tests
  BOUNDARY,      // Boundary condition tests
  EDGE_CASE,     // Edge case tests
  INVALID,       // Invalid input tests
  REFERENCE_KERNEL // Exhaustive reference kernel tests only
};

/** @brief Data type combinations for testing */
enum class DataTypeCombination {
  F32_F32_F32,   // Input, Weight, Output
  BF16_BF16_BF16,
  BF16_F32_BF16,
  F32_BF16_F32,
  S8_S8_S8,
  S4_S4_S4,
  U8_U8_U8,
  S32_S32_S32
};

/** @brief Post-op configuration for comprehensive testing */
struct PostOpConfig {
  std::vector<post_op_type_t> post_ops;
  std::vector<std::string> binary_tensor_names;
  std::string config_name;

  PostOpConfig() {}
};

/** @brief AI-specific matmul parameter structure */
struct MatmulParamsAI {
  uint64_t m, n, k;
  DataTypeCombination data_types;
  TestCategory category;
  PostOpConfig post_op_config;
  bool trans_a, trans_b;
  bool expect_success;
  std::string test_name;

  MatmulParamsAI() : m(1), n(1), k(1),
    data_types(DataTypeCombination::F32_F32_F32),
    category(TestCategory::ACCURACY),
    trans_a(false), trans_b(false),
    expect_success(true),
    test_name("") {}
};




/** @brief AI-specific utility functions */
class AITestUtils {
 private:
  static std::mt19937 rng;

 public:
  // Sampling and comparison utilities
  static std::vector<size_t> get_sample_indices(size_t total_elements,
      size_t max_samples = AI_MAX_VALIDATION_ELEMENTS);
  static bool compare_sampled_tensors(const tensor_t &tensor1,
                                      const tensor_t &tensor2,
                                      float tolerance = 1e-5f);
  // Kernel support utilities
  static bool is_aocl_kernel_supported(data_type_t input_dtype,
                                       data_type_t weight_dtype,
                                       data_type_t output_dtype,
                                       const std::vector<post_op_type_t> &post_ops);
  static bool is_reference_implementation_supported(data_type_t input_dtype,
      data_type_t weight_dtype,
      data_type_t output_dtype,
      const std::vector<post_op_type_t> &post_ops);

  // Data type utilities
  static data_type_t get_input_dtype(DataTypeCombination combo);
  static data_type_t get_weight_dtype(DataTypeCombination combo);
  static data_type_t get_output_dtype(DataTypeCombination combo);
  static bool is_valid_data_type_combination(DataTypeCombination combo);

  // Post-op configuration utilities
  static std::vector<PostOpConfig> get_all_post_op_configs();
  static PostOpConfig create_binary_add_config();
  static PostOpConfig create_binary_mul_config();
  static PostOpConfig create_relu_config();
  static PostOpConfig create_silu_config();
  static PostOpConfig create_mixed_post_op_config();
  static PostOpConfig create_softmax_config();
  static PostOpConfig create_abs_config();
  static PostOpConfig create_square_config();
  static PostOpConfig create_sqrt_config();
  static PostOpConfig create_exp_config();
  static PostOpConfig create_log_config();
  static PostOpConfig create_leaky_relu_config();
  static PostOpConfig create_elu_config();
  static PostOpConfig create_relu_clip_config();
  static PostOpConfig create_binary_add_mul_config();
  static PostOpConfig create_gelu_tanh_config();
  static PostOpConfig create_gelu_erf_config();
  static PostOpConfig create_sigmoid_config();
  static PostOpConfig create_tanh_config();
  static PostOpConfig create_clip_config();

  // Validation utilities
  static bool validate_dimensions(uint64_t m, uint64_t n, uint64_t k);
  static bool validate_tensor_compatibility(const tensor_t &tensor,
      const std::vector<uint64_t> &expected_dims,
      data_type_t expected_dtype);

  // Error testing utilities
  static status_t test_invalid_context_creation(const MatmulParamsAI &params);
  static status_t test_invalid_operator_creation(const MatmulParamsAI &params);
  static status_t test_missing_tensors(const MatmulParamsAI &params);

  // Reference implementation utilities
  static status_t run_reference_matmul(tensor_t &input,
                                       tensor_t &weights,
                                       tensor_t &bias,
                                       tensor_t &output,
                                       const PostOpConfig &post_op_config,
                                       std::vector<tensor_t> &binary_postop_tensors);

  // Debug and logging utilities
  static void log_tensor_info(const tensor_t &tensor, const std::string &name);

  // Debug print utility
  static void debug_print(const std::string &msg);

  // Memory management utilities
  static void cleanup_tensors(std::vector<tensor_t> &tensors);
  static size_t get_tensor_memory_usage(const tensor_t &tensor);

  // Unique name generation
  static std::string generate_unique_name(const std::string &prefix);
};

/** @brief AI-specific tensor factory for comprehensive testing */
class AITensorFactory {
 private:
  static std::mt19937 rng;
  static std::atomic<uint64_t> tensor_counter;

  static void fill_uniform_data(void *ptr, size_t nelem, data_type_t dtype);
  // Fills a raw data buffer with boundary values for stress-testing matmul numerical stability.
  // The values alternate between large, small, positive, and negative values for each supported type.
  // Used by create_boundary_tensor to populate tensor data.
  static void fill_boundary_data(void *ptr, size_t nelem, data_type_t dtype);

 public:
  static tensor_t create_uniform_tensor(const std::vector<uint64_t> &dims,
                                        data_type_t dtype,
                                        const std::string &name = "");

  static tensor_t create_zero_tensor(const std::vector<uint64_t> &dims,
                                     data_type_t dtype,
                                     const std::string &name = "");

  static tensor_t create_boundary_tensor(const std::vector<uint64_t> &dims,
                                         data_type_t dtype,
                                         const std::string &name = "");
};

/** @brief Comprehensive test parameter generator */
class ParameterGenerator {
 public:
  static std::vector<DataTypeCombination> supported_combinations;
  static std::vector<MatmulParamsAI> generate_comprehensive_test_suite();
  static std::vector<MatmulParamsAI> generate_minimal_test_suite();
  static std::vector<MatmulParamsAI> generate_category_specific_params(
    TestCategory category);
 private:
  static void generate_reference_kernel_exhaustive_params(
    std::vector<MatmulParamsAI> &params);
  static void add_accuracy_params(std::vector<MatmulParamsAI> &params);
  static void add_boundary_params(std::vector<MatmulParamsAI> &params);
  static void add_edge_case_params(std::vector<MatmulParamsAI> &params);
  static void add_invalid_params(std::vector<MatmulParamsAI> &params);
  static MatmulParamsAI create_param(uint64_t m, uint64_t n, uint64_t k,
                                     DataTypeCombination data_types,
                                     TestCategory category,
                                     const PostOpConfig &post_op_config,
                                     bool expect_success = true);
};

} // namespace ai_gtests

#endif // _GTEST_UTILS_AI_HPP_
